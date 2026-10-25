import os
import io
import base64
import requests
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template

import torch
import torchvision.transforms as transforms
import torchvision.models as models

app = Flask(__name__)

# --- Load Irrigation Decision Model ---
MODEL_PATH = "irrigation_model.pkl"
rf_model = None
if os.path.exists(MODEL_PATH):
    try:
        import joblib
        rf_model = joblib.load(MODEL_PATH)
        print("✓ Agronomic model loaded.")
    except Exception as e:
        print(f"Warning: Could not load irrigation model: {e}")

# --- Tomato Disease Classes & Farmer Guidance ---
TOMATO_CLASSES = [
    "Tomato - Healthy",
    "Tomato - Early Blight",
    "Tomato - Late Blight",
    "Tomato - Leaf Mold",
    "Tomato - Bacterial Spot",
    "Tomato - Yellow Leaf Curl Virus"
]

TREATMENTS = {
    "Tomato - Healthy": "Crop leaves are healthy. Keep standard irrigation active.",
    "Tomato - Early Blight": "Foliage infection detected. Apply Mancozeb or Copper spray. Remove affected lower leaves.",
    "Tomato - Late Blight": "High humidity risk! Apply systemic fungicide (Metalaxyl). Avoid wet leaves.",
    "Tomato - Leaf Mold": "Improve row spacing for better air circulation. Apply bio-fungicide or copper spray.",
    "Tomato - Bacterial Spot": "Bacterial spotting seen. Spray copper bactericide. Sanitize cutting tools.",
    "Tomato - Yellow Leaf Curl Virus": "Viral symptom spread by whiteflies. Spray Neem oil or recommended insecticide."
}

vision_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = torch.device("cpu")
mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
mobilenet.classifier[1] = torch.nn.Linear(mobilenet.last_channel, len(TOMATO_CLASSES))
mobilenet.eval()

# Global State Formatted for Farmer Dashboard
latest_state = {
    "field_moisture": 0.0,
    "root_zone_moisture": 0.0,
    "air_temp": 0.0,
    "air_humidity": 0.0,
    "rain_chance": 0,
    "cloud_cover": 0,
    "wind_speed": 0.0,
    "uv_index": 0.0,
    "vpd": 0.0,
    "irrigation_status": "OFF",
    "farm_action_summary": "Syncing crop data...",
    "crop_health_score": 95,
    "leaf_image": None,
    "leaf_diagnosis": "No scan performed yet",
    "diagnosis_accuracy": 0,
    "treatment_advice": "Hold mobile camera or farm scanner facing tomato foliage."
}

def is_leaf_present(pil_img):
    img_rgb = np.array(pil_img.convert("RGB"), dtype=np.float32)
    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    exg = (2.0 * g) - r - b
    green_pixels = np.count_nonzero(exg > 15)
    total_pixels = img_rgb.shape[0] * img_rgb.shape[1]
    return ((green_pixels / total_pixels) * 100.0) >= 12.0

def fetch_weather():
    """Fetches full agronomic atmospheric indicators from Open-Meteo."""
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast?"
            "latitude=12.9141&longitude=74.8560"
            "&current=cloud_cover,wind_speed_10m,uv_index"
            "&hourly=precipitation_probability"
            "&forecast_hours=3"
        )
        resp = requests.get(url, timeout=3)
        if resp.status_code == 200:
            d = resp.json()
            cloud = d.get("current", {}).get("cloud_cover", 0)
            wind = d.get("current", {}).get("wind_speed_10m", 0.0)
            uv = d.get("current", {}).get("uv_index", 0.0)
            rain_hourly = d.get("hourly", {}).get("precipitation_probability", [0])
            rain_prob = rain_hourly[0] if rain_hourly else 0
            return int(rain_prob), int(cloud), float(wind), float(uv)
    except Exception as e:
        print(f"Weather error: {e}")
    return 0, 0, 0.0, 0.0

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/api/state", methods=["GET"])
def get_state():
    resp = jsonify(latest_state)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp, 200

@app.route("/api/telemetry", methods=["POST"])
def telemetry():
    global latest_state
    try:
        data = request.get_json(force=True, silent=True) or {}
        
        # Ingest field sensors
        s1 = float(data.get("soil_1", 0.0))
        s2 = float(data.get("soil_2", 0.0))
        temp = float(data.get("temp", 28.5))
        hum = float(data.get("humidity", 72.0))

        # Effective root zone for tomato
        s_effective = round((0.35 * s1) + (0.65 * s2), 1)

        # Weather factors
        rain_prob, cloud, wind, uv = fetch_weather()

        # Vapor Pressure Deficit (VPD in kPa)
        svp = 0.61078 * np.exp((17.27 * temp) / (temp + 237.3))
        avp = svp * (hum / 100.0)
        vpd = round(max(0.0, float(svp - avp)), 2)

        # Irrigation Decision
        relay_cmd = "OFF"
        summary = "Moisture is balanced. Roots are well-hydrated."

        if rf_model is not None:
            n_features = getattr(rf_model, "n_features_in_", 5)
            if n_features == 6:
                feat = np.array([[s1, s2, temp, hum, rain_prob, vpd]])
            else:
                feat = np.array([[s_effective, temp, hum, rain_prob, cloud]])
            pred = rf_model.predict(feat)[0]
            if pred == 1:
                relay_cmd = "ON"
                summary = "Drip watering active: Tomato root moisture reached irrigation threshold."
            else:
                relay_cmd = "OFF"
                if rain_prob >= 50:
                    summary = f"Conserving water: Rain forecast imminent ({rain_prob}% chance)."
                else:
                    summary = "Soil water reserves adequate. Pump is resting."
        else:
            if s_effective < 35.0 and rain_prob < 50:
                relay_cmd = "ON"
                summary = "Drip watering active: Soil is dry."
            else:
                relay_cmd = "OFF"
                summary = "Moisture optimal."

        # Compute Crop Health Index (0-100)
        score = 100
        if s_effective < 25.0: score -= 25
        elif s_effective < 35.0: score -= 10
        if vpd > 1.6: score -= 10
        if "Blight" in latest_state.get("leaf_diagnosis", ""): score -= 30

        latest_state.update({
            "field_moisture": s_effective,
            "root_zone_moisture": s2,
            "air_temp": temp,
            "air_humidity": hum,
            "rain_chance": rain_prob,
            "cloud_cover": cloud,
            "wind_speed": wind,
            "uv_index": uv,
            "vpd": vpd,
            "irrigation_status": relay_cmd,
            "farm_action_summary": summary,
            "crop_health_score": max(20, min(score, 100))
        })

        return jsonify({"relay": relay_cmd, "reason": summary}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/upload-leaf", methods=["POST"])
def upload_leaf():
    global latest_state
    if "imageFile" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["imageFile"]
    img_bytes = file.read()

    try:
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG", quality=70)
        latest_state["leaf_image"] = base64.b64encode(buffered.getvalue()).decode("utf-8")

        if not is_leaf_present(pil_img):
            latest_state["leaf_diagnosis"] = "No Leaf Detected"
            latest_state["diagnosis_accuracy"] = 0
            latest_state["treatment_advice"] = "Please center a clear tomato leaf in the frame."
            return jsonify({"status": "rejected"}), 200

        tensor = vision_transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = mobilenet(tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probs, 0)

        conf_score = int(round(confidence.item() * 100.0))

        if conf_score < 45:
            d_name = "Unclear Foliage Image"
            t_advice = "Improve lighting and keep camera 10–15 cm from leaf."
        else:
            d_name = TOMATO_CLASSES[predicted_idx.item()]
            t_advice = TREATMENTS.get(d_name, "Check crop regularly.")

        latest_state["leaf_diagnosis"] = d_name
        latest_state["diagnosis_accuracy"] = conf_score
        latest_state["treatment_advice"] = t_advice

        return jsonify({"status": "success", "diagnosis": d_name}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
