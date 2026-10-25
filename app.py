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

# --- Load Irrigation ML Model ---
MODEL_PATH = "irrigation_model.pkl"
rf_model = None
if os.path.exists(MODEL_PATH):
    try:
        import joblib
        rf_model = joblib.load(MODEL_PATH)
        print("✓ Irrigation model loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load irrigation model: {e}")

# --- Tomato Disease Classes & Prescriptions ---
TOMATO_CLASSES = [
    "Tomato - Healthy",
    "Tomato - Early Blight",
    "Tomato - Late Blight",
    "Tomato - Leaf Mold",
    "Tomato - Bacterial Spot",
    "Tomato - Yellow Leaf Curl Virus"
]

TREATMENTS = {
    "Tomato - Healthy": "Crop foliage is healthy. Maintain regular drip irrigation.",
    "Tomato - Early Blight": "Fungicide required (Mancozeb or Copper spray). Prune lower infected leaves.",
    "Tomato - Late Blight": "High humidity risk! Apply Metalaxyl immediately and avoid wetting leaves.",
    "Tomato - Leaf Mold": "Improve canopy ventilation. Apply bio-fungicide or copper hydroxide.",
    "Tomato - Bacterial Spot": "Apply copper-based bactericide. Disinfect pruning shears to prevent spread.",
    "Tomato - Yellow Leaf Curl Virus": "Vector-borne (Whiteflies). Apply Neem oil or systemic insecticide."
}

# --- Vision Preprocessing Pipeline ---
vision_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = torch.device("cpu")
mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
mobilenet.classifier[1] = torch.nn.Linear(mobilenet.last_channel, len(TOMATO_CLASSES))
mobilenet.eval()

# Global State for Telemetry & Dashboard View
latest_state = {
    "soil_1": 0.0,
    "soil_2": 0.0,
    "soil_avg": 0.0,
    "temp": 0.0,
    "humidity": 0.0,
    "rain_prob": 0,
    "cloud_cover": 0,
    "relay": "OFF",
    "decision_reason": "Waiting for sensor telemetry...",
    "leaf_image": None,
    "disease": "Awaiting leaf scan...",
    "confidence": 0.0,
    "treatment": "Hold camera 10–15 cm from tomato foliage."
}

def is_leaf_present(pil_img):
    """Vegetation index gate to filter out non-plant objects."""
    img_rgb = np.array(pil_img.convert("RGB"), dtype=np.float32)
    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    exg = (2.0 * g) - r - b
    green_pixels = np.count_nonzero(exg > 15)
    total_pixels = img_rgb.shape[0] * img_rgb.shape[1]
    return ((green_pixels / total_pixels) * 100.0) >= 12.0

def fetch_weather():
    try:
        url = "https://api.open-meteo.com/v1/forecast?latitude=12.9141&longitude=74.8560&current=cloud_cover&hourly=precipitation_probability&forecast_hours=3"
        resp = requests.get(url, timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            cloud = data.get("current", {}).get("cloud_cover", 0)
            rain_hourly = data.get("hourly", {}).get("precipitation_probability", [0])
            rain_prob = rain_hourly[0] if rain_hourly else 0
            return int(rain_prob), int(cloud)
    except Exception as e:
        print(f"Weather API error: {e}")
    return 0, 0

# --- Dashboard View ---
@app.route("/", methods=["GET"])
def dashboard():
    return render_template("index.html")

# --- Live Telemetry API (with full CORS and No-Cache) ---
@app.route("/api/state", methods=["GET"])
def get_state():
    resp = jsonify(latest_state)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp, 200

# --- ESP32 Sensor Ingestion ---
@app.route("/api/telemetry", methods=["POST"])
def telemetry():
    global latest_state
    try:
        data = request.get_json(force=True, silent=True) or {}
        s1 = float(data.get("soil_1", 0.0))
        s2 = float(data.get("soil_2", 0.0))
        s_effective = (0.35 * s1) + (0.65 * s2)
        temp = float(data.get("temp", 28.0))
        hum = float(data.get("humidity", 70.0))

        rain_prob, cloud = fetch_weather()

        # Tetens equation for VPD
        svp = 0.61078 * np.exp((17.27 * temp) / (temp + 237.3))
        avp = svp * (hum / 100.0)
        vpd = max(0.0, float(svp - avp))

        relay_cmd = "OFF"
        reason = "Optimal moisture balance for tomato roots."

        if rf_model is not None:
            # Check model feature dimensions dynamically
            n_features = getattr(rf_model, "n_features_in_", 5)
            if n_features == 6:
                features = np.array([[s1, s2, temp, hum, rain_prob, vpd]])
            else:
                features = np.array([[s_effective, temp, hum, rain_prob, cloud]])

            pred = rf_model.predict(features)[0]
            if pred == 1:
                relay_cmd = "ON"
                reason = f"Tomato Deficit: Root zone {s_effective:.1f}%, VPD {vpd:.2f} kPa, Rain {rain_prob}%."
            else:
                relay_cmd = "OFF"
                reason = f"Tomato Hydrated: Moisture adequate ({s_effective:.1f}%), Rain risk {rain_prob}%."
        else:
            # Fallback agronomic rule
            if s_effective < 35.0 and rain_prob < 50:
                relay_cmd = "ON"
                reason = f"Fallback Rule: Low root zone moisture ({s_effective:.1f}%)."
            else:
                relay_cmd = "OFF"
                reason = f"Fallback Rule: Moisture adequate ({s_effective:.1f}%)."

        latest_state.update({
            "soil_1": s1,
            "soil_2": s2,
            "soil_avg": s_effective,
            "temp": temp,
            "humidity": hum,
            "rain_prob": rain_prob,
            "cloud_cover": cloud,
            "relay": relay_cmd,
            "decision_reason": reason
        })

        return jsonify({"relay": relay_cmd, "reason": reason}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# --- ESP32-CAM Image Ingestion ---
@app.route("/upload-leaf", methods=["POST"])
def upload_leaf():
    global latest_state
    if "imageFile" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["imageFile"]
    img_bytes = file.read()

    try:
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Encode captured frame as base64 for display on web dashboard
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG", quality=70)
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        latest_state["leaf_image"] = img_base64

        # Discard non-plant foliage
        if not is_leaf_present(pil_img):
            latest_state["disease"] = "No Tomato Leaf Detected"
            latest_state["confidence"] = 0.0
            latest_state["treatment"] = "Point ESP32-CAM directly at a tomato leaf. Non-plant object ignored."
            return jsonify({"status": "rejected", "reason": "No leaf found"}), 200

        # CNN Classification
        tensor = vision_transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = mobilenet(tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)

        conf_score = round(confidence.item() * 100.0, 1)

        if conf_score < 45.0:
            disease_name = "Unclear Foliage / Low Light"
            treatment = "Adjust ambient lighting and distance to 10–15 cm."
        else:
            disease_name = TOMATO_CLASSES[predicted_idx.item()]
            treatment = TREATMENTS.get(disease_name, "Inspect leaf regularly.")

        latest_state["disease"] = disease_name
        latest_state["confidence"] = conf_score
        latest_state["treatment"] = treatment

        return jsonify({
            "status": "success",
            "disease": disease_name,
            "confidence": conf_score,
            "treatment": treatment
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
