import os
import requests
import joblib
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
import torch
import torchvision.transforms as transforms
import torchvision.models as models

app = Flask(__name__)
os.makedirs("static", exist_ok=True)

# Farm Coordinates (Coastal Karnataka)
FARM_LAT = 12.9141
FARM_LON = 74.8560

# Load Irrigation ML Model
try:
    irrigation_model = joblib.load('irrigation_model.pkl')
    print("Irrigation ML model loaded.")
except Exception as e:
    print("Warning: Model not found. Run train_models.py first.", e)
    irrigation_model = None

# Load MobileNetV2 Vision CNN
DISEASE_CLASSES = [
    "Tomato - Early Blight", "Tomato - Late Blight", "Tomato - Healthy",
    "Potato - Early Blight", "Potato - Late Blight", "Potato - Healthy"
]
cnn_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
cnn_model.classifier[1] = torch.nn.Linear(cnn_model.classifier[1].in_features, len(DISEASE_CLASSES))
cnn_model.eval()

img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

TREATMENTS = {
    "Tomato - Early Blight": "Prune lower infected leaves; apply copper-based organic fungicide.",
    "Tomato - Late Blight": "Remove affected foliage immediately; maintain strict root drip watering.",
    "Tomato - Healthy": "Optimal foliage health. Maintain regular drip schedule.",
    "Potato - Early Blight": "Apply organic neem extract or Mancozeb spray.",
    "Potato - Late Blight": "Ensure good furrow drainage; discard decomposing foliage.",
    "Potato - Healthy": "Crop in excellent condition."
}

state = {
    "soil_avg": 0.0, "soil_1": 0.0, "soil_2": 0.0,
    "temp": 0.0, "humidity": 0.0, "rain_prob": 0, "cloud_cover": 0,
    "pump_cmd": "OFF", "disease_name": "Awaiting scan",
    "disease_conf": 0.0, "treatment": "No leaf scanned yet",
    "image_url": "/static/latest_leaf.jpg",
    "decision_reason": "Waiting for sensor telemetry"
}

def fetch_weather():
    try:
        url = (f"https://api.open-meteo.com/v1/forecast?latitude={FARM_LAT}&longitude={FARM_LON}"
               f"&current=cloud_cover,rain&hourly=precipitation_probability&forecast_hours=3")
        r = requests.get(url, timeout=3)
        if r.status_code == 200:
            d = r.json()
            state["cloud_cover"] = d.get("current", {}).get("cloud_cover", 0)
            probs = d.get("hourly", {}).get("precipitation_probability", [0])
            state["rain_prob"] = max(probs[:2]) if probs else 0
    except Exception as e:
        print("Weather API sync failed:", e)

@app.route('/api/telemetry', methods=['POST'])
def receive_telemetry():
    data = request.get_json(force=True)
    s1 = data.get("soil_1", 0.0)
    s2 = data.get("soil_2", 0.0)
    s_avg = round((s1 + s2) / 2.0, 1)

    state["soil_1"] = s1
    state["soil_2"] = s2
    state["soil_avg"] = s_avg
    state["temp"] = data.get("temp", 0.0)
    state["humidity"] = data.get("humidity", 0.0)

    fetch_weather()

    if irrigation_model:
        features = np.array([[s_avg, state["temp"], state["humidity"], state["rain_prob"], state["cloud_cover"]]])
        pred = irrigation_model.predict(features)[0]
        state["pump_cmd"] = "ON" if pred == 1 else "OFF"
        if pred == 1:
            state["decision_reason"] = f"ML Triggered: Soil deficit ({s_avg}%), Rain prob {state['rain_prob']}%."
        else:
            state["decision_reason"] = f"ML Idle: Soil moisture adequate or rain expected ({state['rain_prob']}%)."
    else:
        state["pump_cmd"] = "ON" if s_avg < 35 and state["rain_prob"] < 50 else "OFF"

    return jsonify({"relay": state["pump_cmd"]}), 200

@app.route('/upload-leaf', methods=['POST'])
def upload_leaf():
    if 'imageFile' not in request.files:
        return jsonify({"error": "No image received"}), 400

    img_file = request.files['imageFile']
    save_path = os.path.join("static", "latest_leaf.jpg")
    img_file.save(save_path)

    image = Image.open(save_path).convert('RGB')
    tensor = img_transforms(image).unsqueeze(0)

    with torch.no_grad():
        outputs = cnn_model(tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, pred_idx = torch.max(probs, 0)

    detected = DISEASE_CLASSES[pred_idx.item()]
    conf = round(confidence.item() * 100, 1)

    state["disease_name"] = detected
    state["disease_conf"] = conf
    state["treatment"] = TREATMENTS.get(detected, "Monitor crop closely.")
    state["image_url"] = f"/static/latest_leaf.jpg?t={os.urandom(4).hex()}"

    print(f"Leaf Diagnosed: {detected} ({conf}%)")
    return jsonify({"status": "success", "disease": detected, "confidence": conf}), 200

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    return jsonify(state)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)