import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Synthetic training dataset representing farm dynamics
np.random.seed(42)
n_samples = 1500

soil_moist = np.random.uniform(10, 80, n_samples)
temp = np.random.uniform(18, 42, n_samples)
humidity = np.random.uniform(25, 95, n_samples)
rain_prob = np.random.uniform(0, 100, n_samples)
cloud_cover = np.random.uniform(0, 100, n_samples)

pump_decision = []
for s, t, h, r, c in zip(soil_moist, temp, humidity, rain_prob, cloud_cover):
    if r >= 50.0:
        pump_decision.append(0)  # Rain expected -> do not irrigate
    elif s < 35.0:
        pump_decision.append(1)  # Dry soil and clear skies -> irrigate
    elif (35.0 <= s <= 45.0) and (t > 33.0) and (h < 40.0) and (r < 30.0):
        pump_decision.append(1)  # High evapotranspiration -> irrigate
    else:
        pump_decision.append(0)  # Soil moisture optimal -> idle

X = pd.DataFrame({
    'soil_moist': soil_moist,
    'temp': temp,
    'humidity': humidity,
    'rain_prob': rain_prob,
    'cloud_cover': cloud_cover
})
y = pump_decision

clf = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42)
clf.fit(X, y)
joblib.dump(clf, 'irrigation_model.pkl')
print("Model saved: irrigation_model.pkl")