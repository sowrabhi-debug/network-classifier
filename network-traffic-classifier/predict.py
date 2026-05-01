import pandas as pd
import joblib

model = joblib.load("models/model.pkl")

sample = pd.read_csv("data/processed/test.csv").drop("label", axis=1).iloc[:1]

prediction = model.predict(sample)[0]
prob = model.predict_proba(sample)[0][1]

print("Risk Score:", prob)

if prediction == 1:
    print("⚠️ ATTACK DETECTED")
else:
    print("✅ NORMAL TRAFFIC")
