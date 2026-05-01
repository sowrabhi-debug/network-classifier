import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os

print("Loading model...")
model = joblib.load("models/model.pkl")

print("Loading processed data...")
data = pd.read_csv("data/processed/test.csv")
X = data.drop("label", axis=1)

# Use sample for performance
X_sample = X.sample(1000, random_state=42)

print("Initializing SHAP TreeExplainer...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

os.makedirs("explanations", exist_ok=True)

# 1️⃣ Global Summary Plot
print("Generating Global Summary Plot...")
plt.figure()
shap.summary_plot(shap_values, X_sample, show=False)
plt.tight_layout()
plt.savefig("explanations/global_summary.png")
plt.close()

# 2️⃣ Feature Importance (Bar)
print("Generating Feature Importance Plot...")
plt.figure()
shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("explanations/feature_importance.png")
plt.close()

# 3️⃣ Local Explanation for a single sample
print("Generating Local Explanation...")
sample_index = 0
force_plot = shap.force_plot(
    explainer.expected_value,
    shap_values[sample_index],
    X_sample.iloc[sample_index],
    matplotlib=True,
    show=False
)

plt.tight_layout()
plt.savefig("explanations/local_force_plot.png")
plt.close()

# 4️⃣ Dependence Plot for top feature
feature_importance = np.abs(shap_values).mean(0)
top_feature_index = np.argmax(feature_importance)
top_feature_name = X_sample.columns[top_feature_index]

print(f"Generating Dependence Plot for: {top_feature_name}")
plt.figure()
shap.dependence_plot(
    top_feature_name,
    shap_values,
    X_sample,
    show=False
)
plt.tight_layout()
plt.savefig("explanations/dependence_plot.png")
plt.close()

print("All SHAP explanations generated in /explanations folder.")
