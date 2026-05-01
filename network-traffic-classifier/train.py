import pandas as pd
import joblib
import os
from sklearn.metrics import classification_report, roc_auc_score
from model import get_model

def main():
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")

    X_train = train.drop("label", axis=1)
    y_train = train["label"]

    X_test = test.drop("label", axis=1)
    y_test = test["label"]

    model = get_model()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:,1]

    print(classification_report(y_test, preds))
    print("ROC-AUC:", roc_auc_score(y_test, proba))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")

    print("Model saved.")

if __name__ == "__main__":
    main()
