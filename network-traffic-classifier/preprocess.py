import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

COLUMN_NAMES = [
"duration","protocol_type","service","flag","src_bytes","dst_bytes",
"land","wrong_fragment","urgent","hot","num_failed_logins",
"logged_in","num_compromised","root_shell","su_attempted","num_root",
"num_file_creations","num_shells","num_access_files","num_outbound_cmds",
"is_host_login","is_guest_login","count","srv_count","serror_rate",
"srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
"diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
"dst_host_same_srv_rate","dst_host_diff_srv_rate",
"dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
"dst_host_serror_rate","dst_host_srv_serror_rate",
"dst_host_rerror_rate","dst_host_srv_rerror_rate",
"label","difficulty"
]

def load_data():
    train = pd.read_csv("data/KDDTrain+.txt", names=COLUMN_NAMES)
    test = pd.read_csv("data/KDDTest+.txt", names=COLUMN_NAMES)
    return train, test

def preprocess(df):
    df = df.drop("difficulty", axis=1)

    # Binary labels
    df["label"] = df["label"].apply(lambda x: 0 if x == "normal" else 1)

    # Encode categorical columns
    categorical_cols = ["protocol_type", "service", "flag"]
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    return df

def main():
    train, test = load_data()

    train = preprocess(train)
    test = preprocess(test)

    os.makedirs("data/processed", exist_ok=True)

    train.to_csv("data/processed/train.csv", index=False)
    test.to_csv("data/processed/test.csv", index=False)

    print("Preprocessing complete.")

if __name__ == "__main__":
    main()
