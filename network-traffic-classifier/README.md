# Network Traffic Classifier - NSL-KDD

Binary Network Intrusion Detection System using NSL-KDD dataset.

## Features
- Binary classification (Normal vs Attack)
- XGBoost model
- SHAP explainability
- Streamlit Web App
- Model persistence

## Dataset
Place these files inside /data:
- KDDTrain+.txt
- KDDTest+.txt

## Run Steps

1. Install dependencies
pip install -r requirements.txt

2. Preprocess
python preprocess.py

3. Train
python train.py

4. Predict
python predict.py

5. Explain
python explain.py

6. Run App
streamlit run app.py
