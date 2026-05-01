Network Traffic Classifier (Explainable IDS using NSL-KDD)

An end-to-end **Explainable Intrusion Detection System (IDS)** built using machine learning on the NSL-KDD dataset.
This project combines **classification, explainability (SHAP), and an interactive dashboard** to simulate a real-world cybersecurity analytics tool.


Key Features

* Binary Classification (Normal vs Attack)
  
* High-performance XGBoost model
  
* SHAP Explainability
  * Global feature importance
  * Local (per-sample) explanations
    
* Tunable Risk Threshold
  * Dynamically adjust attack sensitivity
  * Observe real-time impact on predictions
    
* Interactive **Streamlit Dashboard
  * Attack distribution (pie chart)
  * Risk score histogram
  * Live metrics (attack count, normal traffic)
    
* Model persistence using `joblib`
  
* Downloadable prediction results

---

Key Concept

Unlike traditional classifiers, this system:

> Outputs probabilities (risk scores) instead of fixed labels
> and allows dynamic threshold tuning to control detection sensitivity.

This mimics real-world IDS systems where:

* Lower thresholds → more attack detection (higher false positives)
* Higher thresholds → fewer alerts (risk of missed attacks)

---

Dataset

This project uses the **NSL-KDD**.

### Required Files:

Place inside `/data` folder:

```
KDDTrain+.txt
KDDTest+.txt
```

---

Project Structure

```
network-traffic-classifier/
│
├── data/
│   ├── KDDTrain+.txt
│   ├── KDDTest+.txt
│   └── processed/
│
├── models/
│   └── model.pkl
│
├── explanations/
│   └── (SHAP plots)
│
├── preprocess.py
├── model.py
├── train.py
├── predict.py
├── explain.py
├── app.py
│
├── requirements.txt
└── README.md
```

---

Installation & Setup

1️⃣ Clone Repository

```bash
git clone <your-repo-url>
cd network-traffic-classifier
```

---

2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
```

---

3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

How to Run

🔹 Step 1: Preprocess Data

```bash
python preprocess.py
```

---

🔹 Step 2: Train Model

```bash
python train.py
```

Outputs:

* Classification report
* ROC-AUC score (~0.97)
* Saved model (`models/model.pkl`)

---

🔹 Step 3: Make Predictions

```bash
python predict.py
```

---

🔹 Step 4: Generate Explainability

```bash
python explain.py
```

Outputs:

* Global SHAP summary
* Feature importance plot
* Local explanation
* Dependence plot

Saved in `/explanations`

---

🔹 Step 5: Run Dashboard

```bash
streamlit run app.py
```

---

Streamlit Dashboard Features

* Upload processed CSV data
* Adjust **risk threshold slider**
* View:

  * Attack vs Normal distribution
  * Risk score histogram
  * SHAP feature importance
* Inspect **row-level explanations**
* Download prediction results

---

Model Performance

* Accuracy: ~79%
* ROC-AUC: ~0.97

> High ROC-AUC indicates strong class separation capability.

---

Explainability (SHAP)

This project uses SHAP to:

* Identify **important network features**
* Explain **why a specific traffic instance is flagged**
* Improve transparency in decision-making

---

Tech Stack

* Python
* Pandas / NumPy
* Scikit-learn
* XGBoost
* SHAP
* Matplotlib
* Streamlit

---

Future Improvements

* Multi-class classification (DoS, Probe, R2L, U2R)
* Real-time packet capture integration
* Adaptive threshold optimization
* Drift detection & model monitoring
* REST API deployment

---

Use Cases

* Intrusion Detection Systems (IDS)
* Cybersecurity analytics dashboards
* Explainable AI demonstrations
* ML portfolio / academic projects

---

⭐ If You Like This Project

Give it a star ⭐ and feel free to fork or contribute!
