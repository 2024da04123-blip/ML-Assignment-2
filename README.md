# ML Assignment 2 — Fraud Detection (Streamlit)

This repository trains and compares **6 classification models** and provides a **Streamlit** app that lets a user upload a CSV, select a model, and view metrics + a confusion matrix.

## Assignment checklist mapping (high level)
- **Dataset**: from **Kaggle/UCI**, with at least **500 rows** and at least **12 features**
- **Models (6)**: Logistic Regression, Decision Tree, k-Nearest Neighbor, Naive Bayes, Random Forest, XGBoost
- **Metrics**: Accuracy, AUC, Precision, Recall, F1, MCC
- **Streamlit app**: CSV upload + model dropdown + metrics display + confusion matrix

## Dataset
- **File**: `Digital_Payment_Fraud_Detection_Dataset.csv`
- **Target**: `fraud_label` (0 = non-fraud, 1 = fraud)
- **Rows / Columns**: 7500 rows, 15 columns
- **Kaggle/UCI source**: [Digital Payment Fraud Detection (Kaggle)](https://www.kaggle.com/datasets/jayjoshi37/digital-payment-fraud-detection)

We drop ID-like columns (`transaction_id`, `user_id`) before training. After dropping them, there are still **12+ features**, satisfying the minimum feature requirement.

## Project structure
- `train.py`: trains all models, computes metrics, saves artifacts
- `streamlit_app.py`: Streamlit app (upload CSV, pick model, predict, evaluate)
- `models/models.joblib`: saved pipelines + label encoder (generated)
- `reports/metrics.csv`: metrics comparison table (generated)
- `reports/test_data.csv`: holdout test split for demo uploads (generated)

## How to run

### 1) Install dependencies
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) Train models + generate reports
```bash
python train.py
```

Outputs:
- `models/models.joblib`
- `reports/metrics.csv`
- `reports/metrics.json`
- `reports/test_data.csv`

### 3) Run Streamlit
```bash
streamlit run streamlit_app.py
```

Upload `reports/test_data.csv` to see metrics + confusion matrix immediately.


