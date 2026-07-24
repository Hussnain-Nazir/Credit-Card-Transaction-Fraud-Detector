# Credit Card Transaction Fraud Detector

A complete, production-style machine learning project that detects fraudulent
credit card transactions using engineered transaction features, class-imbalance
handling (SMOTE), and gradient-boosted trees - wrapped in a clean Streamlit
web application.

## Overview

Credit card fraud is a classic **imbalanced classification** problem: fraudulent
transactions typically represent less than 1–2% of all transactions. This
project builds a full pipeline - from raw data to a deployable web app - that:

- Loads and explores a real transaction-level dataset
- Engineers meaningful features (distance between cardholder and merchant,
  cardholder age, time-of-day patterns)
- Handles severe class imbalance using **SMOTE**
- Trains and compares **Logistic Regression**, **Random Forest**, and **XGBoost**
- Selects the best model based on F1 score and ROC-AUC
- Serves real-time predictions through a minimalistic **Streamlit** app

## Features

- End-to-end Jupyter notebook (EDA → preprocessing → modeling → evaluation)
- Multiple models trained and benchmarked side by side
- Class imbalance handled properly with SMOTE (applied to training data only)
- Feature engineering: haversine distance, cardholder age, hour/day/month
- Saved, reusable model + preprocessing pipeline (`model.pkl`, `scaler.pkl`)
- Clean, minimalistic Streamlit UI (black sidebar, white main content)
- Clear, modular project structure

## Tech Stack

| Layer | Tools |
|---|---|
| Data & EDA | pandas, numpy, matplotlib, seaborn |
| Modeling | scikit-learn, XGBoost, imbalanced-learn (SMOTE) |
| Serialization | joblib |
| Web App | Streamlit |
| Notebook | Jupyter |

## Project Structure

```
Credit_Card_Transaction_Fraud_Detector/
│
├── data/
│   └── Credit_Card_Transactions.csv     # Dataset
│
├── models/
│   ├── model.pkl                        # Trained Best Model
│   ├── scaler.pkl                        # Fitted Preprocessing Pipeline
│   └── metadata.pkl                      # Deature list, Categories, Metrics
│
├── notebook/
│   └── Credit_Card_Transaction_Fraud_Detector_Pipeline.ipynb   # Complete ML Pipeline Notebook
│
├── app.py                                # Streamlit Web App
├── requirements.txt                      # Python Dependencies
└── README.md
```

## Installation Steps

1. **Clone or download** this project folder.

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Add the dataset**: place your `Credit_Card_Transactions.csv` file inside
   the `data/` folder (already included in this project).

## How to Run the Notebook

1. Launch Jupyter:
   ```bash
   jupyter notebook notebook/Credit_Card_Transaction_Fraud_Detector_Pipeline.ipynb
   ```
2. Run all cells from top to bottom. This will:
   - Explore the dataset
   - Engineer features
   - Train Logistic Regression, Random Forest, and XGBoost
   - Compare models and select the best one
   - Save `model.pkl`, `scaler.pkl`, and `metadata.pkl` into `models/`

> The `models/` folder already ships with pre-trained artifacts, so you can
> skip straight to running the app if you just want to try it out.

## How to Run the Streamlit App

From the project root directory:

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal (typically `http://localhost:8501`).

**Using the app:**
1. Enter the transaction amount, merchant category, cardholder details, and
   location information in the form.
2. Click **Predict**.
3. The app displays whether the transaction is likely **fraudulent** or
   **genuine**, along with the estimated fraud probability.

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| XGBoost (best) | 0.998 | 0.90 | 0.86 | 0.88 | 0.999 |
| Random Forest | 0.994 | 0.67 | 0.86 | 0.75 | 0.992 |
| Logistic Regression | 0.861 | 0.05 | 0.74 | 0.10 | 0.917 |

*XGBoost was selected as the production model based on the best balance of
precision, recall, and ROC-AUC.*

## Future Improvements

- Hyperparameter tuning with GridSearchCV / Optuna
- Add behavioral/rolling features per cardholder (e.g. average spend, transaction velocity)
- Threshold tuning to optimize for business-specific precision/recall trade-offs
- Model explainability with SHAP values in the app
- Batch prediction support (CSV upload) in the Streamlit app
- Deploy as a REST API (FastAPI) alongside the Streamlit front end
- Add automated model monitoring for data/concept drift in production

---

**Disclaimer:** This project is for educational purposes. Do not use it as-is
in a real financial system without further validation, security review, and
compliance checks.
