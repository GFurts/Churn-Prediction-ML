# Churn Prediction ML

![CI](https://github.com/GFurts/Churn-Prediction-ML/actions/workflows/ci.yml/badge.svg)

[![API](https://img.shields.io/badge/API-Live-brightgreen)](https://churn-prediction-ml-e4f1.onrender.com/docs)

> End-to-end machine learning project for telecom customer churn prediction — from exploratory analysis to a production-ready inference API.

---

## Overview

This project predicts which telecom customers are most likely to cancel their service (**churn**), using tabular data with demographic, contractual, and usage features.

Built as a **complete ML Engineering pipeline**, it covers every stage from raw data exploration to a served REST API — applying software engineering best practices throughout.

---

## Business Problem

In telecommunications, churn directly impacts revenue and operational costs:

- Each lost customer represents recurring revenue loss
- Acquiring a new customer costs 5–7× more than retaining an existing one
- Targeted retention campaigns require identifying at-risk customers **before** they cancel

A reliable churn prediction model enables proactive, cost-effective retention strategies.

---

## Dataset

**Telco Customer Churn** (IBM Sample Dataset) — 7,043 customers, 19 features, binary target (`Churn`: Yes/No).

Key features: `tenure`, `Contract`, `MonthlyCharges`, `TotalCharges`, `InternetService`, `TechSupport`, `PaymentMethod`.

> Class imbalance: ~26.5% churn rate — addressed through stratified cross-validation and threshold tuning.

---

## Project Stages

### Stage 1 — EDA & Baselines

- Full exploratory data analysis (distributions, correlations, missing values)
- Data quality treatment — including business-aware imputation of `TotalCharges`
- Metric selection for imbalanced classification: **ROC-AUC**, **PR-AUC**, **F1-score**
- Baseline models: `DummyClassifier` and `LogisticRegression` (scikit-learn)
- Experiment tracking with **MLflow**

### Stage 2 — Neural Network with PyTorch

- MLP architecture built in **PyTorch**
- Training loop with **batching** and **early stopping** (best model checkpoint restored)
- Evaluation across 6 metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
- Full comparison: MLP vs. Logistic Regression vs. DummyClassifier

### Stage 3 — ML Engineering & API

- Code refactored into modular `src/` package
- Reproducible preprocessing pipeline serialized with `joblib`
- REST API built with **FastAPI** + **Pydantic** validation
- Structured logging with `loguru`
- Automated tests with **pytest** (smoke, schema, API)
- Linting with **ruff**

---

## Key Results & Serving Decision

| Model | ROC-AUC | F1-Score | PR-AUC |
|---|---|---|---|
| DummyClassifier | ~0.50 | ~0.28 | ~0.27 |
| Logistic Regression | **~0.84** | **~0.61** | **~0.70** |
| MLP (PyTorch) | ~0.83 | ~0.60 | ~0.69 |

**The Logistic Regression was chosen for the serving layer** — it matched the MLP's performance while offering simpler operability, faster inference, and full interpretability (coefficients expose feature importance directly).

> The MLP is retained in `notebooks/` as documented experimental evidence of the comparison process.

This reflects a core ML Engineering principle: **deploy the simplest model that meets performance requirements.**

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data & EDA | pandas, numpy, matplotlib, seaborn |
| Modeling | scikit-learn, PyTorch |
| Experiment Tracking | MLflow |
| API | FastAPI, Pydantic, Uvicorn |
| Testing | pytest, pandera |
| Code Quality | ruff, loguru |
| Packaging | pyproject.toml, joblib |

---

## Project Structure

```
churn-prediction-ml/
├── data/
│   └── raw/                  # Original dataset (never modified)
├── docs/                     # Model card and architecture docs
├── notebooks/
│   ├── 01_eda_baseline.ipynb
│   └── 02_mlp_pytorch.ipynb
├── src/
│   ├── __init__.py
│   ├── api.py                # FastAPI app
│   ├── config.py             # Paths and constants
│   ├── data.py               # Data loading and cleaning
│   ├── features.py           # Feature engineering
│   ├── logger.py             # Structured logging setup
│   ├── predict.py            # Inference logic
│   ├── preprocessamento.py   # Preprocessing pipeline
│   ├── schemas.py            # Pydantic schemas
│   └── train_baseline.py     # Model training script
├── tests/                    # Automated tests
├── .gitignore
├── Makefile
├── pyproject.toml
└── requirements.txt
```

---

## API Endpoints

### `GET /health`

Returns API status.

```json
{ "status": "ok" }
```

### `POST /predict`

Receives customer data and returns churn prediction and probability.

**Request payload:**
```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 89.5,
  "TotalCharges": 1074.0
}
```

**Response:**
```json
{
  "prediction": 1,
  "churn_probability": 0.7138
}
```

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/GFurts/Churn-Prediction-ML.git
cd churn-prediction-ml
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model
```bash
python -m src.train_baseline
```

### 5. Start the API
```bash
uvicorn src.api:app --reload
```

### 6. Access interactive docs
Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Running Tests

```bash
pytest -v
```

## Code Quality

```bash
ruff check .
```

Or use the Makefile:

```bash
make lint
make test
make run
```

---

## Next Steps

- [x] Deploy API to cloud — [Live on Render](https://churn-prediction-ml-e4f1.onrender.com/docs)
- [ ] Add GitHub Actions CI/CD pipeline
- [ ] Implement data drift monitoring
- [ ] Explore MLP serving as an alternative endpoint
- [ ] Add SHAP values for prediction explainability

---

## Author

**Gabriel Furtado**
Postgraduate student in Machine Learning Engineering

[![GitHub](https://img.shields.io/badge/GitHub-GFurts-181717?logo=github)](https://github.com/GFurts)
