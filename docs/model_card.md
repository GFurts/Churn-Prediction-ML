# Model Card — Churn Prediction (Logistic Regression)

## Model Overview

| Field | Details |
|---|---|
| **Model type** | Logistic Regression (scikit-learn) |
| **Task** | Binary classification — customer churn prediction |
| **Serving version** | v1.0 |
| **Trained on** | Telco Customer Churn dataset (IBM) |
| **Served via** | FastAPI REST API (`/predict`) |

---

## Intended Use

**Primary use case:** Identify telecom customers at high risk of cancellation to enable targeted retention campaigns.

**Intended users:** Data and business teams running proactive churn prevention workflows.

**Out-of-scope uses:**
- Predicting churn in non-telecom industries without retraining
- Real-time scoring at high volume without latency evaluation
- Automated cancellation or penalization of customers based solely on model output

---

## Training Data

- **Dataset:** [Telco Customer Churn — IBM Sample](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,043 customers, 19 features
- **Target:** `Churn` (Yes/No) → binary (1/0)
- **Class distribution:** ~73.5% No Churn / ~26.5% Churn
- **Preprocessing:** Stratified train/validation/test split; standard scaling for numeric features; one-hot encoding for categorical features; business-aware imputation of `TotalCharges` (filled with 0 for customers with `tenure = 0`)

---

## Evaluation Results

Evaluated on a held-out test set with stratified split.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| DummyClassifier | 0.7346 | 0.0000 | 0.0000 | 0.0000 | NaN | NaN |
| **Logistic Regression** | **0.8055** | **0.6572** | **0.5588** | **0.6040** | **0.8421** | **0.6343** |
| MLP (PyTorch) | 0.7913 | 0.6087 | 0.5989 | 0.6038 | 0.8423 | 0.6341 |

**Primary metrics:** ROC-AUC and PR-AUC, chosen due to class imbalance (~26.5% positive rate).

---

## Serving Decision

The Logistic Regression was chosen for the production API over the MLP for the following reasons:

- **Competitive performance:** ROC-AUC 0.8421 vs 0.8423 — a difference of 0.0002, operationally negligible
- **Higher precision:** 0.6572 vs 0.6087 — fewer false positives, meaning fewer customers incorrectly targeted for retention campaigns
- **Interpretability:** Logistic Regression coefficients directly expose feature importance, enabling business explanation of individual predictions
- **Operational simplicity:** No deep learning dependencies, faster inference, easier maintenance for a v1 production system

> The MLP is retained in `notebooks/02_mlp_pytorch.ipynb` as documented experimental evidence.

---

## Limitations

- **Class imbalance:** The dataset has ~26.5% churn rate. The model has lower recall (0.5588) — it misses roughly 44% of actual churners. This is a known trade-off favoring precision.
- **Recall gap:** In high-cost churn scenarios (where missing a churner is expensive), a lower threshold or a recall-optimized model may be preferable.
- **Domain specificity:** Trained exclusively on one telecom provider's data. Performance may degrade on different customer bases or contract structures without retraining.
- **Static model:** No drift detection implemented. Model performance may degrade over time as customer behavior evolves.
- **No causal inference:** The model identifies correlation, not causation. A customer flagged as high-risk may not churn due to the features the model weighted — external factors are not captured.

---

## Bias & Fairness Considerations

- The dataset includes `gender` and `SeniorCitizen` as features. These were retained for predictive performance but introduce potential demographic bias.
- No fairness audit was performed across demographic groups. Before deploying in a real business context, a fairness evaluation (e.g., equal opportunity, demographic parity) across `gender` and `SeniorCitizen` subgroups is recommended.
- Retention campaign decisions based on model output should be reviewed by a human before execution.

---

## Failure Scenarios

| Scenario | Expected behavior | Risk level |
|---|---|---|
| Input with missing features | Pydantic validation error (422) | Low — handled by API |
| Customer profile outside training distribution | Prediction may be unreliable | Medium |
| Data drift over time (new contract types, pricing) | Silent performance degradation | High — requires monitoring |
| Extreme class imbalance shift | ROC-AUC may remain high while recall collapses | High — monitor PR-AUC separately |

---

## Monitoring Recommendations

- Track **PR-AUC** and **Recall** monthly on new labeled data — these are the first metrics to degrade with drift
- Monitor **input feature distributions** for shift (especially `MonthlyCharges`, `tenure`, `Contract`)
- Set alert threshold: retrain if ROC-AUC drops below 0.80 on a recent validation window
- Log all predictions with timestamps for audit trail

---

## Model Details

| Field | Value |
|---|---|
| Algorithm | Logistic Regression (`sklearn.linear_model.LogisticRegression`) |
| Regularization | L2 (default) |
| Solver | lbfgs |
| Preprocessing | `sklearn.pipeline.Pipeline` with `StandardScaler` + `OneHotEncoder` |
| Serialization | `joblib` |
| Artifact path | `models/churn_logreg_pipeline.joblib` |
| Seeds | Fixed (`random_state=42`) for reproducibility |

---

*Model Card authored by Gabriel Furtado. Last updated: May 2026.*
