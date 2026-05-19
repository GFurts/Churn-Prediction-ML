import joblib
import pandas as pd

from src.config import MODEL_PATH


def load_model():
    """Carrega o pipeline treinado salvo em disco."""
    return joblib.load(MODEL_PATH)


def predict(data: dict) -> dict:
    """Realiza predição a partir de um dicionário com os dados de entrada."""
    model = load_model()

    input_df = pd.DataFrame([data])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return {
        "prediction": int(prediction),
        "churn_probability": float(probability),
    }