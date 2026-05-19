import pandas as pd

from src.config import DATASET_PATH


def load_raw_data() -> pd.DataFrame:
    """Carrega o dataset bruto a partir do caminho configurado."""
    return pd.read_csv(DATASET_PATH)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica limpeza básica e tratamento inicial dos dados."""
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    return df


def get_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa variáveis explicativas e variável alvo."""
    X = df.drop(columns=["Churn"])
    y = (df["Churn"] == "Yes").astype(int)
    return X, y