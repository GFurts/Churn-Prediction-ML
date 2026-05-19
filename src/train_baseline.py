import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import MODEL_PATH, MODELS_DIR, RANDOM_STATE, TEST_SIZE
from src.data import clean_data, get_features_and_target, load_raw_data
from src.features import build_preprocessor


def train_and_save_model() -> None:
    """Treina a Regressão Logística e salva o pipeline em disco."""
    df = load_raw_data()
    df = clean_data(df)

    X, y = get_features_and_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = build_preprocessor(X_train)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)),
        ]
    )

    pipeline.fit(X_train, y_train)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    print(f"Modelo salvo em: {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save_model()