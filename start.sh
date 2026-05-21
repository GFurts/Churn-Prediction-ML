python -m src.train_baseline
uvicorn src.api:app --host 0.0.0.0 --port $PORT