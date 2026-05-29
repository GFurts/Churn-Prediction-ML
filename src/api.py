import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from src.logger import get_logger
from src.predict import predict
from src.schemas import ChurnInput

logger = get_logger(__name__)

app = FastAPI(title="Churn Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    response.headers["X-Process-Time"] = str(process_time)
    logger.info(
        f"{request.method} {request.url.path} completed in {process_time:.4f}s"
    )

    return response


@app.get("/health")
def health():
    logger.info("Health check endpoint called")
    return {"status": "ok"}


@app.post("/predict")
def predict_churn(payload: ChurnInput):
    logger.info("Prediction endpoint called")
    result = predict(payload.model_dump())
    logger.info(
        f"Prediction completed | prediction={result['prediction']} | "
        f"probability={result['churn_probability']:.4f}"
    )
    return result