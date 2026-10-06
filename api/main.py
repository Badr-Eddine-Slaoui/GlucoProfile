import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PatientFeatures,
    PredictionResponse,
)
from .utils import FEATURES, ModelNotAvailableError, ModelService, get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("api")

settings = get_settings()
model_service = ModelService(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model_service.load()
    except ModelNotAvailableError as error:
        logger.warning("Démarrage sans modèle : %s", error)
    yield


app = FastAPI(
    title="GlucoProfile API",
    description="Estimation du risque de diabète à partir de constantes cliniques.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ModelNotAvailableError)
async def model_unavailable_handler(_: Request, exc: ModelNotAvailableError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


def require_admin_key(x_api_key: str | None = Header(default=None)) -> None:
    if settings.admin_api_key and x_api_key != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="Clé API invalide ou manquante.")

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health():
    loaded = model_service._loaded
    return HealthResponse(
        model_loaded=loaded is not None,
        model_name=loaded.name if loaded else None,
        model_version=loaded.version if loaded else None,
    )


@app.get("/ready", response_model=HealthResponse, tags=["Monitoring"],
        responses={503: {"model": ErrorResponse}})
def ready():
    loaded = model_service.loaded
    return HealthResponse(model_loaded=True, model_name=loaded.name, model_version=loaded.version)


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Modèle"],
        responses={503: {"model": ErrorResponse}})
def model_info():
    m = model_service.loaded
    return ModelInfoResponse(
        model_name=m.name,
        model_version=m.version,
        stage=m.stage,
        model_uri=m.uri,
        run_id=m.run_id,
        features=FEATURES,
        cluster_mapping={str(k): v for k, v in m.cluster_mapping.items()},
        loaded_at=m.loaded_at,
        tracking_uri=settings.tracking_uri,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inférence"],
        responses={503: {"model": ErrorResponse}})
def predict(patient: PatientFeatures):
    try:
        return model_service.predict([patient])[0]
    except ModelNotAvailableError:
        raise
    except Exception as error:
        logger.exception("Erreur d'inférence")
        raise HTTPException(status_code=500, detail=f"Erreur d'inférence : {error}") from error


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inférence"],
        responses={503: {"model": ErrorResponse}})
def predict_batch(request: BatchPredictionRequest):
    try:
        predictions = model_service.predict(request.patients)
    except ModelNotAvailableError:
        raise
    except Exception as error:
        logger.exception("Erreur d'inférence batch")
        raise HTTPException(status_code=500, detail=f"Erreur d'inférence : {error}") from error
    return BatchPredictionResponse(count=len(predictions), predictions=predictions)


@app.post("/model/reload", response_model=ModelInfoResponse, tags=["Modèle"],
        dependencies=[Depends(require_admin_key)],
        responses={503: {"model": ErrorResponse}})
def reload_model():
    model_service.load()
    return model_info()