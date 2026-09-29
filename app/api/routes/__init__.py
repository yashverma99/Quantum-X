from fastapi import APIRouter
from app.api.routes.health import router as health_router
from app.api.routes.datasets import router as datasets_router
from app.api.routes.preprocessing import router as preprocessing_router
from app.api.routes.features import router as features_router
from app.api.routes.models import router as models_router
from app.api.routes.benchmark import router as benchmark_router
from app.api.routes.explainability import router as explainability_router
from app.api.routes.experiments import router as experiments_router
from app.api.routes.prediction import router as prediction_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(datasets_router)
api_router.include_router(preprocessing_router)
api_router.include_router(features_router)
api_router.include_router(models_router)
api_router.include_router(benchmark_router)
api_router.include_router(explainability_router)
api_router.include_router(experiments_router)
api_router.include_router(prediction_router)
