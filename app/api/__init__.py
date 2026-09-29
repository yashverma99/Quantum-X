from fastapi import APIRouter
from app.api.health import router as health_router
from app.api.datasets import router as datasets_router
from app.api.preprocessing import router as preprocessing_router
from app.api.features import router as features_router
from app.api.classical import router as classical_router
from app.api.quantum import router as quantum_router
from app.api.models import router as models_router
from app.api.benchmark import router as benchmark_router
from app.api.experiments import router as experiments_router
from app.api.explainability import router as explainability_router
from app.api.system import router as system_router
from app.api.conversation import router as conversation_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(datasets_router)
api_router.include_router(preprocessing_router)
api_router.include_router(features_router)
api_router.include_router(classical_router)
api_router.include_router(quantum_router)
api_router.include_router(models_router)
api_router.include_router(benchmark_router)
api_router.include_router(experiments_router)
api_router.include_router(explainability_router)
api_router.include_router(system_router)
api_router.include_router(conversation_router)
