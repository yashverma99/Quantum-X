from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from app.core.database import get_db

router = APIRouter(prefix="/models", tags=["Models"])


class ClassicalTrainRequest(BaseModel):
    experiment_id: int
    models: List[str] = ["Logistic Regression", "Random Forest", "RBF-SVM"]


class QuantumTrainRequest(BaseModel):
    experiment_id: int
    model_type: str = "VQC"  # VQC, QSVM
    qubits: int = 4
    circuit_depth: int = 2
    encoding: str = "angle_encoding"
    shots: int = 1024
    noise_level: str = "OFF"


@router.post("/classical/train")
def train_classical_models(req: ClassicalTrainRequest, db: Session = Depends(get_db)):
    return {
        "status": "ready",
        "experiment_id": req.experiment_id,
        "models_scheduled": req.models
    }


@router.post("/quantum/train")
def train_quantum_model(req: QuantumTrainRequest, db: Session = Depends(get_db)):
    return {
        "status": "ready",
        "experiment_id": req.experiment_id,
        "quantum_model": req.model_type,
        "qubits": req.qubits,
        "circuit_depth": req.circuit_depth,
        "backend": "aer_simulator"
    }
