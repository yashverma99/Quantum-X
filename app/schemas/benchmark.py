from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, ConfigDict


class ModelMetricSummary(BaseModel):
    model_name: str
    model_type: str  # classical | quantum
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    specificity: float
    f1_score: float
    roc_auc: float
    training_time_seconds: float
    inference_time_seconds: float


class BenchmarkResponse(BaseModel):
    id: int
    experiment_id: int
    quantum_utility_status: str  # "Classical ML Superior" | "QML Superior" | "Comparable"
    best_classical_model: Optional[str] = None
    best_quantum_model: Optional[str] = None
    balanced_accuracy_diff: Optional[float] = None
    f1_diff: Optional[float] = None
    roc_auc_diff: Optional[float] = None
    latency_ratio: Optional[float] = None
    models: List[ModelMetricSummary] = []
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
