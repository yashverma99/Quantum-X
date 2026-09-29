from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, ConfigDict


class ExperimentConfig(BaseModel):
    dataset_id: int
    name: str
    feature_selection_method: str = "PCA"  # Variance, Mutual Info, PCA, LASSO
    selected_features_count: int = 4  # 4, 8, 16, 32
    random_seed: int = 42
    test_split_ratio: float = 0.2
    qubits: int = 4
    circuit_depth: int = 2
    encoding: str = "angle_encoding"
    noise_level: str = "OFF"  # OFF, LOW, MEDIUM, HIGH
    notes: Optional[str] = None


class ExperimentResponse(BaseModel):
    id: int
    experiment_code: str
    name: str
    dataset_id: int
    status: str
    feature_selection_method: str
    original_features_count: int
    selected_features_count: int
    random_seed: int
    test_split_ratio: float
    config: Optional[Dict[str, Any]] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
