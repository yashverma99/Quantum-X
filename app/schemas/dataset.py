from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, ConfigDict


class DatasetBase(BaseModel):
    name: str
    description: Optional[str] = None
    target_column: Optional[str] = None


class DatasetCreate(DatasetBase):
    pass


class DatasetResponse(DatasetBase):
    id: int
    file_path: str
    file_hash: Optional[str] = None
    row_count: int
    col_count: int
    class_distribution: Optional[Dict[str, int]] = None
    missing_ratio: float
    duplicate_count: int
    numeric_features_count: int
    categorical_features_count: int
    feature_names: Optional[List[str]] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DataQualityReport(BaseModel):
    row_count: int
    feature_count: int
    missing_percentage: float
    duplicate_rows: int
    class_imbalance: str
    leakage_safe: bool = True
