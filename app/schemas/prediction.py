from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, ConfigDict


class PredictionRequest(BaseModel):
    model_id: int
    features: Dict[str, float]
    sample_id: Optional[str] = None


class PredictionResponse(BaseModel):
    id: int
    model_id: int
    sample_id: Optional[str] = None
    predicted_risk_label: str  # "HIGH_RISK" | "LOW_RISK"
    model_probability: float  # e.g., 0.87
    model_confidence_score: float  # e.g., 0.85
    data_quality_score: float  # e.g., 0.94
    ood_status: str  # "LOW" | "MODERATE" | "HIGH"
    human_review_status: str  # "PENDING" | "REVIEWED" | "OVERRIDDEN"
    disclaimer: str = "Research decision-support prototype. Not a certified clinical diagnosis."
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class HumanReviewUpdate(BaseModel):
    human_review_status: str  # "REVIEWED" | "OVERRIDDEN"
    clinician_notes: str
