from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from app.core.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False)
    input_sample_id = Column(String(100), nullable=True)
    input_features = Column(JSON, nullable=False)

    # Risk screening signal
    predicted_risk_label = Column(String(50), nullable=False)  # "HIGH_RISK" | "LOW_RISK"
    model_probability = Column(Float, nullable=False)

    # Confidence & Uncertainty Decomposition
    model_confidence_score = Column(Float, nullable=False)
    data_quality_score = Column(Float, default=1.0)
    ood_status = Column(String(50), default="LOW")  # LOW, MODERATE, HIGH

    # Human-in-the-loop review
    human_review_status = Column(String(50), default="PENDING")  # PENDING, REVIEWED, OVERRIDDEN
    clinician_notes = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    model = relationship("ModelRecord", back_populates="predictions")
