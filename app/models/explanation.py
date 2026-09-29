from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from app.core.database import Base


class ExplanationRecord(Base):
    __tablename__ = "explanations"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False, unique=True)
    method = Column(String(100), default="permutation_importance")
    feature_contributions = Column(JSON, nullable=False)
    top_features = Column(JSON, nullable=False)
    interpretation_notes = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    model = relationship("ModelRecord", back_populates="explanation")
