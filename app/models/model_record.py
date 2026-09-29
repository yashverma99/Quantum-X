from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from app.core.database import Base


class ModelRecord(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=False)
    model_name = Column(String(100), nullable=False)
    model_type = Column(String(50), nullable=False)  # "classical" | "quantum"
    hyperparameters = Column(JSON, nullable=True)

    # Core Metrics
    accuracy = Column(Float, nullable=True)
    balanced_accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    sensitivity = Column(Float, nullable=True)
    specificity = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    f1 = Column(Float, nullable=True)
    macro_f1 = Column(Float, nullable=True)
    roc_auc = Column(Float, nullable=True)

    # Performance timings
    training_time_seconds = Column(Float, nullable=True)
    inference_time_seconds = Column(Float, nullable=True)
    training_time_ms = Column(Float, nullable=True)
    inference_time_ms = Column(Float, nullable=True)

    # Visualization structures
    confusion_matrix = Column(JSON, nullable=True)
    roc_curve_data = Column(JSON, nullable=True)
    artifact_path = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    experiment = relationship("Experiment", back_populates="models")
    quantum_run = relationship("QuantumRun", back_populates="model", uselist=False, cascade="all, delete-orphan")
    explanation = relationship("ExplanationRecord", back_populates="model", uselist=False, cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="model", cascade="all, delete-orphan")


ModelResult = ModelRecord

