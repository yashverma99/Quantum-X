from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from app.core.database import Base


class BenchmarkResult(Base):
    __tablename__ = "benchmarks"

    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=False)
    quantum_utility_status = Column(String(100), nullable=False)  # "Classical ML Superior" | "QML Superior" | "Comparable"
    best_classical_model = Column(String(100), nullable=True)
    best_quantum_model = Column(String(100), nullable=True)
    balanced_accuracy_diff = Column(Float, nullable=True)
    f1_diff = Column(Float, nullable=True)
    roc_auc_diff = Column(Float, nullable=True)
    latency_ratio = Column(Float, nullable=True)
    comparison_table = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    experiment = relationship("Experiment", back_populates="benchmarks")
