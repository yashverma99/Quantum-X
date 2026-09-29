from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from app.core.database import Base


class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, index=True)
    experiment_code = Column(String(50), unique=True, index=True, nullable=False)
    name = Column(String(150), nullable=False)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    status = Column(String(50), default="pending")  # pending, running, completed, failed
    feature_selection_method = Column(String(100), default="PCA")
    original_features_count = Column(Integer, default=0)
    selected_features_count = Column(Integer, default=4)
    random_seed = Column(Integer, default=42)
    test_split_ratio = Column(Float, default=0.2)
    config = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    dataset = relationship("Dataset", back_populates="experiments")
    creator = relationship("User", back_populates="experiments")
    models = relationship("ModelRecord", back_populates="experiment", cascade="all, delete-orphan")
    benchmarks = relationship("BenchmarkResult", back_populates="experiment", cascade="all, delete-orphan")
