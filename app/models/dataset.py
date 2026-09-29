from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.orm import relationship
from app.core.database import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), index=True, nullable=False)
    description = Column(String(500), nullable=True)
    file_path = Column(String(255), nullable=False)
    file_hash = Column(String(64), nullable=True)
    row_count = Column(Integer, default=0)
    col_count = Column(Integer, default=0)
    target_column = Column(String(100), nullable=True)
    class_distribution = Column(JSON, nullable=True)
    missing_ratio = Column(Float, default=0.0)
    duplicate_count = Column(Integer, default=0)
    numeric_features_count = Column(Integer, default=0)
    categorical_features_count = Column(Integer, default=0)
    feature_names = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    experiments = relationship("Experiment", back_populates="dataset", cascade="all, delete-orphan")
