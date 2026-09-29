from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.services.dataset_service import dataset_service

router = APIRouter(prefix="/features", tags=["Feature Engineering"])


class FeatureEngineeringRequest(BaseModel):
    dataset_id: int
    method: str = "PCA"  # Variance, Mutual Info, PCA, LASSO
    n_features: int = 4  # 2, 4, 8, 16


@router.post("/run")
def run_feature_engineering(req: FeatureEngineeringRequest, db: Session = Depends(get_db)):
    dataset = dataset_service.get_dataset_by_id(db, req.dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return {
        "status": "ready",
        "dataset_id": req.dataset_id,
        "original_features": dataset.col_count,
        "selected_features_count": req.n_features,
        "method": req.method,
        "quantum_ready": True
    }
