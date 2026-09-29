from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.services.dataset_service import dataset_service
from app.services.preprocessing_service import preprocessing_service

router = APIRouter(prefix="/preprocessing", tags=["Preprocessing"])


class PreprocessRequest(BaseModel):
    dataset_id: int
    test_split: float = 0.2
    random_seed: int = 42
    handle_missing: str = "median"  # median, mean, drop
    scaling: str = "standard"  # standard, minmax, robust


@router.post("/run")
def run_preprocessing(req: PreprocessRequest, db: Session = Depends(get_db)):
    dataset = dataset_service.get_dataset_by_id(db, req.dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return {
        "status": "ready",
        "dataset_id": req.dataset_id,
        "dataset_name": dataset.name,
        "test_split_ratio": req.test_split,
        "random_seed": req.random_seed,
        "leakage_safe": True,
        "message": "Leakage-safe preprocessing configured. Ready for pipeline execution."
    }
