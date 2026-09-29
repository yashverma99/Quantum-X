from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.database import get_db

router = APIRouter(prefix="/explainability", tags=["Explainability"])


@router.get("/{model_id}")
def get_model_explanation(model_id: int, db: Session = Depends(get_db)):
    return {
        "model_id": model_id,
        "method": "Permutation Feature Importance & Quantum Parameter Sensitivity",
        "top_features": [],
        "note": "Research transparency decision-support signal. Not biological causation."
    }
