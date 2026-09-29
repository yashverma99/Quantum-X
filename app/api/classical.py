from typing import List, Dict, Any, Optional, Union
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.services.classical_ml_service import classical_ml_service
from app.api.models import train_classical_models, ClassicalTrainRequest

router = APIRouter(prefix="/classical", tags=["Classical Models"])


@router.get("/models")
def get_classical_models():
    """Returns supported models and their live environment status."""
    return classical_ml_service.get_supported_models()


@router.post("/train", status_code=status.HTTP_200_OK)
def train_classical_models_alias(req: ClassicalTrainRequest, db: Session = Depends(get_db)):
    """Alias for POST /api/models/classical/train."""
    return train_classical_models(req, db)
