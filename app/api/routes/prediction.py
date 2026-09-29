from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.prediction import Prediction
from app.schemas.prediction import PredictionRequest, PredictionResponse, HumanReviewUpdate

router = APIRouter(prefix="/prediction", tags=["Predictions & Risk Screening"])


@router.post("", response_model=PredictionResponse)
def generate_prediction(req: PredictionRequest, db: Session = Depends(get_db)):
    pred = Prediction(
        model_id=req.model_id,
        input_sample_id=req.sample_id,
        input_features=req.features,
        predicted_risk_label="HIGH_RISK",
        model_probability=0.87,
        model_confidence_score=0.82,
        data_quality_score=0.94,
        ood_status="LOW",
        human_review_status="PENDING",
    )
    db.add(pred)
    db.commit()
    db.refresh(pred)
    return pred


@router.post("/{prediction_id}/review", response_model=PredictionResponse)
def submit_human_review(
    prediction_id: int,
    review: HumanReviewUpdate,
    db: Session = Depends(get_db)
):
    pred = db.query(Prediction).filter(Prediction.id == prediction_id).first()
    if not pred:
        raise HTTPException(status_code=404, detail="Prediction record not found")

    pred.human_review_status = review.human_review_status
    pred.clinician_notes = review.clinician_notes
    db.commit()
    db.refresh(pred)
    return pred
