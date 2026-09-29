from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.experiment import Experiment
from app.schemas.experiment import ExperimentConfig, ExperimentResponse

router = APIRouter(prefix="/experiments", tags=["Experiments"])


@router.get("", response_model=List[ExperimentResponse])
def get_experiments(db: Session = Depends(get_db)):
    return db.query(Experiment).order_by(Experiment.created_at.desc()).all()


@router.post("", response_model=ExperimentResponse, status_code=status.HTTP_201_CREATED)
def create_experiment(config: ExperimentConfig, db: Session = Depends(get_db)):
    exp_count = db.query(Experiment).count()
    exp_code = f"EXP-{exp_count + 1:03d}"

    exp = Experiment(
        experiment_code=exp_code,
        name=config.name,
        dataset_id=config.dataset_id,
        feature_selection_method=config.feature_selection_method,
        selected_features_count=config.selected_features_count,
        random_seed=config.random_seed,
        test_split_ratio=config.test_split_ratio,
        config=config.model_dump(),
        status="created"
    )
    db.add(exp)
    db.commit()
    db.refresh(exp)
    return exp


@router.get("/{experiment_id}", response_model=ExperimentResponse)
def get_experiment(experiment_id: int, db: Session = Depends(get_db)):
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return exp
