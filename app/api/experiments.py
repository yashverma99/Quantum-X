from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.experiment import Experiment
from app.models.dataset import Dataset
from app.schemas.experiment import ExperimentConfig, ExperimentResponse

router = APIRouter(prefix="/experiments", tags=["Experiments"])


@router.get("")
def get_experiments(db: Session = Depends(get_db)):
    experiments = db.query(Experiment).order_by(Experiment.created_at.desc()).all()
    results = []
    for exp in experiments:
        results.append({
            "id": exp.id,
            "experiment_code": exp.experiment_code,
            "name": exp.name,
            "dataset_id": exp.dataset_id,
            "status": exp.status,
            "feature_selection_method": exp.feature_selection_method,
            "original_features_count": exp.original_features_count,
            "selected_features_count": exp.selected_features_count,
            "random_seed": exp.random_seed,
            "test_split_ratio": exp.test_split_ratio,
            "config": exp.config,
            "created_at": exp.created_at.isoformat() if exp.created_at else None
        })
    return {"experiments": results}


@router.post("", status_code=status.HTTP_201_CREATED)
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
        status="Awaiting Run"
    )
    db.add(exp)
    db.commit()
    db.refresh(exp)
    return {
        "id": exp.id,
        "experiment_code": exp.experiment_code,
        "name": exp.name,
        "status": exp.status,
        "created_at": exp.created_at.isoformat() if exp.created_at else None
    }


@router.get("/{experiment_id}")
def get_experiment(experiment_id: str, db: Session = Depends(get_db)):
    if experiment_id.isdigit():
        exp = db.query(Experiment).filter(Experiment.id == int(experiment_id)).first()
    else:
        exp = db.query(Experiment).filter(Experiment.experiment_code == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return {
        "id": exp.id,
        "experiment_code": exp.experiment_code,
        "name": exp.name,
        "dataset_id": exp.dataset_id,
        "status": exp.status,
        "feature_selection_method": exp.feature_selection_method,
        "original_features_count": exp.original_features_count,
        "selected_features_count": exp.selected_features_count,
        "random_seed": exp.random_seed,
        "test_split_ratio": exp.test_split_ratio,
        "config": exp.config,
        "created_at": exp.created_at.isoformat() if exp.created_at else None
    }


@router.get("/{experiment_id}/preprocessing")
def get_experiment_preprocessing(experiment_id: str, db: Session = Depends(get_db)):
    if experiment_id.isdigit():
        exp = db.query(Experiment).filter(Experiment.id == int(experiment_id)).first()
    else:
        exp = db.query(Experiment).filter(Experiment.experiment_code == experiment_id).first()
        
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found")
        
    ds = db.query(Dataset).filter(Dataset.id == exp.dataset_id).first()
    cfg = exp.config or {}
    
    # Dataset information
    ds_name = ds.name if ds else cfg.get("dataset_name", "Unknown Dataset")
    ds_rows = ds.row_count if ds else (cfg.get("train_samples", 0) + cfg.get("test_samples", 0))
    ds_features = (ds.col_count - 1) if ds and ds.col_count > 0 else exp.original_features_count
    
    # Target & classes
    target_col = cfg.get("target_column") or (ds.target_column if ds else "diagnosis")
    classes = []
    if cfg.get("classes"):
        if isinstance(cfg["classes"], dict):
            classes = list(cfg["classes"].keys())
        elif isinstance(cfg["classes"], list):
            classes = cfg["classes"]
    elif ds and ds.class_distribution:
        classes = list(ds.class_distribution.keys())
        
    # Split
    test_ratio = exp.test_split_ratio if exp.test_split_ratio else 0.2
    train_rows = cfg.get("train_samples") or int(round(ds_rows * (1.0 - test_ratio)))
    test_rows = cfg.get("test_samples") or (ds_rows - train_rows)
    
    # Features
    orig_feat = exp.original_features_count or cfg.get("original_features_count", ds_features)
    var_feat = cfg.get("features_after_variance") or cfg.get("features_after_variance_filter", orig_feat)
    pca_feat = exp.selected_features_count or cfg.get("pca_components") or cfg.get("pca_feature_count", 4)
    
    # PCA
    exp_var = cfg.get("explained_variance_ratio", [])
    cum_var = cfg.get("cumulative_explained_variance", 0.0)
    
    return {
        "experiment_id": exp.experiment_code,
        "status": "PREPROCESSED" if exp.status in ["PREPROCESSED", "COMPLETED", "READY"] else exp.status,
        "dataset": {
            "name": ds_name,
            "rows": ds_rows,
            "features": orig_feat
        },
        "target": {
            "column": target_col,
            "classes": classes
        },
        "split": {
            "train_rows": train_rows,
            "test_rows": test_rows
        },
        "features": {
            "original": orig_feat,
            "after_variance_filter": var_feat,
            "after_pca": pca_feat
        },
        "pca": {
            "components": pca_feat,
            "explained_variance": exp_var,
            "cumulative_explained_variance": cum_var
        }
    }

