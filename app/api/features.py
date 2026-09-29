from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.dataset import Dataset
from app.models.experiment import Experiment
from app.services.dataset_service import dataset_service
from app.services.preprocessing_service import preprocessing_service
from app.services.feature_service import feature_service

router = APIRouter(prefix="/features", tags=["Feature Engineering"])


class FeatureRunRequest(BaseModel):
    dataset_id: int
    target_column: str
    variance_threshold: float = Field(default=0.0, ge=0.0)
    correlation_threshold: Optional[float] = Field(default=None, ge=0.5, le=1.0)
    pca_components: int = Field(default=4, ge=1, le=64)
    random_seed: int = Field(default=42, ge=0)


@router.post("/run", status_code=status.HTTP_200_OK)
def run_features(req: FeatureRunRequest, db: Session = Depends(get_db)):
    d = dataset_service.get_dataset_by_id(db, req.dataset_id)
    if not d:
        raise HTTPException(status_code=404, detail=f"Dataset with ID {req.dataset_id} not found.")

    try:
        df = dataset_service.load_dataframe(d)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load dataset: {str(e)}")

    if req.target_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{req.target_column}' does not exist in dataset."
        )

    # Preprocess
    preprocessed = preprocessing_service.run_pipeline(
        df=df,
        target_col=req.target_column,
        test_size=0.2,
        random_state=req.random_seed,
        missing_strategy="median",
        scaler_type="standard"
    )

    X_train = preprocessed["_X_train"]
    X_test = preprocessed["_X_test"]
    orig_features = preprocessed["feature_names"]

    # Variance filtering
    X_train_var, X_test_var, retained_var_names, count_after_var = feature_service.apply_variance_threshold(
        X_train=X_train,
        X_test=X_test,
        feature_names=orig_features,
        threshold=req.variance_threshold
    )

    # Correlation filtering
    count_after_corr = count_after_var
    retained_corr_names = retained_var_names
    X_train_filtered = X_train_var
    X_test_filtered = X_test_var

    if req.correlation_threshold is not None:
        X_train_filtered, X_test_filtered, retained_corr_names, count_after_corr = feature_service.apply_correlation_filter(
            X_train=X_train_var,
            X_test=X_test_var,
            feature_names=retained_var_names,
            threshold=req.correlation_threshold
        )

    # PCA reduction
    X_train_pca, X_test_pca, explained_ratios, cumulative_var = feature_service.apply_pca(
        X_train=X_train_filtered,
        X_test=X_test_filtered,
        n_components=req.pca_components
    )

    # Quantum mapping
    X_train_q, X_test_q, sample_vector, encoding_type = feature_service.map_to_quantum_state(
        X_train_pca=X_train_pca,
        X_test_pca=X_test_pca
    )

    # Create Experiment
    exp_count = db.query(Experiment).count()
    exp_code = f"EXP-{exp_count + 1:03d}"

    exp = Experiment(
        experiment_code=exp_code,
        name=f"Feature Pipeline ({d.name}) - PCA-{req.pca_components}",
        dataset_id=d.id,
        status="PREPROCESSED",
        feature_selection_method=f"VarianceThreshold({req.variance_threshold}) + PCA({req.pca_components})",
        original_features_count=len(orig_features),
        selected_features_count=req.pca_components,
        random_seed=req.random_seed,
        test_split_ratio=0.2,
        config={
            "dataset_id": d.id,
            "target_column": req.target_column,
            "variance_threshold": req.variance_threshold,
            "correlation_threshold": req.correlation_threshold,
            "pca_components": req.pca_components,
            "original_features_count": len(orig_features),
            "features_after_variance": count_after_var,
            "features_after_correlation": count_after_corr,
            "pca_feature_count": req.pca_components,
            "explained_variance_ratio": explained_ratios,
            "cumulative_explained_variance": cumulative_var,
            "sample_quantum_vector": sample_vector,
            "encoding": encoding_type,
            "retained_feature_names": retained_corr_names[:20]
        }
    )
    db.add(exp)
    db.commit()
    db.refresh(exp)

    return {
        "experiment_id": exp.id,
        "experiment_code": exp.experiment_code,
        "original_features_count": len(orig_features),
        "filtered_features_count": count_after_corr,
        "pca_feature_count": req.pca_components,
        "explained_variance_ratio": explained_ratios,
        "cumulative_explained_variance": cumulative_var,
        "quantum_ready_dimensions": req.pca_components,
        "quantum_encoding": encoding_type,
        "sample_quantum_vector": sample_vector,
        "retained_feature_names": retained_corr_names[:30]
    }


@router.get("/{experiment_id}")
def get_features_for_experiment(experiment_id: int, db: Session = Depends(get_db)):
    exp = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment with ID {experiment_id} not found.")

    cfg = exp.config or {}
    return {
        "experiment_id": exp.id,
        "experiment_code": exp.experiment_code,
        "original_features_count": exp.original_features_count,
        "filtered_features_count": cfg.get("features_after_correlation", exp.original_features_count),
        "pca_feature_count": exp.selected_features_count,
        "explained_variance_ratio": cfg.get("explained_variance_ratio", []),
        "cumulative_explained_variance": cfg.get("cumulative_explained_variance", 0.0),
        "sample_quantum_vector": cfg.get("sample_quantum_vector", []),
        "quantum_encoding": cfg.get("encoding", "Angle Encoding [0, π]"),
        "retained_feature_names": cfg.get("retained_feature_names", [])
    }
