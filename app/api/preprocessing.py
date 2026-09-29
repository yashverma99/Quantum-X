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

router = APIRouter(prefix="/preprocessing", tags=["Preprocessing"])


class PreprocessRunRequest(BaseModel):
    dataset_id: int
    target_column: str
    test_size: float = Field(default=0.2, ge=0.05, le=0.5)
    random_seed: int = Field(default=42, ge=0)
    missing_strategy: str = Field(default="median")  # mean, median, most_frequent
    scaler: str = Field(default="standard")  # standard, minmax
    variance_threshold: float = Field(default=0.0, ge=0.0)
    correlation_threshold: Optional[float] = Field(default=None, ge=0.5, le=1.0)
    pca_components: int = Field(default=4, ge=1, le=64)


@router.post("/run", status_code=status.HTTP_200_OK)
def run_preprocessing(req: PreprocessRunRequest, db: Session = Depends(get_db)):
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

    # 1. Leakage-safe Preprocessing (split first, fit on train only)
    try:
        preprocessed = preprocessing_service.run_pipeline(
            df=df,
            target_col=req.target_column,
            test_size=req.test_size,
            random_state=req.random_seed,
            missing_strategy=req.missing_strategy,
            scaler_type=req.scaler
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {str(e)}")

    X_train = preprocessed["_X_train"]
    X_test = preprocessed["_X_test"]
    feature_names = preprocessed["feature_names"]
    orig_features_count = len(feature_names)

    # 2. Variance Threshold Filtering (fit on train only)
    X_train_var, X_test_var, retained_var_names, count_after_var = feature_service.apply_variance_threshold(
        X_train=X_train,
        X_test=X_test,
        feature_names=feature_names,
        threshold=req.variance_threshold
    )

    # 3. Optional Correlation Filtering (computed on train only)
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

    # 4. PCA Dimensionality Reduction (fit on train only)
    X_train_pca, X_test_pca, explained_ratios, cumulative_var = feature_service.apply_pca(
        X_train=X_train_filtered,
        X_test=X_test_filtered,
        n_components=req.pca_components
    )

    # 5. Quantum-Ready Representation Mapping
    X_train_q, X_test_q, sample_quantum_vector, encoding_type = feature_service.map_to_quantum_state(
        X_train_pca=X_train_pca,
        X_test_pca=X_test_pca
    )

    # 6. Store Experiment Record in Database
    exp_count = db.query(Experiment).count()
    exp_code = f"EXP-{exp_count + 1:03d}"

    exp_config = {
        "dataset_id": req.dataset_id,
        "dataset_name": d.name,
        "target_column": req.target_column,
        "test_size": req.test_size,
        "random_seed": req.random_seed,
        "missing_strategy": req.missing_strategy,
        "scaler": req.scaler,
        "variance_threshold": req.variance_threshold,
        "correlation_threshold": req.correlation_threshold,
        "pca_components": req.pca_components,
        "explained_variance_ratios": explained_ratios,
        "cumulative_explained_variance": cumulative_var,
        "sample_quantum_vector": sample_quantum_vector,
        "quantum_encoding": encoding_type,
    }

    exp = Experiment(
        experiment_code=exp_code,
        name=f"Pipeline Run ({d.name}) - PCA-{req.pca_components}",
        dataset_id=d.id,
        status="PREPROCESSED",
        feature_selection_method=f"VarianceThreshold({req.variance_threshold}) + PCA({req.pca_components})",
        original_features_count=orig_features_count,
        selected_features_count=req.pca_components,
        random_seed=req.random_seed,
        test_split_ratio=req.test_size,
        config=exp_config
    )
    db.add(exp)
    db.commit()
    db.refresh(exp)

    return {
        "status": "PREPROCESSED",
        "experiment_id": exp.id,
        "experiment_code": exp.experiment_code,
        "dataset_id": d.id,
        "dataset_name": d.name,
        "target_column": req.target_column,
        "leakage_safe": True,
        "random_seed": req.random_seed,
        "train_samples": preprocessed["train_samples"],
        "test_samples": preprocessed["test_samples"],
        "classes": preprocessed["classes"],
        "train_class_distribution": preprocessed["train_class_distribution"],
        "test_class_distribution": preprocessed["test_class_distribution"],
        "original_features_count": orig_features_count,
        "features_after_variance_filter": count_after_var,
        "features_after_correlation_filter": count_after_corr,
        "pca_components": req.pca_components,
        "explained_variance_ratio": explained_ratios,
        "cumulative_explained_variance": cumulative_var,
        "quantum_ready_dimensions": req.pca_components,
        "quantum_encoding": encoding_type,
        "sample_quantum_vector": sample_quantum_vector,
        "scaler_parameters": preprocessed["scaler_parameters"],
        "created_at": exp.created_at.isoformat() if exp.created_at else None
    }
