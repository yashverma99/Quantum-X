import logging
from typing import List, Dict, Any, Optional, Union
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.model_record import ModelRecord
from app.models.experiment import Experiment
from app.models.quantum_run import QuantumRun
from app.services.classical_ml_service import classical_ml_service
from app.services.quantum_service import quantum_service

logger = logging.getLogger("mediqai.api.models")
router = APIRouter(prefix="/models", tags=["Models"])


class ClassicalTrainRequest(BaseModel):
    experiment_id: Union[str, int] = Field(default="EXP-004", description="Experiment code (e.g. EXP-004) or database ID")
    models: Optional[List[str]] = Field(
        default=["logistic_regression", "random_forest", "rbf_svm"],
        description="List of classical models to train"
    )


@router.get("")
def get_models(experiment_id: Optional[Union[str, int]] = None, db: Session = Depends(get_db)):
    """Lists model records, optionally filtered by experiment_id."""
    query = db.query(ModelRecord)
    if experiment_id is not None:
        if isinstance(experiment_id, int) or (isinstance(experiment_id, str) and experiment_id.isdigit()):
            query = query.filter(ModelRecord.experiment_id == int(experiment_id))
        else:
            exp = db.query(Experiment).filter(Experiment.experiment_code == str(experiment_id)).first()
            if exp:
                query = query.filter(ModelRecord.experiment_id == exp.id)
            else:
                return {"models": []}

    models = query.order_by(ModelRecord.created_at.desc()).all()
    results = []
    for m in models:
        exp_code = m.experiment.experiment_code if m.experiment else f"EXP-{m.experiment_id:03d}"
        results.append({
            "id": m.id,
            "experiment_id": m.experiment_id,
            "experiment_code": exp_code,
            "model_name": m.model_name,
            "model_type": m.model_type,
            "accuracy": m.accuracy,
            "balanced_accuracy": m.balanced_accuracy,
            "precision": m.precision,
            "recall": m.recall,
            "sensitivity": m.sensitivity or m.recall,
            "specificity": m.specificity,
            "f1_score": m.f1_score or m.f1,
            "f1": m.f1 or m.f1_score,
            "macro_f1": m.macro_f1,
            "roc_auc": m.roc_auc,
            "training_time_seconds": m.training_time_seconds,
            "training_time_ms": m.training_time_ms or (round(m.training_time_seconds * 1000.0, 2) if m.training_time_seconds else None),
            "inference_time_seconds": m.inference_time_seconds,
            "inference_time_ms": m.inference_time_ms or (round(m.inference_time_seconds * 1000.0, 2) if m.inference_time_seconds else None),
            "confusion_matrix": m.confusion_matrix,
            "roc_curve": m.roc_curve_data,
            "hyperparameters": m.hyperparameters,
            "created_at": m.created_at.isoformat() if m.created_at else None
        })
    return {"models": results}


@router.get("/classical/status")
def get_classical_status():
    """Returns available classical algorithms and runtime XGBoost availability."""
    return classical_ml_service.get_supported_models()


@router.get("/classical/{experiment_id}")
def get_classical_models_for_experiment(experiment_id: Union[str, int], db: Session = Depends(get_db)):
    """Retrieves all classical model evaluation results for a specific experiment."""
    if isinstance(experiment_id, int) or (isinstance(experiment_id, str) and experiment_id.isdigit()):
        exp = db.query(Experiment).filter(Experiment.id == int(experiment_id)).first()
    else:
        exp = db.query(Experiment).filter(Experiment.experiment_code == str(experiment_id)).first()

    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found.")

    records = (
        db.query(ModelRecord)
        .filter(ModelRecord.experiment_id == exp.id, ModelRecord.model_type == "classical")
        .order_by(ModelRecord.balanced_accuracy.desc().nullslast())
        .all()
    )

    models_data = []
    for r in records:
        models_data.append({
            "id": r.id,
            "model_name": r.model_name,
            "model_type": r.model_type,
            "accuracy": r.accuracy,
            "balanced_accuracy": r.balanced_accuracy,
            "precision": r.precision,
            "recall": r.recall,
            "sensitivity": r.sensitivity or r.recall,
            "specificity": r.specificity,
            "f1": r.f1 or r.f1_score,
            "macro_f1": r.macro_f1,
            "roc_auc": r.roc_auc,
            "training_time_ms": r.training_time_ms or (round(r.training_time_seconds * 1000.0, 2) if r.training_time_seconds else None),
            "inference_time_ms": r.inference_time_ms or (round(r.inference_time_seconds * 1000.0, 2) if r.inference_time_seconds else None),
            "confusion_matrix": r.confusion_matrix,
            "roc_curve": r.roc_curve_data,
            "hyperparameters": r.hyperparameters,
            "created_at": r.created_at.isoformat() if r.created_at else None
        })

    best_model = None
    if models_data:
        best_model = {
            "name": models_data[0]["model_name"],
            "balanced_accuracy": models_data[0]["balanced_accuracy"]
        }

    return {
        "experiment_id": exp.experiment_code,
        "status": exp.status,
        "dataset_name": exp.dataset.name if exp.dataset else "Biomedical Dataset",
        "best_model_in_experiment": best_model,
        "models": models_data,
        "validation": (exp.config or {}).get("validation")
    }


@router.get("/classical/{experiment_id}/audit")
def audit_classical_models(experiment_id: Union[str, int], db: Session = Depends(get_db)):
    """
    Performs an independent scientific validation audit of the classical results
    for an experiment: leakage audit, duplicate audit, target leakage audit,
    reproducibility verification, and 5-fold cross-validation.
    """
    try:
        audit_result = classical_ml_service.audit_experiment(db, experiment_id, mark_validated=True)
        return audit_result
    except Exception as e:
        logger.error(f"Audit failed for experiment '{experiment_id}': {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/classical/train", status_code=status.HTTP_200_OK)
def train_classical_models(req: ClassicalTrainRequest, db: Session = Depends(get_db)):
    """
    Executes real classical machine learning training using the exact leakage-safe
    data split and PCA orthogonal feature representation produced in Phase 2.
    """
    # 1. Load experiment artifacts & validate leakage safeguards
    try:
        data = classical_ml_service.load_experiment_data(db, req.experiment_id)
    except ValueError as ve:
        # Mark experiment as failed if it was found
        if isinstance(req.experiment_id, int) or (isinstance(req.experiment_id, str) and str(req.experiment_id).isdigit()):
            exp = db.query(Experiment).filter(Experiment.id == int(req.experiment_id)).first()
        else:
            exp = db.query(Experiment).filter(Experiment.experiment_code == str(req.experiment_id)).first()

        if exp:
            exp.status = "FAILED"
            exp_cfg = dict(exp.config or {})
            exp_cfg["error"] = str(ve)
            exp.config = exp_cfg
            db.commit()

        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected pipeline error: {str(e)}")

    exp = data["experiment"]
    ds = data["dataset"]
    models_to_train = req.models or ["logistic_regression", "random_forest", "rbf_svm"]

    # 2. Train selected models and compute genuine metrics
    try:
        results = classical_ml_service.train_models(data, models_to_train)
    except Exception as e:
        exp.status = "FAILED"
        exp_cfg = dict(exp.config or {})
        exp_cfg["error"] = f"Training failed: {str(e)}"
        exp.config = exp_cfg
        db.commit()
        raise HTTPException(status_code=500, detail=f"Model training execution failed: {str(e)}")

    # 3. Persist model records to database
    saved_records = classical_ml_service.save_model_results(db, exp, results)

    # 4. Construct response payload
    best_model = None
    if results:
        best_rec = max(results, key=lambda r: r["balanced_accuracy"])
        best_model = {
            "name": best_rec["model_name"],
            "balanced_accuracy": best_rec["balanced_accuracy"],
            "note": "Best balanced accuracy in this experiment."
        }

    models_output = []
    for r in results:
        models_output.append({
            "model_name": r["model_name"],
            "model_type": r["model_type"],
            "accuracy": r["accuracy"],
            "balanced_accuracy": r["balanced_accuracy"],
            "precision": r["precision"],
            "recall": r["recall"],
            "sensitivity": r["sensitivity"],
            "specificity": r["specificity"],
            "f1": r["f1"],
            "macro_f1": r["macro_f1"],
            "roc_auc": r["roc_auc"],
            "training_time_ms": r["training_time_ms"],
            "inference_time_ms": r["inference_time_ms"],
            "confusion_matrix": r["confusion_matrix"],
            "roc_curve": r["roc_curve_data"],
            "hyperparameters": r["hyperparameters"]
        })

    return {
        "status": "COMPLETED",
        "experiment_id": exp.experiment_code,
        "dataset": {
            "name": ds.name,
            "target_column": data["target_column"],
            "train_samples": len(data["y_train"]),
            "test_samples": len(data["y_test"]),
            "features_count": data.get("pca_components", data["X_train"].shape[1]),
            "random_seed": data["random_seed"]
        },
        "best_model_in_experiment": best_model,
        "models": models_output
    }


class VQCTrainRequest(BaseModel):
    experiment_id: Union[str, int] = Field(default="EXP-004", description="Experiment code (e.g. EXP-004)")
    qubits: int = Field(default=4, ge=2, le=16)
    depth: int = Field(default=2, ge=1, le=10)
    encoding: str = Field(default="angle")
    optimizer: str = Field(default="COBYLA")
    max_iterations: int = Field(default=50, ge=5, le=100)
    shots: int = Field(default=1024, ge=100, le=8192)
    seed: int = Field(default=42)


class QSVMTrainRequest(BaseModel):
    experiment_id: Union[str, int] = Field(default="EXP-004", description="Experiment code (e.g. EXP-004)")
    qubits: int = Field(default=4, ge=2, le=16)
    reps: int = Field(default=1, ge=1, le=4)
    C: float = Field(default=1.0, gt=0.0)
    feature_map: str = Field(default="ZZFeatureMap")
    seed: int = Field(default=42)


@router.post("/quantum/vqc/train", status_code=status.HTTP_200_OK)
def train_quantum_vqc(req: VQCTrainRequest, db: Session = Depends(get_db)):
    """
    Executes real Variational Quantum Classifier (VQC) training on Qiskit Aer
    using the exact leakage-safe data partition from the validated classical baseline.
    """
    try:
        result = quantum_service.train_vqc(
            db=db,
            experiment_id=req.experiment_id,
            qubits=req.qubits,
            depth=req.depth,
            encoding=req.encoding,
            optimizer_name=req.optimizer,
            max_iterations=req.max_iterations,
            shots=req.shots,
            seed=req.seed
        )
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=503, detail=str(re))
    except Exception as e:
        logger.error(f"VQC training execution error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"VQC execution failed: {str(e)}")


@router.post("/quantum/qsvm/train", status_code=status.HTTP_200_OK)
def train_quantum_qsvm(req: QSVMTrainRequest, db: Session = Depends(get_db)):
    """
    Executes real Quantum Support Vector Machine (QSVM) kernel evaluation and training on Qiskit Aer
    using the exact leakage-safe data partition from the validated classical baseline.
    """
    try:
        result = quantum_service.train_qsvm(
            db=db,
            experiment_id=req.experiment_id,
            qubits=req.qubits,
            reps=req.reps,
            C=req.C,
            feature_map_name=req.feature_map,
            seed=req.seed
        )
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=503, detail=str(re))
    except Exception as e:
        logger.error(f"QSVM training execution error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"QSVM execution failed: {str(e)}")


@router.get("/quantum/circuit/preview")
def preview_quantum_circuit(
    qubits: int = 4,
    depth: int = 2,
    encoding: str = "angle",
    ansatz: str = "RealAmplitudes"
):
    """Generates and returns programmatic VQC circuit metadata, gate counts, and ASCII diagram."""
    try:
        circuit_pkg = quantum_service.build_vqc_circuit(
            qubits=qubits,
            depth=depth,
            encoding=encoding,
            ansatz_type=ansatz
        )
        return {
            "status": "ready",
            "circuit": circuit_pkg["metadata"]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/quantum/qsvm/circuit/preview")
def preview_qsvm_circuit(
    qubits: int = 4,
    reps: int = 1,
    feature_map: str = "ZZFeatureMap"
):
    """Generates and returns programmatic QSVM feature map circuit metadata, gate counts, and ASCII diagram."""
    try:
        circuit_pkg = quantum_service.build_qsvm_circuit(
            qubits=qubits,
            reps=reps,
            feature_map_name=feature_map
        )
        return {
            "status": "ready",
            "circuit": circuit_pkg["metadata"]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/quantum/runs/{experiment_id}")
def get_quantum_runs_for_experiment(experiment_id: Union[str, int], db: Session = Depends(get_db)):
    """Retrieves all quantum runs recorded for an experiment."""
    if isinstance(experiment_id, int) or (isinstance(experiment_id, str) and experiment_id.isdigit()):
        exp = db.query(Experiment).filter(Experiment.id == int(experiment_id)).first()
    else:
        exp = db.query(Experiment).filter(Experiment.experiment_code == str(experiment_id)).first()

    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found.")

    runs = db.query(QuantumRun).filter(QuantumRun.experiment_id == exp.id).order_by(QuantumRun.created_at.desc()).all()
    results = []
    for r in runs:
        results.append({
            "id": r.id,
            "experiment_id": exp.experiment_code,
            "model_name": r.model_name,
            "backend": r.backend,
            "qubits": r.qubits,
            "circuit_depth": r.circuit_depth,
            "encoding": r.encoding,
            "ansatz": r.ansatz,
            "optimizer": r.optimizer,
            "iterations": r.iterations,
            "final_loss": r.final_loss,
            "shots": r.shots,
            "accuracy": r.accuracy,
            "balanced_accuracy": r.balanced_accuracy,
            "precision": r.precision,
            "recall": r.recall,
            "sensitivity": r.sensitivity,
            "specificity": r.specificity,
            "f1": r.f1,
            "macro_f1": r.macro_f1,
            "roc_auc": r.roc_auc,
            "training_time_ms": r.training_time_ms,
            "inference_time_ms": r.inference_time_ms,
            "confusion_matrix": r.confusion_matrix,
            "roc_curve": r.roc_curve_data,
            "circuit_summary": r.circuit_summary,
            "circuit_text": r.circuit_text,
            "validation_status": r.validation_status or "VALIDATED",
            "created_at": r.created_at.isoformat() if r.created_at else None
        })
    return {"quantum_runs": results}
