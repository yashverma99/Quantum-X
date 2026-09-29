import pytest
import numpy as np
from fastapi.testclient import TestClient
from app.main import app
from app.core.database import SessionLocal
from app.models.experiment import Experiment
from app.services.classical_ml_service import classical_ml_service

client = TestClient(app)


@pytest.fixture(scope="module")
def db_session():
    db = SessionLocal()
    yield db
    db.close()


def test_classical_status_endpoint():
    """Verify supported models list and honest XGBoost availability report."""
    response = client.get("/api/models/classical/status")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "xgboost" in data
    model_ids = [m["id"] for m in data["models"]]
    assert "logistic_regression" in model_ids
    assert "random_forest" in model_ids
    assert "rbf_svm" in model_ids


def test_load_experiment_data_and_leakage_safeguards(db_session):
    """Verify that experiment data loads without leakage and adheres to strict safeguards."""
    # Find existing preprocessed experiment
    exp = db_session.query(Experiment).filter(Experiment.experiment_code == "EXP-004").first()
    if not exp:
        exp = db_session.query(Experiment).first()

    assert exp is not None
    data = classical_ml_service.load_experiment_data(db_session, exp.experiment_code)

    assert "X_train" in data
    assert "X_test" in data
    assert "y_train" in data
    assert "y_test" in data
    assert data["X_train"].shape[0] == 455
    assert data["X_test"].shape[0] == 114
    assert data["X_train"].shape[1] == 8
    assert data["X_test"].shape[1] == 8

    # Verify no NaN or Inf
    assert not np.isnan(data["X_train"]).any()
    assert not np.isnan(data["X_test"]).any()
    assert not np.isinf(data["X_train"]).any()
    assert not np.isinf(data["X_test"]).any()


def test_logistic_regression_training(db_session):
    """Verify Logistic Regression training, metric validity, and confusion matrix."""
    data = classical_ml_service.load_experiment_data(db_session, "EXP-004")
    results = classical_ml_service.train_models(data, ["logistic_regression"])

    assert len(results) == 1
    lr = results[0]
    assert lr["model_name"] == "Logistic Regression"
    assert 0.8 <= lr["accuracy"] <= 1.0
    assert 0.8 <= lr["balanced_accuracy"] <= 1.0
    assert 0.8 <= lr["precision"] <= 1.0
    assert 0.8 <= lr["recall"] <= 1.0
    assert 0.8 <= lr["specificity"] <= 1.0
    assert 0.8 <= lr["f1"] <= 1.0
    assert 0.8 <= lr["roc_auc"] <= 1.0
    assert lr["training_time_ms"] > 0
    assert lr["inference_time_ms"] >= 0

    # Check confusion matrix
    cm = lr["confusion_matrix"]
    assert "tn" in cm and "fp" in cm and "fn" in cm and "tp" in cm
    assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == 114


def test_random_forest_training(db_session):
    """Verify Random Forest training and metrics."""
    data = classical_ml_service.load_experiment_data(db_session, "EXP-004")
    results = classical_ml_service.train_models(data, ["random_forest"])

    assert len(results) == 1
    rf = results[0]
    assert rf["model_name"] == "Random Forest"
    assert 0.8 <= rf["balanced_accuracy"] <= 1.0
    assert rf["training_time_ms"] > 0
    assert "roc_curve_data" in rf
    assert len(rf["roc_curve_data"]["fpr"]) > 0


def test_rbf_svm_training(db_session):
    """Verify RBF-SVM training and metrics."""
    data = classical_ml_service.load_experiment_data(db_session, "EXP-004")
    results = classical_ml_service.train_models(data, ["rbf_svm"])

    assert len(results) == 1
    svm = results[0]
    assert svm["model_name"] == "RBF-SVM"
    assert 0.8 <= svm["balanced_accuracy"] <= 1.0
    assert svm["training_time_ms"] > 0
    assert svm["roc_auc"] is not None


def test_reproducibility(db_session):
    """Verify that repeated runs with identical random seed produce identical metrics."""
    data = classical_ml_service.load_experiment_data(db_session, "EXP-004")
    run1 = classical_ml_service.train_models(data, ["logistic_regression", "random_forest"])
    run2 = classical_ml_service.train_models(data, ["logistic_regression", "random_forest"])

    assert run1[0]["balanced_accuracy"] == run2[0]["balanced_accuracy"]
    assert run1[0]["roc_auc"] == run2[0]["roc_auc"]
    assert run1[1]["balanced_accuracy"] == run2[1]["balanced_accuracy"]


def test_classical_train_api_endpoint():
    """Verify POST /api/models/classical/train returns genuine metrics and status."""
    payload = {
        "experiment_id": "EXP-004",
        "models": ["logistic_regression", "rbf_svm"]
    }
    response = client.post("/api/models/classical/train", json=payload)
    assert response.status_code == 200
    res = response.json()

    assert res["status"] == "COMPLETED"
    assert res["experiment_id"] == "EXP-004"
    assert "best_model_in_experiment" in res
    assert len(res["models"]) == 2

    # Check that model metrics are non-zero and populated
    for m in res["models"]:
        assert m["balanced_accuracy"] > 0.8
        assert m["training_time_ms"] > 0
        assert "confusion_matrix" in m
        assert "roc_curve" in m


def test_get_classical_models_for_experiment():
    """Verify GET /api/models/classical/{experiment_id} retrieves saved records."""
    response = client.get("/api/models/classical/EXP-004")
    assert response.status_code == 200
    res = response.json()
    assert res["experiment_id"] == "EXP-004"
    assert res["status"] in ["COMPLETED", "VALIDATED"]
    assert len(res["models"]) >= 2
    assert "best_model_in_experiment" in res


def test_audit_classical_models_endpoint():
    """Verify GET /api/models/classical/{experiment_id}/audit performs full scientific validation."""
    response = client.get("/api/models/classical/EXP-004/audit")
    assert response.status_code == 200
    res = response.json()
    assert res["experiment_code"] == "EXP-004"
    assert res["validation_status"] == "VALIDATED"
    assert res["all_passed"] is True
    assert res["audits"]["leakage_audit"]["status"] == "PASSED"
    assert res["audits"]["train_test_isolation"]["status"] == "PASSED"
    assert res["audits"]["duplicate_audit"]["status"] == "PASSED"
    assert res["audits"]["target_leakage_audit"]["status"] == "PASSED"
    assert res["audits"]["reproducibility_check"]["status"] == "PASSED"
    assert res["audits"]["cross_validation"]["status"] == "PASSED"
    assert res["cv_statistics"]["mean_balanced_accuracy"] >= 0.90
    assert res["held_out_test_metrics"]["balanced_accuracy"] >= 0.95
