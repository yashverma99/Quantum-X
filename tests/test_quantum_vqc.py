import pytest
import numpy as np
from fastapi.testclient import TestClient
from app.main import app
from app.core.database import SessionLocal
from app.models.experiment import Experiment
from app.services.quantum_service import QuantumService

client = TestClient(app)


@pytest.fixture(scope="module")
def db_session():
    db = SessionLocal()
    yield db
    db.close()


def test_qiskit_aer_availability():
    """Verify that Qiskit Aer executes a real 1-qubit circuit and reports availability."""
    health = QuantumService.check_qiskit_availability()
    assert health["available"] is True
    assert health["backend"] == "AerSimulator"
    assert health["test"] == "passed"
    assert "counts" in health


def test_build_vqc_circuit():
    """Verify programmatic parameterized circuit construction with Angle Encoding and RealAmplitudes."""
    circuit_pkg = QuantumService.build_vqc_circuit(
        qubits=4,
        depth=2,
        encoding="angle",
        ansatz_type="RealAmplitudes"
    )
    meta = circuit_pkg["metadata"]
    assert meta["qubits"] == 4
    assert meta["circuit_depth"] > 0
    assert meta["variational_parameters"] == 12  # 4 qubits * (2 reps + 1)
    assert meta["feature_parameters"] == 4
    assert "ry" in meta["gate_counts"]
    assert "cx" in meta["gate_counts"]
    assert "ascii_diagram" in meta
    assert len(meta["ascii_diagram"]) > 0


def test_circuit_preview_endpoint():
    """Verify the GET /api/models/quantum/circuit/preview endpoint."""
    response = client.get("/api/models/quantum/circuit/preview?qubits=4&depth=2")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["circuit"]["qubits"] == 4
    assert data["circuit"]["num_parameters"] == 16


def test_verify_experiment_integrity_rejection(db_session):
    """Verify that quantum training strictly rejects mismatched or unvalidated configurations."""
    # Test invalid experiment code
    with pytest.raises(ValueError) as excinfo:
        QuantumService.verify_experiment_integrity(db_session, "EXP-INVALID-999")
    assert "does not exist" in str(excinfo.value).lower() or "not found" in str(excinfo.value).lower()


def test_vqc_training_endpoint_small(db_session):
    """
    Verify real VQC training via API with minimal iterations to validate end-to-end execution,
    metric computation, persistence, and fair classical baseline comparison.
    """
    response = client.post(
        "/api/models/quantum/vqc/train",
        json={
            "experiment_id": "EXP-004",
            "qubits": 4,
            "depth": 1,
            "max_iterations": 5,
            "shots": 200,
            "seed": 42
        }
    )
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "COMPLETED"
    assert res["model_name"] == "VQC"
    assert res["experiment_id"] == "EXP-004"
    assert "metrics" in res
    m = res["metrics"]
    assert 0.0 <= m["balanced_accuracy"] <= 1.0
    assert 0.0 <= m["accuracy"] <= 1.0
    assert 0.0 <= m["f1"] <= 1.0
    assert "confusion_matrix" in m
    cm = m["confusion_matrix"]
    assert (cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"]) == 114

    assert "comparison" in res
    comp = res["comparison"]
    assert "classical_baseline" in comp
    assert "quantum_model" in comp
    assert comp["classical_baseline"]["model_name"] == "Logistic Regression"
    assert "research_statement" in comp


def test_validate_quantum_metrics():
    """Verify that independent mathematical audit confirms valid metrics and flags inconsistencies."""
    cm = {"tn": 68, "fp": 4, "fn": 1, "tp": 41}
    sens = 41 / 42
    spec = 68 / 72
    bal_acc = (sens + spec) / 2.0

    valid_res = QuantumService.validate_quantum_metrics(cm, sens, spec, bal_acc, total_samples=114)
    assert valid_res["validation_status"] == "VALIDATED"
    assert valid_res["sample_count_check"]["passed"] is True
    assert valid_res["sensitivity_check"]["passed"] is True
    assert valid_res["specificity_check"]["passed"] is True
    assert valid_res["balanced_accuracy_check"]["passed"] is True

    # Mutated inconsistent metric test:
    bad_res = QuantumService.validate_quantum_metrics(cm, 0.50, spec, bal_acc, total_samples=114)
    assert bad_res["validation_status"] == "REQUIRES_REVIEW"
    assert bad_res["sensitivity_check"]["passed"] is False


def test_build_qsvm_circuit():
    """Verify ZZFeatureMap circuit construction and decomposed depth metadata."""
    circuit_pkg = QuantumService.build_qsvm_circuit(qubits=4, reps=1, feature_map_name="ZZFeatureMap")
    meta = circuit_pkg["metadata"]
    assert meta["qubits"] == 4
    assert meta["logical_reps"] == 1
    assert meta["circuit_depth"] > 0
    assert meta["decomposed_circuit_depth"] == meta["circuit_depth"]
    assert "depth_terminology_note" in meta
    assert "ascii_diagram" in meta


def test_qsvm_preview_endpoint():
    """Verify GET /api/models/quantum/qsvm/circuit/preview endpoint."""
    response = client.get("/api/models/quantum/qsvm/circuit/preview?qubits=4&reps=1")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["circuit"]["qubits"] == 4
    assert data["circuit"]["logical_reps"] == 1


def test_qsvm_training_endpoint(db_session):
    """
    Verify real QSVM training via API with Qiskit Aer statevector kernel evaluation,
    SVM precomputed kernel fit, metric validation, and fair classical baseline comparison.
    """
    response = client.post(
        "/api/models/quantum/qsvm/train",
        json={
            "experiment_id": "EXP-004",
            "qubits": 4,
            "reps": 1,
            "C": 1.0,
            "feature_map": "ZZFeatureMap",
            "seed": 42
        }
    )
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "COMPLETED"
    assert res["model_name"] == "QSVM"
    assert res["experiment_id"] == "EXP-004"
    assert "kernel" in res
    assert res["kernel"]["kernel_train_time_ms"] > 0
    assert "kernel_matrix_sample" in res["kernel"]
    assert "metrics" in res
    m = res["metrics"]
    assert 0.0 <= m["balanced_accuracy"] <= 1.0
    assert 0.0 <= m["accuracy"] <= 1.0
    assert 0.0 <= m["f1"] <= 1.0
    assert "confusion_matrix" in m
    cm = m["confusion_matrix"]
    assert (cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"]) == 114

    assert res["validation_audit"]["validation_status"] == "VALIDATED"
    assert "comparison" in res
    assert res["comparison"]["classical_baseline"]["model_name"] == "Logistic Regression"


def test_get_quantum_runs_endpoint():
    """Verify retrieval of persisted quantum runs (both VQC and QSVM) for an experiment."""
    response = client.get("/api/models/quantum/runs/EXP-004")
    assert response.status_code == 200
    data = response.json()
    assert "quantum_runs" in data
    assert len(data["quantum_runs"]) >= 1
    models_found = {r["model_name"] for r in data["quantum_runs"]}
    assert "VQC" in models_found or "QSVM" in models_found
    for r in data["quantum_runs"]:
        assert r["validation_status"] in ["VALIDATED", "REQUIRES_REVIEW"]
