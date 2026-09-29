import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["project"] == "MediQAI"
    assert data["status"] == "online"
    assert "disclaimer" in data


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["ok", "degraded"]
    assert data["service"] == "MediQAI Backend"
    assert "version" in data
    assert "python" in data
    assert data["quantum_simulator"] in ["available", "unavailable"]
    assert data["database"] in ["configured", "not_connected"]


def test_datasets_endpoint():
    response = client.get("/api/datasets")
    assert response.status_code == 200
    data = response.json()
    assert "datasets" in data
    assert isinstance(data["datasets"], list)


def test_experiments_endpoint():
    response = client.get("/api/experiments")
    assert response.status_code == 200
    data = response.json()
    assert "experiments" in data
    assert isinstance(data["experiments"], list)


def test_models_endpoint():
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert isinstance(data["models"], list)


def test_system_status_endpoint():
    response = client.get("/api/system/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["ok", "degraded"]
    assert "healthcare_safety_policy" in data


def test_quantum_system_endpoint():
    response = client.get("/api/system/quantum")
    assert response.status_code == 200
    data = response.json()
    assert "available" in data
    if data["available"]:
        assert data["provider"] == "Qiskit Aer"
        assert data["test"] == "passed"
        assert "qiskit_version" in data
        assert "aer_version" in data


def test_experiment_preprocessing_endpoint():
    # Fetch experiments to test on an existing one if available
    exp_res = client.get("/api/experiments")
    assert exp_res.status_code == 200
    experiments = exp_res.json().get("experiments", [])
    if len(experiments) > 0:
        exp_id = experiments[0]["experiment_code"]
        res = client.get(f"/api/experiments/{exp_id}/preprocessing")
        assert res.status_code == 200
        p = res.json()
        assert "dataset" in p
        assert "target" in p
        assert "split" in p
        assert "features" in p
        assert "pca" in p

