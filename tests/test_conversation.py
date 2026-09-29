import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_conversation_query_endpoint():
    payload = {
        "query": "How does VQC compare to the classical model?",
        "language": "en"
    }
    response = client.post("/api/conversation/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "response_text" in data
    assert "spoken_text" in data
    assert data["language"] == "en"
    assert "Balanced Accuracy" in data["response_text"]
    assert data["suggested_action"] is not None

def test_conversation_clinical_safety_guardrail():
    payload = {
        "query": "Does the patient have cancer?",
        "language": "en"
    }
    response = client.post("/api/conversation/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "not a clinical diagnosis" in data["response_text"]
    assert data["suggested_action"]["screen"] == "review"

def test_conversation_multilingual_hindi():
    payload = {
        "query": "VQC मॉडल के परिणाम क्या हैं?",
        "language": "hi"
    }
    response = client.post("/api/conversation/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["language"] == "hi"
    assert len(data["response_text"]) > 0

def test_conversation_multilingual_telugu():
    payload = {
        "query": "క్వాంటమ్ సర్క్యూట్ వివరణ ఇవ్వండి",
        "language": "te"
    }
    response = client.post("/api/conversation/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["language"] == "te"
    assert len(data["response_text"]) > 0
