"""
Unit tests for FastAPI serving endpoints in api/main.py.
"""

from fastapi.testclient import TestClient

from api.main import MODEL_STATE, app


def test_health_check():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "edgetune-api"}


def test_model_info_unloaded():
    client = TestClient(app)
    # Ensure model is None
    MODEL_STATE["model"] = None
    response = client.get("/model-info")
    assert response.status_code == 503
    assert "Model not loaded" in response.json()["detail"]


def test_generate_unloaded():
    client = TestClient(app)
    MODEL_STATE["model"] = None
    MODEL_STATE["tokenizer"] = None
    response = client.post("/generate", json={"prompt": "Hello", "max_new_tokens": 16})
    assert response.status_code == 503


def test_generate_empty_prompt():
    client = TestClient(app)
    response = client.post("/generate", json={"prompt": "", "max_new_tokens": 16})
    assert response.status_code in [400, 503]
