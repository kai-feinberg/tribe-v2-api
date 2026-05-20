import importlib
import time

from fastapi.testclient import TestClient


def test_health_and_metadata(monkeypatch, tmp_path):
    monkeypatch.setenv("TRIBE_FAKE_INFERENCE", "true")
    monkeypatch.setenv("TRIBE_JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("TRIBE_CACHE_DIR", str(tmp_path / "cache"))

    app_module = importlib.import_module("tribev2_api.app")
    app_module = importlib.reload(app_module)

    with TestClient(app_module.app) as client:
        health = client.get("/health")
        metadata = client.get("/metadata")

    assert health.status_code == 200
    assert metadata.status_code == 200
    assert metadata.json()["model"]["repo"] == "facebook/tribev2"


def test_fake_text_job_completes(monkeypatch, tmp_path):
    monkeypatch.setenv("TRIBE_FAKE_INFERENCE", "true")
    monkeypatch.setenv("TRIBE_JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("TRIBE_CACHE_DIR", str(tmp_path / "cache"))

    app_module = importlib.import_module("tribev2_api.app")
    app_module = importlib.reload(app_module)

    with TestClient(app_module.app) as client:
        created = client.post("/predict/text", json={"text": "hello from the test"})
        assert created.status_code == 202
        job = created.json()

        status = None
        for _ in range(30):
            response = client.get(job["status_url"])
            assert response.status_code == 200
            status = response.json()
            if status["status"] == "completed":
                break
            time.sleep(0.05)

        assert status is not None
        assert status["status"] == "completed"
        assert client.get(f"/jobs/{job['job_id']}/result.json").status_code == 200
        preds = client.get(f"/jobs/{job['job_id']}/preds.norm.f16.bin")
        assert preds.status_code == 200
        assert len(preds.content) == 3 * 20484 * 2
