import io
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from audio_studio.api import create_app
from audio_studio.jobs import JobManager
from audio_studio.store import write_json


@pytest.fixture
def client(tmp_path: Path):
    with TestClient(create_app(tmp_path)) as session:
        yield session


def test_presets_persist_and_builtins_are_protected(tmp_path: Path):
    with TestClient(create_app(tmp_path)) as client:
        presets = client.get("/api/presets").json()
        assert len({x["category"] for x in presets}) == 5
        assert client.delete("/api/presets/zh-narration").status_code == 400
        preset = {key: value for key, value in presets[0].items() if key not in {"id", "builtin"}}
        preset["name"] = "我的旁白"
        created = client.post("/api/presets", json=preset)
        assert created.status_code == 200, created.text
        preset_id = created.json()["id"]
        assert client.post("/api/presets", json=dict(preset, voice="../../arbitrary")).status_code == 422
        assert client.post("/api/presets", json=dict(preset, engine="f5", model_id="f5-base")).status_code == 422
    with TestClient(create_app(tmp_path)) as reopened:
        assert any(x["id"] == preset_id for x in reopened.get("/api/presets").json())
        assert reopened.delete(f"/api/presets/{preset_id}").status_code == 200


def test_invalid_or_uninstalled_synthesis_is_not_queued(client):
    assert client.post("/api/synthesize", json={"preset_id": "zh-narration", "text": "  "}).status_code == 422
    assert client.post("/api/synthesize", json={"preset_id": "missing", "text": "你好"}).status_code == 404
    response = client.post("/api/synthesize", json={"preset_id": "zh-narration", "text": "你好"})
    assert response.status_code == 400
    assert "模型尚未就绪" in response.json()["detail"]
    assert client.get("/api/jobs").json() == []


def test_reference_upload_checks_real_audio_and_does_not_trust_filename(client, tmp_path):
    buffer = io.BytesIO()
    signal = np.sin(2 * np.pi * 220 * np.arange(48000) / 24000).astype("float32") * .2
    sf.write(buffer, signal, 24000, format="WAV")
    response = client.post("/api/references", files={"file": ("../../escape.wav", buffer.getvalue(), "audio/wav")})
    assert response.status_code == 200, response.text
    path = Path(response.json()["path"])
    assert path.parent == tmp_path / "references"
    assert path.is_file()
    bad = client.post("/api/references", files={"file": ("bad.wav", b"invalid", "audio/wav")})
    assert bad.status_code == 400
    assert len(list((tmp_path / "references").iterdir())) == 1


def test_api_auth_and_cross_origin_protection(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIO_STUDIO_TOKEN", "test-token")
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/api/presets").status_code == 401
        client.headers["Authorization"] = "Bearer test-token"
        assert client.get("/api/presets").status_code == 200
        assert client.get("/api/presets", headers={"Origin": "https://untrusted.example"}).status_code == 403
    monkeypatch.delenv("AUDIO_STUDIO_TOKEN")
    with TestClient(create_app(tmp_path / "unprotected"), base_url="http://attacker.example") as client:
        assert client.get("/api/presets").status_code == 403


def test_dataset_reports_missing_recordings_without_success(client):
    response = client.post("/api/datasets/validate", json={"csv_path": "missing.csv", "wavs_dir": "missing"})
    assert response.status_code == 200
    assert response.json()["valid"] is False
    assert response.json()["errors"]


def test_audio_path_traversal_and_missing_files(client):
    assert client.get("/api/audio/catalogue.json").status_code == 404
    assert client.get("/api/audio/" + "0" * 32 + ".wav").status_code == 404


def test_worker_failure_is_durable_and_queue_keeps_running(tmp_path):
    (tmp_path / "jobs").mkdir()
    jobs = JobManager(tmp_path / "jobs", lambda request, result: result)
    try:
        first = jobs.submit("unsupported", {})
        second = jobs.submit("validate", {"csv_path": "missing.csv", "wavs_dir": "missing"})
        deadline = time.monotonic() + 30
        while jobs.get(second["id"])["status"] not in {"succeeded", "failed"} and time.monotonic() < deadline:
            time.sleep(.05)
        assert jobs.get(first["id"])["status"] == "failed"
        assert jobs.get(first["id"])["logs"]
        assert jobs.get(second["id"])["status"] == "succeeded"
        assert jobs.get(second["id"])["result"]["valid"] is False
    finally:
        jobs.close()
    recovered = JobManager(tmp_path / "jobs", lambda request, result: result)
    try:
        assert recovered.get(first["id"])["status"] == "failed"
    finally:
        recovered.close()


def test_interrupted_jobs_are_marked_failed_on_restart(tmp_path):
    folder = tmp_path / "interrupted"
    folder.mkdir()
    write_json(folder / "job.json", {"id": "interrupted", "kind": "train", "status": "running", "error": None, "result": None})
    jobs = JobManager(tmp_path, lambda request, result: result)
    try:
        assert jobs.get("interrupted")["status"] == "failed"
        assert "服务中断" in jobs.get("interrupted")["error"]
    finally:
        jobs.close()


def test_cancelling_running_and_queued_workers_releases_queue(tmp_path, monkeypatch):
    original_popen = subprocess.Popen
    processes = []

    def start_worker(command, **kwargs):
        if command[1:3] != ["-m", "audio_studio.worker"]:
            # Cancellation also launches Windows taskkill; it is not a worker.
            return original_popen(command, **kwargs)
        request = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
        if request["kind"] == "long-running-test":
            command = [sys.executable, "-c", "import time; print('started', flush=True); time.sleep(60)"]
        process = original_popen(command, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr("audio_studio.jobs.subprocess.Popen", start_worker)
    jobs = JobManager(tmp_path, lambda request, result: result)
    try:
        active = jobs.submit("long-running-test", {})
        queued = jobs.submit("long-running-test", {})
        deadline = time.monotonic() + 5
        while not processes and time.monotonic() < deadline:
            time.sleep(.02)
        assert processes
        assert jobs.cancel(queued["id"])["status"] == "cancelled"
        assert jobs.cancel(active["id"])["status"] == "cancelled"
        following = jobs.submit("validate", {"csv_path": "missing.csv", "wavs_dir": "missing"})
        deadline = time.monotonic() + 20
        while jobs.get(following["id"])["status"] not in {"succeeded", "failed"} and time.monotonic() < deadline:
            time.sleep(.05)
        assert processes[0].poll() is not None
        assert jobs.get(following["id"])["status"] == "succeeded"
        assert len(processes) == 2  # Cancelled queued work never starts a process.
    finally:
        jobs.close()
