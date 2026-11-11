"""Reuse creative inputs without returning private engine configuration."""

from fastapi.testclient import TestClient
from audio_studio.api import create_app
from audio_studio.store import write_json


def test_reuse_survives_reload_and_keeps_only_editor_fields(tmp_path):
    folder = tmp_path / "jobs" / "saved"
    write_json(folder / "job.json", {"id": "saved", "kind": "sing", "status": "succeeded", "result": {}})
    score = {"title": "原作", "notes": [{"lyric": "你", "midi": 60, "start": 0, "duration": .5}]}
    write_json(folder / "request.json", {"kind": "sing", "payload": {"score": score, "steps": 20,
        "target_model": {"id": "voice-a", "private_config": "not-for-editor"}, "secret": "not-for-editor"}})
    with TestClient(create_app(tmp_path, database_url="")) as client:
        reply = client.get("/api/jobs/saved/inputs")
        assert reply.status_code == 200
        assert reply.json() == {"kind": "sing", "values": {"score": score, "steps": 20, "target_voice_id": "voice-a"}}
        assert client.get("/api/jobs/missing/inputs").status_code == 404
        assert client.get("/static/workbench.js").status_code == 200
        assert client.get("/static/workbench.css").status_code == 200


def test_training_reuse_restores_data_and_tuning_without_engine_secrets(tmp_path):
    folder = tmp_path / "jobs" / "training"
    write_json(folder / "job.json", {"id": "training", "kind": "train_voice", "status": "failed", "result": {}})
    write_json(folder / "request.json", {"kind": "train_voice", "payload": {"name": "我的音色", "config": {
        "wavs_dir": "/data/recordings", "epochs": 20, "learning_rate": .0001, "secret": "hidden"}}})
    with TestClient(create_app(tmp_path, database_url="")) as client:
        assert client.get("/api/jobs/training/inputs").json() == {"kind": "train_voice", "values": {
            "name": "我的音色", "wavs_dir": "/data/recordings", "epochs": 20, "learning_rate": .0001}}
