"""Music input validation and download-integrity regressions, no neural substitutes."""

import hashlib
import io
from pathlib import Path

from fastapi.testclient import TestClient

from audio_studio.api import create_app
from audio_studio.hub_download import download_file


def test_music_routes_gate_missing_runtime_and_reject_invalid_duration(tmp_path):
    with TestClient(create_app(tmp_path, database_url="")) as client:
        status = client.get("/api/music/status").json()
        assert status["model"]["engine"] == "ace_step" and not status["model"]["ready"]
        assert status["contains_accompaniment"] and not status["supports_target_voice"]
        assert len(client.get("/api/presets").json()) == 8
        request = {"lyrics": "你好，阳光", "duration": 30}
        assert client.post("/api/music/generate", json=request).status_code == 400
        assert not client.get("/api/jobs").json()
        assert client.post("/api/music/generate", json=dict(request, duration=500)).status_code == 422
        assert client.get("/static/music.js").status_code == 200


def test_download_exposes_only_verified_bytes_and_reuses_complete_target(tmp_path, monkeypatch):
    data = b"official model data" * 100
    digest = hashlib.sha256(data).hexdigest()
    target = tmp_path / "model.safetensors"

    class Reply(io.BytesIO):
        status = 200
        headers = {}

    class Opener:
        def open(self, *args, **kwargs):
            return Reply(data)

    monkeypatch.setattr("audio_studio.hub_download.build_opener", lambda *args: Opener())
    download_file("owner/repo", "revision", "model.safetensors", target, len(data), digest, lambda _: None)
    assert target.read_bytes() == data and not target.with_name(target.name + ".download").exists()
    monkeypatch.setattr("audio_studio.hub_download.build_opener", lambda *args: (_ for _ in ()).throw(AssertionError("unexpected download")))
    download_file("owner/repo", "revision", "model.safetensors", target, len(data), digest, lambda _: None)
