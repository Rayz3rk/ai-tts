"""Real PostgreSQL integration of migration, application and job persistence."""

import io
import json
import os
from pathlib import Path
import shutil
import time
from uuid import uuid4

import numpy as np
import pytest
import soundfile as sf
from dotenv import dotenv_values
from fastapi.testclient import TestClient

from audio_studio.api import create_app
from audio_studio.db_setup import provision_database
from audio_studio.schemas import Preset
from audio_studio.store import Store, write_json


@pytest.fixture
def dsn():
    value = os.environ.get("AUDIO_STUDIO_TEST_DATABASE_URL")
    if not value:
        pytest.skip("Set AUDIO_STUDIO_TEST_DATABASE_URL for real PostgreSQL integration")
    return value


def make_wav(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    signal = np.sin(2 * np.pi * 220 * np.arange(48000) / 24000).astype("float32") * .2
    sf.write(path, signal, 24000)


def test_legacy_import_db_authority_audio_search_and_preset_deletion(tmp_path: Path, dsn):
    original = Store(tmp_path)
    custom = original.save_preset(Preset(name="历史音色"))
    audio_id, job_id = uuid4().hex, uuid4().hex
    audio = tmp_path / "audio" / f"{audio_id}.wav"
    make_wav(audio)
    write_json(tmp_path / "jobs" / job_id / "job.json", {"id": job_id, "kind": "synthesize", "status": "succeeded",
        "created_at": "2026-10-24T01:00:00+00:00", "error": None, "result": {"path": str(audio), "audio_url": f"/api/audio/{audio.name}"}})
    write_json(tmp_path / "jobs" / job_id / "request.json", {"kind": "synthesize", "payload": {"model": original.models()[0],
        "preset": custom, "text": "历史文本", "output_path": str(audio)}})
    with TestClient(create_app(tmp_path, database_url=dsn)) as client:
        assert client.get("/api/status").json()["storage"]["backend"] == "postgresql"
        page = client.get("/api/audio-assets").json()
        assert page["total"] == 1
        asset = page["items"][0]
        assert asset["job_id"] == job_id and asset["text"] == "历史文本"
        assert client.get(asset["audio_url"]).content == audio.read_bytes()
        edited = client.patch("/api/audio-assets/" + asset["id"], json={"name": "我的收藏", "favorite": True, "tags": ["有声书", "有声书"]})
        assert edited.status_code == 200, edited.text
        assert edited.json()["tags"] == ["有声书"]
        assert client.get("/api/audio-assets", params={"query": "有声书"}).json()["total"] == 1
        assert client.get("/api/audio-assets", params={"query": "' OR true --"}).json()["total"] == 0
        changed = {key: value for key, value in custom.items() if key != "builtin"}
        changed["name"] = "数据库修改"
        assert client.post("/api/presets", json=changed).status_code == 200
        assert client.get("/api/audio-assets?limit=0").status_code == 422
    assert (tmp_path / "catalogue.pre-postgresql.json").is_file()
    with TestClient(create_app(tmp_path, database_url=dsn)) as reopened:
        assert next(p for p in reopened.get("/api/presets").json() if p["id"] == custom["id"])["name"] == "数据库修改"
        page = reopened.get("/api/audio-assets").json()
        assert page["total"] == 1 and page["items"][0]["favorite"] is True
        assert page["items"][0]["name"] == "我的收藏"
        assert reopened.delete("/api/presets/" + custom["id"]).status_code == 200
        assert reopened.get("/api/audio-assets").json()["items"][0]["preset_id"] is None


def test_reference_upload_and_real_worker_persist_in_postgres(tmp_path: Path, dsn):
    with TestClient(create_app(tmp_path, database_url=dsn)) as client:
        sample = io.BytesIO()
        sf.write(sample, .2 * np.sin(2 * np.pi * np.arange(48000) * 220 / 24000), 24000, format="WAV")
        response = client.post("/api/references", files={"file": ("reference.wav", sample.getvalue(), "audio/wav")})
        assert response.status_code == 200, response.text
        assert client.get("/api/audio-assets", params={"query": "参考音频"}).json()["total"] == 1
        manager = client.app.state.jobs
        submitted = manager.submit("validate", {"csv_path": "missing.csv", "wavs_dir": "missing"})
        deadline = time.monotonic() + 20
        while manager.get(submitted["id"])["status"] not in {"succeeded", "failed"} and time.monotonic() < deadline:
            time.sleep(.05)
        assert manager.get(submitted["id"])["status"] == "succeeded"
        assert client.app.state.store.database.load_jobs()[0]["request"]["kind"] == "validate"
    with TestClient(create_app(tmp_path, database_url=dsn)) as reopened:
        assert reopened.get("/api/jobs/" + submitted["id"]).json()["status"] == "succeeded"


def test_database_workspace_survives_directory_move(tmp_path: Path, dsn):
    original_root = tmp_path / "original"
    original = Store(original_root, database_url=dsn)
    audio = original_root / "audio" / f"{uuid4().hex}.wav"
    make_wav(audio)
    asset = original.record_audio(audio, {"name": "迁移素材"})
    moved = tmp_path / "moved"
    shutil.copytree(original_root, moved)
    reopened = Store(moved, database_url=dsn)
    assert reopened.database.workspace_id == original.database.workspace_id
    current = reopened.database.get_audio(asset["id"])
    assert Path(current["path"]).is_relative_to(moved)
    assert Path(current["path"]).is_file()


def test_provisioning_creates_dedicated_role_and_is_idempotent(tmp_path: Path, dsn):
    import psycopg
    from psycopg.conninfo import conninfo_to_dict
    parameters = conninfo_to_dict(dsn)
    if parameters.get("port") != "55432":
        pytest.skip("Provisioning test only runs on the isolated local test cluster")
    config = tmp_path / ".env"
    config.write_text("AUDIO_STUDIO_PG_ADMIN_HOST=127.0.0.1\nAUDIO_STUDIO_PG_ADMIN_PORT=55432\n"
                      "AUDIO_STUDIO_PG_ADMIN_USER=postgres\nAUDIO_STUDIO_PG_ADMIN_PASSWORD='test-only'\n", encoding="utf-8")
    suffix = uuid4().hex[:8]
    created = provision_database(config, "test_audio_" + suffix, "test_audio_app_" + suffix)
    assert created["created"] is True
    values = dotenv_values(config, interpolate=False)
    assert values["AUDIO_STUDIO_PG_ADMIN_PASSWORD"] == ""
    with psycopg.connect(values["AUDIO_STUDIO_DATABASE_URL"]) as connection:
        flags = connection.execute("SELECT rolsuper, rolcreatedb, rolcreaterole FROM pg_roles WHERE rolname=current_user").fetchone()
        assert flags == (False, False, False)
    again = provision_database(config)
    assert again["created"] is False and again["database"] == created["database"]


def test_svc_dataset_without_csv_and_audio_inputs_persist_in_postgres(tmp_path: Path, dsn):
    recordings = tmp_path / "recordings"
    make_wav(recordings / "voice.wav")
    with TestClient(create_app(tmp_path / "studio", database_url=dsn)) as client:
        status = client.get("/api/covers/status")
        assert status.status_code == 200
        assert status.json()["assets"]["engine"] == "rvc"
        report = client.post("/api/covers/datasets/validate", json={"wavs_dir": str(recordings)})
        assert report.status_code == 200, report.text
        assert report.json()["valid"] and report.json()["csv_path"] == ""
        database = client.app.state.store.database
        with database._connection() as connection:
            row = connection.execute("SELECT csv_path, metadata->>'task' AS task FROM audio_studio.datasets "
                "WHERE workspace_id=%s AND id=%s", (database.workspace_id, report.json()["dataset_id"])).fetchone()
        assert row == {"csv_path": "", "task": "svc"}
        uploaded = client.post("/api/covers/audio", files={"file": ("vocal.wav", (recordings / "voice.wav").read_bytes(), "audio/wav")})
        assert uploaded.status_code == 200, uploaded.text
        assert database.get_audio(uploaded.json()["audio_asset_id"])["kind"] == "source"
