"""Offline contract tests and opt-in tests against an actual PostgreSQL server.

Set AUDIO_STUDIO_TEST_DATABASE_URL to a disposable application database to run
the integration tests. Every test owns a random workspace and removes it after
completion; there is no in-memory database substitute.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from importlib.resources import files
from uuid import uuid4

import pytest

import audio_studio.database as database_module
from audio_studio.database import Database, DatabaseError, _audio_values, _created_at, _public_row, _rebase_paths


def test_database_configuration_errors_do_not_echo_inputs():
    with pytest.raises(ValueError, match="AUDIO_STUDIO_DATABASE_URL"):
        Database("")
    with pytest.raises(ValueError, match="workspace_id"):
        Database("postgresql://private:secret@localhost/example", workspace_id="")
    assert "secret" not in repr(Database("postgresql://private:secret@localhost/example"))


def test_missing_driver_has_actionable_credential_safe_error(monkeypatch):
    monkeypatch.setattr(database_module, "psycopg", None)
    repository = Database("postgresql://private:secret@localhost/example")
    with pytest.raises(DatabaseError) as caught:
        repository.list_models()
    assert "psycopg" in str(caught.value)
    assert "secret" not in str(caught.value)


def test_driver_errors_are_redacted_without_public_chaining(monkeypatch):
    if database_module.psycopg is None:
        pytest.skip("psycopg is unavailable")

    def failed_connection(*args, **kwargs):
        raise database_module.psycopg.OperationalError("password secret from private connection string")

    monkeypatch.setattr(database_module.psycopg, "connect", failed_connection)
    with pytest.raises(DatabaseError) as caught:
        Database("postgresql://private:secret@localhost/example").list_models()
    assert "secret" not in str(caught.value)
    assert caught.value.__suppress_context__
    assert caught.value.__cause__ is None


def test_relational_columns_override_metadata_and_snapshot():
    timestamp = datetime(2026, 10, 24, tzinfo=timezone.utc)
    row = {"workspace_id": "internal", "id": "real-id", "status": "succeeded", "created_at": timestamp,
           "metadata": {"id": "stale", "status": "queued", "custom": {"labels": ["中文"]}},
           "snapshot": {"id": "old-snapshot", "status": "running", "result": None},
           "result": {"audio_url": "/api/audio/test.wav"}, "updated_at": timestamp}
    item = _public_row(row)
    assert item["id"] == "real-id"
    assert item["status"] == "succeeded"
    assert item["result"]["audio_url"].endswith("test.wav")
    assert item["custom"] == {"labels": ["中文"]}
    assert item["created_at"] == timestamp.isoformat()
    assert not {"workspace_id", "metadata", "snapshot", "updated_at"} & item.keys()


def test_catalogue_dictionaries_do_not_gain_strict_payload_fields():
    item = _public_row({"id": "preset", "name": "测试", "position": 1,
                        "created_at": datetime.now(timezone.utc), "metadata": {"steps": 32}})
    assert item == {"id": "preset", "name": "测试", "steps": 32}


@pytest.mark.parametrize("updates", [
    {"tags": "not-an-array"}, {"tags": [123]}, {"favorite": "false"},
    {"duration_seconds": -1}, {"duration_seconds": float("nan")}, {"duration_seconds": float("inf")},
    {"sample_rate": 0}, {"sample_rate": True},
])
def test_audio_validation_rejects_invalid_database_values(updates):
    with pytest.raises(ValueError):
        _audio_values({"path": r"E:\audio\sample.wav", **updates})


def test_audio_normalization_handles_windows_paths_and_utc_dates():
    item = _audio_values({"path": r"E:\audio\sample.wav", "tags": ["中文", "旁白"], "created_at": "2026-10-24T08:00:00Z"})
    assert item["filename"] == "sample.wav"
    assert item["name"] == "sample.wav"
    assert item["created_at"] == datetime(2026, 10, 24, 8, tzinfo=timezone.utc)
    assert _created_at({"created_at": "2026-10-24T08:00:00"}).tzinfo == timezone.utc
    with pytest.raises(ValueError, match="ISO"):
        _created_at({"created_at": "invalid"})


@pytest.mark.parametrize("kwargs", [{"limit": 0}, {"limit": 201}, {"limit": True}, {"offset": -1}, {"query": None}])
def test_pagination_validation_runs_before_connecting(kwargs):
    with pytest.raises(ValueError):
        Database("unused").list_audio(**kwargs)


def test_migration_is_packaged_with_all_relational_entities():
    source = files("audio_studio").joinpath("migrations", "001_initial.sql").read_text(encoding="utf-8")
    for table in ("workspaces", "models", "presets", "jobs", "audio_assets", "datasets", "training_runs"):
        assert f"CREATE TABLE IF NOT EXISTS audio_studio.{table}" in source
    assert "FOREIGN KEY (workspace_id, preset_id)" in source
    assert "ON DELETE SET NULL (preset_id)" in source


def test_path_rebasing_handles_nested_windows_metadata_and_keeps_external_paths():
    item = {"path": r"E:\old\studio\models\f5", "reference_audio": r"D:\external\reference.wav",
            "request": {"payload": {"output_path": r"E:\old\studio\audio\a.wav"}},
            "samples": [{"audio_path": r"E:\old\studio\wavs\one.wav"}], "text": r"E:\old\studio\words"}
    rebased = _rebase_paths(item, r"E:\old\studio", "/mnt/studio")
    assert rebased["path"] == "/mnt/studio/models/f5"
    assert rebased["reference_audio"] == item["reference_audio"]
    assert rebased["request"]["payload"]["output_path"] == "/mnt/studio/audio/a.wav"
    assert rebased["samples"][0]["audio_path"] == "/mnt/studio/wavs/one.wav"
    assert rebased["text"] == item["text"]
    assert item["path"] == r"E:\old\studio\models\f5"


@pytest.fixture
def repository():
    dsn = os.environ.get("AUDIO_STUDIO_TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("AUDIO_STUDIO_TEST_DATABASE_URL is not configured; PostgreSQL integration was not run")
    if database_module.psycopg is None:
        pytest.fail("PostgreSQL integration was requested but psycopg is unavailable")
    repository = Database(dsn, workspace_id="pytest-" + uuid4().hex)
    repository.migrate()
    try:
        yield repository
    finally:
        with repository._connection() as connection:
            connection.execute("DELETE FROM audio_studio.workspaces WHERE id = %s", (repository.workspace_id,))


def seed(repository: Database) -> None:
    repository.initialize_catalogue(
        [{"id": "kokoro-base", "name": "Kokoro", "engine": "kokoro", "path": "/data/models/kokoro", "builtin": True},
         {"id": "f5-base", "name": "F5", "engine": "f5", "path": "/data/models/f5", "builtin": True}],
        [{"id": "zh-narration", "name": "中文旁白", "engine": "kokoro", "model_id": "kokoro-base",
          "category": "narration", "voice": "zf_xiaobei", "language": "z", "speed": 1.0, "steps": 32, "builtin": True}])


def test_postgres_migrations_and_catalogue_are_idempotent(repository):
    seed(repository)
    repository.migrate()
    repository.initialize_catalogue([], [])
    reopened = Database(repository._dsn, workspace_id=repository.workspace_id)
    assert [item["id"] for item in reopened.list_models()] == ["kokoro-base", "f5-base"]
    assert reopened.get("presets", "zh-narration")["steps"] == 32
    repository.initialize_catalogue([{"id": "ignored", "name": "Ignored", "engine": "f5"}], [])
    assert len(repository.list_models()) == 2
    with pytest.raises(ValueError, match="内置"):
        repository.delete_preset("zh-narration")
    with pytest.raises(ValueError, match="内置"):
        repository.save_preset(dict(repository.get("presets", "zh-narration"), name="不可覆盖"))


def test_postgres_original_music_model_job_and_audio_are_linked(repository):
    seed(repository)
    model = repository.upsert_model({"id": "ace-step-15", "name": "自动创作", "engine": "ace_step",
        "path": "/data/models/ace-step", "runtime_path": "/data/runtimes/ace-step"})
    repository.save_job({"id": "music-job", "kind": "compose", "status": "succeeded", "model_id": model["id"]})
    audio = repository.register_audio({"path": "/data/audio/song.wav", "model_id": model["id"],
        "job_id": "music-job", "kind": "music", "contains_accompaniment": True,
        "duration_seconds": 30, "sample_rate": 48000, "tags": ["原创歌曲"]})
    repository.migrate()
    assert repository.get("models", model["id"])["runtime_path"] == "/data/runtimes/ace-step"
    assert repository.get_audio(audio["id"])["contains_accompaniment"]


def test_postgres_jobs_preserve_requests_results_and_snapshots(repository):
    seed(repository)
    job = {"id": "job-a", "kind": "synthesize", "status": "queued", "model_id": "kokoro-base",
           "name": "你好", "error": None, "result": None, "created_at": "2026-10-24T08:00:00Z", "custom": "snapshot"}
    repository.save_job(job, {"text": "原始文本", "preset": {"id": "zh-narration", "speed": 1.0}})
    repository.save_job(dict(job, status="succeeded", result={"path": "/data/audio/a.wav", "duration_seconds": 1.25}))
    saved = Database(repository._dsn, workspace_id=repository.workspace_id).load_jobs()[0]
    assert saved["status"] == "succeeded"
    assert saved["request"]["payload"]["text"] == "原始文本"
    assert saved["request"]["payload"]["preset"]["speed"] == 1.0
    assert saved["result"]["duration_seconds"] == 1.25
    assert saved["custom"] == "snapshot"
    assert saved["created_at"] == "2026-10-24T08:00:00+00:00"


def test_postgres_audio_history_search_pagination_and_custom_preset_deletion(repository):
    seed(repository)
    preset = repository.save_preset({"name": "自定义", "engine": "kokoro", "model_id": "kokoro-base", "voice": "zf_xiaobei"})
    repository.save_job({"id": "audio-job", "kind": "synthesize", "status": "succeeded", "model_id": "kokoro-base"})
    first = repository.register_audio({"path": "/data/audio/one.wav", "text": "中文旁白 100%_",
        "model_id": "kokoro-base", "preset_id": preset["id"], "job_id": "audio-job", "duration_seconds": 1.0, "sample_rate": 24000,
        "created_at": "2026-10-24T08:00:00Z", "extra": {"source": "synthesis"}})
    changed = repository.update_audio(first["id"], {"name": "我的音频", "tags": ["新闻", "中文"], "favorite": True})
    assert changed["favorite"] and changed["extra"] == {"source": "synthesis"}
    rescanned = repository.register_audio({"path": "/data/audio/one.wav", "sample_rate": 24000})
    assert rescanned["id"] == first["id"]
    assert rescanned["name"] == "我的音频" and rescanned["favorite"]
    repository.register_audio({"path": "/data/audio/two.wav", "text": "other", "created_at": "2026-10-24T09:00:00Z"})
    assert repository.list_audio(query="100%_")["total"] == 1
    assert repository.list_audio(query="新闻")["items"][0]["id"] == first["id"]
    assert repository.list_audio(query="' OR 1=1 --")["total"] == 0
    page = repository.list_audio(limit=1, offset=1)
    assert page["total"] == 2 and page["items"][0]["id"] == first["id"]
    assert repository.list_audio(offset=99)["total"] == 2
    repository.delete_preset(preset["id"])
    assert repository.get_audio(first["id"])["preset_id"] is None


def test_postgres_workspace_isolation_and_foreign_keys(repository):
    seed(repository)
    other = Database(repository._dsn, workspace_id="pytest-" + uuid4().hex)
    other.migrate()
    try:
        assert other.list_models() == []
        with pytest.raises(KeyError):
            other.get("models", "kokoro-base")
        with pytest.raises(DatabaseError):
            other.save_preset({"id": "cross-workspace", "name": "拒绝", "model_id": "kokoro-base", "engine": "kokoro"})
        repository.register_audio({"id": "private-audio", "path": "/data/audio/private.wav"})
        assert other.list_audio()["total"] == 0
        with pytest.raises(KeyError):
            other.get_audio("private-audio")
    finally:
        with other._connection() as connection:
            connection.execute("DELETE FROM audio_studio.workspaces WHERE id = %s", (other.workspace_id,))


def test_postgres_datasets_and_training_runs_preserve_reports(repository):
    seed(repository)
    dataset = repository.save_dataset({"id": "dataset-a", "csv_path": "/data/train.csv", "wavs_dir": "/data/wavs",
        "valid": True, "row_count": 2, "sample_count": 2, "total_duration_seconds": 7.5, "warnings": ["短数据集"]})
    repository.save_job({"id": "train-job", "kind": "train", "status": "running", "model_id": "f5-base"})
    run = repository.save_training_run({"id": "run-a", "dataset_id": dataset["id"], "model_id": "f5-base", "job_id": "train-job",
        "status": "succeeded", "run_dir": "/data/runs/a", "config": {"epochs": 2}, "result": {"loss": 0.2}, "extra": "kept"})
    assert dataset["warnings"] == ["短数据集"]
    assert run["config"] == {"epochs": 2} and run["result"] == {"loss": 0.2} and run["extra"] == "kept"
    with repository._connection() as connection:
        stored = connection.execute("SELECT sample_count FROM audio_studio.datasets WHERE workspace_id = %s AND id = %s",
                                    (repository.workspace_id, dataset["id"])).fetchone()
    assert stored["sample_count"] == 2


def test_postgres_simultaneous_audio_registration_deduplicates(repository):
    def register(index):
        return repository.register_audio({"path": "/data/audio/shared.wav", "text": "重复导入", "sample_rate": 24000})

    with ThreadPoolExecutor(max_workers=4) as pool:
        ids = list(pool.map(register, range(8)))
    assert len({item["id"] for item in ids}) == 1
    assert repository.list_audio()["total"] == 1


def test_postgres_rebasing_updates_binding_and_keeps_audio_ids(repository):
    seed(repository)
    repository.set_workspace_root("/data")
    repository.upsert_model(dict(repository.get("models", "f5-base"), checkpoint_path="/data/models/f5/model.pt"))
    custom = repository.save_preset({"name": "外部参考", "engine": "f5", "model_id": "f5-base",
                                     "reference_audio": "/external/reference.wav"})
    audio = repository.register_audio({"path": "/data/audio/a.wav"})
    repository.save_job({"id": "rebased-job", "kind": "synthesize", "status": "succeeded",
                         "result": {"path": "/data/audio/a.wav"}}, {"output_path": "/data/audio/a.wav"})
    repository.rebase_paths("/data", "/moved")
    assert repository.get_workspace()["root_path"] == "/moved"
    assert repository.get("models", "f5-base")["checkpoint_path"] == "/moved/models/f5/model.pt"
    assert repository.get("presets", custom["id"])["reference_audio"] == "/external/reference.wav"
    assert repository.get_audio(audio["id"])["path"] == "/moved/audio/a.wav"
    assert repository.load_jobs()[0]["request"]["payload"]["output_path"] == "/moved/audio/a.wav"
