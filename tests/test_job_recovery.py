"""Reconcile terminal recovery journals with an actual PostgreSQL repository."""

import os
from pathlib import Path
from uuid import uuid4

import pytest

from audio_studio.database import Database
from audio_studio.jobs import JobManager
from audio_studio.store import write_json


@pytest.fixture
def repository():
    dsn = os.environ.get("AUDIO_STUDIO_TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("Set AUDIO_STUDIO_TEST_DATABASE_URL for real PostgreSQL integration")
    database = Database(dsn, workspace_id="recovery-" + uuid4().hex)
    database.migrate()
    try:
        yield database
    finally:
        with database._connection() as connection:
            connection.execute("DELETE FROM audio_studio.workspaces WHERE id = %s", (database.workspace_id,))


@pytest.mark.parametrize("local_request", [True, False])
def test_terminal_training_journal_repairs_database_and_training_record(tmp_path: Path, repository: Database, local_request: bool):
    job_id = uuid4().hex
    request = {"kind": "train", "job_id": job_id, "payload": {
        "name": "恢复验收", "run_dir": str(tmp_path / "training"), "config": {"steps": 1},
    }}
    running = {"id": job_id, "kind": "train", "status": "running", "created_at": "2026-10-24T01:00:00+00:00",
               "result": None, "error": None}
    repository.save_job(running, request)
    repository.save_training_run({"id": job_id, "job_id": job_id, "status": "running"})
    recovered = dict(running, status="succeeded", result={"steps": 1, "loss": .25},
                     finished_at="2026-10-24T01:01:00+00:00")
    write_json(tmp_path / job_id / "job.json", recovered)
    if local_request:
        write_json(tmp_path / job_id / "request.json", request)

    manager = JobManager(tmp_path, lambda request, result: result, repository)
    try:
        saved = repository.load_jobs()[0]
        assert saved["status"] == "succeeded"
        assert saved["result"] == recovered["result"]
        assert saved["finished_at"] == recovered["finished_at"]
        assert saved["request"] == request
        with repository._connection() as connection:
            training = connection.execute(
                "SELECT status, config, result FROM audio_studio.training_runs WHERE workspace_id = %s AND id = %s",
                (repository.workspace_id, job_id),
            ).fetchone()
        assert training["status"] == "succeeded"
        assert training["config"] == request["payload"]["config"]
        assert training["result"] == recovered["result"]
    finally:
        manager.close()

    restarted = JobManager(tmp_path, lambda request, result: result, repository)
    try:
        assert restarted.get(job_id)["status"] == "succeeded"
        assert repository.load_jobs()[0]["result"] == recovered["result"]
    finally:
        restarted.close()
