"""A serial queue of isolated workers, with durable state and cancellation."""

import json
import os
import queue
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from uuid import uuid4

from audio_studio.store import write_json
from audio_studio.database import Database, DatabaseError

TERMINAL = {"succeeded", "failed", "cancelled"}


def stop_worker_process(process: subprocess.Popen) -> None:
    """Stop the owned worker tree, including Windows venv's Python child."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW, check=False)
        if process.poll() is None:
            process.terminate()
    else:
        process.terminate()


class JobManager:
    def __init__(self, root: Path, on_success: Callable[[dict, dict], dict], database: Database | None = None):
        self.root, self.on_success = root, on_success
        self.database = database
        self.lock = threading.RLock()
        self.pending: queue.Queue[str | None] = queue.Queue()
        self.jobs: dict[str, dict] = {}
        self.requests: dict[str, dict] = {}
        self.process: subprocess.Popen | None = None
        self.current_id: str | None = None
        stored = {job["id"]: job for job in database.load_jobs()} if database else {}
        for path in sorted(root.glob("*/job.json")):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                continue
            request_path = path.parent / "request.json"
            if request_path.exists():
                self.requests[job["id"]] = json.loads(request_path.read_text(encoding="utf-8"))
            if job["id"] not in stored:
                stored[job["id"]] = job
                if database:
                    database.save_job(job, self.requests.get(job["id"]))
            elif stored[job["id"]]["status"] not in TERMINAL and job["status"] in TERMINAL:
                if job["id"] not in self.requests and stored[job["id"]].get("request"):
                    self.requests[job["id"]] = stored[job["id"]]["request"]
                stored[job["id"]] = job
                self._save(job)
        for job in sorted(stored.values(), key=lambda item: item.get("created_at", "")):
            if job.get("request"):
                self.requests[job["id"]] = job["request"]
            if job["status"] not in TERMINAL:
                job.update(status="failed", error="服务中断，任务未完成。请重新提交。")
                self._save(job)
            self.jobs[job["id"]] = job
        self.thread = threading.Thread(target=self._run, name="audio-jobs", daemon=True)
        self.thread.start()

    def _save(self, job: dict) -> None:
        job["updated_at"] = datetime.now(timezone.utc).isoformat()
        if self.database:
            request = self.requests.get(job["id"])
            self.database.save_job(job, request)
            if job["kind"] in {"train", "train_voice"} and request:
                payload = request.get("payload", {})
                result = job.get("result") or {}
                self.database.save_training_run({"id": job["id"], "job_id": job["id"], "name": payload.get("name", ""),
                    "dataset_id": payload.get("dataset_id"), "model_id": payload.get("model_id") or job.get("model_id"),
                    "status": job["status"], "run_dir": payload.get("run_dir", ""), "config": payload.get("config", {}),
                    "result": job.get("result"), "created_at": job["created_at"], "output_model_id": result.get("model_id")})
        write_json(self.root / job["id"] / "job.json", job)

    def _save_recovery(self, job: dict) -> None:
        """Keep a terminal recovery journal when PostgreSQL is unavailable."""
        try:
            self._save(job)
        except DatabaseError:
            write_json(self.root / job["id"] / "job.json", job)

    def submit(self, kind: str, payload: dict) -> dict:
        with self.lock:
            if kind == "download":
                existing = next((x for x in self.jobs.values() if x["kind"] == kind and x["status"] not in TERMINAL
                                 and x.get("model_id") == payload.get("model_id")), None)
                if existing:
                    return self.get(existing["id"])
            if sum(x["status"] not in TERMINAL for x in self.jobs.values()) >= 32:
                raise ValueError("任务队列已满，请等待现有任务完成")
            job_id = uuid4().hex
            job = {"id": job_id, "kind": kind, "status": "queued", "created_at": datetime.now(timezone.utc).isoformat(),
                   "error": None, "result": None, "name": payload.get("name"),
                   "model_id": payload.get("model_id", payload.get("model", {}).get("id"))}
            folder = self.root / job_id
            folder.mkdir(parents=True)
            request = {"kind": kind, "payload": payload, "job_id": job_id}
            write_json(folder / "request.json", request)
            self.requests[job_id] = request
            try:
                self._save(job)
            except Exception:
                self.requests.pop(job_id, None)
                raise
            self.jobs[job_id] = job
            self.pending.put(job_id)
            return self.get(job_id)

    def get(self, job_id: str) -> dict:
        with self.lock:
            job = dict(self.jobs[job_id])
        log_path = self.root / job_id / "worker.log"
        if log_path.exists():
            with log_path.open("rb") as handle:
                handle.seek(max(0, log_path.stat().st_size - 24000))
                job["logs"] = handle.read().decode("utf-8", errors="replace").splitlines()[-120:]
        else:
            job["logs"] = []
        return job

    def list(self) -> list[dict]:
        with self.lock:
            ids = list(self.jobs)[-100:][::-1]
        return [self.get(job_id) for job_id in ids]

    def cancel(self, job_id: str) -> dict:
        with self.lock:
            job = self.jobs[job_id]
            if job["status"] not in TERMINAL:
                job["status"] = "cancelled"
                job["finished_at"] = datetime.now(timezone.utc).isoformat()
                self._save_recovery(job)
                if self.current_id == job_id and self.process and self.process.poll() is None:
                    stop_worker_process(self.process)
        return self.get(job_id)

    def _run(self) -> None:
        while (job_id := self.pending.get()) is not None:
            folder = self.root / job_id
            try:
                with self.lock:
                    job = self.jobs[job_id]
                    if job["status"] == "cancelled":
                        continue
                    job["status"] = "running"
                    job["started_at"] = datetime.now(timezone.utc).isoformat()
                    self.current_id = job_id
                    self._save(job)
                with (folder / "worker.log").open("wb") as log_file:
                    with self.lock:
                        if job["status"] == "cancelled":
                            continue
                        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONUTF8="1")
                        self.process = subprocess.Popen([sys.executable, "-m", "audio_studio.worker", str(folder / "request.json")],
                            stdout=log_file, stderr=subprocess.STDOUT, env=env, cwd=str(Path(__file__).resolve().parent.parent),
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                    return_code = self.process.wait()
                with self.lock:
                    if job["status"] == "cancelled":
                        continue
                    result_path = folder / "result.json"
                    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
                    if return_code or "error" in result:
                        raise RuntimeError(result.get("error", f"工作进程异常退出 ({return_code})，请查看日志"))
                    job["result"] = self.on_success(json.loads((folder / "request.json").read_text(encoding="utf-8")), result)
                    job["status"] = "succeeded"
                    job["finished_at"] = datetime.now(timezone.utc).isoformat()
                    self._save(job)
            except Exception as exc:
                with self.lock:
                    if job["status"] != "cancelled":
                        job.update(status="failed", error=str(exc), finished_at=datetime.now(timezone.utc).isoformat())
                        self._save_recovery(job)
            finally:
                with self.lock:
                    self.process, self.current_id = None, None

    def close(self) -> None:
        with self.lock:
            for job_id, job in self.jobs.items():
                if job["status"] not in TERMINAL:
                    self.cancel(job_id)
            self.pending.put(None)
        self.thread.join(timeout=5)
        if self.process and self.process.poll() is None:
            stop_worker_process(self.process)
