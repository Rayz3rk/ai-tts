"""Import existing JSON history into PostgreSQL without moving audio bytes."""

import json
import logging
from pathlib import Path

from audio_studio.store import Store
from audio_studio.database import DatabaseError

logger = logging.getLogger(__name__)


def import_history(store: Store, jobs: dict[str, dict]) -> dict:
    if not store.database:
        return {"audio_imported": 0}
    database = store.database
    existing_paths: set[str] = set()
    offset = 0
    while True:
        page = database.list_audio(limit=200, offset=offset)
        existing_paths.update(item["path"] for item in page["items"])
        offset += len(page["items"])
        if offset >= page["total"]:
            break
    sources: dict[Path, dict] = {}
    for job in jobs.values():
        request = job.get("request")
        if not request:
            request_path = store.root / "jobs" / job["id"] / "request.json"
            if request_path.exists():
                request = json.loads(request_path.read_text(encoding="utf-8"))
        payload = (request or {}).get("payload", {})
        result = job.get("result") or {}
        if job["kind"] in {"synthesize", "cover", "sing"} and job["status"] == "succeeded":
            path = Path(result.get("path") or payload.get("output_path", ""))
            if path.is_file():
                preset = payload.get("preset", {})
                preset_id = preset.get("id")
                if preset_id:
                    try:
                        store.get("presets", preset_id)
                    except KeyError:
                        preset_id = None
                sources[path.resolve()] = {"kind": "cover" if job["kind"] == "cover" else "generated", "job_id": job["id"], "model_id": (payload.get("model") or {}).get("id"),
                    "preset_id": preset_id, "text": payload.get("text", ""), "name": payload.get("text", "")[:80] or path.name,
                    "created_at": job["created_at"], "preset_snapshot": preset, "parameters": {key: preset.get(key) for key in ("voice", "language", "speed", "seed", "steps")}}
        if job["kind"] in {"train", "train_voice"}:
            config = payload.get("config", {})
            dataset_id = payload.get("dataset_id")
            if not dataset_id and config.get("wavs_dir") and job["kind"] == "train_voice":
                from audio_studio.svc_training import validate_voice_dataset
                dataset_id = store.record_dataset(validate_voice_dataset(Path(config["wavs_dir"])))
            elif not dataset_id and config.get("csv_path") and config.get("wavs_dir"):
                from audio_studio.datasets import validate_dataset
                dataset_id = store.record_dataset(validate_dataset(Path(config["csv_path"]), Path(config["wavs_dir"])))
            database.save_training_run({"id": job["id"], "job_id": job["id"], "name": payload.get("name", ""),
                "dataset_id": dataset_id, "model_id": payload.get("model_id") or job.get("model_id") or ("rvc-support" if job["kind"] == "train_voice" else "f5-base"),
                "status": job["status"], "run_dir": payload.get("run_dir", ""), "config": config, "result": job.get("result"),
                "created_at": job["created_at"], "output_model_id": result.get("model_id")})
    for folder, kind, tags in ((store.root / "audio", "generated", []), (store.root / "references", "reference", ["参考音频"]),
                               (store.root / "inputs", "source", ["翻唱输入"]),
                               (store.root.parent / "outputs", "sample", ["验收样例"])):
        if folder.is_dir():
            for path in folder.rglob("*.wav"):
                sources.setdefault(path.resolve(), {"kind": kind, "tags": tags, "name": path.stem})
    imported = 0
    for path, metadata in sources.items():
        if str(path) in existing_paths:
            continue
        try:
            store.record_audio(path, metadata)
        except DatabaseError:
            raise
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning("音频历史登记失败：%s (%s)", path.name, type(exc).__name__)
        else:
            imported += 1
    return {"audio_imported": imported}
