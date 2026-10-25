"""Run `python -m audio_studio --help` for local CLI commands."""

import argparse
import json
import os
import threading
from pathlib import Path


def main() -> None:
    from audio_studio.config import load_local_config
    load_local_config()
    parser = argparse.ArgumentParser(description="本地音频工作台")
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("AUDIO_STUDIO_DATA", "studio_data")))
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="启动 Web 工作台和 API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=7861)
    serve.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    commands.add_parser("status", help="查看模型和依赖状态")
    download = commands.add_parser("download", help="显式下载模型")
    download.add_argument("engine", choices=["kokoro", "f5"])
    validate = commands.add_parser("validate", help="检查录音和 CSV")
    validate.add_argument("csv_path", type=Path)
    validate.add_argument("wavs_dir", type=Path)
    synth = commands.add_parser("synthesize", help="使用保存的预设生成音频")
    synth.add_argument("--preset", default="zh-narration")
    synth.add_argument("--text", required=True)
    synth.add_argument("--output", type=Path, required=True)
    synth.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    train = commands.add_parser("train", help="以 JSON 配置执行本地微调")
    train.add_argument("config", type=Path)
    train.add_argument("--run-dir", type=Path, required=True)
    database = commands.add_parser("db-init", help="创建专用 PostgreSQL 数据库、应用账号并迁移现有记录")
    database.add_argument("--env-file", type=Path, default=Path(".env"))
    database.add_argument("--database", default="audio_studio")
    database.add_argument("--role", default="audio_studio_app")
    args = parser.parse_args()
    if args.command == "db-init":
        from audio_studio.db_setup import provision_database
        from audio_studio.store import Store, write_json
        from audio_studio.jobs import JobManager
        from audio_studio.history import import_history
        try:
            result = provision_database(args.env_file, args.database, args.role)
            # Explicit values override any stale environment inherited by this CLI.
            from dotenv import dotenv_values
            settings = dotenv_values(args.env_file, interpolate=False)
            store = Store(args.data_dir, database_url=settings["AUDIO_STUDIO_DATABASE_URL"])
            jobs = JobManager(store.root / "jobs", lambda request, value: value, database=store.database)
            try:
                result.update(import_history(store, jobs.jobs))
                result.update(workspace_id=store.database.workspace_id, models=len(store.models()), presets=len(store.presets()), jobs=len(jobs.jobs))
            finally:
                jobs.close()
            write_json(store.root / "database-setup.json", result)
        except (ValueError, RuntimeError) as exc:
            parser.error(str(exc))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "serve":
        if args.host not in {"127.0.0.1", "localhost", "::1"} and not os.environ.get("AUDIO_STUDIO_TOKEN"):
            parser.error("对外监听前请设置 AUDIO_STUDIO_TOKEN")
        import uvicorn
        from audio_studio.api import create_app
        uvicorn.run(create_app(args.data_dir, args.device), host=args.host, port=args.port)
        return
    from audio_studio.store import Store
    store = Store(args.data_dir)
    if args.command == "status":
        import importlib.util
        from audio_studio.engines import model_readiness
        result = {"models": [dict(x, **model_readiness(x)) for x in store.models()],
                  "storage": store.storage_status(),
                  "dependencies": {x: importlib.util.find_spec(x) is not None for x in ("torch", "f5_tts", "kokoro")}}
    elif args.command == "download":
        from audio_studio.engines import download_model
        model = store.get("models", args.engine + "-base")
        result = download_model(args.engine, Path(model["path"]), print)
        store.update_model(model["id"], result)
    elif args.command == "validate":
        from audio_studio.datasets import validate_dataset
        result = validate_dataset(args.csv_path, args.wavs_dir)
        result["dataset_id"] = store.record_dataset(result)
    elif args.command == "synthesize":
        from audio_studio.engines import synthesize
        preset = store.get("presets", args.preset)
        result = synthesize(store.get("models", preset["model_id"]), preset, args.text, args.output, args.device)
        asset = store.record_audio(Path(result["path"]), {"kind": "generated", "model_id": preset["model_id"],
            "preset_id": preset["id"], "text": args.text, "name": args.text[:80], "preset_snapshot": preset,
            "parameters": {key: preset.get(key) for key in ("voice", "language", "speed", "steps", "seed")}})
        if asset:
            result["audio_asset_id"] = asset["id"]
    else:
        from audio_studio.training import run_training
        config = json.loads(args.config.read_text(encoding="utf-8"))
        allowed = {"csv_path", "wavs_dir", "base_checkpoint", "vocab_path", "epochs", "batch_size",
                   "learning_rate", "device", "grad_accumulation_steps"}
        if not isinstance(config, dict) or set(config) - allowed:
            parser.error("训练配置必须是 JSON 对象，且仅包含文档中的公开训练参数")
        if not {"csv_path", "wavs_dir", "base_checkpoint", "vocab_path"}.issubset(config):
            parser.error("训练配置需要 csv_path、wavs_dir、base_checkpoint 和 vocab_path")
        result = run_training(config, args.run_dir, print, threading.Event())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.command == "validate" and not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
