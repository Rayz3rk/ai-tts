"""Local web UI and HTTP API, usable without loading any TTS models."""

import importlib.util
import os
import secrets
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, Query
from fastapi.responses import FileResponse, JSONResponse

from audio_studio.jobs import JobManager
from audio_studio.schemas import AudioAssetUpdate, DatasetRequest, Preset, RegisterModel, SynthesisRequest, TrainingRequest
from audio_studio.store import Store
from audio_studio.database import DatabaseError


@lru_cache(maxsize=1)
def hardware() -> dict:
    if importlib.util.find_spec("torch") is None:
        return {"device": "cpu", "gpu": None}
    import torch
    return {"device": "cuda" if torch.cuda.is_available() else "cpu",
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def create_app(data_dir: Path | None = None, device: str | None = None, database_url: str | None = None) -> FastAPI:
    root = Path(data_dir or os.environ.get("AUDIO_STUDIO_DATA", "studio_data")).resolve()
    selected_device = device or os.environ.get("AUDIO_STUDIO_DEVICE", "auto")
    if selected_device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device 必须为 auto、cpu 或 cuda")
    store = Store(root, database_url=database_url)
    token = os.environ.get("AUDIO_STUDIO_TOKEN", "")

    def completed(request: dict, result: dict) -> dict:
        payload = request["payload"]
        if request["kind"] == "download":
            store.update_model(payload["model_id"], result)
        elif request["kind"] == "synthesize":
            preset = payload["preset"]
            preset_id = preset.get("id")
            if preset_id:
                try:
                    store.get("presets", preset_id)
                except KeyError:
                    preset_id = None
            asset = store.record_audio(Path(result["path"]), {"kind": "generated", "job_id": request.get("job_id"),
                "model_id": payload["model"]["id"], "preset_id": preset_id, "text": payload["text"],
                "name": payload["text"][:80], "preset_snapshot": preset,
                "parameters": {key: preset.get(key) for key in ("voice", "language", "speed", "steps", "seed")}})
            if asset:
                result["audio_asset_id"] = asset["id"]
        elif request["kind"] == "train":
            model = store.register_model(payload["name"], result["checkpoint_path"], result["vocab_path"], payload["vocoder_path"])
            result["model_id"] = model["id"]
            result["vocoder_path"] = payload["vocoder_path"]
        elif request["kind"] == "train_voice":
            model = store.register_voice_model(payload["name"], result["checkpoint_path"], result.get("index_path", ""))
            result.update(model_id=model["id"], engine="rvc")
        elif request["kind"] == "cover":
            metadata = {"kind": "cover", "job_id": request.get("job_id"), "model_id": payload["model"]["id"],
                "name": "翻唱 · " + payload["model"]["name"], "tags": ["翻唱"],
                "source_audio": payload["source_audio"], "accompaniment_audio": payload.get("accompaniment_audio", ""),
                "parameters": {key: payload.get(key) for key in ("pitch_shift", "index_rate", "speaker_id", "seed", "vocal_gain", "accompaniment_gain")}}
            asset = store.record_audio(Path(result["path"]), metadata)
            if asset:
                result["audio_asset_id"] = asset["id"]
            if result.get("vocal_path"):
                vocal = store.record_audio(Path(result["vocal_path"]), dict(metadata, name=metadata["name"] + " · 干声", tags=["翻唱", "干声"]))
                if vocal:
                    result["vocal_asset_id"] = vocal["id"]
        elif request["kind"] == "sing":
            target = payload.get("target_model")
            metadata = {"kind": "singing", "job_id": request.get("job_id"),
                "model_id": target["id"] if target else payload["model"]["id"],
                "source_model_id": payload["model"]["id"], "name": payload["score"]["title"],
                "text": "".join(note["lyric"] for note in payload["score"]["notes"] if note["lyric"] != "-"),
                "tags": ["歌声合成", "目标音色"] if target else ["歌声合成"], "score": payload["score"],
                "attribution": payload["model"].get("attribution", ""), "allow_output_training": False,
                "parameters": {key: payload.get(key) for key in ("steps", "expressiveness", "pitch_shift", "index_rate")}}
            asset = store.record_audio(Path(result["path"]), metadata)
            if asset:
                result["audio_asset_id"] = asset["id"]
            if result.get("synthesized_path"):
                source = store.record_audio(Path(result["synthesized_path"]), dict(metadata,
                    model_id=payload["model"]["id"], name=metadata["name"] + " · 声库原音色", tags=["歌声合成", "原音色"]))
                if source:
                    result["synthesized_asset_id"] = source["id"]
        elif request["kind"] == "compose":
            asset = store.record_audio(Path(result["path"]), {"kind": "music", "job_id": request.get("job_id"),
                "model_id": payload["model_id"], "name": payload["title"], "text": payload["lyrics"],
                "tags": ["原创歌曲", "AI 音乐"], "attribution": result["attribution"],
                "contains_accompaniment": True,
                "parameters": {key: payload.get(key) for key in ("style", "duration", "bpm", "seed", "language")}})
            if asset:
                result["audio_asset_id"] = asset["id"]
        return result

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.jobs = JobManager(root / "jobs", completed, database=store.database)
        try:
            if store.database:
                from audio_studio.history import import_history
                app.state.imported_history = import_history(store, app.state.jobs.jobs)
            yield
        finally:
            app.state.jobs.close()

    app = FastAPI(title="本地音频工作台", version="0.1.0", lifespan=lifespan)
    app.state.store = store

    @app.middleware("http")
    async def local_access(request: Request, call_next):
        if request.url.path.startswith("/api/"):
            if token:
                supplied = request.headers.get("authorization", "")
                if not secrets.compare_digest(supplied.encode("utf-8"), ("Bearer " + token).encode("utf-8")):
                    return JSONResponse({"detail": "请提供工作台访问令牌"}, status_code=401)
            elif request.url.hostname not in {"localhost", "127.0.0.1", "::1", "testserver"}:
                return JSONResponse({"detail": "远程访问需要设置 AUDIO_STUDIO_TOKEN"}, status_code=403)
            origin = request.headers.get("origin")
            if origin and urlsplit(origin).netloc != request.url.netloc:
                return JSONResponse({"detail": "不接受跨站请求"}, status_code=403)
        return await call_next(request)

    @app.exception_handler(KeyError)
    async def missing_handler(request: Request, exc: KeyError):
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.exception_handler(ValueError)
    async def validation_handler(request: Request, exc: ValueError):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(DatabaseError)
    async def database_handler(request: Request, exc: DatabaseError):
        return JSONResponse({"detail": str(exc)}, status_code=503)

    def require_model(model_id: str) -> dict:
        from audio_studio.engines import model_readiness
        model = store.get("models", model_id)
        readiness = model_readiness(model)
        if not readiness["ready"]:
            raise ValueError("模型尚未就绪，请先下载或注册完整模型：" + ", ".join(readiness["missing"]))
        return model

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @app.get("/static/{filename}", include_in_schema=False)
    def static_asset(filename: str):
        if filename not in {"covers.js", "covers.css", "singing.js", "music.js"}:
            raise HTTPException(404, "文件不存在")
        return FileResponse(Path(__file__).parent / "static" / filename)

    @app.get("/api/status")
    def status():
        from audio_studio.engines import model_readiness
        models = [dict(model, **model_readiness(model)) for model in store.models()]
        return {**hardware(), "selected_device": selected_device, "data_dir": str(root), "models": models, "storage": store.storage_status(),
                "dependencies": {name: importlib.util.find_spec(name) is not None for name in ("torch", "f5_tts", "kokoro", "soundfile", "en_core_web_sm")}}

    @app.get("/api/presets")
    def presets():
        return store.presets()

    @app.post("/api/presets")
    def save_preset(preset: Preset):
        return store.save_preset(preset)

    @app.delete("/api/presets/{preset_id}")
    def delete_preset(preset_id: str):
        store.delete_preset(preset_id)
        return {"deleted": preset_id}

    @app.post("/api/models/{model_id}/download", status_code=202)
    def download(model_id: str):
        model = store.get("models", model_id)
        if not model.get("builtin"):
            raise ValueError("自定义模型使用注册入口")
        return app.state.jobs.submit("download", {"model_id": model_id, "name": model["name"], "engine": model["engine"], "path": model["path"]})

    @app.post("/api/models/register")
    def register(model: RegisterModel):
        return store.register_model(**model.model_dump())

    @app.post("/api/synthesize", status_code=202)
    def synthesize(request: SynthesisRequest):
        preset = store.get("presets", request.preset_id)
        model = require_model(preset["model_id"])
        if request.speed is not None:
            preset["speed"] = request.speed
        return app.state.jobs.submit("synthesize", {"model": model, "preset": preset, "text": request.text, "device": selected_device,
            "output_path": str(root / "audio" / f"{uuid4().hex}.wav")})

    @app.post("/api/datasets/validate")
    def validate(request: DatasetRequest):
        from audio_studio.datasets import validate_dataset
        report = validate_dataset(Path(request.csv_path), Path(request.wavs_dir))
        report["dataset_id"] = store.record_dataset(report)
        return report

    @app.post("/api/train", status_code=202)
    def train(request: TrainingRequest):
        from audio_studio.datasets import validate_dataset
        model = require_model(request.model_id)
        if model["engine"] != "f5":
            raise ValueError("本地微调仅支持 F5-TTS v1 架构")
        report = validate_dataset(Path(request.csv_path), Path(request.wavs_dir))
        if not report["valid"]:
            raise ValueError("数据集检查失败：" + "; ".join(report["errors"][:10]))
        dataset_id = store.record_dataset(report)
        checkpoint = model.get("checkpoint_path", str(Path(model["path"]) / "F5TTS_v1_Base" / "model_1250000.safetensors"))
        vocab = model.get("vocab_path", str(Path(model["path"]) / "F5TTS_v1_Base" / "vocab.txt"))
        vocoder = model.get("vocoder_path", str(Path(model["path"]) / "vocos"))
        config = {key: value for key, value in request.model_dump().items() if key not in {"name", "model_id"}}
        config.update(base_checkpoint=checkpoint, vocab_path=vocab, device=selected_device)
        return app.state.jobs.submit("train", {"name": request.name, "model_id": request.model_id, "dataset_id": dataset_id, "config": config,
            "run_dir": str(root / "runs" / uuid4().hex), "vocoder_path": vocoder})

    @app.post("/api/references")
    async def upload_reference(file: UploadFile = File(...)):
        import soundfile as sf
        path = root / "references" / f"{uuid4().hex}.wav"
        size = 0
        try:
            with path.open("wb") as handle:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > 30 * 1024 * 1024:
                        raise ValueError("参考音频不能超过 30MB")
                    handle.write(chunk)
            info = sf.info(path)
            if info.format != "WAV" or not 1 <= info.duration <= 12:
                raise ValueError("参考音频须为 1–12 秒 WAV，请提供与音频完全对应的转写")
            audio, _ = sf.read(path)
            import numpy as np
            if not np.isfinite(audio).all() or np.max(np.abs(audio)) < 1e-5:
                raise ValueError("参考音频包含无效数值或只有静音")
            store.record_audio(path, {"kind": "reference", "name": Path(file.filename or "参考音频").stem,
                                     "tags": ["参考音频"]})
        except DatabaseError:
            raise
        except Exception as exc:
            path.unlink(missing_ok=True)
            raise ValueError(f"参考音频无效：{exc}") from exc
        finally:
            await file.close()
        return {"path": str(path), "duration_seconds": info.duration}

    def require_database():
        if not store.database:
            raise HTTPException(503, "尚未配置 PostgreSQL，请先填写 .env 并运行 audio-studio db-init")
        return store.database

    def audio_item(item: dict) -> dict:
        return dict(item, audio_url=f"/api/audio-assets/{item['id']}/content", available=Path(item["path"]).is_file())

    @app.get("/api/audio-assets")
    def audio_assets(query: str = Query(default="", max_length=300), limit: int = Query(default=20, ge=1, le=200), offset: int = Query(default=0, ge=0)):
        page = require_database().list_audio(query=query, limit=limit, offset=offset)
        page["items"] = [audio_item(item) for item in page["items"]]
        return page

    @app.patch("/api/audio-assets/{asset_id}")
    def update_audio_asset(asset_id: str, updates: AudioAssetUpdate):
        return audio_item(require_database().update_audio(asset_id, updates.model_dump(exclude_unset=True)))

    @app.get("/api/audio-assets/{asset_id}/content")
    def audio_asset_content(asset_id: str):
        item = require_database().get_audio(asset_id)
        path = Path(item["path"])
        if not path.is_file():
            raise HTTPException(404, "音频文件已移动或不存在")
        return FileResponse(path, media_type="audio/wav", filename=item["filename"])

    @app.get("/api/jobs")
    def jobs():
        return app.state.jobs.list()

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        return app.state.jobs.get(job_id)

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        return app.state.jobs.cancel(job_id)

    @app.get("/api/audio/{filename}")
    def audio(filename: str):
        if len(filename) != 36 or not filename.endswith(".wav") or any(c not in "0123456789abcdef" for c in filename[:-4]):
            raise HTTPException(404, "找不到音频")
        path = root / "audio" / filename
        if not path.is_file():
            raise HTTPException(404, "找不到音频")
        return FileResponse(path, media_type="audio/wav", filename=filename)

    from audio_studio.covers import install_cover_routes
    install_cover_routes(app, store, selected_device)
    from audio_studio.singing_api import install_singing_routes
    install_singing_routes(app, store, selected_device)
    from audio_studio.music_api import install_music_routes
    install_music_routes(app, store)
    return app
