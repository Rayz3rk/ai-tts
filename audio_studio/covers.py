"""Local target-voice singing conversion routes and audio input validation."""

from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, File, UploadFile

from audio_studio.schemas import CoverRequest, RegisterVoiceModel, VoiceDatasetRequest, VoiceTrainingRequest
from audio_studio.store import Store


def validate_audio(path: Path, *, maximum_seconds: float = 600) -> dict:
    import soundfile as sf
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError("音频文件不存在，请上传音频或检查本机路径")
    try:
        info = sf.info(path)
    except (RuntimeError, OSError):
        raise ValueError("无法读取音频，请使用 WAV、FLAC 或可解码的音频文件") from None
    if not 0.5 <= info.duration <= maximum_seconds:
        raise ValueError(f"音频时长须为 0.5–{maximum_seconds:g} 秒")
    if not 8000 <= info.samplerate <= 96000 or not 1 <= info.channels <= 2:
        raise ValueError("音频须为 8–96kHz，单声道或立体声")
    return {"path": str(path), "duration_seconds": info.duration, "sample_rate": info.samplerate, "channels": info.channels}


def install_cover_routes(app: FastAPI, store: Store, device: str) -> None:
    @app.get("/api/covers/test-vocal")
    def test_vocal():
        candidates = [store.root.parent / "outputs" / "samples" / "singing-dry-demo.wav",
                      store.root.parent / "outputs" / "validation" / "singing-four-notes.wav"]
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            raise ValueError("本机尚无测试干声，可在歌词与旋律页面使用原音色生成一段演唱后载入")
        return dict(validate_audio(path), name="无伴奏歌声测试样例", allow_output_training=False)

    @app.get("/api/covers/status")
    def status():
        from audio_studio.svc import model_readiness
        assets = store.svc_assets()
        return {"assets": dict(assets, **model_readiness(assets)),
            "voices": [dict(model, **model_readiness(model)) for model in store.models()
                       if model["engine"] == "rvc" and model.get("role") == "voice"],
            "input": {"min_seconds": 0.5, "max_seconds": 600, "max_upload_mb": 150},
            "device": device}

    @app.post("/api/covers/assets", status_code=202)
    def download_assets():
        assets = store.svc_assets()
        return app.state.jobs.submit("download", {"engine": "rvc", "model_id": assets["id"],
            "name": "下载 RVC 转换与训练组件", "path": assets["path"]})

    @app.post("/api/covers/models")
    def register_model(model: RegisterVoiceModel):
        return store.register_voice_model(**model.model_dump())

    @app.post("/api/covers/audio")
    async def upload_audio(kind: Literal["source", "accompaniment"] = "source", file: UploadFile = File(...)):
        import numpy as np
        import soundfile as sf
        identity = uuid4().hex
        original = store.root / "inputs" / f"{identity}.upload"
        normalized = store.root / "inputs" / f"{identity}.part.wav"
        target = store.root / "inputs" / f"{identity}.wav"
        try:
            size = 0
            with original.open("wb") as output:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > 150 * 1024 * 1024:
                        raise ValueError("翻唱输入音频不能超过 150MB")
                    output.write(chunk)
            metadata = validate_audio(original)
            peak = 0.0
            with sf.SoundFile(str(normalized), "w", samplerate=metadata["sample_rate"],
                              channels=metadata["channels"], format="WAV", subtype="PCM_16") as output:
                for block in sf.blocks(original, blocksize=65536, dtype="float32", always_2d=True):
                    if not np.isfinite(block).all():
                        raise ValueError("音频包含无效数值")
                    peak = max(peak, float(np.abs(block).max(initial=0)))
                    output.write(block)
            if peak < 1e-5:
                raise ValueError("音频只有静音，请选择包含人声或伴奏的文件")
            normalized.replace(target)
            metadata["path"] = str(target)
            asset = store.record_audio(target, {"name": Path(file.filename or "翻唱输入").stem[:200],
                "kind": kind, "tags": ["翻唱干声" if kind == "source" else "伴奏"]})
            return dict(metadata, kind=kind, audio_asset_id=asset["id"] if asset else None)
        finally:
            original.unlink(missing_ok=True)
            normalized.unlink(missing_ok=True)
            await file.close()

    @app.post("/api/covers/convert", status_code=202)
    def convert(request: CoverRequest):
        from audio_studio.svc import model_readiness
        model = store.get("models", request.model_id)
        if model["engine"] != "rvc" or model.get("role") != "voice":
            raise ValueError("请选择已注册或训练完成的 RVC 目标音色")
        ready = model_readiness(model)
        if not ready["ready"]:
            raise ValueError("目标音色尚未就绪，请检查模型与转换组件：" + ", ".join(ready["missing"]))
        source = validate_audio(Path(request.source_audio))
        if request.accompaniment_audio:
            validate_audio(Path(request.accompaniment_audio))
        speaker_count = int(model.get("speaker_count", 1))
        if request.speaker_id >= speaker_count:
            raise ValueError(f"该模型的音色编号须小于 {speaker_count}")
        payload = request.model_dump(exclude={"model_id"})
        payload.update(model=model, source_audio=source["path"], device=device,
            name="目标音色翻唱 · " + model["name"], output_path=str(store.root / "audio" / f"{uuid4().hex}.wav"))
        return app.state.jobs.submit("cover", payload)

    @app.post("/api/covers/datasets/validate")
    def validate_dataset(request: VoiceDatasetRequest):
        from audio_studio.svc_training import validate_voice_dataset
        report = validate_voice_dataset(Path(request.wavs_dir))
        report["dataset_id"] = store.record_dataset(report)
        return report

    @app.post("/api/covers/train", status_code=202)
    def train(request: VoiceTrainingRequest):
        from audio_studio.svc import model_readiness
        from audio_studio.svc_training import validate_voice_dataset
        assets = store.svc_assets()
        ready = model_readiness(assets)
        if not ready["ready"]:
            raise ValueError("请先下载完整的 RVC 转换与训练组件")
        report = validate_voice_dataset(Path(request.wavs_dir))
        if not report["valid"]:
            raise ValueError("音色训练数据检查失败：" + "; ".join(report["errors"][:10]))
        dataset_id = store.record_dataset(report)
        config = request.model_dump(exclude={"name"}, exclude_none=True)
        config.update(assets=assets, device=device)
        return app.state.jobs.submit("train_voice", {"name": request.name, "model_id": assets["id"],
            "dataset_id": dataset_id, "config": config, "run_dir": str(store.root / "runs" / uuid4().hex)})
