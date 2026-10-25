"""Lyrics/score/MIDI entry points for real local singing synthesis."""

import importlib.util
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, UploadFile

from audio_studio.schemas import BulkScoreRequest, SingingRequest, SingingScore
from audio_studio.store import Store


def install_singing_routes(app: FastAPI, store: Store, device: str) -> None:
    @app.post("/api/singing/score/from-text")
    def bulk_score(request: BulkScoreRequest):
        from audio_studio.score import score_from_text
        return score_from_text(**request.model_dump())

    @app.get("/api/singing/status")
    def status():
        from audio_studio.engines import model_readiness
        model = store.singing_assets()
        return {"model": dict(model, **model_readiness(model)),
            "voices": [dict(voice, **model_readiness(voice)) for voice in store.models()
                if voice["engine"] == "rvc" and voice.get("role") == "voice"],
            "dependencies": {name: importlib.util.find_spec(name) is not None for name in ("onnxruntime", "mido", "pypinyin", "yaml")},
            "supported_language": "普通话汉字 / 拼音", "device": "cpu"}

    @app.post("/api/singing/assets", status_code=202)
    def download():
        model = store.singing_assets()
        return app.state.jobs.submit("download", {"engine": "diffsinger", "model_id": model["id"],
            "name": "下载 DiffSinger 绮萱声库与声码器", "path": model["path"]})

    @app.post("/api/singing/midi")
    async def import_melody(file: UploadFile = File(...), lyrics: str = Form(""), track_index: int | None = Form(None)):
        from audio_studio.score import import_midi
        try:
            data = await file.read(4 * 1024 * 1024 + 1)
            if len(data) > 4 * 1024 * 1024:
                raise ValueError("MIDI 文件不能超过 4MB")
            return import_midi(data, lyrics, track_index)
        finally:
            await file.close()

    @app.post("/api/singing/score/validate")
    def validate(score: SingingScore):
        from audio_studio.singing import validate_score
        return validate_score(store.singing_assets(), score.model_dump())

    @app.post("/api/singing/generate", status_code=202)
    def generate(request: SingingRequest):
        from audio_studio.engines import model_readiness
        from audio_studio.singing import validate_score
        model = store.get("models", request.model_id)
        if model["engine"] != "diffsinger":
            raise ValueError("歌词与旋律合成需要 DiffSinger 歌声模型")
        readiness = model_readiness(model)
        if not readiness["ready"]:
            raise ValueError("请先下载完整 DiffSinger 声库及声码器")
        score = request.score.model_dump()
        validate_score(model, score)
        target = store.get("models", request.target_voice_id) if request.target_voice_id else None
        if target:
            if target["engine"] != "rvc" or target.get("role") != "voice" or not model_readiness(target)["ready"]:
                raise ValueError("请选择已就绪的 RVC 目标音色，或先使用声库原音色生成")
        return app.state.jobs.submit("sing", {"name": score["title"], "model_id": model["id"], "model": model,
            "score": score, "steps": request.steps, "expressiveness": request.expressiveness,
            "target_model": target, "pitch_shift": request.pitch_shift, "index_rate": request.index_rate,
            "device": device, "output_path": str(store.root / "audio" / f"{uuid4().hex}.wav")})
