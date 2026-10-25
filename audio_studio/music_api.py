"""Lyrics-to-song generation routes; explicit weights and isolated runtime."""

from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI

from audio_studio.schemas import MusicRequest
from audio_studio.store import Store


def install_music_routes(app: FastAPI, store: Store) -> None:
    @app.get("/api/music/status")
    def status():
        from audio_studio.music_assets import model_readiness
        model = store.music_assets()
        return {"model": dict(model, **model_readiness(model)), "max_duration": 180,
            "supports_target_voice": False, "contains_accompaniment": True}

    @app.post("/api/music/assets", status_code=202)
    def download():
        model = store.music_assets()
        return app.state.jobs.submit("download", {"engine": "ace_step", "model_id": model["id"],
            "name": "下载 ACE-Step 自动创作权重", "path": model["path"]})

    @app.post("/api/music/generate", status_code=202)
    def generate(request: MusicRequest):
        from audio_studio.music_assets import model_readiness
        model = store.music_assets()
        if not model_readiness(model)["ready"]:
            raise ValueError("自动创作模型或独立运行环境未就绪，请先完成安装与权重下载")
        return app.state.jobs.submit("compose", dict(model, **request.model_dump(),
            model_id=model["id"], name=request.title,
            output_path=str(store.root / "audio" / f"{uuid4().hex}.wav")))
