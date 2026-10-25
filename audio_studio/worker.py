"""Subprocess entry point. No model state is shared with the HTTP service."""

import json
import sys
import threading
import traceback
from pathlib import Path
from uuid import uuid4

from audio_studio.store import write_json


def execute(request: dict, folder: Path) -> dict:
    kind, payload = request["kind"], request["payload"]
    if kind == "download":
        from audio_studio.engines import download_model
        return download_model(payload["engine"], Path(payload["path"]), print)
    if kind == "synthesize":
        from audio_studio.engines import synthesize
        result = synthesize(payload["model"], payload["preset"], payload["text"], Path(payload["output_path"]), payload["device"])
        result["audio_url"] = "/api/audio/" + Path(payload["output_path"]).name
        return result
    if kind == "train":
        from audio_studio.training import run_training
        return run_training(payload["config"], Path(payload["run_dir"]), print, threading.Event())
    if kind == "validate":
        from audio_studio.datasets import validate_dataset
        return validate_dataset(Path(payload["csv_path"]), Path(payload["wavs_dir"]))
    if kind == "train_voice":
        from audio_studio.svc_training import run_training
        return run_training(payload["config"], Path(payload["run_dir"]), print, threading.Event())
    if kind == "cover":
        from audio_studio.svc import convert
        output = Path(payload["output_path"])
        vocal = output.parent / f"{uuid4().hex}.wav" if payload.get("accompaniment_audio") else output
        result = convert(payload["model"], Path(payload["source_audio"]), vocal,
            pitch_shift=payload.get("pitch_shift", 0), index_rate=payload.get("index_rate", .5),
            speaker_id=payload.get("speaker_id", 0), device=payload.get("device", "auto"), seed=payload.get("seed", 42))
        if payload.get("accompaniment_audio"):
            from audio_studio.audio_processing import mix_accompaniment
            result.update(mix_accompaniment(vocal, Path(payload["accompaniment_audio"]), output,
                vocal_gain=payload.get("vocal_gain", 1), accompaniment_gain=payload.get("accompaniment_gain", .7)))
            result.update(vocal_path=str(vocal), vocal_audio_url="/api/audio/" + vocal.name)
        result["audio_url"] = "/api/audio/" + output.name
        return result
    if kind == "sing":
        from audio_studio.singing import synthesize
        output = Path(payload["output_path"])
        source = output.parent / f"{uuid4().hex}.wav" if payload.get("target_model") else output
        result = synthesize(payload["model"], payload["score"], source,
            steps=payload.get("steps", 20), expressiveness=payload.get("expressiveness", .7))
        if payload.get("target_model"):
            from audio_studio.svc import convert
            result.update(convert(payload["target_model"], source, output,
                pitch_shift=payload.get("pitch_shift", 0), index_rate=payload.get("index_rate", .5),
                speaker_id=int(payload["target_model"].get("speaker_id", 0)), device=payload.get("device", "auto")))
            result.update(synthesized_path=str(source), synthesized_audio_url="/api/audio/" + source.name,
                source_sample_rate=44100)
        result["audio_url"] = "/api/audio/" + output.name
        return result
    if kind == "compose":
        import os
        import subprocess
        runtime = Path(payload["runtime_path"])
        python = runtime / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        input_path, result_path = folder / "music-input.json", folder / "music-result.json"
        write_json(input_path, payload)
        environment = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")
        for key in ("AUDIO_STUDIO_DATABASE_URL", "AUDIO_STUDIO_TOKEN"):
            environment.pop(key, None)
        subprocess.run([str(python), str(Path(__file__).with_name("music_runner.py")),
            str(input_path.resolve()), str(result_path.resolve())], cwd=runtime, env=environment,
            stdin=subprocess.DEVNULL, check=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["audio_url"] = "/api/audio/" + Path(payload["output_path"]).name
        return result
    raise ValueError(f"未知任务: {kind}")


def main() -> None:
    path = Path(sys.argv[1])
    try:
        result = execute(json.loads(path.read_text(encoding="utf-8")), path.parent)
        write_json(path.parent / "result.json", result)
    except Exception as exc:
        traceback.print_exc()
        write_json(path.parent / "result.json", {"error": str(exc)})
        raise SystemExit(1)


if __name__ == "__main__":
    main()
