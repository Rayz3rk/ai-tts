"""Generate a real local audio sample for every built-in preset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import soundfile as sf

from audio_studio.engines import synthesize
from audio_studio.store import Store


def main() -> None:
    from audio_studio.config import load_local_config
    load_local_config()
    import torch
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    root = Path(__file__).resolve().parents[1]
    store = Store(root / "studio_data")
    folder = root / "outputs" / "preset-verification"
    folder.mkdir(parents=True, exist_ok=True)
    results = []
    for preset in store.presets():
        if not preset.get("builtin"):
            continue
        text = "欢迎使用本地语音工作台。今天的任务已经准备好了。" if preset["language"] == "z" else "Welcome to the local audio studio. Your voice is ready."
        path = folder / f"{preset['id']}.wav"
        result = synthesize(store.get("models", preset["model_id"]), preset, text, path, "cpu")
        info = sf.info(path)
        if info.channels != 1 or info.samplerate != 24000 or info.subtype != "PCM_16":
            raise RuntimeError(f"Unexpected audio format for {preset['id']}: {info}")
        result.update(preset_id=preset["id"], category=preset["category"], language=preset["language"], voice=preset["voice"],
                      sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        results.append(result)
        print(f"{preset['id']}: {info.duration:.2f}s", flush=True)
    if len(results) < 8 or len({item["category"] for item in results}) < 5:
        raise RuntimeError("Expected eight working presets across all five use categories")
    (folder / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Verified {len(results)} real preset outputs: {folder}", flush=True)


if __name__ == "__main__":
    main()
