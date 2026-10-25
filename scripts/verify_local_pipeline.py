"""Verify real HTTP synthesis, fine-tuning, registration and redeployment.

Requires the local server and both downloaded models. The training recording is
explicitly synthetic and verifies the software pipeline, not personal voice quality.
Run with the project's Python: python scripts/verify_local_pipeline.py --train
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:7861")
    parser.add_argument("--train", action="store_true", help="Execute one real full-model optimizer update")
    parser.add_argument("--output", type=Path, default=Path("outputs/validation"))
    args = parser.parse_args()
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    evidence: dict = {"server": args.server, "synthetic_training_recording": True}

    def save() -> None:
        (folder / "pipeline.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")

    token = os.environ.get("AUDIO_STUDIO_TOKEN", "")
    headers = {"Authorization": "Bearer " + token} if token else {}
    with httpx.Client(base_url=args.server, timeout=180, trust_env=False, headers=headers) as client:
        def request(method: str, path: str, **kwargs):
            response = client.request(method, path, **kwargs)
            response.raise_for_status()
            return response.json()

        def job(kind: str, path: str, payload: dict) -> dict:
            submitted = request("POST", path, json=payload)
            job_id = submitted["id"]
            print(f"{kind}: {job_id}", flush=True)
            previous = None
            deadline = time.monotonic() + 3600
            while time.monotonic() < deadline:
                current = request("GET", f"/api/jobs/{job_id}")
                progress = (current["status"], (current.get("logs") or [""])[-1])
                if previous != progress:
                    print(f"{kind}: {progress[0]} {progress[1]}", flush=True)
                    previous = progress
                if current["status"] in {"succeeded", "failed", "cancelled"}:
                    evidence[kind] = current
                    save()
                    if current["status"] != "succeeded":
                        raise RuntimeError(f"{kind} failed: {current.get('error')}")
                    return current["result"]
                time.sleep(2)
            raise TimeoutError(f"{kind}: job {job_id} is still running; inspect it before retrying")

        def audio(result: dict, filename: str) -> Path:
            response = client.get(result["audio_url"])
            response.raise_for_status()
            path = folder / filename
            path.write_bytes(response.content)
            waveform, rate = sf.read(path)
            if rate != 24000 or waveform.size == 0 or not np.isfinite(waveform).all() or np.max(np.abs(waveform)) <= 1e-6:
                raise RuntimeError(f"Invalid real model output: {path}")
            return path

        evidence["status"] = request("GET", "/api/status")
        if not all(model["ready"] for model in evidence["status"]["models"] if model.get("builtin")):
            raise RuntimeError("Download Kokoro and F5 base models before running this verification")

        reference_text = "你好，这是本地音频工作台。我们正在测试语音生成与本地部署。"
        zh = job("kokoro_zh", "/api/synthesize", {"preset_id": "zh-narration", "text": reference_text})
        reference_path = audio(zh, "kokoro-zh.wav")
        english = job("kokoro_en", "/api/synthesize", {"preset_id": "en-narration", "text": "Hello. This voice is generated entirely on the local computer."})
        audio(english, "kokoro-en.wav")
        with reference_path.open("rb") as handle:
            uploaded = request("POST", "/api/references", files={"file": ("reference.wav", handle, "audio/wav")})
        preset = {"id": "validation-f5-base", "name": "F5 基础模型 · 链路验证", "category": "custom", "engine": "f5",
                  "model_id": "f5-base", "language": "z", "voice": "zf_xiaobei", "reference_audio": uploaded["path"],
                  "reference_text": reference_text, "steps": 16, "seed": 42, "speed": 1.0}
        request("POST", "/api/presets", json=preset)
        base = job("f5_base", "/api/synthesize", {"preset_id": preset["id"], "text": "这段声音使用本地参考音色生成。"})
        audio(base, "f5-base.wav")

        if args.train:
            # This manifest intentionally uses a synthetic recording for one
            # update, solely to verify the full production architecture and API.
            manifest = folder / "synthetic-recording.csv"
            import csv
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["audio_path", "text"])
                writer.writeheader()
                writer.writerow({"audio_path": str(reference_path), "text": reference_text})
            report = request("POST", "/api/datasets/validate", json={"csv_path": str(manifest), "wavs_dir": str(folder)})
            evidence["dataset"] = report
            if not report["valid"]:
                raise RuntimeError(str(report["errors"]))
            trained = job("training", "/api/train", {"name": "F5 训练链路验证 · 合成样本", "model_id": "f5-base",
                "csv_path": str(manifest), "wavs_dir": str(folder), "epochs": 1, "batch_size": 1,
                "grad_accumulation_steps": 1, "learning_rate": 1e-5})
            if trained["optimizer_updates"] != 1 or not np.isfinite(trained["final_loss"]):
                raise RuntimeError("Training did not produce one finite optimizer update")
            base_path = next(model["checkpoint_path"] for model in evidence["status"]["models"] if model["id"] == "f5-base")
            evidence["weight_change"] = verify_weight_change(Path(base_path), Path(trained["checkpoint_path"]))
            status = request("GET", "/api/status")
            if not any(model["id"] == trained["model_id"] and model["ready"] for model in status["models"]):
                raise RuntimeError("Training artifact was not registered as a ready model")
            preset.update(id="validation-f5-trained", name="F5 微调产物 · 链路验证", model_id=trained["model_id"])
            request("POST", "/api/presets", json=preset)
            deployed = job("f5_trained", "/api/synthesize", {"preset_id": preset["id"], "text": "这段声音由本地微调后的模型生成。"})
            audio(deployed, "f5-trained.wav")
        evidence["complete"] = True
        save()
        print(f"Real pipeline verification passed: {folder / 'pipeline.json'}", flush=True)


def verify_weight_change(base_path: Path, trained_path: Path) -> dict:
    """Inspect real tensors without loading two whole checkpoints into RAM."""
    from safetensors import safe_open

    with safe_open(str(base_path), framework="pt", device="cpu") as base, safe_open(str(trained_path), framework="pt", device="cpu") as trained:
        for key in sorted(set(base.keys()) & set(trained.keys())):
            if not key.startswith("ema_model.transformer."):
                continue
            before, after = base.get_tensor(key), trained.get_tensor(key)
            difference = float((before.float() - after.float()).abs().max().item())
            if np.isfinite(difference) and difference > 0:
                return {"tensor": key, "max_absolute_change": difference}
    raise RuntimeError("Exported model weights did not change after training")


if __name__ == "__main__":
    main()
