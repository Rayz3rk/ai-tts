"""Exercise full RVC assets, one GAN update, registration and real conversion.

Uses the existing Kokoro-generated acceptance recording, never personal voice
quality claims. Task IDs are saved before polling so interrupted observations
resume the same jobs rather than submitting duplicates.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import time

import httpx
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "outputs" / "validation" / "svc-pipeline.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:7861")
    args = parser.parse_args()
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8")) if EVIDENCE.exists() else {
        "synthetic_acceptance_data": True, "personal_voice_quality_verified": False, "jobs": {}}

    def save():
        EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
        EVIDENCE.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")

    with httpx.Client(base_url=args.server, timeout=120, trust_env=False) as client:
        def request(method, path, **kwargs):
            response = client.request(method, path, **kwargs)
            response.raise_for_status()
            return response.json()

        status = request("GET", "/api/covers/status")
        assert status["assets"]["ready"], "Wait for complete verified RVC components before running"
        evidence["assets"] = status["assets"]
        dataset = ROOT / "work" / "svc-validation-recordings"
        dataset.mkdir(parents=True, exist_ok=True)
        # Use the known live PostgreSQL synthesis recording, not a random user's
        # file: its asset ID is part of the previous reproducible acceptance.
        assets = request("GET", "/api/audio-assets", params={"limit": 200})
        accepted = next(item for item in assets["items"] if item["id"] == "bf65b645d86e4d9390335d88702b82a0")
        source = Path(accepted["path"])
        shutil.copyfile(source, dataset / "kokoro-synthetic-acceptance.wav")
        report = request("POST", "/api/covers/datasets/validate", json={"wavs_dir": str(dataset)})
        assert report["valid"], report
        evidence["dataset"] = report

        def job(stage, endpoint, payload):
            if stage not in evidence["jobs"]:
                submitted = request("POST", endpoint, json=payload)
                evidence["jobs"][stage] = {"id": submitted["id"]}
                save()
            identity = evidence["jobs"][stage]["id"]
            previous = None
            while True:
                current = request("GET", "/api/jobs/" + identity)
                progress = (current["status"], (current.get("logs") or [""])[-1])
                if progress != previous:
                    print(stage, identity, progress, flush=True)
                    previous = progress
                if current["status"] in {"succeeded", "failed", "cancelled"}:
                    evidence["jobs"][stage] = current
                    save()
                    assert current["status"] == "succeeded", current.get("error")
                    return current["result"]
                time.sleep(2)

        trained = job("training", "/api/covers/train", {"wavs_dir": str(dataset), "name": "RVC 合成样本验收音色",
            "epochs": 1, "batch_size": 1, "max_steps": 1, "learning_rate": 1e-4})
        assert trained["steps"] == 1 and Path(trained["checkpoint_path"]).is_file()
        import torch
        exported = torch.load(trained["checkpoint_path"], map_location="cpu", weights_only=True)
        initial = torch.load(status["assets"]["pretrained_g_path"], map_location="cpu", weights_only=True)["model"]
        changed = next(name for name, value in exported["weight"].items() if not torch.equal(value, initial[name]))
        evidence["weight_change"] = {"parameter": changed,
            "max_absolute_change": float((exported["weight"][changed] - initial[changed]).abs().max()),
            "exported_parameters": sum(value.numel() for value in exported["weight"].values())}
        assert np.isfinite(trained["loss"]) and evidence["weight_change"]["max_absolute_change"] > 0
        del exported, initial
        converted = job("conversion", "/api/covers/convert", {"model_id": trained["model_id"],
            "source_audio": str(source), "pitch_shift": 0, "index_rate": .5})
        audio = client.get(converted["audio_url"]).raise_for_status().content
        samples, rate = sf.read(io.BytesIO(audio))
        assert rate == 40000 and np.isfinite(samples).all() and np.abs(samples).max() > 1e-5
        assert abs(len(samples) / rate - sf.info(source).duration) < .011
        evidence["audio"] = {"path": converted["path"], "sample_rate": rate, "seconds": len(samples) / rate,
            "sha256": hashlib.sha256(audio).hexdigest(), "finite_non_silent": True, "duration_preserved": True,
            "asset_id": converted["audio_asset_id"], "index_rate": converted["index_rate"]}
        save()
        print("PASS: complete pretrained encoder/F0, actual full RVC update, exported model reload, conversion and indexing", flush=True)


if __name__ == "__main__":
    main()
