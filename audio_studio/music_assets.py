"""Pinned ACE-Step runtime and model assets, separate from speech dependencies."""

import json
import os
from pathlib import Path
from urllib.request import getproxies

from audio_studio.store import write_json

SOURCE = "https://github.com/ace-step/ACE-Step-1.5"
REVISION = "ca1e85fe9430179831e6bc6be790c332190a3866"
LM = "acestep-5Hz-lm-0.6B"
DIT = "acestep-v15-turbo"


def model_metadata(data_root: Path) -> dict:
    return {"id": "ace-step-15", "name": "ACE-Step 1.5 · 歌词自动创作", "engine": "ace_step",
        "role": "music", "builtin": True, "path": str((data_root / "models" / "ace-step").resolve()),
        "runtime_path": str((data_root / "runtimes" / "ace-step").resolve()),
        "source": SOURCE, "revision": REVISION, "sample_rate": 48000,
        "dit_model": DIT, "lm_model": LM, "license": "MIT",
        "attribution": "AI 音乐生成：ACE-Step 1.5"}


def model_readiness(model: dict) -> dict:
    root, runtime = Path(model["path"]), Path(model["runtime_path"])
    required = [root / DIT / "config.json", root / DIT / "model.safetensors",
        root / DIT / "silence_latent.pt", root / "vae" / "diffusion_pytorch_model.safetensors",
        root / "Qwen3-Embedding-0.6B" / "model.safetensors", root / LM / "model.safetensors",
        root / "download-verification.json", runtime / "acestep" / "handler.py",
        runtime / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python"),
        runtime / "studio-runtime.json"]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if not missing:
        try:
            runtime_info = json.loads((runtime / "studio-runtime.json").read_text(encoding="utf-8"))
            manifest = json.loads((root / "download-verification.json").read_text(encoding="utf-8"))["files"]
            if runtime_info.get("revision") != REVISION or not all(runtime_info.get(key) is True for key in ("cuda_verified", "imports_verified", "wavelets_verified")):
                missing.append("独立运行环境需要重新检查 CUDA 与代码版本")
            for item in manifest:
                path = (root / item["destination"]).resolve()
                # Upstream synchronizes these Python files from the pinned
                # runtime source before loading. Weight/config sizes stay exact.
                if not path.is_relative_to(root.resolve()) or not path.is_file() or (path.suffix != ".py" and path.stat().st_size != item["size"]):
                    missing.append(item["destination"])
        except (ValueError, OSError, KeyError, TypeError):
            missing.append("模型或运行环境校验记录无效")
    return {"ready": not missing, "missing": missing, "device": "cuda", "runtime_path": str(runtime)}


def download_assets(root: Path, log=print) -> dict:
    """Download only 2B turbo, text encoder, VAE and 0.6B LM; verify official hashes."""
    for scheme, proxy in getproxies().items():
        if scheme in {"http", "https"}:
            os.environ.setdefault(scheme.upper() + "_PROXY", proxy)
    from huggingface_hub import HfApi
    from audio_studio.hub_download import download_file
    api = HfApi(endpoint="https://huggingface.co")
    manifest_path = root / "source-manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = []
        for repo, prefix in (("ACE-Step/Ace-Step1.5", ""), ("ACE-Step/" + LM, LM + "/")):
            info = api.model_info(repo, files_metadata=True, timeout=40)
            for file in info.siblings:
                if not prefix and not any(file.rfilename.startswith(folder + "/") for folder in (DIT, "Qwen3-Embedding-0.6B", "vae")):
                    continue
                if file.rfilename.startswith("."):
                    continue
                item = {"repo": repo, "revision": info.sha, "file": file.rfilename,
                    "destination": prefix + file.rfilename, "size": file.size}
                if file.lfs:
                    item["sha256"] = file.lfs.sha256
                else:
                    item["git_sha1"] = file.blob_id
                manifest.append(item)
        write_json(manifest_path, manifest)
    for item in manifest:
        target = root / item["destination"]
        if not target.resolve().is_relative_to(root.resolve()):
            raise ValueError("模型清单含无效路径")
        if "sha256" not in item:
            # Verify ordinary Git blobs using the official object hash too.
            import hashlib
            from urllib.request import build_opener, ProxyHandler
            def valid(content):
                return len(content) == item["size"] and hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest() == item["git_sha1"]
            if target.is_file() and valid(target.read_bytes()):
                continue
            for domain, proxies in (("hf-mirror.com", {}), ("huggingface.co", getproxies())):
                try:
                    with build_opener(ProxyHandler(proxies)).open(
                            f"https://{domain}/{item['repo']}/resolve/{item['revision']}/{item['file']}", timeout=45) as response:
                        content = response.read(32 * 1024 * 1024)
                    if not valid(content):
                        raise RuntimeError("模型配置 Git 校验失败：" + item["file"])
                    break
                except (OSError, RuntimeError):
                    if domain == "huggingface.co":
                        raise
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        else:
            download_file(item["repo"], item["revision"], item["file"], target, item["size"], item["sha256"], log)
    write_json(root / "download-verification.json", {"source": SOURCE, "files": manifest})
    return {"engine": "ace_step", "path": str(root.resolve()), "source": SOURCE}
