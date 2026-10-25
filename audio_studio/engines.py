"""Local TTS backends. Downloads are explicit; synthesis only reads local assets.

The model registry supplies JSON-compatible dictionaries so the same adapters work
from the web job worker, a command-line process, and the deployment API.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
import gc
import hashlib
from http.client import IncompleteRead
import importlib.util
import math
import os
from pathlib import Path
import re
import sys
import time
from threading import RLock
from typing import Any, Iterator
from urllib.request import Request, urlopen


KOKORO_REPO = "hexgrad/Kokoro-82M"
F5_REPO = "SWivid/F5-TTS"
VOCOS_REPO = "charactr/vocos-mel-24khz"
KOKORO_CHECKPOINT = "kokoro-v1_0.pth"
F5_CHECKPOINT = "F5TTS_v1_Base/model_1250000.safetensors"
F5_VOCAB = "F5TTS_v1_Base/vocab.txt"
KOKORO_PRESET_VOICES = (
    "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
    "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
    "af_heart", "am_adam", "bf_emma", "bm_george",
)

_LOCK = RLock()
_CACHE: dict[str, Any] = {}


def _require_file(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"缺少本地模型文件：{path}。请先下载模型或检查导入路径。")
    return path


def _asset_path(model: dict, key: str, default: str) -> Path:
    base = Path(model["path"]).expanduser().resolve()
    asset = Path(model.get(key) or default).expanduser()
    return asset.resolve() if asset.is_absolute() else (base / asset).resolve()


def model_readiness(model: dict) -> dict:
    """Check files without importing Torch or loading a checkpoint."""
    engine = model.get("engine")
    if engine == "ace_step":
        from audio_studio.music_assets import model_readiness as music_readiness
        return music_readiness(model)
    if engine == "diffsinger":
        from audio_studio.singing import model_readiness as singing_readiness
        return singing_readiness(model)
    if engine == "rvc":
        from audio_studio.svc import model_readiness as svc_readiness
        return svc_readiness(model)
    if engine == "kokoro":
        required = [
            _asset_path(model, "config_path", "config.json"),
            _asset_path(model, "checkpoint_path", KOKORO_CHECKPOINT),
            *(Path(model["path"]) / "voices" / f"{voice}.pt" for voice in KOKORO_PRESET_VOICES),
        ]
        voices = sorted((Path(model["path"]) / "voices").glob("*.pt"))
        missing = [str(p) for p in required if not p.is_file() or p.stat().st_size == 0]
        return {"ready": not missing, "missing": missing, "voices": [p.stem for p in voices]}
    if engine == "f5":
        vocoder = _asset_path(model, "vocoder_path", "vocos")
        required = [
            _asset_path(model, "checkpoint_path", F5_CHECKPOINT),
            _asset_path(model, "vocab_path", F5_VOCAB),
            vocoder / "config.yaml",
            vocoder / "pytorch_model.bin",
        ]
        missing = [str(p) for p in required if not p.is_file() or p.stat().st_size == 0]
        return {"ready": not missing, "missing": missing}
    return {"ready": False, "missing": [f"不支持的引擎：{engine}"]}


def _download_weight(api: Any, repo: str, revision: str, filename: str, root: Path, log: Callable[[str], None]) -> None:
    """Resume official LFS bytes, then verify integrity before exposing a model."""
    from huggingface_hub import hf_hub_url

    metadata = api.model_info(repo, revision=revision, files_metadata=True, timeout=30)
    sibling = next(item for item in metadata.siblings if item.rfilename == filename)
    expected_hash = sibling.lfs.sha256
    expected_size = sibling.size
    destination = root / filename
    destination.parent.mkdir(parents=True, exist_ok=True)

    def verified(path: Path) -> bool:
        if not path.is_file() or path.stat().st_size != expected_size:
            return False
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest() == expected_hash

    if verified(destination):
        return
    partial = destination.with_name(destination.name + ".download")
    if verified(partial):
        partial.replace(destination)
        return
    if partial.exists() and partial.stat().st_size >= expected_size:
        partial.unlink()
    url = hf_hub_url(repo_id=repo, filename=filename, revision=revision) + "?download=true"
    for attempt in range(3):
        offset = partial.stat().st_size if partial.exists() else 0
        log(f"{filename}：{offset / 1048576:.1f} / {expected_size / 1048576:.1f} MiB")
        try:
            last_report = offset
            while offset < expected_size:
                # Bounded ranges also work through proxies which buffer large
                # responses instead of streaming an entire multi-GB checkpoint.
                end = min(expected_size - 1, offset + 16 * 1024 * 1024 - 1)
                request = Request(url, headers={"Range": f"bytes={offset}-{end}"})
                with urlopen(request, timeout=60) as response:
                    if response.status == 206:
                        if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                            raise RuntimeError("服务器返回的续传范围与本地文件不一致。")
                        mode = "ab"
                    elif response.status == 200:
                        mode, offset = "wb", 0
                    else:
                        raise RuntimeError(f"下载服务器返回状态 {response.status}")
                    before = offset
                    with partial.open(mode) as stream:
                        while block := response.read(1024 * 1024):
                            stream.write(block)
                            if stream.tell() - last_report >= 64 * 1024 * 1024:
                                last_report = stream.tell()
                                log(f"{filename}：{last_report / 1048576:.1f} / {expected_size / 1048576:.1f} MiB")
                    offset = partial.stat().st_size
                    if offset <= before:
                        raise RuntimeError("下载服务器没有返回新的模型数据。")
            if not verified(partial):
                if partial.exists() and partial.stat().st_size >= expected_size:
                    partial.unlink()
                raise RuntimeError(f"{filename} 的文件大小或 SHA256 校验失败。")
            partial.replace(destination)
            log(f"{filename}：SHA256 校验通过。")
            return
        except (OSError, RuntimeError, IncompleteRead) as exc:
            if attempt == 2:
                raise RuntimeError(f"下载失败（保留已下载数据以便重试）：{filename}；{exc}") from exc
            log(f"下载连接中断，正在续传 {filename}（第 {attempt + 2} 次尝试）。")
            time.sleep(attempt + 1)


def download_model(engine: str, model_dir: Path, log: Callable[[str], None]) -> dict:
    """Download the official model plus every asset required for offline inference.

    Hugging Face resumable local-dir downloads keep successful files when a
    network error occurs. No random audio or demo model is substituted on error.
    """
    if engine == "ace_step":
        from audio_studio.music_assets import download_assets
        return download_assets(model_dir, log)
    if engine == "diffsinger":
        from audio_studio.singing import download_assets
        return download_assets(model_dir, log)
    if engine == "rvc":
        from audio_studio.svc import download_assets
        return download_assets(model_dir, log)
    if engine not in {"kokoro", "f5"}:
        raise ValueError(f"不支持的引擎：{engine}")
    # The Windows Xet transport can abort sockets on managed workstations.
    # Respect an explicit override; otherwise use resumable HTTPS in this process.
    if sys.platform == "win32":
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        # httpx does not discover the Windows Internet Settings proxy itself.
        # Use the existing system setting without changing the user's registry.
        from urllib.request import getproxies

        for scheme, proxy in getproxies().items():
            if scheme in {"http", "https"}:
                os.environ.setdefault(f"{scheme.upper()}_PROXY", proxy)
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub import constants as hub_constants

    if os.environ.get("HF_HUB_DISABLE_XET") == "1":
        hub_constants.HF_HUB_DISABLE_XET = True

    model_dir = Path(model_dir).expanduser().resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    if engine == "kokoro":
        log("下载 Kokoro 82M 权重、配置和预设音色。")
        revision = api.model_info(KOKORO_REPO, timeout=30).sha
        _download_weight(api, KOKORO_REPO, revision, KOKORO_CHECKPOINT, model_dir, log)
        snapshot_download(
            repo_id=KOKORO_REPO,
            revision=revision,
            local_dir=str(model_dir),
            max_workers=1,
            allow_patterns=["config.json", "voices/*.pt", "LICENSE", "README.md"],
        )
        result = {
            "engine": engine,
            "path": str(model_dir),
            "checkpoint_path": str(model_dir / KOKORO_CHECKPOINT),
            "config_path": str(model_dir / "config.json"),
            "source": KOKORO_REPO,
            "revision": revision,
            "sample_rate": 24000,
        }
    else:
        log("下载 F5-TTS v1 权重与词表。")
        revision = api.model_info(F5_REPO, timeout=30).sha
        _download_weight(api, F5_REPO, revision, F5_CHECKPOINT, model_dir, log)
        snapshot_download(
            repo_id=F5_REPO,
            revision=revision,
            local_dir=str(model_dir),
            max_workers=1,
            allow_patterns=[F5_VOCAB, "LICENSE", "README.md"],
        )
        log("下载本地 Vocos 声码器。")
        vocoder_revision = api.model_info(VOCOS_REPO, timeout=30).sha
        vocoder_path = model_dir / "vocos"
        _download_weight(api, VOCOS_REPO, vocoder_revision, "pytorch_model.bin", vocoder_path, log)
        snapshot_download(
            repo_id=VOCOS_REPO,
            revision=vocoder_revision,
            local_dir=str(vocoder_path),
            max_workers=1,
            allow_patterns=["config.yaml", "LICENSE", "README.md"],
        )
        result = {
            "engine": engine,
            "path": str(model_dir),
            "checkpoint_path": str(model_dir / F5_CHECKPOINT),
            "vocab_path": str(model_dir / F5_VOCAB),
            "vocoder_path": str(vocoder_path),
            "model_type": "F5TTS_v1_Base",
            "source": F5_REPO,
            "revision": revision,
            "vocoder_revision": vocoder_revision,
            "sample_rate": 24000,
        }
    readiness = model_readiness(result)
    if not readiness["ready"]:
        raise RuntimeError("模型下载不完整：" + ", ".join(readiness["missing"]))
    log("本地模型文件已就绪。")
    return result


@contextmanager
def _offline() -> Iterator[None]:
    """Make accidental Hugging Face lookups fail promptly during inference."""
    from huggingface_hub import constants as hub_constants

    names = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    previous = {name: os.environ.get(name) for name in names}
    previous_hub_offline = hub_constants.HF_HUB_OFFLINE
    transformers_hub = sys.modules.get("transformers.utils.hub")
    previous_transformers_offline = getattr(transformers_hub, "_is_offline_mode", previous_hub_offline)
    hub_constants.HF_HUB_OFFLINE = True
    if transformers_hub is not None and hasattr(transformers_hub, "_is_offline_mode"):
        transformers_hub._is_offline_mode = True
    for name in names:
        os.environ[name] = "1"
    try:
        yield
    finally:
        hub_constants.HF_HUB_OFFLINE = previous_hub_offline
        transformers_hub = sys.modules.get("transformers.utils.hub")
        if transformers_hub is not None and hasattr(transformers_hub, "_is_offline_mode"):
            transformers_hub._is_offline_mode = previous_transformers_offline
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def release_models() -> None:
    """Release cached engines before local training or a model/device change."""
    with _LOCK:
        _CACHE.clear()
        gc.collect()
        if importlib.util.find_spec("torch") is not None:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _device(requested: str) -> str:
    import torch

    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("device 必须为 auto、cpu、cuda 或 mps。")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前环境没有可用 CUDA GPU，请使用 cpu 或安装兼容的 PyTorch。")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("当前环境没有可用 MPS 设备。")
    if requested == "cpu":
        # Match the local trainer and avoid oversubscribing CPU kernels in each
        # isolated worker on machines with many logical cores.
        torch.set_num_threads(min(torch.get_num_threads(), 8))
    return requested


def _file_key(path: Path) -> tuple[str, int, int]:
    info = path.stat()
    return str(path), info.st_mtime_ns, info.st_size


def _kokoro(model: dict, preset: dict, text: str, device: str) -> tuple[Any, int]:
    import numpy as np
    import torch
    from kokoro import KModel, KPipeline

    language = str(preset.get("language", "a")).lower()
    if language not in {"a", "b", "z"}:
        raise ValueError("Kokoro 当前支持 a（美式英语）、b（英式英语）、z（普通话）。")
    voice = str(preset.get("voice") or "")
    if not re.fullmatch(r"[abz][fm]_[a-z0-9_]+", voice) or voice[0] != language:
        raise ValueError("请选择与语言一致的 Kokoro 预设音色。")
    root = Path(model["path"]).expanduser().resolve()
    voice_path = _require_file(root / "voices" / f"{voice}.pt")
    checkpoint = _require_file(_asset_path(model, "checkpoint_path", KOKORO_CHECKPOINT))
    config = _require_file(_asset_path(model, "config_path", "config.json"))
    if language in {"a", "b"}:
        import spacy.util

        if not spacy.util.is_package("en_core_web_sm"):
            raise RuntimeError("英文分词模型未安装。请先安装 en_core_web_sm，再离线合成英文。")
    key = ("kokoro", _file_key(checkpoint), _file_key(config), device)
    if _CACHE.get("key") != key:
        release_models()
        _CACHE.update(key=key, engine=KModel(repo_id=KOKORO_REPO, config=str(config), model=str(checkpoint)).to(device).eval(), pipelines={})
    pipelines = _CACHE["pipelines"]
    if language not in pipelines:
        pipelines[language] = KPipeline(lang_code=language, repo_id=KOKORO_REPO, model=_CACHE["engine"], device=device)
    chunks = []
    # Upstream truncates Chinese phonemes at 510 characters. Measure the actual
    # G2P output and split the source text before invoking the model.
    segments = _text_segments(text, max_chars=80 if language == "z" else 250)
    with torch.inference_mode():
        for segment in segments:
            pipeline = pipelines[language]
            if language == "z":
                results = (
                    result
                    for _, phonemes in _phoneme_segments(segment, pipeline.g2p)
                    for result in pipeline.generate_from_tokens(phonemes, voice=str(voice_path), speed=float(preset.get("speed", 1.0)))
                )
            else:
                results = pipeline(segment, voice=str(voice_path), speed=float(preset.get("speed", 1.0)))
            for result in results:
                audio = result.audio if hasattr(result, "audio") else result[2]
                if audio is not None:
                    chunks.append(audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio))
    if not chunks:
        raise RuntimeError("模型没有生成音频，请检查文本是否包含可朗读的内容。")
    return np.concatenate(chunks), 24000


def _phoneme_segments(text: str, phonemize: Callable[[str], tuple[str, object]], limit: int = 510) -> list[tuple[str, str]]:
    """Preserve every source character while respecting the model phoneme limit."""
    pending = [text]
    result = []
    while pending:
        segment = pending.pop()
        phonemes, _ = phonemize(segment)
        if len(phonemes) <= limit:
            if phonemes:
                result.append((segment, phonemes))
            continue
        if len(segment) < 2:
            raise ValueError("单个字符的音素长度超出模型限制，请检查输入文本。")
        cut = len(segment) // 2
        pending.extend((segment[cut:], segment[:cut]))
    return result


def _text_segments(text: str, max_chars: int) -> list[str]:
    sentences = re.split(r"(?<=[。！？!?;；\n])", text)
    segments = []
    for sentence in sentences:
        sentence = sentence.strip()
        while len(sentence) > max_chars:
            cut = max(sentence.rfind(mark, 0, max_chars) for mark in (" ", "，", ",", "、"))
            cut = cut + 1 if cut >= max_chars // 2 else max_chars
            segments.append(sentence[:cut])
            sentence = sentence[cut:].strip()
        if sentence:
            segments.append(sentence)
    return segments


def _f5(model: dict, preset: dict, text: str, device: str) -> tuple[Any, int]:
    import numpy as np
    import soundfile as sf

    ref_text = str(preset.get("reference_text") or "").strip()
    if not ref_text:
        raise ValueError("F5-TTS 需要参考音频的准确文本；填写后可完全离线合成。")
    reference = Path(preset.get("reference_audio") or "").expanduser().resolve()
    _require_file(reference)
    info = sf.info(str(reference))
    if not 0.5 <= info.duration <= 12.0:
        raise ValueError("参考音频需为 0.5–12 秒，并与参考文本完整对应。")
    reference_wave, _ = sf.read(str(reference), dtype="float32")
    if not np.isfinite(reference_wave).all() or float(np.abs(reference_wave).max()) <= 1e-6:
        raise ValueError("参考音频没有有效语音，请使用清晰、非静音的录音。")
    checkpoint = _require_file(_asset_path(model, "checkpoint_path", F5_CHECKPOINT))
    vocab = _require_file(_asset_path(model, "vocab_path", F5_VOCAB))
    vocoder = _asset_path(model, "vocoder_path", "vocos")
    _require_file(vocoder / "config.yaml")
    _require_file(vocoder / "pytorch_model.bin")
    model_type = model.get("model_type", "F5TTS_v1_Base")
    if model_type not in {"F5TTS_v1_Base", "F5TTS_Base"}:
        raise ValueError(f"不支持的 F5 模型结构：{model_type}")
    use_ema = bool(model.get("use_ema", True))
    key = ("f5", _file_key(checkpoint), _file_key(vocab), _file_key(vocoder / "pytorch_model.bin"), model_type, use_ema, device)
    if _CACHE.get("key") != key:
        from f5_tts.api import F5TTS

        release_models()
        engine = F5TTS(
            model=model_type,
            ckpt_file=str(checkpoint),
            vocab_file=str(vocab),
            vocoder_local_path=str(vocoder),
            device=device,
            use_ema=use_ema,
        )
        _CACHE.update(key=key, engine=engine)
    wave, sample_rate, _ = _CACHE["engine"].infer(
        ref_file=str(reference),
        ref_text=ref_text,
        gen_text=text,
        show_info=lambda _: None,
        progress=None,
        nfe_step=int(preset.get("steps", 32)),
        speed=float(preset.get("speed", 1.0)),
        seed=int(preset.get("seed", 42)),
    )
    return wave, int(sample_rate)


def synthesize(model: dict, preset: dict, text: str, output_path: Path, device: str = "auto") -> dict:
    """Synthesize real PCM WAV audio from an explicitly installed local model."""
    import numpy as np
    import soundfile as sf

    text = text.strip()
    if not text:
        raise ValueError("请输入要合成的文本。")
    speed = float(preset.get("speed", 1.0))
    if not math.isfinite(speed) or not 0.5 <= speed <= 2.0:
        raise ValueError("语速必须在 0.5–2.0 之间。")
    steps = int(preset.get("steps", 32))
    if not 4 <= steps <= 128:
        raise ValueError("采样步数必须在 4–128 之间。")
    seed = int(preset.get("seed", 42))
    if not 0 <= seed < 2**32:
        raise ValueError("随机种子必须在 0–4294967295 之间。")
    output_path = Path(output_path).expanduser().resolve()
    if output_path.suffix.lower() != ".wav":
        raise ValueError("本地推理输出格式为 WAV，请使用 .wav 文件扩展名。")
    readiness = model_readiness(model)
    if not readiness["ready"]:
        raise FileNotFoundError("模型文件尚未就绪：" + ", ".join(readiness["missing"]))
    with _LOCK, _offline():
        import torch

        selected_device = _device(device)
        torch.manual_seed(seed)
        if model.get("engine") == "kokoro":
            wave, sample_rate = _kokoro(model, preset, text, selected_device)
        elif model.get("engine") == "f5":
            wave, sample_rate = _f5(model, preset, text, selected_device)
        else:
            raise ValueError(f"不支持的引擎：{model.get('engine')}")
        wave = np.asarray(wave, dtype=np.float32).reshape(-1)
        if wave.size == 0 or not np.isfinite(wave).all() or float(np.abs(wave).max()) <= 1e-8:
            raise RuntimeError("模型未生成有效音频，请检查参考音频和模型文件。")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), wave, sample_rate, subtype="PCM_16")
        return {
            "path": str(output_path),
            "duration": round(len(wave) / sample_rate, 4),
            "duration_seconds": round(len(wave) / sample_rate, 4),
            "sample_rate": sample_rate,
            "device": selected_device,
        }
