"""Real local RVC v2 singing voice conversion, with explicit model assets.

Large libraries are imported only by workers and checkpoint inspection; status
checks never load Torch or make a network request. Inference uses local files.
"""

from __future__ import annotations

from collections.abc import Callable
import importlib.util
import math
import os
from pathlib import Path
import sys
from typing import Any


CONTENTVEC_REPO = "IAHispano/Applio"
CONTENTVEC_WEIGHT = "Resources/embedders/contentvec/pytorch_model.bin"
CONTENTVEC_CONFIG = "Resources/embedders/contentvec/config.json"
RVC_REPO = "lj1995/VoiceConversionWebUI"
PRETRAINED_G = "pretrained_v2/f0G40k.pth"
PRETRAINED_D = "pretrained_v2/f0D40k.pth"
DEFAULT_CONFIG_40K = [1025, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11],
    [[1, 3, 5], [1, 3, 5], [1, 3, 5]], [10, 10, 2, 2], 512, [16, 16, 4, 4], 109, 256, 40000]


def _path(model: dict, key: str, default: str) -> Path:
    root = Path(model["path"]).expanduser().resolve()
    value = Path(model.get(key) or default).expanduser()
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def model_readiness(model: dict) -> dict:
    encoder = _path(model, "encoder_path", "contentvec")
    required = [encoder / "config.json", encoder / "pytorch_model.bin", _path(model, "f0_path", "rmvpe.pt")]
    if model.get("role") == "support":
        required += [_path(model, "pretrained_g_path", "f0G40k.pth"), _path(model, "pretrained_d_path", "f0D40k.pth")]
    else:
        required.append(_path(model, "checkpoint_path", "model.pth"))
        if model.get("index_path"):
            required.append(_path(model, "index_path", ""))
    missing = [str(path) for path in required if not path.is_file() or not path.stat().st_size]
    return {"ready": not missing, "missing": missing, "role": model.get("role", "voice")}


def download_assets(model_dir: Path, log: Callable[[str], None] = print) -> dict:
    """Download official ContentVec, RMVPE and v2 40k pretrained GAN weights."""
    from urllib.request import getproxies
    from audio_studio.engines import _download_weight

    if sys.platform == "win32":
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        for scheme, proxy in getproxies().items():
            if scheme in {"http", "https"}:
                os.environ.setdefault(scheme.upper() + "_PROXY", proxy)
    from huggingface_hub import HfApi, snapshot_download, constants
    if os.environ.get("HF_HUB_DISABLE_XET") == "1":
        constants.HF_HUB_DISABLE_XET = True
    root = Path(model_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    encoder_revision = api.model_info(CONTENTVEC_REPO, timeout=30).sha
    rvc_revision = api.model_info(RVC_REPO, timeout=30).sha
    log("下载 ContentVec 内容编码器、RMVPE 音高模型和 RVC v2 40k 预训练模型。")
    _download_weight(api, CONTENTVEC_REPO, encoder_revision, CONTENTVEC_WEIGHT, root, log)
    snapshot_download(repo_id=CONTENTVEC_REPO, revision=encoder_revision, local_dir=str(root),
        allow_patterns=[CONTENTVEC_CONFIG, "README.md"], max_workers=1)
    for filename in ("rmvpe.pt", PRETRAINED_G, PRETRAINED_D):
        _download_weight(api, RVC_REPO, rvc_revision, filename, root, log)
    snapshot_download(repo_id=RVC_REPO, revision=rvc_revision, local_dir=str(root / "rvc-model-card"),
        allow_patterns=["README.md", "LICENSE*"], max_workers=1)
    result = {"engine": "rvc", "role": "support", "path": str(root),
        "encoder_path": str(root / Path(CONTENTVEC_CONFIG).parent), "f0_path": str(root / "rmvpe.pt"),
        "pretrained_g_path": str(root / PRETRAINED_G), "pretrained_d_path": str(root / PRETRAINED_D),
        "source": RVC_REPO, "revision": rvc_revision, "encoder_source": CONTENTVEC_REPO,
        "encoder_revision": encoder_revision, "sample_rate": 40000, "version": "v2", "f0": 1}
    if not model_readiness(result)["ready"]:
        raise RuntimeError("RVC 支持模型下载不完整，请重试下载。")
    log("RVC 支持模型已完成大小与 SHA256 校验，目标音色仍需训练或注册。")
    return result


def _checkpoint(path: Path) -> tuple[dict, list, dict]:
    import torch

    path = Path(path).expanduser().resolve()
    if not path.is_file() or path.suffix.lower() not in {".pth", ".pt"}:
        raise ValueError("RVC 音色检查点必须是本地 .pth 或 .pt 文件。")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError("无法安全读取 RVC 检查点，请使用包含张量权重和配置的 RVC v2 导出文件。") from exc
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("config"), (list, tuple)) or not isinstance(checkpoint.get("weight"), dict):
        raise ValueError("该文件不是 RVC 导出的音色模型：需要 config 和 weight。")
    if checkpoint.get("version", "v1") != "v2" or checkpoint.get("f0", 1) != 1:
        raise ValueError("翻唱目前仅支持带音高条件的 RVC v2 模型；不支持 v1 或无 F0 模型。")
    config, weights = list(checkpoint["config"]), checkpoint["weight"]
    if len(config) != 18 or not all(isinstance(value, torch.Tensor) for value in weights.values()):
        raise ValueError("RVC 模型配置或权重格式无效。")
    rate = {"32k": 32000, "40k": 40000, "48k": 48000}.get(str(config[-1]), config[-1])
    if rate not in {32000, 40000, 48000}:
        raise ValueError("RVC 模型采样率须为 32000、40000 或 48000。")
    speaker_weight = weights.get("emb_g.weight")
    phone_weight = weights.get("enc_p.emb_phone.weight")
    if speaker_weight is None or speaker_weight.ndim != 2 or not 1 <= speaker_weight.shape[0] <= 10000:
        raise ValueError("RVC 模型缺少有效的说话人嵌入。")
    if phone_weight is None or phone_weight.ndim != 2 or phone_weight.shape[1] != 768:
        raise ValueError("RVC v2 模型必须使用 768 维内容特征。")
    if not all(isinstance(config[i], int) and 0 < config[i] <= 8192 for i in (0, 1, 2, 3, 4, 5, 6, 7, 13, 16)):
        raise ValueError("RVC 模型网络尺寸配置无效。")
    config[-3], config[-1] = speaker_weight.shape[0], int(rate)
    details = {"checkpoint_path": str(path), "version": "v2", "f0": 1, "sample_rate": int(rate),
        "speaker_count": int(speaker_weight.shape[0]), "feature_dim": 768}
    return checkpoint, config, details


def inspect_checkpoint(path: Path) -> dict:
    """Read tensor-only RVC exports and return JSON-compatible voice metadata."""
    network, details = _load_generator(path, "cpu")
    del network
    return details


def load_encoder(encoder_path: Path, device: str = "cpu") -> Any:
    import torch
    from transformers import HubertModel
    from audio_studio.engines import _offline

    path = Path(encoder_path).expanduser().resolve()
    if not all((path / name).is_file() for name in ("config.json", "pytorch_model.bin")):
        raise ValueError("缺少 ContentVec 编码器，请先下载 RVC 支持模型。")
    with _offline():
        encoder = HubertModel.from_pretrained(str(path), local_files_only=True, dtype=torch.float32,
            attn_implementation="eager")
    return encoder.to(device).eval()


def extract_features(encoder: Any, audio16k: Any, device: str = "cpu") -> Any:
    """Return native 50Hz [time,768] ContentVec features as a float32 ndarray."""
    import numpy as np
    import torch

    waveform = np.asarray(audio16k, dtype="float32")
    if waveform.ndim != 1 or len(waveform) < 400 or not np.isfinite(waveform).all():
        raise ValueError("内容特征输入须为有效的 16kHz 单声道音频。")
    with torch.inference_mode():
        features = encoder(torch.from_numpy(waveform).unsqueeze(0).to(device))["last_hidden_state"][0]
    if features.shape[-1] != 768 or not torch.isfinite(features).all():
        raise RuntimeError("ContentVec 返回了无效或不兼容的内容特征。")
    return features.float().cpu().numpy()


def load_pitch_extractor(f0_path: Path, device: str = "cpu") -> Any:
    from audio_studio.vendor.rvc.rmvpe import RMVPE

    path = Path(f0_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError("缺少 RMVPE 音高模型，请先下载 RVC 支持模型。")
    return RMVPE(str(path), is_half=False, device=device)


def extract_pitch(extractor: Any, audio16k: Any, frames: int, pitch_shift: int = 0) -> tuple[Any, Any]:
    """Return RVC pitch bins and continuous Hz at 10ms, padding only at the end."""
    import numpy as np

    if not 1 <= frames <= 200000 or not -24 <= pitch_shift <= 24:
        raise ValueError("音高帧数或移调范围无效。")
    f0 = np.asarray(extractor.infer_from_audio(np.asarray(audio16k, dtype="float32"), thred=0.03), dtype="float32")
    if not len(f0) or not np.isfinite(f0).all():
        raise RuntimeError("RMVPE 返回了无效的音高。")
    if len(f0) < frames:
        f0 = np.pad(f0, (0, frames - len(f0)), mode="edge")
    f0 = np.maximum(f0[:frames], 0) * (2.0 ** (pitch_shift / 12.0))
    mel = 1127 * np.log1p(f0 / 700)
    low, high = 1127 * math.log1p(50 / 700), 1127 * math.log1p(1100 / 700)
    mel[mel > 0] = (mel[mel > 0] - low) * 254 / (high - low) + 1
    coarse = np.rint(np.clip(mel, 1, 255)).astype("int64")
    return coarse, f0.astype("float32")


def _load_generator(path: Path, device: str) -> tuple[Any, dict]:
    from audio_studio.vendor.rvc.models import SynthesizerTrnMs768NSFsid

    checkpoint, config, details = _checkpoint(path)
    network = SynthesizerTrnMs768NSFsid(*config, is_half=False)
    del network.enc_q
    weights = {key: value for key, value in checkpoint["weight"].items() if not key.startswith("enc_q.")}
    try:
        network.load_state_dict(weights, strict=True)
    except RuntimeError as exc:
        raise ValueError("RVC 权重与其网络配置不匹配，无法加载该音色。") from exc
    return network.float().to(device).eval(), details


def _index(model: dict, index_rate: float) -> tuple[Any, Any]:
    if not model.get("index_path") or index_rate == 0:
        return None, None
    if importlib.util.find_spec("faiss") is None:
        raise RuntimeError("使用 RVC 检索索引需要安装 faiss-cpu；也可将索引混合率设为 0。")
    import faiss
    import numpy as np

    path = _path(model, "index_path", "")
    if not path.is_file():
        raise ValueError("登记的 RVC 索引文件不存在。")
    index = faiss.read_index(str(path))
    if index.d != 768 or index.ntotal < 1:
        raise ValueError("RVC 索引须含非空的 768 维内容特征。")
    vectors = np.asarray(index.reconstruct_n(0, index.ntotal), dtype="float32")
    if not np.isfinite(vectors).all():
        raise ValueError("RVC 检索索引包含无效数值。")
    return index, vectors


def convert(model: dict, source_path: Path, output_path: Path, pitch_shift: int = 0, index_rate: float = .5,
            speaker_id: int = 0, device: str = "auto", seed: int = 42, log: Callable[[str], None] = print) -> dict:
    """Convert an existing sung vocal with a trained RVC v2 target voice."""
    if model.get("role") == "support":
        raise ValueError("RVC 支持模型用于提取特征和训练，请先训练或注册目标音色。")
    if not -24 <= pitch_shift <= 24 or not math.isfinite(index_rate) or not 0 <= index_rate <= 1:
        raise ValueError("移调须在 -24 至 24 半音之间，索引混合率须在 0 至 1 之间。")
    ready = model_readiness(model)
    if not ready["ready"]:
        raise ValueError("目标音色缺少本地模型资源：" + ", ".join(ready["missing"]))
    import numpy as np
    import torch
    import soundfile as sf
    from scipy import signal
    from audio_studio.engines import _device

    selected = _device(device)
    torch.manual_seed(seed)
    if selected == "cuda":
        torch.cuda.manual_seed_all(seed)
    source = Path(source_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if source == output:
        raise ValueError("输出文件须与输入演唱文件不同。")
    waveform, source_rate = sf.read(source, dtype="float32", always_2d=True)
    if not 8000 <= source_rate <= 192000 or not 1 <= waveform.shape[1] <= 8:
        raise ValueError("输入演唱的采样率或声道数不受支持。")
    if not .3 <= len(waveform) / source_rate <= 1800 or not np.isfinite(waveform).all():
        raise ValueError("输入须为 0.3–1800 秒且数值有效的演唱音频。")
    mono = waveform.mean(axis=1)
    if np.max(np.abs(mono)) < 1e-5:
        raise ValueError("输入演唱仅包含静音。")
    divisor = math.gcd(source_rate, 16000)
    audio = signal.resample_poly(mono, 16000 // divisor, source_rate // divisor).astype("float32")
    audio = signal.sosfilt(signal.butter(5, 48, fs=16000, btype="highpass", output="sos"), audio).astype("float32")
    peak = float(np.max(np.abs(audio)))
    if peak > .95:
        audio *= .95 / peak
    network, details = _load_generator(_path(model, "checkpoint_path", "model.pth"), selected)
    if not 0 <= speaker_id < details["speaker_count"]:
        raise ValueError("说话人编号超出该 RVC 模型的范围。")
    encoder = load_encoder(_path(model, "encoder_path", "contentvec"), selected)
    extractor = load_pitch_extractor(_path(model, "f0_path", "rmvpe.pt"), selected)
    index, vectors = _index(model, index_rate)
    effective_rate = index_rate if index is not None else 0.0
    rate = details["sample_rate"]
    context, chunk_size = 16000, 320000
    padded = np.pad(audio, (context, context), mode="reflect")
    pieces = []
    for start in range(0, len(audio), chunk_size):
        end = min(start + chunk_size, len(audio))
        part = padded[start:end + 2 * context]
        extra = (-len(part)) % 160
        if extra:
            part = np.pad(part, (0, extra), mode="reflect")
        frames = len(part) // 160
        log(f"RVC 翻唱：{start / 16000:.1f}–{end / 16000:.1f} 秒 / {len(audio) / 16000:.1f} 秒。")
        features = extract_features(encoder, part, selected)
        original = features.copy()
        if index is not None:
            scores, neighbors = index.search(np.ascontiguousarray(features), min(8, index.ntotal))
            if (neighbors < 0).any():
                raise ValueError("RVC 索引返回了无效的检索位置。")
            weights = 1 / np.maximum(scores, 1e-6) ** 2
            weights /= weights.sum(axis=1, keepdims=True)
            retrieved = np.sum(vectors[neighbors] * weights[..., None], axis=1)
            features = features * (1 - index_rate) + retrieved * index_rate
        features = np.repeat(features, 2, axis=0)
        original = np.repeat(original, 2, axis=0)
        if len(features) < frames:
            features = np.pad(features, ((0, frames - len(features)), (0, 0)), mode="edge")
            original = np.pad(original, ((0, frames - len(original)), (0, 0)), mode="edge")
        features, original = features[:frames], original[:frames]
        coarse, continuous = extract_pitch(extractor, part, frames, pitch_shift)
        unvoiced = continuous < 1
        features[unvoiced] = features[unvoiced] * .33 + original[unvoiced] * .67
        with torch.inference_mode():
            result = network.infer(torch.from_numpy(features).unsqueeze(0).to(selected),
                torch.tensor([frames], device=selected), torch.from_numpy(coarse).unsqueeze(0).to(selected),
                torch.from_numpy(continuous).unsqueeze(0).to(selected), torch.tensor([speaker_id], device=selected))[0][0, 0]
        converted = result.float().cpu().numpy()
        count = round((end - start) / 16000 * rate)
        core = converted[rate:rate + count]
        if len(core) != count:
            raise RuntimeError("RVC 模型生成长度与配置采样率不一致。")
        pieces.append(core)
    joined = np.concatenate(pieces)
    expected = round(len(waveform) / source_rate * rate)
    if abs(len(joined) - expected) > 2:
        raise RuntimeError("RVC 输出长度与源演唱不一致。")
    joined = np.pad(joined, (0, max(0, expected - len(joined))))[:expected]
    if not np.isfinite(joined).all() or np.max(np.abs(joined)) < 1e-7:
        raise RuntimeError("RVC 返回无效或静音音频，未保存生成结果。")
    maximum = float(np.max(np.abs(joined)))
    if maximum > .99:
        joined *= .99 / maximum
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, joined, rate, subtype="PCM_16")
    return {"path": str(output), "source_path": str(source), "engine": "rvc", "sample_rate": rate,
        "duration_seconds": expected / rate, "pitch_shift": pitch_shift, "index_rate": effective_rate,
        "speaker_id": speaker_id, "seed": seed, "version": "v2"}
