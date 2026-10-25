"""Local RVC v2/40 kHz fine-tuning with actual adversarial updates.

Speech-content features and F0 labels come from the same local encoders used by
conversion. Recordings need no transcript. The trainer performs no downloads and
does not modify the source recordings.
"""

from __future__ import annotations

import gc
import importlib.util
import json
import math
from pathlib import Path
import random
import threading
from typing import Any, Callable
from uuid import uuid4

from audio_studio.training import TrainingCancelled

SAMPLE_RATE = 40_000
HOP_LENGTH = 400
FFT_SIZE = 2048
N_MELS = 125
MIN_DURATION = .5
MAX_DURATION = 600


def validate_voice_dataset(wavs_dir: Path) -> dict[str, Any]:
    """Inspect clean target-speaker WAVs; long recordings are split on preparation."""
    import numpy as np
    import soundfile as sf

    directory = Path(wavs_dir).expanduser().resolve()
    report: dict[str, Any] = {"valid": False, "csv_path": "", "wavs_dir": str(directory),
        "errors": [], "warnings": [], "samples": [], "sample_count": 0, "row_count": 0,
        "total_duration_seconds": 0.0, "task": "svc"}
    if not directory.is_dir():
        report["errors"].append("目标音色录音目录不存在。")
        return report
    paths = sorted(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() == ".wav")
    report["row_count"] = len(paths)
    if not paths:
        report["errors"].append("目录内没有 WAV 录音。")
    for path in paths:
        try:
            info = sf.info(path)
            if info.format != "WAV":
                raise ValueError("文件实际格式不是 WAV")
            if not MIN_DURATION <= info.duration <= MAX_DURATION:
                raise ValueError("每段录音须为 0.5–600 秒")
            if not 8000 <= info.samplerate <= 192000 or not 1 <= info.channels <= 8:
                raise ValueError("录音采样率或声道数不受支持")
            waveform, _ = sf.read(path, dtype="float32", always_2d=True)
            if not np.isfinite(waveform).all():
                raise ValueError("录音包含非有限数值")
            mono = waveform.mean(axis=1)
            peak = float(np.max(np.abs(mono)))
            if peak < 1e-5 or float(np.sqrt(np.mean(mono.astype(np.float64) ** 2))) < 1e-6:
                raise ValueError("录音只有静音，或合并声道后信号抵消")
            if info.channels > 1:
                report["warnings"].append(f"{path.name}：准备时会转换为单声道。")
            if peak >= .99:
                report["warnings"].append(f"{path.name}：峰值接近满幅，请检查削波。")
            if info.duration > 3.5:
                report["warnings"].append(f"{path.name}：准备时会切分为约 3 秒片段。")
            report["samples"].append({"audio_path": str(path.resolve()), "duration_seconds": info.duration,
                "sample_rate": info.samplerate, "channels": info.channels})
            report["total_duration_seconds"] += info.duration
        except (OSError, RuntimeError, ValueError) as exc:
            report["errors"].append(f"{path.name}：{exc}")
    report["sample_count"] = len(report["samples"])
    if report["total_duration_seconds"] and report["total_duration_seconds"] < 300:
        report["warnings"].append("有效录音不足 5 分钟；可验证训练流程，但音色相似度、音域覆盖及泛化需要更多干净录音评估。")
    report["valid"] = bool(report["samples"]) and not report["errors"]
    return report


def _check_cancel(cancel: threading.Event) -> None:
    if cancel.is_set():
        raise TrainingCancelled("目标音色训练已取消。")


def _integer(config: dict, name: str, default: int) -> int:
    value = config.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} 必须为正整数")
    return value


def _prepare_dataset(report: dict, assets: dict, destination: Path, device: str,
                     log: Callable[[str], None], cancel: threading.Event) -> list[dict]:
    import librosa
    import numpy as np
    import soundfile as sf
    from scipy import signal
    import torch
    from audio_studio.svc import load_encoder, extract_features, load_pitch_extractor, extract_pitch

    destination.mkdir(parents=True, exist_ok=True)
    encoder = load_encoder(Path(assets["encoder_path"]), device)
    pitch_extractor = load_pitch_extractor(Path(assets["f0_path"]), device)
    records: list[dict] = []
    highpass = signal.butter(5, 48, fs=SAMPLE_RATE, btype="highpass", output="sos")
    try:
        for source in report["samples"]:
            _check_cancel(cancel)
            audio, rate = sf.read(source["audio_path"], dtype="float32", always_2d=True)
            audio = audio.mean(axis=1)
            if rate != SAMPLE_RATE:
                audio = librosa.resample(audio, orig_sr=rate, target_sr=SAMPLE_RATE)
            audio = signal.sosfilt(highpass, audio).astype(np.float32)
            # Use the upstream RVC amplitude mixture rather than hard-normalizing
            # every sample independently and erasing the recording's dynamics.
            peak = float(np.max(np.abs(audio)))
            audio = audio / max(peak, 1e-6) * .9 * .75 + audio * .25
            for start in range(0, len(audio), 3 * SAMPLE_RATE):
                _check_cancel(cancel)
                chunk = audio[start:start + 3 * SAMPLE_RATE]
                if len(chunk) < MIN_DURATION * SAMPLE_RATE or float(np.max(np.abs(chunk))) < 1e-5:
                    continue
                audio16k = librosa.resample(chunk, orig_sr=SAMPLE_RATE, target_sr=16000).astype(np.float32)
                phones = np.repeat(extract_features(encoder, audio16k, device), 2, axis=0)
                frames = min(len(phones), len(chunk) // HOP_LENGTH)
                if frames < 32:
                    continue
                coarse, continuous = extract_pitch(pitch_extractor, audio16k, frames)
                if not np.isfinite(phones).all() or not np.isfinite(continuous).all():
                    raise ValueError("音色特征或 F0 标签包含非有限数值。")
                sample_id = f"{len(records):06d}"
                wav_path, feature_path = destination / f"{sample_id}.wav", destination / f"{sample_id}.npz"
                sf.write(wav_path, chunk[:frames * HOP_LENGTH], SAMPLE_RATE, subtype="PCM_16")
                np.savez(feature_path, phone=phones[:frames].astype(np.float32), pitch=coarse.astype(np.int64),
                         pitchf=continuous.astype(np.float32))
                records.append({"audio_path": str(wav_path), "feature_path": str(feature_path), "frames": frames,
                    "source_path": source["audio_path"], "source_offset_seconds": start / SAMPLE_RATE})
                log(f"已准备音色训练片段 {len(records)}：{frames / 100:.2f} 秒")
    finally:
        del encoder, pitch_extractor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not records:
        raise ValueError("没有得到可训练的非静音片段。")
    (destination / "manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return records


def _spectrogram(waveform: Any) -> Any:
    import torch
    import torch.nn.functional as F
    padded = F.pad(waveform.unsqueeze(1), ((FFT_SIZE - HOP_LENGTH) // 2,) * 2, mode="reflect").squeeze(1)
    spectrum = torch.stft(padded, FFT_SIZE, hop_length=HOP_LENGTH, win_length=FFT_SIZE,
        window=torch.hann_window(FFT_SIZE, device=waveform.device, dtype=waveform.dtype),
        center=False, return_complex=True)
    return torch.sqrt(spectrum.real.square() + spectrum.imag.square() + 2e-7)


def _mel(spectrum: Any, basis: Any) -> Any:
    import torch
    return torch.log(torch.clamp(torch.matmul(basis, spectrum.float()), min=2e-6))


def _batch(records: list[dict], device: str) -> tuple[Any, ...]:
    import numpy as np
    import soundfile as sf
    import torch
    items = []
    for item in records:
        wave, rate = sf.read(item["audio_path"], dtype="float32")
        if rate != SAMPLE_RATE or wave.ndim != 1:
            raise ValueError("准备后的录音必须是 40kHz 单声道。")
        with np.load(item["feature_path"], allow_pickle=False) as labels:
            phone = torch.from_numpy(labels["phone"].copy())
            pitch = torch.from_numpy(labels["pitch"].copy())
            pitchf = torch.from_numpy(labels["pitchf"].copy())
        waveform = torch.from_numpy(wave).unsqueeze(0)
        spec = _spectrogram(waveform)[0]
        frames = min(len(phone), len(pitch), len(pitchf), spec.shape[-1])
        if frames < 1 or phone.ndim != 2 or phone.shape[1] != 768:
            raise ValueError("音色特征形状无效。")
        items.append((phone[:frames], pitch[:frames], pitchf[:frames], spec[:, :frames], waveform[:, :frames * HOP_LENGTH]))
    frames = [len(item[0]) for item in items]
    longest = max(frames)
    phone = torch.zeros(len(items), longest, 768)
    pitch = torch.zeros(len(items), longest, dtype=torch.long)
    pitchf = torch.zeros(len(items), longest)
    spec = torch.zeros(len(items), FFT_SIZE // 2 + 1, longest)
    waveform = torch.zeros(len(items), 1, longest * HOP_LENGTH)
    for index, (item, length) in enumerate(zip(items, frames)):
        phone[index, :length], pitch[index, :length], pitchf[index, :length] = item[:3]
        spec[index, :, :length], waveform[index, :, :length * HOP_LENGTH] = item[3:]
    lengths = torch.tensor(frames, dtype=torch.long)
    speaker = torch.zeros(len(items), dtype=torch.long)
    return tuple(item.to(device) for item in (phone, lengths, pitch, pitchf, spec, lengths, speaker, waveform))


def _load_pretrained(network: Any, path: Path) -> None:
    import torch
    if not path.is_file():
        raise ValueError(f"缺少本地 RVC 预训练权重：{path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise ValueError("训练需要包含 model 状态字典的完整 RVC 预训练权重。")
    try:
        network.load_state_dict(checkpoint["model"], strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ValueError("预训练权重与 RVC v2/40k 训练架构不匹配。") from exc


def _atomic_save(value: dict, path: Path) -> None:
    import torch
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run_training(config: dict[str, Any], run_dir: Path, log: Callable[[str], None] = print,
                 cancel: threading.Event | None = None) -> dict[str, Any]:
    """Fine-tune RVC v2 with LSGAN, feature matching, Mel and KL objectives.

    Required: wavs_dir, assets (encoder_path, f0_path, pretrained_g_path,
    pretrained_d_path). Tunables: epochs, batch_size, learning_rate, device,
    segment_frames; max_steps permits bounded acceptance runs. Private model
    settings are only used by small-network regression tests, never the Web API.
    """
    import librosa
    import torch
    import torch.nn.functional as F
    from audio_studio.svc import DEFAULT_CONFIG_40K
    from audio_studio.vendor.rvc.models import SynthesizerTrnMs768NSFsid, MultiPeriodDiscriminatorV2
    from audio_studio.vendor.rvc import commons

    cancel = cancel if cancel is not None else threading.Event()
    _check_cancel(cancel)
    epochs = _integer(config, "epochs", 20)
    batch_size = _integer(config, "batch_size", 1)
    segment_frames = _integer(config, "segment_frames", 32)
    max_steps = _integer(config, "max_steps", 2 ** 63 - 1)
    if not 8 <= segment_frames <= 128:
        raise ValueError("segment_frames 必须介于 8–128。")
    learning_rate = float(config.get("learning_rate", 1e-4))
    if not math.isfinite(learning_rate) or not 1e-8 <= learning_rate <= 1e-2:
        raise ValueError("learning_rate 必须介于 1e-8–1e-2。")
    requested = config.get("device", "auto")
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device 必须为 auto、cpu 或 cuda。")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("已请求 CUDA，但当前 PyTorch 没有可用 CUDA 设备。")
    device = "cuda" if requested != "cpu" and torch.cuda.is_available() else "cpu"
    assets = config.get("assets", {})
    if not all(assets.get(key) for key in ("encoder_path", "f0_path", "pretrained_g_path", "pretrained_d_path")):
        raise ValueError("音色训练需要本地编码器、F0 和 G/D 预训练权重。")
    for key in ("pretrained_g_path", "pretrained_d_path"):
        if not Path(assets[key]).is_file():
            raise ValueError(f"缺少本地 RVC 预训练权重：{assets[key]}")
    report = validate_voice_dataset(Path(config["wavs_dir"]))
    if not report["valid"]:
        raise ValueError("音色数据检查失败：" + "; ".join(report["errors"][:10]))
    run_dir = Path(run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if any((run_dir / filename).exists() for filename in ("model.pth", "training-state.pt", "metrics.jsonl")):
        raise ValueError("训练目录已有产物，请使用新的训练目录。")
    model_config = list(config.get("_model_config", DEFAULT_CONFIG_40K))
    model_config[1] = segment_frames
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    records = _prepare_dataset(report, assets, run_dir / "dataset", device, log, cancel)
    if min(item["frames"] for item in records) < segment_frames:
        raise ValueError("部分训练片段短于 segment_frames，请降低片段长度设置。")
    torch.manual_seed(1234)
    generator = SynthesizerTrnMs768NSFsid(*model_config, is_half=False)
    discriminator = MultiPeriodDiscriminatorV2(False)
    if config.get("_discriminator_indices") is not None:
        discriminator.discriminators = torch.nn.ModuleList([discriminator.discriminators[index]
            for index in config["_discriminator_indices"]])
    _load_pretrained(generator, Path(assets["pretrained_g_path"]))
    _load_pretrained(discriminator, Path(assets["pretrained_d_path"]))
    generator, discriminator = generator.to(device).train(), discriminator.to(device).train()
    optim_g = torch.optim.AdamW(generator.parameters(), lr=learning_rate, betas=(.8, .99), eps=1e-9)
    optim_d = torch.optim.AdamW(discriminator.parameters(), lr=learning_rate, betas=(.8, .99), eps=1e-9)
    schedule_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=.999875)
    schedule_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=.999875)
    mel_basis = torch.from_numpy(librosa.filters.mel(sr=SAMPLE_RATE, n_fft=FFT_SIZE, n_mels=N_MELS,
        fmin=0, fmax=None)).to(device)
    bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda" and not bf16)
    metrics_path = run_dir / "metrics.jsonl"
    steps = 0
    final_metric: dict[str, Any] = {}
    log(f"开始 RVC v2/40k 训练：{len(records)} 段，设备 {device}；未进行音色质量承诺。")

    def save_state(epoch: int) -> None:
        _atomic_save({"generator": generator.state_dict(), "discriminator": discriminator.state_dict(),
            "optimizer_g": optim_g.state_dict(), "optimizer_d": optim_d.state_dict(), "scaler": scaler.state_dict(),
            "scheduler_g": schedule_g.state_dict(), "scheduler_d": schedule_d.state_dict(),
            "model_config": model_config, "epoch": epoch, "steps": steps,
            "torch_rng_state": torch.get_rng_state()}, run_dir / "training-state.pt")

    for epoch in range(1, epochs + 1):
        order = list(range(len(records)))
        random.Random(1234 + epoch).shuffle(order)
        for start in range(0, len(order), batch_size):
            _check_cancel(cancel)
            phone, lengths, pitch, pitchf, spec, spec_lengths, speaker, wave = _batch(
                [records[index] for index in order[start:start + batch_size]], device)
            optim_g.zero_grad(set_to_none=True)
            optim_d.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=device == "cuda"):
                y_hat, ids_slice, _, z_mask, (_, z_p, m_p, logs_p, _, logs_q) = generator(
                    phone, lengths, pitch, pitchf, spec, spec_lengths, speaker)
                real = commons.slice_segments(wave, ids_slice * HOP_LENGTH, segment_frames * HOP_LENGTH)
                real_scores, fake_scores, _, _ = discriminator(real, y_hat.detach())
                loss_d = sum(((1 - real_score.float()).square().mean() + fake_score.float().square().mean())
                    for real_score, fake_score in zip(real_scores, fake_scores))
            if not torch.isfinite(loss_d):
                raise ValueError("判别器 loss 非有限，已停止训练。")
            scaler.scale(loss_d).backward()
            scaler.unscale_(optim_d)
            norm_d = torch.nn.utils.clip_grad_norm_(discriminator.parameters(), float("inf"), error_if_nonfinite=True)
            scaler.step(optim_d)
            discriminator.requires_grad_(False)
            try:
                with torch.autocast(device_type=device, dtype=amp_dtype, enabled=device == "cuda"):
                    _, fake_scores, real_maps, fake_maps = discriminator(real, y_hat)
                adversarial = sum((1 - score.float()).square().mean() for score in fake_scores)
                matching = sum((real_map.float().detach() - fake_map.float()).abs().mean()
                    for real_group, fake_group in zip(real_maps, fake_maps)
                    for real_map, fake_map in zip(real_group, fake_group)) * 2
                mel_loss = F.l1_loss(_mel(_spectrogram(real.float().squeeze(1)), mel_basis),
                    _mel(_spectrogram(y_hat.float().squeeze(1)), mel_basis)) * 45
                kl = logs_p.float() - logs_q.float() - .5
                kl = kl + .5 * (z_p.float() - m_p.float()).square() * torch.exp(-2 * logs_p.float())
                kl_loss = (kl * z_mask.float()).sum() / z_mask.float().sum()
                loss_g = adversarial + matching + mel_loss + kl_loss
                if not torch.isfinite(loss_g):
                    raise ValueError("生成器 loss 非有限，已停止训练。")
                scaler.scale(loss_g).backward()
                scaler.unscale_(optim_g)
                norm_g = torch.nn.utils.clip_grad_norm_(generator.parameters(), float("inf"), error_if_nonfinite=True)
                scaler.step(optim_g)
                scaler.update()
            finally:
                discriminator.requires_grad_(True)
            steps += 1
            final_metric = {"step": steps, "epoch": epoch, "generator_loss": float(loss_g.detach()),
                "discriminator_loss": float(loss_d.detach()), "mel_loss": float(mel_loss.detach()),
                "feature_loss": float(matching.detach()), "adversarial_loss": float(adversarial.detach()),
                "kl_loss": float(kl_loss.detach()), "generator_grad_norm": float(norm_g), "discriminator_grad_norm": float(norm_d)}
            with metrics_path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(final_metric) + "\n")
            log(f"epoch {epoch}/{epochs} · step {steps} · G {final_metric['generator_loss']:.4f} · D {final_metric['discriminator_loss']:.4f}")
            if steps % 100 == 0:
                save_state(epoch)
            if steps >= max_steps:
                break
        schedule_g.step()
        schedule_d.step()
        if steps >= max_steps:
            break
    _check_cancel(cancel)
    save_state(epoch)
    output_path = run_dir / "model.pth"
    export = {"weight": {key: value.detach().cpu().float() for key, value in generator.state_dict().items() if "enc_q" not in key},
        "config": model_config, "f0": 1, "version": "v2", "sr": "40k", "info": f"{epoch}epoch {steps}steps",
        "training": {"steps": steps, "sample_count": len(records), "metrics": final_metric}}
    _atomic_save(export, output_path)
    index_path = ""
    if importlib.util.find_spec("faiss") is not None:
        import faiss
        import numpy as np
        # Bound the index memory even for long recording collections. Sampling
        # across all prepared chunks preserves coverage without duplicating WAVs.
        per_record = max(1, 20000 // len(records))
        vectors = []
        for record in records:
            _check_cancel(cancel)
            with np.load(record["feature_path"]) as features:
                phone = features["phone"]
                positions = np.linspace(0, len(phone) - 1, min(len(phone), per_record), dtype=int)
                vectors.append(phone[positions])
        index = faiss.IndexFlatL2(768)
        for vector in vectors:
            index.add(np.ascontiguousarray(vector, dtype=np.float32))
        destination = run_dir / "model.index"
        faiss.write_index(index, str(destination))
        index_path = str(destination)
        log(f"已导出目标音色检索索引：{index.ntotal} 个内容特征。")
    log(f"目标音色训练完成，已导出可独立部署的 RVC 检查点：{output_path.name}")
    return {"checkpoint_path": str(output_path), "sample_rate": SAMPLE_RATE, "version": "v2", "steps": steps,
        "metrics_path": str(metrics_path), "dataset_path": str(run_dir / "dataset"),
        "training_checkpoint_path": str(run_dir / "training-state.pt"), "loss": final_metric["generator_loss"], "index_path": index_path}
