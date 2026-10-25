"""Real, single-process F5-TTS fine-tuning with portable inference exports."""

from __future__ import annotations

import gc
import json
import math
import random
import threading
from pathlib import Path
from typing import Any, Callable

from .datasets import prepare_dataset, read_vocab

F5_MODEL_ARCH = {
    "dim": 1024,
    "depth": 22,
    "heads": 16,
    "ff_mult": 2,
    "text_dim": 512,
    "conv_layers": 4,
    "text_mask_padding": True,
    "checkpoint_activations": True,
}
MEL_CONFIG = {
    "target_sample_rate": 24_000,
    "n_mel_channels": 100,
    "hop_length": 256,
    "win_length": 1024,
    "n_fft": 1024,
    "mel_spec_type": "vocos",
}


class TrainingCancelled(RuntimeError):
    """The caller requested cancellation before a training batch."""


def _check_cancelled(cancelled: threading.Event) -> None:
    if cancelled.is_set():
        raise TrainingCancelled("Training cancelled.")


def _positive_integer(config: dict[str, Any], key: str, default: int) -> int:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{key} must be a positive integer.")
    return value


def _load_base_weights(model: Any, checkpoint_path: Path) -> None:
    """Load only model weights with a strict architecture/vocabulary match."""
    import torch

    if checkpoint_path.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file

        checkpoint = load_file(str(checkpoint_path), device="cpu")
    elif checkpoint_path.suffix.lower() in {".pt", ".pth"}:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    else:
        raise ValueError("Base checkpoint must be .safetensors, .pt or .pth.")
    if not isinstance(checkpoint, dict):
        raise ValueError("Base checkpoint must contain a model state dictionary.")
    state = checkpoint.get("ema_model_state_dict", checkpoint.get("model_state_dict", checkpoint))
    if not isinstance(state, dict):
        raise ValueError("Checkpoint model weights are invalid.")
    legacy_buffers = {"mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"}
    weights = {
        key.removeprefix("ema_model."): value
        for key, value in state.items()
        if key not in {"initted", "step", "update"}
        and key.removeprefix("ema_model.") not in legacy_buffers
        and torch.is_tensor(value)
    }
    # A mismatch must fail rather than silently train a partially random model.
    model.load_state_dict(weights, strict=True)
    del checkpoint, state, weights
    gc.collect()


def run_training(
    config: dict[str, Any],
    run_dir: Path,
    log: Callable[[str], None],
    cancelled: threading.Event,
) -> dict[str, Any]:
    """Fine-tune F5TTS_v1_Base and export an EMA safetensors + unchanged vocab.

    Intended to run inside an isolated worker process. Required configuration:
    csv_path, wavs_dir, base_checkpoint, vocab_path. Tunables: epochs, batch_size
    (samples), learning_rate, device, grad_accumulation_steps. No remote data or
    model downloads are performed. Private _model_arch permits a small real model
    in integration tests and is deliberately not an HTTP API option.

    This uses upstream CFM, DiT, Trainer, EMA and Accelerate APIs. The local loop
    fixes upstream 1.1.21's persistent_workers=0 and tiny-run scheduler bugs and
    provides per-batch cancellation and finite-loss checks.
    """
    import numpy as np
    import soundfile as sf
    import torch
    from datasets import Dataset as ArrowDataset
    from f5_tts.model import CFM, DiT, Trainer
    from f5_tts.model.dataset import CustomDataset, collate_fn
    from safetensors.torch import save_file
    from torch.optim.lr_scheduler import LambdaLR
    from torch.utils.data import DataLoader

    _check_cancelled(cancelled)
    epochs = _positive_integer(config, "epochs", 10)
    batch_size = _positive_integer(config, "batch_size", 1)
    accumulation = _positive_integer(config, "grad_accumulation_steps", 1)
    learning_rate = float(config.get("learning_rate", 1e-5))
    if not math.isfinite(learning_rate) or not 1e-8 <= learning_rate <= 1e-2:
        raise ValueError("learning_rate must be between 1e-8 and 1e-2.")
    requested_device = config.get("device", "auto")
    if requested_device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu or cuda.")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but a usable CUDA GPU is unavailable.")
    device = "cuda" if requested_device != "cpu" and torch.cuda.is_available() else "cpu"
    precision = "bf16" if device == "cuda" and torch.cuda.is_bf16_supported() else "fp16" if device == "cuda" else "no"
    checkpoint_path = Path(config["base_checkpoint"]).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise ValueError(f"Base checkpoint is missing: {checkpoint_path}")
    vocab_path = Path(config["vocab_path"]).expanduser().resolve()
    vocab = read_vocab(vocab_path)
    run_dir = Path(run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        raise ValueError("Training checkpoint directory must be empty; use a new run directory.")
    checkpoint_dir.mkdir(exist_ok=True)
    log("Validating recordings and preparing a self-contained 24000 Hz dataset.")
    prepared = prepare_dataset(Path(config["csv_path"]), Path(config["wavs_dir"]), run_dir / "dataset", vocab_path)
    _check_cancelled(cancelled)
    for warning in prepared["warnings"]:
        log(f"Warning: {warning}")
    log(f"Prepared {prepared['sample_count']} samples / {prepared['total_duration_seconds']:.1f} seconds.")
    architecture = dict(config.get("_model_arch", F5_MODEL_ARCH))
    random.seed(666)
    np.random.seed(666)
    torch.manual_seed(666)
    if device == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 8))
    log(f"Loading F5 model on {device} with {precision} precision; strict checkpoint matching enabled.")
    model = CFM(
        transformer=DiT(**architecture, text_num_embeds=len(vocab), mel_dim=100),
        mel_spec_kwargs=dict(MEL_CONFIG),
        vocab_char_map=vocab,
    )
    _load_base_weights(model, checkpoint_path)
    _check_cancelled(cancelled)

    class LocalDataset(CustomDataset):
        # SoundFile avoids a hidden system FFmpeg/torchcodec dependency on Windows.
        def __getitem__(self, index: int) -> dict[str, Any]:
            row = self.data[index]
            audio, rate = sf.read(row["audio_path"], dtype="float32", always_2d=True)
            if rate != 24_000 or audio.shape[1] != 1 or not np.isfinite(audio).all():
                raise ValueError("Prepared audio changed after validation.")
            signal = torch.from_numpy(audio[:, 0]).unsqueeze(0)
            mel = self.mel_spectrogram(signal).squeeze(0)
            return {"mel_spec": mel, "text": row["text"]}

    class LocalTrainer(Trainer):
        def train_local(self, dataset: Any) -> dict[str, Any]:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=collate_fn,
                pin_memory=device == "cuda",
                generator=torch.Generator().manual_seed(666),
            )
            total_updates = math.ceil(len(loader) / accumulation) * epochs
            warmup = min(100, total_updates // 10)

            def schedule(step: int) -> float:
                if warmup and step < warmup:
                    return (step + 1) / warmup
                return max(0.05, (total_updates - step) / max(1, total_updates - warmup))

            self.scheduler = LambdaLR(self.optimizer, schedule)
            loader, self.scheduler = self.accelerator.prepare(loader, self.scheduler)
            update = 0
            last_loss: float | None = None
            metrics_path = run_dir / "metrics.jsonl"
            self.optimizer.zero_grad(set_to_none=True)
            with metrics_path.open("w", encoding="utf-8", buffering=1) as metrics:
                for epoch in range(epochs):
                    self.model.train()
                    epoch_loss = 0.0
                    batches = 0
                    for batch_index, batch in enumerate(loader):
                        _check_cancelled(cancelled)
                        with self.accelerator.accumulate(self.model):
                            loss, _, _ = self.model(
                                batch["mel"].permute(0, 2, 1),
                                text=batch["text"],
                                lens=batch["mel_lengths"],
                            )
                            last_loss = float(loss.detach().float().item())
                            if not math.isfinite(last_loss):
                                raise RuntimeError("Training loss became non-finite; reduce learning rate or inspect audio.")
                            # Accelerate divides by the configured accumulation
                            # count even when the final group has fewer batches.
                            # Normalize that group by its actual size so short
                            # datasets and epoch remainders keep the same scale.
                            group_start = batch_index // accumulation * accumulation
                            group_batches = min(accumulation, len(loader) - group_start)
                            self.accelerator.backward(loss * (accumulation / group_batches))
                            if self.accelerator.sync_gradients:
                                norm = self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                                if not bool(torch.isfinite(norm)):
                                    raise RuntimeError("Training gradients became non-finite.")
                            self.optimizer.step()
                            self.scheduler.step()
                            self.optimizer.zero_grad(set_to_none=True)
                        epoch_loss += last_loss
                        batches += 1
                        if self.accelerator.sync_gradients:
                            self.ema_model.update()
                            update += 1
                            item = {
                                "epoch": epoch + 1,
                                "epochs": epochs,
                                "update": update,
                                "total_updates": total_updates,
                                "loss": last_loss,
                                "learning_rate": self.scheduler.get_last_lr()[0],
                            }
                            metrics.write(json.dumps(item) + "\n")
                            log(f"Epoch {epoch + 1}/{epochs}, update {update}/{total_updates}, loss={last_loss:.6f}")
                            if update % 100 == 0:
                                self.save_checkpoint(update, last=True)
                    log(f"Epoch {epoch + 1} mean loss: {epoch_loss / max(1, batches):.6f}")
                _check_cancelled(cancelled)
                if not update:
                    raise RuntimeError("Training completed without an optimizer update.")
                self.save_checkpoint(update, last=True)
            return {"optimizer_updates": update, "final_loss": last_loss, "metrics_path": str(metrics_path)}

    trainer = LocalTrainer(
        model,
        epochs=epochs,
        learning_rate=learning_rate,
        checkpoint_path=str(checkpoint_dir),
        batch_size_per_gpu=batch_size,
        batch_size_type="sample",
        grad_accumulation_steps=accumulation,
        logger=None,
        log_samples=False,
        accelerate_kwargs={"cpu": device == "cpu", "mixed_precision": precision},
        ema_kwargs={"update_after_step": 0, "update_every": 1},
    )
    try:
        arrow = ArrowDataset.from_file(prepared["arrow_path"])
        durations = json.loads(Path(prepared["duration_path"]).read_text(encoding="utf-8"))["duration"]
        dataset = LocalDataset(arrow, durations=durations, **MEL_CONFIG)
        metrics = trainer.train_local(dataset)
        _check_cancelled(cancelled)
        # F5 inference use_ema=True expects the ema_model.* prefix in safetensors.
        export_path = run_dir / "model.safetensors"
        export_weights = {
            f"ema_model.{key}": value.detach().cpu().contiguous()
            for key, value in trainer.ema_model.ema_model.state_dict().items()
        }
        save_file(export_weights, str(export_path), metadata={"format": "pt", "model": "F5TTS_v1_Base"})
        del export_weights
        result = {
            "checkpoint_path": str(export_path),
            "vocab_path": prepared["vocab_path"],
            "training_checkpoint_path": str(checkpoint_dir / "model_last.pt"),
            "model_name": "F5TTS_v1_Base",
            "model_arch": architecture,
            "sample_count": prepared["sample_count"],
            "total_duration_seconds": prepared["total_duration_seconds"],
            "epochs": epochs,
            "batch_size": batch_size,
            "grad_accumulation_steps": accumulation,
            "device": device,
            "precision": precision,
            **metrics,
        }
        (run_dir / "training.json").write_text(json.dumps({"config": config, "result": result}, indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"Training complete. Exported EMA weights: {export_path}")
        return result
    finally:
        trainer.accelerator.end_training()
        del trainer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
