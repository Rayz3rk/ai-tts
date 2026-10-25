"""Integration verification uses the actual F5 network and optimizer, with tiny dimensions."""

import json
import threading
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_studio.datasets import read_vocab
from audio_studio.training import MEL_CONFIG, TrainingCancelled, _load_base_weights, run_training


@pytest.fixture
def tiny_config(tmp_path: Path) -> dict:
    torch = pytest.importorskip("torch")
    f5 = pytest.importorskip("f5_tts.model")
    from safetensors.torch import save_file

    architecture = {"dim": 32, "depth": 1, "heads": 2, "dim_head": 16, "ff_mult": 2, "text_dim": 32, "conv_layers": 0}
    vocab = tmp_path / "vocab.txt"
    vocab.write_text(" \nh\ne\nl\no\n", encoding="utf-8")
    network = f5.CFM(
        transformer=f5.DiT(**architecture, text_num_embeds=5, mel_dim=100),
        mel_spec_kwargs=dict(MEL_CONFIG),
        vocab_char_map=read_vocab(vocab),
    )
    base = tmp_path / "base.safetensors"
    save_file({f"ema_model.{key}": value.contiguous() for key, value in network.state_dict().items()}, str(base))
    samples = np.arange(12_000) / 24_000
    for sample_id, frequency in (("001", 220), ("002", 330)):
        sf.write(tmp_path / f"{sample_id}.wav", (0.1 * np.sin(2 * np.pi * frequency * samples)).astype(np.float32), 24_000)
    metadata = tmp_path / "recordings.csv"
    metadata.write_text("id,text\n001,hello\n002,hello\n", encoding="utf-8")
    assert isinstance(torch.__version__, str)
    return {"csv_path": str(metadata), "wavs_dir": str(tmp_path), "base_checkpoint": str(base),
            "vocab_path": str(vocab), "epochs": 2, "batch_size": 1, "learning_rate": 1e-4,
            "grad_accumulation_steps": 2, "device": "cpu", "_model_arch": architecture}


def test_real_cpu_training_updates_weights_and_exports_loadable_ema(tmp_path: Path, tiny_config: dict) -> None:
    from f5_tts.model import CFM, DiT
    from safetensors.torch import load_file

    messages: list[str] = []
    result = run_training(tiny_config, tmp_path / "run", messages.append, threading.Event())
    assert result["optimizer_updates"] == 2
    assert np.isfinite(result["final_loss"])
    assert Path(result["training_checkpoint_path"]).is_file()
    base = load_file(tiny_config["base_checkpoint"])
    trained = load_file(result["checkpoint_path"])
    assert base.keys() == trained.keys()
    assert any(not np.array_equal(base[key].numpy(), trained[key].numpy()) for key in base)
    metrics = [json.loads(line) for line in Path(result["metrics_path"]).read_text().splitlines()]
    assert [item["update"] for item in metrics] == [1, 2]
    assert all(np.isfinite(item["loss"]) for item in metrics)
    # Exercise the same strict EMA extraction used for another fine-tuning run.
    model = CFM(transformer=DiT(**tiny_config["_model_arch"], text_num_embeds=5, mel_dim=100),
                mel_spec_kwargs=dict(MEL_CONFIG), vocab_char_map=read_vocab(Path(result["vocab_path"])))
    _load_base_weights(model, Path(result["checkpoint_path"]))
    assert any("Training complete" in message for message in messages)


def test_cancelled_training_does_not_create_a_checkpoint(tmp_path: Path, tiny_config: dict) -> None:
    event = threading.Event()
    event.set()
    with pytest.raises(TrainingCancelled):
        run_training(tiny_config, tmp_path / "cancelled", lambda _: None, event)
    assert not (tmp_path / "cancelled" / "model.safetensors").exists()


def test_invalid_training_settings_fail_before_dataset_preparation(tmp_path: Path, tiny_config: dict) -> None:
    config = {**tiny_config, "learning_rate": float("nan")}
    with pytest.raises(ValueError, match="learning_rate"):
        run_training(config, tmp_path / "invalid", lambda _: None, threading.Event())
    assert not (tmp_path / "invalid" / "dataset").exists()


@pytest.mark.parametrize("sample_count,accumulation", [(2, 3), (3, 2), (2, 2)])
def test_accumulation_remainders_preserve_mean_gradient(
    tmp_path: Path, tiny_config: dict, monkeypatch: pytest.MonkeyPatch, sample_count: int, accumulation: int
) -> None:
    """Exercise real Accelerate backward/optimizer with a known constant gradient."""
    from accelerate import Accelerator
    from f5_tts.model import CFM

    metadata = Path(tiny_config["csv_path"])
    if sample_count == 3:
        (tmp_path / "003.wav").write_bytes((tmp_path / "001.wav").read_bytes())
        with metadata.open("a", encoding="utf-8") as handle:
            handle.write("003,hello\n")
    gradients: list[float] = []
    original_clip = Accelerator.clip_grad_norm_

    def known_loss(self: CFM, *_args, **_kwargs):
        return next(self.parameters()).flatten()[0] * 0.25, None, None

    def capture_gradient(self: Accelerator, parameters, max_norm, *args, **kwargs):
        parameters = list(parameters)
        gradients.append(parameters[0].grad.flatten()[0].item())
        return original_clip(self, parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(CFM, "forward", known_loss)
    monkeypatch.setattr(Accelerator, "clip_grad_norm_", capture_gradient)
    config = {**tiny_config, "epochs": 1, "grad_accumulation_steps": accumulation}
    result = run_training(config, tmp_path / "run", lambda _: None, threading.Event())
    assert gradients == pytest.approx([0.25] * result["optimizer_updates"])
    assert result["optimizer_updates"] == (sample_count + accumulation - 1) // accumulation
