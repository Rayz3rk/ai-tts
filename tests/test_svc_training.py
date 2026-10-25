"""Actual small RVC GAN updates, exports and reloads; no voice-quality claim.

The test supplies deterministic content/F0 labels instead of downloading the
pretrained HuBERT/RMVPE encoders. It tests real RVC generator/discriminator
forward/backward, all four losses, optimization and independent inference reload.
The complete pretrained-encoder chain requires separate local acceptance.
"""

import copy
import json
from pathlib import Path
import threading

import numpy as np
import pytest
import soundfile as sf

from audio_studio.svc_training import run_training, validate_voice_dataset
from audio_studio.training import TrainingCancelled


def test_voice_dataset_requires_wavs_and_rejects_silence_and_nonfinite(tmp_path: Path):
    assert not validate_voice_dataset(tmp_path / "missing")["valid"]
    assert not validate_voice_dataset(tmp_path)["valid"]
    sf.write(tmp_path / "silence.wav", np.zeros(40000, dtype=np.float32), 40000)
    invalid = np.ones(40000, dtype=np.float32) * .1
    invalid[4] = np.nan
    sf.write(tmp_path / "invalid.wav", invalid, 40000, subtype="FLOAT")
    report = validate_voice_dataset(tmp_path)
    assert report["valid"] is False
    assert report["row_count"] == 2 and report["sample_count"] == 0
    assert len(report["errors"]) == 2


def test_voice_dataset_needs_no_transcript_and_reports_stereo(tmp_path: Path):
    wave = .2 * np.sin(2 * np.pi * 220 * np.arange(48000) / 48000)
    sf.write(tmp_path / "voice.WAV", np.stack((wave, wave), axis=1), 48000)
    report = validate_voice_dataset(tmp_path)
    assert report["valid"] is True and report["csv_path"] == ""
    assert report["sample_count"] == report["row_count"] == 1
    assert report["total_duration_seconds"] == pytest.approx(1)
    assert report["samples"][0]["channels"] == 2 and report["warnings"]


@pytest.fixture
def tiny_training_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    torch = pytest.importorskip("torch")
    import audio_studio.svc as svc
    from audio_studio.vendor.rvc.models import SynthesizerTrnMs768NSFsid, MultiPeriodDiscriminatorV2

    original_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    architecture = copy.deepcopy(svc.DEFAULT_CONFIG_40K)
    architecture[1:9] = [8, 16, 16, 32, 2, 1, 3, 0]
    architecture[10:12] = [[3], [[1, 3, 5]]]
    architecture[13] = 16
    architecture[15:17] = [1, 8]
    generator = SynthesizerTrnMs768NSFsid(*architecture, is_half=False)
    generator_path, discriminator_path = tmp_path / "G.pth", tmp_path / "D.pth"
    torch.save({"model": generator.state_dict()}, generator_path)
    del generator
    discriminator = MultiPeriodDiscriminatorV2(False)
    discriminator.discriminators = torch.nn.ModuleList([discriminator.discriminators[1]])
    torch.save({"model": discriminator.state_dict()}, discriminator_path)
    del discriminator
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    t = np.arange(40000) / 40000
    wave = (.2 * np.sin(2 * np.pi * 220 * t) + .05 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(recordings / "voice.wav", wave, 40000)
    monkeypatch.setattr(svc, "load_encoder", lambda *_args: None)
    monkeypatch.setattr(svc, "load_pitch_extractor", lambda *_args: None)

    def content_labels(_encoder, audio16k, _device):
        count = max(1, (len(audio16k) - 400) // 320 + 1)
        return np.random.default_rng(2).normal(0, .05, (count, 768)).astype(np.float32)

    def pitch_labels(_extractor, _audio16k, frames, pitch_shift=0):
        return np.full(frames, 100, dtype=np.int64), np.full(frames, 220, dtype=np.float32)

    monkeypatch.setattr(svc, "extract_features", content_labels)
    monkeypatch.setattr(svc, "extract_pitch", pitch_labels)
    config = {"wavs_dir": str(recordings), "assets": {"encoder_path": str(tmp_path), "f0_path": str(tmp_path / "f0.pt"),
        "pretrained_g_path": str(generator_path), "pretrained_d_path": str(discriminator_path)},
        "epochs": 2, "batch_size": 1, "segment_frames": 8, "max_steps": 2, "learning_rate": 1e-4,
        "device": "cpu", "_model_config": architecture, "_discriminator_indices": [1]}
    try:
        yield config
    finally:
        torch.set_num_threads(original_threads)


def test_actual_rvc_adversarial_update_export_and_inference_reload(tmp_path: Path, tiny_training_config: dict):
    import torch
    from audio_studio.vendor.rvc.models import SynthesizerTrnMs768NSFsid

    result = run_training(tiny_training_config, tmp_path / "run", lambda _: None)
    assert result["steps"] == 2 and result["sample_rate"] == 40000
    metrics = [json.loads(line) for line in Path(result["metrics_path"]).read_text().splitlines()]
    assert len(metrics) == 2
    assert all(np.isfinite(value) for metric in metrics for value in metric.values())
    assert all(metric["generator_grad_norm"] > 0 and metric["discriminator_grad_norm"] > 0 for metric in metrics)
    export = torch.load(result["checkpoint_path"], map_location="cpu", weights_only=True)
    assert export["f0"] == 1 and export["version"] == "v2" and export["sr"] == "40k"
    assert not any("enc_q" in name for name in export["weight"])
    before = torch.load(tiny_training_config["assets"]["pretrained_g_path"], map_location="cpu", weights_only=True)["model"]
    assert any(not torch.equal(before[name], value) for name, value in export["weight"].items())
    saved = torch.load(result["training_checkpoint_path"], map_location="cpu", weights_only=True)
    assert saved["steps"] == 2 and saved["optimizer_g"]["state"] and saved["optimizer_d"]["state"]
    reloaded = SynthesizerTrnMs768NSFsid(*export["config"], is_half=False)
    del reloaded.enc_q
    reloaded.load_state_dict(export["weight"], strict=True)
    reloaded.eval()
    with torch.inference_mode():
        generated = reloaded.infer(torch.randn(1, 50, 768) * .05, torch.tensor([50]),
            torch.full((1, 50), 100, dtype=torch.long), torch.full((1, 50), 220.), torch.zeros(1, dtype=torch.long))[0]
    assert generated.shape[-1] == 20000
    assert torch.isfinite(generated).all() and generated.abs().max() > 0


def test_training_cancelled_and_invalid_rate_do_not_prepare_data(tmp_path: Path, tiny_training_config: dict):
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(TrainingCancelled):
        run_training(tiny_training_config, tmp_path / "cancelled", lambda _: None, cancel)
    with pytest.raises(ValueError, match="learning_rate"):
        run_training({**tiny_training_config, "learning_rate": float("nan")}, tmp_path / "invalid", lambda _: None)
    assert not (tmp_path / "invalid").exists()
