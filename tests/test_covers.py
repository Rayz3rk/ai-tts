"""SVC input, registration and mixing contracts alongside the TTS workbench."""

import copy
import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from audio_studio.api import create_app
from audio_studio.audio_processing import mix_accompaniment


def wav_bytes(*, silence: bool = False, seconds: float = 1, subtype: str = "PCM_16") -> bytes:
    output = io.BytesIO()
    wave = np.zeros(round(24000 * seconds)) if silence else .2 * np.sin(2 * np.pi * 220 * np.arange(round(24000 * seconds)) / 24000)
    sf.write(output, wave, 24000, format="WAV", subtype=subtype)
    return output.getvalue()


def test_cover_status_keeps_tts_models_and_requires_real_assets(tmp_path: Path):
    with TestClient(create_app(tmp_path, database_url="")) as client:
        assert len(client.get("/api/status").json()["models"]) == 2
        status = client.get("/api/covers/status").json()
        assert status["assets"]["engine"] == "rvc" and not status["assets"]["ready"]
        assert status["voices"] == []
        assert len(client.get("/api/presets").json()) == 8
        assert client.post("/api/covers/train", json={"wavs_dir": str(tmp_path)}).status_code == 400
        assert client.post("/api/covers/convert", json={"model_id": "kokoro-base", "source_audio": "x.wav"}).status_code == 400
        assert client.post("/api/covers/convert", json={"model_id": "kokoro-base", "source_audio": "x.wav", "pitch_shift": 25}).status_code == 422
        assert client.get("/static/covers.js").status_code == 200
        assert client.get("/static/unknown.py").status_code == 404


def test_cover_upload_normalizes_wav_and_rejects_invalid_inputs(tmp_path: Path):
    with TestClient(create_app(tmp_path, database_url="")) as client:
        response = client.post("/api/covers/audio", files={"file": ("source.wav", wav_bytes(), "audio/wav")})
        assert response.status_code == 200, response.text
        stored = Path(response.json()["path"])
        assert stored.is_relative_to(tmp_path / "inputs") and sf.info(stored).subtype == "PCM_16"
        for contents in (wav_bytes(silence=True), wav_bytes(seconds=.1), b"not audio"):
            assert client.post("/api/covers/audio", files={"file": ("invalid.wav", contents, "audio/wav")}).status_code == 400
        assert len(list((tmp_path / "inputs").iterdir())) == 1
        assert client.post("/api/covers/audio?kind=unknown", files={"file": ("x.wav", wav_bytes())}).status_code == 422


def test_voice_dataset_has_no_csv_requirement(tmp_path: Path):
    (tmp_path / "voice.wav").write_bytes(wav_bytes())
    with TestClient(create_app(tmp_path / "studio", database_url="")) as client:
        result = client.post("/api/covers/datasets/validate", json={"wavs_dir": str(tmp_path)}).json()
        assert result["valid"] and result["sample_count"] == 1 and result["dataset_id"]
        assert result["csv_path"] == ""


def test_builtin_vocal_can_be_loaded_without_starting_conversion(tmp_path: Path):
    sample = tmp_path / "outputs" / "samples" / "singing-dry-demo.wav"
    sample.parent.mkdir(parents=True)
    sample.write_bytes(wav_bytes())
    with TestClient(create_app(tmp_path / "studio", database_url="")) as client:
        response = client.get("/api/covers/test-vocal")
        assert response.status_code == 200, response.text
        assert response.json()["path"] == str(sample.resolve())
        assert response.json()["duration_seconds"] == 1
        assert not response.json()["allow_output_training"]
        assert client.get("/api/jobs").json() == []


def test_mixing_resamples_keeps_stereo_tail_and_prevents_clipping(tmp_path: Path):
    vocal = tmp_path / "vocal.wav"
    backing = tmp_path / "backing.wav"
    sf.write(vocal, np.full(40000, .8), 40000)
    sf.write(backing, np.full((48000, 2), .8), 24000)
    result = mix_accompaniment(vocal, backing, tmp_path / "mixed.wav", 1, 1)
    mixed, rate = sf.read(result["path"])
    assert rate == 40000 and mixed.shape == (80000, 2)
    assert np.max(np.abs(mixed)) <= .981 and np.min(mixed[-100:]) > .1
    assert result["peak_normalized"]


def test_registration_checks_complete_network_and_speaker_id(tmp_path: Path):
    torch = pytest.importorskip("torch")
    from audio_studio.svc import DEFAULT_CONFIG_40K
    from audio_studio.vendor.rvc.models import SynthesizerTrnMs768NSFsid
    config = copy.deepcopy(DEFAULT_CONFIG_40K)
    config[1:9] = [8, 16, 16, 32, 2, 1, 3, 0]
    config[10:12] = [[3], [[1, 3, 5]]]
    config[13], config[15], config[16] = 16, 1, 8
    network = SynthesizerTrnMs768NSFsid(*config, is_half=False)
    weights = {key: value for key, value in network.state_dict().items() if not key.startswith("enc_q.")}
    checkpoint = tmp_path / "voice.pth"
    torch.save({"config": config, "weight": weights, "version": "v2", "f0": 1}, checkpoint)
    with TestClient(create_app(tmp_path / "studio", database_url="")) as client:
        result = client.post("/api/covers/models", json={"name": "Test voice", "checkpoint_path": str(checkpoint)})
        assert result.status_code == 200, result.text
        assert result.json()["speaker_count"] == 1 and result.json()["engine"] == "rvc"
        invalid = client.post("/api/covers/models", json={"name": "Invalid speaker", "checkpoint_path": str(checkpoint), "speaker_id": 1})
        assert invalid.status_code == 400
        weights.pop("dec.conv_pre.weight")
        torch.save({"config": config, "weight": weights, "version": "v2", "f0": 1}, checkpoint)
        invalid = client.post("/api/covers/models", json={"name": "Broken model", "checkpoint_path": str(checkpoint)})
        assert invalid.status_code == 400
