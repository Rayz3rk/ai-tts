import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_studio.datasets import prepare_dataset, read_vocab, validate_dataset


def make_audio(path: Path, *, rate: int = 24_000, channels: int = 1, duration: float = 0.5) -> None:
    samples = np.arange(int(rate * duration)) / rate
    audio = 0.1 * np.sin(2 * np.pi * 220 * samples)
    if channels > 1:
        audio = np.column_stack([audio] * channels)
    sf.write(path, audio.astype(np.float32), rate, subtype="FLOAT")


def test_existing_recording_script_format_and_resampling_report(tmp_path: Path) -> None:
    make_audio(tmp_path / "001.wav", rate=48_000, channels=2)
    metadata = tmp_path / "recordings.csv"
    metadata.write_text("id,category,style,text,record_notes\n001,daily,neutral,hello,clear voice\n", encoding="utf-8-sig")
    report = validate_dataset(metadata, tmp_path)
    assert report["valid"]
    assert report["sample_count"] == 1
    assert report["total_duration_seconds"] == pytest.approx(0.5)
    assert any("convert 48000 Hz / 2" in warning for warning in report["warnings"])
    assert report["samples"][0]["audio_path"] == str((tmp_path / "001.wav").resolve())


@pytest.mark.parametrize("header,delimiter", [("audio_path", "|"), ("audio_file", "\t")])
def test_relative_and_absolute_audio_paths(tmp_path: Path, header: str, delimiter: str) -> None:
    path = tmp_path / "one.wav"
    make_audio(path)
    metadata = tmp_path / "recordings.csv"
    for location in (path.name, str(path)):
        metadata.write_text(f"{header}{delimiter}text\n{location}{delimiter}hello\n", encoding="utf-8")
        assert validate_dataset(metadata, tmp_path)["valid"]


def test_missing_duplicate_and_empty_transcripts_are_reported(tmp_path: Path) -> None:
    make_audio(tmp_path / "001.wav")
    metadata = tmp_path / "recordings.csv"
    metadata.write_text("id,text\n001,hello\n001,again\n002,missing\n003,\n", encoding="utf-8")
    report = validate_dataset(metadata, tmp_path)
    assert not report["valid"]
    assert any("duplicate" in error for error in report["errors"])
    assert any("missing" in error for error in report["errors"])
    assert any("transcript is empty" in error for error in report["errors"])


@pytest.mark.parametrize("kind", ["nan", "silent", "short", "corrupt"])
def test_unusable_wavs_fail_without_silently_skipping(tmp_path: Path, kind: str) -> None:
    audio_path = tmp_path / "001.wav"
    if kind == "nan":
        sf.write(audio_path, np.full(12_000, np.nan, dtype=np.float32), 24_000, subtype="FLOAT")
    elif kind == "silent":
        sf.write(audio_path, np.zeros(12_000, dtype=np.float32), 24_000)
    elif kind == "short":
        make_audio(audio_path, duration=0.1)
    else:
        audio_path.write_bytes(b"invalid WAV data")
    metadata = tmp_path / "recordings.csv"
    metadata.write_text("id,text\n001,hello\n", encoding="utf-8")
    report = validate_dataset(metadata, tmp_path)
    assert not report["valid"]
    assert report["sample_count"] == 0
    assert any("invalid audio" in error for error in report["errors"])


def test_vocabulary_indices_must_be_preserved(tmp_path: Path) -> None:
    vocab = tmp_path / "vocab.txt"
    vocab.write_text(" \nh\ne\nl\no\n", encoding="utf-8")
    assert read_vocab(vocab) == {" ": 0, "h": 1, "e": 2, "l": 3, "o": 4}
    vocab.write_text("h\ne\n", encoding="utf-8")
    with pytest.raises(ValueError, match="space token"):
        read_vocab(vocab)
    vocab.write_text(" \nh\nh\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        read_vocab(vocab)


def test_opposite_stereo_channels_are_rejected_before_mono_conversion(tmp_path: Path) -> None:
    audio = 0.1 * np.sin(2 * np.pi * 220 * np.arange(12_000) / 24_000)
    sf.write(tmp_path / "001.wav", np.column_stack([audio, -audio]).astype(np.float32), 24_000, subtype="FLOAT")
    metadata = tmp_path / "recordings.csv"
    metadata.write_text("id,text\n001,hello\n", encoding="utf-8")
    report = validate_dataset(metadata, tmp_path)
    assert not report["valid"]
    assert any("cancel to silence" in error for error in report["errors"])


def test_prepare_creates_self_contained_f5_dataset(tmp_path: Path) -> None:
    pytest.importorskip("f5_tts.model.utils")
    datasets = pytest.importorskip("datasets")
    make_audio(tmp_path / "001.wav", rate=48_000, channels=2)
    metadata = tmp_path / "recordings.csv"
    metadata.write_text("id,text\n001,hello\n", encoding="utf-8")
    vocab = tmp_path / "vocab.txt"
    vocab.write_text(" \nh\ne\nl\no\n", encoding="utf-8")
    result = prepare_dataset(metadata, tmp_path, tmp_path / "prepared", vocab)
    arrow = datasets.Dataset.from_file(result["arrow_path"])
    item = arrow[0]
    assert item["text"] == list("hello")
    assert Path(item["audio_path"]).is_relative_to(tmp_path / "prepared")
    audio, rate = sf.read(item["audio_path"])
    assert rate == 24_000 and audio.ndim == 1
    assert item["duration"] == pytest.approx(len(audio) / rate)
    assert Path(result["vocab_path"]).read_bytes() == vocab.read_bytes()
    assert json.loads(Path(result["duration_path"]).read_text())["duration"] == [item["duration"]]
    with pytest.raises(ValueError, match="empty"):
        prepare_dataset(metadata, tmp_path, tmp_path / "prepared", vocab)


def test_prepare_rejects_unknown_tokens(tmp_path: Path) -> None:
    pytest.importorskip("f5_tts.model.utils")
    make_audio(tmp_path / "001.wav")
    metadata = tmp_path / "recordings.csv"
    metadata.write_text("id,text\n001,hello\n", encoding="utf-8")
    vocab = tmp_path / "vocab.txt"
    vocab.write_text(" \nh\n", encoding="utf-8")
    with pytest.raises(ValueError, match="absent from the base vocabulary"):
        prepare_dataset(metadata, tmp_path, tmp_path / "prepared", vocab)
