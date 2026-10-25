"""Validate local speech recordings and prepare portable F5 training datasets."""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

TARGET_SAMPLE_RATE = 24_000
MIN_DURATION = 0.3
MAX_DURATION = 30.0


def _read_rows(csv_path: Path, wavs_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(8192)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",|\t;")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(handle, dialect=dialect)
        names = [name.strip() for name in (reader.fieldnames or []) if name is not None]
        if len(names) != len(set(names)) or any(not name for name in names):
            return [], ["CSV contains empty or duplicate column names."]
        fields = set(names)
        audio_field = next((key for key in ("audio_path", "audio_file", "id") if key in fields), None)
        if "text" not in fields or audio_field is None:
            return [], ["CSV requires text and one of id, audio_path or audio_file columns."]
        seen_paths: set[str] = set()
        seen_ids: set[str] = set()
        for line_number, raw in enumerate(reader, start=2):
            if not any(str(value).strip() for value in raw.values() if value is not None):
                continue
            if None in raw:
                errors.append(f"Row {line_number}: extra CSV values; quote text that contains the delimiter.")
                continue
            row = {key.strip(): value for key, value in raw.items()}
            text = (row.get("text") or "").strip()
            location = (row.get(audio_field) or "").strip()
            sample_id = (row.get("id") or "").strip()
            if not text:
                errors.append(f"Row {line_number}: transcript is empty.")
            if not location:
                errors.append(f"Row {line_number}: audio location is empty.")
                continue
            if audio_field == "id":
                if Path(location).name != location or location in {".", ".."}:
                    errors.append(f"Row {line_number}: id must be a filename without a directory.")
                    continue
                audio_path = wavs_dir / (location if location.lower().endswith(".wav") else f"{location}.wav")
            else:
                audio_path = Path(location).expanduser()
                if not audio_path.is_absolute():
                    audio_path = wavs_dir / audio_path
            audio_path = audio_path.resolve()
            normalized = str(audio_path).casefold()
            if normalized in seen_paths:
                errors.append(f"Row {line_number}: duplicate audio file: {audio_path}")
                continue
            if sample_id and sample_id in seen_ids:
                errors.append(f"Row {line_number}: duplicate id: {sample_id}")
                continue
            seen_paths.add(normalized)
            if sample_id:
                seen_ids.add(sample_id)
            rows.append({"row": line_number, "id": sample_id, "audio_path": str(audio_path), "text": text})
    return rows, errors


def validate_dataset(csv_path: Path, wavs_dir: Path) -> dict[str, Any]:
    """Inspect every recording without modifying it; all findings are JSON serializable.

    CSV supports UTF-8 with/without BOM and comma, pipe, tab or semicolon delimiters.
    Relative audio_path/audio_file values and id.wav are resolved under wavs_dir.
    """
    import numpy as np
    import soundfile as sf

    csv_path = Path(csv_path).expanduser().resolve()
    wavs_dir = Path(wavs_dir).expanduser().resolve()
    report: dict[str, Any] = {
        "valid": False,
        "csv_path": str(csv_path),
        "wavs_dir": str(wavs_dir),
        "row_count": 0,
        "sample_count": 0,
        "total_duration_seconds": 0.0,
        "errors": [],
        "warnings": [],
        "samples": [],
    }
    if not csv_path.is_file():
        report["errors"].append(f"CSV does not exist: {csv_path}")
        return report
    try:
        rows, errors = _read_rows(csv_path, wavs_dir)
    except (OSError, UnicodeError, csv.Error, ValueError, RuntimeError) as exc:
        report["errors"].append(f"Cannot read CSV: {exc}")
        return report
    report["errors"].extend(errors)
    report["row_count"] = len(rows)
    if not rows:
        report["errors"].append("Dataset contains no usable rows.")
    for row in rows:
        audio_path = Path(row["audio_path"])
        prefix = f"Row {row['row']} ({audio_path.name})"
        if not row["text"]:
            continue
        if not audio_path.is_file():
            report["errors"].append(f"{prefix}: audio file is missing: {audio_path}")
            continue
        try:
            info = sf.info(audio_path)
            if info.format not in {"WAV", "WAVEX", "RF64"}:
                raise ValueError("training recordings must be WAV files")
            duration = info.frames / info.samplerate if info.samplerate > 0 else 0.0
            if not math.isfinite(duration) or not MIN_DURATION <= duration <= MAX_DURATION:
                raise ValueError(f"duration must be between {MIN_DURATION} and {MAX_DURATION} seconds; got {duration:.3f}")
            peak = 0.0
            energy = 0.0
            mono_energy = 0.0
            count = 0
            for block in sf.blocks(audio_path, blocksize=65_536, dtype="float32", always_2d=True):
                if not np.isfinite(block).all():
                    raise ValueError("recording contains NaN or infinite audio values")
                peak = max(peak, float(np.max(np.abs(block))))
                energy += float(np.sum(np.square(block, dtype=np.float64)))
                mono_energy += float(np.sum(np.square(block.mean(axis=1), dtype=np.float64)))
                count += block.size
            if count == 0 or energy <= 1e-12:
                raise ValueError("recording is empty or completely silent")
            if mono_energy <= 1e-12:
                raise ValueError("stereo channels cancel to silence when converted to mono")
            if peak >= 0.999:
                report["warnings"].append(f"{prefix}: possible clipping (peak {peak:.4f}); check the recording.")
            if info.samplerate != TARGET_SAMPLE_RATE or info.channels != 1:
                report["warnings"].append(f"{prefix}: will convert {info.samplerate} Hz / {info.channels} channels to 24000 Hz mono.")
            report["samples"].append({**row, "duration": duration, "sample_rate": info.samplerate, "channels": info.channels})
        except (OSError, RuntimeError, ValueError) as exc:
            report["errors"].append(f"{prefix}: invalid audio: {exc}")
    report["sample_count"] = len(report["samples"])
    report["total_duration_seconds"] = round(sum(item["duration"] for item in report["samples"]), 6)
    if report["sample_count"] and report["total_duration_seconds"] < 600:
        report["warnings"].append("Less than 10 minutes of audio: suitable for a pipeline check; voice quality requires more clean recordings.")
    report["valid"] = bool(report["sample_count"] and not report["errors"])
    return report


def read_vocab(vocab_path: Path) -> dict[str, int]:
    """Preserve checkpoint token indices; never derive a new fine-tuning vocabulary."""
    tokens = Path(vocab_path).read_text(encoding="utf-8").splitlines()
    if not tokens or tokens[0] != " ":
        raise ValueError("F5 vocabulary must start with one space token at index 0.")
    if any(not token for token in tokens) or len(set(tokens)) != len(tokens):
        raise ValueError("F5 vocabulary contains empty or duplicate tokens.")
    return {token: index for index, token in enumerate(tokens)}


def prepare_dataset(csv_path: Path, wavs_dir: Path, output_dir: Path, vocab_path: Path) -> dict[str, Any]:
    """Create F5 raw.arrow, duration.json, vocab.txt and copied/resampled WAVs.

    Writes only under the requested output directory, which must be empty. The
    supplied vocabulary must match the base checkpoint and is copied unchanged.
    """
    import numpy as np
    import soundfile as sf
    from datasets.arrow_writer import ArrowWriter
    from f5_tts.model.utils import convert_char_to_pinyin
    from scipy.signal import resample_poly

    report = validate_dataset(csv_path, wavs_dir)
    if not report["valid"]:
        raise ValueError("Dataset validation failed: " + "; ".join(report["errors"][:20]))
    vocab_path = Path(vocab_path).expanduser().resolve()
    vocab = read_vocab(vocab_path)
    texts = convert_char_to_pinyin([sample["text"] for sample in report["samples"]], polyphone=True)
    unknown = sorted({token for tokens in texts for token in tokens if token not in vocab})
    if unknown:
        raise ValueError(f"Transcripts contain tokens absent from the base vocabulary: {unknown[:30]}")
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Prepared dataset destination must be empty: {output_dir}")
    audio_dir = output_dir / "wavs"
    audio_dir.mkdir(parents=True, exist_ok=True)
    durations: list[float] = []
    arrow_path = output_dir / "raw.arrow"
    with ArrowWriter(path=str(arrow_path)) as writer:
        for index, (sample, tokens) in enumerate(zip(report["samples"], texts, strict=True)):
            audio, rate = sf.read(sample["audio_path"], dtype="float32", always_2d=True)
            mono = audio.mean(axis=1)
            if rate != TARGET_SAMPLE_RATE:
                divisor = math.gcd(rate, TARGET_SAMPLE_RATE)
                mono = resample_poly(mono, TARGET_SAMPLE_RATE // divisor, rate // divisor).astype(np.float32)
            # Keep valid floating-point WAV values even if a source was louder than full scale.
            peak = float(np.max(np.abs(mono)))
            if peak > 1.0:
                mono = mono / peak * 0.99
            output_path = audio_dir / f"{index + 1:06d}.wav"
            sf.write(output_path, mono, TARGET_SAMPLE_RATE, subtype="FLOAT")
            duration = len(mono) / TARGET_SAMPLE_RATE
            durations.append(duration)
            writer.write({"audio_path": str(output_path), "text": tokens, "duration": duration})
        writer.finalize()
    duration_path = output_dir / "duration.json"
    duration_path.write_text(json.dumps({"duration": durations}), encoding="utf-8")
    copied_vocab = output_dir / "vocab.txt"
    shutil.copy2(vocab_path, copied_vocab)
    result = {
        **report,
        "dataset_path": str(output_dir),
        "arrow_path": str(arrow_path),
        "duration_path": str(duration_path),
        "vocab_path": str(copied_vocab),
        "vocab_size": len(vocab),
        "target_sample_rate": TARGET_SAMPLE_RATE,
    }
    (output_dir / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result
