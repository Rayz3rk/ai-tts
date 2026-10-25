"""Exercise legacy methods without booting their network-heavy Gradio imports."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np


VOICEPRO = Path(__file__).resolve().parents[1] / "voice-pro-4.0.0" / "app"


def legacy_method(filename: str, class_name: str, method_name: str, namespace: dict):
    module = ast.parse((VOICEPRO / filename).read_text(encoding="utf-8"))
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    method.decorator_list = []
    namespace.update(gr=SimpleNamespace(Progress=lambda: None), logger=Mock(), np=np, os=os)
    exec(compile(ast.Module(body=[method], type_ignores=[]), filename, "exec"), namespace)
    return namespace[method_name]


def kokoro_namespace(tmp_path: Path, audio_chunks: list) -> dict:
    return {
        "KPipeline": Mock(return_value=lambda *args, **kwargs: (("text", "phonemes", audio) for audio in audio_chunks)),
        "path_dubbing_folder": lambda: str(tmp_path),
        "path_new_filename": lambda **kwargs: "raw.wav",
        "path_add_postfix": lambda path, postfix: str(tmp_path / "trimmed.wav"),
        "AbusText": SimpleNamespace(normalize_text=lambda text: text),
        "AbusAudio": SimpleNamespace(trim_silence_file=Mock()),
        "ffmpeg_to_stereo": Mock(),
        "sf": SimpleNamespace(write=Mock()),
    }


def test_kokoro_preserves_every_generated_chunk(tmp_path, monkeypatch):
    namespace = kokoro_namespace(tmp_path, [np.array([0.1, 0.2]), np.array([0.3])])
    method = legacy_method("abus_tts_kokoro.py", "KokoroTTS", "request_tts", namespace)
    monkeypatch.setattr(os, "remove", Mock())
    voice = SimpleNamespace(lang_code="a", voice_code="af_heart")
    assert method(None, "long text", str(tmp_path / "output.wav"), voice, 1.0, "wav") is True
    np.testing.assert_allclose(namespace["sf"].write.call_args.args[1], [0.1, 0.2, 0.3])


def test_kokoro_empty_generator_does_not_postprocess_missing_file(tmp_path):
    namespace = kokoro_namespace(tmp_path, [])
    method = legacy_method("abus_tts_kokoro.py", "KokoroTTS", "request_tts", namespace)
    voice = SimpleNamespace(lang_code="a", voice_code="af_heart")
    assert method(None, "text", str(tmp_path / "output.wav"), voice, 1.0, "wav") is False
    namespace["sf"].write.assert_not_called()
    namespace["AbusAudio"].trim_silence_file.assert_not_called()


def test_f5_inference_exception_reports_failure():
    namespace = {"infer_process": Mock(side_effect=RuntimeError("inference failed")), "sf": SimpleNamespace(write=Mock())}
    method = legacy_method("abus_tts_f5.py", "F5TTS", "generate_audio", namespace)
    engine = SimpleNamespace(ema_model=object(), vocoder=object())
    assert method(engine, "text", "output.wav", "reference.wav", "reference text", 1.0) is False
    namespace["sf"].write.assert_not_called()


def test_f5_request_stops_when_generation_failed(tmp_path):
    namespace = {
        "path_dubbing_folder": lambda: str(tmp_path),
        "path_new_filename": lambda **kwargs: "raw.wav",
        "AbusText": SimpleNamespace(normalize_text=lambda text: text),
        "AbusAudio": SimpleNamespace(trim_silence_file=Mock()),
    }
    method = legacy_method("abus_tts_f5.py", "F5TTS", "request_tts", namespace)
    engine = SimpleNamespace(generate_audio=Mock(return_value=False))
    assert method(engine, "text", "output.wav", "reference.wav", "reference text", 1.0, "wav") is False
    namespace["AbusAudio"].trim_silence_file.assert_not_called()
