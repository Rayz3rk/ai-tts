import json
import shutil
from pathlib import Path

from audio_studio.schemas import Preset
from audio_studio.store import Store


def test_moving_data_directory_preserves_trained_model_and_reference(tmp_path: Path):
    original = tmp_path / "original"
    store = Store(original)
    run = original / "runs" / "example"
    run.mkdir(parents=True)
    for name in ("model.safetensors", "vocab.txt", "reference.wav"):
        (run / name).write_bytes(b"fixture")
    vocoder = original / "models" / "f5" / "vocos"
    vocoder.mkdir(parents=True)
    for name in ("config.yaml", "pytorch_model.bin"):
        (vocoder / name).write_bytes(b"fixture")
    model = store.register_model("迁移模型", str(run / "model.safetensors"), str(run / "vocab.txt"))
    preset = store.save_preset(Preset(name="迁移音色", engine="f5", model_id=model["id"],
        reference_audio=str(run / "reference.wav"), reference_text="测试音频"))
    moved = tmp_path / "moved"
    shutil.copytree(original, moved)
    reopened = Store(moved)
    assert Path(reopened.get("models", model["id"])["checkpoint_path"]).is_file()
    assert Path(reopened.get("presets", preset["id"])["reference_audio"]).is_file()
    assert reopened.get("models", "kokoro-base")["path"] == str(moved / "models" / "kokoro")


def test_windows_catalogue_paths_rebase_inside_container_only(tmp_path: Path):
    initial = Store(tmp_path)
    catalogue = initial.data
    catalogue["data_root"] = r"E:\project\studio_data"
    catalogue["models"][0]["path"] = r"E:\project\studio_data\models\kokoro"
    catalogue["models"][0]["checkpoint_path"] = r"E:\project\studio_data\models\kokoro\kokoro-v1_0.pth"
    catalogue["presets"][0]["reference_audio"] = r"D:\external\reference.wav"
    initial.path.write_text(json.dumps(catalogue), encoding="utf-8")
    reopened = Store(tmp_path)
    assert reopened.get("models", "kokoro-base")["checkpoint_path"] == str(tmp_path / "models" / "kokoro" / "kokoro-v1_0.pth")
    assert reopened.presets()[0]["reference_audio"] == r"D:\external\reference.wav"


def test_legacy_catalogue_without_root_can_be_moved(tmp_path: Path):
    initial = Store(tmp_path)
    catalogue = initial.data
    del catalogue["data_root"]
    catalogue["models"][0]["path"] = r"C:\old\studio_data\models\kokoro"
    catalogue["models"][1]["path"] = r"C:\old\studio_data\models\f5"
    initial.path.write_text(json.dumps(catalogue), encoding="utf-8")
    reopened = Store(tmp_path)
    assert reopened.get("models", "f5-base")["path"] == str(tmp_path / "models" / "f5")


def test_music_runtime_directory_rebases_with_data_directory(tmp_path: Path):
    original = tmp_path / "original"
    store = Store(original)
    store.music_assets()
    moved = tmp_path / "moved"
    shutil.copytree(original, moved)
    model = Store(moved).get("models", "ace-step-15")
    assert model["path"] == str(moved / "models" / "ace-step")
    assert model["runtime_path"] == str(moved / "runtimes" / "ace-step")
