"""Atomic local metadata storage. Model bytes remain on the local disk."""

import json
import os
import hashlib
import shutil
import threading
from pathlib import Path, PurePosixPath, PureWindowsPath
from uuid import uuid4

from audio_studio.schemas import Preset


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class Store:
    def __init__(self, root: Path, database_url: str | None = None, workspace_id: str | None = None):
        self.root = root.resolve()
        self.lock = threading.RLock()
        self.database = None
        for folder in ("models", "jobs", "audio", "references", "runs", "inputs"):
            (self.root / folder).mkdir(parents=True, exist_ok=True)
        self.path = self.root / "catalogue.json"
        dsn = database_url if database_url is not None else os.environ.get("AUDIO_STUDIO_DATABASE_URL", "")
        if self.path.exists():
            if dsn and not (self.root / "catalogue.pre-postgresql.json").exists():
                shutil.copy2(self.path, self.root / "catalogue.pre-postgresql.json")
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            self._rebase_catalogue()
        else:
            self.data = {"data_root": str(self.root), "models": self._models(), "presets": self._presets()}
            write_json(self.path, self.data)
        if dsn:
            from audio_studio.database import Database
            identity_path = self.root / "workspace.json"
            identity = workspace_id or os.environ.get("AUDIO_STUDIO_WORKSPACE_ID")
            if not identity:
                if identity_path.exists():
                    identity = json.loads(identity_path.read_text(encoding="utf-8"))["id"]
                else:
                    identity = uuid4().hex
                    write_json(identity_path, {"id": identity})
            self.database = Database(dsn, identity)
            self.database.migrate()
            self.database.initialize_catalogue(self.data["models"], self.data["presets"])
            previous_root = self.database.get_workspace().get("root_path")
            if previous_root and previous_root != str(self.root):
                self.database.rebase_paths(previous_root, str(self.root))
            else:
                self.database.set_workspace_root(str(self.root))

    def storage_status(self) -> dict:
        return {"backend": "postgresql" if self.database else "json", "configured": self.database is not None,
                "workspace_id": self.database.workspace_id if self.database else ""}

    def svc_assets(self) -> dict:
        """Create the optional SVC catalogue entry only when the feature is used."""
        with self.lock:
            try:
                return self.get("models", "rvc-support")
            except KeyError:
                folder = self.root / "models" / "rvc"
                item = {"id": "rvc-support", "name": "RVC · 音色转换与训练组件", "engine": "rvc", "role": "support",
                    "builtin": True, "path": str(folder), "encoder_path": str(folder / "contentvec"),
                    "f0_path": str(folder / "rmvpe.pt"), "pretrained_g_path": str(folder / "f0G40k.pth"),
                    "pretrained_d_path": str(folder / "f0D40k.pth")}
                return self.add_model(item)

    def singing_assets(self) -> dict:
        with self.lock:
            try:
                return self.get("models", "diffsinger-qixuan")
            except KeyError:
                from audio_studio.singing import model_metadata
                item = model_metadata(self.root / "models" / "diffsinger")
                item.update(id="diffsinger-qixuan", name="绮萱 · DiffSinger 歌声合成", builtin=True)
                return self.add_model(item)

    def music_assets(self) -> dict:
        with self.lock:
            try:
                return self.get("models", "ace-step-15")
            except KeyError:
                from audio_studio.music_assets import model_metadata
                return self.add_model(model_metadata(self.root))

    def add_model(self, item: dict) -> dict:
        with self.lock:
            if self.database:
                return self.database.upsert_model(item)
            self.data["models"] = [x for x in self.data["models"] if x["id"] != item["id"]] + [item]
            write_json(self.path, self.data)
            return dict(item)

    def register_voice_model(self, name: str, checkpoint_path: str, index_path: str = "", speaker_id: int = 0) -> dict:
        from audio_studio.svc import inspect_checkpoint
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        details = inspect_checkpoint(checkpoint)
        if not 0 <= speaker_id < details["speaker_count"]:
            raise ValueError("音色编号超出该 RVC 模型的范围")
        index = Path(index_path).expanduser().resolve() if index_path else None
        if index and (not index.is_file() or index.suffix.lower() != ".index"):
            raise ValueError("检索索引须为存在的 .index 文件；没有索引可留空")
        assets = self.svc_assets()
        item = {**details, "id": "rvc-" + uuid4().hex[:12], "name": name, "engine": "rvc", "role": "voice",
            "builtin": False, "path": str(checkpoint.parent), "checkpoint_path": str(checkpoint),
            "index_path": str(index) if index else "", "speaker_id": speaker_id,
            "encoder_path": assets["encoder_path"], "f0_path": assets["f0_path"]}
        return self.add_model(item)

    def _rebase_catalogue(self) -> None:
        """Keep assets inside the data directory usable after a move or mount.

        External model/recording paths stay explicit: users must mount or register
        those separately. Pure paths also recognize a Windows catalogue on Linux.
        """
        old_root = self.data.get("data_root")
        if not old_root:
            builtin = next((m for m in self.data["models"] if m.get("builtin") and m.get("path")), None)
            if builtin:
                source = builtin["path"]
                path_type = PureWindowsPath if PureWindowsPath(source).drive else PurePosixPath
                old_root = str(path_type(source).parent.parent)
        if old_root:
            path_type = PureWindowsPath if PureWindowsPath(old_root).drive else PurePosixPath
            previous = path_type(old_root)
            for item in self.data["models"] + self.data["presets"]:
                for key in ("path", "checkpoint_path", "config_path", "vocab_path", "vocoder_path", "reference_audio",
                            "index_path", "encoder_path", "f0_path", "pretrained_g_path", "pretrained_d_path", "runtime_path"):
                    value = item.get(key)
                    if not value:
                        continue
                    try:
                        relative = path_type(value).relative_to(previous)
                    except ValueError:
                        continue
                    item[key] = str(self.root.joinpath(*relative.parts))
        self.data["data_root"] = str(self.root)
        write_json(self.path, self.data)

    def _models(self) -> list[dict]:
        return [{"id": f"{engine}-base", "name": name, "engine": engine,
                 "path": str(self.root / "models" / engine), "builtin": True}
                for engine, name in (("kokoro", "Kokoro 82M · 中英文"), ("f5", "F5-TTS v1 · 参考音色 / 微调"))]

    @staticmethod
    def _presets() -> list[dict]:
        specs = [("zh-narration", "清晰旁白", "narration", "z", "zf_xiaobei", 1.0),
                 ("zh-conversation", "日常对话", "conversation", "z", "zf_xiaoyi", 1.05),
                 ("zh-news", "新闻播报", "news", "z", "zm_yunxi", 1.1),
                 ("zh-audiobook", "有声书", "audiobook", "z", "zm_yunjian", .9),
                 ("zh-assistant", "语音助手", "assistant", "z", "zf_xiaoxiao", 1.0),
                 ("en-narration", "English narration", "narration", "a", "af_heart", 1.0),
                 ("en-dialogue", "English dialogue", "conversation", "a", "am_adam", 1.0),
                 ("en-british", "British audiobook", "audiobook", "b", "bf_emma", .95)]
        return [dict(Preset(id=i, name=n, category=c, language=l, voice=v, speed=s).model_dump(), builtin=True)
                for i, n, c, l, v, s in specs]

    def models(self) -> list[dict]:
        if self.database:
            return self.database.list_models()
        with self.lock:
            return [dict(x) for x in self.data["models"]]

    def presets(self) -> list[dict]:
        if self.database:
            return self.database.list_presets()
        with self.lock:
            return [dict(x) for x in self.data["presets"]]

    def get(self, kind: str, item_id: str) -> dict:
        if self.database:
            return self.database.get(kind, item_id)
        with self.lock:
            for item in self.data[kind]:
                if item["id"] == item_id:
                    return dict(item)
        raise KeyError(f"找不到 {kind}: {item_id}")

    def update_model(self, model_id: str, metadata: dict) -> dict:
        with self.lock:
            model = self.get("models", model_id)
            model.update(metadata)
            if self.database:
                return self.database.upsert_model(model)
            self.data["models"] = [model if x["id"] == model_id else x for x in self.data["models"]]
            write_json(self.path, self.data)
            return model

    def register_model(self, name: str, checkpoint_path: str, vocab_path: str, vocoder_path: str = "") -> dict:
        checkpoint, vocab = Path(checkpoint_path).resolve(), Path(vocab_path).resolve()
        if not checkpoint.is_file() or checkpoint.suffix not in {".pt", ".safetensors"}:
            raise ValueError("模型检查点必须是本地 .pt 或 .safetensors 文件")
        if not vocab.is_file() or not vocab.stat().st_size:
            raise ValueError("词表不存在或为空")
        base = self.get("models", "f5-base")
        vocoder = Path(vocoder_path or base.get("vocoder_path", str(Path(base["path"]) / "vocos"))).resolve()
        if not all((vocoder / name).is_file() for name in ("config.yaml", "pytorch_model.bin")):
            raise ValueError("缺少本地 Vocos 声码器，请先下载 F5 基础模型或指定 vocoder_path")
        model = {"id": "f5-" + uuid4().hex[:12], "name": name, "engine": "f5", "path": str(checkpoint.parent),
                 "checkpoint_path": str(checkpoint), "vocab_path": str(vocab), "vocoder_path": str(vocoder), "builtin": False}
        with self.lock:
            if self.database:
                return self.database.upsert_model(model)
            self.data["models"].append(model)
            write_json(self.path, self.data)
        return model

    def save_preset(self, preset: Preset) -> dict:
        model = self.get("models", preset.model_id)
        if model["engine"] != preset.engine:
            raise ValueError("预设引擎与模型不匹配")
        if preset.engine == "f5" and not Path(preset.reference_audio).is_file():
            raise ValueError("参考音频不存在")
        item = preset.model_dump()
        item["id"] = item["id"] or "preset-" + uuid4().hex[:12]
        with self.lock:
            existing = next((x for x in self.presets() if x["id"] == item["id"]), None)
            if existing and existing.get("builtin"):
                raise ValueError("内置预设不可覆盖，请另存为自定义预设")
            item["builtin"] = False
            if self.database:
                return self.database.save_preset(item)
            self.data["presets"] = [x for x in self.data["presets"] if x["id"] != item["id"]] + [item]
            write_json(self.path, self.data)
        return item

    def delete_preset(self, item_id: str) -> None:
        with self.lock:
            item = self.get("presets", item_id)
            if item.get("builtin"):
                raise ValueError("内置预设不可删除")
            if self.database:
                self.database.delete_preset(item_id)
                return
            self.data["presets"] = [x for x in self.data["presets"] if x["id"] != item_id]
            write_json(self.path, self.data)

    def record_dataset(self, report: dict) -> str:
        key = hashlib.sha256((report["csv_path"] + "\0" + report["wavs_dir"]).encode("utf-8")).hexdigest()[:32]
        if self.database:
            self.database.save_dataset(dict(report, id=key, name=Path(report["csv_path"]).stem or Path(report["wavs_dir"]).name))
        return key

    def record_audio(self, path: Path, metadata: dict | None = None) -> dict | None:
        if not self.database:
            return None
        import soundfile as sf
        path = path.resolve()
        info = sf.info(path)
        item = dict(metadata or {})
        item.update(path=str(path), filename=path.name, duration_seconds=info.duration, sample_rate=info.samplerate)
        return self.database.register_audio(item)
