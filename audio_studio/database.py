"""Synchronous PostgreSQL metadata repository with workspace isolation.

Each operation opens its own connection and transaction. A Database instance can
therefore be shared by API and worker threads without sharing connection state.
Call ``migrate()`` once before using a newly configured database. The connection
URL is intentionally never included in public errors or representations.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Literal
from uuid import uuid4

try:
    import psycopg
    from psycopg import sql
    from psycopg.rows import dict_row
    from psycopg.types.json import Jsonb
except ImportError:  # The filesystem backend remains usable without this extra.
    psycopg = None  # type: ignore[assignment]


class DatabaseError(RuntimeError):
    """A credential-safe database failure suitable for displaying to the user."""


_INTERNAL = {"workspace_id", "metadata", "position", "snapshot", "updated_at"}
_JOB_STATUSES = {"queued", "running", "succeeded", "failed", "cancelled"}
_MIGRATION_LOCK = 7654819205184
_PATH_FIELDS = {"path", "output_path", "audio_path", "csv_path", "wavs_dir", "run_dir", "checkpoint_path",
                "config_path", "vocab_path", "vocoder_path", "reference_audio", "base_checkpoint", "dataset_path", "metrics_path",
                "index_path", "encoder_path", "f0_path", "pretrained_g_path", "pretrained_d_path", "source_audio",
                "accompaniment_audio", "vocal_path", "mixed_path", "source_path", "feature_path", "training_checkpoint_path", "runtime_path"}


def _json_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat() if value.tzinfo is not None else value.replace(tzinfo=timezone.utc).isoformat()
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _public_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct the existing JSON interfaces, with columns authoritative."""
    result = dict(row.get("metadata") or {})
    result.update(row.get("snapshot") or {})
    result.update({key: value for key, value in row.items() if key not in _INTERNAL})
    if "position" in row:
        # Catalogue dictionaries are also used as StrictModel request payloads.
        result.pop("created_at", None)
    return _json_value(result)


def _metadata(item: Mapping[str, Any], columns: Mapping[str, Any]) -> dict[str, Any]:
    return _json_value({key: value for key, value in item.items() if key not in columns and key not in _INTERNAL})


def _rebase_paths(value: Any, previous_root: str, root_path: str) -> Any:
    """Rebase only recognized path fields; external paths and transcripts stay."""
    if isinstance(value, dict):
        previous = PureWindowsPath(previous_root) if PureWindowsPath(previous_root).drive else PurePosixPath(previous_root)
        current = PureWindowsPath(root_path) if PureWindowsPath(root_path).drive else PurePosixPath(root_path)
        result = {}
        for key, item in value.items():
            if key in _PATH_FIELDS and isinstance(item, str) and item:
                source = PureWindowsPath(item) if isinstance(previous, PureWindowsPath) else PurePosixPath(item)
                try:
                    relative = source.relative_to(previous)
                except ValueError:
                    result[key] = item
                else:
                    result[key] = str(current.joinpath(*relative.parts))
            else:
                result[key] = _rebase_paths(item, previous_root, root_path)
        return result
    if isinstance(value, list):
        return [_rebase_paths(item, previous_root, root_path) for item in value]
    return value


def _required(item: Mapping[str, Any], key: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} 必须为非空字符串")
    return value


def _created_at(item: Mapping[str, Any]) -> str | datetime:
    value = item.get("created_at")
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError("created_at 必须为有效的 ISO 时间") from None
    if not isinstance(value, datetime):
        raise ValueError("created_at 必须为有效的 ISO 时间")
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _status(item: Mapping[str, Any], default: str = "queued") -> str:
    value = item.get("status", default)
    if value not in _JOB_STATUSES:
        raise ValueError("任务状态无效")
    return value


def _audio_values(item: Mapping[str, Any]) -> dict[str, Any]:
    path = _required(item, "path")
    filename = item.get("filename") or path.replace("\\", "/").rsplit("/", 1)[-1]
    tags = item.get("tags", [])
    if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
        raise ValueError("tags 必须为字符串列表")
    if not isinstance(item.get("favorite", False), bool):
        raise ValueError("favorite 必须为布尔值")
    duration = item.get("duration_seconds")
    if duration is not None and (isinstance(duration, bool) or not isinstance(duration, (int, float))
                                 or not math.isfinite(duration) or duration < 0):
        raise ValueError("音频时长必须为非负有限数值")
    sample_rate = item.get("sample_rate")
    if sample_rate is not None and (isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0):
        raise ValueError("采样率必须为正整数")
    return {"id": item.get("id") or uuid4().hex, "path": path, "filename": filename,
            "name": item.get("name") or filename, "text": item.get("text", ""),
            "model_id": item.get("model_id") or None, "preset_id": item.get("preset_id") or None,
            "job_id": item.get("job_id") or None, "duration_seconds": duration, "sample_rate": sample_rate,
            "tags": tags, "favorite": item.get("favorite", False), "created_at": _created_at(item)}


class Database:
    def __init__(self, dsn: str, workspace_id: str = "default"):
        if not isinstance(dsn, str) or not dsn.strip():
            raise ValueError("请设置 AUDIO_STUDIO_DATABASE_URL")
        if not isinstance(workspace_id, str) or not workspace_id.strip() or len(workspace_id) > 200:
            raise ValueError("workspace_id 必须为 1–200 字符的标识符")
        self._dsn = dsn
        self.workspace_id = workspace_id

    @contextmanager
    def _connection(self) -> Iterator[Any]:
        if psycopg is None:
            raise DatabaseError("未安装 PostgreSQL 驱动，请安装项目依赖 psycopg[binary]")
        try:
            with psycopg.connect(self._dsn, connect_timeout=5, row_factory=dict_row) as connection:
                yield connection
        except psycopg.Error:
            # Driver errors may include a connection string or server-provided
            # details. Do not retain them as a chained public exception.
            raise DatabaseError("PostgreSQL 操作失败，请检查数据库连接、权限与数据配置") from None

    def migrate(self) -> None:
        """Apply packaged migrations atomically, including concurrent startups."""
        migrations = sorted((path for path in files("audio_studio").joinpath("migrations").iterdir()
                             if path.name.endswith(".sql")), key=lambda path: path.name)
        with self._connection() as connection:
            connection.execute("SELECT pg_advisory_xact_lock(%s)", (_MIGRATION_LOCK,))
            connection.execute("CREATE SCHEMA IF NOT EXISTS audio_studio")
            connection.execute("""CREATE TABLE IF NOT EXISTS audio_studio.schema_migrations (
                version integer PRIMARY KEY, name text NOT NULL, checksum text NOT NULL,
                applied_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
            applied = {row["version"]: row for row in connection.execute(
                "SELECT version, name, checksum FROM audio_studio.schema_migrations").fetchall()}
            for migration in migrations:
                version = int(migration.name.split("_", 1)[0])
                source = migration.read_text(encoding="utf-8")
                checksum = hashlib.sha256(source.encode("utf-8")).hexdigest()
                if version in applied:
                    if applied[version]["checksum"] != checksum:
                        raise DatabaseError("数据库迁移文件与已应用版本不一致，请恢复原始迁移文件")
                    continue
                connection.execute(source)
                connection.execute("INSERT INTO audio_studio.schema_migrations(version, name, checksum) VALUES (%s, %s, %s)",
                                   (version, migration.name, checksum))
            connection.execute("INSERT INTO audio_studio.workspaces(id) VALUES (%s) ON CONFLICT (id) DO NOTHING",
                               (self.workspace_id,))

    def get_workspace(self) -> dict:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM audio_studio.workspaces WHERE id = %s", (self.workspace_id,)).fetchone()
        if row is None:
            raise KeyError("找不到工作区，请先执行数据库迁移")
        return _public_row(row)

    def set_workspace_root(self, root_path: str) -> None:
        if not isinstance(root_path, str) or not root_path.strip():
            raise ValueError("root_path 必须为非空路径")
        with self._connection() as connection:
            connection.execute("UPDATE audio_studio.workspaces SET metadata = metadata || %s WHERE id = %s",
                               (Jsonb({"root_path": root_path}), self.workspace_id))

    def rebase_paths(self, previous_root: str, root_path: str) -> None:
        """Move internal metadata paths and the workspace binding atomically.

        This also covers persisted job snapshots and training reports so opening
        an older result after moving a data directory still resolves local files.
        """
        if not previous_root or not root_path:
            raise ValueError("数据目录重定位需要有效的原始路径与目标路径")
        with self._connection() as connection:
            connection.execute("SELECT id FROM audio_studio.workspaces WHERE id = %s FOR UPDATE", (self.workspace_id,))
            for table in ("models", "presets", "jobs", "audio_assets", "datasets", "training_runs"):
                rows = connection.execute(sql.SQL("SELECT * FROM {} WHERE workspace_id = %s FOR UPDATE").format(
                    sql.Identifier("audio_studio", table)), (self.workspace_id,)).fetchall()
                for row in rows:
                    rebased = _rebase_paths(row, previous_root, root_path)
                    changed = {key: value for key, value in rebased.items() if value != row[key]}
                    if not changed:
                        continue
                    assignments = sql.SQL(", ").join(sql.SQL("{} = %s").format(sql.Identifier(key)) for key in changed)
                    parameters = [Jsonb(value) if key in {"metadata", "request", "result", "snapshot", "config"} else value
                                  for key, value in changed.items()]
                    connection.execute(sql.SQL("UPDATE {} SET {}, updated_at = CURRENT_TIMESTAMP WHERE workspace_id = %s AND id = %s").format(
                        sql.Identifier("audio_studio", table), assignments), (*parameters, self.workspace_id, row["id"]))
            connection.execute("UPDATE audio_studio.workspaces SET metadata = metadata || %s WHERE id = %s",
                               (Jsonb({"root_path": root_path}), self.workspace_id))

    def _upsert(self, connection: Any, table: str, values: dict[str, Any], metadata: dict[str, Any],
                *, conflict: tuple[str, ...] = ("workspace_id", "id"),
                preserve_missing: set[str] | None = None, protect_builtin: bool = False) -> dict[str, Any]:
        columns = ["workspace_id", *values, "metadata"]
        parameters = [self.workspace_id, *values.values(), Jsonb(metadata)]
        assignments = []
        for key in values:
            if key in {"id", "created_at"}:
                continue
            if preserve_missing and key in preserve_missing:
                continue
            assignments.append(sql.SQL("{} = EXCLUDED.{}").format(sql.Identifier(key), sql.Identifier(key)))
        assignments.extend([
            sql.SQL("metadata = {}.metadata || EXCLUDED.metadata").format(sql.Identifier(table)),
            sql.SQL("updated_at = CURRENT_TIMESTAMP"),
        ])
        query = sql.SQL("INSERT INTO {} ({}) VALUES ({}) ON CONFLICT ({}) DO UPDATE SET {}{} RETURNING *").format(
            sql.Identifier("audio_studio", table), sql.SQL(", ").join(map(sql.Identifier, columns)),
            sql.SQL(", ").join(sql.Placeholder() for _ in columns),
            sql.SQL(", ").join(map(sql.Identifier, conflict)), sql.SQL(", ").join(assignments),
            sql.SQL(" WHERE NOT {}.builtin").format(sql.Identifier(table)) if protect_builtin else sql.SQL(""))
        row = connection.execute(query, parameters).fetchone()
        if row is None:
            raise ValueError("内置预设不可覆盖，请另存为自定义预设")
        return _public_row(row)

    @staticmethod
    def _model_values(item: dict[str, Any]) -> dict[str, Any]:
        engine = _required(item, "engine")
        if engine not in {"kokoro", "f5", "rvc", "diffsinger", "ace_step"}:
            raise ValueError("模型引擎无效")
        return {"id": _required(item, "id"), "name": _required(item, "name"), "engine": engine,
                "path": item.get("path", ""), "builtin": bool(item.get("builtin", False)), "created_at": _created_at(item)}

    @staticmethod
    def _preset_values(item: dict[str, Any]) -> dict[str, Any]:
        return {"id": _required(item, "id"), "name": _required(item, "name"),
                "model_id": _required(item, "model_id"), "engine": _required(item, "engine"),
                "category": item.get("category", "custom"), "language": item.get("language", "z"),
                "voice": item.get("voice", ""), "speed": item.get("speed", 1.0),
                "builtin": bool(item.get("builtin", False)), "created_at": _created_at(item)}

    def initialize_catalogue(self, models: list[dict], presets: list[dict]) -> None:
        """Import a legacy catalogue only into an entirely empty workspace."""
        with self._connection() as connection:
            connection.execute("SELECT id FROM audio_studio.workspaces WHERE id = %s FOR UPDATE", (self.workspace_id,))
            populated = connection.execute("""SELECT EXISTS(
                SELECT 1 FROM audio_studio.models WHERE workspace_id = %s
                UNION ALL SELECT 1 FROM audio_studio.presets WHERE workspace_id = %s) AS populated""",
                (self.workspace_id, self.workspace_id)).fetchone()["populated"]
            if populated:
                return
            for item in models:
                values = self._model_values(item)
                self._upsert(connection, "models", values, _metadata(item, values))
            for item in presets:
                values = self._preset_values(item)
                self._upsert(connection, "presets", values, _metadata(item, values))

    def _list(self, kind: Literal["models", "presets"]) -> list[dict]:
        with self._connection() as connection:
            rows = connection.execute(sql.SQL("SELECT * FROM {} WHERE workspace_id = %s ORDER BY position").format(
                sql.Identifier("audio_studio", kind)), (self.workspace_id,)).fetchall()
        return [_public_row(row) for row in rows]

    def list_models(self) -> list[dict]:
        return self._list("models")

    def list_presets(self) -> list[dict]:
        return self._list("presets")

    def get(self, kind: Literal["models", "presets"], item_id: str) -> dict:
        if kind not in {"models", "presets"}:
            raise ValueError("元数据类型无效")
        with self._connection() as connection:
            row = connection.execute(sql.SQL("SELECT * FROM {} WHERE workspace_id = %s AND id = %s").format(
                sql.Identifier("audio_studio", kind)), (self.workspace_id, item_id)).fetchone()
        if row is None:
            raise KeyError(f"找不到 {kind}: {item_id}")
        return _public_row(row)

    def upsert_model(self, item: dict) -> dict:
        values = self._model_values(item)
        with self._connection() as connection:
            return self._upsert(connection, "models", values, _metadata(item, values))

    def save_preset(self, item: dict) -> dict:
        item = dict(item, id=item.get("id") or "preset-" + uuid4().hex[:12], builtin=False)
        values = self._preset_values(item)
        with self._connection() as connection:
            return self._upsert(connection, "presets", values, _metadata(item, values), protect_builtin=True)

    def delete_preset(self, item_id: str) -> None:
        with self._connection() as connection:
            item = connection.execute("SELECT builtin FROM audio_studio.presets WHERE workspace_id = %s AND id = %s FOR UPDATE",
                                      (self.workspace_id, item_id)).fetchone()
            if item is None:
                raise KeyError(f"找不到 presets: {item_id}")
            if item["builtin"]:
                raise ValueError("内置预设不可删除")
            connection.execute("DELETE FROM audio_studio.presets WHERE workspace_id = %s AND id = %s", (self.workspace_id, item_id))

    def load_jobs(self) -> list[dict]:
        with self._connection() as connection:
            rows = connection.execute("SELECT * FROM audio_studio.jobs WHERE workspace_id = %s ORDER BY created_at, id",
                                      (self.workspace_id,)).fetchall()
        return [_public_row(row) for row in rows]

    def save_job(self, job: dict, payload: dict | None = None) -> None:
        request = job.get("request")
        if payload is not None:
            request = payload if "kind" in payload and "payload" in payload else {"kind": job["kind"], "payload": payload}
        values = {"id": _required(job, "id"), "kind": _required(job, "kind"), "status": _status(job),
                  "name": job.get("name"), "model_id": job.get("model_id") or None, "error": job.get("error"),
                  "result": Jsonb(_json_value(job.get("result"))), "snapshot": Jsonb(_json_value(job)),
                  "created_at": _created_at(job)}
        if request is not None:
            values["request"] = Jsonb(_json_value(request))
        with self._connection() as connection:
            self._upsert(connection, "jobs", values, _metadata(job, values))

    def register_audio(self, metadata: dict) -> dict:
        """Register a file once; repeated scans preserve user-edited fields."""
        values = _audio_values(metadata)
        preserve_missing = {key for key in values if key not in metadata and key not in {"path", "filename"}}
        with self._connection() as connection:
            return self._upsert(connection, "audio_assets", values, _metadata(metadata, values),
                                conflict=("workspace_id", "path"), preserve_missing=preserve_missing)

    def list_audio(self, query: str = "", limit: int = 50, offset: int = 0) -> dict:
        if not isinstance(query, str) or not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 200:
            raise ValueError("音频查询参数无效，limit 必须为 1–200")
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise ValueError("offset 必须为非负整数")
        predicate = """workspace_id = %s AND strpos(lower(concat_ws(' ', name, filename, text,
            array_to_string(tags, ' '))), lower(%s)) > 0"""
        parameters = (self.workspace_id, query.strip())
        with self._connection() as connection:
            connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
            total = connection.execute("SELECT count(*) AS total FROM audio_studio.audio_assets WHERE " + predicate,
                                       parameters).fetchone()["total"]
            rows = connection.execute("SELECT * FROM audio_studio.audio_assets WHERE " + predicate +
                                      " ORDER BY created_at DESC, id DESC LIMIT %s OFFSET %s", (*parameters, limit, offset)).fetchall()
        return {"items": [_public_row(row) for row in rows], "total": total, "limit": limit, "offset": offset}

    def get_audio(self, item_id: str) -> dict:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM audio_studio.audio_assets WHERE workspace_id = %s AND id = %s",
                                     (self.workspace_id, item_id)).fetchone()
        if row is None:
            raise KeyError(f"找不到 audio: {item_id}")
        return _public_row(row)

    def update_audio(self, item_id: str, updates: dict) -> dict:
        if set(updates) - {"name", "text", "tags", "favorite"}:
            raise ValueError("仅可修改音频名称、文本、标签与收藏状态")
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM audio_studio.audio_assets WHERE workspace_id = %s AND id = %s FOR UPDATE",
                                     (self.workspace_id, item_id)).fetchone()
            if row is None:
                raise KeyError(f"找不到 audio: {item_id}")
            item = dict(_public_row(row), **updates)
            values = _audio_values(item)
            return self._upsert(connection, "audio_assets", values, _metadata(item, values))

    def save_dataset(self, item: dict) -> dict:
        csv_path = item.get("csv_path", "") if item.get("task") == "svc" else _required(item, "csv_path")
        if not isinstance(csv_path, str):
            raise ValueError("csv_path 必须为字符串")
        values = {"id": _required(item, "id"), "name": item.get("name", ""),
                  "csv_path": csv_path, "wavs_dir": _required(item, "wavs_dir"),
                  "valid": bool(item.get("valid", False)), "row_count": item.get("row_count", 0),
                  "sample_count": item.get("sample_count", 0), "total_duration_seconds": item.get("total_duration_seconds", 0),
                  "created_at": _created_at(item)}
        with self._connection() as connection:
            return self._upsert(connection, "datasets", values, _metadata(item, values))

    def save_training_run(self, item: dict) -> dict:
        values = {"id": _required(item, "id"), "name": item.get("name", ""),
                  "dataset_id": item.get("dataset_id") or None, "model_id": item.get("model_id") or None,
                  "job_id": item.get("job_id") or None, "status": _status(item), "run_dir": item.get("run_dir", ""),
                  "config": Jsonb(_json_value(item.get("config", {}))), "result": Jsonb(_json_value(item.get("result"))),
                  "created_at": _created_at(item)}
        with self._connection() as connection:
            return self._upsert(connection, "training_runs", values, _metadata(item, values))
