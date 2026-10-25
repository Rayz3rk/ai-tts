"""Provision this application's database using locally supplied admin details."""

import json
import re
import secrets
from pathlib import Path

import psycopg
from dotenv import dotenv_values, set_key
from psycopg import sql
from psycopg.conninfo import make_conninfo, conninfo_to_dict


def provision_database(env_path: Path, database_name: str = "audio_studio", role_name: str = "audio_studio_app") -> dict:
    env_path = env_path.resolve()
    if not env_path.is_file():
        raise ValueError("请先复制 .env.example 为 .env，填写 PostgreSQL 管理员用户名与密码")
    values = dotenv_values(env_path, interpolate=False)
    existing = values.get("AUDIO_STUDIO_DATABASE_URL")
    if existing:
        try:
            with psycopg.connect(existing, connect_timeout=5) as connection:
                database, user = connection.execute("SELECT current_database(), current_user").fetchone()
            parameters = conninfo_to_dict(existing)
            return {"database": database, "user": user, "host": parameters.get("host", "localhost"),
                    "port": int(parameters.get("port", "5432")), "created": False}
        except psycopg.Error:
            raise ValueError("已有应用数据库连接失败，请检查 .env 中的 AUDIO_STUDIO_DATABASE_URL") from None
    for name in (database_name, role_name):
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,47}", name):
            raise ValueError("数据库与角色名须以小写字母开头，仅含小写字母、数字、下划线，最多 48 字符")
    host = values.get("AUDIO_STUDIO_PG_ADMIN_HOST") or "127.0.0.1"
    port = int(values.get("AUDIO_STUDIO_PG_ADMIN_PORT") or "5432")
    user = values.get("AUDIO_STUDIO_PG_ADMIN_USER") or "postgres"
    password = values.get("AUDIO_STUDIO_PG_ADMIN_PASSWORD")
    if not password:
        raise ValueError("请在项目 .env 中填写 AUDIO_STUDIO_PG_ADMIN_USER 和 AUDIO_STUDIO_PG_ADMIN_PASSWORD")
    try:
        with psycopg.connect(host=host, port=port, user=user, password=password, dbname="postgres", connect_timeout=5, autocommit=True) as connection:
            # Existing objects may belong to another project. Give this app new
            # names rather than resetting credentials or replacing a database.
            if connection.execute("SELECT 1 FROM pg_database WHERE datname=%s", (database_name,)).fetchone():
                database_name += "_" + secrets.token_hex(3)
            if connection.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (role_name,)).fetchone():
                role_name += "_" + secrets.token_hex(3)
            app_password = secrets.token_urlsafe(36)
            connection.execute(sql.SQL("CREATE ROLE {} LOGIN PASSWORD {} NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION").format(
                sql.Identifier(role_name), sql.Literal(app_password)))
            connection.execute(sql.SQL("CREATE DATABASE {} OWNER {} ENCODING 'UTF8' TEMPLATE template0").format(
                sql.Identifier(database_name), sql.Identifier(role_name)))
        app_dsn = make_conninfo(host=host, port=port, dbname=database_name, user=role_name, password=app_password)
        with psycopg.connect(app_dsn, connect_timeout=5) as connection:
            connection.execute("SELECT 1")
    except psycopg.Error as exc:
        code = exc.sqlstate or "connection"
        raise ValueError(f"PostgreSQL 初始化失败（{code}），请检查管理员凭据及建库、建角色权限") from None
    set_key(env_path, "AUDIO_STUDIO_DATABASE_URL", app_dsn, quote_mode="always")
    set_key(env_path, "AUDIO_STUDIO_PG_ADMIN_PASSWORD", "", quote_mode="always")
    return {"database": database_name, "user": role_name, "host": host, "port": port, "created": True}
