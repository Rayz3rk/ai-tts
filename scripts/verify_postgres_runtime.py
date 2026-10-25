"""Verify a completed live synthesis and CLI indexing against real PostgreSQL.

Run after the service has generated an audio clip:
    .venv/Scripts/python.exe scripts/verify_postgres_runtime.py --job JOB_ID
This also generates a short English CLI verification clip in work/.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import subprocess
import sys

import httpx
import numpy as np
import psycopg
from dotenv import dotenv_values
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:7861")
    args = parser.parse_args()
    with httpx.Client(base_url=args.url, trust_env=False, timeout=30) as client:
        status = client.get("/api/status").raise_for_status().json()
        assert status["storage"]["backend"] == "postgresql"
        job = client.get("/api/jobs/" + args.job).raise_for_status().json()
        assert job["status"] == "succeeded", job["status"]
        asset_id = job["result"]["audio_asset_id"]
        wave = client.get(f"/api/audio-assets/{asset_id}/content").raise_for_status().content
        samples, rate = sf.read(io.BytesIO(wave))
        assert rate == 24000 and len(samples) > 0
        assert np.isfinite(samples).all() and np.max(np.abs(samples)) > 1e-5
        workspace = status["storage"]["workspace_id"]

    cli_path = ROOT / "work" / "postgres-cli-validation.wav"
    process = subprocess.run([sys.executable, "-m", "audio_studio", "synthesize", "--preset", "en-narration",
        "--text", "The audio library now uses PostgreSQL.", "--output", str(cli_path), "--device", "cpu"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
    # Upstream speech libraries may log before the final CLI JSON object.
    result = json.loads(process.stdout[process.stdout.rfind("\n{") + 1:] if "\n{" in process.stdout else process.stdout)
    assert result["audio_asset_id"] and cli_path.is_file()
    settings = dotenv_values(ROOT / ".env", interpolate=False)
    with psycopg.connect(settings["AUDIO_STUDIO_DATABASE_URL"]) as connection:
        flags = connection.execute("SELECT rolsuper, rolcreatedb, rolcreaterole FROM pg_roles WHERE rolname=current_user").fetchone()
        assert flags == (False, False, False)
        row = connection.execute("SELECT job_id, model_id, preset_id, duration_seconds FROM audio_studio.audio_assets "
            "WHERE workspace_id=%s AND id=%s", (workspace, asset_id)).fetchone()
        assert row[:3] == (args.job, "kokoro-base", "zh-narration")
        assert row[3] == len(samples) / rate
        cli_row = connection.execute("SELECT path, text FROM audio_studio.audio_assets WHERE workspace_id=%s AND id=%s",
            (workspace, result["audio_asset_id"])).fetchone()
        assert cli_row == (str(cli_path), "The audio library now uses PostgreSQL.")
        counts = {}
        for table in ("models", "presets", "jobs", "audio_assets", "datasets", "training_runs"):
            # Identifiers are a fixed allowlist, values remain bound parameters.
            counts[table] = connection.execute(f"SELECT count(*) FROM audio_studio.{table} WHERE workspace_id=%s", (workspace,)).fetchone()[0]
        evidence = {"database": connection.info.dbname, "user": connection.info.user, "workspace_id": workspace,
            "app_role_privileges": {"superuser": False, "createdb": False, "createrole": False}, "counts": counts,
            "web_synthesis": {"job_id": args.job, "asset_id": asset_id, "seconds": len(samples) / rate,
                "sample_rate": rate, "finite_non_silent": True, "foreign_keys_verified": True},
            "cli_synthesis": {"asset_id": result["audio_asset_id"], "path": str(cli_path), "indexed_immediately": True},
            "admin_password_cleared": not settings.get("AUDIO_STUDIO_PG_ADMIN_PASSWORD")}
    destination = ROOT / "outputs" / "validation" / "postgres-runtime.json"
    destination.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(evidence, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
