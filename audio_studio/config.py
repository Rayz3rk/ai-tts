"""Load project configuration without exporting provisioning credentials."""

import os
from pathlib import Path

from dotenv import dotenv_values

RUNTIME_KEYS = {"AUDIO_STUDIO_DATABASE_URL", "AUDIO_STUDIO_DATA", "AUDIO_STUDIO_WORKSPACE_ID", "AUDIO_STUDIO_TOKEN", "AUDIO_STUDIO_DEVICE"}


def load_local_config(path: Path | None = None) -> None:
    source = path or Path.cwd() / ".env"
    if source.is_file():
        for key, value in dotenv_values(source, interpolate=False).items():
            if key in RUNTIME_KEYS and value is not None:
                os.environ.setdefault(key, value)
