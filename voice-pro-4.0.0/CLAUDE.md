# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Voice-Pro is a Gradio web application for AI-powered dubbing: it downloads/ingests media, separates voices, runs speech-to-text, translates, and re-synthesizes speech (TTS / voice cloning). Runtime is Python 3.12, Torch 2.8.0+cu128, Gradio 6.20.0. Windows + NVIDIA GPU is the primary (verified) target; Linux/macOS paths exist but are unverified, and the code branches heavily on `platform.system()`.

## Running & environment

There is no system-wide Python install and **no test suite**. The app runs inside a self-contained **uv-managed** environment under `installer_files/` (uv binary + project-local Python 3.12 + venv + cache), bootstrapped by the launcher scripts. Do not assume `python`/`pip` on PATH is the app's interpreter — the interpreter is `installer_files/env/Scripts/python.exe` (`bin/python` on POSIX).

- **Windows:** `configure.bat` (one-time, optional: installs git/ffmpeg via choco; CUDA Toolkit is NOT needed — torch wheels bundle the CUDA runtime), then `start.bat`. `update.bat` re-syncs deps, `uninstall.bat` removes (`uninstall.bat silent` for unattended). If ffmpeg is not on PATH, `start.bat` downloads a portable copy into `installer_files/ffmpeg/` — so on machines without admin rights nothing needs installing at all.
- **macOS/Linux:** `configure.sh`, `start.sh`, `update.sh`, `uninstall.sh`.
- **Launch chain:** `start.bat`/`start.sh` → downloads uv into `installer_files/uv/` → `uv sync --frozen --extra gpu|cpu` (installs Python + all deps from `uv.lock`) → `python start-abus.py voice` (`one_click.py` `OneClick` only verifies/repairs) → `python start-voice.py` → `abus_app_voice.create_ui()` → Gradio server on `http://127.0.0.1:7870`.
- **GPU/CPU selection:** `GPU_CHOICE` env var (`G`/`C`) > saved `installer_files/gpu_choice.txt` > NVIDIA autodetect. The extras `gpu`/`cpu` are mutually exclusive (`[tool.uv] conflicts`).
- **Troubleshooting (from README):** most issues are fixed by deleting `installer_files/` and re-running `configure` then `start`.
- Dependencies live in `pyproject.toml` + committed `uv.lock` (universal lockfile — locks all platforms). To change deps: edit `pyproject.toml`, run `installer_files/uv/uv.exe lock`, commit both. Key version constraints: numba caps `numpy<2.5`; transformers is pinned to 5.13.0 and requires two behavior patches in the vendored `cosyvoice/llm/llm.py` (`Qwen2Encoder`: `dtype=torch.float32` on `from_pretrained`, and a full-length decode attention mask) — without them CosyVoice2/3 silently synthesize wrong content. whisperX was removed in v4.0 (its `huggingface-hub<1.0` pin blocked gradio 6). pynini/WeTextProcessing must never return — no Windows wheels; the vendored CosyVoice uses `wetext` instead.

## Architecture

### App layering (everything in `app/`)

The `app/` directory is organized in three cooperating layers — when adding a feature you typically touch all three:

1. **`tab_*.py`** — builds the Gradio UI for one tab: widgets, layout, and event wiring (`.click`/`.change` handlers). Instantiates the controller(s) below.
2. **`gradio_*.py`** — controller classes named `GradioXxx`. Hold UI state and the handler logic that the `tab_*` events call, orchestrating the core modules. `GradioGulliver` (in `gradio_gulliver.py`) is the central controller for the Dubbing Studio and wires together the whole download→ASR→translate→TTS pipeline.
3. **`abus_*.py`** — core processing modules with no Gradio dependency: ASR engines (`abus_asr_*`), TTS engines (`abus_tts_*`), translators (`abus_translate_*`), voice managers (`abus_voice_*`), plus `abus_downloader` (yt-dlp), `abus_ffmpeg`, `abus_demucs`/`abus_mdx` (source separation), `abus_subtitle`, and `abus_path` (all filesystem path helpers).

`abus_app_*.py` are the top-level UI assemblers (one per product variant); `abus_app_voice.py` is the live one and defines the tab structure. Other `abus_app_*` / `tab_*` (aicover, kara, gulliver-as-app, upscaler, vsr, rvc, demixing) are alternate/legacy variants — several are commented out of `abus_app_voice.py`, so check what's actually imported before assuming a tab is active.

### The tabs are the pipeline

`abus_app_voice.create_ui()` composes: **Dubbing Studio** (`gulliver_tab`, the all-in-one download→transcribe→translate→dub flow), **Whisper subtitles** (`subtitle_tab`), **Translation** (VOD `translate_tab` + Live `live_translate_tab`, Live is Windows-only), and **Speech Generation** (Edge/Azure-TTS, F5-TTS single & multi, CosyVoice, kokoro).

### Swappable engines

- **ASR:** `GradioGulliver.switch_case()` selects one of `faster-whisper` / `whisper` / `whisper-timestamped` by name, returning the matching `*Inference` class from `abus_asr_*.py`. Unknown names (e.g. `whisperX` persisted in an old `config-user.json5`) fall back to faster-whisper.
- **Azure vs. free:** `abus_genuine.azure_text_api_working()` (backed by `abus_config` + `.env`) decides at runtime whether translation/TTS use Azure (`AzureTranslator`/`AzureTTS`) or the free path (`DeepTranslator`/`EdgeTTS`). Copy `.env.example` → `.env` to enable Azure.

### Configuration

`src/config.py` `UserConfig` loads `app/config-user.json5` (JSON5), overlaying a large `default_user_config` dict of defaults. UI widgets read initial values via `user_config.get(key, default)` and persist changes via `user_config.set(key, value)` (which writes the file immediately). Add new persisted settings to both the JSON5 file and the defaults dict.

### Internationalization

**All user-facing UI strings must be wrapped in `i18n("...")`** (`from src.i18n.i18n import I18nAuto; i18n = I18nAuto()`). The English string is the lookup key; translations live in `src/i18n/locale/*.json` (8 languages). `src/i18n/scan_i18n.py` extracts keys from source. A new UI string with no locale entry falls back to the key itself.

### Models & assets

Model weights are **not** in the repo. On startup `start-voice.py` calls `AbusHuggingFace.hf_download_models(...)` (`app/abus_hf.py`), which reads manifests `app/abus_hf_files-*.json` and pulls files from HuggingFace `ABUS-AI/*` repos into `model/`. CosyVoice2-0.5B is ~9GB, so first run is slow. The optional Fun-CosyVoice3-0.5B model (Korean + 8 more languages) is downloaded on first use from the official `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` HF repo (see `app/abus_tts_cosyvoice.py`); the model is selectable in the CosyVoice tab and persisted as `cosy_model`.

The vendored `cosyvoice/` directory tracks upstream `FunAudioLLM/CosyVoice` main (re-vendored 2026-07, commit 074ca6d). When updating it, re-copy from upstream wholesale — there are no local patches — and keep `third_party/Matcha-TTS` at the upstream submodule pin (it is byte-identical).

### Paths / working directories (`app/abus_path.py`)

All paths are resolved relative to the current working directory (the repo root at launch): `workspace/` holds run outputs (subfolders timestamped per job), `model/` holds weights, `installer_files/gradio/` is Gradio temp. Use the `path_*` helpers rather than hardcoding paths, and `sanitize_filename` / `path_shorten` for user-derived names (Windows 260-char limit handling lives here).

`third_party/Matcha-TTS` is manually appended to `sys.path` (required by CosyVoice) — preserve those `sys.path.append` lines when refactoring entry points.

## Conventions

- Logging uses `structlog` (`logger = structlog.get_logger()`); log messages are prefixed with `[filename.py] function_name - ...`. `LOG_LEVEL` env var controls verbosity (`start.bat` sets `DEBUG`).
- Many modules use `from x import *` to flatten the `abus_*` namespace into the `gradio_*` controllers — new core functions become available app-wide, so keep names unambiguous.
- Inline comments and some log strings are in Korean; this is expected.
