"""Exercise the real local UI with a temporary catalogue, without model downloads.

Run with a Python that has Playwright and a browser installed, for example:
    python scripts/browser_smoke.py --browser msedge --screenshots outputs
The application runs with .venv's Python unless --app-python is supplied.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request

from playwright.sync_api import sync_playwright


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def free_port() -> int:
    with socket.socket() as connection:
        connection.bind(("127.0.0.1", 0))
        return connection.getsockname()[1]


def wait_for_server(process: subprocess.Popen, url: str, server_logs: list[str]) -> None:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Test server exited:\n" + "".join(server_logs))
        try:
            with opener.open(url, timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.1)
    raise TimeoutError("Test server did not become ready:\n" + "".join(server_logs))


def exercise_ui(url: str, browser_channel: str, screenshots: Path | None) -> None:
    errors: list[str] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel=browser_channel or None, headless=True)
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 1040}, device_scale_factor=1)
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
            page.goto(url, wait_until="networkidle")
            page.locator("#connection-status.ok").wait_for()
            assert page.locator("#connection-status").inner_text() == "服务已连接", ascii(page.locator("#connection-error").inner_text())
            builtin_count = page.locator("#generation-preset option").count()
            assert builtin_count >= 5

            page.locator("#tab-presets").click()
            page.locator("#new-preset").click()
            page.locator("#preset-name").fill("Browser smoke preset")
            assert page.locator("#preset-engine").input_value() == "kokoro"
            page.locator("#preset-language").select_option("b")
            assert page.locator("#preset-voice").input_value() == "bf_emma"
            page.locator("#save-preset").click()
            page.wait_for_function("count => document.querySelectorAll('#preset-list .preset-card').length === count", arg=builtin_count + 1)
            page.wait_for_function("document.querySelector('#preset-id').value.startsWith('preset-')")
            page.locator("#preset-name").fill("<img src=x onerror=alert(1)>")
            page.locator("#preset-speed").fill("0.9")
            page.locator("#save-preset").click()
            page.wait_for_function("document.querySelector('#preset-list').textContent.includes('<img src=x onerror=alert(1)>')")
            assert page.locator("#preset-list img").count() == 0
            page.locator("#use-edited-preset").click()
            assert page.locator("#generation-preset-summary h3").inner_text() == "<img src=x onerror=alert(1)>"
            assert page.locator("#generation-speed").input_value() == "0.9"
            page.locator("#tab-presets").click()
            page.locator("#delete-preset").click()
            page.wait_for_function("count => document.querySelectorAll('#preset-list .preset-card').length === count", arg=builtin_count)

            page.locator("#preset-list .preset-card").first.click()
            assert page.locator("#delete-preset").is_hidden()
            assert "副本" in page.locator("#preset-name").input_value()
            page.locator("#preset-engine").select_option("f5")
            assert page.locator("#preset-f5-fields").is_visible()

            page.locator("#tab-training").click()
            page.locator("#training-csv").fill(str(PROJECT_ROOT / "missing-smoke-dataset" / "metadata.csv"))
            page.locator("#training-wavs").fill(str(PROJECT_ROOT / "missing-smoke-dataset" / "wavs"))
            page.locator("#validate-dataset").click()
            page.locator("#dataset-report .report-bad").wait_for()
            assert "CSV does not exist" in page.locator("#dataset-report").inner_text()
            assert page.locator("#training-model option").count() >= 1

            page.locator("#tab-models").click()
            assert page.locator("#model-list .model-card").count() >= 2
            assert page.locator("#dependency-list .dependency-row").count() >= 4
            page.locator("#register-details summary").click()
            assert page.locator("#register-form").is_visible()
            page.locator("#tab-generate").click()
            page.locator("#fill-example").click()
            assert page.locator("#synthesis-text").bounding_box()["width"] > 500
            page.wait_for_function("document.querySelectorAll('.toast').length === 0", timeout=15000)
            if screenshots:
                screenshots.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(screenshots / "studio-desktop-verified.png"), full_page=True)
            page.set_viewport_size({"width": 390, "height": 844})
            if screenshots:
                page.screenshot(path=str(screenshots / "studio-mobile.png"), full_page=True)
            for name in ("generate", "presets", "training", "models"):
                page.locator("#tab-" + name).click()
                assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), name + " overflows"
            assert not errors, errors
        finally:
            browser.close()


def main() -> None:
    default_python = PROJECT_ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-python", type=Path, default=default_python)
    parser.add_argument("--browser", default="msedge" if os.name == "nt" else "chromium", help="Playwright browser channel; use an empty string for bundled Chromium")
    parser.add_argument("--screenshots", type=Path)
    args = parser.parse_args()
    if not args.app_python.is_file():
        parser.error("Application Python was not found; run setup or pass --app-python.")
    with tempfile.TemporaryDirectory(prefix="audio-studio-browser-") as directory:
        folder = Path(directory)
        env = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")
        env.pop("AUDIO_STUDIO_TOKEN", None)
        # The project's .env may contain its production database. Explicitly
        # disable that binding for this isolated filesystem/UI regression run.
        env["AUDIO_STUDIO_DATABASE_URL"] = ""
        port = free_port()
        process = subprocess.Popen(
            [str(args.app_python.resolve()), "-m", "audio_studio", "--data-dir", str(folder / "data"), "serve", "--port", str(port), "--device", "cpu"],
            cwd=PROJECT_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        server_logs: list[str] = []

        def read_logs() -> None:
            if process.stdout is not None:
                server_logs.extend(process.stdout)

        reader = threading.Thread(target=read_logs, daemon=True)
        reader.start()
        try:
            url = f"http://127.0.0.1:{port}"
            # Hardware discovery can cold-load Torch; finish it before browser navigation.
            wait_for_server(process, url + "/api/status", server_logs)
            try:
                exercise_ui(url, args.browser, args.screenshots)
            except Exception:
                print("".join(server_logs))
                raise
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            reader.join(timeout=5)
            if process.stdout is not None:
                process.stdout.close()
    print("PASS: real-API preset CRUD, voice matching, safe text, dataset errors, model panels, desktop/mobile layout; no browser errors or model downloads.")


if __name__ == "__main__":
    main()
