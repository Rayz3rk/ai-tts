"""Real cover-page upload, conversion/mixing, playback and download acceptance."""

import hashlib
import json
import math
from pathlib import Path
import struct
import time
import urllib.request
import wave

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:7861"
OUTPUT = ROOT / "outputs" / "validation"
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def get(path: str) -> dict:
    with opener.open(BASE + path, timeout=30) as response:
        return json.load(response)


def main() -> None:
    pipeline = json.loads((OUTPUT / "svc-pipeline.json").read_text(encoding="utf-8"))
    model_id = pipeline["jobs"]["training"]["result"]["model_id"]
    source = ROOT / "work" / "svc-validation-recordings" / "kokoro-synthetic-acceptance.wav"
    backing = ROOT / "work" / "svc-backing-acceptance.wav"
    with wave.open(str(backing), "wb") as stream:
        stream.setparams((2, 2, 24000, 0, "NONE", "not compressed"))
        stream.writeframes(b"".join(struct.pack("<hh", *([round(1500 * math.sin(2 * math.pi * 110 * i / 24000))] * 2)) for i in range(round(4.7 * 24000))))
    evidence_path = OUTPUT / "cover-browser.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8")) if evidence_path.exists() else {}
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 1100})
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(BASE, wait_until="networkidle")
            page.locator("#tab-covers").click()
            page.locator(f"#cover-model option[value='{model_id}']").wait_for(state="attached")
            page.locator("#cover-model").select_option(model_id)
            if not evidence.get("job_id"):
                with page.expect_response(lambda r: "/api/covers/audio?kind=source" in r.url) as uploaded:
                    page.locator("#cover-source-file").set_input_files(source)
                assert uploaded.value.status == 200, uploaded.value.text()
                page.locator("#cover-form summary").click()
                with page.expect_response(lambda r: "/api/covers/audio?kind=accompaniment" in r.url) as uploaded_backing:
                    page.locator("#cover-accompaniment-file").set_input_files(backing)
                assert uploaded_backing.value.status == 200, uploaded_backing.value.text()
                page.wait_for_function("document.querySelector('#cover-accompaniment').value.length > 0")
                with page.expect_response(lambda r: "/api/covers/convert" in r.url) as submitted:
                    page.locator("#cover-submit").click()
                assert submitted.value.status == 202, submitted.value.text()
                evidence.update(job_id=submitted.value.json()["id"], uploads=True, model_id=model_id)
                evidence_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
            identity = evidence["job_id"]
            previous = None
            while True:
                job = get("/api/jobs/" + identity)
                if job["status"] != previous:
                    print(identity, job["status"], flush=True)
                    previous = job["status"]
                if job["status"] in {"succeeded", "failed", "cancelled"}:
                    assert job["status"] == "succeeded", job.get("error")
                    break
                time.sleep(2)
            page.wait_for_function("document.querySelectorAll('#cover-job-detail audio').length === 2", timeout=30000)
            players = page.locator("#cover-job-detail audio")
            players.first.evaluate("a => a.play()")
            page.wait_for_function("document.querySelector('#cover-job-detail audio').currentTime > 0")
            evidence["playback"] = players.first.evaluate("a => ({duration:a.duration,currentTime:a.currentTime,readyState:a.readyState})")
            players.first.evaluate("a => a.pause()")
            with page.expect_download() as pending:
                page.locator("#cover-job-detail a[download]").first.click()
            received = Path(pending.value.path()).read_bytes()
            with opener.open(BASE + job["result"]["audio_url"], timeout=30) as response:
                expected = response.read()
            assert received == expected and received[:4] == b"RIFF"
            evidence.update(download_bytes=len(received), download_sha256=hashlib.sha256(received).hexdigest(),
                two_players=True, mixed_asset_id=job["result"]["audio_asset_id"], vocal_asset_id=job["result"]["vocal_asset_id"])
            page.screenshot(path=str(OUTPUT / "cover-ready-desktop.png"), full_page=True)
            page.set_viewport_size({"width": 390, "height": 844})
            page.screenshot(path=str(OUTPUT / "cover-ready-mobile.png"), full_page=True)
            layout = page.evaluate("({viewport:innerWidth,body:document.body.scrollWidth,document:document.documentElement.scrollWidth})")
            assert layout["body"] <= layout["viewport"] and layout["document"] <= layout["viewport"], layout
            evidence.update(mobile_layout=layout, browser_errors=errors)
            assert not errors, errors
            evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
        finally:
            browser.close()
    print("PASS: real UI uploads, conversion+stereo mixing, two audio players, matching WAV download, 390px layout", flush=True)


if __name__ == "__main__":
    main()
