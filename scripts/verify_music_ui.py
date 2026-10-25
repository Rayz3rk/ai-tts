"""Real GPU original-song creation through the browser, with durable evidence."""

import hashlib
import json
from pathlib import Path
import time

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "validation"


def main():
    evidence_path = OUT / "music-browser.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8")) if evidence_path.exists() else {}
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(base_url="http://127.0.0.1:7861", viewport={"width": 1440, "height": 1100})
        page.set_default_timeout(90000)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto("/", wait_until="networkidle")
        page.locator("#tab-music").click()
        page.wait_for_function("!document.querySelector('#music-submit').disabled")
        if not evidence.get("job_id"):
            page.locator("#music-title").fill("自动创作验收 · 晨光里的小路")
            page.locator("#music-bpm").fill("100")
            with page.expect_response(lambda r: "/api/music/generate" in r.url) as submitted:
                page.locator("#music-submit").click()
            assert submitted.value.status == 202, submitted.value.text()
            evidence["job_id"] = submitted.value.json()["id"]
            evidence_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        job_id = evidence["job_id"]
        previous = None
        deadline = time.monotonic() + 1200
        while time.monotonic() < deadline:
            job = page.request.get("/api/jobs/" + job_id).json()
            latest = job["status"], (job.get("logs") or [""])[-1]
            if latest != previous:
                print(latest, flush=True)
                previous = latest
            if job["status"] in {"succeeded", "failed", "cancelled"}:
                assert job["status"] == "succeeded", job.get("error")
                break
            time.sleep(2)
        else:
            raise TimeoutError("Music validation is still running; inspect the persisted job before resuming")
        assert job["result"]["engine"] == "ace_step" and job["result"]["device"] == "cuda"
        assert job["result"]["sampler_correction"] == "DCW double"
        assert 29 <= job["result"]["duration_seconds"] <= 31
        page.locator("#music-result audio").wait_for()
        player = page.locator("#music-result audio")
        player.evaluate("a=>a.play()")
        page.wait_for_function("document.querySelector('#music-result audio').currentTime>.1")
        playback = player.evaluate("a=>({duration:a.duration,currentTime:a.currentTime})")
        player.evaluate("a=>a.pause()")
        with page.expect_download() as download:
            page.locator("#music-result a[download]").click()
        data = Path(download.value.path()).read_bytes()
        assert data == page.request.get(job["result"]["audio_url"]).body()
        assets = page.request.get("/api/audio-assets", params={"limit": 200}).json()["items"]
        asset = next(item for item in assets if item["id"] == job["result"]["audio_asset_id"])
        assert asset["kind"] == "music" and asset["contains_accompaniment"]
        page.screenshot(path=str(OUT / "music-ready-desktop.png"), full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(path=str(OUT / "music-ready-mobile.png"), full_page=True)
        layout = page.evaluate("({body:document.body.scrollWidth,document:document.documentElement.scrollWidth})")
        assert max(layout.values()) <= 390, layout
        assert not errors, errors
        evidence.update(result=job["result"], playback=playback, download_matches=True,
            download_sha256=hashlib.sha256(data).hexdigest(), audio_asset=asset, mobile=layout,
            browser_errors=errors)
        evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
        browser.close()
    print("PASS: real ACE-Step CUDA song, browser playback/download, PostgreSQL asset, mobile")


if __name__ == "__main__":
    main()
