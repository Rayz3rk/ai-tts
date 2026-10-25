"""Verify the live PostgreSQL audio library and restore edited metadata.

Requires Playwright and Edge, for example:
    python scripts/verify_postgres_ui.py --job-id <succeeded-synthesis-job>
The application must already be running. No synthesis is submitted by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.parse
import urllib.request
from uuid import uuid4

from playwright.sync_api import expect, sync_playwright


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7861")
    parser.add_argument("--job-id")
    parser.add_argument("--browser", default="msedge")
    parser.add_argument("--evidence", type=Path, default=Path("outputs/validation"))
    args = parser.parse_args()
    base = args.url.rstrip("/")
    token = os.environ.get("AUDIO_STUDIO_TOKEN", "")
    headers = {"Authorization": "Bearer " + token} if token else {}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(path: str, data: dict | None = None) -> dict:
        payload = json.dumps(data).encode("utf-8") if data is not None else None
        request_headers = dict(headers)
        if payload is not None:
            request_headers["Content-Type"] = "application/json"
        req = urllib.request.Request(base + path, payload, request_headers, method="PATCH" if payload is not None else "GET")
        with opener.open(req, timeout=30) as response:
            return json.load(response)

    status = request("/api/status")
    assert status["storage"]["backend"] == "postgresql", "This check requires a real PostgreSQL backend"
    job = None
    if args.job_id:
        deadline = time.monotonic() + 180
        while True:
            job = request("/api/jobs/" + args.job_id)
            if job["status"] in {"succeeded", "failed", "cancelled"}:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("The existing synthesis task has not completed")
            time.sleep(.5)
        assert job["status"] == "succeeded", "The existing synthesis task did not succeed"
    assets = request("/api/audio-assets?limit=200")
    target_id = (job.get("result") or {}).get("audio_asset_id") if job else None
    candidates = [asset for asset in assets["items"] if asset.get("available") and (not target_id or asset["id"] == target_id)]
    assert candidates, "No available audio asset found"
    asset = candidates[0]
    original = {key: asset[key] for key in ("name", "tags", "favorite")}
    endpoint = "/api/audio-assets/" + asset["id"]
    marker = "UI-验收-" + uuid4().hex[:8]
    edited_name = "数据库素材验收 " + marker
    errors: list[str] = []
    evidence = {"backend": "postgresql", "asset_id": asset["id"], "job_id": asset.get("job_id"), "total_assets": assets["total"]}
    args.evidence.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel=args.browser or None, headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 1040}, extra_http_headers=headers, accept_downloads=True)
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
        try:
            page.goto(base, wait_until="networkidle")
            page.locator("#connection-status.ok").wait_for()
            page.locator("#tab-assets").click()
            expect(page.locator("#asset-list")).to_have_attribute("aria-busy", "false")
            page.locator("#asset-query").fill(asset["filename"])
            page.locator("#asset-search-form").get_by_role("button", name="搜索", exact=True).click()
            card = page.locator(".asset-card")
            expect(card).to_have_count(1)
            expect(card.locator(".asset-name")).to_have_text(original["name"])
            evidence["search"] = True

            card.locator("[data-asset-edit]").click()
            card.get_by_role("textbox", name="素材名称", exact=True).fill(edited_name)
            card.get_by_role("textbox", name="素材标签", exact=True).fill(marker + "，验收")
            card.get_by_role("button", name="保存修改", exact=True).click()
            expect(card.locator(".asset-name")).to_have_text(edited_name)
            updated = request("/api/audio-assets?query=" + urllib.parse.quote(marker))
            assert len(updated["items"]) == 1 and updated["items"][0]["tags"] == [marker, "验收"]
            evidence["rename_tags_persisted"] = True

            favorite = card.locator(".asset-favorite")
            favorite.click()
            expect(favorite).to_have_attribute("aria-pressed", str(not original["favorite"]).lower())
            assert request("/api/audio-assets?query=" + urllib.parse.quote(marker))["items"][0]["favorite"] is not original["favorite"]
            evidence["favorite_persisted"] = True

            card.get_by_role("button", name=marker, exact=True).click()
            expect(page.locator("#asset-query")).to_have_value(marker)
            expect(card).to_have_count(1)
            evidence["tag_search"] = True

            card.get_by_role("button", name="试听 ▷", exact=True).click()
            audio = card.locator("audio")
            audio.wait_for()
            page.wait_for_function("() => { const a = document.querySelector('.asset-card audio'); return a && a.readyState >= 2 && a.currentTime > 0; }")
            playback = audio.evaluate("a => ({duration:a.duration,currentTime:a.currentTime,source:a.src,readyState:a.readyState})")
            assert playback["source"].startswith("blob:") and playback["duration"] > 0
            audio.evaluate("a => a.pause()")
            evidence["playback"] = playback

            with page.expect_download() as pending_download:
                card.get_by_role("button", name="下载音频 ↓", exact=True).click()
            downloaded = pending_download.value
            downloaded_bytes = Path(downloaded.path()).read_bytes()
            with opener.open(urllib.request.Request(base + asset["audio_url"], headers=headers), timeout=30) as response:
                original_bytes = response.read()
            assert downloaded_bytes == original_bytes and downloaded_bytes[:4] == b"RIFF"
            evidence["download"] = {"bytes": len(downloaded_bytes), "sha256": hashlib.sha256(downloaded_bytes).hexdigest(), "matches_registered_audio": True}
            page.screenshot(path=str(args.evidence / "postgres-library-desktop.png"), full_page=True)

            page.set_viewport_size({"width": 390, "height": 844})
            page.screenshot(path=str(args.evidence / "postgres-library-mobile.png"), full_page=True)
            layout = page.evaluate("() => ({viewport:innerWidth,body:document.body.scrollWidth,document:document.documentElement.scrollWidth})")
            assert layout["document"] <= layout["viewport"] and layout["body"] <= layout["viewport"], layout
            evidence["mobile_layout"] = layout
            page.locator("#tab-models").click()
            expect(page.locator("#storage-info")).to_contain_text("PostgreSQL")
            evidence["storage_status"] = True
            assert not errors, errors
        finally:
            try:
                request(endpoint, original)
                restored = request("/api/audio-assets?query=" + urllib.parse.quote(asset["filename"]))["items"]
                restored = next(item for item in restored if item["id"] == asset["id"])
                assert all(restored[key] == value for key, value in original.items())
                evidence["original_metadata_restored"] = True
            finally:
                evidence["browser_errors"] = errors
                (args.evidence / "postgres-ui.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
                browser.close()
    print("PASS: real PostgreSQL library search, edit, tags, favorite, playback, WAV download, mobile layout; metadata restored")


if __name__ == "__main__":
    main()
