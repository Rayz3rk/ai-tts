"""Browser checks for bulk score editing and the real dry-vocal sample."""

import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "validation"


def main():
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(base_url="http://127.0.0.1:7861", viewport={"width": 1440, "height": 1100})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto("http://127.0.0.1:7861/", wait_until="networkidle")
        page.locator("#tab-covers").click()
        page.locator("#cover-model option[value='rvc-e87029619731']").wait_for(state="attached")
        assert "仅" in page.locator("#cover-voice-note").inner_text()
        before = page.request.get("/api/jobs").json()
        with page.expect_response(lambda r: "/api/covers/test-vocal" in r.url) as loaded:
            page.locator("#cover-load-test").click()
        sample = loaded.value.json()
        assert loaded.value.status == 200
        assert Path(page.locator("#cover-source").input_value()).resolve() == ROOT / "outputs/samples/singing-dry-demo.wav"
        assert len(page.request.get("/api/jobs").json()) == len(before)
        page.locator("#tab-singing").click()
        assert not page.locator("#singing-advanced").evaluate("e=>e.open")
        quality = json.loads((OUT / "singing-quality-job.json").read_text(encoding="utf-8"))
        page.locator("#singing-job-list button").filter(has_text=quality["score"]["title"]).first.click()
        page.locator("#singing-job-detail audio").first.wait_for()
        player = page.locator("#singing-job-detail audio").first
        player.evaluate("a=>a.play()")
        page.wait_for_function("document.querySelector('#singing-job-detail audio').currentTime > .1")
        playback = player.evaluate("a=>({duration:a.duration,currentTime:a.currentTime})")
        player.evaluate("a=>a.pause()")
        assert abs(playback["duration"] - sample["duration_seconds"]) < .02
        page.locator("#singing-bpm").fill("180")
        with page.expect_response(lambda r: "/api/singing/score/from-text" in r.url) as compiled:
            page.locator("#singing-validate").click()
        report = compiled.value.json()
        assert compiled.value.status == 200
        assert abs(report["duration_seconds"] - 3) < .00001
        page.wait_for_function("document.querySelector('#singing-validation').textContent.includes('检查通过')")
        # Recompile changed lyrics automatically; reject a count mismatch.
        page.locator("#singing-lyrics").fill("你好")
        with page.expect_response(lambda r: "/api/singing/score/from-text" in r.url) as rejected:
            page.locator("#singing-validate").click()
        assert rejected.value.status == 400
        page.locator("#singing-lyrics").fill("你好世界")
        with page.expect_response(lambda r: "/api/singing/midi" in r.url) as imported:
            page.locator("#singing-midi").set_input_files(ROOT / "work/singing-validation.mid")
        assert imported.value.json()["requires_track_selection"]
        with page.expect_response(lambda r: "/api/singing/midi" in r.url) as selected:
            page.locator("#singing-track").select_option("1")
        assert [n["duration"] for n in selected.value.json()["notes"]] == [.6, .6, .5, .5]
        # The bulk BPM is still 180; it must not overwrite imported MIDI timing.
        with page.expect_response(lambda r: "/api/singing/score/validate" in r.url) as validated:
            page.locator("#singing-validate").click()
        assert validated.value.status == 200 and validated.value.json()["duration_seconds"] == 2.2
        page.locator("#singing-example").click()
        assert page.locator("#singing-validation").inner_text() == ""
        page.locator(".toast.error").wait_for(state="hidden")
        page.screenshot(path=str(OUT / "song-bulk-desktop.png"), full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(path=str(OUT / "song-bulk-mobile.png"), full_page=True)
        layout = page.evaluate("({body:document.body.scrollWidth,document:document.documentElement.scrollWidth})")
        assert max(layout.values()) <= 390, layout
        assert not errors, errors
        (OUT / "song-workflows-browser.json").write_text(json.dumps({"sample": sample,
            "automatic_recompile": True, "bpm180_duration": report["duration_seconds"],
            "playback": playback, "midi_preserves_tempo": True,
            "mismatch_rejected": True, "advanced_collapsed": True, "mobile": layout,
            "browser_errors": errors}, ensure_ascii=False, indent=2), encoding="utf-8")
        browser.close()
    print("PASS: dry sample, demo warning, bulk BPM, automatic recompile, mismatch and mobile")


if __name__ == "__main__":
    main()
