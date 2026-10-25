"""Real MIDI/lyrics -> DiffSinger -> target RVC voice browser acceptance."""

import hashlib
import json
from pathlib import Path
import struct
import time
import urllib.request

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "validation"
BASE = "http://127.0.0.1:7861"
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def get(path):
    with opener.open(BASE + path, timeout=30) as response:
        return json.load(response)


def make_midi(path):
    def chunk(payload):
        return b"MTrk" + struct.pack(">I", len(payload)) + payload + b""
    end = b"\0\xff\x2f\0"
    tempo = b"\0\xff\x51\x03\x09\x27\xc0\x87\x40\xff\x51\x03\x07\xa1\x20" + end
    melody = b"\0\xff\x03\x06Melody"
    for pitch in (60, 62, 64, 65):
        melody += bytes([0, 0x90, pitch, 80]) + bytes([0x83, 0x60, 0x80, pitch, 0])
    melody += end
    chords = b"\0\xff\x03\x06Chords\0\x90\x3c\x50\0\x90\x40\x50\x87\x40\x80\x3c\0\0\x80\x40\0" + end
    path.write_bytes(b"MThd" + struct.pack(">IHHH", 6, 1, 3, 480) + chunk(tempo) + chunk(melody) + chunk(chords))


def main():
    evidence_path = OUT / "singing-browser.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8")) if evidence_path.exists() else {}
    pipeline = json.loads((OUT / "svc-pipeline.json").read_text(encoding="utf-8"))
    target = pipeline["jobs"]["training"]["result"]["model_id"]
    midi = ROOT / "work" / "singing-validation.mid"
    make_midi(midi)
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 1100})
            page.set_default_timeout(120000)
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(BASE, wait_until="networkidle")
            page.locator("#tab-singing").click()
            page.locator(f"#singing-target option[value='{target}']").wait_for(state="attached")
            if not evidence.get("job_id"):
                page.locator("#singing-title").fill("歌词旋律串联验收")
                with page.expect_response(lambda r: "/api/singing/midi" in r.url) as imported:
                    page.locator("#singing-midi").set_input_files(midi)
                assert imported.value.status == 200 and imported.value.json()["requires_track_selection"]
                with page.expect_response(lambda r: "/api/singing/midi" in r.url) as selected:
                    page.locator("#singing-track").select_option("1")
                result = selected.value.json()
                assert not result["errors"] and result["tempo_changes"] == 2
                assert [note["duration"] for note in result["notes"]] == [.6, .6, .5, .5]
                with page.expect_response(lambda r: "/api/singing/score/validate" in r.url) as validated:
                    page.locator("#singing-validate").click()
                assert validated.value.status == 200, validated.value.text()
                page.locator("#singing-target").select_option(target)
                page.locator("#singing-steps").fill("12")
                with page.expect_response(lambda r: "/api/singing/generate" in r.url) as submitted:
                    page.locator("#singing-submit").click()
                assert submitted.value.status == 202, submitted.value.text()
                evidence.update(job_id=submitted.value.json()["id"], midi_import=True, explicit_melody_track=True,
                    tempo_changes=2, imported_notes=result["notes"], score_validated=True, target_voice_id=target)
                evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
            identity = evidence["job_id"]
            previous = None
            while True:
                job = get("/api/jobs/" + identity)
                latest = (job["status"], (job.get("logs") or [""])[-1])
                if latest != previous:
                    print(identity, latest, flush=True)
                    previous = latest
                if job["status"] in {"succeeded", "failed", "cancelled"}:
                    assert job["status"] == "succeeded", job.get("error")
                    break
                time.sleep(2)
            page.wait_for_function("document.querySelectorAll('#singing-job-detail audio').length === 2", timeout=30000)
            player = page.locator("#singing-job-detail audio").first
            player.evaluate("a=>a.play()")
            page.wait_for_function("document.querySelector('#singing-job-detail audio').currentTime>0")
            evidence["playback"] = player.evaluate("a=>({duration:a.duration,currentTime:a.currentTime,readyState:a.readyState})")
            player.evaluate("a=>a.pause()")
            with page.expect_download() as pending:
                page.locator("#singing-job-detail a[download]").first.click()
            actual = Path(pending.value.path()).read_bytes()
            with opener.open(BASE + job["result"]["audio_url"], timeout=30) as response:
                expected = response.read()
            assert actual == expected
            page.screenshot(path=str(OUT / "singing-ready-desktop.png"), full_page=True)
            page.set_viewport_size({"width": 390, "height": 844})
            page.screenshot(path=str(OUT / "singing-ready-mobile.png"), full_page=True)
            layout = page.evaluate("({viewport:innerWidth,body:document.body.scrollWidth,document:document.documentElement.scrollWidth})")
            assert layout["document"] <= 390 and layout["body"] <= 390, layout
            assert not errors, errors
            evidence.update(result=job["result"], download_sha256=hashlib.sha256(actual).hexdigest(),
                download_matches=True, two_players=True, mobile_layout=layout, browser_errors=errors)
            evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
        finally:
            browser.close()
    print("PASS: MIDI track/tempo/lyrics, real DiffSinger+RVC chain, playback/download, mobile layout", flush=True)


if __name__ == "__main__":
    main()
