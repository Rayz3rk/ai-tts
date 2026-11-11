"""Capture real workbench pages with built-in presets and no private history."""

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7861")
    args = parser.parse_args()
    output = Path(__file__).resolve().parents[1] / "docs" / "images" / "workbench"
    output.mkdir(parents=True, exist_ok=True)
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
        page.on("pageerror", lambda error: errors.append(str(error)))

        def public_data(route):
            path = route.request.url.split("?", 1)[0].rsplit("/api/", 1)[-1]
            if path == "jobs":
                route.fulfill(json=[])
                return
            response = route.fetch()
            data = response.json()
            if path == "presets":
                data = [preset for preset in data if preset.get("builtin") and preset["engine"] == "kokoro"]
            elif path == "status":
                data["data_dir"] = "studio_data"
                data["storage"]["workspace_id"] = "demo"
                data["models"] = [model for model in data["models"] if model.get("builtin")]
                for model in data["models"]:
                    for key in list(model):
                        if key == "path" or key.endswith("_path"):
                            model[key] = "本地模型目录"
                    model["missing"] = ["需要准备模型文件"] if model.get("missing") else []
            route.fulfill(response=response, json=data)

        for endpoint in ("jobs", "status", "presets"):
            page.route(args.url + "/api/" + endpoint, public_data)
        page.goto(args.url, wait_until="networkidle")
        page.locator("#connection-status.ok").wait_for()
        assert page.locator("#home-recent .job-card").count() == 0
        page.screenshot(path=str(output / "home.png"), full_page=True)
        page.locator("[data-create=generate]").click()
        page.locator("#synthesis-text").fill("清晨的阳光洒在窗边。新的一天，从一段好声音开始。")
        page.screenshot(path=str(output / "narration.png"), full_page=True)
        page.locator("#nav-voices").click()
        assert page.locator("#voice-grid .voice-card").count() == 8
        page.screenshot(path=str(output / "voices.png"), full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        page.locator("#nav-home").click()
        assert page.evaluate("document.documentElement.scrollWidth<=innerWidth")
        page.screenshot(path=str(output / "mobile.png"), full_page=True)
        assert not errors, errors
        browser.close()
    print("Captured 4 real UI screenshots using built-in presets, demo text and an empty history")


if __name__ == "__main__":
    main()
