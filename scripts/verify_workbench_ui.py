"""Task-first navigation, editor reuse, native validation and responsive checks."""

from pathlib import Path
import json
from playwright.sync_api import sync_playwright
import browser_smoke as smoke


def exercise(url, channel, screenshots):
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=channel, headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1050})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url, wait_until="networkidle")
        page.locator("#connection-status.ok").wait_for()
        assert page.locator(".workspace-nav > button").count() == 6
        assert page.locator("#view-home").is_visible()
        assert page.locator("[data-create]").count() == 4
        for mode in ("generate", "covers", "singing", "music"):
            page.locator("#nav-home").click()
            page.locator(f"[data-create={mode}]").click()
            assert page.locator(f"#view-{mode}").is_visible()
            assert page.locator(f"#view-{mode} .flow-guide").is_visible()
        page.locator("#nav-voices").click()
        assert page.locator("#voice-grid .voice-card").count() >= 8
        assert "可用于朗读" in page.locator("#voice-grid").inner_text()
        page.locator("#voice-grid [data-favorite]").first.click()
        page.locator("#voice-filters").get_by_role("button", name="已收藏", exact=True).click()
        assert page.locator("#voice-grid .voice-card").count() == 1
        page.reload(wait_until="networkidle")
        page.locator("#connection-status.ok").wait_for()
        page.locator("#nav-voices").click()
        page.locator("#voice-grid [data-favorite]").first.wait_for()
        assert page.locator("#voice-grid [data-favorite]").first.get_attribute("aria-pressed") == "true"
        page.get_by_role("button", name="管理朗读预设", exact=True).click()
        page.locator("#new-preset").click()
        page.locator("#preset-name").fill("改版回归音色")
        page.locator("#save-preset").click()
        page.wait_for_function("document.querySelector('#preset-id').value.startsWith('preset-')")
        page.locator("#use-edited-preset").click()
        assert "改版回归音色" in page.locator("#generation-preset-summary").inner_text()
        page.locator("#synthesis-text").fill("切换页面后内容应保留。")
        page.locator("#nav-home").click()
        page.locator("[data-create=generate]").click()
        assert page.locator("#synthesis-text").input_value() == "切换页面后内容应保留。"
        page.locator("#nav-trainhub").click()
        page.locator("#view-trainhub .task-card").filter(has_text="准备翻唱录音 →").click()
        assert page.locator("#cover-training-form").is_visible()
        page.locator("#nav-settings").click()
        assert page.locator("#model-list").is_visible()
        assert page.locator("#job-list").is_hidden()
        page.locator("#nav-tasks").click()
        page.locator("[data-task-filter=failed]").click()
        assert page.locator("[data-task-filter=failed]").get_attribute("aria-pressed") == "true"
        # Real compile and restoration; no model invocation is needed here.
        score = page.request.post(url + "/api/singing/score/from-text", data={"title": "复用验收", "lyrics": "你好", "melody": "1 2", "bpm": 120}).json()["score"]
        page.evaluate("v=>{switchTab('singing');StudioSinging.loadScore(v);}", {"score": score, "steps": 20})
        assert "复用验收" == page.locator("#singing-title").input_value()
        assert "1.00 秒" in page.locator("#singing-score-summary").inner_text()
        page.locator("#nav-home").click()
        if screenshots:
            screenshots.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(screenshots / "workbench-home-desktop.png"), full_page=True)
        for width in (390, 768):
            page.set_viewport_size({"width": width, "height": 900})
            for name in ("home", "voices", "trainhub", "tasks", "settings", "assets"):
                page.locator("#nav-" + name).click()
                assert page.evaluate("document.documentElement.scrollWidth<=innerWidth"), (name, width)
            page.locator("#nav-home").click()
            for mode in ("generate", "covers", "singing", "music"):
                page.locator("[data-create=" + mode + "]").click()
                assert page.evaluate("document.documentElement.scrollWidth<=innerWidth"), (mode, width)
                page.locator("#nav-home").click()
            if screenshots:
                page.screenshot(path=str(screenshots / f"workbench-home-{width}.png"), full_page=True)
        assert not errors, errors
        browser.close()
    print("PASS: six sections, four workflows, preset CRUD/use, score reuse, preserved inputs, responsive layout")


if __name__ == "__main__":
    smoke.exercise_ui = exercise
    smoke.main()
