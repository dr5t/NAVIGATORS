from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(
        viewport={"width": 390, "height": 844},
        device_scale_factor=2,
        is_mobile=True,
        has_touch=True
    )
    page.goto('http://localhost:8000/?mode=mobile')
    page.wait_for_timeout(2000)
    page.screenshot(path='mobile_screenshot.png')
    browser.close()
