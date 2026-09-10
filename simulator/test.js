const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  page.on('console', msg => console.log('BROWSER:', msg.text()));
  await page.goto('http://127.0.0.1:8000/?mode=dashboard');
  await page.waitForTimeout(2000);
  const text = await page.evaluate(() => document.getElementById('loadingStatusText').innerText);
  console.log('STATUS TEXT:', text);
  await browser.close();
})();
