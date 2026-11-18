// Screenshot spikes/img3d_eval/compare.html for every item: photo | GT | SAM 3D | TRELLIS.
//   node spikes/img3d_eval/compare_shots.js BASE_URL ITEMS_JSON OUT_DIR
// BASE_URL serves the repo root (python -m http.server); ITEMS_JSON is a list of
// {item, photo, gt, sam3d, trellis, gtsub, s3sub, trsub} with paths relative to BASE_URL.
const { chromium } = require('playwright');
const fs = require('fs');
(async () => {
  const [base, itemsPath, outDir] = process.argv.slice(2);
  const items = JSON.parse(fs.readFileSync(itemsPath, 'utf8'));
  fs.mkdirSync(outDir, { recursive: true });
  const browser = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
  const page = await browser.newPage({ viewport: { width: 1600, height: 420 }, deviceScaleFactor: 1 });
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  for (const it of items) {
    const q = new URLSearchParams();
    for (const [k, v] of Object.entries(it)) if (v) q.set(k, k === 'item' || k.endsWith('sub') ? v : `${base}/${v}`);
    await page.goto(`${base}/spikes/img3d_eval/compare.html?${q}`);
    await page.waitForFunction(() => window.__ready === true, null, { timeout: 120000 }).catch(() => errors.push(`${it.item}: timeout`));
    await page.waitForTimeout(300);
    await page.screenshot({ path: `${outDir}/${it.item}.png` });
  }
  console.log(errors.length ? errors.join('\n') : 'ok');
  await browser.close();
})();
