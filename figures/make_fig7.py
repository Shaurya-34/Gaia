"""Figure 7: self-contained interactive page (no external libraries) with
sliders for surviving members and distance, returning the fitted model's
detection probability. Coefficients are copied from fit_detectability.py's
output (results/detectability/fit.json)."""
import json

from figstyle import OUT, ROOT

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Detection Probability Explorer</title>
<style>
:root {
  color-scheme: light;
  --surface: #fcfcfb; --page: #f9f9f7; --ink: #0b0b0b; --ink2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --accent: #2a78d6; --accent-soft: rgba(42,120,214,0.14); --violet: #4a3aa7;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface: #1a1a19; --page: #0d0d0d; --ink: #ffffff; --ink2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --accent: #3987e5; --accent-soft: rgba(57,135,229,0.18); --violet: #9085e9;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface: #1a1a19; --page: #0d0d0d; --ink: #ffffff; --ink2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --accent: #3987e5; --accent-soft: rgba(57,135,229,0.18); --violet: #9085e9;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
  font-family: "Helvetica Neue", Helvetica, Arial, system-ui, sans-serif; }
main { max-width: 760px; margin: 0 auto; padding: 40px 16px 56px; }
h1 { font-weight: 400; font-size: clamp(28px, 5vw, 40px); letter-spacing: -0.01em; margin: 0 0 8px; }
p.lede { color: var(--ink2); margin: 0 0 28px; line-height: 1.5; max-width: 60ch; }
.panel { background: var(--surface); border: 1px solid var(--grid); border-radius: 8px; padding: 20px; }
.readout { display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap; margin-bottom: 18px; }
.prob { font-size: clamp(48px, 11vw, 76px); font-weight: 400; line-height: 1; font-variant-numeric: tabular-nums; }
.prob-label { color: var(--ink2); font-size: 15px; }
label { display: block; font-size: 14px; color: var(--ink2); margin: 16px 0 6px; }
label b { color: var(--ink); font-weight: 500; font-variant-numeric: tabular-nums; }
input[type=range] { width: 100%; accent-color: var(--accent); }
svg { width: 100%; height: auto; display: block; margin-top: 18px; }
.note { color: var(--muted); font-size: 13px; line-height: 1.5; margin-top: 18px; }
</style>
</head>
<body>
<main>
  <h1>How small a group can this search find?</h1>
  <p class="lede">The fitted detectability model from a Gaia DR3 injection-recovery test in one sky region (block 6).
  Move the sliders to see the probability that the baseline detector finds a co-moving group (CST &gt; __TAU__).</p>
  <div class="panel">
    <div class="readout"><span class="prob" id="prob">-</span><span class="prob-label" id="probLabel"></span></div>
    <label for="n">Surviving member stars: <b id="nVal"></b></label>
    <input id="n" type="range" min="0" max="1000" value="400" aria-describedby="nVal">
    <label for="d">Distance: <b id="dVal"></b></label>
    <input id="d" type="range" min="50" max="500" step="5" value="250" aria-describedby="dVal">
    <svg id="curve" viewBox="0 0 640 220" role="img" aria-label="Detection probability against surviving members at the chosen distance"></svg>
    <p class="note">Valid only inside the fitted range: block 6's four fields, 50-500 pc, __NMIN__-2,000 surviving members,
    the PREREG quality cuts, Plummer clusters with twin-star photometry. Parallax error and local density are held at the
    block-6 medians (__PERR__ mas, rho = __RHO__). Model: P = 1 / (1 + exp(-(log10 n - a) / sigma)),
    a = a0 + a1 d/100 + a2 log10(parallax error) + a3 log10(rho). Fit on __NFIT__ trials; checked on __NHELD__ held-out trials,
    not on unseen sky.</p>
  </div>
</main>
<script>
const P = __PARAMS__;
const REF = { perr: __PERR__, rho: __RHO__ };
const NMIN = __NMIN__, NMAX = 2000;
const nFromSlider = v => Math.round(Math.pow(10, Math.log10(NMIN) + (Math.log10(NMAX) - Math.log10(NMIN)) * v / 1000));
function prob(n, d) {
  const a = P.a0 + P.a1 * d / 100 + P.a2 * Math.log10(REF.perr) + P.a3 * Math.log10(REF.rho);
  return 1 / (1 + Math.exp(-(Math.log10(n) - a) / P.sigma));
}
const svg = document.getElementById('curve');
const W = 640, H = 220, L = 44, R = 12, T = 12, B = 34;
const x = n => L + (Math.log10(n) - Math.log10(NMIN)) / (Math.log10(NMAX) - Math.log10(NMIN)) * (W - L - R);
const y = p => T + (1 - p) * (H - T - B);
function draw(n, d) {
  const css = getComputedStyle(document.documentElement);
  let pts = [];
  for (let i = 0; i <= 200; i++) {
    const nn = Math.pow(10, Math.log10(NMIN) + (Math.log10(NMAX) - Math.log10(NMIN)) * i / 200);
    pts.push(`${x(nn).toFixed(1)},${y(prob(nn, d)).toFixed(1)}`);
  }
  const ticks = [5, 10, 25, 60, 150, 500, 2000].filter(t => t >= NMIN);
  svg.innerHTML = `
    ${[0, 0.5, 1].map(p => `<line x1="${L}" x2="${W - R}" y1="${y(p)}" y2="${y(p)}" stroke="var(--grid)" stroke-width="1"/>
      <text x="${L - 8}" y="${y(p) + 4}" text-anchor="end" font-size="11" fill="var(--muted)">${p}</text>`).join('')}
    ${ticks.map(t => `<text x="${x(t)}" y="${H - 12}" text-anchor="middle" font-size="11" fill="var(--muted)">${t}</text>`).join('')}
    <text x="${(L + W - R) / 2}" y="${H}" text-anchor="middle" font-size="11" fill="var(--ink2)">surviving members (log scale)</text>
    <polyline points="${pts.join(' ')}" fill="none" stroke="var(--accent)" stroke-width="2"/>
    <line x1="${x(n)}" x2="${x(n)}" y1="${T}" y2="${H - B}" stroke="var(--violet)" stroke-width="1" stroke-dasharray="3 3"/>
    <circle cx="${x(n)}" cy="${y(prob(n, d))}" r="6" fill="var(--violet)" stroke="var(--surface)" stroke-width="2"/>`;
}
function update() {
  const n = nFromSlider(+document.getElementById('n').value);
  const d = +document.getElementById('d').value;
  const p = prob(n, d);
  document.getElementById('nVal').textContent = n.toLocaleString();
  document.getElementById('dVal').textContent = d + ' pc';
  document.getElementById('prob').textContent = (100 * p).toFixed(p > 0.995 || p < 0.005 ? 1 : 0) + '%';
  document.getElementById('probLabel').textContent = `chance of detection for ${n.toLocaleString()} stars at ${d} pc`;
  draw(n, d);
}
document.getElementById('n').addEventListener('input', update);
document.getElementById('d').addEventListener('input', update);
update();
</script>
</body>
</html>
"""


def main():
    fit = json.loads((ROOT / "results" / "detectability" / "fit.json").read_text())
    ref = fit["covariate_reference"]
    html = (TEMPLATE.replace("__PARAMS__", json.dumps(fit["params"]))
            .replace("__PERR__", f"{ref['median_parallax_error']:.4f}").replace("__RHO__", f"{ref['rho_data']:.4f}")
            .replace("__TAU__", str(fit["tau"])).replace("__NMIN__", "5")
            .replace("__NFIT__", str(fit["trials_fit"])).replace("__NHELD__", str(fit["trials_held_out"])))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fig7_interactive.html").write_text(html, encoding="utf-8")
    (OUT / "fig7_caption.txt").write_text(
        "Interactive detection-probability explorer: sliders for surviving members and distance return the fitted "
        "PREREG section 6 model's probability that the baseline detects a group (CST > 3), with parallax error and "
        "density held at block-6 medians. Limitation: valid only inside block 6's fitted range and the PREREG cuts; "
        "checked on held-out trials, not unseen sky.\n", encoding="utf-8")
    print(f"saved {OUT / 'fig7_interactive.html'}")


if __name__ == "__main__":
    main()
