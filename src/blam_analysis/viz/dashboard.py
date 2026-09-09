import json

import pandas as pd

from blam_analysis.viz.common import named_row, task_short_label
from blam_analysis.viz.theta import compute_theta_rows


def build_dashboard_payload(
    theta_df: pd.DataFrame, spec_df: pd.DataFrame, task_df: pd.DataFrame, mae_df: pd.DataFrame = None,
    display_names: dict = None,
) -> dict:
    display_names = display_names or {}

    theta_rows = compute_theta_rows(theta_df, display_names)

    tasks = []
    for row in task_df.itertuples(index=False):
        rows = spec_df[spec_df["task"] == row.task].sort_values("u_mean", ascending=False)
        entry = {
            "key": row.task,
            "label": task_short_label(row.task),
            "mu_mean": float(row.mu_mean),
            "models": [named_row(r.model, r.u_mean, r.u_std, display_names) for r in rows.itertuples(index=False)],
        }
        if mae_df is not None:
            mrows = mae_df[mae_df["task"] == row.task].sort_values("mae_mean", ascending=True)
            entry["mae"] = [named_row(r.model, r.mae_mean, r.mae_se, display_names) for r in mrows.itertuples(index=False)]
        tasks.append(entry)
    return {"theta": theta_rows, "tasks": tasks, "has_mae": mae_df is not None}


def render_dashboard_html(payload: dict) -> str:
    data_json = json.dumps(payload).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>BLAM dashboard</title>
<style>
  :root {{
    color-scheme: light;
    --surface-1: #fcfcfb;
    --surface-2: #f9f9f7;
    --text-primary: #0b0b0b;
    --text-secondary: #52514e;
    --text-muted: #898781;
    --gridline: #e1e0d9;
    --baseline: #c3c2b7;
    --border: rgba(11,11,11,0.10);
    --strong: #2a78d6;
    --weak: #e34948;
    --neutral: #898781;
    --accent-wash: rgba(42,120,214,0.12);
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      color-scheme: dark;
      --surface-1: #1a1a19;
      --surface-2: #0d0d0d;
      --text-primary: #ffffff;
      --text-secondary: #c3c2b7;
      --text-muted: #898781;
      --gridline: #2c2c2a;
      --baseline: #383835;
      --border: rgba(255,255,255,0.10);
      --strong: #3987e5;
      --weak: #e66767;
      --neutral: #a3a19a;
      --accent-wash: rgba(57,135,229,0.16);
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: var(--surface-2);
    color: var(--text-primary);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  }}
  .layout {{
    display: flex;
    min-height: 100vh;
  }}
  .sidebar {{
    width: 260px;
    flex-shrink: 0;
    background: var(--surface-1);
    border-right: 1px solid var(--border);
    padding: 16px 0;
    overflow-y: auto;
  }}
  .sidebar h2 {{
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--text-muted);
    margin: 0 16px 8px;
  }}
  .task-btn {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    width: 100%;
    text-align: left;
    background: none;
    border: none;
    border-left: 3px solid transparent;
    padding: 7px 16px;
    font-size: 13px;
    color: var(--text-primary);
    cursor: pointer;
    font-family: inherit;
  }}
  .task-btn:hover {{ background: var(--accent-wash); }}
  .task-btn.active {{ background: var(--accent-wash); border-left-color: var(--strong); }}
  .task-metric {{ color: var(--text-muted); font-size: 11px; flex-shrink: 0; padding-left: 8px; }}
  main {{ flex: 1; padding: 24px 32px; min-width: 0; }}
  .mode-toggle {{ display: flex; gap: 6px; margin-bottom: 18px; }}
  .mode-btn {{
    font-size: 12px; color: var(--text-secondary); background: var(--surface-1);
    border: 1px solid var(--border); border-radius: 6px; padding: 6px 12px; cursor: pointer;
    font-family: inherit;
  }}
  .mode-btn.active {{ color: var(--text-primary); border-color: var(--strong); background: var(--accent-wash); }}
  h1 {{ font-size: 16px; font-weight: 700; margin: 0 0 2px; }}
  .subtitle {{ color: var(--text-secondary); font-size: 12px; margin: 0 0 20px; max-width: 640px; line-height: 1.5; }}
  .toolbar {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }}
  .legend {{ display: flex; gap: 16px; font-size: 12px; color: var(--text-secondary); }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; }}
  .swatch {{ width: 10px; height: 10px; border-radius: 2px; display: inline-block; }}
  .view-toggle {{
    font-size: 12px; color: var(--text-secondary); background: var(--surface-1);
    border: 1px solid var(--border); border-radius: 6px; padding: 5px 10px; cursor: pointer;
    font-family: inherit;
  }}
  .chart-wrap {{ overflow-x: auto; }}
  svg text {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif; fill: var(--text-primary); }}
  .row-label {{ fill: var(--text-primary); font-size: 11px; }}
  .tick-label {{ fill: var(--text-muted); font-size: 10px; }}
  .bar-strong {{ fill: var(--strong); }}
  .bar-weak {{ fill: var(--weak); }}
  .bar-neutral {{ fill: var(--neutral); }}
  .bar-strong.hovered, .bar-weak.hovered, .bar-neutral.hovered {{ filter: brightness(1.15); }}
  .errorbar {{ stroke: var(--text-secondary); stroke-width: 1; }}
  .gridline {{ stroke: var(--gridline); stroke-width: 1; }}
  .baseline {{ stroke: var(--baseline); stroke-width: 1; }}
  .hit-row {{ fill: transparent; }}
  table {{ border-collapse: collapse; font-size: 12px; }}
  table th, table td {{ text-align: left; padding: 5px 12px; border-bottom: 1px solid var(--gridline); }}
  table th {{ color: var(--text-muted); font-weight: 500; }}
  #tooltip {{
    position: fixed; pointer-events: none; z-index: 10; display: none;
    background: var(--surface-1); border: 1px solid var(--border); border-radius: 6px;
    padding: 6px 10px; font-size: 12px; box-shadow: 0 2px 8px var(--border);
  }}
  #tooltip .tt-value {{ color: var(--text-primary); font-weight: 700; }}
  #tooltip .tt-name {{ color: var(--text-secondary); }}
</style>
</head>
<body>
<div class="layout">
  <nav class="sidebar" id="sidebar">
    <h2>Tasks</h2>
    <div id="task-list"></div>
  </nav>
  <main>
    <div class="mode-toggle" role="tablist">
      <button class="mode-btn active" id="mode-theta" data-mode="theta">Latent ability</button>
      <button class="mode-btn" id="mode-specialization" data-mode="specialization">Per-task specialization</button>
      <button class="mode-btn" id="mode-mae" data-mode="mae" style="display:none">Raw MAE per task</button>
    </div>
    <h1 id="chart-title"></h1>
    <p class="subtitle" id="chart-subtitle"></p>
    <div class="toolbar">
      <div class="legend">
        <span class="legend-item"><span class="swatch" style="background:var(--strong)"></span><span id="legend-strong"></span></span>
        <span class="legend-item"><span class="swatch" style="background:var(--weak)"></span><span id="legend-weak"></span></span>
      </div>
      <button class="view-toggle" id="view-toggle">View as table</button>
    </div>
    <div class="chart-wrap" id="chart-wrap">
      <svg id="chart" width="100%"></svg>
    </div>
    <div id="table-wrap" style="display:none"></div>
  </main>
</div>
<div id="tooltip"></div>
<script>
const DATA = {data_json};

const CHART_WIDTH = 920;
const LEFT_MARGIN = 220;
const RIGHT_MARGIN = 60;
const TOP_MARGIN = 12;
const BOTTOM_MARGIN = 30;
const ROW_H = 16;
const BAR_H = 11;
const CORNER_R = 4;

const MODES = {{
  theta: {{
    title: "BLAM latent ability (theta) per model",
    subtitle: "theta: overall latent ability across all tasks (sum-to-zero across the "
      + "compared models). Higher = better. Error bars are the posterior std dev.",
    strongLabel: "Above average (better)",
    weakLabel: "Below average (worse)",
    axisMode: "diverging",
    isStrong: (v, i) => v >= 0,
    symbol: "\\u03b8",
  }},
  specialization: {{
    title: "Per-model specialization",
    subtitle: "u (specialization): how much each model's error on this task deviates "
      + "from what its overall latent ability alone would predict. Negative = does better than "
      + "expected here (a relative strength for this task); positive = does worse than "
      + "expected here. Error bars are the posterior std dev.",
    strongLabel: "Stronger than expected here",
    weakLabel: "Weaker than expected here",
    axisMode: "diverging",
    isStrong: (v, i) => v < 0,
    symbol: "u",
  }},
  mae: {{
    title: "Raw MAE per task",
    subtitle: "Mean absolute error of each model on this task's held-out samples -- the "
      + "un-modeled numbers the BLAM fit is built from. Lower is better. Error bars are the "
      + "standard error of the mean.",
    strongLabel: "Best (lowest MAE)",
    weakLabel: "",
    axisMode: "zero",
    isStrong: (v, i) => i === 0,
    falseClass: "bar-neutral",
    symbol: "MAE",
  }},
}};

function barClassFor(r, i) {{
  const cfg = MODES[currentMode];
  return cfg.isStrong(r.value, i) ? "bar-strong" : (cfg.falseClass || "bar-weak");
}}

let currentMode = "theta";
let currentKey = DATA.tasks[0].key;
let showTable = false;

function formatValue(v, maxAbs) {{
  const decimals = maxAbs < 0.01 ? 4 : (maxAbs < 0.1 ? 3 : 2);
  return v.toFixed(decimals);
}}

function niceTicks(maxAbs) {{
  return [-maxAbs, -maxAbs / 2, 0, maxAbs / 2, maxAbs];
}}

function svgEl(tag, attrs) {{
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const k in attrs) el.setAttribute(k, attrs[k]);
  return el;
}}

function roundedBarPath(baselineX, endX, y, h, r) {{
  const left = Math.min(baselineX, endX);
  const right = Math.max(baselineX, endX);
  const rr = Math.min(r, (right - left) / 2, h / 2);
  if (endX >= baselineX) {{
    return `M ${{left}},${{y}} L ${{right - rr}},${{y}} Q ${{right}},${{y}} ${{right}},${{y + rr}} `
         + `L ${{right}},${{y + h - rr}} Q ${{right}},${{y + h}} ${{right - rr}},${{y + h}} L ${{left}},${{y + h}} Z`;
  }}
  return `M ${{right}},${{y}} L ${{left + rr}},${{y}} Q ${{left}},${{y}} ${{left}},${{y + rr}} `
       + `L ${{left}},${{y + h - rr}} Q ${{left}},${{y + h}} ${{left + rr}},${{y + h}} L ${{right}},${{y + h}} Z`;
}}

function renderSidebar() {{
  const list = document.getElementById("task-list");
  list.innerHTML = "";
  for (const t of DATA.tasks) {{
    const btn = document.createElement("button");
    btn.className = "task-btn" + (t.key === currentKey ? " active" : "");
    btn.dataset.key = t.key;
    const name = document.createElement("span");
    name.textContent = t.label;
    const metric = document.createElement("span");
    metric.className = "task-metric";
    metric.textContent = "\\u03bc " + t.mu_mean.toFixed(2);
    metric.title = "Task difficulty (mean log\\u2081\\u2080 MAE)";
    btn.appendChild(name);
    btn.appendChild(metric);
    btn.addEventListener("click", () => selectTask(t.key));
    list.appendChild(btn);
  }}
}}

function getCurrentRows() {{
  if (currentMode === "theta") return DATA.theta;
  if (currentMode === "mae") return DATA.tasks.find(e => e.key === currentKey).mae;
  return DATA.tasks.find(e => e.key === currentKey).models;
}}

function updateHeader() {{
  const cfg = MODES[currentMode];
  let title = cfg.title;
  if (currentMode === "specialization" || currentMode === "mae") {{
    title += " \\u2013 " + DATA.tasks.find(e => e.key === currentKey).label;
  }}
  document.getElementById("chart-title").textContent = title;
  document.getElementById("chart-subtitle").textContent = cfg.subtitle;
  document.getElementById("legend-strong").textContent = cfg.strongLabel;
  document.getElementById("legend-weak").textContent = cfg.weakLabel;
  document.querySelector(".legend-item:nth-child(2)").style.display = cfg.weakLabel ? "flex" : "none";
}}

function renderCurrent() {{
  const rows = getCurrentRows();
  if (showTable) renderTable(rows); else renderChart(rows);
}}

function switchMode(mode) {{
  currentMode = mode;
  for (const btn of document.querySelectorAll(".mode-btn")) {{
    btn.classList.toggle("active", btn.dataset.mode === mode);
  }}
  document.getElementById("sidebar").style.display = mode === "theta" ? "none" : "block";
  updateHeader();
  renderCurrent();
}}

function selectTask(key) {{
  currentKey = key;
  for (const btn of document.querySelectorAll(".task-btn")) {{
    btn.classList.toggle("active", btn.dataset.key === key);
  }}
  updateHeader();
  renderCurrent();
}}

function renderChart(rows) {{
  const svg = document.getElementById("chart");
  svg.innerHTML = "";
  const n = rows.length;
  const plotWidth = CHART_WIDTH - LEFT_MARGIN - RIGHT_MARGIN;
  const zeroBased = MODES[currentMode].axisMode === "zero";
  const maxAbs = zeroBased
    ? Math.max(...rows.map(r => r.value + r.std), 1e-9)
    : Math.max(...rows.map(r => Math.abs(r.value) + r.std), 1e-9);
  const height = TOP_MARGIN + n * ROW_H + BOTTOM_MARGIN;
  svg.setAttribute("viewBox", `0 0 ${{CHART_WIDTH}} ${{height}}`);
  svg.setAttribute("height", height);

  const baselineX = zeroBased ? LEFT_MARGIN : LEFT_MARGIN + plotWidth / 2;
  const scale = zeroBased
    ? (v => LEFT_MARGIN + (v / maxAbs) * plotWidth)
    : (v => baselineX + (v / maxAbs) * (plotWidth / 2));
  const ticks = zeroBased ? [0, maxAbs / 4, maxAbs / 2, (3 * maxAbs) / 4, maxAbs] : niceTicks(maxAbs);
  const plotBottom = TOP_MARGIN + n * ROW_H;

  for (const t of ticks) {{
    const x = scale(t);
    svg.appendChild(svgEl("line", {{
      class: "gridline", x1: x, x2: x, y1: TOP_MARGIN, y2: plotBottom,
    }}));
    const label = svgEl("text", {{
      class: "tick-label", x: x, y: plotBottom + 16, "text-anchor": "middle",
    }});
    label.textContent = formatValue(t, maxAbs);
    svg.appendChild(label);
  }}

  rows.forEach((r, i) => {{
    const y = TOP_MARGIN + i * ROW_H;
    const barY = y + (ROW_H - BAR_H) / 2;
    const endX = scale(r.value);
    const cls = barClassFor(r, i);

    const hit = svgEl("rect", {{
      class: "hit-row", x: 0, y: y, width: CHART_WIDTH, height: ROW_H,
    }});

    const bar = svgEl("path", {{ class: cls, d: roundedBarPath(baselineX, endX, barY, BAR_H, CORNER_R) }});

    const loX = scale(r.value - r.std);
    const hiX = scale(r.value + r.std);
    const cy = y + ROW_H / 2;
    const errGroup = svgEl("g", {{ class: "errorbar" }});
    errGroup.appendChild(svgEl("line", {{ x1: loX, x2: hiX, y1: cy, y2: cy }}));
    errGroup.appendChild(svgEl("line", {{ x1: loX, x2: loX, y1: cy - 3, y2: cy + 3 }}));
    errGroup.appendChild(svgEl("line", {{ x1: hiX, x2: hiX, y1: cy - 3, y2: cy + 3 }}));

    const label = svgEl("text", {{
      class: "row-label", x: LEFT_MARGIN - 10, y: cy + 4, "text-anchor": "end",
    }});
    label.textContent = r.name;

    hit.addEventListener("pointerenter", () => bar.classList.add("hovered"));
    hit.addEventListener("pointerleave", () => {{ bar.classList.remove("hovered"); hideTooltip(); }});
    hit.addEventListener("pointermove", ev => showTooltip(ev, r, maxAbs));

    // hit-row goes on top (last) so it wins hit-testing over the bar/label
    // beneath it -- pointerenter/leave don't bubble between siblings, so a
    // listener on a lower rect would never fire while the pointer is over
    // the (higher, unlistened) bar.
    svg.appendChild(bar);
    svg.appendChild(errGroup);
    svg.appendChild(label);
    svg.appendChild(hit);
  }});

  svg.appendChild(svgEl("line", {{
    class: "baseline", x1: baselineX, x2: baselineX, y1: TOP_MARGIN, y2: plotBottom,
  }}));
}}

function showTooltip(ev, r, maxAbs) {{
  const tt = document.getElementById("tooltip");
  tt.innerHTML = "";
  const name = document.createElement("div");
  name.className = "tt-name";
  name.textContent = r.name;
  const value = document.createElement("div");
  value.className = "tt-value";
  value.textContent = `${{MODES[currentMode].symbol}} = ${{formatValue(r.value, maxAbs)}} \\u00b1 ${{formatValue(r.std, maxAbs)}}`;
  tt.appendChild(name);
  tt.appendChild(value);
  tt.style.display = "block";
  tt.style.left = (ev.clientX + 14) + "px";
  tt.style.top = (ev.clientY + 14) + "px";
}}

function hideTooltip() {{
  document.getElementById("tooltip").style.display = "none";
}}

function renderTable(rows) {{
  const wrap = document.getElementById("table-wrap");
  wrap.innerHTML = "";
  const symbol = MODES[currentMode].symbol;
  const table = document.createElement("table");
  const thead = document.createElement("tr");
  for (const h of ["", "Model", `${{symbol}}_mean`, `${{symbol}}_std`]) {{
    const th = document.createElement("th");
    th.textContent = h;
    thead.appendChild(th);
  }}
  table.appendChild(thead);
  rows.forEach((r, i) => {{
    const tr = document.createElement("tr");
    const swatchCell = document.createElement("td");
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = `var(--${{barClassFor(r, i).replace("bar-", "")}})`;
    swatchCell.appendChild(swatch);
    const nameCell = document.createElement("td");
    nameCell.textContent = r.name;
    const meanCell = document.createElement("td");
    meanCell.textContent = r.value.toFixed(4);
    const stdCell = document.createElement("td");
    stdCell.textContent = r.std.toFixed(4);
    tr.appendChild(swatchCell);
    tr.appendChild(nameCell);
    tr.appendChild(meanCell);
    tr.appendChild(stdCell);
    table.appendChild(tr);
  }});
  wrap.appendChild(table);
}}

document.getElementById("view-toggle").addEventListener("click", () => {{
  showTable = !showTable;
  document.getElementById("view-toggle").textContent = showTable ? "View as chart" : "View as table";
  document.getElementById("chart-wrap").style.display = showTable ? "none" : "block";
  document.getElementById("table-wrap").style.display = showTable ? "block" : "none";
  renderCurrent();
}});

document.getElementById("mode-theta").addEventListener("click", () => switchMode("theta"));
document.getElementById("mode-specialization").addEventListener("click", () => switchMode("specialization"));
document.getElementById("mode-mae").addEventListener("click", () => switchMode("mae"));
if (DATA.has_mae) document.getElementById("mode-mae").style.display = "inline-block";

renderSidebar();
switchMode("theta");
</script>
</body>
</html>
"""


def write_dashboard_html(
    theta_df: pd.DataFrame, spec_df: pd.DataFrame, task_df: pd.DataFrame, output: str, mae_df: pd.DataFrame = None,
    display_names: dict = None,
) -> None:
    payload = build_dashboard_payload(theta_df, spec_df, task_df, mae_df, display_names)
    with open(output, "w") as f:
        f.write(render_dashboard_html(payload))
    print(f"Wrote {output}")
