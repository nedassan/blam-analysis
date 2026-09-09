import json

import pandas as pd

from blam_analysis.viz.common import (
    COLOR_ABOVE,
    COLOR_BASELINE,
    COLOR_BELOW,
    COLOR_GRID,
    COLOR_TEXT_MUTED,
    COLOR_TEXT_PRIMARY,
    SURFACE,
    resolve_display,
    task_short_label,
)


def _bar_color(row, alpha):
    if row["p_value"] >= alpha:
        return COLOR_TEXT_MUTED
    return COLOR_ABOVE if row["prob_beat_baseline"] >= 0.5 else COLOR_BELOW


def plot_comparison(df: pd.DataFrame, output: str, baseline: str, alpha: float, display_names: dict = None) -> None:
    import matplotlib.pyplot as plt

    display_names = display_names or {}
    colors = [_bar_color(row, alpha) for _, row in df.iterrows()]
    labels = [resolve_display(n, display_names) for n in df["model"]]
    baseline_label = resolve_display(baseline, display_names)

    fig_height = max(4, 0.22 * len(df))
    fig, ax = plt.subplots(figsize=(9, fig_height), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    y = range(len(df))
    ax.barh(y, df["prob_beat_baseline"], height=0.6, color=colors)

    ax.axvline(0.5, color=COLOR_BASELINE, linewidth=1, zorder=0)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=7, color=COLOR_TEXT_PRIMARY)
    ax.set_ylim(-0.6, len(df) - 0.4)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])

    ax.set_xlabel(f"P(model beats {baseline_label}) over BLAM posterior draws", color=COLOR_TEXT_MUTED, fontsize=9)
    ax.tick_params(axis="x", colors=COLOR_TEXT_MUTED, labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=COLOR_GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(COLOR_BASELINE)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_ABOVE, label=f"Significantly better (p < {alpha:g})"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_BELOW, label=f"Significantly worse (p < {alpha:g})"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_TEXT_MUTED, label=f"Not significant (p ≥ {alpha:g})"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=False, fontsize=8, labelcolor=COLOR_TEXT_MUTED)

    fig.tight_layout()
    fig.savefig(output, dpi=150, facecolor=SURFACE)
    print(f"Wrote {output}")


def _comparison_row_dicts(df: pd.DataFrame, display_names: dict) -> list:
    rows = df.sort_values("prob_beat_baseline", ascending=False)
    return [
        {"name": resolve_display(r.model, display_names), "value": float(r.prob_beat_baseline), "p_value": float(r.p_value)}
        for r in rows.itertuples(index=False)
    ]


def build_task_comparison_payload(
    comparison_df: pd.DataFrame, task_df: pd.DataFrame, baseline: str, alpha: float, display_names: dict = None,
    new_task_df: pd.DataFrame = None,
) -> dict:
    display_names = display_names or {}
    baseline_label = resolve_display(baseline, display_names)
    tasks = []
    if new_task_df is not None:
        tasks.append({
            "key": "__new_task__",
            "label": "New / unseen task",
            "mu_mean": None,
            "rows": _comparison_row_dicts(new_task_df, display_names),
            "synthetic": True,
        })
    for row in task_df.itertuples(index=False):
        rows = comparison_df[comparison_df["task"] == row.task]
        tasks.append({
            "key": row.task,
            "label": task_short_label(row.task),
            "mu_mean": float(row.mu_mean),
            "rows": _comparison_row_dicts(rows, display_names),
        })
    return {"baseline": baseline_label, "alpha": alpha, "tasks": tasks}


def render_task_comparison_html(payload: dict) -> str:
    data_json = json.dumps(payload).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>BLAM comparison per task</title>
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
  .layout {{ display: flex; min-height: 100vh; }}
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
  .task-btn.synthetic {{ font-style: italic; border-bottom: 1px solid var(--border); margin-bottom: 4px; padding-bottom: 9px; }}
  .task-metric {{ color: var(--text-muted); font-size: 11px; flex-shrink: 0; padding-left: 8px; }}
  main {{ flex: 1; padding: 24px 32px; min-width: 0; }}
  h1 {{ font-size: 16px; font-weight: 700; margin: 0 0 2px; }}
  .subtitle {{ color: var(--text-secondary); font-size: 12px; margin: 0 0 20px; max-width: 640px; line-height: 1.5; }}
  .toolbar {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }}
  .legend {{ display: flex; gap: 16px; font-size: 12px; color: var(--text-secondary); flex-wrap: wrap; }}
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
    <h1 id="chart-title"></h1>
    <p class="subtitle" id="chart-subtitle"></p>
    <div class="toolbar">
      <div class="legend">
        <span class="legend-item"><span class="swatch" style="background:var(--strong)"></span><span id="legend-strong"></span></span>
        <span class="legend-item"><span class="swatch" style="background:var(--weak)"></span><span id="legend-weak"></span></span>
        <span class="legend-item"><span class="swatch" style="background:var(--neutral)"></span><span id="legend-neutral"></span></span>
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
const DOMAIN_MIN = 0;
const DOMAIN_MAX = 1;
const BASELINE_VALUE = 0.5;

let currentKey = DATA.tasks[0].key;
let showTable = false;

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

function currentTask() {{
  return DATA.tasks.find(e => e.key === currentKey);
}}

function barClass(r) {{
  if (r.p_value >= DATA.alpha) return "bar-neutral";
  return r.value >= BASELINE_VALUE ? "bar-strong" : "bar-weak";
}}

function renderSidebar() {{
  const list = document.getElementById("task-list");
  list.innerHTML = "";
  for (const t of DATA.tasks) {{
    const btn = document.createElement("button");
    btn.className = "task-btn" + (t.key === currentKey ? " active" : "") + (t.synthetic ? " synthetic" : "");
    btn.dataset.key = t.key;
    const name = document.createElement("span");
    name.textContent = t.label;
    const metric = document.createElement("span");
    metric.className = "task-metric";
    metric.textContent = t.mu_mean === null ? "" : "\\u03bc " + t.mu_mean.toFixed(2);
    metric.title = "Task difficulty (mean log\\u2081\\u2080 MAE)";
    btn.appendChild(name);
    btn.appendChild(metric);
    btn.addEventListener("click", () => selectTask(t.key));
    list.appendChild(btn);
  }}
}}

function updateHeader() {{
  const t = currentTask();
  document.getElementById("chart-title").textContent = `P(model beats ${{DATA.baseline}}) \\u2013 ${{t.label}}`;
  document.getElementById("chart-subtitle").textContent = t.synthetic
    ? "This is the model's answer for a hypothetical task it has never seen. Significance uses a "
      + "two-sided p-value analog: p = 2 * min(P, 1-P), flagged "
      + `at alpha = ${{DATA.alpha}}.`
    : "This reflects what was actually observed on this task, not a hypothetical rerun. Significance "
      + "uses a two-sided p-value "
      + `analog: p = 2 * min(P, 1-P), flagged at alpha = ${{DATA.alpha}}.`;
  document.getElementById("legend-strong").textContent = `Significantly better (p < ${{DATA.alpha}})`;
  document.getElementById("legend-weak").textContent = `Significantly worse (p < ${{DATA.alpha}})`;
  document.getElementById("legend-neutral").textContent = `Not significant (p \\u2265 ${{DATA.alpha}})`;
}}

function renderCurrent() {{
  const rows = currentTask().rows;
  if (showTable) renderTable(rows); else renderChart(rows);
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
  const height = TOP_MARGIN + n * ROW_H + BOTTOM_MARGIN;
  svg.setAttribute("viewBox", `0 0 ${{CHART_WIDTH}} ${{height}}`);
  svg.setAttribute("height", height);

  const scale = v => LEFT_MARGIN + ((v - DOMAIN_MIN) / (DOMAIN_MAX - DOMAIN_MIN)) * plotWidth;
  const baselineX = scale(BASELINE_VALUE);
  const plotBottom = TOP_MARGIN + n * ROW_H;

  for (const t of [0, 0.25, 0.5, 0.75, 1.0]) {{
    const x = scale(t);
    svg.appendChild(svgEl("line", {{
      class: "gridline", x1: x, x2: x, y1: TOP_MARGIN, y2: plotBottom,
    }}));
    const label = svgEl("text", {{
      class: "tick-label", x: x, y: plotBottom + 16, "text-anchor": "middle",
    }});
    label.textContent = t.toFixed(2);
    svg.appendChild(label);
  }}

  rows.forEach((r, i) => {{
    const y = TOP_MARGIN + i * ROW_H;
    const barY = y + (ROW_H - BAR_H) / 2;
    const endX = scale(r.value);

    const hit = svgEl("rect", {{
      class: "hit-row", x: 0, y: y, width: CHART_WIDTH, height: ROW_H,
    }});

    const bar = svgEl("path", {{ class: barClass(r), d: roundedBarPath(baselineX, endX, barY, BAR_H, CORNER_R) }});

    const label = svgEl("text", {{
      class: "row-label", x: LEFT_MARGIN - 10, y: y + ROW_H / 2 + 4, "text-anchor": "end",
    }});
    label.textContent = r.name;

    hit.addEventListener("pointerenter", () => bar.classList.add("hovered"));
    hit.addEventListener("pointerleave", () => {{ bar.classList.remove("hovered"); hideTooltip(); }});
    hit.addEventListener("pointermove", ev => showTooltip(ev, r));

    // hit-row goes on top (last) so it wins hit-testing over the bar/label
    // beneath it -- pointerenter/leave don't bubble between siblings, so a
    // listener on a lower rect would never fire while the pointer is over
    // the (higher, unlistened) bar.
    svg.appendChild(bar);
    svg.appendChild(label);
    svg.appendChild(hit);
  }});

  svg.appendChild(svgEl("line", {{
    class: "baseline", x1: baselineX, x2: baselineX, y1: TOP_MARGIN, y2: plotBottom,
  }}));
}}

function showTooltip(ev, r) {{
  const tt = document.getElementById("tooltip");
  tt.innerHTML = "";
  const name = document.createElement("div");
  name.className = "tt-name";
  name.textContent = r.name;
  const value = document.createElement("div");
  value.className = "tt-value";
  value.textContent = `P(beats ${{DATA.baseline}}) = ${{r.value.toFixed(3)}}, p = ${{r.p_value.toFixed(3)}}`;
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
  const table = document.createElement("table");
  const thead = document.createElement("tr");
  for (const h of ["", "Model", `P(beats ${{DATA.baseline}})`, "p-value"]) {{
    const th = document.createElement("th");
    th.textContent = h;
    thead.appendChild(th);
  }}
  table.appendChild(thead);
  for (const r of rows) {{
    const tr = document.createElement("tr");
    const swatchCell = document.createElement("td");
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = `var(${{barClass(r) === "bar-strong" ? "--strong" : barClass(r) === "bar-weak" ? "--weak" : "--neutral"}})`;
    swatchCell.appendChild(swatch);
    const nameCell = document.createElement("td");
    nameCell.textContent = r.name;
    const valueCell = document.createElement("td");
    valueCell.textContent = r.value.toFixed(4);
    const pCell = document.createElement("td");
    pCell.textContent = r.p_value.toFixed(4);
    tr.appendChild(swatchCell);
    tr.appendChild(nameCell);
    tr.appendChild(valueCell);
    tr.appendChild(pCell);
    table.appendChild(tr);
  }}
  wrap.appendChild(table);
}}

document.getElementById("view-toggle").addEventListener("click", () => {{
  showTable = !showTable;
  document.getElementById("view-toggle").textContent = showTable ? "View as chart" : "View as table";
  document.getElementById("chart-wrap").style.display = showTable ? "none" : "block";
  document.getElementById("table-wrap").style.display = showTable ? "block" : "none";
  renderCurrent();
}});

renderSidebar();
updateHeader();
renderCurrent();
</script>
</body>
</html>
"""


def write_task_comparison_html(
    comparison_df: pd.DataFrame, task_df: pd.DataFrame, baseline: str, alpha: float, output: str,
    display_names: dict = None, new_task_df: pd.DataFrame = None,
) -> None:
    payload = build_task_comparison_payload(comparison_df, task_df, baseline, alpha, display_names, new_task_df)
    with open(output, "w") as f:
        f.write(render_task_comparison_html(payload))
    print(f"Wrote {output}")
