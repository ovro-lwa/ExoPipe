#!/usr/bin/env python3
"""Generate Phase 2 pipeline dashboard as a self-contained HTML page.

Scans Phase 1 and Phase 2 directories, reads QA metrics, and produces
a single HTML file with completion status, ionosphere quality, thermal
noise, and flux check results.

Usage:
    # Generate once:
    python generate_dashboard.py

    # Auto-refresh every 5 minutes:
    watch -n 300 python generate_dashboard.py

    # Serve to team:
    python -m http.server 8080 -d /fast/gh/main/phase2/
    # Then browse to http://lwacalim07:8080/phase2_dashboard.html

Output: {PHASE2_ROOT}/phase2_dashboard.html
"""
import os
import sys
import glob
import json
from datetime import datetime
from collections import defaultdict

from exopipe.common import phase2_config as cfg

PHASE1_ROOT = cfg.CELERY_IMAGE_ROOT
PHASE2_ROOT = cfg.PHASE2_WORKING_DIR
PHASE2_LUSTRE = '/lustre/pipeline/exopipe/phase2'
OUTPUT_HTML = os.path.join(PHASE2_ROOT, "phase2_dashboard.html")


def scan_all_data():
    """Scan Phase 1 and Phase 2, return structured data."""
    data = defaultdict(lambda: defaultdict(lambda: {
        'phase1': False, 'phase2': False, 'phase3': False, 'qa': {}
    }))

    # Phase 1
    for lst_dir in sorted(glob.glob(os.path.join(PHASE1_ROOT, "*h"))):
        lst = os.path.basename(lst_dir)
        for date_dir in sorted(glob.glob(os.path.join(lst_dir, "20*"))):
            date = os.path.basename(date_dir)
            if glob.glob(os.path.join(date_dir, "Run_*")):
                data[lst][date]['phase1'] = True

    # Phase 2 + QA (scan both NVMe and Lustre)
    for p2_root in [PHASE2_ROOT, PHASE2_LUSTRE]:
        if not os.path.isdir(p2_root):
            continue
        for lst_dir in sorted(glob.glob(os.path.join(p2_root, "*h"))):
            lst = os.path.basename(lst_dir)
            for date_dir in sorted(glob.glob(os.path.join(lst_dir, "20*"))):
                date = os.path.basename(date_dir)
                science_dirs = sorted(glob.glob(
                    os.path.join(date_dir, "Science_*")))
                if not science_dirs:
                    continue

                sd = science_dirs[-1]  # most recent run
                if os.path.isdir(os.path.join(sd, "Red")):
                    data[lst][date]['phase2'] = True
                    # Only read QA if we haven't already (prefer NVMe)
                    if not data[lst][date]['qa']:
                        data[lst][date]['qa'] = read_qa(sd)

                # Phase 3: photometry + transient search
                det_dir = os.path.join(sd, "detections")
                if os.path.isdir(det_dir) and glob.glob(
                        os.path.join(det_dir, "*.csv")):
                    data[lst][date]['phase3'] = True
                if glob.glob(os.path.join(sd, "samples", "*.csv")):
                    data[lst][date]['phase3'] = True

    return data


def read_qa(science_dir):
    """Read QA metrics from a completed Phase 2 run."""
    qa = {}

    # Ionosphere disturbance (from 73 MHz warp stats)
    for ws in sorted(glob.glob(os.path.join(
            science_dir, "Dewarp_Diagnostics", "*warp_stats_*73MHz*.json"))):
        try:
            with open(ws) as f:
                d = json.load(f)
            qa['disturbance_rms'] = d.get('disturbance_rms_arcmin')
            qa['disturbance_quality'] = d.get('disturbance_quality', '')
            qa['median_shift'] = d.get('median_shift_arcmin')
            qa['n_anchors'] = d.get('n_anchors')
        except Exception:
            pass

    # Ionosphere variability
    for vf in sorted(glob.glob(os.path.join(
            science_dir, "Dewarp_Diagnostics",
            "*ionosphere_variability_*.json"))):
        try:
            with open(vf) as f:
                d = json.load(f)
            qa['variability'] = d.get('mean_variability')
            qa['variability_quality'] = d.get('quality', '')
        except Exception:
            pass

    # Thermal noise (median across subbands, deep only)
    for nc in sorted(glob.glob(os.path.join(
            science_dir, "QA", "*thermal_noise.csv"))):
        try:
            import pandas as pd
            df = pd.read_csv(nc)
            deep = df[df.get('category', 'deep') == 'deep'] if 'category' in df.columns else df
            if not deep.empty:
                qa['noise_median_mJy'] = float(deep['rms'].median() * 1000)
                qa['noise_best_mJy'] = float(deep['rms'].min() * 1000)
        except Exception:
            pass

    # Flux check
    fc_pngs = glob.glob(os.path.join(science_dir, "QA", "*Flux_Check_*.png"))
    qa['flux_check'] = len(fc_pngs) > 0

    return qa


def quality_color(quality_str):
    """Map quality string to CSS color."""
    if not quality_str:
        return '#999'
    q = quality_str.lower()
    if 'excellent' in q or 'stable' in q:
        return '#2ecc71'
    elif 'good' in q or 'slow' in q:
        return '#27ae60'
    elif 'moderate' in q:
        return '#f39c12'
    elif 'poor' in q:
        return '#e74c3c'
    return '#999'


def generate_html(data):
    """Generate the dashboard HTML."""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Compute stats
    n_p1 = sum(1 for lst in data for date in data[lst]
               if data[lst][date]['phase1'])
    n_p2 = sum(1 for lst in data for date in data[lst]
               if data[lst][date]['phase2'])
    n_p3 = sum(1 for lst in data for date in data[lst]
               if data[lst][date]['phase3'])

    # Per-LST stats
    lst_stats = {}
    for lst in sorted(data.keys(), key=lambda x: int(x.replace('h', ''))):
        dates = data[lst]
        p1 = sum(1 for d in dates.values() if d['phase1'])
        p2 = sum(1 for d in dates.values() if d['phase2'])
        p3 = sum(1 for d in dates.values() if d['phase3'])
        lst_stats[lst] = {'p1': p1, 'p2': p2, 'p3': p3}

    # Build rows for detail table
    rows_html = ""
    for lst in sorted(data.keys(), key=lambda x: int(x.replace('h', ''))):
        for date in sorted(data[lst].keys()):
            d = data[lst][date]
            if not d['phase1']:
                continue
            qa = d.get('qa', {})

            p2_status = '✓' if d['phase2'] else '—'
            p2_class = 'done' if d['phase2'] else 'pending'
            p3_status = '✓' if d['phase3'] else '—'
            p3_class = 'done' if d['phase3'] else 'pending'

            dist = qa.get('disturbance_rms')
            dist_str = f"{dist:.2f}'" if dist else '—'
            dist_qual = qa.get('disturbance_quality', '')
            dist_color = quality_color(dist_qual)

            var = qa.get('variability')
            var_str = f"{var:.3f}" if var is not None else '—'
            var_qual = qa.get('variability_quality', '')
            var_color = quality_color(var_qual)

            noise = qa.get('noise_median_mJy')
            noise_str = f"{noise:.1f}" if noise else '—'

            flux = '✓' if qa.get('flux_check') else '—'

            rows_html += f"""<tr class="{p2_class}">
                <td>{lst}</td><td>{date}</td>
                <td class="{p2_class}">{p2_status}</td>
                <td class="{p3_class}">{p3_status}</td>
                <td style="color:{dist_color}" title="{dist_qual}">{dist_str}</td>
                <td style="color:{var_color}" title="{var_qual}">{var_str}</td>
                <td>{noise_str}</td>
                <td>{flux}</td>
            </tr>\n"""

    # LST bar chart data
    bar_html = ""
    for lst, s in lst_stats.items():
        p1_h = max(s['p1'] * 8, 2)
        p2_h = max(s['p2'] * 8, 0)
        p3_h = max(s['p3'] * 8, 0)
        bar_html += f"""<div class="bar-group">
            <div class="bar p1" style="height:{p1_h}px"
                 title="Phase 1: {s['p1']}"></div>
            <div class="bar p2" style="height:{p2_h}px"
                 title="Phase 2: {s['p2']}"></div>
            <div class="bar p3" style="height:{p3_h}px"
                 title="Phase 3: {s['p3']}"></div>
            <div class="bar-label">{lst.replace('h','')}</div>
        </div>\n"""

    pct = 100 * n_p2 / n_p1 if n_p1 > 0 else 0

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="300">
<title>ExoPipe Phase 2 Dashboard</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: -apple-system, 'Segoe UI', Roboto, sans-serif;
            background: #0a0e17; color: #e0e0e0; padding: 20px; }}
    h1 {{ color: #fff; margin-bottom: 5px; }}
    .subtitle {{ color: #888; margin-bottom: 20px; }}
    .cards {{ display: flex; gap: 15px; margin-bottom: 20px; flex-wrap: wrap; }}
    .card {{ background: #161b2e; border-radius: 8px; padding: 15px 20px;
             min-width: 150px; }}
    .card .label {{ color: #888; font-size: 12px; text-transform: uppercase; }}
    .card .value {{ font-size: 28px; font-weight: bold; color: #fff; }}
    .card .value.green {{ color: #2ecc71; }}
    .progress-bar {{ background: #1a1f33; border-radius: 4px; height: 24px;
                      margin-bottom: 20px; overflow: hidden; }}
    .progress-fill {{ background: linear-gradient(90deg, #2ecc71, #27ae60);
                       height: 100%; border-radius: 4px;
                       transition: width 0.5s; display: flex; align-items: center;
                       justify-content: center; font-size: 12px; font-weight: bold; }}
    .chart {{ display: flex; align-items: flex-end; gap: 3px; height: 260px;
              background: #161b2e; border-radius: 8px; padding: 15px;
              margin-bottom: 20px; }}
    .bar-group {{ display: flex; flex-direction: column; align-items: center;
                   flex: 1; }}
    .bar {{ width: 100%; border-radius: 2px 2px 0 0; min-width: 8px; }}
    .bar.p1 {{ background: #2c3e6b; }}
    .bar.p2 {{ background: #2ecc71; }}
    .bar.p3 {{ background: #9b59b6; }}
    .bar-label {{ font-size: 10px; color: #888; margin-top: 4px; }}
    .legend {{ display: flex; gap: 15px; margin-bottom: 10px; font-size: 12px; }}
    .legend span {{ display: flex; align-items: center; gap: 5px; }}
    .legend .dot {{ width: 10px; height: 10px; border-radius: 2px; }}
    table {{ width: 100%; border-collapse: collapse; background: #161b2e;
             border-radius: 8px; overflow: hidden; }}
    th {{ background: #1a2240; padding: 10px; text-align: left; font-size: 12px;
          text-transform: uppercase; color: #888; position: sticky; top: 0; }}
    td {{ padding: 8px 10px; border-top: 1px solid #1a1f33; font-size: 13px; }}
    tr.done td:nth-child(3) {{ color: #2ecc71; font-weight: bold; }}
    tr.pending td:nth-child(3) {{ color: #555; }}
    tr:hover {{ background: #1a2240; }}
    .section-title {{ color: #fff; margin: 20px 0 10px; font-size: 16px; }}
</style>
</head><body>
<h1>OVRO-LWA ExoPipe Phase 2</h1>
<p class="subtitle">Updated {now} · auto-refreshes every 5 min</p>

<div class="cards">
    <div class="card">
        <div class="label">Phase 1</div>
        <div class="value">{n_p1}</div>
    </div>
    <div class="card">
        <div class="label">Phase 2 Complete</div>
        <div class="value green">{n_p2}</div>
    </div>
    <div class="card">
        <div class="label">Phase 3 Complete</div>
        <div class="value" style="color:#9b59b6">{n_p3}</div>
    </div>
    <div class="card">
        <div class="label">Remaining</div>
        <div class="value">{n_p1 - n_p2}</div>
    </div>
    <div class="card">
        <div class="label">Est. Time (4 parallel)</div>
        <div class="value">{(n_p1 - n_p2) * 31.5 / 4 / 60:.1f}h</div>
    </div>
</div>

<div class="progress-bar">
    <div class="progress-fill" style="width:{pct:.1f}%">{pct:.1f}%</div>
</div>

<div class="section-title">Hours by LST</div>
<div class="legend">
    <span><div class="dot" style="background:#2c3e6b"></div>Phase 1</span>
    <span><div class="dot" style="background:#2ecc71"></div>Phase 2</span>
    <span><div class="dot" style="background:#9b59b6"></div>Phase 3</span>
</div>
<div class="chart">
{bar_html}
</div>

<div class="section-title">Detail</div>
<div style="max-height:600px; overflow-y:auto; border-radius:8px;">
<table>
<thead><tr>
    <th>LST</th><th>Date</th><th>P2</th><th>P3</th>
    <th>Disturbance</th><th>Variability</th>
    <th>Noise (mJy)</th><th>Flux</th>
</tr></thead>
<tbody>
{rows_html}
</tbody>
</table>
</div>

</body></html>"""
    return html


def main():
    data = scan_all_data()
    html = generate_html(data)
    with open(OUTPUT_HTML, 'w') as f:
        f.write(html)
    print(f"Dashboard written to {OUTPUT_HTML}", file=sys.stderr)


if __name__ == "__main__":
    main()
