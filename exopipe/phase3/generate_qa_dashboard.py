#!/usr/bin/env python3
"""Generate interactive QA comparison dashboard.

Scans all Phase 2 QA data and produces a self-contained HTML page with:
  1. Thermal noise vs frequency (overlay multiple nights, group by LST)
  2. Flux scale check (measured/model ratio across nights)
  3. 3-color PNG gallery
  4. Ionosphere quality timeline (disturbance + variability)

Filterable by LST, date, and quality. Uses Chart.js from CDN.

Usage:
    python generate_qa_dashboard.py
    # Serve: python -m http.server 8888 -d /fast/gh/main/phase2/
    # Browse: http://lwacalim07:8888/qa_dashboard.html
"""
import os, sys, glob, json
from collections import defaultdict
from datetime import datetime
from exopipe.common import phase2_config as cfg

PHASE2_ROOTS = [cfg.PHASE2_WORKING_DIR, '/lustre/pipeline/exopipe/phase2']
OUTPUT = os.path.join(cfg.PHASE2_WORKING_DIR, "qa_dashboard.html")


def scan_qa():
    """Scan all Phase 2 runs and extract QA data."""
    runs = []
    seen = set()

    for root in PHASE2_ROOTS:
        if not os.path.isdir(root):
            continue
        for lst_dir in sorted(glob.glob(os.path.join(root, "*h"))):
            lst = os.path.basename(lst_dir)
            for date_dir in sorted(glob.glob(os.path.join(lst_dir, "20*"))):
                date = os.path.basename(date_dir)
                key = (lst, date)
                if key in seen:
                    continue

                for sd in sorted(glob.glob(os.path.join(date_dir, "Science_*")))[::-1]:
                    if not os.path.isdir(os.path.join(sd, "Red")):
                        continue
                    seen.add(key)
                    run = {'lst': lst, 'date': date, 'path': sd,
                           'noise': [], 'flux': [], 'iono': {},
                           'three_color': None}

                    # Thermal noise
                    for nc in sorted(glob.glob(os.path.join(sd, "QA", "*thermal_noise.csv"))):
                        try:
                            import pandas as pd
                            df = pd.read_csv(nc)
                            deep = df[df['category'] == 'deep'] if 'category' in df.columns else df
                            for _, row in deep.iterrows():
                                run['noise'].append({
                                    'freq': float(row['freq']),
                                    'rms': float(row['rms'])
                                })
                        except Exception:
                            pass

                    # Flux check
                    for fc in sorted(glob.glob(os.path.join(sd, "QA", "*flux_check_hybrid.csv"))):
                        try:
                            import pandas as pd
                            df = pd.read_csv(fc)
                            for _, row in df.iterrows():
                                if row.get('imstat_flux') and row.get('model_flux'):
                                    ratio = row['imstat_flux'] / row['model_flux']
                                    if 0 < ratio < 10:
                                        run['flux'].append({
                                            'source': row.get('source', ''),
                                            'freq': float(row['freq']),
                                            'ratio': float(ratio),
                                            'model': float(row['model_flux']),
                                            'measured': float(row['imstat_flux']),
                                        })
                        except Exception:
                            pass

                    # Ionosphere
                    for ws in sorted(glob.glob(os.path.join(
                            sd, "Dewarp_Diagnostics", "*warp_stats_*73MHz*.json"))):
                        try:
                            with open(ws) as f:
                                d = json.load(f)
                            run['iono']['disturbance'] = d.get('disturbance_rms_arcmin')
                            run['iono']['dist_quality'] = d.get('disturbance_quality', '')
                            run['iono']['median_shift'] = d.get('median_shift_arcmin')
                        except Exception:
                            pass

                    for vf in sorted(glob.glob(os.path.join(
                            sd, "Dewarp_Diagnostics", "*ionosphere_variability_*.json"))):
                        try:
                            with open(vf) as f:
                                d = json.load(f)
                            run['iono']['variability'] = d.get('mean_variability')
                            run['iono']['var_quality'] = d.get('quality', '')
                        except Exception:
                            pass

                    # 3-color PNG (relative path for serving)
                    pngs = sorted(glob.glob(os.path.join(sd, "QA", "*3color*.png")))
                    if pngs:
                        # Make path relative to PHASE2_ROOT for web serving
                        run['three_color'] = os.path.relpath(
                            pngs[0], cfg.PHASE2_WORKING_DIR)

                    runs.append(run)
                    break  # only most recent Science_ dir

    return runs


def generate_html(runs):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # Clean runs for JSON (remove path)
    data = []
    for r in runs:
        d = {k: v for k, v in r.items() if k != 'path'}
        data.append(d)

    data_json = json.dumps(data, indent=None)

    return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="300">
<title>ExoPipe QA Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,'Segoe UI',Roboto,sans-serif; background:#0a0e17; color:#e0e0e0; }}
.header {{ padding:20px 24px 0; }}
h1 {{ color:#fff; font-size:22px; }}
.subtitle {{ color:#666; font-size:12px; margin-top:4px; }}
.tabs {{ display:flex; gap:0; margin:16px 24px 0; border-bottom:1px solid #1a2240; }}
.tab {{ padding:10px 20px; cursor:pointer; color:#888; font-size:13px; font-weight:600;
        border-bottom:2px solid transparent; transition:all 0.2s; }}
.tab:hover {{ color:#ccc; }}
.tab.active {{ color:#fff; border-bottom-color:#2ecc71; }}
.controls {{ display:flex; gap:12px; padding:12px 24px; align-items:center; flex-wrap:wrap; }}
.controls label {{ color:#888; font-size:12px; }}
.controls select {{ background:#161b2e; color:#e0e0e0; border:1px solid #2a3050;
                     border-radius:4px; padding:5px 8px; font-size:12px; }}
.panel {{ display:none; padding:16px 24px; }}
.panel.active {{ display:block; }}
.chart-container {{ background:#161b2e; border-radius:8px; padding:16px; margin-bottom:16px; }}
canvas {{ max-height:500px; }}
.gallery {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); gap:12px; }}
.gallery-item {{ background:#161b2e; border-radius:8px; overflow:hidden; }}
.gallery-item img {{ width:100%; display:block; }}
.gallery-item .caption {{ padding:8px 12px; font-size:12px; color:#aaa; }}
.stats-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(220px,1fr)); gap:12px; }}
.stat-card {{ background:#161b2e; border-radius:8px; padding:14px; }}
.stat-card .label {{ font-size:11px; color:#888; text-transform:uppercase; }}
.stat-card .value {{ font-size:20px; font-weight:bold; margin-top:4px; }}
.stat-card .sub {{ font-size:11px; color:#666; margin-top:2px; }}
.good {{ color:#2ecc71; }} .moderate {{ color:#f39c12; }} .poor {{ color:#e74c3c; }} .neutral {{ color:#aaa; }}
table {{ width:100%; border-collapse:collapse; background:#161b2e; border-radius:8px; overflow:hidden; font-size:13px; }}
th {{ background:#1a2240; padding:8px 10px; text-align:left; color:#888; font-size:11px; text-transform:uppercase; }}
td {{ padding:7px 10px; border-top:1px solid #111827; }}
tr:hover {{ background:#1a2240; }}
</style>
</head><body>

<div class="header">
<h1>OVRO-LWA ExoPipe — QA Dashboard</h1>
<p class="subtitle">Updated {now} · {len(runs)} runs · auto-refreshes every 5 min</p>
</div>

<div class="tabs">
<div class="tab active" onclick="showTab('noise')">Thermal Noise</div>
<div class="tab" onclick="showTab('flux')">Flux Scale</div>
<div class="tab" onclick="showTab('gallery')">3-Color Gallery</div>
<div class="tab" onclick="showTab('iono')">Ionosphere</div>
<div class="tab" onclick="showTab('table')">Summary Table</div>
</div>

<div class="controls">
<label>LST:</label>
<select id="lstFilter" onchange="updateAll()">
<option value="all">All LSTs</option>
</select>
<label>Sort:</label>
<select id="sortBy" onchange="updateAll()">
<option value="date">By Date</option>
<option value="lst">By LST</option>
<option value="noise">By Noise (best first)</option>
<option value="disturbance">By Disturbance (best first)</option>
</select>
</div>

<div class="panel active" id="panel-noise">
<div class="chart-container"><canvas id="noiseChart"></canvas></div>
</div>

<div class="panel" id="panel-flux">
<div class="chart-container"><canvas id="fluxChart"></canvas></div>
</div>

<div class="panel" id="panel-gallery">
<div class="gallery" id="galleryGrid"></div>
</div>

<div class="panel" id="panel-iono">
<div class="stats-grid" id="ionoStats"></div>
<div class="chart-container" style="margin-top:16px"><canvas id="ionoChart"></canvas></div>
</div>

<div class="panel" id="panel-table">
<table id="summaryTable">
<thead><tr><th>LST</th><th>Date</th><th>Noise (mJy)</th><th>Disturbance</th><th>Variability</th><th>Flux Check</th></tr></thead>
<tbody></tbody>
</table>
</div>

<script>
const DATA = {data_json};

const COLORS = [
'#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6','#1abc9c',
'#e67e22','#2980b9','#27ae60','#8e44ad','#d35400','#16a085',
'#c0392b','#2c3e50','#f1c40f','#7f8c8d','#e91e63','#00bcd4',
'#ff5722','#607d8b','#4caf50','#ff9800','#795548','#673ab7',
'#03a9f4','#8bc34a','#ffc107','#9e9e9e','#cddc39','#00e5ff'
];

let noiseChart, fluxChart, ionoChart;

// Populate LST filter
const lsts = [...new Set(DATA.map(d=>d.lst))].sort((a,b)=>parseInt(a)-parseInt(b));
const lstSel = document.getElementById('lstFilter');
lsts.forEach(l => {{
    const o = document.createElement('option');
    o.value = l; o.textContent = l;
    lstSel.appendChild(o);
}});

function getFiltered() {{
    let f = DATA.slice();
    const lst = document.getElementById('lstFilter').value;
    if (lst !== 'all') f = f.filter(d => d.lst === lst);
    const sort = document.getElementById('sortBy').value;
    if (sort === 'date') f.sort((a,b) => a.date.localeCompare(b.date));
    else if (sort === 'lst') f.sort((a,b) => parseInt(a.lst) - parseInt(b.lst) || a.date.localeCompare(b.date));
    else if (sort === 'noise') f.sort((a,b) => {{
        const na = a.noise.length ? a.noise.reduce((s,n)=>s+n.rms,0)/a.noise.length : 999;
        const nb = b.noise.length ? b.noise.reduce((s,n)=>s+n.rms,0)/b.noise.length : 999;
        return na - nb;
    }});
    else if (sort === 'disturbance') f.sort((a,b) => (a.iono.disturbance||999) - (b.iono.disturbance||999));
    return f;
}}

function updateNoise() {{
    const runs = getFiltered();
    if (noiseChart) noiseChart.destroy();
    const datasets = runs.filter(r=>r.noise.length>0).map((r,i) => ({{
        label: `${{r.date}} (${{r.lst}})`,
        data: r.noise.sort((a,b)=>a.freq-b.freq).map(n=>({{x:n.freq, y:n.rms}})),
        borderColor: COLORS[i % COLORS.length],
        backgroundColor: 'transparent',
        pointRadius: 3, borderWidth: 1.5, tension: 0.3
    }}));
    noiseChart = new Chart(document.getElementById('noiseChart'), {{
        type: 'scatter',
        data: {{ datasets }},
        options: {{
            responsive: true, showLine: true,
            scales: {{
                x: {{ title: {{ display:true, text:'Frequency (MHz)', color:'#888' }},
                       grid: {{ color:'#1a2240' }}, ticks: {{ color:'#888' }} }},
                y: {{ title: {{ display:true, text:'RMS (Jy/beam)', color:'#888' }},
                       grid: {{ color:'#1a2240' }}, ticks: {{ color:'#888' }},
                       type: 'logarithmic' }}
            }},
            plugins: {{
                legend: {{ position:'right', labels: {{ color:'#ccc', font: {{ size:11 }}, boxWidth:12 }} }}
            }}
        }}
    }});
}}

function updateFlux() {{
    const runs = getFiltered();
    if (fluxChart) fluxChart.destroy();
    const datasets = runs.filter(r=>r.flux.length>0).map((r,i) => ({{
        label: `${{r.date}} (${{r.lst}})`,
        data: r.flux.map(f=>({{x:f.freq, y:f.ratio}})),
        borderColor: COLORS[i % COLORS.length],
        backgroundColor: COLORS[i % COLORS.length] + '40',
        pointRadius: 4, borderWidth: 0, pointStyle: 'circle'
    }}));
    // Add unity line
    datasets.unshift({{
        label: 'Unity', data: [{{x:15,y:1}},{{x:90,y:1}}],
        borderColor: '#ffffff40', borderWidth: 1, borderDash: [5,5],
        pointRadius: 0, showLine: true
    }});
    fluxChart = new Chart(document.getElementById('fluxChart'), {{
        type: 'scatter',
        data: {{ datasets }},
        options: {{
            responsive: true,
            scales: {{
                x: {{ title: {{ display:true, text:'Frequency (MHz)', color:'#888' }},
                       grid: {{ color:'#1a2240' }}, ticks: {{ color:'#888' }} }},
                y: {{ title: {{ display:true, text:'Measured / Model', color:'#888' }},
                       grid: {{ color:'#1a2240' }}, ticks: {{ color:'#888' }},
                       min: 0, max: 2 }}
            }},
            plugins: {{
                legend: {{ position:'right', labels: {{ color:'#ccc', font: {{ size:11 }}, boxWidth:12 }} }}
            }}
        }}
    }});
}}

function updateGallery() {{
    const runs = getFiltered();
    const grid = document.getElementById('galleryGrid');
    grid.innerHTML = '';
    runs.filter(r=>r.three_color).forEach(r => {{
        grid.innerHTML += `<div class="gallery-item">
            <img src="${{r.three_color}}" loading="lazy"
                 onerror="this.parentElement.style.display='none'">
            <div class="caption">${{r.lst}} · ${{r.date}}</div>
        </div>`;
    }});
    if (!grid.innerHTML) grid.innerHTML = '<p style="padding:20px;color:#666">No 3-color images found</p>';
}}

function qualClass(q) {{
    if (!q) return 'neutral';
    q = q.toLowerCase();
    if (q.includes('excellent') || q.includes('good')) return 'good';
    if (q.includes('moderate')) return 'moderate';
    if (q.includes('poor')) return 'poor';
    return 'neutral';
}}

function updateIono() {{
    const runs = getFiltered().filter(r => r.iono.disturbance != null);
    if (ionoChart) ionoChart.destroy();

    const labels = runs.map(r => `${{r.date}}\\n${{r.lst}}`);
    ionoChart = new Chart(document.getElementById('ionoChart'), {{
        type: 'bar',
        data: {{
            labels,
            datasets: [
                {{ label: 'Disturbance (arcmin)', data: runs.map(r=>r.iono.disturbance||0),
                   backgroundColor: runs.map(r => {{
                       const d = r.iono.disturbance||0;
                       return d < 0.1 ? '#2ecc71' : d < 0.2 ? '#27ae60' : d < 0.4 ? '#f39c12' : '#e74c3c';
                   }}), yAxisID: 'y' }},
                {{ label: 'Variability', data: runs.map(r=>r.iono.variability||0),
                   type: 'line', borderColor: '#9b59b6', backgroundColor: 'transparent',
                   pointRadius: 4, borderWidth: 2, yAxisID: 'y1' }}
            ]
        }},
        options: {{
            responsive: true,
            scales: {{
                x: {{ ticks: {{ color:'#888', font:{{ size:10 }} }}, grid: {{ color:'#1a2240' }} }},
                y: {{ position:'left', title: {{ display:true, text:"Disturbance (arcmin)", color:'#888' }},
                      grid: {{ color:'#1a2240' }}, ticks: {{ color:'#888' }} }},
                y1: {{ position:'right', title: {{ display:true, text:"Variability", color:'#9b59b6' }},
                       grid: {{ drawOnChartArea:false }}, ticks: {{ color:'#9b59b6' }},
                       min: 0 }}
            }},
            plugins: {{ legend: {{ labels: {{ color:'#ccc' }} }} }}
        }}
    }});
}}

function updateTable() {{
    const runs = getFiltered();
    const tbody = document.querySelector('#summaryTable tbody');
    tbody.innerHTML = '';
    runs.forEach(r => {{
        const noiseMed = r.noise.length ? (r.noise.reduce((s,n)=>s+n.rms,0)/r.noise.length*1000).toFixed(1) : '—';
        const dist = r.iono.disturbance != null ? r.iono.disturbance.toFixed(2)+"'" : '—';
        const distQ = qualClass(r.iono.dist_quality);
        const varV = r.iono.variability != null ? r.iono.variability.toFixed(3) : '—';
        const varQ = qualClass(r.iono.var_quality);
        const flux = r.flux.length > 0 ? '✓ ('+r.flux.length+')' : '—';
        tbody.innerHTML += `<tr>
            <td>${{r.lst}}</td><td>${{r.date}}</td>
            <td>${{noiseMed}}</td>
            <td class="${{distQ}}">${{dist}}</td>
            <td class="${{varQ}}">${{varV}}</td>
            <td>${{flux}}</td>
        </tr>`;
    }});
}}

function showTab(name) {{
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    document.getElementById('panel-'+name).classList.add('active');
    event.target.classList.add('active');
    updateAll();
}}

function updateAll() {{
    updateNoise(); updateFlux(); updateGallery(); updateIono(); updateTable();
}}

updateAll();
</script>
</body></html>"""


if __name__ == "__main__":
    runs = scan_qa()
    html = generate_html(runs)
    with open(OUTPUT, 'w') as f:
        f.write(html)
    print(f"QA dashboard: {OUTPUT} ({len(runs)} runs)", file=sys.stderr)
