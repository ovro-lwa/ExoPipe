#!/usr/bin/env python3
"""Build a flare-star / active-binary / ultracool-dwarf cross-match catalog for the
Stokes V transient search. FP-conscious (the transient xmatch already covers ~1/1000
of the sky), so the catalog is kept TIGHT: UCDs distance-limited, no faint distant
brown-dwarf bulk, and distances filled from Gaia DR3 at a tight match radius.

  GCVS   B/gcvs/gcvs_cat  VarType UV/UVN/BY/RS/FKCOM  (the magnetically-active coherent
                          emitters: UV Cet flares, BY Dra spotted dwarfs, RS CVn binaries,
                          FK Com rapid rotators)  -- a deliberate subset of GCVS.
  Reyle 2018  J/A+A/619/L8  Gaia DR2 ultracool dwarfs, CAPPED at <50 pc (matches the
                          10/25/50pc catalogs; distant UCDs would be implausibly luminous
                          radio sources and just add false positives).
  SIMBAD basic.otype in Fl*/BY*/RS*/Er*  (named active stars; BD* brown dwarfs dropped --
                          distant/faint, FP-prone, and nearby UCDs already come from Reyle).

Distances: Reyle from Gaia DR2 Plx; GCVS/SIMBAD backfilled by Gaia DR3 cross-match (5").
Output: clv15 catalog format (common_name,ra_current,dec_current,coords_sexagesimal,distance).
"""
import urllib.request, urllib.parse, io, csv
import numpy as np
from astropy.coordinates import SkyCoord, search_around_sky
import astropy.units as u

VIZ = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"
SIM = "https://simbad.cds.unistra.fr/simbad/sim-tap/sync"
# ASU mirrors, tried in order (the Strasbourg database has outages)
ASU_MIRRORS = ["https://vizier.cds.unistra.fr/viz-bin/asu-tsv",
               "https://vizier.cfa.harvard.edu/viz-bin/asu-tsv"]
XM  = "http://cdsxmatch.u-strasbg.fr/xmatch/api/v1/sync"
OUT = "/lustre/gh/main/clv15/flare_ucd.csv"
# UltracoolSheet v2.1.0 (Best et al.), pinned Zenodo record 15802304,
# staged locally next to the other clv15 input catalogs
UCS_LOCAL = "/lustre/gh/main/clv15/ultracoolsheet_main_v2.1.0.csv"
UCS_URL = ("https://zenodo.org/api/records/15802304/files/"
           "UltracoolSheet%20-%20Main.csv/content")
UCD_MAX_PC = 50.0
GAIA_RADIUS_ARCSEC = 5.0


def _get(url):
    return urllib.request.urlopen(url, timeout=120).read().decode('utf-8', 'replace')


def asu(query_suffix):
    err = None
    for base in ASU_MIRRORS:
        try:
            txt = _get(base + query_suffix)
            head = txt[:3000]
            if ('QUERY_STATUS=ERROR' in head or 'Error=' in head
                    or 'not currently reachable' in head):
                err = f'{base}: server-side error page'
                continue
            return txt
        except Exception as e:
            err = e
    raise RuntimeError(f'all ASU mirrors failed: {err}')


def tap(base, adql):
    return _get(base + "?" + urllib.parse.urlencode(
        {"request": "doQuery", "lang": "ADQL", "format": "csv", "query": adql}))


def parse_csv(txt):
    return list(csv.DictReader(io.StringIO(txt)))


def gaia_backfill(missing):
    """missing: list of (key, ra, dec). Returns {key: distance_pc} via Gaia DR3 (tight)."""
    if not missing:
        return {}
    buf = io.StringIO(); w = csv.writer(buf); w.writerow(['id', 'RA', 'Dec'])
    for k, ra, dec in missing:
        w.writerow([k, f'{ra:.6f}', f'{dec:.6f}'])
    sample = buf.getvalue()
    b = '----xmb'
    def fld(n, v): return f'--{b}\r\nContent-Disposition: form-data; name="{n}"\r\n\r\n{v}\r\n'
    body = ''.join(fld(n, v) for n, v in [
        ('request', 'xmatch'), ('distMaxArcsec', str(GAIA_RADIUS_ARCSEC)),
        ('RESPONSEFORMAT', 'csv'), ('cat2', 'vizier:I/355/gaiadr3'),
        ('colRA1', 'RA'), ('colDec1', 'Dec')])
    body += (f'--{b}\r\nContent-Disposition: form-data; name="cat1"; filename="c.csv"\r\n'
             f'Content-Type: text/csv\r\n\r\n{sample}\r\n--{b}--\r\n')
    req = urllib.request.Request(XM, data=body.encode(),
                                 headers={'Content-Type': f'multipart/form-data; boundary={b}'})
    out = urllib.request.urlopen(req, timeout=300).read().decode('utf-8', 'replace')
    best = {}   # key -> (angDist, distance_pc)  keep nearest valid-Plx match
    for r in parse_csv(out):
        try:
            k = r['id']; ad = float(r['angDist']); plx = float(r['Plx']) if r.get('Plx', '').strip() else 0.0
            if plx > 0 and (k not in best or ad < best[k][0]):
                best[k] = (ad, 1000.0/plx)
        except Exception:
            pass
    return {k: f'{v[1]:.1f}' for k, v in best.items()}


rows = []   # [name, ra, dec, distance_str]

# --- GCVS active subset ---
print("GCVS...", flush=True)
adql = ("SELECT GCVS,RAJ2000,DEJ2000,VarType FROM \"B/gcvs/gcvs_cat\" WHERE RAJ2000 IS NOT NULL "
        "AND (VarType LIKE '%UV%' OR VarType LIKE '%BY%' OR VarType LIKE '%RS%' OR VarType LIKE '%FKCOM%')")
for r in parse_csv(tap(VIZ, adql)):
    try:
        rows.append([f"{r['GCVS'].strip()} [GCVS:{r['VarType'].strip()}]",
                     float(r['RAJ2000']), float(r['DEJ2000']), ''])
    except Exception:
        pass
print(f"  GCVS: {len(rows)}", flush=True)

# --- Reyle UCDs, distance-limited ---
print("Reyle UCDs (<%g pc)..." % UCD_MAX_PC, flush=True)
n0 = len(rows)
txt = asu("?-source=J/A%2BA/619/L8&-out.max=999999&-out=Source,RAICRS,DEICRS,Plx,sptphot")
for ln in txt.splitlines():
    if not ln or ln.startswith('#') or ln.startswith('-') or 'RAICRS' in ln or ln.startswith(' \t'):
        continue
    p = ln.split('\t')
    if len(p) < 4:
        continue
    try:
        src = p[0].strip(); ra = float(p[1]); dec = float(p[2])
        plx = float(p[3]) if p[3].strip() else 0.0
        if plx <= 0:
            continue
        dist = 1000.0/plx
        if dist >= UCD_MAX_PC:
            continue
        spt = p[4].strip() if len(p) > 4 else ''
        rows.append([f"Gaia DR2 {src} [UCD:{spt}]", ra, dec, f'{dist:.1f}'])
    except Exception:
        pass
print(f"  Reyle <{UCD_MAX_PC:g}pc: {len(rows)-n0}", flush=True)
if len(rows) - n0 < 1000:
    raise SystemExit(f'REFUSING: Reyle returned {len(rows)-n0} rows '
                     f'(<50pc reference count is ~1376; 2026-08-23 '
                     f'regression caught a truncated mirror response)')

# --- Kirkpatrick 2021 full-sky 20-pc L/T/Y census (added 2026-08-23:
# --- Gaia-selected Reyle contains no T dwarfs; Gregg-directed after the
# --- 2MASS J06254230+5646268 T6 near-miss) ---
def _spt_decode(v):
    # Kirkpatrick numeric code: <0 late-M (e.g. -2 = M8), 0-9.5 L, 10-19.5 T,
    # >=20 Y
    if v < 0:
        return f'M{10 + v:g}'
    if v < 10:
        return f'L{v:g}'
    if v < 20:
        return f'T{v - 10:g}'
    return f'Y{v - 20:g}'


print("Kirkpatrick 2021 20-pc census...", flush=True)
n0 = len(rows)
txt = asu("?-source=J/ApJS/253/7/tablea1&-out.max=999999"
          "&-out=FName,RACdeg,DECdeg,_RA,_DE,SpAd,plx")
for ln in txt.splitlines():
    if (not ln or ln.startswith('#') or ln.startswith('-')
            or 'FName' in ln or ln.startswith(' ') or ln.startswith('deg')):
        continue
    p = ln.split('\t')
    if len(p) < 7:
        continue
    try:
        name = p[0].strip()
        plx = float(p[6]) if p[6].strip() else 0.0
        if plx <= 0:
            continue
        dist = 1000.0 / plx
        if dist >= UCD_MAX_PC:
            continue
        ra = float(p[1]) if p[1].strip() else float(p[3])
        dec = float(p[2]) if p[2].strip() else float(p[4])
        spt = _spt_decode(float(p[5])) if p[5].strip() else ''
        rows.append([f"{name} [UCD:{spt}]", ra, dec, f'{dist:.1f}'])
    except Exception:
        pass
print(f"  Kirkpatrick <{UCD_MAX_PC:g}pc: {len(rows)-n0}", flush=True)

# --- UltracoolSheet v2.1.0 (pinned Zenodo), distance-limited ---
print("UltracoolSheet...", flush=True)
n0 = len(rows)
import os
if os.path.exists(UCS_LOCAL):
    ucs_txt = open(UCS_LOCAL, encoding='utf-8', errors='replace').read()
else:
    ucs_txt = _get(UCS_URL)
    open(UCS_LOCAL, 'w').write(ucs_txt)
for r in csv.DictReader(io.StringIO(ucs_txt)):
    try:
        plx = 0.0
        for col in ('plx_formula', 'plx_lit', 'plx_Gaia'):
            v = (r.get(col) or '').strip()
            if v and v.lower() != 'nan':
                plx = float(v)
                break
        if plx <= 0:
            continue
        dist = 1000.0 / plx
        if dist >= UCD_MAX_PC:
            continue
        spt = next((r[c].strip() for c in ('spt_ir', 'spt_opt')
                    if (r.get(c) or '').strip().lower() not in ('', 'null',
                                                                'nan')), '')
        rows.append([f"{r['name'].strip()} [UCS:{spt}]",
                     float(r['ra_j2000_formula']),
                     float(r['dec_j2000_formula']), f'{dist:.1f}'])
    except Exception:
        pass
print(f"  UltracoolSheet <{UCD_MAX_PC:g}pc: {len(rows)-n0}", flush=True)

# --- SIMBAD active (no brown dwarfs) ---
print("SIMBAD active (<100pc)...", flush=True)
n0 = len(rows)
# plx>10 mas (<100pc): the all-sky Fl*/BY* class is 172k (mostly distant ZTF flare
# candidates -> pure FP); the nearby subset complements GCVS without truncation.
adql = ("SELECT main_id,ra,dec,plx_value,otype FROM basic "
        "WHERE otype IN ('Fl*','BY*','RS*','Er*') AND ra IS NOT NULL AND plx_value > 10")
for r in parse_csv(tap(SIM, adql)):
    try:
        plx = float(r['plx_value']) if r.get('plx_value', '').strip() else 0.0
        dist = f"{1000.0/plx:.1f}" if plx > 0 else ''
        rows.append([f"{r['main_id'].strip()} [SIMBAD:{r['otype'].strip()}]",
                     float(r['ra']), float(r['dec']), dist])
    except Exception:
        pass
print(f"  SIMBAD: {len(rows)-n0}", flush=True)

# --- Pinned supplement: previously-cataloged active stars that a later SIMBAD
# --- query no longer returned (otype/main-row drift upstream). The 2026-08-30
# --- rebuild silently dropped these 13; restored 2026-08-31 (Gregg-approved).
# --- Pinned here so no upstream change can drop them again; dedup below merges
# --- them if SIMBAD returns them once more.
PINNED = [
    ("L  120-120 [SIMBAD:Er*]",      351.353743, -67.668656, "22.6"),
    ("CD-47   277 [SIMBAD:Er*]",      13.985524, -47.149560, "38.9"),
    ("RX J0314.8-6241 [SIMBAD:Er*]",  48.713798, -62.694947, "27.8"),
    ("CD-36  1252 [SIMBAD:Er*]",      49.742586, -36.393025, "21.8"),
    ("BD+01  2684 [SIMBAD:Er*]",     185.106387,   0.583568, "17.5"),
    ("StKM 1-1021 [SIMBAD:Er*]",     188.109685,  20.390993, "22.8"),
    ("Wolf 1351 [SIMBAD:Er*]",       309.336727,  21.947850, "33.1"),
    ("V* AS Dra [SIMBAD:RS*]",       185.548896,  73.248483, "42.8"),
    ("G 227-22 [SIMBAD:Er*]",        270.569183,  64.262289, "7.8"),
    ("G 271-149 [SIMBAD:Er*]",        26.653508,  -8.649336, "25.6"),
    ("BD-21  1074A [SIMBAD:Er*]",     76.707984, -21.585869, "19.8"),
    ("HD 358623 [SIMBAD:BY*]",       314.011411, -17.181612, "45.8"),
    ("HD 239960B [SIMBAD:Er*]",      336.999148,  57.697152, "4.0"),
]
rows.extend([list(p) for p in PINNED])
print(f"  pinned supplement: {len(PINNED)}", flush=True)

# --- Gaia DR3 distance backfill for rows still lacking a distance ---
missing = [(str(i), rows[i][1], rows[i][2]) for i in range(len(rows)) if rows[i][3] == '']
print(f"Gaia backfill for {len(missing)} rows ({GAIA_RADIUS_ARCSEC}\")...", flush=True)
filled = gaia_backfill(missing)
for k, d in filled.items():
    rows[int(k)][3] = d
print(f"  filled {len(filled)}", flush=True)

# --- de-duplicate within 5 arcsec (vectorized) ---
print(f"de-dup {len(rows)}...", flush=True)
sc = SkyCoord([r[1] for r in rows]*u.deg, [r[2] for r in rows]*u.deg)
i1, i2, _, _ = search_around_sky(sc, sc, 5.0*u.arcsec)
parent = list(range(len(rows)))
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
for a, b in zip(i1.tolist(), i2.tolist()):
    ra_, rb_ = find(a), find(b)
    if ra_ != rb_:
        parent[max(ra_, rb_)] = min(ra_, rb_)
clusters = {}
for i in range(len(rows)):
    clusters.setdefault(find(i), []).append(i)
keep = []
for grp in clusters.values():
    labels = " | ".join(sorted({rows[j][0] for j in grp}))
    dist = next((rows[j][3] for j in grp if rows[j][3] != ''), '')
    keep.append((labels, rows[grp[0]][1], rows[grp[0]][2], dist))
print(f"  unique: {len(keep)}", flush=True)

with open(OUT, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['common_name', 'ra_current', 'dec_current', 'coords_sexagesimal', 'distance'])
    nd = 0
    for name, ra, dec, dist in keep:
        sx = SkyCoord(ra*u.deg, dec*u.deg).to_string('hmsdms', sep=':', precision=1)
        w.writerow([name, f"{ra:.6f}", f"{dec:.6f}", sx, dist])
        nd += dist != ''
print(f"wrote {len(keep)} -> {OUT}  ({nd} with distance, {100*nd//len(keep)}%)")
