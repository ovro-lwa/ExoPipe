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
ASU = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv"
XM  = "http://cdsxmatch.u-strasbg.fr/xmatch/api/v1/sync"
OUT = "/lustre/gh/main/clv15/flare_ucd.csv"
UCD_MAX_PC = 50.0
GAIA_RADIUS_ARCSEC = 5.0


def _get(url):
    return urllib.request.urlopen(url, timeout=120).read().decode('utf-8', 'replace')


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
txt = _get(ASU + "?-source=J/A%2BA/619/L8&-out.max=999999&-out=Source,RAICRS,DEICRS,Plx,sptphot")
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
