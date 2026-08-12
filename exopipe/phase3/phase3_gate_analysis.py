#!/usr/bin/env python3
"""
phase3_gate_analysis.py — apply Gregg's veto SEQUENCE to the gate ledger and track
losses through each gate. Sequence (Gregg 2026-07-01):
  mstruct (all)  ->  TAG image by post-mstruct >=7sigma count
    Low <=2 | Intermediate 3-10 | High >10 (informational)
  I/V 20x rb=0 (all)  ->  bilobe (Intermediate+High only)  ->  gaussfit (all)
Reports the funnel (counts + per-gate loss) overall / by band / by FP-class, at >=5s
and >=7s, plus the image-tag distribution. Boundaries/flags are constants up top so any
tweak re-runs instantly.  Usage: python phase3_gate_analysis.py
"""
import csv
from collections import defaultdict, Counter
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation, get_body
from astropy.time import Time

LEDGER = '/data07/gh/phase3/transients/Run_GATES/gate_ledger.csv'
LOW_MAX, HIGH_MIN = 2, 11        # Low: <=2 ; High: >=11 ; Int: 3-10  (post-mstruct >=7s)
IV_THRESH = 20.0
BILOBE_ON = ('Int', 'High')      # bilobe applied only to these image classes
# Gregg 2026-07-01: mstruct subsumes the >30 Jy bright-mask; keep only A-team + Jupiter.
ATEAM = SkyCoord([299.868, 350.866, 83.633, 187.706] * u.deg,
                 [40.734, 58.811, 22.015, 12.391] * u.deg)   # CygA CasA TauA VirA
ATEAM_DEG = 2.0
JUP_DEG = {'Red': 5.0, 'Full': 5.0, 'Green': 0.5, 'Blue': 0.5}
OVRO = EarthLocation(lat=37.2339 * u.deg, lon=-118.2817 * u.deg, height=1222 * u.m)
STAGES = ['raw', 'mstruct', 'iv', 'bilobe', 'gaussfit', 'ateam', 'jupiter']
_JUP = {}


def _jup(date):
    if date not in _JUP:
        try:
            b = get_body('jupiter', Time(f'{date}T10:00:00'), OVRO); _JUP[date] = SkyCoord(b.ra, b.dec)
        except Exception:
            _JUP[date] = None
    return _JUP[date]


def load():
    rows = list(csv.DictReader(open(LEDGER)))
    for r in rows:
        r['snr'] = abs(float(r['snr'])); r['mstruct'] = float(r['mstruct'])
        r['iv0'] = float(r['iv0']); r['bilobe'] = int(r['bilobe']); r['gaussfit'] = int(r['gaussfit'])
    return rows


def tag_of(mstruct_survivors):
    n7 = sum(1 for c in mstruct_survivors if c['snr'] >= 7)
    return 'Low' if n7 <= LOW_MAX else ('High' if n7 >= HIGH_MIN else 'Int')


def _radec(cs):
    return SkyCoord([float(c['ra']) for c in cs] * u.deg, [float(c['dec']) for c in cs] * u.deg) if cs else None


def sequence(cands):
    """Return dict stage->list of surviving candidates, for one image."""
    ms = [c for c in cands if c['snr'] >= 5 * max(c['mstruct'], 1)]
    tag = tag_of(ms)
    iv = [c for c in ms if c['iv0'] <= IV_THRESH]
    bl = [c for c in iv if not c['bilobe']] if tag in BILOBE_ON else list(iv)
    gf = [c for c in bl if not c['gaussfit']]
    # A-team 2deg
    at = [c for c in gf
          if float(SkyCoord(float(c['ra']) * u.deg, float(c['dec']) * u.deg).separation(ATEAM).deg.min()) > ATEAM_DEG]
    # Jupiter (band radius)
    band = cands[0]['band'] if cands else 'Full'; jr = JUP_DEG.get(band, 5.0)
    ju = []
    for c in at:
        jp = _jup(c['date'])
        if jp is None or SkyCoord(float(c['ra']) * u.deg, float(c['dec']) * u.deg).separation(jp).deg > jr:
            ju.append(c)
    return tag, {'raw': cands, 'mstruct': ms, 'iv': iv, 'bilobe': bl, 'gaussfit': gf,
                 'ateam': at, 'jupiter': ju}


def funnel(images, snr_cut=5.0):
    tot = {s: 0 for s in STAGES}
    for _, st in images:
        for s in STAGES:
            tot[s] += sum(1 for c in st[s] if c['snr'] >= snr_cut)
    return tot


def show(name, images, snr_cut):
    t = funnel(images, snr_cut)
    print(f"  {name:16s}", end='')
    prev = None
    for s in STAGES:
        loss = '' if prev is None else f'(-{prev - t[s]})'
        print(f" {s}={t[s]}{loss}", end='')
        prev = t[s]
    print()


def main():
    rows = load()
    byimg = defaultdict(list)
    for r in rows:
        byimg[(r['lst'], r['date'], r['band'])].append(r)
    images = []
    for k, cs in byimg.items():
        tag, st = sequence(cs)
        images.append((tag, st, k))
    imgs = [(t, s) for t, s, k in images]

    print(f"=== {len(rows)} raw candidates over {len(byimg)} images ===\n")
    tags = Counter(t for t, _, _ in images)
    print(f"image TAG distribution (post-mstruct >=7s): "
          f"Low(<=2)={tags['Low']}  Int(3-10)={tags['Int']}  High(>=11)={tags['High']}\n")

    print("FUNNEL (>=5sigma candidates, per-gate loss):")
    show('ALL', imgs, 5.0)
    for b in ['Red', 'Green', 'Blue', 'Full']:
        show(b, [(t, s) for t, s, k in images if k[2] == b], 5.0)
    print("\nFUNNEL (>=7sigma candidates):")
    show('ALL', imgs, 7.0)
    for b in ['Red', 'Green', 'Blue', 'Full']:
        show(b, [(t, s) for t, s, k in images if k[2] == b], 7.0)

    print("\nFUNNEL by FP-class (>=7sigma):")
    for cl in ['Low', 'Int', 'High']:
        show(cl, [(t, s) for t, s, k in images if t == cl], 7.0)

    # final survivors per night (>=7s)
    print("\nfinal survivors (>=7s, after full sequence) per night — top 12:")
    bd = defaultdict(int)
    for t, s, k in images:
        bd[k[1]] += sum(1 for c in s['gaussfit'] if c['snr'] >= 7)
    for d, n in sorted(bd.items(), key=lambda x: -x[1])[:12]:
        print(f"    {d}: {n}")
    print(f"    ... [{len(bd)} nights] total final >=7s = {sum(bd.values())}")


if __name__ == '__main__':
    main()
