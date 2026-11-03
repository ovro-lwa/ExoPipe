#!/usr/bin/env python3
"""The size / axis rule's CALIBRATION with the library (lib/search/size_rule.py; spec plan/specs/fp_gates_v2.md section 3): beam-
shaped point sources injected ON THE SKY into a tier's own residual frames, fitted exactly as the search fits detections,
aggregated into the product the rule reads (exopipe/common/size_envelope_<tier>.json) - the procedure of the reference
implementation (refactor_analysis/bright_source_contamination_20261020/size_filter2.py calibrate, 2026-10-22).

Procedure: the frames holding the tier's served detections (--members: cand_id, st, ra, dec), per Stokes x band class
--frames/4 of them at random (--seed); in each, --per-frame positions at random that are inside the search's domain
(size_rule.in_domain at the frame's mid-time: deep LST hour + 0.5 h, 10-minute the window's middle), >= 3 beams from every
served detection of the frame, on a stamp that is finite with <= 10% exact zeros; at each, a beam (the frame's BMAJ, BMIN,
BPA) at S/N 5, 6, 7, 8, 10, 12, 15, 20, 30 x the annulus rms is added and fitted (size_rule.measure_detection); the product is
size_rule.envelope_from_injections of the rows. With --reference the report compares the new product with it per Stokes,
band class and S/N: the percentiles the rule reads, the fixed 1.5 / 1.5 rule's loss and the rule's own false-flag rate (the
reference product's thresholds looked up at each injection's measured S/N), each with its binomial / bootstrap spread.

Read-only on the run and the frames; writes only under --out (injections.csv, size_envelope.json, report.txt).
  python tools/size_rule_calibrate.py deep   --members <size_detections_v2.csv> --out DIR [--reference PRODUCT] [--nproc 6]
  python tools/size_rule_calibrate.py sublst --members <...> --out DIR [--reference PRODUCT]
"""
import argparse
import collections
import csv
import json
import math
import os
import random
import re
import sys
import warnings

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
warnings.filterwarnings('ignore')

import numpy as np                                                  # noqa: E402
from exopipe.phase3.lib.search import size_rule as SZ               # noqa: E402

csv.field_size_limit(10 ** 9)
DEEP_RUN = '/data07/gh/phase3/transients/Run_P3S5'
SUBLST_TREE = '/lustre/gh/main/phase3/sublst/tenmin_root_x2'
CID = {'deep': re.compile(r'_(\d\dh)_(\d{4}-\d\d-\d\d)_([^_]+)_\d+$'),
       'sublst': re.compile(r'_(\d\dh)_(\d{4}-\d\d-\d\dT\d\d)_([^_]+)_\d+$')}


class Frames:
    """The tier's frame of a key (Stokes, LST, band, date-or-window) and its mid-time LST angle (deg)."""

    def __init__(self, tier, run=None, tree=None):
        self.tier, self.run, self.tree, self._fr, self._ls = tier, run or DEEP_RUN, tree or SUBLST_TREE, {}, {}

    def exists(self, p):
        """os.path.exists by one listing per directory (thousands of frames: one Lustre metadata call per directory)."""
        d, b = os.path.split(p)
        if d not in self._ls:
            try:
                self._ls[d] = set(os.listdir(d))
            except OSError:
                self._ls[d] = set()
        return b in self._ls[d]

    def path(self, st, lst, band, key):
        if self.tier == 'deep':
            return '%s/%sresid/%s_%s/%s_%s_redo_resid_%s.fits' % (self.run, st.lower(), lst, band, key, st, band)
        if (lst, band) not in self._fr:
            self._fr[(lst, band)] = {}
            p = '%s/%s_%s/frames.csv' % (self.tree, lst, band)
            if os.path.exists(p):
                for r in csv.DictReader(open(p)):
                    self._fr[(lst, band)][(r['key'], r['stokes'])] = r['sub_path']
        return self._fr[(lst, band)].get((key, st), '')

    def lst_deg(self, lst, key):
        if self.tier == 'deep':
            return 15.0 * (int(lst[:2]) + 0.5)                    # the deep tables' and conform's mid-hour convention
        return 15.0 * (int(lst[:2]) + (int(key.split('T')[1]) + 0.5) / 6.0)   # window w of six in the LST hour


def members(path, tier):
    out = []
    for r in csv.DictReader(open(path)):
        m = CID[tier].search(r['cand_id'])
        if m and r.get('st') in SZ.STOKES:
            out.append(dict(st=r['st'], ra=float(r['ra']), dec=float(r['dec']), lst=m.group(1), key=m.group(2), band=m.group(3)))
    return out


def pick_frames(mem, fr, n_frames, seed):
    """[(Stokes, LST, band, key)]: n_frames / 4 per Stokes x band class, drawn at random among the frames of served detections
    (the reference's draw: keys in first-seen order, shuffled with the seed)."""
    random.seed(seed)
    by = collections.OrderedDict()
    for m in mem:
        by.setdefault((m['st'], m['lst'], m['band'], m['key']), []).append(m)
    keys = [k for k in by if fr.path(*k) and fr.exists(fr.path(*k))]
    pick = []
    for st in SZ.STOKES:
        for cls in SZ.CLASSES:
            ks = [k for k in keys if k[0] == st and SZ.band_class(k[2]) == cls]
            random.shuffle(ks)
            pick += ks[:n_frames // 4]
    return pick, by


def _inject_frame(args):
    """Worker: the injections of one frame (its own seeded generator)."""
    from astropy.io import fits
    from astropy.wcs import WCS
    tier, run, tree, k, avoid_sky, per_frame, seed = args
    fr = Frames(tier, run, tree)
    rng = random.Random(f'{seed}:{"|".join(k)}')
    p = fr.path(*k)
    out, refused = [], collections.Counter()
    with fits.open(p, memmap=True) as hh:
        hd = hh[0].header; sec = hh[0].section; lead = (0,) * (hd['NAXIS'] - 2)
        ny, nx = hd['NAXIS2'], hd['NAXIS1']; w = WCS(hd).celestial
        cd = abs(float(hd['CDELT1'])); bmaj, bmin, bpa = float(hd['BMAJ']), float(hd['BMIN']), float(hd.get('BPA', 0.0))
        bpix = bmaj / cd; Rb = SZ.rms_radius(bpix); R = SZ.stamp_radius(bpix)
        L = fr.lst_deg(k[1], k[3]); beam = SZ.beam_image(bmaj, bmin, bpa, cd, R)
        avoid = [w.world_to_pixel_values(ra % 360, dec) for ra, dec in avoid_sky]
        done = tries = 0
        while done < per_frame and tries < per_frame * 40:
            tries += 1
            x, y = rng.randint(Rb, nx - Rb - 1), rng.randint(Rb, ny - Rb - 1)
            ra, de = w.pixel_to_world_values(x, y)
            if not SZ.in_domain(float(ra), float(de), L):
                refused['outside the search domain'] += 1; continue
            if any(math.hypot(x - ax, y - ay) < 3 * bpix for ax, ay in avoid):
                refused['near a served detection'] += 1; continue
            big = np.array(sec[lead + (slice(y - Rb, y + Rb + 1), slice(x - Rb, x + Rb + 1))], dtype=np.float64)
            if not SZ.stamp_ok(big):
                refused['non-finite or > 10% zeros'] += 1; continue
            rms0 = SZ.annulus_rms(big, Rb, bpix, 1.0)
            if not np.isfinite(rms0) or rms0 <= 0:
                refused['no rms'] += 1; continue
            for snr in SZ.SNR_GRID:
                b = big.copy(); b[Rb - R:Rb + R + 1, Rb - R:Rb + R + 1] += snr * rms0 * beam
                r = SZ.measure_detection(b, Rb, bpix, cd, bmaj, bmin)
                o = dict(st=k[0], cls=SZ.band_class(k[2]), band=k[2], lst=k[1], key=k[3], snr=snr, fit=int(bool(r)), x=x, y=y,
                         ra=round(float(ra), 5), dec=round(float(de), 5))
                if r:
                    o.update(size=r['size'], axis=r['axis'], err=r['err'], eff=r['eff'], snr_meas=r['snr'])
                out.append(o)
            done += 1
    return out, refused


def rule_rates(rows, env):
    """{(Stokes, class, S/N): (n, the rule's flags, the fixed 1.5 / 1.5 rule's flags)} - the rule looked up at each injection's
    measured S/N in ``env``."""
    out = collections.defaultdict(lambda: [0, 0, 0])
    for o in rows:
        if not int(o['fit']) or not np.isfinite(float(o['snr_meas'])):
            continue
        rec = dict(size=float(o['size']), axis=float(o['axis']), eff=float(o['eff']), snr=float(o['snr_meas']))
        v = SZ.size_verdict(env, o['st'], 'Red' if o['cls'] == 'low' else 'Full', rec)[0]
        k = (o['st'], o['cls'], float(o['snr']))
        out[k][0] += 1
        out[k][1] += v != ''
        out[k][2] += (rec['eff'] > SZ.SIZE_FLOOR) or (rec['axis'] > SZ.AXIS_FLOOR)
    return out


def q99_spread(rows, st, cls, snr, key, n_boot=200, seed=5):
    """The bootstrap 68% half-width of the 99th percentile of ``key`` ('eff' or 'axis') at one injected S/N."""
    v = np.array([float(o[key]) for o in rows if o['st'] == st and o['cls'] == cls and float(o['snr']) == snr and int(o['fit'])])
    if v.size < 30:
        return float('nan')
    g = np.random.default_rng(seed)
    b = [np.percentile(g.choice(v, v.size), 99) for _ in range(n_boot)]
    return float((np.percentile(b, 84) - np.percentile(b, 16)) / 2)


def main(argv=None):
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument('tier', choices=('deep', 'sublst'))
    ap.add_argument('--members', required=True, help='CSV of the served detections: cand_id, st, ra, dec')
    ap.add_argument('--out', required=True)
    ap.add_argument('--reference', default='', help='a product to compare with (e.g. exopipe/common/size_envelope_deep.json)')
    ap.add_argument('--run', default=None, help=f'deep: the run directory (default {DEEP_RUN})')
    ap.add_argument('--tree', default=None, help=f'sublst: the window tree (default {SUBLST_TREE})')
    ap.add_argument('--frames', type=int, default=240)
    ap.add_argument('--per-frame', type=int, default=12)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--nproc', type=int, default=6)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    lines = []

    def say(s):
        print(s, flush=True); lines.append(s)
    fr = Frames(a.tier, a.run, a.tree)
    mem = members(a.members, a.tier)
    pick, by = pick_frames(mem, fr, a.frames, a.seed)
    say(f'{a.tier}: {len(mem)} served detections on {len(by)} frames; {len(pick)} frames drawn (seed {a.seed}), '
        f'{a.per_frame} positions each, S/N {SZ.SNR_GRID}')
    jobs = [(a.tier, fr.run, fr.tree, k, [(m['ra'], m['dec']) for m in by[k]], a.per_frame, a.seed) for k in pick]
    rows, refused = [], collections.Counter()
    with Pool(a.nproc) as pool:
        for i, (out, ref) in enumerate(pool.imap(_inject_frame, jobs)):
            rows += out; refused.update(ref)
            if i % 40 == 0:
                print(f'  frames {i + 1}/{len(jobs)}, injections {len(rows)}', flush=True)
    keys = list(dict.fromkeys(kk for o in rows for kk in o))
    with open(os.path.join(a.out, 'injections.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for o in rows:
            w.writerow({kk: o.get(kk, '') for kk in keys})
    cal = SZ.envelope_from_injections(rows)
    with open(os.path.join(a.out, 'size_envelope.json'), 'w') as f:
        json.dump(cal, f, indent=1)
    say(f'{len(rows)} injections ({sum(int(o["fit"]) for o in rows)} fitted); positions refused: {dict(refused)}')
    env_new = SZ.Envelope(cal).require()
    ref = SZ.Envelope.load(a.reference).require() if a.reference else None
    rates_new = rule_rates(rows, env_new)
    rates_ref = rule_rates(rows, ref) if ref else None
    for st in SZ.STOKES:
        for cls in SZ.CLASSES:
            say(f'{st} {cls}:')
            for snr in SZ.SNR_GRID:
                k = '%s_%s_%g' % (st, cls, snr)
                c = cal.get(k)
                if not c:
                    say(f'  {snr:>4}: not calibrated')
                    continue
                n, fl_new, fixed = rates_new[(st, cls, float(snr))]
                se = math.sqrt(max(fixed, 1) * (n - fixed) / n) / n if n else float('nan')
                line = (f'  {snr:>4}: n {c["n"]} (fail {c["n_fail"]}) | eff_q99 {c["eff_q99"]:.2f} +- '
                        f'{q99_spread(rows, st, cls, snr, "eff"):.2f}, axis_q99 {c["axis_q99"]:.2f} +- '
                        f'{q99_spread(rows, st, cls, snr, "axis"):.2f} | fixed 1.5 / 1.5 rule removes {100 * fixed / n:.1f}% '
                        f'+- {100 * se:.1f} | the rule (this product) {100 * fl_new / n:.1f}%')
                if ref:
                    rc = ref.cal.get(k, {})
                    _, fl_ref, _ = rates_ref[(st, cls, float(snr))]
                    line += (f' || reference eff_q99 {rc.get("eff_q99", float("nan")):.2f}, axis_q99 '
                             f'{rc.get("axis_q99", float("nan")):.2f}, fixed rule {100 * rc.get("gate_rate", float("nan")):.1f}%'
                             f'; the reference product\'s rule on these injections {100 * fl_ref / n:.1f}%')
                say(line)
    with open(os.path.join(a.out, 'report.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
