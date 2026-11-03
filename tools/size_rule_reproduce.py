#!/usr/bin/env python3
"""ACCEPTANCE of the size / axis rule (lib/search/size_rule.py; plan/specs/fp_gates_v2.md section 5): the library reproduces the
reference implementation - the deep page filters of 2026-10-22 (refactor_analysis/bright_source_contamination_20261020:
fp_size_common.py, size_filter2.py, deep/deep_det_filters.py) - as the never-merged branch fp-gates-v2 checked it
(tools/fp_v2_reproduce_deep.py, "REPRODUCED: size 7,122 / 7,122 states"), on today's code.

Run_P3S5 was re-published since (2026-10-30/31: 222 of 912 cell-Stokes re-searched; its catalogs re-built), so the reference's
inputs are read from what was recorded on 2026-10-22, never from today's catalogs:
  detections   deep/size_detections_v2.csv (7,122 rows, 2026-10-22 04:55): each detection's frame key and the reference's own
               measurement (size, axis, err, eff, local S/N; 4 / 3 decimals)
  identities   fp_sample_20261022/page_states_deep{I,V}.json (the served pages, 2026-10-22 07:23): the identity of every detection
  states       Run_P3S5/fp_filters.csv.vgti_v1_20261022 (the second build of 2026-10-22, 05:02 - the branch's reference)
  envelope     the repo's exopipe/common/size_envelope_deep.json (must be byte-identical to deep/size_cal_v2.json)
  injections   deep/size_cal_injections_v2.csv (25,920 rows) and tenmin/size_cal_injections_v2.csv (25,920)

Checks (each printed with its counts; the exit status is 0 only when every check passes):
  verdicts     the library's verdict and per-instant carry on the reference's own measurements = the page states, every
               detection, and its size / excess values = the page's (3 decimals)
  measure      the library's measurement on each detection's frame = the reference's (size, axis, err, eff to 4 decimals, S/N to
               3); the frame is the one Run_P3S5 links today, and for a cell-Stokes re-searched after 2026-10-22 05:00 the
               attempt published before then (the search register, newest first). Then the states from these measurements.
  calibration  the envelope rebuilt by the library from the reference's injections = the product (every number), and the
               rates the notes quote: the fixed 1.5 / 1.5 rule's loss and the rule's own false flags (looked up at the measured
               S/N) per Stokes, band class and injected S/N

Read-only (Run_P3S5, its attempts, the analysis files); writes only --out. Usage:
  python tools/size_rule_reproduce.py [--only verdicts|measure|calibration] [--out report.txt] [--nproc N]
"""
import argparse
import collections
import csv
import glob
import json
import math
import os
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
RUN = '/data07/gh/phase3/transients/Run_P3S5'
ANA = '/lustre/gh/main/phase3/refactor_analysis/bright_source_contamination_20261020'
REF_STATES = RUN + '/fp_filters.csv.vgti_v1_20261022'
PAGE_STATES = ANA + '/fp_sample_20261022/page_states_deep%s.json'
REGISTER = '/lustre/gh/main/exopipe_ops/p3s5_state/published_%s.csv'
COLLECT = '/lustre/gh/main/phase3_collect/p3s5_collect_%s/attempts'
CUTOFF = '20261022T050000'           # the reference's frames: published before 2026-10-22 05:00 UTC
CID = re.compile(r'_(\d\dh)_(\d{4}-\d\d-\d\d)_([^_]+)_\d+$')
ENVELOPE = os.path.join(REPO, 'exopipe', 'common', 'size_envelope_deep.json')
ST_SIZE = ['larger than a point source at its SNR', 'more elongated than a point source at its SNR',
           'point-like itself; another detection of the same source (identity) in the same epoch is larger or more elongated',
           'consistent with a point source', 'not measured']


def state_index(text):
    for i, s in enumerate(ST_SIZE):
        if text.startswith(s[:24]):
            return i
    return -1


def reference_detections():
    rows = list(csv.DictReader(open(os.path.join(ANA, 'deep', 'size_detections_v2.csv'))))
    for r in rows:
        m = CID.search(r['cand_id'])
        r['lst'], r['date'], r['img_band'] = m.group(1), m.group(2), m.group(3)
    return rows


def identities():
    ident = {}
    for st in ('I', 'V'):
        d = json.load(open(PAGE_STATES % st))
        for i in d['identities']:
            for x in i['detections']:
                ident[x['cand_id']] = (st, i['identity'])
    return ident


def reference_states():
    rows = [r for r in csv.DictReader(l for l in open(REF_STATES) if not l.startswith('#'))]
    return {(r['cand_id'], r['ra'], r['dec']): r for r in rows}


def judge_rows(env, rows, ident, key_rec):
    """{key: (state index, size, sx, ax)} from the library's verdict on each row's record (key_rec(row) -> rec or None) and the
    per-instant carry over identities - deep_det_filters.py's states."""
    v = {}
    for r in rows:
        rec = key_rec(r)
        k = (r['cand_id'], r['ra'], r['dec'])
        if rec is None or not np.isfinite(rec['snr']):
            v[k] = None
            continue
        verdict, sx, ax = SZ.size_verdict(env, r['st'], r['img_band'], rec)
        v[k] = (verdict, rec['size'], sx, ax)
    carried = SZ.carry_instant((ident.get(r['cand_id'], (r['st'], r['cand_id'])), (r['lst'], r['date']),
                                v.get((r['cand_id'], r['ra'], r['dec'])) is not None
                                and v[(r['cand_id'], r['ra'], r['dec'])][0] != '') for r in rows)
    out = {}
    for r in rows:
        k = (r['cand_id'], r['ra'], r['dec'])
        if v[k] is None:
            out[k] = (4, None, None, None)
            continue
        verdict, size, sx, ax = v[k]
        idx = (0 if verdict == 'gaussfit-size' else 1 if verdict == 'axis-ratio'
               else 2 if (ident.get(r['cand_id'], (r['st'], r['cand_id'])), (r['lst'], r['date'])) in carried else 3)
        out[k] = (idx, size, sx, ax)
    return out


def compare_states(mine, ref, out, what):
    n = bad_state = bad_val = missing = 0
    ex = []
    by = collections.Counter()
    for k, (idx, size, sx, ax) in mine.items():
        r = ref.get(k)
        if r is None:
            missing += 1
            continue
        n += 1
        theirs = state_index(r['fp_c_size'])
        by[ST_SIZE[idx][:24] if idx >= 0 else '?'] += 1
        if idx != theirs:
            bad_state += 1
            if len(ex) < 8:
                ex.append(f'   state differs: {k[0]} mine {idx} page {theirs} (sx {sx} ax {ax}; page {r.get("fp_m_sizex")} '
                          f'{r.get("fp_m_axisx")})')
        if size is not None and r.get('fp_m_sizex', '') != '':
            if (round(size, 3) != float(r['fp_m_size']) or round(sx, 3) != float(r['fp_m_sizex'])
                    or round(ax, 3) != float(r['fp_m_axisx'])):
                bad_val += 1
                if len(ex) < 8:
                    ex.append(f'   value differs: {k[0]} size {size:.4f} sx {sx:.4f} ax {ax:.4f}; page {r["fp_m_size"]} '
                              f'{r["fp_m_sizex"]} {r["fp_m_axisx"]}')
    out(f'{what}: {n} detections compared ({missing} not in the reference states); state differs {bad_state}; '
        f'size / excess values differ (3 decimals) {bad_val}; states: ' + ', '.join(f'{s} {c}' for s, c in sorted(by.items())))
    for e in ex:
        out(e)
    return bad_state + bad_val + missing


def check_verdicts(out):
    env = SZ.Envelope.load(ENVELOPE)
    if env.md5 != SZ.md5_file(os.path.join(ANA, 'deep', 'size_cal_v2.json')):
        out('verdicts: the repo envelope is NOT the reference size_cal_v2.json')
        return 1
    rows = reference_detections()
    ident = identities()
    out(f'verdicts: {len(rows)} reference detections (I {sum(r["st"] == "I" for r in rows)}, V {sum(r["st"] == "V" for r in rows)}),'
        f' {sum(r["cand_id"] in ident for r in rows)} with a page identity; envelope {env.label()}')

    def rec(r):
        if r.get('eff', '') == '' or r.get('snr_meas', '') == '':
            return None
        return dict(size=float(r['size']), axis=float(r['axis']), eff=float(r['eff']), snr=float(r['snr_meas']))
    return compare_states(judge_rows(env, rows, ident, rec), reference_states(), out, 'verdicts (reference measurements)')


def register(st):
    out = collections.defaultdict(list)
    for line in open(REGISTER % st):
        p = line.strip().split(',')
        if len(p) == 3 and p[0] and re.fullmatch(r'\d{8}T\d{6}', p[2]):
            out[p[0]].append((p[2], p[1]))
    return {k: sorted(v, reverse=True) for k, v in out.items()}


def frame_candidates(st, lst, band, date, reg):
    """Frames to read for a reference detection, in order: today's link; then, when that attempt is newer than CUTOFF, the
    attempts published before CUTOFF, newest first."""
    sub = 'iresid' if st == 'I' else 'vresid'
    name = f'{date}_{st}_redo_resid_{band}.fits'
    link = os.path.join(RUN, sub, f'{lst}_{band}')
    out = [('link', os.path.join(link, name))]
    tgt = os.path.realpath(link) if os.path.islink(link) else ''
    tag = re.search(r'_(\d{8}T\d{6})/', tgt + '/')
    if tag and tag.group(1) > CUTOFF:
        for t, host in reg.get(f'{lst}_{band}', ()):
            if t < CUTOFF:
                out.append((f'register {t}', os.path.join(COLLECT % host, f'{lst}_{band}_{st}_{t}', 'Run_P3S5', sub,
                                                          f'{lst}_{band}', name)))
    return out


def _measure_group(args):
    """Worker: every reference detection of one frame key, tried on each candidate frame in turn."""
    from astropy.io import fits
    from astropy.wcs import WCS
    cands, dets = args
    res = []
    for how, p in cands:
        if not os.path.exists(p):
            continue
        with fits.open(p, memmap=True) as hh:
            hd = hh[0].header; sec = hh[0].section; lead = (0,) * (hd['NAXIS'] - 2)
            ny, nx = hd['NAXIS2'], hd['NAXIS1']; w = WCS(hd).celestial
            cd = abs(float(hd['CDELT1'])); bmaj, bmin = float(hd['BMAJ']), float(hd['BMIN']); bpix = bmaj / cd
            got = []
            for d in dets:
                x, y = w.world_to_pixel_values(float(d['ra']) % 360, float(d['dec']))
                x, y = int(round(float(x))), int(round(float(y)))
                rec = None
                for R_ in (SZ.rms_radius(bpix), SZ.edge_radius(bpix)):
                    if R_ <= x < nx - R_ and R_ <= y < ny - R_:
                        big = np.nan_to_num(np.array(sec[lead + (slice(y - R_, y + R_ + 1), slice(x - R_, x + R_ + 1))],
                                                     dtype=np.float64))
                        rec = SZ.measure_detection(big, R_, bpix, cd, bmaj, bmin)
                        break
                got.append(rec)
        ok = [_same(d, r) for d, r in zip(dets, got)]
        res.append((how, p, got, ok))
        if all(ok):
            break
    return res


def _same(d, rec):
    if rec is None:
        return d.get('eff', '') == ''
    if d.get('eff', '') == '':
        return False
    vals = (round(rec['size'], 4) == float(d['size']) and round(rec['axis'], 4) == float(d['axis'])
            and round(rec['eff'], 4) == float(d['eff'])
            and ((d['err'] == '' and not np.isfinite(rec['err'])) or (d['err'] != '' and round(rec['err'], 4) == float(d['err']))))
    snr = ((d['snr_meas'] == '' and not np.isfinite(rec['snr']))
           or (d['snr_meas'] != '' and np.isfinite(rec['snr']) and round(rec['snr'], 3) == float(d['snr_meas'])))
    return vals and snr


def check_measure(out, nproc):
    from multiprocessing import Pool
    rows = reference_detections()
    reg = {st: register(st) for st in ('I', 'V')}
    grp = collections.defaultdict(list)
    for r in rows:
        grp[(r['st'], r['lst'], r['img_band'], r['date'])].append(r)
    jobs = [(frame_candidates(st, lst, band, date, reg[st]), dets) for (st, lst, band, date), dets in sorted(grp.items())]
    out(f'measure: {len(rows)} reference detections on {len(jobs)} frames ({nproc} processes)')
    recs, how_n, frames_bad = {}, collections.Counter(), []
    with Pool(nproc) as pool:
        for (cands, dets), res in zip(jobs, pool.imap(_measure_group, jobs, chunksize=4)):
            if not res:
                how_n['no frame'] += len(dets)
                frames_bad.append(f'{dets[0]["cand_id"]}: no frame ({cands[0][1]})')
                continue
            how, p, got, ok = res[-1]
            if all(ok):
                how_n[f'identical ({how.split()[0]})'] += len(dets)
            else:
                how_n['differ'] += sum(not o for o in ok)
                how_n[f'identical ({how.split()[0]})'] += sum(ok)
                if len(frames_bad) < 8:
                    i = ok.index(False)
                    frames_bad.append(f'{dets[i]["cand_id"]}: tried {[h for h, *_ in res]}; mine {got[i]} ref '
                                      f'{ {k: dets[i][k] for k in ("size", "axis", "err", "eff", "snr_meas")} }')
            for d, g in zip(dets, got):
                recs[(d['cand_id'], d['ra'], d['dec'])] = g
    out('measure: library measurement vs the reference (4 decimals, S/N 3): ' + ', '.join(f'{k} {v}' for k, v in sorted(how_n.items())))
    for e in frames_bad:
        out('   ' + e)
    bad = how_n['differ'] + how_n['no frame']
    env = SZ.Envelope.load(ENVELOPE)
    ident, ref = identities(), reference_states()

    def as_page(r):
        """The library's measurement as the page read it: size_detections_v2.csv's text (4 decimals, S/N 3) parsed back - the
        fitter's numbers are numpy scalars, whose own rounding differs from Python's at a last-digit tie."""
        g = recs.get((r['cand_id'], r['ra'], r['dec']))
        if g is None:
            return None
        return dict(size=float(str(round(g['size'], 4))), axis=float(str(round(g['axis'], 4))),
                    eff=float(str(round(g['eff'], 4))),
                    snr=float(str(round(g['snr'], 3))) if np.isfinite(g['snr']) else float('nan'))
    bad += compare_states(judge_rows(env, rows, ident, as_page), ref, out, 'states (library measurements, read as the page did)')
    full = judge_rows(env, rows, ident, lambda r: recs.get((r['cand_id'], r['ra'], r['dec'])))
    dev = sorted(max(abs(v[1] - float(ref[k]['fp_m_size'])), abs(v[2] - float(ref[k]['fp_m_sizex'])),
                     abs(v[3] - float(ref[k]['fp_m_axisx']))) / max(1.0, abs(v[1]), abs(v[2]), abs(v[3]))
                 for k, v in full.items() if v[1] is not None and ref.get(k, {}).get('fp_m_sizex', '') != '')
    nst = sum(full[k][0] != state_index(ref[k]['fp_c_size']) for k in full if k in ref)
    out(f'states (library measurements, full precision): state differs {nst}; size / excess values within {dev[-1]:.2e} '
        f'(relative to max(1, |value|)) of the page\'s 3 decimals over {len(dev)} detections')
    return bad + nst


def check_calibration(out):
    bad = 0
    for tier, prod in (('deep', 'size_envelope_deep.json'), ('tenmin', 'size_envelope_sublst.json')):
        p_inj = os.path.join(ANA, tier, 'size_cal_injections_v2.csv')
        ref = json.load(open(os.path.join(ANA, tier, 'size_cal_v2.json')))
        mine_prod = json.load(open(os.path.join(REPO, 'exopipe', 'common', prod)))
        rows = list(csv.DictReader(open(p_inj)))
        for o in rows:
            o['snr'] = float(o['snr'])
        cal = SZ.envelope_from_injections(rows)
        diff = [k for k in set(cal) | set(ref) if k not in cal or k not in ref
                or any(abs(cal[k][q] - ref[k][q]) > 1e-12 * max(1.0, abs(ref[k][q])) for q in ref[k])]
        same_prod = mine_prod == ref
        out(f'calibration {tier}: {len(rows)} injections; the library\'s envelope = the reference product: '
            f'{"yes" if not diff else "NO (" + ", ".join(sorted(diff)[:6]) + ")"}; the repo product = the reference: '
            f'{"yes" if same_prod else "NO"}')
        bad += len(diff) + (not same_prod)
        env = SZ.Envelope(cal)
        rate = collections.defaultdict(list)
        for o in rows:
            if int(o['fit']) and o['snr_meas'] != '' and np.isfinite(float(o['snr_meas'])):
                rec = dict(size=float(o['size']), axis=float(o['axis']), eff=float(o['eff']), snr=float(o['snr_meas']))
                rate[(o['st'], o['cls'], o['snr'])].append(SZ.size_verdict(env, o['st'], 'Red' if o['cls'] == 'low' else 'Full', rec)[0] != '')
        for st in ('I', 'V'):
            out(f'  {tier} {st}: the fixed 1.5 / 1.5 rule removes (low / high bands): ' + ', '.join(
                f'{s:g}: {100 * cal["%s_low_%g" % (st, s)]["gate_rate"]:.1f} / {100 * cal["%s_high_%g" % (st, s)]["gate_rate"]:.1f}%'
                for s in SZ.SNR_GRID))
            out(f'  {tier} {st}: the rule flags (lookup at the measured S/N, low / high): ' + ', '.join(
                f'{s:g}: {100 * np.mean(rate[(st, "low", float(s))]):.1f} / {100 * np.mean(rate[(st, "high", float(s))]):.1f}%'
                for s in SZ.SNR_GRID))
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', choices=('verdicts', 'measure', 'calibration'), default=None)
    ap.add_argument('--out', default='')
    ap.add_argument('--nproc', type=int, default=6)
    a = ap.parse_args(argv)
    lines = []

    def out(s):
        print(s, flush=True)
        lines.append(s)
    bad = 0
    if a.only in (None, 'verdicts'):
        bad += check_verdicts(out)
    if a.only in (None, 'calibration'):
        bad += check_calibration(out)
    if a.only in (None, 'measure'):
        bad += check_measure(out, a.nproc)
    out('RESULT: %s' % ('REPRODUCED' if bad == 0 else 'DIFFERS (%d)' % bad))
    if a.out:
        with open(a.out, 'w') as f:
            f.write('\n'.join(lines) + '\n')
    return 0 if bad == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
