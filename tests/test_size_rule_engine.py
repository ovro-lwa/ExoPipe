#!/usr/bin/env python3
"""STANDING TEST (lwacalim07 only: /data07) - the size / axis rule in the ENGINE, end to end (review 2026-11-03 of
wt-20261102-S-size, finding 4: no engine run under the rule had been tested; the gate-ledger content test had never read a
ledger the engine wrote).

run_deep_cell (lib/search/deep_search) on a read-only copy of the installed cell 05h_78MHz (records copied, FITS and flows
linked; tests/test_search_admitted.py's driver and engine(): x2 flow refinement off, v110 off, incremental for TWO member
nights, seeded with the served search's sigma stack and frames as the base it appends to), one Stokes per run as the search
step runs the engine, under the ruled 'envelope' (the step's environment: size_rule.step_env_for) and under 'production', the
same nights, removal rules and inputs:
  envelope    size_rule.json (size-rule-2, the product's md5) beside a ledger of GLED_COLS + GLED_COLS_SIZE; the search
              step's content tests pass on the outputs (gate_ledger - with size_rule_rows: EVERY row judged or recorded, a
              'gate' row's removal the rule's own verdict, a 'record' row an earlier gate's, every final within its limits -
              and image_ledger); Stokes I: the local-SNR rows judged as the fork's member pool ('gate'), the identification
              and master-structure removals recorded ('record');
  production  no record, the 35-column ledger; the SAME candidate rows (night, Stokes, pixel) as under 'envelope', and
              every row whose fate differs differs by a size verdict (the rule's, production's fixed one, or the member
              pool's) or by the one population-coupled gate after it, the row line (>= 5 finals of an epoch on one pixel row:
              a detection the rule keeps can complete a row, one it removes can break one) - the rule changes nothing but its
              own decisions;
  frames      every judged row measured again by size_rule.measure_at on the ARCHIVED search frame (float32, as
              tools/size_rule_reproduce.py reads it) with the epoch beam the engine used. The fit (core.fitting.
              gaussian_photometry: curve_fit, 7 free parameters) is ILL-CONDITIONED on some rows - mostly faint rows in
              structured regions, but not only: a 1e-7 relative change of the pixels moves it (2026-11-03, 05h_78MHz 2024-12-18
              I, 1,234 rows: median 7% in size / axis / S/N, 85 rows another verdict between the engine's float64 residual and
              its float32 archive) - so a row is STABLE when three such perturbations keep its verdict and move it < 1%; among
              the stable rows away from a limit at most 2% may get another verdict and the median size difference is <= 1% (a
              wrong pixel, beam, frame or Stokes would move most of them); the rest are counted. The recorded values are the
              rule's inputs: the post-run filters read them, never a refit.
Run (lwacalim07, niced):  python tests/test_size_rule_engine.py      (~10-20 minutes: four engine runs)
"""
import csv
import glob
import json
import os
import shutil
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'tests'))
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')

import numpy as np                                                  # noqa: E402
import test_search_admitted as SA                                   # noqa: E402  (its driver, engine(), seeds; CELL 05h_78MHz)
from exopipe.phase3.config import CFG                               # noqa: E402
from exopipe.phase3.lib.search import size_rule as SZ               # noqa: E402

T = '/tmp/exopipe3_tests_size_rule_engine'
CELL, LST, SC = SA.CELL, SA.LST, SA.SC
LABEL = 'ADMITTEST'                                                 # the driver's label
FAILS, N = [], [0]
SIZE_REASONS = set(SZ.REMOVING) | {SZ.MEMBER_REASON + v for v in SZ.REMOVING}


def check(name, ok, detail=''):
    N[0] += 1
    print(('PASS  ' if ok else 'FAIL  ') + name + ('' if ok else f'  ({str(detail)[:1500]})'), flush=True)
    if not ok:
        FAILS.append(name)


def cell_copy(root):
    """root/<CELL>: the installed cell, read only - its small records copied (CSV, JSON), every FITS and flow linked, file by
    file inside real directories (whatever the engine wrote would land here, never in the installed cell)."""
    src, dst = os.path.join(CFG.paths.COADD_SB_ROOT, CELL), os.path.join(root, CELL)
    for dp, dn, fn in os.walk(src):
        rel = os.path.relpath(dp, src)
        if rel.split(os.sep)[0] in ('variants_rebuilt', 'coadd_provisional') or rel.startswith('stale_ref'):
            dn[:] = []
            continue
        os.makedirs(os.path.join(dst, rel), exist_ok=True)
        for f in fn:
            s, d = os.path.join(dp, f), os.path.join(dst, rel, f)
            if f.endswith(('.csv', '.json', '.txt')) and os.path.getsize(s) < 50e6:
                shutil.copy2(s, d)
            else:
                os.symlink(s, d)
    return dst


def nights_of(cd):
    """Two member nights of the cell with a night flow, the served search's I and V frames (the seed's population)."""
    w = list(csv.DictReader(open(os.path.join(cd, 'coadd_final', 'weights.csv'))))
    srv = SA.served_noise()
    ok = sorted(r['date'] for r in w if r.get('in_master') == '1' and r.get('qa3_guard') == 'pass'
                and os.path.exists(os.path.join(cd, 'flow', f'{r["date"]}_scaled_flow.npy'))
                and r['date'] in srv.get('I', {}) and r['date'] in srv.get('V', {}))
    return ok[:2]


def run(name, st, nights, rule, wt, work):
    """One engine run (one Stokes, the given nights) under ``rule``, in a fresh attempt <T>/<name>; returns (result, log,
    attempt, run dir)."""
    att = os.path.join(T, name)
    shutil.rmtree(att, ignore_errors=True)
    sg = os.path.join(att, 'transients', f'Run_{LABEL}', 'sigstack')
    os.makedirs(sg, exist_ok=True)
    shutil.copy2(os.path.join(CFG.calibrate.SERVED_RUN['deep'], 'sigstack', f'{CELL}_{st}_sigstack.fits'), sg)
    keep = {k: os.environ.pop(k, None) for k in (SZ.RULE_ENV, SZ.ENVELOPE_ENV, SZ.ENVELOPE_MD5_ENV)}
    try:
        if rule == 'production':
            os.environ[SZ.RULE_ENV] = 'production'
        else:                                       # what the search step hands its engine child (deep_step.size_rule_env)
            from exopipe.phase3.lib.search import deep_step
            os.environ.update(SZ.step_env_for('deep', deep_step.size_rule_frames()))
        t1 = time.time()
        res, log = SA.engine(wt, work, att, 'admitted', nights, stokes=st)
        cnt = [ln.strip() for ln in log.splitlines() if 'gate-ledger rows' in ln]
        print(f'  engine {name}: {time.time() - t1:.0f} s wall; {cnt[-1] if cnt else "(no summary line)"}', flush=True)
        for sub in ('iresid', 'vresid'):            # the step's layout <attempt>/Run_<label>/<sub> (the harness writes <attempt>/<sub>)
            if os.path.isdir(os.path.join(att, sub)):
                os.makedirs(os.path.join(att, f'Run_{LABEL}'), exist_ok=True)
                os.symlink(os.path.join(att, sub), os.path.join(att, f'Run_{LABEL}', sub))
    finally:
        for k, v in keep.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
    return res, log, att, os.path.join(att, 'transients', f'Run_{LABEL}')


def ledger(rd):
    with open(os.path.join(rd, 'v_deep_ledger.csv'), newline='') as f:
        rdr = csv.DictReader(f)
        return list(rdr.fieldnames or []), list(rdr)


def stable_fit(img, x, y, bmaj, bmin, cd, rng, env, st, band, n=3):
    """(the archived-frame record, its verdict, stable): the row measured on the archived frame and again on ``n`` copies of its
    pixels perturbed by 1e-7 (relative, random); STABLE when every copy keeps the verdict and moves size, axis and S/N by < 1%
    (relative) - else the fit is ill-conditioned there and float32 rounding moves it as much."""
    R = SZ.rms_radius(bmaj / cd)                       # the pixels any stamp of the row reads (frame_stamp: R, else 6 beams)
    y0, x0 = max(0, y - R - 1), max(0, x - R - 1)
    box = img[y0:y + R + 2, x0:x + R + 2]              # the same stamp, the same edge fallback as on the whole frame
    recs = [SZ.measure_at(box if k == 0 else box.astype(np.float64) * (1 + 1e-7 * rng.standard_normal(box.shape)),
                          x - x0, y - y0, bmaj, bmin, cd) for k in range(n + 1)]
    verdicts = [SZ.judge(env, st, band, q)['size_verdict'] for q in recs]
    if any(q is None for q in recs):
        return recs[0], verdicts[0], all(q is None for q in recs)
    spread = max(abs(q[k] - recs[0][k]) / max(abs(recs[0][k]), 1e-9) for q in recs[1:] for k in ('size', 'axis', 'snr'))
    return recs[0], verdicts[0], spread < 0.01 and len(set(verdicts)) == 1


def frame_checks(rows, att, st, cd):
    """(problems, counts) of the engine's recorded values against the archived frame (module docstring 'frames'): every judged
    row (verdict known) measured with stable_fit; among the STABLE ones (no recorded excess within 2% of a limit) at most 2%
    another verdict and the median relative size difference <= 1% - a wrong pixel, beam, frame or Stokes would move most of
    them; the ill-conditioned ones counted (not reproducible from a float32 frame by construction)."""
    from astropy.io import fits
    c = dict(n=0, unstable=0, stable=0, near=0, differ=0)
    dsize = []
    rng = np.random.default_rng(20261103)
    env = SZ.Envelope.load(SZ.resolve(CFG.search_detect.SIZE_ENVELOPE), CFG.search_detect.SIZE_ENVELOPE_MD5).require()
    by = {}
    for r in rows:
        if r.get('size_verdict') not in ('fit-failed', 'no-snr', ''):
            by.setdefault(r['date'], []).append(r)
    examples = []
    for d, rs in sorted(by.items()):
        fp = os.path.join(att, 'iresid' if st == 'I' else 'vresid', CELL, f'{d}_{st}_redo_resid_{SC}.fits')
        sub = fits.getheader(os.path.join(SA.CFG.paths.COADD_SB_ROOT, CELL, 'subtracted', f'{d}_{st}_sub_{SC}.fits'))
        with fits.open(fp, memmap=True) as h:
            img = np.array(h[0].data, dtype=np.float32)          # a copy: the file closes below
            cdelt = abs(float(h[0].header['CDELT1']))
        img = img.reshape(img.shape[-2:])
        for r in rs:
            c['n'] += 1
            rec, v, ok = stable_fit(img, int(float(r['pix_x'])), int(float(r['pix_y'])), float(sub['BMAJ']), float(sub['BMIN']),
                                    cdelt, rng, env, st, SC)
            if not ok:
                c['unstable'] += 1
                continue
            if min(abs(float(r['size_excess']) - 1), abs(float(r['axis_excess']) - 1)) < 0.02:
                c['near'] += 1
                continue
            c['stable'] += 1
            if v != r['size_verdict']:
                c['differ'] += 1
                examples.append(f"{st} {d} ({r['pix_x']},{r['pix_y']}): engine {r['size_verdict']} (size {r['size_beams']}, axis "
                                f"{r['axis_norm']}, S/N {r['size_snr']}), archived frame {v}")
            if rec is not None and r.get('size_beams') not in ('', None):
                dsize.append(abs(float(r['size_beams']) - rec['size']) / max(rec['size'], 1e-9))
    probs = []
    if c['stable'] < 20:
        probs.append(f"only {c['stable']} stable rows away from a limit (no power)")
    if c['differ'] > max(1, 0.02 * c['stable']):
        probs.append(f"{c['differ']} of {c['stable']} stable rows get another verdict on the archived frame: {examples[:3]}")
    if dsize and float(np.median(dsize)) > 0.01:
        probs.append(f'median relative size difference {float(np.median(dsize)):.4f} > 1%')
    c['median_dsize'] = round(float(np.median(dsize)), 5) if dsize else None
    return probs, c


def main():
    t0 = time.time()
    shutil.rmtree(T, ignore_errors=True)
    os.makedirs(T)
    SA.T = T
    from exopipe.phase3.lib.search import tests as ST
    try:
        wt, work = os.path.join(T, 'wt'), os.path.join(T, 'work')
        cd = cell_copy(wt)
        nights = nights_of(cd)
        check(f'setup: two member nights of {CELL} with flows and served frames ({nights})', len(nights) == 2, nights)
        if len(nights) != 2:
            return 1
        for st in ('I', 'V'):
            re_, le_, ae_, rde = run(f'attempt_{st}_envelope', st, nights, 'envelope', wt, work)
            rp_, lp_, ap_, rdp = run(f'attempt_{st}_production', st, nights, 'production', wt, work)
            check(f'{st}: the engine runs under the rule and under production', re_['ok'] and rp_['ok'],
                  (re_.get('error'), le_[-800:], rp_.get('error'), lp_[-800:]))
            if not (re_['ok'] and rp_['ok']):
                continue
            rec = SZ.read_run_record(rde)
            check(f'{st}: the run records its rule (size-rule-2, the configured product), production none',
                  rec['rule'] == 'envelope' and rec['version'] == SZ.RECORD_VERSION
                  and rec['md5'] == CFG.search_detect.SIZE_ENVELOPE_MD5 and not os.path.exists(os.path.join(rdp, SZ.RUN_RECORD)),
                  (rec, os.listdir(rdp)))
            ce, re = ledger(rde)
            cp, rp = ledger(rdp)
            check(f"{st}: ledgers - GLED_COLS + the rule's columns under the rule ({len(re)} rows), production's GLED_COLS "
                  f"({len(rp)} rows)", ce == ST.GLED_COLS + ST.GLED_COLS_SIZE and cp == ST.GLED_COLS and re and rp,
                  (len(ce), len(cp), len(re), len(rp)))
            ct = [ST.complete(os.path.join(rde, 'v_deep_ledger.csv'), stokes=st, cell=CELL, tier='deep'),
                  ST.one_row_per_searched_epoch(os.path.join(rde, 'v_deep_images.csv'), stokes=st, cell=CELL, tier='deep'),
                  ST.complete(os.path.join(rdp, 'v_deep_ledger.csv'), stokes=st, cell=CELL, tier='deep')]
            check(f"{st}: the search step's content tests pass on the engine's outputs under the rule (gate ledger with every "
                  f"row judged or recorded, image ledger) and under production", all(x[0] for x in ct), [x[2][:400] for x in ct])
            roles = {}
            for r in re:
                roles.setdefault((r['size_role'], r['veto_reason'].split(':')[0] if r['veto_reason'] else 'final'), 0)
                roles[(r['size_role'], r['veto_reason'].split(':')[0] if r['veto_reason'] else 'final')] += 1
            print(f'  {st}: roles x reasons under the rule: ' + json.dumps({f'{a} {b}': n for (a, b), n in sorted(roles.items())}))
            gate_snr = [r for r in re if r['veto_reason'].startswith('reject_local_snr')]
            if st == 'I':
                check('I: every local-SNR row judged as the fork\'s member pool (gate), every identification / master-structure '
                      'removal recorded', gate_snr and all(r['size_role'] == 'gate' for r in gate_snr)
                      and all(r['size_role'] == 'record' for r in re if r['veto_reason'] in ('mstruct_I', 'reject_merged',
                                                                                              'reject_sigma_stack')),
                      roles)
            else:
                check('V: every identification removal recorded', all(r['size_role'] == 'record' for r in re
                                                                     if r['veto_reason'].startswith('reject_')), roles)
            # the same candidate rows; every difference of fate a size verdict
            key = lambda r: (r['date'], r['stokes'], r['pix_x'], r['pix_y'], r['ra'], r['dec'])
            ke = {}
            for r in re:
                ke.setdefault(key(r), []).append(r)
            kp = {}
            for r in rp:
                kp.setdefault(key(r), []).append(r)
            diff = []
            for k in set(ke) | set(kp):
                a, b = sorted(x['veto_reason'] for x in ke.get(k, [])), sorted(x['veto_reason'] for x in kp.get(k, []))
                if len(a) != len(b):
                    diff.append((k, 'rows', a, b))
                    continue
                for x, y in zip(a, b):
                    if x != y and not ({x, y} & (SIZE_REASONS | set(SZ.REMOVING) | {'row-line'})):
                        diff.append((k, 'fate', x, y))
            chg = [(k, sorted(x['veto_reason'] for x in ke[k]), sorted(x['veto_reason'] for x in kp[k])) for k in ke
                   if k in kp and sorted(x['veto_reason'] for x in ke[k]) != sorted(x['veto_reason'] for x in kp[k])]
            nrl = sum(1 for _k, a, b in chg if 'row-line' in a + b and not set(a + b) & SIZE_REASONS)
            check(f'{st}: the same candidate rows under both rules, every changed fate a size verdict or the row line it '
                  f'couples ({len(chg)} changed of {len(ke)}, {nrl} by the row line)', not diff, diff[:5])
            probs, fc = frame_checks(re, ae_, st, cd)
            check(f"{st}: {fc['n']} judged rows measured again on the archived float32 frames - {fc['unstable']} ill-conditioned "
                  f"(a 1e-7 perturbation moves the fit > 1%: not reproducible from a float32 frame), {fc['near']} within 2% of a "
                  f"limit, {fc['stable']} stable: {fc['differ']} another verdict, median size difference {fc['median_dsize']}",
                  not probs, probs)
    finally:
        if os.environ.get('EXOPIPE3_TEST_KEEP') != '1':
            shutil.rmtree(T, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed in {time.time() - t0:.0f} s')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
