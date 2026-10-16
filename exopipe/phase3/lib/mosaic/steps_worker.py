"""Child process of `exopipe3 mosaic`: one task per process, `python -m exopipe.phase3.lib.mosaic.steps_worker TASK.json`.

Started by steps_common.run_tasks through core.legacy.run (site interpreter, P3_* cleared, NUMPY_MADVISE_HUGEPAGE=0, parity
guard, code fingerprint at exit). A task pins itself to its lane's cores, installs the legacy finder (old dotted names such
as exopipe.phase3.pb_apply resolve to the moved modules), runs the production code IN PLACE, and writes a JSON result:
{'status': 'ok' | 'refused' | 'error', 'message': ..., ...}. The process exits 0 whenever it wrote a result.

Task kinds (each calls exactly what production calls, with the same arguments):
  stage    exopipe_ops/mosaic_stage_from_lustre.py run as __main__ (runpy) with MOSAIC_STAGE=<stage>, MOSAIC_BANDS=<scope>
           and argv [script, '0', '1'] - symlinks of the scope's current masters + the weighting PB maps (STG:31-95).
  census   per tile of one job, the engine's own guard arithmetic on the staged master x weighting PB (CJ:200-221,
           identical to the conformance pass membership CP:104-113): verdict, sigma, weight 1/sigma^2.
  coadd    lib/mosaic/coadd_jysr.coadd_jysr(lsts, variant, scope, root=<stage>, workroot=None, last_it=None,
           out_dir=<tmp>, tag='', product='I', domain=D) after setting the WINDOW-GLOBAL module globals; stdout tee'd and
           parsed for the per-tile verdict lines ('[lst] added', 'MISSING', 'no valid pixels', 'CORRUPT-FLAT').
  render   lib/mosaic/band_products.build(variant, <link dir>, '_' + D, <tmp>, bands=[scope], moll_W, car_cdelt) with
           BEAMS = mosaic_render.BEAM_FALLBACK_ARCMIN.
  qa       exopipe_ops/mosaic_v2_flux_anchor_measure.py as __main__ with argv [script, <link dir>, <anchor csv>]; then
           verify_mosaic.astrometry_check(scope, map, nsrc, minflux) per jysr map -> the VLSSr table.
"""
import json
import os
import re
import sys
import traceback


class _Tee:
    """stdout that keeps every line it passes on (the engine prints its per-tile verdicts)."""

    def __init__(self, real):
        self.real, self.buf, self.lines = real, '', []

    def write(self, s):
        self.real.write(s)
        self.buf += s
        while '\n' in self.buf:
            line, self.buf = self.buf.split('\n', 1)
            self.lines.append(line)
        return len(s)

    def flush(self):
        self.real.flush()


class _Refused(Exception):
    pass


def _runpy(script, argv, env=None):
    import runpy
    from exopipe.phase3.core.legacy import refuse_patch_path
    from exopipe.phase3.core.record import Refusal
    try:                                       # B-01 (b): no patch path, also for scripts run by path
        refuse_patch_path(script, 'mosaic worker')
    except Refusal as e:                       # the worker's own refusal (status 'refused'), not an error
        raise _Refused(str(e))
    saved_argv, saved_path = list(sys.argv), list(sys.path)
    saved_env = {k: os.environ.get(k) for k in (env or {})}
    try:
        sys.argv = [script] + [str(a) for a in argv]
        os.environ.update({k: str(v) for k, v in (env or {}).items()})
        try:
            g = runpy.run_path(script, run_name='__main__')
        except SystemExit as e:
            if e.code not in (None, 0):
                raise _Refused(str(e.code))
            g = {}
        return g
    finally:
        sys.argv, sys.path[:] = saved_argv, saved_path
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def task_stage(t):
    g = _runpy(t['script'], ['0', '1'], env={'MOSAIC_STAGE': t['stage'], 'MOSAIC_BANDS': t['scope']})
    return {'status': 'ok', 'ROOT': g.get('ROOT'), 'VARMAP': g.get('VARMAP'), 'links': g.get('done'),
            'absent': g.get('skipped'), 'pb_failures': g.get('pberr')}


def task_census(t):
    import numpy as np
    from astropy.io import fits
    floor, min_px, thr = float(t['pb_floor']), int(t['min_valid_px']), float(t['flat_sigma'])
    tiles = {}
    for lst in t['lsts']:
        mp = os.path.join(t['stage'], lst, t['master_name'])
        pbp = os.path.join(t['stage'], lst, t['pb_name'])
        if not (os.path.exists(mp) and os.path.exists(pbp)):
            tiles[lst] = {'verdict': 'MISSING'}
            continue
        raw = fits.getdata(mp).squeeze().astype(np.float32)
        fin = np.isfinite(raw)
        v = raw[fin]
        mad_i = float(1.4826 * np.median(np.abs(v - np.median(v)))) if v.size else float('nan')
        intr = np.nan_to_num(raw)                                          # CJ:208 / CP:104
        pb = np.nan_to_num(fits.getdata(pbp).squeeze().astype(np.float32))   # CJ:200 / CP:105
        app = (intr * pb)[pb > floor]                                      # CJ:209
        app = app[np.isfinite(app)]                                        # CJ:210
        rec = {'finite_any': bool(fin.any()), 'mad_intrinsic': mad_i, 'n_app': int(app.size),
               'master_realpath': os.path.realpath(mp), 'pb_shape': list(pb.shape), 'master_shape': list(raw.shape)}
        if app.size < min_px:                                              # CJ:211
            rec['verdict'] = 'no valid pixels'
        else:
            med = np.median(app)
            sigma = 1.4826 * np.median(np.abs(app - med)) or 1e-6            # CJ:214 / CP:111
            if not np.isfinite(sigma) or sigma < thr:                      # CJ:219
                rec['verdict'] = 'CORRUPT-FLAT'
            else:
                rec['verdict'] = 'added'
            rec['sigma'] = float(sigma)
            rec['weight'] = float(1.0 / sigma ** 2)                        # CP:118 WEIGHT
        tiles[lst] = rec
    return {'status': 'ok', 'tiles': tiles}


_VERDICT = re.compile(r'^\[(\d\dh)\] (added|MISSING|no valid pixels|CORRUPT-FLAT|CORRUPT \(census\))')


def task_coadd(t):
    from exopipe.phase3.lib.mosaic import coadd_jysr as mcj
    for k, v in t['globals'].items():
        setattr(mcj, k, v)
    mcj.BEAM_OVERRIDE.clear()                 # July repair-ledger modes are never used by v2 (retired keys, spec 3)
    mcj.SKIP_CELLS.clear()
    os.makedirs(t['out_tmp'], exist_ok=True)
    tee = _Tee(sys.stdout)
    sys.stdout = tee
    try:
        try:
            mcj.coadd_jysr(list(t['lsts']), t['variant'], t['scope'], root=t['stage'], workroot=None, last_it=None,
                           out_dir=t['out_tmp'], tag='', product='I', domain=t['domain'])
        except SystemExit as e:              # 'NO TILES CONTRIBUTED' (CJ:240-242) -> gate G10
            raise _Refused(f'G10: {e.code}')
    finally:
        sys.stdout = tee.real
    verdicts, wrote = {}, []
    for line in tee.lines:
        m = _VERDICT.match(line)
        if m:
            verdicts[m.group(1)] = 'CORRUPT-FLAT' if m.group(2).startswith('CORRUPT-FLAT') else m.group(2)
        elif line.startswith('WROTE '):
            wrote.append(line[6:].strip())
    return {'status': 'ok', 'verdicts': verdicts, 'wrote': wrote}


def task_render(t):
    from exopipe.phase3.lib.mosaic import band_products as mbp
    mbp.BEAMS = dict(t['beams'])
    os.makedirs(t['out_tmp'], exist_ok=True)
    mbp.build(t['variant'], t['hpx_dir'], '_' + t['domain'], t['out_tmp'], bands=[t['scope']], moll_W=int(t['moll_W']),
              car_cdelt=float(t['car_cdelt_deg']), dec_cut=t.get('dec_cut'))
    return {'status': 'ok', 'files': sorted(os.listdir(t['out_tmp']))}


def _vlssr_row(m, nsrc, minflux):
    import numpy as np
    from exopipe.phase3.lib.mosaic import verify_mosaic as vm
    row = {'variant': m['variant'], 'band': m['scope'], 'n_matched': 0}
    try:
        dra, ddec, _cat = vm.astrometry_check(m['scope'], m['jysr'], nsrc=nsrc, minflux=minflux)
    except Exception as e:                      # e.g. no match at all: sep.max() of an empty array (VM:79)
        row['note'] = f'{type(e).__name__}: {e}'[:120]
        return row
    dra, ddec = np.asarray(dra, float), np.asarray(ddec, float)
    sep = np.hypot(dra, ddec)

    def mad(x):
        return 1.4826 * np.median(np.abs(x - np.median(x)))
    row.update(n_matched=int(len(dra)), dra_med_arcmin=f'{np.median(dra):.3f}', dra_mad_arcmin=f'{mad(dra):.3f}',
               ddec_med_arcmin=f'{np.median(ddec):.3f}', ddec_mad_arcmin=f'{mad(ddec):.3f}',
               offset_med_arcmin=f'{np.median(sep):.3f}', offset_p90_arcmin=f'{np.percentile(sep, 90):.3f}',
               offset_max_arcmin=f'{sep.max():.3f}')
    return row


VLSSR_COLUMNS = ['variant', 'band', 'n_matched', 'dra_med_arcmin', 'dra_mad_arcmin', 'ddec_med_arcmin',
                 'ddec_mad_arcmin', 'offset_med_arcmin', 'offset_p90_arcmin', 'offset_max_arcmin', 'note']


def task_qa(t):
    import csv
    _runpy(t['anchor_script'], [t['link_dir'], t['anchor_out']])
    rows = [_vlssr_row(m, int(t['nsrc']), float(t['minflux'])) for m in t['maps']]
    tmp = t['vlssr_out'] + f'.tmp.{os.getpid()}'
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=VLSSR_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, t['vlssr_out'])
    return {'status': 'ok', 'vlssr_rows': len(rows)}


KINDS = {'stage': task_stage, 'census': task_census, 'coadd': task_coadd, 'render': task_render, 'qa': task_qa}


def main(argv):
    with open(argv[0]) as f:
        t = json.load(f)
    if t.get('cores'):
        try:
            os.sched_setaffinity(0, set(int(c) for c in t['cores']))
        except (AttributeError, OSError):
            pass
    try:
        from exopipe.phase3.core import env
        env.legacy()
        res = KINDS[t['kind']](t)
    except _Refused as e:
        res = {'status': 'refused', 'message': str(e)}
    except Exception as e:
        res = {'status': 'error', 'message': f'{type(e).__name__}: {e}', 'traceback': traceback.format_exc()[-3000:]}
    tmp = t['result'] + f'.tmp.{os.getpid()}'
    with open(tmp, 'w') as f:
        json.dump(res, f, indent=1, default=str)
    os.replace(tmp, t['result'])
    sys.stdout.flush()
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
