#!/usr/bin/env python
"""Allocation-safe Orca launcher for the gh account (2026-07-14).

Patches NVME_BASE_DIR ('/fast/pipeline', pipeline-account-owned) to
'/fast/gh/main' in BOTH the config module and every module that imported the
constant, then runs either a celery worker or the subband_celery controller.

Usage:
  python orca_gh.py worker  -Q calim07 -n gh07@lwacalim07 -c 10
  python orca_gh.py control --range 14-15 --date ... (subband_celery args)
"""
import sys

sys.path.insert(0, '/opt/devel/nkosogor/nkosogor/distributed-pipeline')

GH_BASE = '/fast/gh/main'

import orca.resources.subband_config as sc
sc.NVME_BASE_DIR = GH_BASE
import orca.tasks.subband_tasks as st
if hasattr(st, 'NVME_BASE_DIR'):
    st.NVME_BASE_DIR = GH_BASE
# archive to a gh-writable Lustre area (the default targets the
# pipeline-account archive and throws Errno 13)
for _m in (st,):
    for _k in dir(_m):
        if 'ARCHIVE' in _k and isinstance(getattr(_m, _k), str)                 and getattr(_m, _k).startswith('/lustre/pipeline'):
            setattr(_m, _k, '/lustre/gh/main/hwjms_p1')
# pre-validate MS lists before concat so corrupt files are dropped in ONE
# pass instead of one-per-retry (heal cap was exhausted on 69MHz 14h)
import orca.transform.subband_processing as _sp
_orig_concat = _sp.concatenate_ms
def _deep_repair(ms):
    """Rewrite an MS via deep copy — fixes storage-manager-level damage that
    logical reads miss but ms::concatenate trips on (root-caused 2026-07-15,
    10/358 files on 69MHz 14h; repair-not-prune, zero data loss)."""
    import shutil, os
    from casacore.tables import table as _ct
    fix = ms[:-3] + '_fx.ms'
    if os.path.exists(fix): shutil.rmtree(fix)
    t = _ct(ms, ack=False); t.copy(fix, deep=True); t.close()
    shutil.rmtree(ms); os.rename(fix, ms)

def _concat_prevalidated(ms_files, work_dir, subband, max_retries=10, **kw):
    """Drop-in for concatenate_ms: casatasks.concat directly with
    deep-repair-on-blame — NEVER prunes or deletes inputs."""
    import os, re, shutil
    from casatasks import concat as _concat
    from casacore.tables import table as _ct, addImagingColumns as _aic
    out = os.path.join(work_dir, f"{subband}_concat.ms")
    mslist = sorted(ms_files)
    # ROOT CAUSE (2026-07-15): concat MUTATES inputs (adds missing MODEL_DATA
    # scratch columns mid-append) — a failure mid-mutation bucket-corrupts the
    # input, so failed concats mint new "corrupt" files. Pre-add the imaging
    # columns so concat has nothing to mutate.
    for m in mslist:
        try:
            t = _ct(m, ack=False); need = 'MODEL_DATA' not in t.colnames(); t.close()
            if need: _aic(m, ack=False)
        except Exception:
            _deep_repair(m)
            try: _aic(m, ack=False)
            except Exception: pass
    for attempt in range(30):
        if os.path.exists(out):
            shutil.rmtree(out)
        clog = os.path.join(work_dir, f'_gh_concat_{attempt}.casalog')
        try:
            from casatasks import casalog as _cl
            _cl.setlogfile(clog)
        except Exception:
            pass
        try:
            _concat(vis=mslist, concatvis=out, timesort=True)
            return out
        except Exception as e:
            msg = str(e)
        blamed = None
        try:
            apps = re.findall(r'Appending (/\S+_averaged\.ms)', open(clog).read())
            if apps and apps[-1] in mslist:
                blamed = apps[-1]
        except Exception:
            pass
        if blamed is None:
            print(f'[orca_gh] concat failed, no blamable file: {msg[:120]}')
            return None
        print(f'[orca_gh] deep-repairing {os.path.basename(blamed)} (attempt {attempt})')
        _deep_repair(blamed)
    print('[orca_gh] concat failed after 30 repairs')
    return None
_sp.concatenate_ms = _concat_prevalidated
if hasattr(st, 'concatenate_ms'):
    st.concatenate_ms = _concat_prevalidated
try:
    import orca.transform.subband_processing as sp
    if hasattr(sp, 'NVME_BASE_DIR'):
        sp.NVME_BASE_DIR = GH_BASE
except Exception:
    pass

mode = sys.argv[1]
argv = sys.argv[2:]

if mode == 'worker':
    from orca.celery import app
    app.worker_main(['worker', '--loglevel=INFO'] + argv)
elif mode == 'control':
    import runpy
    sys.argv = ['subband_celery.py'] + argv
    runpy.run_path('/opt/devel/nkosogor/nkosogor/distributed-pipeline/'
                   'pipeline/subband_celery.py', run_name='__main__')
else:
    raise SystemExit('mode must be worker|control')
