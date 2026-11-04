#!/usr/bin/env python3
"""PARITY of the combined snapshot view (`serve --tier snapshot --family comb`, lib/serve/snapshot_view.py) against PRODUCTION's
own builder and driver (exopipe_ops snapshot_combined_view.py + snapshot_batch/comb_build_20261102.sh steps 2-3), on the same
inputs, into scratch. Spec: docs/specs/snapshot_combined_view.md, "Parity target". Never writes /data07 or a production root.

    PY=/opt/devel/pipeline/envs/py38_orca_nkosogor/bin/python3
    $PY tests/parity_snapshot_view.py prod    WORK REFUSALS FLEET RUN [RUN ...]   production's side -> WORK/prod/<VIEW>
    $PY tests/parity_snapshot_view.py port    WORK REFUSALS FLEET RUN [RUN ...]   the stage (this tree)  -> WORK/port/snapshot/<VIEW>
    $PY tests/parity_snapshot_view.py compare PROD_VIEW PORT_VIEW [OUT_JSON]          file by file

  REFUSALS  a copy of the deep refusal record (production's /fast/gh/main/phase3/Run_P3S5/refusals is on lwacalim07 only)
  FLEET     a copy of the fleet records of the runs' LSTs (noise_*_median*.csv, wideband_members_*.csv), frozen: the live tree
            is written by the snapshot chat and both sides must decide on the same files
  RUN       the served batch runs, in the view's order - COPIES on the scratch file system (the panels are hard-linked from
            them by production; read only by both sides)
prod: production's snapshot_combined_view.py run UNCHANGED under its interpreter (site PYTHON 'py36'); the only thing set is
      snapshot_deep_omission.decide's two data roots (REFUSALS, FLEET) before the script imports the module; then
      snapshot_filter_catalog.py over the runs' fit stores (catalog_snapshot.BATCH_DIR/rechain_<run>/fits_store.csv) and the
      driver's card2 / FILTER-INPUTS heredoc, VERBATIM (read from comb_build_20261102.sh); then snapshot_localize_tiers.sh with
      P3_LOC_SHARDS=6 SNAP_LOC_FIG_PROCS=6 (speed only; production's driver used 12 shards on /data07 - 6 keeps to the Lustre
      rule: at most 6 readers).
port: `exopipe3 serve --tier snapshot --family comb --stages omit,build,filtercat,localize,ready` of THIS tree as a --parity run
      with the production parity file, --set catalog_omission.REFUSALS=REFUSALS paths.SNAPSHOT_FLEET_OUT=FLEET and the same
      localization environment. WORK must lie outside every production root (the CLI refuses /lustre/gh/main/phase3 ...).
      Under --parity the guard refuses a hard link of a file outside the parity roots, so the stage COPIES the panels there
      (recorded as panels_copied; a production view links them).
compare: every relative path present in both views; a panel (a file of a candidates/<S>/ page directory other than
      detections.csv and astrometry.png) the same inode, else the same bytes; combined_view_summary.json equal but for 'seconds';
      every other file byte for byte; the directory sets equal. Prints a table per file class and every difference (up to 50
      per class in OUT_JSON); exit 0 only on no difference.
Run on lwacalim04 (the review host is busy), nice, on the recorder-free cores.
"""
import collections
import filecmp
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPS = '/lustre/gh/main/exopipe_ops'
PY36 = '/opt/devel/pipeline/envs/development/bin/python'
PY38 = '/opt/devel/pipeline/envs/py38_orca_nkosogor/bin/python3'
VIEW = 'Run_SNAPCOMB_PARITY_20261103'
LOC_ENV = {'P3_LOC_SHARDS': '6', 'SNAP_LOC_FIG_PROCS': '6'}
FORBIDDEN = ('/data07', '/lustre/gh/main/exopipe_ops', '/lustre/gh/main/clv15', '/lustre/gh/main/exopipe/')

PROD_HARNESS = r'''
import os, runpy, sys
OPS = '/lustre/gh/main/exopipe_ops'
rdir, proto, out, runs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
sys.path[:] = [p for p in sys.path if p not in ('', os.path.dirname(os.path.abspath(__file__)))]
sys.path.insert(0, OPS)                       # as `python snapshot_combined_view.py` run from exopipe_ops
import snapshot_deep_omission as SDO
SDO.decide.__defaults__ = (rdir, proto)       # the only change: the decisions' two data roots
sys.argv = [os.path.join(OPS, 'snapshot_combined_view.py'), out] + runs
runpy.run_path(os.path.join(OPS, 'snapshot_combined_view.py'), run_name='__main__')
'''


def _refuse_path(p):
    rp = os.path.realpath(p)
    if any(rp == f.rstrip('/') or rp.startswith(f.rstrip('/') + '/') for f in FORBIDDEN):
        sys.exit(f'REFUSING: {p} is under a production tree ({rp})')


def _env():
    e = {k: v for k, v in os.environ.items() if not k.startswith(('P3_', 'SNAP', 'SUBLST_', 'EXOPIPE3_PARITY', 'EXOPIPE3_INJ'))}
    e['NUMPY_MADVISE_HUGEPAGE'] = '0'
    return e


def driver_heredoc():
    """comb_build_20261102.sh's step-2 heredoc (between <<'PYC' and PYC), verbatim."""
    src = open(os.path.join(OPS, 'snapshot_batch', 'comb_build_20261102.sh')).read()
    return src.split("<<'PYC'", 1)[1].split('\nPYC\n', 1)[0].split('\n', 1)[1]


def fit_stores(runs):
    return [os.path.join(OPS, 'snapshot_batch', f'rechain_{os.path.basename(r.rstrip("/"))}', 'fits_store.csv') for r in runs]


def prod(work, rdir, fleet, runs):
    _refuse_path(work)
    out = os.path.join(work, 'prod', VIEW)
    if os.path.exists(out) or os.path.exists(out + '.part'):
        sys.exit(f'REFUSING: {out} exists')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    h = os.path.join(work, 'prod_harness.py')
    with open(h, 'w') as f:
        f.write(PROD_HARNESS)
    run = lambda argv, **k: subprocess.run(argv, env=_env(), **k)
    r = run(['nice', PY36, '-u', h, rdir, fleet, out] + list(runs))
    if r.returncode or not os.path.isdir(out):
        sys.exit('PROD BUILD REFUSED/FAILED')
    stores = fit_stores(runs)
    for s in stores:
        if not (os.path.isfile(s) and os.path.getsize(s)):
            sys.exit(f'REFUSING: no fit store {s}')
    if run(['nice', PY36, 'snapshot_filter_catalog.py', out] + stores, cwd=OPS).returncode:
        sys.exit('FILTER-CATALOG FAILED')
    if run(['nice', PY36, '-', out] + list(runs), input=driver_heredoc(), universal_newlines=True, cwd=OPS).returncode:
        sys.exit('FILTER-INPUTS check FAILED')
    log = os.path.join(work, 'prod', 'localize.log')
    with open(log, 'w') as lf:
        rc = subprocess.run(['nice', 'bash', 'snapshot_localize_tiers.sh', out], cwd=OPS, env=dict(_env(), **LOC_ENV),
                            stdout=lf, stderr=subprocess.STDOUT).returncode
    if rc or 'LOCALIZE-TIERS-OK' not in open(log).read():
        sys.exit(f'LOCALIZATION FAILED ({log})')
    print(f'PROD VIEW-READY-FOR-HIDES {out}')


def port(work, rdir, fleet, runs):
    _refuse_path(work)
    out = os.path.join(work, 'port')
    loc = "{%s}" % ', '.join(f"'{k}': '{v}'" for k, v in LOC_ENV.items())
    argv = [PY38, '-m', 'exopipe.phase3', 'serve', '--tier', 'snapshot', '--family', 'comb', '--run', VIEW,
            '--view-runs', ','.join(runs), '--out', out, '--set', f'catalog_omission.REFUSALS={rdir}',
            '--set', f'paths.SNAPSHOT_FLEET_OUT={fleet}', '--set', f'serve.SNAPVIEW_LOC_ENV={loc}', '--parity',
            '--config', os.path.join(REPO, 'exopipe', 'phase3', 'parity', 'production_20261013.cfg'),
            '--stages', 'omit,build,filtercat,localize,ready']
    e = dict(_env(), PYTHONPATH=REPO, EXOPIPE3_SITE=os.environ.get('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py'))
    rc = subprocess.run(['nice'] + argv, cwd=REPO, env=e).returncode
    print(f'PORT exit {rc}: {os.path.join(out, "snapshot", VIEW)}')
    return rc


def cls(rel):
    """The class of a view file (the comparison's table)."""
    p = rel.split(os.sep)
    if p[0].startswith('queue_'):
        return 'queue'
    if p[0] == 'source_catalogs':
        return 'general catalogue'
    if p[0].startswith('source_catalogs_loc_'):
        return 'localization (%s)' % ('figure' if p[-1].endswith('.png') else p[-1])
    if '_locin_' in p[0]:
        return 'localization input'
    if p[0].startswith('source_catalogs_'):
        return 'per-tier view'
    if p[0] == 'candidates' and len(p) > 1 and p[1] == '_identity':
        return 'identity detections'
    if p[0] == 'candidates' and p[-1] == 'detections.csv':
        return 'page detections'
    if p[0] == 'candidates' and p[-1] == 'astrometry.png':
        return 'page astrometry figure'
    if p[0] == 'candidates':
        return 'panel'
    if p[0].startswith('deep_omission'):
        return 'omission'
    if p[0] in ('master_catalog.csv', 'card2_filter.csv', 'coadd_root.txt'):
        return 'filter'
    return p[0]


def _tree(d):
    files, dirs = {}, set()
    for dp, _dn, fns in os.walk(d):
        dirs.add(os.path.relpath(dp, d))
        for fn in fns:
            files[os.path.relpath(os.path.join(dp, fn), d)] = os.path.join(dp, fn)
    return files, dirs


def _same(rel, pa, pb):
    """(equal, seconds) of one file present in both views (the module docstring's rules)."""
    if rel == 'combined_view_summary.json':
        ja, jb = json.load(open(pa)), json.load(open(pb))
        sec = (ja.pop('seconds', None), jb.pop('seconds', None))
        return ja == jb and list(ja) == list(jb), sec
    if cls(rel) == 'panel' and os.stat(pa).st_ino == os.stat(pb).st_ino:
        return True, None
    return filecmp.cmp(pa, pb, shallow=False), None


def compare(a, b, out_json=None, workers=6):
    """The comparison (module docstring), on ``workers`` reader threads (the Lustre rule: at most 6)."""
    from concurrent.futures import ThreadPoolExecutor
    ta, da = _tree(a)
    tb, db = _tree(b)
    n, same, diff = collections.Counter(), collections.Counter(), collections.defaultdict(list)
    seconds = None
    rels = sorted(set(ta) | set(tb))
    both = [r for r in rels if r in ta and r in tb]
    with ThreadPoolExecutor(workers) as ex:
        res = dict(zip(both, ex.map(lambda r: _same(r, ta[r], tb[r]), both, chunksize=64)))
    for rel in rels:
        c = cls(rel)
        n[c] += 1
        if rel not in res:
            diff[c].append(('only in ' + ('port' if rel not in ta else 'prod'), rel))
            continue
        ok, sec = res[rel]
        if sec is not None:
            seconds = sec
        if ok:
            same[c] += 1
        else:
            diff[c].append(('content', rel))
    tot = sum(len(v) for v in diff.values()) + len(da ^ db)
    for c in sorted(n):
        print('%-40s %7d files, %7d identical, %5d different' % (c, n[c], same[c], len(diff[c])))
    print(f'directories: prod {len(da)}, port {len(db)}, only prod {len(da - db)}, only port {len(db - da)}')
    print(f'summary seconds (prod, port): {seconds}')
    for c, v in sorted(diff.items()):
        for x in v[:10]:
            print('  DIFF %s: %s' % (c, x))
    print('PARITY %s: %d files (prod %d, port %d), %d differences'
          % ('PASS' if not tot else 'DIFFERENCES', len(set(ta) | set(tb)), len(ta), len(tb), tot))
    if out_json:
        with open(out_json, 'w') as f:
            json.dump({'prod': a, 'port': b, 'files': {'prod': len(ta), 'port': len(tb)},
                       'directories': {'prod': len(da), 'port': len(db), 'only_prod': sorted(da - db)[:50],
                                       'only_port': sorted(db - da)[:50]},
                       'classes': {c: {'files': n[c], 'identical': same[c], 'different': len(diff[c])} for c in sorted(n)},
                       'differences': {c: v[:50] for c, v in diff.items()}, 'summary_seconds': seconds}, f, indent=1)
    return 0 if not tot else 1


def main(argv):
    if len(argv) >= 6 and argv[1] in ('prod', 'port'):
        work, rdir, fleet, runs = argv[2], os.path.abspath(argv[3]), os.path.abspath(argv[4]), [os.path.abspath(r) for r in argv[5:]]
        return prod(os.path.abspath(work), rdir, fleet, runs) if argv[1] == 'prod' else port(os.path.abspath(work), rdir, fleet, runs)
    if len(argv) in (4, 5) and argv[1] == 'compare':
        return compare(argv[2], argv[3], argv[4] if len(argv) == 5 else None)
    sys.exit(__doc__)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
