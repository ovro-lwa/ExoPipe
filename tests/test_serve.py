#!/usr/bin/env python3
"""Tests of `exopipe3 serve` (unit serve). Plain asserts, no pytest; run with the py38 interpreter:

    NUMPY_MADVISE_HUGEPAGE=0 EXOPIPE3_SITE=/lustre/gh/main/exopipe_ops/exopipe3_site.py \
        /opt/devel/pipeline/envs/py38_orca_nkosogor/bin/python3 tests/test_serve.py

Synthetic data under /tmp/exopipe3_tests_serve/ (removed at the end):
  1. PORT EQUIVALENCE: the production launchers' heredocs (extracted from the exopipe_ops scripts at run time, never
     copied) and the in-place scripts run on one synthetic deep run and one synthetic sub-LST build; lib/serve's ports run
     on a second copy; the two trees must be byte-identical (queues, views, unions). The HOLD-rule and journal-append ports
     likewise against their heredocs.
  2. the D9 residual (own-Stokes review_dir), the pages-merge check, the exclusion gate keyers and verdicts, hub registry
     edits, the generation-mix gate, every content test (positive and negative).
  3. process control on high ports (>= 28700, never a production port): a stand-in review server (a script named
     classify_transients.py in the scratch tree), listeners, readiness, exact-pid stop, replace, inspection twin,
     --restart through the CLI, served_page / stopped_page tests.
  4. deep `views` stage end to end through record.Run (records under the scratch root).
  5. (review 2026-10-13) a refused stage stops the later ones (deep G2, sub-LST G8 / G11, snapshot G17), the in-place
     exclusion gate and core.exclusions agree, the forced switch probes before it stops the old server.
READ-ONLY checks on production: the planned argv of every page against the live servers (titles' served time taken from
the live argv), `serve --tier deep --run P3S5 --dry-run`, content tests on real queues and views.
"""
import csv
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the tree under test
sys.path.insert(0, ROOT)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')
T = '/tmp/exopipe3_tests_serve'
OPS = '/lustre/gh/main/exopipe_ops'
PY = '/opt/devel/pipeline/envs/py38_orca_nkosogor/bin/python3'
BASE_PORT = 28700            # the stand-in servers bind 28700-28899 (production 8765-8801, parity offset >= 1000 -> 9765+)

from exopipe.phase3.core import env as _env   # noqa: E402
_env.legacy()
from exopipe.phase3.config import CFG          # noqa: E402
from exopipe.phase3.core import validate      # noqa: E402
from exopipe.phase3.core.record import Refusal  # noqa: E402
from exopipe.phase3.lib.serve import (exclusion_gate, hub, journals, procs, queues, scripts, servers,  # noqa: E402
                                      sublst_build, views)
from exopipe.phase3.lib.serve import plan as planmod  # noqa: E402

PASSED = []


def ok(name):
    PASSED.append(name)
    print(f'ok   {name}')


# ---- helpers ---------------------------------------------------------------------------------------------------------
def heredoc(script, tag, nth=0):
    """The text of the nth heredoc <<'TAG' ... TAG of a production launcher (read at run time, never copied)."""
    lines = open(os.path.join(OPS, script)).read().splitlines()
    starts = [i for i, l in enumerate(lines) if f"<<'{tag}'" in l]
    i = starts[nth]
    j = next(k for k in range(i + 1, len(lines)) if lines[k] == tag)
    return '\n'.join(lines[i + 1:j]) + '\n'


def run_heredoc(code, args, env=None):
    e = dict(os.environ, **(env or {}))
    p = subprocess.run([PY, '-'] + [str(a) for a in args], input=code.encode(), stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, env=e)
    return p.returncode, p.stdout.decode()


def wcsv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in header})


def tree(d):
    out = {}
    for dp, dn, fn in os.walk(d):
        for f in fn:
            p = os.path.join(dp, f)
            rel = os.path.relpath(p, d)
            if rel.endswith('incremental.log') or '/_records/' in p:
                continue
            out[rel] = open(p, 'rb').read()
    return out


def same_tree(a, b, what):
    ta, tb = tree(a), tree(b)
    only = sorted(set(ta) ^ set(tb))
    diff = sorted(k for k in set(ta) & set(tb) if ta[k] != tb[k])
    assert not only and not diff, f'{what}: files only on one side {only[:5]}; differing {diff[:5]}'
    return len(ta)


def touch_png(U, rel):
    d = os.path.join(U, rel)
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, 'rank0001_04h_2025-01-01_Red_x.png'), 'w').close()


# ---- 1a. synthetic deep run ------------------------------------------------------------------------------------------
SURV = ['rank', 'source_id', 'identity_id', 'band', 'n_det', 'best_cand_id', 'best_date', 'best_snr', 'review_dir',
        'snr_rep']
MEM = ['source_id', 'band', 'cand_id', 'det_date']
MASTER = ['source_id', 'band', 'ra']
PIN = ['_sid', 'lst', 'date', 'band', 'n_det', 'snr', 'label', 'snr_rep']
IDX = ['source_id', 'band', 'rank', 'review_dir']


def pid(j, lst, date, band):
    return f'{j}_{lst}_{date}_{band}'


def make_deep(U):
    """Two Stokes, a wideband (Red) and a subband (23MHz) tier, the fork tiers, a pages-merged union with 'stokes'."""
    src = {   # tier -> [(J-name, band, lst, date, n_det, snr, label)]
        'V_wb': [('J0001+0001', 'Red', '04h', '2025-01-01', 2, 9.0, ''), ('J0002+0002', 'Red', '05h', '2025-01-02', 1, 8.0, ''),
                 ('J0003+0003', 'Red', '06h', '2025-01-03', 1, 7.5, 'marginal'),
                 ('J0005+0005', 'Red', '07h', '2025-01-04', 1, 6.0, '')],
        'V_sb': [('J0004+0004', '23MHz', '04h', '2025-01-01', 3, 12.0, ''), ('J0006+0006', '23MHz', '05h', '2025-01-02', 1, 5.5, '')],
        'I_wb': [('J0001+0001', 'Red', '04h', '2025-01-01', 1, 20.0, ''), ('J0007+0007', 'Red', '08h', '2025-01-05', 2, 15.0, '')],
        'I_sb': [('J0008+0008', '23MHz', '09h', '2025-01-06', 1, 10.0, '')],
    }
    union = {}
    for K, rows in src.items():
        S, t = K.split('_')
        surv, mem, mas, pin, q = [], [], [], [], []
        for n, (j, b, lst, date, nd, snr, lab) in enumerate(rows, 1):
            p_ = pid(j, lst, date, b)
            cid = f'{S}D_{lst}_{date}_{b}_{n:04d}'
            bd = '2025-01-08' if (j == 'J0005+0005' and S == 'V') else date    # D9 case: no exact / dated page match
            surv.append(dict(rank=f'rank{n:04d}', source_id=j, identity_id=j, band=b, n_det=nd, best_cand_id=cid,
                             best_date=bd, best_snr=snr, review_dir='', snr_rep=snr / 2))
            mem.append(dict(source_id=j, band=b, cand_id=cid, det_date=date))
            mas.append(dict(source_id=j, band=b, ra=n))
            pin.append(dict(_sid=j, lst=lst, date=date, band=b, n_det=nd, snr=snr, label=lab, snr_rep=snr / 2))
            q.append(p_)
            touch_png(U, f'candidates/I/{p_}')
        for b in sorted({r['band'] for r in surv}):
            wcsv(f'{U}/source_catalogs_{K}/{b}_sources_survivors.csv', SURV, [r for r in surv if r['band'] == b])
            wcsv(f'{U}/source_catalogs_{K}/{b}_source_members.csv', MEM, [r for r in mem if r['band'] == b])
            wcsv(f'{U}/source_catalogs_{K}/{b}_sources_master.csv', MASTER, [r for r in mas if r['band'] == b])
            for r in surv:
                if r['band'] == b:
                    rr = dict(r, stokes=S)
                    union.setdefault(b, []).append(rr)
        wcsv(f'{U}/source_catalogs/pages_input_{K}.csv', PIN, pin)
        os.makedirs(f'{U}/judgements', exist_ok=True)
        with open(f'{U}/judgements/queue_p3s3_{K}.txt', 'w') as f:
            f.write('# queue_name: i14_posthoc_std7\n# selection: raw\n' + '\n'.join(sorted(q, key=lambda x: x)) + '\n')
    # page indexes per Stokes (the singles sets), with a D9-residual case: V J0005+0005 has its page in 07h, I index
    # (shared candidates/I) names a J0005+0005 Red page of 10h that the unkeyed merge picked for the V union row
    idx = {'V': [], 'I': []}
    for K, rows in src.items():
        S = K[0]
        for n, (j, b, lst, date, nd, snr, lab) in enumerate(rows, 1):
            p_ = pid(j, lst, date, b)
            idx[S].append(dict(source_id=p_, band=b, rank=n, review_dir=f'candidates/I/{p_}'))
    other = pid('J0005+0005', '10h', '2025-01-09', 'Red')
    touch_png(U, f'candidates/I/{other}')
    idx['I'].append(dict(source_id=other, band='Red', rank=9, review_dir=f'candidates/I/{other}'))
    for S in 'VI':
        for b in sorted({r['band'] for r in idx[S]}):
            wcsv(f'{U}/source_catalogs/{b}_pages_index_{S}.csv', IDX, [r for r in idx[S] if r['band'] == b])
    for b, rows in union.items():
        for r in rows:
            c = r['best_cand_id'].split('_')
            r['review_dir'] = f"candidates/I/{pid(r['source_id'], c[1], c[2], b)}"
            if r['source_id'] == 'J0005+0005' and r['stokes'] == 'V':
                r['review_dir'] = f'candidates/I/{other}'          # what the unkeyed merge (all indexes) left
        wcsv(f'{U}/source_catalogs/{b}_sources_survivors.csv', SURV + ['stokes'], rows)
    # the fork (deep multiples): V_fwb only
    fsurv = [dict(rank='rank0001', source_id='J0001+0001', identity_id='J0001+0001', band='Red', n_det=2,
                  best_cand_id='VD_04h_2025-01-01_Red_0001', best_date='2025-01-01', best_snr=9.0, review_dir='', snr_rep=4.0),
             dict(rank='rank0002', source_id='J0009+0009', identity_id='J0009+0009', band='Red', n_det=1,
                  best_cand_id='VD_05h_2025-01-02_Red_0002', best_date='2025-01-02', best_snr=8.0, review_dir='', snr_rep=6.0)]
    wcsv(f'{U}/source_catalogs_V_fwb/Red_sources_survivors.csv', SURV, fsurv)
    wcsv(f'{U}/source_catalogs_V_fwb/Red_source_members.csv', MEM,
         [dict(source_id=r['source_id'], band='Red', cand_id=r['best_cand_id'], det_date=r['best_date']) for r in fsurv])
    wcsv(f'{U}/source_catalogs_V_fwb/Red_sources_master.csv', MASTER, [dict(source_id=r['source_id'], band='Red', ra=1)
                                                                         for r in fsurv])
    fp = [pid('J0001+0001', '04h', '2025-01-01', 'Red'), pid('J0009+0009', '05h', '2025-01-02', 'Red')]
    wcsv(f'{U}/source_catalogs/pages_input_V_fwb.csv', PIN, [dict(_sid='J0001+0001', lst='04h', date='2025-01-01', band='Red',
                                                                  n_det=2, snr=9.0), dict(_sid='J0009+0009', lst='05h',
                                                                                          date='2025-01-02', band='Red',
                                                                                          n_det=1, snr=8.0)])
    with open(f'{U}/judgements/queue_p3s3_V_fwb_raw.txt', 'w') as f:
        f.write('# queue_name: i14_posthoc_std7\n' + '\n'.join(fp) + '\n')
    for p_ in fp:
        touch_png(U, f'candidates/I/{p_}')
    wcsv(f'{U}/source_catalogs/Red_pages_index_V_fwb.csv', IDX,
         [dict(source_id=p_, band='Red', rank=n, review_dir=f'candidates/I/{p_}') for n, p_ in enumerate(fp, 1)])
    # the merge (pages) ran after the last page index
    t = time.time()
    for f in os.listdir(f'{U}/source_catalogs'):
        os.utime(f'{U}/source_catalogs/{f}', (t - 100, t - 100) if '_pages_index_' in f else (t, t))


def test_deep_ports():
    base = f'{T}/deep_base/Run_TEST'
    make_deep(base)
    A, B = f'{T}/deep_prod/Run_TEST', f'{T}/deep_new/Run_TEST'
    shutil.copytree(base, A)
    shutil.copytree(base, B)
    # production: p3s3_serve_tier.sh PYQ (its hard-coded run root replaced by the scratch copy), p3s3_stokes_queue.py,
    # p3s3_serve_stokes.sh PYV, p3s3_serve_multiples.sh PYQ
    pyq = heredoc('p3s3_serve_tier.sh', 'PYQ')
    needle = "U=f\"/data07/gh/phase3/transients/Run_{os.environ.get('P3S_LABEL','P3S3')}\""
    assert needle in pyq, 'p3s3_serve_tier.sh PYQ changed: update the test'
    pyq = pyq.replace(needle, f'U={A!r}')
    for K in ('V_wb', 'V_sb', 'I_wb', 'I_sb'):
        rc, out = run_heredoc(pyq, [K])
        assert rc == 0, out
    for S in 'VI':
        assert subprocess.run([PY, f'{OPS}/p3s3_stokes_queue.py', A, S], stdout=subprocess.DEVNULL).returncode == 0
        rc, out = run_heredoc(heredoc('p3s3_serve_stokes.sh', 'PYV'), [A, S])
        assert rc == 0, out
    rc, out = run_heredoc(heredoc('p3s3_serve_multiples.sh', 'PYQ'), [A, 'V'], env={'P3S_LABEL': 'P3S5'})
    assert rc == 0, out
    # new code
    for K in ('V_wb', 'V_sb', 'I_wb', 'I_sb'):
        queues.section_tier_queue(B, K, 'deep')
    for S in 'VI':
        assert scripts.run('stokes_queue', [B, S]).ok
        views.deep_stokes_view(B, S, review_dir='union')
    q, keep, secs = queues.deep_multi_queue(B, 'V')
    views.deep_multi_view(B, 'V', keep)
    n = same_tree(A, B, 'deep ports vs production heredocs')
    ok(f'deep: tier sectioning, Stokes queues, per-Stokes views and the fork multiples queue/view byte-identical to the '
       f'production heredocs ({n} files)')
    # G3: no fork for I
    try:
        queues.deep_multi_queue(B, 'I')
        raise AssertionError('G3 did not refuse')
    except Refusal as e:
        assert 'G3' in str(e)
    ok('deep G3: multiples without the fork refuse')
    # D9 residual: own-Stokes review_dir
    C = f'{T}/deep_own/Run_TEST'
    shutil.copytree(base, C)
    for K in ('V_wb', 'V_sb'):
        queues.section_tier_queue(C, K, 'deep')
    v = views.deep_stokes_view(C, 'V', review_dir='own-stokes')
    ch = {(a, b): (o, n_) for a, b, o, n_ in v['review_dir_changed']}
    assert ch == {('J0005+0005', 'Red'): ('candidates/I/J0005+0005_10h_2025-01-09_Red',
                                          'candidates/I/J0005+0005_07h_2025-01-04_Red')}, ch
    ok('deep D9 residual: own-Stokes index resolves the V row to its own page (1 row changed, as expected)')
    # G2
    D = f'{T}/deep_g2/Run_TEST'
    shutil.copytree(base, D)
    p = f'{D}/source_catalogs/Red_sources_survivors.csv'
    rows = list(csv.DictReader(open(p)))
    wcsv(p, SURV, rows)
    try:
        views.deep_stokes_view(D, 'V')
        raise AssertionError('G2 did not refuse')
    except Refusal as e:
        assert 'G2' in str(e)
    ok('deep G2: a union without stokes refuses')
    # pages merge check
    good, detail = views.deep_merge_current(base)
    assert good, detail
    t = time.time() + 10
    os.utime(f'{D}/source_catalogs/Red_pages_index_V.csv', (t, t))
    stale, detail2 = views.deep_merge_current(D)
    assert not stale and 'older than the page index' in detail2, detail2
    E = f'{T}/deep_hdr/Run_TEST'                     # a header-only union file: the merge never rewrites it
    shutil.copytree(base, E)
    wcsv(f'{E}/source_catalogs/41MHz_sources_survivors.csv', SURV + ['stokes'], [])
    t0 = time.time() - 1000
    os.utime(f'{E}/source_catalogs/41MHz_sources_survivors.csv', (t0, t0))
    good3, detail3 = views.deep_merge_current(E)
    assert good3, detail3
    ok('deep: the pages merge is required current (merged union newer than every page index; a header-only union file, '
       'which the merge leaves untouched, does not count)')
    return B


def test_queue_view_content(B):
    for name in ('queue_p3s3_V_wb.txt', 'queue_p3s3_V.txt', 'queue_p3s3_I.txt', 'queue_p3s3_V_multi.txt'):
        res = validate.run('served_queue', f'{B}/judgements/{name}')
        bad = [r for r in res if not r['passed']]
        assert not bad, (name, bad)
    for d in ('source_catalogs_V', 'source_catalogs_I', 'views/source_catalogs_V_multi'):
        res = validate.run('served_view', f'{B}/{d}', tier='deep')
        bad = [r for r in res if not r['passed']]
        assert not bad, (d, bad)
    ok('content tests pass on the synthetic served queues and views (served_queue x4, served_view x3)')
    # negatives
    q = f'{B}/judgements/queue_p3s3_V.txt'
    lines = open(q).read().splitlines()
    j = next(l for l in lines if l.startswith('J'))
    bad_q = f'{T}/neg/judgements/queue_p3s3_V.txt'
    os.makedirs(os.path.dirname(bad_q), exist_ok=True)
    open(bad_q, 'w').write('\n'.join(lines + [j]) + '\n')
    r = {x['name']: x for x in validate.run('served_queue', bad_q)}
    assert not r['served_queue.protocol']['passed']
    open(bad_q, 'w').write('# queue_name: p3s3_V_deep\n# selection: single-block sources (SNR-sorted)\n')
    r = {x['name']: x for x in validate.run('served_queue', bad_q)}
    assert not r['served_queue.nonempty']['passed']
    E = f'{T}/neg_view/Run_TEST'
    shutil.copytree(B, E)
    p = f'{E}/source_catalogs_V/Red_sources_survivors.csv'
    rows = list(csv.DictReader(open(p)))
    rows[0]['best_snr'] = '99.9'
    rows.append(dict(rows[1]))
    wcsv(p, list(rows[0].keys()), rows)
    r = {x['name']: x for x in validate.run('served_view', f'{E}/source_catalogs_V', tier='deep')}
    assert not r['served_view.stokes_equals_catalogue']['passed'] and not r['served_view.unique_keys']['passed']
    os.remove(f'{E}/source_catalogs_V/23MHz_sources_survivors.csv')
    r = {x['name']: x for x in validate.run('served_view', f'{E}/source_catalogs_V', tier='deep')}
    assert not r['served_view.complete']['passed']
    ok('content tests fail on a duplicated id, an empty queue, a changed / duplicated view row and a missing view file')


# ---- 1b. synthetic sub-LST build -------------------------------------------------------------------------------------
def make_sublst(B):
    src = {'V_sb': [('J0101+0101', '23MHz', '04h', '2025-01-01T02', 2, 9.0), ('J0102+0102', '23MHz', '05h', '2025-01-02T03', 1, 6.0)],
           'V_wb': [('J0101+0101', 'Red', '04h', '2025-01-01T02', 1, 7.0)],
           'I_sb': [('J0101+0101', '23MHz', '04h', '2025-01-01T02', 1, 8.0), ('J0103+0103', '23MHz', '06h', '2025-01-03T04', 3, 11.0)],
           'I_wb': []}
    for K, rows in src.items():
        S, t = K.split('_')
        surv, mem, mas, pin, idx, q = [], [], [], [], [], []
        for n, (j, b, lst, date, nd, snr) in enumerate(rows, 1):
            p_ = pid(j, lst, date, b)
            cid = f'{S}1_{lst}_{date}_{b}_{n:04d}'
            surv.append(dict(rank=f'rank{n:04d}', source_id=j, identity_id=j, band=b, n_det=nd, best_cand_id=cid,
                             best_date=date, best_snr=snr, review_dir=''))
            mem.append(dict(source_id=j, band=b, cand_id=cid, det_date=date))
            mas.append(dict(source_id=j, band=b, ra=n))
            pin.append(dict(_sid=j, lst=lst, date=date, band=b, n_det=nd, snr=snr))
            idx.append(dict(source_id=p_, band=b, rank=n, review_dir=f'candidates/{S}/{p_}'))
            q.append(p_)
            touch_png(B, f'candidates/{S}/{p_}')
        for b in sorted({r['band'] for r in surv}):
            wcsv(f'{B}/source_catalogs_{K}/{b}_sources_survivors.csv', SURV[:-1], [r for r in surv if r['band'] == b])
            wcsv(f'{B}/source_catalogs_{K}/{b}_source_members.csv', MEM, [r for r in mem if r['band'] == b])
            wcsv(f'{B}/source_catalogs_{K}/{b}_sources_master.csv', MASTER, [r for r in mas if r['band'] == b])
            wcsv(f'{B}/source_catalogs/{b}_pages_index_{S}.csv', IDX, [r for r in idx if r['band'] == b])
        if rows:
            wcsv(f'{B}/source_catalogs/pages_input_{K}.csv', PIN[:-2], pin)
            os.makedirs(f'{B}/judgements', exist_ok=True)
            open(f'{B}/judgements/queue_p3s3_{K}.txt', 'w').write('\n'.join(q) + '\n')


def test_sublst_ports():
    base = f'{T}/sub_base/Run_SLSTB.build'
    make_sublst(base)
    A, B = f'{T}/sub_prod/Run_SLSTB.build', f'{T}/sub_new/Run_SLSTB.build'
    shutil.copytree(base, A)
    shutil.copytree(base, B)
    LST = '04h 05h 06h'
    menv = {'P3S_MERGE_INDEX_ONLY': '1', 'P3S_MERGE_STOKES_KEY': '1'}
    for S in 'VI':     # production: sublst_serve.sh prepare, once per Stokes (sublst_cumulative.sh:71-74)
        rc, out = run_heredoc(heredoc('sublst_serve.sh', 'PYU'), [A])
        assert rc == 0, out
        p = subprocess.run([PY, f'{OPS}/p3s3_merge_review_dir.py', A], env=dict(os.environ, **menv), stdout=subprocess.PIPE)
        assert p.returncode == 0, p.stdout
        for K in (f'{S}_sb', f'{S}_wb'):
            if K.endswith('_wb') and not os.path.isfile(f'{A}/judgements/queue_p3s3_{K}.txt'):
                continue
            rc, out = run_heredoc(heredoc('sublst_serve.sh', 'PYQ'), [A, K, LST])
            assert rc == 0, out
        assert subprocess.run([PY, f'{OPS}/p3s3_stokes_queue.py', A, S], stdout=subprocess.DEVNULL).returncode == 0
        rc, out = run_heredoc(heredoc('sublst_serve.sh', 'PYV'), [A, S])
        assert rc == 0, out
    for S in 'VI':     # multi-integration (sublst_cumulative.sh:77-95)
        Q, QM, V = f'{A}/judgements/queue_p3s3_{S}.txt', f'{A}/judgements/queue_p3s3_{S}_multi.txt', f'{A}/source_catalogs_{S}_multi'
        subprocess.run(f"awk '/^# selection: MULTI-INTEGRATION/{{m=1; print; next}} /^# selection:/{{m=0}} m' {Q} > {QM}",
                       shell=True, check=True)
        if subprocess.run(['grep', '-q', '^J', QM]).returncode != 0:
            continue
        rc, out = run_heredoc(heredoc('sublst_cumulative.sh', 'PYM'), [A, S, QM, V])
        assert rc == 0, out
    # new code (union and merge once: idempotent; see lib/serve/sublst_build)
    views.sublst_union(B)
    assert scripts.run('merge_review_dir', [B], env=views.pages_merge_env('sublst')).ok
    for S in 'VI':
        for t in ('sb', 'wb'):
            K = f'{S}_{t}'
            if t == 'wb' and not os.path.isfile(queues.tier_queue_path(B, K)):
                continue
            queues.section_tier_queue(B, K, 'sublst', lsts_text=LST)
        assert scripts.run('stokes_queue', [B, S]).ok
        views.sublst_stokes_view(B, S)
    for S in 'VI':
        Q, QM = f'{B}/judgements/queue_p3s3_{S}.txt', f'{B}/judgements/queue_p3s3_{S}_multi.txt'
        if queues.sublst_multi_queue(Q, QM):
            views.sublst_multi_view(B, S, QM)
    n = same_tree(A, B, 'sub-LST ports vs production heredocs')
    ok(f'sub-LST: union, Stokes-keyed merge (in place), sectioning, Stokes queues, views and multi-integration queues/views '
       f'byte-identical to the production heredocs ({n} files)')
    for d in ('source_catalogs_V', 'source_catalogs_I', 'source_catalogs_I_multi'):
        res = [r for r in validate.run('served_view', f'{B}/{d}', tier='sublst') if r['name'] != 'served_view.tier_rules']
        assert all(r['passed'] for r in res), (d, res)
    ok('sub-LST views pass served_view (Stokes, keys, complete, review_dir page)')
    # G6: a view key held twice
    C = f'{T}/sub_g6/Run_SLSTB.build'
    shutil.copytree(B, C)
    p = f'{C}/source_catalogs/23MHz_sources_survivors.csv'
    rows = list(csv.DictReader(open(p)))
    rows.append(dict(rows[0]))
    wcsv(p, list(rows[0].keys()), rows)
    try:
        views.sublst_stokes_view(C, rows[0]['stokes'])
        raise AssertionError('G6 did not refuse')
    except Refusal as e:
        assert 'G6' in str(e)
    ok('sub-LST G6: a (J-name, band) key held twice refuses')


def test_generation_mix():
    R = f'{T}/lstruns'
    for l, (pert, posk) in {'04h': (True, True), '05h': (False, True), '06h': (True, False)}.items():
        d = f'{R}/Run_SLST{l[:-1]}B'
        os.makedirs(f'{d}/candidates', exist_ok=True)
        if pert:
            open(f'{d}/pertie_detections.csv', 'w').write('cand_id\n')
        open(f'{d}/master_catalog.csv', 'w').write('cand_id,ra' + (',pos_kind' if posk else '') + '\n')
    orig = sublst_build.lst_run
    sublst_build.lst_run = lambda l: f'{R}/Run_SLST{l[:-1]}B'
    try:
        bad = dict(sublst_build.generation_mix(['04h', '05h', '06h', '07h']))
    finally:
        sublst_build.lst_run = orig
    assert set(bad) == {'05h', '06h', '07h'} and 'pertie' in bad['05h'] and 'pos_kind' in bad['06h'], bad
    ok('sub-LST G8: LST runs without the window tie / fitted positions / candidates are named (generation mix)')


# ---- 1c. journals ------------------------------------------------------------------------------------------------------
def test_journals():
    old, new = f'{T}/j_old', f'{T}/j_new'
    os.makedirs(f'{old}/judgements')
    os.makedirs(f'{new}/judgements')
    recs = [dict(source_id='J1', scope='source', action='classify', judge_confidence='real', judge_note='n' * 300),
            dict(source_id='J2', scope='source', action='classify', judge_confidence='FP'),
            dict(source_id='J3', scope='source', action='classify', judge_confidence='lean-FP'),
            dict(source_id='J3', scope='source', action='retract'),
            dict(source_id='J4', scope='source', action='classify', judge_confidence='probably-real')]
    open(f'{old}/judgements/judgements.jsonl', 'w').write(''.join(json.dumps(r) + '\n' for r in recs))
    wcsv(f'{new}/judgements/carry_report.csv', ['new_source', 'n_detections', 'old_sources', 'outcome', 'verdicts'],
         [dict(new_source='K4', old_sources='J4:1/2', outcome='carried')])
    code = heredoc('snapshot_rechain_batch.sh', 'PYM')
    rc, out = run_heredoc(code, [old, new])
    assert rc == 0, out
    prod_missed = open(f'{new}/judgements/carry_missed.csv').read()
    notfp, n = journals.hold_check(old, new, reader='driver')
    assert open(f'{new}/judgements/carry_missed.csv').read() == prod_missed
    assert out.strip() == '%d|%s' % (len(notfp), '; '.join(notfp)), (out, notfp)
    ok('HOLD rule (driver reader) byte-identical to the batch driver heredoc (carry_missed.csv and the not-FP list)')
    assert notfp == ['J1 (real)', 'J3 ()'], notfp
    notfp2, _ = journals.hold_check(old, new, reader='carry')
    assert notfp2 == ['J1 (real)'], notfp2
    ok(f'HOLD rule (carry reader, the R6 fix): a retracted verdict is not a missed verdict ({notfp} -> {notfp2})')
    assert CFG.serve.HOLD_JOURNAL_READER == 'driver' and journals.hold_check(old, new)[0] == notfp
    assert open(f'{new}/judgements/carry_missed.csv').read() == prod_missed
    ok('HOLD rule default = the batch driver reader (production; R6 not ruled): byte-identical to the heredoc')
    open(f'{new}/judgements/judgements.jsonl', 'w').write('')
    r = {x['name']: x for x in validate.run('judgement_journal', f'{new}/judgements/judgements.jsonl')}
    assert not r['judgement_journal.hold_rule']['passed']
    ok('judgement_journal.hold_rule fails when a non-FP verdict is missed')
    # sub-LST journal copy / swap / append (sublst_cumulative.sh:98-120)
    U, Bd = f'{T}/swap/Run_SLSTB', f'{T}/swap/Run_SLSTB.build'
    for J in CFG.serve.SUBLST_JOURNALS:
        os.makedirs(f'{U}/{J}')
        open(f'{U}/{J}/judgements.jsonl', 'w').write('{"a": 1}\n')
    os.makedirs(f'{Bd}/candidates')
    journals.sublst_copy_journals(U, Bd)
    with open(f'{U}/judgements_V/judgements.jsonl', 'a') as f:
        f.write('{"b": 2}\n')                                          # a verdict between the copy and the swap
    prev = journals.sublst_swap(U, Bd, U + '.prev_X')
    shutil.copytree(U, f'{T}/swap_prod')
    rc, out = run_heredoc(heredoc('sublst_cumulative.sh', 'PYJ'), [prev, f'{T}/swap_prod'])
    assert rc == 0, out
    journals.append_between(prev, U)
    same_tree(U, f'{T}/swap_prod', 'journal append')
    assert open(f'{U}/judgements_V/judgements.jsonl').read() == '{"a": 1}\n{"b": 2}\n'
    r = {x['name']: x for x in validate.run('judgement_journal', f'{U}/judgements_V/judgements.jsonl',
                                            carried_from=f'{prev}/judgements_V/judgements.jsonl')}
    assert r['judgement_journal.parses_append_only']['passed'], r
    ok('sub-LST journal copy, swap and between-lines append identical to the production heredoc; append-only test passes')


# ---- 2. exclusion gate, hub ------------------------------------------------------------------------------------------
def test_exclusion_gate():
    import importlib.util
    spec = importlib.util.spec_from_file_location('gate_inplace', f'{OPS}/snapshot_serve_exclusion_gate.py')
    G = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(G)
    rows = [{'cand_id': '18h_2025-04-11_36MHz_V_10s_20250411_034510'}, {'lst': '18h', 'date': '2025-04-11', 'band': 'Red'},
            {'cell': '18h_2025-04-11', 'subband': '41MHz'}, {'cell': '18h_2025-04-11', 'member_sb': '46MHz'},
            {'cell': '18h_2025-04-11', 'bands': 'Red;Green'}, {'cand_id': 'VD_04h_2025-01-01_Red_0001'}, {'x': '1'}]
    for r in rows:
        assert exclusion_gate.keys_snapshot(r) == G.keys_of(r), r
    ok('exclusion gate: snapshot row keying identical to the in-place gate (keys_of)')
    assert exclusion_gate.keys_tier({'cand_id': 'VD_04h_2025-01-01_Red_0001'}) == [('04h', '2025-01-01', 'Red')]
    assert exclusion_gate.keys_tier({'best_cand_id': 'V1_00h_2025-09-13T02_18MHz_0001'}) == [('00h', '2025-09-13', '18MHz')]
    assert exclusion_gate.keys_tier({'source_id': 'J1134+3339_14h_2025-05-01T04_41MHz'}) == [('14h', '2025-05-01', '41MHz')]
    assert exclusion_gate.keys_tier({'source_id': 'J1134+3339'}) == []
    ok('exclusion gate: deep / sub-LST keyers (cand_id, best_cand_id, page id)')
    from exopipe.phase3.core.exclusions import Exclusions
    dd = f'{T}/data_decisions.csv'
    wcsv(dd, ['lst', 'subband', 'date', 'verdict', 'reason'],
         [dict(lst='04h', subband='64MHz', date='2025-01-01', verdict='exclude', reason='x'),
          dict(lst='05h', subband='18MHz', date='2025-01-02', verdict='exclude', reason='x')])
    ex = Exclusions(dd)
    V = f'{T}/gate_views/source_catalogs_V'
    wcsv(f'{V}/Blue_sources_survivors.csv', SURV, [dict(rank=1, source_id='J1', identity_id='', band='Blue', n_det=1,
                                                        best_cand_id='VD_06h_2025-01-03_Blue_0001', best_date='', best_snr=9,
                                                        review_dir='', snr_rep=4),
                                                   dict(rank=2, source_id='J2', identity_id='', band='Red', n_det=1,
                                                        best_cand_id='VD_06h_2025-01-03_Red_0001', best_date='', best_snr=9,
                                                        review_dir='', snr_rep=4)])
    wcsv(f'{V}/Blue_source_members.csv', MEM, [dict(source_id='J1', band='Blue', cand_id='VD_04h_2025-01-01_Blue_0001'),
                                               dict(source_id='J2', band='Red', cand_id='VD_05h_2025-01-02_Red_0001')])
    scan = exclusion_gate.scan_page(os.path.dirname(V), V, None, ex=ex)
    assert scan.per_key() == {('04h', '2025-01-01', 'Blue'): 1}, scan.summary()    # 64 MHz excludes Blue; 18 MHz not Red
    ok('exclusion gate: verdicts from core.exclusions (a 64 MHz exclusion excludes Blue; the 18 MHz carve-out keeps Red)')
    ruled = CFG.serve.EXCLUSION_GATE_TIERS
    assert set(ruled) == {'snapshot', 'snapshot-forced', 'deep', 'sublst', 'deep-strat', 'deep-deep-forced'}, ruled
    assert all(exclusion_gate.refuses(*k) for k in (('snapshot',), ('snapshot', 'forced'), ('deep',), ('sublst',),
                                                     ('deep', 'strat'), ('deep', 'deep-forced')))
    assert not exclusion_gate.refuses('jupiter')
    CFG.serve.EXCLUSION_GATE_TIERS = ('snapshot',)                  # production (the parity file)
    try:
        assert exclusion_gate.refuses('snapshot') and not exclusion_gate.refuses('deep') and not exclusion_gate.refuses(
            'sublst') and not exclusion_gate.refuses('snapshot', 'forced')
    finally:
        CFG.serve.EXCLUSION_GATE_TIERS = ruled
    ok('exclusion gate refuses for serve.EXCLUSION_GATE_TIERS: ruled snapshot, snapshot forced, deep (review, stratified, '
       'deep forced) and sub-LST pages (Gregg 2026-10-14); the production value refuses snapshot runs only')


def test_exclusion_gate_scan_page():
    """Deep / sub-LST: the gate reads exactly what the page's review server reads (classify_transients.load_sources) -
    each band's survivors and members, each survivor's page folder (detections.csv, non-blind panels), the queue - and
    nothing else in the folder (2026-10-14: 34 rows of 2026-10-07 index files the server never reads were counted)."""
    from exopipe.phase3.core.exclusions import Exclusions
    dd = f'{T}/scan_page/data_decisions.csv'
    wcsv(dd, ['lst', 'subband', 'date', 'verdict', 'reason'],
         [dict(lst='04h', subband='64MHz', date='2025-01-01', verdict='exclude', reason='x')])      # excludes 64MHz, Blue, Full
    ex = Exclusions(dd)
    U = f'{T}/scan_page/Run_X'
    C = f'{U}/source_catalogs_V'
    bad_c, good_c = 'VD_04h_2025-01-01_64MHz_0001', 'VD_05h_2025-01-02_64MHz_0001'
    rows = dict(rank=1, identity_id='', n_det=1, best_date='', best_snr=9, snr_rep=4)
    wcsv(f'{C}/64MHz_sources_survivors.csv', SURV, [
        dict(rows, source_id='J1', band='64MHz', best_cand_id=good_c, review_dir='candidates/V/p1'),
        dict(rows, source_id='J2', band='64MHz', best_cand_id=good_c, review_dir='candidates/V/p2'),
        dict(rows, source_id='J3', band='64MHz', best_cand_id=good_c, review_dir='candidates/V/missing')])
    wcsv(f'{C}/64MHz_source_members.csv', MEM, [dict(source_id='J1', band='64MHz', cand_id=good_c, det_date='')])
    os.makedirs(f'{U}/candidates/V/p1')
    os.makedirs(f'{U}/candidates/V/p2')
    for f in ('rank00001_05h_2025-01-02_64MHz.png', 'rank00002_04h_2025-01-01_Blue_blind.png', 'notes.png'):
        open(f'{U}/candidates/V/p1/{f}', 'w').close()
    wcsv(f'{U}/candidates/V/p2/detections.csv', ['cand_id', 'lst', 'date', 'band'],
         [dict(cand_id=good_c, lst='05h', date='2025-01-02', band='64MHz')])
    q = f'{U}/judgements/queue_V.txt'
    os.makedirs(os.path.dirname(q))
    open(q, 'w').write('J1_05h_2025-01-02_64MHz\nJ2_05h_2025-01-02_64MHz\n')
    # files the server never reads, each holding a row on the excluded night: none may count
    wcsv(f'{C}/64MHz_pages_index_V.csv', IDX, [dict(source_id='J9_04h_2025-01-01_64MHz', band='64MHz', rank=1, review_dir='')])
    wcsv(f'{C}/Foo_sources_survivors.csv', SURV, [dict(rows, source_id='J8', band='Foo', best_cand_id=bad_c, review_dir='')])
    wcsv(f'{C}/master_catalog_removed.csv', ['cand_id'], [dict(cand_id=bad_c)])
    # member rows the server does not show: of no survivor; of a survivor whose folder holds a detections.csv (J2, p2)
    with open(f'{C}/64MHz_source_members.csv', 'a') as fh:
        fh.write(f'J99,64MHz,{bad_c},\nJ2,64MHz,{bad_c},\n')
    s0 = exclusion_gate.scan_page(U, C, q, ex=ex)
    assert not s0.hits and s0.n_files == 4 and s0.n_rows == 8, (s0.summary(), s0.n_files, s0.n_rows)
    ok('exclusion gate per page: files the server never reads (an old pages index, a band it does not list, a removed-rows '
       'table) and blind panels are not scanned; survivors, members, detections.csv, panels and the queue are (0 hits)')
    # each path the server reads, one at a time, carries a row on the excluded night
    def hit(label, edit):
        shutil.copytree(U, f'{U}_{label}')
        edit(f'{U}_{label}')
        sc = exclusion_gate.scan_page(f'{U}_{label}', f'{U}_{label}/source_catalogs_V', q.replace(U, f'{U}_{label}'), ex=ex)
        assert len(sc.hits) == 1 and sc.per_key() == {('04h', '2025-01-01', '64MHz'): 1} or label in ('blue_panel',) and \
            sc.per_key() == {('04h', '2025-01-01', 'Blue'): 1}, (label, sc.summary())
    hit('survivor', lambda R: wcsv(f'{R}/source_catalogs_V/64MHz_sources_survivors.csv', SURV, [
        dict(rows, source_id='J1', band='64MHz', best_cand_id=bad_c, review_dir='candidates/V/p1')]))
    hit('member', lambda R: wcsv(f'{R}/source_catalogs_V/64MHz_source_members.csv', MEM,
                                 [dict(source_id='J1', band='64MHz', cand_id=bad_c, det_date='')]))
    hit('detection', lambda R: wcsv(f'{R}/candidates/V/p2/detections.csv', ['cand_id', 'lst', 'date', 'band'],
                                    [dict(cand_id=bad_c, lst='04h', date='2025-01-01', band='64MHz')]))
    hit('panel', lambda R: open(f'{R}/candidates/V/p1/rank00003_04h_2025-01-01_64MHz.png', 'w').close())
    hit('blue_panel', lambda R: open(f'{R}/candidates/V/p1/rank00004_04h_2025-01-01_Blue.png', 'w').close())
    hit('queue', lambda R: open(f'{R}/judgements/queue_V.txt', 'a').write('J7_04h_2025-01-01_64MHz\n'))
    hit('snapshot_token_panel', lambda R: open(f'{R}/candidates/V/p1/rank00005_04h_2025-01-01_64MHz20250101043000.png',
                                               'w').close())
    ok('exclusion gate per page: a row on an excluded night in a survivors table, a shown member, a page folder\'s '
       'detections.csv, a panel image name (a band token with its instant too) or the queue is found (one hit each)')
    # --loc-offsets: the offsets rows of the cand_ids the page shows are page text; other rows are not read
    lo = f'{U}/source_catalogs/localization_offsets.csv'
    wcsv(lo, ['cand_id', 'identity_id', 'sep_arcsec', 'sep_beams', 'offset_pct'],
         [dict(cand_id=bad_c, identity_id='J9', sep_arcsec=1, sep_beams=0.1, offset_pct=10)])
    assert not exclusion_gate.scan_page(U, C, q, ex=ex, loc_offsets=lo).hits     # bad_c is shown by no row of the page
    shutil.copytree(U, f'{U}_lo')
    wcsv(f'{U}_lo/candidates/V/p2/detections.csv', ['cand_id', 'lst', 'date', 'band'],
         [dict(cand_id=bad_c.replace('04h_2025-01-01', '05h_2025-01-02'), lst='05h', date='2025-01-02', band='64MHz')])
    wcsv(f'{U}_lo/source_catalogs/localization_offsets.csv', ['cand_id', 'identity_id', 'sep_arcsec', 'sep_beams', 'offset_pct'],
         [dict(cand_id=bad_c.replace('04h_2025-01-01', '05h_2025-01-02'), identity_id='J2', sep_arcsec=1, sep_beams=0.1,
               offset_pct=10)])
    s_no = exclusion_gate.scan_page(f'{U}_lo', f'{U}_lo/source_catalogs_V', None, ex=ex)
    s_lo = exclusion_gate.scan_page(f'{U}_lo', f'{U}_lo/source_catalogs_V', None, ex=ex,
                                    loc_offsets=f'{U}_lo/source_catalogs/localization_offsets.csv')
    assert (s_lo.n_files, s_lo.n_rows) == (s_no.n_files + 1, s_no.n_rows + 1), (s_no.summary(), s_lo.summary())
    ok('exclusion gate per page: --loc-offsets rows count only for the cand_ids the page shows')
    # report only (strict False): an unreadable table is listed, never raised; refusing (strict True): it refuses
    shutil.copytree(U, f'{U}_ur')
    os.remove(f'{U}_ur/source_catalogs_V/64MHz_source_members.csv')
    os.makedirs(f'{U}_ur/source_catalogs_V/64MHz_source_members.csv')             # exists, cannot be read
    sr = exclusion_gate.scan_page(f'{U}_ur', f'{U}_ur/source_catalogs_V', None, ex=ex, strict=False)
    assert sr.unreadable == ['source_catalogs_V/64MHz_source_members.csv'] and 'not readable' in sr.summary(), sr.summary()
    try:
        exclusion_gate.scan_page(f'{U}_ur', f'{U}_ur/source_catalogs_V', None, ex=ex, strict=True)
        raise AssertionError('an unreadable table did not refuse a refusing scan')
    except Refusal as e:
        assert 'could not be read' in str(e), e
    ok('exclusion gate per page: an unreadable file is listed in a report and refuses only when the gate refuses')


def test_exclusion_gate_server_mirrors():
    """The gate's mirrors of the review server (band list, panel-name pattern) equal classify_transients.py's own."""
    import ast
    tree = ast.parse(open(f'{OPS}/classify_transients.py').read())
    got = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            n = node.targets[0].id
            if n == 'BANDS':
                got['BANDS'] = tuple(ast.literal_eval(node.value))
            elif n == 'PNG_RE':
                got['PNG_RE'] = ast.literal_eval(node.value.args[0])     # re.compile(r'...' r'...'): one string constant
    assert got.get('BANDS') == exclusion_gate.SERVER_BANDS, (got.get('BANDS'), exclusion_gate.SERVER_BANDS)
    assert got.get('PNG_RE') == exclusion_gate.SERVER_PANEL_RE.pattern, (got.get('PNG_RE'),
                                                                         exclusion_gate.SERVER_PANEL_RE.pattern)
    ok('exclusion gate mirrors: SERVER_BANDS and SERVER_PANEL_RE equal classify_transients.BANDS / PNG_RE')


def test_hub():
    reg = {'round2': {'port': 8767}, 'V': {'1-hour': {'port': 8766}}, 'I': {}, 'tunnel_extra': [8800]}
    hub.apply_set(reg, 'V', '1-hour', ['run=Run_X', 'generation=g', 'served=s', 'live=1', 'port=18766'])
    assert reg['V']['1-hour'] == {'port': 18766, 'run': 'Run_X', 'generation': 'g', 'served': 's', 'live': True}
    reg['V']['10-min-sublst'] = {'port': 8790}
    hub.remove_tiers(reg, ('10-min-sublst',))
    assert '10-min-sublst' not in reg['V']
    assert hub.tunnel_extra_with(reg, [8800]) is None and hub.tunnel_extra_with(reg, [8801, 18766]) == [8800, 8801]
    ok('hub registry: --set rule, retired tier removal, tunnel_extra union (button ports not duplicated)')


# ---- 3. process control with stand-in servers ---------------------------------------------------------------------------
FAKE = r'''
import argparse, http.server, os, urllib.parse
ap = argparse.ArgumentParser()
for a in ('--run', '--queue', '--labels-dir', '--catalog-dir', '--title', '--loc-dir', '--default-fp-reason', '--lists-dir'):
    ap.add_argument(a)
ap.add_argument('--port', type=int)
ap.add_argument('--sibling', action='append')
ap.add_argument('--inspect', action='store_true')
ap.add_argument('--loc-offsets', action='store_true')
a = ap.parse_args()
ids = [l.split()[0] for l in open(a.queue) if l.startswith('J')]
body = ''.join('<a href="/source/%s">x</a>\n' % urllib.parse.quote(i) for i in ids).encode()
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(body)
    def log_message(self, *x): pass
http.server.ThreadingHTTPServer(('127.0.0.1', a.port), H).serve_forever()
'''


def test_processes():
    import argparse
    CFG.set('serve.READY_POLL_S', 0.3)
    CFG.set('serve.START_SETTLE_S', 0.2)
    fake_dir = f'{T}/fake'
    os.makedirs(fake_dir)
    open(f'{fake_dir}/classify_transients.py', 'w').write(FAKE)
    offset = 20000                               # 8766 -> 28766 (BASE_PORT range)
    a = argparse.Namespace(tier='deep', run_label='TEST', port_offset=offset, pages=None, stokes=None, lsts=None, label=None,
                           source=None, family=None, forced_tag=None, restart=None, stop=None)
    out = f'{T}/procs_out'
    plan = planmod.make(a, out)
    assert plan.parity and not plan.gates, plan.gates
    U = plan.U
    os.makedirs(f'{U}/judgements', exist_ok=True)
    q = f'{U}/judgements/queue_p3s3_V.txt'
    open(q, 'w').write('# queue_name: p3s3_V_deep\nJ0001+0001_04h_2025-01-01_Red\nJ0002+0002_05h_2025-01-02_Red\n'
                       'J0001+0001_04h_2025-01-03_Red\n')
    pg = next(p for p in plan.pages if p.key == 'V')
    assert pg.port == 28766 and pg.twin == 28796

    def fake_argv(page, utc, root=None):
        return [PY, f'{fake_dir}/classify_transients.py', '--run', U, '--queue', q, '--labels-dir', f'{U}/judgements_V',
                '--catalog-dir', f'{U}/source_catalogs_V', '--port', str(page.port), '--title', f'test page - served {utc}']
    plan.argv = fake_argv
    for port in (28766, 28796):
        assert not procs.port_in_use(port), f'port {port} busy: the test needs it free'
    notes = []
    path, rec = servers.replace(plan, pg, 'UTC', expected={'n': 3, 'kind': 'queue ids'}, notes=notes)
    assert procs.listeners(28766) == [rec['pid']]
    r = {x['name']: x for x in validate.run('served_page', path)}
    assert all(x['passed'] for x in r.values()), r
    ok(f'replace: stand-in review server started on 28766 (pid {rec["pid"]}), served_page live / same_process / '
       f'sources_listed pass')
    old_pid = rec['pid']
    path, rec = servers.replace(plan, pg, 'UTC2', expected={'n': 3, 'kind': 'queue ids'}, notes=notes)
    assert rec['pid'] != old_pid and not procs.alive(old_pid)
    ok('replace again: the old server stopped by exact pid, a new one serves (G14)')
    tpath, trec = servers.start_twin(plan, pg, notes=notes, expected={'n': 3})
    assert '--inspect' in trec['argv'] and trec['argv'][trec['argv'].index('--title') + 1].endswith('INSPECTION (read-only)')
    r = {x['name']: x for x in validate.run('served_page', tpath)}
    assert all(x['passed'] for x in r.values()), r
    ok('inspection twin: cloned argv with --inspect and the read-only title, judge environment, tests pass (28796)')
    # a server of a run outside the parity root on the port refuses (G23)
    plan2 = planmod.make(argparse.Namespace(**dict(vars(a), run_label='OTHER')), f'{T}/procs_out2')
    try:
        servers.to_stop(plan2, 28766)
        raise AssertionError('G23 did not refuse')
    except Refusal as e:
        assert 'G23' in str(e)
    ok('G23: a parity plan never replaces a server whose run lies outside its --out')
    # --restart through the CLI (exact argv, cwd, log, environment)
    env = dict(os.environ, PYTHONPATH=ROOT)
    p = subprocess.run([PY, '-m', 'exopipe.phase3', 'serve', '--restart', '28766', '--out', out, '--port-offset', str(offset),
                        '--cores', 'none'], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=400)
    assert p.returncode == 0, p.stdout.decode()[-800:]
    rr = json.load(open(f'{out}/_records/serve/pages/restart_28766.json'))
    assert rr['argv'] == rec['argv'] and rr['pid'] != rec['pid'] and procs.listeners(28766) == [rr['pid']]
    ok('serve --restart 28766 (CLI): same argv, new pid, exit 0')
    # G17 at the restart (review 2026-10-14 m4): the page's files hold a row on an excluded night -> refused before the
    # running server is stopped
    from exopipe.phase3.core.exclusions import Exclusions
    exr = Exclusions()
    l_, s_, d_ = sorted(k for k in exr.rows if k not in exr.perband and k[1].isdigit())[0]
    sp = f'{U}/source_catalogs_V/{s_}MHz_sources_survivors.csv'
    wcsv(sp, SURV, [dict(rank=1, source_id='J0009+0009', identity_id='', band=f'{s_}MHz', n_det=1,
                         best_cand_id=f'VD_{l_}_{d_}_{s_}MHz_0001', best_date=d_, best_snr=9, review_dir='', snr_rep=4)])
    try:
        p = subprocess.run([PY, '-m', 'exopipe.phase3', 'serve', '--restart', '28766', '--out', out, '--port-offset',
                            str(offset), '--cores', 'none'], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           timeout=400)
        rr2 = json.load(open(f'{out}/_records/serve/restart_28766.json'))
        assert p.returncode == 2 and rr2['status'] == 'refused' and 'G17' in rr2['reason'], (p.stdout.decode()[-500:], rr2)
        assert procs.listeners(28766) == [rr['pid']], 'the running server was stopped'
    finally:
        os.remove(sp)
    ok(f'serve --restart 28766 refused by G17 ({l_} {d_} {s_}MHz excluded now) before the running server is stopped')
    p = subprocess.run([PY, '-m', 'exopipe.phase3', 'serve', '--restart', '8767', '--out', out, '--cores', 'none'], env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
    ur = json.load(open(f'{out}/_records/serve/restart_8767.json'))
    assert p.returncode == 2 and ur['status'] == 'refused' and "not serve's" in ur['reason'], (p.stdout.decode()[-300:], ur)
    ok('serve --restart 8767 refused (frozen Round-2 port, not serve\'s) - exit 2, nothing touched')
    # --stop both
    for port in (28766, 28796):
        spath, srec = servers.stop_port(port, out, notes, offset=offset)
        r = {x['name']: x for x in validate.run('stopped_page', spath)}
        assert all(x['passed'] for x in r.values()), r
    ok('stop by exact pid: stopped_page not_listening passes on both ports')
    # http.server page (the Jupiter v2 site program) and its identity
    www = f'{T}/www'
    os.makedirs(www)
    open(f'{www}/index.html', 'w').write('<a href="/source/J0001%2B0001">x</a> <a href="/source/J0002+0002">y</a>\n')
    hp = procs.start([PY, '-m', 'http.server', '28801', '--bind', '127.0.0.1', '--directory', www], www, dict(os.environ),
                     f'{T}/http.log')
    good, detail = procs.wait_ready(28801, hp, 30, 0.3, f'{T}/http.log')
    assert good, detail
    info = procs.proc_info(hp.pid)
    assert procs.program_of(info) == 'http' and procs.server_port(info) == '28801'
    st, body = procs.http_get(28801, '/index.html')
    assert procs.index_source_ids(body) == {'J0001+0001', 'J0002+0002'}
    procs.stop_exact(info)
    assert not procs.alive(hp.pid)
    ok('http.server page: identity (program http, port from argv), index source links URL-decoded, exact-pid stop')


# ---- 4. the deep views stage end to end (records) -------------------------------------------------------------------------
def test_views_stage():
    import argparse
    from exopipe.phase3.core.record import Run
    from exopipe.phase3.lib.serve import deep, flow
    a = argparse.Namespace(tier='deep', run_label='TEST', port_offset=20000, pages=None, stokes=None, lsts=None, label=None,
                           source=None, family=None, forced_tag=None, restart=None, stop=None, carry_from=None)
    out = f'{T}/stage_out'
    plan = planmod.make(a, out)
    plan.U_src = f'{T}/deep_base/Run_TEST'           # the synthetic run stands in for the production one (mirror source)
    run = Run('serve', ['test'], out, CFG, config_sections=None)
    state = flow.new_state()
    deep.views_stage(run, plan, state)
    u = run.units[-1]
    assert u.status == 'done', (u.status, u.reason, u.failed_tests()[:3])
    assert 'I-multi' in state['refused'] and 'G3' in state['refused']['I-multi']
    rec = json.load(open(f'{out}/_records/serve/deep_Run_TEST_views.json'))
    assert rec['status'] == 'done' and len(rec['outputs']) >= 8
    assert os.path.islink(f'{plan.U}/candidates') and not os.path.islink(f'{plan.U}/judgements')
    ok(f'deep views stage through record.Run: done, {len(rec["outputs"])} outputs tested on write, I-multi refused (G3), '
       f'parity mirror (candidates linked, judgements copied)')


def test_exclusion_gate_per_page_stage():
    """The test stage gates PER PAGE (Gregg 2026-10-14): a page whose server reads a row on an excluded night is refused
    (its start unit refuses; its current server keeps running), the other pages are not; a stale file the server never
    reads changes nothing; under the production value (the parity file) nothing is refused and the hits are reported."""
    from exopipe.phase3.core.exclusions import Exclusions
    from exopipe.phase3.core.record import Run
    from exopipe.phase3.lib.serve import deep, flow
    dd = f'{T}/perpage/data_decisions.csv'
    wcsv(dd, ['lst', 'subband', 'date', 'verdict', 'reason'],          # the V pages' J0002 (Red) and J0006 (23MHz)
         [dict(lst='05h', subband='23MHz', date='2025-01-02', verdict='exclude', reason='x')])
    ex = Exclusions(dd)
    orig, ruled = exclusion_gate.fresh_exclusions, CFG.serve.EXCLUSION_GATE_TIERS
    exclusion_gate.fresh_exclusions = lambda: ex
    try:
        for value, label in ((ruled, 'ruled'), (('snapshot',), 'production')):
            CFG.serve.EXCLUSION_GATE_TIERS = value
            a = _ns(tier='deep', run_label='TEST')
            out = f'{T}/perpage_out_{label}'
            plan = planmod.make(a, out)
            plan.U_src = f'{T}/deep_base/Run_TEST'
            run = Run('serve', ['test'], out, CFG, config_sections=None)
            state = flow.new_state()
            deep.views_stage(run, plan, state)
            pre = dict(state['refused'])
            # a stale file in the I page's folder that its server never reads, holding a row on the excluded night
            wcsv(f'{plan.U}/source_catalogs_I/Red_pages_index_I.csv', IDX,
                 [dict(source_id='J0002+0002_05h_2025-01-02_Red', band='Red', rank=1, review_dir='')])
            u = flow.test_stage(run, plan, state, 'deep_Run_TEST_test', probe=False)
            assert u.status == 'done', (label, u.status, u.reason)
            g = u.meta['exclusion_gate']
            hit = {k for k, v in g['pages'].items() if v['hits']}
            assert 'V' in hit and 'I' not in hit and 'I' in g['pages'], (label, g)
            new = {k: v for k, v in state['refused'].items() if k not in pre}
            if label == 'ruled':
                assert g['refuses'] and set(new) == hit and all(v.startswith('G17') for v in new.values()), (new, hit)
                # start: the refused pages' start units refuse, the others start; the run exits 2
                started = []
                for pg_ in plan.pages:                    # the App probe is not run here (probe=False): stand-in counts
                    state['expected'].setdefault(pg_.key, {'n': 1, 'kind': 'stand-in'})
                orig_rep = servers.replace
                servers.replace = lambda plan_, pg_, utc_, root=None, expected=None, notes=None: (
                    started.append(pg_.key) or (f'{out}/stub_{pg_.key}.json', {}))
                try:
                    flow.start_stage(run, plan, state)
                finally:
                    servers.replace = orig_rep
                su = {x.meta.get('page'): x for x in run.units if x.meta.get('stage') == 'start'}
                assert set(started) == {k for k in su if k not in state['refused']} and 'I' in started, (started, list(su))
                assert all(su[k].status == 'refused' and su[k].reason.startswith('G17') for k in new), \
                    {k: (su[k].status, su[k].reason[:80]) for k in new}
                assert run.exit_code() == 2
                # a later start in ANOTHER invocation after data_decisions.csv changed: the test record is not current (G28)
                open(dd, 'a').write('08h,23MHz,2025-01-05,exclude,y\n')
                try:
                    run2 = Run('serve', ['test'], out, CFG, config_sections=None)
                    try:
                        flow.expected_from_record(run2, 'deep_Run_TEST_test', 'start')
                        raise AssertionError('a start after an exclusion change used the old test record')
                    except Refusal as e:
                        assert 'G28' in str(e) and 'data_decisions' in str(e), e
                finally:
                    with open(dd) as fh:
                        keep = [l for l in fh if not l.startswith('08h,23MHz,2025-01-05')]
                    open(dd, 'w').writelines(keep)
                # the start re-scans each deep page (its files may change after the test stage)
                mem = f'{plan.U}/source_catalogs_I/Red_source_members.csv'
                orows = list(csv.DictReader(open(mem)))
                srow = next(r for r in orows)
                wcsv(mem, list(orows[0].keys()), orows + [dict(srow, cand_id='ID_05h_2025-01-02_Red_0099')])
                state3 = {'refused': {k: v for k, v in state['refused'].items() if k != 'I'}, 'expected': state['expected'],
                          'served': {}}
                run3 = Run('serve', ['test'], out, CFG, config_sections=None)
                servers.replace = lambda *a_, **k_: (_ for _ in ()).throw(AssertionError('I was replaced'))
                try:
                    flow.start_stage(run3, plan, state3)
                finally:
                    servers.replace = orig_rep
                    wcsv(mem, list(orows[0].keys()), orows)
                ui = next(x for x in run3.units if x.meta.get('page') == 'I')
                assert ui.status == 'refused' and 'G17 serve-time exclusion gate at start' in ui.reason, (ui.status, ui.reason)
            else:
                assert not g['refuses'] and not new, new
        ok(f'exclusion gate in the test stage: under the ruled value exactly the pages whose server reads rows on the excluded '
           f'night are refused ({sorted(hit)}) and their start units refuse while the I page (a stale index file only) starts '
           f'(exit 2); a start in another invocation after data_decisions.csv changed refuses (G28); the start re-scans a '
           f'page whose files changed after the test stage; under the production value nothing is refused')
        # sub-LST: the swap is gated on the build directory (every page's server reads the swapped run by path)
        CFG.serve.EXCLUSION_GATE_TIERS = ruled
        R = f'{T}/lstruns'
        olr = sublst_build.lst_run
        sublst_build.lst_run = lambda l: f'{R}/Run_SLST{l[:-1]}B'
        try:
            a2 = _ns(tier='sublst', lsts='05h')
            out2 = f'{T}/perpage_sublst_out'
            plan2 = planmod.make(a2, out2)
            B = sublst_build.build_dir(plan2)
            vpg = next(x for x in plan2.pages if x.key == 'V')
            wcsv(os.path.join(vpg.at('catalog', B), '23MHz_sources_survivors.csv'), SURV,
                 [dict(rank=1, source_id='J0006+0006', identity_id='', band='23MHz', n_det=1,
                       best_cand_id='V1_05h_2025-01-02T03_23MHz_0001', best_date='2025-01-02', best_snr=9, review_dir='',
                       snr_rep=4)])
            run4 = Run('serve', ['test'], out2, CFG, config_sections=None)
            u4 = sublst_build.carry_swap_stage(run4, plan2, flow.new_state())
            assert u4.status == 'refused' and 'G17 serve-time exclusion gate at the swap' in u4.reason, (u4.status, u4.reason)
            assert not os.path.exists(plan2.U) or not os.path.samefile(plan2.U, B)
        finally:
            sublst_build.lst_run = olr
        ok('exclusion gate at the sub-LST swap: a build holding a row on an excluded night is not swapped in (no page served)')
    finally:
        exclusion_gate.fresh_exclusions, CFG.serve.EXCLUSION_GATE_TIERS = orig, ruled


# ---- 5. a refused stage stops the later stages (review 2026-10-13) ----------------------------------------------------------
def _ns(**kw):
    import argparse
    base = dict(tier='deep', run_label=None, port_offset=20000, pages=None, stokes=None, lsts=None, label=None, source=None,
                family=None, forced_tag=None, restart=None, stop=None, carry_from=None, no_twins=False, no_hub=False,
                forced_refresh_siblings=False)
    base.update(kw)
    return argparse.Namespace(**base)


def test_stage_halting():
    from exopipe.phase3.core.exclusions import Exclusions
    from exopipe.phase3.core.record import Run
    from exopipe.phase3.lib.serve import deep, snapshot_copy

    def no_start(*a, **k):
        raise AssertionError('a server was (re)started after a refused stage')
    orig = (servers.replace, servers.start_twin)
    servers.replace = servers.start_twin = no_start
    try:
        # deep: G2 (a union without 'stokes') refuses the views -> no test, no start (the previous views are never served)
        src = f'{T}/halt_deep_src/Run_TEST'
        shutil.copytree(f'{T}/deep_base/Run_TEST', src)
        p = f'{src}/source_catalogs/Red_sources_survivors.csv'
        wcsv(p, SURV, list(csv.DictReader(open(p))))
        a = _ns(tier='deep', run_label='TEST')
        out = f'{T}/halt_deep_out'
        plan = planmod.make(a, out)
        plan.U_src = src
        run = Run('serve', ['test'], out, CFG, config_sections=None)
        deep.run(a, run, plan, deep.STAGES)
        assert [u.name for u in run.units] == ['deep_Run_TEST_views'], [u.name for u in run.units]
        assert run.units[0].status == 'refused' and 'G2' in run.units[0].reason and run.exit_code() == 2
        ok('deep: G2 refuses the views stage and nothing after it runs (no test, no server started) - exit 2')
        # sub-LST: G8 refuses the build -> the views stage never runs on a stale build directory
        R = f'{T}/lstruns'                          # made by test_generation_mix: 05h has no pertie_detections.csv
        olr = sublst_build.lst_run
        sublst_build.lst_run = lambda l: f'{R}/Run_SLST{l[:-1]}B'
        try:
            a2 = _ns(tier='sublst', lsts='05h')
            out2 = f'{T}/halt_sublst_out'
            plan2 = planmod.make(a2, out2)
            B = sublst_build.build_dir(plan2)
            os.makedirs(f'{B}/candidates/I')        # a stale build left by an earlier failed build
            run2 = Run('serve', ['test'], out2, CFG, config_sections=None)
            sublst_build.run(a2, run2, plan2, sublst_build.STAGES)
        finally:
            sublst_build.lst_run = olr
        assert [u.name for u in run2.units] == ['sublst_Run_SLSTB_build'] and run2.units[0].status == 'refused', \
            [(u.name, u.status) for u in run2.units]
        assert 'G8' in run2.units[0].reason and run2.exit_code() == 2 and not os.path.exists(plan2.U)
        ok('sub-LST: G8 refuses the build and nothing after it runs (the stale build is never viewed or swapped in)')
        # sub-LST G11: a page refused on the build (here by the App probe) refuses the swap - nothing swapped in
        state = {'refused': {'V': 'App probe: 2 queue ids the server cannot resolve (404) - not served',
                             'V-multi': 'NO-INPUT: V: no multi-integration sources - multiples page not served'},
                 'expected': {}, 'served': {}}
        u = sublst_build.carry_swap_stage(run2, plan2, state)
        assert u.status == 'refused' and 'G11' in u.reason and 'V-multi' not in u.reason, (u.status, u.reason)
        assert os.path.isdir(B) and not os.path.exists(plan2.U)
        ok('sub-LST G11: a page refused on the build refuses the swap (a multiples page with no source does not)')
        # snapshot: the serve-time exclusion gate refuses the source -> nothing copied
        ex = Exclusions()
        l_, s_, d_ = sorted(k for k in ex.rows if k not in ex.perband and k[1].isdigit())[0]
        srcs = f'{T}/halt_snap_src/Run_SNAPTEST'
        wcsv(f'{srcs}/detections.csv', ['cand_id', 'snr'], [dict(cand_id=f'{l_}_{d_}_{s_}MHz_V_10s_20250101_000000', snr=9)])
        a3 = _ns(tier='snapshot', run_label='Run_SNAPTEST', source=srcs, label='halt test')
        out3 = f'{T}/halt_snap_out'
        plan3 = planmod.make(a3, out3)
        run3 = Run('serve', ['test'], out3, CFG, config_sections=None)
        snapshot_copy.run(a3, run3, plan3, snapshot_copy.STAGES)
        assert [u.name for u in run3.units] == ['snapshot_Run_SNAPTEST_gate'], [u.name for u in run3.units]
        assert run3.units[0].status == 'refused' and 'G17' in run3.units[0].reason and run3.exit_code() == 2
        assert not os.path.exists(plan3.U)
        ok(f'snapshot: G17 refuses the chain run ({l_} {d_} {s_}MHz excluded now) and nothing is copied or served')
    finally:
        servers.replace, servers.start_twin = orig


def test_inplace_gate_agrees():
    """G17 two readers: production's gate (snapshot_serve_exclusion_gate.py through core.legacy.run, its reader
    coadd_full_pipeline.is_excluded) and core.exclusions refuse the same rows (subband exact; a wideband through a member
    subband) - FOLD_IN row f1996ee."""
    from exopipe.phase3.core.exclusions import Exclusions
    ex = Exclusions()
    l_, s_, d_ = sorted(k for k in ex.rows if k not in ex.perband and k[1].isdigit())[0]
    R = f'{T}/gate_run/Run_SNAPG'
    wcsv(f'{R}/detections.csv', ['cand_id', 'cell', 'bands'],
         [dict(cand_id=f'{l_}_{d_}_{s_}MHz_V_10s_x'), dict(cand_id='04h_2030-01-01_Red_V_10s_x'),
          dict(cell=f'{l_}_{d_}', bands='Red;Green')])
    wcsv(f'{R}/judgements/carry_report.csv', ['cand_id'], [dict(cand_id=f'{l_}_{d_}_{s_}MHz_V_10s_x')])  # never scanned
    scan = exclusion_gate.scan_snapshot_run(R)
    rc, out, refusals = exclusion_gate.run_inplace_gate(R)
    agree, detail = exclusion_gate.agree(scan, refusals, rc)
    assert rc == 1 and agree and scan.hits and ('04h', '2030-01-01', 'Red') not in scan.per_key(), (rc, detail)
    ok(f'G17 two readers: the in-place gate (core.legacy.run) and core.exclusions refuse the same rows ({detail})')


def test_forced_probe_before_stop():
    """forced switch: the App probe runs before the old server is stopped (a failed probe keeps the old page served)."""
    from exopipe.phase3.core.record import Run
    from exopipe.phase3.lib.serve import exclusion_gate as eg, flow, forced
    a = _ns(tier='snapshot', family='forced', forced_tag='TEST', pages='forced-V-10s')
    out = f'{T}/forced_out'
    plan = planmod.make(a, out)
    pg = plan.pages[0]
    stopped = []

    class Res:
        ok, rc, tail = True, 0, 'AD Leo: absent'
    old = {'pid': 1, 'comm': 'python3', 'argv': [PY, 'classify_transients.py', '--run', f'{out}/snapshot/OLDRUN', '--port',
                                                 str(pg.port)], 'cwd': OPS}

    def probe_fails(*x, **k):
        raise Refusal('App probe of page forced-V-10s failed (rc 1): test')
    saved = (forced.ready, forced.missing_files, forced._parity_copy, scripts.run, eg.scan_snapshot_run, forced._old_server,
             servers.app_probe, procs.stop_exact)
    forced.ready = lambda name: (True, 'FT-READY')
    forced.missing_files = lambda pg_, D: []
    forced._parity_copy = lambda pg_: None
    scripts.run = lambda *x, **k: Res()
    eg.scan_snapshot_run = lambda D, ex=None: eg.Scan()
    forced._old_server = lambda pg_: (old, {})
    servers.app_probe = probe_fails
    procs.stop_exact = lambda info, **k: stopped.append(info['pid']) or 'stopped'
    try:
        run = Run('serve', ['test'], out, CFG, config_sections=None)
        forced.switch_page(run, plan, pg, flow.new_state(), 'UTC')
    finally:
        (forced.ready, forced.missing_files, forced._parity_copy, scripts.run, eg.scan_snapshot_run, forced._old_server,
         servers.app_probe, procs.stop_exact) = saved
    u = run.units[-1]
    assert u.status == 'refused' and 'App probe' in u.reason and not stopped, (u.status, u.reason, stopped)
    ok('forced switch: a failed App probe refuses BEFORE the old server is stopped (the old page stays served)')


# ---- read-only checks on production -----------------------------------------------------------------------------------------
def test_server_env_mirrors_launcher():
    """serve.SERVER_ENV['snapshot'] = the environment the production launcher gives classify_transients.py
    (snapshot_serve.sh, read at run time): same variable names, same literal values (templated ones are filled per run).
    The staged astrometry bundle added P3_LOC_COMMON_LINKS=1 on install (2026-10-14) - a drift here fails."""
    import re
    lines = [l for l in open(os.path.join(OPS, 'snapshot_serve.sh')) if 'classify_transients.py --run' in l and 'nohup' in l]
    assert lines, 'no classify_transients.py launch line in snapshot_serve.sh'
    want = CFG.serve.SERVER_ENV['snapshot']
    for l in lines:
        env = dict(re.findall(r'\b([A-Z][A-Z0-9_]+)=(\S+)', l.split('nohup')[0]))
        assert set(env) == set(want), ('SERVER_ENV[snapshot] names', sorted(set(env) ^ set(want)))
        lit = {k: v for k, v in want.items() if '{' not in str(v)}
        assert all(env[k] == str(v) for k, v in lit.items()), ('literal values', {k: (env[k], v) for k, v in lit.items()})
    ok(f'SERVER_ENV[snapshot] = snapshot_serve.sh launch environment ({len(lines)} launch lines, {len(want)} variables)')


def test_production_readonly():
    import argparse
    import re
    n_cmp = 0
    for tier, run, extra in (('deep', 'P3S5', {}), ('sublst', None, {}), ('snapshot', 'Run_SNAPB1R_20261004', {}),
                             ('snapshot', None, {'family': 'forced', 'forced_tag': '20261007'}), ('jupiter', None, {})):
        a = argparse.Namespace(tier=tier, run_label=run, port_offset=0, pages=None, stokes=None, lsts=None, label=None,
                               source=None, family=extra.get('family'), forced_tag=extra.get('forced_tag'), restart=None,
                               stop=None)
        p = planmod.make(a, CFG.paths.SNAPSHOT_REVIEW_ROOT if tier == 'snapshot' else CFG.paths.TRANSIENTS_ROOT)
        for pg in p.pages:
            pids = procs.listeners(pg.port)
            if not pids:
                continue
            live = procs.proc_info(pids[0])['argv']
            t = procs.argv_value(live, '--title') or ''
            m = re.search(r'served (\d{4}-\d\d-\d\d \d\d:\d\d UT)', t)
            if tier == 'snapshot' and pg.family == 'review':
                p.label = re.search(r'\((.*)\)$', t).group(1)
            if tier == 'sublst':
                p.lsts_text = re.search(r'LSTs ([0-9h ]+?) - ', t).group(1)
                live = [live[0], f'{p.ops}/{live[1]}'] + live[2:] if not live[1].startswith('/') else live
            mine = p.argv(pg, m.group(1) if m else '')
            assert mine == live, (tier, pg.key, [x for x in zip(live, mine) if x[0] != x[1]][:3])
            n_cmp += 1
    ok(f'planned argv == the live production servers\' argv on {n_cmp} pages (sub-LST: the current launcher\'s absolute '
       f'script path; served time and LST list taken from the live title)')
    env = dict(os.environ, PYTHONPATH=ROOT)
    p = subprocess.run([PY, '-m', 'exopipe.phase3', 'serve', '--tier', 'deep', '--run', 'P3S5', '--dry-run', '--cores', 'none'],
                       env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
    txt = p.stdout.decode()
    assert p.returncode == 0 and 'PLAN exopipe3 serve --tier deep' in txt and 'page V-multi: port 8772' in txt, txt[-600:]
    ok('serve --tier deep --run P3S5 --dry-run: plan printed, exit 0, nothing written')
    D = CFG.paths.TRANSIENTS_ROOT
    for path in (f'{D}/Run_P3S5/judgements/queue_p3s3_V_multi.txt', f'{D}/Run_SLSTB/judgements/queue_p3s3_V.txt'):
        assert all(r['passed'] for r in validate.run('served_queue', path)), path
    assert all(r['passed'] for r in validate.run('served_view', f'{D}/Run_SLSTB/source_catalogs_V'))
    assert all(r['passed'] for r in validate.run('hub_page', f'{D}/review_hub/index.html'))
    ok('content tests on real products (read-only): Run_P3S5 V multiples queue, Run_SLSTB V queue and view, the review hub')


def main():
    t0 = time.time()
    if os.path.exists(T):
        shutil.rmtree(T)
    os.makedirs(T)
    try:
        B = test_deep_ports()
        test_queue_view_content(B)
        test_sublst_ports()
        test_generation_mix()
        test_journals()
        test_exclusion_gate()
        test_exclusion_gate_scan_page()
        test_exclusion_gate_server_mirrors()
        test_hub()
        test_processes()
        test_views_stage()
        test_exclusion_gate_per_page_stage()
        test_stage_halting()
        test_inplace_gate_agrees()
        test_forced_probe_before_stop()
        test_server_env_mirrors_launcher()
        test_production_readonly()
    finally:
        for port in (28766, 28796, 28801):           # never leave a stand-in server behind
            for pid in procs.listeners(port):
                info = procs.proc_info(pid)
                if info and any(T in x for x in info['argv']):
                    try:
                        procs.stop_exact(info)
                    except Exception:
                        pass
        shutil.rmtree(T, ignore_errors=True)
    print(f'ALL {len(PASSED)} CHECKS PASSED in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
