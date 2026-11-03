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
     edits, the generation-mix gate, every content test (positive and negative). THE HUB PAGE (exopipe_ops 4c42f91, Gregg
     2026-10-27): production's build_review_hub.py (read at run time, its hub directory replaced) and lib/serve/hub.render
     write the same page and registry, byte for byte; the content test rejects a tunnel line, the earlier layout, a stale
     page. THE MULTIPLES CHAIN (same commit): the singles pages' --mchain-* arguments, the App probe (the server's own
     loader) with the chain, the chain refusals, the exclusion gate over the chain.
  3. process control on high ports (>= 28700, never a production port): a review page is never given a server of its own;
     a stand-in review server (a script named classify_transients.py in the scratch tree) left on a page's port: listeners,
     readiness, --restart through the CLI, exact-pid --stop; replace() of an http.server page; served_page / stopped_page.
  4. deep `views` stage end to end through record.Run (records under the scratch root).
  5. (review 2026-10-13) a refused stage stops the later ones (deep G2, sub-LST G8 / G11, snapshot G17), the in-place
     exclusion gate and core.exclusions agree, the forced switch probes before anything is registered.
  6. REGISTRATION on the two-port server (lib/serve/register.py; production since 2026-10-22), every server an ISOLATED
     two-port server of this tree on free test ports (>= 19000) over a scratch registry - the live registry, the live server
     and 8765 / 8795 are never touched: the registration = production's register_page.py byte for byte (run on a sandbox);
     the server's registry change = inode, size or time; the publish hold; the build-state wait (queued behind a build,
     failed, replaced, unknown, stalled); the start stage registering the pages in one registry change, the hold lifted or
     kept, the served_page / page_registration tests.
READ-ONLY checks on production: the planned argv of every page against the live servers (titles' served time taken from
the live argv), the planned multiples-chain arguments and the planned registry entries against the live two-port registry
entries, `serve --tier deep --run P3S5 --dry-run`, content tests on real queues and views and on the review hub.
"""
import csv
import glob
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
TWO_PORT_REGISTRY = '/lustre/gh/main/phase3/serve/registry.json'   # the live review server's pages (read only)
PY = '/opt/devel/pipeline/envs/py38_orca_nkosogor/bin/python3'
BASE_PORT = 28700            # the stand-in servers bind 28700-28899 (production 8765-8801, parity offset >= 1000 -> 9765+)

from exopipe.phase3.core import env as _env   # noqa: E402
_env.legacy()
from exopipe.phase3.config import CFG          # noqa: E402
from exopipe.phase3.core import validate      # noqa: E402
from exopipe.phase3.core.record import Refusal  # noqa: E402
from exopipe.phase3.lib.serve import (exclusion_gate, hub, journals, procs, queues, register, scripts,  # noqa: E402
                                      servers, sublst_build, two_port, views)
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


def add_fork(U, ST, rows):
    """Add a deep multiples fork for Stokes ST to the synthetic run U (make_deep gives V one; I has none, the G3 case):
    the wideband fork tier <ST>_fwb with rows [(J-name, lst, date, n_det)] in Red, as make_deep writes V's. The new page
    index keeps the oldest index mtime, so the pages merge stays current."""
    fsurv = [dict(rank=f'rank{n:04d}', source_id=j, identity_id=j, band='Red', n_det=nd,
                  best_cand_id=f'{ST}D_{lst}_{date}_Red_{n:04d}', best_date=date, best_snr=10.0 - n, review_dir='',
                  snr_rep=6.0 - n) for n, (j, lst, date, nd) in enumerate(rows, 1)]
    wcsv(f'{U}/source_catalogs_{ST}_fwb/Red_sources_survivors.csv', SURV, fsurv)
    wcsv(f'{U}/source_catalogs_{ST}_fwb/Red_source_members.csv', MEM,
         [dict(source_id=r['source_id'], band='Red', cand_id=r['best_cand_id'], det_date=r['best_date']) for r in fsurv])
    wcsv(f'{U}/source_catalogs_{ST}_fwb/Red_sources_master.csv', MASTER,
         [dict(source_id=r['source_id'], band='Red', ra=1) for r in fsurv])
    fp = [pid(j, lst, date, 'Red') for j, lst, date, nd in rows]
    t_idx = min(os.path.getmtime(p) for p in glob.glob(f'{U}/source_catalogs/*_pages_index_*.csv'))
    wcsv(f'{U}/source_catalogs/pages_input_{ST}_fwb.csv', PIN,
         [dict(_sid=j, lst=lst, date=date, band='Red', n_det=nd, snr=10.0 - n) for n, (j, lst, date, nd) in enumerate(rows, 1)])
    with open(f'{U}/judgements/queue_p3s3_{ST}_fwb_raw.txt', 'w') as f:
        f.write('# queue_name: i14_posthoc_std7\n' + '\n'.join(fp) + '\n')
    for p_ in fp:
        touch_png(U, f'candidates/I/{p_}')
    ix = f'{U}/source_catalogs/Red_pages_index_{ST}_fwb.csv'
    wcsv(ix, IDX, [dict(source_id=p_, band='Red', rank=n, review_dir=f'candidates/I/{p_}') for n, p_ in enumerate(fp, 1)])
    os.utime(ix, (t_idx, t_idx))
    return fp


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
    reg['V']['10-min-forced'] = {'href': '/s/forced_tiers/10min_V.html', 'live': True}     # a static site: no port
    assert hub.tunnel_extra_with(reg, [8802]) == [8800, 8802]
    ok('hub registry: --set rule, retired tier removal, tunnel_extra union (button ports not duplicated; an href entry '
       'has no port)')


def _hub_registry():
    """A registry with every case the 2026-10-27 page distinguishes (build_review_hub.py, exopipe_ops 4c42f91)."""
    def e(port, run, generation='gen', served='2026-10-31 10:27 UT', live=True):
        return dict(port=port, run=run, generation=generation, served=served, live=live)
    return {'round2': {'port': 8767, 'label': 'Round-2 (v1 masters, frozen)'},
            'V': {'1-hour': e(8766, 'Run_X', generation='generation 4 (Run_20260909_000000)'),
                  '1-hour-multiples': e(8772, 'Run_X'),                                  # no button (Gregg 2026-10-27)
                  '1-hour-forced': e(9122, 'Run_F', generation='forced <deep> & "quoted" HWJ'),   # title escaped
                  '10-min': e(8787, 'Run_SLSTB', generation='LSTs: 00h 01h'),
                  '10-min-forced': {'href': '/s/forced_tiers/10min_V.html?x=1&y=2', 'run': 'Run_G', 'live': False,
                                    'note': 'awaiting the tier evaluations'},          # a static site, not yet served
                  '10-min-groupB-preview': e(9301, 'Run_P'),                            # no button
                  '1-min': e(8781, 'Run_SNAPB'),
                  '1-min-comb': e(8988, 'Run_SNAPCOMB', generation='COMBINED VIEW', served='2026-10-28 02:21 UT'),  # wins
                  '10-sec': e(8779, 'Run_SNAPB'),
                  '10-sec-comb': e(8986, 'Run_SNAPCOMB', live=False),                   # not live: the tier's own entry
                  '10-sec-B6X': e(9070, 'Run_SNAPB6X')},                                # a batch: no button
            'I': {'1-hour': e(8768, 'Run_X', served='', live=False),                    # "served not yet", greyed
                  '10-sec': e(8780, 'Run_SNAPB'),
                  '10-sec-forced': e(8785, 'Run_F')},                                   # I 10-min, 1-min: placeholders
            'tunnel_extra': [8800, 8801]}


def _registry_matches(path, **ctx):
    return {x['name']: x for x in validate.run('hub_page', path, **ctx)}['hub_page.registry_matches']


def _prod_hub(D, args=(), script=None):
    """Production's build_review_hub.py (read at run time) with its hard-coded hub directory replaced by D; returns rc,
    output. ``script``: another version's text (e.g. the layout before 4c42f91)."""
    code = script if script is not None else open(f'{OPS}/build_review_hub.py').read()
    needle = "D = '/data07/gh/phase3/transients/review_hub'"
    assert code.count(needle) == 1, 'build_review_hub.py changed its hub directory line: update the test'
    return run_heredoc(code.replace(needle, f'D = {D!r}'), list(args))


def test_hub_page():
    """THE HUB PAGE (Gregg 2026-10-27; exopipe_ops 4c42f91): lib/serve/hub.render IS the page production's
    build_review_hub.py writes (run here on scratch registries), its --set and --tunnel-extra registry writes equal
    hub.apply_set / tunnel_extra_with; the content test passes on such a page and fails on a page with a tunnel line, the
    previous layout, a stale page or a malformed entry; a parity serve writes the registry and the page under --out."""
    from types import SimpleNamespace
    D = f'{T}/hub_prod'
    os.makedirs(D)
    reg = _hub_registry()
    with open(f'{D}/registry.json', 'w') as f:
        json.dump(reg, f, indent=1)
    rc, out = _prod_hub(D)
    assert rc == 0, out
    page = open(f'{D}/index.html').read()
    when = hub.page_generated(page)
    assert when and hub.render(json.load(open(f'{D}/registry.json')), when) == page, \
        hub.first_difference(hub.render(reg, when or ''), page)
    for must in ('>1-hour</a>', 'href="http://localhost:8988/"', 'href="http://localhost:8779/"',
                 'href="/s/forced_tiers/10min_V.html?x=1&amp;y=2"', '&lt;deep&gt; &amp; &quot;quoted&quot;',
                 '<span class="b off" title="not yet served">1-min</span>', 'served not yet', '/s/jupiter/', '/s/planets/'):
        assert must in page, must
    for never in ('8772', '9301', '9070', '8986', 'Tunnel', 'ssh ', '8767', 'multiples'):
        assert never not in page, never
    # --set and --tunnel-extra: the registry production writes = the refactor's rule (byte for byte), the page = render()
    ref = json.load(open(f'{D}/registry.json'))
    kvs = ['run=Run_SLSTB', 'generation=new chain, LSTs: 00h 01h 02h', 'served=2026-11-02 01:00 UT', 'live=1', 'port=8787']
    rc, out = _prod_hub(D, ['--set', 'I', '10-min'] + kvs)
    assert rc == 0, out
    hub.apply_set(ref, 'I', '10-min', kvs)
    hub._dump(ref, f'{T}/hub_ref/registry.json')
    assert open(f'{D}/registry.json').read() == open(f'{T}/hub_ref/registry.json').read()
    te = hub.tunnel_extra_with(ref, [18800])
    rc, out = _prod_hub(D, ['--tunnel-extra', ','.join(map(str, te))])
    assert rc == 0, out
    ref['tunnel_extra'] = te
    hub._dump(ref, f'{T}/hub_ref/registry.json')
    assert open(f'{D}/registry.json').read() == open(f'{T}/hub_ref/registry.json').read()
    page = open(f'{D}/index.html').read()
    assert page == hub.render(ref, hub.page_generated(page)) and '18800' not in page
    ok('hub page: lib/serve/hub.render == the page production\'s build_review_hub.py writes (V over I; per tier its page - '
       'the live combined view first - and its forced button, placeholders while unserved; three links; no multiples / '
       'preview / batch button, no Round-2, no tunnel line), and its --set / --tunnel-extra registry writes, byte for byte')
    # the content test on the production-written hub
    exp = [('I', '10-min', dict(kv.split('=', 1) for kv in kvs))]
    r = {x['name']: x for x in validate.run('hub_page', f'{D}/index.html', expected=exp)}
    assert r['hub_page.registry_matches']['passed'], r
    assert _registry_matches(f'{D}/registry.json', expected=exp)['passed']
    exp_bad = [('I', '10-min', dict(exp[0][2], served='2026-11-02 02:00 UT'))]
    assert not _registry_matches(f'{D}/index.html', expected=exp_bad)['passed']

    def failing(label, why, edit_reg=None, script=None, edit_page=None):
        """The content test on a copy of the hub: the registry edited, then (script) the page written by another version
        of build_review_hub.py, then the page edited; it must fail, saying ``why``."""
        d2 = f'{T}/hub_neg_{label}'
        shutil.copytree(D, d2)
        if edit_reg:
            g = json.load(open(f'{d2}/registry.json'))
            edit_reg(g)
            hub._dump(g, f'{d2}/registry.json')
        if script is not None:
            rc_, out_ = _prod_hub(d2, script=script)
            assert rc_ == 0, out_
        if edit_page:
            p_ = f'{d2}/index.html'
            text = edit_page(open(p_).read())
            open(p_, 'w').write(text)
        res = _registry_matches(f'{d2}/index.html')
        assert not res['passed'] and why in res['detail'], (label, res)
        return res['detail']
    failing('tunnel', 'is not the page', edit_page=lambda s: s.replace(
        '</body>', '<div class="note">Tunnel: ssh -YA calim7 -L 8765:127.0.0.1:8765</div>\n</body>'))
    old = subprocess.run(['git', '-C', OPS, 'show', '4c42f91^:build_review_hub.py'], stdout=subprocess.PIPE, check=True)
    # the layout before 4c42f91 (a static-site entry has no port there: left out)
    det = failing('old_layout', 'is not the page', edit_reg=lambda g: g['V'].pop('10-min-forced'),
                  script=old.stdout.decode())
    failing('stale', 'is not the page', edit_reg=lambda g: g['V']['1-hour'].update(served='2026-11-02 03:00 UT'))
    failing('handmade', 'not generated from registry.json',
            edit_page=lambda s: s.replace(f'Page generated {hub.page_generated(s)} from registry.json', 'Edited'))
    failing('no_port', 'not an int (and no href)', edit_reg=lambda g: g['I'].update({'1-min': {'run': 'Run_Q', 'live': True}}))
    ok(f'hub_page.registry_matches: passes on the page production writes (and its expected entries), fails on a tunnel line, '
       f'the layout before 4c42f91 ({det[:70]}...), a registry changed after the page, a page not generated from the '
       f'registry and an entry with neither port nor href')
    # parity: the registry and the page are written under --out, from the production registry (here: the scratch hub)
    orig = CFG.serve.HUB_DIR
    CFG.serve.HUB_DIR = D
    try:
        plan = SimpleNamespace(parity=True, hub_dir=f'{T}/hub_parity/review_hub')
        entries = [('V', '1-hour', ['run=Run_TEST', 'generation=generation 4 (Run_20260909_000000)',
                                    'served=2026-11-02 04:00 UT', 'live=1', 'port=18766'])]
        R, index, planned = hub.update(plan, entries, remove=('10-sec-B6X',), tunnel=[18800, 18801])
    finally:
        CFG.serve.HUB_DIR = orig
    assert index == f'{plan.hub_dir}/index.html' and R == f'{plan.hub_dir}/registry.json', (R, index)
    want = json.load(open(f'{D}/registry.json'))
    hub.apply_set(want, *entries[0])
    hub.remove_tiers(want, ('10-sec-B6X',))
    want['tunnel_extra'] = [8800, 8801, 18800, 18801]
    assert json.load(open(R)) == want and planned[-1] == ['--tunnel-extra', '8800,8801,18800,18801'], planned
    pp = open(index).read()
    assert pp == hub.render(want, hub.page_generated(pp)) and 'href="http://localhost:18766/"' in pp
    r = validate.run('hub_page', index, expected=[(s, t, dict(kv.split('=', 1) for kv in k)) for s, t, k in entries])
    assert all(x['passed'] for x in r), r
    ok('hub, parity: registry.json (production\'s + this run\'s --set, the retired tier removed, tunnel_extra) and '
       'index.html (the page build_review_hub.py would write from it) under --out; hub_page tests pass')


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
    """Process control for the servers serve still handles by port (a review server left on a page's port from before the
    registration, the Jupiter pages): a review page is never given a server of its own (replace refuses it); --restart through
    the CLI (exact argv, cwd, log, environment), G17 at the restart, the frozen 8767 refused, exact-pid --stop; replace() of an
    http.server page (the Jupiter v2 site program) with its served_page tests, a second replace stopping the first by pid."""
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
    try:
        servers.replace(plan, pg, 'UTC', expected={'n': 3, 'kind': 'queue ids'})
        raise AssertionError('a review page was given a server of its own')
    except Refusal as e:
        assert 'registered on the two-port server' in str(e), e
    ok('replace: a review page is never given a server of its own (it is registered on the two-port server)')
    for port in (28766, 28801):
        assert not procs.port_in_use(port), f'port {port} busy: the test needs it free'
    argv = [PY, f'{fake_dir}/classify_transients.py', '--run', U, '--queue', q, '--labels-dir', f'{U}/judgements_V',
            '--catalog-dir', f'{U}/source_catalogs_V', '--port', '28766', '--title', 'test page - served UTC']
    sp = procs.start(argv, OPS, plan.server_env(), f'{U}/server_V.log')      # a per-port review server from before
    good, detail = procs.wait_ready(28766, sp, 30, 0.3, f'{U}/server_V.log')
    assert good, detail
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
    assert rr['argv'] == argv and rr['pid'] != sp.pid and procs.listeners(28766) == [rr['pid']]
    ok('serve --restart 28766 (CLI): same argv, new pid, exit 0')
    # G17 at the restart (review 2026-10-14 m4): the page's files hold a row on an excluded night -> refused before the
    # running server is stopped
    from exopipe.phase3.core.exclusions import Exclusions
    exr = Exclusions()
    l_, s_, d_ = sorted(k for k in exr.rows if k not in exr.perband and k[1].isdigit())[0]
    sp_ = f'{U}/source_catalogs_V/{s_}MHz_sources_survivors.csv'
    wcsv(sp_, SURV, [dict(rank=1, source_id='J0009+0009', identity_id='', band=f'{s_}MHz', n_det=1,
                          best_cand_id=f'VD_{l_}_{d_}_{s_}MHz_0001', best_date=d_, best_snr=9, review_dir='', snr_rep=4)])
    try:
        p = subprocess.run([PY, '-m', 'exopipe.phase3', 'serve', '--restart', '28766', '--out', out, '--port-offset',
                            str(offset), '--cores', 'none'], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           timeout=400)
        rr2 = json.load(open(f'{out}/_records/serve/restart_28766.json'))
        assert p.returncode == 2 and rr2['status'] == 'refused' and 'G17' in rr2['reason'], (p.stdout.decode()[-500:], rr2)
        assert procs.listeners(28766) == [rr['pid']], 'the running server was stopped'
    finally:
        os.remove(sp_)
    ok(f'serve --restart 28766 refused by G17 ({l_} {d_} {s_}MHz excluded now) before the running server is stopped')
    p = subprocess.run([PY, '-m', 'exopipe.phase3', 'serve', '--restart', '8767', '--out', out, '--cores', 'none'], env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
    ur = json.load(open(f'{out}/_records/serve/restart_8767.json'))
    assert p.returncode == 2 and ur['status'] == 'refused' and "not serve's" in ur['reason'], (p.stdout.decode()[-300:], ur)
    ok('serve --restart 8767 refused (frozen Round-2 port, not serve\'s) - exit 2, nothing touched')
    notes = []
    spath, srec = servers.stop_port(28766, out, notes, offset=offset)
    r = {x['name']: x for x in validate.run('stopped_page', spath)}
    assert all(x['passed'] for x in r.values()), r
    ok('stop by exact pid: stopped_page not_listening passes')
    # replace() of a page with a server of its own: the Jupiter v2 site program (http.server) over a scratch site
    www = f'{T}/www'
    os.makedirs(www)
    open(f'{www}/index.html', 'w').write('<a href="/source/J0001%2B0001">x</a> <a href="/source/J0002+0002">y</a>\n')
    out_j = f'{T}/procs_out_j'
    plan_j = planmod.make(_ns(tier='jupiter', port_offset=offset), out_j)
    hpg = next(p_ for p_ in plan_j.pages if p_.key == 'v2-site')
    hpg.root = www
    assert hpg.port == 28801 and hpg.program == 'http'
    path, rec = servers.replace(plan_j, hpg, 'UTC', notes=notes)
    r = {x['name']: x for x in validate.run('served_page', path)}
    assert all(x['passed'] for x in r.values()) and procs.listeners(28801) == [rec['pid']], r
    info = procs.proc_info(rec['pid'])
    assert procs.program_of(info) == 'http' and procs.server_port(info) == '28801'
    st, body = procs.http_get(28801, '/index.html')
    assert procs.index_source_ids(body) == {'J0001+0001', 'J0002+0002'}
    old_pid = rec['pid']
    path, rec = servers.replace(plan_j, hpg, 'UTC2', notes=notes)
    assert rec['pid'] != old_pid and not procs.alive(old_pid) and procs.listeners(28801) == [rec['pid']]
    spath, srec = servers.stop_port(28801, out_j, notes, offset=offset)
    assert not procs.alive(rec['pid'])
    ok('http.server page (the Jupiter v2 site program): replace() starts it (served_page live / same_process pass; identity '
       'program http, port from argv; index links URL-decoded), a second replace stops the first by exact pid, --stop')


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
    hp = register.hold_file(plan.U)
    txt = open(hp).read()
    assert not os.path.islink(hp) and 'exopipe3 serve rewrites its run' in txt and any('publish marker' in n for n in u.notes), txt
    src_hold = register.hold_file(plan.U_src)        # a hold that stands - the production publish's, which the parity
    with open(src_hold, 'w') as f:                   # mirror links - is kept with its own text, never written through
        f.write('the publish\'s own text\n')
    try:
        run2 = Run('serve', ['test'], out, CFG, config_sections=None)
        deep.views_stage(run2, plan, flow.new_state())
        assert run2.units[-1].status == 'done' and os.path.islink(hp) and open(hp).read() == 'the publish\'s own text\n'
        lifted = register.lift_hold(plan.U)          # lifting it renames the mirror's link, not the production file
        assert os.path.islink(lifted) and open(src_hold).read() == 'the publish\'s own text\n' and not os.path.lexists(hp)
    finally:
        os.remove(src_hold)
    stale = f'{T}/stage_stale_src/Run_TEST'          # the pages merge stale: refused BEFORE the marker is written
    shutil.copytree(plan.U_src, stale)
    t_ = time.time() + 10
    os.utime(f'{stale}/source_catalogs/Red_pages_index_V.csv', (t_, t_))
    plan3 = planmod.make(a, f'{T}/stage_out_stale')
    plan3.U_src = stale
    run3 = Run('serve', ['test'], plan3.out, CFG, config_sections=None)
    deep.views_stage(run3, plan3, flow.new_state())
    assert run3.units[-1].status == 'refused' and not os.path.lexists(register.hold_file(plan3.U)), run3.units[-1].reason
    ok(f'deep views stage through record.Run: done, {len(rec["outputs"])} outputs tested on write, I-multi refused (G3), '
       f'parity mirror (candidates linked, judgements copied); THE PUBLISH MARKER written once the pages merge is current (a '
       f'standing marker - the production run\'s, linked by the mirror - kept with its own text and removed by renaming the '
       f'link only; a stale merge refuses before any marker is written)')


def test_mchain():
    """THE MULTIPLES CHAIN (Gregg 2026-10-27, exopipe_ops 4c42f91): the deep and 10-minute singles pages are served with
    --mchain-queue (deep: and --mchain-catalog-dir) right after their title, as their two-port registry entries carry
    them, and no other page is; the review server defines both options; the App probe counts the chain identities that
    join the page; as production, a page is served with the chain on disk whatever its multiples page became in the same
    serve (an empty multiples queue, refused G1, leaves an empty chain: the review of 2026-11-02), refused only when its
    --mchain-queue FILE is absent (the server dies reading it), an absent chain view noted; the sub-LST views write the
    multi-integration queue for every Stokes they build, as sublst_cumulative.sh:79 does."""
    import ast
    from exopipe.phase3.core.record import Run
    from exopipe.phase3.lib.serve import common, deep, flow
    out = f'{T}/mchain_out'
    plan = planmod.make(_ns(tier='deep', run_label='TEST'), out)
    assert plan.parity and not plan.gates, plan.gates
    U = plan.U
    for pg in plan.pages:
        a = plan.argv(pg, 'UTC')
        if pg.kind == 'single':
            tail = a[a.index('--title') + 2:]
            assert tail == ['--mchain-queue', f'{U}/judgements/queue_p3s3_{pg.stokes}_multi.txt', '--mchain-catalog-dir',
                            f'{U}/views/source_catalogs_{pg.stokes}_multi', '--lists-dir', plan.lists_dir], (pg.key, tail)
        else:
            assert not [x for x in a if x.startswith('--mchain')], (pg.key, a)
    p2 = planmod.make(_ns(tier='sublst', lsts='05h'), f'{T}/mchain_sublst_out')
    B = sublst_build.build_dir(p2)
    for pg in p2.pages:
        a = p2.argv(pg, 'UTC', root=B)
        want = ['--mchain-queue', f'{B}/judgements/queue_p3s3_{pg.stokes}_multi.txt'] if pg.kind == 'single' else []
        assert a[a.index('--title') + 2:] == want + ['--lists-dir', p2.lists_dir], (pg.key, a[-6:])
    others = [planmod.make(_ns(tier='snapshot', run_label='Run_SNAPTEST', source='/nowhere', label='t'), f'{T}/mchain_s'),
              planmod.make(_ns(tier='snapshot', family='forced', forced_tag='20261007'), f'{T}/mchain_f')]
    assert not [x for p_ in others for pg in p_.pages for x in p_.argv(pg, 'UTC') if x.startswith('--mchain')]
    tree = ast.parse(open(f'{OPS}/classify_transients.py').read())
    opts = {n.args[0].value for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, 'attr', '') == 'add_argument'
            and n.args and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str)}
    assert {'--mchain-queue', '--mchain-catalog-dir'} <= opts, sorted(o for o in opts if 'mchain' in o)
    ok('multiples chain: deep V/I pages carry --mchain-queue + --mchain-catalog-dir, sub-LST V/I --mchain-queue, after the '
       'title (before a parity --lists-dir); no multiples, snapshot or forced page does; classify_transients.py defines both')
    # the stages below read an exclusion list with no row (the gate is not what they test)
    from exopipe.phase3.core.exclusions import Exclusions
    dd = f'{T}/mchain_dd/data_decisions.csv'
    wcsv(dd, ['lst', 'subband', 'date', 'verdict', 'reason'], [])
    ex0, orig_fx = Exclusions(dd), exclusion_gate.fresh_exclusions
    exclusion_gate.fresh_exclusions = lambda: ex0
    try:
        # the App probe (the server's own loader, the deep pages' environment) on a synthetic deep run: V's fork has J0001
        # (a page source) and J0009 (fork only); I has no fork (I-multi refused, G3: no chain written)
        plan.U_src = f'{T}/deep_base/Run_TEST'
        run = Run('serve', ['test'], out, CFG, config_sections=None)
        state = flow.new_state()
        deep.views_stage(run, plan, state)
        assert run.units[-1].status == 'done' and 'G3' in state['refused'].get('I-multi', ''), state['refused']
        V = next(p for p in plan.pages if p.key == 'V')
        r = servers.app_probe(plan, V, U)
        c = V.c
        V.c = {k: v for k, v in c.items() if not k.startswith('mchain')}
        try:
            r0 = servers.app_probe(plan, V, U)
        finally:
            V.c = c
        assert (r['n'], r['mchain_ids'], r['n_unexplained']) == (r0['n'] + 1, 2, 0) and r0['mchain_ids'] is None, (r, r0)
        ok(f'multiples chain: the App probe loads the page as its server does - {r["n"]} index sources with the chain (2 '
           f'chain identities, the fork-only one joins), {r0["n"]} without')
        # the test stage: V is served with its chain (inputs of the stage); I's chain queue was never written (I-multi G3) -
        # its server would die reading it, so I is not served; an absent chain VIEW is only noted (the server skips it)
        u = flow.test_stage(run, plan, state, 'deep_Run_TEST_test', probe=False)
        assert u.status == 'done', (u.status, u.reason)
        why = state['refused'].get('I', '')
        assert why.startswith('multiples chain: --mchain-queue') and 'is absent' in why, why
        assert 'V' not in state['refused'], state['refused']
        assert f'{U}/judgements/queue_p3s3_V_multi.txt' in u.inputs and \
            f'{U}/views/source_catalogs_V_multi/Red_sources_survivors.csv' in u.inputs, sorted(u.inputs)[:6]
        V.c = dict(c, mchain_catalog='views/none')
        try:
            why_v, notes_v = flow.mchain_check(V, U)
        finally:
            V.c = c
        assert why_v is None and len(notes_v) == 1 and 'is absent' in notes_v[0], (why_v, notes_v)
        ok('multiples chain: a page whose --mchain-queue file is absent is not served (I: I-multi refused G3 wrote no chain - '
           'its server would die reading it); an absent --mchain-catalog-dir is noted and the page served; the chain queue '
           'and view are test-stage inputs')
        # the review's scenario (2026-11-02): the fork lists nothing - an empty deep multiples queue, V-multi refused (G1) -
        # and the V page is served all the same, with an empty chain, as production serves it
        src2 = f'{T}/mchain_empty_src/Run_TEST'
        shutil.copytree(f'{T}/deep_base/Run_TEST', src2)
        open(f'{src2}/judgements/queue_p3s3_V_fwb_raw.txt', 'w').write('# queue_name: i14_posthoc_std7\n')
        pe = planmod.make(_ns(tier='deep', run_label='TEST', pages='V,V-multi'), f'{T}/mchain_empty_out')
        pe.U_src = src2
        rune = Run('serve', ['test'], pe.out, CFG, config_sections=None)
        ste = flow.new_state()
        deep.views_stage(rune, pe, ste)
        assert ste['refused'].get('V-multi', '').startswith('G1 empty queue'), ste['refused']
        ue = flow.test_stage(rune, pe, ste, 'deep_Run_TEST_test')
        assert ue.status == 'done' and 'V' not in ste['refused'], (ue.status, ue.reason, ste['refused'])
        xv = ste['expected']['V']
        assert (xv['n'], xv['mchain_ids'], xv['n_unexplained']) == (r0['n'], 0, 0), xv
    finally:
        exclusion_gate.fresh_exclusions = orig_fx
    ok(f'multiples chain: an empty deep multiples queue (V-multi refused, G1) leaves the V page served with an empty chain '
       f'({xv["n"]} index sources, 0 chain identities - App probe), as production serves it')
    # sub-LST: the multi-integration queue is written whenever the Stokes is built (here the V singles page alone)
    p5 = planmod.make(_ns(tier='sublst', lsts='04h,05h,06h', pages='V'), f'{T}/mchain_sublst_views')
    B5 = sublst_build.build_dir(p5)
    shutil.copytree(f'{T}/sub_base/Run_SLSTB.build', B5)
    run5 = Run('serve', ['test'], p5.out, CFG, config_sections=None)
    st5 = flow.new_state()
    u5 = sublst_build.views_stage(run5, p5, st5)
    assert u5.status == 'done' and not st5['refused'], (u5.status, u5.reason, st5)
    assert common.has_j_line(f'{B5}/judgements/queue_p3s3_V_multi.txt') and not os.path.exists(f'{B5}/source_catalogs_V_multi')
    assert not os.path.exists(f'{B5}/judgements/queue_p3s3_I_multi.txt')
    ok('multiples chain, sub-LST: the views stage writes the multi-integration queue of every Stokes it builds (the singles '
       'page reads it after the swap), the multiples view only for a served multiples page')


def test_hub_stage():
    """The hub stage as production's serves set the hub today: the entries of the pages served and - production's
    10-minute two-port path (sublst_cumulative.sh:298-299) - the multiples entry of a sub-LST page not served for NO-INPUT
    only; a page refused for any other reason gets none; no separate inspection hub page and no hub HTTP server (both
    retired by production on 2026-10-27: the review hub is served on both listeners of the two-port server)."""
    from exopipe.phase3.core.record import Run
    from exopipe.phase3.lib.serve import flow
    D = f'{T}/hubstage_prod'
    hub._dump(_hub_registry(), f'{D}/registry.json')
    plan = planmod.make(_ns(tier='sublst', lsts='04h,05h,06h'), f'{T}/hubstage_out')
    run = Run('serve', ['test'], plan.out, CFG, config_sections=None)
    state = flow.new_state()
    state['served'].update({'V': ('stub', {}), 'I': ('stub', {})})
    state['refused'].update({'V-multi': 'NO-INPUT: V: no multi-integration sources - multiples page not served',
                             'I-multi': 'G17 serve-time exclusion gate: 1 rows ... - NOT served'})
    orig = CFG.serve.HUB_DIR
    CFG.serve.HUB_DIR = D                          # the parity hub starts from this scratch 'production' registry
    try:
        flow.hub_stage(run, plan, state, remove=CFG.serve.HUB_REMOVE_TIERS['sublst'], utc='2026-11-02 05:00 UT')
    finally:
        CFG.serve.HUB_DIR = orig
    u = run.units[-1]
    assert u.status == 'done', (u.status, u.reason, u.failed_tests()[:2])
    reg = json.load(open(f'{plan.out}/review_hub/registry.json'))
    for S_, t, port in (('V', '10-min', 28787), ('I', '10-min', 28788), ('V', '10-min-multiples', 28789)):
        e = reg[S_].get(t) or {}
        assert (e.get('port'), e.get('served'), e.get('run')) == (port, '2026-11-02 05:00 UT', 'Run_SLSTB'), (S_, t, e)
    assert '10-min-multiples' not in reg['I'], reg['I'].get('10-min-multiples')
    assert hub.SERVED_NOTE in u.notes and not os.path.exists(f'{plan.out}/inspection_hub'), u.notes
    assert not [c for c in u.meta.get('legacy_calls', []) if 'build_inspection_hub' in ' '.join(map(str, c.get('argv', [])))]
    ok('hub stage: the served pages\' entries, and the 10-minute multiples entry of a page not served for NO-INPUT only (as '
       'production\'s two-port path); a page refused otherwise gets none; no inspection hub page, no hub HTTP server')


def test_exclusion_gate_mchain():
    """The serve-time exclusion gate reads a page's multiples chain as its server does (classify_transients.load_sources
    extra_dir, exopipe_ops 4c42f91): the chain queue's page ids and, from the chain's view, only the identities the page's
    own catalogue does not hold - their rows, page folders and members; a held identity's chain rows are never read."""
    from exopipe.phase3.core.exclusions import Exclusions
    U = f'{T}/scan_chain/Run_X'
    C, M = f'{U}/source_catalogs_V', f'{U}/views/source_catalogs_V_multi'
    row = dict(rank=1, identity_id='', n_det=2, best_date='', best_snr=9, snr_rep=4, band='64MHz')
    wcsv(f'{C}/64MHz_sources_survivors.csv', SURV, [dict(row, source_id='JA', best_cand_id='VD_04h_2025-01-01_64MHz_0001',
                                                         review_dir='candidates/V/pa')])
    wcsv(f'{M}/64MHz_sources_survivors.csv', SURV, [
        dict(row, source_id='JA', best_cand_id='VD_06h_2025-01-03_64MHz_0001', review_dir='candidates/V/pa2'),   # held
        dict(row, source_id='JB', best_cand_id='VD_07h_2025-01-04_64MHz_0002', review_dir='candidates/V/pb')])   # chain only
    wcsv(f'{M}/64MHz_source_members.csv', MEM, [dict(source_id='JA', band='64MHz', cand_id='VD_06h_2025-01-03_64MHz_0003'),
                                               dict(source_id='JB', band='64MHz', cand_id='VD_07h_2025-01-04_64MHz_0004')])
    for d, png in (('pa', '04h_2025-01-01'), ('pa2', '06h_2025-01-03'), ('pb', '07h_2025-01-04')):
        os.makedirs(f'{U}/candidates/V/{d}')
        open(f'{U}/candidates/V/{d}/rank00001_{png}_64MHz.png', 'w').close()
    os.makedirs(f'{U}/judgements')
    q, mq = f'{U}/judgements/queue_p3s3_V.txt', f'{U}/judgements/queue_p3s3_V_multi.txt'
    open(q, 'w').write('JA_04h_2025-01-01_64MHz\n')
    open(mq, 'w').write('JA_04h_2025-01-01_64MHz\nJB_07h_2025-01-04_64MHz\n')

    def excluding(lst, date):
        dd = f'{T}/scan_chain/dd_{lst}.csv'
        wcsv(dd, ['lst', 'subband', 'date', 'verdict', 'reason'],
             [dict(lst=lst, subband='64MHz', date=date, verdict='exclude', reason='x')])
        return Exclusions(dd)
    s_held = exclusion_gate.scan_page(U, C, q, ex=excluding('06h', '2025-01-03'), mchain_queue=mq, mchain_catalog=M)
    assert not s_held.hits, s_held.summary()
    ex3 = excluding('07h', '2025-01-04')
    s_b = exclusion_gate.scan_page(U, C, q, ex=ex3, mchain_queue=mq, mchain_catalog=M)
    assert sorted(h[0] for h in s_b.hits) == ['candidates/V/pb/rank00001_07h_2025-01-04_64MHz.png',
                                              'judgements/queue_p3s3_V_multi.txt',
                                              'views/source_catalogs_V_multi/64MHz_source_members.csv',
                                              'views/source_catalogs_V_multi/64MHz_sources_survivors.csv'], s_b.summary()
    assert not exclusion_gate.scan_page(U, C, q, ex=ex3).hits
    argv = [PY, 'classify_transients.py', '--run', U, '--queue', q, '--catalog-dir', C, '--port', '1', '--title', 't',
            '--mchain-queue', mq, '--mchain-catalog-dir', M]
    assert sorted(exclusion_gate.scan_server_argv(argv, ex=ex3).hits) == sorted(s_b.hits)
    ok('exclusion gate per page, multiples chain: the chain-only identity\'s survivor row, panel and member and the chain '
       'queue line are found (4 hits); the held identity\'s chain rows are never read (0 hits on their night); a running '
       'server\'s argv is scanned the same way')


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
    # a run with both multiples forks, as production's (each singles page serves its Stokes' chain, 4c42f91): make_deep's
    # (V only) plus an I fork - J0007 (an I page source) and J0010 (fork only), on nights the exclusion leaves alone
    src = f'{T}/perpage_src/Run_TEST'
    shutil.copytree(f'{T}/deep_base/Run_TEST', src)
    add_fork(src, 'I', [('J0007+0007', '08h', '2025-01-05', 2), ('J0010+0010', '09h', '2025-01-06', 1)])
    orig, ruled = exclusion_gate.fresh_exclusions, CFG.serve.EXCLUSION_GATE_TIERS
    exclusion_gate.fresh_exclusions = lambda: ex
    try:
        for value, label in ((ruled, 'ruled'), (('snapshot',), 'production')):
            CFG.serve.EXCLUSION_GATE_TIERS = value
            a = _ns(tier='deep', run_label='TEST')
            out = f'{T}/perpage_out_{label}'
            plan = planmod.make(a, out)
            plan.U_src = src
            run = Run('serve', ['test'], out, CFG, config_sections=None)
            state = flow.new_state()
            deep.views_stage(run, plan, state)
            pre = dict(state['refused'])
            assert not pre, pre                             # both forks: every page's queue and view built
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
                # start: the refused pages' start units refuse, the others are registered (an isolated two-port server on
                # the plan's parity listeners, the stand-in page code); the run exits 2
                for pg_ in plan.pages:                    # the App probe is not run here (probe=False): the stand-in's count
                    state['expected'].setdefault(pg_.key, {'n': queue_identities(pg_.at('queue', plan.U)),
                                                           'kind': 'stand-in index'})
                plan.ops = standin_dir()
                srv = TwoPortServer(register.registry_dir(plan), *register.listeners(plan))   # stopped after run3 (else main)
                flow.start_stage(run, plan, state)
                su = {x.meta.get('page'): x for x in run.units if x.meta.get('stage') == 'start' and x.meta.get('page')}
                started = {k for k, x in su.items() if x.status == 'done'}
                reg_ids = {e['id'] for e in json.load(open(f'{register.registry_dir(plan)}/registry.json'))['pages']}
                assert started == {k for k in su if k not in state['refused']} and 'I' in started, (started, list(su))
                assert reg_ids == {f'p{p_.port}' for p_ in plan.pages if p_.key in started}, (reg_ids, started)
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
                orig_w = register.write
                register.write = lambda *a_, **k_: (_ for _ in ()).throw(AssertionError('I was registered'))
                all_pages, plan.pages = plan.pages, [p_ for p_ in plan.pages if p_.key == 'I']
                try:
                    flow.start_stage(run3, plan, state3)
                finally:
                    register.write = orig_w
                    plan.pages = all_pages
                    wcsv(mem, list(orows[0].keys()), orows)
                    srv.stop()
                ui = next(x for x in run3.units if x.meta.get('page') == 'I')
                assert ui.status == 'refused' and 'G17 serve-time exclusion gate at start' in ui.reason, (ui.status, ui.reason)
            else:
                assert not g['refuses'] and not new, new
        ok(f'exclusion gate in the test stage: under the ruled value exactly the pages whose server reads rows on the excluded '
           f'night are refused ({sorted(hit)}) and their start units refuse while the I page (a stale index file only) is '
           f'registered with the others (one registration; exit 2); a start in another invocation after data_decisions.csv '
           f'changed refuses (G28); the start re-scans a page whose files changed after the test stage (not registered); under '
           f'the production value nothing is refused')
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
                family=None, forced_tag=None, restart=None, stop=None, carry_from=None, no_hub=False,
                forced_refresh_siblings=False)
    base.update(kw)
    return argparse.Namespace(**base)


def test_stage_halting():
    from exopipe.phase3.core.exclusions import Exclusions
    from exopipe.phase3.core.record import Run
    from exopipe.phase3.lib.serve import deep, snapshot_copy

    from exopipe.phase3.lib.serve import flow

    def no_start(*a, **k):
        raise AssertionError('a page was registered (or a server started) after a refused stage')
    orig = (servers.replace, flow.register_pages, register.write)
    servers.replace = flow.register_pages = register.write = no_start
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
        ok('deep: G2 refuses the views stage and nothing after it runs (no test, nothing registered) - exit 2')
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
        servers.replace, flow.register_pages, register.write = orig


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
    """forced switch: the App probe runs before anything is registered - a failed probe refuses that page and, no other page
    passing, nothing is registered (the page keeps the run its entry serves; snapshot_forced_gen5_serve_20261026.sh: 'A page
    that fails a check before the registry call keeps its old run')."""
    from exopipe.phase3.core.record import Run
    from exopipe.phase3.lib.serve import exclusion_gate as eg, flow, forced
    a = _ns(tier='snapshot', family='forced', forced_tag='TEST', pages='forced-V-10s')
    out = f'{T}/forced_out'
    plan = planmod.make(a, out)
    pg = plan.pages[0]
    written = []

    class Res:
        ok, rc, tail = True, 0, 'AD Leo: absent'

    def probe_fails(*x, **k):
        raise Refusal('App probe of page forced-V-10s failed (rc 1): test')
    saved = (forced.ready, forced.missing_files, forced._parity_copy, scripts.run, eg.scan_snapshot_run, servers.app_probe,
             register.write, register.check_server)
    forced.ready = lambda name: (True, 'FT-READY')
    forced.missing_files = lambda pg_, D: []
    forced._parity_copy = lambda pg_: None
    scripts.run = lambda *x, **k: Res()
    eg.scan_snapshot_run = lambda D, ex=None: eg.Scan()
    servers.app_probe = probe_fails
    register.write = lambda prep: written.append(prep) or {}
    register.check_server = lambda *x, **k: {}
    try:
        run = Run('serve', ['test'], out, CFG, config_sections=None)
        forced.switch(run, plan, flow.new_state(), 'UTC')
    finally:
        (forced.ready, forced.missing_files, forced._parity_copy, scripts.run, eg.scan_snapshot_run, servers.app_probe,
         register.write, register.check_server) = saved
    u = {x.name: x for x in run.units}
    pu, ru = u[pg.unit], u['snapshot_forced_TEST_register']
    assert pu.status == 'refused' and 'App probe' in pu.reason and not written, (pu.status, pu.reason, written)
    assert ru.status == 'no-input' and any('REFUSED' in n and 'App probe' in n for n in ru.notes), (ru.status, ru.notes)
    ok('forced switch: a failed App probe refuses the page before anything is registered (no registration: the page keeps '
       'the run its entry serves)')


# ---- 6. REGISTRATION on the two-port server (lib/serve/register.py; production since 2026-10-22) -------------------------------
# A stand-in page module shaped like classify_transients.py for the two-port server (module globals, an App, a Handler with
# its app, main() with argparse and ThreadingHTTPServer): its index lists the distinct J-names of its queue, shows its title,
# its build's stamp and - as the page code does (classify_transients.py:1242) - the publish marker's text when its run holds it.
STANDIN = r'''
import argparse, html, os, time, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
INSPECT = False
STAMP = '%.6f' % time.time()

class App:
    TITLE = ''
    def __init__(self, run_dir, ids):
        self.run_dir, self.ids = run_dir, ids
        try:
            self.publish_note = open(os.path.join(run_dir, 'PUBLISH_IN_PROGRESS.txt')).read().strip()
        except OSError:
            self.publish_note = ''
    def index(self):
        banner = '<div class="banner">%s</div>' % html.escape(self.publish_note) if self.publish_note else ''
        links = ''.join('<a href="/source/%s">%s</a>\n' % (urllib.parse.quote(i), html.escape(i)) for i in self.ids)
        return ('<html><head><title>%s</title></head><body>%s<h1>%s</h1> stamp=%s\n%s</body></html>'
                % (html.escape(self.TITLE), banner, html.escape(self.TITLE), STAMP, links))

class Handler(BaseHTTPRequestHandler):
    app = None
    def do_GET(self):
        body = self.app.index().encode()
        self.send_response(200); self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
    def log_message(self, *a):
        pass

def main():
    global INSPECT
    ap = argparse.ArgumentParser()
    for a in ('--run', '--queue', '--labels-dir', '--catalog-dir', '--title', '--loc-dir', '--default-fp-reason', '--lists-dir',
              '--mchain-queue', '--mchain-catalog-dir'):
        ap.add_argument(a)
    ap.add_argument('--port', type=int); ap.add_argument('--sibling', action='append'); ap.add_argument('--inspect', action='store_true')
    ap.add_argument('--loc-offsets', action='store_true'); ap.add_argument('--slow', type=float, default=0)
    ap.add_argument('--fail', action='store_true'); ap.add_argument('--die', action='store_true')
    a = ap.parse_args()
    if a.fail:
        raise SystemExit('REFUSING: this stand-in page refuses to start')
    if a.die:
        os._exit(3)                       # a build that kills the process hosting it (the two-port backend)
    time.sleep(a.slow)
    INSPECT = a.inspect
    App.TITLE = a.title or ''
    ids = sorted({l.split()[0].split('_')[0] for l in open(a.queue) if l.startswith('J')})
    Handler.app = App(os.path.realpath(a.run), ids)
    ThreadingHTTPServer(('127.0.0.1', a.port), Handler).serve_forever()
'''


def standin_dir():
    d = f'{T}/standin'
    if not os.path.isdir(d):
        os.makedirs(d)
        with open(f'{d}/classify_transients.py', 'w') as f:
            f.write(STANDIN)
    return d


def queue_identities(q):
    """What the stand-in page's index lists for a queue: its distinct J-names."""
    with open(q) as f:
        return len({l.split()[0].split('_')[0] for l in f if l.startswith('J')})


LIVE_SERVERS = []          # every TwoPortServer running now: main() stops what a failed test left running


class TwoPortServer:
    """An isolated two-port server - this tree's lib/serve/two_port.py as run_two_port.sh runs production's frozen copy (its
    environment; ``tree``: another tree's, e.g. production's frozen two_port_d07fa79, run read only) - on free test ports
    (>= 19000; never 8765 / 8795) serving the scratch registry <d>/registry.json, its sockets in a short directory under the
    scratch root; stopped by exact pid with every backend of its socket directory."""

    def __init__(self, d, review, inspection, publish_hold=False, tree=None):
        assert min(review, inspection) >= 19000, (review, inspection)
        self.d, self.ports = d, (review, inspection)
        self.reg = os.path.join(d, 'registry.json')
        os.makedirs(d, exist_ok=True)
        if not os.path.exists(self.reg):
            with open(self.reg, 'w') as f:
                json.dump({'pages': []}, f)
        for p in self.ports:
            assert not procs.port_in_use(p), f'port {p} is in use: the test needs it free'
        self.socks = f'{T}/s{review}'
        self.log = os.path.join(d, 'two_port.log')
        argv = register.server_argv(self.reg, review, inspection, self.socks, python=PY, publish_hold=publish_hold)
        env = register.server_env(dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        self.proc = procs.start(argv, tree or ROOT, env, self.log)
        LIVE_SERVERS.append(self)
        t0 = time.time()
        while register.server_status(review, timeout=5) is None:
            assert self.proc.poll() is None, open(self.log).read()[-800:]
            assert time.time() - t0 < 60, 'the test two-port server did not answer within 60 s'
            time.sleep(0.2)

    def get(self, pid, fresh=False, path='/'):
        """(status, headers {lower name: value}, body text) of GET /p/<pid><path> on the review listener."""
        import urllib.error
        import urllib.request
        req = urllib.request.Request(f'http://127.0.0.1:{self.ports[0]}/p/{pid}{path}',
                                     headers={two_port.FRESH_HEADER: '1'} if fresh else {})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(req, timeout=120) as r:
                return r.status, {k.lower(): v for k, v in r.headers.items()}, r.read().decode('utf-8', 'replace')
        except urllib.error.HTTPError as e:
            return e.code, {k.lower(): v for k, v in e.headers.items()}, e.read().decode('utf-8', 'replace')

    def backend(self):
        st = register.server_status(self.ports[0])
        return (st.get('backends') or [{}])[0] if st else {}

    def stop(self):
        if self in LIVE_SERVERS:
            LIVE_SERVERS.remove(self)
        info = procs.proc_info(self.proc.pid)
        if info and info['state'] != 'Z':
            procs.stop_exact(info, timeout=20, poll=0.2)
        self.proc.wait()
        for pid in os.listdir('/proc'):
            if pid.isdigit():
                i = procs.proc_info(pid)
                if i and i['state'] != 'Z' and any(self.socks in a for a in i['argv']):
                    procs.stop_exact(i, timeout=20, poll=0.2)


def _stamp(body):
    import re
    m = re.search(r'stamp=([0-9.]+)', body)
    return m and m.group(1)


def test_register_matches_production():
    """register.register = production's register_page.py 'register --entry' (/lustre/gh/main/phase3/serve/register_page.py, read
    at run time and run on a sandbox - TWO_PORT_DIR, no bytecode written - with greenlit_overlay.py copied beside it), byte for
    byte: two registrations in a row (a page replacing its seeded entry, a new page, a page of Gregg's greenlit list with its
    own inspection title, one without) leave the same registry.json AND registered_pages.json, both mode 0664, each write with
    its modification time strictly later in whole seconds (two writes within one second); an entry the server's loader refuses
    (argv with --port) changes neither file in either; the production registry is refused (G23) before anything is read."""
    prod = '/lustre/gh/main/phase3/serve'
    A, B, site = f'{T}/regpar_prod', f'{T}/regpar_port', f'{T}/regpar_site'
    os.makedirs(site)

    def rev(i, title, run='/data99/r/Run_A', extra=()):
        return dict(id=f'p{i}', kind='review', code='/x/ops/classify_transients.py',
                    argv=['--run', run, '--queue', f'{run}/q.txt', '--title', title] + list(extra), env={'P3_X': str(i)},
                    legacy_port=i, greenlit=False, inspection_legacy_port=None, inspection_title=None, title=title)
    base = {'pages': [dict(id='hub', kind='static', root=site, greenlit=False), rev(9001, 'seeded A'), rev(9002, 'seeded B')]}
    green = {'pages': {'p9001': {'since': 't', 'by': 'test'},
                       'p9003': {'since': 't', 'by': 'test', 'inspection_title': 'C - INSPECTION (read-only)'}}}
    for D in (A, B):
        os.makedirs(D)
        for name, obj in (('registry.json', base), ('registered_pages.json', {'pages': [rev(9002, 'registered B')]}),
                          ('greenlit_pages.json', green)):
            with open(f'{D}/{name}', 'w') as f:
                json.dump(obj, f, indent=1)
    shutil.copy2(f'{prod}/greenlit_overlay.py', f'{A}/greenlit_overlay.py')     # register_page.py imports it from TWO_PORT_DIR
    env = dict(os.environ, TWO_PORT_DIR=A, PYTHONDONTWRITEBYTECODE='1')
    rounds = [[rev(9001, 'new A', extra=('--loc-offsets',)), rev(9003, 'new C')], [rev(9004, 'new D', run='/data99/r/Run_D')]]
    for k, entries in enumerate(rounds):
        ep = f'{T}/regpar_entries_{k}.json'
        with open(ep, 'w') as f:
            json.dump(entries, f)
        before = {D: os.stat(f'{D}/registry.json').st_mtime for D in (A, B)}
        p = subprocess.run([PY, f'{prod}/register_page.py', 'register', '--entry', ep], env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
        assert p.returncode == 0, p.stdout.decode()[-600:]
        res = register.register(entries, B)
        for name in ('registry.json', 'registered_pages.json'):
            a_, b_ = open(f'{A}/{name}', 'rb').read(), open(f'{B}/{name}', 'rb').read()
            assert a_ == b_, (k, name, a_[:300], b_[:300])
            assert all(os.stat(f'{D}/{name}').st_mode & 0o777 == 0o664 for D in (A, B)), name
        for D in (A, B):
            assert int(os.stat(f'{D}/registry.json').st_mtime) > int(before[D]), (k, D)
        assert [e['id'] for e in res['entries']] == [e['id'] for e in entries]
    reg = {e['id']: e for e in json.load(open(f'{B}/registry.json'))['pages']}
    assert reg['p9001']['greenlit'] is True and 'inspection_title' not in reg['p9001'], reg['p9001']
    assert reg['p9003']['inspection_title'] == 'C - INSPECTION (read-only)' and reg['p9002']['title'] == 'registered B'
    bad = [rev(9005, 'bad', extra=('--port', '5'))]
    with open(f'{T}/regpar_bad.json', 'w') as f:
        json.dump(bad, f)
    snap = {(D, n): (open(f'{D}/{n}', 'rb').read(), os.stat(f'{D}/{n}').st_ino) for D in (A, B)
            for n in ('registry.json', 'registered_pages.json')}
    p = subprocess.run([PY, f'{prod}/register_page.py', 'register', '--entry', f'{T}/regpar_bad.json'], env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert p.returncode != 0 and b'argv carries --port' in p.stdout, p.stdout.decode()[-300:]
    try:
        register.register(bad, B)
        raise AssertionError('a registration the server\'s loader refuses was written')
    except Refusal as e:
        assert 'nothing written' in str(e) and 'argv carries --port' in str(e), e
    assert all((open(f'{D}/{n}', 'rb').read(), os.stat(f'{D}/{n}').st_ino) == v for (D, n), v in snap.items())
    live = os.stat(TWO_PORT_REGISTRY)
    try:
        register.prepare([rev(9006, 'x')], '/lustre/gh/main/phase3/serve')
        raise AssertionError('the production registry was not refused')
    except Refusal as e:
        assert 'G23' in str(e), e
    s2 = os.stat(TWO_PORT_REGISTRY)
    assert (s2.st_ino, s2.st_size, s2.st_mtime_ns) == (live.st_ino, live.st_size, live.st_mtime_ns)
    try:
        register.guard('/data07/gh/phase3/transients/Run_P3S5', 'the publish hold of')
        raise AssertionError('a production run\'s hold was not refused')
    except Refusal as e:
        assert 'G23' in str(e)
    ok('registration = production\'s register_page.py register --entry, byte for byte (registry.json AND '
       'registered_pages.json, two registrations, the greenlit layer, mode 0664, the time strictly later on each write); a '
       'refused entry writes neither file; the production registry and a production run\'s hold are refused (G23)')


def _rules_entry(d, pid_, title='Page', run=None, extra=()):
    run = run or f'{d}/runs/{pid_}'
    os.makedirs(f'{run}/judgements', exist_ok=True)
    q = f'{run}/judgements/queue.txt'
    if not os.path.exists(q):
        with open(q, 'w') as f:
            f.write('# queue_name: test\nJ0001+0001_04h_2025-01-01_Red\nJ0002+0002_05h_2025-01-02_Red\n')
    return dict(id=pid_, kind='review', code=f'{standin_dir()}/classify_transients.py',
                argv=['--run', run, '--queue', q, '--title', title] + list(extra), env={'P3_TEST': pid_},
                legacy_port=None, greenlit=False, inspection_legacy_port=None, inspection_title=None, title=title)


def test_two_port_rules(srv):
    """The server (an isolated instance of this tree's two_port.py, default options): (1) a REGISTRY CHANGE is a change of the
    file's inode, size OR modification time (two_port.registry_state) - a new file of the same size and the same time (the
    mtime alone, the rule before, saw nothing: B6A p9024), a rewrite in place of another size and the same time, the time
    alone - and nothing else reloads; (2) NO PUBLISH HOLD by default (production, exopipe_ops 4a39c6b): a page of a run that
    holds PUBLISH_IN_PROGRESS.txt is rebuilt at a registry change like any other - the build before answers meanwhile (stale,
    queued, never held) - and the rebuilt page shows the marker's text as a banner."""
    d = srv.d
    st0 = register.server_status(srv.ports[0])
    assert st0['build_state'] == two_port.BUILD_STATE and st0['publish_hold'] is False, st0
    e = _rules_entry(d, 'pcd', 'Title A')
    register.register([e], d)
    st_, h, b = srv.get('pcd')
    assert st_ == 200 and 'Title A' in b, (st_, b[:200])
    s0 = os.stat(srv.reg)
    txt = open(srv.reg).read().replace('Title A', 'Title B')
    assert len(txt) == s0.st_size
    with open(srv.reg + '.new', 'w') as f:
        f.write(txt)
    os.replace(srv.reg + '.new', srv.reg)
    os.utime(srv.reg, ns=(s0.st_atime_ns, s0.st_mtime_ns))
    s1 = os.stat(srv.reg)
    assert s1.st_ino != s0.st_ino and (s1.st_size, s1.st_mtime_ns) == (s0.st_size, s0.st_mtime_ns)
    st_, h, b = srv.get('pcd')
    assert 'Title B' in b, b[:200]
    with open(srv.reg, 'w') as f:                    # the same file (inode), another size, the same time
        f.write(txt.replace('Title B', 'Title CC'))
    os.utime(srv.reg, ns=(s0.st_atime_ns, s0.st_mtime_ns))
    s2 = os.stat(srv.reg)
    assert s2.st_ino == s1.st_ino and s2.st_size != s1.st_size and s2.st_mtime_ns == s1.st_mtime_ns
    st_, h, b = srv.get('pcd')
    assert 'Title CC' in b, b[:200]
    g0 = srv.backend().get('gen')
    os.utime(srv.reg, ns=(s2.st_atime_ns, s2.st_mtime_ns + 10 ** 9))      # the time alone
    srv.get('pcd')
    g1 = srv.backend().get('gen')
    srv.get('pcd')
    srv.get('pcd')
    g2 = srv.backend().get('gen')
    assert g1 == g0 + 1 and g2 == g1, (g0, g1, g2)
    ok('two-port server: a registry change = a new inode, another size or another time (a new file of the same size and time '
       'reloads - the time alone missed it), and nothing else reloads')
    RH = f'{d}/runs/pmark'
    eh = _rules_entry(d, 'pmark', 'Marked page', run=RH)
    register.register([eh], d)
    st_, h, b = srv.get('pmark', fresh=True)
    s_1 = _stamp(b)
    assert st_ == 200 and s_1 and 'class="banner"' not in b, b[:200]
    register.set_hold(RH, 'publish in progress (test)')
    os.utime(srv.reg, ns=(time.time_ns(), os.stat(srv.reg).st_mtime_ns + 10 ** 9))      # any registry change
    st_, h, b = srv.get('pmark')
    be = srv.backend()
    assert _stamp(b) == s_1 and two_port.STALE_HEADER.lower() in h and two_port.HELD_HEADER.lower() not in h, (h, b[:200])
    assert not be['held'] and not be['built']['pmark']['held'] and 'pmark' in be['rebuild_queue'], be
    st_, h, b = srv.get('pmark', fresh=True)
    assert _stamp(b) not in (None, s_1) and 'publish in progress (test)' in b, b[:300]
    register.lift_hold(RH)
    ok('two-port server, default (production): no publish hold - a page of a marked run is rebuilt at a registry change (the '
       'build before answers meanwhile, stale, queued, never held) and the rebuilt page shows the marker\'s text')


def test_publish_hold_option(srvh):
    """THE PUBLISH HOLD, an amendment option (two_port serve --publish-hold; serve.TWO_PORT_PUBLISH_HOLD, default False =
    production): with it, while <run>/PUBLISH_IN_PROGRESS.txt exists the run's built page is not rebuilt at a registry change -
    it answers from its build, a fresh GET too, with X-Two-Port-Held and X-Two-Port-Stale, and /_status reports it held and
    not queued; the build wait refuses it (G16); lifted, its next request queues its rebuild and a fresh GET gets the rebuilt
    page; a page whose own entry changed while held is built at once (with the banner). serve refuses a server whose hold mode
    differs from the configuration (G13)."""
    d = srvh.d
    assert register.server_status(srvh.ports[0])['publish_hold'] is True
    try:
        register.check_server(srvh.ports, d)
        raise AssertionError('a server with the hold passed under serve.TWO_PORT_PUBLISH_HOLD False')
    except Refusal as e:
        assert 'G13' in str(e) and 'publish hold on' in str(e), e
    CFG.set('serve.TWO_PORT_PUBLISH_HOLD', True)
    try:
        register.check_server(srvh.ports, d)
    finally:
        CFG.set('serve.TWO_PORT_PUBLISH_HOLD', False)
    RH = f'{d}/runs/phold'
    eh = _rules_entry(d, 'phold', 'Held page', run=RH)
    res = register.register([eh], d)
    st_, h, b = srvh.get('phold', fresh=True)
    s_1 = _stamp(b)
    assert st_ == 200 and s_1 and 'class="banner"' not in b, b[:200]
    hp, now = register.set_hold(RH, 'publish in progress (test)')
    assert now and open(hp).read() == 'publish in progress (test)\n'
    os.utime(srvh.reg, ns=(time.time_ns(), os.stat(srvh.reg).st_mtime_ns + 10 ** 9))      # any registry change
    st_, h, b = srvh.get('phold')
    assert _stamp(b) == s_1 and h.get(two_port.HELD_HEADER.lower()) == hp and two_port.STALE_HEADER.lower() in h, (h, b[:200])
    t = time.time()
    st_, h, b = srvh.get('phold', fresh=True)
    assert _stamp(b) == s_1 and time.time() - t < 5 and 'class="banner"' not in b, b[:200]
    be = srvh.backend()
    assert be['built']['phold']['held'] and 'phold' in be['held'] and 'phold' not in be['rebuild_queue'], be
    try:
        register.wait_built(srvh.ports[0], res['entries'][0], time.time(), d=d, poll=0.2, stall_s=10)
        raise AssertionError('a held page passed the build wait')
    except Refusal as x:
        assert 'G16' in str(x) and 'publish hold' in str(x), x
    lifted = register.lift_hold(RH)
    assert lifted and os.path.exists(lifted) and not os.path.exists(hp) and '.removed_' in lifted
    st_, h, b = srvh.get('phold')
    be = srvh.backend()
    assert _stamp(b) == s_1 and ('phold' in be['rebuild_queue'] or 'phold' in be['building']) and not be['held'], be
    st_, h, b = srvh.get('phold', fresh=True)
    assert _stamp(b) not in (None, s_1) and 'class="banner"' not in b and two_port.HELD_HEADER.lower() not in h, b[:200]
    register.set_hold(RH, 'publish in progress (test 2)')
    register.register([dict(eh, argv=eh['argv'][:-1] + ['Held page 2'], title='Held page 2')], d)
    st_, h, b = srvh.get('phold')
    assert 'Held page 2' in b and 'publish in progress (test 2)' in b and two_port.HELD_HEADER.lower() not in h, b[:300]
    register.lift_hold(RH)
    ok('publish hold OPTION (--publish-hold): a held page keeps its build at a registry change (a fresh GET too; held and stale '
       'headers; /_status held, not queued; the build wait refuses it, G16); lifted, its rebuild is queued and a fresh GET '
       'gets the rebuilt page; a page whose entry changed while held is built at once, with the banner; serve refuses a '
       'server whose hold mode differs from serve.TWO_PORT_PUBLISH_HOLD (G13)')


def test_old_server_refused():
    """A two-port server older than this tree's two_port.py - production's frozen two_port_d07fa79, the live server's code, run
    here read only as an isolated instance on test ports - reports no build state: serve refuses AT ONCE, before anything is
    written (G16: it must be redeployed with this two_port.py first), and a build wait against it refuses at once instead of
    waiting for the stall limit (the review of 2026-11-02 reproduced 'states seen: waiting' while the page answered 200)."""
    frozen = '/lustre/gh/main/phase3/serve/two_port_d07fa79'
    d = f'{T}/old_server'
    old = TwoPortServer(d, 29945, 29946, tree=frozen)
    try:
        st = register.server_status(29945)
        assert st is not None and 'build_state' not in st and register.old_server(st), st
        reg0 = open(old.reg, 'rb').read()
        t = time.time()
        try:
            register.check_server(old.ports, d)
            raise AssertionError('an old two-port server passed the server check')
        except Refusal as e:
            assert 'G16' in str(e) and 'redeployed with this two_port.py first' in str(e) and time.time() - t < 30, e
        try:
            register.wait_built(29945, _rules_entry(d, 'pold'), time.time(), d=None, poll=0.2, stall_s=600)
            raise AssertionError('a build wait against an old server passed')
        except Refusal as e:
            assert 'G16' in str(e) and 'redeployed' in str(e) and time.time() - t < 30, e
        assert open(old.reg, 'rb').read() == reg0
    finally:
        old.stop()
    ok(f'an older two-port server (production\'s frozen two_port_d07fa79: /_status without the build state) is refused at once '
       f'(G16, redeploy first; {time.time() - t:.1f} s), nothing registered; a build wait against it refuses at once too')


def test_build_state_wait(srv):
    """register.wait_built waits on the page's BUILD STATE (/_status), not a number of tries: a page queued behind another
    page's build is waited for longer than the stall limit because builds progress; a failed build is refused at once with
    the page's own error (G24); an entry replaced by another registration meanwhile is refused (G13); a page the server does
    not know is refused (G16, 404); no build starting or finishing on the server for the stall limit is refused (G16)."""
    import threading
    d, port = srv.d, srv.ports[0]
    ea, eb = _rules_entry(d, 'pslowa', extra=('--slow', '3')), _rules_entry(d, 'pslowb', extra=('--slow', '3'))
    res = register.register([ea, eb], d)
    th = threading.Thread(target=lambda: srv.get('pslowa', fresh=True))
    th.start()
    time.sleep(0.4)
    t = time.time()
    b = register.wait_built(port, res['entries'][1], res['t'], d=d, poll=0.2, stall_s=4)
    dt = time.time() - t
    th.join()
    assert b['state'] == 'built' and dt > 4 and 'building' in b['states'] and b['entry'] == two_port.entry_md5(res['entries'][1]) \
        and b['build_start'] >= res['t'], (round(dt, 1), b)
    ok(f'build wait: a page queued behind another page\'s build is waited for {dt:.1f} s (stall limit 4 s: builds progressed; '
       f'states {" -> ".join(b["states"])})')
    ef = _rules_entry(d, 'pfail', extra=('--fail',))
    res = register.register([ef], d)
    t = time.time()
    try:
        register.wait_built(port, res['entries'][0], res['t'], d=d, poll=0.2, stall_s=60)
        raise AssertionError('a failed build passed')
    except Refusal as x:
        assert 'G24' in str(x) and 'refuses to start' in str(x) and time.time() - t < 15, (x, time.time() - t)
    ok(f'build wait: a failed build is refused at once (G24, {time.time() - t:.1f} s) with the page\'s own error')
    er = _rules_entry(d, 'prepl', extra=('--slow', '2'))
    res = register.register([er], d)
    got = {}

    def waiter():
        try:
            register.wait_built(port, res['entries'][0], res['t'], d=d, poll=0.2, stall_s=30)
            got['r'] = 'built'
        except Refusal as x:
            got['r'] = str(x)
    th = threading.Thread(target=waiter)
    th.start()
    time.sleep(0.6)
    register.register([dict(er, argv=er['argv'][:4] + ['--title', 'replaced'], title='replaced')], d)
    th.join()
    assert 'G13' in got['r'] and 'replaced by another registration' in got['r'], got
    try:
        register.wait_built(port, _rules_entry(d, 'pnone'), time.time(), d=None, poll=0.2, stall_s=10)
        raise AssertionError('an unknown page passed')
    except Refusal as x:
        assert 'G16' in str(x) and '404' in str(x), x
    ok('build wait: an entry replaced meanwhile is refused (G13); a page the server does not know is refused (G16, 404)')
    time.sleep(2.5)                                  # the replaced page's build ends: nothing builds on the server now
    # the registration's time is taken BEFORE the registry write: a build that starts as soon as the server sees the new
    # registry (here a request in that very window) is the registered page's build
    ee = _rules_entry(d, 'pearly')
    orig_w = register.atomic_write

    def write_then_build(path, obj):
        orig_w(path, obj)
        if os.path.basename(path) == 'registry.json':
            got['early'] = srv.get('pearly', fresh=True)
    register.atomic_write = write_then_build
    try:
        res = register.register([ee], d)
    finally:
        register.atomic_write = orig_w
    t_after = time.time()
    b = register.wait_built(port, res['entries'][0], res['t'], d=d, poll=0.2, stall_s=3)
    assert got['early'][0] == 200 and b['state'] == 'built' and res['t'] <= b['build_start'] < t_after \
        and b['waited_s'] < 3, (got['early'][0], b, res['t'], t_after)
    ok('build wait: the registration time is taken before the registry write - a build started in the window between the write '
       'and the end of the registration is the page\'s build (accepted at once, not waited for until the stall limit)')
    # a page whose build kills its backend (each request restarts the backend, so builds "progress"): refused after
    # TWO_PORT_MAX_FAILED_REQUESTS requests in a row without a 200
    ed = _rules_entry(d, 'pdie', extra=('--die',))
    res = register.register([ed], d)
    t = time.time()
    try:
        register.wait_built(port, res['entries'][0], res['t'], d=d, poll=0.2, stall_s=60)
        raise AssertionError('a page whose build kills its backend passed')
    except Refusal as x:
        assert 'G24' in str(x) and f'{CFG.serve.TWO_PORT_MAX_FAILED_REQUESTS} requests in a row got no 200' in str(x), x
    ok(f'build wait: a page whose build kills its backend is refused after {CFG.serve.TWO_PORT_MAX_FAILED_REQUESTS} requests '
       f'in a row without a 200 (G24, {time.time() - t:.1f} s; the stall limit was 60 s)')
    es = _rules_entry(d, 'pstall', extra=('--slow', '8'))
    res = register.register([es], d)
    t = time.time()
    try:
        register.wait_built(port, res['entries'][0], res['t'], d=d, poll=0.2, stall_s=2)
        raise AssertionError('a build without progress passed')
    except Refusal as x:
        assert 'G16' in str(x) and 'no build started or finished' in str(x), x
    ok(f'build wait: no build starting or finishing on the server for the stall limit (2 s) is refused (G16, after '
       f'{time.time() - t:.1f} s)')
    time.sleep(7)                                    # let the stall page's build end before the server stops


def test_start_registers():
    """THE START STAGE REGISTERS (production since 2026-10-22, deep_reregister_after_publish_20261031.py, sublst_register_pages.py,
    snapshot_register_page.py): through record.Run, the deep pages of a parity plan (offset 21000; the stand-in page code)
    - with no two-port server answering: refused (G16) before anything is written, the publish marker standing; the command it
      prints is run_two_port.sh's (its environment, nice, a short socket directory); a server whose socket directory is too
      long for a Unix socket refuses to start (instead of answering 502 for every page);
    - with one: ONE registration (registry.json and registered_pages.json written once each for both pages), the publish
      marker removed right before it, no server on any page port, entries as production's (p<port>, argv without --port, the
      P3_* environment, legacy and inspection ports, greenlit by the layer only), each page's build waited on and its
      served_page record passing its content tests (the page at /p/<id>/, the registry entry and the server's build of it, the
      sources); no inspection page built before: the warm-up has nothing to do;
    - a second serve with a page refused at start: the other page registered, the marker removed all the same (production:
      unconditionally, right before the registration), the refused page rebuilt by the server like any other (no hold), the
      registered page's previously built INSPECTION page requested once (the warm-up) and built from the new entry.
    A review page is never given a server of its own (servers.replace refuses it)."""
    from exopipe.phase3.core.record import Run
    from exopipe.phase3.lib.serve import flow
    out = f'{T}/reg_out'
    plan = planmod.make(_ns(tier='deep', run_label='TEST', port_offset=21000), out)
    assert plan.parity and not plan.gates, plan.gates
    plan.ops = standin_dir()                         # the registered page code: the stand-in (classify_transients.py-shaped)
    U = plan.U
    os.makedirs(f'{U}/judgements')
    for S_, ids in (('V', ('J0001+0001_04h_2030-01-01_Red', 'J0002+0002_05h_2030-01-02_Red', 'J0001+0001_06h_2030-01-03_Red')),
                    ('I', ('J0007+0007_08h_2030-01-05_Red', 'J0008+0008_09h_2030-01-06_23MHz'))):   # no night excluded (G17)
        with open(f'{U}/judgements/queue_p3s3_{S_}.txt', 'w') as f:
            f.write('# queue_name: p3s3_%s_deep\n' % S_ + '\n'.join(ids) + '\n')
    V = next(p_ for p_ in plan.pages if p_.key == 'V')
    I_ = next(p_ for p_ in plan.pages if p_.key == 'I')

    def state_for(refused):
        st = flow.new_state()
        st['refused'].update({'V-multi': 'NO-INPUT: test', 'I-multi': 'NO-INPUT: test'})
        st['refused'].update(refused)
        for pg_ in (V, I_):
            if pg_.key not in st['refused']:
                st['expected'][pg_.key] = {'n': queue_identities(pg_.at('queue', U)), 'kind': 'stand-in index'}
        return st
    try:
        servers.replace(plan, V, 'UTC')
        raise AssertionError('a review page got a server of its own')
    except Refusal as e:
        assert 'registered on the two-port server' in str(e), e
    hp, _ = register.set_hold(U, 'publish (test)')
    d, ports = register.registry_dir(plan), register.listeners(plan)
    assert ports == (29765, 29795) and d == f'{out}/two_port'
    os.makedirs(d)
    with open(f'{d}/registry.json', 'w') as f:
        json.dump({'pages': []}, f)
    with open(f'{d}/greenlit_pages.json', 'w') as f:                 # Gregg's list: p29766 on the inspection listener
        json.dump({'pages': {'p29766': {'since': 'test', 'by': 'test'}}}, f)
    reg0 = open(f'{d}/registry.json', 'rb').read()
    run0 = Run('serve', ['test'], out, CFG, config_sections=None)
    flow.start_stage(run0, plan, state_for({}))
    u = {x.name: x for x in run0.units}
    ru = u['deep_Run_TEST_register']
    assert ru.status == 'refused' and 'G16 two-port server: nothing answers' in ru.reason, (ru.status, ru.reason)
    sk = register.sock_dir(29765)
    assert all(x in ru.reason for x in ('NUMPY_MADVISE_HUGEPAGE=0', 'OMP_NUM_THREADS=1', 'nice -n10', f'--sock-dir {sk}',
                                        f'--review-port 29765 --inspection-port 29795')), ru.reason
    assert len(os.path.join(sk, 'tp_0123456789abcdef.sock')) <= two_port.SOCK_PATH_MAX and len(sk) < len(d)
    assert u[V.unit].status == 'refused' and 'not registered' in u[V.unit].reason, (u[V.unit].status, u[V.unit].reason)
    assert open(f'{d}/registry.json', 'rb').read() == reg0 and os.path.exists(hp) and run0.exit_code() == 2
    longsk = f'{T}/' + 'x' * 90
    p_ = subprocess.run(register.server_argv(f'{d}/registry.json', 29765, 29795, longsk, python=PY), cwd=ROOT,
                        env=register.server_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    assert p_.returncode != 0 and b'too long' in p_.stdout and not procs.port_in_use(29765), p_.stdout[-300:]
    ok('start stage: no two-port server answering - refused (G16) before anything is written, the publish marker standing; '
       'the command it prints is run_two_port.sh\'s (environment, nice, a short socket directory); a server whose socket '
       'directory is too long refuses to start')
    srv = TwoPortServer(d, *ports)
    writes = {}
    orig_w = register.atomic_write

    def counting(path, obj):
        writes[os.path.basename(path)] = writes.get(os.path.basename(path), 0) + 1
        return orig_w(path, obj)
    register.atomic_write = counting
    try:
        run1 = Run('serve', ['test'], out, CFG, config_sections=None)
        state1 = state_for({})
        flow.start_stage(run1, plan, state1, utc='2026-11-02 06:00 UT')
        u = {x.name: x for x in run1.units}
        assert u['deep_Run_TEST_register'].status == 'done', (u['deep_Run_TEST_register'].status, u['deep_Run_TEST_register'].reason)
        assert all(u[pg_.unit].status == 'done' for pg_ in (V, I_)), {n: (x.status, x.reason) for n, x in u.items()}
        assert {u[p_.unit].status for p_ in plan.pages if p_.multi} == {'no-input'}
        assert u['deep_Run_TEST_register_warm'].status == 'no-input', u['deep_Run_TEST_register_warm'].status
        assert writes == {'registered_pages.json': 1, 'registry.json': 1}, writes
        assert not os.path.exists(hp) and glob.glob(hp + '.removed_*'), os.listdir(U)
        assert not any(procs.port_in_use(p_.port) for p_ in plan.pages) and run1.exit_code() == 0
        reg = {e['id']: e for e in json.load(open(f'{d}/registry.json'))['pages']}
        e = reg['p29766']
        assert set(reg) == {'p29766', 'p29768'} and e['code'] == f'{plan.ops}/classify_transients.py', reg.keys()
        want = plan.argv(V, '2026-11-02 06:00 UT')[2:]
        del want[want.index('--port'):want.index('--port') + 2]
        assert '--port' not in e['argv'] and e['argv'] == want, (e['argv'], want)
        assert e['env'] == {k: v for k, v in sorted(plan.server_env_extra().items()) if k.startswith('P3_')}
        assert (e['legacy_port'], e['inspection_legacy_port'], e['greenlit'], 'inspection_title' in e) == (29766, 29796, True, False)
        assert reg['p29768']['greenlit'] is False and reg['p29768']['inspection_title'] is None
        assert e['title'] == e['argv'][e['argv'].index('--title') + 1] and 'served 2026-11-02 06:00 UT' in e['title']
        rec = json.load(open(V.record_path()))
        assert rec['registered'] and rec['url'] == 'http://127.0.0.1:29765/p/p29766/' and rec['build']['state'] == 'built'
        r = {x['name']: x for x in validate.run('served_page', V.record_path())}
        assert all(x['passed'] for x in r.values()), r
        rr = {x['name']: x for x in validate.run('page_registration', rec['registration'])}
        assert all(x['passed'] for x in rr.values()), rr
        ok('start stage: ONE registration of both pages (registry.json and registered_pages.json written once each), the '
           'publish marker removed right before it, no server on any page port; entries in production\'s form (the greenlit '
           'layer from the list); each page\'s build waited on; served_page (/p/<id>/, the registry entry and its build, the '
           'sources) and page_registration tests pass; nothing to warm')
        import urllib.request
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open('http://127.0.0.1:29795/p/p29766/',
                                                                               timeout=60) as r_:
            assert r_.status == 200                          # a team member opens p29766 on the inspection listener
        st_, h, b = srv.get('p29768')
        s_old = _stamp(b)
        hp2, _ = register.set_hold(U, 'publish 2 (test)')
        writes.clear()
        run2 = Run('serve', ['test'], out, CFG, config_sections=None)
        flow.start_stage(run2, plan, state_for({'I': 'G17 serve-time exclusion gate: 1 rows (test) - NOT registered'}),
                         utc='2026-11-02 07:00 UT')
        u = {x.name: x for x in run2.units}
        ru = u['deep_Run_TEST_register']
        assert ru.status == 'done' and not os.path.exists(hp2) and len(glob.glob(hp + '.removed_*')) == 2, (ru.status, ru.notes)
        assert u[V.unit].status == 'done' and u[I_.unit].status == 'refused' and run2.exit_code() == 2
        assert writes == {'registered_pages.json': 1, 'registry.json': 1}, writes
        st_, h, b = srv.get('p29766')
        assert 'served 2026-11-02 07:00 UT' in b and 'publish 2 (test)' not in b, b[:300]
        st_, h, b = srv.get('p29768')
        be = srv.backend()
        assert _stamp(b) == s_old and two_port.HELD_HEADER.lower() not in h and not be['held'], (h, be['held'])
        assert 'p29768' in be['rebuild_queue'] or 'p29768' in be['building'] or _stamp(srv.get('p29768', fresh=True)[2]) != s_old
        wu = u['deep_Run_TEST_register_warm']
        assert wu.status == 'done', (wu.status, wu.reason)
        rows = list(csv.DictReader(open(wu.outputs[0]['path'])))
        assert [(x['page'], x['status'], x['registered_now']) for x in rows] == [('p29766', '200', '1')], rows
        x = srv.backend()['built'].get('p29766:inspection') or {}
        assert x.get('entry') == two_port.entry_md5(next(e_ for e_ in json.load(open(f'{d}/registry.json'))['pages']
                                                         if e_['id'] == 'p29766')), x
        ok('start stage, a page refused at start: the other page registered (one change), the publish marker removed all the '
           'same (as production), the refused page rebuilt like any other (no hold), and the registered page\'s previously '
           'built inspection page requested once and built from the new entry (the warm-up)')
    finally:
        register.atomic_write = orig_w
        srv.stop()


def test_forced_checks_server_first():
    """forced switch: the two-port server is checked BEFORE any page is checked or any judgement carried - with none answering
    the registration unit refuses (G16) and nothing is carried (the review of 2026-11-02)."""
    from exopipe.phase3.core.record import Run
    from exopipe.phase3.lib.serve import flow, forced
    a = _ns(tier='snapshot', family='forced', forced_tag='TEST', pages='forced-V-10s', port_offset=21200)
    out = f'{T}/forced_out_server'
    plan = planmod.make(a, out)
    assert not procs.port_in_use(register.listeners(plan)[0])
    touched = []
    saved = (forced.check_page, journals.snapshot_carry)
    forced.check_page = lambda *x, **k: touched.append('check') or (_ for _ in ()).throw(AssertionError('checked'))
    journals.snapshot_carry = lambda *x, **k: touched.append('carry') or (_ for _ in ()).throw(AssertionError('carried'))
    try:
        run = Run('serve', ['test'], out, CFG, config_sections=None)
        forced.switch(run, plan, flow.new_state(), 'UTC')
    finally:
        forced.check_page, journals.snapshot_carry = saved
    ru = {x.name: x for x in run.units}['snapshot_forced_TEST_register']
    assert ru.status == 'refused' and 'G16' in ru.reason and not touched, (ru.status, ru.reason, touched)
    ok('forced switch: the two-port server is checked before any page check or judgement carry (none answering: refused, '
       'nothing carried)')


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


def _frozen_to_live(live, pid, ops):
    """B-51 (spec serve.md section 9; not ruled): production serves a sub-LST page from the frozen code tree its 10-minute
    run was launched from (sublst_code_root.sh: SUBLST_CODE_ROOT in the server's environment, $OPSC/classify_transients.py);
    the branch serves from the live tree during the window. The live argv with the tree's prefix mapped to the live code
    root, after checking the tree is a complete frozen tree and its server script IS the live one modulo the freeze's
    code-location rewrite (a frozen server running older code than the live tree is a real difference). (argv, root|None)"""
    root = (procs.environ(pid).get('SUBLST_CODE_ROOT') or '').rstrip('/')
    if not root:
        return live, None
    base = os.path.dirname(ops)                            # the live code root (/lustre/gh/main), as sublst_code_root.sh's XC
    assert os.path.isfile(os.path.join(root, 'FROZEN.ok')), f'{root}: no FROZEN.ok - not a complete frozen tree'
    frozen = open(os.path.join(root, 'exopipe_ops', 'classify_transients.py')).read().replace(root, base)
    assert frozen == open(os.path.join(ops, 'classify_transients.py')).read(), \
        f'{root}: the frozen classify_transients.py differs from the live one beyond the code-root rewrite'
    return [base + x[len(root):] if x.startswith(root + '/') else x for x in live], root


def test_production_readonly():
    import argparse
    import re
    n_cmp, frozen = 0, set()
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
                live, root = _frozen_to_live(live, pids[0], p.ops)
                if root:
                    frozen.add(root)
            mine = p.argv(pg, m.group(1) if m else '')
            assert mine == live, (tier, pg.key, [x for x in zip(live, mine) if x[0] != x[1]][:3])
            n_cmp += 1
    ok(f'planned argv == the live production servers\' argv on {n_cmp} pages (sub-LST: the current launcher\'s absolute '
       f'script path; served time and LST list taken from the live title'
       + (f'; served from frozen tree(s) {sorted(os.path.basename(r) for r in frozen)} whose server script is the live one '
          f'(B-51: the branch serves the live tree in the window)' if frozen else '') + ')')
    # the pages as production serves them: since 2026-10-22 the review pages are entries of the two-port server's registry
    # (no per-port server, so the loop above compares none of them). The planned argv after the script, without --port (the
    # two-port server routes by path; the title's served time and LST list taken from the entry), and the environment
    # (but NUMPY_MADVISE_HUGEPAGE, which the two-port server sets for every page, run_two_port.sh:5) equal each deep and
    # sub-LST entry's - the multiples chain (4c42f91), the deep pages' environment (2026-10-30) and the deep multiples
    # pages' title (2026-10-31) included. Read only.
    entries = {e.get('legacy_port'): e for e in json.load(open(TWO_PORT_REGISTRY))['pages'] if e.get('kind') == 'review'}
    n_pages, n_chain = 0, 0
    for tier, run in (('deep', 'P3S5'), ('sublst', None)):
        a = argparse.Namespace(tier=tier, run_label=run, port_offset=0, pages=None, stokes=None, lsts=None, label=None,
                               source=None, family=None, forced_tag=None, restart=None, stop=None)
        p = planmod.make(a, CFG.paths.TRANSIENTS_ROOT)
        for pg in p.pages:
            e = entries.get(pg.port)
            if e is None:
                continue
            t = procs.argv_value(e['argv'], '--title') or ''
            m = re.search(r'served (\d{4}-\d\d-\d\d \d\d:\d\d UT)', t)
            if tier == 'sublst':
                p.lsts_text = re.search(r'LSTs ([0-9h ]+?) - ', t).group(1)
            mine = p.argv(pg, m.group(1) if m else '')
            i = mine.index('--port')
            mine = mine[2:i] + mine[i + 2:]
            assert mine == e['argv'], (tier, pg.key, [x for x in zip(mine, e['argv']) if x[0] != x[1]][:3])
            env = {k: v for k, v in p.server_env_extra().items() if k != 'NUMPY_MADVISE_HUGEPAGE'}
            assert env == e['env'], (tier, pg.key, env, e['env'])
            n_pages += 1
            n_chain += '--mchain-queue' in mine
    assert n_pages == 8 and n_chain == 4, (n_pages, n_chain)
    ok(f'planned argv and environment == the live two-port registry entries on {n_pages} deep and sub-LST pages (multiples '
       f'chain on {n_chain}: p8766/p8768 queue + view, p8787/p8788 queue; the deep pages\' P3_LOC_COMMON_LINKS / '
       f'P3_XTIER_FRAME; the deep multiples pages\' title), read only')
    # the ENTRY each page would be registered as (register.page_entry with Gregg's greenlit layer, both read only) = the live
    # entry, every key: deep and 10-minute (the 10-minute pages' code is the frozen tree their run was published from, B-51:
    # same file name, the live tree in the window) and the forced pages (their siblings from the live registry, as the gen5
    # serve's AFTER; the title in the gen5 serve's words, served time from the live title)
    green = register.greenlit(os.path.join(CFG.serve.TWO_PORT_DIR, CFG.serve.TWO_PORT_GREENLIT))
    n_entries = 0
    for tier, run, extra in (('deep', 'P3S5', {}), ('sublst', None, {}),
                             ('snapshot', None, {'family': 'forced', 'forced_tag': 'gen5'})):
        a = argparse.Namespace(tier=tier, run_label=run, port_offset=0, pages=None, stokes=None, lsts=None, label=None,
                               source=None, family=extra.get('family'), forced_tag=extra.get('forced_tag'), restart=None,
                               stop=None)
        p = planmod.make(a, CFG.paths.SNAPSHOT_REVIEW_ROOT if tier == 'snapshot' else CFG.paths.TRANSIENTS_ROOT)
        for pg in p.pages:
            e = entries.get(pg.port)
            if e is None:
                continue
            t = procs.argv_value(e['argv'], '--title') or ''
            m = re.search(r'served (\d{4}-\d\d-\d\d \d\d:\d\d UT)', t)
            if tier == 'sublst':
                p.lsts_text = re.search(r'LSTs ([0-9h ]+?) - ', t).group(1)
            mine = register.greenlit_apply({'pages': [register.page_entry(p, pg, m.group(1) if m else '')]}, green)['pages'][0]
            live = dict(e)
            if tier == 'sublst':
                assert os.path.basename(live['code']) == os.path.basename(mine['code']), (live['code'], mine['code'])
                live['code'] = mine['code']
            assert mine == live, (tier, pg.key, {k: (mine.get(k), live.get(k)) for k in set(mine) | set(live)
                                                 if mine.get(k) != live.get(k)})
            n_entries += 1
    assert n_entries == 12, n_entries
    ok(f'planned registry ENTRIES == the live two-port registry entries on {n_entries} pages (deep, 10-minute, forced: id, '
       f'kind, code, argv without --port, env, legacy and inspection ports, the greenlit layer, title - the forced pages\' '
       f'in the gen5 words), read only')
    hubreg = json.load(open(f'{CFG.serve.HUB_DIR}/registry.json'))
    a = argparse.Namespace(tier='snapshot', run_label=None, port_offset=0, pages=None, stokes=None, lsts=None, label=None,
                           source=None, family='forced', forced_tag='gen5', restart=None, stop=None)
    p = planmod.make(a, CFG.paths.SNAPSHOT_REVIEW_ROOT)
    for pg in p.pages:
        S_, tier_, kvs = p.hub_entry(pg, 'UTC')
        e = hubreg[S_][tier_]
        assert dict(kv.split('=', 1) for kv in kvs)['generation'] == e['generation'], (pg.key, e['generation'])
    ok('forced hub entries: the planned generation text == the live hub registry\'s (gen5 words), read only')
    env = dict(os.environ, PYTHONPATH=ROOT)
    p = subprocess.run([PY, '-m', 'exopipe.phase3', 'serve', '--tier', 'deep', '--run', 'P3S5', '--dry-run', '--cores', 'none'],
                       env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
    txt = p.stdout.decode()
    assert p.returncode == 0 and 'PLAN exopipe3 serve --tier deep' in txt and 'page V-multi: p8772 [registered' in txt \
        and f'pages registered in {TWO_PORT_REGISTRY} (one registration)' in txt, txt[-600:]
    ok('serve --tier deep --run P3S5 --dry-run: plan printed (the pages registered in the live registry, read only), exit 0, '
       'nothing written')
    D = CFG.paths.TRANSIENTS_ROOT
    for path in (f'{D}/Run_P3S5/judgements/queue_p3s3_V_multi.txt', f'{D}/Run_SLSTB/judgements/queue_p3s3_V.txt'):
        assert all(r['passed'] for r in validate.run('served_queue', path)), path
    assert all(r['passed'] for r in validate.run('served_view', f'{D}/Run_SLSTB/source_catalogs_V'))
    hr = validate.run('hub_page', f'{D}/review_hub/index.html')
    assert all(r['passed'] for r in hr), hr
    ok('content tests on real products (read-only): Run_P3S5 V multiples queue, Run_SLSTB V queue and view, the review hub '
       '(index.html is the page lib/serve/hub.render makes from registry.json, byte for byte)')


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
        test_hub_page()
        test_hub_stage()
        test_exclusion_gate_mchain()
        test_processes()
        test_views_stage()
        test_mchain()
        test_exclusion_gate_per_page_stage()
        test_stage_halting()
        test_inplace_gate_agrees()
        test_forced_probe_before_stop()
        test_register_matches_production()
        srv = TwoPortServer(f'{T}/tp_rules', 29865, 29895)
        try:
            test_two_port_rules(srv)
            test_build_state_wait(srv)
        finally:
            srv.stop()
        srvh = TwoPortServer(f'{T}/tp_hold', 29875, 29885, publish_hold=True)
        try:
            test_publish_hold_option(srvh)
        finally:
            srvh.stop()
        test_old_server_refused()
        test_start_registers()
        test_forced_checks_server_first()
        test_server_env_mirrors_launcher()
        test_production_readonly()
    finally:
        for srv_ in list(LIVE_SERVERS):              # never leave a test two-port server behind
            try:
                srv_.stop()
            except Exception:
                pass
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
