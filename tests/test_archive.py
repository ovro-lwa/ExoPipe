#!/usr/bin/env python3
"""Tests of `exopipe3 archive` (unit archive; plain asserts, no pytest needed):

    NUMPY_MADVISE_HUGEPAGE=0 EXOPIPE3_SITE=/lustre/gh/main/exopipe_ops/exopipe3_site.py \
        /opt/devel/pipeline/envs/py38_orca_nkosogor/bin/python3 tests/test_archive.py [--no-real]

Synthetic part: a MIRROR archive tree, attempts, staging trees, snapshot work dirs and fit stores under a fresh
/tmp/exopipe3_tests_archive/<pid>/ (removed at the end); every action runs through the real CLI (cli.main) with
--archive-root on the mirror, so applies run in the refactor window without touching production. Exclusions are the REAL
data_decisions.csv (read only): the synthetic keys are chosen among its excluded / admitted (12h, 27MHz, date) keys.
Real part (read only; --no-real skips it): the content tests on a few production manifests; results are printed, and
only 'the test ran and returned a verdict' is asserted (production state may legitimately fail a test - that is the
finding).
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
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')

import numpy as np                                   # noqa: E402
from astropy.io import fits                          # noqa: E402

from exopipe.phase3 import cli                       # noqa: E402
from exopipe.phase3.config import CFG, ConfigRefusal  # noqa: E402
from exopipe.phase3.core import validate             # noqa: E402
from exopipe.phase3.core.validate import content_test  # noqa: E402

BASE = f'/tmp/exopipe3_tests_archive/{os.getpid()}'
RUN = 'Run_20260909_000000'
LEDGER_ID = 'FINDING-EXCLUDED-NIGHTS-ARCHIVE-PRODUCTS-NOT-RETIRED-20261012'
RULING = f"Gregg 2026-10-13 'test only' 05:00 UTC {LEDGER_ID}"
EPOCH_CLASSES = "archive_push.EPOCH_NAME_CLASSES=(('', None, 'test_fits_copy'),)"
RESULTS = []


@content_test('test_fits_copy')
def _fits_ok(path, **ctx):
    with fits.open(path) as h:
        d = h[0].data
    return d is not None and np.isfinite(d).any(), None, 'synthetic FITS readable'


def ok(cond, what):
    RESULTS.append((bool(cond), what))
    print(('PASS ' if cond else 'FAIL ') + what, flush=True)
    assert cond, what


def main_cli(argv):
    try:
        return cli.main(['archive'] + argv)
    finally:
        CFG.reset()


def rec(out, unit):
    return json.load(open(os.path.join(out, '_records', 'archive', f'{unit}.json')))


def write_fits(path, value, prodname=None, mtime=None, shape=(8, 8), extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    d = np.full(shape, value, dtype=np.float32)
    d[0, 0] = np.nan
    h = fits.Header()
    if prodname:
        h['PRODNAME'] = prodname
    for k, v in (extra or {}).items():
        h[k] = v
    fits.writeto(path, d, h, overwrite=True)
    if mtime:
        os.utime(path, (mtime, mtime))
    return path


def frame_name(scope, st, date, pbcorr=False):
    rob = 'Robust-0.75' if st == 'I' else 'Robust+0.0'
    return (f'{scope}_{st}_deep_Taper_{rob}{"_pbcorr" if pbcorr else ""}_dewarped_aligned_msub_ssub_LST12h_'
            f'{date.replace("-", "")}T050000')


def archive_frame(root, date, scope, st, name, cadence='deep'):
    return os.path.join(root, '12h', date, RUN, scope, st, cadence, name + '.fits')


def pick_dates():
    from exopipe.phase3.core import exclusions
    ex = exclusions.load()
    cands = [f'2025-{m:02d}-{d:02d}' for m in (4, 5) for d in range(1, 31)]
    excl = [d for d in cands if ex.is_excluded('12h', '27MHz', d)]
    adm = [d for d in cands if not ex.is_excluded('12h', '27MHz', d) and not ex.is_excluded('12h', 'Red', d)]
    assert excl and len(adm) >= 4, (excl, adm)
    return excl[0], adm[:4]


def test_config():
    for s in ('archive', 'archive_quarantine', 'archive_sync', 'archive_retire', 'archive_push', 'archive_snapshot',
              'archive_rollback'):
        ok(s in CFG.sections(), f'config section {s} loaded')
    for key in ('archive.NEVER_DELETE', 'archive.SAME_FILESYSTEM_ONLY', 'archive.RULING_REQUIRED',
                'archive_quarantine.SELECT_RULE', 'archive_sync.SAME_DATA_RTOL'):
        try:
            CFG.set(key, 'False' if 'NEVER' in key or 'SAME_FILE' in key else "'x'")
            refused = False
        except (ConfigRefusal, ValueError):
            refused = True
        finally:
            CFG.reset()
        ok(refused, f'--set {key} refused (protected)')
    ok(CFG.ruled('archive_retire.EXCLUDED_QUARANTINE_RUN') == ('FINDING-EXCLUDED-NIGHTS-ARCHIVE-PRODUCTS-NOT-RETIRED-20261012',),
       'archive_retire.EXCLUDED_QUARANTINE_RUN carries its ruling')


def test_deep(out, root, E, A):
    a, b, c, d = A
    # ---- the mirror archive: 27MHz V and I frames on 5 nights, epoch images, a 10min file of the excluded night
    for date in (a, b, d, E):
        for st in ('V', 'I'):
            write_fits(archive_frame(root, date, '27MHz', st, frame_name('27MHz', st, date)), 1.0,
                       prodname=frame_name('27MHz', st, date), mtime=time.time() - 86400 * 3)
        write_fits(archive_frame(root, date, '27MHz', 'V', f'27MHz_V_deep_Taper_Robust+0.0_dewarped_aligned_LST12h_'
                                 f'{date.replace("-", "")}T050000'), 3.0)
    write_fits(archive_frame(root, E, '27MHz', 'V', f'27MHz_V_10min_x_{E}_msub', cadence='10min'), 2.0)
    n_excl = len(glob.glob(os.path.join(root, '12h', E, RUN, '*', '*', '*', '*.fits')))

    # ---- dry run writes nothing
    rc = main_cli(['retire-excluded', '--archive-root', root, '--lsts', '12h', '--out', out, '--dry-run'])
    ok(rc == 0 and not os.path.exists(os.path.join(out, '_records')), 'retire-excluded --dry-run: exit 0, nothing written')

    # ---- per-hour selection (--epochs): only that (LST, night); an action without per-hour items refuses the flag
    eo = out + '_epochs'
    rc = main_cli(['retire-excluded', '--archive-root', root, '--epochs', f'12h_{a}', '--out', eo])
    ok(rc == 0 and rec(eo, 'retire-excluded_deep')['meta']['n_files'] == 0, 'retire-excluded --epochs <admitted hour>: 0 files')
    rc = main_cli(['retire-excluded', '--archive-root', root, '--epochs', f'12h_{E}', '--out', eo])
    ok(rc == 0 and rec(eo, 'retire-excluded_deep')['meta']['n_files'] == n_excl,
       f'retire-excluded --epochs <excluded hour>: the {n_excl} files of that hour only')
    ok(all(t['passed'] for t in rec(eo, 'retire-excluded_deep')['tests']), 'retirement audit honours --epochs')
    rc = main_cli(['quarantine', '--archive-root', root, '--epochs', f'12h_{a}', '--out', eo])
    ok(rc == 1, 'quarantine --epochs: error (not a per-hour selector of that action; never silently ignored)')
    shutil.rmtree(eo, ignore_errors=True)

    # ---- retire-excluded plan
    rc = main_cli(['retire-excluded', '--archive-root', root, '--lsts', '12h', '--out', out])
    ok(rc == 0, 'retire-excluded plan exit 0')
    plan = glob.glob(os.path.join(out, 'manifests', 'retire_excluded', '*', 'DRYRUN_excluded_archive_products_*.csv'))
    ok(len(plan) == 1, 'combined DRYRUN plan written')
    rows = list(csv.DictReader(open(plan[0])))
    ok(len(rows) == n_excl and all(r['date'] == E for r in rows), f'plan lists the {n_excl} files of the excluded night only')
    ok(set(rows[0]) == set(CFG.archive_retire.MANIFEST_COLUMNS), 'plan columns = the 2026-10-12 dry run columns')
    ok(any(r['kind'] == 'epoch_image' for r in rows) and any(r['kind'] == 'search_frame' for r in rows), 'kind column')
    s = rec(out, 'retire-excluded_deep')
    ok(s['status'] == 'done' and all(t['passed'] for t in s['tests']), 'summary unit done, retirement audit (plan) passes')

    # ---- apply: refused without ruling, without manifest, into production; then done on the mirror
    rc = main_cli(['retire-excluded', '--archive-root', root, '--lsts', '12h', '--out', out, '--apply'])
    ok(rc == 2 and 'G10' in rec(out, 'retire-excluded_deep_run')['reason'], 'apply without --apply-ruling: G10 (exit 2)')
    rc = main_cli(['retire-excluded', '--archive-root', root, '--lsts', '12h', '--out', out, '--apply',
                   '--apply-ruling', RULING])
    ok(rc == 2 and 'G11' in rec(out, 'retire-excluded_deep_run')['reason'], 'apply without --manifest: G11 (exit 2)')
    rc = main_cli(['retire-excluded', '--lsts', '12h', '--out', out, '--apply', '--apply-ruling', RULING,
                   '--manifest', plan[0]])
    ok(rc == 2 and 'G2' in rec(out, 'retire-excluded_deep_run')['reason'], 'apply on the production archive: G2 (exit 2)')
    for rp in glob.glob(os.path.join(out, '_records', 'archive', 'retire-excluded_*.json')):   # as if written from a
        r_ = json.load(open(rp))                                                                 # clean tree (resumable)
        r_['commit'] = str(r_.get('commit') or '').replace('-dirty', '')
        json.dump(r_, open(rp, 'w'))
    rc = main_cli(['retire-excluded', '--archive-root', root, '--lsts', '12h', '--out', out, '--apply', '--apply-ruling',
                   RULING, '--manifest', plan[0], '--resume'])
    ok(rc == 0, 'retire-excluded apply on the mirror (with --resume: the done PLAN records never vouch for the apply): exit 0')
    left = glob.glob(os.path.join(root, '12h', E, RUN, '*', '*', '*', '*.fits'))
    moved = glob.glob(os.path.join(root, 'Run_EXCLUDED_QUARANTINE_20261012', '12h', E, RUN, '*', '*', '*', '*.fits'))
    ok(not left and len(moved) == n_excl, 'every excluded file moved to Run_EXCLUDED_QUARANTINE_20261012')
    ok(len(glob.glob(os.path.join(root, '12h', a, RUN, '*', '*', '*', '*.fits'))) == 3, 'admitted nights untouched')
    applied = glob.glob(os.path.join(out, 'manifests', 'retire_excluded', '*', 'APPLIED_excluded_archive_products_*.csv'))
    ok(len(applied) == 1 and len(list(csv.DictReader(open(applied[0])))) == n_excl, 'APPLIED manifest, one row per move')
    r = validate.run('quarantine_manifest', applied[0])
    ok(all(x['passed'] for x in r), 'APPLIED manifest passes rows_quarantined: ' + r[0]['detail'])
    aud = glob.glob(os.path.join(out, 'manifests', 'retire_excluded', '*', 'retirement_audit_applied_*.json'))
    r = validate.run('retirement_audit', aud[0])
    ok(all(x['passed'] for x in r), 'retirement audit (applied) passes: ' + r[0]['detail'])
    r = validate.run('quarantine_manifest', plan[0])
    ok(not r[0]['passed'] and 'planned source missing' in r[0]['detail'], 'the old plan now fails (its files moved)')

    # ---- attempt for sync / retire-refused (27MHz V): a keep, b replace, c publish, d retire (refused night)
    att = os.path.join(BASE, 'collect', 'attempts', '12h_27MHz_V_20261004T000000')
    fd = os.path.join(att, 'Run_P3S5', 'vresid', '12h_27MHz')
    write_fits(os.path.join(fd, f'{a}_V_redo_resid_27MHz.fits'), 1.0, prodname=frame_name('27MHz', 'V', a))
    write_fits(os.path.join(fd, f'{b}_V_redo_resid_27MHz.fits'), 5.0, prodname=frame_name('27MHz', 'V', b))
    write_fits(os.path.join(fd, f'{c}_V_redo_resid_27MHz.fits'), 1.0, prodname=frame_name('27MHz', 'V', c))
    sync = ['--archive-root', root, '--cells', '12h_27MHz', '--stokes', 'V', '--attempt', att, '--out', out,
            '--set', EPOCH_CLASSES]
    rc = main_cli(['retire-refused'] + sync)
    rr = rec(out, 'retire-refused_12h_27MHz_V')
    ok(rc == 0 and rr['meta']['n_refused_night_frames'] == 1, 'retire-refused plan: 1 frame of a gate-refused night')
    rc = main_cli(['sync'] + sync)
    sr = rec(out, 'sync_12h_27MHz_V')
    ok(rc == 0 and sr['meta']['counts'] == {'keep': 1, 'replace': 1, 'publish': 1, 'retire': 1},
       f"sync plan counts {sr['meta'].get('counts')}")
    ok(all(t['passed'] for t in sr['tests']), 'sync plan record passes archive_equals_attempt')
    rc = main_cli(['sync', '--apply'] + sync)
    ok(rc == 2 and 'G10' in rec(out, 'sync_12h_27MHz_V')['reason'], 'sync apply with a retire row needs a ruling (G10)')
    ok(os.path.exists(archive_frame(root, d, '27MHz', 'V', frame_name('27MHz', 'V', d))), 'nothing moved by the refusal')
    rc = main_cli(['sync', '--apply', '--apply-ruling', RULING] + sync)
    sr = rec(out, 'sync_12h_27MHz_V')
    ok(rc == 0 and sr['status'] == 'done', f"sync apply on the mirror: {sr['status']} {sr['reason']}")
    with fits.open(archive_frame(root, b, '27MHz', 'V', frame_name('27MHz', 'V', b))) as h:
        ok(float(np.nanmax(h[0].data)) == 5.0, 'replaced archive frame = the attempt frame')
    ok(os.path.exists(archive_frame(root, c, '27MHz', 'V', frame_name('27MHz', 'V', c))), 'published frame in place')
    ok(os.path.exists(os.path.join(root, 'Run_SEARCH_QUARANTINE_20261004', '12h', d, RUN, '27MHz', 'V', 'deep',
                                   frame_name('27MHz', 'V', d) + '.fits')), 'retired frame in the sync quarantine')
    ok(not glob.glob(os.path.join(root, '12h', '*', RUN, '27MHz', 'V', 'deep', '*.part')), 'no .part left')

    # ---- G20: an attempt holding a frame of the EXCLUDED night (its archive copy retired above) would publish it back
    att2 = os.path.join(BASE, 'collect', 'attempts', '12h_27MHz_V_20261005T000000')
    fd2 = os.path.join(att2, 'Run_P3S5', 'vresid', '12h_27MHz')
    write_fits(os.path.join(fd2, f'{a}_V_redo_resid_27MHz.fits'), 1.0, prodname=frame_name('27MHz', 'V', a))
    write_fits(os.path.join(fd2, f'{E}_V_redo_resid_27MHz.fits'), 1.0, prodname=frame_name('27MHz', 'V', E))
    s2 = ['--archive-root', root, '--cells', '12h_27MHz', '--stokes', 'V', '--attempt', att2, '--out', out,
          '--set', EPOCH_CLASSES]
    rc = main_cli(['sync'] + s2)
    sr = rec(out, 'sync_12h_27MHz_V')
    ok(rc == 2 and 'G20' in sr['reason'], f"sync plan with a publish row on an excluded night: refused G20 ({sr['reason'][:80]})")
    rc = main_cli(['sync', '--apply', '--apply-ruling', RULING] + s2)
    sr = rec(out, 'sync_12h_27MHz_V')
    ok(rc == 2 and 'G20' in sr['reason'] and not os.path.exists(archive_frame(root, E, '27MHz', 'V', frame_name('27MHz', 'V', E)))
       and os.path.exists(archive_frame(root, b, '27MHz', 'V', frame_name('27MHz', 'V', b))),
       'sync apply refused by G20 before anything moved (excluded frame not published, retire rows not moved)')
    rc = main_cli(['sync', '--waive', 'archive.G20_exclusion_no_readmit', '--ruling', RULING] + s2)
    ok(rc == 2 and 'never waivable' in rec(out, 'sync_deep_run')['reason'], 'G20 is never waivable')

    # ---- pre-search quarantine (27MHz I): old frames (3 days) before the launch, a new one after
    newf = write_fits(archive_frame(root, c, '27MHz', 'I', frame_name('27MHz', 'I', c)), 1.0, prodname='x')
    launch = str(int(time.time() - 3600))
    q = ['--archive-root', root, '--cells', '12h_27MHz', '--stokes', 'I', '--launch-epoch', launch, '--round', 'T1',
         '--out', out, '--no-membership-precheck']
    rc = main_cli(['quarantine'] + q)
    ok(rc == 2 and 'G10' in rec(out, 'quarantine_deep_run')['reason'],
       '--no-membership-precheck needs --apply-ruling')
    rc = main_cli(['quarantine'] + q + ['--apply-ruling', RULING])
    qr = rec(out, 'quarantine_12h_27MHz_I')
    ok(rc == 0 and qr['meta']['n_planned'] == 3, f"quarantine plan: 3 earlier frames ({qr['meta'].get('n_planned')})")
    blocker = os.path.join(root, 'Run_SEARCH_QUARANTINE_T2', os.path.relpath(
        archive_frame(root, a, '27MHz', 'I', frame_name('27MHz', 'I', a)), root))
    write_fits(blocker, 0.0)
    rc = main_cli(['quarantine'] + q[:-3] + ['--round', 'T2', '--out', out, '--no-membership-precheck', '--apply-ruling',
                                             RULING])
    ok(rc == 2 and 'G4' in rec(out, 'quarantine_12h_27MHz_I')['reason'], 'taken quarantine destination: G4 before any move')
    rc = main_cli(['quarantine', '--apply', '--apply-ruling', RULING] + q)
    qr = rec(out, 'quarantine_12h_27MHz_I')
    ok(rc == 0 and qr['status'] == 'done', 'quarantine apply on the mirror')
    ok(os.path.exists(newf) and not os.path.exists(archive_frame(root, a, '27MHz', 'I', frame_name('27MHz', 'I', a))),
       'old frames moved, the frame newer than the launch kept')
    man = os.path.join(out, 'manifests', 'frame_quarantine', '12h_27MHz_I.csv')
    ok(len(list(csv.DictReader(open(man)))) == 3, 'manifest of record: 3 rows (path,dest,bytes,md5,cell,stokes,moved_at)')
    raw = open(man, 'rb').read()
    ok(raw.startswith(b','.join(c.encode() for c in CFG.archive.QUARANTINE_COLUMNS) + b'\r\n')
       and raw.count(b'\r\n') == 4 and raw.count(b'\n') == 4,
       'manifest lines are CRLF like production (p3s5_quarantine_frames.py csv.DictWriter; frame_quarantine/*.csv)')

    # ---- rollback of that quarantine
    since = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(time.time() - 600))
    rb = ['--archive-root', root, '--cells', '12h_27MHz', '--stokes', 'I', '--manifest', man, '--since', since, '--out', out]
    rc = main_cli(['rollback'] + rb)
    ok(rc == 0 and rec(out, 'rollback_12h_27MHz_I')['meta']['n_rows'] == 3, 'rollback plan: 3 rows')
    rc = main_cli(['rollback', '--apply'] + rb)
    ok(rc == 2 and 'G10' in rec(out, 'rollback_12h_27MHz_I')['reason'], 'rollback apply needs a ruling')
    rc = main_cli(['rollback', '--apply', '--apply-ruling', RULING] + rb)
    ok(rc == 0 and os.path.exists(archive_frame(root, a, '27MHz', 'I', frame_name('27MHz', 'I', a))), 'rollback restored')
    r = validate.run('quarantine_manifest', man)
    ok(r[0]['passed'] and 'rolled back' in r[0]['detail'], 'quarantine manifest after rollback: ' + r[0]['detail'])
    rc = main_cli(['rollback'] + rb)
    ok(rc == 2 and 'G15' in rec(out, 'rollback_12h_27MHz_I')['reason'] and 'occupied' in rec(out, 'rollback_12h_27MHz_I')['reason'],
       'second rollback plan: originals occupied -> G15 refused, nothing moved (the P10f case)')
    # G20: a quarantined frame of the EXCLUDED night is never rolled back into the generation Run
    xp = archive_frame(root, E, '27MHz', 'I', frame_name('27MHz', 'I', E))
    xq = os.path.join(root, 'Run_SEARCH_QUARANTINE_T9', os.path.relpath(xp, root))
    write_fits(xq, 1.0)
    from exopipe.phase3.lib.archive import common as K
    xman = os.path.join(BASE, 'xman', '12h_27MHz_I.csv')
    K.write_csv(xman, CFG.archive.QUARANTINE_COLUMNS, [dict(path=xp, dest=xq, bytes=os.path.getsize(xq),
                                                             md5=K.checksum(xq), cell='12h_27MHz', stokes='I',
                                                             moved_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))])
    rc = main_cli(['rollback', '--archive-root', root, '--cells', '12h_27MHz', '--stokes', 'I', '--manifest', xman,
                   '--since', since, '--out', out])
    ok(rc == 2 and 'G20' in rec(out, 'rollback_12h_27MHz_I')['reason'] and not os.path.exists(xp),
       'rollback of a frame of an excluded night: G20 refused, nothing restored')
    # conformant names (Gregg 2026-10-16): a quarantined frame whose product is in the archive under its new name (the rename
    # inserted shflux) is never restored beside it
    from exopipe.phase3.core import products as P
    op = archive_frame(root, a, '27MHz', 'I', frame_name('27MHz', 'I', a))          # restored by the rollback above
    g = P.parse_name(os.path.basename(op))
    np_ = os.path.join(os.path.dirname(op), P.build_name(**dict(g, proc=list(g['proc']) + ['shflux'])))
    os.rename(op, np_)
    rq = os.path.join(root, 'Run_SEARCH_QUARANTINE_T10', os.path.relpath(op, root))
    write_fits(rq, 2.0)
    rman = os.path.join(BASE, 'rman', '12h_27MHz_I.csv')
    K.write_csv(rman, CFG.archive.QUARANTINE_COLUMNS, [dict(path=op, dest=rq, bytes=os.path.getsize(rq), md5=K.checksum(rq),
                                                             cell='12h_27MHz', stokes='I',
                                                             moved_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))])
    rc = main_cli(['rollback', '--archive-root', root, '--cells', '12h_27MHz', '--stokes', 'I', '--manifest', rman,
                   '--since', since, '--out', out])
    why = rec(out, 'rollback_12h_27MHz_I')['reason']
    ok(rc == 2 and 'G15' in why and 'under another name' in why and not os.path.exists(op) and os.path.exists(np_),
       'rollback of a frame whose product was renamed since (conformant names): G15 refused, never restored beside it')
    ok(K.product_key(os.path.basename(np_)) == K.product_key(os.path.basename(op))
       and K.product_key(os.path.basename(op).replace('_I_', '_V_')) != K.product_key(os.path.basename(op))
       and K.occupant(op) == np_, 'product_key ignores only the rename tokens; occupant finds the renamed product')
    os.rename(np_, op)                                  # the mirror as the later tests expect
    os.remove(rq)
    return att


def test_push_epoch(out, root, E, A):
    a, b, c, d = A
    stg = os.path.join(BASE, 'staging')
    # epoch a: keep (same bytes), replace (different pixels), publish (new); a .part intermediate (refused item)
    same = archive_frame(root, a, '27MHz', 'V', frame_name('27MHz', 'V', a))
    os.makedirs(os.path.join(stg, os.path.dirname(os.path.relpath(same, root))), exist_ok=True)
    shutil.copy2(same, os.path.join(stg, os.path.relpath(same, root)))
    write_fits(os.path.join(stg, os.path.relpath(archive_frame(root, a, '27MHz', 'V', f'27MHz_V_deep_Taper_Robust+0.0_'
               f'dewarped_aligned_LST12h_{a.replace("-", "")}T050000'), root)), 7.0)
    write_fits(os.path.join(stg, os.path.relpath(archive_frame(root, a, '27MHz', 'V',
               frame_name('27MHz', 'V', a, pbcorr=True)), root)), 1.5)
    open(os.path.join(stg, '12h', a, RUN, '27MHz', 'V', 'deep', 'x.fits.part'), 'w').write('x')
    p = ['push', '--epochs', f'12h_{a}', '--staging-root', stg, '--archive-root', root, '--out', out, '--set', EPOCH_CLASSES]
    rc = main_cli(p)
    u = rec(out, f'push_epoch_12h_{a}')
    ok(rc == 0 and u['meta']['counts'] == {'keep': 1, 'publish': 1, 'replace': 1, 'keep-or-replace': 0},
       f"per-hour push plan: {u['meta'].get('counts')}")
    ok(len(u['refused_items']) == 1 and u['refused_items'][0]['gate'] == 'G9', 'intermediate file refused, never pushed')
    rc = main_cli(p + ['--apply', '--round', 'P1'])
    u = rec(out, f'push_epoch_12h_{a}')
    ok(rc == 0 and u['status'] == 'done', f"per-hour push apply on the mirror: {u['status']} {u['reason']}")
    img = archive_frame(root, a, '27MHz', 'V', f'27MHz_V_deep_Taper_Robust+0.0_dewarped_aligned_LST12h_'
                                               f'{a.replace("-", "")}T050000')
    with fits.open(img) as h:
        ok(float(np.nanmax(h[0].data)) == 7.0, 'replaced epoch image = the staged one')
    ok(os.path.exists(os.path.join(root, 'Run_SEARCH_QUARANTINE_P1', os.path.relpath(img, root))), 'old one quarantined')
    # the excluded night: every staged file refused -> unit refused
    write_fits(os.path.join(stg, '12h', E, RUN, '27MHz', 'V', 'deep', frame_name('27MHz', 'V', E) + '.fits'), 1.0)
    rc = main_cli(['push', '--epochs', f'12h_{E}', '--staging-root', stg, '--archive-root', root, '--out', out])
    ok(rc == 2 and rec(out, f'push_epoch_12h_{E}')['refused_items'][0]['gate'] == 'exclusion',
       'a staged product of an excluded night is refused (exit 2)')
    # audit-only (no staging tree): the archive's own files of the hour
    rc = main_cli(['push', '--epochs', f'12h_{a}', '--archive-root', root, '--out', out, '--set', EPOCH_CLASSES])
    ok(rc == 0 and rec(out, f'push_epoch_12h_{a}')['meta']['mode'] == 'audit-only', 'per-hour audit-only mode')
    # sub-LST: nothing archived
    rc = main_cli(['push', '--tier', 'sublst', '--epochs', f'12h_{a}', '--out', out])
    ok(rc == 3, 'push --tier sublst: no-input (the sub-LST tier archives nothing today)')


def test_mosaic(out, root):
    stg = os.path.join(BASE, 'mosaic_staging')
    n1 = 'Blue_I_deep_Taper_Robust+0.0_coadd_mosaic_jybeam_feather_molrender_allsky_20241218-20251029_N024.fits'
    n2 = 'Blue_I_deep_Taper_Robust+0.0_coadd_mosaic_jybeam_feather_allsky_20241218-20251029_N024.fits'
    hdr = {'BUNIT': 'Jy/beam', 'PBCORR': False, 'NSTACK': 24, 'DATE-OBS': '2024-12-18T03:04:12.100',
           'DATE-END': '2025-10-29T08:21:24.560'}
    write_fits(os.path.join(stg, 'renders', 'wideband', 'mollweide', n1), 1.0, extra=hdr, shape=(200, 4))
    hp = os.path.join(stg, 'wideband', n2)
    os.makedirs(os.path.dirname(hp), exist_ok=True)
    ph = fits.PrimaryHDU()
    ph.header.update(hdr)
    fits.HDUList([ph, fits.BinTableHDU.from_columns([fits.Column(name='SIGNAL', format='E',
                                                                 array=np.arange(300, dtype=np.float32))])]).writeto(hp)
    tab = os.path.join(BASE, 'mosaic_qa', 'flux_anchor.csv')
    os.makedirs(os.path.dirname(tab), exist_ok=True)
    open(tab, 'w').write('a,b\n1,2\n')
    entry = {'enabled': True, 'src_globs': (f'{stg}/wideband/*.fits', f'{stg}/renders/wideband/*/*.fits'), 'tables': (tab,),
             'run_dir': '{root}/mosaic/Run_FEATHER2_{d0}-{d1}', 'taper_only': True, 'existing': 'refuse'}
    s = ['--set', f'archive_push.CLASSES[mosaic]={entry!r}']
    rc = main_cli(['push', '--class', 'mosaic', '--archive-root', root, '--out', out] + s)
    ok(rc == 0 and rec(out, 'push_mosaic')['meta']['n_files'] == 3, 'mosaic push plan: 3 rows (2 FITS + table)')
    rc = main_cli(['push', '--class', 'mosaic', '--archive-root', root, '--out', out, '--apply'] + s)
    u = rec(out, 'push_mosaic')
    ok(rc == 0 and u['status'] == 'done', f"mosaic push apply on the mirror: {u['status']} {u['reason']}")
    dst = os.path.join(root, 'mosaic', 'Run_FEATHER2_20241218-20251029', 'Blue', 'I', 'mollweide', n1)
    ok(os.path.exists(dst) and os.path.exists(dst.replace('mollweide/' + n1, 'healpix/' + n2)), 'layout <band>/I/<sub>/')
    ok(any(t['name'] == 'archive_mosaic_copy.class_audit' and t['passed'] for t in u['tests']), 'mosaic content audit')
    rc = main_cli(['push', '--class', 'mosaic', '--archive-root', root, '--out', out] + s)
    ok(rc == 2 and 'already exist' in rec(out, 'push_mosaic')['reason'], 'second push refused: destinations exist (G9)')
    man = glob.glob(os.path.join(out, 'manifests', 'push', 'MANIFEST_MOSAIC_*.csv'))[0]
    rc = main_cli(['audit', '--class', 'mosaic', '--manifest', man, '--archive-root', root, '--out', out])
    ok(rc == 0, 'audit --class mosaic of the push manifest passes')
    with fits.open(dst, mode='update') as h:
        h[0].header['NSTACK'] = 23
    rc = main_cli(['audit', '--class', 'mosaic', '--manifest', man, '--archive-root', root, '--out', out])
    ok(rc == 2, 'audit catches a changed file (NSTACK, md5): exit 2')
    rc = main_cli(['push', '--class', 'qa', '--out', out])
    ok(rc == 2, 'push --class qa: disabled in config, refused')


def test_snapshot(out):
    # ---- supersede through the production ship script on mirror roots
    cell = '21h_2025-08-26'
    w = os.path.join(BASE, 'ft_work')
    ship, phot, quar = (os.path.join(BASE, x) for x in ('ship', 'phot', 'quar'))
    for f in ('noise_36MHz_I_median_1min.csv', 'runinfo_36MHz_I_median.json', 'wideband_members_I.csv'):
        os.makedirs(os.path.join(w, 'out'), exist_ok=True)
        open(os.path.join(w, 'out', f), 'w').write('new ' + f)
    os.makedirs(os.path.join(ship, cell), exist_ok=True)
    for f in ('noise_36MHz_I_median_1min.csv', 'runinfo_36MHz_I_median.json', 'group_I.log'):
        open(os.path.join(ship, cell, f), 'w').write('old ' + f)
    sargs = ['--tier', 'snapshot', '--cells', cell, '--workdir', w, '--ship-root', ship, '--phot-root', phot,
             '--quar-root', quar, '--out', out]
    rc = main_cli(['supersede'] + sargs)
    u = rec(out, f'supersede_{cell}')
    ok(rc == 0 and u['meta']['plan_counts']['replaced_fleet'] == 2 and u['meta']['plan_counts']['superseded_no_successor'] == 1,
       f"supersede plan (ship listing ported): {u['meta'].get('plan_counts')}")
    # G14 pre-ship: the launcher ships only after the kept 10-s check and content tests A-E passed on the work dir
    rc = main_cli(['supersede', '--apply'] + sargs)
    u = rec(out, f'supersede_{cell}')
    ok(rc == 2 and 'G14' in u['reason'] and open(os.path.join(ship, cell, 'group_I.log')).read().startswith('old'),
       'supersede apply without passing pre-ship checks: G14 refused, nothing shipped')
    open(os.path.join(w, 'kept_10s_check.log'), 'w').close()
    for k in 'ABCDE':
        open(os.path.join(w, f'test_{k}.log'), 'w').write(f'TEST {k} {cell}: synthetic, 0 failures -> '
                                                          f'{"FAIL" if k == "C" else "PASS"}\n')
    rc = main_cli(['supersede', '--apply'] + sargs)
    ok(rc == 2 and 'content test C did not pass' in rec(out, f'supersede_{cell}')['reason'], 'a FAIL test log refuses (G14)')
    open(os.path.join(w, 'test_C.log'), 'w').write(f'TEST C {cell}: synthetic, 0 failures -> PASS\n')
    rc = main_cli(['supersede', '--apply'] + sargs)
    u = rec(out, f'supersede_{cell}')
    ok(rc == 0 and u['status'] == 'done' and u['meta']['shipped']['quarantined'] == 3,
       f"supersede apply (snapshot_frametime_ship.sh in place): {u['status']} {u['meta'].get('shipped')}")
    ok(open(os.path.join(ship, cell, 'noise_36MHz_I_median_1min.csv')).read().startswith('new'), 'new file shipped')
    ok(not os.path.exists(os.path.join(ship, cell, 'group_I.log')), 'superseded file without successor quarantined')
    # ---- quarantine-rows through the production script
    fields = json.loads(os.popen(f"{CFG.site.PYTHON['py36']} -c \"import sys, json; sys.path.insert(0, '/lustre/gh/main/"
                                 f"clv15'); from phase3_snapshot_fit import STORE_FIELDS as S, TEMPLATE_FIELDS as T; "
                                 f"print(json.dumps([S, T]))\"").read())
    sdir = os.path.join(BASE, 'stores')
    os.makedirs(sdir, exist_ok=True)
    store, tmpl = os.path.join(sdir, 'fits_21h_2025-08-26.csv'), os.path.join(sdir, 'fits_21h_2025-08-26_template.csv')
    base = dict(cell=cell, stokes='I', tier='1min', ts='2025-08-26T05:00:00', ra_deg='10.0', dec_deg='20.0')
    rows = [dict(base, subband='36MHz', image='a'), dict(base, subband='Red', image='x identity screen', ts='2025-08-26T05:01:00'),
            dict(base, subband='Red', image='old', ts='2025-08-26T05:02:00')]
    for p, f, rr in ((store, fields[0], rows), (tmpl, fields[1], rows[2:])):
        with open(p, 'w', newline='') as fh:
            wr = csv.DictWriter(fh, fieldnames=f, extrasaction='ignore')
            wr.writeheader()
            wr.writerows(rr)
    qd = os.path.join(BASE, 'qdir_rows')
    ra = ['--tier', 'snapshot', '--stores', store, '--qdir', qd, '--quiet-min', '0', '--batch-name', 'T', '--out', out]
    rc = main_cli(['quarantine-rows'] + ra)
    u = rec(out, 'rows_T')
    ok(rc == 0 and u['meta']['per_store'][store]['quarantined'] == 1 and not os.path.exists(qd),
       'quarantine-rows plan (--dry, per store): 1 unmarked wideband row; nothing written')
    rc = main_cli(['quarantine-rows', '--apply'] + ra)
    u = rec(out, 'rows_T')
    ok(rc == 0 and u['status'] == 'done', f"quarantine-rows apply on mirror stores: {u['status']} {u['reason']}")
    ok(len(list(csv.DictReader(open(store)))) == 2, 'store keeps 2 rows')


def test_misc(out, root):
    rc = main_cli(['retire-excluded', '--archive-root', root, '--lsts', '12h', '--out', out, '--waive',
                   'retirement_audit.no_excluded_product_left', '--ruling', RULING])
    ok(rc == 2, '--waive of a never-waivable test is refused')
    rc = main_cli(['supersede', '--tier', 'deep', '--out', out])
    ok(rc == 1, 'supersede is not a deep action (exit 1)')
    rc = main_cli(['retire-excluded', '--tier', 'snapshot', '--cells', '21h_2025-08-06', '--out', out, '--apply',
                   '--archive-root', root])
    ok(rc == 2 and 'report only' in rec(out, 'retire-excluded_snapshot_run')['reason'],
       'retire-excluded --tier snapshot --apply refused (report only), not merely by G2')
    # every class the per-epoch audit assigns is a registered class; a pbcorr sibling is a search_frame
    from exopipe.phase3.lib.archive.sync import class_of
    known = set(validate.classes())
    missing = [c for _, _, c in CFG.archive_push.EPOCH_NAME_CLASSES if c not in known]
    ok(not missing, f'archive_push.EPOCH_NAME_CLASSES classes are registered (missing: {missing})')
    ok(class_of('Blue_V_deep_Taper_Robust+0.0_pbcorr_dewarped_aligned_msub_ssub_LST12h_20250506T045557.fits') == 'search_frame'
       and class_of('18MHz_V_deep_Taper_Robust+0.0_loo-20250806_shflux_mdewarp_dewarped_aligned_coadd_LST21h_20250508-20250906_N014.fits')
       == 'loo_master', 'class_of: pbcorr sibling -> search_frame, LOO master -> loo_master')
    # G12 on the production archive: a live claim (this process) refuses, a dead pid does not
    from exopipe.phase3.core import env
    from exopipe.phase3.core.record import Refusal
    from exopipe.phase3.lib.archive import common as K
    cdir = os.path.join(BASE, 'claims')
    os.makedirs(os.path.join(cdir, 'claims_V', '12h_27MHz'))
    open(os.path.join(cdir, 'claims_V', '12h_27MHz', 'owner'), 'w').write(f'{env.host()}:{os.getpid()}\n')
    os.makedirs(os.path.join(cdir, 'claims_I', '12h_27MHz'))
    open(os.path.join(cdir, 'claims_I', '12h_27MHz', 'owner'), 'w').write(f'{env.host()}:4194303\n')

    class _Ctx:
        production_archive = True
    try:
        CFG.set('archive.WRITER_CLAIM', repr(cdir + '/claims_{stokes}/{cell}/owner'))
        try:
            K.refuse_live_writers(_Ctx(), [('12h_27MHz', 'V')], 'test')
            live_refused = False
        except Refusal as e:
            live_refused = 'G12' in str(e)
        K.refuse_live_writers(_Ctx(), [('12h_27MHz', 'I')], 'test')        # dead pid: no refusal
        _Ctx.production_archive = False
        K.refuse_live_writers(_Ctx(), [('12h_27MHz', 'V')], 'test')        # a mirror archive: no fleet claim applies
    finally:
        CFG.reset()
    ok(live_refused, 'G12: a live fleet claim refuses an apply on the production archive; a dead pid / a mirror does not')


def test_real():
    """READ ONLY: content tests on production manifests. Prints the verdicts; asserts only that each test ran."""
    real = [('quarantine_manifest', '/lustre/gh/main/exopipe_ops/p3s5_state/frame_quarantine/23h_Green_I.csv', {}),
            ('quarantine_manifest', '/lustre/pipeline/exopipe/phase3/Run_SNAP_FRAMETIME_QUARANTINE_20261007/'
                                    '21h_2025-08-26/20261011T060312Z/MANIFEST.csv', {}),
            ('row_quarantine_manifest', '/lustre/gh/main/exopipe_ops/snapshot_batch/'
                                        'fits_quarantine_wideband_double_registered_20261007/MANIFEST.csv', {}),
            ('restore_manifest', '/lustre/gh/main/exopipe_ops/membership_rebuild_20261007/'
                                 'restore_refused_frames_20261008T163213Z.csv', {'checksum': False}),
            ('quarantine_manifest', '/lustre/gh/main/exopipe_ops/excluded_quarantine_20261012/'
                                    'DRYRUN_excluded_archive_products.csv', {})]
    for cls, p, ctx in real:
        if not os.path.exists(p):
            print(f'REAL skip (missing) {cls} {p}')
            continue
        t0 = time.time()
        r = validate.run(cls, p, **ctx)
        for x in r:
            print(f"REAL {'PASS' if x['passed'] else 'FAIL'} {x['name']} {os.path.basename(p)} ({time.time() - t0:.0f} s): "
                  f"{str(x['detail'])[:300]}", flush=True)
        ok(r and all('test raised' not in str(x['detail']) for x in r), f'real {cls} test ran on {os.path.basename(p)}')


def main():
    t0 = time.time()
    os.makedirs(BASE)
    try:
        out, root = os.path.join(BASE, 'out'), os.path.join(BASE, 'archive')
        E, A = pick_dates()
        print(f'synthetic keys: excluded 12h 27MHz {E}; admitted {A}')
        test_config()
        test_deep(out, root, E, A)
        test_push_epoch(out, root, E, A)
        test_mosaic(out, root)
        test_snapshot(out)
        test_misc(out, root)
        if '--no-real' not in sys.argv:
            test_real()
    finally:
        shutil.rmtree(BASE, ignore_errors=True)
        try:
            os.rmdir(os.path.dirname(BASE))
        except OSError:
            pass
    print(f'{sum(1 for r in RESULTS if r[0])} of {len(RESULTS)} checks pass ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
