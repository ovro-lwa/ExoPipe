"""Tests of the catalog-tiers unit (catalog --tier sublst | snapshot): pytest-free, plain asserts, run with python.

    EXOPIPE3_SITE=/lustre/gh/main/exopipe_ops/exopipe3_site.py NUMPY_MADVISE_HUGEPAGE=0 \
        /opt/devel/pipeline/envs/py38_orca_nkosogor/bin/python3 tests/test_catalog-tiers.py

1. config: the four sections load, PROTECTED_KEYS ([lit] and pinned keys) refuse overrides, [env]/[arg] keys do not, every
   RULED ID is a row of the fix ledger;
2. binding: every [lit] key of both tiers equals its in-place literal (read-only AST of the production files);
3. ported shell / heredoc logic on synthetic data (split, TABLE-NOTES, tables check, veto pairs, the elevation floor at
   the fitted positions against production's own heredoc (24d2b39), flux bake, search
   complete, compare copy, cells file, code sha256, frame-time marks, unmarked rows, combined store, prefit counts, tie
   shards, memory model, argparse defaults, chain arguments);
4. content tests on synthetic files (pass and fail cases);
5. end-to-end through the real CLI on STUB in-place modules (a temporary site file whose LEGACY_CODE_ROOTS are stub trees):
   sub-LST preflight (incl. the pcheck child on synthetic FITS masters) + tier-rules + `--from` restore; snapshot preflight,
   store, prefit with a fit round run by core/runner on localhost, tie pass as runner items, chain, test;
6. dry-runs on real items (sub-LST 14h, snapshot 2 cells of batch 2) and content tests on real production products
   (read-only; results printed).
Synthetic data in /tmp/exopipe3_tests_catalog-tiers/ (removed at the end). Nothing is written under a production root.
"""
import csv
import glob
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')
REAL_SITE = os.environ['EXOPIPE3_SITE']
T = '/tmp/exopipe3_tests_catalog-tiers'
PY = sys.executable
OPS_REAL, CLV_REAL = '/lustre/gh/main/exopipe_ops', '/lustre/gh/main/clv15'
RESULTS = []


def ok(name, cond, detail=''):
    RESULTS.append((name, bool(cond), detail))
    print(('PASS ' if cond else 'FAIL ') + name + (('  ' + str(detail)[:300]) if detail else ''), flush=True)
    assert cond, f'{name}: {detail}'


def wcsv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def rcsv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def cli(args, site, timeout=300):
    env = dict(os.environ, EXOPIPE3_SITE=site, PYTHONPATH=REPO)
    p = subprocess.run([PY, '-m', 'exopipe.phase3'] + args, cwd=REPO, env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True, timeout=timeout)
    return p.returncode, p.stdout


# =====================================================================================================================
def test_config():
    from exopipe.phase3.config import CFG, ConfigRefusal
    from exopipe.phase3.core import ledger
    for sec in ('catalog_sublst', 'catalog_sublst_tie', 'catalog_sublst_veto', 'catalog_snapshot'):
        ok(f'config section {sec} loaded', sec in CFG.sections())
    for key in ('catalog_sublst_tie.WIDEN_DEG=(1.0,)', 'catalog_sublst.TIER_RULE3_PIN=/x', 'catalog_snapshot.GHOST_DATES=("x",)',
                'catalog_snapshot.GATE3D_NTHR=()', 'catalog_sublst.ELEV_FLOOR_DEG=20.0', 'catalog_sublst_veto.EL_BINS_DEG=()'):
        try:
            CFG.apply_override_strings([key])
            refused = False
        except ConfigRefusal:
            refused = True
        ok(f'protected override refused: {key.split("=")[0]}', refused)
    CFG.authorize(out='/tmp/exopipe3_tests_catalog-tiers_out')
    CFG.apply_override_strings(['catalog_sublst_veto.ELEV_FLOOR_DEG=28.0', 'catalog_snapshot.STORE_FLOOR=7.5'])
    ok('[env]/[arg] keys overridable into a non-production out', CFG.catalog_sublst_veto.ELEV_FLOOR_DEG == 28.0
       and CFG.catalog_snapshot.STORE_FLOOR == 7.5)
    CFG.reset()
    CFG.authorize()
    missing = []
    for sec in ('catalog_sublst', 'catalog_sublst_tie', 'catalog_sublst_veto', 'catalog_snapshot'):
        for k, ids in (getattr(CFG.sections()[sec], 'RULED', {}) or {}).items():
            for lid in (ids if isinstance(ids, tuple) else (ids,)):
                if not ledger.has_id(lid, CFG.paths.FIX_LEDGER):
                    missing.append(f'{sec}.{k}={lid}')
        for k in getattr(CFG.sections()[sec], 'PROTECTED_KEYS', ()):
            if not hasattr(CFG.sections()[sec], k):
                missing.append(f'{sec}.PROTECTED_KEYS names unknown {k}')
    ok('every RULED ID is a fix-ledger row; every PROTECTED_KEYS entry exists', not missing, missing)
    ok('no lower-case marker attributes left', not any(hasattr(CFG.sections()[s], 'protected') for s in
                                                       ('catalog_sublst', 'catalog_sublst_tie', 'catalog_snapshot')))


def test_binding():
    from exopipe.phase3.lib.catalog import tier_sublst as S, tier_snapshot as P
    rows = S.binding_check(S.LIT)
    bad = [r for r in rows if r[1] != 'ok']
    ok(f'sub-LST bindings: {len(rows)} [lit] keys of this unit equal their in-place literals', not bad, bad[:3])
    # the template K is a SHARED key (fitting.K_TEMPLATE, refactor chat): reported, not asserted - a production edit in flight
    # (2026-10-13 ~05:00 UTC: K re-measured after the template mirror fix) makes the sub-LST preflight refuse until folded in
    kb = S.k_binding(OPS_REAL)
    print('INFO K binding (shared key fitting.K_TEMPLATE[sublst]): ' + '; '.join(f'{r[0]} {r[1]} config={r[2]!r} module={r[3]!r}'
                                                                                for r in kb), flush=True)
    ok('K binding detects a difference between config and sublst_catalogs.sh FITPOS_K (or agrees)',
       all(r[1] in ('ok', 'MISMATCH') for r in kb))
    rows = S.binding_check(P.LIT)
    bad = [r for r in rows if r[1] != 'ok']
    ok(f'snapshot bindings: {len(rows)} [lit] keys equal their in-place literals', not bad, bad[:3])
    # a literal that differs is detected
    os.makedirs(f'{T}/lit', exist_ok=True)
    open(f'{T}/lit/m.py', 'w').write("A = 1.0\nB = (2, 3)\nC = os.environ.get('X', '7')\nD = f'{A}x'\nE = foo()\n")
    lits = S.module_literals(f'{T}/lit/m.py')
    ok('module_literals reads literals, env defaults, f-strings; marks calls unresolved',
       lits['A'] == 1.0 and lits['B'] == (2, 3) and lits['C'] == '7' and lits['D'] == '1.0x' and lits['E'] is S._Unresolved)
    ok('binding compare: 1 == 1.0, (2,3) != (2,4)', S._norm(1) == S._norm(1.0) and S._norm((2, 3)) != S._norm((2, 4)))
    d = P.argparse_defaults(f'{CLV_REAL}/phase3_snapshot_chain.py')
    ok('argparse defaults of the snapshot chain read', d.get('--store-floor') == 7.0 and d.get('--dirty-i-floor') == 0.0
       and d.get('--samples-dir') == '/lustre/gh/main/clv15', {k: d.get(k) for k in ('--store-floor', '--dirty-i-floor')})


# =====================================================================================================================
def test_ported_sublst():
    from exopipe.phase3.lib.catalog import tier_sublst as S
    from exopipe.phase3.core.record import Refusal
    U = f'{T}/U1'
    hdr = ['cand_id', 'lst', 'date', 'band', 'stokes', 'cadence', 'ra', 'dec', 'snr']
    rows = [['V1', '14h', '2025-04-06T00', '18MHz', 'V', '10min', '200', '40', '6'],
            ['V2', '14h', '2025-04-06T01', 'Red', 'V', '10min', '201', '41', '6'],
            ['I1', '14h', '2025-04-06T02', '41MHz', 'I', '10min', '202', '42', '7'],
            ['I2', '14h', '2025-04-06T03', 'Full', 'I', '10min', '203', '43', '8'],
            ['X1', '14h', '2025-04-06T03', 'Weird', 'I', '10min', '203', '43', '8']]
    wcsv(f'{U}/selected_all.csv', hdr, rows)
    lines = S.split(U, ('Red', 'Green', 'Blue', 'Full'), 'MHz')
    got = {k: [r['cand_id'] for r in rcsv(f'{U}/selected_{k}.csv')] for k in ('V_sb', 'V_wb', 'I_sb', 'I_wb')}
    ok('split by Stokes into sb (MHz in band) and wb groups', got == {'V_sb': ['V1'], 'V_wb': ['V2'], 'I_sb': ['I1'],
                                                                     'I_wb': ['I2']} and len(lines) == 4, got)
    wcsv(f'{U}/rem.csv', ['cand_id', 'removal_reason', 'conform_note'],
         [['a', 'table: [el35-50] table: parent S=3 at 1 bw (image-pixel) < 2', ''],
          ['b', 'table: parent S=3 at 1 bw < 2', ''], ['c', 'elevation', '']])
    line, n, bad = S.table_notes(f'{U}/rem.csv')
    ok('TABLE-NOTES counts table removals without [el and (image-pixel)', (n, bad) == (2, 1) and line.startswith('TABLE-NOTES 2'))
    json.dump({'tables': {'sublst_V_18MHz_el25-35': {'metric': 'image-pixel'}, 'sublst_V_18MHz': {'metric': 'image-pixel'},
                          '10min_V_18MHz': {'metric': 'angular'}}}, open(f'{U}/t.json', 'w'))
    ok('sub-LST tables check (keys, elevation band, metric)', S.tables_present(f'{U}/t.json', 'image-pixel') == (True, 2))
    json.dump({'tables': {'sublst_V_18MHz': {'metric': 'angular'}}}, open(f'{U}/t2.json', 'w'))
    ok('sub-LST tables check refuses no _el / other metric', S.tables_present(f'{U}/t2.json', 'image-pixel')[0] is False)
    wcsv(f'{U}/master_catalog.csv', hdr, rows)
    ok('veto pairs read band/stokes by name, sorted as sort -u', S.veto_pairs(U) == [
        ('18MHz', 'V'), ('41MHz', 'I'), ('Full', 'I'), ('Red', 'V'), ('Weird', 'I')], S.veto_pairs(U))
    wcsv(f'{U}/master_catalog.csv', ['cand_id', 'lst', 'date', 'stokes', 'band'], [['a', '14h', 'd', 'V', 'Red']])
    try:
        S.veto_pairs(U)
        r = False
    except Refusal:
        r = True
    ok('veto pairs refuse when columns 4-5 are not band, stokes', r)
    # compare copy + byte check
    for p, txt in (('a.csv', 'x\n1\n'), ('catalogs.log', 'log\n'), ('pertie/pertie_detections.csv', 'cand_id\nV1\n'),
                   ('source_catalogs_noveto_V_sb/18MHz_sources_survivors.csv', 'source_id\nJ1\n')):
        os.makedirs(os.path.dirname(f'{U}/{p}'), exist_ok=True)
        open(f'{U}/{p}', 'w').write(txt)
    S.compare_copy(U, f'{T}/cmp/catalog')
    ok('compare copy: small products and the window-tie tables whole',
       S.files_equal(f'{U}/pertie/pertie_detections.csv', f'{T}/cmp/catalog/pertie/pertie_detections.csv')
       and os.path.exists(f'{T}/cmp/catalog/source_catalogs_noveto_V_sb/18MHz_sources_survivors.csv'))
    from exopipe.phase3.lib.catalog import tests_sublst as TS
    ok('compare_copy content test passes against its source', TS.compare_copy_equal(f'{T}/cmp/catalog', source=U)[0])
    open(f'{T}/cmp/catalog/a.csv', 'w').write('changed\n')
    ok('compare_copy content test fails on a changed file', not TS.compare_copy_equal(f'{T}/cmp/catalog', source=U)[0])


def test_ported_sublst_floor_fitted():
    """exopipe_ops 24d2b39: the port of the PYE2 heredoc writes the bytes production's own heredoc (read from the live
    sublst_catalogs.sh) writes, on a catalogue that repeats column names as master_catalog.csv does after the fit."""
    from exopipe.phase3.lib.catalog import tier_sublst as S, tests_sublst as TS
    src = open(f'{OPS_REAL}/sublst_catalogs.sh').read()
    a, b = src.find("<<'PYE2'"), src.find('\nPYE2\n')
    ok('production sublst_catalogs.sh holds the PYE2 heredoc (24d2b39) - if not, the port is stale', 0 < a < b)
    heredoc = src[src.index('\n', a) + 1:b + 1]
    hdr = ['cand_id', 'lst', 'date', 'band', 'stokes', 'ra', 'dec', 'axis_flag', 'size_flag', 'peak_ra', 'peak_dec',
           'axis_flag', 'size_flag', 'pos_kind']
    rows = [['I1', '14h', '2025-04-06T00', '41MHz', 'I', '210.0', '37.0', 'a1', 's1', '210.0', '37.0', 'a2', 's2', 'gaussian'],
            ['I2', '14h', '2025-04-06T02', 'Full', 'I', '30.0', '-60.0', 'a1', 's1', '30.1', '-59.9', 'a2', 's2', 'template'],
            ['V1', '14h', '2025-04-06T01', '18MHz', 'V', '211.0', '38.0', '', 's1', '211.0', '38.0', 'x', '', 'peak']]
    for d in ('fa', 'fb'):
        wcsv(f'{T}/{d}/master_catalog.csv', hdr, rows)
    p = subprocess.run([PY, '-', f'{T}/fa'], input=heredoc, universal_newlines=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
    line = S.elevation_floor_fitted(f'{T}/fb', OPS_REAL)
    names = ('master_catalog.csv', 'master_catalog_elevation_floor_fitted.csv')
    same = p.returncode == 0 and all(open(f'{T}/fa/{n}', 'rb').read() == open(f'{T}/fb/{n}', 'rb').read() for n in names)
    with open(f'{T}/fb/master_catalog_elevation_floor_fitted.csv', newline='') as f:
        gone = list(csv.reader(f))
    with open(f'{T}/fb/master_catalog.csv', newline='') as f:
        kept_hdr = next(csv.reader(f))
    ok('fitted-position floor: the port writes the bytes of production\'s heredoc; repeated columns kept; I2 removed',
       same and line in p.stdout and kept_hdr == hdr and [r[0] for r in gone[1:]] == ['I2'], (p.returncode, p.stdout[-300:]))
    r = TS.tier_rules_recomputed(f'{T}/fb/master_catalog_elevation_floor_fitted.csv')
    ok('tier_rule_removals passes on the fitted-floor removals (below the floor, elevation written)', r[0] and r[1] == 1, r)
    gone[1][-1] = '31.00'
    wcsv(f'{T}/fc/master_catalog_elevation_floor_fitted.csv', gone[0], gone[1:])
    ok('tier_rule_removals fails on a wrong elevation value', not TS.tier_rules_recomputed(
        f'{T}/fc/master_catalog_elevation_floor_fitted.csv')[0])


def test_ported_snapshot():
    from exopipe.phase3.lib.catalog import tier_snapshot as P
    from exopipe.phase3.config import CFG
    d = f'{T}/snap'
    os.makedirs(d, exist_ok=True)
    open(f'{d}/cells.txt', 'w').write('# the cells\n14h 2025-03-09\n20h_2025-05-03\n\n14h_2025-03-09\n  \n')
    ok('cells file parsed as the awk rule (LST DATE, comments, dedupe)',
       P.parse_cells_file(f'{d}/cells.txt') == ['14h_2025-03-09', '20h_2025-05-03'])
    open(f'{d}/f1', 'w').write('one\n')
    open(f'{d}/f2', 'w').write('two\n')
    open(f'{d}/code.sha256', 'w').write('%s  %s\n%s  %s\n' % (P.sha256_file(f'{d}/f1'), f'{d}/f1', '0' * 64, f'{d}/f2'))
    bad = P.verify_code_sha256(f'{d}/code.sha256')
    ok('tested-code sha256 verification flags the changed file only', len(bad) == 1 and 'f2' in bad[0], bad)
    wcsv(f'{d}/elig.csv', ['cell', 'status'], [['a', 'TO-PREFIT'], ['b', 'DIRTY-I'], ['c', 'WRITER-ACTIVE']])
    ok('eligibility filter', P.eligibility_bad(f'{d}/elig.csv', CFG.catalog_snapshot.ELIGIBLE_STATUSES) == ['b DIRTY-I'])
    for c_, sb, ri in (('c1', '18MHz', {'frame_time_1min': 'B'}), ('c1', '23MHz', {'frame_time_1min': 'old'}),
                       ('c1', 'Red', {'wideband_pairing': 1}), ('c2', 'Green', {})):
        os.makedirs(f'{d}/fleet/{c_}', exist_ok=True)
        json.dump(ri, open(f'{d}/fleet/{c_}/runinfo_{sb}_median.json', 'w'))
    ok('frame-time marks (n, pre-fix 1-min, cells, pre-fix wideband, cells)',
       P.frame_time_marks(f'{d}/fleet', ['c1', 'c2'], 'B') == (4, 1, 1, 1, 1))
    H = 'cell,subband,stokes,tier,ts,ra_deg,dec_deg,snr,fit_status,image'
    open(f'{d}/b1.csv', 'w').write(H + '\nc0,18MHz,I,10s,t0,1,2,8,ok,raw+flow\n')
    open(f'{d}/fits_c1.csv', 'w').write(H + '\nc1,Red,I,10s,t1,1,2,8,ok,identity screen x\n')
    open(f'{d}/fits_c1_template.csv', 'w').write('k\nT1\n')
    open(f'{d}/fits_c2.csv', 'w').write(H + '\nc2,Blue,V,10s,t2,1,2,8,ok,dewarped twice\n')
    ok('unmarked wideband rows counted', P.unmarked_wideband_rows(f'{d}/fits_c2.csv') == 1
       and P.unmarked_wideband_rows(f'{d}/fits_c1.csv') == 0)
    n, nt = P.combine_stores(f'{d}/b1.csv', [f'{d}/fits_c1.csv', f'{d}/fits_c2.csv'], f'{d}/S.csv')
    ok('combined store = batch-1 + cell rows; template created by the first copy',
       (n, nt) == (3, 1) and open(f'{d}/S_template.csv').read() == 'k\nT1\n', (n, nt))
    open(f'{d}/fits_bad.csv', 'w').write('other,header\nx,y\n')
    try:
        P.combine_stores(f'{d}/b1.csv', [f'{d}/fits_bad.csv'], f'{d}/S2.csv')
        r = False
    except ValueError:
        r = True
    ok('combined store refuses a store whose header differs', r)
    for c_, nfs, nl in (('c1', 40000, 100), ('c2', 10, 7)):
        os.makedirs(f'{d}/pf/{c_}', exist_ok=True)
        json.dump({'prefit': {'n_at_per_frame_fit_step': nfs}, 'n_loaded_%s' % c_: nl}, open(f'{d}/pf/{c_}/chain_summary.json', 'w'))
    ndet, nload, per = P.prefit_counts(f'{d}/pf')
    ok('prefit counts', (ndet, nload, per) == (40010, 107, {'c1': 40000, 'c2': 10}))
    rule = CFG.catalog_snapshot.TIE_SHARDS
    ok('tie shards: >= 30,000 at the fit step -> 16, else 8', P.tie_shards(40000, rule) == 16 and P.tie_shards(10, rule) == 8)
    m = CFG.catalog_snapshot.MEM_MODEL
    # batch 2: 17.5 M loaded, 1.46 M at the fit step, 82 cells, LC cache 4000 -> 60 + 38.5 + 14.6 + 2460 x 0.031 = 189.4 -> 250
    ok('memory model: batch-2 numbers give the 250 GB floor', P.memory_need(m, 17.5e6, 1.46e6, 82, 4000) == 250)
    ok('memory model formula', P.memory_need(m, 100e6, 6e6, 82, 4000) == int(__import__('math').ceil(
        60 + 2.2 * 100 + 10 * 6 + min(4000, 30 * 82) * 0.031)))


# =====================================================================================================================
def test_content_tests():
    from exopipe.phase3.lib.catalog import tests_sublst as TS, tests_snapshot as TP
    U = f'{T}/ct'
    hdr = ['cand_id', 'lst', 'date', 'band', 'stokes', 'cadence', 'ra', 'dec', 'snr']
    wcsv(f'{U}/master_catalog.csv', hdr, [['V1', '14h', '2025-04-06T00', '18MHz', 'V', '10min', '200', '40', '6'],
                                          ['I1', '14h', '2025-04-06T02', '41MHz', 'I', '10min', '202', '42', '7']])
    wcsv(f'{U}/selected_V_sb.csv', hdr, [['V1'] + [''] * 8])
    wcsv(f'{U}/selected_I_sb.csv', hdr, [['I1'] + [''] * 8])
    pd = f'{U}/pertie/pertie_detections.csv'
    wcsv(pd, ['cand_id', 'tie_status', 'sys_arcmin', 'sys_source'], [['V1', 'tied', '1.2', 'per-case'],
                                                                     ['I1', 'untied-few', '2.0', 'fallback-p90 x']])
    ok('window_tie passes on full coverage', TS.window_tie_rows(pd)[0])
    wcsv(pd, ['cand_id', 'tie_status', 'sys_arcmin', 'sys_source'], [['V1', 'tied', '1.2', 'per-case']])
    r = TS.window_tie_rows(pd)
    ok('window_tie fails when a selected detection has no row', not r[0], r[2])
    vt = f'{U}/leakveto/leakveto_14h_18MHz_V.csv'
    vh = ['cand_id', 'lst', 'band', 'stokes', 'elevation_deg', 'master_jy', 'affected_side', 'veto']
    wcsv(vt, vh, [['V1', '14h', '18MHz', 'V', '20', '1.0', '1', 'elevation']])
    ok('leak_veto_table passes (rows = conformant candidates, rule kept)', TS.leak_veto_rows(vt)[0], TS.leak_veto_rows(vt)[2])
    wcsv(vt, vh, [['V1', '14h', '18MHz', 'V', '20', '1.0', '1', '']])
    ok('leak_veto_table fails on a clear case not vetoed', not TS.leak_veto_rows(vt)[0])
    # exopipe_ops 7119fad: the Master floor on the deep flux scale, restore-only - each row's master_floor_jy = max(0.5, 0.5 x
    # window_flux_factor); a Master between 0.5 Jy and the row's floor is no clear case; an elevation veto below the floor is off
    # the rule; a floor that is not max(0.5, 0.5 F) (or below 0.5 where F < 1) fails
    vh2 = vh + ['window_flux_factor', 'master_floor_jy']
    for label, row, want in (('a Master below the scaled floor is kept', ['0.8', '1', '', '2.0000', '1.000'], True),
                             ('an elevation veto at the scaled floor', ['1.0', '1', 'elevation', '2.0000', '1.000'], True),
                             ('a clear case above the scaled floor not vetoed', ['1.2', '1', '', '2.0000', '1.000'], False),
                             ('an elevation veto below the scaled floor', ['0.8', '1', 'elevation', '2.0000', '1.000'], False),
                             ('a floor that is not max(0.5, 0.5 F)', ['0.8', '1', '', '2.0000', '0.500'], False),
                             ('F < 1: the floor stays 0.5 (restore-only)', ['0.45', '1', '', '0.8000', '0.500'], True),
                             ('F < 1: a floor below 0.5', ['0.45', '1', '', '0.8000', '0.400'], False),
                             ('F < 1: a clear case above 0.5 not vetoed', ['0.6', '1', '', '0.8000', '0.500'], False)):
        wcsv(vt, vh2, [['V1', '14h', '18MHz', 'V', '20', row[0], row[1], row[2], row[3], row[4]]])
        r = TS.leak_veto_rows(vt)
        ok(f'leak_veto_table (7119fad floor): {label} - {"passes" if want else "fails"}', r[0] == want, r[2])
    import numpy as np
    rh = ['veto', 'cell', 'subband', 'stokes', 'tier', 'ts', 'ra_deg', 'dec_deg', 'snr', 'pos_err_arcsec', 'cand_id', 'detail']
    rj = f'{U}/rejected.csv'
    offs = []
    with open(rj, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(rh)
        for i in range(3):
            offs.append(f.tell())
            w.writerow(['size', '14h_18MHz', '18MHz', 'V', '10min', '20250406_085331', '1', '2', '5', '60', f'R{i}', 'x'])
    np.savez_compressed(f'{U}/rejected_index.npz', ra=np.ones(3), dec=np.ones(3), err=np.ones(3), t0=np.ones(3),
                        dur=np.ones(3), band=np.zeros(3, np.int16), band_names=np.array(['18MHz']), offset=np.array(offs))
    wcsv(f'{U}/detection_ts.csv', ['cand_id', 'ts', 'tier'], [['V1', 't', '10min'], ['I1', 't', '10min']])
    ok('rejected_record passes (times, index offsets, served detections timed)', TS.rejected_record_consistent(rj)[0],
       TS.rejected_record_consistent(rj)[2])
    wcsv(f'{U}/detection_ts.csv', ['cand_id', 'ts', 'tier'], [['V1', 't', '10min']])
    ok('rejected_record fails when a served detection has no detection_ts', not TS.rejected_record_consistent(rj)[0])
    wcsv(f'{U}/master_catalog_removed.csv', ['cand_id', 'removal_reason', 'conform_note'],
         [['x', 'table: [el35-50] parent S=1 (image-pixel)', '']])
    ok('conformant_catalog sublst_table_notes passes / n.a. on a deep product',
       TS.sublst_table_notes(f'{U}/master_catalog.csv')[0] and TS.sublst_table_notes(f'{U}/master_catalog.csv', tier='deep')[1] == 'n/a')
    ok('fitted_catalog sublst test fails without pos_kind (catalogued before 70003ad)',
       not TS.sublst_fitted_positions(f'{U}/master_catalog.csv')[0])
    from exopipe.phase3.core import exclusions
    ex = exclusions.load()
    lst, sb, date = sorted(k for k in ex.rows if k[1].isdigit())[0]          # a real excluded (LST, subband, night)
    wcsv(f'{U}/mc_ok/master_catalog.csv', hdr, [['V1', lst, '2030-01-01T00', sb + 'MHz', 'V', '10min', '1', '1', '6']])
    wcsv(f'{U}/mc_bad/master_catalog.csv', hdr, [['V1', lst, date + 'T00', sb + 'MHz', 'V', '10min', '1', '1', '6']])
    ok('merged_record sublst: passes without excluded epochs, fails on a row of an excluded (LST, band, night)',
       TS.sublst_no_excluded_epoch(f'{U}/mc_ok/master_catalog.csv')[0]
       and not TS.sublst_no_excluded_epoch(f'{U}/mc_bad/master_catalog.csv')[0], (lst, sb, date))
    rep = f'{U}/preflight.json'
    json.dump({'plan': ['a'], 'binding': [{'key': 'k', 'status': 'ok'}]}, open(rep, 'w'))
    ok('catalog_preflight passes', TS.preflight_report(rep)[0])
    json.dump({'plan': ['a'], 'binding': [{'key': 'k', 'status': 'MISMATCH'}]}, open(rep, 'w'))
    ok('catalog_preflight fails on a binding mismatch', not TS.preflight_report(rep)[0])
    run = f'{T}/snaprun'
    write_snapshot_run(run, ['14h_2025-03-09'])
    r = TP.snapshot_run_content(run)
    ok('snapshot_run passes on a conformant synthetic run', r[0], r[2])
    # R3 (exopipe_ops 56810b9): a removed ghost leaks only when a served row of its cand_id lies within 1 deg of it - a cand_id
    # names a frame, so another source of the ghost's frame is not a leak; an unreadable served position is
    g3 = f'{T}/snaprun_r3'
    write_snapshot_run(g3, ['14h_2025-03-09'])
    S3 = json.load(open(f'{g3}/chain_summary.json'))
    S3['domain_dropped'] = {'ghost': 1}
    json.dump(S3, open(f'{g3}/chain_summary.json', 'w'))
    wcsv(f'{g3}/ghost_removed.csv', ['cand_id', 'ra_deg', 'dec_deg'], [['c1', '40.0', '0.0']])
    det = f'{g3}/candidates/Red/rank0001_Red_J0000+0000/detections.csv'
    hdr3 = ['cand_id', 'ra', 'dec', 'pos_err_stat_arcsec', 'pos_err_arcsec']
    res3 = {}
    for label, pos in (('another source of the frame', ('10.0', '0.0')), ('the same detection', ('40.3', '0.2')),
                       ('an unreadable position', ('', ''))):
        wcsv(det, hdr3, [['c1', pos[0], pos[1], '30', '60']])
        res3[label] = dict((n, (p_, d_)) for n, p_, d_ in TP.run_checks(g3))['R3']
    ok('snapshot_run R3: a served row of the removed cand_id 30 deg away is not a leak; within 1 deg, or unreadable, is',
       res3['another source of the frame'][0] and not res3['the same detection'][0]
       and not res3['an unreadable position'][0], res3)
    snapshot_run_card2_restore(TP)
    # per-tier pages (clv15 82792d5): a union identity that qualifies in no tier (served_tiers.csv) is not viewed - allowed;
    # any other unviewed union row fails
    nt = f'{T}/snaprun_notier'
    write_snapshot_run(nt, ['14h_2025-03-09'])
    with open(f'{nt}/source_catalogs/Red_sources_survivors.csv', 'a') as f:
        f.write('J0001+0001,V,11.0\n')
    stc = ['identity_jname', 'chain_outcome', 'tier', 'qualifies']
    wcsv(f'{nt}/served_tiers.csv', stc, [['J0000+0000', 'served', '10s', '1'], ['J0001+0001', 'served', '10s', '0'],
                                          ['J0001+0001', 'served', '1min', '0']])
    v_ = dict((n, (p_, d_)) for n, p_, d_ in TP.run_checks(nt))['views']
    wcsv(f'{nt}/served_tiers.csv', stc, [['J0000+0000', 'served', '10s', '1'], ['J0001+0001', 'served', '10s', '0'],
                                          ['J0001+0001', 'served', '1min', '1']])
    v2_ = dict((n, (p_, d_)) for n, p_, d_ in TP.run_checks(nt))['views']
    ok('snapshot_run views: a union identity that qualifies in no tier is not viewed (allowed, counted); one that qualifies '
       'in a tier must be', v_[0] and 'qualify in no tier' in v_[1] and not v2_[0], (v_, v2_))
    with open(f'{run}/source_catalogs/Red_sources_survivors.csv', 'a') as f:
        f.write('J9999+0000,V,-5.0\n')
    r = TP.snapshot_run_content(run)
    ok('snapshot_run fails on an RA outside [0, 360) and an unviewed union row', not r[0] and 'R7 FAIL' in r[2], r[2][-300:])
    tc = f'{T}/tc/tie_14h_2025-03-09.csv'
    wcsv(tc, ['cell', 'sb', 'st', 'tier', 'ts', 'ra_pre', 'dec_pre', 'tie_status', 'ra', 'dec', 'errmodel'],
         [['14h_2025-03-09', '18MHz', 'I', '10s', 't', '1.00000', '2.00000', 'tied', '1.01', '2.01', 'M2']])
    ok('tie_cache passes with rows of the installed model', TP.tie_cache_rows(tc, errmodel='M2')[0])
    ok('tie_cache fails without rows of the installed model', not TP.tie_cache_rows(tc, errmodel='M3')[0])


def write_record(path, cols, rows):
    import gzip
    with gzip.open(path, 'wt', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)


def snapshot_run_card2_restore(TP):
    """snapshot_run's card2 and restore checks (clv15 1279bd4 Card 2 in the chain; b8d1989 + bf7dce8 the restore-only 3-D gate),
    on synthetic runs: each passes on a consistent run and fails on each inconsistency it reads back."""
    def checks(run):
        return dict((n, (p_, d_)) for n, p_, d_ in TP.run_checks(run))
    run = f'{T}/snaprun_c2'
    write_snapshot_run(run, ['14h_2025-03-09'])
    c = checks(run)
    ok('snapshot_run card2 / restore: a run of a chain before both passes (no summary keys, no files, no record)',
       c['card2'][0] and c['restore'][0], (c['card2'], c['restore']))
    S = json.load(open(f'{run}/chain_summary.json'))
    S['card2'] = dict(ratio=0.05, tiers=['10s'], evaluated=2, not_evaluated=1, removed=1, v_10s_rows=3)
    json.dump(S, open(f'{run}/chain_summary.json', 'w'))
    c2h = ['cand_id', 'cell', 'band', 'ts', 'snr', 'peak_jyb', 'ra', 'dec', 'i_master_pix_jyb', 'scale', 'scale_source', 'i_here_jy',
           'v_over_i_here']
    c2row = ['14h_2025-03-09_18MHz_V_10s_20250309_101010', '14h_2025-03-09', '18MHz', '20250309_101010', '-9.000', '-0.5', '10.00000',
             '20.00000', '20', '1.0000', 'lc', '20', '0.02500']
    wcsv(f'{run}/card2_removed.csv', c2h, [c2row])
    rh = ['det_uid', 'cand_id', 'ra_peak', 'dec_peak', 'served', 'first_removed_by', 'gate_3d', 'gate_3d_installed', 'gate_3d_restore',
          'restored_by']
    rec = [['u1', c2row[0], '10.00000', '20.00000', '0', 'card2', '', '', '', ''],
           ['u2', 'c1', '10.00000', '0.00000', '1', '', 'pass', '', '', '']]
    write_record(f'{run}/detection_record.csv.gz', rh, rec)
    c = checks(run)
    ok('snapshot_run card2 passes: card2_removed.csv = the summary, |V|/I under the ratio, the record agrees', c['card2'][0],
       c['card2'][1])
    for label, edit in (('a row above the ratio', lambda r, s_, rr: r.__setitem__(12, '0.06000')),
                        ('a Stokes I row', lambda r, s_, rr: r.__setitem__(0, r[0].replace('_V_', '_I_'))),
                        ('counts that do not add up', lambda r, s_, rr: s_['card2'].__setitem__('not_evaluated', 2)),
                        ('the record off the file', lambda r, s_, rr: rr[0].__setitem__(5, 'seed')),
                        ('a card2 detection served', lambda r, s_, rr: rr[0].__setitem__(4, '1'))):
        r_, s_, rr = list(c2row), json.loads(json.dumps(S)), [list(x) for x in rec]
        edit(r_, s_, rr)
        wcsv(f'{run}/card2_removed.csv', c2h, [r_])
        json.dump(s_, open(f'{run}/chain_summary.json', 'w'))
        write_record(f'{run}/detection_record.csv.gz', rh, rr)
        c = checks(run)
        ok(f'snapshot_run card2 fails on {label}', not c['card2'][0], c['card2'][1])
    wcsv(f'{run}/card2_removed.csv', c2h, [c2row])
    json.dump(S, open(f'{run}/chain_summary.json', 'w'))
    # the restore-only gate: a restored detection (installed pass removes, restore pass keeps) tagged, served, in detections.csv
    RB = 'restore-only 3-D gate 20261104'
    rec_ro = rec + [['u3', 'c3', '11.00000', '0.50000', '1', '', 'pass', 'remove', 'pass', RB],
                    ['u4', 'c4', '12.00000', '0.50000', '0', 'bright-parent-3d', 'remove', 'remove', 'remove', '']]
    write_record(f'{run}/detection_record.csv.gz', rh, rec_ro)
    c = checks(run)
    ok('snapshot_run restore fails on a tagged record without the summary key (no restore table in the run)', not c['restore'][0],
       c['restore'][1])
    S['gate3d_restore_only'] = dict(table='t.json', removed_installed_pass=2, removed_restore_pass=1, removed=1, restored=1,
                                    restore_pass_only=0)
    json.dump(S, open(f'{run}/chain_summary.json', 'w'))
    det = f'{run}/candidates/Red/rank0001_Red_J0000+0000/detections.csv'
    dh = ['cand_id', 'ra', 'ra_peak', 'dec_peak', 'pos_err_stat_arcsec', 'pos_err_arcsec', 'restored_by']
    wcsv(det, dh, [['c1', '10.0', '10.00000', '0.00000', '30', '60', ''], ['c3', '11.0', '11.00000', '0.50000', '30', '60', RB]])
    c = checks(run)
    ok('snapshot_run restore passes: tags = the pair, = the summary\'s restored, served tagged rows = the record\'s',
       c['restore'][0] and c['card2'][0], (c['restore'][1], c['card2'][1]))
    for label, do in (
            ('the summary\'s restored off the record', lambda: S['gate3d_restore_only'].__setitem__('restored', 2)),
            ('a summary that does not add up', lambda: S['gate3d_restore_only'].__setitem__('removed_installed_pass', 3)),
            ('a tag without the pair', lambda: rec_ro[2].__setitem__(8, 'remove')),
            ('the pair without a tag', lambda: rec_ro[2].__setitem__(9, '')),
            ('a served tagged row missing from detections.csv', lambda: wcsv(det, dh, [['c1', '10.0', '10.00000', '0.00000', '30',
                                                                                         '60', '']])),
            ('a detections.csv without the column', lambda: wcsv(det, dh[:-1], [['c1', '10.0', '10.00000', '0.00000', '30', '60']]))):
        S0, rec0 = json.loads(json.dumps(S)), [list(x) for x in rec_ro]
        do()
        json.dump(S, open(f'{run}/chain_summary.json', 'w'))
        write_record(f'{run}/detection_record.csv.gz', rh, rec_ro)
        c = checks(run)
        ok(f'snapshot_run restore fails on {label}', not c['restore'][0], c['restore'][1])
        S.clear()
        S.update(S0)
        rec_ro[:] = rec0
        wcsv(det, dh, [['c1', '10.0', '10.00000', '0.00000', '30', '60', ''], ['c3', '11.0', '11.00000', '0.50000', '30', '60', RB]])


def test_restore_marker():
    """exopipe_ops fa90670: the approval marker's CURVE_CONSTANTS_RESTORE=<table> line (the first, up to a space) passes
    --curve-constants-restore <table> to every chain call (chainargs); none without the line."""
    import types
    from exopipe.phase3.lib.catalog import tier_snapshot as P
    from exopipe.phase3.config import CFG
    ok('restore table: the first CURVE_CONSTANTS_RESTORE= line, up to a space; none when empty or absent',
       P.restore_table(['x', 'CURVE_CONSTANTS_RESTORE=/a/b.json extra', 'CURVE_CONSTANTS_RESTORE=/c']) == '/a/b.json'
       and P.restore_table(['CURVE_CONSTANTS_RESTORE=']) is None and P.restore_table(None) is None
       and P.restore_table([' CURVE_CONSTANTS_RESTORE=/x']) is None)
    ns = types.SimpleNamespace(c=CFG.catalog_snapshot, cfg=CFG, S='/s/fits_store.csv', chain_py='/nonexistent/chain.py', T=None,
                               analysis=[], restore='/t/restore.json')
    a = P.Ctx.chainargs(ns)
    i = a.index('--curve-constants-restore') if '--curve-constants-restore' in a else -1
    ns.restore = None
    ok('chainargs: --curve-constants-restore <table> right after production\'s CHAINARGS; absent without a table',
       i == 10 and a[i + 1] == '/t/restore.json' and '--curve-constants-restore' not in P.Ctx.chainargs(ns), a[:12])


def write_snapshot_run(run, cells):
    os.makedirs(f'{run}/candidates/Red/rank0001_Red_J0000+0000', exist_ok=True)
    json.dump({'cells': cells, 'prefit': {'n_without_fit': 0, 'n_at_per_frame_fit_step': 3}, 'n_served_with_unfitted_frames': 0,
               'n_unfitted_frames_under_served': 0, 'domain_dropped': {'ghost': 0}, 'tie_cache': {'from_cache': 3, 'tied_now': 0},
               'gate3d_match_percentile': 95.0, 'member_order': 'time, subband, Stokes', 'ghost_gate': {'dates': ['2024-12-28']},
               'n_served': 1}, open(f'{run}/chain_summary.json', 'w'))
    wcsv(f'{run}/ghost_removed.csv', ['cand_id'], [])
    wcsv(f'{run}/candidates/Red/rank0001_Red_J0000+0000/detections.csv', ['cand_id', 'ra', 'pos_err_stat_arcsec', 'pos_err_arcsec'],
         [['c1', '10.0', '30', '60']])
    wcsv(f'{run}/source_catalogs/Red_sources_survivors.csv', ['source_id', 'stokes', 'ra'], [['J0000+0000', 'V', '10.0']])
    wcsv(f'{run}/source_catalogs_V/18MHz_sources_survivors.csv', ['source_id', 'identity_id', 'ra'],
         [['J0000+0000_18MHz', 'J0000+0000', '10.0']])


# =====================================================================================================================
# end-to-end on stub in-place modules
# =====================================================================================================================

STUB_TIER_RULES = '''
DEEP_REFUSALS = "stub pin (md5 x)"
def deep_search_refused(lst, date, band, stokes):
    return "beam" if (band == "23MHz" and stokes == "V") else ""
def flux_scale_failed(lst, date, band):
    return "degenerate-flatDC qa3_scale 5" if date == "2025-04-07" else ""
'''
STUB_GATE_INPUTS = 'exec(compile(open("/lustre/gh/main/exopipe_ops/gate_inputs.py").read(), "gate_inputs_real", "exec"))\n'
STUB_MEMGATE = '#!/bin/bash\necho "$(date -u) MEM-GATE $3: pass on $(hostname -s) (stub, need $1)"\nexit 0\n'
SNAP_STUBS = {
    'clv15/phase3_snapshot_tiecache.py': 'def _errmodel():\n    return "stat0.42floor5_newmodel"\n',
    'clv15/phase3_snapshot_search.py': 'TIER1MIN_TIME_BASIS = "integration-midpoint-dateobs-v1"\n',
    'clv15/phase3_snapshot_fit.py': r'''
import csv, sys
def _group_order(r):
    return (0, 0, False, r["subband"], r["stokes"], r["ts"])
if __name__ == "__main__":
    a = sys.argv[1:]
    pend, store = a[a.index("--pending") + 1], a[a.index("--store") + 1]
    rows = list(csv.DictReader(open(pend)))
    with open(store, "w") as f:
        f.write("cell,subband,stokes,tier,ts,ra_deg,dec_deg,snr,fit_status,image\n")
        for r in rows:
            f.write("%s,%s,%s,%s,%s,%s,%s,%s,ok,raw+flow\n" % (r["cell"], r["subband"], r["stokes"], r["tier"], r["ts"],
                                                            r["ra_deg"], r["dec_deg"], r["snr"]))
    print("fit-grouped DONE: %d" % len(rows))
''',
    'clv15/phase3_snapshot_chain.py': r'''
import csv, json, os, sys
a = sys.argv[1:]
def opt(k, n=1):
    i = a.index(k); return a[i + 1] if n == 1 else a[i + 1:]
out, store = opt("--out"), opt("--fit-store")
cells = []
i = a.index("--cells") + 1
while i < len(a) and not a[i].startswith("--"):
    cells.append(a[i]); i += 1
os.makedirs(out, exist_ok=True)
fitted = set()
if os.path.exists(store):
    for r in csv.DictReader(open(store)):
        if r["fit_status"] == "ok":
            fitted.add(r["cell"])
if "--prefit-list-only" in a:
    req = [c for c in cells if c not in fitted]
    with open(os.path.join(out, "prefit_requests.csv"), "w") as f:
        f.write("cell,subband,stokes,tier,ts,ra_deg,dec_deg,snr\n")
        for c in req:
            f.write("%s,18MHz,I,10s,20250309_100000,1.0,2.0,9.0\n" % c)
    json.dump(dict([("prefit", {"n_at_per_frame_fit_step": 1, "n_without_fit": len(req)})] +
                   [("n_loaded_%s" % c, 10) for c in cells]), open(os.path.join(out, "chain_summary.json"), "w"))
    print("PREFIT-LIST-ONLY: %d to fit" % len(req)); sys.exit(0)
if "--stop-after-tie" in a:
    d = os.environ["P3_TIE_CACHE_DIR"]; os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "tie_%s.csv" % cells[0]), "a") as f:
        f.write("cell,sb,st,tier,ts,ra_pre,dec_pre,tie_status,ra,dec,errmodel\n")
        f.write("%s,18MHz,I,10s,t,1.00000,2.00000,tied,1.01,2.01,stat0.42floor5_newmodel\n" % cells[0])
    print("tie cache: 3 detections from %s, 1 tied now" % d)
    json.dump({"prefit": {"n_without_fit": 0, "n_at_per_frame_fit_step": 1}, "tie_cache": {"from_cache": 3, "tied_now": 1}},
              open(os.path.join(out, "chain_summary.json"), "w"))
    print("STOP-AFTER-TIE"); sys.exit(0)
assert os.environ.get("P3_VETO_PROCS") == "24" and os.environ.get("P3_REFASTRO_SHARDS") == "24"
os.makedirs(os.path.join(out, "candidates/Red/rank0001_Red_J0000+0000"))
json.dump({"cells": cells, "prefit": {"n_without_fit": 0, "n_at_per_frame_fit_step": 3}, "n_served_with_unfitted_frames": 0,
           "n_unfitted_frames_under_served": 0, "domain_dropped": {"ghost": 0}, "tie_cache": {"from_cache": 3, "tied_now": 0},
           "gate3d_match_percentile": 95.0, "member_order": "time, subband, Stokes", "ghost_gate": {"dates": ["2024-12-28"]},
           "n_served": 1, "per_instant_vetoes": {"rule": "per instant (stub)"}}, open(os.path.join(out, "chain_summary.json"), "w"))
open(os.path.join(out, "ghost_removed.csv"), "w").write("cand_id\n")
# the veto records the chain writes since clv15 99fe3ea (content test snapshot_run.per_instant): none removed here
open(os.path.join(out, "vetoed_sources.csv"), "w").write("veto,cell,subband,stokes,tier,ts,ra_deg,dec_deg,snr,n_members,"
                                                          "n_instants,bands,routes,pos_err_arcsec,detail\n")
open(os.path.join(out, "vetoed_detections.csv"), "w").write("removed_by,cell,subband,stokes,tier,ts,ra_deg,dec_deg,snr,instant,"
                                                             "instant_kind,neg_member,asym_member\n")
open(os.path.join(out, "candidates/Red/rank0001_Red_J0000+0000/detections.csv"), "w").write(
    "cand_id,ra,pos_err_stat_arcsec,pos_err_arcsec\nc1,10.0,30,60\n")
os.makedirs(os.path.join(out, "source_catalogs")); os.makedirs(os.path.join(out, "source_catalogs_V"))
open(os.path.join(out, "source_catalogs/Red_sources_survivors.csv"), "w").write("source_id,stokes,ra\nJ0000+0000,V,10.0\n")
open(os.path.join(out, "source_catalogs_V/18MHz_sources_survivors.csv"), "w").write(
    "source_id,identity_id,ra\nJ0000+0000_18MHz,J0000+0000,10.0\n")
print("chain done")
''',
    'exopipe_ops/p3s3_astrom.py': 'def band_class(b):\n    return {"41MHz": "Green", "64MHz": "Blue"}.get(b, "Red")\n',
    'exopipe_ops/snapshot_batch_audit.py': r'''
import sys
a = sys.argv[1:]
cf, out = a[a.index("--cells-file") + 1], a[a.index("--out") + 1]
cells = []
for l in open(cf):
    f = l.split()
    if f and not l.startswith("#"):
        c = f[0] + "_" + f[1] if len(f) >= 2 else f[0]
        if c not in cells: cells.append(c)
open(out, "w").write("cell,status\n" + "".join("%s,TO-PREFIT\n" % c for c in cells))
print("audit stub %d cells" % len(cells))
''',
    'exopipe_ops/snapshot_store_repair.py': r'''
import sys
a = [x for x in sys.argv[1:] if not x.startswith("--")]
open(a[1], "w").write("cell,subband,stokes,tier,ts,ra_deg,dec_deg,snr\n")
print("REPAIR-LIST %s 0 0 0 0" % a[0])
''',
    'exopipe_ops/snapshot_store_ready.py': 'import sys\nfor c in sys.argv[1:]:\n    print("READY %s stub" % c)\n',
    'exopipe_ops/snapshot_store_append.py': r'''
import csv, os, sys
a = sys.argv[1:]
also = a[a.index("--also") + 1] if "--also" in a else None
F = a[a.index("--fits-dir") + 1]
srcs = [x for i, x in enumerate(a) if not x.startswith("--") and (i == 0 or a[i - 1] not in ("--also", "--fits-dir"))]
n = 0
for s in srcs:
    lines = open(s).read().splitlines()
    for l in lines[1:]:
        cell = l.split(",")[0]
        for p in [os.path.join(F, "fits_%s.csv" % cell)] + ([also] if also else []):
            new = not os.path.exists(p)
            with open(p, "a") as f:
                if new: f.write(lines[0] + "\n")
                f.write(l + "\n")
        n += 1
print("APPENDED %d rows, 0 template" % n)
''',
    'exopipe_ops/snapshot_rechain_content_test.py': 'print("R1 PASS stub")\nprint("B2R-RUN-CONTENT PASS")\n',
}


def make_site(stub_root):
    host = socket.gethostname()
    site = f'{T}/site.py'
    open(site, 'w').write(
        "NODE_POOLS = {'snapshot': [('localhost', 1)]}\nGPU_HOSTS = []\n"
        "PYTHON = {'default': '/opt/devel/pipeline/envs/py38_orca_nkosogor/bin/python3', "
        "'flow': '/opt/devel/pipeline/envs/py310_orca/bin/python3', 'py36': '/opt/devel/pipeline/envs/development/bin/python'}\n"
        f"SCRATCH_ROOT = {{{host!r}: '{T}/scratch'}}\nNODE_LOCAL_ROOTS = {{}}\n"
        f"LEGACY_CODE_ROOTS = ['{stub_root}/clv15', '{stub_root}/exopipe_ops']\n")
    return site


def build_stubs():
    st = f'{T}/stub'
    for d in ('clv15', 'exopipe_ops', 'exopipe/exopipe/common'):
        os.makedirs(f'{st}/{d}', exist_ok=True)
    open(f'{st}/clv15/phase3_tier_rules.py', 'w').write(STUB_TIER_RULES)
    open(f'{st}/exopipe_ops/gate_inputs.py', 'w').write(STUB_GATE_INPUTS)
    open(f'{st}/exopipe_ops/node_mem_gate.sh', 'w').write(STUB_MEMGATE)
    from exopipe.phase3.lib.catalog import tier_sublst as S
    open(f'{st}/exopipe_ops/sublst_catalogs.sh', 'w').write('#!/bin/bash\n  FITPOS_K="%s"\n' % ' '.join(S.k_arg()))
    for p in glob.glob(f'{OPS_REAL}/sublst_pertie_*.json'):
        shutil.copy(p, f'{st}/exopipe_ops/')
    json.dump({'tables': {'sublst_V_18MHz_el25-35': {'metric': 'image-pixel'}}},
              open(f'{st}/exopipe/exopipe/common/parent_curve_tables_sublst.json', 'w'))
    for rel, txt in SNAP_STUBS.items():
        open(f'{st}/{rel}', 'w').write(txt)
    return st


def test_e2e_sublst(site):
    import numpy as np
    from astropy.io import fits
    # production's p3s2_conform.py as the pending patch leaves it (its P3_CONFORM_AXIS switch, default on): under the ruled
    # catalog_conform.CONFORM_AXIS_UNDER_RULE 'replace' (R-REFACTOR-20261104-AXIS-RULE-ALONE-FULLBAND-SIZE-OFF-FROM-PULSARS) the
    # preflight of the rule's search jobs refuses at S-G7 while the script lacks it (sg13_size_rule tests both)
    open(f'{T}/stub/exopipe_ops/p3s2_conform.py', 'w').write(STUB_CONFORM_SUBLST.replace(
        'AXIS_STEP = True   # AXIS-SWITCH', "AXIS_STEP = os.environ.get('P3_CONFORM_AXIS', '1') != '0'   # AXIS-SWITCH"))
    sl, ops = f'{T}/sl', f'{T}/ops'
    os.makedirs(f'{sl}/tenmin_root_x2/14h_18MHz', exist_ok=True)
    wcsv(f'{sl}/tenmin_root_x2/14h_18MHz/frames.csv', ['key', 'stokes', 'bmaj_deg', 'fluxbake'], [['2025-04-06T00', 'I', '0.2', '1']])
    for sub in ('done', 'claims', 'failed'):
        os.makedirs(f'{ops}/sublst_search_state_B/{sub}', exist_ok=True)
    for i in range(35):
        open(f'{ops}/sublst_search_state_B/done/14h_job{i:02d}', 'w').close()
    m = f'{sl}/coadd_14h/14h_18MHz/coadd_final/I_deep_Taper_R075_18MHz.fits'
    os.makedirs(os.path.dirname(m))
    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    fits.writeto(m, data)
    os.makedirs(f'{sl}/parent_catalogs_14h')
    pc = f'{sl}/parent_catalogs_14h/master_R075_14h_18MHz.gaul.csv'
    open(pc, 'w').write('# gaul\n0,0,0,0,200.0,0,40.0,0,0,0,5.0\n')
    json.dump({'master': os.path.basename(m), 'master_md5_data': hashlib.md5(np.ascontiguousarray(data).tobytes()).hexdigest()},
              open(pc + '.gen.json', 'w'))
    U = f'{T}/out_sublst'
    # the conformance of this file is a stand-in script (STUB_CONFORM_SUBLST): the table-noise / pulsar-evidence record
    # (catalog_conform.TABLE_NOISE_RECORD, on for this tier) runs production's module through five exact-once text patches
    # and refuses any other text, so it is off here (a ruled key: this --out is not a production root); the record, the
    # frozen mode and the stage's wiring are tested against production's module in tests/test_catalog_conform_noise.py
    common = ['catalog', '--tier', 'sublst', '--lsts', '14h', '--out', U, '--cores', 'none',
              '--set', f'paths.SUBLST_ROOT={sl}', '--set', f'paths.OPS_ROOT={ops}',
              '--set', 'catalog_conform.TABLE_NOISE_RECORD[sublst]=False']
    rc, out = cli(common + ['--stages', 'preflight'], site)
    ok('sub-LST preflight refuses an incomplete search (35 of 36 done): exit 2, S-G1', rc == 2 and 'S-G1' in out, out[-400:])
    open(f'{ops}/sublst_search_state_B/done/14h_job35', 'w').close()
    # review 2026-10-13: as the production launcher (sublst_catalog_publish.sh:39, :44-47), no run log / no '4 tree B built'
    # line in it refuses; a search end after the tree-B line passes
    rc, out = cli(common + ['--stages', 'preflight'], site)
    ok('sub-LST preflight refuses without the run log (S-G1, sublst_catalog_publish.sh:44)', rc == 2 and 'run log' in out,
       out[-400:])
    open(f'{ops}/sublst_run_14h.log', 'w').write('Tue Oct 13 01:00:00 UTC 2026 3 stacks done\n')
    rc, out = cli(common + ['--stages', 'preflight'], site)
    ok("sub-LST preflight refuses a run log without '4 tree B built' (S-G1)", rc == 2 and 'tree B built' in out, out[-400:])
    open(f'{ops}/sublst_run_14h.log', 'a').write('Tue Oct 13 02:00:00 UTC 2026 4 tree B built\n'
                                                 'Tue Oct 13 03:00:00 UTC 2026 5 search complete\n')
    rc, out = cli(common + ['--stages', 'preflight'], site)
    rec = json.load(open(f'{U}/_records/catalog/14h_B.preflight.json'))
    ok('sub-LST preflight done (S-G1, S-G2 pcheck child on a synthetic master, tables, pinned md5, bindings)',
       rc == 0 and rec['status'] == 'done' and any('PARENT-CHECK 14h: 1 subbands, 0 refused' in (t.get('detail') or '')
                                                  for t in rec['tests']), out[-600:])
    ok('the pcheck child ran through core/legacy.run (recorded with its code fingerprint)',
       any('pcheck' in ' '.join(c['argv']) for c in rec['meta'].get('legacy_calls', [])))
    # S-G7, the size / axis rule of the search jobs (lib/search/size_rule; the 2026-11-03 review, finding 5): the jobs must share
    # ONE rule and it must be the CONFIGURED one (search_sublst.SIZE_RULE) - checked at the preflight on real job directories
    from exopipe.phase3.lib.search import size_rule as SZ, deep_search as DS
    srec = SZ.configured_record('10min')
    jobs = {}
    for job in ('14h_18MHz_I', '14h_18MHz_V'):
        jobs[job] = rd = f'{sl}/search_B/{job}/transients/Run_SLST14B'
        os.makedirs(rd, exist_ok=True)
        open(f'{rd}/v_deep_ledger.csv', 'w').write(','.join(DS.GLED_COLS + DS.GLED_COLS_SIZE) + '\n')
        json.dump(srec, open(f'{rd}/{SZ.RUN_RECORD}', 'w'))
    rc, out = cli(common + ['--stages', 'preflight'], site)
    rec = json.load(open(f'{U}/_records/catalog/14h_B.preflight.json'))
    g7 = [t for t in rec['tests'] if t['name'].endswith('S-G7.size-rule')]
    ok("S-G7: two search jobs searched under the configured rule (their size_rule.json): the preflight passes, its S-G7 check "
       "names the rule", rc == 0 and g7 and g7[0]['passed'] and g7[0].get('value') == 'envelope', (rc, g7, out[-400:]))
    os.remove(f"{jobs['14h_18MHz_V']}/{SZ.RUN_RECORD}")              # one job of production's: two rules among the jobs
    open(f"{jobs['14h_18MHz_V']}/v_deep_ledger.csv", 'w').write(','.join(DS.GLED_COLS) + '\n')
    rc, out = cli(common + ['--stages', 'preflight'], site)
    ok('S-G7: search jobs of two size rules refuse the preflight (exit 2)', rc == 2 and 'S-G7' in out and 'size rules' in out,
       out[-400:])
    os.remove(f"{jobs['14h_18MHz_I']}/{SZ.RUN_RECORD}")              # every job production's, the configuration the rule's
    open(f"{jobs['14h_18MHz_I']}/v_deep_ledger.csv", 'w').write(','.join(DS.GLED_COLS) + '\n')
    rc, out = cli(common + ['--stages', 'preflight'], site)
    ok("S-G7: production's search jobs under the configured rule refuse the preflight (as the deep check refuses), naming the "
       "configured rule", rc == 2 and 'S-G7' in out and 'the configuration applies envelope' in out, out[-400:])
    rc, out = cli(common + ['--stages', 'preflight', '--set', 'search_sublst.SIZE_RULE=production'], site)
    rec = json.load(open(f'{U}/_records/catalog/14h_B.preflight.json'))
    g7 = [t for t in rec['tests'] if t['name'].endswith('S-G7.size-rule')]
    ok("S-G7: the same jobs with the configuration set to production (the parity file's value): the preflight passes",
       rc == 0 and g7 and g7[0]['passed'] and g7[0].get('value') == 'production', (rc, g7, out[-400:]))
    for rd in jobs.values():                                         # back to the configured rule for the stages below
        open(f'{rd}/v_deep_ledger.csv', 'w').write(','.join(DS.GLED_COLS + DS.GLED_COLS_SIZE) + '\n')
        json.dump(srec, open(f'{rd}/{SZ.RUN_RECORD}', 'w'))
    rc, out = cli(common + ['--stages', 'preflight'], site)
    ok('S-G7: the jobs back under the configured rule: the preflight passes again', rc == 0, out[-400:])
    # a merge record (fabricated: the merge stage's classes are the catalog-deep unit's) and its stage copy
    from exopipe.phase3.core import record
    hdr = ['cand_id', 'lst', 'date', 'band', 'stokes', 'cadence', 'ra', 'dec', 'snr']
    rows = [['V1', '14h', '2025-04-06T00', '18MHz', 'V', '10min', '210.0', '37.0', '6'],      # kept
            ['V2', '14h', '2025-04-06T00', '23MHz', 'V', '10min', '210.0', '37.0', '6'],      # tier rule 3
            ['I1', '14h', '2025-04-07T01', '41MHz', 'I', '10min', '210.0', '37.0', '7'],      # tier rule 4
            ['I2', '14h', '2025-04-06T02', 'Full', 'I', '10min', '30.0', '-60.0', '8']]       # below 25 deg
    wcsv(f'{U}/master_catalog.csv', hdr, rows)
    sc = f'{U}/_records/catalog/14h_B/stage_merge'
    os.makedirs(sc, exist_ok=True)
    shutil.copy2(f'{U}/master_catalog.csv', sc)
    record.atomic_write_json(record.record_path(U, 'catalog', '14h_B.merge'), {
        'status': 'done', 'unit': '14h_B.merge',
        'meta': {'output_fingerprints': {f'{U}/master_catalog.csv': record.fingerprint(f'{U}/master_catalog.csv')}}})
    rc, out = cli(common + ['--stages', 'tier-rules'], site)
    kept = [r['cand_id'] for r in rcsv(f'{U}/master_catalog.csv')]
    gone = {n: [r['cand_id'] for r in rcsv(f'{U}/{n}')] for n in ('master_catalog_deepsearch_refused.csv',
                                                                     'master_catalog_fluxscale_refused.csv',
                                                                     'master_catalog_elevation_floor.csv')}
    rec = json.load(open(f'{U}/_records/catalog/14h_B.tier-rules.json'))
    ok('tier-rules: rule 3, rule 4 and the elevation floor remove exactly their rows; content tests pass',
       rc == 0 and kept == ['V1'] and gone == {'master_catalog_deepsearch_refused.csv': ['V2'],
                                               'master_catalog_fluxscale_refused.csv': ['I1'],
                                               'master_catalog_elevation_floor.csv': ['I2']}
       and rec['status'] == 'done' and all(t['passed'] for t in rec['tests']), (rc, kept, gone, out[-500:]))
    first = open(f'{U}/master_catalog_elevation_floor.csv', 'rb').read()
    open(f'{U}/master_catalog.csv', 'a').write('Z9,14h,x,y,z,10min,1,1,1\n')        # the in-place file changed since
    rc, out = cli(common + ['--from', 'tier-rules', '--until', 'tier-rules'], site)
    rec = json.load(open(f'{U}/_records/catalog/14h_B.tier-rules.json'))
    ok('--from tier-rules restores master_catalog.csv from the merge stage copy and reproduces the outputs',
       rc == 0 and any('restored' in n for n in rec['notes']) and open(f'{U}/master_catalog_elevation_floor.csv', 'rb').read()
       == first, out[-400:])
    open(f'{sc}/master_catalog.csv', 'a').write('tampered\n')
    open(f'{U}/master_catalog.csv', 'a').write('Z9,14h,x,y,z,10min,1,1,1\n')
    rc, out = cli(common + ['--from', 'tier-rules', '--until', 'tier-rules'], site)
    ok('--from refuses when the in-place file and its stage copy both changed (exit 2)', rc == 2, out[-300:])
    rc, out = cli(common + ['--stages', 'tier-rules', '--set', 'catalog_sublst.ELEV_FLOOR_DEG=20.0'], site)
    ok('an override of a [lit] key is refused by config (exit 2)', rc == 2 and 'PROTECTED' in out, out[-300:])
    # review 2026-10-13: a gap in --stages is checked like a --from (conform without a tie-pulsars record refuses)
    rc, out = cli(common + ['--stages', 'preflight,conform'], site)
    rec = json.load(open(f'{U}/_records/catalog/14h_B.conform.json'))
    ok('--stages with a gap (preflight,conform): conform refuses without the record of tie-pulsars (exit 2)',
       rc == 2 and rec['status'] == 'refused' and '14h_B.tie-pulsars' in (rec.get('reason') or ''), (rc, rec.get('reason')))
    sg13_size_rule(site, common)


# S-G13 under the size / axis rule: the conformance stage on the rule's search jobs, production's p3s2_conform.py stood in for by
# a stub that applies production's AXIS step (|S/N| > 8 and axis_norm > 2.0 x beam removed in its words, p3s2_conform.py:635-639)
# - with no switch, as production today, or with the pending patch's P3_CONFORM_AXIS switch
STUB_CONFORM_SUBLST = r'''
import csv, os, shutil, sys
AXIS_STEP = True   # AXIS-SWITCH
U = sys.argv[1]
rows = list(csv.DictReader(open(os.path.join(U, "master_catalog.csv"))))
cols = list(rows[0].keys()) if rows else ["cand_id"]
shutil.copy(os.path.join(U, "master_catalog.csv"), os.path.join(U, "master_catalog_asrun.csv"))
out, rem = [], []
for r in rows:
    r = dict(r, axis_flag="", fp_prediction=r.get("fp_prediction") or ("inconclusive" if r["stokes"] == "V" else ""))
    an, snr = float(r.get("axis_norm") or "nan"), abs(float(r["snr"]))
    if an == an and snr > 8.0 and an > 2.0:
        if AXIS_STEP:
            rem.append(dict(r, removal_reason="axis ratio %s x beam > 2.0 above 8.0 sigma (Gregg 2026-09-05)" % r["axis_norm"]))
            continue
        r["axis_flag"] = "axis > 2.0 x beam (kept: point-like at its S/N by the size / axis rule)"
    out.append(r)
oc = cols + [c for c in ("fp_prediction", "axis_flag") if c not in cols]
for name, rr, cc in (("master_catalog.csv", out, oc), ("master_catalog_removed.csv", rem, oc + ["removal_reason"])):
    with open(os.path.join(U, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cc, extrasaction="ignore"); w.writeheader(); w.writerows(rr)
print("conform: %d raw -> %d conformant | removed %d (stub)" % (len(rows), len(out), len(rem)))
'''


def sg13_size_rule(site, common):
    """The 10-minute conformance (S-G13) on search jobs of the size / axis rule (lib/search/size_rule; the 2026-11-03 review
    finding 1 and its verification), with production's axis step in the stand-in conformance: one Stokes I row at S/N 9 with an
    axis ratio of 2.4 x beam (kept by the rule: its limit there is above 3) among kept rows. 'replace' (the ruled default,
    R-REFACTOR-20261104-AXIS-RULE-ALONE-FULLBAND-SIZE-OFF-FROM-PULSARS) without the switch: refused before the script runs,
    naming the pending patch (S-G13 and the preflight's S-G7); 'second_gate' (production's behavior, the parity file's value)
    with production's script as it is: the stage completes, the row removed in production's words and recorded
    (S-G13.size-rule-axis-step, value 1), the content tests of the conformant catalog passing; 'replace' with the switch:
    P3_CONFORM_AXIS=0 reaches the script, the row kept and flagged, value 0; 'second_gate' with it: removed as today."""
    from exopipe.phase3.core import record
    from exopipe.phase3.lib.search import size_rule as SZ
    U = f'{T}/out_sublst_sg13'                                        # its own run directory: the stages above stay as they are
    common = [U if x == f'{T}/out_sublst' else x for x in common]
    assert U in common
    conform = f'{T}/stub/exopipe_ops/p3s2_conform.py'                 # build_stubs' tree, the site's LEGACY_CODE_ROOTS

    def stub(kind):
        sw = "AXIS_STEP = os.environ.get('P3_CONFORM_AXIS', '1') != '0'"
        line = {'today': 'AXIS_STEP = True', 'patched': sw}[kind] + '   # AXIS-SWITCH'
        open(conform, 'w').write(STUB_CONFORM_SUBLST.replace('AXIS_STEP = True   # AXIS-SWITCH', line))

    def setup(pre=(), conf=()):
        """A fresh run directory: the preflight (``pre``: its --set), the conformance's input (a fabricated tie-pulsars record),
        then the conformance stage alone (``conf``)."""
        shutil.rmtree(U, ignore_errors=True)
        rc, out = cli(common + ['--stages', 'preflight'] + list(pre), site)
        if rc != 0:
            return rc, out
        hdr = ['cand_id', 'lst', 'date', 'band', 'stokes', 'cadence', 'ra', 'dec', 'snr', 'peak_jy', 'axis_norm', 'size_beams']
        wcsv(f'{U}/master_catalog.csv', hdr, [['V1', '14h', '2025-04-06T00', '18MHz', 'V', '10min', '210.0', '37.0', '6', '0.5', '1.1',
                                               '1.0'],
                                              ['I3', '14h', '2025-04-06T01', '18MHz', 'I', '10min', '211.0', '38.0', '9', '2.0', '2.4',
                                               '1.6'],
                                              ['I4', '14h', '2025-04-06T02', '18MHz', 'I', '10min', '212.0', '39.0', '9.5', '2.1',
                                               '1.8', '1.2']])
        record.atomic_write_json(record.record_path(U, 'catalog', '14h_B.tie-pulsars'), {
            'status': 'done', 'unit': '14h_B.tie-pulsars',
            'meta': {'output_fingerprints': {f'{U}/master_catalog.csv': record.fingerprint(f'{U}/master_catalog.csv')}}})
        return cli(common + ['--from', 'conform', '--until', 'conform'] + list(conf), site)

    def conform_rec():
        p_ = f'{U}/_records/catalog/14h_B.conform.json'
        return json.load(open(p_)) if os.path.exists(p_) else {}
    SG = ['--set', 'catalog_conform.CONFORM_AXIS_UNDER_RULE=second_gate']   # production's behavior (a ruled key: this --out is
    stub('today')                                                            # not a production root)
    rc, out = setup(pre=SG)
    rec = conform_rec()
    ok("S-G13 under the size / axis rule, 'replace' (the ruled default) while production's script has no P3_CONFORM_AXIS "
       "switch: the stage refuses before the script runs (exit 2; no master_catalog_asrun.csv), naming the pending patch and "
       "the decision", rc == 2 and rec.get('status') == 'refused' and SZ.CONFORM_AXIS_PATCH in (rec.get('reason') or '')
       and SZ.PULSAR_DECISION in (rec.get('reason') or '') and not os.path.exists(f'{U}/master_catalog_asrun.csv'),
       (rc, rec.get('status'), (rec.get('reason') or '')[:300], out[-300:]))
    rc, out = cli(common + ['--stages', 'preflight'], site)
    rec = json.load(open(f'{U}/_records/catalog/14h_B.preflight.json'))
    ok("S-G7 under 'replace' without the switch: the preflight refuses too (exit 2), naming the pending patch",
       rc == 2 and rec.get('status') == 'refused' and SZ.CONFORM_AXIS_PATCH in (rec.get('reason') or ''),
       (rc, rec.get('status'), (rec.get('reason') or '')[:300], out[-300:]))
    rc, out = setup(pre=SG, conf=SG)
    rec = conform_rec()
    g = [t for t in rec.get('tests', []) if t['name'] == 'S-G13.size-rule-axis-step']
    rem = [r['cand_id'] for r in rcsv(f'{U}/master_catalog_removed.csv')] if os.path.exists(f'{U}/master_catalog_removed.csv') else None
    ok("S-G13, 'second_gate' (production's behavior, the parity file's value) with production's script as it is: the stage "
       "completes, the 9-sigma axis-2.4 row removed in production's words and recorded (S-G13.size-rule-axis-step 1), the "
       "conformant catalog's content tests passing", rc == 0 and rec.get('status') == 'done' and rem == ['I3'] and g
       and g[0]['value'] == 1 and g[0]['passed'] and 'second gate' in g[0]['detail'] and all(t['passed'] for t in rec['tests']),
       (rc, rec.get('status'), rec.get('reason'), rem, g, out[-500:]))
    stub('patched')
    rc, out = setup()
    rec = conform_rec()
    g = [t for t in rec.get('tests', []) if t['name'] == 'S-G13.size-rule-axis-step']
    kept = {r['cand_id']: r for r in rcsv(f'{U}/master_catalog.csv')} if rc == 0 else {}
    ok("S-G13, 'replace' (the default) with the pending patch's switch: P3_CONFORM_AXIS=0 reaches the script (the stand-in "
       "keeps the row only then), the axis-2.4 row kept and flagged, nothing removed by the axis step "
       "(S-G13.size-rule-axis-step 0), the content tests passing", rc == 0 and rec.get('status') == 'done'
       and sorted(kept) == ['I3', 'I4', 'V1'] and kept['I3'].get('axis_flag', '').startswith('axis > 2.0 x beam (kept')
       and g and g[0]['value'] == 0 and 'replaced (P3_CONFORM_AXIS=0)' in g[0]['detail']
       and not rcsv(f'{U}/master_catalog_removed.csv') and all(t['passed'] for t in rec['tests']),
       (rc, rec.get('status'), rec.get('reason'), g, sorted(kept), out[-500:]))
    rc, out = setup(pre=SG, conf=SG)
    rem = [r['cand_id'] for r in rcsv(f'{U}/master_catalog_removed.csv')] if rc == 0 else None
    ok("S-G13, 'second_gate' with the patched script (P3_CONFORM_AXIS unset: its default on): the row removed as today",
       rc == 0 and rem == ['I3'], (rc, rem, out[-300:]))
    os.remove(conform)


def test_e2e_snapshot(site):
    name = 'Run_SNAPT_20261013'
    cells = ['14h_2025-03-09', '20h_2025-05-03']
    open(f'{T}/cells.txt', 'w').write('# test cells\n14h 2025-03-09\n20h_2025-05-03\n')
    H = 'cell,subband,stokes,tier,ts,ra_deg,dec_deg,snr,fit_status,image\n'
    os.makedirs(f'{T}/b1', exist_ok=True)
    open(f'{T}/b1/fits_store_batch1.csv', 'w').write(H + 'c0,18MHz,I,10s,t0,1,2,8,ok,raw+flow\n')
    os.makedirs(f'{T}/batch/fits', exist_ok=True)
    os.makedirs(f'{T}/batch/tiecache', exist_ok=True)
    open(f'{T}/batch/fits/fits_14h_2025-03-09.csv', 'w').write(H + '14h_2025-03-09,Red,I,10s,t1,1,2,8,error,identity screen\n')
    old = time.time() - 3600
    for p in glob.glob(f'{T}/batch/fits/*') + [f'{T}/b1/fits_store_batch1.csv']:
        os.utime(p, (old, old))
    for c in cells:
        os.makedirs(f'{T}/fleet/{c}', exist_ok=True)
        json.dump({'frame_time_1min': 'integration-midpoint-dateobs-v1'}, open(f'{T}/fleet/{c}/runinfo_18MHz_median.json', 'w'))
    os.makedirs(f'{T}/served', exist_ok=True)
    json.dump({'cells': ['14h_2025-03-09']}, open(f'{T}/served/chain_summary.json', 'w'))
    stub = f'{T}/stub'
    open(f'{T}/code_{name}.sha256', 'w').write('%s  %s\n' % (hashlib.sha256(open(f'{stub}/clv15/phase3_snapshot_chain.py', 'rb')
                                                                           .read()).hexdigest(), f'{stub}/clv15/phase3_snapshot_chain.py'))
    RUN = f'{T}/out_snap/{name}'
    common = ['catalog', '--tier', 'snapshot', '--run', name, '--cells', f'@{T}/cells.txt', '--out', RUN, '--cores', 'none',
              '--work', f'{T}/scratch', '--snap-chain-nodes', 'localhost:1', '--snap-served-run', f'{T}/served',
              '--snap-approval', 'test approval, 2026-10-13T05:00Z, TEST-ONLY',
              '--set', f'paths.SNAPSHOT_FLEET_OUT={T}/fleet', '--set', f'catalog_snapshot.BATCH_DIR={T}/batch',
              '--set', f'catalog_snapshot.BATCH1_STORE={T}/b1/fits_store_batch1.csv',
              '--set', f'catalog_snapshot.CODE_SHA256={T}/code_{{name}}.sha256',
              '--set', f'runner.JOBS_ROOT={T}/jobs', '--set', 'runner.POLL_S=1']
    rc, out = cli(common + ['--stages', 'preflight'], site)
    rec = json.load(open(f'{RUN}/_records/catalog/{name}.preflight.json'))
    ok('snapshot preflight done (approval, code sha256, eligibility, served cells kept, Q14, frame time, tie model child)',
       rc == 0 and rec['status'] == 'done' and rec['meta'].get('tie_model') == 'stat0.42floor5_newmodel'
       and rec['meta'].get('n_new_cells') == 1, out[-800:])
    rc, out = cli(common + ['--stages', 'store'], site)
    rec = json.load(open(f'{RUN}/_records/catalog/{name}.store.json'))
    S = f'{RUN}/_side/rechain_{name}/fits_store.csv'
    ok('store: copies, repair lists (0 rows), combined store = batch-1 + cell rows, store test READY',
       rc == 0 and rec['status'] == 'done' and len(rcsv(S)) == 2 and os.path.exists(f'{RUN}/_side/fits/fits_14h_2025-03-09.csv'),
       out[-800:])
    t0 = time.time()
    rc, out = cli(common + ['--stages', 'prefit'], site, timeout=240)
    rec = json.load(open(f'{RUN}/_records/catalog/{name}.prefit.json'))
    items = glob.glob(f'{RUN}/_records/catalog/{name}.fit.prefit1.part*.json')
    ok('prefit: round 1 lists 2 unfitted, the fit round runs as a core/runner job on localhost, appends; round 2 lists 0',
       rc == 0 and rec['status'] == 'done' and rec['meta'].get('prefit_iterations') == 2 and items
       and len([r for r in rcsv(S) if r['fit_status'] == 'ok']) >= 3, (rc, out[-900:], round(time.time() - t0)))
    rc, out = cli(common + ['--stages', 'tie'], site, timeout=240)
    rec = json.load(open(f'{RUN}/_records/catalog/{name}.tie.json'))
    titems = sorted(glob.glob(f'{RUN}/_records/catalog/{name}.tie.*_*.json'))
    ok('tie: one runner item per cell (--stop-after-tie into the tie-cache COPY), tie_cache outputs pass',
       rc == 0 and rec['status'] == 'done' and len(titems) == 2 and rec['meta'].get('tie_tied_now') == 2
       and os.path.exists(f'{RUN}/_side/tiecache/tie_20h_2025-05-03.csv'), (rc, out[-900:]))
    rc, out = cli(common + ['--stages', 'chain,test'], site)
    rc_rec = json.load(open(f'{RUN}/_records/catalog/{name}.chain.json'))
    t_rec = json.load(open(f'{RUN}/_records/catalog/{name}.test.json'))
    ok('chain (memory model 250 GB, env 24/24/4000) and the run content test: snapshot_run passes',
       rc == 0 and rc_rec['status'] == 'done' and t_rec['status'] == 'done' and rc_rec['meta'].get('mem_need_gb') == 250,
       out[-900:])
    ok('nothing written to the production batch dir copies source (stores unchanged)',
       open(f'{T}/batch/fits/fits_14h_2025-03-09.csv').read().count('\n') == 2)


# =====================================================================================================================
def test_dry_runs():
    out_s = f'{T}/dry_sublst'
    rc, out = cli(['catalog', '--tier', 'sublst', '--lsts', '14h', '--dry-run', '--cores', 'none', '--out', out_s], REAL_SITE)
    mism = [ln for ln in out.splitlines() if ln.strip().startswith('MISMATCH')]
    ok('dry-run sub-LST 14h on real inputs: every stage planned, [lit] mismatches only on the shared K keys, nothing written',
       rc == 0 and '[14h_B.reftie]' in out and 'binding check:' in out and not os.path.exists(out_s)
       and all('K_TEMPLATE' in ln or 'SUBLST_K_ARG' in ln for ln in mism), mism or out[-300:])
    cf = f'{T}/b2two.txt'
    open(cf, 'w').write('02h_2024-12-28\n02h_2024-12-29\n')
    out_p = f'{T}/dry_snap'
    rc, out = cli(['catalog', '--tier', 'snapshot', '--run', 'Run_SNAPB2R_20261007', '--cells', f'@{cf}', '--dry-run',
                   '--cores', 'none', '--out', out_p, '--snap-served-run',
                   '/data07/gh/phase3/snapshot_search/review/Run_SNAPB1R_20261004'], REAL_SITE)
    from exopipe.phase3.lib.catalog import tier_snapshot as P
    ok(f'dry-run snapshot (2 cells of batch 2): every stage planned, 0 binding mismatches ({len(P.LIT)} [lit] keys), the restore-only '
       f'3-D gate off (no marker line), nothing written',
       rc == 0 and '[Run_SNAPB2R_20261007.test]' in out and f'binding check: {len(P.LIT)} ok, 0 mismatch' in out
       and 'P-G21 restore-only 3-D gate: off' in out and not os.path.exists(out_p), out[-300:])


def test_real_products():
    """Read-only content tests on production products (results printed; known pre-rule products are expected to fail)."""
    from exopipe.phase3.core import validate
    C = '/lustre/gh/main/phase3/sublst/compare_14h/new_B/catalog'
    for cls, p, expect in (('window_tie', f'{C}/pertie/pertie_detections.csv', True),
                           ('leak_veto_table', f'{C}/leakveto/leakveto_14h_18MHz_I.csv', True),
                           ('rejected_record', f'{C}/rejected.csv', True),
                           ('compare_copy', C, True),
                           ('tier_rule_removals', f'{C}/master_catalog_fluxscale_refused.csv', True),
                           ('tie_cache', '/lustre/gh/main/exopipe_ops/snapshot_batch/tiecache/tie_00h_2025-09-01.csv', True)):
        if not os.path.exists(p):
            print(f'SKIP real {cls}: {p} absent')
            continue
        res = validate.run(cls, p)
        mine = [r for r in res if r['name'].split('.')[1] in ('window_tie_rows', 'leak_veto_rows', 'rejected_record_consistent',
                                                              'compare_copy_equal', 'tier_rules_recomputed', 'tie_cache_rows')]
        ok(f'real {cls} {os.path.basename(p)}: {mine[0]["detail"][:150]}', all(r['passed'] for r in mine) == expect)


def main():
    t0 = time.time()
    shutil.rmtree(T, ignore_errors=True)
    os.makedirs(T)
    try:
        test_config()
        test_binding()
        test_ported_sublst()
        test_ported_sublst_floor_fitted()
        test_ported_snapshot()
        test_restore_marker()
        test_content_tests()
        stub = build_stubs()
        site = make_site(stub)
        test_e2e_sublst(site)
        test_e2e_snapshot(site)
        test_dry_runs()
        test_real_products()
    finally:
        npass = sum(1 for r in RESULTS if r[1])
        print(f'\n{npass} of {len(RESULTS)} checks passed in {time.time() - t0:.0f} s')
        stop_supervisors()
        shutil.rmtree(T, ignore_errors=True)


def stop_supervisors(wait_s=45):
    """The runner supervisors of this test's jobs poll their job record every runner.POLL_S (30 s, not forwarded to the
    supervisor): wait for them to see their job finished, then stop any left, before the job records are removed."""
    def mine():
        p = subprocess.run(['ps', '-eo', 'pid,args'], stdout=subprocess.PIPE, universal_newlines=True).stdout.splitlines()
        return [int(ln.split()[0]) for ln in p if 'run supervise' in ln and T in ln and 'grep' not in ln]
    t0 = time.time()
    while mine() and time.time() - t0 < wait_s:
        time.sleep(2)
    for pid in mine():
        try:
            os.kill(pid, 15)
        except OSError:
            pass


if __name__ == '__main__':
    main()
