#!/usr/bin/env python3
"""STANDING TEST - the per-instant veto rule as a content test, every tier (lib/catalog/tests_per_instant.py; Gregg 2026-10-18,
R-GREGG-20261018-VETOES-PER-FRAME-INDEPENDENTLY / -REFINED-FREQUENCY-YES-TIME-NO: a veto measured at one instant removes that
instant's detections in any band and Stokes, never another instant's, never another tier's).

  kinds       the reason grammar READ from the writers (lib/search/removal_rules.py, parent_curve_table.py): the reasons
              removal_rules' own functions write classify to their kinds, every template rendered classifies to its kind,
              a reason that adds words to a template ('... of night 2025-01-03') matches none - whole or cut at 160
              characters; the module's mirrors of emit_rejected_record.py (the 160-character cut) and p3s2_frames.py (the
              deep masters) are drift-checked against production
  deep        a synthetic deep conformance record: T1 tier, T2 undeclared reason, T3 the Jupiter track naming another night
              decided by the master's own header (unmasked: the imprint; JUPMASK or a jupmask COADDOP: refused; a master
              newer than the record or none: the tier rule, said so), T4 a pulsar group bounded by its instant's detections
              (refused when its n exceeds them), and exactly through a group record
  sublst      the 10-minute record: a track removal naming another night decided by the searched frame's JUPMASK (masked:
              refused; unmasked: the imprint; no frame record: the tier rule, masked - refused), a window that is not a
              window refused, the night-keyed noise test refused
  rejected    rejected.csv (reason cut at 160 characters): a cut template matches, a cut probe does not; a pulsar group
              needs a second detection of its own window
  fork        the deep multiples fork: EVERY record (master_catalog_removed.csv, each run's _<tag>.csv, subdirectories,
              rowline_removed.csv) judged - an undeclared reason in the validity run's _A record refuses (the review probe)
  snapshot    a synthetic chain run: per-instant record passes; a detection removed although its own instant passed, a member
              of another instant / tier, a count mismatch, a missing record, and source-level vetoes (before clv15 99fe3ea)
              each refuse; the tests are registered on their classes (conformant_catalog, rejected_record, snapshot_run)
  real        READ-ONLY on served products of every tier: deep Run_P3S5 (and every record of its multiples fork), 10-minute
              Run_SLST07B rejected.csv and the 14h compare copy, snapshot Run_SNAPB6H_20261023 (10-s and 1-minute) pass; the
              source-level batch Run_SNAPB2A_20261014 refuses

Scratch: /tmp/exopipe3_tests_per_instant/ (removed at the end). Production trees are only read.
Run standalone:  python tests/test_per_instant.py [test_name ...]
"""
import csv
import gzip
import json
import os
import re
import shutil
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')
SCRATCH = '/tmp/exopipe3_tests_per_instant'
PSR = 'PSR J0218+4232'                      # a pulsar of catalog.PULSARS_CSV (its position is read from the file)
PROBE = ' of night 2025-01-03'              # words naming another instant, added to a template (the review's probe)


def _fresh(name):
    d = os.path.join(SCRATCH, name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    return d


def _wcsv(path, cols, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _PI():
    from exopipe.phase3.lib.catalog import tests_per_instant as PI
    return PI


def _RR():
    from exopipe.phase3.lib.search import removal_rules as RR
    return RR


def _psr_pos():
    return _PI().pulsar_positions()[PSR]


# ---- reasons from the writers' own templates (never hand-written) ---------------------------------------------------
_SAMPLES = ('el50-90', '2025-01-02', PSR, 'flare_ucd:CV Com [GCVS:UV]', 'V', '5', '1.0', '1.00', '1.000', '1.25')


def render(node, vals=None, opt=True):
    """A reason of a template tree (tests_per_instant.reason_templates): named fields from ``vals``, the others the first
    sample their pattern takes; optional parts in (``opt``) or out; the first alternative."""
    vals = vals or {}
    k = node[0]
    if k == 'lit':
        return node[1]
    if k == 'pat':
        if node[3] and node[3] in vals:
            return vals[node[3]]
        for c in _SAMPLES:
            if re.fullmatch(node[1], c):
                return c
        raise AssertionError(f'no sample for the field pattern {node[1]}')
    if k == 'seq':
        return ''.join(render(x, vals, opt) for x in node[1])
    if k == 'opt':
        return render(node[1], vals, opt) if opt else ''
    return render(node[1][0], vals, opt)


def templates():
    return {kind: node for kind, _i, _t, node in _PI().reason_templates()}


def r_table():
    return render(templates()['table'])


def r_psr(n, bare=False):
    """A pulsar-group removal of PSR with n members (production's conformance wraps the group's words in the table reason;
    the B-02 pulsar stage writes them bare)."""
    return render(templates()['pulsar-group'][1][1 if bare else 0], {'pulsar': PSR, 'n': str(n), 'sep': '730', 'r99': '365'})


def r_jup(night):
    return _RR().jupiter_track_reason(10.0, 20.0, 'Red', [(10.0, 20.0, night)])


def r_elev():
    return _RR().elevation_reason(24.4, 'at mid-hour', floor=25.0)


def r_day():
    return render(templates()['daytime'])


CAT = ['cand_id', 'lst', 'date', 'band', 'stokes', 'cadence', 'ra', 'dec', 'snr']


def make_conform(root, tier='deep', removed=(), asrun_extra=()):
    """A conformance record: master_catalog.csv (one kept row), master_catalog_asrun.csv (kept + removed + asrun_extra) and
    master_catalog_removed.csv (``removed``: dicts with cand_id, date, reason, ra, dec[, jupiter_track_night])."""
    cad = 'deep' if tier == 'deep' else '10min'
    kept = [dict(cand_id='VD_05h_2025-01-02_Red_0001', lst='05h', date='2025-01-02' if tier == 'deep' else '2025-01-02T00',
                 band='Red', stokes='V', cadence=cad, ra='10.0', dec='20.0', snr='8')]
    rem = [dict(cand_id=x['cand_id'], lst='05h', date=x['date'], band='Red', stokes='V', cadence=x.get('cadence', cad),
                ra=x.get('ra', '11.0'), dec=x.get('dec', '21.0'), snr='7', removal_reason=x['reason'],
                jupiter_track_night=x.get('jupiter_track_night', '')) for x in removed]
    _wcsv(os.path.join(root, 'master_catalog.csv'), CAT, kept)
    _wcsv(os.path.join(root, 'master_catalog_asrun.csv'), CAT, kept + rem + list(asrun_extra))
    _wcsv(os.path.join(root, 'master_catalog_removed.csv'), CAT + ['removal_reason', 'jupiter_track_night'], rem)
    return os.path.join(root, 'master_catalog.csv')


def near(cand_id, date, k=0, cadence='deep'):
    """An as-run row 0.2 deg from the pulsar."""
    pra, pdec = _psr_pos()
    return dict(cand_id=cand_id, lst='05h', date=date, band='Red', stokes='V', cadence=cadence, ra=f'{pra + 0.2 + 0.01 * k:.5f}',
                dec=f'{pdec:.5f}', snr='6')


def fits_file(path, **cards):
    """A tiny FITS image whose primary header carries ``cards`` (a master or a searched frame)."""
    import numpy as np
    from astropy.io import fits
    os.makedirs(os.path.dirname(path), exist_ok=True)
    h = fits.Header()
    for k, v in cards.items():
        h[k] = v
    fits.writeto(path, np.zeros((4, 4), 'float32'), h, overwrite=True)
    return path


def age(path, seconds):
    """Set a file's modification time ``seconds`` before now (a master older than the record)."""
    t = time.time() - seconds
    os.utime(path, (t, t))


# ---------------------------------------------------------------------------------------------------------------------

def test_kinds():
    """The grammar is the writers' own (ast of removal_rules.py / parent_curve_table.py): their functions' reasons classify,
    every template renders to its kind, added words match nothing (whole or cut); the mirrors are drift-checked."""
    PI, RR = _PI(), _RR()
    T = templates()
    assert set(T) == {'pulsar-group', 'table', 'duplicate', 'excluded epoch', 'elevation floor', 'daytime', 'twilight',
                      'daytime unreadable', 'jupiter-track', 'noise sanity', 'row-line', 'no ledger verdict', 'size',
                      'axis ratio'}, sorted(T)
    # the reasons removal_rules' own functions and constants write
    _cols, size = RR.size_axis({'size_beams': '3.0', 'snr': '9', 'axis_ratio': '', 'axis_norm': ''}, size_step=True)
    _cols, axis = RR.size_axis({'size_beams': '1.0', 'snr': '9', 'axis_ratio': '', 'axis_norm': '2.5'}, size_step=True)
    written = {'elevation floor': [RR.elevation_reason(24.4, 'at mid-hour', floor=25.0),
                                   RR.elevation_reason(20.0, 'at the frame mid-time', floor=25.0)],
               'jupiter-track': [r_jup('2024-12-18')], 'noise sanity': [RR.noise_reason('V'), RR.noise_reason('I')],
               'row-line': [RR.rowline_reason(10.0, 20.0, [(10.0, 20.0)])], 'size': [size], 'axis ratio': [axis],
               'duplicate': [RR.DUPLICATE], 'excluded epoch': [RR.EXCLUDED], 'no ledger verdict': [RR.UNBACKED],
               'daytime unreadable': [RR.DAYTIME_UNREADABLE]}
    for kind, reasons in written.items():
        for r in reasons:
            assert r and PI.reason_kind(r)[0] == kind, (kind, r, PI.reason_kind(r))
            for probe in (r[:-1] + PROBE + r[-1:], r + PROBE, PROBE.strip() + ': ' + r):
                assert PI.reason_kind(probe) == (None, None), (kind, probe)
    # every template, rendered with and without its optional parts, classifies to its kind; with the probe to nothing
    for kind, node in T.items():
        for opt in (True, False):
            r = render(node, opt=opt)
            assert PI.reason_kind(r)[0] == kind, (kind, opt, r)
            assert PI.reason_kind(r[:-1] + PROBE + r[-1:]) == (None, None), (kind, r)
    # production's own reasons: the noise test is declared for the deep instant only; the bare B-02 pulsar-group words
    assert PI.reason_kind(RR.noise_reason('V'), 'deep') == ('noise sanity', 'own')
    assert PI.reason_kind(RR.noise_reason('V'), 'sublst') == ('noise sanity', None)
    assert PI.reason_kind(r_psr(3, bare=True)) == ('pulsar-group', 'pulsar-group') == PI.reason_kind(r_psr(3))
    assert PI.reason_kind(RR.table_reason(render(T['table'][1][1]))) == ('table', 'own')     # table_reason(note), as written
    # a reason cut at 160 characters (rejected.csv) matches a template's prefix; a cut probe does not
    long = r_psr(2)
    assert len(long) > PI.REJECTED_REASON_CHARS
    cut = long[:PI.REJECTED_REASON_CHARS]
    assert PI.reason_kind(cut) == (None, None) and PI.reason_kind(cut, truncated=True) == ('pulsar-group', 'pulsar-group')
    probe = (long[:40] + PROBE + long[40:])[:PI.REJECTED_REASON_CHARS]
    assert PI.reason_kind(probe, truncated=True) == (None, None), probe
    # the mirrors of production text: emit_rejected_record.py's cut, p3s2_frames.master_path's masters
    from exopipe.phase3.core import env
    ops = env.legacy_root('exopipe_ops')
    m = re.search(r"veto=\(r\.get\('removal_reason'\) or 'removed'\)\[:(\d+)\]", open(os.path.join(ops, 'emit_rejected_record.py')).read())
    assert m and int(m.group(1)) == PI.REJECTED_REASON_CHARS, m
    fr = open(os.path.join(ops, 'p3s2_frames.py')).read()
    assert (f"f'{PI.DEEP_MASTERS['I']}' if stokes == 'I'" in fr and f"else f'{PI.DEEP_MASTERS['V']}'" in fr), 'p3s2_frames drift'
    # a writer whose template changed (a field without a declared pattern) refuses - it never classifies by guess
    tree = PI._tree('removal_rules')
    old = PI._TREES['removal_rules']
    try:
        import ast
        src = open(PI.WRITERS['removal_rules']).read().replace("{_cc().ROWLINE_MIN} finals", "{_cc().ROWLINE_NEW} finals")
        PI._TREES['removal_rules'] = (old[0], ast.parse(src))
        PI._GRAMMAR.clear()
        try:
            PI.reason_templates()
            raise AssertionError('a template field without a pattern was accepted')
        except PI.TemplateError as e:
            assert 'ROWLINE_NEW' in str(e), e
        ok, val, det = PI.check_removals('deep', [dict(id='x', lst='05h', date='2025-01-02', cadence='deep', reason=r_elev())])
        assert not ok and 'cannot be read from the writers' in det, det
    finally:
        PI._TREES['removal_rules'] = old
        PI._GRAMMAR.clear()
    assert tree is PI._tree('removal_rules')


def test_deep():
    from exopipe.phase3.core import validate
    PI = _PI()
    root = _fresh('deep')
    own = '2025-01-02'
    # the deep masters of the record's cell: the co-add root of 05h_Red -> coadd_final/<DEEP_MASTERS>
    roots = os.path.join(root, 'roots.csv')
    with open(roots, 'w') as f:
        f.write(f"05h_Red,,{os.path.join(root, 'coadd')}\n")
    mV = fits_file(os.path.join(root, 'coadd', '05h_Red', 'coadd_final', PI.DEEP_MASTERS['V'].format(band='Red')),
                   COADDOP='ivw-mean-v3')
    age(mV, 3600)
    removed = [dict(cand_id='VD_05h_2025-01-02_Red_0002', date=own, reason=r_table()),
               dict(cand_id='VD_05h_2025-01-02_Red_0003', date=own, reason=r_elev()),
               dict(cand_id='VD_05h_2025-01-02_Red_0004', date=own, reason=r_jup('2024-12-18'), jupiter_track_night='2024-12-18'),
               dict(cand_id='VD_05h_2025-01-02_Red_0005', date=own, reason=r_jup(own))]
    p = make_conform(root, 'deep', removed)
    ok, val, det = PI.conform_per_instant(p, deep_roots_csv=roots)
    assert ok and val == 4 and "1 as the master's imprint (unmasked by the product's own files 1, by the tier rule 0)" in det, det
    names = {r['name']: r for r in validate.run('conformant_catalog', p, deep_roots_csv=roots)}
    assert names['conformant_catalog.per_instant']['passed'], names['conformant_catalog.per_instant']
    # T3 keyed on the master's own header: a master carrying the moving mask (JUPMASK, or a jupmask COADDOP) refuses
    for cards in ({'COADDOP': 'ivw-mean-v3', 'JUPMASK': 120.0}, {'COADDOP': 'ivw-mean-v3-jupmask'}):
        fits_file(mV, **cards)
        age(mV, 3600)
        ok, val, det = PI.conform_per_instant(p, deep_roots_csv=roots)
        assert not ok and 'T3 VD_05h_2025-01-02_Red_0004' in det and 'night 2024-12-18' in det and 'Red_0005' not in det, det
    # a master NEWER than the record is not the product's own (a later rebuild): the tier rule decides, and says so
    fits_file(mV, COADDOP='ivw-mean-v3', JUPMASK=120.0)
    age(mV, -3600)
    ok, val, det = PI.conform_per_instant(p, deep_roots_csv=roots)
    assert ok and 'by the tier rule 1' in det, det
    # no master at all: the tier rule (deep: unmasked); the configuration is never read (a flip cannot change a retest)
    os.remove(mV)
    ok, val, det = PI.conform_per_instant(p, deep_roots_csv=roots)
    assert ok and 'by the tier rule 1' in det, det
    import inspect
    assert 'JUPITER_MASK_DEEP' not in inspect.getsource(PI), 'the per-instant test reads the masters configuration'
    # T2 undeclared reason, T1 a 10-minute row in the deep record and a deep row without a night
    ok, val, det = PI.conform_per_instant(make_conform(_fresh('deep_t12'), 'deep', removed[:3] + [
        dict(cand_id='VD_05h_2025-01-02_Red_0006', date=own, reason='negative lobe measured on night 2025-01-03'),
        dict(cand_id='VD_05h_2025-01-02_Red_0016', date=own, reason=r_table()[:-1] + PROBE + ')'),
        dict(cand_id='V1_05h_2025-01-02T00_Red_0007', date='2025-01-02T00', reason=r_table(), cadence='10min'),
        dict(cand_id='VD_05h_x_Red_0008', date='', reason=r_table())]), deep_roots_csv=roots)
    assert not ok and val == 4 and 'T2 VD_05h_2025-01-02_Red_0006' in det and 'T2 VD_05h_2025-01-02_Red_0016' in det \
        and 'T1 V1_05h_2025-01-02T00_Red_0007' in det and 'T1 VD_05h_x_Red_0008' in det, det
    # T4 as-run bound: the removed row and one more detection of ITS night near the pulsar (n=2) pass; n=4 with the two
    # other members on other nights refuses (a group across nights decided the removal)
    pg = dict(near('VD_05h_2025-01-02_Red_0009', own), reason=r_psr(2))
    extra = [near('VD_05h_2025-01-02_Red_0010', own, 1), near('VD_05h_2025-01-03_Red_0001', '2025-01-03', 2),
             near('VD_05h_2025-01-04_Red_0001', '2025-01-04', 3),
             dict(near('VD_05h_2025-01-02_Red_0011', own, 4), cand_id='VD_05h_2025-01-02_Red_0011')]
    day = dict(near('VD_05h_2025-01-02_Red_0011', own, 4), reason=r_day())      # in the as-run, never in a group
    ok, val, det = PI.conform_per_instant(make_conform(_fresh('deep_t4'), 'deep', [pg, day], extra[:4]))
    assert ok and '1 (as-run bound)' in det, det
    ok, val, det = PI.conform_per_instant(make_conform(_fresh('deep_t4x'), 'deep', [dict(pg, reason=r_psr(4)), day], extra[:4]))
    assert not ok and 'T4 VD_05h_2025-01-02_Red_0009' in det and 'has 4 members' in det and 'holds 2 detection(s)' in det, det
    ok, val, det = PI.conform_per_instant(make_conform(_fresh('deep_t4y'), 'deep', [dict(pg, reason=r_psr(3)), day], extra[:4]))
    assert not ok and 'holds 2 detection(s)' in det, det                           # the daytime row does not count
    # T4 exactly, through the pulsar stage's group record (B-02 check: pulsar_groups.csv; the stage writes the words bare)
    from exopipe.phase3.lib.catalog import conform_check as CC
    gcols = CC.PULSAR_GROUP_COLS
    for name, groups, want in (
            ('per-epoch group of its night', [dict(pulsar=PSR, n=2, members='VD_05h_2025-01-02_Red_0009|VD_05h_2025-01-02_Red_0010',
                                                   sep_arcsec=730, r99_arcsec=365, consistent=0, grouping='per-epoch')], True),
            ('only an across-epoch group', [dict(pulsar=PSR, n=3, members='VD_05h_2025-01-02_Red_0009|VD_05h_2025-01-03_Red_0001|'
                                                 'VD_05h_2025-01-04_Red_0001', sep_arcsec=730, r99_arcsec=365, consistent=0,
                                                 grouping='across-epoch')], False),
            ('a per-epoch group holding another night', [dict(pulsar=PSR, n=2, members='VD_05h_2025-01-02_Red_0009|'
                                                              'VD_05h_2025-01-03_Red_0001', sep_arcsec=730, r99_arcsec=365,
                                                              consistent=0, grouping='per-epoch')], False)):
        d = _fresh('deep_groups')
        p = make_conform(d, 'deep', [dict(pg, reason=r_psr(2, bare=True))], extra[:2])
        _wcsv(os.path.join(d, CC.PULSAR_GROUPS), gcols, groups)
        ok, val, det = PI.conform_per_instant(p)
        assert ok == want and ('(group record)' in det), (name, det)


def test_sublst():
    """10-minute: T3 reads the searched frame's JUPMASK through the window tree's frames.csv (as the conformance does)."""
    PI = _PI()
    w = '2025-01-02T03'
    root = _fresh('sublst')
    tree = os.path.join(root, 'tenmin_root_x2')
    frame = fits_file(os.path.join(tree, '05h_Red', 'subtracted', f'{w}_V_sub_Red.fits'), JUPMASK=68.98)
    age(frame, 3600)
    _wcsv(os.path.join(tree, '05h_Red', 'frames.csv'), ['key', 'date', 'bin', 'stokes', 'src', 'sub_path'],
          [dict(key=w, date=w[:10], bin='3', stokes='V', src='', sub_path=frame)])
    removed = [dict(cand_id=f'V1_05h_{w}_Red_0001', date=w, reason=r_table()),
               dict(cand_id=f'V1_05h_{w}_Red_0002', date=w, reason=r_jup('2025-01-02')),       # its own night: own instant
               dict(cand_id=f'V1_05h_{w}_Red_0003', date=w, reason=r_jup('2025-01-05'))]
    p = make_conform(_fresh('sublst_rec'), 'sublst', removed)
    ok, val, det = PI.conform_per_instant(p, tier='sublst', tenmin_tree=tree)
    assert not ok and val == 1 and f'T3 V1_05h_{w}_Red_0003' in det and 'JUPMASK 68.98' in det, det
    # a searched frame without the mask (production keeps the track there): the master's imprint
    fits_file(frame, COADDOP='sublst-ivw-window-mean-v1')
    age(frame, 3600)
    ok, val, det = PI.conform_per_instant(p, tenmin_tree=tree)
    assert ok and 'unmasked by the product\'s own files 1' in det and 'of the sublst record' in det, det
    # no record of the frame: the tier rule (the window masters carry the moving mask) refuses, and says so
    ok, val, det = PI.conform_per_instant(p, tenmin_tree=os.path.join(root, 'nowhere'))
    assert not ok and 'the tier rule: RULING-MOVING-JUPITER-MASK-IN-MASTERS' in det, det
    # the window tree from the record's own run name (Run_SLST<hh><arm>) - here no such name: unknown, the tier rule
    ok, val, det = PI.conform_per_instant(p)
    assert not ok and 'names no 10-minute arm' in det, det
    run = os.path.join(_fresh('sublst_run'), 'Run_SLST05B')
    os.makedirs(run)
    assert PI._tenmin_tree(os.path.join(run, 'rejected.csv')).endswith('/tenmin_root_x2')
    ok, val, det = PI.conform_per_instant(make_conform(_fresh('sublst3'), 'sublst', [dict(removed[0], date='2025-01-02')]))
    assert not ok and 'is not a sublst instant' in det, det
    ok, val, det = PI.conform_per_instant(make_conform(_fresh('sublst4'), 'sublst', [dict(
        removed[0], reason=_RR().noise_reason('V'))]))
    assert not ok and 'not declared for the sublst tier' in det, det        # the night-keyed noise test: across windows


def test_rejected():
    """rejected.csv cuts the reason at 160 characters (emit_rejected_record.py): a cut template matches, a cut probe does
    not; the pulsar is read, n is not - a group needs a second detection of the row's own window."""
    PI = _PI()
    w = '2025-01-02T03'
    root = _fresh('rejected')
    pra, pdec = _psr_pos()
    cols = ['veto', 'cell', 'subband', 'stokes', 'tier', 'ts', 'ra_deg', 'dec_deg', 'snr', 'pos_err_arcsec', 'cand_id', 'detail']

    def rej(cid, reason, date=w, ra=pra + 0.2):
        return dict(veto=reason[:PI.REJECTED_REASON_CHARS], cell='05h_Red', subband='Red', stokes='V', tier='10min',
                    ts='20250102_051000', ra_deg=f'{ra:.6f}', dec_deg=f'{pdec:.6f}', snr='7', pos_err_arcsec='60',
                    cand_id=cid, detail=f'{cid} {date} peak 1.0 Jy')
    rows = [rej(f'V1_05h_{w}_Red_0001', r_table(), ra=pra + 5.0), rej(f'V1_05h_{w}_Red_0002', r_psr(5))]   # table row: far
    assert '(n=' not in rows[1]['veto'] and len(rows[1]['veto']) == PI.REJECTED_REASON_CHARS
    _wcsv(os.path.join(root, 'rejected.csv'), cols, rows)
    _wcsv(os.path.join(root, 'master_catalog.csv'), CAT, [near(f'V1_05h_{w}_Red_0009', w, 1, '10min'),
                                                          near('V1_05h_2025-01-03T03_Red_0001', '2025-01-03T03', 2, '10min')])
    ok, val, det = PI.rejected_per_instant(os.path.join(root, 'rejected.csv'))
    assert ok and val == 2 and 'pulsar-group 1' in det, det
    _wcsv(os.path.join(root, 'master_catalog.csv'), CAT, [near('V1_05h_2025-01-03T03_Red_0001', '2025-01-03T03', 2, '10min')])
    ok, val, det = PI.rejected_per_instant(os.path.join(root, 'rejected.csv'))
    assert not ok and 'at least (n cut from the record)' in det and 'holds 1 detection(s)' in det, det
    # a cut reason that adds words before the cut is undeclared
    long = r_psr(2)
    _wcsv(os.path.join(root, 'rejected.csv'), cols, [rej(f'V1_05h_{w}_Red_0002', long[:30] + PROBE + long[30:])])
    ok, val, det = PI.rejected_per_instant(os.path.join(root, 'rejected.csv'))
    assert not ok and 'T2 V1_05h_' in det, det
    from exopipe.phase3.core import validate
    assert 'rejected_record.per_instant' in {r['name'] for r in validate.run('rejected_record', os.path.join(root, 'rejected.csv'))}


def test_fork():
    """The deep multiples fork: every record of fork_<S>/ is judged (the review's probe: an undeclared reason in the validity
    run's master_catalog_removed_A.csv, which the next conformance run's master_catalog_removed.csv no longer shows)."""
    PI = _PI()
    run = _fresh('fork')
    fd = os.path.join(run, 'fork_V')
    os.makedirs(os.path.join(fd, 'members'))
    os.makedirs(os.path.join(run, 'source_catalogs_V_fsb'))
    own = '2025-01-02'
    rem_b = [dict(cand_id='VD_05h_2025-01-02_Red_0002', date=own, reason=r_table())]
    make_conform(fd, 'deep', rem_b)                                    # the table run B: its removed file is the last one
    shutil.copy(os.path.join(fd, 'master_catalog_removed.csv'), os.path.join(fd, 'master_catalog_removed_B.csv'))
    cols = CAT + ['removal_reason', 'jupiter_track_night']

    def rec(cid, reason, date=own, cadence='deep'):
        return dict(cand_id=cid, lst='05h', date=date, band='Red', stokes='V', cadence=cadence, ra='11.0', dec='21.0', snr='7',
                    removal_reason=reason)
    _wcsv(os.path.join(fd, 'master_catalog_removed_A.csv'), cols, [rec('VD_05h_2025-01-02_Red_0003', r_elev()),
                                                                   rec('VD_05h_2025-01-02_Red_0004', r_day())])
    _wcsv(os.path.join(fd, 'members', 'master_catalog_removed_A_members.csv'), cols, [rec('VD_05h_2025-01-02_Red_0005', r_day())])
    _wcsv(os.path.join(fd, 'rowline_removed.csv'), CAT + ['rowline_pass'], [dict(rec('VD_05h_2025-01-02_Red_0006', ''),
                                                                                  rowline_pass='pass1')])
    cat = os.path.join(run, 'source_catalogs_V_fsb')
    ok, val, det = PI.fork_per_instant(cat)
    assert ok and val == 6 and '5 fork records judged' in det and 'rowline_removed.csv 1' in det, det
    # the review's probe: an undeclared reason in the validity run's record (the last record does not show it) refuses
    _wcsv(os.path.join(fd, 'master_catalog_removed_A.csv'), cols, [rec('VD_05h_2025-01-02_Red_0003', r_elev()),
                                                                   rec('VD_05h_2025-01-02_Red_0007',
                                                                       'negative lobe measured on night 2025-01-03')])
    ok, val, det = PI.fork_per_instant(cat)
    assert not ok and 'master_catalog_removed_A.csv: T2 VD_05h_2025-01-02_Red_0007' in det, det
    # a 10-minute row in a fork record refuses (T1); a Jupiter track there is judged on the deep master (T3)
    _wcsv(os.path.join(fd, 'master_catalog_removed_A.csv'), cols, [rec('V1_05h_2025-01-02T00_Red_0008', r_elev(),
                                                                       date='2025-01-02T00', cadence='10min')])
    ok, val, det = PI.fork_per_instant(cat)
    assert not ok and 'T1 V1_05h_2025-01-02T00_Red_0008' in det, det
    from exopipe.phase3.core import validate
    assert 'fork_catalog.per_instant' in {r['name'] for r in validate.run('fork_catalog', cat)}
    assert not PI.fork_per_instant(fd)[0]                                  # not a fork catalogue directory


# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# snapshot runs
# ---------------------------------------------------------------------------------------------------------------------

VD_COLS = ['removed_by', 'cell', 'subband', 'stokes', 'tier', 'ts', 'ra_deg', 'dec_deg', 'snr', 'instant', 'instant_kind',
           'neg_member', 'asym_applies', 'asym_member', 'source_best_cand_id', 'source_instants_vetoed', 'source_outcome']
VS_COLS = ['veto', 'cell', 'subband', 'stokes', 'tier', 'ts', 'ra_deg', 'dec_deg', 'snr', 'n_members', 'n_instants', 'bands',
           'routes', 'pos_err_arcsec', 'detail']
REC_COLS = ['det_uid', 'cand_id', 'cell', 'lst', 'date', 'subband', 'band', 'stokes', 'tier', 'ts', 'instant_key', 'snr',
            'gate_neg_lobe', 'gate_asymmetry', 'neg_member', 'asym_member', 'first_removed_by', 'removed_detail']
CELL = '12h_2024-12-28'


def snap_run(root, per_instant=True, record=True, mutate=None):
    """A chain run whose source has three 10-s instants (I1, I2, I3) and one 1-minute instant (M1): the negative lobe fails at
    I1 (its 55 MHz V member measured; the Green I member of I1 goes with it - across frequency), the asymmetry at M1; I2 and I3
    pass. Record rows carry each detection's own instant verdict. ``mutate(vd, rec)`` edits the rows before writing."""
    os.makedirs(root, exist_ok=True)
    I1, I2, M1 = '20241228_132444', '20241228_132454', '20241228_1327'
    rec = [dict(cell=CELL, subband='55MHz', stokes='V', tier='10s', ts='20241228_132443', instant_key=I1, gate_neg_lobe='remove',
                gate_asymmetry='n/a', neg_member=f'{CELL} 55MHz 20241228_132443', first_removed_by='neg-lobe'),
           dict(cell=CELL, subband='Green', stokes='I', tier='10s', ts='20241228_132443', instant_key=I1, gate_neg_lobe='remove',
                gate_asymmetry='n/a', neg_member=f'{CELL} 55MHz 20241228_132443', first_removed_by='neg-lobe'),
           dict(cell=CELL, subband='55MHz', stokes='V', tier='10s', ts='20241228_132453', instant_key=I2, gate_neg_lobe='pass',
                gate_asymmetry='n/a', neg_member=f'{CELL} 55MHz 20241228_132453', first_removed_by=''),
           dict(cell=CELL, subband='64MHz', stokes='I', tier='1min', ts='20241228_132700', instant_key=M1,
                gate_neg_lobe='would-remove', gate_asymmetry='remove', neg_member=f'{CELL} 64MHz 1min 20241228_132700',
                asym_member=f'{CELL} 64MHz I 1min 20241228_132700', first_removed_by='asymmetry')]
    vd = [dict(removed_by='negative-lobe', cell=CELL, subband=r['subband'], stokes=r['stokes'], tier='10s', ts=r['ts'],
               instant=f'{CELL} 10s {I1}', instant_kind='10s-subband', neg_member=r['neg_member'], asym_applies='1')
          for r in rec[:2]]
    vd.append(dict(removed_by='reflection-asymmetry', cell=CELL, subband='64MHz', stokes='I', tier='1min', ts='20241228_132700',
                   instant=f'{CELL} 1min {M1}', instant_kind='1min', neg_member=rec[3]['neg_member'], asym_applies='1',
                   asym_member=rec[3]['asym_member']))
    if mutate:
        mutate(vd, rec)
    summ = {'cells': [CELL], 'n_served': 1}
    if per_instant:
        summ['per_instant_vetoes'] = {'rule': 'per instant (synthetic)'}
        _wcsv(os.path.join(root, 'vetoed_detections.csv'), VD_COLS, vd)
    json.dump(summ, open(os.path.join(root, 'chain_summary.json'), 'w'))
    _wcsv(os.path.join(root, 'vetoed_sources.csv'), VS_COLS,
          [] if per_instant else [dict(veto='negative-lobe', cell=CELL, subband='55MHz', stokes='V', tier='10s',
                                       ts='20241228_132443', detail='neg -19.6 rms, 0.45 of peak at 0.74 beams')])
    if record:
        with gzip.open(os.path.join(root, 'detection_record.csv.gz'), 'wt', newline='') as f:
            w = csv.DictWriter(f, fieldnames=REC_COLS, extrasaction='ignore')
            w.writeheader()
            for r in rec:
                w.writerow(dict({k: '' for k in REC_COLS}, **r))
    return root


def test_snapshot():
    from exopipe.phase3.core import validate
    PI = _PI()
    run = snap_run(_fresh('snap_ok'))
    ok, val, det = PI.check_snapshot_run(run)
    assert ok and val == 3 and '10s 2, 1min 1' in det, det
    names = {r['name']: r for r in validate.run('snapshot_run', run)}
    assert names['snapshot_run.per_instant']['passed'], names['snapshot_run.per_instant']

    def bad(name, mutate, want, **kw):
        ok, val, det = PI.check_snapshot_run(snap_run(_fresh('snap_' + name), mutate=mutate, **kw))
        assert not ok and want in det, (name, det)

    def other_instant(vd, rec):              # I2's 55 MHz detection removed by the lobe its own instant passed
        rec[2].update(first_removed_by='neg-lobe')
        vd.append(dict(vd[0], ts='20241228_132453', neg_member=f'{CELL} 55MHz 20241228_132443'))
    bad('other_instant', other_instant, "removed by the negative-lobe veto while its own instant's verdict is 'pass'")

    def member_elsewhere(vd, rec):           # the lobe that removed I1 was measured at I2's frame
        for r in rec[:2] + vd[:2]:
            r['neg_member'] = f'{CELL} 55MHz 20241228_132453'
    bad('member_elsewhere', member_elsewhere, 'not at its own instant 20241228_132444')

    def member_tier(vd, rec):                # a 10-s detection removed by the 1-minute block's measurement
        vd[0]['neg_member'] = f'{CELL} 64MHz 1min 20241228_132700'
    bad('member_tier', member_tier, 'another tier')

    def instant_tier(vd, rec):               # a 1-minute instant recorded on a 10-s row
        vd[0]['instant'] = f'{CELL} 1min 20241228_1324'
    bad('instant_tier', instant_tier, 'is not of its own cell and tier')

    def count(vd, rec):                      # the record removed one more detection by the lobe than the veto record lists
        rec.append(dict(rec[0], subband='Blue'))
    bad('count', count, 'the record says 3 detections were removed by negative-lobe, vetoed_detections.csv lists 2')
    bad('no_record', None, 'no per-detection record', record=False)
    bad('per_source', None, 'S1: 1 sources removed', per_instant=False, record=False)
    ok, val, det = PI.check_snapshot_run(snap_run(_fresh('snap_none'), mutate=lambda vd, rec: vd.clear() or rec.clear()))
    assert ok and val == 0, det
    os.remove(os.path.join(SCRATCH, 'snap_none', 'vetoed_sources.csv'))
    ok, val, det = PI.check_snapshot_run(os.path.join(SCRATCH, 'snap_none'))
    assert not ok and 'no vetoed_sources.csv' in det, det


# ---------------------------------------------------------------------------------------------------------------------
# real served products, read only
# ---------------------------------------------------------------------------------------------------------------------

DEEP = '/data07/gh/phase3/transients/Run_P3S5/master_catalog.csv'
DEEP_FORK = '/data07/gh/phase3/transients/Run_P3S5/source_catalogs_I_fsb'
TENMIN = '/data07/gh/phase3/transients/Run_SLST07B/rejected.csv'
TENMIN_CONFORM = '/lustre/gh/main/phase3/sublst/compare_14h/new_B/catalog/master_catalog.csv'
SNAP = '/data07/gh/phase3/snapshot_search/review/Run_SNAPB6H_20261023'
SNAP_SOURCE_LEVEL = '/data07/gh/phase3/snapshot_search/review/Run_SNAPB2A_20261014'


def test_real_products():
    PI = _PI()
    report = []
    for tier, cls, p, fn in (('deep', 'conformant_catalog', DEEP, PI.conform_per_instant),
                             ('deep multiples fork', 'fork_catalog', DEEP_FORK, PI.fork_per_instant),
                             ('10-minute', 'rejected_record', TENMIN, PI.rejected_per_instant),
                             ('10-minute', 'conformant_catalog', TENMIN_CONFORM, PI.conform_per_instant),
                             ('10-s + 1-minute', 'snapshot_run', SNAP, PI.check_snapshot_run)):
        if not os.path.exists(p):
            print(f'    SKIP {tier} {p}: absent')
            continue
        ok, val, det = fn(p)
        assert ok, (tier, p, det)
        if p == DEEP:                            # the deep track's master imprint and pulsar groups are there to be judged
            assert 'pulsar-group removals checked against one instant: 0' not in det and ' 0 as the master' not in det, det
        if p == SNAP:                            # both snapshot tiers carry veto removals
            assert ' 10s ' in det and ' 1min ' in det, det
        report.append(f'{tier} {cls} {p}: PASS ({det[:160]})')
    if os.path.exists(SNAP_SOURCE_LEVEL):        # chained under the source-level rule (before clv15 99fe3ea): refused
        ok, val, det = PI.check_snapshot_run(SNAP_SOURCE_LEVEL)
        assert not ok and det.startswith('S1:'), det
        report.append(f'source-level batch {SNAP_SOURCE_LEVEL}: REFUSED ({det[:120]})')
    assert report, 'no real product found'
    print('    ' + '\n    '.join(report))


TESTS = [test_kinds, test_deep, test_sublst, test_rejected, test_fork, test_snapshot, test_real_products]


def main(argv):
    want = set(argv[1:])
    t0 = time.time()
    failed = []
    os.makedirs(SCRATCH, exist_ok=True)
    try:
        for t in TESTS:
            if want and t.__name__ not in want:
                continue
            t1 = time.time()
            try:
                t()
                print(f'PASS {t.__name__} ({time.time() - t1:.1f} s)')
            except Exception:
                import traceback
                failed.append(t.__name__)
                print(f'FAIL {t.__name__}\n{traceback.format_exc()}')
    finally:
        shutil.rmtree(SCRATCH, ignore_errors=True)
    n = len([t for t in TESTS if not want or t.__name__ in want])
    print(f'{len(failed)} failed: {failed}' if failed else f'{n} of {n} passed ({time.time() - t0:.1f} s)')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
