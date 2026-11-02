#!/usr/bin/env python3
"""STANDING TEST - the deep omission of the short tiers (lib/catalog/deep_omission.py; Gregg 2026-10-24,
R-GREGG-20261024-DEEP-OMISSION-DESIGN-APPROVED-AND-10MIN-HOLD-RELEASED).

  drift      the port's rule and layout = production's modules (exopipe_ops sublst_deep_omission.py, snapshot_deep_omission.py,
             sublst_deep_omission_drop.py, snapshot_combined_view.py): the propagating gates, band groups, SIGMA, the refusal
             record, the window-rms table, the snapshot tiers and fleet root, every decision text, the cand_id and panel patterns
  snapshot   a synthetic fleet (two LSTs, eight nights, subbands and band groups, both Stokes and tiers, refusals of every gate
             kind, an older attempt, a stale_input row) and two served batch runs: production's own code (snapshot_deep_omission
             .decide and the combined view's step 0, run in this process, read only) and the port - the decision table byte for
             byte, the omitted detections and the dropped identities equal; hand-checked keys (the 18 MHz carve-out, member OMIT
             and ABSENT, the latest attempt, the Stokes scopes, ABSENT, fewer than 3 nights, rms and rate); the per-tier
             consequences (source dropped, partial re-pointed, member rows, panels) against the rule written out here
  tenmin     a synthetic window tree, searched lists, window-rms table, published LST runs and wideband windows (WBMEMB):
             production's sublst_deep_omission.decide (its roots patched) and the port byte for byte; the detections the merge's
             drop removes = production's sublst_deep_omission_drop.filter_rows on the member tables
  gates      G1 no record / no file / a foreign row, G2 a run without catalogue, G3 an identity without detections, G4 a source of
             no served identity, G6 served detections in an ABSENT key ('refuse'; 'production' keeps and counts)
  content    omission_decisions and omission_record refuse a changed verdict, a member row without its member, a removal whose
             key is not OMIT, a why of another key; omission_record.per_instant refuses O1 (another night's cand_id), O2 (an
             undeclared reason), O3 (a member of another tier); the templates are the writer's own
  stage      `exopipe3 catalog --tier snapshot --stages omit` and `--tier sublst --stages omit` through the step's run()
  real       READ ONLY: the port's decisions = production's decide for one LST of the served batch B6H (served counts of B6H)

Scratch: /tmp/exopipe3_tests_deep_omission/ (removed at the end). Production trees are only read.
Run standalone:  python tests/test_deep_omission.py [test_name ...]
"""
import collections
import csv
import filecmp
import json
import os
import random
import shutil
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')
SCRATCH = '/tmp/exopipe3_tests_deep_omission'
OPS = '/lustre/gh/main/exopipe_ops'
TAG, OLD = '20261006T000000', '20261001T000000'
NIGHTS = ['2025-01-0%d' % i for i in range(1, 9)]
SUBBANDS = ('18MHz', '23MHz', '27MHz', '46MHz', '55MHz')
csv.field_size_limit(10 ** 9)


def _fresh(name):
    d = os.path.join(SCRATCH, name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    return d


def _wcsv(path, cols, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _rows(p):
    return list(csv.DictReader(open(p, newline='')))


def _DO():
    from exopipe.phase3.lib.catalog import deep_omission as DO
    return DO


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def _prod():
    """production's modules, imported read only from exopipe_ops."""
    if OPS not in sys.path:
        sys.path.insert(0, OPS)
    import snapshot_deep_omission as SDO
    import sublst_deep_omission as PDO
    import sublst_deep_omission_drop as PDROP
    return SDO, PDO, PDROP


class Patch:
    """Set config / module attributes for a block, restore after."""

    def __init__(self, *items):
        self.items, self.saved = items, []

    def __enter__(self):
        for obj, k, v in self.items:
            self.saved.append((obj, k, getattr(obj, k)))
            setattr(obj, k, v)
        return self

    def __exit__(self, *a):
        for obj, k, v in reversed(self.saved):
            setattr(obj, k, v)


# ---------------------------------------------------------------------------------------------------------------------
# a synthetic snapshot world: refusal record, fleet records, two served batch runs
# ---------------------------------------------------------------------------------------------------------------------

def refusal_record(rdir):
    """Every gate kind: searchtier per Stokes (with an older attempt), beam (both, values), stale_input (not propagated),
    sanity_V / sanity_I (one Stokes; with and without values), nightimage (no values), epochgate of a band group, sanity
    (both, values), a band group's searchtier; 06h's 27 MHz has records of only two nights (no threshold)."""
    os.makedirs(rdir, exist_ok=True)
    files = {
        '05h_46MHz_searchtier': [['05h', '46MHz', NIGHTS[0], 'I', '9.1', '4.5', TAG], ['05h', '46MHz', NIGHTS[1], 'V', '8.0', '4.0', TAG],
                                 ['05h', '46MHz', NIGHTS[4], 'I', '7.2', '4.5', TAG], ['05h', '46MHz', NIGHTS[2], 'I', '9.9', '4.5', OLD]],
        '05h_46MHz_beam': [['05h', '46MHz', NIGHTS[3], '3.73', '1.5', TAG]],
        '05h_46MHz_stale_input': [['05h', '46MHz', NIGHTS[5], 'V product unreadable', TAG]],
        '05h_18MHz_sanity_V': [['05h', '18MHz', NIGHTS[0], '1181.9', '286.6', TAG]],
        '05h_23MHz_sanity_I': [['05h', '23MHz', NIGHTS[0], TAG], ['05h', '23MHz', NIGHTS[6], TAG]],
        '05h_55MHz_nightimage': [['05h', '55MHz', NIGHTS[1], TAG], ['05h', '55MHz', NIGHTS[7], TAG]],
        '05h_Green_epochgate': [['05h', 'Green', NIGHTS[2], TAG]],
        '05h_Red_searchtier': [['05h', 'Red', NIGHTS[4], 'V', '3.1', '2.0', TAG]],
        '06h_27MHz_searchtier': [['06h', '27MHz', NIGHTS[1], 'I', '5.5', '2.0', TAG]],
        '06h_46MHz_sanity': [['06h', '46MHz', NIGHTS[2], '40.1', '3.3', TAG]],
    }
    for name, rows in files.items():
        with open(os.path.join(rdir, name + '_excluded.csv'), 'w', newline='') as f:
            csv.writer(f).writerows(rows)
    return rdir


NOISE_COLS = ['idx', 'ts', 'date_obs', 'elev_bin', 'n_subbands', 'wsum', 'admitted', 'global_rms_jyb', 'n_components',
              'flood_refused', 'refuse_reason']


def searched(lst, ni, band):
    """the synthetic search ran there (05h 55 MHz has no night 8; 06h 27 MHz only nights 1-2)."""
    return not ((lst, band, ni) == ('05h', '55MHz', 7) or (lst == '06h' and band == '27MHz' and ni > 1))


def fleet(proto, rng):
    """Noise tables per (cell, band, Stokes, tier) and the cells' wideband member lists. Fixed points: 05h 46 MHz I 10-s is
    noisy on night 1 (rms OMIT); 05h 55 MHz has no records of night 8 (ABSENT); 06h 27 MHz only nights 1-2 (< 3 nights).
    Every table has a refused (admitted 0) row, an 'True' row, an empty and a non-numeric rms."""
    for lst in ('05h', '06h'):
        for ni, night in enumerate(NIGHTS):
            cell = os.path.join(proto, f'{lst}_{night}')
            os.makedirs(cell, exist_ok=True)
            for band in SUBBANDS + ('Red', 'Green'):
                if not searched(lst, ni, band):
                    continue
                for S in ('I', 'V'):
                    for T, sfx in (('10s', ''), ('1min', '_1min')):
                        n = rng.randint(6, 12)
                        base = 1.0 + 0.05 * rng.random()
                        if (lst, band, S, T, ni) == ('05h', '46MHz', 'I', '10s', 0):
                            base = 4.0
                        rows = [dict(idx=i, ts='x', admitted='1', global_rms_jyb='%.6f' % (base * (1 + 0.02 * rng.random())))
                                for i in range(n)]
                        rows += [dict(idx=n, admitted='0', global_rms_jyb='99.0'), dict(idx=n + 1, admitted='True',
                                                                                         global_rms_jyb='%.6f' % base),
                                 dict(idx=n + 2, admitted='1', global_rms_jyb=''), dict(idx=n + 3, admitted='1', global_rms_jyb='nan?')]
                        _wcsv(os.path.join(cell, f'noise_{band}_{S}_median{sfx}.csv'), NOISE_COLS, rows)
            for S in ('I', 'V'):
                mem = [('Red', '18MHz'), ('Red', '23MHz'), ('Green', '55MHz')] + ([('Red', '27MHz')] if ni % 2 else []) \
                    + ([('Green', '46MHz')] if ni != 3 else [])
                _wcsv(os.path.join(cell, f'wideband_members_{S}.csv'), ['group', 'ts', 'subband'],
                      [dict(group=g, ts='x', subband=sb) for g, sb in mem])
    return proto


DET_COLS = ['lst', 'date', 'band', 'subband', 'stokes', 'ts', 'snr', 'ra', 'dec', 'tier', 'instant', 'cand_id', 'ra_peak',
            'dec_peak']
BAND_CLASS = {'18MHz': 'Red', '23MHz': 'Red', '27MHz': 'Red', 'Red': 'Red', '46MHz': 'Green', '55MHz': 'Green', 'Green': 'Green'}


def detection(lst, night, sb, S, T, k, snr, rng):
    ts = '%s_%02d%02d%02d' % (night.replace('-', ''), int(lst[:2]), k // 6, (k % 6) * 10)
    if T == '1min':
        ts = ts[:-2] + '00'
    ra, dec = 10 + rng.random(), 20 + rng.random()
    return dict(lst=lst, date=night, band=BAND_CLASS[sb], subband=sb, stokes=S, ts=ts, snr='%.2f' % snr, ra='%.5f' % ra,
                dec='%.5f' % dec, tier=T, instant=ts[:13] if T == '1min' else ts, cand_id=f'{lst}_{night}_{sb}_{S}_{T}_{ts}',
                ra_peak='%.5f' % (ra + 0.001), dec_peak='%.5f' % dec)


def batch_run(root, name, cells, rng, fixed=()):
    """A served batch run: identities of 1-6 detections in its cells (plus ``fixed`` identities: (J, [detections args])),
    the general catalogue (source_catalogs/), the per-tier views (source_catalogs_<S>_<T>/: survivors, members), every
    page directory with its detections.csv and one panel + page.json per detection."""
    run = os.path.join(root, name)
    idents = []
    for i in range(36):
        dets = []
        while len(dets) < rng.randint(1, 6):
            lst, night = rng.choice(cells)
            sb = rng.choice(SUBBANDS + ('Red', 'Green'))
            if searched(lst, NIGHTS.index(night), sb):     # a detection only where the search ran (its noise table exists)
                dets.append(detection(lst, night, sb, rng.choice('IV'), rng.choice(('10s', '1min')), rng.randint(0, 59),
                                      6 + 20 * rng.random(), rng))
        idents.append((f'J{name[-12:-9]}{i:03d}+0000', dets))
    for J, spec in fixed:
        idents.append((J, [detection(*a, rng) for a in spec]))
    gen = []
    for J, dets in idents:
        rd = f'candidates/_identity/{J}'
        _wcsv(os.path.join(run, rd, 'detections.csv'), DET_COLS, dets)
        gen.append(dict(rank='rank0001', source_id=J, review_dir=rd, ra=dets[0]['ra'], dec=dets[0]['dec'],
                        best_snr=max(d['snr'] for d in dets)))
        by = collections.defaultdict(list)
        for d in dets:
            by[(d['subband'], d['stokes'], d['tier'])].append(d)
        for (sb, S, T), dd in by.items():
            pd = f'candidates/{S}/{J}_{sb}' + ('_1min' if T == '1min' else '')
            _wcsv(os.path.join(run, pd, 'detections.csv'), DET_COLS, dd)
            for d in dd:
                stem = f"rank0001_{d['lst']}_{d['date']}_{sb}{d['ts'].replace('_', '')}" + ('1min' if T == '1min' else '')
                for ext in ('.png', '.page.json'):
                    open(os.path.join(run, pd, stem + ext), 'w').close()
            open(os.path.join(run, pd, 'astrometry.png'), 'w').close()
            best = max(dd, key=lambda d: float(d['snr']))
            surv = os.path.join(run, f'source_catalogs_{S}_{T}', f'{sb}_sources_survivors.csv')
            mem = os.path.join(run, f'source_catalogs_{S}_{T}', f'{sb}_source_members.csv')
            for p, cols, rows in ((surv, ['rank', 'source_id', 'identity_id', 'review_dir', 'best_snr', 'n_det', 'best_cand_id',
                                          'ts', 'subband'],
                                   [dict(rank='rank0001', source_id=f'{J}_{sb}', identity_id=J, review_dir=pd, best_snr=best['snr'],
                                         n_det=len(dd), best_cand_id=best['cand_id'], ts=best['ts'], subband=sb)]),
                                  (mem, ['source_id', 'cand_id', 'det_snr', 'det_date', 'band'],
                                   [dict(source_id=f"{J}_{d['lst']}_{d['date']}_{sb}", cand_id=d['cand_id'], det_snr=d['snr'],
                                         det_date=d['date'], band=sb) for d in dd])):
                old = _rows(p) if os.path.exists(p) else []
                _wcsv(p, cols, old + rows)
    _wcsv(os.path.join(run, 'source_catalogs', 'Red_sources_survivors.csv'),
          ['rank', 'source_id', 'review_dir', 'ra', 'dec', 'best_snr'], gen)
    return run


def snapshot_world(name='snap', seed=7):
    root = _fresh(name)
    rng = random.Random(seed)
    rdir = refusal_record(os.path.join(root, 'refusals'))
    proto = fleet(os.path.join(root, 'fleet'), rng)
    cells_a = [(l, n) for l in ('05h', '06h') for n in NIGHTS[:4]]
    cells_b = [(l, n) for l in ('05h', '06h') for n in NIGHTS[4:]]
    # fixed identities: one entirely on the noisy key (dropped), one partial whose best detection is omitted (re-pointed), and
    # the rate case: 05h 46 MHz V 1-min night 2 served 40 times
    fixed_a = [('J0000+0001', [('05h', NIGHTS[0], '46MHz', 'I', '10s', 3, 30.0), ('05h', NIGHTS[0], '46MHz', 'I', '10s', 4, 20.0)]),
               ('J0000+0002', [('05h', NIGHTS[0], '46MHz', 'I', '10s', 5, 50.0), ('05h', NIGHTS[1], '46MHz', 'I', '10s', 5, 9.0),
                               ('05h', NIGHTS[1], '46MHz', 'I', '10s', 6, 8.0)]),
               ('J0000+0003', [('05h', NIGHTS[1], '46MHz', 'V', '1min', k, 7.0 + k) for k in range(40)])]
    runs = [batch_run(os.path.join(root, 'review'), 'Run_SNAPT1A_20261101', cells_a, rng, fixed_a),
            batch_run(os.path.join(root, 'review'), 'Run_SNAPT1B_20261101', cells_b, rng)]
    return root, rdir, proto, runs


def prod_snapshot(runs, rdir, proto, outdir):
    """production's own code: the combined view's source up to its merge (step 0: identities, decide, drop) in this process
    with snapshot_deep_omission.decide's record and fleet roots given; returns (DEC, REC, n_all)."""
    SDO, _p, _d = _prod()
    orig = SDO.decide
    SDO.decide = lambda lsts, served: orig(lsts, served, rdir=rdir, proto=proto)
    SDO._NS.clear()
    SDO._WM.clear()
    try:
        src = open(os.path.join(OPS, 'snapshot_combined_view.py')).read()
        g = {'__name__': 'prod_step0'}
        argv = sys.argv
        sys.argv = ['snapshot_combined_view.py', os.path.join(outdir, 'NEVER_WRITTEN')] + list(runs)
        try:
            exec(compile(src[:src.index("# ---- 2. the chain's merge")], 'snapshot_combined_view.py', 'exec'), g)
        finally:
            sys.argv = argv
    finally:
        SDO.decide = orig
        SDO._NS.clear()
        SDO._WM.clear()
    assert not os.path.exists(os.path.join(outdir, 'NEVER_WRITTEN'))
    SDO.write(os.path.join(outdir, 'prod_deep_omission.csv'), g['DEC'])
    return g['DEC'], g['REC'], g['n_all']


def _dec_by_key(rows):
    out = collections.defaultdict(list)
    for r in rows:
        out[(r['lst'], r['band'], r['night'], r['stokes'], r['tier'])].append(r)
    return out


# ---------------------------------------------------------------------------------------------------------------------

def test_drift():
    """The port's constants and texts are production's (the modules read as text / imported, nothing executed beyond import)."""
    SDO, PDO, PDROP = _prod()
    DO, C = _DO(), _cfg()
    c = C.catalog_omission
    assert tuple(c.PROPAGATE) == tuple(PDO.PROPAGATE), (c.PROPAGATE, PDO.PROPAGATE)
    assert {g: list(v) for g, v in C.instrument.SUBBAND_BANDS.items()} == PDO.GROUPS
    assert c.SIGMA == PDO.SIGMA and c.REFUSALS == PDO.REFUSALS and c.TENMIN_FRAME_RMS == PDO.FRAME_RMS
    assert tuple(c.SNAPSHOT_TIERS) == tuple(SDO.TIERS) and C.paths.SNAPSHOT_FLEET_OUT == SDO.PROTO
    assert os.path.join(C.paths.SUBLST_ROOT) == PDO.SL and C.paths.TRANSIENTS_ROOT == PDO.RUNS
    src_s = open(os.path.join(OPS, 'snapshot_deep_omission.py')).read()
    src_t = open(os.path.join(OPS, 'sublst_deep_omission.py')).read()
    src_d = open(os.path.join(OPS, 'sublst_deep_omission_drop.py')).read()
    src_v = open(os.path.join(OPS, 'snapshot_combined_view.py')).read()
    for txt, where in ((DO.PASSES, src_s), (DO.RMS_ABOVE, src_s), (DO.RATE_HIGH, src_s), (DO.PASSES, src_t), (DO.RMS_ABOVE, src_t),
                       (DO.RATE_HIGH, src_t), (DO.TENMIN_NO_WINDOWS, src_t), (DO.TENMIN_NIGHT_ABSENT, src_t),
                       (DO.SANITY_MEANING, src_s), (DO.SANITY_MEANING, src_t), (DO.SNAPSHOT_PARTIAL_NOTE, src_v),
                       ("% (', '.join(sorted(why)), g)", src_s), ("'member ' + ', '.join(sorted(why))", src_s),
                       ("'member ' + ', '.join(sorted(why))", src_t), ("'%s %s' % (r['band'], r['verdict'])", src_s),
                       ("'%s %s' % (r['band'], r['verdict'])", src_t), ("'%s: %s' % (r.get('gates', ''), r.get('reason', ''))", src_d),
                       ("' | '.join(WHY[o])", src_d)):
        assert txt in where, txt
    # the rate definitions (string literals of production's code, read with ast) and the record names
    import ast
    lit = {}
    for name, src in (('s', src_s), ('t', src_t)):
        for n in ast.walk(ast.parse(src)):
            if isinstance(n, ast.Assign) and len(n.targets) == 1:
                t = n.targets[0]
                sl = (t.slice.value if isinstance(t.slice, ast.Index) else t.slice) if isinstance(t, ast.Subscript) else None
                k = t.id if isinstance(t, ast.Name) else (sl.value if isinstance(sl, ast.Constant) else None)
                if k in ('RATE_DEF', 'rate_definition') and isinstance(n.value, (ast.Constant, ast.JoinedStr)):
                    lit[(name, k)] = ast.literal_eval(n.value)
    assert lit[('s', 'rate_definition')] == DO.SNAPSHOT_RATE_DEF and lit[('t', 'RATE_DEF')] == DO.TENMIN_RATE_DEF, lit
    for T, sfx in (('10s', ''), ('1min', '_1min')):
        assert c.SNAPSHOT_NOISE[T].format(band='46MHz', stokes='V') == 'noise_%s_%s_median%s.csv' % ('46MHz', 'V', sfx)
    assert "'noise_%s_%s_median%s.csv' % (band, S, '' if T == '10s' else '_1min')" in src_s
    assert c.SNAPSHOT_MEMBERS.format(stokes='I') == 'wideband_members_%s.csv' % 'I' and "'wideband_members_%s.csv' % S" in src_s
    assert c.TENMIN_RUN.format(hh='05', arm='B') == 'Run_SLST%sB' % '05' and "'Run_SLST%sB' % lst[:-1]" in src_t
    # the parameterized texts render production's words
    assert DO.snapshot_absent_reason('10s') == ('no searched %s data of this night, band and Stokes in the tier (nothing to drop)'
                                                % '10s')
    assert "no searched %s data of this night, band and Stokes in the tier (nothing to drop)" in src_s
    assert DO.snapshot_member_reason('46MHz OMIT', 'Green') == ("wideband-member: %s this night, a member of the cell\'s %s frames"
                                                                 % ('46MHz OMIT', 'Green'))
    assert "wideband-member: %s this night, a member of the cell\\'s %s frames" in src_s
    assert DO.tenmin_member_reason('46MHz OMIT') == 'wideband-member: %s this night, in the WBMEMB of its wideband windows' % '46MHz OMIT'
    assert "wideband-member: %s this night, in the WBMEMB of its wideband windows" in src_t
    # the patterns of keys: snapshot_combined_view CIDK / PANEL, sublst_deep_omission_drop CID
    assert DO.SNAP_CID.pattern in src_v and DO.SNAP_PANEL.pattern in src_v and DO.TENMIN_CID.pattern in src_d
    assert DO.MEMBER_SOURCE.pattern in src_v
    assert 'RATE FLOOR' in src_t and '1.0 / max(nwm, 1.0)' in src_t and c.RATE_FLOOR_PER_NIGHT == 1.0
    assert "x['band'] != '18MHz'" in src_t and "r['band'] != '18MHz'" in src_s and tuple(c.WIDEBAND_EXCEPT) == ('18MHz',)


def test_snapshot():
    """The port = production's own code on a synthetic world; hand-checked keys; per-tier consequences."""
    DO = _DO()
    root, rdir, proto, runs = snapshot_world()
    out = os.path.join(root, 'port')
    summ = DO.run_snapshot(runs, out, rdir=rdir, proto=proto)
    DEC, REC, n_all = prod_snapshot(runs, rdir, proto, root)
    assert filecmp.cmp(os.path.join(out, 'deep_omission.csv'), os.path.join(root, 'prod_deep_omission.csv'), shallow=False), \
        'decision tables differ'
    rec = _rows(os.path.join(out, 'deep_omission_dropped.csv'))
    K = lambda r: (r['batch'], r['identity'], r['cand_id'], r['lst'], r['band'], r['night'], r['stokes'], r['tier'], r['snr'])
    port_det = collections.Counter(K(r) for r in rec if r['kind'] == 'detection')
    prod_det = collections.Counter(K(r) for r in REC if r['kind'] == 'detection')
    assert port_det == prod_det and sum(port_det.values()) == summ['omitted'] > 0, (len(port_det), len(prod_det))
    assert {(r['batch'], r['identity']) for r in rec if r['kind'] == 'identity dropped'} == \
        {(r['batch'], r['identity']) for r in REC if r['kind'] == 'identity dropped'}
    assert summ['served'] == n_all and summ['kept'] == n_all - summ['omitted']
    # hand-checked decisions
    D = _dec_by_key(_rows(os.path.join(out, 'deep_omission.csv')))
    one = lambda k, kind='decision': [r for r in D[k] if (r['gates'].startswith('member ')) == (kind == 'member')]
    r = one(('05h', '46MHz', NIGHTS[0], 'I', '10s'))[0]
    assert (r['verdict'], r['reason'], r['rms_pass']) == ('OMIT', DO.RMS_ABOVE, '0'), r            # the noisy night
    assert not D[('05h', '46MHz', NIGHTS[0], 'V', '10s')]                                          # searchtier: its Stokes only
    assert not D[('05h', '46MHz', NIGHTS[2], 'I', '10s')]                                          # an older attempt: not read
    assert not D[('05h', '46MHz', NIGHTS[5], 'V', '10s')]                                          # stale_input: not propagated
    assert one(('05h', '46MHz', NIGHTS[3], 'I', '1min')) and one(('05h', '46MHz', NIGHTS[3], 'V', '1min'))   # beam: both
    r = one(('05h', '46MHz', NIGHTS[1], 'V', '1min'))[0]
    assert (r['verdict'], r['reason'], r['rms_pass'], r['rate_pass']) == ('OMIT', DO.RATE_HIGH, '1', '0'), r   # the rate case
    assert one(('05h', '18MHz', NIGHTS[0], 'V', '10s')) and not one(('05h', '18MHz', NIGHTS[0], 'I', '10s'))   # sanity_V: V
    for T in ('10s', '1min'):                                                                      # 18 MHz never omits Red
        for S in ('I', 'V'):
            for m in one(('05h', 'Red', NIGHTS[0], S, T), 'member'):
                assert '18MHz' not in m['gates'], m
    r = one(('05h', '55MHz', NIGHTS[7], 'I', '10s'))[0]
    assert (r['verdict'], r['reason']) == ('ABSENT', DO.snapshot_absent_reason('10s')), r            # no record of the night
    m = one(('05h', 'Green', NIGHTS[7], 'I', '10s'), 'member')                                     # ABSENT member -> Green
    assert m and '55MHz ABSENT' in m[0]['gates'] and m[0]['verdict'] == 'OMIT', m
    r = one(('06h', '27MHz', NIGHTS[1], 'I', '10s'))[0]
    assert (r['verdict'], r['tier_rms_threshold'], r['rms_pass']) == ('OMIT', 'nan', '0'), r         # fewer than 3 nights
    assert one(('05h', 'Green', NIGHTS[2], 'V', '10s'))                                            # a deep-refused group night
    # per-tier consequences, the rule written out: J0000+0001 dropped, J0000+0002's 46 MHz I 10-s source re-pointed
    kinds = collections.Counter(r['kind'] for r in rec)
    assert kinds['identity dropped'] >= 1 and kinds['source dropped'] >= 1 and kinds['member row'] >= 1 and kinds['panel'] >= 1
    part = [r for r in rec if r['kind'] == 'source partial' and r['identity'] == 'J0000+0002']
    assert part and 're-pointed to 05h_%s_46MHz_I_10s' % NIGHTS[1] in part[0]['note'], part
    assert any(r['kind'] == 'source dropped' and r['identity'] == 'J0000+0001' and r['note'] == 'identity left with no detection'
               for r in rec)
    OM = DO.omitted_keys(_rows(os.path.join(out, 'deep_omission.csv')))
    for r in rec:
        if r['kind'] in ('panel', 'detection') or (r['kind'] == 'member row' and r['lst']):
            assert (r['lst'], r['band'], r['night'], r['stokes'], r['tier']) in OM, r
    # the content tests of both classes and the per-instant test pass on the stage's products
    from exopipe.phase3.core import validate
    for cls, p in (('omission_decisions', 'deep_omission.csv'), ('omission_record', 'deep_omission_dropped.csv')):
        for t in validate.run(cls, os.path.join(out, p), tier='snapshot'):
            assert t['passed'], t
    pi = [t for t in validate.run('omission_record', os.path.join(out, 'deep_omission_dropped.csv')) if t['name'].endswith('per_instant')]
    assert pi and pi[0]['passed'] and pi[0]['value'] > 0, pi


# ---------------------------------------------------------------------------------------------------------------------
# a synthetic 10-minute world
# ---------------------------------------------------------------------------------------------------------------------

def tenmin_world(name='tenmin', seed=11):
    """SUBLST root (tenmin_root_x2/<LST>_<band>/frames.csv + window files, search_B/.../v_deep_images.csv), the window-rms
    table, the published runs Run_SLST<hh>B (master_catalog.csv, member tables) and Green wideband windows with WBMEMB."""
    import numpy as np
    from astropy.io import fits
    root = _fresh(name)
    rng = random.Random(seed)
    sl, tr = os.path.join(root, 'sublst'), os.path.join(root, 'transients')
    rdir = refusal_record(os.path.join(root, 'refusals'))
    frms = []
    served = collections.defaultdict(list)
    for lst in ('05h', '06h'):
        for band in SUBBANDS + ('Red', 'Green'):
            fr = []
            for ni, night in enumerate(NIGHTS):
                if (lst, band, ni) == ('05h', '55MHz', 7) or (lst == '06h' and band == '27MHz' and ni > 1):
                    continue
                for b in range(3):
                    key = f'{night}T{b:02d}'
                    for S in ('I', 'V'):
                        p = os.path.join(sl, 'tenmin_root_x2', f'{lst}_{band}', 'subtracted', f'{key}_{S}_sub_{band}.fits')
                        os.makedirs(os.path.dirname(p), exist_ok=True)
                        h = fits.Header()
                        if band == 'Green':
                            h['WBMEMB'] = '46MHz,55MHz' if ni != 3 else '55MHz'
                        fits.writeto(p, np.zeros((2, 2), 'float32'), h, overwrite=True)
                        st = os.stat(p)
                        rms = 1.0 + 0.05 * rng.random() if not (lst, band, S, ni) == ('05h', '46MHz', 'I', 0) else 4.0
                        frms.append(dict(path=p, size=st.st_size, mtime=int(st.st_mtime), rms='%.6f' % rms))
                        fr.append(dict(key=key, date=night, bin=b, stokes=S, src='', sub_path=p))
                        if rng.random() < 0.3:
                            served[lst].append(dict(band=band, date=key, stokes=S))
            _wcsv(os.path.join(sl, 'tenmin_root_x2', f'{lst}_{band}', 'frames.csv'), ['key', 'date', 'bin', 'stokes', 'src', 'sub_path'], fr)
            for S in ('I', 'V'):
                _wcsv(os.path.join(sl, 'search_B', f'{lst}_{band}_{S}', 'transients', f'Run_SLST{lst[:2]}B', 'v_deep_images.csv'),
                      ['date'], [dict(date=x['key']) for x in fr if x['stokes'] == S])
    for k in range(40):                                                           # the rate case: 05h 46 MHz V night 2
        served['05h'].append(dict(band='46MHz', date=f'{NIGHTS[1]}T00', stokes='V'))
    for lst in ('05h', '06h'):
        R = os.path.join(tr, f'Run_SLST{lst[:2]}B')
        rows = [dict(r, cand_id=f"{r['stokes']}1_{lst}_{r['date']}_{r['band']}_{i:04d}") for i, r in enumerate(served[lst])]
        _wcsv(os.path.join(R, 'master_catalog.csv'), ['cand_id', 'band', 'date', 'stokes'], rows)
        for S in ('I', 'V'):
            for T, bands in (('sb', SUBBANDS), ('wb', ('Red', 'Green'))):
                for band in bands:
                    mem = [dict(source_id=f'J{i:04d}_{band}', cand_id=r['cand_id'], band=band, det_snr='7')
                           for i, r in enumerate(rows) if r['stokes'] == S and r['band'] == band]
                    _wcsv(os.path.join(R, f'source_catalogs_{S}_{T}', f'{band}_source_members.csv'),
                          ['source_id', 'cand_id', 'band', 'det_snr'], mem)
    _wcsv(os.path.join(root, 'page_context_frames.csv'), ['path', 'size', 'mtime', 'rms'], frms)
    return root, rdir, sl, tr, os.path.join(root, 'page_context_frames.csv')


def test_tenmin():
    """The port = production's sublst_deep_omission.decide (roots patched) byte for byte; the detections the merge drop removes
    = production's sublst_deep_omission_drop.filter_rows on every member table."""
    DO, C = _DO(), _cfg()
    SDO, PDO, PDROP = _prod()
    root, rdir, sl, tr, frms = tenmin_world()
    lsts = ['05h', '06h']
    os.makedirs(os.path.join(root, 'prodcache'))           # production's driver makes it (sublst_cumulative.sh mkdir -p)
    with Patch((PDO, 'SL', sl), (PDO, 'RUNS', tr), (PDO, 'FRAME_RMS', frms), (PDO, '_FR', None), (PDO, '_WBW', {}), (PDO, '_WBM', {})):
        prows = []
        for L in lsts:
            prows += PDO.decide(L, rdir, os.path.join(root, 'prodcache'))
    for r in prows:
        r['rate_definition'] = DO.TENMIN_RATE_DEF
    with open(os.path.join(root, 'prod.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(prows[0]))
        w.writeheader()
        w.writerows(prows)
    with Patch((C.paths, 'SUBLST_ROOT', sl), (C.paths, 'TRANSIENTS_ROOT', tr)):
        summ = DO.run_tenmin(lsts, os.path.join(root, 'port'), rdir=rdir, seed_dir='', frame_rms=frms)
    assert filecmp.cmp(os.path.join(root, 'prod.csv'), os.path.join(root, 'port', 'deep_omission.csv'), shallow=False), \
        'decision tables differ'
    assert summ['window_rms'].get('page-context', 0) > 0 and not summ['window_rms'].get('recomputed'), summ['window_rms']
    rows = _rows(os.path.join(root, 'port', 'deep_omission.csv'))
    V = {(r['lst'], r['band'], r['night'], r['stokes'], r['gates'].startswith('member')): r for r in rows}
    assert V[('05h', '46MHz', NIGHTS[0], 'I', False)]['verdict'] == 'OMIT'
    assert V[('05h', '46MHz', NIGHTS[1], 'V', False)]['reason'] == DO.RATE_HIGH
    assert V[('05h', '55MHz', NIGHTS[7], 'I', False)]['reason'] == DO.TENMIN_NIGHT_ABSENT
    g = V.get(('05h', 'Green', NIGHTS[7], 'I', True))
    assert g and '55MHz ABSENT' in g['gates'], g                        # by the windows' own WBMEMB card
    assert ('05h', 'Green', NIGHTS[3], 'I', True) not in V or '46MHz' not in V[('05h', 'Green', NIGHTS[3], 'I', True)]['gates']
    # the merge drop's detections: production's filter_rows over every member table = the stage's record
    PDROP.OM.clear()
    PDROP.WHY.clear()
    del PDROP.REC[:]
    PDROP.CNT.clear()
    PDROP._CTX.clear()
    PDROP.load(os.path.join(root, 'port', 'deep_omission.csv'))
    for L in lsts:
        R = os.path.join(tr, f'Run_SLST{L[:2]}B')
        for S in ('I', 'V'):
            for T in ('sb', 'wb'):
                d = os.path.join(R, f'source_catalogs_{S}_{T}')
                for nm in sorted(os.listdir(d)):
                    PDROP.filter_rows(L, R, f'source_catalogs_{S}_{T}/{nm}', _rows(os.path.join(d, nm)))
    want = collections.Counter((r['lst'], r['stokes'], r['view'], r['source_id'], r['band'], r['cand_id'], r['night'], r['why'])
                               for r in PDROP.REC if r['kind'] == 'detection')
    got = collections.Counter((r['lst'], r['stokes'], r['view'], r['source_id'], r['band'], r['cand_id'], r['night'], r['why'])
                              for r in _rows(os.path.join(root, 'port', 'deep_omission_dropped.csv')))
    assert want == got and sum(got.values()) == summ['omitted'] > 0, (sum(want.values()), sum(got.values()))
    from exopipe.phase3.core import validate
    for cls, p in (('omission_decisions', 'deep_omission.csv'), ('omission_record', 'deep_omission_dropped.csv')):
        for t in validate.run(cls, os.path.join(root, 'port', p), tier='sublst'):
            assert t['passed'], t


# ---------------------------------------------------------------------------------------------------------------------

def test_gates():
    from exopipe.phase3.core.record import Refusal
    DO, C = _DO(), _cfg()
    root, rdir, proto, runs = snapshot_world('gates', seed=3)

    def refused(fn, want):
        try:
            fn()
        except Refusal as e:
            assert want in str(e), (want, str(e))
            return
        raise AssertionError(f'no refusal ({want})')
    refused(lambda: DO.run_snapshot(runs, os.path.join(root, 'o1'), rdir=os.path.join(root, 'none'), proto=proto), 'G1: no deep')
    os.makedirs(os.path.join(root, 'empty'))
    refused(lambda: DO.run_snapshot(runs, os.path.join(root, 'o2'), rdir=os.path.join(root, 'empty'), proto=proto), 'holds no')
    bad = os.path.join(root, 'foreign')
    shutil.copytree(rdir, bad)
    with open(os.path.join(bad, '05h_46MHz_beam_excluded.csv'), 'a') as f:
        f.write('06h,46MHz,2025-01-05,1,1,%s\n' % TAG)
    refused(lambda: DO.run_snapshot(runs, os.path.join(root, 'o3'), rdir=bad, proto=proto), 'not of its LST 05h')
    refused(lambda: DO.run_snapshot(runs + [os.path.join(root, 'nothing')], os.path.join(root, 'o4'), rdir=rdir, proto=proto),
            'G2:')
    p = os.path.join(runs[0], 'candidates', '_identity', 'J0000+0003', 'detections.csv')
    os.rename(p, p + '.x')
    refused(lambda: DO.run_snapshot(runs, os.path.join(root, 'o5'), rdir=rdir, proto=proto), 'G3: T1A identity J0000+0003')
    os.rename(p + '.x', p)
    cat = os.path.join(runs[1], 'source_catalogs_V_10s', 'stray_sources_survivors.csv')
    _wcsv(cat, ['source_id', 'identity_id', 'review_dir'], [dict(source_id='Jx_stray', identity_id='J9999+9999', review_dir='x')])
    refused(lambda: DO.run_snapshot(runs, os.path.join(root, 'o6'), rdir=rdir, proto=proto), 'G4:')
    os.remove(cat)
    # G6: a served detection in an ABSENT key (05h 55 MHz night 8 has no record): refused; 'production' keeps and counts it
    _wcsv(os.path.join(runs[1], 'candidates', '_identity', 'J0000+0009', 'detections.csv'), DET_COLS,
          [detection('05h', NIGHTS[7], '55MHz', 'I', '10s', 1, 8.0, random.Random(1))])
    g = os.path.join(runs[1], 'source_catalogs', 'Red_sources_survivors.csv')
    gen = _rows(g)
    _wcsv(g, list(gen[0]), gen + [dict(gen[0], source_id='J0000+0009', review_dir='candidates/_identity/J0000+0009')])
    refused(lambda: DO.run_snapshot(runs, os.path.join(root, 'o7'), rdir=rdir, proto=proto), 'G6: 1 deep-refused keys')
    with Patch((C.catalog_omission, 'ABSENT_WITH_SERVED', 'production')):
        s = DO.run_snapshot(runs, os.path.join(root, 'o8'), rdir=rdir, proto=proto)
    assert s['absent_with_served'] == 1, s


def test_content():
    """The content tests refuse what they must; the per-instant test reads the writer's own templates."""
    DO = _DO()
    from exopipe.phase3.lib.catalog import tests_deep_omission as TD
    from exopipe.phase3.lib.catalog import tests_per_instant as PI
    root, rdir, proto, runs = snapshot_world('content', seed=5)
    out = os.path.join(root, 'port')
    DO.run_snapshot(runs, out, rdir=rdir, proto=proto)
    dec, rec = os.path.join(out, 'deep_omission.csv'), os.path.join(out, 'deep_omission_dropped.csv')
    assert TD.check_decisions(dec)[0] and TD.check_record(rec)[0] and PI.check_omission(rec)[0]
    rows, hdr = _rows(dec), list(_rows(dec)[0])
    # the omission kinds are the writer's: its texts classify, added words do not
    for r in rows:
        if r['verdict'] == 'OMIT':
            assert PI.omission_kind(DO.why_part(r['gates'], r['reason'])), r
    assert PI.omission_kind('beam: tier rms above threshold of night 2025-01-03') is None
    assert PI.omission_kind('stale_input: tier rms above threshold') is None
    assert set(k for k, *_ in PI.omission_templates()) == {'deep omission', 'deep omission (wideband member)'}

    def mutated(path, rows_, fix, check, want):
        bad = os.path.join(root, 'mut', os.path.basename(path))
        os.makedirs(os.path.dirname(bad), exist_ok=True)
        rr = [dict(r) for r in rows_]
        fix(rr)
        _wcsv(bad, list(rows_[0]), rr)
        if bad.endswith('_dropped.csv'):
            shutil.copy(dec, os.path.join(os.path.dirname(bad), 'deep_omission.csv'))
        ok, val, det = check(bad)
        assert not ok and want in det, (want, det)

    def flip(rr):                                     # a KEEP made OMIT (its values pass)
        r = next(r for r in rr if r['verdict'] == 'KEEP')
        r['verdict'] = 'OMIT'
    mutated(dec, rows, flip, TD.check_decisions, 'with rms_pass 1 rate_pass 1')

    def orphan(rr):                                   # a member row whose member has no decision of its own night
        r = next(r for r in rr if r['gates'].startswith('member '))
        r['night'] = '2024-12-31'
    mutated(dec, rows, orphan, TD.check_decisions, 'has no decision row of its own night')
    rr0 = _rows(rec)

    def not_omit(rr):                                 # a removal whose key is no OMIT key
        r = next(r for r in rr if r['kind'] == 'detection')
        r['night'] = '2024-12-31'
    mutated(rec, rr0, not_omit, TD.check_record, 'is not an OMIT key')
    mutated(rec, rr0, not_omit, PI.check_omission, 'O1')

    def other_why(rr):                                # the text of another key
        ds = [r for r in rr if r['kind'] == 'detection']
        a = next(r for r in ds if r['why'] != ds[0]['why'])
        a['why'] = ds[0]['why']
    mutated(rec, rr0, other_why, PI.check_omission, 'O3')

    def undeclared(rr):
        r = next(r for r in rr if r['kind'] == 'detection')
        r['why'] = r['why'] + ' of night 2025-01-03'
    mutated(rec, rr0, undeclared, PI.check_omission, 'O2')

    def other_tier(rr):                               # a 1-minute removal recorded as 10-s
        r = next(r for r in rr if r['kind'] == 'detection' and r['tier'] == '1min')
        r['tier'] = '10s'
    mutated(rec, rr0, other_tier, PI.check_omission, 'O1')


def test_stage():
    """`exopipe3 catalog --tier snapshot|sublst --stages omit` through the step: records, classes, refusal on a missing record."""
    import argparse
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import record
    from exopipe.phase3.steps import catalog as step
    DO = _DO()
    root, rdir, proto, runs = snapshot_world('stage', seed=9)
    p = argparse.ArgumentParser()
    step.add_arguments(p)
    p.add_argument('--tier')
    p.add_argument('--run', dest='run_label')
    p.add_argument('--lsts')
    p.add_argument('--epochs')
    for argv, want in ((['--tier', 'snapshot', '--stages', 'omit', '--omit-runs', ','.join(runs), '--omit-refusals', rdir], 'done'),
                       (['--tier', 'snapshot', '--stages', 'omit', '--omit-runs', ','.join(runs), '--omit-refusals',
                         os.path.join(root, 'none')], 'refused')):
        a = p.parse_args(argv)
        assert DO.wants(a)
        out = _fresh('stage_out')
        run = record.Run('catalog', argv, out, CFG)
        with Patch((CFG.paths, 'SNAPSHOT_FLEET_OUT', proto)):
            step.run(a, run)
        st = [u.status for u in run.units]
        assert st == [want], (st, [u.reason for u in run.units])
        recp = os.path.join(out, '_records', 'catalog', 'Run_SNAPOMIT_T1A_T1B.omit.json')
        r = json.load(open(recp))
        assert r['status'] == want
        if want == 'done':
            assert {o['class'] for o in r['outputs']} == {'omission_decisions', 'omission_record'}
            assert all(t['passed'] for t in r['tests']) and any(t['name'] == 'omission_record.per_instant' for t in r['tests'])
            assert any(p_.startswith(rdir) for p_ in r['inputs']), 'the refusal files are inputs'
    assert DO.wants(p.parse_args(['--tier', 'sublst', '--stages', 'omit'])) and not DO.wants(p.parse_args(['--tier', 'sublst']))


def test_real_one_lst():
    """READ ONLY: the port's decisions = production's snapshot_deep_omission.decide for 07h on the served counts of B6H."""
    DO = _DO()
    SDO, PDO, _d = _prod()
    run = '/data07/gh/phase3/snapshot_search/review/Run_SNAPB6H_20261023'
    if not (os.path.isdir(run) and os.path.isdir(PDO.REFUSALS)):
        print('    SKIP: no served B6H or no refusal record on this host')
        return
    reader = DO.Reader()
    pending = DO.served_identities([run], reader)
    served, n = DO.served_counts(pending)
    t0 = time.time()
    SDO._NS.clear()
    SDO._WM.clear()
    prod = SDO.decide(['07h'], served)
    port = DO.decide_snapshot(['07h'], served, reader=DO.Reader())
    assert prod == port and prod, (len(prod), len(port))
    print(f'    07h on B6H ({n} served detections): {len(port)} decisions identical to production ({time.time() - t0:.0f} s)')


TESTS = [test_drift, test_snapshot, test_tenmin, test_gates, test_content, test_stage, test_real_one_lst]


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
