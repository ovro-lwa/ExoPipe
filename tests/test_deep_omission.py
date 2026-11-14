#!/usr/bin/env python3
"""STANDING TEST - the deep omission of the short tiers (lib/catalog/deep_omission.py; Gregg 2026-10-24,
R-GREGG-20261024-DEEP-OMISSION-DESIGN-APPROVED-AND-10MIN-HOLD-RELEASED).

  pin        production's ported files are the pinned ones (tests/production_pin.py; V_combview review F3): the only test
             production's later commits fail; every other test reads production at the pin
  drift      the port's rule and layout = production's modules at the pin (exopipe_ops sublst_deep_omission.py,
             snapshot_deep_omission.py, sublst_deep_omission_drop.py, snapshot_combined_view.py): the propagating gates, band
             groups, SIGMA, the refusal record, the window-rms table, the snapshot tiers and fleet root, every decision text, the
             cand_id and panel patterns
  snapshot   a synthetic fleet (two LSTs, eight nights, subbands and band groups, both Stokes and tiers, refusals of every gate
             kind, an older attempt, a stale_input row) and two served batch runs: production's own code (snapshot_deep_omission
             .decide and the combined view's step 0, run in this process, read only) and the port - the decision table byte for
             byte, the omitted detections and the dropped identities equal; hand-checked keys (the 18 MHz carve-out, member OMIT
             and ABSENT, the latest attempt, the Stokes scopes, ABSENT, fewer than 3 nights, rms and rate); the per-tier
             consequences (source dropped, partial re-pointed, member rows, panels) against the rule written out here
  tenmin     a synthetic window tree, searched lists, window-rms table, published LST runs and wideband windows (WBMEMB):
             production's sublst_deep_omission.decide (its roots patched) and the port byte for byte; the detections the merge's
             drop removes = production's sublst_deep_omission_drop.filter_rows on the member tables
  serve      `serve --tier sublst` build: the stage's table written outside the build, recorded, passed to the in-place merge
             as SUBLST_DEEP_OMISSION (the merge itself stubbed)
  serve_merge the same build with the REAL in-place merge (production's sublst_cumulative.py + sublst_deep_omission_drop.py,
             copied with their two data roots pointed at the fixture, run through core.legacy.run - its SUBLST_* clearing
             included): the merge drops what the stage predicts (its record = the stage's), the build's record registered and
             passing its content and per-instant tests; no /data07 needed (Q review F9)
  gates      G1 no record / no file / a foreign row, G2 a run without catalogue, G3 an identity without detections, G4 a source of
             no served identity, G4b a batch without served_tiers.csv or with a per-tier catalogue lost, G4c a kept source whose
             page holds no detection (V_combview review F4), G6 served detections in an ABSENT key ('refuse'; 'production' keeps
             and counts)
  content    omission_decisions and omission_record refuse a changed verdict, a member row without its member, a removal whose
             key is not OMIT, a why of another key; omission_record.per_instant refuses O1 (another night's cand_id), O2 (an
             undeclared reason), O3 (a member of another tier); the templates are the writer's own
  kept       the kept product deep_omission_kept.csv (Q review F2): its kept detections = production's step 0 clusters (the
             merge's input) exactly; omission_kept refuses a kept detection of an OMIT key, one the record omits, a missing or an
             extra per-tier row, a re-pointed source whose new best is not a kept detection of its identity, a count off the summary;
             and (Q fix verification N2) a detection key kept twice, a kept detection the served product never had or moved to
             another identity (the served-set digests of the summary), a summary without digests, a record detection row lost;
             (V_combview review F5) K5 / omission_served: the served counts the decisions were made with - a count changed, a key
             lost, the file missing
  stage      `exopipe3 catalog --tier snapshot --stages omit` and `--tier sublst --stages omit` through the step's run();
             --omit-refusals is an override of the RULED catalog_omission.REFUSALS (refused into a production root without
             --ruling-for, allowed and recorded with one or into a non-production --out; Q review F8)
  pending    production's two values RULED 2026-11-04 (RATE_FLOOR_PER_NIGHT, WIDEBAND_MEMBER_VERDICTS: R-GREGG-20261104-DEEP-
             OMISSION-SHORT-TIERS-ANSWERS) are not PROTECTED and their alternatives work (0 = the card's rule without a floor;
             ('OMIT',) = no ABSENT propagation); ABSENT_WITH_SERVED's 'refuse' is RULED (R-REFACTOR-20261104-ABSENT-WITH-SERVED-
             REFUSE) with production's value in the parity file and the WAVE2 G6 marks (Q review F3, F5); an invalid value
             refuses at --set through the real CLI, before the run (N3)
  prodbugs   production's bugs carried for parity, pinned as they are today (Q review F1: a wideband detection dropped for a member
             its own frame / window lacks; F4: deep_value / deep_threshold in the record's listing order, gates sorted, the
             meaning from the first sorted gate; F10: 'tier rms above threshold' without a threshold) - a fix shows here; the
             10-minute F1 world is also decided by PRODUCTION's sublst_deep_omission.decide, its table = the port's (N5)
  injrecord  an ordinary run refuses an injected run's refusal record however it is named (N1): run_snapshot, tenmin_table,
             the real CLI with --set catalog_omission.REFUSALS= and --omit-refusals, dry run included
  real       READ ONLY: the port's decisions = production's decide for one LST of the served batch B6H (served counts of B6H);
             SKIP (reported as SKIP, not PASS) on a host without the served batch

Scratch: /tmp/exopipe3_tests_deep_omission/ (removed at the end). Production trees are only read.
Run standalone:  python tests/test_deep_omission.py [test_name ...]
"""
import collections
import csv
import filecmp
import glob
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


# THE PRODUCTION PIN (V_combview review F3; tests/production_pin.py): the production modules this stage PORTS are read at the
# pinned exopipe_ops commit, so a later production commit never breaks the port's own tests - it fails test_production_pin alone,
# naming the commits to port. The pinned files are materialized under SCRATCH and imported from there (first on sys.path; exopipe_ops
# after them for what is not pinned).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import production_pin as PIN        # noqa: E402
import snapshot_chain_fixture       # noqa: E402
PINNED = ('snapshot_combined_view.py', 'snapshot_deep_omission.py', 'snapshot_rejudge.py', 'sublst_deep_omission.py',
          'sublst_deep_omission_drop.py', 'sublst_cumulative.py')
_PIN_DIR = []


def pinned_ops():
    """The directory holding the pinned production files (materialized once per test run, under SCRATCH), with the 10-minute
    builder build_source_catalogs_v3.py beside them as production has it (the drop's re-judging reads its THR there, edf3263;
    the builder runs in place: the live file)."""
    if not _PIN_DIR:
        d = PIN.materialize(os.path.join(SCRATCH, 'production_pin'), PINNED)
        shutil.copy(os.path.join(OPS, 'build_source_catalogs_v3.py'), d)
        _PIN_DIR.append(d)
    return _PIN_DIR[0]


def _prod():
    """production's modules at the pin (read only)."""
    pin = pinned_ops()
    if OPS not in sys.path:
        sys.path.insert(0, OPS)
    sys.path[:] = [pin] + [x for x in sys.path if x != pin]
    import snapshot_deep_omission as SDO
    import snapshot_rejudge as SRJ          # cached here: the view's source imports it after putting exopipe_ops first
    import sublst_deep_omission as PDO
    import sublst_deep_omission_drop as PDROP
    for m in (SDO, SRJ, PDO, PDROP):
        assert os.path.dirname(os.path.abspath(m.__file__)) == os.path.abspath(pin), (m.__name__, m.__file__)
    return SDO, PDO, PDROP


def _ops_src(name):
    """A pinned production file's text."""
    return open(os.path.join(pinned_ops(), name)).read()


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
            for sb in SUBBANDS:                       # the per-subband admission tables (dde1760's fail-closed rule reads them)
                if searched(lst, ni, sb):
                    _wcsv(os.path.join(cell, f'admission_{sb}.csv'), ['idx', 'ts', 'admitted'],
                          [dict(idx=k, ts=frame_ts(lst, night, k), admitted='1') for k in range(0, 60, 9)])
            if lst != '05h':                          # 06h: searched before the frame-time fix, no member table (fail closed)
                continue
            for S in ('I', 'V'):
                mem = [('Red', '18MHz'), ('Red', '23MHz'), ('Green', '55MHz')] + ([('Red', '27MHz')] if ni % 2 else []) \
                    + ([('Green', '46MHz')] if ni != 3 else [])
                rows = [dict(group=g, ts=frame_ts(lst, night, k), date_obs=date_obs(frame_ts(lst, night, k)), subband=sb)
                        for k in range(60) for g, sb in mem if not ((g, sb) == ('Green', '46MHz') and k % 5 == 0)]
                _wcsv(os.path.join(cell, f'wideband_members_{S}.csv'), ['group', 'ts', 'date_obs', 'subband'], rows)
    return proto


def frame_ts(lst, night, k):
    """The 10-s frame k of a cell (detection()'s ts)."""
    return '%s_%02d%02d%02d' % (night.replace('-', ''), int(lst[:2]), k // 6, (k % 6) * 10)


def date_obs(ts):
    """The DATE-OBS of a frame whose integration midpoint is ts (clv15 INTEGRATION_S / 2 before it)."""
    import datetime
    t = datetime.datetime.strptime(ts, '%Y%m%d_%H%M%S') - datetime.timedelta(seconds=5.0155102)
    return t.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]


DET_COLS = ['lst', 'date', 'band', 'subband', 'stokes', 'ts', 'snr', 'ra', 'dec', 'tier', 'instant', 'cand_id', 'ra_peak',
            'dec_peak']
BAND_CLASS = {'18MHz': 'Red', '23MHz': 'Red', '27MHz': 'Red', 'Red': 'Red', '46MHz': 'Green', '55MHz': 'Green', 'Green': 'Green'}


def detection(lst, night, sb, S, T, k, snr, rng, pos=None, elev_bin=None):
    """pos: (ra, dec) in deg (drawn from rng when absent); elev_bin: the chain record's (2 when absent; '-1' = the hour's first
    1-min block, re-measured by the 1-minute elevation floor, exopipe_ops b114ff3)."""
    snr = snr + 0.013 if min(abs(snr - 10.0), abs(snr - 8.0)) < 0.006 else snr   # clear of the record's rounding at a floor
    ts = '%s_%02d%02d%02d' % (night.replace('-', ''), int(lst[:2]), k // 6, (k % 6) * 10)
    if T == '1min':
        ts = ts[:-2] + '00'
    ra, dec = pos or (10 + rng.random(), 20 + rng.random())
    d = dict(lst=lst, date=night, band=BAND_CLASS[sb], subband=sb, stokes=S, ts=ts, snr='%.2f' % snr, ra='%.5f' % ra,
             dec='%.5f' % dec, tier=T, instant=ts[:13] if T == '1min' else ts, cand_id=f'{lst}_{night}_{sb}_{S}_{T}_{ts}',
             ra_peak='%.5f' % (ra + 0.001), dec_peak='%.5f' % dec)
    if elev_bin is not None:
        d['elev_bin'] = elev_bin
    return d


def batch_run(root, name, cells, rng, fixed=(), card2=()):
    """A served batch run: identities of 1-6 detections in its cells (plus ``fixed`` identities: (J, [detections args])),
    the general catalogue (source_catalogs/), the chain record, chain_summary.json and served_tiers.csv
    (tests/snapshot_chain_fixture.py: the chain's own route functions decide which (identity, tier) qualifies), the per-tier
    views of the QUALIFYING (identity, tier) (source_catalogs_<S>_<T>/: survivors, members), their page directories with
    detections.csv and one panel + page.json per detection, card2_vleak.csv (every 10-s Stokes V detection; ``card2``: the
    cand_ids Card 2 flags), FLUXSCAL_APPLIED.json with no scaled hour-band (production's flux-scale gate, exopipe_ops 5c49199)."""
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
        idents.append((J, [detection(*a[:7], rng, **(a[7] if len(a) > 7 else {})) for a in spec]))
    gen = []
    for J, dets in idents:
        rd = f'candidates/_identity/{J}'
        _wcsv(os.path.join(run, rd, 'detections.csv'), DET_COLS, dets)
        gen.append(dict(rank='rank0001', source_id=J, review_dir=rd, ra=dets[0]['ra'], dec=dets[0]['dec'],
                        best_snr=max(d['snr'] for d in dets)))
    qual = snapshot_chain_fixture.write_chain(run, idents, sorted({'%s_%s' % c for c in cells}))
    for J, dets in idents:
        by = collections.defaultdict(list)
        for d in dets:
            if qual.get((J, d['tier'])):
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
    card2_file(run, idents, card2)
    with open(os.path.join(run, 'FLUXSCAL_APPLIED.json'), 'w') as f:
        json.dump({'factors': {}}, f)
    return run


C2_COLS = ['det_uid', 'cand_id', 'ra_peak', 'dec_peak', 'identity_jname', 'stokes', 'tier', 'served', 'peak_jyb', 'i_here_jy',
           'v_over_i_here', 'card2_ratio', 'card2_would_remove']


def card2_file(run, idents, flagged):
    """card2_vleak.csv (phase3_snapshot_card2.py's columns the view reads): every 10-s Stokes V detection, by the chain record's
    det_uid, the ratio to 5 decimals and the flag at 0.05 (0.03 for the cand_ids in ``flagged``, 0.4 otherwise)."""
    rows = []
    tag = os.path.basename(run)
    for i, (J, dets) in enumerate(idents):
        for j, d in enumerate(dets):
            if (d['tier'], d['stokes']) != ('10s', 'V'):
                continue
            ratio = 0.03 if d['cand_id'] in flagged else 0.4
            pk = abs(float(d['snr'])) * 0.1
            rows.append(dict(det_uid='%s:%d:%d' % (tag, i, j), cand_id=d['cand_id'], ra_peak=d['ra_peak'], dec_peak=d['dec_peak'],
                             identity_jname=J, stokes='V', tier='10s', served='1', peak_jyb='%.4g' % pk,
                             i_here_jy='%.6g' % (pk / ratio), v_over_i_here='%.5f' % ratio, card2_ratio='0.05',
                             card2_would_remove='1' if ratio <= 0.05 else '0'))
    _wcsv(os.path.join(run, 'card2_vleak.csv'), C2_COLS, rows)


def write_served_tiers(run):
    """served_tiers.csv of a synthetic batch run: each general-catalogue identity 'served' in both tiers, qualifying in a tier
    exactly when it holds a per-tier source there (production's relation, G4b; V_combview review F4) - written from the run's
    own files, so a fixture that edits its catalogues afterwards calls it again."""
    import glob as _g
    tiers_of = collections.defaultdict(set)
    for T in ('10s', '1min'):
        for p in _g.glob(os.path.join(run, f'source_catalogs_[IV]_{T}', '*_sources_survivors.csv')):
            for r in _rows(p):
                tiers_of[r['identity_id']].add(T)
    ids = sorted({r['source_id'] for p in _g.glob(os.path.join(run, 'source_catalogs', '*_sources_survivors.csv'))
                  for r in _rows(p)} | set(tiers_of))
    _wcsv(os.path.join(run, 'served_tiers.csv'), ['identity_jname', 'chain_outcome', 'tier', 'qualifies'],
          [dict(identity_jname=J, chain_outcome='served', tier=T, qualifies='1' if T in tiers_of[J] else '0')
           for J in ids for T in ('10s', '1min')])


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
               ('J0000+0003', [('05h', NIGHTS[1], '46MHz', 'V', '1min', k, 7.0 + k) for k in range(40)]),
               # Card 2 (dbc40a6): one of the two 10-s Stokes V frames flagged - its member row and panel leave the view
               ('J0000+0004', [('05h', NIGHTS[2], '23MHz', 'V', '10s', 7, 12.0), ('05h', NIGHTS[2], '23MHz', 'V', '10s', 8, 11.0)]),
               # the 1-minute elevation floor (b114ff3): an elev_bin -1 1-min detection at dec -55 deg (below 30 deg at every
               # time: removed - its panel and member row leave the page, which keeps a later block's detection, re-pointed to
               # it) beside a kept 10-s one, and one at dec +89 deg (37 +- 1 deg: kept)
               ('J0000+0005', [('05h', NIGHTS[0], '23MHz', 'I', '10s', 20, 15.0, dict(pos=(10.5, -55.0))),
                               ('05h', NIGHTS[0], '23MHz', 'I', '1min', 21, 14.0, dict(pos=(10.5, -55.0), elev_bin='-1')),
                               ('05h', NIGHTS[0], '23MHz', 'I', '1min', 45, 12.5, dict(pos=(10.5, -55.0)))]),
               ('J0000+0006', [('05h', NIGHTS[0], '23MHz', 'I', '1min', 22, 13.0, dict(pos=(10.5, 89.0), elev_bin='-1'))])]
    flagged = {'05h_%s_23MHz_V_10s_%s' % (NIGHTS[2], frame_ts('05h', NIGHTS[2], 8))}
    runs = [batch_run(os.path.join(root, 'review'), 'Run_SNAPT1A_20261101', cells_a, rng, fixed_a, card2=flagged),
            batch_run(os.path.join(root, 'review'), 'Run_SNAPT1B_20261101', cells_b, rng)]
    return root, rdir, proto, runs


def prod_snapshot(runs, rdir, proto, outdir):
    """production's own code: the combined view's source up to its merge (step 0: identities, decide, drop) in this process
    with snapshot_deep_omission.decide's record and fleet roots given; returns (DEC, REC, n_all)."""
    SDO, _p, _d = _prod()
    orig = SDO.decide
    SDO.decide = lambda lsts, served, cells=None: orig(lsts, served, rdir=rdir, proto=proto, cells=cells)
    saved = (SDO.wb_image_members.__defaults__, SDO.wb_image_omitted.__defaults__)
    SDO.wb_image_members.__defaults__ = SDO.wb_image_omitted.__defaults__ = (proto,)   # the per-image member tables (dde1760)
    for c_ in (SDO._NS, SDO._WM, SDO._SA, SDO._WIM):
        c_.clear()
    try:
        src = _ops_src('snapshot_combined_view.py')
        g = {'__name__': 'prod_step0'}
        argv = sys.argv
        sys.argv = ['snapshot_combined_view.py', os.path.join(outdir, 'NEVER_WRITTEN')] + list(runs)
        try:
            exec(compile(src[:src.index("# ---- 2. the chain's merge")], 'snapshot_combined_view.py', 'exec'), g)
        finally:
            sys.argv = argv
    finally:
        SDO.decide = orig
        SDO.wb_image_members.__defaults__, SDO.wb_image_omitted.__defaults__ = saved
        for c_ in (SDO._NS, SDO._WM, SDO._SA, SDO._WIM):
            c_.clear()
    assert not os.path.exists(os.path.join(outdir, 'NEVER_WRITTEN'))
    SDO.write(os.path.join(outdir, 'prod_deep_omission.csv'), g['DEC'])
    kept = collections.Counter((g['TAG'][r], J) + tuple(g['dkey'](m['_row'])) for (r, J, _grs), c in zip(g['cinfo'], g['clusters'])
                               for m in c)         # step 0's kept identities and detections: the merge's input
    return g['DEC'], g['REC'], g['n_all'], kept


def _dec_by_key(rows):
    out = collections.defaultdict(list)
    for r in rows:
        out[(r['lst'], r['band'], r['night'], r['stokes'], r['tier'])].append(r)
    return out


# ---------------------------------------------------------------------------------------------------------------------

def test_production_pin():
    """THE EXPLICIT DRIFT TEST (V_combview review F3): the production files this port reproduces are the pinned ones
    (tests/production_pin.py). When production changes one, this test fails, naming the files and the commits to port; the
    port's own tests keep running against the pin. Moving the pin is the port of those commits."""
    d = PIN.drift(PINNED)
    assert not d, PIN.drift_message(d)


def test_drift():
    """The port's constants and texts are production's at the pin (the modules read as text / imported, nothing executed beyond
    import)."""
    SDO, PDO, PDROP = _prod()
    DO, C = _DO(), _cfg()
    c = C.catalog_omission
    assert tuple(c.PROPAGATE) == tuple(PDO.PROPAGATE), (c.PROPAGATE, PDO.PROPAGATE)
    assert {g: list(v) for g, v in C.instrument.SUBBAND_BANDS.items()} == PDO.GROUPS
    assert c.SIGMA == PDO.SIGMA and c.REFUSALS == PDO.REFUSALS and c.TENMIN_FRAME_RMS == PDO.FRAME_RMS
    assert tuple(c.SNAPSHOT_TIERS) == tuple(SDO.TIERS) and C.paths.SNAPSHOT_FLEET_OUT == SDO.PROTO
    assert os.path.join(C.paths.SUBLST_ROOT) == PDO.SL and C.paths.TRANSIENTS_ROOT == PDO.RUNS
    src_s = _ops_src('snapshot_deep_omission.py')
    src_t = _ops_src('sublst_deep_omission.py')
    src_d = _ops_src('sublst_deep_omission_drop.py')
    src_v = _ops_src('snapshot_combined_view.py')
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
    # the member reasons since dde1760: per image (the frame exception) and fail closed (no member table)
    flat_s = src_s.replace("'\n                                      '", '')
    assert ("'wideband-member: %s this night - omitted only in the %s images that include it (frame exception)'" in src_s and
            DO.snapshot_member_image_reason('%s', '%s') == 'wideband-member: %s this night - omitted only in the %s images that '
                                                          'include it (frame exception)')
    assert ('wideband-member: %s this night - the cell has no wideband member table (searched before 2026-10-07): every %s '
            'image of the night is taken to include it (fail closed)') in flat_s and \
        DO.snapshot_member_failclosed_reason('%s', '%s') == ('wideband-member: %s this night - the cell has no wideband member '
                                                             'table (searched before 2026-10-07): every %s image of the night is '
                                                             'taken to include it (fail closed)')
    assert DO.snapshot_member_reason('46MHz OMIT', 'Green') == ("wideband-member: %s this night, a member of the cell's %s frames"
                                                                 % ('46MHz OMIT', 'Green'))      # before dde1760 (old tables)
    # the gate pairing (dde1760, both tiers) and the rate population (dbc40a6)
    assert "gs = sorted(gates, key=lambda g: g[0])" in src_s and "gs = sorted(gates, key=lambda g: g[0])" in src_t
    assert "deep_value=';'.join(g[1] for g in gs), deep_threshold=';'.join(g[2] for g in gs)" in src_s
    assert 'inview = (lambda nt: True) if cells is None else (lambda nt: "%s_%s" % (lst, nt) in cells)' in src_s
    assert DO.tenmin_member_reason('46MHz OMIT') == 'wideband-member: %s this night, in the WBMEMB of its wideband windows' % '46MHz OMIT'
    assert "wideband-member: %s this night, in the WBMEMB of its wideband windows" in src_t
    # the patterns of keys: snapshot_combined_view CIDK / PANEL, sublst_deep_omission_drop CID
    assert DO.SNAP_CID.pattern in src_v and DO.SNAP_PANEL.pattern in src_v and DO.TENMIN_CID.pattern in src_d
    assert DO.MEMBER_SOURCE.pattern in src_v
    assert 'RATE FLOOR' in src_t and '1.0 / max(nwm, 1.0)' in src_t and c.RATE_FLOOR_PER_NIGHT == 1.0
    assert "x['band'] != '18MHz'" in src_t and "r['band'] != '18MHz'" in src_s and tuple(c.WIDEBAND_EXCEPT) == ('18MHz',)
    assert "x['verdict'] in ('OMIT', 'ABSENT')" in src_t and "r['verdict'] in ('OMIT', 'ABSENT')" in src_s
    assert tuple(c.WIDEBAND_MEMBER_VERDICTS) == ('OMIT', 'ABSENT')     # production's value (pending Gregg; Q review F3)


def test_snapshot():
    """The port = production's own code on a synthetic world; hand-checked keys; per-tier consequences."""
    DO = _DO()
    root, rdir, proto, runs = snapshot_world()
    out = os.path.join(root, 'port')
    summ = DO.run_snapshot(runs, out, rdir=rdir, proto=proto)
    DEC, REC, n_all, prod_kept = prod_snapshot(runs, rdir, proto, root)
    assert filecmp.cmp(os.path.join(out, 'deep_omission.csv'), os.path.join(root, 'prod_deep_omission.csv'), shallow=False), \
        'decision tables differ'
    rec = _rows(os.path.join(out, 'deep_omission_dropped.csv'))
    K = lambda r: (r['batch'], r['identity'], r['cand_id'], r['lst'], r['band'], r['night'], r['stokes'], r['tier'], r['snr'])
    port_det = collections.Counter(K(r) for r in rec if r['kind'] == 'detection')
    prod_det = collections.Counter(K(r) for r in REC if r['kind'] == 'detection')
    assert port_det == prod_det and sum(port_det.values()) == summ['omitted'] > 0, (len(port_det), len(prod_det))
    assert {(r['batch'], r['identity']) for r in rec if r['kind'] == 'identity dropped'} == \
        {(r['batch'], r['identity']) for r in REC if r['kind'] == 'identity dropped'}
    assert summ['served'] == n_all and summ['kept'] == n_all - summ['omitted'] - summ['card2'] - summ['elev1min']
    c2r = collections.Counter(K(r) for r in REC if r['kind'] == 'card2 detection')
    assert collections.Counter(K(r) for r in rec if r['kind'] == 'card2 detection') == c2r and sum(c2r.values()) == summ['card2'] > 0
    # the 1-minute elevation floor (b114ff3): J0000+0005's dec -55 deg 1-min detection removed, J0000+0006's dec +89 deg one kept
    elr = collections.Counter(K(r) for r in REC if r['kind'] == 'elevation-floor 1min detection')
    assert collections.Counter(K(r) for r in rec if r['kind'] == 'elevation-floor 1min detection') == elr and \
        sum(elr.values()) == summ['elev1min'] == 1, (elr, summ['elev1min'])
    assert {k[1] for k in elr} == {'J0000+0005'}, elr
    ec = summ['elev1min_counts']
    assert (ec['rows_1min_elev_bin_minus1'], ec['below_floor_records'], ec['below_floor_served'], ec['removed_from_view']) == \
        (2, 1, 1, 1) and ec['rows_1min'] > 2, ec
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
    dec_ = _rows(os.path.join(out, 'deep_omission.csv'))
    OM, OMWB = DO.omitted_keys(dec_), DO.omitted_wb(dec_)
    for r in rec:
        if r['kind'] in ('panel', 'detection') or (r['kind'] == 'member row' and r['lst']):
            k = (r['lst'], r['band'], r['night'], r['stokes'], r['tier'])
            assert k in OM or k in OMWB, r                                  # a whole-night key, or a per-image one (dde1760)
    assert any(k[1] in ('Red', 'Green') for k in OMWB), sorted(OMWB)[:3]        # the world has per-image member rows
    assert any(r['kind'] == 'detection' and (r['lst'], r['band'], r['night'], r['stokes'], r['tier']) in OMWB for r in rec)
    # the content tests of both classes and the per-instant test pass on the stage's products
    from exopipe.phase3.core import validate
    for cls, p in (('omission_decisions', 'deep_omission.csv'), ('omission_record', 'deep_omission_dropped.csv')):
        for t in validate.run(cls, os.path.join(out, p), tier='snapshot'):
            assert t['passed'], t
    pi = [t for t in validate.run('omission_record', os.path.join(out, 'deep_omission_dropped.csv')) if t['name'].endswith('per_instant')]
    assert pi and pi[0]['passed'] and pi[0]['value'] > 0, pi
    # the kept product (Q review F2): its kept detections ARE production's step 0 clusters, and its content test passes
    kp = _rows(os.path.join(out, 'deep_omission_kept.csv'))
    port_kept = collections.Counter((r['batch'], r['identity'], r['cand_id'], r['ra_peak'], r['dec_peak']) for r in kp
                                    if r['kind'] == 'detection kept')
    assert port_kept == prod_kept and sum(port_kept.values()) == summ['kept'] > 0, (sum(port_kept.values()), sum(prod_kept.values()))
    for t in validate.run('omission_kept', os.path.join(out, 'deep_omission_kept.csv'), tier='snapshot'):
        assert t['passed'] and t['value'] == summ['kept'], t
    rp = [r for r in kp if r['kind'] == 'source re-pointed' and r['identity'] == 'J0000+0002']
    # dbc40a6: the new best is the kept detection of largest |S/N| (cand_id / snr of the row, named in the note), the survivors
    # row's best_cand_id '<identity>_<lst>_<date>_<band>'
    assert rp and rp[0]['cand_id'] == part[0]['note'].split('re-pointed to ')[1].split(' ')[0], rp
    assert json.loads(rp[0]['fields'])['best_cand_id'] == 'J0000+0002_05h_%s_46MHz' % NIGHTS[1], rp


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
    set_job_rule(sl, 'envelope')            # searched under the refactor's default rule (G12 reads each job's size_rule.json)
    return root, rdir, sl, tr, os.path.join(root, 'page_context_frames.csv')


_ENV_REC = []


def _envelope_record():
    """The configured 10-minute envelope's run record (lib/search/size_rule.configured_record under 'envelope'; loaded once)."""
    if not _ENV_REC:
        from exopipe.phase3.lib.search import size_rule as SZ
        C = _cfg()
        with Patch((C.search_sublst, 'SIZE_RULE', 'envelope')):
            _ENV_REC.append(SZ.configured_record('10min'))
    return dict(_ENV_REC[0])


def set_job_rule(sl, rule, lsts=None):
    """Every 10-minute search job of the fixture (sl/search_B/<lst>_<band>_<S>/transients/Run_SLST<hh>B) searched under
    ``rule``: 'envelope' writes its size_rule.json (the engine's record), 'production' removes it (production writes none)."""
    from exopipe.phase3.lib.search import size_rule as SZ
    rec = _envelope_record() if rule == 'envelope' else None
    for d in glob.glob(os.path.join(sl, 'search_B', '*_*_[IV]', 'transients', 'Run_SLST*B')):
        if lsts and os.path.basename(os.path.dirname(os.path.dirname(d))).split('_')[0] not in lsts:
            continue
        p = os.path.join(d, SZ.RUN_RECORD)
        if rec is None:
            if os.path.exists(p):
                os.remove(p)
        else:
            with open(p, 'w') as f:
                json.dump(rec, f, indent=1, sort_keys=True)


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


def test_serve_build():
    """`serve --tier sublst` build: the decision table (this stage's) is written outside the build, recorded (class
    omission_decisions) and passed to the in-place merge as SUBLST_DEEP_OMISSION (sublst_cumulative.sh:98-100)."""
    import types
    from exopipe.phase3.core import record
    from exopipe.phase3.lib.serve import sublst_build
    DO, C = _DO(), _cfg()
    root, rdir, sl, tr, frms = tenmin_world('serve_build', seed=13)
    calls = []

    def fake_run(name, args=(), env=None, **kw):              # the merge is not run: its call is the test
        calls.append((name, list(args), dict(env or {})))
        return types.SimpleNamespace(ok=False, rc=1, tail='stopped by the test', reason='stopped by the test', refused=False)
    out = os.path.join(root, 'out')
    U = os.path.join(out, 'sublst', 'Run_SLSTB')
    plan = types.SimpleNamespace(run_name='Run_SLSTB', lsts=['05h', '06h'], parity=True, U=U, U_src=U, out=out)
    run = record.Run('serve', ['test_serve_build'], out, C)
    with Patch((C.paths, 'SUBLST_ROOT', sl), (C.paths, 'TRANSIENTS_ROOT', tr), (C.catalog_omission, 'REFUSALS', rdir),
               (C.catalog_omission, 'TENMIN_FRAME_RMS', frms), (C.catalog_omission, 'TENMIN_RMS_CACHE_SEED', ''),
               (sublst_build.scripts, 'run', fake_run), (sublst_build, 'generation_mix', lambda lsts: []),
               (sublst_build, 'lst_run', lambda l: os.path.join(tr, f'Run_SLST{l[:2]}B'))):
        # the fixture's LST runs are production builds (no grouping_table.json): under the design-B default the build refuses
        # them before the merge (G8 design-B grouping mix, item F.3) ...
        u0 = sublst_build.build_stage(record.Run('serve', ['test_serve_build_designb'], out, C), plan)
        assert u0.status == 'refused' and 'G8 design-B grouping mix' in u0.reason and not calls, (u0.status, u0.reason, calls)
        # ... and the parity plan runs production's grouping, as the parity file sets it (catalog_grouping.GROUPING =
        # MULTIPLES_RULE = 'production')
        with Patch((C.catalog_grouping, 'GROUPING', 'production'), (C.catalog_grouping, 'MULTIPLES_RULE', 'production')):
            u = sublst_build.build_stage(run, plan)
        DO.run_tenmin(plan.lsts, os.path.join(root, 'stage'), seed_dir='')
    assert u.status == 'refused' and 'merge failed' in u.reason, (u.status, u.reason)
    assert len(calls) == 1 and calls[0][0] == 'sublst_cumulative' and calls[0][1][1:] == plan.lsts, calls
    # the V3 round: the merge's drop judges a remaining single with the builder's gate under the configured 10-minute size rule -
    # the refactor's default 'envelope': P3_SIZE_GATE_7_8 = inf (no fixed gate); production's rule: nothing set (the drop's 1.7)
    assert calls[0][2].get('P3_SIZE_GATE_7_8') == 'inf' and u.meta['rejudge_size_gate']['rule'] == 'envelope', (calls[0][2], u.meta)
    with Patch((C.paths, 'SUBLST_ROOT', sl), (C.paths, 'TRANSIENTS_ROOT', tr), (C.catalog_omission, 'REFUSALS', rdir),
               (C.catalog_omission, 'TENMIN_FRAME_RMS', frms), (C.catalog_omission, 'TENMIN_RMS_CACHE_SEED', ''),
               (sublst_build.scripts, 'run', fake_run), (sublst_build, 'generation_mix', lambda lsts: []),
               (sublst_build, 'lst_run', lambda l: os.path.join(tr, f'Run_SLST{l[:2]}B')),
               (C.catalog_grouping, 'GROUPING', 'production'), (C.catalog_grouping, 'MULTIPLES_RULE', 'production'),
               (C.search_sublst, 'SIZE_RULE', 'production')):
        set_job_rule(sl, 'production')
        u2 = sublst_build.build_stage(record.Run('serve', ['test_serve_build_prodrule'], out, C), plan)
        set_job_rule(sl, 'envelope')
    assert len(calls) == 2 and 'P3_SIZE_GATE_7_8' not in calls[1][2] and u2.meta['rejudge_size_gate'] == dict(
        rule='production', gate_beams='1.7', env={}), (calls[1][2], u2.meta.get('rejudge_size_gate'))
    table = calls[0][2].get('SUBLST_DEEP_OMISSION', '')
    assert table.startswith(os.path.join(out, 'sublst', 'deep_omission', 'deep_omission_')) and table.endswith('Z.csv'), table
    assert filecmp.cmp(table, os.path.join(root, 'stage', 'deep_omission.csv'), shallow=False), 'the build table is the stage\'s'
    outs = {o['class']: o['path'] for o in u.outputs}
    assert outs.get('omission_decisions') == table and all(t['passed'] for t in u.tests), (outs, u.tests)
    assert os.path.exists(os.path.join(out, 'sublst', 'deep_omission', 'cache', 'window_rms_05h.csv'))


def merge_ready(tr, lsts):
    """What production's cumulative merge needs of each published LST run beyond tenmin_world's tables: master_catalog.csv
    with pos_kind, pertie_detections.csv (one row per detection), a survivors table per members table (one row per
    (source_id, band), identity_id, best = its first member), candidates/<S>/<page>/ with one block per member detection
    (rank0001_<lst>_<night>_bin<k>_<band>.png, the drop's BLOCK pattern)."""
    DO = _DO()
    for L in lsts:
        R = os.path.join(tr, f'Run_SLST{L[:2]}B')
        mc = _rows(os.path.join(R, 'master_catalog.csv'))
        _wcsv(os.path.join(R, 'master_catalog.csv'), ['cand_id', 'band', 'date', 'stokes', 'pos_kind', 'peak_jy'],
              [dict(r, pos_kind='fitted', peak_jy='1.0') for r in mc])
        _wcsv(os.path.join(R, 'pertie_detections.csv'), ['cand_id', 'tie_sys_arcsec'],
              [dict(cand_id=r['cand_id'], tie_sys_arcsec='30') for r in mc])
        for S in ('I', 'V'):
            os.makedirs(os.path.join(R, 'candidates', S), exist_ok=True)
            for T in ('sb', 'wb'):
                d = os.path.join(R, f'source_catalogs_{S}_{T}')
                for nm in sorted(os.listdir(d)):
                    if not nm.endswith('_source_members.csv'):
                        continue
                    band, mem = nm[:-len('_source_members.csv')], _rows(os.path.join(d, nm))
                    first = collections.OrderedDict()
                    for m in mem:
                        first.setdefault(m['source_id'], m)
                    _wcsv(os.path.join(d, f'{band}_sources_survivors.csv'),
                          ['source_id', 'band', 'identity_id', 'n_det', 'best_cand_id', 'best_snr'],
                          [dict(source_id=sid, band=band, identity_id=sid.split('_')[0], best_cand_id=m['cand_id'], best_snr='7',
                                n_det=sum(1 for x in mem if x['source_id'] == sid)) for sid, m in first.items()])
                    for m in mem:
                        c = DO.TENMIN_CID.match(m['cand_id'])
                        page = os.path.join(R, 'candidates', S, f"{m['source_id']}_{L}_{c.group(3)}_{band}")
                        os.makedirs(page, exist_ok=True)
                        b = int(m['cand_id'].split('T')[1][:2])
                        open(os.path.join(page, f'rank0001_{L}_{c.group(3)}_bin{b}_{band}.png'), 'w').close()


# the 10-minute RE-JUDGING world (edf3263; the V3 round): partial survivors in 05h's Stokes I 46 MHz table - one member on the OMIT
# key (05h 46 MHz 2025-06-01 I) and one kept (2025-06-07); (source, kept |S/N|, size, axis, promotion, xm_best)
REJUDGE_SOURCES = (('JR01_46MHz', 7.5, 1.85, 1.10, '', ''),        # a 7-8 sigma single above 1.7 beams: out under production only
                   ('JR02_46MHz', 6.5, 1.00, 1.00, '', ''),        # below the survivor threshold: no route (both rules)
                   ('JR03_46MHz', 9.5, 2.50, 1.00, '', ''),        # above 8 sigma: survives (both)
                   ('JR04_46MHz', 7.4, 1.20, 1.00, '', ''),        # a compact 7-8 sigma single: survives (both)
                   ('JR05_46MHz', 6.8, 1.00, 1.00, 'RPT', ''),     # a repeat promotion void with one member: no route (both)
                   ('JR06_46MHz', 7.5, 1.90, 1.00, '', 'gaia:1'),   # a cross-matched single: the gate does not apply (both)
                   ('JR08_46MHz', 7.5, 1.95, 1.00, '', '', 0.35))   # 1.95 beams less its 0.35 error = 1.60: survives (both;
                                                                    # exopipe_ops 40d6f42 - the raw size, compared before, removed it)
REJUDGE_OUT = {'production': {'JR01_46MHz', 'JR02_46MHz', 'JR05_46MHz'}, 'envelope': {'JR02_46MHz', 'JR05_46MHz'}}


def rejudge_ready(tr, lst='05h', extra=()):
    """Add REJUDGE_SOURCES (+ ``extra``) to the LST run of a merge_ready world: members, survivors (promotion, xm_best), master
    rows (size_beams, size_err_beams when given, axis_norm), page directories with their blocks."""
    R = os.path.join(tr, f'Run_SLST{lst[:2]}B')
    gone_n, keep_n = f'{NIGHTS[0]}T00', f'{NIGHTS[6]}T01'
    mp = os.path.join(R, 'master_catalog.csv')
    mc = _rows(mp)
    d = os.path.join(R, 'source_catalogs_I_sb')
    mem = _rows(os.path.join(d, '46MHz_source_members.csv'))
    sv = _rows(os.path.join(d, '46MHz_sources_survivors.csv'))
    for i, (sid, snr, sz, ax, promo, xm, *se) in enumerate(REJUDGE_SOURCES + tuple(extra)):
        g, k = f'I1_{lst}_{gone_n}_46MHz_{9000 + 2 * i}', f'I1_{lst}_{keep_n}_46MHz_{9001 + 2 * i}'
        mc += [dict(cand_id=g, band='46MHz', date=gone_n, stokes='I', pos_kind='fitted', peak_jy='1.0', size_beams='1.0',
                    axis_norm='1.0'),
               dict(cand_id=k, band='46MHz', date=keep_n, stokes='I', pos_kind='fitted', peak_jy='1.0', size_beams='%.2f' % sz,
                    size_err_beams='%.2f' % se[0] if se else '', axis_norm='%.2f' % ax)]
        mem += [dict(source_id=sid, cand_id=g, band='46MHz', det_snr='9.0'), dict(source_id=sid, cand_id=k, band='46MHz',
                                                                                  det_snr='%.2f' % snr)]
        sv.append(dict(source_id=sid, band='46MHz', identity_id=sid.split('_')[0], n_det=2, best_cand_id=g, best_snr='9.0',
                       promotion=promo, xm_best=xm))
        for c_, night in ((g, NIGHTS[0]), (k, NIGHTS[6])):
            page = os.path.join(R, 'candidates', 'I', f'{sid}_{lst}_{night}_46MHz')
            os.makedirs(page, exist_ok=True)
            b = int(c_.split('T')[1][:2])
            open(os.path.join(page, f'rank0001_{lst}_{night}_bin{b}_46MHz.png'), 'w').close()
    _wcsv(mp, ['cand_id', 'band', 'date', 'stokes', 'pos_kind', 'peak_jy', 'size_beams', 'size_err_beams', 'axis_norm'], mc)
    _wcsv(os.path.join(d, '46MHz_source_members.csv'), ['source_id', 'cand_id', 'band', 'det_snr'], mem)
    _wcsv(os.path.join(d, '46MHz_sources_survivors.csv'),
          ['source_id', 'band', 'identity_id', 'n_det', 'best_cand_id', 'best_snr', 'promotion', 'xm_best'], sv)


def prod_rejudge(PDROP, table, tr, lsts, gate, approved):
    """Production's drop (pinned) on the decision table: its re-judging (_ctx) per LST with SIZE_GATE_7_8 = ``gate`` (what
    P3_SIZE_GATE_7_8 sets; unset = 1.7), then filter_rows over every members table - (REJ, its 'detection' REC rows)."""
    with Patch((PDROP, 'SIZE_GATE_7_8', gate), (PDROP, 'APPROVED', approved)):
        PDROP.OM.clear()
        PDROP.WHY.clear()
        del PDROP.REC[:]
        PDROP.CNT.clear()
        PDROP._CTX.clear()
        PDROP.REJ.clear()
        PDROP.load(table)
        for L in lsts:
            R = os.path.join(tr, f'Run_SLST{L[:2]}B')
            PDROP._ctx(L, R)
            for S in ('I', 'V'):
                for T in ('sb', 'wb'):
                    dd = os.path.join(R, f'source_catalogs_{S}_{T}')
                    for nm in sorted(os.listdir(dd)):
                        if nm.endswith('_source_members.csv'):
                            PDROP.filter_rows(L, R, f'source_catalogs_{S}_{T}/{nm}', _rows(os.path.join(dd, nm)))
        return dict(PDROP.REJ), [dict(r) for r in PDROP.REC if r['kind'] == 'detection']


def test_tenmin_rejudge():
    """THE 10-MINUTE RE-JUDGING (exopipe_ops edf3263 + 35aad1a; the V3 round): on partial survivors the stage's prediction
    (run_tenmin: 'source re-judged' rows, the reasons) = production's drop (pinned sublst_deep_omission_drop: _ctx's REJ), for
    BOTH size rules - under 'envelope' (the refactor's default) the remaining single carries the rule's verdict and no fixed
    7-8 sigma gate acts (the drop given P3_SIZE_GATE_7_8 = inf, as the builder), under 'production' (the parity file) the drop's
    1.7 beams; the members of a re-judged-out source get no 'detection' row (production's filter_rows); the summary records the
    gate; G11 refuses without the go file; G10 refuses where survival rests on a statistic not re-derived (production too)."""
    DO, C = _DO(), _cfg()
    SDO, PDO, PDROP = _prod()
    root, rdir, sl, tr, frms = tenmin_world('rejudge', seed=11)
    lsts = ['05h', '06h']
    merge_ready(tr, lsts)
    rejudge_ready(tr)
    go = os.path.join(root, 'go.txt')
    open(go, 'w').write('the test go\n')
    for rule, gate, gate_txt in (('production', 1.7, '1.7'), ('envelope', float('inf'), 'inf')):
        out = os.path.join(root, f'stage_{rule}')
        set_job_rule(sl, rule)                      # the jobs searched under the rule the configuration applies (G12)
        with Patch((C.paths, 'SUBLST_ROOT', sl), (C.paths, 'TRANSIENTS_ROOT', tr), (C.search_sublst, 'SIZE_RULE', rule),
                   (C.catalog_omission, 'TENMIN_REJUDGE_APPROVED', go)):
            assert DO.tenmin_size_gate()[:2] == (gate, {'P3_SIZE_GATE_7_8': 'inf'} if rule == 'envelope' else {}), rule
            summ = DO.run_tenmin(lsts, out, rdir=rdir, seed_dir='', frame_rms=frms)
        rec = _rows(os.path.join(out, 'deep_omission_dropped.csv'))
        port = {(r['lst'], r['stokes'], r['view'], r['source_id'], r['band']): r['why'] for r in rec if r['kind'] == 'source re-judged'}
        REJ, pdet = prod_rejudge(PDROP, os.path.join(out, 'deep_omission.csv'), tr, lsts, gate, go)
        prod = {k: 're-judged on the members it keeps: %s' % v for k, v in REJ.items()}
        assert port == prod, (rule, sorted(set(port.items()) ^ set(prod.items()))[:4])
        assert {k[3] for k in port} == REJUDGE_OUT[rule], (rule, sorted(port))
        assert summ['rejudged'] == len(REJUDGE_OUT[rule]) and summ['rejudge_size_gate'] == dict(
            rule=rule, gate_beams=gate_txt, lst_rules={L: rule for L in lsts}), summ['rejudge_size_gate']
        K = lambda r: (r['lst'], r['stokes'], r['view'], r['source_id'], r['band'], r['cand_id'], r['night'], r['why'])
        got = collections.Counter(K(r) for r in rec if r['kind'] == 'detection')
        assert got == collections.Counter(K(r) for r in pdet), (rule, sum(got.values()), len(pdet))
        assert not [r for r in rec if r['kind'] == 'detection' and r['source_id'] in REJUDGE_OUT[rule]], rule
        assert any(r['kind'] == 'detection' and r['source_id'] == 'JR03_46MHz' for r in rec)    # a survivor's omitted member: listed
        from exopipe.phase3.core import validate
        for t in validate.run('omission_record', os.path.join(out, 'deep_omission_dropped.csv'), tier='sublst'):
            assert t['passed'], (rule, t)
    # the 7-8 sigma reason names production's gate and the measured size; under 'envelope' no size reason at all
    why1 = [r['why'] for r in _rows(os.path.join(root, 'stage_production', 'deep_omission_dropped.csv'))
            if r['kind'] == 'source re-judged' and r['source_id'] == 'JR01_46MHz']
    assert why1 and '7-8 sigma size/axis gate' in why1[0] and 'size 1.85' in why1[0] and '> 1.7' in why1[0], why1
    assert not [r for r in _rows(os.path.join(root, 'stage_envelope', 'deep_omission_dropped.csv')) if 'size/axis gate' in r['why']]
    # G12: the LSTs' search jobs must be searched under the configured rule - a job without the envelope's record (production's
    # search) under the default, or envelope records under production's rule, refuse; and an LST without any job under 'envelope'
    for rule, jobs_rule, lsts_off in (('envelope', 'production', ['06h']), ('production', 'envelope', None)):
        set_job_rule(sl, 'envelope')
        set_job_rule(sl, jobs_rule, lsts=lsts_off)
        with Patch((C.paths, 'SUBLST_ROOT', sl), (C.paths, 'TRANSIENTS_ROOT', tr), (C.search_sublst, 'SIZE_RULE', rule),
                   (C.catalog_omission, 'TENMIN_REJUDGE_APPROVED', go)):
            try:
                DO.run_tenmin(lsts, os.path.join(root, f'stage_g12_{rule}'), rdir=rdir, seed_dir='', frame_rms=frms)
                raise AssertionError('G12 did not refuse')
            except Exception as e:
                assert 'G12' in str(e) and ('06h' in str(e) if lsts_off else '05h' in str(e)), e
    set_job_rule(sl, 'envelope')
    bare = os.path.join(root, 'bare_sl')
    shutil.copytree(sl, bare, ignore=shutil.ignore_patterns('search_B'))
    with Patch((C.paths, 'SUBLST_ROOT', bare), (C.paths, 'TRANSIENTS_ROOT', tr), (C.catalog_omission, 'TENMIN_REJUDGE_APPROVED', go)):
        try:
            DO.tenmin_rule_check(lsts)
            raise AssertionError('G12 accepted an LST without search jobs under the envelope')
        except Exception as e:
            assert 'G12' in str(e) and 'no 10-minute search job' in str(e), e
    # G11: re-judging removes sources and no go is recorded - refused before anything is written
    with Patch((C.paths, 'SUBLST_ROOT', sl), (C.paths, 'TRANSIENTS_ROOT', tr),
               (C.catalog_omission, 'TENMIN_REJUDGE_APPROVED', os.path.join(root, 'no_go.txt'))):
        try:
            DO.run_tenmin(lsts, os.path.join(root, 'stage_nogo'), rdir=rdir, seed_dir='', frame_rms=frms)
            raise AssertionError('G11 did not refuse')
        except Exception as e:
            assert 'G11' in str(e), e
    assert not os.path.exists(os.path.join(root, 'stage_nogo', 'deep_omission_dropped.csv'))
    # G10: a cross-match E promotion rests on the members left - refused (production refuses the same)
    root2, rdir2, sl2, tr2, frms2 = tenmin_world('rejudge_g10', seed=11)
    merge_ready(tr2, lsts)
    rejudge_ready(tr2, extra=(('JR07_46MHz', 6.5, 1.0, 1.0, 'E:gaia', ''),))
    with Patch((C.paths, 'SUBLST_ROOT', sl2), (C.paths, 'TRANSIENTS_ROOT', tr2), (C.catalog_omission, 'TENMIN_REJUDGE_APPROVED', go)):
        try:
            DO.run_tenmin(lsts, os.path.join(root2, 'stage'), rdir=rdir2, seed_dir='', frame_rms=frms2)
            raise AssertionError('G10 did not refuse')
        except Exception as e:
            assert 'G10' in str(e) and 'JR07_46MHz' in str(e), e
    try:
        prod_rejudge(PDROP, os.path.join(root, 'stage_production', 'deep_omission.csv'), tr2, lsts, 1.7, go)
        raise AssertionError('production did not refuse')
    except SystemExit as e:
        assert 'REFUSING re-judging' in str(e) and 'E:gaia' in str(e), e


def prod_merge_copy(ops_dir, tr, sl):
    """production's sublst_cumulative.py with its two data roots (T, SL) pointed at the fixture - the one line rewritten,
    refused if production changed it - and sublst_deep_omission_drop.py verbatim, into ``ops_dir`` (the merge imports the drop
    and, since d1616d6, gate_inputs from its own directory)."""
    os.makedirs(ops_dir, exist_ok=True)
    src = _ops_src('sublst_cumulative.py')
    line = "T = '/data07/gh/phase3/transients'; SL = '/lustre/gh/main/phase3/sublst'"
    assert src.count(line) == 1, 'production changed the merge\'s root line: update the fixture rewrite'
    open(os.path.join(ops_dir, 'sublst_cumulative.py'), 'w').write(src.replace(line, f'T = {tr!r}; SL = {sl!r}'))
    shutil.copy(os.path.join(pinned_ops(), 'sublst_deep_omission_drop.py'), ops_dir)
    shutil.copy(os.path.join(pinned_ops(), 'build_source_catalogs_v3.py'), ops_dir)    # the drop's re-judging reads its THR (edf3263)
    # the merge's ghost-track refusal (exopipe_ops d1616d6) imports gate_inputs from its own directory: production's live file, run
    # in place by the port too (lib/catalog/tier_sublst.py), so not pinned - as the builder beside the pinned files
    shutil.copy(os.path.join(OPS, 'gate_inputs.py'), ops_dir)


def test_serve_build_merge():
    """Q review F9: `serve --tier sublst` build with the REAL in-place merge (production's code, its roots pointed at a tiny
    fixture, run through scripts.run / core.legacy.run - SUBLST_DEEP_OMISSION survives its SUBLST_* clearing): the merge drops
    exactly the detections the stage predicts, the build's record is registered (class omission_record) and passes its
    content and per-instant tests, no omitted row is left in the build, the LST runs keep their files. The build is stopped
    after the merge (build_localizations stubbed). The V3 round: partial survivors (REJUDGE_SOURCES) - the merge's own
    re-judging removes exactly the sources the stage predicts, under the refactor's default size rule ('envelope': the merge
    given P3_SIZE_GATE_7_8 = inf) and under production's (the parity file: 1.7 beams), JR01 only under the latter."""
    import types
    from exopipe.phase3.core import record
    from exopipe.phase3.lib.serve import sublst_build
    DO, C = _DO(), _cfg()
    root, rdir, sl, tr, frms = tenmin_world('serve_merge', seed=13)
    lsts = ['05h', '06h']
    merge_ready(tr, lsts)
    rejudge_ready(tr)
    ops_dir = os.path.join(root, 'ops')
    prod_merge_copy(ops_dir, tr, sl)
    real_run, calls = sublst_build.scripts.run, []

    def run_(name, args=(), **kw):
        calls.append(name)
        if name == 'sublst_cumulative':
            return real_run(name, args, **kw)
        return types.SimpleNamespace(ok=False, rc=1, tail='stopped by the test', reason='stopped by the test', refused=False)
    out = os.path.join(root, 'out')
    U = os.path.join(out, 'sublst', 'Run_SLSTB')
    plan = types.SimpleNamespace(run_name='Run_SLSTB', lsts=lsts, parity=True, U=U, U_src=U, out=out)
    run = record.Run('serve', ['test_serve_build_merge'], out, C)
    os.environ['SUBLST_DEEP_OMISSION'] = '/nonexistent/an_inherited_table.csv'     # cleared by core.legacy.run, then set
    try:
        with Patch((C.paths, 'SUBLST_ROOT', sl), (C.paths, 'TRANSIENTS_ROOT', tr), (C.catalog_omission, 'REFUSALS', rdir),
                   (C.catalog_omission, 'TENMIN_FRAME_RMS', frms), (C.catalog_omission, 'TENMIN_RMS_CACHE_SEED', ''),
                   (sublst_build.scripts, 'run', run_), (sublst_build.scripts, 'ops_root', lambda: ops_dir),
                   (sublst_build, 'lst_run', lambda l: os.path.join(tr, f'Run_SLST{l[:2]}B')),
                   # production-built LST runs, a parity plan: production's grouping, as the parity file sets it (item F.3)
                   (C.catalog_grouping, 'GROUPING', 'production'), (C.catalog_grouping, 'MULTIPLES_RULE', 'production')):
            u = sublst_build.build_stage(run, plan)
            summ = DO.run_tenmin(lsts, os.path.join(root, 'stage'), seed_dir='')
    finally:
        os.environ.pop('SUBLST_DEEP_OMISSION', None)
    B = sublst_build.build_dir(plan)
    assert calls == ['sublst_cumulative', 'build_localizations'], calls
    assert u.status == 'refused' and 'G9 localizations failed' in u.reason, (u.status, u.reason)
    table = [o['path'] for o in u.outputs if o['class'] == 'omission_decisions'][0]
    assert filecmp.cmp(table, os.path.join(B, 'deep_omission.csv'), shallow=False), 'the merge copied another table'
    K = lambda r: (r['lst'], r['stokes'], r['view'], r['source_id'], r['band'], r['cand_id'], r['night'], r['why'])
    merged = collections.Counter(K(r) for r in _rows(os.path.join(B, 'deep_omission_dropped.csv')) if r['kind'] == 'detection')
    stage = collections.Counter(K(r) for r in _rows(os.path.join(root, 'stage', 'deep_omission_dropped.csv'))
                                if r['kind'] == 'detection')
    assert merged == stage and sum(stage.values()) == summ['omitted'] > 0, (sum(merged.values()), sum(stage.values()))
    outs = {o['class']: o for o in u.outputs}
    assert outs['omission_record']['path'] == os.path.join(B, 'deep_omission_dropped.csv')
    names = {t['name'] for t in u.tests}
    assert {'omission_record.removals_follow_decisions', 'omission_record.per_instant'} <= names, names
    assert all(t['passed'] for t in u.tests), [t for t in u.tests if not t['passed']]
    OM4 = {k[:4] for k in DO.omitted_keys(_rows(table))}
    left = [r['cand_id'] for r in _rows(os.path.join(B, 'master_catalog.csv'))
            if (lambda m: m and (m.group(2), m.group(4), m.group(3), r['stokes']) in OM4)(DO.TENMIN_CID.match(r['cand_id']))]
    assert not left, f'omitted rows left in the build master_catalog: {left[:3]}'
    gone, kept_in_run = 0, 0
    for S in ('I', 'V'):
        for page in os.listdir(os.path.join(tr, 'Run_SLST05B', 'candidates', S)) + os.listdir(os.path.join(tr, 'Run_SLST06B', 'candidates', S)):
            for L in lsts:
                d = os.path.join(tr, f'Run_SLST{L[:2]}B', 'candidates', S, page)
                for fn in (os.listdir(d) if os.path.isdir(d) else []):
                    parts = fn[:-4].split('_')          # rank0001 <lst> <night> bin<k> <band>
                    if (parts[1], parts[4], parts[2], S) in OM4:
                        kept_in_run += 1
                        gone += not os.path.exists(os.path.join(B, 'candidates', S, page, fn))
    assert kept_in_run and gone == kept_in_run, (gone, kept_in_run)    # every omitted block left the build, none left a run
    # the re-judging under both size rules: the merge's own (in place) = the stage's prediction
    RJ = lambda p: {(r['lst'], r['stokes'], r['view'], r['source_id'], r['band']): r['why'] for r in _rows(p)
                    if r['kind'] == 'source re-judged'}
    assert RJ(os.path.join(B, 'deep_omission_dropped.csv')) == RJ(os.path.join(root, 'stage', 'deep_omission_dropped.csv'))
    assert {k[3] for k in RJ(os.path.join(B, 'deep_omission_dropped.csv'))} == REJUDGE_OUT['envelope']
    calls.clear()
    with Patch((C.paths, 'SUBLST_ROOT', sl), (C.paths, 'TRANSIENTS_ROOT', tr), (C.catalog_omission, 'REFUSALS', rdir),
               (C.catalog_omission, 'TENMIN_FRAME_RMS', frms), (C.catalog_omission, 'TENMIN_RMS_CACHE_SEED', ''),
               (sublst_build.scripts, 'run', run_), (sublst_build.scripts, 'ops_root', lambda: ops_dir),
               (sublst_build, 'lst_run', lambda l: os.path.join(tr, f'Run_SLST{l[:2]}B')),
               (C.catalog_grouping, 'GROUPING', 'production'), (C.catalog_grouping, 'MULTIPLES_RULE', 'production'),
               (C.search_sublst, 'SIZE_RULE', 'production')):
        set_job_rule(sl, 'production')
        u2 = sublst_build.build_stage(record.Run('serve', ['test_serve_build_merge_prodrule'], out, C), plan)
        DO.run_tenmin(lsts, os.path.join(root, 'stage_prodrule'), seed_dir='')
        set_job_rule(sl, 'envelope')
    assert u2.status == 'refused' and 'G9 localizations failed' in u2.reason, (u2.status, u2.reason)
    assert all(t['passed'] for t in u2.tests), [t for t in u2.tests if not t['passed']]
    got = RJ(os.path.join(B, 'deep_omission_dropped.csv'))
    assert got == RJ(os.path.join(root, 'stage_prodrule', 'deep_omission_dropped.csv')) and {k[3] for k in got} == \
        REJUDGE_OUT['production'], sorted(got)
    merged2 = collections.Counter(K(r) for r in _rows(os.path.join(B, 'deep_omission_dropped.csv')) if r['kind'] == 'detection')
    stage2 = collections.Counter(K(r) for r in _rows(os.path.join(root, 'stage_prodrule', 'deep_omission_dropped.csv'))
                                 if r['kind'] == 'detection')
    assert merged2 == stage2, (sum(merged2.values()), sum(stage2.values()))


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
    # G4b / G4c (V_combview review F4): incomplete batch inputs refuse, never booked as a deep omission nor lost unrecorded
    st = os.path.join(runs[0], 'served_tiers.csv')
    os.rename(st, st + '.x')
    refused(lambda: DO.run_snapshot(runs, os.path.join(root, 'o6b'), rdir=rdir, proto=proto), 'G4b: T1A has no served_tiers.csv')
    os.rename(st + '.x', st)
    cdir = os.path.join(runs[0], 'source_catalogs_I_10s')       # I3: a per-tier catalogue lost (survivors + members)
    lost = sorted(f for f in os.listdir(cdir) if f.endswith('_sources_survivors.csv'))
    keep = {f: open(os.path.join(cdir, f), 'rb').read() for f in os.listdir(cdir)}
    for f in os.listdir(cdir):
        os.remove(os.path.join(cdir, f))
    refused(lambda: DO.run_snapshot(runs, os.path.join(root, 'o6c'), rdir=rdir, proto=proto), 'G4b: T1A tier 10s')
    for f, b in keep.items():
        open(os.path.join(cdir, f), 'wb').write(b)
    assert lost and not os.path.exists(os.path.join(root, 'o6b')) and not os.path.exists(os.path.join(root, 'o6c'))
    # I2: a kept per-tier source's page detections.csv emptied (its member rows kept or not: G4c either way) - a source of an
    # identity the intact world keeps (a dropped identity's sources are dropped before their pages are read)
    DO.run_snapshot(runs, os.path.join(root, 'o6ok'), rdir=rdir, proto=proto)
    gone = {r['identity'] for r in _rows(os.path.join(root, 'o6ok', 'deep_omission_dropped.csv'))
            if r['kind'] == 'identity dropped' and r['batch'] == 'T1A'}
    src = next(r for f in lost for r in _rows(os.path.join(cdir, f)) if r['identity_id'] not in gone)
    dp = os.path.join(runs[0], src['review_dir'], 'detections.csv')
    keep_d = open(dp, 'rb').read()
    _wcsv(dp, DET_COLS, [])
    refused(lambda: DO.run_snapshot(runs, os.path.join(root, 'o6d'), rdir=rdir, proto=proto),
            f'G4c: T1A source_catalogs_I_10s source {src["source_id"]}: its page')
    open(dp, 'wb').write(keep_d)
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


def test_kept_product():
    """Q review F2: the kept product's content test (class omission_kept, K1-K4) refuses what the cumulative step must never
    apply: a kept detection of an OMIT key, a kept detection the record omits, a per-tier row missing or added against the
    record, a re-pointed source whose new best is not a kept detection of its identity, kept detections off the summary."""
    DO = _DO()
    from exopipe.phase3.lib.catalog import tests_deep_omission as TD
    root, rdir, proto, runs = snapshot_world('kept', seed=7)
    out = os.path.join(root, 'port')
    summ = DO.run_snapshot(runs, out, rdir=rdir, proto=proto)
    kp = os.path.join(out, 'deep_omission_kept.csv')
    ok, val, det = TD.check_kept(kp)
    assert ok and val == summ['kept'], det
    rows = _rows(kp)
    kinds = collections.Counter(r['kind'] for r in rows)
    assert set(kinds) <= set(DO.KEPT_KINDS) and {'detection kept', 'source dropped', 'source re-pointed', 'member row dropped',
                                                 'panel dropped', 'card2 member row dropped', 'card2 panel dropped',
                                                 'elevation-floor member row dropped', 'elevation-floor panel dropped',
                                                 'source routes'} <= set(kinds), kinds     # the re-judged kinds: test_snapshot_view
    rec = _rows(os.path.join(out, 'deep_omission_dropped.csv'))
    OM = DO.omitted_keys(_rows(os.path.join(out, 'deep_omission.csv')))

    def mutated(fix, want):
        d = os.path.join(root, 'mut_kept')
        shutil.rmtree(d, ignore_errors=True)
        shutil.copytree(out, d)
        rr = [dict(r) for r in rows]
        fix(rr)
        _wcsv(os.path.join(d, 'deep_omission_kept.csv'), DO.KEPT_COLS, rr)
        ok_, _v, det_ = TD.check_kept(os.path.join(d, 'deep_omission_kept.csv'))
        assert not ok_ and want in det_, (want, det_)

    def omit_key(rr):                                   # a kept detection moved onto an OMIT key
        k = sorted(OM)[0]
        r = next(r for r in rr if r['kind'] == 'detection kept')
        r.update(lst=k[0], band=k[1], night=k[2], stokes=k[3], tier=k[4])
    mutated(omit_key, 'is an OMIT key')
    det0 = next(r for r in rec if r['kind'] == 'detection')

    def resurrect(rr):                                  # an omitted detection listed as kept (its own key unchanged)
        r = next(r for r in rr if r['kind'] == 'detection kept' and r['batch'] == det0['batch'])
        r.update(identity=det0['identity'], cand_id=det0['cand_id'])
    mutated(resurrect, 'the record removes it')
    mutated(lambda rr: rr.remove(next(r for r in rr if r['kind'] == 'member row dropped')), 'member row dropped: 0 rows not in')
    mutated(lambda rr: rr.append(dict(next(r for r in rr if r['kind'] == 'panel dropped'))), 'panel dropped: 1 rows not in')

    def bad_best(rr):                                   # a re-pointed source pointing at a detection its identity does not keep
        r = next(r for r in rr if r['kind'] == 'source re-pointed')
        f = json.loads(r['fields'])
        f['best_cand_id'] = r['cand_id'] = r['cand_id'][:-1] + ('1' if r['cand_id'][-1] != '1' else '2')
        r['fields'] = json.dumps(f, sort_keys=True)
    mutated(bad_best, 'is no kept detection of its identity')
    mutated(lambda rr: rr.remove(next(r for r in rr if r['kind'] == 'detection kept')), 'K2 ')
    mutated(lambda rr: next(r for r in rr if r['kind'] == 'source dropped').update(catalog='source_catalogs_I_10s/x_source_members.csv'),
            'is not its tier')
    # Q fix verification N2: the three corruptions the test could not see (vrq_kept_gap.py) refuse now - a kept detection listed
    # twice in place of another (counts unchanged), one the served product never had, one moved to another kept identity
    kept_i = [i for i, r in enumerate(rows) if r['kind'] == 'detection kept']

    def swap_dup(rr):
        rr[kept_i[1]] = dict(rr[kept_i[0]])
    mutated(swap_dup, 'this detection key is kept twice')

    def swap_foreign(rr):
        r = rr[kept_i[0]]
        r['cand_id'] = r['cand_id'][:-2] + '99'
        r['ra_peak'] = '0.00001'
    mutated(swap_foreign, "is not the summary's kept digest")

    def swap_identity(rr):
        nid = collections.Counter((rr[i]['batch'], rr[i]['identity']) for i in kept_i)
        a = rr[next(i for i in kept_i if nid[(rr[i]['batch'], rr[i]['identity'])] >= 2)]
        a['identity'] = next(rr[i] for i in kept_i if rr[i]['batch'] == a['batch'] and rr[i]['identity'] != a['identity'])['identity']
    mutated(swap_identity, "is not the summary's kept digest")
    # the summary's digests: present, additive (kept + omitted = served), and required
    sj = json.load(open(os.path.join(out, 'deep_omission_summary.json')))
    dg = {k: DO.MultisetDigest(v) for k, v in sj['digests'].items()}
    assert dg['kept'] + dg['omitted'] + dg['card2'] + dg['elev1min'] == dg['served'] and dg['kept'].n == summ['kept'] and \
        dg['omitted'].n == summ['omitted'] and dg['card2'].n == summ['card2'] and dg['elev1min'].n == summ['elev1min'] == 1, sj
    d = os.path.join(root, 'mut_kept_nodigest')
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(out, d)
    json.dump(dict(sj, digests={}), open(os.path.join(d, 'deep_omission_summary.json'), 'w'))
    ok_, _v, det_ = TD.check_kept(os.path.join(d, 'deep_omission_kept.csv'))
    assert not ok_ and 'has no served / kept / omitted / card2 / elev1min digests' in det_, det_
    # the record's detection rows are the omitted detections of the digest: one removed refuses
    shutil.copy(os.path.join(out, 'deep_omission_summary.json'), os.path.join(d, 'deep_omission_summary.json'))
    rp_ = os.path.join(d, 'deep_omission_dropped.csv')
    rr_ = _rows(rp_)
    rr_.remove(next(r for r in rr_ if r['kind'] == 'detection'))
    _wcsv(rp_, DO.RECORD_COLS, rr_)
    ok_, _v, det_ = TD.check_kept(os.path.join(d, 'deep_omission_kept.csv'))
    assert not ok_ and "record's detection rows are not the omitted detections" in det_, det_
    # K5 / omission_served (V_combview review F5): the served counts the decisions were made with = the kept + omitted detections
    # of each key, summing to the summary's served; a count changed, a key lost, or the file missing refuses
    sv = os.path.join(out, 'deep_omission_served.csv')
    ok_, val_, det_ = TD.check_served(sv)
    assert ok_ and val_ == summ['served'] - summ['card2_served'] - summ['elev1min_served'] and summ['elev1min_served'] == 1, \
        det_    # dbc40a6 / b114ff3: Card 2 and below-floor 1-min detections not counted
    pend_ = DO.served_identities(runs, DO.Reader())
    assert DO.read_served(sv) == DO.served_counts(pend_, DO.Card2(runs), DO.Elev1min(runs))[0]   # less Card 2 and the floor
    srows = _rows(sv)
    for name, fix, want_k5, want_sv in (('count', lambda rr: rr[0].update(n=str(int(rr[0]['n']) + 1)), 'K5 1 served counts',
                                         'the summary\'s served'),
                                        ('lost', lambda rr: rr.pop(), 'K5 1 served counts', 'the summary\'s served'),
                                        ('missing', None, 'K5 no deep_omission_served.csv', None)):
        d = os.path.join(root, 'mut_served')
        shutil.rmtree(d, ignore_errors=True)
        shutil.copytree(out, d)
        if fix is None:
            os.remove(os.path.join(d, 'deep_omission_served.csv'))
        else:
            rr = [dict(r) for r in srows]
            fix(rr)
            _wcsv(os.path.join(d, 'deep_omission_served.csv'), DO.SERVED_COLS, rr)
            ok_, _v, det_ = TD.check_served(os.path.join(d, 'deep_omission_served.csv'))
            assert not ok_ and want_sv in det_, (name, det_)
        ok_, _v, det_ = TD.check_kept(os.path.join(d, 'deep_omission_kept.csv'))
        assert not ok_ and want_k5 in det_, (name, det_)


def test_pending_values():
    """Q review F3 / F5 and the 2026-11-04 decision. The rate floor and the ABSENT propagation to band groups were production's
    values pending Gregg; on 2026-11-04 Gregg delegated and the refactor chat decided, values unchanged
    (R-GREGG-20261104-DEEP-OMISSION-SHORT-TIERS-ANSWERS): both RULED under that row, not PROTECTED - their alternatives still
    work through the config into a non-production --out (into a production root only with --ruling-for) and change the
    decisions as stated (floor 0 = the card's median + 5 x 1.4826 MAD; ('OMIT',) = no ABSENT member). G6's 'refuse' is RULED by its
    own row (R-REFACTOR-20261104-ABSENT-WITH-SERVED-REFUSE), not PROTECTED; production's 'production' in the parity file, the code
    sites marked WAVE2 G6."""
    from exopipe.phase3.config import CFG, ConfigRefusal
    DO = _DO()
    c = CFG.catalog_omission
    for k in ('RATE_FLOOR_PER_NIGHT', 'WIDEBAND_MEMBER_VERDICTS'):
        assert CFG.ruled(f'catalog_omission.{k}') and 'R-GREGG-20261104-DEEP-OMISSION-SHORT-TIERS-ANSWERS' in \
            ((CFG.ruled(f'catalog_omission.{k}'),) if isinstance(CFG.ruled(f'catalog_omission.{k}'), str)
             else tuple(CFG.ruled(f'catalog_omission.{k}'))) and not CFG.protected(f'catalog_omission.{k}'), k
    assert CFG.ruled('catalog_omission.ABSENT_WITH_SERVED') == ('R-REFACTOR-20261104-ABSENT-WITH-SERVED-REFUSE',) and \
        not CFG.protected('catalog_omission.ABSENT_WITH_SERVED')            # ruled 2026-11-04, overridable (parity file 'production')
    for k in ('PROPAGATE', 'SIGMA', 'WIDEBAND_EXCEPT'):
        assert CFG.ruled(f'catalog_omission.{k}') and CFG.protected(f'catalog_omission.{k}'), k
    assert (c.RATE_FLOOR_PER_NIGHT, tuple(c.WIDEBAND_MEMBER_VERDICTS), c.ABSENT_WITH_SERVED) == (1.0, ('OMIT', 'ABSENT'), 'refuse')
    src_s = _ops_src('snapshot_deep_omission.py')
    src_t = _ops_src('sublst_deep_omission.py')
    assert "r['verdict'] in ('OMIT', 'ABSENT')" in src_s and "x['verdict'] in ('OMIT', 'ABSENT')" in src_t   # production's values
    root, rdir, proto, runs = snapshot_world('pending', seed=7)
    DO.run_snapshot(runs, os.path.join(root, 'base'), rdir=rdir, proto=proto)
    try:
        CFG.authorize(out=DO.default_out(type('A', (), {'tier': 'sublst'})()))   # a production root: refused without a ruling
        try:
            CFG.set('catalog_omission.RATE_FLOOR_PER_NIGHT', '0')
            raise AssertionError('a RULED key overridden into a production root without --ruling-for')
        except ConfigRefusal as e:
            assert 'carries a ruling' in str(e), e
        CFG.authorize(out=os.path.join(root, 'alt'))                              # a scratch --out: allowed, recorded
        CFG.set('catalog_omission.RATE_FLOOR_PER_NIGHT', '0')
        CFG.set('catalog_omission.WIDEBAND_MEMBER_VERDICTS', "('OMIT',)")
        assert all(str(o[4]).startswith('non-production out:') for o in CFG.overrides) and len(CFG.ruled_overrides()) == 2, \
            CFG.overrides
        DO.run_snapshot(runs, os.path.join(root, 'alt'), rdir=rdir, proto=proto)
        from exopipe.phase3.core import validate
        for cls, f in (('omission_decisions', 'deep_omission.csv'), ('omission_record', 'deep_omission_dropped.csv'),
                       ('omission_kept', 'deep_omission_kept.csv')):
            for t in validate.run(cls, os.path.join(root, 'alt', f), tier='snapshot'):
                assert t['passed'], t
        for bad_k, bad_v in (('WIDEBAND_MEMBER_VERDICTS', "('ABSENT',)"), ('RATE_FLOOR_PER_NIGHT', '-1.0')):
            CFG.set(f'catalog_omission.{bad_k}', bad_v)
            try:
                DO.member_verdicts() if bad_k == 'WIDEBAND_MEMBER_VERDICTS' else DO.rate_floor()
                raise AssertionError(f'{bad_k}={bad_v} accepted')
            except ValueError as e:
                assert bad_k in str(e), e
            assert any(bad_k in p for p in DO.check_values()), DO.check_values()      # N3: named before any run
    finally:
        CFG.reset()
        CFG.authorize()
    assert DO.check_values() == [], DO.check_values()
    # Q fix verification N3: through the real CLI an invalid value REFUSES at --set (exit 2, 'invalid value', before the run) -
    # it no longer starts a run that stops late with an error; the floor is refused without any cell of 3 nights
    for bad_k, bad_v in (('WIDEBAND_MEMBER_VERDICTS', "('ABSENT',)"), ('RATE_FLOOR_PER_NIGHT', '-1.0'),
                         ('ABSENT_WITH_SERVED', 'keep')):
        r = _cli('catalog', '--tier', 'snapshot', '--stages', 'omit', '--omit-runs', ','.join(runs), '--omit-refusals', rdir,
                 '--set', f'catalog_omission.{bad_k}={bad_v}', '--out', os.path.join(root, 'n3_out'), '--cores', 'none')
        assert r.returncode == 2 and 'invalid value' in r.stderr and bad_k in r.stderr, (bad_k, r.returncode, r.stderr[-500:])
        assert not os.path.exists(os.path.join(root, 'n3_out', 'deep_omission.csv')), 'a run started'
    B, A = (_dec_by_key(_rows(os.path.join(root, x, 'deep_omission.csv'))) for x in ('base', 'alt'))
    member = lambda D, k: [r for r in D[k] if r['gates'].startswith('member ')]
    g8 = ('05h', 'Green', NIGHTS[7], 'I', '10s')
    assert member(B, g8) and '55MHz ABSENT' in member(B, g8)[0]['gates'] and not member(A, g8)    # ABSENT no longer propagates
    assert all('ABSENT' not in r['gates'] for k in A for r in member(A, k)) and any(member(A, k) for k in A)   # OMIT still does
    lower = strict = 0
    for k in B:
        for rb, ra in zip([r for r in B[k] if not r['gates'].startswith('member ')], [r for r in A[k] if not r['gates'].startswith('member ')]):
            fb, fa = DO_float(rb['rate_threshold']), DO_float(ra['rate_threshold'])
            if fb is not None and fa is not None:
                assert fa <= fb * (1 + 1e-3), (k, rb['rate_threshold'], ra['rate_threshold'])     # the floor only raises it
                lower += 1
                strict += fa < fb * (1 - 1e-3)
    assert lower and strict, (lower, strict)
    # F5: production's G6 value in the parity file, the code sites marked
    par = open(os.path.join(REPO, 'exopipe', 'phase3', 'parity', 'production_20261013.cfg')).read().splitlines()
    i = par.index("catalog_omission.ABSENT_WITH_SERVED = 'production'")
    assert any(ln.startswith('# WAVE2 G6') for ln in par[max(0, i - 3):i]), par[max(0, i - 3):i + 1]
    do_src = open(os.path.join(REPO, 'exopipe', 'phase3', 'lib', 'catalog', 'deep_omission.py')).read()
    sb_src = open(os.path.join(REPO, 'exopipe', 'phase3', 'lib', 'serve', 'sublst_build.py')).read()
    assert "if bad and mode == 'refuse':    # WAVE2 G6" in do_src and '# WAVE2 G6' in sb_src


def DO_float(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def _pin_world():
    """A fleet (fleet()), a refusal record with two-gate keys, and one served batch run whose fixed identities hold Green I
    10-s detections of 05h night 1 on two frames: frame A of 46 + 55 MHz, frame B of 55 MHz only (wideband_members_I.csv of
    the cell, by ts)."""
    root = _fresh('pins')
    rng = random.Random(23)
    proto = fleet(os.path.join(root, 'fleet'), rng)
    rdir = os.path.join(root, 'refusals')
    os.makedirs(rdir)
    files = {'05h_46MHz_searchtier': [['05h', '46MHz', NIGHTS[0], 'I', '9.1', '4.5', TAG]],      # 10-s: OMIT (the noisy night)
             '05h_27MHz_searchtier': [['05h', '27MHz', NIGHTS[1], 'I', '2895.8', '166.5', TAG]],  # F4: two gates of one key
             '05h_27MHz_beam': [['05h', '27MHz', NIGHTS[1], '3.96', '1.5', TAG]],
             '05h_23MHz_sanity': [['05h', '23MHz', NIGHTS[2], '40.1', '3.3', TAG]],               # F4: 'beam;sanity'
             '05h_23MHz_beam': [['05h', '23MHz', NIGHTS[2], '3.73', '1.5', TAG]],
             '06h_27MHz_searchtier': [['06h', '27MHz', NIGHTS[1], 'I', '5.5', '2.0', TAG]]}       # F10: two nights only
    for name, rows in files.items():
        with open(os.path.join(rdir, name + '_excluded.csv'), 'w', newline='') as f:
            csv.writer(f).writerows(rows)
    ts = lambda k: detection('05h', NIGHTS[0], 'Green', 'I', '10s', k, 9.0, random.Random(0))['ts']
    _wcsv(os.path.join(proto, f'05h_{NIGHTS[0]}', 'wideband_members_I.csv'), ['group', 'ts', 'subband'],
          [dict(group='Green', ts=ts(1), subband='46MHz'), dict(group='Green', ts=ts(1), subband='55MHz'),
           dict(group='Green', ts=ts(2), subband='55MHz')])
    fixed = [('J0000+0101', [('05h', NIGHTS[0], 'Green', 'I', '10s', 1, 11.0)]),          # frame A (46 + 55 MHz)
             ('J0000+0102', [('05h', NIGHTS[0], 'Green', 'I', '10s', 2, 12.0)]),          # frame B (55 MHz only)
             ('J0000+0103', [('06h', NIGHTS[0], '46MHz', 'I', '10s', 3, 8.0)])]           # 06h served: its LST is decided
    runs = [batch_run(os.path.join(root, 'review'), 'Run_SNAPP1A_20261101', [('05h', n) for n in NIGHTS[1:4]], rng, fixed)]
    return root, rdir, proto, runs, ts


def test_production_bugs_pinned():
    """Production's bugs the port carries or carried for parity (Q review F1, F4, F10), pinned AS THEY ARE TODAY - production's
    own code agrees on every pinned case. Production fixed F1 for the 10-s / 1-minute tiers (per image) and F4 (gate pairing,
    both tiers) in exopipe_ops dde1760, and the port followed (the V2 follow-up round): those asserts changed on purpose; the
    10-minute F1 and F10 stand (dde1760 left them; the review's T9: production first)."""
    DO = _DO()
    SDO, PDO, PDROP = _prod()
    root, rdir, proto, runs, ts = _pin_world()
    out = os.path.join(root, 'port')
    DO.run_snapshot(runs, out, rdir=rdir, proto=proto)
    DEC, REC, _n, _k = prod_snapshot(runs, rdir, proto, root)
    assert filecmp.cmp(os.path.join(out, 'deep_omission.csv'), os.path.join(root, 'prod_deep_omission.csv'), shallow=False)
    D = _dec_by_key(_rows(os.path.join(out, 'deep_omission.csv')))
    rec = _rows(os.path.join(out, 'deep_omission_dropped.csv'))
    # F1 FIXED (10-s / 1-minute, dde1760): 46 MHz night 1 is OMIT at 10 s; the Green row is per image - the detection of frame A
    # (46 + 55 MHz) is omitted, the one of frame B (55 MHz only) is KEPT, in production and in the port
    assert D[('05h', '46MHz', NIGHTS[0], 'I', '10s')][0]['verdict'] == 'OMIT'
    m = [r for r in D[('05h', 'Green', NIGHTS[0], 'I', '10s')] if r['gates'].startswith('member ')]
    assert m and m[0]['gates'] == 'member 46MHz OMIT' and m[0]['wb_omitted_members'] == '46MHz', m
    assert m[0]['reason'] == DO.snapshot_member_image_reason('46MHz OMIT', 'Green'), m
    om = {(r['identity'], r['cand_id']) for r in rec if r['kind'] == 'detection'}
    frame_a, frame_b = f'05h_{NIGHTS[0]}_Green_I_10s_{ts(1)}', f'05h_{NIGHTS[0]}_Green_I_10s_{ts(2)}'
    assert ('J0000+0101', frame_a) in om and ('J0000+0102', frame_b) not in om, sorted(om)[:4]
    prod_om = {(r['identity'], r['cand_id']) for r in REC if r['kind'] == 'detection'}
    assert ('J0000+0101', frame_a) in prod_om and ('J0000+0102', frame_b) not in prod_om       # production too
    # F4 FIXED (dde1760): names, values and thresholds in ONE order - the gates sorted by name, stably; whatever the listing
    r27 = [r for r in D[('05h', '27MHz', NIGHTS[1], 'I', '10s')] if not r['gates'].startswith('member ')][0]
    assert (r27['gates'], r27['deep_value'], r27['deep_threshold']) == ('beam;searchtier', '3.96;2895.8', '1.5;166.5'), r27
    for listing in (lambda d, l: sorted(glob_(d, l), reverse=True), lambda d, l: sorted(glob_(d, l))):
        with Patch((DO, 'refusal_files', listing)):
            g = DO.load_refusals(rdir, '05h')[('27MHz', NIGHTS[1], 'I')]
            b = DO._base('05h', '27MHz', NIGHTS[1], 'I', g)
        assert (b['gates'], b['deep_value'], b['deep_threshold']) == ('beam;searchtier', '3.96;2895.8', '1.5;166.5'), b
    r23 = [r for r in D[('05h', '23MHz', NIGHTS[2], 'I', '10s')] if not r['gates'].startswith('member ')][0]
    assert r23['gates'] == 'beam;sanity' and r23['deep_threshold_meaning'] == DO.GATE_MEANING, r23   # sanity's median, read
    # F10: fewer than 3 nights - no threshold, never passed, and the reason still 'tier rms above threshold'      # as a threshold
    r06 = [r for r in D[('06h', '27MHz', NIGHTS[1], 'I', '10s')] if not r['gates'].startswith('member ')][0]
    assert (r06['verdict'], r06['tier_rms_threshold'], r06['rms_pass'], r06['reason']) == ('OMIT', 'nan', '0', DO.RMS_ABOVE), r06
    # F1 (10-minute): ONE window of the night whose WBMEMB holds the omitted subband omits the group's whole night - the Green
    # detection of a window stacked without 46 MHz is dropped too (production's merge drop agrees)
    from astropy.io import fits
    import numpy as np
    root2, rdir2, sl, tr, frms = tenmin_world('pins_tenmin', seed=11)
    w1 = os.path.join(sl, 'tenmin_root_x2', '05h_Green', 'subtracted', f'{NIGHTS[0]}T01_I_sub_Green.fits')
    h = fits.Header()
    h['WBMEMB'] = '55MHz'
    fits.writeto(w1, np.zeros((2, 2), 'float32'), h, overwrite=True)
    st = os.stat(w1)
    fr = _rows(frms)
    for r in fr:
        if r['path'] == w1:
            r.update(size=st.st_size, mtime=int(st.st_mtime))
    _wcsv(frms, ['path', 'size', 'mtime', 'rms'], fr)
    R = os.path.join(tr, 'Run_SLST05B')
    cid = f'I1_05h_{NIGHTS[0]}T01_Green_9001'
    mc = _rows(os.path.join(R, 'master_catalog.csv'))
    _wcsv(os.path.join(R, 'master_catalog.csv'), ['cand_id', 'band', 'date', 'stokes'],
          mc + [dict(cand_id=cid, band='Green', date=f'{NIGHTS[0]}T01', stokes='I')])
    mp = os.path.join(R, 'source_catalogs_I_wb', 'Green_source_members.csv')
    _wcsv(mp, ['source_id', 'cand_id', 'band', 'det_snr'], _rows(mp) + [dict(source_id='J9001_Green', cand_id=cid, band='Green',
                                                                            det_snr='9')])
    C = _cfg()
    with Patch((C.paths, 'SUBLST_ROOT', sl), (C.paths, 'TRANSIENTS_ROOT', tr)):
        DO.run_tenmin(['05h'], os.path.join(root2, 'port'), rdir=rdir2, seed_dir='', frame_rms=frms)
    T = _dec_by_key([dict(r, tier='10min') for r in _rows(os.path.join(root2, 'port', 'deep_omission.csv'))])
    g = [r for r in T[('05h', 'Green', NIGHTS[0], 'I', '10min')] if r['gates'].startswith('member ')]
    assert g and '46MHz OMIT' in g[0]['gates'], g
    assert cid in {r['cand_id'] for r in _rows(os.path.join(root2, 'port', 'deep_omission_dropped.csv'))}
    # Q fix verification N5: PRODUCTION'S OWN decide on this pinned world (its roots patched, as test_tenmin) - its table equals
    # the port's byte for byte and holds the same Green member row, and production's merge drop on PRODUCTION's table omits the
    # window-without-46-MHz detection; a production fix of F1 inside sublst_deep_omission.decide fails here
    os.makedirs(os.path.join(root2, 'prodcache'))
    with Patch((PDO, 'SL', sl), (PDO, 'RUNS', tr), (PDO, 'FRAME_RMS', frms), (PDO, '_FR', None), (PDO, '_WBW', {}), (PDO, '_WBM', {})):
        prows = PDO.decide('05h', rdir2, os.path.join(root2, 'prodcache'))
    for r in prows:
        r['rate_definition'] = DO.TENMIN_RATE_DEF
    with open(os.path.join(root2, 'prod.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(prows[0]))
        w.writeheader()
        w.writerows(prows)
    assert filecmp.cmp(os.path.join(root2, 'prod.csv'), os.path.join(root2, 'port', 'deep_omission.csv'), shallow=False), \
        'the pinned 10-minute world: production\'s decide table differs from the port\'s'
    gp = [r for r in prows if (r['band'], r['night'], r['stokes']) == ('Green', NIGHTS[0], 'I') and r['gates'].startswith('member ')]
    assert gp and '46MHz OMIT' in gp[0]['gates'] and gp[0]['verdict'] == 'OMIT', gp
    PDROP.OM.clear()
    PDROP.WHY.clear()
    PDROP.load(os.path.join(root2, 'prod.csv'))
    assert PDROP.omitted(cid, 'I'), 'production\'s merge drop on production\'s own table keeps it'
    PDROP.OM.clear()
    PDROP.WHY.clear()
    PDROP.load(os.path.join(root2, 'port', 'deep_omission.csv'))
    assert PDROP.omitted(cid, 'I'), 'production\'s merge drop keeps it'


def glob_(rdir, lst):
    import glob as _g
    return _g.glob(os.path.join(rdir, '%s_*_excluded.csv' % lst))


def _cli(*args, timeout=900):
    """The real CLI (python -m exopipe.phase3) of this tree, an ordinary run (no inherited injected flag)."""
    import subprocess
    e = {k: v for k, v in os.environ.items() if k not in ('EXOPIPE3_INJECTED', 'P3_SNAP_INJECTED', 'EXOPIPE3_PARITY_ROOT')}
    e['PYTHONPATH'] = REPO + (os.pathsep + e['PYTHONPATH'] if e.get('PYTHONPATH') else '')
    return subprocess.run([sys.executable, '-m', 'exopipe.phase3'] + [str(a) for a in args], cwd=REPO, env=e,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=timeout)


def _inject_mark(root):
    """An injected run's start record under ``root`` (core/record.INJECTED_DIR, as cli writes it): the products under ``root``
    carry synthetic sources."""
    from exopipe.phase3.core import record
    p = os.path.join(root, '_records', record.INJECTED_DIR, 'search_20261103T000000Z_test_1.json')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w') as f:
        json.dump({'run_id': 'search_20261103T000000Z_test_1', 'step': 'search', 'injected': 'test: an injected run'}, f)
    return p


def test_injected_record():
    """Q fix verification N1: an ordinary run never takes its decisions from an injected run's refusal record, however the record
    is named - run_snapshot and tenmin_table (every caller: the omit stage, the 10-minute cumulative build) refuse it, and the
    real CLI refuses `--set catalog_omission.REFUSALS=<it>` (the config route the flag check could not see) as it refuses
    `--omit-refusals <it>`, dry run included; an injected run may read it."""
    from exopipe.phase3.core import injected
    from exopipe.phase3.core.record import Refusal
    DO = _DO()
    root, rdir, proto, runs = snapshot_world('inj_record', seed=7)
    inj = os.path.join(root, 'injout')
    _inject_mark(inj)
    irdir = os.path.join(inj, 'refusals')
    shutil.copytree(rdir, irdir)
    for what, fn in (('run_snapshot', lambda: DO.run_snapshot(runs, os.path.join(root, 'o1'), rdir=irdir, proto=proto)),
                     ('tenmin_table', lambda: DO.tenmin_table(['05h'], os.path.join(root, 'o2.csv'), rdir=irdir))):
        try:
            fn()
            raise AssertionError(f'{what}: an injected run\'s refusal record was read')
        except Refusal as e:
            assert 'G1 INJECTED-RUN GUARD' in str(e) and irdir in str(e), (what, str(e))
    assert not os.path.exists(os.path.join(root, 'o1')) and not os.path.exists(os.path.join(root, 'o2.csv'))
    saved = os.environ.get(injected.ENV)
    os.environ[injected.ENV] = 'test: an injected consumer'
    try:
        DO.injected_record_gate(irdir)            # an injected run may read an injected run's record
    finally:
        if saved is None:
            os.environ.pop(injected.ENV, None)
        else:
            os.environ[injected.ENV] = saved
    DO.injected_record_gate(rdir)                 # an ordinary record, an ordinary run
    for extra in (['--set', f'catalog_omission.REFUSALS={irdir}'], ['--omit-refusals', irdir]):
        for dry in (['--dry-run'], []):
            r = _cli('catalog', '--tier', 'snapshot', '--stages', 'omit', '--omit-runs', ','.join(runs), *extra,
                     '--out', os.path.join(root, 'cli_out'), '--cores', 'none', *dry)
            assert r.returncode == 2 and 'INJECTED-RUN GUARD' in r.stderr, (extra, dry, r.returncode, r.stderr[-500:])
    assert not os.path.exists(os.path.join(root, 'cli_out', 'deep_omission.csv'))


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
        CFG.authorize(out=out)                  # as the CLI: --omit-refusals overrides a RULED key into a non-production --out
        run = record.Run('catalog', argv, out, CFG)
        try:
            with Patch((CFG.paths, 'SNAPSHOT_FLEET_OUT', proto)):
                step.run(a, run)
            hdr = run.header()
        finally:
            CFG.reset()
            CFG.authorize()
        st = [u.status for u in run.units]
        assert st == [want], (st, [u.reason for u in run.units])
        recp = os.path.join(out, '_records', 'catalog', 'Run_SNAPOMIT_T1A_T1B.omit.json')
        r = json.load(open(recp))
        assert r['status'] == want
        ro = [o for o in hdr['ruled_overrides'] if o['key'] == 'catalog_omission.REFUSALS']
        assert ro and ro[0]['authority'].startswith('non-production out:'), hdr['ruled_overrides']   # Q review F8: recorded
        if want == 'done':
            assert {o['class'] for o in r['outputs']} == {'omission_decisions', 'omission_record', 'omission_served',
                                                           'omission_rejudged', 'omission_kept'}
            assert any(t['name'] == 'omission_served.served_counts_add_up' and t['passed'] for t in r['tests'])
            assert all(t['passed'] for t in r['tests']) and any(t['name'] == 'omission_record.per_instant' for t in r['tests'])
            assert any(t['name'].startswith('omission_kept.') for t in r['tests'])
            assert any(p_.startswith(rdir) for p_ in r['inputs']), 'the refusal files are inputs'
    assert DO.wants(p.parse_args(['--tier', 'sublst', '--stages', 'omit'])) and not DO.wants(p.parse_args(['--tier', 'sublst']))
    assert CFG.catalog_omission.REFUSALS == DO._cfg().catalog_omission.REFUSALS and not CFG.overrides


def test_refusals_override():
    """Q review F8: --omit-refusals is an override of the RULED catalog_omission.REFUSALS through the config - refused into a
    production root without --ruling-for (a run-level refusal, the config unchanged), allowed with --ruling-for (authority
    'ruling:<ID>') or into a non-production --out (authority 'non-production out:'), recorded as an override of a ruled key."""
    import argparse
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core.record import Refusal
    DO = _DO()
    assert CFG.ruled('catalog_omission.REFUSALS') and not CFG.protected('catalog_omission.REFUSALS')
    default = CFG.catalog_omission.REFUSALS
    other = _fresh('override_refusals')
    a = argparse.Namespace(omit_refusals=other)
    prod_out = DO.default_out(argparse.Namespace(tier='sublst'))         # <TRANSIENTS_ROOT>/deep_omission: never written here
    assert CFG.production_root_of(prod_out), prod_out
    lid = 'R-GREGG-20261031-SNAPSHOT-RESEARCH-LC-DEEPOMIT-GPU-RELEASE'
    try:
        CFG.authorize(out=prod_out)
        try:
            DO.refusals_arg(a)
            raise AssertionError('--omit-refusals into a production root without --ruling-for was not refused')
        except Refusal as e:
            assert '--omit-refusals' in str(e) and 'carries a ruling' in str(e), str(e)
        assert CFG.catalog_omission.REFUSALS == default and not CFG.overrides
        assert DO.refusals_arg(argparse.Namespace(omit_refusals=None)) == default      # no flag: the ruled record
        CFG.authorize(rulings=[f'catalog_omission.REFUSALS={lid}'], out=prod_out)
        assert DO.refusals_arg(a) == os.path.abspath(other) == CFG.catalog_omission.REFUSALS
        assert CFG.ruled_overrides() == [{'key': 'catalog_omission.REFUSALS', 'ruled_by': list(CFG.ruled('catalog_omission.REFUSALS')),
                                          'authority': f'ruling:{lid}'}], CFG.ruled_overrides()
        CFG.reset()
        CFG.authorize(out=os.path.join(other, 'out'))
        assert DO.refusals_arg(a) == os.path.abspath(other)
        assert CFG.ruled_overrides()[0]['authority'].startswith('non-production out:'), CFG.ruled_overrides()
    finally:
        CFG.reset()
        CFG.authorize()
    assert CFG.catalog_omission.REFUSALS == default


class Skip(Exception):
    """A test that cannot run on this host (its read-only production inputs are absent): reported as SKIP, never PASS."""


def test_real_one_lst():
    """READ ONLY: the port's decisions = production's snapshot_deep_omission.decide for 07h on the served counts of B6H."""
    DO = _DO()
    SDO, PDO, _d = _prod()
    run = '/data07/gh/phase3/snapshot_search/review/Run_SNAPB6H_20261023'
    if not (os.path.isdir(run) and os.path.isdir(PDO.REFUSALS)):
        raise Skip('no served B6H or no refusal record on this host (lwacalim07 only)')
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


TESTS = [test_production_pin, test_drift, test_snapshot, test_tenmin, test_tenmin_rejudge, test_serve_build, test_serve_build_merge,
         test_gates, test_content,
         test_kept_product, test_stage, test_refusals_override, test_pending_values, test_production_bugs_pinned,
         test_injected_record, test_real_one_lst]


def main(argv):
    want = set(argv[1:])
    t0 = time.time()
    failed, skipped = [], []
    os.makedirs(SCRATCH, exist_ok=True)
    try:
        for t in TESTS:
            if want and t.__name__ not in want:
                continue
            t1 = time.time()
            try:
                t()
                print(f'PASS {t.__name__} ({time.time() - t1:.1f} s)')
            except Skip as e:
                skipped.append(t.__name__)
                print(f'SKIP {t.__name__}: {e}')
            except Exception:
                import traceback
                failed.append(t.__name__)
                print(f'FAIL {t.__name__}\n{traceback.format_exc()}')
    finally:
        shutil.rmtree(SCRATCH, ignore_errors=True)
    n = len([t for t in TESTS if not want or t.__name__ in want])
    passed = n - len(failed) - len(skipped)
    print((f'{len(failed)} failed: {failed}; ' if failed else '') + f'{passed} of {n} passed'
          + (f', {len(skipped)} skipped: {skipped}' if skipped else '') + f' ({time.time() - t0:.1f} s)')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
