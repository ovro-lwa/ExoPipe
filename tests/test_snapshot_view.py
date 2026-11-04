#!/usr/bin/env python3
"""STANDING TEST - the combined snapshot view, `serve --tier snapshot --family comb` (lib/serve/snapshot_view.py; spec
docs/specs/snapshot_combined_view.md): the snapshot tier's cumulative step, the port of exopipe_ops snapshot_combined_view.py
and its driver snapshot_batch/comb_build_20261102.sh.

  drift      the port's texts, patterns and layout are production's (snapshot_combined_view.py, comb_build_20261102.sh read as
             text): the queue header lines, the summary lines, the tests line, the page suffixes, the stores' path, the
             interpreter, the localization's environment, the re-point fields and their order
  parity     a synthetic world - three served batch runs with merged identities across batches, a name clash, a dropped
             identity, a dropped per-tier source, partial sources (one re-pointed: a non-production best_cand_id; one that keeps
             its omitted best: production's best_cand_id - PRODUCTION BUG RP), member rows and panels on OMIT keys, both Stokes
             and tiers, wideband pages, a panel-name clash, card2 files, a detection_ts.csv, fit stores - built by PRODUCTION's
             own snapshot_combined_view.py (under its interpreter; snapshot_deep_omission.decide's roots pointed at the world)
             with the driver's filter catalogue and card2 / FILTER-INPUTS heredoc, and by the stage through the real CLI (omit,
             build, filtercat): every file of the two views equal byte for byte but the summary's 'seconds', the panels hard
             links of the same inodes; the stage's records done, every content test passed
  kept       G-K / G-V4: the build refuses (nothing moved into place) on a kept detection moved to another identity, a per-tier
             row removed from the kept product, a served run changed after the omit stage, an omit product of other runs, served
             counts that are not the served set's; a refusal after the .part was made moves it aside to <view>.refused_<UTC>
             (F7), a .part the build did not make stays where it is
  gates      G-V1 (no name, no runs, a run twice, a run without catalogues), G-V2 (the view or its .part exists: refused BEFORE
             the omit stage writes, its omit product unchanged - F2), G-V5 (another file system, simulated, a production view),
             G-V6 (an injected run's products), --view-omit with --stages omit
  incomplete G4b / G4c at the omit stage and G-V7 in the build child (F4): a batch's per-tier catalogue lost (I3), a kept
             source's page detections.csv lost with its member rows (I2) - refused, never booked as a deep omission
  stale      G-V8 (F5): the build re-decides from the refusal record and fleet records it reads now - a refusal added after the omit
             stage, a noise table rewritten, a product decided from another record refuse; K5 tampered served counts
  copy       G-V5 in the child (batch runs on /dev/shm): a production view refuses; a scratch / parity view copies every panel
             (as under the parity guard) and equals the hard-linked view byte for byte
  content    snapshot_view refuses a queue that lost a name, a page holding an omitted detection, a page directory of two tiers,
             an identity detection the kept product does not keep, a record off the omit stage's, a panel of an omitted instant
             on a page (V3, its own key) and a page's files not its member batch sources' less the panel drops (V7) - F1;
             snapshot_card2_union a changed line; filter_catalog (per detection) a missing row
  readers    exopipe_ops 1b9475c: both filter-catalogue stages always pass SNAPCAT_PROCS (serve.SNAPVIEW_FILTERCAT_PROCS,
             pages_snapshot.FILTERCAT_PROCS: 6, the Lustre rule), never an inherited value; a count not >= 1 refuses (F-G4 / P22)
  m1         F1 proven on the reviewer's mutation M1 (a private copy of this tree whose build links an omitted panel it records as
             dropped): the build's read-back refuses, the view is moved aside (F7), nothing is left under its name
  seed       F10: the localize stage runs snapshot_localize_tiers.sh with PYTHONHASHSEED = serve.SNAPSHOT_FIGURE_HASHSEED
  ready      the ready stage on the built view (stub localizations): G17 passes, the final tests pass, VIEW-READY-FOR-HIDES
  py36       the child's module parses and loads under production's interpreter (Python 3.6)
  pin        production's ported files are the pinned ones (tests/production_pin.py; F3): the only test production's later
             commits fail - the stage's own tests above build the stage's view alone, production's side only in drift / parity /
             repoint, from the pinned files

Scratch: /tmp/exopipe3_tests_snapshot_view/ (removed at the end). Production trees are only read.
Run standalone:  python tests/test_snapshot_view.py [test_name ...]
"""
import collections
import csv
import filecmp
import glob
import json
import os
import random
import shutil
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import snapshot_chain_fixture       # noqa: E402
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')
SCRATCH = '/tmp/exopipe3_tests_snapshot_view'
OPS = '/lustre/gh/main/exopipe_ops'
PY36 = '/opt/devel/pipeline/envs/development/bin/python'
TAG, OLD = '20261006T000000', '20261001T000000'
NIGHTS = ['2025-06-0%d' % i for i in range(1, 9)]       # no night of data_decisions.csv for 05h / 06h (the G17 gate at ready)
SUBBANDS = ('18MHz', '23MHz', '27MHz', '46MHz', '55MHz')
SB_CLASS = {'18MHz': 'Red', '23MHz': 'Red', '27MHz': 'Red', 'Red': 'Red', '46MHz': 'Green', '55MHz': 'Green', 'Green': 'Green'}
VIEW = 'Run_SNAPCOMB_T2_20261103'
csv.field_size_limit(10 ** 9)


def _wcsv(path, cols, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _rows(p):
    with open(p, newline='') as f:
        return list(csv.DictReader(f))


def _fresh(name):
    d = os.path.join(SCRATCH, name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    return d


def _SV():
    from exopipe.phase3.lib.serve import snapshot_view as SV
    return SV


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def _cli(*args, timeout=1800, repo=REPO):
    """The real CLI of this tree (an ordinary run); ``repo``: another tree's (test_m1_mutation's mutated copy)."""
    e = {k: v for k, v in os.environ.items() if k not in ('EXOPIPE3_INJECTED', 'P3_SNAP_INJECTED', 'EXOPIPE3_PARITY_ROOT')}
    e['PYTHONPATH'] = repo + (os.pathsep + e['PYTHONPATH'] if e.get('PYTHONPATH') else '')
    return subprocess.run([sys.executable, '-m', 'exopipe.phase3'] + [str(a) for a in args], cwd=repo, env=e,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=timeout)


def unit_records(out):
    """{unit: record} of the serve units under ``out``."""
    recs = {}
    for p in glob.glob(os.path.join(out, '_records', 'serve', '*.json')):
        j = json.load(open(p))
        if isinstance(j, dict) and 'unit' in j:
            recs[j['unit']] = j
    return recs


# ---------------------------------------------------------------------------------------------------------------------
# the synthetic world: a deep refusal record, fleet records, three served batch runs, their fit stores
# ---------------------------------------------------------------------------------------------------------------------

def refusal_record(rdir):
    """Every gate kind (tests/test_deep_omission.py's record): 05h 46 MHz I night 1 noisy (OMIT at 10 s), 06h 27 MHz with two
    nights of records (no threshold: OMIT), sanity / nightimage / beam / epochgate rows, an older attempt, a stale_input row."""
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
        '06h_27MHz_searchtier': [['06h', '27MHz', NIGHTS[1], 'I', '5.5', '2.0', TAG]],
        '06h_46MHz_sanity': [['06h', '46MHz', NIGHTS[2], '40.1', '3.3', TAG]],
    }
    for name, rows in files.items():
        with open(os.path.join(rdir, name + '_excluded.csv'), 'w', newline='') as f:
            csv.writer(f).writerows(rows)
    return rdir


NOISE_COLS = ['idx', 'ts', 'admitted', 'global_rms_jyb']


def searched(lst, ni, band):
    return not ((lst, band, ni) == ('05h', '55MHz', 7) or (lst == '06h' and band == '27MHz' and ni > 1))


def frame_ts(lst, night, k):
    """The 10-s frame k of a cell (an integration label YYYYMMDD_HHMMSS; det()'s ts)."""
    return '%s_%02d%02d%02d' % (night.replace('-', ''), int(lst[:2]), k // 6, (k % 6) * 10)


def date_obs(ts):
    """The DATE-OBS of a frame whose integration midpoint is ts (clv15 INTEGRATION_S / 2 = 5.0155102 s before it): its 1-minute
    block (integration_minute_key + '00') is ts's minute."""
    import datetime
    t = datetime.datetime.strptime(ts, '%Y%m%d_%H%M%S') - datetime.timedelta(seconds=5.0155102)
    return t.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]


def gap_frame(k):
    """The Green frames stacked without 46 MHz (the frame exception, dde1760): every frame k with k % 5 == 0."""
    return k % 5 == 0


def fleet(proto, rng):
    """Noise tables per (cell, band, Stokes, tier) (05h 46 MHz I 10-s night 1 noisy); per-subband admission tables; the wideband
    member tables of the 05h cells, one row per (frame, group, subband) with its ts and DATE-OBS (dde1760 applies a member's
    omission per image: Green holds 46 MHz in every frame but gap_frame's), and none in the 06h cells (searched before the
    frame-time fix: dde1760 fails closed there, on the admission tables - 06h 27 MHz admits no frame on night 1)."""
    for lst in ('05h', '06h'):
        for ni, night in enumerate(NIGHTS):
            cell = os.path.join(proto, f'{lst}_{night}')
            os.makedirs(cell, exist_ok=True)
            for band in SUBBANDS + ('Red', 'Green'):
                if not searched(lst, ni, band):
                    continue
                for S in ('I', 'V'):
                    for T, sfx in (('10s', ''), ('1min', '_1min')):
                        base = 4.0 if (lst, band, S, T, ni) == ('05h', '46MHz', 'I', '10s', 0) else 1.0 + 0.05 * rng.random()
                        rows = [dict(idx=i, ts='x', admitted='1', global_rms_jyb='%.6f' % (base * (1 + 0.02 * rng.random())))
                                for i in range(rng.randint(30, 40))]
                        _wcsv(os.path.join(cell, f'noise_{band}_{S}_median{sfx}.csv'), NOISE_COLS, rows)
            for sb in SUBBANDS:
                if not searched(lst, ni, sb):
                    continue
                adm = '0' if (lst, sb, ni) == ('06h', '27MHz', 0) else '1'
                _wcsv(os.path.join(cell, f'admission_{sb}.csv'), ['idx', 'ts', 'date_obs', 'admitted'],
                      [dict(idx=k, ts=frame_ts(lst, night, k), date_obs=date_obs(frame_ts(lst, night, k)), admitted=adm)
                       for k in range(0, 120, 7)])
            if lst != '05h':
                continue
            for S in ('I', 'V'):
                rows = []
                for k in range(120):
                    ts = frame_ts(lst, night, k)
                    for g, sb in (('Red', '18MHz'), ('Red', '23MHz'), ('Green', '55MHz'), ('Green', '46MHz')):
                        if (g, sb) == ('Green', '46MHz') and gap_frame(k):
                            continue
                        rows.append(dict(group=g, ts=ts, date_obs=date_obs(ts), subband=sb))
                _wcsv(os.path.join(cell, f'wideband_members_{S}.csv'), ['group', 'ts', 'date_obs', 'subband'], rows)
    return proto


DET_COLS = ['lst', 'date', 'band', 'subband', 'stokes', 'ts', 'snr', 'ra', 'dec', 'tier', 'instant', 'cand_id', 'ra_peak',
            'dec_peak', 'pos_err_stat_arcsec', 'pos_err_sys_arcsec']


def det(lst, night, sb, S, T, k, snr, ra, dec):
    ts = '%s_%02d%02d%02d' % (night.replace('-', ''), int(lst[:2]), k // 6, (k % 6) * 10)
    if T == '1min':
        ts = ts[:-2] + '00'
    return dict(lst=lst, date=night, band=SB_CLASS[sb], subband=sb, stokes=S, ts=ts, snr='%.2f' % snr, ra='%.5f' % ra,
                dec='%.5f' % dec, tier=T, instant=ts[:13] if T == '1min' else ts, cand_id=f'{lst}_{night}_{sb}_{S}_{T}_{ts}',
                ra_peak='%.5f' % (ra + 0.001), dec_peak='%.5f' % dec, pos_err_stat_arcsec='40.0', pos_err_sys_arcsec='60.0')


def batch_run(root, name, idents, cells, extra_files=(), card2=()):
    """A served batch run like production's: the general catalogue per band class (a row per identity and class, its own
    review_dir with that class's detections, anchored); the CHAIN RECORD, chain_summary.json (``cells``) and served_tiers.csv
    (tests/snapshot_chain_fixture.py: the chain's own route functions decide which (identity, tier) qualifies); per-tier views
    of the QUALIFYING (identity, tier) only (survivors: one row per (identity, band) with production's best_cand_id
    '<identity>_<cell>_<band>' and the best detection's ts / subband; members '<identity>_<cell>_<band>'), page directories per
    (band, tier) with panels (.png + .page.json per detection, astrometry.png, ``extra_files``), the queues with their header
    lines, card2_filter.csv, card2_vleak.csv (every 10-s Stokes V detection; ``card2`` = the cand_ids Card 2 flags, a row not
    evaluated, a flagged row the chain did not serve). idents: [(J, (ra, dec, err_arcsec), [detections])]."""
    run = os.path.join(root, name)
    gen = collections.defaultdict(list)
    per = collections.defaultdict(list)
    mem = collections.defaultdict(list)
    queues = collections.defaultdict(list)
    for J, (ra, dec, err), dets in idents:
        by_class = collections.defaultdict(list)
        for d in dets:
            by_class[SB_CLASS[d['subband']]].append(d)
        for cls, dd in sorted(by_class.items()):
            rd = f'candidates/{cls}/rank0001_{cls}_{J}'
            _wcsv(os.path.join(run, rd, 'detections.csv'), DET_COLS, dd)
            gen[cls].append(dict(rank='rank0001', source_id=J, review_dir=rd, ra='%.5f' % ra, dec='%.5f' % dec,
                                 best_snr='%.2f' % max(abs(float(x['snr'])) for x in dd), n_det=len(dd),
                                 best_cand_id=f'{J}_{dd[0]["lst"]}_{dd[0]["date"]}_{cls}',
                                 ra_common='%.5f' % (ra + 0.002), dec_common='%.5f' % dec, pos_err_common_arcsec='%.1f' % err))
    qual = snapshot_chain_fixture.write_chain(run, [(J, dets) for J, _a, dets in idents], cells)
    for J, (ra, dec, err), dets in idents:
        by = collections.defaultdict(list)
        for d in dets:
            if qual.get((J, d['tier'])):
                by[(d['subband'], d['stokes'], d['tier'])].append(d)
        for (sb, S, T), dd in sorted(by.items()):
            pd = f'candidates/{S}/{J}_{sb}' + ('_1min' if T == '1min' else '')
            _wcsv(os.path.join(run, pd, 'detections.csv'), DET_COLS, dd)
            for d in dd:
                stem = f"rank0001_{d['lst']}_{d['date']}_{sb}{d['ts'].replace('_', '')}" + ('1min' if T == '1min' else '')
                for ext in ('.png', '.page.json'):
                    with open(os.path.join(run, pd, stem + ext), 'w') as f:
                        f.write(f'{name} {pd} {stem}{ext}\n')
            with open(os.path.join(run, pd, 'astrometry.png'), 'w') as f:
                f.write('figure\n')
            for fn in extra_files:
                with open(os.path.join(run, pd, fn), 'w') as f:
                    f.write(f'{name} {fn}\n')
            best = max(dd, key=lambda d: abs(float(d['snr'])))
            per[(S, T, sb)].append(dict(rank='rank0001', source_id=f'{J}_{sb}', identity_id=J, review_dir=pd, ra=best['ra'],
                                        dec=best['dec'], best_snr=best['snr'], n_det=len(dd), n_det_identity=len(dets),
                                        best_cand_id=f'{J}_{best["lst"]}_{best["date"]}_{sb}', peak_jy_best='12.5', stokes=S,
                                        tier=T, subband=sb, ts=best['ts']))
            for d in dd:
                mem[(S, T, sb)].append(dict(source_id=f"{J}_{d['lst']}_{d['date']}_{sb}", cand_id=d['cand_id'], det_snr=d['snr'],
                                            det_date=d['date'], band=sb))
            queues[(S, T)].append((-abs(float(best['snr'])), J))
    for cls, rr in gen.items():
        _wcsv(os.path.join(run, 'source_catalogs', f'{cls}_sources_survivors.csv'),
              ['rank', 'source_id', 'review_dir', 'ra', 'dec', 'best_snr', 'n_det', 'best_cand_id', 'ra_common', 'dec_common',
               'pos_err_common_arcsec'], rr)
    pcols = ['rank', 'source_id', 'identity_id', 'review_dir', 'ra', 'dec', 'best_snr', 'n_det', 'n_det_identity', 'best_cand_id',
             'peak_jy_best', 'stokes', 'tier', 'subband', 'ts']
    for (S, T, sb), rr in per.items():
        _wcsv(os.path.join(run, f'source_catalogs_{S}_{T}', f'{sb}_sources_survivors.csv'), pcols, rr)
        _wcsv(os.path.join(run, f'source_catalogs_{S}_{T}', f'{sb}_source_members.csv'),
              ['source_id', 'cand_id', 'det_snr', 'det_date', 'band'], mem[(S, T, sb)])
    for S in 'VI':
        for T in ('10s', '1min'):
            os.makedirs(os.path.join(run, f'source_catalogs_{S}_{T}'), exist_ok=True)
            with open(os.path.join(run, f'queue_{S}_{T}.txt'), 'w') as f:
                f.write(f'# queue_name: snapshot Stokes {S}, {T} tier\n# run: {name}\n# selection: identities with a panelled '
                        f'{T} detection in Stokes {S}\n# per-tier serving: synthetic\n')
                for _s, J in sorted(set(queues[(S, T)])):
                    f.write(J + '\n')
    with open(os.path.join(run, 'card2_filter.csv'), 'w') as f:
        f.write('# card2 filter (synthetic)\n# columns: cand_id, verdict\ncand_id,verdict\n')
        for J, _a, dets in idents:
            for d in dets[:2]:
                f.write(f'{d["cand_id"]},keep\n')
        f.write('shared_row,keep\n')
    card2_file(run, idents, card2)
    return run


C2_COLS = ['det_uid', 'cand_id', 'ra_peak', 'dec_peak', 'identity_jname', 'stokes', 'tier', 'served', 'peak_jyb', 'i_here_jy',
           'v_over_i_here', 'card2_ratio', 'card2_would_remove']


def card2_file(run, idents, flagged):
    """card2_vleak.csv of a synthetic batch (phase3_snapshot_card2.py's columns the view reads): every 10-s Stokes V detection, the
    ratio |peak| / i_here written to 5 decimals and the flag at 0.05 - flagged (0.03) for the cand_ids in ``flagged``, 0.4
    otherwise, the first one not evaluated (no Stokes I pixel) - and a flagged row the chain did not serve."""
    rows, n = [], 0
    tag = os.path.basename(run)
    for i, (J, _a, dets) in enumerate(idents):
        for j, d in enumerate(dets):
            if (d['tier'], d['stokes']) != ('10s', 'V'):
                continue
            n += 1
            uid = '%s:%d:%d' % (tag, i, j)          # the chain record's det_uid (tests/snapshot_chain_fixture.py)
            pk = abs(float(d['snr'])) * 0.1
            if n == 1 and d['cand_id'] not in flagged:
                rows.append(dict(det_uid=uid, cand_id=d['cand_id'], ra_peak=d['ra_peak'], dec_peak=d['dec_peak'],
                                 identity_jname=J, stokes='V', tier='10s', served='1', peak_jyb='%.4g' % pk, i_here_jy='',
                                 v_over_i_here='', card2_ratio='0.05', card2_would_remove='0'))
                continue
            ratio = 0.03 if d['cand_id'] in flagged else 0.4
            rows.append(dict(det_uid=uid, cand_id=d['cand_id'], ra_peak=d['ra_peak'], dec_peak=d['dec_peak'],
                             identity_jname=J, stokes='V', tier='10s', served='1', peak_jyb='%.4g' % pk,
                             i_here_jy='%.6g' % (pk / ratio), v_over_i_here='%.5f' % ratio, card2_ratio='0.05',
                             card2_would_remove='1' if ratio <= 0.05 else '0'))
    rows.append(dict(det_uid='u_unserved', cand_id='05h_2025-06-01_23MHz_V_10s_20250601_051950', ra_peak='1.00000',
                     dec_peak='2.00000', identity_jname='', stokes='V', tier='10s', served='0', peak_jyb='0.5', i_here_jy='25',
                     v_over_i_here='0.02000', card2_ratio='0.05', card2_would_remove='1'))
    _wcsv(os.path.join(run, 'card2_vleak.csv'), C2_COLS, rows)


def fit_store(batch_dir, run):
    """The batch's fit store (rechain_<run>/fits_store.csv): a Gaussian fit for every other detection of its pages."""
    rows = []
    for p in sorted(glob.glob(os.path.join(run, 'candidates', '[IV]', '*', 'detections.csv'))):
        for i, d in enumerate(_rows(p)):
            if i % 2:
                continue
            rows.append(dict(cell=f"{d['lst']}_{d['date']}", subband=d['subband'], stokes=d['stokes'], tier=d['tier'], ts=d['ts'],
                             ra_deg=d['ra_peak'], dec_deg=d['dec_peak'], snr=d['snr'], fit_status='ok', fit_maj_asec='900.0',
                             beam_maj_asec='600.0', fit_min_asec='700.0'))
    _wcsv(os.path.join(batch_dir, f'rechain_{os.path.basename(run)}', 'fits_store.csv'),
          ['cell', 'subband', 'stokes', 'tier', 'ts', 'ra_deg', 'dec_deg', 'snr', 'fit_status', 'fit_maj_asec', 'beam_maj_asec',
           'fit_min_asec'], rows)


def _off_floor(snr):
    """An S/N clear of the record's rounding at a singleton floor (10 / 8 sigma; the chain fixture refuses one there)."""
    return snr + 0.013 if min(abs(snr - 10.0), abs(snr - 8.0)) < 0.006 else snr


def world(name='world', seed=31):
    """(root, rdir, proto, runs, batch_dir) of the synthetic world (module docstring)."""
    root = _fresh(name)
    rng = random.Random(seed)
    rdir = refusal_record(os.path.join(root, 'refusals'))
    proto = fleet(os.path.join(root, 'fleet'), rng)
    review = os.path.join(root, 'review')
    N = NIGHTS
    c2 = {}                                  # batch -> the cand_ids Card 2 flags
    a = [  # batch A: 05h nights 1-2 (+ 06h night 2)
        ('J0520+3000', (80.0, 30.0, 60.0), [det('05h', N[0], '46MHz', 'I', '10s', 3, 30.0, 80.0, 30.0),
                                            det('05h', N[1], '18MHz', 'V', '1min', 4, 12.0, 80.0, 30.0)]),
        ('J0600+2500', (90.0, 25.0, 60.0), [det('05h', N[1], '23MHz', 'I', '10s', 5, 15.0, 90.0, 25.0)]),
        ('J0510+2000', (77.5, 20.0, 60.0), [det('05h', N[0], '46MHz', 'I', '10s', 6, 11.0, 77.5, 20.0),     # all omitted:
                                            det('05h', N[0], '46MHz', 'I', '10s', 7, 9.0, 77.5, 20.0)]),    # identity dropped
        ('J0530+3500', (82.5, 35.0, 60.0), [det('05h', N[0], '46MHz', 'I', '10s', 8, 25.0, 82.5, 35.0),     # partial, best
                                            det('05h', N[1], '46MHz', 'I', '10s', 8, 10.6, 82.5, 35.0)]),   # omitted: re-pointed
        ('J0545+3700', (86.0, 37.0, 60.0), [det('05h', N[0], '46MHz', 'I', '10s', 9, 28.0, 86.0, 37.0),     # partial, best
                                            det('05h', N[1], '46MHz', 'I', '10s', 9, 11.0, 86.0, 37.0)]),   # omitted, re-pointed
        ('J0550+3800', (87.5, 38.0, 60.0), [det('05h', N[0], '46MHz', 'I', '10s', 10, 13.0, 87.5, 38.0),    # omitted; its 18 MHz
                                            det('05h', N[1], '18MHz', 'V', '10s', 10, 9.0, 87.5, 38.0),     # V one Card 2: the
                                            det('05h', N[1], 'Green', 'I', '10s', 11, 8.5, 87.5, 38.0)]),   # 10-s re-judged out
        ('J0535+3600', (84.0, 36.0, 60.0), [det('05h', N[0], 'Green', 'I', '10s', 12, 12.0, 84.0, 36.0),    # its frame holds
                                            det('05h', N[0], 'Green', 'I', '10s', 15, 9.5, 84.0, 36.0),     # 46 MHz: omitted per
                                            det('05h', N[1], '23MHz', 'I', '10s', 13, 16.0, 84.0, 36.0)]),  # image; frame 15 not
        ('J0612+2200', (93.0, 22.0, 60.0), [det('06h', N[1], 'Red', 'I', '10s', 14, 16.0, 93.0, 22.0)]),    # fail closed (06h)
    ]
    c2['A'] = {a[5][2][1]['cand_id']}
    b = [  # batch B: 05h nights 3-4, 06h night 3
        ('J0520+3001', (80.0, 30.0, 60.0), [det('05h', N[2], '46MHz', 'I', '10s', 3, 20.0, 80.0, 30.0),     # merges with A's
                                            det('05h', N[3], 'Red', 'I', '1min', 3, 9.5, 80.0, 30.0),       # (and its 18 MHz V
                                            det('05h', N[2], '18MHz', 'V', '1min', 4, 11.0, 80.0, 30.0)]),  # 1-min page: a clash)
        ('J0610+3100', (92.5, 31.0, 60.0), [det('06h', N[2], '55MHz', 'V', '10s', 5, 14.0, 92.5, 31.0)]),  # Card 2: dropped
        ('J0615+3300', (93.7, 33.0, 60.0), [det('06h', N[2], '46MHz', 'I', '1min', 6, 16.0, 93.7, 33.0),
                                            det('05h', N[3], '27MHz', 'I', '10s', 6, 8.8, 93.7, 33.0)]),
    ]
    c2['B'] = {b[1][2][0]['cand_id']}
    c = [  # batch C: 06h nights 5-6
        ('J0600+2500', (95.0, 35.0, 60.0), [det('06h', N[4], '18MHz', 'I', '10s', 5, 18.0, 95.0, 35.0)]),  # a name clash
        ('J0615+3301', (93.7, 33.0, 60.0), [det('06h', N[5], '46MHz', 'I', '1min', 6, 22.0, 93.7, 33.0)]),  # merges with B's
        ('J0620+2900', (95.0, 29.0, 60.0), [det('06h', N[5], 'Green', 'V', '10s', 7, 10.5, 95.0, 29.0),
                                            det('06h', N[4], '23MHz', 'V', '1min', 7, 9.0, 95.0, 29.0)]),
        ('J0625+2700', (96.2, 27.0, 60.0), [det('06h', N[4], '46MHz', 'V', '10s', 30, 12.0, 96.2, 27.0),    # one frame Card 2:
                                            det('06h', N[4], '46MHz', 'V', '10s', 31, 11.0, 96.2, 27.0)]),  # its panel, member row
    ]
    c2['C'] = {c[3][2][1]['cand_id']}
    rnd = []
    for i in range(24):                    # filler identities (no merge: far apart), every band and tier
        bi = i % 3
        lst = ('05h', '05h', '06h')[bi]
        nights = ((N[0], N[1]), (N[2], N[3]), (N[4], N[5]))[bi]
        ra, dec = 70.0 + 1.3 * i, 21.0 + 0.7 * i
        dets = []
        for k in range(rng.randint(1, 4)):
            sb = rng.choice(SUBBANDS + ('Red', 'Green'))
            nt = rng.choice(nights)
            if lst == '06h' and sb == '27MHz' and nt not in (N[0], N[1]):
                sb = '23MHz'
            dets.append(det(lst, nt, sb, rng.choice('IV'), rng.choice(('10s', '1min')), 20 + 3 * i + k,
                            _off_floor(7.5 + 10 * rng.random()), ra, dec))
        rnd.append((bi, (f'J{4 + i // 10:02d}{i % 60:02d}+{20 + i:02d}00', (ra, dec, 60.0), dets)))
    a += [x for bi, x in rnd if bi == 0]
    b += [x for bi, x in rnd if bi == 1]
    c += [x for bi, x in rnd if bi == 2]
    cells = {'A': ['05h_' + N[0], '05h_' + N[1], '06h_' + N[1]], 'B': ['05h_' + N[2], '05h_' + N[3], '06h_' + N[2]],
             'C': ['06h_' + N[4], '06h_' + N[5]]}
    runs = [batch_run(review, 'Run_SNAPT2A_20261103', a, cells['A'], extra_files=('notes.txt',), card2=c2['A']),
            batch_run(review, 'Run_SNAPT2B_20261103', b, cells['B'], extra_files=('notes.txt',), card2=c2['B']),
            batch_run(review, 'Run_SNAPT2C_20261103', c, cells['C'], card2=c2['C'])]
    _wcsv(os.path.join(runs[1], 'detection_ts.csv'), ['cand_id', 'ts_utc'],
          [dict(cand_id=d['cand_id'], ts_utc=d['ts']) for _J, _a, dd in b for d in dd])
    batch_dir = os.path.join(root, 'snapshot_batch')
    for r in runs:
        fit_store(batch_dir, r)
    return root, rdir, proto, runs, batch_dir


# ---------------------------------------------------------------------------------------------------------------------
# production's side and the stage's side
# ---------------------------------------------------------------------------------------------------------------------

# THE PRODUCTION PIN (V_combview review F3; tests/production_pin.py): the production code this stage PORTS - the view's builder,
# the deep-omission decisions it imports, the driver whose heredoc is ported - is read at the pinned exopipe_ops commit, so a later
# production commit never breaks the stage's own tests; it shows up in test_production_pin. Production code the stage RUNS IN PLACE
# (the chain's merge, snapshot_filter_catalog.py) is live on both sides.
import production_pin as PIN        # noqa: E402
PINNED = ('snapshot_combined_view.py', 'snapshot_deep_omission.py', 'snapshot_rejudge.py', 'sublst_deep_omission.py',
          'snapshot_batch/comb_build_20261104.sh')
_PIN_DIR = []


def pinned_ops():
    """The directory holding the pinned production files (materialized once per test run, under SCRATCH)."""
    if not _PIN_DIR:
        _PIN_DIR.append(PIN.materialize(os.path.join(SCRATCH, 'production_pin'), PINNED))
    return _PIN_DIR[0]


PROD_HARNESS = r'''
import os, runpy, sys
pin, rdir, proto, out, runs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:]
sys.path[:] = [p for p in sys.path if p not in ('', os.path.dirname(os.path.abspath(__file__)))]
sys.path.insert(0, pin)                 # the PINNED snapshot_deep_omission (+ the sublst_deep_omission beside it) is imported
import snapshot_deep_omission as SDO    # first: the builder's own `import snapshot_deep_omission` then finds it cached
import snapshot_rejudge as SRJ
assert os.path.dirname(os.path.abspath(SDO.__file__)) == os.path.abspath(pin), SDO.__file__
assert os.path.dirname(os.path.abspath(SRJ.__file__)) == os.path.abspath(pin), SRJ.__file__
SDO.decide.__defaults__ = (rdir, proto, None)               # decide(lsts, served, rdir, proto, cells=None): the data roots
SDO.wb_image_members.__defaults__ = (proto,)                # the per-image member tables (dde1760) read the same fleet
SDO.wb_image_omitted.__defaults__ = (proto,)
sys.argv = [os.path.join(pin, 'snapshot_combined_view.py'), out] + runs
runpy.run_path(os.path.join(pin, 'snapshot_combined_view.py'), run_name='__main__')
'''


def prod_driver_heredoc():
    """The driver's step-2 heredoc (comb_build_20261104.sh at the pin, between <<'PYC' and PYC), verbatim."""
    src = open(os.path.join(pinned_ops(), 'snapshot_batch', 'comb_build_20261104.sh')).read()
    return src.split("<<'PYC'", 1)[1].split('\nPYC\n', 1)[0].split('\n', 1)[1]


def prod_view(root, rdir, proto, runs, batch_dir, out):
    """Production's builder (pinned) + the driver's filter catalogue (in place, live) and heredoc (pinned) into ``out``
    (production's interpreter)."""
    h = os.path.join(root, 'prod_harness.py')
    with open(h, 'w') as f:
        f.write(PROD_HARNESS)
    env = dict(os.environ, NUMPY_MADVISE_HUGEPAGE='0')
    for k in [k for k in env if k.startswith(('P3_', 'SNAP', 'EXOPIPE3_'))]:
        env.pop(k)
    r = subprocess.run([PY36, '-u', h, pinned_ops(), rdir, proto, out] + runs, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       universal_newlines=True, env=env, cwd=root, timeout=1800)
    assert r.returncode == 0 and 'COMBINED-VIEW-OK' in r.stdout, r.stdout[-3000:]
    stores = [os.path.join(batch_dir, f'rechain_{os.path.basename(x)}', 'fits_store.csv') for x in runs]
    r2 = subprocess.run([PY36, os.path.join(OPS, 'snapshot_filter_catalog.py'), out] + stores, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, universal_newlines=True, env=env, cwd=OPS, timeout=1800)
    assert r2.returncode == 0, r2.stdout[-2000:]
    r3 = subprocess.run([PY36, '-', out] + runs, input=prod_driver_heredoc(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        universal_newlines=True, env=env, cwd=OPS, timeout=600)
    assert r3.returncode == 0 and 'FILTER-INPUTS OK' in r3.stdout, r3.stdout[-2000:]
    return r.stdout


def port_view(root, rdir, proto, runs, batch_dir, stages='omit,build,filtercat', name=VIEW, extra=()):
    """The stage through the real CLI into <root>/port (a parity --out): returns (CompletedProcess, view dir, out)."""
    out = os.path.join(root, 'port')
    r = _cli('serve', '--tier', 'snapshot', '--family', 'comb', '--run', name, '--view-runs', ','.join(runs), '--out', out,
             '--set', f'catalog_omission.REFUSALS={rdir}', '--set', f'paths.SNAPSHOT_FLEET_OUT={proto}',
             '--set', f'catalog_snapshot.BATCH_DIR={batch_dir}', '--stages', stages, '--cores', 'none', *extra)
    return r, os.path.join(out, 'snapshot', name), out


def tree(d):
    out = {}
    for dp, _dn, fns in os.walk(d):
        for fn in fns:
            p = os.path.join(dp, fn)
            out[os.path.relpath(p, d)] = p
    return out


def compare_views(a, b):
    """[differences] between two view trees: the same files; every file byte for byte, but the summary's 'seconds'; a panel
    (a file of a candidates/<S>/ page directory other than detections.csv) the same inode (both hard links of one input)."""
    ta, tb = tree(a), tree(b)
    diff = [f'only in {os.path.basename(os.path.dirname(a))}: {x}' for x in sorted(set(ta) - set(tb))[:5]]
    diff += [f'only in {os.path.basename(os.path.dirname(b))}: {x}' for x in sorted(set(tb) - set(ta))[:5]]
    for rel in sorted(set(ta) & set(tb)):
        pa, pb = ta[rel], tb[rel]
        if rel == 'combined_view_summary.json':
            ja, jb = json.load(open(pa)), json.load(open(pb))
            ja.pop('seconds', None)
            jb.pop('seconds', None)
            if ja != jb or list(ja) != list(jb):
                diff.append(f'{rel}: {set(k for k in ja if ja.get(k) != jb.get(k))}')
            continue
        parts = rel.split(os.sep)
        if parts[0] == 'candidates' and parts[1] in ('I', 'V') and parts[-1] != 'detections.csv':
            if os.stat(pa).st_ino != os.stat(pb).st_ino:
                diff.append(f'{rel}: another inode')
            continue
        if not filecmp.cmp(pa, pb, shallow=False):
            diff.append(f'{rel}: bytes differ')
    return diff


# ---------------------------------------------------------------------------------------------------------------------

def test_production_pin():
    """THE EXPLICIT DRIFT TEST (V_combview review F3): the production files this stage ports are the pinned ones
    (tests/production_pin.py). When production changes one, this test fails, naming the files and the commits to port - the
    stage's own tests keep running against the pin. Moving the pin is the port of those commits."""
    d = PIN.drift(PINNED)
    assert not d, PIN.drift_message(d)


def test_drift():
    """The port's texts, patterns and layout are production's at the pin (read as text): the view's (dbc40a6), the decisions'
    (dde1760 / dbc40a6) and the re-judging's (snapshot_rejudge.py)."""
    SV = _SV()
    from exopipe.phase3.lib.catalog import deep_omission as DO
    from exopipe.phase3.lib.catalog import rejudge as RJ
    src = open(os.path.join(pinned_ops(), 'snapshot_combined_view.py')).read()
    srj = open(os.path.join(pinned_ops(), 'snapshot_rejudge.py')).read()
    sdo = open(os.path.join(pinned_ops(), 'snapshot_deep_omission.py')).read()
    drv = open(os.path.join(pinned_ops(), 'snapshot_batch', 'comb_build_20261104.sh')).read()
    flat = src.replace('"\n                     "', '').replace('"\n                    "', '').replace('"\n                       "', '')
    flat = flat.replace('"\n                         "', '').replace('"\n               "', '')
    for txt in ('# queue_name: snapshot Stokes %s, %s tier - COMBINED VIEW of %d served batches\\n',
                '# run: %s (combined view; batches %s)\\n',
                '%d detections omitted by the deep-omission rule (Gregg 2026-10-24)" % n_om',
                'served detections %d, kept %d; decisions %d (%d OMIT keys%s); %s\\n'):
        assert txt in src, txt
    assert SV.QUEUE_NAME.replace('\n', '\\n') in src and SV.QUEUE_RUN.replace('\n', '\\n') in src
    assert SV.SUMMARY_LINE1 in src and SV.SUMMARY_LINE2.replace('\n', '\\n') in src
    assert SV.SUMMARY_LINE1_NEW in flat and SV.SUMMARY_WB in src and SV.DEEP_OMISSION_WB in src
    assert 'tests="T1 T2 T3 T4 T5 T6%s PASS" % (" T7" if NEW_ON else "")' in src and SV.TESTS_PASSED == 'T1 T2 T3 T4 T5 T6 T7 PASS'
    for line in (SV.VIEW_LIMITS, SV.DEEP_OMISSION_LINE, SV.CARD2_REJUDGE_LINE):   # the long header lines, joined from their literals
        assert line.replace('\n', '\\n') in flat, line[:60]
    assert 'TIERS, STOKES = ("10s", "1min"), ("V", "I")' in src
    S = _cfg().serve
    assert tuple(S.SNAPVIEW_TIERS) == ('10s', '1min') and tuple(S.SNAPVIEW_STOKES) == ('V', 'I')
    assert 'TSFX = {"10s": "", "1min": "_1min"}' in src and dict(S.SNAPVIEW_PAGE_SUFFIX) == {'10s': '', '1min': '_1min'}
    assert SV.PANEL_TOKEN.pattern == r"^rank\d+_\d\dh_\d{4}-\d\d-\d\d_([A-Za-z0-9]+)\.(?:png|page\.json)$" and \
        'PANEL_TOKEN = re.compile(r"^rank\\d+_\\d\\dh_\\d{4}-\\d\\d-\\d\\d_([A-Za-z0-9]+)\\.(?:png|page\\.json)$")' in src
    assert 're.match(r"^(.*)_(\\d{2}h_\\d{4}-\\d{2}-\\d{2})_([^_]+)$", row["source_id"])' in src and \
        SV.MEMBER_ROW.pattern == r"^(.*)_(\d{2}h_\d{4}-\d{2}-\d{2})_([^_]+)$"
    assert DO.SNAP_CID.pattern in src and DO.SNAP_PANEL.pattern in src
    # the re-point (dbc40a6): best by (ts, subband), best_cand_id '<identity>_<lst>_<date>_<band>', the fields' order and note
    assert 'row = dict(row, best_cand_id="%s_%s_%s_%s" % (row["identity_id"], top["lst"], top["date"], band),' in src
    assert 'best_snr=top["snr"], ts=top["ts"], subband=top["subband"], fit_verdict=top.get("fit_verdict", ""),' in src
    assert DO.REPOINT_FIELDS == ('best_cand_id', 'best_snr', 'ts', 'subband', 'fit_verdict', 'peak_jy_best', 'deep_omission')
    assert ('peak_jy_best="", deep_omission="%s' % DO.REPOINT_NOTE[:50]) in src.replace('"\n                                   "', '')
    assert 'note="best %s %s (%s) removed; re-pointed to %s (%s)" % (' in src
    for t in (DO.SOURCE_EMPTY_NOTE, DO.SOURCE_EMPTY_CARD2_NOTE, DO.IDENTITY_EMPTY_NOTE):
        assert '"%s"' % t in src, t
    assert 'rejudged_routes=REJ_ROUTES[(r, row["identity_id"], T)]' in src
    # Card 2 (dbc40a6): the key at 5 decimals, the 10-s Stokes V rule, the ratio, the per-frame rule of member rows and panels
    assert 'return (cand, "%.5f" % float(ra), "%.5f" % float(dec))' in src and DO.nk('a', '1', '2') == ('a', '1.00000', '2.00000')
    assert 'd.get("tier") == "10s" and d.get("stokes") == "V"' in src
    assert 'CARD2_RATIO = 0.05' in srj and _cfg().catalog_omission.CARD2_RATIO == 0.05
    assert 'return JC_C2[(r, J, cand)] > 0 and JC_C2[(r, J, cand)] == JC_ALL[(r, J, cand)]' in src
    assert 'if pm and S == "V" and T == "10s" and JC_C2 and jc_card2(' in src
    # the decisions (dde1760 / dbc40a6): the member reasons, the members' order, the fail-closed rule, the rate population
    flat_d = sdo.replace("'\n                                      '", '')
    assert DO.snapshot_member_image_reason('%s', '%s') in flat_d.replace("' % (', '.join(sorted(why)), g))", '')
    assert DO.snapshot_member_failclosed_reason('%s', '%s') in flat_d.replace("' % (', '.join(sorted(why)), g))", '')
    assert "' '.join(sorted({w.split()[0] for w in why}, key=lambda b: int(b[:-3])))" in sdo
    assert "elif not has_t and searched_sb(lst, r['night'], r['band'], proto):" in sdo
    assert 'ap = bool(not np.isfinite(rthr) or night not in rate or rate[night] <= rthr)' in sdo
    assert "rate=('%.4g' % rate[night]) if night in rate else ''" in sdo
    assert "iid = r['ts'] if T == '10s' else integration_minute_key(r['date_obs']) + '00'" in sdo
    assert 'return True if mem is None else bool(mem & om)' in sdo
    # the re-judging (snapshot_rejudge.py): the port's rejudge is production's, line for line but the refusal and the docstring
    import ast

    def body(srcs, name):
        f = next(n for n in ast.walk(ast.parse(srcs)) if isinstance(n, ast.FunctionDef) and n.name == name)
        return [ast.dump(x) for x in f.body[1:]]          # the docstring left out
    norm = lambda xs: [x.replace("Name(id='RejudgeRefusal', ctx=Load())", "Name(id='RuntimeError', ctx=Load())") for x in xs]
    port_src = open(RJ.__file__).read()
    prod_body, port_body = body(srj, 'rejudge'), norm(body(port_src, 'rejudge'))
    diff = [i for i, (x, y) in enumerate(zip(prod_body, port_body)) if x != y]
    assert len(prod_body) == len(port_body) and len(diff) <= 12, (len(prod_body), len(port_body), diff[:5])
    # the driver: stores, interpreter, localization environment
    assert 's=snapshot_batch/rechain_$(basename $r)/fits_store.csv' in drv and S.SNAPVIEW_FIT_STORE == \
        '{batch_dir}/rechain_{run}/fits_store.csv' and _cfg().catalog_snapshot.BATCH_DIR == os.path.join(OPS, 'snapshot_batch')
    assert 'PY=/opt/devel/pipeline/envs/development/bin/python' in drv and \
        _cfg().site.PYTHON[S.SNAPVIEW_PYTHON] == '/opt/devel/pipeline/envs/development/bin/python'
    assert 'P3_LOC_SHARDS=${P3_LOC_SHARDS:-12}' in drv and S.SNAPVIEW_LOC_ENV == {'P3_LOC_SHARDS': '12'}
    assert "'# combined view %s: the union of its %d member batches\\' card2_filter.csv'" in drv
    assert 'sys.path.insert(0, "/lustre/gh/main/clv15")' in src and 'import phase3_snapshot_chain as CH' in src


def test_py36():
    """The child's module (and deep_omission.py, which it loads from its file) parse and load under production's interpreter."""
    sv = os.path.join(REPO, 'exopipe', 'phase3', 'lib', 'serve', 'snapshot_view.py')
    do = os.path.join(REPO, 'exopipe', 'phase3', 'lib', 'catalog', 'deep_omission.py')
    code = ('import sys, importlib.util\nsv, do = sys.argv[1], sys.argv[2]\nfor p in (sv, do):\n    compile(open(p).read(), p, "exec")\n'
            'spec = importlib.util.spec_from_file_location("sv", sv); m = importlib.util.module_from_spec(spec)\n'
            'spec.loader.exec_module(m)\nD = m.load_standalone(do)\nprint("OK", m.STAGES, D.batch_tag("/x/Run_SNAPB6I2_20261101"))\n')
    r = subprocess.run([PY36, '-c', code, sv, do], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True,
                       timeout=300)
    assert r.returncode == 0 and 'OK' in r.stdout and 'B6I2' in r.stdout, r.stdout[-1500:]


_W = {}


def shared_world():
    """The world and the STAGE's view (omit, build, filtercat), built once - production's side is not involved (F3: the stage's
    own tests never wait on production's builder); test_parity and test_repoint_pinned add production's view (prod_world)."""
    if not _W:
        root, rdir, proto, runs, batch_dir = world()
        r, port, out = port_view(root, rdir, proto, runs, batch_dir)
        _W.update(root=root, rdir=rdir, proto=proto, runs=runs, batch_dir=batch_dir, port=port, out=out, cli=r)
        assert r.returncode == 0, ('the stage\'s view of the shared world', r.returncode, r.stderr[-3000:])
    return _W


def prod_world():
    """shared_world + production's view of the same world (the pinned builder; built once)."""
    W = shared_world()
    if 'prod' not in W:
        prod_out = os.path.join(W['root'], 'prod', VIEW)
        W['prod_log'] = prod_view(W['root'], W['rdir'], W['proto'], W['runs'], W['batch_dir'], prod_out)
        W['prod'] = prod_out
    return W


def test_parity():
    """Production's builder + driver steps 1-2 and the stage (omit, build, filtercat) on the synthetic world: equal views."""
    from exopipe.phase3.lib.catalog import deep_omission as DO
    W = prod_world()
    r = W['cli']
    assert r.returncode == 0, (r.returncode, r.stderr[-3000:])
    diff = compare_views(W['prod'], W['port'])
    assert not diff, diff[:10]
    # the world exercised what it was built for
    summ = json.load(open(os.path.join(W['port'], 'combined_view_summary.json')))
    rec = _rows(os.path.join(W['port'], 'deep_omission_dropped.csv'))
    kinds = collections.Counter(x['kind'] for x in rec)
    assert summ['multi_batch_identities'] >= 2 and summ['renamed'] >= 1, summ
    for k in ('detection', 'identity dropped', 'source dropped', 'member row', 'panel', 'source partial', 'card2 detection',
              'card2 member row', 'card2 panel', 'source re-judged', 'member row re-judged'):
        assert kinds.get(k), (k, kinds)
    # dde1760: a per-image wideband key (05h Green I night 1, 46 MHz omitted) omitted one Green detection (its frame holds 46
    # MHz) and kept the other (a frame stacked without it); a fail-closed key (06h, no member table) dropped J0612+2200
    assert summ['deep_omission']['omit_wb_image_keys'] >= 1, summ['deep_omission']
    om_c = {x['cand_id'] for x in rec if x['kind'] == 'detection'}
    assert any(c.endswith('_050200') and '_Green_I_10s_' in c for c in om_c), sorted(om_c)[:5]
    assert not any(c.endswith('_050230') and '_Green_I_10s_' in c for c in om_c)
    assert ('T2A', 'J0612+2200') in {(x['batch'], x['identity']) for x in rec if x['kind'] == 'identity dropped'}
    assert summ['card2']['removed_from_view'] >= 2 and summ['rejudge']['rejudged_out'] >= 1, (summ['card2'], summ['rejudge'])
    assert os.path.getsize(os.path.join(W['port'], 'rejudged.csv')) > 100
    names = {r_['source_id'] for p in glob.glob(os.path.join(W['port'], 'source_catalogs', '*_sources_survivors.csv'))
             for r_ in _rows(p)}
    assert 'J0600+2500' in names and any(n.startswith('J0600+2500~') for n in names), sorted(names)[:8]
    assert os.path.isdir(os.path.join(W['port'], 'candidates', 'I', 'J0615p3300_46MHz_1min')) or \
        os.path.isdir(os.path.join(W['port'], 'candidates', 'I', 'J0615p3301_46MHz_1min')), 'the merged 1-minute page'
    assert any(fn.startswith('T2B_notes.txt') for fn in sum((os.listdir(d) for d in glob.glob(os.path.join(W['port'], 'candidates',
                                                                                                          '[IV]', '*'))), [])), \
        'a panel-name clash gets the batch tag'
    # the stage's records: every unit done, every content test passed
    recs = unit_records(W['out'])
    for st in ('omit', 'build', 'filtercat'):
        j = recs.get(f'snapshot_{VIEW}_{st}')
        assert j and j['status'] == 'done', (st, j and (j['status'], j['reason']))
        assert j['tests'] and all(t['passed'] for t in j['tests']), [t for t in j['tests'] if not t['passed']][:2]
    names_t = {t['name'] for t in recs[f'snapshot_{VIEW}_build']['tests']}
    for t in ('snapshot_view.files', 'snapshot_view.queues', 'snapshot_view.pages', 'snapshot_view.panels',
              'snapshot_view.identities', 'snapshot_view.record', 'G-V8_decisions_current',
              'G-V3 omission_served.served_counts_add_up', 'G-V3 omission_rejudged.rejudged_table'):
        assert t in names_t, (t, names_t)
    assert {'omission_served.served_counts_add_up', 'omission_rejudged.rejudged_table'} <= \
        {t['name'] for t in recs[f'snapshot_{VIEW}_omit']['tests']}
    assert recs[f'snapshot_{VIEW}_build']['meta']['build']['stale_best'] == 0      # dbc40a6: every row ranked by a kept detection
    assert recs[f'snapshot_{VIEW}_filtercat']['meta']['filtercat_procs'] == 6    # 1b9475c: SNAPCAT_PROCS, the Lustre rule
    # the omit product is the deep-omission stage's (deep_omission.csv = the view's, byte for byte)
    od = os.path.join(W['out'], 'snapshot', 'deep_omission', VIEW)
    assert filecmp.cmp(os.path.join(od, 'deep_omission.csv'), os.path.join(W['prod'], 'deep_omission.csv'), shallow=False)
    assert json.load(open(os.path.join(od, 'deep_omission_summary.json')))['digests']['served'].startswith(f'{summ["deep_omission"]["served"]}:')
    assert DO.omitted_keys(_rows(os.path.join(od, 'deep_omission.csv')))


def test_repoint():
    """THE RE-POINT, fixed (exopipe_ops dbc40a6; the port review's defect RP): a partial per-tier source whose best detection - the
    one of its own (ts, subband) - was removed is re-pointed to the largest-|S/N| detection it keeps (production and the port):
    J0530+3500 and J0545+3700 (their night-1 bests omitted), best_cand_id '<identity>_<lst>_<date>_<band>', fit_verdict, the
    note; a source still qualifying after re-judging carries rejudged_routes."""
    from exopipe.phase3.lib.catalog import deep_omission as DO
    W = prod_world()
    for view in (W['prod'], W['port']):
        rows = {r['source_id']: r for r in _rows(os.path.join(view, 'source_catalogs_I_10s', '46MHz_sources_survivors.csv'))}
        for sid, snr in (('J0545+3700_46MHz', '11.00'), ('J0530+3500_46MHz', '10.60')):
            rp = rows[sid]
            assert rp['best_snr'] == snr and rp['deep_omission'] == DO.REPOINT_NOTE, rp
            assert rp['best_cand_id'] == '%s_05h_%s_46MHz' % (sid.split('_')[0], NIGHTS[1]) and rp['ts'].startswith(
                NIGHTS[1].replace('-', '')), rp
            assert rp.get('rejudged_routes'), rp                                        # still qualifies: its routes
            dets = _rows(os.path.join(view, rp['review_dir'], 'detections.csv'))
            assert [d['snr'] for d in dets] == [snr], dets                              # its page holds only the kept one
    rec = _rows(os.path.join(W['port'], 'deep_omission_dropped.csv'))
    part = sorted(x['source'] for x in rec if x['kind'] == 'source partial')
    assert {'J0530+3500_46MHz', 'J0545+3700_46MHz'} <= set(part), part


def _mutated_omit(W, name, fix):
    """A copy of the stage's omit product with ``fix`` applied to its kept rows (the summary unchanged)."""
    from exopipe.phase3.lib.catalog import deep_omission as DO
    od = os.path.join(W['out'], 'snapshot', 'deep_omission', VIEW)
    d = os.path.join(W['root'], 'mut_' + name)
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(od, d)
    rr = _rows(os.path.join(d, 'deep_omission_kept.csv'))
    fix(rr)
    _wcsv(os.path.join(d, 'deep_omission_kept.csv'), DO.KEPT_COLS, rr)
    return d


def _build_job(W, omit_dir, out, runs=None, copy_fallback=False):
    """The build child directly (what the build stage runs), under production's interpreter: (rc, log)."""
    from exopipe.phase3.lib.catalog import deep_omission
    SV = _SV()
    S = _cfg().serve
    co = _cfg().catalog_omission
    proto = json.load(open(os.path.join(omit_dir, 'deep_omission_summary.json')))['proto']
    job = dict(out=out, runs=runs or W['runs'], kept_dir=omit_dir, tiers=list(S.SNAPVIEW_TIERS), stokes=list(S.SNAPVIEW_STOKES),
               suffix=dict(S.SNAPVIEW_PAGE_SUFFIX), clv15='/lustre/gh/main/clv15', ops=OPS,
               deep_omission_py=os.path.abspath(deep_omission.__file__), result=out + '.result.json',
               copy_fallback=copy_fallback, proto=proto,
               names=dict(noise=dict(co.SNAPSHOT_NOISE), members=co.SNAPSHOT_MEMBERS, admission=co.SNAPSHOT_ADMISSION),
               card2_ratio=co.CARD2_RATIO, card2_file=co.SNAPSHOT_CARD2, read_workers=co.READ_WORKERS)
    jp = out + '.job.json'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(job, open(jp, 'w'))
    r = subprocess.run([PY36, os.path.abspath(SV.__file__), 'build', jp], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       universal_newlines=True, timeout=1800)
    return r.returncode, r.stdout


def test_kept_refusals():
    """G-K and G-V3 / G-V4: the build refuses a kept product that does not apply exactly, and moves nothing into place; a
    refusal after the build made its .part moves the .part aside to <view>.refused_<UTC> (F7: the view name stays free, nothing
    is deleted); a .part the build did not make (G-V2) is never touched."""
    W = shared_world()
    cases = []

    def moved(rr):                                   # a kept detection moved to another identity of its batch
        a = next(r for r in rr if r['kind'] == 'detection kept' and r['identity'] == 'J0520+3000')
        a['identity'] = 'J0600+2500'
    cases.append(('moved', moved, 'G-K'))
    cases.append(('member', lambda rr: rr.remove(next(r for r in rr if r['kind'] == 'member row dropped')),
                  "production's rule drops it, the kept product keeps it"))
    cases.append(('panel_extra', lambda rr: rr.append(dict(next(r for r in rr if r['kind'] == 'panel dropped'),
                                                           note='rank0001_05h_2025-01-02_46MHz20250102050000.png')),
                  'the kept product holds rows no served row matched'))
    for name, fix, want in cases:
        d = _mutated_omit(W, name, fix)
        out = os.path.join(W['root'], 'refused', name, VIEW)
        rc, log = _build_job(W, d, out)
        assert rc == 2 and 'REFUSING' in log and want in log, (name, rc, log[-1500:])
        assert not os.path.exists(out) and not os.path.exists(out + '.part'), (name, 'a refused build left its view or .part')
        aside = glob.glob(out + '.refused_*')
        assert len(aside) == (1 if 'moved aside to' in log else 0), (name, aside, log[-600:])
        if name != 'moved':          # refused after the .part was made (step 0 refuses 'moved' before): moved aside (F7)
            assert aside and os.path.isdir(os.path.join(aside[0], 'source_catalogs')) and aside[0] in log, (name, aside, log[-600:])
    # G-V2 in the child: a .part this build did not make (another build's, running or refused) stays where it is
    out = os.path.join(W['root'], 'refused', 'other_part', VIEW)
    os.makedirs(out + '.part')
    rc, log = _build_job(W, os.path.join(W['out'], 'snapshot', 'deep_omission', VIEW), out)
    assert rc == 2 and 'G-V2' in log and os.path.isdir(out + '.part') and not glob.glob(out + '.refused_*'), log[-800:]
    # G-V4, the counts (F5): served counts that are not the served set's (the child; the step side's K5 refuses them earlier)
    d = _mutated_omit(W, 'counts', lambda rr: None)
    sp = os.path.join(d, 'deep_omission_served.csv')
    rr = _rows(sp)
    rr[0]['n'] = str(int(rr[0]['n']) + 1)
    _wcsv(sp, list(rr[0]), rr)
    rc, log = _build_job(W, d, os.path.join(W['root'], 'refused', 'counts', VIEW))
    assert rc == 2 and "G-V4 the omit product's served counts" in log, log[-800:]
    # G-V3: the omit product of other runs (the runs reversed)
    od = os.path.join(W['out'], 'snapshot', 'deep_omission', VIEW)
    rc, log = _build_job(W, od, os.path.join(W['root'], 'refused', 'order', VIEW), runs=list(reversed(W['runs'])))
    assert rc == 2 and 'G-V3' in log, log[-800:]
    # G-V4: a served run changed after the omit stage (one detection more on a page of the general catalogue)
    changed = os.path.join(W['root'], 'changed')
    shutil.rmtree(changed, ignore_errors=True)
    runs2 = []
    for r in W['runs']:
        shutil.copytree(r, os.path.join(changed, os.path.basename(r)), symlinks=True)
        runs2.append(os.path.join(changed, os.path.basename(r)))
    d = _mutated_omit(W, 'runs2', lambda rr: None)
    sj = json.load(open(os.path.join(d, 'deep_omission_summary.json')))
    sj['runs'] = runs2
    json.dump(sj, open(os.path.join(d, 'deep_omission_summary.json'), 'w'))
    p = os.path.join(runs2[2], 'candidates', 'Red', 'rank0001_Red_J0600+2500', 'detections.csv')
    rr = _rows(p)
    _wcsv(p, DET_COLS, rr + [dict(rr[0], cand_id=rr[0]['cand_id'][:-1] + '9', ra_peak='95.00999')])
    rc, log = _build_job(W, d, os.path.join(W['root'], 'refused', 'changed', VIEW), runs=runs2)
    assert rc == 2 and 'G-V4' in log, log[-800:]


def test_gates():
    """G-V1, G-V2, G-V5, G-V6 and the --view-omit / --stages omit conflict, through the real CLI (exit 2, nothing written).
    G-V2 refuses BEFORE any stage when omit or build is asked (V_combview review F2): an existing view's omit product is its
    product of record, never rewritten by a re-run of its name."""
    W = shared_world()
    root, runs = W['root'], W['runs']
    base = ['serve', '--tier', 'snapshot', '--family', 'comb', '--set', f'catalog_omission.REFUSALS={W["rdir"]}', '--cores',
            'none', '--set', f'paths.SNAPSHOT_FLEET_OUT={W["proto"]}']
    gout = os.path.join(root, 'gates_out')
    for extra, want in ((['--view-runs', ','.join(runs)], 'G-V1 view name required'),
                        (['--run', 'Run_SNAPCOMB_G', '--view-runs', ''], 'G-V1 no batch run'),
                        (['--run', 'Run_SNAPCOMB_G', '--view-runs', f'{runs[0]},{runs[0]}'], 'listed twice'),
                        (['--run', 'Run_SNAPCOMB_G', '--view-runs', f'{runs[0]},{root}'], 'without source_catalogs'),
                        (['--run', 'NOT_A_SNAP', '--view-runs', runs[0]], "starts with 'Run_SNAP'"),
                        (['--run', 'Run_SNAPCOMB_G', '--view-runs', runs[0], '--view-omit', root, '--stages', 'omit,build'],
                         'do not also run the omit stage')):
        r = _cli(*base, *extra, '--out', gout)
        assert r.returncode == 2 and want in r.stderr, (extra, r.returncode, r.stderr[-600:])
    # G-V2: the view exists (the stage's view of this world) - for omit,build, omit alone and build alone the gate refuses
    # before any stage: the omit product of the existing view is not rewritten (the reviewer's case O1: a refusal file added
    # to the record, the same command again), its files' bytes and times unchanged (F2)
    od = os.path.join(W['out'], 'snapshot', 'deep_omission', VIEW)

    def state(d):
        return {rel: (os.stat(p).st_mtime_ns, open(p, 'rb').read()) for rel, p in sorted(tree(d).items())}
    rdir2 = shutil.copytree(W['rdir'], os.path.join(root, 'gates_refusals'))
    with open(os.path.join(rdir2, '06h_55MHz_searchtier_excluded.csv'), 'w') as f:
        f.write('06h,55MHz,%s,V,9.0,4.0,%s\n' % (NIGHTS[2], TAG))
    before = state(od)
    time.sleep(1.1)
    for st in ('omit,build', 'omit', 'build'):
        r = _cli(*base, '--run', VIEW, '--view-runs', ','.join(runs), '--out', W['out'], '--stages', st,
                 '--set', f'catalog_snapshot.BATCH_DIR={W["batch_dir"]}', '--set', f'catalog_omission.REFUSALS={rdir2}')
        assert r.returncode == 2 and 'G-V2' in r.stderr and 'exists' in r.stderr, (st, r.returncode, r.stderr[-600:])
    assert state(od) == before, 'the existing view\'s omit product was rewritten'
    # a .part alone (a build running, or one that failed) blocks the name the same way, before anything is written
    gp = os.path.join(gout, 'snapshot', 'Run_SNAPCOMB_GP')
    os.makedirs(gp + '.part')
    r = _cli(*base, '--run', 'Run_SNAPCOMB_GP', '--view-runs', ','.join(runs), '--out', gout, '--stages', 'omit,build',
             '--set', f'catalog_snapshot.BATCH_DIR={W["batch_dir"]}')
    assert r.returncode == 2 and 'G-V2' in r.stderr, r.stderr[-600:]
    assert not os.path.exists(os.path.join(gout, 'snapshot', 'deep_omission', 'Run_SNAPCOMB_GP')) and not os.path.exists(gp)
    # G-V6: an injected run's products as a view run (the CLI's source-flag guard, then the stage's own)
    inj = os.path.join(root, 'inj')
    shutil.rmtree(inj, ignore_errors=True)
    shutil.copytree(runs[2], os.path.join(inj, os.path.basename(runs[2])), symlinks=True)
    from exopipe.phase3.core import record
    p = os.path.join(inj, '_records', record.INJECTED_DIR, 'search_x.json')
    os.makedirs(os.path.dirname(p))
    json.dump({'injected': 'test'}, open(p, 'w'))
    r = _cli(*base, '--run', 'Run_SNAPCOMB_G', '--view-runs', f'{runs[0]},{os.path.join(inj, os.path.basename(runs[2]))}',
             '--out', gout, '--dry-run')
    assert r.returncode == 2 and 'INJECTED-RUN GUARD' in r.stderr, r.stderr[-600:]
    lst = os.path.join(root, 'view_runs.txt')
    open(lst, 'w').write(f'{runs[0]}  # first\n{os.path.join(inj, os.path.basename(runs[2]))}\n')
    r = _cli(*base, '--run', 'Run_SNAPCOMB_G', '--view-runs', '@' + lst, '--out', gout, '--dry-run')
    assert r.returncode == 2 and 'INJECTED-RUN GUARD' in r.stderr, r.stderr[-600:]
    # G-V5: another file system (simulated: the view's parent reports another device)
    SV = _SV()
    import types
    from exopipe.phase3.config import CFG
    from exopipe.phase3.lib.serve import flow
    rr = record.Run('serve', ['test'], os.path.join(root, 'gv5_out'), CFG)
    plan = SV.ViewPlan(types.SimpleNamespace(run_label='Run_SNAPCOMB_GV5', view_runs=','.join(runs), view_omit=None),
                       os.path.join(root, 'gv5_out'))
    plan.omit_dir = os.path.join(W['out'], 'snapshot', 'deep_omission', VIEW)
    plan.parity = False                  # as a production view (a scratch / parity view copies instead: test_copy_fallback)
    real_dev = SV._dev
    SV._dev = lambda p: real_dev(p) + (1 if os.path.abspath(p) in runs else 0)
    try:
        CFG.authorize(out=os.path.join(root, 'gv5_out'))
        u = SV.build_stage(rr, plan)
    finally:
        SV._dev = real_dev
        CFG.reset()
        CFG.authorize()
    assert u.status == 'refused' and 'G-V5' in u.reason, (u.status, u.reason)
    assert not os.path.exists(plan.U) and flow.halts(u, 'build')


def test_filtercat_procs():
    """exopipe_ops 1b9475c (the parallel filter catalogue: SNAPCAT_PROCS readers, merged in glob order): both filter-catalogue
    stages ALWAYS pass SNAPCAT_PROCS - the view's serve.SNAPVIEW_FILTERCAT_PROCS, the snapshot pages' pages_snapshot.FILTERCAT_PROCS,
    6 by default (the Lustre rule; production's default 16) - so an inherited value never reaches the child (SNAPCAT_ is no cleared
    switch family); a reader count that is not an integer >= 1 refuses before the child runs (F-G4 / P22). The built view's
    filtercat unit records its count (test_parity)."""
    import types
    SV = _SV()
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import record
    from exopipe.phase3.lib.serve import scripts
    from exopipe.phase3.lib.pages import common, mirror, tier_snapshot as TSN
    assert CFG.serve.SNAPVIEW_FILTERCAT_PROCS == 6 and CFG.pages_snapshot.FILTERCAT_PROCS == 6
    assert not any('SNAPCAT_'.startswith(x) for x in CFG.runner.CHILD_ENV_CLEAR_PREFIXES)   # so the stages must pass it
    for bad in (0, -2, 2.5, '6', True, None):
        try:
            SV.filtercat_procs(bad)
        except ValueError as e:
            assert 'F-G4' in str(e), e
        else:
            raise AssertionError(f'reader count {bad!r} accepted')
    W = shared_world()
    root = os.path.join(W['root'], 'fcp')
    shutil.rmtree(root, ignore_errors=True)
    seen = []

    class Stub:
        ok, refused, rc, reason, tail = False, True, 2, 'stub child (not run)', ''

    def stub(*a, **kw):
        seen.append(dict(kw.get('env') or {}))
        return Stub()
    real_run, real_lrun, real_gate, real_script = scripts.run, common.lrun, mirror.gate, common.legacy_script
    old_env = os.environ.get('SNAPCAT_PROCS')
    os.environ['SNAPCAT_PROCS'] = '64'                 # an inherited value: never the child's
    scripts.run = stub
    common.lrun = lambda u, script, args, **kw: stub(**kw)
    mirror.gate = lambda *a, **kw: None
    common.legacy_script = lambda roots, name, fn: os.path.join(OPS, fn)
    try:
        for n, want in ((6, 'F-G3'), (3, 'F-G3'), (0, 'F-G4')):
            out = os.path.join(root, f'view_{n}')
            CFG.authorize(out=out)
            CFG.set('catalog_snapshot.BATCH_DIR', W['batch_dir'])
            CFG.set('serve.SNAPVIEW_FILTERCAT_PROCS', str(n))
            rr = record.Run('serve', ['test'], out, CFG)
            plan = SV.ViewPlan(types.SimpleNamespace(run_label='Run_SNAPCOMB_FCP', view_runs=','.join(W['runs']), view_omit=None),
                               out)
            os.makedirs(plan.U)
            k = len(seen)
            u = SV.filtercat_stage(rr, plan)
            assert u.status == 'refused' and want in u.reason, (n, u.status, u.reason)
            assert seen[k:] == ([{'SNAPCAT_PROCS': str(n)}] if n else []), (n, seen[k:])
            assert os.listdir(plan.U) == [], 'the filtercat stage wrote into the view on a refusal'
            CFG.reset()
            CFG.authorize()
        for n, want in ((6, 'P22 filter catalogue failed'), (0, 'P22 F-G4')):
            out = os.path.join(root, f'pages_{n}')
            CFG.authorize(out=out)
            CFG.set('pages_snapshot.FILTERCAT_PROCS', str(n))
            rr = record.Run('pages', ['test'], out, CFG)
            c = types.SimpleNamespace(name='Run_SNAPB_FCP', roots={}, store=os.path.join(out, 'fits_store.csv'),
                                      RUN=os.path.join(out, 'Run_SNAPB_FCP'), W=out)
            os.makedirs(c.RUN)
            open(c.store, 'w').write('cand_id\n')
            k = len(seen)
            done = {'Run_SNAPB_FCP:build': 'done'}
            TSN.stage_filtercat(c, rr, done)
            assert seen[k:] == ([{'SNAPCAT_PROCS': str(n)}] if n else []), (n, seen[k:])
            assert 'Run_SNAPB_FCP:filtercat' not in done
            j = json.load(open(record.record_path(out, 'pages', 'Run_SNAPB_FCP:filtercat')))
            assert j['status'] == 'refused' and want in j['reason'], (n, j['status'], j['reason'])
            CFG.reset()
            CFG.authorize()
    finally:
        scripts.run, common.lrun, mirror.gate, common.legacy_script = real_run, real_lrun, real_gate, real_script
        if old_env is None:
            os.environ.pop('SNAPCAT_PROCS', None)
        else:
            os.environ['SNAPCAT_PROCS'] = old_env
        CFG.reset()
        CFG.authorize()


def _copy_runs(W, name):
    """Copies of the world's batch runs (to damage), in the view's order."""
    d = os.path.join(W['root'], 'runs_' + name)
    shutil.rmtree(d, ignore_errors=True)
    return [shutil.copytree(r, os.path.join(d, os.path.basename(r)), symlinks=True) for r in W['runs']]


def _view_cli(W, runs, out, stages, rdir=None, proto=None, extra=()):
    return _cli('serve', '--tier', 'snapshot', '--family', 'comb', '--run', VIEW, '--view-runs', ','.join(runs), '--out', out,
                '--set', f'catalog_omission.REFUSALS={rdir or W["rdir"]}', '--set', f'paths.SNAPSHOT_FLEET_OUT={proto or W["proto"]}',
                '--set', f'catalog_snapshot.BATCH_DIR={W["batch_dir"]}', '--stages', stages, '--cores', 'none', *extra)


def lose_tier_catalogue(runs):
    """I3: batch A's I 10-s 46 MHz per-tier catalogue lost (survivors and members) - its identities with no other I/V 10-s
    source leave the 10-s tier unrecorded (production: every stage done, every test passed)."""
    for fn in ('46MHz_sources_survivors.csv', '46MHz_source_members.csv'):
        os.remove(os.path.join(runs[0], 'source_catalogs_I_10s', fn))


def lose_page_detections(runs):
    """I2: a KEPT per-tier source's page detections.csv lost with its member rows (batch A, J0600+2500 23 MHz I 10 s) -
    production records it as 'source dropped ... every detection omitted', an input defect booked to the deep rule."""
    os.remove(os.path.join(runs[0], 'candidates', 'I', 'J0600+2500_23MHz', 'detections.csv'))
    mp = os.path.join(runs[0], 'source_catalogs_I_10s', '23MHz_source_members.csv')
    _wcsv(mp, ['source_id', 'cand_id', 'det_snr', 'det_date', 'band'],
          [x for x in _rows(mp) if not x['source_id'].startswith('J0600+2500_')])


def test_incomplete_inputs():
    """F4 (V_combview review): a batch run copied or re-served incompletely refuses - at the omit stage (G4b: a batch's per-tier
    survivors are not exactly the identities its served_tiers.csv serves per tier; G4c: a kept per-tier source whose page holds
    no detection) with nothing written, and in the build child on the same damage after the omit stage (G-V7). Never booked as
    a deep omission, never a source lost unrecorded."""
    W = shared_world()
    for name, damage, want in (('I3', lose_tier_catalogue, 'G4b: T2A tier 10s'),
                               ('I2', lose_page_detections, 'G4c: T2A source_catalogs_I_10s source J0600+2500_23MHz')):
        runs = _copy_runs(W, name)
        damage(runs)
        out = os.path.join(W['root'], 'inc_out_' + name)
        r = _view_cli(W, runs, out, 'omit')
        assert r.returncode == 2 and want in r.stderr, (name, r.returncode, r.stderr[-800:])
        assert not os.path.exists(os.path.join(out, 'snapshot', 'deep_omission', VIEW)), (name, 'the refused omit wrote')
        # the same damage AFTER the omit stage (the product of the intact runs, re-pointed to the damaged copies): the child
        old = {os.path.abspath(a): b for a, b in zip(W['runs'], runs)}

        def repoint(rr):
            for x in rr:
                x['run'] = old.get(x['run'], x['run'])
        d = _mutated_omit(W, 'inc_' + name, repoint)
        sj = json.load(open(os.path.join(d, 'deep_omission_summary.json')))
        sj['runs'] = runs
        json.dump(sj, open(os.path.join(d, 'deep_omission_summary.json'), 'w'))
        bout = os.path.join(W['root'], 'inc_build_' + name, VIEW)
        rc, log = _build_job(W, d, bout, runs=runs)
        want_c = 'G-V7 T2A tier 10s' if name == 'I3' else 'G-V7 T2A source_catalogs_I_10s source J0600+2500_23MHz'
        assert rc == 2 and want_c in log, (name, rc, log[-1200:])
        assert not os.path.exists(bout) and not os.path.exists(bout + '.part'), name


def test_stale_omit():
    """F5 (V_combview review): the build applies only an omit product whose decisions are CURRENT - G-V8 re-decides from the
    refusal record and fleet records the build reads (with the served counts the product records) and refuses a product
    decided from another record, a refusal added after the omit stage, a noise table rewritten after it; K5 (the omit
    product's content test) refuses served counts that are not its kept + omitted detections. Nothing is built."""
    W = shared_world()
    rdir = shutil.copytree(W['rdir'], os.path.join(W['root'], 'stale_refusals'))
    proto = shutil.copytree(W['proto'], os.path.join(W['root'], 'stale_fleet'))
    out = os.path.join(W['root'], 'stale_out')
    view = os.path.join(out, 'snapshot', VIEW)
    r = _view_cli(W, W['runs'], out, 'omit', rdir=rdir, proto=proto)
    assert r.returncode == 0, r.stderr[-2000:]

    def refused_g_v8(want, **kw):
        r = _view_cli(W, W['runs'], out, 'build', **kw)
        assert r.returncode == 2 and 'G-V8' in r.stderr and want in r.stderr, (want, r.returncode, r.stderr[-800:])
        assert not os.path.exists(view) and not os.path.exists(view + '.part')
        u = unit_records(out)[f'snapshot_{VIEW}_build']
        assert u['status'] == 'refused' and u['reason'].startswith('G-V8'), (u['status'], u['reason'][:300])
    refused_g_v8('was decided from the refusal record', rdir=W['rdir'], proto=proto)      # another record than the product's
    extra = os.path.join(rdir, '06h_55MHz_searchtier_excluded.csv')
    with open(extra, 'w') as f:                                                         # a refusal added after the omit stage
        f.write('06h,55MHz,%s,V,9.0,4.0,%s\n' % (NIGHTS[2], TAG))
    refused_g_v8('the omit product is stale', rdir=rdir, proto=proto)
    os.remove(extra)
    p = os.path.join(proto, f'05h_{NIGHTS[0]}', 'noise_46MHz_I_median.csv')             # a noise table rewritten after it
    keep = open(p, 'rb').read()
    _wcsv(p, NOISE_COLS, [dict(x, global_rms_jyb='%.6f' % (float(x['global_rms_jyb']) / 4.0)) for x in _rows(p)])
    refused_g_v8('the omit product is stale', rdir=rdir, proto=proto)
    open(p, 'wb').write(keep)
    # K5: served counts tampered (a copy of the product, given with --view-omit)
    od = os.path.join(out, 'snapshot', 'deep_omission', VIEW)
    d = shutil.copytree(od, os.path.join(W['root'], 'stale_counts'))
    sp = os.path.join(d, 'deep_omission_served.csv')
    rr = _rows(sp)
    rr[0]['n'] = str(int(rr[0]['n']) + 1)
    _wcsv(sp, list(rr[0]), rr)
    r = _view_cli(W, W['runs'], out, 'build', rdir=rdir, proto=proto, extra=('--view-omit', d))
    assert r.returncode == 2 and 'G-V3' in r.stderr, r.stderr[-800:]
    u = unit_records(out)[f'snapshot_{VIEW}_build']
    bad = [t for t in u['tests'] if not t['passed']]
    assert any(t['name'] == 'G-V3 omission_kept.kept_product_follows_the_drop' and 'K5' in str(t['detail']) for t in bad), bad
    assert any(t['name'] == 'G-V3 omission_served.served_counts_add_up' for t in bad), bad
    assert not os.path.exists(view)


M1_OLD = """                        if skip:
                            REC.append(dict(kind='panel', batch=TAG[r], source=row['source_id'], stokes=S, tier=T,
                                            band=pm.group(3), note=fn))
                            continue
"""
M1_NEW = """                        if skip:
                            REC.append(dict(kind='panel', batch=TAG[r], source=row['source_id'], stokes=S, tier=T,
                                            band=pm.group(3), note=fn))
"""


def test_m1_mutation():
    """F1, PROVEN on the reviewer's mutation M1 (V_combview review; RV/tools/mutate.py): a private copy of this tree whose build
    records an omitted panel as dropped but links it anyway (the `continue` removed) - the kept product and the build agree,
    production's T3 counts in the same loop, V5 counts the record. Before F1 every stage ended done and every test passed, the
    view serving 4 panel files of the OMIT key 05h 2025-06-01 46 MHz I 10 s. Now the build's read-back refuses (V3: the panels'
    own keys; V7: the page's files against its member batch sources' less the panel drops), the built view is moved aside (F7)
    and nothing is left under its name."""
    W = shared_world()
    tree_m1 = os.path.join(SCRATCH, 'm1_tree')
    shutil.rmtree(tree_m1, ignore_errors=True)
    shutil.copytree(REPO, tree_m1, ignore=shutil.ignore_patterns('.git', '__pycache__', 'tests', 'docs'))
    p = os.path.join(tree_m1, 'exopipe', 'phase3', 'lib', 'serve', 'snapshot_view.py')
    src = open(p).read()
    assert src.count(M1_OLD) == 1, 'the M1 anchor moved: update M1_OLD to the panel loop of snapshot_view.build'
    open(p, 'w').write(src.replace(M1_OLD, M1_NEW))
    out = os.path.join(W['root'], 'm1_out')
    r = _cli('serve', '--tier', 'snapshot', '--family', 'comb', '--run', VIEW, '--view-runs', ','.join(W['runs']), '--out', out,
             '--set', f'catalog_omission.REFUSALS={W["rdir"]}', '--set', f'paths.SNAPSHOT_FLEET_OUT={W["proto"]}',
             '--set', f'catalog_snapshot.BATCH_DIR={W["batch_dir"]}', '--stages', 'omit,build,filtercat', '--cores', 'none',
             repo=tree_m1)
    view = os.path.join(out, 'snapshot', VIEW)
    recs = unit_records(out)
    assert recs[f'snapshot_{VIEW}_omit']['status'] == 'done', recs[f'snapshot_{VIEW}_omit']['reason']
    u = recs[f'snapshot_{VIEW}_build']
    assert r.returncode == 2 and u['status'] == 'refused', (r.returncode, u['status'], u['reason'][:500], r.stderr[-1500:])
    assert 'fails its read-back tests' in u['reason'] and 'snapshot_view.pages' in u['reason'] and \
        'snapshot_view.panels' in u['reason'], u['reason'][:600]
    bad = {t['name']: str(t['detail']) for t in u['tests'] if not t['passed']}
    assert 'panels of an omitted instant are served' in bad['snapshot_view.pages'], bad
    assert 'files no member batch source keeps' in bad['snapshot_view.panels'], bad
    assert set(bad) == {'snapshot_view.pages', 'snapshot_view.panels'}, sorted(bad)          # the rest of the view is right
    assert f'snapshot_{VIEW}_filtercat' not in recs and not os.path.exists(view) and not os.path.exists(view + '.part')
    aside = glob.glob(view + '.refused_*')
    assert len(aside) == 1 and aside[0] in u['reason'], (aside, u['reason'][:400])
    served = [os.path.relpath(x, aside[0]) for x in glob.glob(os.path.join(aside[0], 'candidates', 'I', 'J0530p3500_46MHz',
                                                                          f'rank0001_05h_{NIGHTS[0]}_46MHz*'))]
    assert served, 'the mutation linked no omitted panel: M1 did not act'
    # control: the unmutated stage's view of the same world passes both (test_parity: every content test passed)


def test_figure_seed():
    """F10 (V_combview review): the localize stage runs snapshot_localize_tiers.sh with PYTHONHASHSEED =
    serve.SNAPSHOT_FIGURE_HASHSEED ('0'), over an inherited seed, besides the view's SNAPVIEW_LOC_ENV - the figure pool's workers
    inherit it, so a figure is a function of its tables; '' leaves the seed as production has it (unset)."""
    import types
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import record
    from exopipe.phase3.lib.serve import scripts, snapshot_copy
    W = shared_world()
    seen = []
    real = scripts.run_shell

    def fake(name, args, env=None, **kw):
        seen.append((name, list(args), dict(env or {}), scripts._base_env(env)))
        return 0, 'stub\nLOCALIZE-TIERS-OK\n'
    out = os.path.join(W['root'], 'seed_out')
    os.makedirs(out, exist_ok=True)
    plan = types.SimpleNamespace(U=os.path.join(out, 'snapshot', VIEW), run_name=VIEW, out=out)
    inherited = os.environ.get('PYTHONHASHSEED')
    os.environ['PYTHONHASHSEED'] = '7'
    scripts.run_shell = fake
    try:
        CFG.authorize(out=out)
        rr = record.Run('serve', ['test'], out, CFG)
        snapshot_copy.localize_stage(rr, plan, env=dict(CFG.serve.SNAPVIEW_LOC_ENV))
        CFG.set('serve.SNAPSHOT_FIGURE_HASHSEED', '')
        snapshot_copy.localize_stage(rr, plan, env=dict(CFG.serve.SNAPVIEW_LOC_ENV))
    finally:
        scripts.run_shell = real
        CFG.reset()
        CFG.authorize()
        if inherited is None:
            os.environ.pop('PYTHONHASHSEED', None)
        else:
            os.environ['PYTHONHASHSEED'] = inherited
    (n1, a1, e1, b1), (n2, a2, e2, b2) = seen
    assert n1 == n2 == 'snapshot_localize' and a1 == a2 == [plan.U]
    assert e1 == dict(CFG.serve.SNAPVIEW_LOC_ENV, PYTHONHASHSEED='0') and b1['PYTHONHASHSEED'] == '0', (e1, b1.get('PYTHONHASHSEED'))
    assert e2 == dict(CFG.serve.SNAPVIEW_LOC_ENV) and b2['PYTHONHASHSEED'] == '7', (e2, b2.get('PYTHONHASHSEED'))   # as production
    assert all(b1[k] == v for k, v in CFG.serve.SNAPVIEW_LOC_ENV.items())


def test_copy_fallback():
    """G-V5 in the child: batch runs on another file system (/dev/shm) - a production view (copy_fallback off) refuses at the
    first panel; a scratch / parity view copies every panel (the parity guard refuses a hard link of a file outside the parity
    roots, core/guard.py) and equals the hard-linked view byte for byte, its copies counted."""
    from exopipe.phase3.lib.catalog import deep_omission as DO
    W = shared_world()
    shm = '/dev/shm/exopipe3_tests_snapshot_view_shm'
    shutil.rmtree(shm, ignore_errors=True)
    try:
        runs = [shutil.copytree(r, os.path.join(shm, os.path.basename(r))) for r in W['runs']]
        if os.stat(shm).st_dev == os.stat(W['root']).st_dev:
            raise AssertionError('/dev/shm is on the scratch file system here: the test cannot cross file systems')
        od = os.path.join(W['root'], 'shm_omit')
        DO.run_snapshot(runs, od, rdir=W['rdir'], proto=W['proto'])
        rc, log = _build_job(W, od, os.path.join(W['root'], 'shm_prod', VIEW), runs=runs, copy_fallback=False)
        assert rc == 2 and 'G-V5 a panel could not be hard-linked' in log, log[-800:]
        assert not os.path.exists(os.path.join(W['root'], 'shm_prod', VIEW))
        out = os.path.join(W['root'], 'shm_scratch', VIEW)
        rc, log = _build_job(W, od, out, runs=runs, copy_fallback=True)
        assert rc == 0 and 'COMBINED-VIEW-OK' in log, log[-1500:]
        st = json.load(open(out + '.result.json'))
        n = json.load(open(os.path.join(out, 'combined_view_summary.json')))['panels_linked']
        assert st['panels_copied'] == n > 0, (st['panels_copied'], n)
        ta, tb = tree(W['port']), tree(out)
        skip = {'combined_view_summary.json', 'master_catalog.csv', 'card2_filter.csv', 'coadd_root.txt'}   # filtercat's
        assert set(ta) - skip == set(tb) - {'combined_view_summary.json'}, sorted(set(ta) ^ set(tb))[:5]
        # PRODUCTION ORDER: the record's rows follow the per-tier catalogues' directory listing (glob, unsorted), which differs
        # between /dev/shm and the scratch file system for the same names - the record compares as a multiset here
        same_rows = lambda a_, b_: sorted(open(a_).read().splitlines()) == sorted(open(b_).read().splitlines())
        bad = [rel for rel in tb if rel not in skip and not filecmp.cmp(ta[rel], tb[rel], shallow=False)
               and not (rel == 'deep_omission_dropped.csv' and same_rows(ta[rel], tb[rel]))]
        assert not bad, bad[:5]
    finally:
        shutil.rmtree(shm, ignore_errors=True)


def test_content():
    """The view's content tests refuse what they must (mutated copies of the parity view)."""
    from exopipe.phase3.core import validate
    W = shared_world()
    kept = os.path.join(W['out'], 'snapshot', 'deep_omission', VIEW, 'deep_omission_kept.csv')
    ok = {t['name']: t for t in validate.run('snapshot_view', W['port'], kept=kept, runs=W['runs'])}
    assert all(t['passed'] for t in ok.values()), [t for t in ok.values() if not t['passed']]

    def mutated(name, fix, test, want):
        d = os.path.join(W['root'], 'cmut', name, VIEW)
        shutil.rmtree(os.path.dirname(d), ignore_errors=True)
        shutil.copytree(W['port'], d, copy_function=shutil.copy2)
        fix(d)
        res = {t['name']: t for t in validate.run('snapshot_view', d, kept=kept, runs=W['runs'])}
        t = res[f'snapshot_view.{test}']
        assert not t['passed'] and want in str(t['detail']), (name, t)

    def lost_name(d):
        p = os.path.join(d, 'queue_I_10s.txt')
        lines = open(p).read().splitlines()
        i = next(i for i, ln in enumerate(lines) if not ln.startswith('#'))
        open(p, 'w').write('\n'.join(lines[:i] + lines[i + 1:]) + '\n')
    mutated('queue', lost_name, 'queues', 'identities of the view not queued')

    def omitted_on_page(d):
        p = next(p for p in sorted(glob.glob(os.path.join(d, 'candidates', 'I', '*_46MHz', 'detections.csv'))))
        rr = _rows(p)
        _wcsv(p, list(rr[0]), rr + [dict(rr[0], date=NIGHTS[0], cand_id='05h_%s_46MHz_I_10s_x' % NIGHTS[0], lst='05h')])
    mutated('omitted', omitted_on_page, 'pages', 'on an OMIT key')

    def two_tiers(d):
        p = next(p for p in sorted(glob.glob(os.path.join(d, 'candidates', '[IV]', '*_1min', 'detections.csv'))))
        open(os.path.join(os.path.dirname(p), 'rank0001_05h_2025-01-01_46MHz20250101050010.png'), 'w').write('x')
    mutated('tiers', two_tiers, 'pages', 'of another tier')

    def foreign(d):
        p = sorted(glob.glob(os.path.join(d, 'candidates', '_identity', '*', 'detections.csv')))[0]
        rr = _rows(p)
        _wcsv(p, list(rr[0]), rr + [dict(rr[0], cand_id=rr[0]['cand_id'] + 'x')])
    mutated('foreign', foreign, 'identities', 'not kept by the kept product')

    def record_off(d):
        p = os.path.join(d, 'deep_omission_dropped.csv')
        rr = _rows(p)
        rr.remove(next(r for r in rr if r['kind'] == 'detection'))
        _wcsv(p, list(rr[0]), rr)
    mutated('record', record_off, 'record', "omitted detections are not the omit stage's")

    # F1: a panel of an omitted instant served on a page (the batch's own file, as M1 links it): V3 by the panel's own key,
    # V7 by the page's member batch sources' listing less the kept product's panel drops
    omitted_panel = f'rank0001_05h_{NIGHTS[0]}_46MHz{NIGHTS[0].replace("-", "")}050120.png'
    src_panel = os.path.join(W['runs'][0], 'candidates', 'I', 'J0530+3500_46MHz', omitted_panel)
    assert os.path.isfile(src_panel) and os.path.isdir(os.path.join(W['port'], 'candidates', 'I', 'J0530p3500_46MHz'))
    assert not os.path.exists(os.path.join(W['port'], 'candidates', 'I', 'J0530p3500_46MHz', omitted_panel))

    def panel_served(d):
        shutil.copy2(src_panel, os.path.join(d, 'candidates', 'I', 'J0530p3500_46MHz', omitted_panel))
    mutated('panel_om', panel_served, 'pages', 'panels of an omitted instant are served')
    mutated('panel_om7', panel_served, 'panels', 'files no member batch source keeps')

    def panel_lost(d):                  # a kept panel missing from its page
        pd = os.path.join(d, 'candidates', 'I', 'J0530p3500_46MHz')
        os.remove(os.path.join(pd, sorted(fn for fn in os.listdir(pd) if fn.startswith('rank') and fn.endswith('.png'))[0]))
    mutated('panel_lost', panel_lost, 'panels', "member batch sources' kept panels missing")

    def panel_foreign(d):               # a kept panel of another source's page linked onto this one (F6's case, read back)
        a = os.path.join(d, 'candidates', 'I', 'J0545p3700_46MHz')
        b = os.path.join(d, 'candidates', 'I', 'J0530p3500_46MHz')
        fn = sorted(x for x in os.listdir(a) if x.startswith('rank'))[0]
        shutil.copy2(os.path.join(a, fn), os.path.join(b, fn))
    mutated('panel_foreign', panel_foreign, 'panels', 'files no member batch source keeps')
    # the card2 union and the per-detection filter catalogue
    c2 = os.path.join(W['port'], 'card2_filter.csv')
    assert validate.run('snapshot_card2_union', c2, runs=W['runs'])[0]['passed']
    d = os.path.join(W['root'], 'cmut', 'card2')
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(os.path.join(d, VIEW))
    txt = open(c2).read().splitlines()
    open(os.path.join(d, VIEW, 'card2_filter.csv'), 'w').write('\n'.join(txt[:-1]) + '\n')
    assert not validate.run('snapshot_card2_union', os.path.join(d, VIEW, 'card2_filter.csv'), runs=W['runs'])[0]['passed']
    mc = os.path.join(W['root'], 'cmut', 'record', VIEW, 'master_catalog.csv')
    rr = _rows(mc)
    _wcsv(mc, list(rr[0]), rr[1:])
    t = validate.run('filter_catalog', mc, per_detection=True)[0]
    assert not t['passed'] and 'without their row' in t['detail'], t


def test_ready():
    """The ready stage on the built view: G17 on the view, the final tests (stub localizations), VIEW-READY-FOR-HIDES."""
    import types
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import record
    W = shared_world()
    SV = _SV()
    for T in ('10s', '1min'):
        ids = {r['source_id'] for p in glob.glob(os.path.join(W['port'], 'source_catalogs', '*_sources_survivors.csv'))
               for r in _rows(p)}
        _wcsv(os.path.join(W['port'], f'source_catalogs_loc_{T}', 'localizations.csv'), ['identity_id', 'ra', 'dec', 'n_det'],
              [dict(identity_id=i, ra='1', dec='2', n_det='1') for i in sorted(ids)[:3]])
    rr = record.Run('serve', ['test'], W['out'], CFG)
    plan = SV.ViewPlan(types.SimpleNamespace(run_label=VIEW, view_runs=','.join(W['runs']), view_omit=None), W['out'])
    try:
        CFG.authorize(out=W['out'])
        u = SV.ready_stage(rr, plan)
    finally:
        CFG.reset()
        CFG.authorize()
    assert u.status == 'done', (u.status, u.reason, [t for t in u.tests if not t['passed']][:2])
    assert any(t['name'] == 'snapshot_view.final' and t['passed'] for t in u.tests)
    assert any(n.startswith('VIEW-READY-FOR-HIDES') for n in u.notes), u.notes


TESTS = [test_production_pin, test_drift, test_py36, test_parity, test_repoint, test_kept_refusals, test_gates, test_filtercat_procs,
         test_incomplete_inputs, test_stale_omit, test_copy_fallback, test_content, test_m1_mutation, test_figure_seed, test_ready]


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
        if os.environ.get('EXOPIPE3_TEST_KEEP') != '1':
            shutil.rmtree(SCRATCH, ignore_errors=True)
    n = len([t for t in TESTS if not want or t.__name__ in want])
    print((f'{len(failed)} failed: {failed}; ' if failed else '') + f'{n - len(failed)} of {n} passed ({time.time() - t0:.1f} s)')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
