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


def fleet(proto, rng):
    """Noise tables per (cell, band, Stokes, tier) and wideband member lists (05h 46 MHz I 10-s night 1 noisy)."""
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
            for S in ('I', 'V'):
                mem = [('Red', '18MHz'), ('Red', '23MHz'), ('Green', '55MHz'), ('Green', '46MHz')]
                _wcsv(os.path.join(cell, f'wideband_members_{S}.csv'), ['group', 'ts', 'subband'],
                      [dict(group=g, ts='x', subband=sb) for g, sb in mem])
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


def batch_run(root, name, idents, extra_files=(), best_cand_detection=()):
    """A served batch run like production's: the general catalogue per band class (a row per identity and class, its own
    review_dir with that class's detections, anchored), per-tier views (survivors: one row per (identity, band) with
    production's best_cand_id '<identity>_<cell>_<band>' - a detection cand_id for the identities in best_cand_detection;
    members '<identity>_<cell>_<band>'), page directories per (band, tier) with panels (.png + .page.json per detection,
    astrometry.png, ``extra_files``), the queues with their header lines, card2_filter.csv. idents: [(J, (ra, dec, err_arcsec),
    [detections])]."""
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
        by = collections.defaultdict(list)
        for d in dets:
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
            bc = best['cand_id'] if J in best_cand_detection else f'{J}_{best["lst"]}_{best["date"]}_{sb}'
            per[(S, T, sb)].append(dict(rank='rank0001', source_id=f'{J}_{sb}', identity_id=J, review_dir=pd, ra=best['ra'],
                                        dec=best['dec'], best_snr=best['snr'], n_det=len(dd), n_det_identity=len(dets),
                                        best_cand_id=bc, peak_jy_best='12.5', stokes=S, tier=T, subband=sb, ts=best['ts']))
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
    # served_tiers.csv (the chain's per-tier serving table): an identity served in a tier holds a per-tier source there, as in
    # production (36 of 36 batches of Run_SNAPCOMB_20261102); a no-route row and a non-qualifying row as production writes them
    tiers_of = collections.defaultdict(set)
    for (S, T, sb), rr in per.items():
        for x in rr:
            tiers_of[x['identity_id']].add(T)
    st = []
    for J, _a, dets in idents:
        for T in ('10s', '1min'):
            st.append(dict(identity_jname=J, chain_outcome='served', tier=T, qualifies='1' if T in tiers_of[J] else '0'))
    st.append(dict(identity_jname='J9999+9999', chain_outcome='no-route', tier='10s', qualifies='0'))
    _wcsv(os.path.join(run, 'served_tiers.csv'), ['identity_jname', 'chain_outcome', 'tier', 'qualifies'], st)
    return run


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


def world(name='world', seed=31):
    """(root, rdir, proto, runs, batch_dir) of the synthetic world (module docstring)."""
    root = _fresh(name)
    rng = random.Random(seed)
    rdir = refusal_record(os.path.join(root, 'refusals'))
    proto = fleet(os.path.join(root, 'fleet'), rng)
    review = os.path.join(root, 'review')
    N = NIGHTS
    a = [  # batch A: 05h nights 1-2
        ('J0520+3000', (80.0, 30.0, 60.0), [det('05h', N[0], '46MHz', 'I', '10s', 3, 30.0, 80.0, 30.0),
                                            det('05h', N[1], '18MHz', 'V', '1min', 4, 12.0, 80.0, 30.0)]),
        ('J0600+2500', (90.0, 25.0, 60.0), [det('05h', N[1], '23MHz', 'I', '10s', 5, 15.0, 90.0, 25.0)]),
        ('J0510+2000', (77.5, 20.0, 60.0), [det('05h', N[0], '46MHz', 'I', '10s', 6, 11.0, 77.5, 20.0),     # all omitted:
                                            det('05h', N[0], '46MHz', 'I', '10s', 7, 9.0, 77.5, 20.0)]),    # identity dropped
        ('J0530+3500', (82.5, 35.0, 60.0), [det('05h', N[0], '46MHz', 'I', '10s', 8, 25.0, 82.5, 35.0),     # partial, best
                                            det('05h', N[1], '46MHz', 'I', '10s', 8, 10.0, 82.5, 35.0)]),   # omitted: RP
        ('J0545+3700', (86.0, 37.0, 60.0), [det('05h', N[0], '46MHz', 'I', '10s', 9, 28.0, 86.0, 37.0),     # partial, best
                                            det('05h', N[1], '46MHz', 'I', '10s', 9, 11.0, 86.0, 37.0)]),   # omitted, re-pointed
        ('J0550+3800', (87.5, 38.0, 60.0), [det('05h', N[0], '46MHz', 'I', '10s', 10, 13.0, 87.5, 38.0),    # source dropped,
                                            det('05h', N[1], '18MHz', 'V', '10s', 10, 9.0, 87.5, 38.0),     # identity kept
                                            det('05h', N[1], 'Green', 'I', '10s', 11, 8.5, 87.5, 38.0)]),
    ]
    b = [  # batch B: 05h nights 3-4, 06h night 3
        ('J0520+3001', (80.0, 30.0, 60.0), [det('05h', N[2], '46MHz', 'I', '10s', 3, 20.0, 80.0, 30.0),     # merges with A's
                                            det('05h', N[3], 'Red', 'I', '1min', 3, 9.5, 80.0, 30.0),       # (and its 18 MHz V
                                            det('05h', N[2], '18MHz', 'V', '1min', 4, 11.0, 80.0, 30.0)]),  # 1-min page: a clash)
        ('J0610+3100', (92.5, 31.0, 60.0), [det('06h', N[2], '55MHz', 'V', '10s', 5, 14.0, 92.5, 31.0)]),
        ('J0615+3300', (93.7, 33.0, 60.0), [det('06h', N[2], '46MHz', 'I', '1min', 6, 16.0, 93.7, 33.0),
                                            det('05h', N[3], '27MHz', 'I', '10s', 6, 8.8, 93.7, 33.0)]),
    ]
    c = [  # batch C: 06h nights 5-6
        ('J0600+2500', (95.0, 35.0, 60.0), [det('06h', N[4], '18MHz', 'I', '10s', 5, 18.0, 95.0, 35.0)]),  # a name clash
        ('J0615+3301', (93.7, 33.0, 60.0), [det('06h', N[5], '46MHz', 'I', '1min', 6, 22.0, 93.7, 33.0)]),  # merges with B's
        ('J0620+2900', (95.0, 29.0, 60.0), [det('06h', N[5], 'Green', 'V', '10s', 7, 10.5, 95.0, 29.0),
                                            det('06h', N[4], '23MHz', 'V', '1min', 7, 9.0, 95.0, 29.0)]),
    ]
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
            dets.append(det(lst, nt, sb, rng.choice('IV'), rng.choice(('10s', '1min')), 20 + 3 * i + k, 7.5 + 10 * rng.random(),
                            ra, dec))
        rnd.append((bi, (f'J{4 + i // 10:02d}{i % 60:02d}+{20 + i:02d}00', (ra, dec, 60.0), dets)))
    a += [x for bi, x in rnd if bi == 0]
    b += [x for bi, x in rnd if bi == 1]
    c += [x for bi, x in rnd if bi == 2]
    runs = [batch_run(review, 'Run_SNAPT2A_20261103', a, extra_files=('notes.txt',), best_cand_detection=('J0545+3700',)),
            batch_run(review, 'Run_SNAPT2B_20261103', b, extra_files=('notes.txt',)),
            batch_run(review, 'Run_SNAPT2C_20261103', c)]
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import production_pin as PIN        # noqa: E402
PINNED = ('snapshot_combined_view.py', 'snapshot_deep_omission.py', 'sublst_deep_omission.py',
          'snapshot_batch/comb_build_20261102.sh')
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
assert os.path.dirname(os.path.abspath(SDO.__file__)) == os.path.abspath(pin), SDO.__file__
SDO.decide.__defaults__ = (rdir, proto)
sys.argv = [os.path.join(pin, 'snapshot_combined_view.py'), out] + runs
runpy.run_path(os.path.join(pin, 'snapshot_combined_view.py'), run_name='__main__')
'''


def prod_driver_heredoc():
    """The driver's step-2 heredoc (comb_build_20261102.sh at the pin, between <<'PYC' and PYC), verbatim."""
    src = open(os.path.join(pinned_ops(), 'snapshot_batch', 'comb_build_20261102.sh')).read()
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
    """The port's texts, patterns and layout are production's at the pin (read as text)."""
    SV = _SV()
    from exopipe.phase3.lib.catalog import deep_omission as DO
    src = open(os.path.join(pinned_ops(), 'snapshot_combined_view.py')).read()
    drv = open(os.path.join(pinned_ops(), 'snapshot_batch', 'comb_build_20261102.sh')).read()
    flat = src.replace('"\n                     "', '').replace('"\n                    "', '').replace('"\n                       "', '')
    for txt in ('# queue_name: snapshot Stokes %s, %s tier - COMBINED VIEW of %d served batches\\n',
                '# run: %s (combined view; batches %s)\\n',
                '%d detections omitted by the deep-omission rule (Gregg 2026-10-24)\\n',
                'served detections %d, kept %d; decisions %d (%d OMIT keys); %s\\n', '"T1 T2 T3 T4 T5 T6 PASS"'):
        assert txt in src, txt
    assert SV.QUEUE_NAME.replace('\n', '\\n') in src and SV.QUEUE_RUN.replace('\n', '\\n') in src
    assert SV.SUMMARY_LINE1.replace('\n', '\\n') in src and SV.SUMMARY_LINE2.replace('\n', '\\n') in src
    assert '"%s"' % SV.TESTS_PASSED in src
    for line in (SV.VIEW_LIMITS, SV.DEEP_OMISSION_LINE):         # the two long header lines, joined from their source literals
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
    assert 'row = dict(row, best_cand_id=top["cand_id"], best_snr=top["snr"], ts=top["ts"], subband=top["subband"], ' \
           'peak_jy_best="",' in src and SV.REPOINT_FIELDS == ('best_cand_id', 'best_snr', 'ts', 'subband', 'peak_jy_best',
                                                             'deep_omission')
    assert 'deep_omission="best detection omitted; re-pointed (survivor gates not re-run)")' in src
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
    for k in ('detection', 'identity dropped', 'source dropped', 'member row', 'panel', 'source partial'):
        assert kinds.get(k), (k, kinds)
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
              'G-V3 omission_served.served_counts_add_up'):
        assert t in names_t, (t, names_t)
    assert 'omission_served.served_counts_add_up' in {t['name'] for t in recs[f'snapshot_{VIEW}_omit']['tests']}
    assert recs[f'snapshot_{VIEW}_build']['meta']['build']['stale_best'] >= 1      # PRODUCTION BUG RP counted (J0530+3500)
    # the omit product is the deep-omission stage's (deep_omission.csv = the view's, byte for byte)
    od = os.path.join(W['out'], 'snapshot', 'deep_omission', VIEW)
    assert filecmp.cmp(os.path.join(od, 'deep_omission.csv'), os.path.join(W['prod'], 'deep_omission.csv'), shallow=False)
    assert json.load(open(os.path.join(od, 'deep_omission_summary.json')))['digests']['served'].startswith(f'{summ["deep_omission"]["served"]}:')
    assert DO.omitted_keys(_rows(os.path.join(od, 'deep_omission.csv')))


def test_repoint_pinned():
    """PRODUCTION BUG RP, pinned as it is today (production and the port): a per-tier source whose best detection is omitted is
    re-pointed only when its best_cand_id is a detection cand_id (J0545+3700, a synthetic row); with production's best_cand_id
    '<identity>_<cell>_<band>' (J0530+3500) it keeps the omitted detection's best_snr and ts. When production's owner fixes it
    and the port follows, these asserts change on purpose."""
    W = prod_world()
    for view in (W['prod'], W['port']):
        rows = {r['source_id']: r for r in _rows(os.path.join(view, 'source_catalogs_I_10s', '46MHz_sources_survivors.csv'))}
        rp = rows['J0545+3700_46MHz']
        assert rp['best_snr'] == '11.00' and rp['deep_omission'].startswith('best detection omitted; re-pointed'), rp
        st = rows['J0530+3500_46MHz']
        assert st['best_snr'] == '25.00' and not st.get('deep_omission'), st          # the omitted night-1 detection's
        dets = _rows(os.path.join(view, st['review_dir'], 'detections.csv'))
        assert [d['snr'] for d in dets] == ['10.00'], dets                            # its page holds only the kept one
    rec = _rows(os.path.join(W['port'], 'deep_omission_dropped.csv'))
    part = [x for x in rec if x['kind'] == 'source partial']
    assert [x['source'] for x in part] == ['J0545+3700_46MHz'], part


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
    job = dict(out=out, runs=runs or W['runs'], kept_dir=omit_dir, tiers=list(S.SNAPVIEW_TIERS), stokes=list(S.SNAPVIEW_STOKES),
               suffix=dict(S.SNAPVIEW_PAGE_SUFFIX), clv15='/lustre/gh/main/clv15', ops=OPS,
               deep_omission_py=os.path.abspath(deep_omission.__file__), result=out + '.result.json',
               copy_fallback=copy_fallback)
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
        return {f: (os.stat(os.path.join(d, f)).st_mtime_ns, open(os.path.join(d, f), 'rb').read()) for f in sorted(os.listdir(d))}
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
        bad = [rel for rel in tb if rel not in skip and not filecmp.cmp(ta[rel], tb[rel], shallow=False)]
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


TESTS = [test_production_pin, test_drift, test_py36, test_parity, test_repoint_pinned, test_kept_refusals, test_gates,
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
