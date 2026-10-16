"""B-02 steps 2 and 4: lib/catalog/conform_check.py - the publish conformance as a check, and the pulsar stage - on synthetic
runs (plan/wave2/B02_DESIGN.md "Steps 2 and 4").

A run of one cell (05h 23MHz - a Red band: a 5-degree Jupiter track - Stokes V, six nights): an attempt searched under 'once-in-search' with its rules record, its
archived search frames (small FITS on a SIN grid about the 05h zenith, each with its own time block and beam), frame_noise.csv,
table_pulsar_decisions.csv, a gate ledger in which every removal rule of the search acts (finals, 'elevation', 'jupiter-track',
'table', 'row-line' rows, chosen on the grid so that each rule's own function decides them), and the merged catalog of its
finals. The bright-source table and the parents are synthetic (a table that cuts the pixels the test names); the elevation
floor, the Jupiter track, the daytime gate, the noise rules, the row line and the backing are the real functions.
  (1) the clean run passes, and writes production's outputs (as-run = the merge's output, nothing removed);
  (2) each rule faulted in each direction is refused with its finding: a final moved below the floor / an 'elevation' row
      above it; a final put on the track / a 'jupiter-track' row off it; the table cutting a final / not cutting a 'table'
      row; a row line among the finals / a 'row-line' row alone; a searched frame in daytime / a daytime refusal at night;
      a searched V frame above the noise rule / a V noise refusal below it; a catalog row without its final / a final without
      its row; a duplicate; a searched night excluded now; a v110 marginal removal; an attempt searched under 'production',
      without a record, or at another rules version;
  (3) the pulsar stage: with no pulsar decision the catalog is unchanged; a consistent group's cut detection is restored
      (a PR_ row from its ledger row) and an inconsistent group's exempted detection removed, both recorded, and the check
      accepts them (and refuses a restored row the table no longer cuts).
Run: python tests/test_conform_check.py
"""
import copy
import csv
import json
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')

import numpy as np                                                  # noqa: E402
from astropy.io import fits                                         # noqa: E402
from astropy.time import Time, TimeDelta                            # noqa: E402
from astropy.wcs import WCS, FITSFixedWarning                       # noqa: E402
import warnings                                                     # noqa: E402
warnings.filterwarnings('ignore', category=FITSFixedWarning)

from exopipe.phase3.core import env as ENV                          # noqa: E402
ENV.legacy()
from exopipe.phase3.config import CFG                               # noqa: E402
from exopipe.phase3.lib.catalog import conform_check as CC          # noqa: E402
from exopipe.phase3.lib.search import removal_rules as RR           # noqa: E402
from exopipe.phase3.lib.search import tests as ST                   # noqa: E402

T = '/tmp/exopipe3_tests_conform_check'
LST, SC, LABEL, CELL = '05h', '23MHz', 'CHKTEST', '05h_23MHz'
NIGHTS = ['2024-12-18', '2024-12-19', '2024-12-20', '2024-12-21', '2024-12-22', '2024-12-23']
N0 = 64
FAILS, N = [], [0]


def check(name, ok, detail=''):
    N[0] += 1
    print(('PASS  ' if ok else 'FAIL  ') + name + ('' if ok else f'\n      {str(detail)[:900]}'), flush=True)
    if not ok:
        FAILS.append(name)


# ---------------------------------------------------------------------------------------------------------------------
# the synthetic run
# ---------------------------------------------------------------------------------------------------------------------

def frame_header(night, k, daytime=False):
    h = fits.Header()
    h['NAXIS'] = 2
    h['NAXIS1'] = h['NAXIS2'] = N0
    h['CTYPE1'], h['CTYPE2'] = 'RA---SIN', 'DEC--SIN'
    h['CRVAL1'], h['CRVAL2'] = 75.0, 37.2
    h['CDELT1'], h['CDELT2'] = -2.0, 2.0
    h['CRPIX1'] = h['CRPIX2'] = N0 / 2 + 0.5
    t0 = (Time(f'{night}T20:00:00') if daytime else Time('2024-12-18T07:03:36') + TimeDelta(86400 * k - 236 * k, format='sec'))
    h['DATE-OBS'] = t0.isot
    h['DATE-END'] = (t0 + TimeDelta(3591, format='sec')).isot
    h['BMAJ'], h['BMIN'], h['BPA'] = 0.2, 0.18, 0.0
    return h


class Run:
    """A synthetic once-in-search attempt and its merged run directory (see the module docstring)."""

    def __init__(self, root, stokes='V'):
        self.root, self.st = root, stokes
        shutil.rmtree(root, ignore_errors=True)
        self.A = os.path.join(root, 'attempts', f'{CELL}_{stokes}_20261016T000000')
        self.rd = os.path.join(self.A, 'transients', f'Run_{LABEL}')
        self.U = os.path.join(root, 'run')
        self.base = os.path.join(root, 'coadd', CELL)
        for d in (self.rd, self.U, os.path.join(self.base, 'coadd_final'), os.path.join(self.rd, 'refusals')):
            os.makedirs(d, exist_ok=True)
        self.headers = {d: frame_header(d, k) for k, d in enumerate(NIGHTS)}
        self.W = WCS(self.headers[NIGHTS[0]]).celestial
        self.cutset, self.notes = set(), {}
        self.ledger, self.fnoise, self.tpd, self.day_ref, self.sanity_v, self.sanity_i = [], [], [], [], [], []
        self.qa3 = {d: 500 + k for k, d in enumerate(NIGHTS)}
        self.rules = RR.rules_record('once-in-search')

    def sky(self, x, y):
        s = self.W.pixel_to_world(x, y)
        return round(float(s.ra.deg), 5), round(float(s.dec.deg), 5)

    def el(self, night, x, y):
        return RR.frame_elevation(LST, *self.sky(x, y), self.headers[night], epoch='frame-mid')[0]

    def pixels(self, night, lo, hi, n, avoid=()):
        """n pixels of the grid whose frame-mid elevation on ``night`` lies in [lo, hi), away from the track and each other."""
        out = []
        for y in range(4, N0 - 4, 3):
            for x in range(4, N0 - 4, 3):
                if (x, y) in avoid or any(abs(x - a) < 3 and abs(y - b) < 3 for a, b in out):
                    continue
                if not np.isfinite(self.sky(x, y)[0]):
                    continue
                e = self.el(night, x, y)
                if lo <= e < hi and not RR.jupiter_track_hit(*self.sky(x, y), SC, self.track):
                    out.append((x, y))
                    if len(out) == n:
                        return out
        raise AssertionError(f'no {n} pixels at {lo}-{hi} deg')

    def led(self, night, x, y, veto='', final=0, **kw):
        r = dict.fromkeys(ST.GLED_COLS, '')
        ra, dec = self.sky(x, y)
        r.update(lst=LST, scope=SC, stokes=self.st, date=night, ra=ra + 0.001, dec=dec - 0.001, pix_x=x, pix_y=y, snr=7.5,
                 peak_jy=0.5, morph_pass=1, veto_reason=veto, final=final, pass_mstruct=1, fp_prediction='inconclusive')
        r.update(kw)
        self.ledger.append(r)
        return r

    def weights(self):
        """The cell's weights.csv: every night in the master, its qa3 rms (the Stokes I noise rule's value)."""
        with open(os.path.join(self.base, 'coadd_final', 'weights.csv'), 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['date', 'weight', 'qa3_scale', 'qa3_tilt', 'qa3_nsrc', 'qa3_guard', 'qa3_rms_mjy', 'sci_dir', 'in_master'])
            for d in NIGHTS:
                w.writerow([d, 1.0, 1.0, 0.0, 500, 'pass', self.qa3[d], '', 1])

    def build(self):
        """The clean run: per night two finals high in the sky, one 'elevation' row low, one 'table' row the table cuts;
        night 1 also a 'jupiter-track' row on the track and a row line of five ('row-line'); night 2 a 'table'-naming
        pulsar decision kept (a final) and one cut."""
        self.weights()
        self.track = RR.jupiter_track(LST, os.path.join(self.base, 'coadd_final', 'weights.csv'))
        for k, d in enumerate(NIGHTS):
            hi = self.pixels(d, 40, 90, 3)
            for x, y in hi[:2]:
                self.led(d, x, y, final=1)
            x, y = hi[2]
            self.cutset.add(self.sky(x, y))
            self.led(d, x, y, veto='table')
            x, y = self.pixels(d, 0, 24, 1)[0]
            self.led(d, x, y, veto='elevation')
            self.fnoise.append(dict(lst=LST, scope=SC, stokes=self.st, date=d, noise_jy=repr(0.010 + 0.0005 * k), admitted=1,
                                    population=1, tblnoise_jy='', vsanmed_jy='', vsanity_x='', attempt='A1'))
        pop = sorted(float(r['noise_jy']) for r in self.fnoise)
        vmed = pop[len(pop) // 2]
        for r in self.fnoise:                # the V noise rule's median and factor on V rows only
            r.update(tblnoise_jy=repr(RR.median_noise(pop)), vsanmed_jy=repr(vmed) if self.st == 'V' else '',
                     vsanity_x=repr(3.0) if self.st == 'V' else '')
        jra, jdec, jd = next(p for p in self.track if p[2] == NIGHTS[0])   # a 'jupiter-track' row on night 1's track point
        jx, jy = (int(round(float(v))) for v in self.W.world_to_pixel_values(jra, jdec))
        self.jpix = (jx, jy)
        self.led(NIGHTS[0], jx, jy, veto=f'jupiter-track:{jd}')
        used = {(int(z['pix_x']), int(z['pix_y'])) for z in self.ledger}
        line = next(L for y in range(20, 44) for L in [[(x, y) for x in range(14, 50, 8)]]
                    if all(self.el(NIGHTS[0], x, y) >= 30 and (x, y) not in used and (x, y + 1) not in used
                           and (x, y - 1) not in used and not RR.jupiter_track_hit(*self.sky(x, y), SC, self.track) for x, y in L))
        for x, y in line:                   # five on one pixel row, above the floor, off the track and the other rows
            self.led(NIGHTS[0], x, y, veto='row-line')
        self.line = line
        return self

    def write(self):
        """The attempt's records, its frames and the merged run directory (what p3s3_merge.py links and merges)."""
        self.weights()
        with open(os.path.join(self.A, RR.RULES_RECORD), 'w') as fh:
            json.dump(self.rules, fh)
        sub = 'vresid' if self.st == 'V' else 'iresid'
        vdir = os.path.join(self.A, f'Run_{LABEL}', sub, CELL)
        os.makedirs(vdir, exist_ok=True)
        for d, h in self.headers.items():
            fits.PrimaryHDU(np.zeros((N0, N0), np.float32), h).writeto(
                os.path.join(vdir, f'{d}_{self.st}_redo_resid_{SC}.fits'), overwrite=True)

        def wcsv(p, cols, rows):
            with open(p, 'w', newline='') as fh:
                w = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
                w.writeheader()
                w.writerows(rows)
        wcsv(os.path.join(self.rd, 'v_deep_ledger.csv'), ST.GLED_COLS, self.ledger)
        wcsv(os.path.join(self.U, 'v_deep_ledger.csv'), ST.GLED_COLS, self.ledger)
        wcsv(os.path.join(self.rd, RR.FRAME_NOISE), RR.FRAME_NOISE_COLS, self.fnoise)
        wcsv(os.path.join(self.rd, RR.TABLE_PULSAR), RR.TABLE_PULSAR_COLS, self.tpd)
        ref = os.path.join(self.rd, 'refusals')
        for name, rows in ((f'{LST}_{SC}_daytime_excluded.csv', self.day_ref), (f'{LST}_{SC}_sanity_V_excluded.csv', self.sanity_v),
                           (f'{LST}_{SC}_sanity_I_excluded.csv', self.sanity_i)):
            p = os.path.join(ref, name)
            if os.path.exists(p):
                os.remove(p)
            if rows:
                with open(p, 'w', newline='') as fh:
                    csv.writer(fh, lineterminator='\n').writerows(rows)
        cat = []
        for j, r in enumerate(x for x in self.ledger if x['final'] in (1, '1')):
            ra, dec = self.sky(int(r['pix_x']), int(r['pix_y']))
            c = dict.fromkeys(ST.CAT_FIELDS, '')
            c.update(cand_id=f'{self.st}D_{LST}_{r["date"]}_{SC}_{j:04d}', lst=LST, date=r['date'], band=SC, stokes=self.st,
                     cadence='deep', ra=ra, dec=dec, peak_jy=r['peak_jy'], snr=r['snr'],
                     fp_prediction='inconclusive' if self.st == 'V' else '')
            cat.append(c)
        self.cat = cat
        wcsv(os.path.join(self.U, 'master_catalog.csv'), ST.CAT_FIELDS, cat)
        self.merged = os.path.join(self.root, 'merged_master_catalog.csv')
        shutil.copy2(os.path.join(self.U, 'master_catalog.csv'), self.merged)
        os.makedirs(os.path.join(self.U, sub), exist_ok=True)
        link = os.path.join(self.U, sub, CELL)
        if not os.path.islink(link):
            os.symlink(vdir, link)
        self.man = os.path.join(self.root, 'harvest_manifest.csv')
        wcsv(self.man, ['cell', 'stokes', 'node', 'att', 'dir', 'harvested', 'label', 'patched', 'patched_from'],
             [dict(cell=CELL, stokes=self.st, node='x', att='20261016T000000', dir=self.A, harvested=1, label=LABEL)])
        for name in (CC.CHECK_REPORT, CC.PULSAR_RECORD, CC.PULSAR_REMOVED, 'frame_times.csv', 'frame_beam_ratio.csv',
                     'frame_beam_bmaj.csv'):
            p = os.path.join(self.U, name)
            if os.path.exists(p):
                os.remove(p)
        return self


class FakeG:
    def parents(self, l, b):
        return np.zeros((0, 3))

    def phase_centre(self, l, b):
        return (75.0, 37.2)

    def deep_target_beam(self, l, b, d, s):
        return 0.2


def deps(run):
    class PCT:
        @staticmethod
        def removals(st, b, parents, noise, beam, ra, dec, elev=None, phase_centre=None):
            assert noise is not None and noise > 0 and beam == 0.2 and elev is not None and len(elev) == len(ra)
            keys = [(round(float(a), 5), round(float(d), 5)) for a, d in zip(ra, dec)]
            cut = np.array([k in run.cutset for k in keys])
            return cut, [run.notes.get(k, 'table: synthetic cut' if c else '') for k, c in zip(keys, cut)]
    return lambda: dict(G=FakeG(), LOC=None, PCT=PCT, excluded=lambda l, b, d: d in run.excluded)


def run_check(run, label='clean'):
    """(passed, the findings {(direction, rule)} , the message) of conform_check.check on ``run`` (pulsar stage first)."""
    run.excluded = getattr(run, 'excluded', set())
    CC._deps = deps(run)
    CC.cell_base = lambda cell, roots_csv=None: run.base
    try:
        with open(os.path.join(run.U, CC.PULSAR_RECORD), 'w', newline='') as fh:
            csv.writer(fh).writerow(CC.PULSAR_RECORD_COLS)
        with open(os.path.join(run.U, CC.PULSAR_REMOVED), 'w', newline='') as fh:
            csv.writer(fh).writerow(ST.CAT_FIELDS + ['removal_reason'])
        res = CC.check(run.U, run.man, LABEL, merged=run.merged, out=lambda *a, **k: None)
        return True, set(), res
    except SystemExit as e:
        rep = os.path.join(run.U, CC.CHECK_REPORT)
        f = {(r['direction'], r['rule']) for r in CC._rows(rep)} if os.path.exists(rep) else set()
        return False, f, f'exit {e.code}'


# ---------------------------------------------------------------------------------------------------------------------
# the tests
# ---------------------------------------------------------------------------------------------------------------------

def fresh(name, stokes='V'):
    return Run(os.path.join(T, name), stokes).build()


def test_clean():
    r = fresh('clean').write()
    ok, f, res = run_check(r)
    check('the clean run passes the check (every rule re-made, none missed)', ok, f)
    if ok:
        asrun = os.path.join(r.U, 'master_catalog_asrun.csv')
        rows = CC._rows(os.path.join(r.U, 'master_catalog.csv'))
        rem = CC._rows(os.path.join(r.U, 'master_catalog_removed.csv'))
        check("passed: production's outputs - as-run = the merge's output byte for byte, every row conformant, none removed, the "
              "conform columns added", open(asrun, 'rb').read() == open(r.merged, 'rb').read() and len(rows) == len(r.cat)
              and not rem and 'conform_note' in rows[0] and all(x['fp_prediction'] for x in rows), (len(rows), len(rem)))
        from exopipe.phase3.lib.catalog import tests as CT
        cr = CT.conformance_record(os.path.join(r.U, 'master_catalog.csv'), merge_copy=r.merged)
        check('passed: the conformant_catalog content test passes on the check\'s output (the floor at the frame mid-time from '
              'frame_times.csv)', cr[0], cr)


def fault(name, edit, want, desc, also=(), stokes='V'):
    """A fresh run, ``edit`` applied before it is written; the check must refuse with exactly the finding ``want`` (and the
    ones ``also`` names, that the same edit implies)."""
    r = fresh(name, stokes)
    edit(r)
    r.write()
    ok, f, res = run_check(r)
    check(f'{desc} - refused with {want}' + (f' (and {sorted(also)})' if also else '') + ', nothing else',
          not ok and f == {want, *also}, (ok, sorted(f), res))


def first(r, pred):
    return next(x for x in r.ledger if pred(x))


def test_faults():
    d0 = NIGHTS[0]

    def final_low(r):                       # (a) a final below the floor
        x, y = r.pixels(NIGHTS[1], 0, 24, 1, avoid={(int(z['pix_x']), int(z['pix_y'])) for z in r.ledger})[0]
        r.led(NIGHTS[1], x, y, final=1)
    fault('a_floor', final_low, ('a', 'elevation'), '(a) a final below the elevation floor at its frame\'s mid-time')

    def elev_high(r):                       # (b) an 'elevation' row above the floor
        x, y = r.pixels(NIGHTS[1], 50, 90, 1, avoid={(int(z['pix_x']), int(z['pix_y'])) for z in r.ledger})[0]
        r.led(NIGHTS[1], x, y, veto='elevation')
    fault('b_floor', elev_high, ('b', 'elevation'), "(b) an 'elevation' row that the floor keeps today")

    def final_track(r):                     # (a) a final on the Jupiter track
        z = first(r, lambda x: x['veto_reason'].startswith('jupiter-track'))
        z['veto_reason'], z['final'] = '', 1
    fault('a_track', final_track, ('a', 'jupiter-track'), '(a) a final on the Jupiter track of a co-added night')

    def track_off(r):                       # (b) a 'jupiter-track' row off the track
        x, y = r.pixels(NIGHTS[2], 50, 90, 1, avoid={(int(z['pix_x']), int(z['pix_y'])) for z in r.ledger})[0]
        r.led(NIGHTS[2], x, y, veto=f'jupiter-track:{NIGHTS[2]}')
    fault('b_track', track_off, ('b', 'jupiter-track'), "(b) a 'jupiter-track' row off every co-added night's track")

    def table_cuts_final(r):                # (a) the table cuts a final
        z = first(r, lambda x: x['final'] == 1)
        r.cutset.add(r.sky(int(z['pix_x']), int(z['pix_y'])))
    fault('a_table', table_cuts_final, ('a', 'table'), '(a) a final the bright-source table cuts today (tables re-derived)')

    def table_keeps(r):                     # (b) a 'table' row the table keeps
        z = first(r, lambda x: x['veto_reason'] == 'table')
        r.cutset.discard(r.sky(int(z['pix_x']), int(z['pix_y'])))
    fault('b_table', table_keeps, ('b', 'table'), "(b) a 'table' row the bright-source table keeps today")

    def finals_line(r):                     # (a) five finals on one pixel row
        for z in [x for x in r.ledger if x['veto_reason'] == 'row-line']:
            z['veto_reason'], z['final'] = '', 1
    fault('a_rowline', finals_line, ('a', 'row-line'), '(a) five finals of one epoch on one pixel row +/-1')

    def lone_line(r):                       # (b) a 'row-line' row the row line does not make
        z = [x for x in r.ledger if x['veto_reason'] == 'row-line']
        for x in z[1:]:
            r.ledger.remove(x)
    fault('b_rowline', lone_line, ('b', 'row-line'), "(b) a 'row-line' row alone on its row")

    def daytime_frame(r):                   # (a) a searched frame in daytime
        r.headers[NIGHTS[3]] = frame_header(NIGHTS[3], 3, daytime=True)
    fault('a_daytime', daytime_frame, ('a', 'daytime'), '(a) a searched frame the daytime gate refuses today',
          also={('a', 'elevation')})          # the frame's new times move its finals below the floor too

    def day_at_night(r):                    # (b) a daytime refusal of a night frame
        r.day_ref.append([LST, SC, NIGHTS[4], 'V', 'daytime: Sun 1.0 to 6.3 deg over the frame (solar gate 2026-09-26/27)', 'A1'])
    fault('b_daytime', day_at_night, ('b', 'daytime'), '(b) a frame refused as daytime that the daytime gate admits today')

    def v_noisy(r):                         # (a) a searched V frame above 3 x its median
        r.fnoise[5]['noise_jy'] = repr(float(r.fnoise[5]['vsanmed_jy']) * 3.5)
    fault('a_vnoise', v_noisy, ('a', 'noise'), '(a) a searched V frame whose noise exceeds 3 x the V median it was judged with')

    def v_quiet_refused(r):                 # (b) a V noise refusal of a quiet frame
        r.sanity_v.append([LST, SC, NIGHTS[2], '10.0', '11.0', 'A1'])
    fault('b_vnoise', v_quiet_refused, ('b', 'noise'), '(b) a V noise refusal whose recorded noise and median do not refuse')

    r = fresh('b_unpublished')              # (b) a final without its catalog row
    r.write()
    rows = CC._rows(os.path.join(r.U, 'master_catalog.csv'))
    with open(os.path.join(r.U, 'master_catalog.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=ST.CAT_FIELDS)
        w.writeheader()
        w.writerows(rows[1:])
    shutil.copy2(os.path.join(r.U, 'master_catalog.csv'), r.merged)
    ok, f, res = run_check(r)
    check("(b) a final detection without its catalog row - refused with ('b', 'unpublished'), nothing else",
          not ok and f == {('b', 'unpublished')}, (ok, sorted(f)))

    r = fresh('a_unbacked')               # (a) a catalog row without its final, and a duplicate
    r.write()
    rows = CC._rows(os.path.join(r.U, 'master_catalog.csv'))
    extra = dict(rows[0], cand_id='VD_X_0001', ra=float(rows[0]['ra']) + 3.0)
    with open(os.path.join(r.U, 'master_catalog.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=ST.CAT_FIELDS)
        w.writeheader()
        w.writerows(rows + [extra, dict(rows[1], cand_id='VD_DUP_0001')])
    shutil.copy2(os.path.join(r.U, 'master_catalog.csv'), r.merged)
    ok, f, res = run_check(r)
    check("(a) a catalog row without a final detection, and a duplicate row - refused with ('a', 'unbacked') and "
          "('a', 'duplicate'), nothing else", not ok and f == {('a', 'unbacked'), ('a', 'duplicate')}, (ok, sorted(f)))

    r = fresh('a_excluded').write()        # (a) a searched night excluded now
    r.excluded = {NIGHTS[2]}
    ok, f, res = run_check(r)
    check("(a) a searched night excluded since the search - refused with ('a', 'excluded'), nothing else",
          not ok and f == {('a', 'excluded')}, (ok, sorted(f)))

    def marginal(r):                        # (b) a v110 marginal removed
        x, y = r.pixels(NIGHTS[3], 50, 90, 1, avoid={(int(z['pix_x']), int(z['pix_y'])) for z in r.ledger})[0]
        r.led(NIGHTS[3], x, y, veto='v110:marginal')
    fault('b_v110', marginal, ('b', 'v110'), '(b) a v110 marginal detection removed (marginal is kept and labelled)')

    r = fresh('i_clean', 'I').write()      # Stokes I: the clean run, then the I noise rule both ways (weights.csv qa3)
    ok, f, res = run_check(r)
    check('the clean Stokes I run passes the check', ok, sorted(f))

    def i_noisy(r):                         # (a) a searched I frame whose qa3 rms exceeds 3 x the cell's (upper) median
        r.qa3[NIGHTS[5]] = 2000.0
    fault('a_inoise', i_noisy, ('a', 'noise'), '(a) a searched I frame whose qa3 rms exceeds 3 x its cell median', stokes='I')

    def i_quiet_refused(r):                 # (b) an I noise refusal the qa3 values do not make
        r.sanity_i.append([LST, SC, NIGHTS[2], '502.0', '503.0', 'A1'])
    fault('b_inoise', i_quiet_refused, ('b', 'noise'), '(b) an I noise refusal that today\'s qa3 values do not make',
          stokes='I')

    for name, rules, desc in (('prod', dict(rules='production', version=None), "searched under 'production'"),
                              ('ver', dict(rules='once-in-search', version='b02-3a'), 'searched at another rules version'),
                              ('none', None, 'without a rules record (a production-fleet attempt)')):
        r = fresh(f'rules_{name}')
        r.write()
        p = os.path.join(r.A, RR.RULES_RECORD)
        if rules is None:
            os.remove(p)
        else:
            json.dump(rules, open(p, 'w'))
        ok, f, res = run_check(r)
        check(f'an attempt {desc} is refused before any row is read (no report)', not ok and not f, (ok, f, res))


def test_pulsar_stage():
    """The stage's plumbing: production's group rule (removal_rules.pulsar_consistency, ported and tested against production
    in test_removal_rules) replaced by one that judges the test's groups."""
    real = RR.pulsar_consistency
    try:
        r = fresh('psr0').write()
        CC._deps = deps(r)
        r.excluded = set()
        n0 = open(os.path.join(r.U, 'master_catalog.csv')).read()
        res = CC.pulsars(r.U, r.man, LABEL, out=lambda *a, **k: None)
        check('pulsar stage, no pulsar decision: the catalog is unchanged, nothing restored or removed, the record written',
              open(os.path.join(r.U, 'master_catalog.csv')).read() == n0 and res['restored'] == res['removed'] == 0
              and os.path.exists(os.path.join(r.U, CC.PULSAR_RECORD)), res)

        r = fresh('psr1')
        d1 = NIGHTS[1]
        kept = first(r, lambda x: x['final'] == 1 and x['date'] == d1)
        cutr = first(r, lambda x: x['veto_reason'] == 'table' and x['date'] == d1)
        for z, c_ in ((kept, 0), (cutr, 1)):
            ra, dec = r.sky(int(z['pix_x']), int(z['pix_y']))
            note = (f'[el50-90] table-exempt: ATNF pulsar PSR J0000+0000 is the parent' if not c_ else
                    f'[el50-90] table: ATNF pulsar PSR J0000+0000 within 1 beam NOT exempting')
            r.notes[(ra, dec)] = note
            r.tpd.append(dict(lst=LST, scope=SC, stokes='V', date=d1, ra=ra, dec=dec, pix_x=z['pix_x'], pix_y=z['pix_y'], cut=c_,
                              note=note))
        r.write()
        CC._deps = deps(r)
        r.excluded = set()

        def judge(rows, cut, day, excluded, G, LOC, psr):    # the kept one inconsistent (removed), the cut one consistent
            for i, (x, nt) in list(cut.items()):
                cut[i] = (not x, f'synthetic group: {"consistent" if x else "inconsistent"} ({nt})')
            return 1, 1, [('PSR J0000+0000', 2, 400, 260)], 1
        RR.pulsar_consistency = judge
        res = CC.pulsars(r.U, r.man, LABEL, out=lambda *a, **k: None)
        cat = CC._rows(os.path.join(r.U, 'master_catalog.csv'))
        rec = CC._rows(os.path.join(r.U, CC.PULSAR_RECORD))
        rem = CC._rows(os.path.join(r.U, CC.PULSAR_REMOVED))
        pr = [x for x in cat if x['cand_id'].startswith('PR_')]
        check('pulsar stage: a consistent group\'s cut detection restored as a PR_ row from its ledger row (position, peak, '
              'S/N), an inconsistent group\'s exempted one removed - each recorded, the removed row with its reason, the '
              'merged columns kept', res['restored'] == 1 and res['removed'] == 1 and len(pr) == 1
              and float(pr[0]['peak_jy']) == float(cutr['peak_jy']) and (float(pr[0]['ra']), float(pr[0]['dec'])) == r.sky(
                  int(cutr['pix_x']), int(cutr['pix_y'])) and sorted(x['action'] for x in rec) == ['removed', 'restored']
              and len(rem) == 1 and rem[0]['removal_reason'].startswith('synthetic group: inconsistent')
              and list(cat[0].keys()) == ST.CAT_FIELDS, (res, rec, [x['cand_id'] for x in rem]))
        RR.pulsar_consistency = real
        CC.cell_base = lambda cell, roots_csv=None: r.base
        try:
            CC.check(r.U, r.man, LABEL, merged=r.merged, out=lambda *a, **k: None)
            ok, f = True, set()
        except SystemExit:
            ok, f = False, {(x['direction'], x['rule']) for x in CC._rows(os.path.join(r.U, CC.CHECK_REPORT))}
        out = CC._rows(os.path.join(r.U, 'master_catalog.csv')) if ok else []
        prr = [x for x in out if x['cand_id'].startswith('PR_')]
        check('the check accepts the pulsar stage\'s decisions: the restored row (a table cut today, as recorded) published with '
              'the stage\'s words in conform_note, the removed one in master_catalog_removed.csv', ok and len(prr) == 1
              and prr[0]['conform_note'].startswith('synthetic group: consistent')
              and len(CC._rows(os.path.join(r.U, 'master_catalog_removed.csv'))) == 1, (ok, sorted(f)))
        # the restored row's table cut lifted: the stage's decision is no longer what the table says - refused
        shutil.copy2(r.merged, os.path.join(r.root, 'merged_keep.csv'))
        r.cutset.discard(r.sky(int(cutr['pix_x']), int(cutr['pix_y'])))
        RR.pulsar_consistency = judge
        shutil.copy2(r.merged, os.path.join(r.U, 'master_catalog.csv'))
        CC.pulsars(r.U, r.man, LABEL, out=lambda *a, **k: None)
        RR.pulsar_consistency = real
        try:
            CC.check(r.U, r.man, LABEL, merged=r.merged, out=lambda *a, **k: None)
            ok, f = True, set()
        except SystemExit:
            ok, f = False, {(x['direction'], x['rule']) for x in CC._rows(os.path.join(r.U, CC.CHECK_REPORT))}
        check("a row the pulsar stage restored as a table cut that the table no longer cuts - refused with ('b', 'pulsar')",
              not ok and ('b', 'pulsar') in f, (ok, sorted(f)))
    finally:
        RR.pulsar_consistency = real


def test_mode():
    check("catalog_conform.MODE is 'check' (ruled) and refuses an unknown value", CC.conform_mode() == 'check'
          and _raises(lambda: CC.conform_mode('chek')), CC.conform_mode())
    pf = open(os.path.join(REPO, 'exopipe', 'phase3', 'parity', 'production_20261013.cfg')).read()
    check("the parity file names catalog_conform.MODE = 'remove' with catalog_conform.DEEP_ELEV_EPOCH = 'mid-hour'",
          "catalog_conform.MODE = 'remove'" in pf and "catalog_conform.DEEP_ELEV_EPOCH = 'mid-hour'" in pf)
    from exopipe.phase3.lib.catalog import deep_chain as D
    check("the deep chain uses MODE and DEEP_ELEV_EPOCH (an override of them is not a dead override) and orders pulsars between "
          "merge and conform", {'MODE', 'DEEP_ELEV_EPOCH'} <= set(D.LIVE_KEYS['catalog_conform'])
          and list(CFG.catalog.DEEP_STAGES[:5]) == ['preflight', 'harvest', 'merge', 'pulsars', 'conform'])


def _raises(fn):
    try:
        fn()
    except Exception:
        return True
    return False


def main():
    shutil.rmtree(T, ignore_errors=True)
    os.makedirs(T)
    keep = (CC._deps, CC.cell_base)
    try:
        test_mode()
        test_clean()
        test_faults()
        test_pulsar_stage()
    finally:
        CC._deps, CC.cell_base = keep
        if os.environ.get('KEEP_T') != '1':
            shutil.rmtree(T, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
