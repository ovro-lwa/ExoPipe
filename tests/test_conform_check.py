"""B-02 steps 2 and 4: lib/catalog/conform_check.py - the publish conformance as a check, and the pulsar stage - on synthetic
runs (plan/wave2/B02_DESIGN.md "Steps 2 and 4").

A run of one cell (05h 23MHz - a Red band: a 5-degree Jupiter track - six nights): attempts searched under 'once-in-search'
with their rules record, archived search frames (small FITS on a SIN grid about the 05h zenith, each with its own time block
and beam), frame_noise.csv (a TBLNOISE of its own per frame), table_pulsar_decisions.csv, a gate ledger in which every
removal rule of the search acts (finals, 'elevation', 'jupiter-track', 'table', 'row-line' rows, chosen on the grid so that
each rule's own function decides them), and the merged catalog of the finals. The bright-source table and the parents are
synthetic (a table that cuts the (frame, position) pairs the test names - the frame told by its TBLNOISE); the elevation
floor, the Jupiter track, the daytime gate, the noise rules, the row line, the backing - and, in (4), production's pulsar
group rule with the served localization's estimator - are the real functions.
  (1) the clean run passes, and writes production's outputs (as-run = the merge's output, nothing removed), which pass the
      conformant_catalog content test; V and I attempts of the cell together (their refusal files share names) pass;
  (2) each rule faulted in each direction is refused with exactly its finding: a final below the floor / an 'elevation' row
      above it; a final on the track / a 'jupiter-track' row off it; the table cutting a final / not cutting a 'table' row
      (one frame each); a row line among the finals / a 'row-line' row alone; a searched frame in daytime / a daytime
      refusal at night; a searched V frame above the noise rule / a V noise refusal below it; the I noise rule both ways;
      a catalog row without its final / a final without its row; a duplicate; a searched night excluded now / a night
      the search left out as excluded that data_decisions admits now and no search took up since (each attempt's own
      record: V and I); a v110
      marginal removal; an axis ratio above the search's limit; attempts under 'production', at another version (b02-3b
      among them: no record of the nights left out as excluded), without a record refused before any row; and not
      refused: a frame whose source extraction was refused, on a night excluded since; a night left out as excluded and
      excluded still, or searched since by an append; a final whose rounded size sits above the search's size floor
      (the search's own gate is not re-run);
  (3) restored rows carry an ID in the catalog's shape (field 2 the LST, as downstream parses it);
  (4) the pulsar stage with production's group rule: a group consistent with its pulsar gets its cut detection back, an
      inconsistent one loses its exempted detections; both recorded (members included) and accepted by the check, which
      refuses a restored row the table no longer cuts; a kept decision that joins neither a catalog row nor a 'row-line'
      ledger row is refused;
  (5) the chain's merged set (deep_chain._merged_attempts): a stale attempt left out, an unharvested one refused; the
      config overrides a unit holds reach the module child (forward_overrides / apply_forwarded).
Run: python tests/test_conform_check.py
"""
import csv
import json
import os
import shutil
import subprocess
import sys
import types

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
QUIET = dict(out=lambda *a, **k: None)


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


def tblnoise(k):
    """Each frame's own TBLNOISE (the synthetic table tells the frame by it)."""
    return 0.0105 + 0.0001 * k


class Attempt:
    """One synthetic once-in-search attempt (one Stokes of the cell) - see the module docstring."""

    def __init__(self, run, stokes):
        self.run, self.st = run, stokes
        self.A = os.path.join(run.root, 'attempts', f'{CELL}_{stokes}_20261016T000000')
        self.rd = os.path.join(self.A, 'transients', f'Run_{LABEL}')
        os.makedirs(os.path.join(self.rd, 'refusals'), exist_ok=True)
        self.headers = {d: frame_header(d, k) for k, d in enumerate(NIGHTS)}
        self.ledger, self.fnoise, self.tpd = [], [], []
        self.day_ref, self.sanity_v, self.sanity_i, self.xref, self.ddrop = [], [], [], [], []
        self.rules = RR.rules_record('once-in-search')

    def led(self, night, x, y, veto='', final=0, **kw):
        r = dict.fromkeys(ST.GLED_COLS, '')
        ra, dec = self.run.sky(x, y)
        r.update(lst=LST, scope=SC, stokes=self.st, date=night, ra=ra + 0.001, dec=dec - 0.001, pix_x=x, pix_y=y, snr=7.5,
                 peak_jy=0.5, morph_pass=1, veto_reason=veto, final=final, pass_mstruct=1,
                 fp_prediction='inconclusive' if self.st == 'V' else '')
        r.update(kw)
        self.ledger.append(r)
        return r

    def build(self):
        """Per night two finals high in the sky, one 'elevation' row low, one 'table' row its frame's table cuts; night 1
        also a 'jupiter-track' row on the track and a row line of five ('row-line')."""
        run = self.run
        for k, d in enumerate(NIGHTS):
            hi = run.pixels(d, 40, 90, 3, avoid=run.used(self))
            for x, y in hi[:2]:
                self.led(d, x, y, final=1)
            x, y = hi[2]
            run.cutset.add((tblnoise(k),) + run.sky(x, y))
            self.led(d, x, y, veto='table')
            x, y = run.pixels(d, 0, 24, 1, avoid=run.used(self))[0]
            self.led(d, x, y, veto='elevation')
            self.fnoise.append(dict(lst=LST, scope=SC, stokes=self.st, date=d, noise_jy=repr(0.010 + 0.0005 * k), admitted=1,
                                    population=1, tblnoise_jy=repr(tblnoise(k)), vsanmed_jy='', vsanity_x='', attempt='A1'))
        pop = sorted(float(r['noise_jy']) for r in self.fnoise)
        if self.st == 'V':                  # the V noise rule's median and factor on V rows only
            for r in self.fnoise:
                r.update(vsanmed_jy=repr(pop[len(pop) // 2]), vsanity_x=repr(3.0))
        jra, jdec, jd = next(p for p in run.track if p[2] == NIGHTS[0])   # a 'jupiter-track' row on night 1's track point
        jx, jy = (int(round(float(v))) for v in run.W.world_to_pixel_values(jra, jdec))
        self.jpix = (jx, jy)
        self.led(NIGHTS[0], jx, jy, veto=f'jupiter-track:{jd}')
        used = run.used(self)
        line = next(L for y in range(20, 44) for L in [[(x, y) for x in range(14, 50, 8)]]
                    if all(run.el(NIGHTS[0], x, y) >= 30 and not ({(x, y), (x, y + 1), (x, y - 1)} & used)
                           and not RR.jupiter_track_hit(*run.sky(x, y), SC, run.track) for x, y in L))
        for x, y in line:                   # five on one pixel row, above the floor, off the track and the other rows
            self.led(NIGHTS[0], x, y, veto='row-line')
        return self

    def write(self):
        run = self.run
        with open(os.path.join(self.A, RR.RULES_RECORD), 'w') as fh:
            json.dump(self.rules, fh)
        sub = 'vresid' if self.st == 'V' else 'iresid'
        fdir = os.path.join(self.A, f'Run_{LABEL}', sub, CELL)
        os.makedirs(fdir, exist_ok=True)
        for d, h in self.headers.items():
            fits.PrimaryHDU(np.zeros((N0, N0), np.float32), h).writeto(os.path.join(fdir, f'{d}_{self.st}_redo_resid_{SC}.fits'),
                                                                       overwrite=True)
        wcsv(os.path.join(self.rd, 'v_deep_ledger.csv'), ST.GLED_COLS, self.ledger)
        wcsv(os.path.join(self.rd, RR.FRAME_NOISE), RR.FRAME_NOISE_COLS, self.fnoise)
        wcsv(os.path.join(self.rd, RR.TABLE_PULSAR), RR.TABLE_PULSAR_COLS, self.tpd)
        ref = os.path.join(self.rd, 'refusals')
        for name, rows in ((f'{LST}_{SC}_daytime_excluded.csv', self.day_ref), (f'{LST}_{SC}_sanity_V_excluded.csv', self.sanity_v),
                           (f'{LST}_{SC}_sanity_I_excluded.csv', self.sanity_i),
                           (f'{LST}_{SC}_extraction_{self.st}_excluded.csv', self.xref),
                           (f'{LST}_{SC}_datadecisions_excluded.csv', self.ddrop)):
            p = os.path.join(ref, name)
            if os.path.exists(p):
                os.remove(p)
            if rows:
                with open(p, 'w', newline='') as fh:
                    csv.writer(fh, lineterminator='\n').writerows(rows)
        os.makedirs(os.path.join(run.U, sub), exist_ok=True)
        link = os.path.join(run.U, sub, CELL)
        if not os.path.islink(link):
            os.symlink(fdir, link)


def wcsv(p, cols, rows):
    with open(p, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


class Run:
    """The cell's coadd root (weights.csv), its attempts and the merged run directory."""

    def __init__(self, root, stokes=('V',)):
        self.root = root
        shutil.rmtree(root, ignore_errors=True)
        self.U = os.path.join(root, 'run')
        self.base = os.path.join(root, 'coadd', CELL)
        for d in (self.U, os.path.join(self.base, 'coadd_final')):
            os.makedirs(d, exist_ok=True)
        self.W = WCS(frame_header(NIGHTS[0], 0)).celestial
        self.cutset, self.notes, self.excluded = set(), {}, set()
        self.qa3 = {d: 500 + k for k, d in enumerate(NIGHTS)}
        self.weights()
        self.track = RR.jupiter_track(LST, os.path.join(self.base, 'coadd_final', 'weights.csv'))
        self.att = {}
        for st in stokes:
            self.att[st] = Attempt(self, st).build()

    def weights(self):
        with open(os.path.join(self.base, 'coadd_final', 'weights.csv'), 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['date', 'weight', 'qa3_scale', 'qa3_tilt', 'qa3_nsrc', 'qa3_guard', 'qa3_rms_mjy', 'sci_dir', 'in_master'])
            for d in NIGHTS:
                w.writerow([d, 1.0, 1.0, 0.0, 500, 'pass', self.qa3[d], '', 1])

    def sky(self, x, y):
        s = self.W.pixel_to_world(x, y)
        return round(float(s.ra.deg), 5), round(float(s.dec.deg), 5)

    def el(self, night, x, y):
        return RR.frame_elevation(LST, *self.sky(x, y), frame_header(night, NIGHTS.index(night)), epoch='frame-mid')[0]

    def used(self, a=None):
        return {(int(z['pix_x']), int(z['pix_y'])) for at in self.att.values() for z in at.ledger} | \
               ({(int(z['pix_x']), int(z['pix_y'])) for z in a.ledger} if a else set())

    def pixels(self, night, lo, hi, n, avoid=()):
        """n pixels of the grid whose frame-mid elevation on ``night`` lies in [lo, hi), off the track, apart."""
        out = []
        for y in range(4, N0 - 4, 3):
            for x in range(4, N0 - 4, 3):
                if (x, y) in avoid or any(abs(x - a) < 3 and abs(y - b) < 3 for a, b in out):
                    continue
                if not np.isfinite(self.sky(x, y)[0]):
                    continue
                if lo <= self.el(night, x, y) < hi and not RR.jupiter_track_hit(*self.sky(x, y), SC, self.track):
                    out.append((x, y))
                    if len(out) == n:
                        return out
        raise AssertionError(f'no {n} pixels at {lo}-{hi} deg')

    def write(self):
        self.weights()
        cat = []
        for st, a in self.att.items():
            a.write()
            for j, r in enumerate(x for x in a.ledger if x['final'] in (1, '1')):
                ra, dec = self.sky(int(r['pix_x']), int(r['pix_y']))
                c = dict.fromkeys(ST.CAT_FIELDS, '')
                c.update(cand_id=f'{st}D_{LST}_{r["date"]}_{SC}_{j:04d}', lst=LST, date=r['date'], band=SC, stokes=st,
                         cadence='deep', ra=ra, dec=dec, peak_jy=r['peak_jy'], snr=r['snr'],
                         fp_prediction='inconclusive' if st == 'V' else '', axis_norm=r.get('axis_norm', ''),
                         size_beams=r.get('size_beams', ''))
                cat.append(c)
        self.cat = cat
        wcsv(os.path.join(self.U, 'master_catalog.csv'), ST.CAT_FIELDS, cat)
        self.merged = os.path.join(self.root, 'merged_master_catalog.csv')
        shutil.copy2(os.path.join(self.U, 'master_catalog.csv'), self.merged)
        self.man = os.path.join(self.root, CC.MERGED_ATTEMPTS)
        wcsv(self.man, ['cell', 'stokes', 'node', 'att', 'dir', 'harvested', 'label', 'patched', 'patched_from'],
             [dict(cell=CELL, stokes=st, node='x', att='20261016T000000', dir=a.A, harvested=1, label=LABEL)
              for st, a in self.att.items()])
        for name in (CC.CHECK_REPORT, CC.PULSAR_RECORD, CC.PULSAR_REMOVED, 'frame_times.csv', 'frame_beam_ratio.csv',
                     'frame_beam_bmaj.csv'):
            p = os.path.join(self.U, name)
            if os.path.exists(p):
                os.remove(p)
        return self


class FakeG:
    def __init__(self, beam=0.2):
        self.beam = beam

    def parents(self, l, b):
        return np.zeros((0, 3))

    def phase_centre(self, l, b):
        return (75.0, 37.2)

    def deep_target_beam(self, l, b, d, s):
        return self.beam


def deps(run, beam=0.2, LOC=None):
    class PCT:
        @staticmethod
        def removals(st, b, parents, noise, beam_, ra, dec, elev=None, phase_centre=None):
            assert noise is not None and noise > 0 and beam_ == 0.2 and elev is not None and len(elev) == len(ra)
            cuts = {(round(c[0], 9),) + c[1:] for c in run.cutset}
            keys = [(round(float(noise), 9), round(float(a), 5), round(float(d), 5)) for a, d in zip(ra, dec)]
            cut = np.array([k in cuts for k in keys])
            return cut, [run.notes.get(k[1:], 'table: synthetic cut' if c else '') for k, c in zip(keys, cut)]
    return lambda: dict(G=FakeG(beam), LOC=LOC, PCT=PCT, excluded=lambda l, b, d: d in run.excluded)


def stage_record(run):
    """An empty pulsar stage record (the stage judged no group) - the check's precondition."""
    wcsv(os.path.join(run.U, CC.PULSAR_RECORD), CC.PULSAR_RECORD_COLS, [])
    wcsv(os.path.join(run.U, CC.PULSAR_REMOVED), ST.CAT_FIELDS + ['removal_reason'], [])


def run_check(run, stage=True):
    """(passed, the findings {(direction, rule)}, the result) of conform_check.check on ``run``."""
    CC._deps = deps(run)
    CC.cell_base = lambda cell, roots_csv=None: run.base
    if stage:
        stage_record(run)
    return _check(run)


def _check(run):
    try:
        res = CC.check(run.U, run.man, LABEL, merged=run.merged, **QUIET)
        return True, set(), res
    except SystemExit as e:
        rep = os.path.join(run.U, CC.CHECK_REPORT)
        f = {(r['direction'], r['rule']) for r in CC._rows(rep)} if os.path.exists(rep) else set()
        return False, f, f'exit {e.code}'


def fresh(name, stokes=('V',)):
    return Run(os.path.join(T, name), stokes)


def first(a, pred):
    return next(x for x in a.ledger if pred(x))


# ---------------------------------------------------------------------------------------------------------------------
# (1) the clean run
# ---------------------------------------------------------------------------------------------------------------------

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
        check("passed: the conformant_catalog content test passes on the check's output (the floor at the frame mid-time from "
              "frame_times.csv)", cr[0], cr)
    r = fresh('clean_iv', stokes=('V', 'I')).write()
    ok, f, res = run_check(r)
    check("V and I attempts of one cell pass together (each attempt's own refusal files - the merge's links would shadow one "
          "Stokes' with the other's)", ok and res['attempts'] == 2, (ok, sorted(f), res))


# ---------------------------------------------------------------------------------------------------------------------
# (2) every rule faulted in each direction
# ---------------------------------------------------------------------------------------------------------------------

def fault(name, edit, want, desc, also=(), stokes=('V',)):
    """A fresh run, ``edit`` applied before it is written; the check must refuse with exactly ``want`` (and ``also``), or,
    with want None, pass."""
    r = fresh(name, stokes)
    edit(r)
    r.write()
    ok, f, res = run_check(r)
    if want:
        check(f'{desc} - refused with {want}' + (f' (and {sorted(also)})' if also else '') + ', nothing else',
              not ok and f == {want, *also}, (ok, sorted(f), res))
    else:
        check(f'{desc} - passes', ok, (ok, sorted(f), res))


def test_faults():
    def final_low(r):
        x, y = r.pixels(NIGHTS[1], 0, 24, 1, avoid=r.used())[0]
        r.att['V'].led(NIGHTS[1], x, y, final=1)
    fault('a_floor', final_low, ('a', 'elevation'), "(a) a final below the elevation floor at its frame's mid-time")

    def elev_high(r):
        x, y = r.pixels(NIGHTS[1], 50, 90, 1, avoid=r.used())[0]
        r.att['V'].led(NIGHTS[1], x, y, veto='elevation')
    fault('b_floor', elev_high, ('b', 'elevation'), "(b) an 'elevation' row that the floor keeps today")

    def final_track(r):
        z = first(r.att['V'], lambda x: x['veto_reason'].startswith('jupiter-track'))
        z['veto_reason'], z['final'] = '', 1
    fault('a_track', final_track, ('a', 'jupiter-track'), '(a) a final on the Jupiter track of a co-added night')

    def track_off(r):
        x, y = r.pixels(NIGHTS[2], 50, 90, 1, avoid=r.used())[0]
        r.att['V'].led(NIGHTS[2], x, y, veto=f'jupiter-track:{NIGHTS[2]}')
    fault('b_track', track_off, ('b', 'jupiter-track'), "(b) a 'jupiter-track' row off every co-added night's track")

    def table_cuts_final(r):
        z = first(r.att['V'], lambda x: x['final'] == 1 and x['date'] == NIGHTS[3])
        r.cutset.add((tblnoise(3),) + r.sky(int(z['pix_x']), int(z['pix_y'])))
    fault('a_table', table_cuts_final, ('a', 'table'), "(a) a final its frame's bright-source table cuts today (tables "
                                                       "re-derived)")

    def table_keeps(r):
        z = first(r.att['V'], lambda x: x['veto_reason'] == 'table' and x['date'] == NIGHTS[2])
        r.cutset.discard((tblnoise(2),) + r.sky(int(z['pix_x']), int(z['pix_y'])))
    fault('b_table', table_keeps, ('b', 'table'), "(b) a 'table' row its frame's table keeps today")

    def finals_line(r):
        for z in [x for x in r.att['V'].ledger if x['veto_reason'] == 'row-line']:
            z['veto_reason'], z['final'] = '', 1
    fault('a_rowline', finals_line, ('a', 'row-line'), '(a) five finals of one epoch on one pixel row +/-1')

    def lone_line(r):
        a = r.att['V']
        for x in [x for x in a.ledger if x['veto_reason'] == 'row-line'][1:]:
            a.ledger.remove(x)
    fault('b_rowline', lone_line, ('b', 'row-line'), "(b) a 'row-line' row alone on its row")

    def daytime_frame(r):
        r.att['V'].headers[NIGHTS[3]] = frame_header(NIGHTS[3], 3, daytime=True)
    fault('a_daytime', daytime_frame, ('a', 'daytime'), '(a) a searched frame the daytime gate refuses today',
          also={('a', 'elevation')})          # the frame's new times move its finals below the floor too

    def day_at_night(r):
        r.att['V'].day_ref.append([LST, SC, NIGHTS[4], 'V', 'daytime: Sun 1.0 to 6.3 deg over the frame', 'A1'])
    fault('b_daytime', day_at_night, ('b', 'daytime'), '(b) a frame refused as daytime that the daytime gate admits today')

    def v_noisy(r):
        fn = r.att['V'].fnoise[5]
        fn['noise_jy'] = repr(float(fn['vsanmed_jy']) * 3.5)
    fault('a_vnoise', v_noisy, ('a', 'noise'), '(a) a searched V frame whose noise exceeds 3 x the V median it was judged with')

    def v_quiet_refused(r):
        r.att['V'].sanity_v.append([LST, SC, NIGHTS[2], '10.0', '11.0', 'A1'])
    fault('b_vnoise', v_quiet_refused, ('b', 'noise'), '(b) a V noise refusal whose recorded noise and median do not refuse')

    def i_noisy(r):
        r.qa3[NIGHTS[5]] = 2000.0
    fault('a_inoise', i_noisy, ('a', 'noise'), '(a) a searched I frame whose qa3 rms exceeds 3 x its cell median', stokes=('I',))

    def i_quiet_refused(r):
        r.att['I'].sanity_i.append([LST, SC, NIGHTS[2], '502.0', '503.0', 'A1'])
    fault('b_inoise', i_quiet_refused, ('b', 'noise'), "(b) an I noise refusal that today's qa3 values do not make",
          stokes=('I',))

    def marginal(r):
        x, y = r.pixels(NIGHTS[3], 50, 90, 1, avoid=r.used())[0]
        r.att['V'].led(NIGHTS[3], x, y, veto='v110:marginal')
    fault('b_v110', marginal, ('b', 'v110'), '(b) a v110 marginal detection removed (marginal is kept and labelled)')

    def axis_wide(r):
        z = first(r.att['V'], lambda x: x['final'] == 1 and x['date'] == NIGHTS[1])
        z.update(snr=9.0, axis_norm=1.52)
    fault('a_axis', axis_wide, ('a', 'axis'), "(a) a final above 8 sigma whose axis ratio exceeds the search's 1.5 x beam")

    def size_rounded(r):                    # Run_P3S5's 04h 69MHz I 2024-12-22 final: 1.6 - 0.099, kept by the search
        z = first(r.att['V'], lambda x: x['final'] == 1 and x['date'] == NIGHTS[1])
        z.update(snr=9.0, size_beams=1.6, size_err_beams=0.099)
    fault('size', size_rounded, None, "a final whose rounded recorded size (1.6 - 0.099) sits above the search's size floor "
                                      "(the search's own gate, not re-run)")

    def xref_excluded(r):                   # a frame whose source extraction was refused, on a night excluded since
        a = r.att['V']
        d = NIGHTS[4]
        a.ledger = [x for x in a.ledger if x['date'] != d]
        a.xref.append([LST, SC, d, 'V', 'source extraction failed: synthetic', 'A1', '2026-10-16T00:00:00Z'])
        r.excluded = {d}
    fault('xref', xref_excluded, None, 'a frame whose source extraction was refused (not searched), on a night excluded since')

    left = '2024-12-24'                     # a night of the cell the search left out as excluded: no frame, no ledger row

    def left_out(r):
        r.att['V'].ddrop.append([LST, SC, left, 'A1'])
        r.excluded = {left}
    fault('dd_kept', left_out, None, 'a night the search left out as excluded by data_decisions, excluded still')

    def appended(r):                        # the reviewer's case: the base left the night out as excluded, an append
        r.att['V'].ddrop.append([LST, SC, NIGHTS[3], 'A0'])   # searched it since (its frame and ledger rows), admitted now
    fault('dd_appended', appended, None, 'a night the base left out as excluded and an append searched since (data_decisions '
                                         'admitting it): the inherited record row is no finding')

    def appended_refused(r):                # ... and an append that built the lifted night's frame, which the V noise rule
        a = r.att['V']                       # then refused (frame_noise admitted 0, its refusal row): still no finding -
        a.ddrop.append([LST, SC, left, 'A0'])   # the attempt took the night up, whatever a per-Stokes gate decided
        fn0 = dict(a.fnoise[0], date=left, admitted=0, population=0)
        fn0['noise_jy'] = repr(float(fn0['vsanmed_jy']) * 4.0)
        a.fnoise.append(fn0)
        a.sanity_v.append([LST, SC, left, fn0['noise_jy'], fn0['vsanmed_jy'], 'A1'])
    fault('dd_appended_refused', appended_refused, None, 'a lifted night an append built the frame of and the V noise rule '
                                                         'refused: the inherited record row is no finding')

    def lifted(r):
        r.att['V'].ddrop.append([LST, SC, left, 'A1'])
    fault('b_excluded', lifted, ('b', 'excluded'), '(b) a night the search left out as excluded by data_decisions that '
                                                   'data_decisions admits now (the exclusion lifted: a night never searched)')

    r = fresh('b_excluded_iv', stokes=('V', 'I'))
    for a in r.att.values():
        a.ddrop.append([LST, SC, left, 'A1'])
    r.write()
    ok, f, res = run_check(r)
    rep = CC._rows(os.path.join(r.U, CC.CHECK_REPORT)) if not ok else []
    check("(b) the lifted exclusion in the V and I attempts of the cell (each attempt's own record) - one finding per attempt, "
          "each naming its Stokes and the night", not ok and f == {('b', 'excluded')}
          and sorted((x['stokes'], x['date']) for x in rep) == [('I', left), ('V', left)],
          (ok, sorted(f), [(x['stokes'], x['date']) for x in rep]))

    r = fresh('wcs_none')                    # a frame whose header reads but holds no sky axes: findings, no crash
    for c_ in ('CTYPE1', 'CTYPE2'):
        del r.att['V'].headers[NIGHTS[2]][c_]
    r.write()
    ok, f, res = run_check(r)
    check("a search frame with no sky axes in its WCS - refused with 'frame' findings, no crash",
          not ok and f and f <= {('a', 'frame'), ('b', 'frame'), ('record', 'frame')}, (ok, sorted(f), res))

    r = fresh('wcs_bad')                     # a frame whose WCS does not parse: one 'frame' finding per row, no crash
    r.att['V'].headers[NIGHTS[2]]['CTYPE1'] = 'RA---XYZ'
    r.write()
    ok, f, res = run_check(r)
    check("a search frame whose WCS does not parse - refused with ('a', 'frame') / ('b', 'frame') findings, no crash, and its "
          "finals still back their catalog rows (no 'unbacked')", not ok and f and f <= {('a', 'frame'), ('b', 'frame'),
                                                                                       ('record', 'frame')}, (ok, sorted(f), res))

    r = fresh('b_unpublished').write()      # (b) a final without its catalog row
    rows = CC._rows(os.path.join(r.U, 'master_catalog.csv'))
    wcsv(os.path.join(r.U, 'master_catalog.csv'), ST.CAT_FIELDS, rows[1:])
    shutil.copy2(os.path.join(r.U, 'master_catalog.csv'), r.merged)
    ok, f, res = run_check(r)
    check("(b) a final detection without its catalog row - refused with ('b', 'unpublished'), nothing else",
          not ok and f == {('b', 'unpublished')}, (ok, sorted(f)))

    r = fresh('a_unbacked').write()         # (a) a catalog row without its final, and a duplicate
    rows = CC._rows(os.path.join(r.U, 'master_catalog.csv'))
    wcsv(os.path.join(r.U, 'master_catalog.csv'), ST.CAT_FIELDS,
         rows + [dict(rows[0], cand_id='VD_X_0001', ra=float(rows[0]['ra']) + 3.0), dict(rows[1], cand_id='VD_DUP_0001')])
    shutil.copy2(os.path.join(r.U, 'master_catalog.csv'), r.merged)
    ok, f, res = run_check(r)
    check("(a) a catalog row without a final detection, and a duplicate row - refused with ('a', 'unbacked') and "
          "('a', 'duplicate'), nothing else", not ok and f == {('a', 'unbacked'), ('a', 'duplicate')}, (ok, sorted(f)))

    r = fresh('a_excluded').write()         # (a) a searched night excluded now
    r.excluded = {NIGHTS[2]}
    ok, f, res = run_check(r)
    check("(a) a searched night excluded since the search - refused with ('a', 'excluded'), nothing else",
          not ok and f == {('a', 'excluded')}, (ok, sorted(f)))

    for name, rules, desc in (('prod', dict(rules='production', version=None), "searched under 'production'"),
                              ('ver', dict(rules='once-in-search', version='b02-3a'), 'searched at another rules version'),
                              ('ver3b', dict(rules='once-in-search', version='b02-3b'),
                               'searched at b02-3b (no record of the nights it left out as excluded)'),
                              ('none', None, 'without a rules record (a production-fleet attempt)')):
        r = fresh(f'rules_{name}').write()
        p = os.path.join(r.att['V'].A, RR.RULES_RECORD)
        if rules is None:
            os.remove(p)
        else:
            json.dump(rules, open(p, 'w'))
        ok, f, res = run_check(r)
        check(f'an attempt {desc} is refused before any row is read (no report)', not ok and not f, (ok, f, res))


# ---------------------------------------------------------------------------------------------------------------------
# (3) and (4) the pulsar stage
# ---------------------------------------------------------------------------------------------------------------------

PSR = 'PSR JTEST+0000'


def psr_run(name, offset_deg):
    """A run with a pulsar group: on nights 2, 3 and 4 a detection at one pixel - two finals kept as the pulsar's parent, one
    a 'table' cut 'within 1 beam NOT exempting' - the pulsar ``offset_deg`` north of them."""
    r = fresh(name)
    a = r.att['V']
    x, y = r.pixels(NIGHTS[1], 55, 90, 1, avoid=r.used())[0]
    ra, dec = r.sky(x, y)
    for d, c_ in ((NIGHTS[1], 0), (NIGHTS[2], 1), (NIGHTS[3], 0)):
        kn = NIGHTS.index(d)
        note = (f'[el50-90] table-exempt: ATNF pulsar {PSR} is the parent' if not c_ else
                f'[el50-90] table: ATNF pulsar {PSR} within 1 beam NOT exempting')
        if c_:
            a.led(d, x, y, veto='table', snr=20.0)
            r.cutset.add((tblnoise(kn), ra, dec))
        else:
            a.led(d, x, y, final=1, snr=20.0)
        a.tpd.append(dict(lst=LST, scope=SC, stokes='V', date=d, ra=ra, dec=dec, pix_x=x, pix_y=y, cut=c_, note=note))
    r.notes = {(ra, dec): f'[el50-90] table-exempt: ATNF pulsar {PSR} is the parent'}   # the kept rows' table note
    r.psr = {PSR: (ra, dec + offset_deg)}
    r.write()
    return r, (x, y)


def psr_run_epoch(name, offset_deg):
    """A per-epoch pulsar group (exopipe_ops a2d37b9): on night 2, one pixel detected in Stokes V and in Stokes I - both finals
    kept as the pulsar's parent, so one epoch (LST, night) holds two exempted detections - the pulsar ``offset_deg`` north."""
    r = fresh(name, stokes=('V', 'I'))
    x, y = r.pixels(NIGHTS[1], 55, 90, 1, avoid=r.used())[0]
    ra, dec = r.sky(x, y)
    note = f'[el50-90] table-exempt: ATNF pulsar {PSR} is the parent'
    for st in ('V', 'I'):
        a = r.att[st]
        a.led(NIGHTS[1], x, y, final=1, snr=20.0)
        a.tpd.append(dict(lst=LST, scope=SC, stokes=st, date=NIGHTS[1], ra=ra, dec=dec, pix_x=x, pix_y=y, cut=0, note=note))
    r.notes = {(ra, dec): note}
    r.psr = {PSR: (ra, dec + offset_deg)}
    r.write()
    return r, (x, y)


def _stage(r, LOC):
    CC._deps = deps(r, beam=2.0, LOC=LOC)   # production's group rule; the pulsar within its search beam (the grid's
    CC.cell_base = lambda cell, roots_csv=None, _b=r.base: _b   # 2-degree pixels place the detections)
    real = RR.pulsar_positions
    RR.pulsar_positions = lambda path=None, _p=r.psr: dict(_p)
    try:
        return CC.pulsars(r.U, r.man, LABEL, **QUIET)
    finally:
        RR.pulsar_positions = real


def test_pulsar_stage_per_epoch():
    """a2d37b9: a per-epoch group inconsistent with the pulsar removes its exempted detections; consistent, it keeps them."""
    import source_localization as LOC
    for name, off, want in (('psr_ep_cons', 0.0, 'consistent'), ('psr_ep_incons', 1.0, 'inconsistent')):
        r, px = psr_run_epoch(name, off)
        res = _stage(r, LOC)
        grp = CC._rows(os.path.join(r.U, CC.PULSAR_GROUPS))
        rec = CC._rows(os.path.join(r.U, CC.PULSAR_RECORD))
        kinds = sorted(g['grouping'] for g in grp)
        if want == 'consistent':
            ok = res['removed'] == 0 and res['restored'] == 0 and kinds == ['across-epoch', 'per-epoch'] and rec == []
        else:
            ok = (res['removed'] == 2 and res['restored'] == 0 and kinds == ['across-epoch', 'per-epoch']
                  and all(g['consistent'] == '0' for g in grp) and sorted(x['action'] for x in rec) == ['removed', 'removed'])
        check(f'pulsar stage, two exempted detections in one epoch ({want}): '
              f'{"kept" if want == "consistent" else "both removed by their per-epoch group"}', ok, (res, grp, rec))
        okc, f, _res = _check(r)
        n_rem = len(CC._rows(os.path.join(r.U, 'master_catalog_removed.csv'))) if okc else -1
        check(f"the check accepts the stage's decisions, one epoch ({want}): "
              f"{'nothing removed' if want == 'consistent' else 'the two removed rows in master_catalog_removed.csv'}",
              okc and n_rem == (0 if want == 'consistent' else 2), (okc, f, n_rem))


def test_pulsar_stage():
    import source_localization as LOC
    r = fresh('psr0').write()
    CC._deps = deps(r)
    n0 = open(os.path.join(r.U, 'master_catalog.csv')).read()
    res = CC.pulsars(r.U, r.man, LABEL, **QUIET)
    check('pulsar stage, no pulsar decision: the catalog is unchanged, nothing restored or removed, the record written',
          open(os.path.join(r.U, 'master_catalog.csv')).read() == n0 and res['restored'] == res['removed'] == 0
          and os.path.exists(os.path.join(r.U, CC.PULSAR_RECORD)), res)
    for name, off, want in (('psr_cons', 0.0, 'consistent'), ('psr_incons', 1.0, 'inconsistent')):
        r, px = psr_run(name, off)
        res = _stage(r, LOC)
        cat = CC._rows(os.path.join(r.U, 'master_catalog.csv'))
        rec = CC._rows(os.path.join(r.U, CC.PULSAR_RECORD))
        grp = CC._rows(os.path.join(r.U, CC.PULSAR_GROUPS))
        pr = [x for x in cat if CC.is_restored(x['cand_id'])]
        if want == 'consistent':
            ok = (res['restored'] == 1 and res['removed'] == 0 and len(pr) == 1 and pr[0]['cand_id'].startswith('VR_')
                  and pr[0]['cand_id'].split('_')[1] == LST and len(grp) == 1 and grp[0]['consistent'] == '1'
                  and grp[0]['grouping'] == 'across-epoch'
                  and len(grp[0]['members'].split('|')) == 3 and [x['action'] for x in rec] == ['restored'])
        else:
            # a2d37b9: one detection per night, so no per-epoch group of two - an across-epoch group alone removes nothing
            ok = (res['restored'] == 0 and res['removed'] == 0 and not pr and len(grp) == 1 and grp[0]['consistent'] == '0'
                  and grp[0]['grouping'] == 'across-epoch' and rec == []
                  and len(CC._rows(os.path.join(r.U, CC.PULSAR_REMOVED))) == 0)
        from exopipe.phase3.lib.catalog import tests as CT
        pt = CT.pulsar_stage_record(os.path.join(r.U, 'master_catalog.csv'), merge_copy=r.merged)
        check(f"the pulsar_catalog content test passes on the stage's output ({want})", pt[0], pt)
        if want == 'consistent':
            bad = os.path.join(T, 'pulsar_catalog_corrupt.csv')
            rows_ = CC._rows(os.path.join(r.U, 'master_catalog.csv'))
            wcsv(bad, ST.CAT_FIELDS, [x for x in rows_ if not CC.is_restored(x['cand_id'])] +
                 [dict(rows_[0], cand_id=f'VR_{LST}_{NIGHTS[0]}_{SC}_00010001')])
            for nm in (CC.PULSAR_RECORD, CC.PULSAR_REMOVED):
                shutil.copy2(os.path.join(r.U, nm), os.path.join(T, nm))
            pb = CT.pulsar_stage_record(bad, merge_copy=r.merged)
            check('the pulsar_catalog content test refuses a stage catalogue whose restored row the record does not name',
                  not pb[0], pb)
            for what, edit in (('a kept row\'s peak changed', lambda x: dict(x, peak_jy='999') if not CC.is_restored(x['cand_id'])
                                and x is rows_[0] else x),
                               ('a restored row moved', lambda x: dict(x, ra=str(float(x['ra']) + 0.01)) if CC.is_restored(x['cand_id'])
                                else x)):
                wcsv(bad, ST.CAT_FIELDS, [edit(x) for x in rows_])
                pe = CT.pulsar_stage_record(bad, merge_copy=r.merged)
                check(f'the pulsar_catalog content test compares whole rows: {what} - refused', not pe[0], pe)
            restored_id = pr[0]['cand_id'] if pr else ''
            check('a restored ID numbers the detection by its pixel (stable from pass to pass)', restored_id ==
                  f'VR_{LST}_{NIGHTS[2]}_{SC}_{px[0]:04d}{px[1]:04d}', restored_id)
        check(f"pulsar stage with production's group rule, a group {want} with its pulsar: "
              + ('its cut detection restored as VR_<lst>_... (field 2 the LST), the group and its 3 members recorded'
                 if want == 'consistent' else 'one detection per night: the across-epoch group alone removes nothing (a2d37b9)'),
              ok, (res, rec, grp, [x['cand_id'] for x in pr]))
        okc, f, _res = _check(r)
        out = CC._rows(os.path.join(r.U, 'master_catalog.csv')) if okc else []
        check(f"the check accepts the stage's decisions ({want}): " + ("the restored row published with the stage's words in "
              "conform_note" if want == 'consistent' else 'nothing removed (a2d37b9: no per-epoch group)'),
              okc and ((want == 'consistent' and any(CC.is_restored(x['cand_id']) and x['conform_note'] for x in out))
                       or (want == 'inconsistent' and len(CC._rows(os.path.join(r.U, 'master_catalog_removed.csv'))) == 0)),
              (okc, sorted(f)))
        if want == 'consistent':            # the restored row's table cut lifted: the stage's decision no longer holds
            from exopipe.phase3.lib.catalog import tests as CT
            cr = CT.conformance_record(os.path.join(r.U, 'master_catalog.csv'), merge_copy=r.merged) if okc else (False,)
            check("the conformant_catalog content test passes with the restored row (named in pulsar_stage.csv)", cr[0], cr)
            r.cutset.discard((tblnoise(2),) + r.sky(*px))
            shutil.copy2(r.merged, os.path.join(r.U, 'master_catalog.csv'))
            _stage(r, LOC)
            okc, f, _res = _check(r)
            check("a row the pulsar stage restored as a table cut that its table no longer cuts - refused with ('b', "
                  "'pulsar'), nothing else", not okc and f == {('b', 'pulsar')}, (okc, sorted(f)))
    r, px = psr_run('psr_join', 0.0)          # a kept decision with no catalog row and no 'row-line' ledger row
    rows = [x for x in CC._rows(os.path.join(r.U, 'master_catalog.csv'))
            if not (x['date'] == NIGHTS[1] and (float(x['ra']), float(x['dec'])) == r.sky(*px))]
    wcsv(os.path.join(r.U, 'master_catalog.csv'), ST.CAT_FIELDS, rows)
    try:
        _stage(r, LOC)
        ok = False
    except SystemExit:
        ok = True
    check("pulsar stage: a kept pulsar decision that joins neither a catalog row nor a 'row-line' ledger row is refused", ok)


# ---------------------------------------------------------------------------------------------------------------------
# (5) the chain's merged set; the overrides the child receives
# ---------------------------------------------------------------------------------------------------------------------

def test_chain_pieces():
    from exopipe.phase3.lib.catalog import deep_chain as D
    d = os.path.join(T, 'merged')
    os.makedirs(d, exist_ok=True)
    man = os.path.join(d, 'harvest_manifest.csv')
    wcsv(man, ['cell', 'stokes', 'node', 'att', 'dir', 'harvested', 'label', 'patched', 'patched_from'],
         [dict(cell=c_, stokes=s_, node='x', att='a', dir=f'/x/{c_}_{s_}', harvested=1, label=LABEL)
          for c_ in ('05h_23MHz', '06h_Red') for s_ in ('V', 'I')])

    class U:
        def __init__(self):
            self.meta = {}

        def refuse(self, why):
            raise RuntimeError(why)

    def fake(meta):
        return types.SimpleNamespace(u=lambda st: f'X.{st}', chain=types.SimpleNamespace(
            meta=lambda unit, k: (meta.get(k) if unit == 'X.merge' else (man if k == 'harvest_manifest' else None))))
    u = U()
    out = D.DeepRun._merged_attempts(fake({'stale_attempts': [['06h_Red', 'V']]}), u, d)
    got = [(x['cell'], x['stokes']) for x in CC._rows(out)]
    check("the merged set: the harvest manifest less the attempts the merge refused as stale", got ==
          [('05h_23MHz', 'V'), ('05h_23MHz', 'I'), ('06h_Red', 'I')] and u.meta['merged_attempts']['n'] == 3, (got, u.meta))
    try:
        D.DeepRun._merged_attempts(fake({'unharvested_warning': 'merge: WARNING 1 published attempts not harvested yet'}),
                                   U(), d)
        ok = False
    except RuntimeError as e:
        ok = 'unharvested' in str(e)
    check("the merged set: an attempt the merge skipped as unharvested refuses (the merged set is not recorded)", ok)
    try:
        CFG.authorize(out=os.path.join(T, 'out'))
        CFG.set('catalog_conform.DEEP_ELEV_EPOCH', 'mid-hour')
        js = CC.forward_overrides(CFG)
        CFG.reset()
        CFG.authorize(out=os.path.join(T, 'out'))
        CFG.set('catalog_conform.DEEP_ELEV_EPOCH', 'mid-hour')
        CFG.set('search_detect.SANITY_X', '3.5')
        CFG.set('catalog_conform.TABLE_ELEV[deep]', 'False')
        want = {o[0]: o[2] for o in CFG.overrides}
        js2 = CC.forward_overrides(CFG)
    finally:
        CFG.reset()
    code = ('import sys; sys.path.insert(0, %r)\nfrom exopipe.phase3.core import env; env.legacy()\n'
            'from exopipe.phase3.lib.catalog import conform_check as CC\nfrom exopipe.phase3.config import CFG\n'
            'import os\nn = CC.apply_forwarded(os.environ[CC.OVERRIDES_ENV])\n'
            'print(n, CFG.catalog_conform.DEEP_ELEV_EPOCH, CFG.overrides[0][3])' % REPO)
    p = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', **{CC.OVERRIDES_ENV: js}))
    check("the unit's config overrides reach the module child: catalog_conform.DEEP_ELEV_EPOCH='mid-hour' set in the parent "
          "is the child's value, as a forwarded override", p.stdout.strip().startswith('1 mid-hour forwarded'),
          (p.stdout[-300:], p.stderr[-600:]))
    code2 = ('import sys; sys.path.insert(0, %r)\nfrom exopipe.phase3.core import env; env.legacy()\n'
             'from exopipe.phase3.lib.catalog import conform_check as CC\nfrom exopipe.phase3.config import CFG\n'
             'import os\nCC.apply_forwarded(os.environ[CC.OVERRIDES_ENV])\n'
             'print(repr({o[0]: o[2] for o in CFG.overrides}))' % REPO)
    p = subprocess.run([sys.executable, '-c', code2], capture_output=True, text=True,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', **{CC.OVERRIDES_ENV: js2}))
    got = {}
    try:
        import ast as _ast
        got = _ast.literal_eval(p.stdout.strip().splitlines()[-1])
    except Exception:
        pass
    check('the forwarded overrides are exact: a str, a float and a dict entry arrive with the parent\'s values and types',
          got == want and all(type(got[k]) is type(want[k]) for k in want), (want, got, p.stderr[-400:]))


def test_literal():
    import ast
    vals = [float('inf'), -float('inf'), (1.0, float('inf')), {('V', 'Red'): 2.0}, ((25, 35), (35, 50)), '5', None, 1e-3, True,
            [1, 2], (3,), {'deep': float('inf')}]
    back = [ast.literal_eval(CC.literal(v)) for v in vals]
    check('override text: every value type (infinities, tuples of tuples, tuple keys, a one-tuple, None, str) round-trips '
          'exactly; a NaN refuses', back == vals and all(type(a) is type(b) for a, b in zip(back, vals))
          and _raises(lambda: CC.literal(float('nan'))), (vals, back))


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
    check('restored IDs: VR_/IR_ + the catalog fields (field 2 the LST); VD_, VM_ and others are not restored rows',
          CC.is_restored('VR_05h_2024-12-18_23MHz_0007') and CC.is_restored('IR_05h_2024-12-18_23MHz_0007')
          and not any(CC.is_restored(x) for x in ('VD_05h_2024-12-18_23MHz_0007', 'VM_05h_2024-12-18_23MHz_0007',
                                                  'PR_V_05h', 'XR_05h')))


def _raises(fn):
    try:
        fn()
    except Exception:
        return True
    return False


def main():
    shutil.rmtree(T, ignore_errors=True)
    os.makedirs(T)
    keep = (CC._deps, CC.cell_base, RR.pulsar_positions)
    try:
        test_mode()
        test_literal()
        test_clean()
        test_faults()
        test_pulsar_stage()
        test_pulsar_stage_per_epoch()
        test_chain_pieces()
    finally:
        CC._deps, CC.cell_base, RR.pulsar_positions = keep
        if os.environ.get('KEEP_T') != '1':
            shutil.rmtree(T, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
