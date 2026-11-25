#!/usr/bin/env python3
"""STANDING TEST - the Phase 1 hot-baseline step excludes flagged visibilities from every statistic (Gregg 2026-11-11, on
the step ignoring the FLAG column: "That's a bug - make the change"), on synthetic measurement sets.

The worker (exopipe/phase1/hot_baseline_worker.py) averages each baseline's amplitude over channels and time and lists
baselines far above the others (a heat map of the cross-hand mean, and amplitude against baseline length); an antenna
with more than 25% of its baselines listed is marked bad. Before this change it ignored FLAG: antennas that were already
flagged but still held junk amplitudes (about 7 times the live data in the Stage II evidence) made about 26% of every
live antenna's baselines look hot, so nearly every antenna was marked bad.

  dead_antennas   50 antennas, 13 of them dead (junk amplitude 7, every visibility flagged; 13 of 49 = 26.5% of each
                  live antenna's baselines) and one genuinely hot live baseline (amplitude 3). With the flags honored only
                  the hot baseline is listed and no antenna is bad; the same set with FLAG cleared (what the worker
                  computed before, since it did not read FLAG) marks every live antenna bad.
  no_data         an all-flagged baseline between live antennas, holding zeros (would read cold) or junk (would read
                  hot), is NaN in every statistic (never 0), is not listed, and the report names it as no data.
  partial         flagged channels of a row and a row flagged in one correlation only leave that correlation's mean.
  refusals        every visibility flagged, no cross-hand (XY and YX) data longer than the UV cut, a FLAG column or
                  ANTENNA table that cannot be read, a FLAG column of the wrong shape, a failed FLAG write, or any other
                  failure before the plots: the step refuses (HotBaselineRefusal) naming the cause and the flag state,
                  and writes a refusal report in place of the previous run's outputs.
  flag_write      the FLAG of every row to be written is backed up first; a write that fails part way is restored and
                  the refusal says how many rows were written and restored; a failed restore is said too (never "no
                  flag was written"), and the next attempt restores the kept backup first; a failure after the write
                  restores it.
  lists           the listed baselines (antennas, statistics, FLAG rows written), the bad antennas and the FLAG rows
                  per baseline, as CSV beside the report, written after flagging with the flag state.
  plots           plots run after flagging; a failed plot is recorded in the report and does not refuse.
  cross_hand      a baseline without cross-hand data is no data for the heat-map detector: counted, named (capped)
                  and listed in full in the CSV; the report's two sections agree.
  same_numbers    the channel average (sum of usable amplitudes over their count; usable = unflagged and finite) equals
                  numpy's nanmean with flagged and non-finite values set to NaN, bit for bit.
  process_subband the production caller (process_subband.run_stage_image): a refusal stops the subband before imaging,
                  records REFUSED.json (work and archive directories) and exits with status 3.
  mnc_lookup      the MNC bad-antenna lookup (pipeline_utils.run_antenna_flagging) converts the set's start (MJD
                  seconds; a real TIME_RANGE value) to MJD days; a helper that exits with an error or prints no valid
                  record, an unreadable start time or a failed flag write refuses; an empty list from a valid record
                  passes. process_subband.run_stage_setup refuses the subband on a failed lookup (exit 3, before
                  calibration) and writes the per-set record (requested and matched MJD, gap in days, bad antennas).
  refused_used    a subband whose REFUSED.json is newer than its images is excluded by the Phase 2 collector (with its
                  reason in excluded_subbands.csv; a refusal during the wait refuses the run) and skipped by the Phase 1
                  wideband stacker; one re-imaged after its refusal is used.
  casacore_ms     the FLAG read, write and restore on a real (tiny) casacore measurement set (skipped, and said so,
                  where python-casacore is not installed, as in the CI image).

Run standalone:  python tests/test_hot_baseline_flags.py [test_name ...]
"""
import os
import shutil
import sys
import tempfile
import types

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

try:                                            # the CI image has no python-casacore; the fake table below replaces it
    import casacore.tables as _ct              # noqa: F401
    HAVE_CASACORE = True
except ImportError:
    HAVE_CASACORE = False
    _cc = types.ModuleType('casacore'); _cct = types.ModuleType('casacore.tables')
    _cct.table = None; _cc.tables = _cct
    sys.modules.setdefault('casacore', _cc); sys.modules.setdefault('casacore.tables', _cct)
for _name, _attrs in (('casatasks', ('casalog', 'concat', 'flagdata')), ('bdsf', ())):   # process_subband's imports
    try:
        __import__(_name)
    except ImportError:
        _m = types.ModuleType(_name)
        for _a in _attrs:
            setattr(_m, _a, None)
        sys.modules.setdefault(_name, _m)

from exopipe.phase1 import hot_baseline_worker as HB   # noqa: E402

N_ANT = 50
DEAD = tuple(range(37, 50))                     # 13 dead antennas: 13 of the 49 baselines of each live antenna (26.5%)
HOT = (0, 1)                                    # a genuinely hot live baseline
NCHAN, NTIME, NCORR = 8, 6, 4


# --------------------------------------------------------------------------------------------------- synthetic sets
class SyntheticMS:
    """An in-memory measurement set: rows of every baseline (autocorrelations included) for NTIME integrations."""

    def __init__(self, seed=1, dead=DEAD, hot=(HOT,), zero_bl=(), junk_bl=(), has_flag=True, flag_shape_bad=False):
        rng = np.random.default_rng(seed)
        self.pos = np.zeros((N_ANT, 3)); self.pos[:, :2] = rng.uniform(-500.0, 500.0, (N_ANT, 2))
        self.freqs = np.linspace(45.7e6, 48.3e6, NCHAN)
        self.names = np.array([f'LWA{i + 1:03d}' for i in range(N_ANT)])
        pairs = [(i, j) for _ in range(NTIME) for i in range(N_ANT) for j in range(i, N_ANT)]
        self.a1 = np.array([p[0] for p in pairs], dtype=np.int32); self.a2 = np.array([p[1] for p in pairs], dtype=np.int32)
        n = len(pairs)
        amp = 1.0 + 0.05 * rng.standard_normal((n, NCHAN, NCORR))
        amp[self.a1 == self.a2] = 100.0                                   # autocorrelations (the worker drops them)
        dead_row = np.isin(self.a1, dead) | np.isin(self.a2, dead)
        amp[dead_row & (self.a1 != self.a2)] = 7.0                        # junk on the dead antennas
        flag = np.zeros((n, NCHAN, NCORR), dtype=bool)
        flag[dead_row] = True
        for (i, j) in hot:
            amp[(self.a1 == i) & (self.a2 == j)] = 3.0
        for (i, j) in zero_bl:                                            # all flagged, zeros: would read cold
            r = (self.a1 == i) & (self.a2 == j); amp[r] = 0.0; flag[r] = True
        for (i, j) in junk_bl:                                            # all flagged, junk: would read hot
            r = (self.a1 == i) & (self.a2 == j); amp[r] = 50.0; flag[r] = True
        phase = rng.uniform(0, 2 * np.pi, (n, NCHAN, NCORR))
        self.data = (amp * np.exp(1j * phase)).astype(np.complex64)
        self.flag = flag
        self.has_flag = has_flag
        self.flag_shape_bad = flag_shape_bad
        self.putcols = []                                                 # (column, rows changed) written by the step

    def cleared(self):
        """The same set with FLAG cleared: what the worker computed before it read FLAG."""
        c = SyntheticMS.__new__(SyntheticMS); c.__dict__.update(self.__dict__)
        c.flag = np.zeros_like(self.flag); c.putcols = []
        return c


REGISTRY = {}


class FakeTable:
    """The casacore table calls the worker makes: nrows, getcol (with startrow/nrow), putcol, close."""

    def __init__(self, name, ack=False, readonly=True, **_k):
        base, _, sub = str(name).partition('::')
        self.ms = REGISTRY[base]; self.sub = sub; self.readonly = readonly

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def colnames(self):
        return list(self._cols())

    def nrows(self):
        return {'ANTENNA': N_ANT, 'SPECTRAL_WINDOW': 1}.get(self.sub, len(self.ms.a1))

    def _cols(self):
        m = self.ms
        if self.sub == 'ANTENNA':
            return {'NAME': m.names, 'POSITION': m.pos}
        if self.sub == 'SPECTRAL_WINDOW':
            return {'CHAN_FREQ': m.freqs[None, :]}
        cols = {'ANTENNA1': m.a1, 'ANTENNA2': m.a2, 'CORRECTED_DATA': m.data}
        if m.has_flag:
            cols['FLAG'] = m.flag[:, :-1, :] if m.flag_shape_bad else m.flag
        return cols

    def getcol(self, col, startrow=0, nrow=-1):
        if self.sub == 'ANTENNA' and col == 'NAME' and getattr(self.ms, 'antenna_unreadable', False):
            raise RuntimeError('Table: cannot read column NAME of ANTENNA (test)')
        cols = self._cols()
        if col not in cols:
            raise RuntimeError(f'Table DataManager error: column {col} does not exist')
        end = None if nrow is None or nrow < 0 else startrow + nrow
        return np.array(cols[col][startrow:end], copy=True)

    def putcol(self, col, value, startrow=0, nrow=-1):
        assert not self.readonly, 'putcol on a table opened read only'
        self.ms.n_putcol = getattr(self.ms, 'n_putcol', 0) + 1
        fail = getattr(self.ms, 'putcol_fails', None)   # True: every call; ('at', n): only call n; ('from', n): n on
        if fail is True or (fail and ((fail[0] == 'at' and self.ms.n_putcol == fail[1])
                                      or (fail[0] == 'from' and self.ms.n_putcol >= fail[1]))):
            raise RuntimeError(f'Table: cannot write column FLAG (test, call {self.ms.n_putcol})')
        old = self.ms.flag[startrow:startrow + len(value)]
        self.ms.putcols.append((col, int((value.any(axis=(1, 2)) & ~old.any(axis=(1, 2))).sum())))
        self.ms.flag[startrow:startrow + len(value)] = value

    def close(self):
        pass


def _identity_map(ms_id_map):
    """One hardware group, ordered by MS antenna index (as the Stage II evidence run replaced the Stage III wiring map)."""
    rows = [dict(corr_num=r, ant_num=a, name=f'LWA{a:03d}', display_name=f'LWA{a:03d}', arx_id='ARX-01', arx_chan=r,
                 snap_id_int=1, snap_id='SNAP-01') for a, r in sorted(ms_id_map.items())]
    return pd.DataFrame(rows).set_index('corr_num').sort_index()


class _Args:
    col = 'CORRECTED_DATA'; uv_cut = 0.0; uv_cut_lambda = 4.0; sigma = 5.0; uv_sigma = 7.0; threshold = 0.25
    uv_window_size = 100; run_uv = True; run_heatmap = True
    apply_antenna_flags = True; apply_baseline_flags = True


def run_step(ms_obj, workdir, label, patch=None, args_over=None):
    """run_diagnostics end to end on a synthetic set (plots off); returns what it listed and the report text.
    patch: extra worker attributes to replace for this run (e.g. a plot that fails); args_over: step arguments."""
    path = os.path.join(workdir, label + '.ms')
    REGISTRY[path] = ms_obj
    cap = {}
    patch = patch or {}
    saved = {k: getattr(HB, k) for k in ('table', 'build_complete_map', 'plot_heatmap', 'plot_diagnosis_figure',
                                         'analyze_amp_vs_uv', 'identify_bad_components', 'get_data_matrix')
             + tuple(patch)}
    _uv, _ibc, _gdm = saved['analyze_amp_vs_uv'], saved['identify_bad_components'], saved['get_data_matrix']

    def uv(*a, **k):
        r = _uv(*a, **k); cap['uv_bad'] = {tuple(sorted(map(int, b))) for b in r}; return r

    def ibc(*a, **k):
        r = _ibc(*a, **k)
        cap['hm_bad'] = {tuple(sorted(map(int, b))) for b in r[0]}; cap['bad_ants'] = set(map(int, r[1])); return r

    def gdm(*a, **k):
        r = _gdm(*a, **k); cap['raw_mean'] = {p: v.copy() for p, v in r[0].items()}; return r

    HB.table = FakeTable; HB.build_complete_map = _identity_map
    HB.plot_heatmap = lambda *a, **k: None; HB.plot_diagnosis_figure = lambda *a, **k: None
    HB.analyze_amp_vs_uv = uv; HB.identify_bad_components = ibc; HB.get_data_matrix = gdm
    for k, v in patch.items():
        setattr(HB, k, v)
    args = _Args(); args.ms = path
    for k, v in (args_over or {}).items():
        setattr(args, k, v)
    cwd = os.getcwd(); os.chdir(workdir)
    try:
        try:
            HB.run_diagnostics(args, HB.logging.getLogger('hot_baseline_test'))
            cap['refused'] = None
        except HB.HotBaselineRefusal as e:
            cap['refused'] = str(e)
    finally:
        os.chdir(cwd)
        for k, v in saved.items():
            setattr(HB, k, v)
    rep = os.path.join(workdir, label + '.ms_hot_component_report.txt')
    cap['report'] = open(rep).read() if os.path.exists(rep) else None
    csv_ = os.path.join(workdir, label + '.ms_no_data_baselines.csv')
    cap['csv'] = pd.read_csv(csv_) if os.path.exists(csv_) else None
    cap['rows_flagged'] = sum(n for _, n in ms_obj.putcols)
    return cap


def _tmp():
    return tempfile.mkdtemp(prefix='hotbl_flags_')


# --------------------------------------------------------------------------------------------------- tests
def test_dead_antennas_with_junk_flagged_do_not_make_live_antennas_hot():
    d = _tmp()
    try:
        ms = SyntheticMS()
        before = run_step(ms.cleared(), d, 'cleared')          # the statistic as the worker formed it before (no FLAG)
        live = set(range(N_ANT)) - set(DEAD)
        assert before['refused'] is None, before['refused']
        assert live <= before['bad_ants'], f"flags cleared: live antennas not marked bad: {sorted(live - before['bad_ants'])}"
        assert any(set(b) & set(DEAD) for b in before['hm_bad']), 'flags cleared: no junk baseline listed hot'

        after = run_step(ms, d, 'flagged')
        assert after['refused'] is None, after['refused']
        assert after['bad_ants'] == set(), f"flags honored: bad antennas {sorted(after['bad_ants'])}"
        assert after['hm_bad'] | after['uv_bad'] == {HOT}, f"listed: {sorted(after['hm_bad'] | after['uv_bad'])}"
        for p, m in after['raw_mean'].items():                 # a dead antenna is no data, not a value
            assert np.isnan(m[list(DEAD), :]).all() and np.isnan(m[:, list(DEAD)]).all(), p
            iu = np.triu_indices(N_ANT, k=1)
            vals = m[iu][np.isfinite(m[iu])]
            assert vals.max() < 3.5 and np.median(vals) < 1.1, (p, vals.max(), np.median(vals))
        assert f'Antennas without cross-hand data on any baseline longer than the UV cut (MS antenna index): ' \
               f'{len(DEAD)}: {list(DEAD)}' in after['report'], after['report']
        # only the hot baseline's rows are newly flagged (6 integrations); flags cleared: nearly every row
        assert after['rows_flagged'] == NTIME, after['rows_flagged']
        assert before['rows_flagged'] > 0.9 * N_ANT * (N_ANT - 1) / 2 * NTIME, before['rows_flagged']
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_all_flagged_baseline_is_no_data_not_cold_or_hot():
    d = _tmp()
    try:
        zero_bl, junk_bl = (5, 6), (7, 8)
        ms = SyntheticMS(zero_bl=(zero_bl,), junk_bl=(junk_bl,))
        before = run_step(ms.cleared(), d, 'cleared')
        assert junk_bl in before['hm_bad'] | before['uv_bad'], 'flags cleared: the junk baseline should read hot'
        assert before['raw_mean']['XY'][zero_bl] < 0.01, 'flags cleared: the zero baseline should read cold'

        after = run_step(ms, d, 'flagged')
        assert after['refused'] is None, after['refused']
        for p, m in after['raw_mean'].items():
            for (i, j) in (zero_bl, junk_bl):
                assert np.isnan(m[i, j]) and np.isnan(m[j, i]), (p, i, j, m[i, j])
        listed = after['hm_bad'] | after['uv_bad']
        assert zero_bl not in listed and junk_bl not in listed, sorted(listed)
        assert after['bad_ants'] == set(), sorted(after['bad_ants'])
        rep = after['report']
        assert 'neither hot nor cold' in rep, rep
        assert 'between two antennas that have cross-hand data on other baselines: 2: 5-6, 7-8' in rep, rep
        assert {(5, 6), (7, 8)} <= set(zip(after['csv'].ms_index_1, after['csv'].ms_index_2)), after['csv']
        n_bl = N_ANT * (N_ANT - 1) // 2
        n_dead_bl = n_bl - (N_ANT - len(DEAD)) * (N_ANT - len(DEAD) - 1) // 2
        assert f'Baselines without unflagged data in XY: {n_dead_bl + 2} of {n_bl}' in rep, rep
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_partially_flagged_rows_use_unflagged_values_only():
    d = _tmp()
    try:
        ms = SyntheticMS(dead=(), hot=())
        r = np.nonzero((ms.a1 == 10) & (ms.a2 == 11))[0]
        amp = np.abs(ms.data)
        ms.data[r[0], :NCHAN // 2, :] = 500.0                 # half the channels of one row: junk, flagged
        ms.flag[r[0], :NCHAN // 2, :] = True
        ms.data[r[1], :, 1] = 500.0                           # one row flagged in XY only (every channel)
        ms.flag[r[1], :, 1] = True
        # expected: per row, the mean of the unflagged channels; per baseline, the mean over rows with any
        exp = {}
        for ci, p in enumerate(('XX', 'XY', 'YX', 'YY')):
            vals = []
            for k, row in enumerate(r):
                ok = ~ms.flag[row, :, ci]
                if ok.any():
                    vals.append(amp[row, ok, ci].mean())
            exp[p] = np.mean(vals)
        n_rows_xy = sum(1 for row in r if (~ms.flag[row, :, 1]).any())
        assert n_rows_xy == NTIME - 1
        REGISTRY[os.path.join(d, 'partial.ms')] = ms
        saved = HB.table
        HB.table = FakeTable
        try:
            mean, std, n = HB.get_data_matrix(os.path.join(d, 'partial.ms'), 'CORRECTED_DATA')
        finally:
            HB.table = saved
        assert n == N_ANT
        for p in exp:
            assert abs(mean[p][10, 11] - exp[p]) < 1e-5 and mean[p][11, 10] == mean[p][10, 11], (p, mean[p][10, 11], exp[p])
            assert mean[p][10, 11] < 1.2, (p, mean[p][10, 11])
            assert np.isfinite(std[p][10, 11]), p
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_every_visibility_flagged_refuses_and_flags_nothing():
    d = _tmp()
    try:
        ms = SyntheticMS()
        ms.flag[:] = True
        cap = run_step(ms, d, 'allflagged')
        assert cap['refused'] and 'refused' in cap['refused'] and 'every cross-correlation visibility is flagged' \
            in cap['refused'], cap['refused']
        assert cap['rows_flagged'] == 0 and ms.putcols == [], ms.putcols
        assert 'REFUSED: every cross-correlation visibility is flagged' in cap['report'], cap['report']
        assert f'(MS antenna index): {N_ANT}:' in cap['report'], cap['report']
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_unreadable_or_misshapen_flag_column_refuses():
    d = _tmp()
    try:
        for label, ms, cause in (('noflag', SyntheticMS(has_flag=False), 'column FLAG does not exist'),
                                 ('badshape', SyntheticMS(flag_shape_bad=True), 'FLAG shape')):
            cap = run_step(ms, d, label)
            assert cap['refused'] and 'could not be read' in cap['refused'] and cause in cap['refused'], \
                (label, cap['refused'])
            assert 'REFUSED: its data and flags could not be read' in cap['report'] and cause in cap['report'], \
                (label, cap['report'])
            assert ms.putcols == [], (label, ms.putcols)
            assert 'hm_bad' not in cap and 'uv_bad' not in cap, label    # nothing listed from unflagged statistics
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_casacore_ms_flag_read():
    if not HAVE_CASACORE:
        print('    SKIPPED: python-casacore is not installed here (the fake-table tests above cover the logic)')
        return
    from casacore import tables as ct
    d = _tmp()
    try:
        src = SyntheticMS()
        path = os.path.join(d, 'tiny.ms')
        desc = ct.maketabdesc([ct.makearrcoldesc('CORRECTED_DATA', 0j, shape=[NCHAN, NCORR], valuetype='complex')])
        t = ct.default_ms(path, desc)
        t.addrows(len(src.a1))
        t.putcol('ANTENNA1', src.a1); t.putcol('ANTENNA2', src.a2)
        t.putcol('CORRECTED_DATA', src.data); t.putcol('FLAG', src.flag)
        t.close()
        a = ct.table(path + '::ANTENNA', readonly=False, ack=False)
        a.addrows(N_ANT); a.putcol('NAME', list(src.names)); a.putcol('POSITION', src.pos); a.close()
        s = ct.table(path + '::SPECTRAL_WINDOW', readonly=False, ack=False)
        s.addrows(1); s.putcell('CHAN_FREQ', 0, src.freqs); s.putcell('NUM_CHAN', 0, NCHAN); s.close()
        mean, std, n = HB.get_data_matrix(path, 'CORRECTED_DATA')
        assert n == N_ANT, n
        for p, m in mean.items():
            assert np.isnan(m[list(DEAD), :]).all() and np.isnan(m[:, list(DEAD)]).all(), p
            assert 2.5 < m[HOT] < 3.5, (p, m[HOT])
        lines, dom = HB.summarize_no_data(mean, n, src.pos, 0.0)
        assert dom['n_cross_hand'] > 0 and any(f'{len(DEAD)}: {list(DEAD)}' in ln for ln in lines), lines
        # the FLAG write and its backup on a real table: written per baseline, then restored exactly
        saved_chunk = HB.FLAG_CHUNK_ROWS
        HB.FLAG_CHUNK_ROWS = 2000
        try:
            bk = HB.flag_backup_dir(path)
            counts = HB.apply_flags_to_ms(path, {5}, [HOT], bk)
            assert counts[HOT] == NTIME and sum(counts.values()) == NTIME * (N_ANT + 1), counts   # 5: 49 + auto; HOT
            with ct.table(path, ack=False) as t:
                f = t.getcol('FLAG')
            sel = (src.a1 == 5) | (src.a2 == 5) | ((src.a1 == HOT[0]) & (src.a2 == HOT[1]))
            assert f[sel].all() and np.array_equal(f[~sel], src.flag[~sel]), 'wrong rows flagged'
            assert HB.restore_flags(path, bk) == NTIME * (N_ANT + 1) and not os.path.exists(bk)
            with ct.table(path, ack=False) as t:
                assert np.array_equal(t.getcol('FLAG'), src.flag), 'FLAG not restored'
        finally:
            HB.FLAG_CHUNK_ROWS = saved_chunk
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _cut_m(ms):
    return 4.0 * 299792458.0 / float(np.mean(ms.freqs))        # the production UV cut, 4 wavelengths


def _live_pairs_above_cut(ms, n, dead=DEAD):
    cut = _cut_m(ms)
    out = []
    for i in range(N_ANT):
        for j in range(i + 1, N_ANT):
            if i in dead or j in dead or (i, j) == HOT:
                continue
            if np.linalg.norm(ms.pos[i] - ms.pos[j]) >= cut:
                out.append((i, j))
    return out[:n]


def _domain_numbers(report):
    import re
    a = re.search(r'longer than the UV cut \(([\d.]+) m\): (\d+); of these with cross-hand data \(XY and YX; the '
                  r'heat-map detector.s data\): (\d+); without \(no data\): (\d+)', report)
    b = re.search(r'Heat-map detector: (\d+) baselines longer than the UV cut have cross-hand data; (\d+) of them '
                  r'already listed by the amplitude-against-length detector, (\d+) tested; (\d+) without cross-hand '
                  r'data are no data', report)
    assert a and b, report
    return [int(x) for x in a.groups()[1:]], [int(x) for x in b.groups()]


def test_no_cross_hand_data_refuses():
    d = _tmp()
    try:
        for label, corr in (('nocross', [1, 2]), ('noyx', [2])):
            ms = SyntheticMS(dead=(), hot=())
            ms.flag[:, :, corr] = True                       # XX and YY keep their data
            cap = run_step(ms, d, label)
            assert cap['refused'] and 'cross-hand (XY and YX)' in cap['refused'], (label, cap['refused'])
            assert 'REFUSED: no baseline longer than the UV cut' in cap['report'], (label, cap['report'])
            assert ms.putcols == [] and 'hm_bad' not in cap, label
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_baseline_without_cross_hand_data_is_no_data_in_both_sections():
    d = _tmp()
    try:
        ms = SyntheticMS(hot=())
        pairs = _live_pairs_above_cut(ms, 3)
        amp = np.abs(ms.data)
        for (i, j) in pairs:                                  # XY flagged, holding junk that would read hot
            r = (ms.a1 == i) & (ms.a2 == j)
            ms.data[r, :, 1] = 50.0; ms.flag[r, :, 1] = True
        cap = run_step(ms, d, 'xyonly')
        assert cap['refused'] is None, cap['refused']
        listed = cap['hm_bad'] | cap['uv_bad']
        assert not set(pairs) & listed, sorted(set(pairs) & listed)
        for (i, j) in pairs:
            assert np.isnan(cap['raw_mean']['XY'][i, j]) and np.isfinite(cap['raw_mean']['XX'][i, j]), (i, j)
        (n_above, n_cross, n_none), (h_cross, h_uv, h_tested, h_none) = _domain_numbers(cap['report'])
        assert n_cross + n_none == n_above and h_cross == n_cross and h_none == n_none, cap['report']
        assert h_uv + h_tested == h_cross, cap['report']
        n_dead_above = sum(1 for i in range(N_ANT) for j in range(i + 1, N_ANT)
                           if (i in DEAD or j in DEAD) and np.linalg.norm(ms.pos[i] - ms.pos[j]) >= _cut_m(ms))
        assert n_none == n_dead_above + len(pairs), (n_none, n_dead_above, len(pairs))
        named = ', '.join(f'{i}-{j}' for i, j in pairs)
        assert f'between two antennas that have cross-hand data on other baselines: {len(pairs)}: {named}' \
            in cap['report'], cap['report']
        c = cap['csv']
        assert len(c) == n_none, (len(c), n_none)
        for (i, j) in pairs:
            row = c[(c.ms_index_1 == i) & (c.ms_index_2 == j)]
            assert len(row) == 1 and int(row.has_XY.iloc[0]) == 0 and int(row.has_XX.iloc[0]) == 1 \
                and int(row.antenna_without_cross_hand_data.iloc[0]) == 0, row
        del amp
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_no_data_list_capped_full_list_in_csv():
    d = _tmp()
    try:
        ms = SyntheticMS(dead=(), hot=())
        pairs = _live_pairs_above_cut(ms, HB.NO_DATA_LIST_CAP + 10, dead=())
        assert len(pairs) == HB.NO_DATA_LIST_CAP + 10
        for (i, j) in pairs:
            ms.flag[(ms.a1 == i) & (ms.a2 == j)] = True
        cap = run_step(ms, d, 'capped')
        assert cap['refused'] is None, cap['refused']
        line = [ln for ln in cap['report'].splitlines() if 'between two antennas that have cross-hand data' in ln][0]
        assert f': {len(pairs)} (first {HB.NO_DATA_LIST_CAP} named; all in the CSV): ' in line, line
        assert len(line.split(': ')[-1].split(', ')) == HB.NO_DATA_LIST_CAP, line
        assert set(zip(cap['csv'].ms_index_1, cap['csv'].ms_index_2)) == set(pairs), cap['csv']
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_previous_outputs_removed_on_refusal_and_on_success():
    d = _tmp()
    try:
        bad = SyntheticMS(); bad.flag[:] = True
        for label, ms, refused in (('stale_refused', bad, True), ('stale_ok', SyntheticMS(), False)):
            for fn in HB.output_names(label + '.ms'):
                with open(os.path.join(d, fn), 'w') as f:
                    f.write('STALE from a previous run\n')
            cap = run_step(ms, d, label)
            assert (cap['refused'] is not None) == refused, (label, cap['refused'])
            assert 'STALE' not in cap['report'] and ('REFUSED' in cap['report']) == refused, (label, cap['report'])
            for fn in HB.output_names(label + '.ms'):     # each output is gone or this run's own
                path = os.path.join(d, fn)
                if os.path.exists(path):
                    assert b'STALE' not in open(path, 'rb').read(), (label, fn)
            if refused:
                assert not [fn for fn in HB.output_names(label + '.ms') if fn.endswith('.png')
                            and os.path.exists(os.path.join(d, fn))], label
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_plot_failure_is_recorded_and_does_not_refuse():
    """plots run after flagging; a failed plot is recorded in the report, and the lists and flags stand"""
    d = _tmp()
    try:
        def boom(*a, **k):
            raise ValueError('plot failed (test)')
        ms = SyntheticMS()
        cap = run_step(ms, d, 'plotfail', patch={'plot_heatmap': boom})
        assert cap['refused'] is None, cap['refused']
        assert '[PLOTS] These plots were not made' in cap['report'] and 'raw mean XX: ValueError: plot failed (test)' \
            in cap['report'], cap['report']
        assert cap['hm_bad'] | cap['uv_bad'] == {HOT}, (cap['hm_bad'], cap['uv_bad'])
        assert ms.flag[(ms.a1 == HOT[0]) & (ms.a2 == HOT[1])].all(), 'the hot baseline was not flagged'
        for fn in HB.list_names('plotfail.ms'):
            assert os.path.exists(os.path.join(d, fn)), fn
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_lists_written_after_flagging_with_the_flag_state():
    """the report's lists as CSV beside it: listed baselines (antennas, statistic, FLAG rows written), bad antennas and
    FLAG rows per baseline, written after the flags, with the flag state"""
    d = _tmp()
    try:
        ms = SyntheticMS()
        cap = run_step(ms, d, 'lists')
        assert cap['refused'] is None, cap['refused']
        hot = pd.read_csv(os.path.join(d, 'lists.ms_hot_baselines.csv'))
        assert list(zip(hot.ms_index_1, hot.ms_index_2)) == [HOT], hot
        r = hot.iloc[0]
        assert r.name_1 == 'LWA001' and r.name_2 == 'LWA002' and r.flag_rows_written == NTIME, r
        assert abs(r.cross_hand_mean - 3.0) < 0.1 and r.length_m > 0, r
        if r.listed_by == 'uv':                      # listed first by amplitude against length: the heat map never tested it
            assert np.isnan(r.heatmap_robust_z) and r.uv_excess_sigma > 7 and r.uv_excess_correlation in HB.POLS, r
        else:
            assert r.heatmap_robust_z > 5, r
        assert r.flag_state.startswith(f'written: {NTIME} rows'), r.flag_state
        assert len(pd.read_csv(os.path.join(d, 'lists.ms_bad_antennas.csv'))) == 0
        rows = pd.read_csv(os.path.join(d, 'lists.ms_flag_rows_written.csv'))
        assert list(zip(rows.ms_index_1, rows.ms_index_2, rows.flag_rows_written)) == [HOT + (NTIME,)], rows
        assert f'FLAG: written: {NTIME} rows' in cap['report'] and 'lists.ms_hot_baselines.csv (1 baselines)' \
            in cap['report'], cap['report']
        assert not os.path.exists(HB.flag_backup_dir(os.path.join(d, 'lists.ms'))), 'backup left after success'
        # flagging disabled: the lists say so and nothing is written
        ms2 = SyntheticMS()
        cap = run_step(ms2, d, 'noflags', args_over=dict(apply_antenna_flags=False, apply_baseline_flags=False))
        assert 'FLAG: flagging is disabled' in cap['report'], cap['report']
        hot = pd.read_csv(os.path.join(d, 'noflags.ms_hot_baselines.csv'))
        assert hot.flag_rows_written.tolist() == [0] and 'no flag was written' in hot.flag_state.iloc[0], hot
        assert ms2.putcols == [], ms2.putcols
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_flag_write_failure_restores_and_says_what_was_written():
    """a FLAG write that fails part way: the rows written so far are restored from the backup taken before writing,
    and the refusal says exactly what was written and restored; when the restore fails too, it says so (never 'no flag
    was written')"""
    d = _tmp()
    saved_chunk = HB.FLAG_CHUNK_ROWS
    try:
        HB.FLAG_CHUNK_ROWS = 2000                     # 4 chunks of the 7,650 rows: each holds rows of the hot baseline
        ms = SyntheticMS(); before = ms.flag.copy()
        ms.putcol_fails = ('at', 2)                   # the first chunk is written, the second write fails
        cap = run_step(ms, d, 'partial')
        assert cap['refused'] and 'the FLAG write failed' in cap['refused'], cap['refused']
        assert 'after 2 rows had been written' in cap['refused'] and 'were restored from the backup, so FLAG is as ' \
            'before the step' in cap['refused'], cap['refused']
        assert np.array_equal(ms.flag, before), 'FLAG not restored'
        assert 'FLAG: the FLAG write failed' in cap['report'] and 'no flag was written' not in cap['report'], cap['report']
        assert not [fn for fn in HB.list_names('partial.ms') if os.path.exists(os.path.join(d, fn))], 'lists left'
        ms = SyntheticMS(); ms.putcol_fails = True    # the first write fails, and so does its restore
        cap = run_step(ms, d, 'writefail')
        assert 'after 0 rows had been written; restoring them from the backup failed' in cap['refused'], cap['refused']
        ms = SyntheticMS()
        ms.putcol_fails = ('from', 2)                 # the restore fails too
        cap = run_step(ms, d, 'norestore')
        assert 'restoring them from the backup failed' in cap['refused'] and 'the backup is kept at' in cap['refused'] \
            and 'no flag was written' not in cap['report'], cap['refused']
        assert os.path.isdir(HB.flag_backup_dir(os.path.join(d, 'norestore.ms')))
        # the next attempt on the same set restores the kept backup before it runs, and says so
        ms.putcol_fails = None; ms.n_putcol = 0
        partial = ms.flag.copy()
        cap = run_step(ms, d, 'norestore')
        assert cap['refused'] is None and "A previous attempt's FLAG write did not finish" in cap['report'], cap
        assert not np.array_equal(partial, before)
    finally:
        HB.FLAG_CHUNK_ROWS = saved_chunk
        shutil.rmtree(d, ignore_errors=True)


def test_failure_after_the_flag_write_restores_the_flags():
    d = _tmp()
    try:
        def boom(*a, **k):
            raise OSError('disk full (test)')
        ms = SyntheticMS(); before = ms.flag.copy()
        cap = run_step(ms, d, 'late', patch={'write_lists': boom})
        assert cap['refused'] and 'the step failed: OSError: disk full (test)' in cap['refused'], cap['refused']
        assert f'{NTIME} rows were written for this step' in cap['refused'] and 'all 6 were restored from the backup' \
            in cap['refused'], cap['refused']
        assert np.array_equal(ms.flag, before), 'FLAG not restored'
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_unreadable_antenna_table_refuses():
    d = _tmp()
    try:
        ms = SyntheticMS(); ms.antenna_unreadable = True
        cap = run_step(ms, d, 'noant')
        assert cap['refused'] and 'its data and flags could not be read' in cap['refused'] \
            and 'cannot read column NAME of ANTENNA' in cap['refused'], cap['refused']
        assert ms.putcols == []
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_channel_average_equals_nanmean_bit_for_bit():
    """the sum-over-count channel average reproduces numpy nanmean with flagged values set to NaN exactly"""
    d = _tmp()
    try:
        ms = SyntheticMS(zero_bl=((5, 6),), junk_bl=((7, 8),))
        rng = np.random.default_rng(7)
        ms.flag |= rng.random(ms.flag.shape) < 0.3            # 30% of the visibilities flagged at random
        bad = rng.random(ms.flag.shape) < 0.02                 # 2% non-finite, flagged or not: no data
        ms.data[bad] = np.where(rng.random(int(bad.sum())) < 0.5, np.nan, np.inf).astype(np.complex64)
        r = (ms.a1 == 9) & (ms.a2 == 10)
        ms.data[r, :, 1] = np.nan; ms.flag[r, :, 1] = False   # unflagged but non-finite in every channel of XY
        path = os.path.join(d, 'same.ms'); REGISTRY[path] = ms
        saved = HB.table; HB.table = FakeTable
        try:
            mean, std, n = HB.get_data_matrix(path, 'CORRECTED_DATA')
        finally:
            HB.table = saved
        # the reference: the previous form (nanmean over NaN-masked amplitudes) with the same accumulation
        x = ms.a1 != ms.a2
        a1, a2, amp = ms.a1[x], ms.a2[x], np.abs(ms.data[x])
        amp[ms.flag[x] | ~np.isfinite(amp)] = np.nan
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            avg = np.nanmean(amp, axis=1)
        for k, p in enumerate(HB.POLS):
            s_ = np.zeros((N_ANT, N_ANT)); ss = np.zeros((N_ANT, N_ANT)); c = np.zeros((N_ANT, N_ANT), dtype=np.int64)
            v = np.isfinite(avg[:, k])
            np.add.at(s_, (a1[v], a2[v]), avg[v, k]); np.add.at(ss, (a1[v], a2[v]), avg[v, k] ** 2)
            np.add.at(c, (a1[v], a2[v]), 1)
            with np.errstate(invalid='ignore', divide='ignore'):
                m = s_ / c; sd = np.sqrt(ss / c - m ** 2)
            m[c == 0] = np.nan; sd[c == 0] = np.nan
            iu = np.triu_indices(N_ANT, k=1)
            m[iu[1], iu[0]] = m[iu]; sd[iu[1], iu[0]] = sd[iu]
            assert np.array_equal(mean[p], m, equal_nan=True), p
            assert np.array_equal(std[p], sd, equal_nan=True), p
        assert np.isnan(mean['XY'][9, 10]) and np.isfinite(mean['XX'][9, 10]), 'non-finite XY is not no data'
    finally:
        shutil.rmtree(d, ignore_errors=True)


class _StopAtImaging(Exception):
    pass


def _run_stage_image(ms_obj, worker_missing=False):
    """process_subband.run_stage_image from a resumed concatenation (no CASA, no wsclean: every external step is
    recorded, not run) up to the first science imaging call. Returns (exit status, commands, log text, dirs)."""
    import io
    import logging as _lg
    from exopipe.phase1 import process_subband as PS
    d = _tmp()
    work = os.path.join(d, 'work'); archive = os.path.join(d, 'archive')
    os.makedirs(os.path.join(work, 'QA'))
    concat = os.path.join(work, '46MHz_concat.ms'); os.makedirs(concat)
    REGISTRY[concat] = ms_obj
    org = types.SimpleNamespace(subband='46MHz', hot_baselines=True, lst_label='05h', run_label='Run_test',
                                start_time='2025-11-01T10:09:18', end_time='2025-11-01T11:09:08', skip_cleanup=True)
    PS.save_state(work, org)
    args = types.SimpleNamespace(mode='image', work_dir=work, resume_ms=concat, resume_science=False, subband='46MHz',
                                 run_label='Run_test', lst_label='05h', start_time='2025-11-01T10:09:18',
                                 end_time='2025-11-01T11:09:08', hot_baselines=True, science_out_dir=None)
    cmds = []

    def run_subprocess(cmd, description):
        cmds.append(description)
        if description.startswith('Imaging'):
            raise _StopAtImaging(description)
    stub = lambda *a, **k: None                                                   # noqa: E731
    saved_ps = {k: getattr(PS, k) for k in ('run_subprocess', 'table', 'hot_baseline_worker')}
    saved_u = {k: getattr(PS.utils, k) for k in ('fix_field_id', 'add_timestamps_to_images', 'analyze_snapshot_quality',
                                                 'plot_snapshot_diagnostics', 'flag_bad_integrations',
                                                 'redirect_casa_log')}
    saved_hb = {k: getattr(HB, k) for k in ('table', 'build_complete_map', 'plot_heatmap', 'plot_diagnosis_figure')}
    saved_arch = PS.config.LUSTRE_ARCHIVE_DIR
    buf = io.StringIO(); h = _lg.StreamHandler(buf); PS.logger.addHandler(h)
    cwd = os.getcwd()
    try:
        PS.run_subprocess = run_subprocess; PS.table = FakeTable
        if worker_missing:
            PS.hot_baseline_worker = None
        for k in saved_u:
            setattr(PS.utils, k, stub)
        PS.utils.analyze_snapshot_quality = lambda *a, **k: ([], [])
        HB.table = FakeTable; HB.build_complete_map = _identity_map
        HB.plot_heatmap = stub; HB.plot_diagnosis_figure = stub
        PS.config.LUSTRE_ARCHIVE_DIR = archive
        try:
            PS.run_stage_image(args)
            status = 0
        except SystemExit as e:
            status = e.code
    finally:
        os.chdir(cwd)
        PS.logger.removeHandler(h)
        for k, v in saved_ps.items():
            setattr(PS, k, v)
        for k, v in saved_u.items():
            setattr(PS.utils, k, v)
        for k, v in saved_hb.items():
            setattr(HB, k, v)
        PS.config.LUSTRE_ARCHIVE_DIR = saved_arch
    return status, cmds, buf.getvalue(), dict(root=d, work=work, archive=os.path.join(archive, '05h', '2025-11-01',
                                                                                      'Run_test', '46MHz'))


def test_process_subband_refusal_stops_the_subband_before_imaging():
    import json
    dirs = None
    try:
        # control: a good set reaches science imaging (stopped there by the test)
        status, cmds, log, dirs = _run_stage_image(SyntheticMS())
        assert any(c.startswith('Imaging') for c in cmds), cmds
        assert not os.path.exists(os.path.join(dirs['work'], 'REFUSED.json')), 'control refused'
        shutil.rmtree(dirs['root'], ignore_errors=True)
        # every visibility flagged: refused, recorded, not imaged, status 3
        bad = SyntheticMS(); bad.flag[:] = True
        status, cmds, log, dirs = _run_stage_image(bad)
        assert status == 3, (status, log)
        assert not any(c.startswith('Imaging') for c in cmds), cmds
        assert 'SUBBAND REFUSED (46MHz, hot-baseline step)' in log and 'every cross-correlation visibility is flagged' \
            in log, log
        for where in ('work', 'archive'):
            rec = json.load(open(os.path.join(dirs[where], 'REFUSED.json')))
            assert rec['status'] == 'REFUSED' and rec['step'] == 'hot-baseline step' \
                and 'every cross-correlation visibility is flagged' in rec['reason'], (where, rec)
        rep = open(os.path.join(dirs['archive'], 'QA', '46MHz_concat.ms_hot_component_report.txt')).read()
        assert 'REFUSED: every cross-correlation visibility is flagged' in rep, rep
        shutil.rmtree(dirs['root'], ignore_errors=True)
        # hot baselines enabled but the worker could not be imported: refused, not imaged
        status, cmds, log, dirs = _run_stage_image(SyntheticMS(), worker_missing=True)
        assert status == 3 and not any(c.startswith('Imaging') for c in cmds), (status, cmds)
        assert 'could not be imported' in json.load(open(os.path.join(dirs['work'], 'REFUSED.json')))['reason']
    finally:
        if dirs:
            shutil.rmtree(dirs['root'], ignore_errors=True)


# A real start time: OBSERVATION TIME_RANGE[0] of /lustre/gh/2019-11-22/concat/07_2019-11-22T073509-080450.ms (MJD
# seconds), whose name and first main-table TIME (5081124915.737229 s, the first integration's center) agree.
REAL_TIME_RANGE = (5081124909.237225, 5081124922.237234)
REAL_START_MJD = 5081124909.237225 / 86400.0        # 58809.316079... d = 2019-11-22T07:35:09.237 UTC


class _ObsTable:
    def __init__(self, path, ack=False, **_k):
        self.path = path

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def getcol(self, col):
        if 'unreadable' in self.path:
            raise RuntimeError('Table OBSERVATION does not exist (test)')
        return np.array([REAL_TIME_RANGE])


def _mnc_env(U, result=None, flagdata_fails=False):
    """Replace the MNC lookup's externals: the OBSERVATION read, the helper run (result: (status, stdout, stderr)) and
    flagdata. Returns (calls, restore)."""
    calls = {'cmd': [], 'flagdata': []}
    saved = {k: getattr(U, k) for k in ('pt', 'subprocess', 'flagdata')}

    def run(cmd, **k):
        calls['cmd'].append(cmd)
        st, out, err = result
        return types.SimpleNamespace(returncode=st, stdout=out, stderr=err)

    def flagdata(**k):
        calls['flagdata'].append(k)
        if flagdata_fails:
            raise RuntimeError('flagdata: table is locked (test)')
    U.pt = types.SimpleNamespace(table=_ObsTable)
    U.subprocess = types.SimpleNamespace(run=run)
    U.flagdata = flagdata

    def restore():
        for k, v in saved.items():
            setattr(U, k, v)
    return calls, restore


ETCD_FAIL = (1, '', 'Traceback (most recent call last):\n  ...\netcd3.exceptions.ConnectionFailedError')


def test_mnc_lookup_uses_mjd_days_and_refuses_on_any_failure():
    """the set's start (MJD seconds) is converted to MJD days for the MNC helper; a failed helper, an exit status other
    than 0, or no valid JSON record raises MncLookupFailed with its cause; an empty bad-antenna list is a success only
    when it comes from a valid record; the record carries the requested and matched MJDs and their gap"""
    from astropy.time import Time
    from exopipe.phase2 import pipeline_utils as U
    import json
    import logging as _lg
    lg = _lg.getLogger('mnc_test')
    m = U.mjd_from_ms_time(REAL_TIME_RANGE[0])
    assert abs(m - 58809.3160791) < 1e-6, m
    assert Time(m, format='mjd', scale='utc').isot == '2019-11-22T07:35:09.237', Time(m, format='mjd').isot
    good = json.dumps(dict(requested_mjd=m, data_timestamp_mjd=m - 0.25, bad_correlator_numbers=[7, 3, 7]))
    cases = [
        ('helper exits 1', ETCD_FAIL, 'exited with status 1 for MJD 58809.316079: Traceback'),
        ('no JSON', (0, 'connecting...\ndone\n', ''), 'printed no valid record'),
        ('JSON without the keys', (0, json.dumps({'bad_correlator_numbers': []}), ''), 'printed no valid record'),
        ('list is not a list', (0, json.dumps(dict(requested_mjd=m, data_timestamp_mjd=m, bad_correlator_numbers=None)),
                                ''), 'printed no valid record'),
    ]
    for label, result, want in cases:
        calls, restore = _mnc_env(U, result)
        try:
            try:
                U.run_antenna_flagging('/x/a.ms', {}, '', lg)
                raise AssertionError(f'{label}: not refused')
            except U.MncLookupFailed as e:
                assert want in str(e) and str(e).startswith('a.ms: '), (label, str(e))
            assert calls['flagdata'] == [], label
            assert float(calls['cmd'][0][-1]) == m, calls['cmd']        # MJD in days, full precision
        finally:
            restore()
    calls, restore = _mnc_env(U, (0, 'log line\n' + good + '\n', ''))
    try:
        rec = U.run_antenna_flagging('/x/a.ms', {}, '', lg)
        assert rec['bad_correlator_numbers'] == [3, 7] and rec['requested_mjd'] == m, rec
        assert abs(rec['matched_health_mjd'] - (m - 0.25)) < 1e-9 and abs(rec['gap_days'] + 0.25) < 1e-9, rec
        assert calls['flagdata'] == [dict(vis='/x/a.ms', mode='manual', antenna='3,7', flagbackup=False)], calls
    finally:
        restore()
    empty = json.dumps(dict(requested_mjd=m, data_timestamp_mjd=m, bad_correlator_numbers=[]))
    calls, restore = _mnc_env(U, (0, empty, ''))
    try:
        rec = U.run_antenna_flagging('/x/a.ms', {}, '', lg)
        assert rec['bad_correlator_numbers'] == [] and rec['gap_days'] == 0.0 and calls['flagdata'] == [], rec
    finally:
        restore()
    calls, restore = _mnc_env(U, (0, good, ''), flagdata_fails=True)
    try:
        try:
            U.run_antenna_flagging('/x/a.ms', {}, '', lg)
            raise AssertionError('flagdata failure not refused')
        except U.MncLookupFailed as e:
            assert 'flagging the bad antennas 3,7 failed' in str(e), str(e)
    finally:
        restore()
    calls, restore = _mnc_env(U, (0, good, ''))
    try:
        try:
            U.run_antenna_flagging('/x/unreadable.ms', {}, '', lg)
            raise AssertionError('unreadable start time not refused')
        except U.MncLookupFailed as e:
            assert 'start time (OBSERVATION TIME_RANGE) could not be read' in str(e) and calls['cmd'] == [], str(e)
    finally:
        restore()


def _run_stage_setup(result):
    """process_subband.run_stage_setup with two sets and the MNC helper's result replaced; stopped at calibration.
    Returns (exit status or 'calibration', log text, dirs)."""
    import io
    import logging as _lg
    from exopipe.phase1 import process_subband as PS
    d = _tmp()
    archive = os.path.join(d, 'archive')
    args = types.SimpleNamespace(mode='setup', subband='46MHz', run_label='Run_test', lst_label='07h',
                                 start_time='2019-11-22T07:35:09', end_time='2019-11-22T08:04:50', input_dir=None,
                                 bp_table='bp', xy_table='xy')
    saved_u = {k: getattr(PS.utils, k) for k in ('redirect_casa_log', 'find_archive_files_for_subband',
                                                 'copy_and_prepare_files', 'apply_calibration')}
    saved = (PS.NVME_WORK_ROOT, PS.config.LUSTRE_ARCHIVE_DIR)

    def copy(files, work_dir, sb, lg):
        return [os.path.join(work_dir, f'{k}.ms') for k in ('20191122_073509_46MHz', '20191122_073519_46MHz')]

    def cal(*a, **k):
        raise _StopAtImaging('calibration')
    buf = io.StringIO(); h = _lg.StreamHandler(buf); PS.logger.addHandler(h)
    calls, restore = _mnc_env(PS.utils, result)
    cwd = os.getcwd()
    try:
        PS.NVME_WORK_ROOT = os.path.join(d, 'fast'); PS.config.LUSTRE_ARCHIVE_DIR = archive
        PS.utils.redirect_casa_log = lambda *a, **k: None
        PS.utils.find_archive_files_for_subband = lambda *a, **k: ['raw.ms']
        PS.utils.copy_and_prepare_files = copy; PS.utils.apply_calibration = cal
        try:
            PS.run_stage_setup(args)
            status = 0
        except SystemExit as e:
            status = e.code
        except _StopAtImaging:
            status = 'calibration'
    finally:
        os.chdir(cwd)
        restore()
        PS.logger.removeHandler(h)
        for k, v in saved_u.items():
            setattr(PS.utils, k, v)
        PS.NVME_WORK_ROOT, PS.config.LUSTRE_ARCHIVE_DIR = saved
    return status, buf.getvalue(), dict(root=d, work=os.path.join(d, 'fast', '07h', '2019-11-22', 'Run_test', '46MHz'),
                                        archive=os.path.join(archive, '07h', '2019-11-22', 'Run_test', '46MHz'))


def test_setup_refuses_a_subband_whose_mnc_lookup_fails():
    """the production caller (process_subband.run_stage_setup): a failed MNC lookup refuses the subband before
    calibration (exit 3, REFUSED.json in the work and archive directories, the per-set record in QA); a valid record,
    empty or not, continues to calibration with the record written"""
    import json
    dirs = None
    try:
        status, log, dirs = _run_stage_setup(ETCD_FAIL)
        assert status == 3, (status, log)
        assert 'SUBBAND REFUSED (46MHz, MNC bad-antenna lookup)' in log and 'exited with status 1' in log, log
        for where in ('work', 'archive'):
            rec = json.load(open(os.path.join(dirs[where], 'REFUSED.json')))
            assert rec['step'] == 'MNC bad-antenna lookup' and 'ConnectionFailedError' in rec['reason'], (where, rec)
            rows = pd.read_csv(os.path.join(dirs[where], 'QA', 'mnc_bad_antennas.csv'))
            assert len(rows) == 1 and rows.status[0].startswith('REFUSED: MncLookupFailed'), rows
        shutil.rmtree(dirs['root'], ignore_errors=True)
        good = json.dumps(dict(requested_mjd=REAL_START_MJD, data_timestamp_mjd=REAL_START_MJD + 0.0125,
                               bad_correlator_numbers=[]))
        status, log, dirs = _run_stage_setup((0, good, ''))
        assert status == 'calibration', (status, log)
        assert not os.path.exists(os.path.join(dirs['work'], 'REFUSED.json'))
        rows = pd.read_csv(os.path.join(dirs['work'], 'QA', 'mnc_bad_antennas.csv'))
        assert len(rows) == 2 and (rows.status == 'no bad antenna in the record').all(), rows
        assert np.allclose(rows.requested_mjd, 58809.316079) and np.allclose(rows.gap_days, 0.0125), rows
    finally:
        if dirs:
            shutil.rmtree(dirs['root'], ignore_errors=True)


def test_phase2_excludes_a_refused_subband_with_stale_images():
    """a subband whose REFUSED.json is newer than its images (an earlier attempt's) is excluded with its reason; one
    re-imaged after the refusal, or never refused, is used"""
    import json
    from exopipe.common import exclusions as EX
    d = _tmp()
    try:
        dirs = {}
        for sb in ('41MHz', '46MHz', '50MHz', '55MHz'):
            dirs[sb] = os.path.join(d, sb); os.makedirs(os.path.join(dirs[sb], 'I', 'deep'))
        def img(sb, t):
            f = os.path.join(dirs[sb], 'I', 'deep', f'{sb}-I-Deep-Taper-Robust-0.75-image.fits')
            open(f, 'w').close(); os.utime(f, (t, t))
        def refused(sb, t):
            f = os.path.join(dirs[sb], 'REFUSED.json')
            json.dump(dict(status='REFUSED', step='hot-baseline step', reason='HotBaselineRefusal: a, b'), open(f, 'w'))
            os.utime(f, (t, t))
        img('41MHz', 1000); refused('41MHz', 2000)         # stale image beside a newer refusal: excluded
        refused('46MHz', 2000)                             # refused, no image: excluded
        refused('50MHz', 1000); img('50MHz', 2000)         # re-imaged after the refusal: used
        img('55MHz', 1000)                                 # never refused: used
        out = EX.phase1_refused_subbands(dirs)
        assert set(out) == {'41MHz', '46MHz'}, out
        assert 'hot-baseline step: HotBaselineRefusal: a; b' in out['41MHz'] and 'newer than all 1 of its images' \
            in out['41MHz'] and ',' not in out['41MHz'], out
        assert 'it has no image' in out['46MHz'], out
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _run_phase2_discovery(make, late=None):
    """phase2_controller.main on a synthetic Phase 1 tree up to its copy step (stopped there). make(sb, sb_dir) builds
    each subband directory; late: (subband, builder) that appears only after Phase 2 starts waiting. Returns (exit
    status or the subband map handed to the copy, log text, work directory)."""
    import io
    import logging as _lg
    from exopipe.phase2 import phase2_controller as PC
    from exopipe.common import exclusions as EX
    d = _tmp()
    root = os.path.join(d, 'phase1'); runs = os.path.join(root, '05h', '2025-11-01', 'Run_20251101_120000')
    sbs = [sb for sb in PC.cfg.ALL_SUBBANDS if sb not in PC.cfg.SKIP_SUBBANDS]
    for sb in sbs:
        if not late or sb != late[0]:
            make(sb, os.path.join(runs, sb))
    saved_cfg = (PC.cfg.CELERY_IMAGE_ROOT, PC.cfg.PHASE2_WORKING_DIR)
    saved = {k: getattr(PC, k) for k in ('copy_to_nvme', 'validate_subband', 'discover_subbands')}
    saved_ex = {k: getattr(EX, k) for k in ('excluded_subbands_for_night', 'dirty_gated_subbands')}
    saved_argv = sys.argv
    got = {}

    def copy(sm, wd):
        got['map'] = dict(sm)
        raise _StopAtImaging('copy')
    n_disc = [0]

    def disc(*a, **k):
        n_disc[0] += 1
        if late and n_disc[0] == 2:
            late[1](late[0], os.path.join(runs, late[0]))
        return saved['discover_subbands'](*a, **k)
    buf = io.StringIO(); h = _lg.StreamHandler(buf); PC.logger.addHandler(h)
    try:
        PC.cfg.CELERY_IMAGE_ROOT = root; PC.cfg.PHASE2_WORKING_DIR = os.path.join(d, 'phase2')
        PC.copy_to_nvme = copy; PC.validate_subband = lambda *a, **k: []; PC.discover_subbands = disc
        EX.excluded_subbands_for_night = lambda *a, **k: []; EX.dirty_gated_subbands = lambda *a, **k: {}
        sys.argv = ['phase2_controller', '--lst', '05h', '--date', '2025-11-01', '--run_label', 'Science_test',
                    '--phase1_wait_max', '5' if late else '0', '--phase1_poll', '0']
        try:
            PC.main()
            status = 0
        except SystemExit as e:
            status = e.code
        except _StopAtImaging:
            status = got['map']
    finally:
        PC.logger.removeHandler(h)
        for hh in list(PC.logger.handlers):
            if isinstance(hh, _lg.FileHandler) and hh.baseFilename.startswith(d):
                PC.logger.removeHandler(hh); hh.close()
        sys.argv = saved_argv
        PC.cfg.CELERY_IMAGE_ROOT, PC.cfg.PHASE2_WORKING_DIR = saved_cfg
        for k, v in saved.items():
            setattr(PC, k, v)
        for k, v in saved_ex.items():
            setattr(EX, k, v)
    return status, buf.getvalue(), os.path.join(d, 'phase2', '05h', '2025-11-01', 'Science_test'), d


def _sb_dir(sb, path, image_t=None, refused_t=None):
    import json
    os.makedirs(os.path.join(path, 'I', 'deep'))
    if image_t is not None:
        f = os.path.join(path, 'I', 'deep', f'{sb}-I-Deep-Taper-Robust-0.75-image.fits')
        open(f, 'w').close(); os.utime(f, (image_t, image_t))
    if refused_t is not None:
        f = os.path.join(path, 'REFUSED.json')
        json.dump(dict(status='REFUSED', step='hot-baseline step', reason='HotBaselineRefusal: test'), open(f, 'w'))
        os.utime(f, (refused_t, refused_t))


def test_phase2_controller_never_uses_a_refused_subband():
    """the Phase 2 collector (phase2_controller.main): a subband whose REFUSED.json is newer than its images is
    excluded with its reason (excluded_subbands.csv; never copied, and its band and Full are not formed); one refused
    while Phase 2 waits for Phase 1 refuses the run, naming it"""
    d = None
    try:
        def make(sb, p):
            _sb_dir(sb, p, image_t=1000, refused_t=2000 if sb == '46MHz' else None)
        status, log, work, d = _run_phase2_discovery(make)
        assert isinstance(status, dict) and '46MHz' not in status and '41MHz' in status, (status, log)
        csv_ = pd.read_csv(os.path.join(work, 'excluded_subbands.csv'))
        assert csv_.subband.tolist() == ['46MHz'] and 'Phase 1 refused it (hot-baseline step' in csv_.reason[0], csv_
        assert "Wideband bands NOT FORMED (excluded constituent): ['Full', 'Green']" in log, log
        shutil.rmtree(d, ignore_errors=True)
        # control: re-imaged after the refusal, so it is used
        def make2(sb, p):
            _sb_dir(sb, p, image_t=2000, refused_t=1000 if sb == '46MHz' else None)
        status, log, work, d = _run_phase2_discovery(make2)
        assert isinstance(status, dict) and '46MHz' in status and not os.path.exists(
            os.path.join(work, 'excluded_subbands.csv')), (status, log)
        shutil.rmtree(d, ignore_errors=True)
        # refused while Phase 2 waited for it
        status, log, work, d = _run_phase2_discovery(lambda sb, p: _sb_dir(sb, p, image_t=1000),
                                                     late=('50MHz', lambda sb, p: _sb_dir(sb, p, refused_t=2000)))
        assert status == 1 and 'REFUSED: subband 50MHz: Phase 1 refused it' in log, (status, log)
    finally:
        if d:
            shutil.rmtree(d, ignore_errors=True)


def test_phase1_stacker_skips_a_refused_subband_with_stale_images():
    """the Phase 1 wideband stacker (post_process_science.run_wideband_stacking) never reads a refused subband's
    earlier-attempt images; it records the subband in Wideband/refused_subbands.csv (as it already skips a subband
    without a V image)"""
    from exopipe.phase1 import post_process_science as PPS
    d = _tmp()
    saved = PPS.get_inner_rms
    try:
        for sb, ref in (('46MHz', 2000), ('50MHz', None)):
            p = os.path.join(d, sb)
            os.makedirs(os.path.join(p, 'V', 'deep'))
            f = os.path.join(p, 'V', 'deep', f'{sb}-V-Taper-Robust-0.75-image.fits')
            open(f, 'w').close(); os.utime(f, (1000, 1000))
            if ref:
                _sb_dir(sb, p, refused_t=ref)
        read = []
        PPS.get_inner_rms = lambda f: read.append(os.path.basename(f)) or float('nan')
        PPS.run_wideband_stacking(d, None)
        assert read == ['50MHz-V-Taper-Robust-0.75-image.fits'], read
        rows = pd.read_csv(os.path.join(d, 'Wideband', 'refused_subbands.csv'))
        assert rows.subband.tolist() == ['46MHz'] and 'Phase 1 refused it' in rows.reason[0], rows
    finally:
        PPS.get_inner_rms = saved
        shutil.rmtree(d, ignore_errors=True)


def _all_tests():
    return sorted((n, f) for n, f in globals().items() if n.startswith('test_') and callable(f))


def main(argv):
    names = argv or [n for n, _ in _all_tests()]
    fails = 0
    for n, f in _all_tests():
        if n not in names:
            continue
        try:
            f()
            print(f'PASS  {n}')
        except Exception as e:                     # noqa: BLE001 - report every failure
            fails += 1
            import traceback
            traceback.print_exc()
            print(f'FAIL  {n}: {e}')
    total = len([n for n, _ in _all_tests() if n in names])
    print(f'{total - fails}/{total} passed')
    return 1 if fails else 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
