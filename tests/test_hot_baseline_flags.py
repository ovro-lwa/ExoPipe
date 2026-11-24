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
  refusals        every visibility flagged, or a FLAG column that cannot be read or has the wrong shape: the step
                  refuses (RuntimeError) and writes no flag.
  casacore_ms     the FLAG read on a real (tiny) casacore measurement set (skipped, and said so, where python-casacore is
                  not installed, as in the CI image).

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
        cols = self._cols()
        if col not in cols:
            raise RuntimeError(f'Table DataManager error: column {col} does not exist')
        end = None if nrow is None or nrow < 0 else startrow + nrow
        return np.array(cols[col][startrow:end], copy=True)

    def putcol(self, col, value, startrow=0, nrow=-1):
        assert not self.readonly, 'putcol on a table opened read only'
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


def run_step(ms_obj, workdir, label):
    """run_diagnostics end to end on a synthetic set (plots off); returns what it listed and the report text."""
    path = os.path.join(workdir, label + '.ms')
    REGISTRY[path] = ms_obj
    cap = {}
    saved = {k: getattr(HB, k) for k in ('table', 'build_complete_map', 'plot_heatmap', 'plot_diagnosis_figure',
                                         'analyze_amp_vs_uv', 'identify_bad_components', 'get_data_matrix')}
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
    args = _Args(); args.ms = path
    cwd = os.getcwd(); os.chdir(workdir)
    try:
        try:
            HB.run_diagnostics(args, HB.logging.getLogger('hot_baseline_test'))
            cap['refused'] = None
        except RuntimeError as e:
            cap['refused'] = str(e)
    finally:
        os.chdir(cwd)
        for k, v in saved.items():
            setattr(HB, k, v)
    rep = os.path.join(workdir, label + '.ms_hot_component_report.txt')
    cap['report'] = open(rep).read() if os.path.exists(rep) else None
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
        assert f'Antennas without unflagged data on any baseline (MS antenna index): {len(DEAD)}: {list(DEAD)}' \
            in after['report'], after['report']
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
        assert 'between two antennas that have data on other baselines: 2: 5-6, 7-8' in rep, rep
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
        assert 'REFUSED: no unflagged cross-correlation visibility' in cap['report'], cap['report']
        assert f'Antennas without unflagged data on any baseline (MS antenna index): {N_ANT}:' in cap['report']
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_unreadable_or_misshapen_flag_column_refuses():
    d = _tmp()
    try:
        for label, ms in (('noflag', SyntheticMS(has_flag=False)), ('badshape', SyntheticMS(flag_shape_bad=True))):
            cap = run_step(ms, d, label)
            assert cap['refused'] and 'could not be read' in cap['refused'], (label, cap['refused'])
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
        lines, any_data = HB.summarize_no_data(mean, n)
        assert any_data and any(f'{len(DEAD)}: {list(DEAD)}' in ln for ln in lines), lines
    finally:
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
