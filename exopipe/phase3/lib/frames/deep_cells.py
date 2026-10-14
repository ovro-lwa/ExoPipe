"""Read-only resolution of a deep cell's inputs, the cell-level pre-gates and the record-based epoch preview.

Nothing here writes. The authoritative gates run inside the engine (deep_search.run_deep_cell); the step evaluates the
cell-level ones here FIRST so a unit refuses before an engine process starts and so --dry-run can print what the engine
will do. Every pre-gate names the production gate it mirrors (spec plan/specs/frames.md section 5.1, D-G<n>) and the
production line it reads; none of them is new behaviour except D-G22 (x2 flows on a host that cannot compute them),
which refuses before the engine would crash its GPU call.

Inputs of a deep cell C = <co-add root>/<lst>_<scope> (spec 4.1):
  C/coadd_final/I_deep_Taper_R075_<scope>.fits, V_deep_Taper_R0_<scope>.fits    masters (phase3_deep.py:945, 997)
  C/coadd_final/loo/<date>_{I,V}_loo_Taper_{R075,R0}_<scope>.fits                LOO masters (:1397-1398)
  C/coadd_final/{weights,night_image_gate,epoch_gate,stage2_verdict}.csv,
  C/residuals_provisional.csv, C/coadd_final/PB_deep[_R075]_<scope>.fits          records and PB maps
  C/subtracted/<date>_{I,V}_sub_<scope>.fits                                      stage-1 products (the epochs)
The co-add root of a cell comes from search_inputs.ROOTS_CSV (cell,,root; p3s5_search_fleet.sh:170 reads
column 3) unless --coadd-root names one.
"""
import csv
import glob
import os
import re

# the engine's epoch list: glob *_I_sub_<scope>.fits, date by this regex (phase3_deep.py:1020-1024)
_SUB_DATE_RE = re.compile(r'(\d{4}-\d\d-\d\d(?:T\d\d)?)_I_sub_')


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def parse_epoch(item):
    """'04h_2025-10-14' -> ('04h', '2025-10-14'): a per-hour item (core.cells.parse_epoch; one LST hour of one night)."""
    from exopipe.phase3.core import cells
    return cells.parse_epoch(str(item).strip())


def expand_epochs(spec):
    """Expand --epochs (the common selector: list, @file, record/job queries) to (lst, date) pairs, in order."""
    from exopipe.phase3.core import cells
    return [cells.parse_epoch(it) for it in cells.expand_epochs(spec)]


_ROOTS = {}


def roots_table(path=None):
    """{cell: co-add root} from the co-add root table (search_inputs.ROOTS_CSV): rows 'cell,,root'."""
    if path is None:
        from exopipe.phase3.lib.frames.deep_engine import engine_values
        path = engine_values('deep')['roots_csv']
    st = os.stat(path)
    key = (path, st.st_mtime_ns, st.st_size)
    if key not in _ROOTS:
        tab = {}
        with open(path, newline='') as f:
            for row in csv.reader(f):
                if len(row) >= 3 and row[0].strip() and row[2].strip():
                    tab[row[0].strip()] = row[2].strip()
        _ROOTS.clear()
        _ROOTS[key] = tab
    return _ROOTS[key]


def coadd_root_of(cell, override=None):
    """The co-add root holding <cell>/ (the --coadd-root override, else the root table). None when unknown."""
    if override:
        return override
    try:
        return roots_table().get(cell)
    except OSError:
        return None


class Cell:
    """One deep cell's inputs (read-only)."""

    def __init__(self, cell, root, cadence='deep'):
        from exopipe.phase3.core import cells
        self.cell = cell
        self.lst, self.scope = cells.parse_cell(cell)
        self.root = root
        self.base = os.path.join(root, cell)
        self.final = os.path.join(self.base, 'coadd_final')
        self.sub_dir = os.path.join(self.base, 'subtracted')
        self.cadence = cadence
        self._hdr = {}

    # ---- paths --------------------------------------------------------------------------------------------------
    def master(self, stokes):
        if stokes == 'I':
            rob = 'R0' if self.cadence == '10min' else 'R075'       # phase3_deep.py:944
            return os.path.join(self.final, f'I_deep_Taper_{rob}_{self.scope}.fits')
        return os.path.join(self.final, f'V_deep_Taper_R0_{self.scope}.fits')          # phase3_deep.py:997

    def loo(self, date, stokes):
        var = 'Taper_R075' if stokes == 'I' else 'Taper_R0'                             # phase3_deep.py:1397-1398
        return os.path.join(self.final, 'loo', f'{date}_{stokes}_loo_{var}_{self.scope}.fits')

    def sub(self, date, stokes):
        return os.path.join(self.sub_dir, f'{date}_{stokes}_sub_{self.scope}.fits')

    def record(self, name):
        if name == 'residuals_provisional.csv':
            return os.path.join(self.base, name)
        return os.path.join(self.final, name)

    def pb_maps(self):
        return sorted(glob.glob(os.path.join(self.final, f'PB_deep*_{self.scope}.fits')))

    # ---- contents -----------------------------------------------------------------------------------------------
    def exists(self):
        return os.path.isdir(self.base)

    def epoch_dates(self):
        """The engine's epoch list (phase3_deep.py:1020-1024): dates of subtracted/*_I_sub_<scope>.fits."""
        out = []
        for c in sorted(glob.glob(os.path.join(self.sub_dir, f'*_I_sub_{self.scope}.fits'))):
            m = _SUB_DATE_RE.search(os.path.basename(c))
            if m:
                out.append(m.group(1))
        return out

    def weights(self):
        p = self.record('weights.csv')
        if not os.path.exists(p):
            return None
        with open(p, newline='') as f:
            return list(csv.DictReader(f))

    def in_master(self):
        """{date: bool} from weights.csv in_master (the build record; phase3_deep.py:1388-1392)."""
        return {r['date']: str(r.get('in_master', '')).strip().lower() in ('1', 'true') for r in (self.weights() or [])}

    def header(self, path):
        if path not in self._hdr:
            from astropy.io import fits
            self._hdr[path] = fits.getheader(path)
        return self._hdr[path]

    def input_paths(self, stokes):
        """Every file the engine reads for this (cell, Stokes): fingerprinted in the unit record (DESIGN 11.5)."""
        from exopipe.phase3.lib.masters import membership as mb
        dates = self.epoch_dates()
        paths = [self.master('I'), self.master('V')] + self.pb_maps()
        paths += [self.record(n) for n in ('weights.csv', 'night_image_gate.csv', 'epoch_gate.csv', 'stage2_verdict.csv',
                                           'residuals_provisional.csv')]
        for d in dates:
            paths += [self.sub(d, 'I'), self.sub(d, 'V'), self.loo(d, stokes)]
        paths += [mb.DEWARP_ACCEPTED, mb.SOLAR_AUDIT, mb.NAMED_OUT_OF_MASTER, _cfg().paths.DATA_DECISIONS]
        seen, out = set(), []
        for p in paths:
            if p not in seen and os.path.exists(p):
                seen.add(p)
                out.append(p)
        return out


def _gate(gid, ok, detail, kind='refuse'):
    return {'gate': gid, 'ok': bool(ok), 'detail': detail, 'kind': kind}


def pregates(c, stokes, tier='deep', date=None, work_cell=None, x2_mode='compute', host=None, excl=None):
    """Cell-level gates of the engine, evaluated before it runs, in the engine's order. Returns [gate dict]; a failing gate
    of kind 'refuse' refuses the unit (exit 2), of kind 'noinput' ends it with nothing to do (exit 3). Every gate is
    evaluated (the dry-run prints them all); the run stops at the first failure.

    date: the per-hour night (incremental unit of one night): it must be one of the cell's epochs and not excluded."""
    from exopipe.phase3.lib.frames.deep_engine import engine_values
    cfg = _cfg()
    v = engine_values(tier, cfg)
    out = []
    mI, mV = c.master('I'), c.master('V')
    if not c.exists():
        return [_gate('D-G1', False, f'no co-add cell directory {c.base} on this node (installed master missing or not '
                                     f'shipped)', 'noinput' if date else 'refuse')]
    out.append(_gate('D-G1', os.path.exists(mI), f'I master {os.path.basename(mI)}'
                     + ('' if os.path.exists(mI) else ' MISSING - inputs missing or unshipped (phase3_deep.py:945-950)')))
    gen = v['master_gen']
    # search_inputs.ALLOW_OLD_MASTERS (P3_ALLOW_OLD_MASTERS=1, an archaeology comparison key, never production) bypasses
    # the generation and membership gates in the engine (phase3_deep.py:971, 984, 1001): the pre-gates follow it
    old_ok = bool(v['allow_old_masters'])
    if os.path.exists(mI):
        opi = c.header(mI).get('COADDOP')
        opv = c.header(mV).get('COADDOP') if os.path.exists(mV) else None
        ok = old_ok or (opi == gen and (opv is None or opv == gen))
        out.append(_gate('D-G2', ok, f'COADDOP I {opi!r}' + (f', V {opv!r}' if os.path.exists(mV) else ', no V master')
                         + f' (required {gen!r}; phase3_deep.py:968-977, 997-1004)'
                         + (' - bypassed: ALLOW_OLD_MASTERS' if old_ok else '')))
    if tier == 'deep' and old_ok:
        out.append(_gate('D-G3', True, 'membership gate bypassed: ALLOW_OLD_MASTERS (phase3_deep.py:984)'))
    elif tier == 'deep':
        try:
            from exopipe.phase3.lib.masters import membership as mb
            viol = mb.verify(c.base) + mb.verify_products(c.base)
            out.append(_gate('D-G3', not viol, 'membership.verify + verify_products: '
                             + ('pass' if not viol else ' | '.join(viol[:3]) + (f' (+{len(viol) - 3} more)' if len(viol) > 3
                                                                                  else ''))))
        except Exception as e:
            out.append(_gate('D-G3', False, f'membership check could not run: {type(e).__name__}: {e}'))
    dates = c.epoch_dates()
    out.append(_gate('D-G4', bool(dates), f'{len(dates)} epochs (subtracted/*_I_sub_{c.scope}.fits)'))
    if dates:
        mn = int(v['min_epochs'][tier])
        out.append(_gate('D-G5', len(dates) >= mn, f'{len(dates)} epochs, minimum {mn} ({tier}): fewer is P3-POLICY-SKIP',
                         'noinput'))
    if date is not None:
        out.append(_gate('epoch-on-disk', date in dates,
                         f'night {date} ' + ('has its stage-1 product in this cell' if date in dates else
                                             f'has no subtracted/{date}_I_sub_{c.scope}.fits: not in this cell\'s master build '
                                             f'(per-hour stage-1 products for nights after the build are not produced yet)'),
                         'noinput'))
        if excl is not None:
            x = excl.is_excluded(c.lst, c.scope, date)
            out.append(_gate('D-G8', not x, f'night {date} ' + ('admitted' if not x else 'EXCLUDED by data_decisions'),
                             'noinput'))
    if tier == 'deep':
        from exopipe.phase3.lib.masters import night_image_gate as ng, epoch_gate as eg
        try:
            ng.failed_dates(c.final)
            out.append(_gate('D-G9', True, f'{ng.GATE_FILE} present'))
        except FileNotFoundError:
            out.append(_gate('D-G9', False, f'no {ng.GATE_FILE} in coadd_final - the night-image gate cannot run '
                                            f'(phase3_deep.py:1101-1105)'))
        try:
            eg.failed_dates(c.final, c.scope)
            out.append(_gate('D-G10', True, f'{eg.GATE_FILE} present and made against the current I master'))
        except FileNotFoundError:
            out.append(_gate('D-G10', False, f'no {eg.GATE_FILE} in coadd_final - the epoch gate cannot run '
                                             f'(phase3_deep.py:1117-1121)'))
        except RuntimeError as e:
            out.append(_gate('D-G10', False, str(e)))
    rp = c.record('residuals_provisional.csv')
    if not os.path.exists(rp):
        out.append(_gate('D-G12', False, 'residuals_provisional.csv missing - the search-tier gate cannot run '
                                         '(phase3_deep.py:1189-1191)'))
    else:
        with open(rp, newline='') as f:
            pdates = {r['date'] for r in csv.DictReader(f) if r.get('date')}
        w = c.weights()
        wdates = {r['date'] for r in w} if w is not None else None
        ok = wdates is None or pdates == wdates
        out.append(_gate('D-G12', ok, 'residuals_provisional.csv date set ' + ('equals weights.csv' if ok else
                         f'differs from weights.csv ({sorted(pdates ^ wdates)[:4]}): stale derivative (phase3_deep.py:'
                         f'1197-1199)')))
    if v['refine_x2'] and dates:
        missing = [d for d in dates if not (work_cell and os.path.exists(os.path.join(work_cell, f'ep_{d}_up2_flow.npy')))]
        gpu = host in (cfg.site.GPU_HOSTS or ())
        if not missing:
            out.append(_gate('D-G22', True, f'x2 flow cache complete ({len(dates)} flows in {work_cell})'))
        elif x2_mode == 'require':
            out.append(_gate('D-G22', False, f'{len(missing)} of {len(dates)} x2 flows missing in {work_cell} and '
                                             f'--deep-x2-flows require'))
        elif not gpu:
            out.append(_gate('D-G22', False, f'{len(missing)} of {len(dates)} x2 flows missing and {host} is not a GPU host '
                                             f'of the site file: the engine would compute them on a GPU it does not have'))
        else:
            out.append(_gate('D-G22', True, f'{len(missing)} of {len(dates)} x2 flows missing: computed by the engine on '
                                            f'this node\'s GPU under the node flow lock (every missing epoch, not only '
                                            f'the unit\'s)'))
    return out


def preview(c, stokes, excl, included=None, tier='deep'):
    """Record-based prediction of the engine's whole-epoch removals, in the engine's order (phase3_deep.py:1088-1180):
    data_decisions exclusions, night-image FAIL, epoch-gate FAIL (deep), the 10x sanity discard. The input-provenance
    and beam gates (:1257-1365) need every stage-1 header and are left to the engine. Returns a dict with the removals,
    the owed epochs (remaining, intersected with ``included``) and, per owed epoch, the master it subtracts."""
    import math
    from exopipe.phase3.lib.frames.deep_engine import engine_values
    v = engine_values(tier)
    dates = c.epoch_dates()
    rem = list(dates)
    res = {'dates': dates}
    res['excluded'] = [d for d in rem if excl.is_excluded(c.lst, c.scope, d)]
    rem = [d for d in rem if d not in res['excluded']]
    res['nightimage'], res['epochgate'] = [], []
    if tier == 'deep':
        from exopipe.phase3.lib.masters import night_image_gate as ng, epoch_gate as eg
        try:
            f = ng.failed_dates(c.final)
            res['nightimage'] = [d for d in rem if d in f]
            rem = [d for d in rem if d not in f]
        except Exception:
            pass
        try:
            f = eg.failed_dates(c.final, c.scope)
            res['epochgate'] = [d for d in rem if d in f]
            rem = [d for d in rem if d not in f]
        except Exception:
            pass
    res['sanity'] = []
    w = c.weights()
    if w is not None:
        wr = {}
        for r in w:
            try:
                q = float(r.get('qa3_rms_mjy'))
            except (TypeError, ValueError):
                continue
            wr[r['date']] = q
        fin = sorted(q for q in wr.values() if math.isfinite(q) and q > 0)
        if len(fin) >= v['sanity_min_nights']:          # deep_search.py:1263 (SANITY_MIN_NIGHTS = P3_SANITY_MIN_NIGHTS)
            med = fin[len(fin) // 2]
            res['sanity'] = [d for d in rem if wr.get(d, 0) > v['sanity_align_x'] * med]
            rem = [d for d in rem if d not in res['sanity']]
    res['remaining'] = rem
    owed = [d for d in rem if (not included or d in included)]
    res['owed'] = owed
    # the master each owed epoch subtracts (phase3_deep.py:1394-1433)
    inm = c.in_master()
    mp = c.master(stokes)
    nco = int(c.header(mp).get('NCOADD', 0) or 0) if os.path.exists(mp) else 0
    lo, hi = v['loo_ncoadd_range']
    use_loo = v['loo']
    loo = {}
    for d in owed:
        lp = c.loo(d, stokes)
        if not use_loo:
            loo[d] = 'central'
        elif not os.path.exists(lp):
            if lo <= nco <= hi and inm.get(d, False):
                loo[d] = f'LOO MISSING for an in-master epoch of an N={nco} stack: the engine REFUSES (D-G16)'
            else:
                loo[d] = 'central' + ('' if inm.get(d, False) else ' (non-member night: no LOO)')
        else:
            lh = c.header(lp)
            mh = c.header(mp)
            pd, cd = str(lh.get('PARENTDG', '') or ''), str(mh.get('CONSTDIG', '') or '')
            if pd and cd and pd != cd:
                loo[d] = f'LOO STALE (PARENTDG {pd[:8]} != CONSTDIG {cd[:8]}): the engine REFUSES (D-G16)'
            elif not pd and lh.get('COADDOP') != mh.get('COADDOP'):
                loo[d] = 'LOO STALE (legacy, COADDOP differs): the engine REFUSES (D-G16)'
            else:
                loo[d] = 'loo'
    res['loo'] = loo
    res['ncoadd'] = nco
    res['in_master'] = inm
    return res
