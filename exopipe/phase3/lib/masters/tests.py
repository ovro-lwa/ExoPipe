"""Content tests of the `exopipe3 masters` product classes of the DEEP tier and the classes shared by the tiers (spec 6).
Discovered by core/validate (lib/<step>/tests*.py); the sub-LST and snapshot classes live in tests_sublst.py /
tests_snapshot.py (tier agents). Every test checks CONTENT (pixels, rows, header values against the data or against the
sibling products), never only presence. A test returns (passed, value, detail).

Classes (path = the file the step wrote; ctx from UnitRecord.add_output may add cell=, lst=, scope=):
  master             a co-add master/variant (coadd_final/<S>_deep_<taper>_<rob>[_shflux]_<scope>.fits)
  reference_master   stage-in's native un-warped reference (coadd_final_ref/{I_deep_Taper_R075,V_deep_Taper_R0}_<scope>.fits)
  loo_master         coadd_final/loo/<date>_<S>_loo_<taper>_<rob>[_shflux]_<scope>.fits
  coadd_epoch        subtracted/<date>_<S>_{epoch,sub,clean}[_shflux]_<scope>.fits
  weights_table      coadd_final/weights.csv (the final record) or <cell>/residuals_provisional.csv (the provisional record)
  membership_record  coadd_final/stage2_verdict.csv, or a legacy cell's residuals_provisional.csv (membership.verify of the
                     cell, night-gate rows, stage-2 metric)
  pb_map             coadd_final/PB_deep_<scope>.fits, PB_deep_R075_<scope>.fits
  pbcorr_sibling     coadd_final/<variant>_pbcorr[_shflux]_<scope>.fits
  vclean_master      coadd_final/V_deep_Taper_R0_clean[_shflux]_<scope>.fits
  master_generation  <cell>/gen_shflux (a directory): T1-T8 of check_rebuild.py (lib/masters/deep_verify.py)
  master_catalog     parent_catalogs/master_R075_<cell>.gaul.csv (deep) or parent_catalogs_<LST>/master_R075_<LST>_<scope>.gaul.csv
                     (sub-LST): sidecar md5 = the master's float32 data now, rows > 0, recipe, deep dewarp consistency
Cell-level checks (all variants present, LOO count per Stokes, every admitted night has its products) are cached per
directory state so a unit that registers 100 files runs them once.
"""
import csv
import glob
import hashlib
import json
import os
import re

import numpy as np

from exopipe.phase3.core.validate import content_test

_MASTER_RE = re.compile(r'^(?P<st>[IV])_deep_(?P<taper>Taper|NoTaper)_(?P<rob>R075|R0)(?P<clean>_clean)?(?P<pb>_pbcorr)?'
                        r'(?P<sh>_shflux)?(?P<md>_mdewarp)?_(?P<fl>[A-Za-z0-9]+)\.fits$')
_LOO_RE = re.compile(r'^(?P<date>\d{4}-\d{2}-\d{2})_(?P<st>[IV])_loo_(?P<taper>Taper|NoTaper)_(?P<rob>R075|R0)(?P<sh>_shflux)?'
                     r'(?P<md>_mdewarp)?_(?P<fl>[A-Za-z0-9]+)\.fits$')
_EPOCH_RE = re.compile(r'^(?P<date>\d{4}-\d{2}-\d{2})_(?P<st>[IV])_(?P<kind>clean|sub|epoch)(?P<sh>_shflux)?(?P<md>_mdewarp)?_'
                       r'(?P<fl>[A-Za-z0-9]+)\.fits$')
_CELL_RE = re.compile(r'^(\d\dh)_(\d+MHz|Red|Green|Blue|Full)$')
SURVEY_VARIANTS = ('I_deep_Taper_R075', 'V_deep_Taper_R0', 'I_deep_Taper_R0', 'I_deep_NoTaper_R0', 'I_deep_NoTaper_R075')
_CACHE = {}


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def _hdr(p):
    from astropy.io import fits
    return fits.getheader(p)


def _data(p):
    from astropy.io import fits
    return np.squeeze(fits.getdata(p))


def _dirstate(d):
    try:
        st = os.stat(d)
        return (os.path.abspath(d), st.st_mtime_ns)
    except OSError:
        return (os.path.abspath(d), None)


def _cached(key, fn):
    if key not in _CACHE:
        if len(_CACHE) > 512:
            _CACHE.clear()
        _CACHE[key] = fn()
    return _CACHE[key]


def locate(path, **ctx):
    """dict(final=<coadd_final dir>, gen=<generation dir holding coadd_final/subtracted>, cell_dir, lst, scope, token).
    The generation dir is the cell dir, or <cell>/<OUTGEN> for the scaled generation (lst/scope from the cell name above)."""
    p = os.path.abspath(path)
    d = os.path.dirname(p)
    if os.path.basename(d) == 'loo':
        d = os.path.dirname(d)
    gen = os.path.dirname(d)                      # holds coadd_final/ and subtracted/
    cell_dir = gen
    if not _CELL_RE.match(os.path.basename(cell_dir)):
        cell_dir = os.path.dirname(gen)           # <cell>/gen_shflux
    m = _CELL_RE.match(os.path.basename(cell_dir))
    lst, scope = (m.group(1), m.group(2)) if m else (ctx.get('lst'), ctx.get('scope'))
    if ctx.get('cell'):
        lst, scope = ctx['cell'].split('_', 1)
    token = _cfg().masters_fluxscale.TOKEN if '_shflux' in os.path.basename(p) else ''
    if lst is None:                               # archive layout: the LST from the PRODNAME card
        try:
            mm = re.search(r'LST(\d\d)h', str(_hdr(p).get('PRODNAME', '')))
            lst = f'{mm.group(1)}h' if mm else None
        except Exception:
            lst = None
    return dict(final=os.path.join(gen, 'coadd_final'), gen=gen, cell_dir=cell_dir, lst=lst, scope=scope, token=token)


def constituent_dates(hdr):
    """The COADD constituent dates of a master header (coadd._constituent_dates, the writer's own convention)."""
    return sorted(str(x).split()[-1] for x in hdr.get('HISTORY', []) if str(x).startswith('COADD constituent'))


def constdig(dates):
    return hashlib.md5(','.join(sorted(set(dates))).encode()).hexdigest()


def _finite_maxabs(d):
    fin = np.isfinite(d)
    mx = float(np.nanmax(np.abs(d))) if fin.any() else float('nan')
    return float(fin.mean()), mx


# =====================================================================================================================
# master
# =====================================================================================================================

@content_test('master')
def header_standard(path, **ctx):
    """The Phase-3 header standard (validate_fits_headers level 'standard') + PRODNAME, RESTFRQ, finite beam cards."""
    from exopipe.common.validate_fits_headers import validate_header
    h = _hdr(path)
    probs = []
    for k in ('PRODNAME', 'RESTFRQ', 'BMAJ', 'BMIN', 'BPA'):
        if k not in h:
            probs.append(f'{k} missing')
    for k in ('RESTFRQ', 'BMAJ', 'BMIN'):
        try:
            if not (np.isfinite(float(h[k])) and float(h[k]) > 0):
                probs.append(f'{k}={h[k]!r}')
        except (KeyError, TypeError, ValueError):
            pass
    rep = validate_header(path, level='standard')
    if not rep.ok:
        probs += [f'standard: {e}' for e in rep.errors[:3]]
    return not probs, len(probs), '; '.join(probs) or 'header standard, PRODNAME, RESTFRQ, beam present'


@content_test('master')
def coaddop_v3(path, **ctx):
    """COADDOP = masters_build.COADDOP (ivw-mean-v3) and FLUXFIT = flat-dc (LANDED-FLAT-DC-FLUX-SCALE-V3, R-TILT-REMOVED-NEXTGEN)."""
    b = _cfg().masters_build
    h = _hdr(path)
    ok = h.get('COADDOP') == b.COADDOP and h.get('FLUXFIT') == b.FLUXFIT
    return ok, h.get('COADDOP'), f"COADDOP {h.get('COADDOP')!r} FLUXFIT {h.get('FLUXFIT')!r} (want {b.COADDOP}, {b.FLUXFIT})"


@content_test('master')
def constdig_matches_history(path, **ctx):
    """CONSTDIG = md5 of the sorted 'COADD constituent' HISTORY dates; NCOADD = their count (coadd_full_pipeline.py:1336-1343)."""
    h = _hdr(path)
    dates = constituent_dates(h)
    ok = bool(dates) and h.get('CONSTDIG') == constdig(dates) and int(h.get('NCOADD') or -1) == len(dates)
    return ok, len(dates), f"{len(dates)} constituents; CONSTDIG {'matches' if h.get('CONSTDIG') == constdig(dates) else 'DIFFERS'}; NCOADD {h.get('NCOADD')}"


@content_test('master')
def no_excluded_constituent(path, **ctx):
    """No constituent night is excluded under data_decisions.csv re-read now (G-M02; core.exclusions)."""
    from exopipe.phase3.core import exclusions
    loc = locate(path, **ctx)
    if not loc['lst'] or not loc['scope']:
        return False, None, 'cannot tell the cell (LST, scope) of this file'
    dates = constituent_dates(_hdr(path))
    bad = exclusions.load().excluded_constituents(loc['lst'], loc['scope'], dates)
    return not bad, len(bad), f'{len(dates)} constituents of {loc["lst"]} {loc["scope"]}; excluded: {bad[:5]}'


def _integrity(path):
    lim = _cfg().masters_verify.INTEGRITY
    fin, mx = _finite_maxabs(_data(path))
    ok = fin >= lim['min_finite'] and mx < lim['max_abs_jy']
    return ok, round(mx, 4) if np.isfinite(mx) else None, f'finite fraction {fin:.3f} (>= {lim["min_finite"]}); max |pixel| {mx:.4g} Jy/beam (< {lim["max_abs_jy"]:g})'


@content_test('master')
def integrity(path, **ctx):
    """|max| < 1e5 Jy/beam and >= 50 per cent finite (verify_master_generation.py:35, 60-98)."""
    return _integrity(path)


@content_test('master')
def all_variants_present(path, **ctx):
    """Cell level: the five I/V survey variants exist beside it with the current COADDOP (LANDED-VERIFIER-ALL-VARIANTS-AND-VCLEAN)."""
    loc = locate(path, **ctx)

    def f():
        miss, wrong = [], []
        for v in SURVEY_VARIANTS:
            p = os.path.join(loc['final'], f'{v}{loc["token"]}_{loc["scope"]}.fits')
            if not os.path.exists(p):
                miss.append(v)
            elif _hdr(p).get('COADDOP') != _cfg().masters_build.COADDOP:
                wrong.append(v)
        return (not miss and not wrong), 5 - len(miss), f'missing {miss}; other COADDOP {wrong}'
    return _cached(('variants',) + _dirstate(loc['final']) + (loc['token'],), f)


# =====================================================================================================================
# reference_master (stage-in)
# =====================================================================================================================

@content_test('reference_master')
def native_unwarped(path, **ctx):
    """No SH*/FLUX*/MDWARP cards; HISTORY names the reconstruction branch (rebuild_cell.sh:74-114)."""
    h = _hdr(path)
    bad = [k for k in h.keys() if k.startswith(tuple(_cfg().masters_stagein.STRIP_CARD_PREFIXES))] + (['MDWARP'] if h.get('MDWARP') else [])
    hist = [str(x) for x in h.get('HISTORY', []) if 'rebuild reference' in str(x)]
    ok = not bad and bool(hist)
    return ok, len(bad), f"cards {bad[:5]}; branch: {hist[-1][:120] if hist else 'NO reconstruction HISTORY'}"


@content_test('reference_master')
def constdig_equals_previous(path, **ctx):
    """CONSTDIG = the md5 of its own constituents, and, when the unit names the previous (current) master (ctx previous=),
    = its CONSTDIG (same constituents) or = the CONSTDIG in its ALIGNREF card (ctx generation='previous-generation': the
    master it replaced; Gregg (A), e5c218e)."""
    h = _hdr(path)
    dates = constituent_dates(h)
    own = h.get('CONSTDIG') == constdig(dates) if dates else False
    prev = ctx.get('previous')
    hp = _hdr(prev) if prev and os.path.exists(prev) else None
    if hp is None:
        pv = None
    elif ctx.get('generation') == 'previous-generation':
        from exopipe.phase3.lib.masters.deep_stagein import alignref_digest
        pv = alignref_digest(hp)
    else:
        pv = hp.get('CONSTDIG')
    ok = own and (pv is None or pv == h.get('CONSTDIG'))
    return ok, h.get('CONSTDIG'), (f'own constituents {"match" if own else "DIFFER"}; previous master '
                                   + (f'CONSTDIG {pv} {"=" if pv == h.get("CONSTDIG") else "!="}' if pv else 'not given'))


@content_test('reference_master')
def pixels_source_over_factor(path, **ctx):
    """Pixels = the source divided by its pixel factor with production's own arithmetic, (float64 / f) -> float32, bit for
    bit (NaN where the source is NaN), when the unit names the source (ctx source=, factor=; rebuild_cell.sh:73, 90, 110;
    branch 3 copies float32 as it is); always: finite fraction >= 0.5 and |max| < 1e5 Jy/beam."""
    ok, val, det = _integrity(path)
    src, f = ctx.get('source'), ctx.get('factor')
    if src and os.path.exists(src):
        d = _data(path)
        s = _data(src)
        want = (s.astype(np.float64) / float(f)).astype(np.float32) if f else s.astype(np.float32)
        same_mask = bool(np.array_equal(np.isfinite(d), np.isfinite(want)))
        k = np.isfinite(want)
        nbad = int((d[k] != want[k]).sum()) if same_mask else -1
        good = same_mask and nbad == 0
        return ok and good, nbad, det + f'; vs source / {f}: NaN mask equal {same_mask}, {nbad} pixel(s) differ (exact)'
    return ok, val, det + '; source not given (alone: integrity only)'


# =====================================================================================================================
# loo_master
# =====================================================================================================================

def _loo_parent(path, loc):
    m = _LOO_RE.match(os.path.basename(path))
    if not m:
        return None, m
    g = m.groupdict()
    return os.path.join(loc['final'], f"{g['st']}_deep_{g['taper']}_{g['rob']}{loc['token']}_{loc['scope']}.fits"), m


@content_test('loo_master')
def parentdg(path, **ctx):
    """PARENTDG = the parent master's CONSTDIG; NCOADD = parent NCOADD - 1; LOOEXCL = the date in the name, a parent
    constituent (LANDED-LOO-PARENT-DIGEST-REFUSAL; FIX-STALE-LOO-MASTERS-NOT-PURGED)."""
    loc = locate(path, **ctx)
    parent, m = _loo_parent(path, loc)
    if parent is None or not os.path.exists(parent):
        return False, None, f'parent master {parent} missing'
    h, ph = _hdr(path), _hdr(parent)
    date = m.group('date')
    probs = []
    if h.get('PARENTDG') != str(ph.get('CONSTDIG', ''))[:32]:
        probs.append(f"PARENTDG {h.get('PARENTDG')} != parent CONSTDIG {ph.get('CONSTDIG')}")
    if int(h.get('NCOADD') or -1) != int(ph.get('NCOADD') or 0) - 1:
        probs.append(f"NCOADD {h.get('NCOADD')} != parent {ph.get('NCOADD')} - 1")
    if h.get('LOOEXCL') != date:
        probs.append(f"LOOEXCL {h.get('LOOEXCL')} != {date}")
    if date not in constituent_dates(ph):
        probs.append(f'{date} is not a constituent of the parent (stale LOO)')
    return not probs, len(probs), '; '.join(probs) or f'derivative of {os.path.basename(parent)} excluding {date}'


@content_test('loo_master')
def count_per_stokes(path, **ctx):
    """Cell level, per Stokes: LOO count = the parent master's NCOADD when 2 <= N <= 20, none otherwise (R-LOO-PER-STOKES-RATIFIED)."""
    loc = locate(path, **ctx)
    parent, m = _loo_parent(path, loc)
    if parent is None or not os.path.exists(parent):
        return False, None, 'parent master missing'
    lo, hi = _cfg().masters_build.LOO_N_RANGE

    def f():
        n = int(_hdr(parent).get('NCOADD') or 0)
        g = m.groupdict()
        have = glob.glob(os.path.join(loc['final'], 'loo', f"*_{g['st']}_loo_{g['taper']}_{g['rob']}{loc['token']}_{loc['scope']}.fits"))
        want = n if lo <= n <= hi else 0
        return len(have) == want, len(have), f"Stokes {g['st']}: {len(have)} LOO masters, parent NCOADD {n} -> {want} expected"
    return _cached(('loocount', parent) + _dirstate(os.path.join(loc['final'], 'loo')), f)


# =====================================================================================================================
# coadd_epoch
# =====================================================================================================================

def _weights_rows(final_dir):
    p = os.path.join(final_dir, 'weights.csv')
    if not os.path.exists(p):
        return None
    return {r['date']: r for r in csv.DictReader(open(p))}


def _weights_dir_for(loc):
    """weights.csv lives in the cell's coadd_final/ (also for the scaled generation's epochs)."""
    return os.path.join(loc['cell_dir'], 'coadd_final')


@content_test('coadd_epoch')
def identity_and_ratio(path, **ctx):
    """epoch - residual - master: robust rms < 1e-3 x the master's; for an in-master night also Stokes I median epoch/master
    at >= 20 compact master peaks in 0.8-1.25 (coadd._epoch_content_arrays; R-EPOCH-CONTENT-GATE-REFUSES-NIGHT...).
    A sub/clean file is tested through its night's triple (computed once)."""
    from exopipe.phase3.core import fitsio
    m = _EPOCH_RE.match(os.path.basename(path))
    if not m:
        return False, None, 'not an epoch/sub/clean file name'
    loc = locate(path, **ctx)
    g = m.groupdict()
    sd = os.path.dirname(os.path.abspath(path))
    tok = loc['token']
    e = os.path.join(sd, f"{g['date']}_{g['st']}_epoch{tok}_{loc['scope']}.fits")
    s_ = os.path.join(sd, f"{g['date']}_{g['st']}_sub{tok}_{loc['scope']}.fits")
    mvar = 'I_deep_Taper_R075' if g['st'] == 'I' else 'V_deep_Taper_R0'
    mp = os.path.join(loc['final'], f'{mvar}{tok}_{loc["scope"]}.fits')
    if g['kind'] == 'clean':
        # the clean product = residual - striation model: same grid, beam and time block as its residual; finite where it is
        d = _data(path)
        r = _data(s_) if os.path.exists(s_) else None
        if r is None or r.shape != d.shape:
            return False, None, f'no residual {os.path.basename(s_)} of the same shape'
        hc, hr = _hdr(path), _hdr(s_)
        same = all(hc.get(k) == hr.get(k) for k in ('BMAJ', 'BMIN', 'BPA', 'RESTFRQ', 'DATE-OBS', 'DATE-END'))
        nan_ok = int((np.isfinite(d) & ~np.isfinite(r)).sum()) == 0
        return same and nan_ok, None, f'clean vs residual: beam/time cards equal {same}; no finite pixel where the residual is NaN {nan_ok}'
    if not (os.path.exists(e) and os.path.exists(mp)):
        return False, None, f'missing {os.path.basename(e) if not os.path.exists(e) else os.path.basename(mp)}'
    W = _weights_rows(_weights_dir_for(loc)) or {}
    member = W.get(g['date'], {}).get('in_master') == '1'

    def f():
        from exopipe.phase3.lib.masters import coadd as CFP
        mdat = np.asarray(fitsio.read_image(mp)[0], dtype=np.float64)       # one read per master (fitsio cache)
        ed = _data(e).astype(np.float64)
        rd = _data(s_).astype(np.float64) if os.path.exists(s_) else None
        ok, why, med, n = CFP._epoch_content_arrays(ed, mdat, rd, g['st'] if member else 'V')
        return ok, (round(med, 4) if med is not None else None), (why + ('' if member else ' (not in the master: identity only)'))
    return _cached(('epoch', e, os.path.getmtime(e), mp, os.path.getmtime(mp), member), f)


@content_test('coadd_epoch')
def header_own_beam(path, **ctx):
    """BEAMSRC = 'epoch'; BMAJ/BMIN/BPA/RESTFRQ finite and named from the night's own Phase-2 source (HISTORY
    'Beam+RESTFRQ (S) from per-epoch source <file>'); when that file is found under the night's Science dir (weights.csv
    sci_dir) its values must equal the product's (R1 amendment #1; coadd.validate_epoch_residual_header rule)."""
    m = _EPOCH_RE.match(os.path.basename(path))
    if not m:
        return False, None, 'not an epoch/sub/clean file name'
    h = _hdr(path)
    probs = []
    if h.get('BEAMSRC') != 'epoch':
        probs.append(f"BEAMSRC {h.get('BEAMSRC')!r}")
    vals = {}
    for k in ('BMAJ', 'BMIN', 'BPA', 'RESTFRQ'):
        try:
            vals[k] = float(h[k])
            if not np.isfinite(vals[k]):
                probs.append(f'{k} not finite')
        except (KeyError, TypeError, ValueError):
            probs.append(f'{k} missing')
    hist = ''.join(str(x) for x in h.get('HISTORY', []))
    mm = re.search(r'Beam\+RESTFRQ \(' + m.group('st') + r'\) from per-epoch source (\S+?\.fits)', hist)
    if not mm:
        return False, len(probs) + 1, '; '.join(probs + ['no HISTORY naming the per-epoch source'])
    srcname = mm.group(1)
    loc = locate(path, **ctx)
    W = _weights_rows(_weights_dir_for(loc)) or {}
    sci = W.get(m.group('date'), {}).get('sci_dir')
    checked = 'source not located (weights.csv sci_dir)'
    if sci and os.path.isdir(sci):
        hits = glob.glob(os.path.join(sci, '*', '*', '*', srcname)) + glob.glob(os.path.join(sci, '*', '*', srcname)) \
            + glob.glob(os.path.join(sci, '*', srcname))
        if hits:
            sh = _hdr(hits[0])
            diff = [k for k in vals if not abs(vals[k] - float(sh.get(k, np.nan))) <= 1e-9 * max(abs(vals[k]), 1e-300)]
            if diff:
                probs.append(f'{"/".join(diff)} differ from the own source {hits[0]}')
            checked = f'equal to own source {os.path.basename(hits[0])}' if not diff else 'source differs'
    return not probs, len(probs), '; '.join(probs) or f'own beam ({checked})'


@content_test('coadd_epoch')
def every_admitted_night_has_products(path, **ctx):
    """Cell level: epoch, sub and clean exist for both Stokes for every non-excluded weights.csv night (two-tier products,
    R-HYS1 rider; INCIDENT-FIRST-INSTALL-01H-18MHZ...)."""
    from exopipe.phase3.core import exclusions
    loc = locate(path, **ctx)
    sd = os.path.dirname(os.path.abspath(path))

    def f():
        W = _weights_rows(_weights_dir_for(loc))
        if W is None:
            return False, None, 'no weights.csv for the cell'
        ex = exclusions.load()
        miss = []
        nights = [d for d in sorted(W) if not ex.is_excluded(loc['lst'], loc['scope'], d)]
        refused_v = _refused_v_nights(loc['cell_dir'])
        for d in nights:
            for st in ('I', 'V'):
                if st == 'V' and d in refused_v:
                    continue
                for kind in ('epoch', 'sub', 'clean'):
                    if not os.path.exists(os.path.join(sd, f'{d}_{st}_{kind}{loc["token"]}_{loc["scope"]}.fits')):
                        miss.append(f'{d}_{st}_{kind}')
        return not miss, len(nights), (f'{len(nights)} admitted nights; missing {miss[:6]}'
                                       + (f'; V input refused (B-03 same-state) on {sorted(refused_v)}' if refused_v else ''))
    return _cached(('nightprods',) + _dirstate(sd) + (loc['token'],) + _filestate(os.path.join(loc['cell_dir'], 'refusals_input.csv')), f)


def _filestate(p):
    try:
        st = os.stat(p)
        return (os.path.abspath(p), st.st_mtime_ns, st.st_size)
    except OSError:
        return (os.path.abspath(p), None, None)


def _refused_v_nights(cell_dir):
    """Nights whose V input the B-03 same-state rule refused (<cell>/refusals_input.csv): no V epoch / residual is made."""
    from exopipe.phase3.lib.masters import input_resolver
    return {r['date'] for r in input_resolver.read_refusals(cell_dir) if r.get('product') == 'V'}


# =====================================================================================================================
# build_stats (D-ADMIT slice 1)
# =====================================================================================================================

@content_test('build_stats')
def build_stats_rederived(path, **ctx):
    """<cell>/coadd_final/build_stats.json: every number re-derived from the files it names - the stage-2, search-tier and
    sanity statistics from the build's records, the beam areas from the images it names (read again) - and the build
    record's own stage-2 thresholds equal to the re-derived high thresholds."""
    from exopipe.phase3.lib.masters import build_stats as BS
    c = _cfg()
    probs = BS.recheck(path, c, (float(c.masters_build.OUTLIER_SIGMA), float(c.masters_build.LOW_OUTLIER_SIGMA)))
    return not probs, len(probs), '; '.join(probs[:4]) or 'every number re-derived from its sources'


# =====================================================================================================================
# input_refusals (B-03 same-state)
# =====================================================================================================================

@content_test('input_refusals')
def refusals_hold(path, **ctx):
    """<cell>/refusals_input.csv: the header, no (night, product) twice, and every row still holds on the Phase 2 tree - the
    named image is the one the finder chooses for the night and product, and its dewarp state differs from the night's
    survey image (core.inputs.state_mismatch)."""
    from exopipe.phase3.core import inputs as I
    from exopipe.phase3.lib.masters import input_resolver
    with open(path, newline='') as f:
        rd = csv.DictReader(f)
        cols, rows = rd.fieldnames, list(rd)
    if cols != input_resolver.REFUSALS_HEADER:
        return False, None, f'columns {cols} != {input_resolver.REFUSALS_HEADER}'
    probs, seen = [], set()
    for r in rows:
        key = (r['date'], r['product'])
        if key in seen:
            probs.append(f'{key} twice')
        seen.add(key)
        sci = I.latest_science_dir(r['lst'], r['date'])
        fq = r['scope'][:-3] if r['scope'].endswith('MHz') else r['scope']
        prod = 'V' if r['product'] == 'V' else r['product'].split('_', 1)[1]
        p = I.product_image(sci, fq, prod) if sci else None
        if p is None or os.path.basename(p) != r['image']:
            probs.append(f"{r['date']} {r['product']}: the finder now chooses {os.path.basename(p) if p else None}, "
                         f"not {r['image']}")
        elif not I.state_mismatch(sci, fq, p, r['product']):
            probs.append(f"{r['date']} {r['product']}: {r['image']} is now in its survey image's state")
    return not probs, len(rows), f'{len(rows)} refusals' + (f'; {probs[:4]}' if probs else ' hold on the Phase 2 tree')


# =====================================================================================================================
# weights_table, membership_record
# =====================================================================================================================

WEIGHTS_COLUMNS = ('date', 'weight', 'qa3_scale', 'qa3_tilt', 'qa3_nsrc', 'qa3_guard', 'qa3_rms_mjy', 'sci_dir', 'in_master',
                   'flux_dc', 'qa3_window')
PROVISIONAL_COLUMNS = ('date', 'i_rms_mjy', 'v_rms_mjy', 'stage2_metric', 'i_outlier', 'v_outlier', 'beam_outlier', 'outlier')


@content_test('weights_table')
def columns_and_rows(path, **ctx):
    """weights.csv: its columns, >= 1 row, every in-master night has flux_dc (T2 part), dates = residuals_provisional.csv
    dates (T7). residuals_provisional.csv: its columns, metric = the ruled stage-2 metric, finite Stokes I rms for every
    non-outlier night."""
    rows = list(csv.DictReader(open(path)))
    cols = set(rows[0].keys()) if rows else set()
    base = os.path.basename(path)
    if base == 'residuals_provisional.csv':
        miss = [c for c in PROVISIONAL_COLUMNS if c not in cols]
        metric = _cfg().masters_build.STAGE2_METRIC
        other = sorted({r.get('stage2_metric') for r in rows} - {metric})
        nonfin = [r['date'] for r in rows if r.get('outlier') not in ('True', '1')
                  and not np.isfinite(float(r.get('i_rms_mjy') or 'nan'))]
        ok = bool(rows) and not miss and not other and not nonfin
        return ok, len(rows), f'{len(rows)} nights; missing columns {miss}; other metrics {other}; non-finite I rms {nonfin}'
    miss = [c for c in WEIGHTS_COLUMNS if c not in cols]
    nodc = [r['date'] for r in rows if r.get('in_master') == '1' and not (r.get('flux_dc') or '').strip()]
    rp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(path))), 'residuals_provisional.csv')
    t7 = 'no residuals_provisional.csv beside the cell'
    t7ok = False
    if os.path.exists(rp):
        a = {r['date'] for r in csv.DictReader(open(rp))}
        b = {r['date'] for r in rows}
        t7ok = a == b
        t7 = f'dates equal residuals_provisional.csv: {t7ok} (differ {sorted(a ^ b)[:4]})'
    ok = bool(rows) and not miss and not nodc and t7ok
    return ok, len(rows), f'{len(rows)} nights, {sum(1 for r in rows if r.get("in_master") == "1")} in master; missing columns {miss}; in-master without flux_dc {nodc}; {t7}'


def _record_dirs(path):
    """(cell_dir, coadd_final dir) of a membership record: <cell>/coadd_final/stage2_verdict.csv, or - for a master built
    before 2026-10-08 (no verdict file; B-04 default (c), masters_build.LEGACY_STAGE2_RECORD) - <cell>/residuals_provisional.csv."""
    p = os.path.abspath(path)
    if os.path.basename(p) == 'residuals_provisional.csv':
        return os.path.dirname(p), os.path.join(os.path.dirname(p), 'coadd_final')
    return os.path.dirname(os.path.dirname(p)), os.path.dirname(p)


@content_test('membership_record')
def verify(path, **ctx):
    """membership.verify(cell) returns no violation (incl. nights absent from the record) - the final stage's own rule
    (coadd_full_pipeline.py:3800-3805; IMPL-MEMBERSHIP-VERIFY-NIGHTS-ABSENT-FROM-RECORD)."""
    from exopipe.phase3.lib.masters import membership as MB
    cell_dir, _ = _record_dirs(path)
    if not _CELL_RE.match(os.path.basename(cell_dir)):
        return False, None, f'{cell_dir}: not a <LST>_<scope> cell directory (membership.verify keys the cell by its name)'
    v = MB.verify(cell_dir)
    return not v, len(v), f'{len(v)} violations {v[:3]}'


@content_test('membership_record')
def night_gate_rows(path, **ctx):
    """A night_image_gate.csv row for every weights.csv night; no FAIL night in the master (T2 part)."""
    _, fd = _record_dirs(path)
    W = _weights_rows(fd)
    g = os.path.join(fd, 'night_image_gate.csv')
    if W is None or not os.path.exists(g):
        return False, None, 'weights.csv or night_image_gate.csv missing'
    G = {r['date']: r for r in csv.DictReader(open(g))}
    miss = sorted(set(W) - set(G))
    fin = sorted(d for d, r in G.items() if r.get('verdict') == 'FAIL' and W.get(d, {}).get('in_master') == '1')
    return not (miss or fin), len(G), f'{len(G)} gate rows; missing {miss}; FAIL in master {fin}'


@content_test('membership_record')
def stage2_metric(path, **ctx):
    """Every row carries the ruled metric (nonmatched-provisional-v1) and its outlier flags equal the production rule
    re-applied to the recorded values (membership.stage2_outliers). stage2_verdict.csv: column 'metric' must equal it. A
    legacy residuals_provisional.csv (builds before 2026-10-07 wrote no metric column): column 'stage2_metric' may be empty,
    never another metric - accepted while masters_build.LEGACY_STAGE2_RECORD = 'accept-residuals_provisional' (B-04 (c);
    membership.stage2_record applies the same rule)."""
    from exopipe.phase3.lib.masters import membership as MB
    if not os.path.exists(path):
        return False, None, f'{os.path.basename(path)} missing'
    rows = list(csv.DictReader(open(path)))
    metric = _cfg().masters_build.STAGE2_METRIC
    legacy = os.path.basename(path) == 'residuals_provisional.csv'
    if legacy:
        if _cfg().masters_build.LEGACY_STAGE2_RECORD != 'accept-residuals_provisional':
            return False, len(rows), 'legacy stage-2 record not accepted (masters_build.LEGACY_STAGE2_RECORD)'
        other = sorted({r.get('stage2_metric') for r in rows} - {None, '', metric})
    else:
        other = sorted({r.get('metric') for r in rows} - {metric})
    if not rows or other:
        return False, len(rows), f'{len(rows)} rows; other metrics {other}'
    dates = [r['date'] for r in rows]
    iv = [MB._num(r.get('i_rms_mjy')) for r in rows]
    vv = [MB._num(r.get('v_rms_mjy')) for r in rows]
    beam = {r['date'] for r in rows if MB._flag(r.get('beam_outlier'))}
    _, _, _, _, out = MB.stage2_outliers(dates, iv, vv, beam)
    out |= {d for d, x in zip(dates, iv) if not np.isfinite(x)}
    dis = [r['date'] for r in rows if MB._flag(r.get('outlier')) != (r['date'] in out)]
    return not dis, len(rows), (f'{len(rows)} rows, metric {metric}' + (' (legacy residuals_provisional record)' if legacy
                                                                        else '') + f'; flags disagreeing with the rule {dis[:4]}')


# =====================================================================================================================
# pb_map, pbcorr_sibling, vclean_master
# =====================================================================================================================

@content_test('pb_map')
def both_maps(path, **ctx):
    """Both PB maps of the cell exist: PB_deep_<scope> (Robust 0) and PB_deep_R075_<scope> (R-FLUX-SCALE-REMEASURE-...-PB-STEP2)."""
    loc = locate(path, **ctx)
    names = [n.format(scope=loc['scope']) for n in _cfg().masters_pbmaps.PB_MAP_NAMES.values()]
    miss = [n for n in names if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(path)), n))]
    return not miss, 2 - len(miss), f'missing {miss}' if miss else 'both PB maps present'


@content_test('pb_map')
def three_factor_cards(path, **ctx):
    """PBLIBV, PBMODEL, PBCHROM (bs-sqbp-v1), PBRESID, PBROBUST and PBBMAJ = the beam source's BMAJ (the I Taper master of
    the map's robust weighting) (phase3_pbcorr_masters.py:76-99; LANDED-THREE-FACTOR-PB-MODEL-CHROMATIC-RESIDUAL)."""
    s = _cfg().masters_pbmaps
    h = _hdr(path)
    probs = [k for k in ('PBLIBV', 'PBMODEL', 'PBCHROM', 'PBRESID', 'PBROBUST', 'PBBMAJ') if k not in h]
    if h.get('PBCHROM') not in (None, s.PB_CHROM_VERSION):
        probs.append(f"PBCHROM {h.get('PBCHROM')!r}")
    if h.get('PBLIBV') not in (None, s.PBLIBV):
        probs.append(f"PBLIBV {h.get('PBLIBV')!r}")
    loc = locate(path, **ctx)
    rob = 'R075' if '_R075_' in os.path.basename(path) else 'R0'
    if h.get('PBROBUST') is not None and str(h.get('PBROBUST')) != ('-0.75' if rob == 'R075' else '0'):
        probs.append(f"PBROBUST {h.get('PBROBUST')!r} for a {rob} map")
    bsrc = os.path.join(os.path.dirname(os.path.abspath(path)), f'I_deep_Taper_{rob}{loc["token"]}_{loc["scope"]}.fits')
    if 'PBBMAJ' in h and os.path.exists(bsrc):
        if not abs(float(h['PBBMAJ']) - float(_hdr(bsrc)['BMAJ'])) <= 1e-9 * abs(float(h['PBBMAJ'])):
            probs.append(f"PBBMAJ {h['PBBMAJ']} != beam source BMAJ {_hdr(bsrc)['BMAJ']}")
    return not probs, len(probs), '; '.join(map(str, probs)) or 'three-factor provenance complete'


@content_test('pb_map')
def values(path, **ctx):
    """Every pixel finite and inside masters_pbmaps.PB_VALUE_RANGE, and >= 1 pixel above PB_FLOOR (refuses the all-zero
    Full map of latent defect D2)."""
    s = _cfg().masters_pbmaps
    d = _data(path).astype(np.float64)
    fin = np.isfinite(d)
    lo, hi = s.PB_VALUE_RANGE
    mn = float(np.nanmin(d)) if fin.any() else float('nan')
    mx = float(np.nanmax(d)) if fin.any() else float('nan')
    above = float((d[fin] > s.PB_FLOOR).mean()) if fin.any() else 0.0
    ok = bool(fin.all()) and lo <= mn and mx <= hi and above > 0
    return ok, round(above, 4), f'finite {fin.mean():.4f}; min {mn:.4g} max {mx:.4g} (range {lo}..{hi}); {above:.3f} of pixels > {s.PB_FLOOR}'


def _sibling(path, **ctx):
    from exopipe.phase3.lib.masters import pb_apply as PB
    b = os.path.basename(path)
    if '_pbcorr' not in b:
        return False, None, 'not a pbcorr sibling name', 0
    plain = os.path.join(os.path.dirname(os.path.abspath(path)), b.replace('_pbcorr', '', 1))
    loc = locate(path, **ctx)
    rob = 'R075' if '_R075' in b else 'R0'
    mp = os.path.join(os.path.dirname(os.path.abspath(path)),
                      _cfg().masters_pbmaps.PB_MAP_NAMES[rob].format(scope=loc['scope']))
    if not (os.path.exists(plain) and os.path.exists(mp)):
        return False, None, f'plain {os.path.basename(plain)} or PB map {os.path.basename(mp)} missing', 0
    ok, why = PB.sibling_check(path, plain, _data(mp))
    mm = re.search(r'at (\d+) pixels', why)
    n = int(mm.group(1)) if mm else 0
    return ok, n, f'{why} (vs {os.path.basename(plain)} x {os.path.basename(mp)})', n


@content_test('pbcorr_sibling')
def sibling_check(path, **ctx):
    """sibling x its robust's PB map = its plain within rel 1e-5 + abs 1e-9 where PB > the floor, NaN elsewhere
    (pb_apply.sibling_check; IMPL-VCLEAN-PBCORR-SIBLINGS-STALE-PB)."""
    ok, n, det, _ = _sibling(path, **ctx)
    return ok, n, det


@content_test('pbcorr_sibling')
def tested_pixels(path, **ctx):
    """More than 0 pixels tested (latent defect D3: a sibling with no pixel above the floor passes sibling_check vacuously)."""
    ok, n, det, npx = _sibling(path, **ctx)
    return npx > 0, npx, f'{npx} pixels tested'


@content_test('pbcorr_sibling')
def count(path, **ctx):
    """Cell level: a sibling for every plain PB variant present (5 after pbmaps, 6 once the cleaned V exists; T4)."""
    loc = locate(path, **ctx)
    fd = os.path.dirname(os.path.abspath(path))

    def f():
        miss = []
        n = 0
        for v in _cfg().masters_pbmaps.PB_VARIANTS:
            plain = os.path.join(fd, f'{v}{loc["token"]}_{loc["scope"]}.fits')
            if os.path.exists(plain):
                n += 1
                if not os.path.exists(os.path.join(fd, f'{v}_pbcorr{loc["token"]}_{loc["scope"]}.fits')):
                    miss.append(v)
        return not miss, n - len(miss), f'{n - len(miss)} of {n} plain variants have a sibling; missing {miss}'
    return _cached(('sibcount',) + _dirstate(fd) + (loc['token'],), f)


@content_test('vclean_master')
def sane(path, **ctx):
    """|max| < 1e6 Jy/beam and finite somewhere (vclean_masters.sane), >= 50 per cent finite (verifier integrity)."""
    fin, mx = _finite_maxabs(_data(path))
    ok = np.isfinite(mx) and mx < _cfg().masters_vclean.SANE_MAX_JY and fin >= _cfg().masters_verify.INTEGRITY['min_finite']
    return ok, round(mx, 4) if np.isfinite(mx) else None, f'max |pixel| {mx:.4g}; finite {fin:.3f}'


@content_test('vclean_master')
def sibling(path, **ctx):
    """Its pbcorr sibling passes sibling_check against PB_deep_<scope> (vclean_masters.write_vclean_sibling)."""
    b = os.path.basename(path)
    sib = os.path.join(os.path.dirname(os.path.abspath(path)), b.replace('_clean', '_clean_pbcorr', 1))
    if not os.path.exists(sib):
        return False, None, f'sibling {os.path.basename(sib)} missing'
    ok, n, det, _ = _sibling(sib, **ctx)
    return ok, n, det


@content_test('vclean_master')
def from_current_v_master(path, **ctx):
    """Built from this generation's V master: CONSTDIG (inherited from the raw V master by the clean engine) and NCOADD equal
    those of V_deep_Taper_R0 beside it; SKYSUB stamped (the engine's own card). Open: spec 12 item 14 asks which card
    links the two - CONSTDIG is what the files carry today (checked on 23h_23MHz 2026-10-13)."""
    loc = locate(path, **ctx)
    v = os.path.join(os.path.dirname(os.path.abspath(path)), f'V_deep_Taper_R0{loc["token"]}_{loc["scope"]}.fits')
    if not os.path.exists(v):
        return False, None, 'V master missing beside it'
    h, hv = _hdr(path), _hdr(v)
    ok = h.get('CONSTDIG') == hv.get('CONSTDIG') and h.get('NCOADD') == hv.get('NCOADD') and bool(h.get('SKYSUB'))
    return ok, h.get('CONSTDIG'), f"CONSTDIG {h.get('CONSTDIG')} vs V master {hv.get('CONSTDIG')}; NCOADD {h.get('NCOADD')}/{hv.get('NCOADD')}; SKYSUB {h.get('SKYSUB')}"


# =====================================================================================================================
# master_generation (T1-T8)
# =====================================================================================================================

def _gen_results(path):
    from exopipe.phase3.lib.masters import deep_verify as DV
    g = os.path.abspath(path.rstrip('/'))
    stamp = []
    for root, _, fs in os.walk(g):
        for f in fs:
            try:
                stamp.append(os.path.getmtime(os.path.join(root, f)))
            except OSError:
                pass
    key = ('gen', g, len(stamp), max(stamp) if stamp else 0)
    return _cached(key, lambda: {n: (ok, v, d) for n, ok, v, d in DV.run_all(g)})


def _gen_test(name):
    def fn(path, **ctx):
        if not os.path.isdir(path):
            return False, None, f'{path} is not a generation directory'
        return _gen_results(path)[name]
    fn.__name__ = name
    fn.__doc__ = f'{name} of check_rebuild.py on the scaled generation (lib/masters/deep_verify.py)'
    return fn


for _n in ('T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8'):
    content_test('master_generation', name=_n)(_gen_test(_n))


# =====================================================================================================================
# master_catalog (deep and sub-LST parent catalogues)
# =====================================================================================================================

def _md5_float32(a):
    return hashlib.md5(np.ascontiguousarray(np.asarray(a).astype(np.float32)).tobytes()).hexdigest()


def catalog_master_path(path, **ctx):
    """The master a parent catalogue was made on: ctx master=, the sidecar's 'source' (sub-LST), or the layout - deep
    <...>/parent_catalogs/master_R075_<cell>.gaul.csv -> paths.COADD_SB_ROOT (or ctx installed_root)/<cell>/coadd_final;
    sub-LST <SL>/parent_catalogs_<LST>/master_R075_<LST>_<scope>.gaul.csv -> <SL>/coadd_<LST>/<LST>_<scope>/coadd_final."""
    if ctx.get('master'):
        return ctx['master']
    side = path + '.gen.json'
    meta = json.load(open(side)) if os.path.exists(side) else {}
    src = str(meta.get('source', ''))
    if ' on /' in src:
        return src.split(' on ', 1)[1].strip()
    b = os.path.basename(path)
    m = re.match(r'^master_R075_(\d\dh)_(\d+MHz|Red|Green|Blue|Full)\.gaul\.csv$', b)
    if not m:
        return None
    lst, scope = m.groups()
    pdir = os.path.dirname(os.path.abspath(path))
    name = meta.get('master') or f'I_deep_Taper_R075_{scope}.fits'
    if os.path.basename(pdir).startswith('parent_catalogs_'):
        return os.path.join(os.path.dirname(pdir), f'coadd_{lst}', f'{lst}_{scope}', 'coadd_final', name)
    root = ctx.get('installed_root') or _cfg().paths.COADD_SB_ROOT
    return os.path.join(root, f'{lst}_{scope}', 'coadd_final', name)


@content_test('master_catalog')
def sidecar_md5(path, **ctx):
    """.gen.json 'master' = the master's file name and master_md5_data = a digest of that master's float32 data re-read now:
    the raw array (p3s5_parents.py:50-53, sublst_parent_catalogs.py:42-46) or the array with its NaN border set to 0 (the
    search's inline catalogue, phase3_deep) - production's own conformance gate accepts either (exopipe_ops/gate_inputs.py
    parents, 53-59); a catalogue of any other master fails (FINDING-10MIN-PARENT-CATALOGUES-PREDATE-REBAKE-REBUILT)."""
    side = path + '.gen.json'
    if not os.path.exists(side):
        return False, None, 'no .gen.json sidecar'
    meta = json.load(open(side))
    mp = catalog_master_path(path, **ctx)
    if not mp or not os.path.exists(mp):
        return False, None, f'master {mp} not found'
    if meta.get('master') and meta.get('master') != os.path.basename(mp):
        return False, None, f"sidecar names {meta.get('master')}, not {os.path.basename(mp)}"
    d = _data(mp).astype(np.float32)
    raw, nan0 = _md5_float32(d), _md5_float32(np.nan_to_num(d))
    got = meta.get('master_md5_data')
    ok = got in (raw, nan0)
    which = 'raw' if got == raw else ('NaN border = 0' if got == nan0 else 'NEITHER')
    return ok, raw[:8], f'sidecar {str(got)[:8]} vs {os.path.basename(mp)} now raw {raw[:8]} / NaN->0 {nan0[:8]} ({which})'


@content_test('master_catalog')
def rows(path, **ctx):
    """More than 0 Gaussian rows (PyBDSF gaul CSV: '#' header lines, then one row per Gaussian)."""
    n = 0
    with open(path) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                n += 1
    return n > 0, n, f'{n} Gaussian rows'


@content_test('master_catalog')
def recipe(path, **ctx):
    """thresh_isl 4.0 and thresh_pix 5.0 recorded in the sidecar (masters_sources.THRESH_ISL / THRESH_PIX)."""
    s = _cfg().masters_sources
    side = path + '.gen.json'
    meta = json.load(open(side)) if os.path.exists(side) else {}
    ok = meta.get('thresh_isl') == s.THRESH_ISL and meta.get('thresh_pix') == s.THRESH_PIX
    return ok, meta.get('thresh_pix'), f"thresh_isl {meta.get('thresh_isl')} thresh_pix {meta.get('thresh_pix')}"


@content_test('master_catalog')
def dewarp_consistent(path, **ctx):
    """Deep: when the archived screen record says the RBF master dewarp was applied to the cell (applied=1), the master the
    catalogue was made on carries MDWARP (DESIGN 11.2: sources after dewarp + install). Sub-LST catalogues: not applicable."""
    from exopipe.phase3.lib.masters import membership as MB
    b = os.path.basename(path)
    m = re.match(r'^master_R075_(\d\dh)_(\d+MHz|Red|Green|Blue|Full)\.gaul\.csv$', b)
    if (not m or ctx.get('tier') in ('sublst', 'snapshot')
            or os.path.basename(os.path.dirname(os.path.abspath(path))).startswith('parent_catalogs_')):
        return True, None, 'not a deep parent catalogue (no screen record applies)'
    rec = _cfg().masters_verify.DEWARP_SCREEN_RECORD
    rows_ = MB._rcsv(rec)
    if not rows_:
        return False, None, f'screen record {rec} unreadable'
    applied = (m.group(1), m.group(2)) in {(r['lst'], r['scope']) for r in rows_ if r.get('applied', '1') == '1'}
    if not applied:
        return True, 0, 'cell not in the applied screen record'
    mp = catalog_master_path(path, **ctx)
    has = bool(mp and os.path.exists(mp) and _hdr(mp).get('MDWARP'))
    return has, 1, f'screen applied; master MDWARP {"present" if has else "MISSING"} ({mp})'
