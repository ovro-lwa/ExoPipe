"""Content tests of the sub-LST frames products (unit frames-tiers; spec plan/specs/frames.md section 6).

Classes this tier writes (each test reads the file's CONTENT - pixels, rows, header values against the data or against
the files it names). The window stacks, window masters and arm-A residuals are masters' products and classes
(lib/masters/tests_sublst.py: sublst_window*, sublst_window_master, sublst_loo_residual; crosscheck decision 6, 4.2).
  sublst_wideband      wideband window products: stack, _pb, _nframes, set-A and set-B residual (phase3_sublst_wideband.py)
  tenmin_tree          tree B: frames.csv, bin_masters/PB_bin<k>_<band>.fits, NO_WINDOWS.txt (phase3_sublst_tenmin_tree.py)
  sublst_residual_set  one LST: content test (b) R1-R3 of exopipe_ops/sublst_reserve_test_residuals.py, ported, plus the
                       cards of every residual (path = the LST's stack root <sublst>/<LST>; ctx x2_root, default <path>_x2)
A test returns (passed, value, detail). Context keys are optional: everything is derived from the path when absent, so
`exopipe3 verify <class> <files>` and --resume re-tests run the same checks.
"""
import csv
import glob
import os
import re

import numpy as np

from exopipe.phase3.core.validate import content_test

_STACK_RE = re.compile(r'^sublst_(?P<lst>\d\dh)_(?P<night>\d{4}-\d\d-\d\d)_(?P<band>\d+MHz|Red|Green|Blue|Full)_(?P<S>[IV])'
                       r'_w(?P<k>\d)(?P<sfx>_pb|_nframes|_resid_loo)?\.fits$')
_GROUPS = ('Red', 'Green', 'Blue', 'Full')


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def _hdr(path):
    from astropy.io import fits
    return fits.getheader(path, 0)


def _cards(path):
    """Primary-header cards read raw (core.record.fits_primary_cards: header blocks only, no astropy) - for the bulk
    header reads of the tree and residual-set tests."""
    from exopipe.phase3.core.record import fits_primary_cards
    return fits_primary_cards(path)


def _data(path):
    from astropy.io import fits
    with fits.open(path, memmap=True) as h:
        return np.array(np.squeeze(h[0].data))


def _patch(path, half=32):
    """Central (2*half)^2 patch of a 2-D FITS image, read through the memory map."""
    from astropy.io import fits
    with fits.open(path, memmap=True) as h:
        d = h[0].data
        while d.ndim > 2:
            d = d[0]
        ny, nx = d.shape
        return np.array(d[ny // 2 - half:ny // 2 + half, nx // 2 - half:nx // 2 + half], dtype=np.float64)


def _frozen_weights(lst, band):
    """The frozen deep co-add record the sub-LST tier reads (masters_sublst; ruling B 2026-10-08) - masters' one resolver."""
    from exopipe.phase3.lib.masters.tier_sublst import frozen_weights
    return frozen_weights(lst, band)


def _excluded(lst, band, night):
    from exopipe.phase3.core import exclusions
    return exclusions.load().is_excluded(lst, band, night)


def _fail(msgs):
    return (not msgs), len(msgs), '; '.join(msgs[:6]) if msgs else 'ok'


def _sublst_root_of(path):
    """(<sublst root>, lst, arm) from a path under <SL>/<LST>[_x2]/<LST>_<night>/<band>/<S>/<file>."""
    parts = os.path.abspath(path).split(os.sep)
    for i in range(len(parts) - 1, 0, -1):
        m = re.match(r'^(\d\dh)(_x2)?$', parts[i])
        if m and i + 1 < len(parts) and parts[i + 1].startswith(m.group(1) + '_'):
            return os.sep.join(parts[:i]) or '/', m.group(1), ('B' if m.group(2) else 'A')
    return None, None, None


# ---------------------------------------------------------------------------------------------------------------------
# per-file checks of a sub-LST residual (arm A, arm B, wideband sets), used by sublst_residual_set
# ---------------------------------------------------------------------------------------------------------------------

def resid_card_problems(path, h, m, master_root=None):
    """The per-file checks of a sub-LST residual (shared with sublst_residual_set)."""
    msgs = []
    k = int(m['k'])
    if h.get('LSTBIN') != k:
        msgs.append(f"LSTBIN {h.get('LSTBIN')!r} != w{k}")
    if not (float(h.get('DURATION', 0) or 0) > 0):
        msgs.append(f"DURATION {h.get('DURATION')!r}")
    if 'BINMSTR' in h:
        msgs.append('carries BINMSTR (the search would add a bin master)')
    if 'SUBTRACT' not in h:
        msgs.append('no SUBTRACT card')
    if 'JUPMASK' not in h:
        msgs.append('no JUPMASK card (DEFECT-SUBLST-ARM-B-LOO-WITHOUT-JUPITER-MASK)')
    lm = str(h.get('LOOMSTR', ''))
    if m['band'] in _GROUPS:
        mm = re.match(r'^sublst_master_%s_(\d+MHz)_%s_w%d\.fits$' % (m['lst'], m['S'], k), lm)
        memb = str(h.get('WBMEMB', '')).split(',')
        if not mm or mm.group(1) not in memb:
            msgs.append(f'LOOMSTR {lm!r} is not a member master of WBMEMB {memb}')
    else:
        want = f"sublst_master_{m['lst']}_{m['band']}_{m['S']}_w{k}.fits"
        if lm != want:
            msgs.append(f'LOOMSTR {lm!r} != {want}')
        elif master_root and not os.path.exists(os.path.join(master_root, m['band'], m['S'], want)):
            msgs.append(f'window master {want} not under {master_root}')
        if not (int(h.get('LOONCOAD', 0) or 0) >= 2):
            msgs.append(f"LOONCOAD {h.get('LOONCOAD')!r} < 2 (per-night rule)")
    return msgs


# ---------------------------------------------------------------------------------------------------------------------
# sublst_wideband
# ---------------------------------------------------------------------------------------------------------------------

@content_test('sublst_wideband')
def wideband_content(path, **ctx):
    """WIDEBAND = the group of the name; WBMEMB >= frames_sublst.WIDEBAND_MIN_MEMBERS members of the group; WBWGTS one per member,
    summing to 1 (1e-3); every member has its five products for both Stokes (stack, _pb, _nframes, set-A residual beside
    the stacks, set-B residual under <LST>_x2); the central 64x64 patch = the WBWGTS-weighted mean of the members' patches
    (minimum for _nframes; NaN propagates) within 1e-3 of the patch scale (WBWGTS carry 4 significant digits); baked flux:
    SHSCALE x FLUXCORR = the weighted mean of the members' SHSCALE x FLUXCORR (1e-3); not excluded under the band rule."""
    from exopipe.phase3.core import cells
    b = os.path.basename(path)
    m = _STACK_RE.match(b)
    if not m or m['band'] not in _GROUPS:
        return False, None, f'not a wideband window product name: {b}'
    sl, lst, arm = _sublst_root_of(path)
    if not sl:
        return False, None, 'path is not under <sublst>/<LST>[_x2]/<LST>_<night>/<group>/<S>/'
    G, S, k, night = m['band'], m['S'], int(m['k']), m['night']
    kind = {None: 'stack', '_pb': 'pb', '_nframes': 'nf', '_resid_loo': 'rA' if arm == 'A' else 'rB'}[m['sfx']]
    h = _hdr(path)
    msgs = []
    if h.get('WIDEBAND') != G:
        msgs.append(f"WIDEBAND {h.get('WIDEBAND')!r} != {G}")
    memb = [x for x in str(h.get('WBMEMB', '')).split(',') if x]
    try:
        w = np.array([float(x) for x in str(h.get('WBWGTS', '')).split(',') if x])
    except ValueError:
        w = np.array([])
    allowed = set(cells.band_members(G))
    if len(memb) < _cfg().frames_sublst.WIDEBAND_MIN_MEMBERS or any(x not in allowed for x in memb):
        msgs.append(f'WBMEMB {memb} (members of {G}: {sorted(allowed)})')
    if len(w) != len(memb) or not len(w) or abs(w.sum() - 1.0) > 1e-3:
        msgs.append(f'WBWGTS {list(w)} do not sum to 1 for {len(memb)} members')
    if msgs:
        return _fail(msgs)
    stack_root, x2_root = os.path.join(sl, lst), os.path.join(sl, f'{lst}_x2')

    def prod(sb, St, kd):
        base = f'sublst_{lst}_{night}_{sb}_{St}_w{k}'
        d = os.path.join(x2_root if kd == 'rB' else stack_root, f'{lst}_{night}', sb, St)
        return os.path.join(d, base + {'stack': '.fits', 'pb': '_pb.fits', 'nf': '_nframes.fits',
                                        'rA': '_resid_loo.fits', 'rB': '_resid_loo.fits'}[kd])
    missing = [prod(sb, St, kd) for sb in memb for St in ('I', 'V') for kd in ('stack', 'pb', 'nf', 'rA', 'rB')
               if not os.path.exists(prod(sb, St, kd))]
    if missing:
        msgs.append(f'{len(missing)} member product(s) missing, e.g. {missing[0]}')
        return _fail(msgs)
    out = _patch(path)
    arrs = np.stack([_patch(prod(sb, S, kind)) for sb in memb])
    exp = np.min(arrs, axis=0) if kind == 'nf' else np.tensordot(w, arrs, axes=1) / w.sum()
    nan_o, nan_e = ~np.isfinite(out), ~np.isfinite(exp)
    scale = float(np.nanmax(np.abs(arrs))) if np.isfinite(arrs).any() else 0.0
    if (nan_o != nan_e).any():
        msgs.append(f'NaN mask of the central patch differs from the members\' ({int((nan_o != nan_e).sum())} px)')
    else:
        dev = float(np.nanmax(np.abs(out - exp))) if (~nan_o).any() else 0.0
        if dev > 1e-3 * scale + 1e-6:
            msgs.append(f'central patch deviates from the weighted member mean by {dev:.3g} (scale {scale:.3g})')
    if kind in ('stack', 'rA', 'rB') and h.get('FLUXBAKE'):
        mh = [_hdr(prod(sb, 'I', 'stack')) for sb in memb]
        bm = np.array([float(x.get('SHSCALE', np.nan)) * float(x.get('FLUXCORR', np.nan)) for x in mh])
        beff = float(np.sum(w * bm) / w.sum())
        got = float(h.get('SHSCALE', np.nan)) * float(h.get('FLUXCORR', np.nan))
        if not (np.isfinite(got) and abs(got - beff) <= 1e-3 * abs(beff)):
            msgs.append(f'pixel factor SHSCALE x FLUXCORR {got:.6g} != weighted member mean {beff:.6g}')
    if _excluded(lst, G, night):
        msgs.append(f'({lst}, {G}, {night}) is excluded under the band rule (data_decisions)')
    return _fail(msgs)


# ---------------------------------------------------------------------------------------------------------------------
# tenmin_tree
# ---------------------------------------------------------------------------------------------------------------------

def _tree_name(path):
    """(lst, band) from <tree>/<LST>_<band>/..."""
    d = os.path.dirname(path)
    if os.path.basename(d) == 'bin_masters':
        d = os.path.dirname(d)
    m = re.match(r'^(\d\dh)_(\d+MHz|Red|Green|Blue|Full)$', os.path.basename(d))
    return (m.group(1), m.group(2), d) if m else (None, None, d)


def _frames_rows(csv_path):
    with open(csv_path, newline='') as fh:
        rd = csv.DictReader(fh)
        return list(rd.fieldnames or []), list(rd)


@content_test('tenmin_tree')
def tree_content(path, **ctx):
    """frames.csv: the 16 columns of the tree writer; every row's sub_path is a link resolving to its src residual, whose
    name carries the row's (night, band, Stokes, window) and whose header has LSTBIN = bin, DURATION, no BINMSTR and
    FLUXBAKE = the fluxbake column; both Stokes per key; PB_bin<k> exists for every bin; no excluded (LST, band, night).
    PB_bin<k>_<band>.fits: finite PB values in [0, 1], PBSRC present, PBNNIGHT = the I rows of bin k, and the central patch
    = the co-add-weight-weighted mean of those nights' window PB maps (frozen deep record weights).
    NO_WINDOWS.txt: frames.csv has no rows and the refine record it names shows every window refused by rule
    (phase3_sublst_tenmin_tree.rule_empty, in place)."""
    lst, band, cdir = _tree_name(path)
    if not lst:
        return False, None, f'not under <tree>/<LST>_<band>/: {path}'
    b = os.path.basename(path)
    cols_want = list(_cfg().frames_sublst.TREE_COLUMNS)
    if b == 'frames.csv':
        cols, rows = _frames_rows(path)
        msgs = []
        if cols != cols_want:
            return False, None, f'columns {cols} != {cols_want}'
        keys = {}
        for r in rows:
            sp, src = r['sub_path'], r['src']
            if not os.path.islink(sp) or os.path.realpath(sp) != os.path.realpath(src) or not os.path.exists(src):
                msgs.append(f'{sp}: not a link to an existing {src}')
                continue
            m = _STACK_RE.match(os.path.basename(src))
            if (not m or m['sfx'] != '_resid_loo' or m['night'] != r['date'] or m['band'] != band
                    or m['S'] != r['stokes'] or int(m['k']) != int(r['bin']) or r['key'] != f"{r['date']}T{int(r['bin']):02d}"):
                msgs.append(f"row {r['key']} {r['stokes']}: src name {os.path.basename(src)} disagrees")
                continue
            h = _cards(src)
            if h.get('LSTBIN') != int(r['bin']) or 'DURATION' not in h or 'BINMSTR' in h:
                msgs.append(f'{src}: LSTBIN/DURATION/BINMSTR')
            if str(int(bool(h.get('FLUXBAKE', False)))) != str(r['fluxbake']).strip():
                msgs.append(f"{src}: FLUXBAKE vs fluxbake column {r['fluxbake']}")
            if _excluded(lst, band, r['date']):
                msgs.append(f"({lst}, {band}, {r['date']}) excluded")
            keys.setdefault(r['key'], set()).add(r['stokes'])
        half = [k for k, s in keys.items() if s != {'I', 'V'}]
        if half:
            msgs.append(f'{len(half)} key(s) without both Stokes, e.g. {half[0]}')
        for kbin in sorted({int(r['bin']) for r in rows}):
            if not os.path.exists(os.path.join(cdir, 'bin_masters', f'PB_bin{kbin}_{band}.fits')):
                msgs.append(f'PB_bin{kbin}_{band}.fits missing')
        if not rows and not os.path.exists(os.path.join(cdir, 'NO_WINDOWS.txt')):
            msgs.append('no rows and no NO_WINDOWS.txt')
        ok, _, det = _fail(msgs)
        return ok, len(rows), det if msgs else f'{len(rows)} rows, {len(keys)} windows'
    if b == 'NO_WINDOWS.txt':
        cols, rows = _frames_rows(os.path.join(cdir, 'frames.csv'))
        if rows:
            return False, len(rows), 'NO_WINDOWS.txt beside a frames.csv with rows'
        txt = open(path).read()
        m = re.search(r'From (\S+\.log):', txt)
        if not m:
            return False, None, 'NO_WINDOWS.txt names no refine record'
        log = m.group(1)
        import phase3_sublst_tenmin_tree as TT
        why = TT.rule_empty(os.path.dirname(os.path.dirname(log)), lst, band)
        return bool(why), len(why or []), ('refine record: every window refused by rule' if why
                                            else f'{log} does not show every window refused by rule')
    m = re.match(r'^PB_bin(\d)_%s\.fits$' % re.escape(band), b)
    if m:
        kbin = int(m.group(1))
        h = _hdr(path)
        d = _data(path)
        msgs = []
        f = d[np.isfinite(d)]
        if not f.size or f.min() < -1e-6 or f.max() > 1.0001:
            msgs.append('PB values outside [0, 1] or none finite')
        if 'PBSRC' not in h:
            msgs.append('no PBSRC card')
        fcsv = os.path.join(cdir, 'frames.csv')
        if os.path.exists(fcsv):
            _, rows = _frames_rows(fcsv)
            nights = sorted({r['date'] for r in rows if int(r['bin']) == kbin and r['stokes'] == 'I'})
            if h.get('PBNNIGHT') != len(nights):
                msgs.append(f"PBNNIGHT {h.get('PBNNIGHT')!r} != {len(nights)} I rows of bin {kbin}")
            srcs = [r['src'] for r in rows if int(r['bin']) == kbin and r['stokes'] == 'I']
            if srcs and nights:
                sl, lst_, _ = _sublst_root_of(srcs[0])
                wp = _frozen_weights(lst, band)
                if sl and os.path.exists(wp):
                    wts = {r['date']: float(r['weight']) for r in csv.DictReader(open(wp)) if r.get('weight') not in (None, '')}
                    acc, wsum = 0.0, 0.0
                    for n in nights:
                        pbp = os.path.join(sl, lst, f'{lst}_{n}', band, 'I', f'sublst_{lst}_{n}_{band}_I_w{kbin}_pb.fits')
                        if n not in wts or not os.path.exists(pbp):
                            msgs.append(f'no weight or window PB for {n}')
                            break
                        acc = acc + wts[n] * np.nan_to_num(_patch(pbp, 16))
                        wsum += wts[n]
                    else:
                        exp = acc / wsum
                        dev = float(np.max(np.abs(_patch(path, 16) - exp)))
                        if dev > 1e-5:
                            msgs.append(f'central patch deviates from the weighted window-PB mean by {dev:.3g}')
        return _fail(msgs)
    return False, None, f'not a tree product: {b}'


# ---------------------------------------------------------------------------------------------------------------------
# sublst_residual_set (content test (b) of sublst_reserve_test_residuals.py, ported)
# ---------------------------------------------------------------------------------------------------------------------

def residual_set_report(stack_root, x2_root=None, lst=None, weights_tmpl=None, subbands=None):
    """R1-R3 of exopipe_ops/sublst_reserve_test_residuals.py (preamble 4, Gregg 2026-10-07 07:03 'approved 1-4'; Q11) for
    one LST, independently of the code that wrote the residuals, plus the per-file cards of every residual.
    Returns (failures [str], summary str)."""
    from phase3_tier_rules import membership_vetoed
    C = _cfg()
    L = lst or os.path.basename(stack_root.rstrip('/'))
    X2 = x2_root or stack_root.rstrip('/') + '_x2'
    weights_tmpl = weights_tmpl or _frozen_weights('{lst}', '{sb}')
    SBS = list(subbands or C.instrument.ALL_SUBBANDS)
    WB = _GROUPS
    fail, lines = [], []
    pat = re.compile(r'sublst_%s_(\d{4}-\d\d-\d\d)_(\w+?)_([IV])_w(\d)(_resid_loo)?\.fits$' % L)
    stacks = {}
    resA, resB = {}, {}
    for p in glob.glob(f'{stack_root}/{L}_*/*/[IV]/sublst_{L}_*.fits'):
        m = pat.search(os.path.basename(p))
        if not m:
            continue
        night, band, S, k, res = m.groups()
        if res:
            resA[(band, S, int(k), night)] = p
        else:
            stacks.setdefault((band, S, int(k)), set()).add(night)
    for p in glob.glob(f'{X2}/{L}_*/*/[IV]/sublst_{L}_*_resid_loo.fits'):
        m = pat.search(os.path.basename(p))
        if m:
            night, band, S, k, _ = m.groups()
            resB[(band, S, int(k), night)] = p
    allowed, nloo, wok = set(), {}, {}
    for sb in SBS:
        wp = weights_tmpl.format(lst=L, sb=sb)
        if not os.path.exists(wp):
            continue
        wts = {}
        for r in csv.DictReader(open(wp)):
            try:
                wts[r['date']] = 0.0 if (str(r.get('in_master', '1')).strip() == '0' or membership_vetoed(L, sb, r['date'])) \
                    else float(r['weight'])
            except (KeyError, ValueError, TypeError):
                pass
        for S in 'IV':
            for k in range(C.frames_sublst.WINDOWS):
                ns = [n for n in stacks.get((sb, S, k), ()) if n in wts]
                nw = sum(1 for n in ns if wts[n] > 0)
                wok[(sb, S, k)] = len(ns) >= 2 and nw >= 2
                for n in ns:
                    c = nw - (1 if wts[n] > 0 else 0)
                    nloo[(sb, S, k, n)] = c
                    if len(ns) >= 2 and nw >= 2 and c >= 2:
                        allowed.add((sb, S, k, n))
    refused_log = set()
    for sb in SBS:
        lp = f'{X2}/logs/{L}_{sb}.log'
        if not os.path.exists(lp):
            continue
        for l in open(lp, errors='replace'):
            q = re.match(r'REFUSED-NIGHT %s w(\d) (\S+): no flow returned' % sb, l)
            if q:
                for S in 'IV':
                    refused_log.add((sb, S, int(q.group(1)), q.group(2)))
            q = re.match(r'REFUSED-WINDOW %s V w(\d): no Stokes I flow' % sb, l)
            if q:
                for n in stacks.get((sb, 'V', int(q.group(1))), ()):
                    refused_log.add((sb, 'V', int(q.group(1)), n))
    cards_bad = []
    from concurrent.futures import ThreadPoolExecutor
    allp = sorted(set(resA.values()) | set(resB.values()))
    with ThreadPoolExecutor(max_workers=16) as ex:        # Lustre metadata latency dominates a serial header scan
        hdr_cache = dict(zip(allp, ex.map(_cards, allp)))

    def hdr(p):
        if p not in hdr_cache:
            hdr_cache[p] = _cards(p)
        return hdr_cache[p]
    for arm, res in (('A', resA), ('x2', resB)):
        sub = {k: p for k, p in res.items() if k[0] not in WB}
        r1 = [(k, nloo.get(k)) for k in sub if k not in allowed]
        r1h = [k for k, p in sub.items() if k in allowed and int(hdr(p).get('LOONCOAD', -1)) != nloo[k]]

        def x2_expected(k):
            sb, S, kk, n = k
            return n in stacks.get((sb, 'I', kk), ()) and wok.get((sb, 'I', kk), False) and k not in refused_log
        miss = sorted(k for k in allowed if k not in sub and (arm == 'A' or x2_expected(k)))
        lines.append(f'arm {arm}: {len(sub)} subband residuals; not allowed {len(r1)}; LOONCOAD != count {len(r1h)}; '
                     f'allowed but missing {len(miss)}')
        if r1 or r1h or miss:
            fail.append(f'arm {arm} R1/R2: not allowed {r1[:2]}, LOONCOAD {r1h[:2]}, missing {miss[:2]}')
        r3 = []
        for k, p in res.items():
            if k[0] not in WB:
                continue
            for b in str(hdr(p).get('WBMEMB', '')).split(','):
                if (b, k[1], k[2], k[3]) not in res:
                    r3.append((k, b))
        lines.append(f'arm {arm}: {sum(1 for k in res if k[0] in WB)} wideband residuals; member without own residual {len(r3)}')
        if r3:
            fail.append(f'wideband {arm} R3: {r3[:2]}')
        for k, p in res.items():
            m = _STACK_RE.match(os.path.basename(p))
            pr = resid_card_problems(p, hdr(p), m) if m else ['unparsable name']
            # LOONCOAD < 2 on a subband residual is R1's business (reported there), not a card problem here
            pr = [x for x in pr if not x.startswith('LOONCOAD')]
            if pr:
                cards_bad.append(f'{os.path.basename(p)}: {pr[0]}')
    if cards_bad:
        fail.append(f'cards: {len(cards_bad)} residual(s), e.g. {cards_bad[0]}')
    lines.append(f'per-night rule refuses {sum(1 for c in nloo.values() if c < 2)} of {len(nloo)} weighted-record stacks')
    return fail, ' | '.join(lines)


@content_test('sublst_residual_set')
def residual_set(path, x2_root=None, lst=None, **ctx):
    """One LST (path = its stack root <sublst>/<LST>): R1 every subband residual (arm A beside the stacks, arm B under
    <LST>_x2) is allowed by the per-night rule and its LOONCOAD equals the count from the frozen deep weights; R2 arm A
    wrote a residual for every allowed (S, window, night), arm B likewise where the night has a Stokes I stack in that
    window, except refusals its refine log records; R3 every wideband residual's WBMEMB members have their own residual of
    the same (night, window, Stokes, arm); and every residual's cards (LSTBIN, DURATION, no BINMSTR, SUBTRACT, JUPMASK,
    LOOMSTR)."""
    if not os.path.isdir(path):
        return False, None, f'{path} is not an LST stack root'
    fail, summary = residual_set_report(path, x2_root=x2_root, lst=lst)
    return (not fail), len(fail), ('; '.join(fail) + ' || ' if fail else '') + summary
