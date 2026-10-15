"""Content tests of the `exopipe3 catalog` product classes - the deep chain and the classes every tier shares
(spec plan/specs/catalog.md section 6). Registered with core.validate.content_test; run by UnitRecord.add_output, by
--resume and by `exopipe3 verify <class> <paths>`. Each test reads the file's CONTENT and returns (passed, value, detail).

A test that needs a sibling product finds it beside the file (the production layout: everything in the run directory)
or, for a stage copy under <out>/_records/catalog/<label>/stage_<name>/, through that stage directory's stage.json
(``run_dir`` and the paths the stage read). Sub-LST and snapshot variants add their own tests to the shared classes in
tests_sublst.py / tests_snapshot.py (catalog-tiers unit); the tests here accept their directory names.

Classes: parent_snapshot, harvest_manifest, merged_record, conformant_catalog, frame_table, fitted_catalog, selection,
source_catalog, fork_catalog, union_catalog, identity_table, localization_table, transient_list, candidate_catalog
(dispatches master_catalog.csv / selected_*.csv to the stage tests above).
"""
import csv
import glob
import hashlib
import json
import math
import os
import random
import re

from exopipe.phase3.core.validate import content_test

KINDS = ('gaussian', 'template', 'peak')
DET_COLS = ('pos_kind', 'pos_ra', 'pos_dec', 'pos_err_maj_arcsec', 'pos_err_min_arcsec', 'pos_err_pa_deg',
            'pos_stat_maj_arcsec', 'pos_stat_min_arcsec', 'pos_sys_arcsec', 'pos_snr', 'pos_offset_arcmin', 'pos_reason',
            'peak_ra', 'peak_dec')        # ops/deep_forced_radio.py:73-74, 255 (DET_COLS)
IDENTITY_COLS = ('identity_id', 'identity_bands', 'identity_tiers', 'identity_n')
WIDEBANDS = ('Red', 'Green', 'Blue', 'Full')


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def _rows(path):
    with open(path, newline='') as f:
        rd = csv.DictReader(f)
        rows = list(rd)
        return list(rd.fieldnames or []), rows


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float('nan')


def _stage_info(path):
    """stage.json of the stage directory holding ``path`` (a stage copy), else {}."""
    p = os.path.join(os.path.dirname(os.path.abspath(path)), 'stage.json')
    if os.path.exists(p):
        try:
            with open(p) as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}
    return {}


def _sibling(path, name, ctx=None):
    """A sibling product: ctx[name] when given, beside the file, else in the run dir named by stage.json."""
    if ctx and ctx.get(name):
        return ctx[name]
    d = os.path.dirname(os.path.abspath(path))
    p = os.path.join(d, name)
    if os.path.exists(p):
        return p
    rd = _stage_info(path).get('run_dir')
    if rd and os.path.exists(os.path.join(rd, name)):
        return os.path.join(rd, name)
    return None


def _run_dir(path, ctx=None):
    if ctx and ctx.get('run_dir'):
        return ctx['run_dir']
    rd = _stage_info(path).get('run_dir')
    return rd or os.path.dirname(os.path.abspath(path))


def _excl():
    from exopipe.phase3.core import exclusions
    return exclusions.load()


def _band_class(band):
    for cls, subs in _cfg().catalog.BAND_CLASS.items():
        if band in subs:
            return cls
    return band


def _fail(msgs, n_ok, what):
    if msgs:
        return False, len(msgs), '; '.join(msgs[:6]) + (f' (+{len(msgs) - 6} more)' if len(msgs) > 6 else '')
    return True, n_ok, what


# ---------------------------------------------------------------------------------------------------------------------
# preflight: the per-pass parent-catalogue snapshot
# ---------------------------------------------------------------------------------------------------------------------

def _read_gaul_finite(p):
    """Rows with finite RA, Dec and peak (columns 4, 6, 10 as ops/gate_inputs._read_gaul)."""
    n = 0
    with open(p) as f:
        for ln in f:
            if ln.startswith('#') or not ln.strip():
                continue
            q = ln.split(',')
            try:
                if all(math.isfinite(float(q[i])) for i in (4, 6, 10)):
                    n += 1
            except (ValueError, IndexError):
                pass
    return n


def master_digests(master_path):
    """(raw, NaN-zeroed) md5 of the float32 data of a master - the two digests ops/gate_inputs.parents accepts."""
    import numpy as np
    from astropy.io import fits
    d = np.squeeze(fits.getdata(master_path)).astype(np.float32)
    return (hashlib.md5(np.ascontiguousarray(d).tobytes()).hexdigest(),
            hashlib.md5(np.ascontiguousarray(np.nan_to_num(d)).tobytes()).hexdigest())


@content_test('parent_snapshot')
def parent_catalogues_current(path, coadd_root=None, sample=None, **ctx):
    """One Robust -0.75 parent catalogue per installed cell, its sidecar naming the cell's master, >= 1 finite row, and for
    a seeded sample of cells the sidecar digest equal to the master's data now (ops/gate_inputs.py:39-63,
    ops/p3s5_parent_union.py:16-33)."""
    cfg = _cfg()
    if not os.path.isdir(path):
        return False, None, f'{path} is not a directory'
    root = coadd_root or cfg.paths.COADD_SB_ROOT          # the deep masters (shared root; catalog.DEEP_COADD_ROOT is gone)
    cats = sorted(glob.glob(os.path.join(path, 'master_R075_*.gaul.csv')))
    msgs = []
    if not cats:
        return False, 0, 'no master_R075_<cell>.gaul.csv in the snapshot'
    have = {}
    for c in cats:
        m = re.match(r'master_R075_(\d\dh)_([A-Za-z0-9]+)\.gaul\.csv$', os.path.basename(c))
        if not m:
            msgs.append(f'unexpected name {os.path.basename(c)}')
            continue
        lst, sc = m.groups()
        side = c + '.gen.json'
        if not os.path.exists(side):
            msgs.append(f'{lst}_{sc}: no sidecar')
            continue
        try:
            g = json.load(open(side))
        except ValueError as e:
            msgs.append(f'{lst}_{sc}: sidecar unreadable ({e})')
            continue
        if g.get('master') != f'I_deep_Taper_R075_{sc}.fits':
            msgs.append(f'{lst}_{sc}: sidecar names {g.get("master")}')
        if _read_gaul_finite(c) < 1:
            msgs.append(f'{lst}_{sc}: no finite row')
        have[f'{lst}_{sc}'] = g.get('master_md5_data')
    if not os.path.isdir(root):
        msgs.append(f'masters not readable here ({root}): coverage and digests cannot be checked on this host')
        return _fail(msgs, len(cats), '')
    installed = sorted(os.path.basename(d) for d in glob.glob(os.path.join(root, '[0-2][0-9]h_*'))
                       if os.path.exists(os.path.join(d, 'coadd_final',
                                                      f'I_deep_Taper_R075_{os.path.basename(d).split("_", 1)[1]}.fits')))
    missing = [c for c in installed if c not in have]
    if missing:
        msgs.append(f'{len(missing)} installed cells without a catalogue, e.g. {missing[:5]}')
    n = int(sample if sample is not None else cfg.catalog.DEEP_PARENT_TEST_SAMPLE)
    pick = sorted(random.Random(20261013).sample(sorted(have), min(n, len(have))))
    for cell in pick:
        sc = cell.split('_', 1)[1]
        mp = os.path.join(root, cell, 'coadd_final', f'I_deep_Taper_R075_{sc}.fits')
        if not os.path.exists(mp):
            msgs.append(f'{cell}: catalogue for a cell with no installed master')
            continue
        if have[cell] not in master_digests(mp):
            msgs.append(f'{cell}: sidecar digest {str(have[cell])[:8]} is not the current master\'s')
    return _fail(msgs, len(cats), f'{len(cats)} catalogues, {len(installed)} installed cells covered, '
                                  f'{len(pick)} digests re-read')


# ---------------------------------------------------------------------------------------------------------------------
# harvest and merge
# ---------------------------------------------------------------------------------------------------------------------

@content_test('harvest_manifest')
def harvest_every_published_attempt(path, **ctx):
    """The harvest stage's manifest: one row per published (cell, Stokes), every attempt present (harvested) under its
    collect root with its gate ledger (ops/p3s3_sources.published_attempts 'harvested')."""
    cols, rows = _rows(path)
    need = ('cell', 'stokes', 'node', 'att', 'dir', 'harvested', 'label')
    miss = [c for c in need if c not in cols]
    if miss:
        return False, None, f'missing columns {miss}'
    if not rows:
        return False, 0, 'no published attempt'
    msgs, seen = [], set()
    for r in rows:
        k = (r['cell'], r['stokes'])
        if k in seen:
            msgs.append(f'{k}: listed twice')
        seen.add(k)
        led = os.path.join(r['dir'], 'transients', f"Run_{r['label']}", 'v_deep_ledger.csv')
        if r['harvested'] != '1' or not os.path.exists(led):
            msgs.append(f"{r['cell']}/{r['stokes']} {r['att']}: not harvested ({led})")
        if 'patched' in r:                     # B-01 (b): the patch flag re-derived from the attempt's own record
            from exopipe.phase3.lib.catalog import deep_ported as DP
            src = DP.patched_from(r['dir'])
            if (r['patched'] == '1') != (src is not None) or (src is not None and r.get('patched_from') != src):
                msgs.append(f"{r['cell']}/{r['stokes']} {r['att']}: patched={r['patched']} from {r.get('patched_from')!r}, "
                            f"the attempt's PATCH_RECORD.json says {src!r}")
    n_p = sum(1 for r in rows if r.get('patched') == '1')
    return _fail(msgs, len(rows), f'{len(rows)} published attempts, every one harvested'
                 + (f'; {n_p} patched Run_P3S4 results (B-01 b)' if 'patched' in cols else ''))


def _merge_key(r):
    return (r['lst'], r['band'], r['stokes'], r['date'], round(_f(r['ra']), 4), round(_f(r['dec']), 4))


@content_test('merged_record')
def merge_gate_record(path, backing_keys=None, **ctx):
    """The merge gate's record (ops/p3s3_merge.py:98-160): > 0 accepted rows; no duplicate (cell, Stokes, date, position)
    key; every refused row has a reason; no accepted row on an epoch data_decisions excludes now; accepted + refused =
    the rows of the attempts read (when the harvest manifest is known); for a seeded sample of (cell, Stokes, date) keys
    every accepted row has a final ledger row within the backing cap and the accepted rows do not outnumber the finals
    (one-to-one), re-run from the ledger; v_deep_images.csv has no exact-duplicate row."""
    cfg = _cfg()
    cols, acc = _rows(path)
    msgs = []
    if not acc:
        return False, 0, 'zero accepted rows (merge gate refuses this)'
    seen = set()
    for r in acc:
        k = _merge_key(r)
        if k in seen:
            msgs.append(f"duplicate accepted key {k[:4]}")
        seen.add(k)
    refp = _sibling(path, 'master_catalog_refused.csv', ctx)
    refused = []
    if refp:
        _, refused = _rows(refp)
        nr = [r for r in refused if not (r.get('refusal_reason') or '').strip()]
        if nr:
            msgs.append(f'{len(nr)} refused rows without a refusal_reason')
    else:
        msgs.append('master_catalog_refused.csv not found')
    ex = _excl()
    bad = [r['cand_id'] for r in acc if ex.is_excluded(r['lst'], r['band'], r['date'])]
    if bad:
        msgs.append(f'{len(bad)} accepted rows on epochs data_decisions excludes now, e.g. {bad[:3]}')
    info = _stage_info(path)
    man = ctx.get('manifest') or info.get('harvest_manifest')
    n_in = None
    if man and os.path.exists(man):
        stale = {tuple(x) for x in (ctx.get('stale_attempts') or info.get('stale_attempts') or [])}
        n_in = 0
        for a in _rows(man)[1]:
            if (a['cell'], a['stokes']) in stale:
                continue                      # refused whole by the staleness gate: its rows are not read
            p = os.path.join(a['dir'], 'transients', f"Run_{a['label']}", 'master_catalog.csv')
            if os.path.exists(p):
                with open(p, newline='') as f:
                    n_in += max(0, sum(1 for _ in csv.reader(f)) - 1)
        if n_in != len(acc) + len(refused):
            msgs.append(f'rows of the attempts read {n_in} != accepted {len(acc)} + refused {len(refused)}')
    imgp = _sibling(path, 'v_deep_images.csv', ctx)
    if imgp:
        with open(imgp, newline='') as f:
            lines = f.read().splitlines()[1:]
        if len(lines) != len(set(lines)):
            msgs.append(f'v_deep_images.csv has {len(lines) - len(set(lines))} exact-duplicate rows')
    ledp = _sibling(path, 'v_deep_ledger.csv', ctx)
    nk = int(backing_keys if backing_keys is not None else cfg.catalog_merge.TEST_BACKING_KEYS)
    checked = 0
    if ledp and nk > 0:
        bykey = {}
        for r in acc:
            bykey.setdefault((r['lst'], r['band'], r['stokes'], r['date']), []).append(r)
        keys = sorted(random.Random(20261013).sample(sorted(bykey), min(nk, len(bykey))))
        want = set(keys)
        finals = {k: [] for k in keys}
        with open(ledp, newline='') as f:
            rd = csv.reader(f)
            h = next(rd)
            ix = {c: i for i, c in enumerate(h)}
            il, isc, ist, idt, ifi, ira, ide = (ix[c] for c in ('lst', 'scope', 'stokes', 'date', 'final', 'ra', 'dec'))
            for row in rd:
                if row[ifi] != '1':
                    continue
                k = (row[il], row[isc], row[ist], row[idt])
                if k in want:
                    finals[k].append((_f(row[ira]), _f(row[ide])))
        cap = cfg.catalog_merge.BACKING_CAP_DEG
        for k in keys:
            fl = finals[k]
            if len(bykey[k]) > len(fl):
                msgs.append(f'{k}: {len(bykey[k])} accepted rows but {len(fl)} final ledger rows (not one-to-one)')
            for r in bykey[k]:
                ra, dec = _f(r['ra']), _f(r['dec'])
                ok = any(math.hypot(((a - ra + 180.0) % 360.0 - 180.0) * math.cos(math.radians(dec)), d - dec) < cap
                         for a, d in fl)
                if not ok:
                    msgs.append(f"{r['cand_id']}: no final ledger row within {cap} deg")
            checked += 1
    return _fail(msgs, len(acc), f'{len(acc)} accepted, {len(refused)} refused'
                                 + (f', {n_in} rows in' if n_in is not None else '')
                                 + f'; backing re-run on {checked} keys')


# ---------------------------------------------------------------------------------------------------------------------
# conformance, frame caches, fitted positions, selection
# ---------------------------------------------------------------------------------------------------------------------

def _mid_el(lst, ra, dec, lat):
    hr = math.radians(15.0 * (int(lst.rstrip('h')) + 0.5) - ra)
    dr, la = math.radians(dec), math.radians(lat)
    return math.degrees(math.asin(math.sin(dr) * math.sin(la) + math.cos(dr) * math.cos(la) * math.cos(hr)))


@content_test('conformant_catalog')
def conformance_record(path, tier='deep', **ctx):
    """ops/p3s2_conform.py:471-559 on content: the as-run copy is the merge output; conformant + removed rows = as-run
    rows + re-admitted VM_ rows (cand_ids); every removed row has a removal_reason; no conformant row on an excluded
    epoch or (deep) below the floor at mid-hour (LST + 0.5 h, judged at the position conformance saw: peak_ra/peak_dec once
    fitted); every conformant V row carries an fp_prediction."""
    cfg = _cfg()
    cols, rows = _rows(path)
    msgs = []
    if not rows:
        return False, 0, 'no conformant rows (conformance refuses this)'
    asp = _sibling(path, 'master_catalog_asrun.csv', ctx)
    rmp = _sibling(path, 'master_catalog_removed.csv', ctx)
    if not asp or not rmp:
        return False, len(rows), f'as-run or removed record not found (asrun {asp}, removed {rmp})'
    _, asrun = _rows(asp)
    _, removed = _rows(rmp)
    merged = ctx.get('merge_copy') or _stage_info(path).get('merge_copy')
    if merged and os.path.exists(merged):
        if _sha1(merged) != _sha1(asp):
            msgs.append('master_catalog_asrun.csv is not the merge output byte for byte')
    a_ids = [r['cand_id'] for r in asrun]
    c_ids = [r['cand_id'] for r in rows]
    r_ids = [r['cand_id'] for r in removed]
    vm = [c for c in c_ids if c.startswith('VM_') and c not in set(a_ids)]
    if sorted(c_ids + r_ids) != sorted(a_ids + vm):
        msgs.append(f'conformant {len(c_ids)} + removed {len(r_ids)} != as-run {len(a_ids)} + re-admitted {len(vm)}')
    nr = [r['cand_id'] for r in removed if not (r.get('removal_reason') or '').strip()]
    if nr:
        msgs.append(f'{len(nr)} removed rows without removal_reason')
    ex = _excl()
    lat = cfg.catalog.SITE_LAT_DEG['conform']
    floor = cfg.catalog_conform.DEEP_ELEV_FLOOR_DEG
    n_ex = n_el = n_fp = 0
    for r in rows:
        if ex.is_excluded(r['lst'], r['band'], r['date']):
            n_ex += 1
        if tier == 'deep' and not r['cand_id'].startswith('VM_'):
            ra, dec = (_f(r['peak_ra']), _f(r['peak_dec'])) if r.get('peak_ra') else (_f(r['ra']), _f(r['dec']))
            if math.isfinite(ra) and _mid_el(r['lst'], ra, dec, lat) < floor - 1e-9:
                n_el += 1
        if r['stokes'] == 'V' and not (r.get('fp_prediction') or '').strip():
            n_fp += 1
    if n_ex:
        msgs.append(f'{n_ex} conformant rows on excluded epochs')
    if n_el:
        msgs.append(f'{n_el} conformant rows below the deep floor {floor:g} deg at mid-hour')
    if n_fp:
        msgs.append(f'{n_fp} conformant V rows without fp_prediction')
    return _fail(msgs, len(rows), f'{len(rows)} conformant, {len(removed)} removed of {len(asrun)} as-run '
                                  f'(+{len(vm)} re-admitted)')


def _sha1(path):
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


@content_test('frame_table')
def frame_cache(path, sample=20, **ctx):
    """frame_times.csv: date_obs < date_end on every row, one row per key; frame_beam_ratio.csv: BMAJ/BMIN >= 1 on every
    row, one row per key; frame_beam_bmaj.csv (D8, ops 283a513): BMAJ > 0, one row per key; for a seeded sample of keys
    the archived searched frame's header (beside the run) gives the cached values (ops/p3s2_conform.py:106-147, 324-337)."""
    cols, rows = _rows(path)
    name = os.path.basename(path)
    msgs = []
    if not rows:
        return False, 0, 'empty frame cache'
    keys = [(r['stokes'], r['lst'], r['band'], r['date']) for r in rows]
    if len(keys) != len(set(keys)):
        msgs.append(f'{len(keys) - len(set(keys))} duplicate keys')
    times = 'date_obs' in cols
    bmaj = 'bmaj_deg' in cols            # frame_beam_bmaj.csv: the searched frame's BMAJ (ops/p3s2_conform.py, D8, ops 283a513)
    if times:
        bad = [k for k, r in zip(keys, rows) if not (r['date_obs'] < r['date_end'])]
        if bad:
            msgs.append(f'{len(bad)} rows with date_obs >= date_end, e.g. {bad[:2]}')
    elif bmaj:
        bad = [k for k, r in zip(keys, rows) if not (_f(r['bmaj_deg']) > 0)]
        if bad:
            msgs.append(f'{len(bad)} rows without a positive BMAJ, e.g. {bad[:2]}')
    elif 'ratio' in cols:
        bad = [k for k, r in zip(keys, rows) if not (_f(r['ratio']) >= 0.999)]
        if bad:
            msgs.append(f'{len(bad)} rows with BMAJ/BMIN < 1, e.g. {bad[:2]}')
    else:
        return False, None, f'{name}: neither date_obs, bmaj_deg nor ratio columns'
    run = _run_dir(path, ctx)
    n_hdr = 0
    pick = random.Random(20261013).sample(range(len(rows)), min(sample, len(rows)))
    from exopipe.phase3.core.record import fits_primary_cards
    for i in pick:
        s, l, b, d = keys[i]
        fr = os.path.join(run, 'iresid' if s == 'I' else 'vresid', f'{l}_{b}', f'{d}_{s}_redo_resid_{b}.fits')
        if not os.path.exists(fr):
            continue
        h = fits_primary_cards(fr)
        n_hdr += 1
        if times:
            if str(h.get('DATE-OBS', ''))[:19] != rows[i]['date_obs'][:19]:
                msgs.append(f'{keys[i]}: DATE-OBS {h.get("DATE-OBS")} != cached {rows[i]["date_obs"]}')
        elif bmaj:
            if h.get('BMAJ') is None or abs(float(h['BMAJ']) - _f(rows[i]['bmaj_deg'])) > 1e-9 * max(1.0, float(h['BMAJ'])):
                msgs.append(f'{keys[i]}: BMAJ {h.get("BMAJ")} != cached {rows[i]["bmaj_deg"]}')
        elif h.get('BMAJ') and h.get('BMIN'):
            if abs(float(h['BMAJ']) / float(h['BMIN']) - _f(rows[i]['ratio'])) > 1e-3:
                msgs.append(f'{keys[i]}: BMAJ/BMIN {float(h["BMAJ"]) / float(h["BMIN"]):.4f} != cached {rows[i]["ratio"]}')
    return _fail(msgs, len(rows), f'{len(rows)} keys; {n_hdr} of {len(pick)} sampled frames re-read')


@content_test('fitted_catalog')
def fitted_positions(path, **ctx):
    """ops/deep_fit_positions.py:138-159 on content: the DET_COLS present; every row a position kind (gaussian, template,
    peak); ra/dec = pos_ra/pos_dec; fitted rows with finite positive statistical errors; same cand_ids in the same order as
    the stage input, whose ra/dec became peak_ra/peak_dec; fitted rows on a pixel centre at most ON_CENTRE_MAX_FRAC
    (fit_positions_summary.json)."""
    cfg = _cfg()
    cols, rows = _rows(path)
    miss = [c for c in DET_COLS if c not in cols]
    if miss:
        return False, None, f'missing fitted-position columns {miss}'
    msgs, zero = [], []
    nk = {k: 0 for k in KINDS}
    for r in rows:
        k = r.get('pos_kind')
        if k not in nk:
            msgs.append(f"{r['cand_id']}: pos_kind {k!r}")
            continue
        nk[k] += 1
        if r['ra'] != r['pos_ra'] or r['dec'] != r['pos_dec']:
            msgs.append(f"{r['cand_id']}: ra/dec is not the fitted position")
        if k != 'peak':
            a, b = _f(r['pos_stat_maj_arcsec']), _f(r['pos_stat_min_arcsec'])
            if not (a > 0 and b > 0):
                zero.append(f"{r['cand_id']}: {k} fit without positive statistical errors (S/N {r.get('pos_snr')})")
    report = ''
    if zero and cfg.catalog_fitpos.ZERO_STAT_ERROR == 'report':
        report = f'; reported (catalog_fitpos.ZERO_STAT_ERROR report): {len(zero)} {zero[:3]}'
    else:
        msgs += zero
    inp = ctx.get('input') or _stage_info(path).get('input')
    if inp and os.path.exists(inp):
        _, irows = _rows(inp)
        if [r['cand_id'] for r in irows] != [r['cand_id'] for r in rows]:
            msgs.append('cand_ids differ from the stage input (set or order)')
        else:
            # compared as numbers at the writer's precision (deep_forced_radio writes 5 decimals: an input '42.9882' becomes
            # '42.98820'; the 2026-10-14 replay's 26,946 text-only differences were all of that kind)
            def _same(x, y):
                try:
                    return abs(float(x) - float(y)) <= 5e-6
                except (TypeError, ValueError):
                    return x == y
            moved = sum(1 for a, b in zip(irows, rows) if not (_same(a['ra'], b['peak_ra']) and _same(a['dec'], b['peak_dec'])))
            if moved:
                msgs.append(f'{moved} rows whose peak_ra/peak_dec is not the input position')
    sp = _sibling(path, 'fit_positions_summary.json', ctx)
    if sp:
        s = json.load(open(sp))
        nf = int(s.get('fitted_rows') or 0)
        if s.get('refused'):
            msgs.append(f'summary records refusals {s["refused"]}')
        if nf and int(s.get('fitted_on_pixel_centre') or 0) / nf > cfg.catalog_fitpos.ON_CENTRE_MAX_FRAC:
            msgs.append(f'{s["fitted_on_pixel_centre"]} of {nf} fitted rows on a pixel centre')
    else:
        msgs.append('fit_positions_summary.json not found')
    return _fail(msgs, len(rows), f'{len(rows)} rows: ' + ', '.join(f'{k} {v}' for k, v in nk.items()) + report)


def _tier_of(band):
    return 'sb' if 'MHz' in band else 'wb'


@content_test('selection')
def selection_rows(path, **ctx):
    """ops/select_v3.py + ops/p3s2_ephem_filter.py + the split (ops/p3s3_pass.sh:66-76) on content: |SNR| >= SNR_MIN; no
    page-side parent exclusion (parent_excluded 0) while catalog_select.PARENT_EXCLUSION is off; one row per cand_id;
    selected_all.csv: every cand_id in master_catalog.csv with ITS fp_prediction (deep, M8), and the four splits
    partition it exactly ('MHz' in band = subband tier); a split file: its rows are its Stokes and tier and in
    selected_all.csv."""
    cfg = _cfg()
    cols, rows = _rows(path)
    msgs = []
    smin = cfg.catalog_select.SNR_MIN
    low = [r['cand_id'] for r in rows if not abs(_f(r.get('snr'))) >= smin]
    if low:
        msgs.append(f'{len(low)} rows with |SNR| < {smin:g}')
    if not cfg.catalog_select.PARENT_EXCLUSION:
        pe = [r['cand_id'] for r in rows if str(r.get('parent_excluded', '0')) not in ('0', '')]
        if pe:
            msgs.append(f'{len(pe)} rows parent-excluded while parent exclusion is off')
    ids = [r['cand_id'] for r in rows]
    if len(ids) != len(set(ids)):
        msgs.append(f'{len(ids) - len(set(ids))} duplicate cand_ids')
    name = os.path.basename(path)
    m = re.match(r'selected_([IV])_(wb|sb)\.csv$', name)
    d = os.path.dirname(os.path.abspath(path))
    if m:
        st, t = m.groups()
        wrong = [r['cand_id'] for r in rows if r['stokes'] != st or _tier_of(r['band']) != t]
        if wrong:
            msgs.append(f'{len(wrong)} rows not of Stokes {st} tier {t}')
        allp = os.path.join(d, 'selected_all.csv')
        if os.path.exists(allp):
            all_ids = {r['cand_id'] for r in _rows(allp)[1]}
            out = [c for c in ids if c not in all_ids]
            if out:
                msgs.append(f'{len(out)} rows not in selected_all.csv')
        return _fail(msgs, len(rows), f'{len(rows)} rows of {st} {t}')
    # M8 (ops f15c489) is the DEEP pass's verdict input (ops/p3s3_pass.sh:57); the sub-LST chain selects without a verdict
    # file (ops/sublst_catalogs.sh:219, spec Q7: 'pending' on every row), so the comparison is deep-only (catalog-tiers F2)
    tier = ctx.get('tier') or ('sublst' if rows and rows[0].get('cadence') == '10min' else 'deep')
    mc = _sibling(path, cfg.catalog_select.VERDICT_INPUT, ctx) if tier == 'deep' else None
    if mc:
        ver = {r['cand_id']: r.get('fp_prediction', '') for r in _rows(mc)[1]}
        out = [c for c in ids if c not in ver]
        if out:
            msgs.append(f'{len(out)} selected cand_ids not in {os.path.basename(mc)}')
        if 'fp_prediction' in cols:
            dif = [r['cand_id'] for r in rows if r['cand_id'] in ver and r.get('fp_prediction', '') != ver[r['cand_id']]]
            if dif:
                msgs.append(f'{len(dif)} rows whose fp_prediction is not the verdict file\'s (M8)')
    parts = [os.path.join(d, f'selected_{s}_{t}.csv') for s in ('V', 'I') for t in ('wb', 'sb')]
    if all(os.path.exists(p) for p in parts):
        got = sorted(r['cand_id'] for p in parts for r in _rows(p)[1])
        if got != sorted(ids):
            msgs.append(f'the four splits hold {len(got)} rows, selected_all {len(ids)} (not a partition)')
    return _fail(msgs, len(rows), f'{len(rows)} selected')


# ---------------------------------------------------------------------------------------------------------------------
# per-source catalogues (shared with the fork) and the fork
# ---------------------------------------------------------------------------------------------------------------------

def _catalogue_key(d):
    """'V_wb' / 'I_fsb' from source_catalogs[_<build>]_<S>_<t>; None when the name carries no key."""
    m = re.search(r'_([IV])_(f?[sw]b)$', os.path.basename(os.path.normpath(d)))
    return f'{m.group(1)}_{m.group(2)}' if m else None


def _catalogue_checks(d, msgs):
    cfg = _cfg()
    surv = sorted(p for p in glob.glob(os.path.join(d, '*_sources_survivors.csv')) if 'ALL_' not in os.path.basename(p))
    n_src = n_mem = 0
    stokes = set()
    thr = cfg.catalog_associate.SURVIVOR_SNR
    for sp in surv:
        band = os.path.basename(sp).split('_sources_')[0]
        _, rr = _rows(sp)
        n_src += len(rr)
        ids = [r['source_id'] for r in rr]
        if len(ids) != len(set(ids)):
            msgs.append(f'{band}: {len(ids) - len(set(ids))} repeated source_id')
        s = [abs(_f(r.get('best_snr'))) for r in rr]
        if any(not (s[i] >= s[i + 1]) for i in range(len(s) - 1)):
            msgs.append(f'{band}: survivors not sorted by |best_snr|')
        weak = [r['source_id'] for r in rr if not (abs(_f(r.get('best_snr'))) >= thr or (r.get('promotion') or '').strip())]
        if weak:
            msgs.append(f'{band}: {len(weak)} survivors below {thr:g} sigma and unpromoted')
        offsky = [r['source_id'] for r in rr if not (0.0 <= _f(r.get('ra')) < 360.0 and -90.0 <= _f(r.get('dec')) <= 90.0)]
        if offsky:
            msgs.append(f'{band}: {len(offsky)} positions off the sky')
        stokes |= {r['stokes'] for r in rr if r.get('stokes')}
        mp = os.path.join(d, f'{band}_source_members.csv')
        if not os.path.exists(mp):
            msgs.append(f'{band}: no members file')
            continue
        _, mm = _rows(mp)
        n_mem += len(mm)
        has = {m['source_id'] for m in mm}
        lone = [x for x in ids if x not in has]
        if lone:
            msgs.append(f'{band}: {len(lone)} survivors without a member')
        gp = os.path.join(d, f'{band}_sources_removed_gate78.csv')
        if os.path.exists(gp):
            g = [r['source_id'] for r in _rows(gp)[1] if not (7.0 <= abs(_f(r.get('best_snr'))) < 8.0)]
            if g:
                msgs.append(f'{band}: {len(g)} gate-7-8 removals outside 7 <= |SNR| < 8')
    if len(stokes) > 1:
        msgs.append(f'more than one Stokes {sorted(stokes)} (M1)')
    return surv, n_src, n_mem


@content_test('source_catalog')
def source_catalogue_dir(path, **ctx):
    """ops/build_source_catalogs_v3.py output on content: per band, survivors with unique source_id, sorted by
    |best_snr|, each at or above SURVIVOR_SNR or promoted, on the sky, each with >= 1 member; one Stokes per directory
    (M1: the selection the members come from holds one Stokes, and every member is one of its rows); no member on an
    epoch data_decisions excludes now (by the member's selection row); gate-7-8 removals at 7 <= |SNR| < 8."""
    if not os.path.isdir(path):
        return False, None, f'{path} is not a directory'
    msgs = []
    surv, n_src, n_mem = _catalogue_checks(path, msgs)
    if not surv:
        return False, 0, 'no <band>_sources_survivors.csv'
    key = _catalogue_key(path)
    if key and not key.split('_')[1].startswith('f'):
        sel = os.path.join(os.path.dirname(os.path.abspath(path)), f'selected_{key}.csv')
        if os.path.exists(sel):
            _, srows = _rows(sel)
            st = {r['stokes'] for r in srows}
            if st - {key[0]}:
                msgs.append(f'selection {os.path.basename(sel)} holds Stokes {sorted(st)}')
            sid = {r['cand_id']: r for r in srows}
            alien = n_exm = 0
            ex = _excl()
            for mp in glob.glob(os.path.join(path, '*_source_members.csv')):
                for m in _rows(mp)[1]:
                    r = sid.get(m['cand_id'])
                    if r is None:
                        alien += 1
                    elif ex.is_excluded(r['lst'], r['band'], r['date'][:10]):
                        n_exm += 1
            if alien:
                msgs.append(f'{alien} members not in {os.path.basename(sel)}')
            if n_exm:
                msgs.append(f'{n_exm} members on epochs data_decisions excludes now')
    return _fail(msgs, n_src, f'{n_src} sources, {n_mem} member detections in {len(surv)} bands')


@content_test('fork_catalog')
def fork_catalogue_dir(path, **ctx):
    """ops/multiples_fork.py check (:497-577) on content, the file-level part: the catalogue checks above; built with the
    sigma-stack gate off (its build log); repeat_shuffle.csv present; every member a fork row of the repeat class
    (fork_<S>/master_catalog.csv fork_repeat 1); a multi-member source spans >= 2 (LST, night) cells. The re-tabling and
    the re-gating of the 5-7 sigma members are the module's own check, run by the stage (its exit status is recorded)."""
    if not os.path.isdir(path):
        return False, None, f'{path} is not a directory'
    msgs = []
    surv, n_src, n_mem = _catalogue_checks(path, msgs)
    key = _catalogue_key(path)
    if not key or not key.split('_')[1].startswith('f'):
        return False, None, f'{os.path.basename(path)} is not a fork catalogue (source_catalogs_<S>_f<wb|sb>)'
    st, t = key[0], key[2:].lstrip('f')
    run = os.path.dirname(os.path.abspath(path))
    if not os.path.exists(os.path.join(path, 'repeat_shuffle.csv')):
        msgs.append('no repeat_shuffle.csv (shuffle companion)')
    lg = os.path.join(run, f'fork_{st}', f'build_{st}_{t}_final.log')
    if 'sigma-stack singleton gate: off' not in (open(lg).read() if os.path.exists(lg) else ''):
        msgs.append(f'not built with the sigma-stack gate off (or no build log {lg})')
    fm = os.path.join(run, f'fork_{st}', 'master_catalog.csv')
    if os.path.exists(fm):
        by = {r['cand_id']: r for r in _rows(fm)[1]}
        for mp in glob.glob(os.path.join(path, '*_source_members.csv')):
            mem = {}
            for m in _rows(mp)[1]:
                mem.setdefault(m['source_id'], []).append(m)
            for sid, ms in mem.items():
                rows = [by.get(m['cand_id']) for m in ms]
                if any(v is None for v in rows):
                    msgs.append(f'{sid}: a member is not a table-passing fork row')
                    continue
                if any(str(v.get('fork_repeat')) != '1' for v in rows):
                    msgs.append(f'{sid}: a member is not in the repeat class')
                if len(ms) >= 2 and len({(v['lst'], v['date'][:10]) for v in rows}) < 2:
                    msgs.append(f'{sid}: {len(ms)} members from one (LST, night) cell')
                ex = _excl()
                if any(ex.is_excluded(v['lst'], v['band'], v['date'][:10]) for v in rows):
                    msgs.append(f'{sid}: a member on an epoch data_decisions excludes now')
    else:
        msgs.append(f'{fm} not found')
    return _fail(msgs, n_src, f'{n_src} fork sources, {n_mem} member detections')


# ---------------------------------------------------------------------------------------------------------------------
# union, identity, localizations, transient list
# ---------------------------------------------------------------------------------------------------------------------

@content_test('union_catalog')
def union_of_primary_tiers(path, **ctx):
    """The union (ops/p3s3_pass.sh:103-129, D9 fix 09995b5): a 'stokes' column; rows only from the four primary tiers;
    one row per (Stokes, source_id); every row equal to its tier row (identity columns excepted: identity rewrites them
    by (source_id, band) across Stokes); every tier row present; header = every tier column."""
    cfg = _cfg()
    m = re.match(r'(.+)_(sources_survivors|sources_master|source_members)\.csv$', os.path.basename(path))
    if not m:
        return False, None, 'not a union catalogue name <band>_<kind>.csv'
    band, kind = m.groups()
    cols, rows = _rows(path)
    if 'stokes' not in cols:
        return False, len(rows), "no 'stokes' column (served views filter on it)"
    run = os.path.dirname(os.path.dirname(os.path.abspath(path)))
    msgs = []
    seen = set()
    tier_rows = {}
    for K in cfg.catalog_identity.UNION_TIERS:
        if _tier_of(band) != K.split('_')[1]:
            continue
        p = os.path.join(run, f'source_catalogs_{K}', f'{band}_{kind}.csv')
        if os.path.exists(p):
            tc, tr = _rows(p)
            miss = [c for c in tc if c not in cols]
            if miss:
                msgs.append(f'header lacks {K} columns {miss[:4]}')
            first = {}
            for r in tr:
                first.setdefault(r.get('source_id'), r)
            tier_rows[K[0]] = first
    for r in rows:
        st, sid = r.get('stokes'), r.get('source_id')
        if st not in ('V', 'I'):
            msgs.append(f'{sid}: stokes {st!r}')
            continue
        if (st, sid) in seen:
            msgs.append(f'({st}, {sid}) twice')
        seen.add((st, sid))
        t = tier_rows.get(st, {}).get(sid)
        if t is None:
            msgs.append(f'({st}, {sid}) is in no primary {st} tier file')
            continue
        dif = [c for c in t if c not in IDENTITY_COLS and c is not None and (t.get(c) or '') != (r.get(c) or '')]
        if dif:
            msgs.append(f'({st}, {sid}) differs from its tier row in {dif[:3]}')
    for st, first in tier_rows.items():
        lost = [sid for sid in first if (st, sid) not in seen]
        if lost:
            msgs.append(f'{len(lost)} {st} tier rows missing from the union')
    return _fail(msgs, len(rows), f'{len(rows)} union rows of {sorted(tier_rows)}')


_IDENT_CACHE = {}


def identity_groups(run):
    """{(file, row index): identity_id} and {(source_id, band): identity_id} recomputed exactly as ops/p3s3_identity.py
    (single linkage, sep < K x hypot(err_a, err_b), same tier+band never linked, named by the smallest error then the
    largest |best_snr|), over the files it reads. Cached per run directory and file state."""
    cfg = _cfg()
    skip = set(cfg.catalog_identity.IDENTITY_SKIP_DIRS)
    files = [f for f in sorted(glob.glob(os.path.join(run, 'source_catalogs_*', '*_sources_survivors.csv')))
             if os.path.basename(os.path.dirname(f)) not in skip]
    state = tuple((f, os.stat(f).st_mtime_ns, os.stat(f).st_size) for f in files)
    if _IDENT_CACHE.get(run, (None,))[0] == state:
        return _IDENT_CACHE[run][1]
    import numpy as np
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    K = cfg.catalog_identity.IDENTITY_K
    dflt = cfg.catalog_identity.IDENTITY_ERR_DEFAULT_ARCSEC
    rows = []
    for f in files:
        tier = os.path.basename(os.path.dirname(f)).replace('source_catalogs_', '')
        for i, r in enumerate(_rows(f)[1]):
            try:
                rows.append((f, i, tier, r['band'], float(r['ra']), float(r['dec']),
                             float(r.get('pos_err_arcsec') or 0) or dflt, r['source_id'], abs(float(r.get('best_snr') or 0))))
            except ValueError:
                continue
    out = ({}, {})
    if rows:
        c = SkyCoord([r[4] for r in rows] * u.deg, [r[5] for r in rows] * u.deg)
        err = np.array([r[6] for r in rows])
        parent = list(range(len(rows)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        i1, i2, sd, _ = c.search_around_sky(c, K * np.sqrt(2.0) * float(err.max()) / 3600.0 * u.deg)
        for a, b, x in zip(i1, i2, sd.arcsec):
            if a >= b or (rows[a][2] == rows[b][2] and rows[a][3] == rows[b][3]):
                continue
            if x < K * np.hypot(err[a], err[b]):
                ra_, rb_ = find(a), find(b)
                if ra_ != rb_:
                    parent[rb_] = ra_
        groups = {}
        for i in range(len(rows)):
            groups.setdefault(find(i), []).append(i)
        for g in groups.values():
            best = min(g, key=lambda i: (err[i], -rows[i][8]))
            for i in g:
                out[0][(rows[i][0], rows[i][1])] = rows[best][7]
                out[1][(rows[i][7], rows[i][3])] = rows[best][7]
    _IDENT_CACHE[run] = (state, out)
    return out


@content_test('identity_table')
def identity_columns(path, **ctx):
    """ops/p3s3_identity.py:44-104 on content: identity_id / identity_bands / identity_tiers / identity_n on every row
    (identity_n >= 1); and every row's identity_id equals the single-linkage grouping recomputed over the run's tier survivors (union rows by (source_id, band), as written)."""
    cols, rows = _rows(path)
    miss = [c for c in IDENTITY_COLS if c not in cols]
    if miss:
        return False, len(rows), f'missing identity columns {miss}'
    msgs = []
    for r in rows:
        try:
            n = int(r['identity_n'])
        except ValueError:
            n = 0
        if not r['identity_id'] or n < 1:
            msgs.append(f"{r.get('source_id')}: identity_id {r['identity_id']!r} identity_n {r['identity_n']!r}")
    d = os.path.dirname(os.path.abspath(path))
    union = os.path.basename(d) == 'source_catalogs'
    run = os.path.dirname(d)
    n_chk = 0
    if os.path.basename(d).startswith('source_catalogs'):
        by_row, by_key = identity_groups(run)
        if union:
            for r in rows:
                want = by_key.get((r.get('source_id'), r.get('band')))
                if want is not None:
                    n_chk += 1
                    if want != r['identity_id']:
                        msgs.append(f"union {r.get('source_id')}: identity_id {r['identity_id']} != {want}")
        else:
            for i, r in enumerate(rows):
                want = by_row.get((os.path.abspath(path), i)) or by_row.get((path, i))
                if want is not None:
                    n_chk += 1
                    if want != r['identity_id']:
                        msgs.append(f"{r.get('source_id')}: identity_id {r['identity_id']} != recomputed {want}")
    return _fail(msgs, len(rows), f'{len(rows)} rows, {n_chk} checked against the recomputed grouping')


@content_test('localization_table')
def localization_rows(path, **ctx):
    """ops/build_localizations.py:26-77 on content. localizations.csv: one row per identity; err >= err_raw and
    err = err_raw x sqrt(chi2) when chi2 > 1 (to the 0.1-arcsec rounding); RA in [0, 360). localization_offsets.csv: each
    cand_id at most once per identity (M12, ops 43db654); det_sigma_arcsec > 0; every identity in localizations.csv."""
    cols, rows = _rows(path)
    msgs = []
    name = os.path.basename(path)
    if 'offsets' in name:
        pairs = [(r['identity_id'], r['cand_id']) for r in rows]
        if len(pairs) != len(set(pairs)):
            msgs.append(f'{len(pairs) - len(set(pairs))} repeated (identity, cand_id) entries (M12)')
        bad = [r['cand_id'] for r in rows if not _f(r.get('det_sigma_arcsec')) > 0]
        if bad:
            msgs.append(f'{len(bad)} detections without a positive det_sigma_arcsec')
        lp = os.path.join(os.path.dirname(os.path.abspath(path)), 'localizations.csv')
        if os.path.exists(lp):
            ids = {r['identity_id'] for r in _rows(lp)[1]}
            out = {i for i, _ in pairs} - ids
            if out:
                msgs.append(f'{len(out)} identities with offsets but no localization')
        return _fail(msgs, len(rows), f'{len(rows)} detection offsets')
    ids = [r['identity_id'] for r in rows]
    if len(ids) != len(set(ids)):
        msgs.append(f'{len(ids) - len(set(ids))} repeated identities')
    for r in rows:
        e, er, c = _f(r['err_arcsec']), _f(r['err_raw_arcsec']), _f(r.get('chi2'))
        if not e >= er - 0.05:
            msgs.append(f"{r['identity_id']}: err {e} < err_raw {er}")
        if c > 1 and abs(e - er * math.sqrt(c)) > 0.15 + 0.01 * e:
            msgs.append(f"{r['identity_id']}: err {e} != err_raw x sqrt(chi2) {er * math.sqrt(c):.1f}")
        if not 0.0 <= _f(r['ra']) < 360.0:
            msgs.append(f"{r['identity_id']}: RA {r['ra']}")
    return _fail(msgs, len(rows), f'{len(rows)} identities localized')


@content_test('transient_list')
def transient_list_rows(path, **ctx):
    """ops/p3s2_transient_list.py:26-101 (M15, ops 0c1580c) on content: the header's per-tier counts equal the rows; one row
    per (tier, Stokes, source_id) over the eight tier catalogues only; V rows before I rows, each by |best_snr|
    descending; every page link uses its tier's port and identity_id else source_id; the _latest symlink beside it
    resolves to the newest list."""
    cfg = _cfg()
    with open(path, newline='') as f:
        text = f.read().splitlines()
    head = [ln for ln in text if ln.startswith('#')]
    body = [ln for ln in text if not ln.startswith('#')]
    rows = list(csv.DictReader(body))
    msgs = []
    if not rows:
        return False, 0, 'no rows'
    if 'tier' not in rows[0]:
        return False, len(rows), "no 'tier' column (a list written before M15)"
    tiers = cfg.catalog_localize.LIST_TIERS
    ports = cfg.catalog_localize.LIST_PORTS
    cnt = {}
    for ln in head:
        m = re.match(r'# Rows per tier[^:]*:\s*(.*)$', ln)
        if m:
            for k, n in re.findall(r'\b([IV]_f?[sw]b) (\d+)', m.group(1)):
                cnt[k] = int(n)
    got = {}
    keys = set()
    for r in rows:
        got[r['tier']] = got.get(r['tier'], 0) + 1
        k = (r['tier'], r['stokes'], r['source_id'])
        if k in keys:
            msgs.append(f'{k} twice')
        keys.add(k)
        if r['tier'] not in tiers:
            msgs.append(f"tier {r['tier']!r} is not one of the eight tier catalogues")
            continue
        if r['stokes'] != r['tier'][0]:
            msgs.append(f"{r['source_id']}: stokes {r['stokes']} in tier {r['tier']}")
        want = f"http://localhost:{ports[(r['stokes'], '_f' in r['tier'])]}/source/" \
               f"{(r.get('identity_id') or '').strip() or r['source_id']}"
        if r.get('page') != want:
            msgs.append(f"{r['source_id']}: page {r.get('page')} != {want}")
    if not cnt:
        msgs.append('no per-tier count line in the header')
    elif cnt != got:
        msgs.append(f'header counts {cnt} != rows {got}')
    order = [(0 if r['stokes'] == 'V' else 1, -abs(_f(r['best_snr']))) for r in rows]
    if any(order[i] > order[i + 1] for i in range(len(order) - 1)):
        msgs.append('rows not V first then I, each by |best_snr| descending')
    m = re.match(r'(TRANSIENT_LIST_.+)_\d{8}T\d{4}\.csv$', os.path.basename(path))
    if m:
        d = os.path.dirname(os.path.abspath(path))
        lists = sorted(glob.glob(os.path.join(d, m.group(1) + '_*T*.csv')))
        latest = os.path.join(d, m.group(1) + '_latest.csv')
        if lists and os.path.abspath(path) == lists[-1] and os.path.lexists(latest):
            if os.path.realpath(latest) != os.path.realpath(path):
                msgs.append(f'{os.path.basename(latest)} does not point at the newest list')
    return _fail(msgs, len(rows), f'{len(rows)} rows: ' + ', '.join(f'{k} {got[k]}' for k in tiers if k in got))


# ---------------------------------------------------------------------------------------------------------------------
# the class core/validate.OWNERS names for this step
# ---------------------------------------------------------------------------------------------------------------------

@content_test('candidate_catalog')
def candidate_catalogue(path, **ctx):
    """The OWNERS class 'candidate_catalog' (core/validate.py): master_catalog.csv as fitted (pos_kind present), as
    conformant (an as-run copy beside it) or as merged; selected_*.csv as a selection."""
    name = os.path.basename(path)
    if name.startswith('selected_'):
        return selection_rows(path, **ctx)
    with open(path, newline='') as f:
        cols = next(csv.reader(f), [])
    if 'pos_kind' in cols:
        return fitted_positions(path, **ctx)
    if _sibling(path, 'master_catalog_asrun.csv', ctx) and _sibling(path, 'master_catalog_removed.csv', ctx):
        return conformance_record(path, **ctx)
    return merge_gate_record(path, **ctx)
