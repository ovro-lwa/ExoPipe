"""Bright-parent tables (deep, sub-LST) and multiples tables (deep) of `exopipe3 calibrate`.

derive  inputs  freeze the harvest list (published_{V,I}.csv) into the version; deep: a current-master parents union
                (ops/p3s5_parent_union.py IN PLACE into node-local scratch, or --parents); multiples: the per-Stokes exclude
                lists; count harvest rows on excluded (LST, scope, date) (gate G23; calibrate_bright_parent.EXCLUDED_ROWS)
        curves  ops/p3s3_curve_derive.py IN PLACE with production's environment and arguments, sub-runs in parallel:
                deep singles r3 (V I) + r2 (V); multiples I r3, V r3, V r2 (--multiplicity harvest --classes multi);
                sub-LST one run (prefix sublst, cadence 10min)
        build   post_process (deep_tables_elev_20261006.post_process, IN PLACE, locked against config) + the ported key
                loops: deep_tables_elev_20261006.build (:60-87) and deep_tables_pixel_20261001.build (all-elevation keys),
                multiples_fork.tables (:595-619), sublst_install_tables.py (:36-64)
diff    the tables key by key (installed vs candidate) and the conformance table step replayed on the served list with
        both table sets (lib/calibrate/cuts.py): newly removed / kept detections, served identities and judgements touched,
        in-search removals a candidate would keep; sub-LST: the TABLE-NOTES test on the candidate's removal notes
install the installed file with the candidate merged (deep: the candidate keys; multiples: every -multi_ key replaced;
        sub-LST: the whole file) written as a NEW versioned file + a mirror at the production path under --out, a config
        patch and a draft ledger row; never the production file (DESIGN 11.1, 11.4)
verify  the content tests of class bright_parent_table on the installed file (sha256 pin, keys, monotone guard, families,
        installed = derivation key by key) or on a candidate
"""
import json
import os
import re
import shutil

from exopipe.phase3.core.record import Refusal
from exopipe.phase3.lib.calibrate import common as K

KEY_SINGLES_ALL = re.compile(r'^deep_([IV])_([^_]+)$')
KEY_SINGLES_BAND = re.compile(r'^deep_([IV])_([^_]+)_el(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)$')
KEY_MULTI = re.compile(r'^deep_([IV])-multi_([^_]+)(?:_el(\d+)-(\d+))?$')
KEY_SUBLST = re.compile(r'^sublst_([IV])_([^_]+)(?:_el(\d+)-(\d+))?$')
KEY_10MIN = re.compile(r'^10min_([IV])_([^_]+)$')


def family(key):
    """The key family of a table key: deep-all, deep-band, deep-multi, sublst, 10min, or 'other'."""
    if KEY_SINGLES_ALL.match(key):
        return 'deep-all'
    if KEY_SINGLES_BAND.match(key):
        return 'deep-band'
    if KEY_MULTI.match(key):
        return 'deep-multi'
    if KEY_SUBLST.match(key):
        return 'sublst'
    if KEY_10MIN.match(key):
        return '10min'
    return 'other'


def band_label(lo, hi):
    return 'el%g-%g' % (lo, hi)


# ---------------------------------------------------------------------------------------------------------------------
# post-processing: IN PLACE, locked against config
# ---------------------------------------------------------------------------------------------------------------------

_PP = {}


def reference_post_process(rec, guard, cap_on):
    """The post-processing rule written from config (EXCESS_GUARD_CANDIDATES, MONOTONE_CAP) - used ONLY to lock the in-place
    function: the step refuses when the module's behaviour differs from the config values."""
    n = rec.get('n_candidates') or 0
    bins = sorted(rec.get('bins', []), key=lambda b: -b['S_lo'])
    out, cap = [], None
    for b in bins:
        R = float(b.get('R2') or 0.0)
        if R > 0 and b.get('marginal_excess', 0.0) * n < guard:
            R = 0.0
        if cap_on and cap is not None and R > cap:
            R = cap
        if R > 0:
            cap = R
            out.append([b['S_lo'], b['S_hi'], R])
    return sorted(out)


LOCK_RECORD = {'n_candidates': 100, 'bins': [
    {'S_lo': 200, 'S_hi': 1e6, 'R2': 3.0, 'marginal_excess': 0.25},     # 25 >= 20: kept
    {'S_lo': 100, 'S_hi': 200, 'R2': 4.0, 'marginal_excess': 0.25},     # larger than the brighter bin: capped to 3.0
    {'S_lo': 60, 'S_hi': 100, 'R2': 2.0, 'marginal_excess': 0.199},     # 19.9 < 20: dropped
    {'S_lo': 35, 'S_hi': 60, 'R2': 1.5, 'marginal_excess': 0.2},        # 20.0 >= 20: kept
    {'S_lo': 20, 'S_hi': 35, 'R2': 0.0, 'marginal_excess': 0.5}]}       # no radius


def post_process():
    """deep_tables_elev_20261006.post_process, imported in place (pure), after the lock check."""
    if 'fn' in _PP:
        return _PP['fn']
    C = K.CFG().calibrate_bright_parent
    mod = K.import_inplace(os.path.splitext(K.CFG().calibrate.LEGACY_SCRIPTS['post_process'])[0])
    fn = mod.post_process
    got = fn(json.loads(json.dumps(LOCK_RECORD)))
    want = reference_post_process(LOCK_RECORD, C.EXCESS_GUARD_CANDIDATES, C.MONOTONE_CAP)
    if K.canon(got) != K.canon(want):
        raise Refusal(f'lock: the in-place post_process ({mod.__file__}) gives {got} on the lock record, config '
                      f'(EXCESS_GUARD_CANDIDATES={C.EXCESS_GUARD_CANDIDATES}, MONOTONE_CAP={C.MONOTONE_CAP}) gives {want} - '
                      f'the config does not describe the code it runs; refused')
    _PP['fn'] = fn
    return fn


def _fmt(tab):
    return ' '.join(f'S {a:g}-{b:g}: {r:.2f}' for a, b, r in tab) or '(none: falls back to the all-elevation table)'


# ---------------------------------------------------------------------------------------------------------------------
# key loops (ported, pure)
# ---------------------------------------------------------------------------------------------------------------------

def _require_meta(D, src, multiplicity=None):
    m = D.get('meta') or {}
    if m.get('metric') != 'image-pixel' or not m.get('elevation_bands'):
        raise Refusal(f'{src} was not derived in the image-pixel metric with elevation bands (gate G6/G7)')
    if multiplicity and m.get('multiplicity') != multiplicity:
        raise Refusal(f'{src}: multiplicity {m.get("multiplicity")!r}, expected {multiplicity!r} (gate G7)')


def build_deep(D3, D2, src3, src2, inst_tables, build_keys, need_meta=True):
    """The deep singles candidate. 'elevation-bands' = deep_tables_elev_20261006.build (:60-87); 'all-elevation' =
    deep_tables_pixel_20261001.build. Loops over the INSTALLED all-elevation singles keys deep_<S>_<band> (the production
    loops also met -multi_ keys and band keys in today's file; those never produce a singles key - section 4 of the note).
    Returns ({key: record}, report lines)."""
    C = K.CFG().calibrate_bright_parent
    pp = post_process()
    if need_meta:
        for D, src in ((D3, src3), (D2, src2)):
            if 'elevation-bands' in build_keys:
                _require_meta(D, src)
            elif (D.get('meta') or {}).get('metric') != 'image-pixel':
                raise Refusal(f'{src} was not derived in the image-pixel metric (gate G6)')
    low = set(C.RATIO_LOW_V_BANDS)
    bands = [tuple(K.num(x) for x in b) for b in C.ELEV_BANDS_INSTALL]
    report_only = [tuple(K.num(x) for x in b) for b in C.ELEV_BANDS_REPORT_ONLY['deep']]
    T = {}
    L = [f'ELEVATION-BANDED DEEP BRIGHT-SOURCE TABLES - rejection radius in beam widths per parent-brightness bin '
         f'(S = parent peak / frame noise), {K.utcnow()[:16]}Z', '']
    for k in sorted(x for x in inst_tables if KEY_SINGLES_ALL.match(x)):
        _, S, b = k.split('_')
        D, src, ratio = (D2, src2, f'{C.RATIO_LOW_V}') if (S == 'V' and b in low) else (D3, src3, f'{C.RATIO["deep"]}')
        ins = inst_tables[k]
        if 'all-elevation' in build_keys:
            if ratio != str(ins.get('ratio')):
                raise Refusal(f'{k} installed at ratio {ins.get("ratio")}, the rebuild would use {ratio}')
            rec = D['curves'].get(k)
            if rec is None:
                raise Refusal(f'{k} missing from {src}')
            T[k] = dict(stokes=S, band=b, ratio=ratio, metric='image-pixel', table=pp(rec), raw_table=rec.get('table', []),
                        n_candidates=rec.get('n_candidates'), n_cells=rec.get('n_cells'), provisional=False,
                        derived_utc=D['meta'].get('derived_utc'), source=os.path.basename(src), parents=ins.get('parents'),
                        beam_unit=ins.get('beam_unit'))
            L.append(f'{k} (ratio {ratio}) all elevations: {_fmt(T[k]["table"])}')
        if 'elevation-bands' in build_keys:
            L.append(f'{k} (ratio {ratio})')
            if 'all-elevation' not in build_keys:
                L.append(f'   all elevations (installed, unchanged): {_fmt(ins["table"])}')
            for lo, hi in report_only + bands:
                kk = f'{k}_{band_label(lo, hi)}'
                rec = D['curves'].get(kk)
                tab = pp(rec) if rec else []
                tag = 'REPORT ONLY' if (lo, hi) in report_only else ('candidate' if tab else 'no table')
                L.append(f'   {lo:g}-{hi:g} deg [{tag}; {rec.get("n_candidates") if rec else 0} detections]: {_fmt(tab)}')
                if tab and (lo, hi) in bands:
                    T[kk] = dict(stokes=S, band=b, ratio=ratio, metric='image-pixel', table=tab, raw_table=rec.get('table', []),
                                 n_candidates=rec.get('n_candidates'), n_cells=rec.get('n_cells'), provisional=False,
                                 elevation_band=[lo, hi], derived_utc=D['meta'].get('derived_utc'),
                                 source=os.path.basename(src), parents=ins.get('parents'), beam_unit=ins.get('beam_unit'))
    L.append(f'\n{len(T)} candidate {"band " if build_keys == ("elevation-bands",) else ""}tables written to')   # + the path (derive_build)
    return T, L


def build_multiples(R3I, R3V, R2, srcs, exclude_cells):
    """multiples_fork.tables key loop (:586-619): every -multi_ key, 20-25 deg report-only, an empty all-elevation table
    installed (nothing removed). srcs = (r3_I, r3_V, r2V) paths; exclude_cells = {S: path}."""
    C = K.CFG()
    B, M = C.calibrate_bright_parent, C.calibrate_multiples
    pp = post_process()
    for D, src in zip((R3I, R3V, R2), srcs):
        _require_meta(D, src, multiplicity=M.MULTIPLICITY)
    R3 = dict(meta=R3I['meta'], curves={**{k: v for k, v in R3I['curves'].items() if k.startswith('deep_I-')},
                                        **{k: v for k, v in R3V['curves'].items() if k.startswith('deep_V-')}})
    low = set(B.RATIO_LOW_V_BANDS)
    report_lo = {str(K.num(b[0])) for b in B.ELEV_BANDS_REPORT_ONLY['deep']}
    Cd, L = {}, []
    for k in sorted(set(R3['curves']) | set(R2['curves'])):
        mm = KEY_MULTI.match(k)
        if not mm:
            continue
        S, b, lo, hi = mm.group(1), mm.group(2), mm.group(3), mm.group(4)
        if lo in report_lo:
            continue
        D, ratio = (R2, f'{M.RATIO_LOW_V}') if (S == 'V' and b in low) else (R3, f'{M.RATIO[S]}')
        rec = D['curves'].get(k)
        if not rec:
            continue
        tab = pp(rec)
        L.append(f'{k} (ratio {ratio}; {rec.get("n_candidates")} detections, {rec.get("n_cells")} cells): '
                 + (' '.join(f'S {a:g}-{c:g}: {r:.2f}' for a, c, r in tab) or 'none'))
        if tab or (lo is None and M.INSTALL_EMPTY_ALL_ELEVATION):
            src = srcs[2] if D is R2 else srcs[0 if S == 'I' else 1]
            Cd[k] = dict(stokes=f'{S}-multi', band=b, ratio=ratio, metric='image-pixel', table=tab,
                         raw_table=rec.get('table', []), n_candidates=rec.get('n_candidates'), n_cells=rec.get('n_cells'),
                         provisional=False, elevation_band=([int(lo), int(hi)] if lo else None),
                         derived_utc=D['meta'].get('derived_utc'), source=os.path.basename(src),
                         parents=D['meta'].get('parents'), beam_unit=D['meta'].get('beam_unit'),
                         derived_utc_class=D['meta'].get('derived_utc'), excluded_cells=exclude_cells.get(S, ''),
                         class_definition=M.CLASS_DEFINITION)
    return Cd, L


def build_sublst(D, src):
    """sublst_install_tables.py (:36-64): an all-elevation table for every (Stokes, band) of SUBLST_BANDS (empty when no
    harvest), band tables only where a bin survives the guard. Returns (file object without installed_utc, report)."""
    B = K.CFG().calibrate_bright_parent
    pp = post_process()
    Cc = D.get('curves') or {}
    metric = (D.get('meta') or {}).get('metric', 'sky')
    if any((c.get('metric') or metric) != metric for c in Cc.values()):
        raise Refusal(f'{src} mixes distance metrics across its curves (gate G8)')
    if not any(k.startswith(B.KEY_PREFIX['sublst'] + '_') for k in Cc):
        raise Refusal(f'{src} holds no {B.KEY_PREFIX["sublst"]}_* key (derive with P3_TABLE_PREFIX=sublst; gate G8)')
    elevb = [band_label(*b) for b in B.ELEV_BANDS_INSTALL]
    T = {'meta': dict(source=src, derivation=D['meta'],
                      rule='all-elevation table per (Stokes, band); elevation tables only where a bin survives the '
                           '20-candidate excess guard, else fallback to the all-elevation table (Gregg 2026-09-27)'),
         'tables': {}}
    rep = []
    pre = B.KEY_PREFIX['sublst']
    for S in ('I', 'V'):
        for b in B.SUBLST_BANDS:
            k = f'{pre}_{S}_{b}'
            rec = Cc.get(k)
            tab = pp(rec) if rec else []
            T['tables'][k] = dict(stokes=S, band=b, metric=metric, table=tab, n_candidates=(rec or {}).get('n_candidates', 0),
                                  note=('measured' if rec else 'no harvest for this band - nothing removed'))
            line = f'{k}: all-elevation {len(tab)} bin(s), n={(rec or {}).get("n_candidates", 0)}'
            for e in elevb:
                ke = f'{k}_{e}'
                re_ = Cc.get(ke)
                te = pp(re_) if re_ else []
                if te:
                    T['tables'][ke] = dict(stokes=S, band=b, elevation=e, metric=metric, table=te,
                                           n_candidates=re_.get('n_candidates', 0))
                    line += f'; {e} {len(te)} bin(s) n={re_.get("n_candidates", 0)}'
                else:
                    line += f'; {e} fallback (n={(re_ or {}).get("n_candidates", 0)})'
            rep.append(line)
    return T, rep


def merge(cls, tier, base, cand):
    """The installed file after the install of a candidate (a NEW object; base is not modified):
    bright-parent deep: the candidate keys replace / add; multiples: every -multi_ key replaced; sub-LST: the candidate's
    tables replace the whole file. Returns (merged, replaced keys, removed keys)."""
    M = K.CFG().calibrate_multiples
    new = json.loads(json.dumps(base))
    ct = cand['tables']
    removed = []
    if tier == 'sublst':
        removed = sorted(set(base.get('tables', {})) - set(ct))
        new = {'meta': dict(installed_utc=K.utcnow(), source=cand['meta'].get('source'), derivation=cand['meta'].get('derivation'),
                            rule=cand['meta'].get('rule')), 'tables': json.loads(json.dumps(ct))}
        return new, sorted(ct), removed
    if cls == 'multiples':
        if any(not KEY_MULTI.match(k) for k in ct):
            raise Refusal('multiples candidate holds a key outside the -multi_ family')
        if M.REPLACE_ALL_MULTI_KEYS:
            removed = sorted(k for k in new['tables'] if '-multi_' in k and k not in ct)
            for k in [k for k in new['tables'] if '-multi_' in k]:
                del new['tables'][k]
    else:
        bad = [k for k in ct if family(k) not in ('deep-all', 'deep-band')]
        if bad:
            raise Refusal(f'bright-parent deep candidate holds keys outside the deep singles family: {bad[:5]}')
    for k, v in ct.items():
        new['tables'][k] = v
    return new, sorted(ct), removed


# ---------------------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------------------

def installed_path(tier):
    T = K.CFG().tables
    return T.BRIGHT_PARENT_DEEP if tier == 'deep' else T.BRIGHT_PARENT_SUBLST


def installed_pin(tier):
    T = K.CFG().tables
    return T.BRIGHT_PARENT_DEEP_SHA256 if tier == 'deep' else T.BRIGHT_PARENT_SUBLST_SHA256


def check_pin(path, tier, what='installed table'):
    sha = K.sha256(K.need_file(path, what))
    if os.path.realpath(path) == os.path.realpath(installed_path(tier)) and sha != installed_pin(tier):
        raise Refusal(f'{what} {path}: sha256 {sha} != the pinned tables.{"BRIGHT_PARENT_DEEP" if tier == "deep" else "BRIGHT_PARENT_SUBLST"}'
                      f'_SHA256 {installed_pin(tier)} - the file changed outside `calibrate install` (gate G19)')
    return sha


def _sublst_names(lst, arm):
    hh = lst.rstrip('h')
    sfx = arm if lst == '19h' else f'{lst}_{arm}'
    return hh, sfx


def harvest_label(ctx):
    B = ctx.C.calibrate_bright_parent
    if ctx.args.run_label:
        return ctx.args.run_label
    if ctx.tier == 'deep':
        return B.HARVEST_LABEL['deep']
    lst = sublst_lst(ctx)
    return B.HARVEST_LABEL['sublst'].format(hh=lst.rstrip('h'), arm=B.SUBLST_ARM)


def sublst_lst(ctx):
    from exopipe.phase3.core import cells
    lsts = cells.expand(ctx.args.lsts, universe=cells.LSTS, kind='LST') if ctx.args.lsts else []
    if len(lsts) != 1:
        raise Refusal('bright-parent --tier sublst derives from ONE LST arm: give --lsts <LST> (today one LST per derivation, '
                      'sublst_catalogs.sh:84)')
    return cells.parse_lst(lsts[0])


def sublst_env(ctx, state_dir, lst=None):
    """The environment sublst_catalogs.sh:24-30 exports for the LST arm, plus the tables stage's own (:87-90)."""
    C = ctx.C
    B = C.calibrate_bright_parent
    lst = lst or sublst_lst(ctx)
    arm = B.SUBLST_ARM
    hh, sfx = _sublst_names(lst, arm)
    SL, F = C.paths.SUBLST_ROOT, B.SUBLST_CATALOGS_ROOT
    foot = f'{SL}/catalog_inputs/bdsf_catalogs_aug' + ('' if lst == '19h' else f'_{lst}')
    label = ctx.args.run_label or B.HARVEST_LABEL['sublst'].format(hh=hh, arm=arm)
    return {'P3S_LABEL': label, 'P3S_RUN_DIR': f'{F}/catalog_{sfx}', 'P3S3_STATE_DIR': state_dir,
            'P3S3_COLLECT_ROOTS': f'{F}/collect_{sfx}', 'P3_CADENCE': B.CADENCE['sublst'], 'P3_TIER': B.CADENCE['sublst'],
            'P3_TENMIN_ROOT': f'{SL}/{B.SUBLST_TENMIN_TREE[arm]}', 'P3_COADD_ROOT': f'{SL}/coadd_{lst}',
            'P3_PARENT_DIR': f'{SL}/' + B.PARENTS_DIR['sublst'].format(lst=lst), 'P3S2_ROOTS_CSV': f'{SL}/catalog_inputs/roots_{lst}.csv',
            'P3_FOOTPRINT_CAT_DIR': foot, 'P3_TABLE_PREFIX': B.KEY_PREFIX['sublst'],
            'P3_CURVE_ELEV_BANDS': B.ELEV_BANDS_DERIVE['sublst'], 'P3S3_CURVE_RATIO': str(B.RATIO['sublst']),
            'P3_GATE_METRIC': B.METRIC['sublst']}


def _published(state_dir):
    rows = []
    for S in ('V', 'I'):
        p = os.path.join(state_dir, f'published_{S}.csv')
        if os.path.exists(p):
            for ln in open(p):
                if ln.strip():
                    rows.append((S,) + tuple(ln.strip().split(',')[:3]))
    return rows


# ---------------------------------------------------------------------------------------------------------------------
# derive: inputs
# ---------------------------------------------------------------------------------------------------------------------

def derive_inputs(ctx):
    C, B = ctx.C, ctx.C.calibrate_bright_parent
    state_src = ctx.args.harvest_state
    if not state_src:
        if ctx.tier == 'deep':
            state_src = B.HARVEST_STATE_DIR['deep']
        else:
            _, sfx = _sublst_names(sublst_lst(ctx), B.SUBLST_ARM)
            state_src = B.HARVEST_STATE_DIR['sublst'].format(sfx=sfx)
    need = [os.path.join(state_src, f'published_{S}.csv') for S in stokes_of(ctx)]
    missing = [p for p in need if not (os.path.exists(p) and os.path.getsize(p) > 0)]
    plan = {'harvest_state': state_src, 'published': need, 'missing': missing}
    if ctx.tier == 'deep':
        plan['parents'] = ctx.args.parents or '(fresh current-master union: p3s5_parent_union.py into node-local scratch)'
    if ctx.cls == 'multiples':
        plan['exclude_cells'] = dict(C.calibrate_multiples.EXCLUDE_CELLS)
    if ctx.dry:
        return plan
    if missing:
        raise Refusal(f'harvest list missing or empty: {missing}' + (' (no collect layout: run the sub-LST catalogue chain prep '
                                                                    'first; sublst_catalogs.sh:83)' if ctx.tier == 'sublst' else ''))
    if ctx.skip_if_current('derive-inputs'):
        return None
    d = K.fresh_stage_dir(ctx.stage_dir('inputs'), ctx.args.fresh)
    with ctx.run.unit(ctx.unit_name('derive-inputs'), cls=ctx.cls, tier=ctx.tier, version=ctx.version, stage='inputs') as u:
        sdir = os.path.join(d, 'state')
        os.makedirs(sdir)
        files = {}
        for p in need:
            u.add_input(p)
            dst = os.path.join(sdir, os.path.basename(p))
            shutil.copy2(p, dst)
            files[dst] = K.sha256(dst)
        man = {'class': ctx.cls, 'tier': ctx.tier, 'version': ctx.version, 'harvest_state_source': state_src,
               'harvest_label': harvest_label(ctx), 'state_dir': sdir, 'files': files, 'stokes': list(stokes_of(ctx))}
        # parents (deep): a fresh current-master union, or --parents
        if ctx.tier == 'deep':
            if ctx.args.parents:
                man['parents_dir'] = K.need_file(ctx.args.parents, '--parents')
            else:
                man['parents_dir'] = parents_union(ctx, u, sdir_harvest=sdir)
        else:
            man['parents_dir'] = sublst_env(ctx, sdir)['P3_PARENT_DIR']
        man['parents_gen'] = add_parent_inputs(u, man['parents_dir'])
        # multiples: per-Stokes exclude lists frozen
        if ctx.cls == 'multiples':
            man['exclude_cells'] = {}
            for S, p in C.calibrate_multiples.EXCLUDE_CELLS.items():
                u.add_input(K.need_file(p, f'calibrate_multiples.EXCLUDE_CELLS[{S}]'))
                dst = os.path.join(d, f'exclude_{S}.txt')
                shutil.copy2(p, dst)
                files[dst] = K.sha256(dst)
                man['exclude_cells'][S] = {'frozen': dst, 'source': p}
        if ctx.args.exclude_cells or (ctx.tier == 'deep' and ctx.args.cells):
            man['subset_exclude_file'] = subset_exclude(ctx, d)
            files[man['subset_exclude_file']] = K.sha256(man['subset_exclude_file'])
        # exclusions (gate G23): count harvest rows on excluded (LST, scope, date)
        n, by = count_excluded_rows(ctx, sdir, man)
        man['excluded_rows'] = {'n': n, 'by_cell': by, 'policy': B.EXCLUDED_ROWS}
        from exopipe.phase3.core import exclusions
        ex = exclusions.load()
        u.add_input(C.paths.DATA_DECISIONS, extra=ex.fingerprint())
        u.note(f'harvest rows on excluded (LST, scope, date): {n} (policy {B.EXCLUDED_ROWS})')
        if n and B.EXCLUDED_ROWS == 'refuse':
            raise Refusal(f'{n} harvest rows lie on excluded (LST, scope, date) and calibrate_bright_parent.EXCLUDED_ROWS = '
                          f"'refuse' (gate G23)")
        mp = K.write_json(os.path.join(d, 'inputs_manifest.json'), man)
        u.meta['parents_dir'] = man['parents_dir']
        u.add_output(mp, 'calibration_inputs')
    return None


def stokes_of(ctx):
    want = [s.strip() for s in (ctx.args.stokes or ','.join(ctx.C.calibrate_bright_parent.STOKES)).split(',') if s.strip()]
    bad = [s for s in want if s not in ('I', 'V')]
    if bad:
        raise ValueError(f'--stokes {ctx.args.stokes}: I and/or V')
    return tuple(s for s in ('V', 'I') if s in want)


def subset_exclude(ctx, d):
    """--cells / --exclude-cells (deep) -> one P3_DERIVE_EXCLUDE_CELLS file (p3s3_curve_derive.py:82 has only an exclude
    list); a subset derivation is recorded and never installed (gate G24)."""
    from exopipe.phase3.core import cells
    allc = set(cells.all_cells())
    keep = set(cells.expand_cells(ctx.args.cells)) if ctx.args.cells else allc
    excl = (allc - keep) | (set(cells.expand_cells(ctx.args.exclude_cells)) if ctx.args.exclude_cells else set())
    p = os.path.join(d, 'subset_exclude_cells.txt')
    K.write_text(p, ''.join(c + '\n' for c in sorted(excl)))
    return p


def parents_union(ctx, u, sdir_harvest):
    """ops/p3s5_parent_union.py IN PLACE into node-local scratch (chain.sh:16); gate G3: a harvest cell with no catalogue on
    its current master refuses (chain.sh only logged the misses)."""
    from exopipe.phase3.core import fitsio
    B = ctx.C.calibrate_bright_parent
    script = K.ops_script('parent_union')
    src = open(script).read()
    for p in B.PARENT_UNION_SOURCES:      # lock: the in-place script's literal list
        if repr(p) not in src and f"'{p}'" not in src:
            raise Refusal(f'lock: {script} no longer lists the parent source {p} (calibrate_bright_parent.PARENT_UNION_SOURCES)')
    root = fitsio.scratch_root(ctx.args.work)
    out = os.path.join(root, f'calibrate_parents_union_{ctx.version}')
    if os.path.exists(out):
        raise Refusal(f'{out} exists (a union of this version): pass --parents {out} to reuse it, or remove it')
    log = os.path.join(ctx.stage_dir('inputs'), 'parent_union.log')
    res = K.run_legacy(u, script, [out], log=log)
    import ast
    miss_line = [ln for ln in res.tail.splitlines() if ' without: ' in ln]
    misses = []
    if miss_line:
        try:
            misses = ast.literal_eval(miss_line[-1].split(' without: ', 1)[1].strip())
        except (ValueError, SyntaxError):
            raise RuntimeError(f'p3s5_parent_union.py: unparseable miss list: {miss_line[-1][:300]}')
    harvest_cells = {r[1] for r in _published(sdir_harvest)}
    hit = sorted({c for c, _why in misses} & harvest_cells)
    u.meta['parents_union_misses'] = misses
    if hit:
        raise Refusal(f'{len(hit)} harvest cell(s) have no parent catalogue on their current master: {hit[:10]} '
                      f'(build them with masters --stages sources; gate G3)')
    return out


def add_parent_inputs(u, parents_dir):
    """Fingerprint the parent catalogues by their .gen.json sidecars (each names the md5 of the master it was made on; spec
    7.2 item 13): a changed union makes the derivation's record stale. Returns the number of sidecars."""
    import glob
    gens = sorted(glob.glob(os.path.join(parents_dir, '*.gen.json'))) if parents_dir and os.path.isdir(parents_dir) else []
    for p in gens:
        u.add_input(p)
    return len(gens)


def count_excluded_rows(ctx, sdir, man):
    """Harvest rows with |snr| >= floor on an excluded (LST, scope, date) - read from the published attempts' gate ledgers
    (p3s3_sources in place, the attempts the derivation reads). Returns (n, {cell: n})."""
    import csv
    from exopipe.phase3.core import exclusions
    B = ctx.C.calibrate_bright_parent
    ex = exclusions.load()
    env = {'P3S_LABEL': man['harvest_label'], 'P3S3_STATE_DIR': sdir}
    if ctx.tier == 'deep':
        env['P3S3_COLLECT_ROOTS'] = ':'.join(B.COLLECT_ROOTS)
    else:
        env['P3S3_COLLECT_ROOTS'] = sublst_env(ctx, sdir)['P3S3_COLLECT_ROOTS']
    PS = K.import_inplace('p3s3_sources', env=env, reload=True)
    n, by = 0, {}
    for a in PS.published_attempts(list(stokes_of(ctx))):
        p = f"{a['dir']}/transients/Run_{PS.LABEL}/v_deep_ledger.csv"
        if not os.path.exists(p):
            continue
        with open(p, newline='') as f:
            for r in csv.DictReader(f):
                try:
                    if abs(float(r['snr'])) < B.FLOOR_SIGMA:
                        continue
                except (KeyError, ValueError):
                    continue
                if ex.is_excluded(r['lst'], r.get('scope') or r.get('band'), str(r.get('date', ''))[:10]):
                    n += 1
                    by[a['cell']] = by.get(a['cell'], 0) + 1
    return n, by


# ---------------------------------------------------------------------------------------------------------------------
# derive: curves (in place)
# ---------------------------------------------------------------------------------------------------------------------

def curve_jobs(ctx, man, d):
    """[(output basename, env, argv)] of the in-place derivation sub-runs, exactly as production's drivers run them."""
    C, B, M = ctx.C, ctx.C.calibrate_bright_parent, ctx.C.calibrate_multiples
    sdir = man.get('state_dir') or os.path.dirname(next(iter(man['files'])))
    st = stokes_of(ctx)
    jobs = []
    if ctx.tier == 'sublst':
        lock_tenmin_floor()
        env = sublst_env(ctx, sdir)
        name = f'sublst_tables_{K.utcstamp()[:-1]}.json'
        jobs.append((name, env, ['--by', B.BY, '--floor', f'{B.FLOOR_SIGMA:g}', '--stokes'] + list(st)))
        return jobs
    base = {'P3S_LABEL': man['harvest_label'], 'P3S3_STATE_DIR': sdir, 'P3_PARENT_DIR': man['parents_dir'],
            'P3_GATE_METRIC': B.METRIC['deep'], 'P3_GATE_TARGET_BEAM': '1' if B.TARGET_BEAM else '0',
            'P3_DEEP_MIN_ELEVATION': f'{B.ELEV_FLOOR_DEG["deep"]:g}', 'P3S3_COLLECT_ROOTS': ':'.join(B.COLLECT_ROOTS),
            'P3_CADENCE': B.CADENCE['deep']}
    sub = man.get('subset_exclude_file')
    if ctx.cls == 'bright-parent':
        e = dict(base, P3_CURVE_ELEV_BANDS=B.ELEV_BANDS_DERIVE['deep'], P3_DERIVE_EXCLUDE_CELLS=sub)
        jobs.append(('deep_elev_r3.json', dict(e, P3S3_CURVE_RATIO=f'{B.RATIO["deep"]:g}'),
                     ['--by', B.BY, '--floor', f'{B.FLOOR_SIGMA:g}', '--stokes'] + list(st)))
        if 'V' in st:
            jobs.append(('deep_elev_r2V.json', dict(e, P3S3_CURVE_RATIO=f'{B.RATIO_LOW_V:g}'),
                         ['--by', B.BY, '--floor', f'{B.FLOOR_SIGMA:g}', '--stokes', 'V']))
        return jobs
    # multiples (derive_conformant.sh:5-11): one derivation per Stokes and ratio, the per-Stokes exclude list
    e = dict(base, P3_CURVE_ELEV_BANDS=M.ELEV_BANDS_DERIVE)
    marg = ['--multiplicity', M.MULTIPLICITY, '--classes', M.CLASSES]
    if M.MATCH_BEAMS != 1.0:
        marg += ['--match-beams', f'{M.MATCH_BEAMS:g}']
    for S in ('I', 'V'):
        if S not in st:
            continue
        ex = _join_excludes(man['exclude_cells'][S]['frozen'], sub, d, S)
        jobs.append((f'multi_r3_{S}.json', dict(e, P3S3_CURVE_RATIO=f'{M.RATIO[S]:g}', P3_DERIVE_EXCLUDE_CELLS=ex),
                     ['--by', B.BY, '--floor', f'{B.FLOOR_SIGMA:g}', '--stokes', S] + marg))
    if 'V' in st:
        ex = _join_excludes(man['exclude_cells']['V']['frozen'], sub, d, 'V')
        jobs.append(('multi_r2V.json', dict(e, P3S3_CURVE_RATIO=f'{M.RATIO_LOW_V:g}', P3_DERIVE_EXCLUDE_CELLS=ex),
                     ['--by', B.BY, '--floor', f'{B.FLOOR_SIGMA:g}', '--stokes', 'V'] + marg))
    return jobs


def lock_tenmin_floor():
    """Lock: the 10-minute elevation floor of the in-place derivation is the literal gate_inputs.TENMIN_ELEV_FLOOR_DEG
    (gate_inputs.py:94; not settable): it must equal calibrate_bright_parent.ELEV_FLOOR_DEG['sublst']."""
    p = os.path.join(K.ops_root(), 'gate_inputs.py')
    m = re.search(r'^TENMIN_ELEV_FLOOR_DEG\s*=\s*([0-9.]+)', open(p).read(), re.M)
    want = K.CFG().calibrate_bright_parent.ELEV_FLOOR_DEG['sublst']
    if not m or float(m.group(1)) != float(want):
        raise Refusal(f'lock: {p} TENMIN_ELEV_FLOOR_DEG = {m.group(1) if m else "?"} != calibrate_bright_parent.'
                      f"ELEV_FLOOR_DEG['sublst'] = {want} (the in-place derivation cannot be given another floor)")


def _join_excludes(frozen, subset, d, S):
    if not subset:
        return frozen
    p = os.path.join(d, f'exclude_{S}_with_subset.txt')
    cells_ = sorted({ln.strip() for f in (frozen, subset) for ln in open(f) if ln.strip()})
    K.write_text(p, ''.join(c + '\n' for c in cells_))
    return p


def _planned_manifest(ctx):
    """--dry-run before the inputs stage exists: the manifest the inputs stage would write (paths only)."""
    idir = ctx.stage_dir('inputs')
    man = {'files': {os.path.join(idir, 'state', f'published_{S}.csv'): '' for S in stokes_of(ctx)},
           'harvest_label': harvest_label(ctx),
           'parents_dir': ctx.args.parents or ('<a fresh current-master union in node-local scratch>' if ctx.tier == 'deep'
                                               else sublst_env(ctx, '')['P3_PARENT_DIR'])}
    if ctx.cls == 'multiples':
        man['exclude_cells'] = {S: {'frozen': os.path.join(idir, f'exclude_{S}.txt')} for S in ('I', 'V')}
    return man


def derive_curves(ctx):
    d = ctx.stage_dir('curves')
    rin = K.load_record(ctx.out, ctx.unit_name('derive-inputs'))
    if ctx.dry and not (rin and rin.get('status') == 'done'):
        jobs = curve_jobs(ctx, _planned_manifest(ctx), d)
        return {'after': 'the inputs stage of this version', 'script': K.ops_script('curve_derive'),
                'jobs': [{'out': os.path.join(d, n), 'env': e, 'argv': a} for n, e, a in jobs]}
    rin = K.require_done(ctx.out, ctx.unit_name('derive-inputs'), 'derive curves')
    man_p = K.outputs_of(rin, 'calibration_inputs')[0]
    man = K.read_json(man_p)
    jobs = curve_jobs(ctx, man, d)
    if ctx.dry:
        return {'script': K.ops_script('curve_derive'), 'jobs': [{'out': os.path.join(d, n), 'env': e, 'argv': a} for n, e, a in jobs]}
    if ctx.skip_if_current('derive-curves'):
        return None
    K.fresh_stage_dir(d, ctx.args.fresh)
    script = K.ops_script('curve_derive')
    with ctx.run.unit(ctx.unit_name('derive-curves'), cls=ctx.cls, tier=ctx.tier, version=ctx.version, stage='curves') as u:
        u.add_input(man_p)
        for f in man['files']:
            u.add_input(f)
        u.add_input(script)
        add_parent_inputs(u, man['parents_dir'])

        def one(job):
            name, env, argv = job
            outp = os.path.join(d, name)
            K.run_legacy(u, script, [outp] + argv, env=env, log=os.path.join(d, name.replace('.json', '.log')))
            if not (os.path.exists(outp) and os.path.getsize(outp) > 0):
                raise RuntimeError(f'{script} wrote no {outp}')
            return outp
        outs = K.parallel([lambda j=j: one(j) for j in jobs], ctx.workers(ctx.C.calibrate.DERIVE_PROCESSES))
        u.meta['jobs'] = [{'out': os.path.join(d, n), 'env': e, 'argv': a} for n, e, a in jobs]
        u.meta['subset'] = bool(man.get('subset_exclude_file')) or tuple(stokes_of(ctx)) != ('V', 'I')
        u.meta['exclude_cells'] = {S: v['frozen'] for S, v in (man.get('exclude_cells') or {}).items()}
        for o in outs:
            u.add_output(o, 'bright_parent_curves')
    return None


# ---------------------------------------------------------------------------------------------------------------------
# derive: build
# ---------------------------------------------------------------------------------------------------------------------

def _curves_files(ctx):
    """The derivation files the build reads: --curves (explicit, e.g. production's own derivation for a parity build) or the
    outputs of this version's curves record. Returns (paths in the order the class needs, subset flag, exclude map)."""
    names = {'bright-parent': {'deep': ('deep_elev_r3.json', 'deep_elev_r2V.json'), 'sublst': None},
             'multiples': {'deep': ('multi_r3_I.json', 'multi_r3_V.json', 'multi_r2V.json')}}[ctx.cls][ctx.tier]
    if ctx.args.curves:
        paths = [p.strip() for p in ctx.args.curves.split(',') if p.strip()]
        want = 1 if names is None else len(names)
        if len(paths) != want:
            raise ValueError(f'--curves needs {want} file(s) for {ctx.cls} {ctx.tier}: '
                             + ('the sub-LST derivation JSON' if names is None else ', '.join(names)))
        for p in paths:
            K.need_file(p, '--curves')
        excl = dict(ctx.C.calibrate_multiples.EXCLUDE_CELLS) if ctx.cls == 'multiples' else {}
        return paths, False, excl, None
    rec = K.require_done(ctx.out, ctx.unit_name('derive-curves'), 'derive build')
    outs = K.outputs_of(rec, 'bright_parent_curves')
    if names is None:
        if len(outs) != 1:
            raise Refusal(f'derive build: the curves record lists {len(outs)} derivations (expected 1)')
        paths = outs
    else:
        by = {os.path.basename(p): p for p in outs}
        miss = [n for n in names if n not in by]
        if miss:
            raise Refusal(f'derive build: the curves record lacks {miss} (a --stokes subset cannot build {ctx.cls})')
        paths = [by[n] for n in names]
    return paths, bool((rec.get('meta') or {}).get('subset')), (rec.get('meta') or {}).get('exclude_cells') or {}, rec


def derive_build(ctx):
    B = ctx.C.calibrate_bright_parent
    d = ctx.stage_dir('build')
    if ctx.dry:
        plan = {'installed': installed_path(ctx.tier), 'build_keys': B.BUILD_KEYS.get(ctx.tier)}
        if not ctx.args.curves and 'curves' in ctx.stages:
            # the curves stage runs EARLIER IN THIS INVOCATION: the build reads the derivations that stage writes and lists
            # in its record (never a glob), so the plan names them instead of refusing on a record that does not exist yet
            names = {'bright-parent': {'deep': ('deep_elev_r3.json', 'deep_elev_r2V.json'), 'sublst': ('sublst_tables_<UTC>.json',)},
                     'multiples': {'deep': ('multi_r3_I.json', 'multi_r3_V.json', 'multi_r2V.json')}}[ctx.cls][ctx.tier]
            return dict(plan, after='the curves stage of this version (earlier in this invocation)',
                        derivations=[os.path.join(ctx.stage_dir('curves'), n) for n in names])
        try:
            paths = _curves_files(ctx)[0]
        except (Refusal, ValueError) as e:
            paths = [f'(not yet: {e})']
        return dict(plan, derivations=paths)
    if ctx.skip_if_current('derive-build'):
        return None
    paths, subset, excl, _rec = _curves_files(ctx)
    K.fresh_stage_dir(d, ctx.args.fresh)
    with ctx.run.unit(ctx.unit_name('derive-build'), cls=ctx.cls, tier=ctx.tier, version=ctx.version, stage='build') as u:
        for p in paths:
            u.add_input(p)
        inst_p = ctx.args.installed or installed_path(ctx.tier)
        u.add_input(inst_p)
        base_meta = {'class': ctx.cls, 'tier': ctx.tier, 'version': ctx.version, 'derivations': paths,
                     'derivation_sha256': {p: K.sha256(p) for p in paths}, 'subset': subset,
                     'installed_at_build': {'path': inst_p, 'sha256': K.sha256(inst_p)}}
        if ctx.cls == 'bright-parent' and ctx.tier == 'deep':
            inst = K.read_json(inst_p)['tables']
            D3, D2 = (K.read_json(p) for p in paths)
            T, L = build_deep(D3, D2, paths[0], paths[1], inst, B.BUILD_KEYS['deep'])
            cand = {'meta': dict(built_utc=K.utcnow(), sources=paths, calibrate=dict(base_meta, build_keys=list(B.BUILD_KEYS['deep']))),
                    'tables': T}
            cp, rp = os.path.join(d, 'deep_elev_candidate.json'), os.path.join(d, 'deep_elev_report.txt')
        elif ctx.cls == 'multiples':
            R3I, R3V, R2 = (K.read_json(p) for p in paths)
            T, L = build_multiples(R3I, R3V, R2, paths, excl)
            cand = {'meta': dict(built_utc=K.utcnow(), sources=paths, kind='MULTIPLES tables deep_<S>-multi_<band>[_el..]',
                                 calibrate=base_meta), 'tables': T}
            cp, rp = os.path.join(d, 'multi_candidate.json'), os.path.join(d, 'multi_report_installed.txt')
        else:
            D = K.read_json(paths[0])
            cand, L = build_sublst(D, paths[0])
            cand['meta']['built_utc'] = K.utcnow()
            cand['meta']['calibrate'] = base_meta
            cp, rp = os.path.join(d, 'parent_curve_tables_sublst.json'), os.path.join(d, 'sublst_tables_report.txt')
        if not cand['tables']:
            raise Refusal(f'the build produced no table (derivations {paths})')
        if L and L[-1].endswith('written to'):
            L[-1] += f' {cp}'
        K.write_text(rp, '\n'.join(L) + '\n')
        K.write_json(cp, cand)
        u.note(f'{len(cand["tables"])} candidate tables; report {rp}')
        u.meta['candidate_sha256'] = K.sha256(cp)
        u.add_output(cp, 'bright_parent_table')
    return None


# ---------------------------------------------------------------------------------------------------------------------
# candidate resolution (diff / install)
# ---------------------------------------------------------------------------------------------------------------------

def candidate_path(ctx):
    if ctx.args.candidate:
        return K.need_file(ctx.args.candidate, '--candidate')
    rec = K.require_done(ctx.out, ctx.unit_name('derive-build'), f'{ctx.action}')
    outs = K.outputs_of(rec, 'bright_parent_table')
    if len(outs) != 1:
        raise Refusal(f'{ctx.action}: the build record lists {len(outs)} candidates')
    return outs[0]


# ---------------------------------------------------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------------------------------------------------

def table_diff(base_tables, merged_tables, keys):
    rows = []
    for k in sorted(keys):
        a, b = base_tables.get(k), merged_tables.get(k)
        ta = K.canon(a['table']) if a else None
        tb = K.canon(b['table']) if b else None
        ch = 'added' if a is None else ('removed' if b is None else ('same' if ta == tb else 'changed'))
        rows.append({'key': k, 'family': family(k), 'change': ch, 'installed_table': json.dumps(ta),
                     'candidate_table': json.dumps(tb), 'n_candidates': (b or a or {}).get('n_candidates', '')})
    return rows


def _diff_list(ctx, run_dir):
    if ctx.args.list:
        return [K.need_file(ctx.args.list, '--list')]
    if ctx.cls == 'multiples':
        ps = [os.path.join(run_dir, f'fork_{S}', 'master_catalog.csv') for S in stokes_of(ctx)]
        return [K.need_file(p, 'multiples diff list (the fork rows)') for p in ps]
    for n in ctx.C.calibrate.DIFF_LIST_CSV:
        p = os.path.join(run_dir, n)
        if os.path.exists(p):
            return [p]
    raise Refusal(f'diff: no detection list {ctx.C.calibrate.DIFF_LIST_CSV} in {run_dir} (pass --list)')


def _gate_inputs(ctx, parents, lst=None):
    """gate_inputs imported in place with the parents of the derivation (P3_PARENT_DIR read at import: reloaded). Sub-LST:
    the LST's own coadd, parents and frame tree (sublst_catalogs.sh:28-29), one import per LST of the list."""
    if ctx.tier == 'sublst':
        e = sublst_env(ctx, '', lst=lst)
        env = {k: e[k] for k in ('P3_TENMIN_ROOT', 'P3_COADD_ROOT', 'P3_PARENT_DIR')}
    else:
        env = {'P3_PARENT_DIR': parents, 'P3_COADD_ROOT': ctx.C.paths.COADD_SB_ROOT,
               'P3_DEEP_MIN_ELEVATION': f'{ctx.C.calibrate_bright_parent.ELEV_FLOOR_DEG["deep"]:g}'}
    return K.import_inplace('gate_inputs', env=env, reload=True), env


def served_replay(ctx, base, merged, parents, run_dir, lists, d, u):
    """Part 2 of the diff: the conformance table step on the served list with the installed and the candidate tables.
    Deep: one gate_inputs import (the derivation's parents). Sub-LST: the tables are global (keys per Stokes and band), so
    every LST of the list is replayed, each with its own gate inputs (coadd, parents, frame tree of that LST)."""
    from exopipe.phase3.lib.calibrate import cuts
    from exopipe.phase3.lib.search import parent_curve_table as P
    C = ctx.C
    lat = C.calibrate.OVRO_LAT_DEG_LEGACY
    cadence = C.calibrate_bright_parent.CADENCE[ctx.tier]
    prefix = C.calibrate_bright_parent.KEY_PREFIX['sublst'] if ctx.tier == 'sublst' else None
    suffix = '-multi' if ctx.cls == 'multiples' else ''
    rows = []
    for lp in lists:
        u.add_input(lp)
        rows += K.read_csv(lp)
    st = set(stokes_of(ctx))
    rows = [r for r in rows if r.get('stokes') in st]
    rms = cuts.rms_by_cell(rows)
    if ctx.tier == 'sublst':           # the sub-LST file is merged over the deep file (parent_curve_table.tables, P3_TABLE_EXTRA)
        deep = K.read_json(ctx.C.tables.BRIGHT_PARENT_DEEP)['tables']
        old_t, new_t = dict(deep, **base['tables']), dict(deep, **merged['tables'])
    else:
        old_t, new_t = base['tables'], merged['tables']
    mode = ctx.args.diff_mode
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault(r['lst'] if ctx.tier == 'sublst' else '', []).append(i)
    old, new, envs = {}, {}, {}
    for lst, idx in sorted(groups.items()):
        G, genv = _gate_inputs(ctx, parents, lst=lst or None)
        envs[lst or ctx.tier] = genv
        sub = [rows[i] for i in idx]
        o = cuts.cuts(sub, rms, old_t, P, G, cadence=cadence, elev_on=(mode != 'elevation'), table_class_suffix=suffix,
                      prefix=prefix, lat_deg=lat)
        n = cuts.cuts(sub, rms, new_t, P, G, cadence=cadence, elev_on=True, table_class_suffix=suffix, prefix=prefix,
                      lat_deg=lat)
        for j, i in enumerate(idx):
            old[i], new[i] = o[j], n[j]
    u.meta['gate_inputs_env'] = envs
    tiers = ('_fwb', '_fsb') if ctx.cls == 'multiples' else C.calibrate.SERVED_CATALOG_TIERS.get(ctx.tier, ('',))
    served = cuts.served_map(run_dir, tiers, stokes=tuple(st))
    judg = cuts.judgements(run_dir, C.calibrate.JUDGEMENTS, stokes=tuple(st))
    out_rows, tally = cuts.compare(rows, old, new, served, judg, cadence=cadence, lat_deg=lat)
    rp = os.path.join(d, 'rows.csv')
    K.write_csv(rp, ['cand_id', 'stokes', 'lst', 'band', 'date', 'ra', 'dec', 'snr', 'elev_deg', 'elev_band', 'change',
                     'served_identity', 'judged', 'installed_note', 'band_note'], out_rows)
    ids = {}
    for x in out_rows:
        if x['served_identity'] and x['change'] == 'newly removed':
            ids.setdefault(f"{x['stokes']}:{x['served_identity']}", []).append(x['judged'] or 'unjudged')
    summ = {'list': lists, 'n_rows': len(rows), 'removed_installed': sum(c for c, _ in old.values()),
            'removed_candidate': sum(c for c, _ in new.values()), 'changed': len(out_rows), 'rows_csv': rp,
            'tally': {'|'.join(k): v for k, v in sorted(tally.items())}, 'served_tiers': list(tiers),
            'served_sources_checked': len(served) > 0, 'n_served_members': len(served), 'n_judged': len(judg),
            'served_identities_losing': {k: v[0] for k, v in sorted(ids.items())}, 'mode': mode}
    # what changes for the served sources (served_change_20261006.py, band-gate part): gone / reduced / unchanged per source,
    # the judged ones listed; the night-rejected and re-searched fates of that script belonged to the P3S4 -> P3S5 hand-over
    src, jud, leave, jrows = cuts.source_fates(rows, old, new, served, cuts.judged_labels(run_dir, C.calibrate.JUDGEMENTS,
                                                                                       stokes=tuple(st)))
    sp_ = os.path.join(d, 'served_sources_judged.csv')
    K.write_csv(sp_, ['stokes', 'source', 'label', 'status', 'detections', 'band_removed'], jrows)
    summ['served_sources'] = {'by_status': {'|'.join(k): v for k, v in sorted(src.items())},
                              'judged_by_status': {'|'.join(k): v for k, v in sorted(jud.items())},
                              'member_detections_leaving': dict(sorted(leave.items())), 'judged_csv': sp_}
    if ctx.cls == 'bright-parent' and ctx.tier == 'deep':
        summ['returning'] = returning(ctx, run_dir, rms, new_t, P, parents, lat)
    if ctx.tier == 'sublst':
        n, bad, ex = cuts.table_notes(rows, new)
        summ['table_notes'] = {'n': n, 'bad': bad, 'example': ex}
        u.check('calibrate.diff.table_notes', bad == 0, value=f'{n} removals, {bad} bad',
                detail=f'every candidate removal note names an elevation band and (image-pixel) (sublst_catalogs.sh:187-201){"; e.g. " + ex if bad else ""}')
    # in-search removals of the published gate ledgers that the candidate would keep (deep singles; :168-182)
    label = ctx.args.ledger_label or C.calibrate.DIFF_LEDGER_LABEL.get(ctx.tier)
    if ctx.cls == 'bright-parent' and ctx.tier == 'deep' and label:
        G, _ = _gate_inputs(ctx, parents)
        summ['in_search'] = in_search_kept(ctx, label, rms, new_t, P, G, lat)
    return summ


def returning(ctx, run_dir, rms, new_t, P, parents, lat):
    """Detections the installed gate removed at the served pass (<run>/master_catalog_removed.csv, a removal reason naming the
    table) that the candidate gate keeps: they return to the list (served_change_20261006.py:76-81). Returns counts by Stokes."""
    import collections
    from exopipe.phase3.lib.calibrate import cuts
    p = os.path.join(run_dir, 'master_catalog_removed.csv')
    if not os.path.exists(p):
        return {'list': None, 'n_table_removed': 0, 'returning': {}}
    st = set(stokes_of(ctx))
    rem = [r for r in K.read_csv(p) if 'table' in (r.get('removal_reason') or r.get('reason') or '') and r.get('stokes') in st]
    G, _ = _gate_inputs(ctx, parents)
    ret = cuts.cuts(rem, rms, new_t, P, G, cadence='deep', elev_on=True, lat_deg=lat) if rem else {}
    n = collections.Counter(r['stokes'] for i, r in enumerate(rem) if not ret[i][0])
    return {'list': p, 'n_table_removed': len(rem), 'returning': dict(sorted(n.items()))}


def in_search_kept(ctx, label, rms, new_t, P, G, lat):
    import collections
    import csv
    from exopipe.phase3.lib.calibrate import cuts
    rin = K.load_record(ctx.out, ctx.unit_name('derive-inputs'))
    env = {'P3S_LABEL': label}
    if rin and rin.get('status') == 'done':
        man = K.read_json(K.outputs_of(rin, 'calibration_inputs')[0])
        env['P3S3_STATE_DIR'] = os.path.dirname(next(iter(man['files'])))
    PS = K.import_inplace('p3s3_sources', env=env, reload=True)
    led = []
    for a in PS.published_attempts(['V', 'I']):
        p = f"{a['dir']}/transients/Run_{label}/v_deep_ledger.csv"
        if os.path.exists(p):
            with open(p, newline='') as f:
                for r in csv.DictReader(f):
                    if (r.get('veto_reason') or '').startswith('table') and r.get('ra'):
                        led.append(dict(lst=r['lst'], band=r.get('scope') or r.get('band'), stokes=r['stokes'], date=r['date'],
                                        ra=r['ra'], dec=r['dec']))
    keep = cuts.cuts(led, rms, new_t, P, G, cadence='deep', elev_on=True, lat_deg=lat)
    kept = [led[i] for i, (c, _) in keep.items() if not c]
    kt = collections.Counter(f"{x['stokes']}|{cuts.eband(float(cuts.elev_mid_hour(x['lst'], float(x['ra']), float(x['dec']), lat)))}"
                             for x in kept)
    return {'state_dir': env.get('P3S3_STATE_DIR', '(p3s3_sources default)'), 'n_ledger_table_removals': len(led),
            'n_kept_by_candidate': len(kept), 'kept_by_stokes_band': dict(sorted(kt.items()))}


def diff(ctx):
    C = ctx.C
    d = ctx.stage_dir('diff')
    run_dir = ctx.args.served_run or C.calibrate.SERVED_RUN.get(ctx.tier)
    if ctx.dry:
        try:
            cp = candidate_path(ctx)
        except Refusal as e:
            cp = f'(not yet: {e})'
        return {'candidate': cp, 'installed': ctx.args.installed or installed_path(ctx.tier), 'served_run': run_dir,
                'served_replay': not ctx.args.table_only, 'mode': ctx.args.diff_mode}
    if ctx.skip_if_current('diff'):
        return None
    cp = candidate_path(ctx)
    inst_p = ctx.args.installed or installed_path(ctx.tier)
    K.fresh_stage_dir(d, ctx.args.fresh)
    with ctx.run.unit(ctx.unit_name('diff'), cls=ctx.cls, tier=ctx.tier, version=ctx.version, stage='diff') as u:
        u.add_input(cp)
        u.add_input(inst_p)
        check_pin(inst_p, ctx.tier)
        cand, base = K.read_json(cp), K.read_json(inst_p)
        merged, replaced, removed = merge(ctx.cls, ctx.tier, base, cand)
        keys = set(replaced) | set(removed)
        if ctx.tier == 'sublst':
            keys |= set(base.get('tables', {}))
        rows = table_diff(base['tables'], merged['tables'], keys)
        tp = os.path.join(d, 'tables_diff.csv')
        K.write_csv(tp, ['key', 'family', 'change', 'installed_table', 'candidate_table', 'n_candidates'], rows)
        summ = {'class': ctx.cls, 'tier': ctx.tier, 'version': ctx.version, 'candidate': cp,
                'candidate_sha256': K.sha256(cp), 'installed': inst_p, 'installed_sha256': K.sha256(inst_p),
                'tables_csv': tp, 'tables': {}, 'served': None}
        for r in rows:
            summ['tables'][r['change']] = summ['tables'].get(r['change'], 0) + 1
        if ctx.args.table_only:
            u.note('served replay NOT run (--table-only)')
            u.check('calibrate.diff.served', False, value='not run',
                    detail='--table-only: no served replay; install needs this waived with Gregg\'s ruling')
        else:
            parents = ctx.args.parents
            if not parents:
                rin = K.load_record(ctx.out, ctx.unit_name('derive-inputs'))
                parents = (rin or {}).get('meta', {}).get('parents_dir') if rin and rin.get('status') == 'done' else None
            if not parents and ctx.tier == 'deep':
                raise Refusal('diff: no parents directory (no derive inputs record of this version): pass --parents '
                              '<current-master union> (gate G3)')
            summ['served'] = served_replay(ctx, base, merged, parents, run_dir, _diff_list(ctx, run_dir), d, u)
            summ['served_run'] = run_dir
        sp = K.write_json(os.path.join(d, 'diff_summary.json'), summ)
        u.meta['candidate_sha256'] = summ['candidate_sha256']
        u.add_output(sp, 'calibration_diff')
    return None


# ---------------------------------------------------------------------------------------------------------------------
# install (window: mirror + versioned file + patch + draft ledger row under --out)
# ---------------------------------------------------------------------------------------------------------------------

def versioned_name(tier, version):
    base = os.path.basename(installed_path(tier))
    stem, ext = os.path.splitext(base)
    return f'{stem}.{version}{ext}'


def install(ctx):
    C = ctx.C
    d = ctx.stage_dir('install')
    inst_p = installed_path(ctx.tier)
    if ctx.dry:
        try:
            cp = candidate_path(ctx)
        except Refusal as e:
            cp = f'(not yet: {e})'
        return {'candidate': cp, 'installed': inst_p, 'mirror': os.path.join(d, 'mirror', inst_p.lstrip('/')),
                'versioned': os.path.join(C.calibrate.VERSIONED_TABLES_ROOT or d, versioned_name(ctx.tier, ctx.version)),
                'patch': os.path.join(d, 'config_patch.diff'), 'ledger_draft': os.path.join(d, 'ledger_row_draft.csv')}
    if ctx.skip_if_current('install'):
        return None
    rid = K.check_ruling(ctx.args)
    cp = candidate_path(ctx)
    cand_sha = K.sha256(cp)
    diff_rec = K.diff_gate(ctx.out, ctx.cls, ctx.tier, ctx.version, cand_sha, installed_sha=K.sha256(K.need_file(inst_p, 'installed table')))
    cand = K.read_json(cp)
    if (cand.get('meta') or {}).get('calibrate', {}).get('subset'):
        raise Refusal('install: the candidate was derived from a subset (--cells / --exclude-cells / --stokes): never '
                      'installed as production (gate G24)')
    K.fresh_stage_dir(d, ctx.args.fresh)
    with ctx.run.unit(ctx.unit_name('install'), cls=ctx.cls, tier=ctx.tier, version=ctx.version, stage='install') as u:
        u.add_input(cp)
        u.add_input(inst_p)
        base_sha = check_pin(inst_p, ctx.tier)
        base = K.read_json(inst_p)
        merged, replaced, removed = merge(ctx.cls, ctx.tier, base, cand)
        if ctx.tier == 'deep':
            M = ctx.C.calibrate_bright_parent
            entry = dict(source=cp, metric='image-pixel', measured=len(replaced), backup=f'{inst_p} (sha256 {base_sha})',
                         ruling=rid + (f' "{ctx.args.install_words}"' if ctx.args.install_words else ''),
                         calibrate_version=ctx.version)
            if ctx.cls == 'multiples':
                entry.update(kind='MULTIPLES tables deep_<S>-multi_<band>[_el..]',
                             elevation_bands='25-35,35-50,50-90 + all-elevation (20-25 report-only)')
            else:
                entry.update(elevation_bands='25-35,35-50,50-90 (20-25 report-only; all-elevation fallback)')
            entry.update(ratio=f'{M.RATIO["deep"]} (V 18-36 MHz {M.RATIO_LOW_V})', floor=M.FLOOR_SIGMA)
            merged.setdefault('meta_deep', {})[K.utcnow()] = entry
        mirror = os.path.join(d, 'mirror', inst_p.lstrip('/'))
        K.write_json(mirror, merged)
        vroot = d
        if C.calibrate.VERSIONED_TABLES_ROOT and C.paths.PRODUCTION_WRITES:
            vroot = C.calibrate.VERSIONED_TABLES_ROOT
        versioned = os.path.join(vroot, versioned_name(ctx.tier, ctx.version))
        shutil.copyfile(mirror, versioned)
        tkey = 'BRIGHT_PARENT_DEEP' if ctx.tier == 'deep' else 'BRIGHT_PARENT_SUBLST'
        rep = K.pointer_replacements(inst_p, versioned, inst_p, versioned)
        patch, changed = K.config_patch(rep)
        pp = K.write_text(os.path.join(d, 'config_patch.diff'), patch)
        ds = K.read_json(K.outputs_of(K.load_record(ctx.out, ctx.unit_name('diff')), 'calibration_diff')[0]) if diff_rec else {}
        desc = (f'calibrate {ctx.cls} --tier {ctx.tier} install {ctx.version}: candidate {cp} (sha256 {cand_sha}) merged into '
                f'{inst_p} (sha256 {base_sha}) -> {versioned} (sha256 {K.sha256(versioned)}); {len(replaced)} keys replaced, '
                f'{len(removed)} removed; diff {diff_rec} (tables {ds.get("tables")}, served changed '
                f'{(ds.get("served") or {}).get("changed")}); config patch {pp} changes {", ".join(f"{s}.{k}" for s, k in changed)}')
        lp = os.path.join(d, 'ledger_row_draft.csv')
        K.ledger_draft(lp, ctx.cls, ctx.tier, ctx.version, [versioned, pp], desc, rid, ctx.args.install_words)
        man = {'class': ctx.cls, 'tier': ctx.tier, 'version': ctx.version, 'family_rule': ctx.cls if ctx.tier == 'deep' else 'sublst',
               'candidate': {'path': cp, 'sha256': cand_sha}, 'base': {'path': inst_p, 'sha256': base_sha},
               'mirror': {'path': mirror, 'sha256': K.sha256(mirror)}, 'versioned': {'path': versioned, 'sha256': K.sha256(versioned)},
               'config_key': f'tables.{tkey}', 'patch': pp, 'patch_changes': list(dict.fromkeys(f'{s}.{k}' for s, k in changed)),
               'ledger_draft': lp, 'diff_record': diff_rec, 'replaced': replaced, 'removed': removed,
               'ruling': {'id': rid, 'words': ctx.args.install_words}, 'production_writes': bool(C.paths.PRODUCTION_WRITES),
               'code_literals': [f'{a} ({b})' for a, b in K.code_literals([inst_p] + list(K.digests(inst_p).values()))]}
        mp = K.write_json(os.path.join(d, 'install_manifest.json'), man)
        u.note(f'mirror {mirror}; versioned {versioned}; patch {pp} ({len(changed)} keys); ledger draft {lp}')
        u.add_output(mp, 'install_bundle')
    return None


# ---------------------------------------------------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------------------------------------------------

def verify_targets(ctx):
    """[(path, label, product class)]: --candidate, else the installed file of the tier (sha256 pin, content tests)."""
    if ctx.args.candidate:
        return [(ctx.args.candidate, 'candidate', 'bright_parent_table')]
    return [(installed_path(ctx.tier), 'installed', 'bright_parent_table')]
