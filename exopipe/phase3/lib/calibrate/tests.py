"""Content tests of every product class `exopipe3 calibrate` writes (discovered by core/validate; `exopipe3 verify <class>`).

Each test reads the file's CONTENT and, where the class allows it, RECOMPUTES the product from its inputs (a sidecar
<file>.provenance.json or, for the installed production file, the tables.*_INPUTS / *_DERIVATIONS keys of config):

  calibration_inputs        files (every frozen input listed with its sha256), harvest (published lines parse)
  bright_parent_curves      meta_form (metric, floor, ratio, elevation bands, multiplicity as config), bins_sane
  bright_parent_table       keys_form, metric, monotone_guard, sha_pinned, sublst_present (G21), equals_derivation (G9)
  calibration_diff          reconciles (summary totals = the rows; sub-LST TABLE-NOTES)
  install_bundle            files_sha, mirror_equals (G9, G10), patch_names, ledger_draft
  vlssr_ratios              columns_rows
  scale_table               rows, recompute (scale_table_v4.py arithmetic from config)
  shflux_stagea             rows_status
  band_vector               norm, converged
  fluxcorr_table            structure, recompute (g, k x g, n, chi, the shcells rows, the version)
  template_k_trials         columns, status_counts
  template_k_record         recompute, bounds, interval, orientation
  calibration_verify        results_pass
A test returns (passed, value, detail).
"""
import json
import math
import os

from exopipe.phase3.core.validate import content_test


def _K():
    from exopipe.phase3.lib.calibrate import common as K
    return K


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def _same_path(a, b):
    return bool(a and b) and os.path.realpath(a) == os.path.realpath(b)


# ---------------------------------------------------------------------------------------------------------------------
# calibration_inputs
# ---------------------------------------------------------------------------------------------------------------------

@content_test('calibration_inputs')
def files(path, **ctx):
    K = _K()
    man = K.read_json(path)
    bad = [p for p, h in man.get('files', {}).items() if not os.path.exists(p) or K.sha256(p) != h]
    return not bad and bool(man.get('files')), len(man.get('files', {})), (f'changed or missing: {bad[:3]}' if bad else
                                                                           'every frozen input has its recorded sha256')


@content_test('calibration_inputs')
def harvest(path, **ctx):
    from exopipe.phase3.core import cells
    man = _K().read_json(path)
    n, bad = 0, []
    for p in man.get('files', {}):
        if not os.path.basename(p).startswith('published_'):
            continue
        for ln in open(p):
            if not ln.strip():
                continue
            n += 1
            f = ln.strip().split(',')
            try:
                cells.parse_cell(f[0])
                ok = len(f) >= 3 and all(f[:3])
            except ValueError:
                ok = man.get('tier') == 'sublst' and len(f) >= 3   # sub-LST attempts name <lst>_<band> too; tolerate
            if not ok:
                bad.append(ln.strip()[:60])
    return n > 0 and not bad, n, (f'unparseable lines: {bad[:3]}' if bad else f'{n} published attempts')


# ---------------------------------------------------------------------------------------------------------------------
# bright_parent_curves (the in-place derivation's JSON)
# ---------------------------------------------------------------------------------------------------------------------

def _curve_kind(D):
    m = D.get('meta') or {}
    multi = any('-multi_' in k for k in D.get('curves', {}))
    tier = 'sublst' if m.get('cadence') == '10min' else 'deep'
    return tier, multi


@content_test('bright_parent_curves')
def meta_form(path, **ctx):
    C = _cfg()
    B, M = C.calibrate_bright_parent, C.calibrate_multiples
    D = _K().read_json(path)
    m = D.get('meta') or {}
    tier, multi = _curve_kind(D)
    want_eb = M.ELEV_BANDS_DERIVE if multi else B.ELEV_BANDS_DERIVE[tier]
    ratios = ({M.RATIO['I'], M.RATIO['V'], M.RATIO_LOW_V} if multi else
              ({B.RATIO['deep'], B.RATIO_LOW_V} if tier == 'deep' else {B.RATIO['sublst']}))
    errs = []
    if m.get('metric') != B.METRIC[tier]:
        errs.append(f"metric {m.get('metric')!r} != {B.METRIC[tier]!r}")
    if m.get('elevation_bands') != want_eb:
        errs.append(f"elevation_bands {m.get('elevation_bands')!r} != {want_eb!r}")
    if m.get('floor') != B.FLOOR_SIGMA:
        errs.append(f"floor {m.get('floor')} != {B.FLOOR_SIGMA}")
    if m.get('ratio') not in ratios:
        errs.append(f"ratio {m.get('ratio')} not in {sorted(ratios)}")
    if multi and m.get('multiplicity') != M.MULTIPLICITY:
        errs.append(f"multiplicity {m.get('multiplicity')!r} != {M.MULTIPLICITY!r}")
    if any((c.get('metric') or m.get('metric')) != m.get('metric') for c in D.get('curves', {}).values()):
        errs.append('mixed metric across curves')
    return not errs, len(D.get('curves', {})), '; '.join(errs) or f'{tier}{" multiples" if multi else ""}: metric, floor, ratio, bands as config'


@content_test('bright_parent_curves')
def bins_sane(path, **ctx):
    D = _K().read_json(path)
    bad = []
    for k, c in D.get('curves', {}).items():
        bins = c.get('bins', [])
        if [b['S_lo'] for b in bins] != sorted(b['S_lo'] for b in bins):
            bad.append(f'{k}: bins not sorted')
        for b in bins:
            if not b['S_lo'] < b['S_hi']:
                bad.append(f'{k}: S_lo >= S_hi')
            r = float(b.get('R2') or 0)
            if r < 0 or r > 40.0 + 1e-9 or abs(r / 0.25 - round(r / 0.25)) > 1e-9:
                bad.append(f'{k}: R2 {r} not on the 0.25-bw grid in [0, 40]')
        u = c.get('union_kill_fraction')
        if u is not None and not 0 <= u <= 1:
            bad.append(f'{k}: union_kill_fraction {u}')
        tab = [[b['S_lo'], b['S_hi'], b['R2']] for b in bins if b['R2'] > 0]
        if json.loads(json.dumps(c.get('table', []))) != json.loads(json.dumps(tab)):
            bad.append(f'{k}: table != bins with R2 > 0')
    return not bad, len(D.get('curves', {})), '; '.join(bad[:4]) or 'bins sorted, radii on the grid, table = bins with a radius'


# ---------------------------------------------------------------------------------------------------------------------
# bright_parent_table (candidate, installed, mirror or versioned file)
# ---------------------------------------------------------------------------------------------------------------------

def _tables_of(path):
    T = _K().read_json(path)
    return T, T.get('tables', {})


@content_test('bright_parent_table')
def keys_form(path, **ctx):
    from exopipe.phase3.lib.calibrate.bright_parent import family
    _, t = _tables_of(path)
    bad = [k for k in t if family(k) == 'other']
    return bool(t) and not bad, len(t), (f'keys outside every family: {bad[:5]}' if bad else
                                         f'{len(t)} keys in the families deep-all/deep-band/deep-multi/sublst/10min')


@content_test('bright_parent_table')
def metric(path, **ctx):
    from exopipe.phase3.lib.calibrate.bright_parent import family
    _, t = _tables_of(path)
    bad = [k for k, v in t.items() if family(k) in ('deep-all', 'deep-band', 'deep-multi', 'sublst')
           and v.get('metric') != 'image-pixel']
    return not bad, len(bad), (f'not image-pixel: {bad[:5]}' if bad else 'every deep / sub-LST table records the image-pixel metric '
                                                                         '(10min_* are the retired sky tables)')


@content_test('bright_parent_table')
def monotone_guard(path, **ctx):
    _, t = _tables_of(path)
    bad = []
    for k, v in t.items():
        tab = v.get('table') or []
        if [x[0] for x in tab] != sorted(x[0] for x in tab):
            bad.append(f'{k}: not sorted')
        for a, b in zip(tab, tab[1:]):
            if a[2] > b[2] + 1e-12:
                bad.append(f'{k}: fainter bin {a[0]}-{a[1]} R {a[2]} > brighter {b[0]}-{b[1]} R {b[2]}')
        if any(not (x[2] > 0 and x[0] < x[1]) for x in tab):
            bad.append(f'{k}: a bin with R <= 0 or S_lo >= S_hi')
    return not bad, len(bad), '; '.join(bad[:3]) or 'every table sorted, R > 0, monotone from the brightest bin down'


@content_test('bright_parent_table')
def sha_pinned(path, **ctx):
    K, T = _K(), _cfg().tables
    for key in ('BRIGHT_PARENT_DEEP', 'BRIGHT_PARENT_SUBLST'):
        if _same_path(path, getattr(T, key)):
            sha = K.sha256(path)
            pin = getattr(T, key + '_SHA256')
            return sha == pin, sha[:16], f'tables.{key}_SHA256 {"=" if sha == pin else "!="} {pin[:16]} (gate G19)'
    return True, None, 'not a pinned path (candidate / mirror / versioned file)'


@content_test('bright_parent_table')
def sublst_present(path, **ctx):
    C = _cfg()
    B = C.calibrate_bright_parent
    _, t = _tables_of(path)
    pre = B.KEY_PREFIX['sublst'] + '_'
    s = [k for k in t if k.startswith(pre)]
    if not s and not _same_path(path, C.tables.BRIGHT_PARENT_SUBLST):
        return True, None, 'no sub-LST keys (deep file)'
    miss = [f'{pre}{S}_{b}' for S in ('I', 'V') for b in B.SUBLST_BANDS if f'{pre}{S}_{b}' not in t]
    nband = sum('_el' in k for k in s)
    notpix = [k for k in s if t[k].get('metric') != 'image-pixel']
    ok = s and not miss and nband > 0 and not notpix
    return bool(ok), len(s), (f'{len(s)} sub-LST tables, {nband} elevation-band tables, all image-pixel (G21)' if ok else
                              f'missing all-elevation keys {miss[:4]}; band keys {nband}; not image-pixel {notpix[:3]} (G21)')


def expected_tables(path):
    """{key: expected table} recomputed from the derivations of the file, or (None, why) when there are none to read."""
    from exopipe.phase3.lib.calibrate import bright_parent as BP
    K, C = _K(), _cfg()
    T, t = _tables_of(path)
    cal = (T.get('meta') or {}).get('calibrate')
    exp = {}
    if cal:
        cls, tier, paths = cal['class'], cal['tier'], cal['derivations']
        miss = [p for p in paths if not os.path.exists(p)]
        if miss:
            return None, f'UNREADABLE: derivations not readable: {miss[:2]}'
        if cls == 'bright-parent' and tier == 'deep':
            inst = K.read_json(cal['installed_at_build']['path'])['tables']
            D3, D2 = (K.read_json(p) for p in paths)
            got, _ = BP.build_deep(D3, D2, paths[0], paths[1], inst, tuple(cal.get('build_keys') or ('elevation-bands',)))
        elif cls == 'multiples':
            got, _ = BP.build_multiples(*(K.read_json(p) for p in paths), paths, {})
        else:
            got = BP.build_sublst(K.read_json(paths[0]), paths[0])[0]['tables']
        exp.update({k: v['table'] for k, v in got.items()})
        return exp, f'candidate of {cls} {tier} {cal.get("version")}'
    if _same_path(path, C.tables.BRIGHT_PARENT_DEEP):
        dv = C.tables.BRIGHT_PARENT_DEEP_DERIVATIONS
        miss = [p for g in dv.values() for p in g if not os.path.exists(p)]
        if miss:      # declared but node-local elsewhere / gone: never a silent pass (verify on the host that holds them)
            return None, f'UNREADABLE: derivations of the installed file not readable on this host: {miss[:2]}'
        a3, a2 = (K.read_json(p) for p in dv['all-elevation'])
        got, _ = BP.build_deep(a3, a2, *dv['all-elevation'], t, ('all-elevation',))
        exp.update({k: v['table'] for k, v in got.items()})
        e3, e2 = (K.read_json(p) for p in dv['elevation-bands'])
        got, _ = BP.build_deep(e3, e2, *dv['elevation-bands'], t, ('elevation-bands',))
        exp.update({k: v['table'] for k, v in got.items()})
        got, _ = BP.build_multiples(*(K.read_json(p) for p in dv['multiples']), dv['multiples'], {})
        exp.update({k: v['table'] for k, v in got.items()})
        return exp, 'installed deep file vs tables.BRIGHT_PARENT_DEEP_DERIVATIONS'
    if _same_path(path, C.tables.BRIGHT_PARENT_SUBLST):
        p = C.tables.BRIGHT_PARENT_SUBLST_DERIVATION
        if not p or not os.path.exists(p):
            return None, 'installed sub-LST file: its derivation is not readable here (lwacalim08 node-local; spec B3)'
        return {k: v['table'] for k, v in BP.build_sublst(K.read_json(p), p)[0]['tables'].items()}, 'installed sub-LST file'
    return None, 'mirror / versioned file: install_bundle.mirror_equals checks it against its candidate and base'


@content_test('bright_parent_table')
def equals_derivation(path, **ctx):
    from exopipe.phase3.lib.calibrate.bright_parent import family
    K = _K()
    exp, why = expected_tables(path)
    if exp is None:       # declared derivations that cannot be read fail; a file with none declared passes as NOT RECOMPUTED
        return not why.startswith('UNREADABLE'), None, (why if why.startswith('UNREADABLE') else f'NOT RECOMPUTED: {why}')
    _, t = _tables_of(path)
    fams = {family(k) for k in exp}
    have = {k for k in t if family(k) in fams}
    bad = [k for k in exp if k not in t or K.canon(t[k].get('table')) != K.canon(exp[k])]
    extra = sorted(have - set(exp))
    ok = not bad and not extra
    return ok, len(exp), (f'{len(exp)} tables = post_process(derivation) key by key ({why})' if ok else
                          f'{len(bad)} differ {bad[:4]}; {len(extra)} keys without a derivation {extra[:4]} ({why})')


# ---------------------------------------------------------------------------------------------------------------------
# calibration_diff
# ---------------------------------------------------------------------------------------------------------------------

@content_test('calibration_diff')
def reconciles(path, **ctx):
    K = _K()
    s = K.read_json(path)
    errs = []
    if s.get('tables_csv'):
        rows = K.read_csv(s['tables_csv'])
        cnt = {}
        for r in rows:
            cnt[r['change']] = cnt.get(r['change'], 0) + 1
        if cnt != s.get('tables'):
            errs.append(f"tables {s.get('tables')} != rows {cnt}")
    sv = s.get('served') or {}
    if sv.get('rows_csv') and 'tally' in sv:
        rows = K.read_csv(sv['rows_csv'])
        if len(rows) != sv.get('changed'):
            errs.append(f"changed {sv.get('changed')} != {len(rows)} rows")
        t = {}
        for r in rows:
            k = f"{r['stokes']}|{r['change']}|{r['elev_band']}"
            t[k] = t.get(k, 0) + 1
        if t != sv.get('tally'):
            errs.append('tally != rows')
        ids = {f"{r['stokes']}:{r['served_identity']}" for r in rows if r['served_identity'] and r['change'] == 'newly removed'}
        if ids != set(sv.get('served_identities_losing') or {}):
            errs.append('served identities losing detections != rows')
        if sv.get('removed_candidate', 0) - sv.get('removed_installed', 0) != \
                sum(1 for r in rows if r['change'] == 'newly removed') - sum(1 for r in rows if r['change'] == 'newly kept'):
            errs.append('removed counts do not reconcile with the changed rows')
        ss = sv.get('served_sources')
        if ss is not None:          # served-source fates reconcile with the changed rows (served_change_20261006.py port)
            for S in ('I', 'V'):
                n_rows = sum(1 for r in rows if r['stokes'] == S and r['change'] == 'newly removed' and r['served_identity'])
                if (ss.get('member_detections_leaving') or {}).get(S, 0) != n_rows:
                    errs.append(f'Stokes {S}: member detections leaving {(ss.get("member_detections_leaving") or {}).get(S, 0)} '
                                f'!= {n_rows} newly removed served rows')
                losing = sum(1 for k in (sv.get('served_identities_losing') or {}) if k.startswith(S + ':'))
                touched = sum(v for k, v in (ss.get('by_status') or {}).items() if k.split('|')[0] == S and k.split('|')[1] != 'unchanged')
                if losing != touched:
                    errs.append(f'Stokes {S}: {touched} sources gone/reduced != {losing} served identities losing detections')
            jp = ss.get('judged_csv')
            if jp and os.path.exists(jp) and len(K.read_csv(jp)) != sum((ss.get('judged_by_status') or {}).values()):
                errs.append('judged sources csv != judged_by_status')
        tn = sv.get('table_notes')
        if tn is not None and tn.get('bad'):
            errs.append(f"TABLE-NOTES: {tn['bad']} of {tn['n']} removals without an elevation band and (image-pixel)")
    elif sv.get('rows_csv'):
        rows = K.read_csv(sv['rows_csv'])
        n = sv.get('n_lst') if 'n_lst' in sv else len(rows)
        if len(rows) != n:
            errs.append(f'{len(rows)} rows != {n}')
    if s.get('rows_csv') and 'k' in s:
        if len(K.read_csv(s['rows_csv'])) != len(s['k']):
            errs.append('K rows != summary')
    if s.get('candidate') and os.path.exists(s['candidate']) and K.sha256(s['candidate']) != s.get('candidate_sha256'):
        errs.append('the candidate changed since the diff')
    return not errs, s.get('candidate_sha256', '')[:12], '; '.join(errs) or 'summary totals = the rows'


# ---------------------------------------------------------------------------------------------------------------------
# install_bundle
# ---------------------------------------------------------------------------------------------------------------------

@content_test('install_bundle')
def files_sha(path, **ctx):
    K = _K()
    m = K.read_json(path)
    errs = []
    for key in ('candidate', 'mirror', 'versioned'):
        x = m.get(key)
        if x and (not os.path.exists(x['path']) or K.sha256(x['path']) != x['sha256']):
            errs.append(f'{key} {x["path"]} missing or changed')
    for name, x in (m.get('files') or {}).items():
        if not os.path.exists(x['path']) or K.sha256(x['path']) != x['sha256']:
            errs.append(f'{name} {x["path"]} missing or changed')
    v, src = m.get('versioned'), m.get('mirror') or m.get('candidate')
    if v and src and os.path.exists(v['path']) and os.path.exists(src['path']) and K.sha256(v['path']) != K.sha256(src['path']):
        errs.append('the versioned file is not byte-equal to the mirror / candidate')
    return not errs, (m.get('versioned') or {}).get('sha256', '')[:12], '; '.join(errs) or 'every bundle file has its recorded sha256'


@content_test('install_bundle')
def mirror_equals(path, **ctx):
    K = _K()
    m = K.read_json(path)
    if not m.get('mirror'):
        return True, None, f"{m.get('class')}: no mirror (the versioned file is the candidate itself)"
    from exopipe.phase3.lib.calibrate.bright_parent import family, merge
    base_p = m['base']['path']
    if not os.path.exists(base_p) or K.sha256(base_p) != m['base']['sha256']:
        return False, None, f'the installed file {base_p} changed since the install bundle was made (re-run install)'
    base, cand, mir = (K.read_json(p) for p in (base_p, m['candidate']['path'], m['mirror']['path']))
    exp, replaced, removed = merge(m['class'], m['tier'], base, cand)
    errs = []
    if K.canon(exp['tables']) != K.canon(mir['tables']):
        diff = sorted(k for k in set(exp['tables']) | set(mir['tables'])
                      if K.canon(exp['tables'].get(k)) != K.canon(mir['tables'].get(k)))
        errs.append(f'tables differ from base + candidate: {diff[:5]} (G9)')
    if m['tier'] == 'deep':
        fam = {'multiples': ('deep-multi',), 'bright-parent': ()}[m['class']]
        for k, v in base['tables'].items():
            if k in cand['tables'] or family(k) in fam:
                continue
            if json.dumps(v, sort_keys=True) != json.dumps(mir['tables'].get(k), sort_keys=True):
                errs.append(f'{k} (outside the class) changed (G10)')
                break
        for top in ('meta', 'meta_10min'):
            if K.canon(base.get(top)) != K.canon(mir.get(top)):
                errs.append(f'{top} changed (G10)')
        bm, mm = base.get('meta_deep') or {}, mir.get('meta_deep') or {}
        if any(K.canon(mm.get(k)) != K.canon(v) for k, v in bm.items()) or len(mm) != len(bm) + 1:
            errs.append('meta_deep: earlier entries changed or not exactly one appended')
    else:
        if set(mir.get('meta', {})) != {'installed_utc', 'source', 'derivation', 'rule'}:
            errs.append(f"sub-LST meta keys {sorted(mir.get('meta', {}))}")
    return not errs, len(replaced), '; '.join(errs) or (f'mirror = base with {len(replaced)} candidate keys ({len(removed)} removed); '
                                                       f'other families byte-equal')


@content_test('install_bundle')
def patch_names(path, **ctx):
    K = _K()
    m = K.read_json(path)
    if not os.path.exists(m.get('patch', '')):
        return False, None, 'config patch missing'
    text = open(m['patch']).read()
    plus = '\n'.join(ln for ln in text.splitlines() if ln.startswith('+') and not ln.startswith('+++'))
    errs = []
    if not m.get('patch_changes'):
        errs.append('the patch changes no key')
    v = m.get('versioned') or {}
    if m.get('class') in ('bright-parent', 'multiples', 'scale', 'fluxcorr'):
        if v.get('path') not in plus:
            errs.append('the patch does not point at the versioned file')
        if v.get('sha256') not in plus and v.get('sha256') != (m.get('base') or {}).get('sha256'):
            errs.append('the patch does not carry the versioned file sha256')
    if m.get('class') == 'template-k':
        cur = _cfg().fitting.K_TEMPLATE
        same = all(tuple(round(x, 3) for x in cur.get((m['tier'], S), ())) == tuple(v_) for S, v_ in (m.get('k_new') or {}).items())
        if same:
            return True, 0, 'K equals the config values: nothing to patch (the patch is empty)'
        for S, (a, b) in (m.get('k_new') or {}).items():
            if f'{a:.3f}, {b:.3f}' not in plus:
                errs.append(f'K {S} not in the patch')
    return not errs, len(m.get('patch_changes') or []), '; '.join(errs) or f"patch changes {', '.join(m.get('patch_changes'))}"


@content_test('install_bundle')
def ledger_draft(path, **ctx):
    K = _K()
    m = K.read_json(path)
    p = m.get('ledger_draft', '')
    if not os.path.exists(p):
        return False, None, 'draft ledger row missing'
    import csv
    with open(p, newline='') as f:
        rd = csv.DictReader(f)
        cols, rows = tuple(rd.fieldnames or ()), list(rd)
    errs = []
    if cols != tuple(_cfg().calibrate.LEDGER_DRAFT_COLUMNS):
        errs.append(f'columns {cols}')
    if len(rows) != 1 or rows[0].get('status') != 'draft':
        errs.append('not exactly one row with status draft')
    elif m['candidate']['sha256'] not in rows[0].get('description', ''):
        errs.append('the description does not name the candidate sha256')
    if (m.get('ruling') or {}).get('id') and rows and m['ruling']['id'] not in rows[0].get('authorized_by', ''):
        errs.append('authorized_by does not name the ruling')
    return not errs, rows[0].get('fix_id') if rows else None, '; '.join(errs) or 'one draft row (fix_ledger columns)'


# ---------------------------------------------------------------------------------------------------------------------
# flux scale
# ---------------------------------------------------------------------------------------------------------------------

@content_test('vlssr_ratios')
def columns_rows(path, **ctx):
    K = _K()
    rows = K.read_csv(path)
    need = ('lst', 'dec', 'ratio', 's_lwa_jy', 's_vlssr_jy')
    if not rows or any(c not in rows[0] for c in need):
        return False, len(rows), f'no rows or missing columns {need}'
    bad = [r for r in rows if not (math.isfinite(float(r['ratio'])) and float(r['ratio']) > 0)]
    lo, hi = _cfg().calibrate_scale.DEC_WINDOW_DEG
    lsts = {r['lst'] for r in rows if lo <= float(r['dec']) <= hi}
    return not bad, len(rows), (f'{len(bad)} non-finite or non-positive ratios' if bad else
                                f'{len(rows)} matches; {len(lsts)} LSTs with matches in the window')


def _scale_inputs(path):
    from exopipe.phase3.lib.calibrate.fluxscale import read_provenance
    K, T = _K(), _cfg().tables
    pv = read_provenance(path)
    if pv:
        return pv.get('ratios'), pv.get('previous')
    if _same_path(path, T.SCALE_TABLE) or K.sha256(path) == T.SCALE_TABLE_SHA256:
        return T.SCALE_TABLE_INPUTS['ratios'], T.SCALE_TABLE_INPUTS['previous']
    return None, None


@content_test('scale_table')
def rows(path, **ctx):
    K = _K()
    rs = K.read_csv(path)
    want = [f'{h:02d}h' for h in range(_cfg().calibrate_scale.LST_COUNT)]
    cols = ('lst', 'scale', 'err', 'nsrc', 'median_ratio', 'scale_old', 'rescale', 'window', 'ref', 'measured_at')
    errs = []
    if [r['lst'] for r in rs] != want:
        errs.append(f'LSTs {[r["lst"] for r in rs][:4]}... != 00h..23h')
    if rs and tuple(rs[0].keys()) != cols:
        errs.append(f'columns {tuple(rs[0].keys())}')
    if any(not float(r['scale']) > 0 for r in rs):
        errs.append('a scale <= 0')
    return not errs, len(rs), '; '.join(errs) or '24 LSTs, production columns, positive factors'


@content_test('scale_table')
def recompute(path, **ctx):
    from exopipe.phase3.lib.calibrate.fluxscale import recompute_scale
    K = _K()
    ratios, prev = _scale_inputs(path)
    if not ratios or not os.path.exists(ratios) or not prev or not os.path.exists(prev):
        return False, None, f'inputs not readable (ratios {ratios}, previous {prev}): the table cannot be recomputed'
    try:
        exp = recompute_scale(ratios, prev)
    except ValueError as e:
        return False, None, f'recompute refused: {e}'
    bad = []
    for r in K.read_csv(path):
        e = exp.get(r['lst'])
        got = (float(r['scale']), float(r['err']), int(r['nsrc']), float(r['median_ratio']), float(r['scale_old']),
               float(r['rescale']))
        if e is None or any(abs(a - b) > 1e-12 for a, b in zip(got, e)):
            bad.append(r['lst'])
    return not bad, 24 - len(bad), (f'rows differ from the recompute: {bad}' if bad else
                                    f'every row = recompute from {os.path.basename(ratios)} and {os.path.basename(prev)}')


@content_test('shflux_stagea')
def rows_status(path, **ctx):
    rs = _K().read_csv(path)
    bad = []
    for r in rs:
        st = r.get('status', '')
        if st == 'OK':
            try:
                if not (float(r['scale']) > 0 and float(r['err']) > 0):
                    bad.append(f"{r['lst']}_{r['scope']}: scale/err <= 0")
            except ValueError:
                bad.append(f"{r['lst']}_{r['scope']}: unparseable")
        elif not st.startswith('REFUSED'):
            bad.append(f"{r.get('lst')}_{r.get('scope')}: status {st!r}")
    n_ok = sum(r.get('status') == 'OK' for r in rs)
    return bool(rs) and not bad, n_ok, '; '.join(bad[:3]) or f'{len(rs)} cells: {n_ok} OK, {len(rs) - n_ok} REFUSED'


@content_test('band_vector')
def norm(path, **ctx):
    F = _cfg().calibrate_fluxcorr
    k = {r['subband']: float(r['factor']) for r in _K().read_csv(path)}
    ok = set(k) == set(F.SUBBANDS) and k.get(F.BANDVEC_NORM_BAND) == 1.0
    return ok, len(k), (f'15 subbands, k({F.BANDVEC_NORM_BAND}) = 1' if ok else
                        f'subbands {sorted(k)}; k({F.BANDVEC_NORM_BAND}) = {k.get(F.BANDVEC_NORM_BAND)}')


@content_test('band_vector')
def converged(path, **ctx):
    from exopipe.phase3.lib.calibrate.fluxscale import read_provenance
    K, C = _K(), _cfg()
    pv = read_provenance(path)
    conv = (pv or {}).get('convergence')
    if not conv and (_same_path(path, C.tables.BAND_VECTOR) or K.sha256(path) == C.tables.BAND_VECTOR_SHA256):
        conv = C.tables.BAND_VECTOR_CONVERGENCE
    if not conv or not os.path.exists(conv):
        return False, None, f'no readable LOO medians measured with this vector ({conv})'
    m = [abs(float(r['raw_median_pct'])) / 100.0 for r in K.read_csv(conv)]
    worst = max(m) if m else float('inf')
    tol = C.calibrate_fluxcorr.BANDVEC_TOL
    return worst < tol, round(worst, 5), f'worst LOO median {100 * worst:.2f} per cent {"<" if worst < tol else ">="} {100 * tol:.1f} (G13)'


def _fc_inputs(path):
    import glob
    from exopipe.phase3.lib.calibrate.fluxscale import read_provenance
    K, T = _K(), _cfg().tables
    pv = read_provenance(path)
    if pv:
        return pv['stagea'], pv['band_vector'], pv['scale_table'], pv['version'], pv['yyyymmdd']
    if _same_path(path, T.FLUXCORR_TABLE) or K.sha256(path) == T.FLUXCORR_TABLE_SHA256:
        v = T.FLUXCORR_VERSION.split('-')      # fluxcorr-v2-20261001
        return (sorted(glob.glob(T.FLUXCORR_INPUTS['stagea_glob'])), T.FLUXCORR_INPUTS['band_vector'],
                T.FLUXCORR_INPUTS['scale_table'], v[1], v[2])
    return None


@content_test('fluxcorr_table')
def structure(path, **ctx):
    F = _cfg().calibrate_fluxcorr
    q = 10.0 ** -F.TABLE_DECIMALS          # one unit of the last written decimal
    rs = _K().read_csv(path)
    errs = []
    if [r['scope'] for r in rs] != list(F.SUBBANDS) + list(F.WIDEBAND):
        errs.append('scopes are not the 15 subbands + Red, Green, Blue, Full')
    if len({(r['g_abs'], r['n_sh_cells'], r['version']) for r in rs}) != 1:
        errs.append('g, n or version differ between rows')
    k = {r['scope']: float(r['k_band']) for r in rs}
    for r in rs:
        kk, gg = float(r['k_band']), float(r['g_abs'])
        # fluxcorr = round6(k x g) of the UNROUNDED k and g (build_table :65-70): against the rounded k, g the exact bound
        # is 0.5 q x (k + g + 1) (production's v2 table reaches 7.8e-7; a wideband k, a mean, reaches 1.3e-6)
        if abs(float(r['fluxcorr']) - kk * gg) > 0.5 * q * (abs(kk) + abs(gg) + 1) + 1e-9:
            errs.append(f"{r['scope']}: fluxcorr != k x g")
            break
    from exopipe.phase3.core.cells import band_members
    for b in F.WIDEBAND:
        # round6 of the mean of the unrounded members vs the mean of the rounded members: within q (0.5 q each side)
        if b in k and abs(k[b] - sum(k[s] for s in band_members(b)) / len(band_members(b))) > q + 1e-9:
            errs.append(f'{b}: k is not the mean of its members')
    return not errs, len(rs), '; '.join(errs) or '19 scopes, one g / n / version, fluxcorr = k x g, wideband k = mean of members'


@content_test('fluxcorr_table')
def recompute(path, **ctx):   # noqa: F811 - one name per class is the registry's convention
    from exopipe.phase3.lib.calibrate.fluxscale import fit_table, TABLE_COLS
    K = _K()
    inp = _fc_inputs(path)
    if not inp:
        return False, None, 'no provenance: the table cannot be recomputed'
    files, vec, scale, version, ymd = inp
    miss = [p for p in list(files) + [vec, scale] if not os.path.exists(p)]
    if miss or not files:
        return False, None, f'inputs not readable: {miss[:2] or "no stage-A file"}'
    rows, sh, fit = fit_table(files, vec, scale, version, ymd)
    got = K.read_csv(path)
    bad = [r['scope'] for r, e in zip(got, rows) if any(str(r[c]) != str(e[c]) for c in TABLE_COLS)]
    if len(got) != len(rows):
        bad.append('row count')
    shp = path.replace('.csv', '_shcells.csv')
    if os.path.exists(shp):
        gs = [tuple(r.values()) for r in K.read_csv(shp)]
        if gs != [tuple(str(x) for x in s) for s in sh]:
            bad.append('shcells rows')
    return not bad, round(fit['g'], 6), (f'rows differ from the recompute: {bad[:5]}' if bad else
                                         f"g {fit['g']:.6f} over {fit['n']} cells, every row and the shcells rows recomputed exactly")


# ---------------------------------------------------------------------------------------------------------------------
# template K
# ---------------------------------------------------------------------------------------------------------------------

def _trials_tier(cols):
    for tier, (st, _ok, a, b) in _cfg().calibrate_template_k.TRIALS_COLUMNS.items():
        if st in cols and a in cols and b in cols:
            return tier
    return None


@content_test('template_k_trials')
def columns(path, **ctx):
    import csv
    with open(path, newline='') as f:
        cols = csv.DictReader(f).fieldnames or []
    tier = _trials_tier(cols)
    need = ('stokes', 'snr_true', 'bmaj_deg', 'bmin_deg')
    miss = [c for c in need if c not in cols]
    return tier is not None and not miss, tier, (f'{tier} trials columns' if tier and not miss else f'missing {miss}; tier {tier}')


@content_test('template_k_trials')
def status_counts(path, **ctx):
    rs = _K().read_csv(path)
    tier = _trials_tier(list(rs[0].keys()) if rs else [])
    if not tier:
        return False, None, 'unknown trials columns'
    st, okv = _cfg().calibrate_template_k.TRIALS_COLUMNS[tier][:2]
    n_ok = {S: sum(r.get('stokes') == S and r.get(st) == okv for r in rs) for S in ('I', 'V')}
    return all(n_ok.values()), n_ok, f'{len(rs)} injections; ok fits per Stokes {n_ok}'


@content_test('template_k_record')
def recompute(path, **ctx):   # noqa: F811
    from exopipe.phase3.lib.calibrate.template_k import summarise
    K = _K()
    rec = K.read_json(path)
    files = sorted(rec.get('trials', {}))
    bad = [p for p in files if not os.path.exists(p) or K.digests(p)['sha1'] != rec['trials'][p]]
    if bad or not files:
        return False, None, f'trials missing or changed: {bad[:2]}'
    again = summarise(files, rec['tier'], rec.get('orientation'))
    diff = [S for S in rec.get('K', {}) if (again['K'][S]['major'], again['K'][S]['minor']) !=
            (rec['K'][S]['major'], rec['K'][S]['minor'])]
    return not diff, {S: (round(v['major'], 3), round(v['minor'], 3)) for S, v in rec.get('K', {}).items()}, \
        (f'K differs from the recompute for {diff}' if diff else f'K recomputed from {len(files)} trials files')


@content_test('template_k_record')
def bounds(path, **ctx):
    lo, hi = _cfg().calibrate_template_k.K_BOUNDS
    rec = _K().read_json(path)
    bad = {S: (v['major'], v['minor']) for S, v in rec.get('K', {}).items() if not (lo < v['major'] < hi and lo < v['minor'] < hi)}
    return bool(rec.get('K')) and not bad, len(rec.get('K', {})), (f'outside ({lo}, {hi}): {bad}' if bad else f'0 < K < 5 (G16)')


@content_test('template_k_record')
def interval(path, **ctx):
    rec = _K().read_json(path)
    bad = [S for S, v in rec.get('K', {}).items()
           if not (v['boot_major_16_84'][0] <= v['major'] <= v['boot_major_16_84'][1]
                   and v['boot_minor_16_84'][0] <= v['minor'] <= v['boot_minor_16_84'][1])]
    return not bad, None, (f'K outside its 16-84 per cent bootstrap interval: {bad}' if bad else 'K inside its bootstrap interval')


@content_test('template_k_record')
def orientation(path, **ctx):
    o = _K().read_json(path).get('orientation')
    return o in ('legacy-mirrored', 'wcs'), o, 'template orientation recorded (G17 compares it at install)'


# ---------------------------------------------------------------------------------------------------------------------
# calibration_verify
# ---------------------------------------------------------------------------------------------------------------------

@content_test('calibration_verify')
def results_pass(path, **ctx):
    K = _K()
    rep = K.read_json(path)
    fails, changed = [], []
    for t in rep.get('targets', []):
        fails += [r['name'] for r in t.get('results', []) if not r.get('passed')]
        if t.get('sha256') and (not os.path.exists(t['path']) or K.sha256(t['path']) != t['sha256']):
            changed.append(t['path'])
        if not t.get('sha256'):
            fails.append(f"{t.get('class')}.exists")
    fails += [c['name'] for c in rep.get('checks', []) if not c.get('passed') and not c.get('waived')]
    return not fails and not changed, len(fails), (f'failed {fails[:4]}; changed since {changed[:2]}' if fails or changed
                                                   else 'every recorded test passed; targets unchanged')
