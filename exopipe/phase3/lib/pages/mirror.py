"""Gate P3 (MIRROR drift): config keys whose values live INSIDE the in-place page modules during the refactor window.

The step does not pass these values to the production modules (they are literals in their source). Before a stage runs
an in-place module, every anchor of that module is read from its SOURCE TEXT (ast for module-level literals, a regular
expression around inline literals; nothing imported or executed) and compared with config. Any difference - or an anchor
that is no longer found (the module was edited) - refuses the stage: config must describe the code that runs. The keys
become live parameters when the modules are absorbed into lib/pages.

ANCHORS: (section, key, entry, legacy root, file, kind, spec, transform)
  kind 'const'  spec = module-level name; its literal must equal transform(value)
  kind 're'     spec = pattern; EVERY match's capture group(s) must equal transform(value) (numbers compared as floats);
                at least one match required
  kind 'flag'   spec = (text, value_when_present); the text's presence gives the module's value
"""
import os
import re

from exopipe.phase3.lib.pages import common

B14, STD7, TEN, SNAP = 'build_i14_pages_std7.py', 'std7_panel.py', 'build_tenmin_pages.py', 'phase3_snapshot_pages.py'

ANCHORS = (
    ('pages', 'PNG_DPI', None, 'ops', B14, 're', r'fig\.savefig\(_png, dpi=(\d+)\)', None),
    ('pages', 'PNG_DPI', None, 'ops', TEN, 're', r"\.png', dpi=(\d+)\)", None),
    ('pages', 'PNG_DPI', None, 'clv15', SNAP, 're', r'fig\.savefig\(job\["dst"\], dpi=(\d+)\)', None),
    ('pages', 'PANEL_CUT_DEG', None, 'ops', STD7, 'const', 'CUT_DEG', None),
    ('pages', 'PANEL_CUT_DEG', None, 'clv15', SNAP, 'const', 'CUT_DEG', None),
    ('pages', 'PANEL_OBS_LAT_DEG', None, 'ops', STD7, 'const', 'LAT', None),
    ('pages', 'PANEL_OBS_LAT_DEG', None, 'ops', TEN, 're', r'lat = _math\.radians\(([\d.]+)\)', None),
    ('pages', 'HOURS_EL_MIN_DEG', 'deep', 'ops', STD7, 'flag', ("EL_MIN = _GIF.TENMIN_ELEV_FLOOR_DEG if (_os.environ.get('P3_TIER') == "
     "'10min' or _os.environ.get('P3_CADENCE') == '10min') else _GIF.DEEP_ELEV_FLOOR_DEG", True), lambda v: True),
    ('pages', 'HOURS_EL_MIN_DEG', 'deep', 'ops', 'gate_inputs.py', 're', r"DEEP_ELEV_FLOOR_DEG = float\(os\.environ\.get\('P3_DEEP_MIN_ELEVATION', '([\d.]+)'\)\)", None),
    ('pages', 'HOURS_EL_MIN_DEG', 'sublst', 'ops', 'gate_inputs.py', 'const', 'TENMIN_ELEV_FLOOR_DEG', None),
    ('pages', 'HOURS_EL_MIN_DEG', 'snapshot', 'clv15', SNAP, 're', r'elevation_deg\(pos\[0\], pos\[1\], lst_deg\) >= ([\d.]+)', None),
    ('pages', 'PANEL_TEXT_WRAP_CHARS', None, 'ops', STD7, 're', r'def _wrap\(line, width=(\d+)', None),
    ('pages', 'FORCED_READING', None, 'ops', STD7, 're', r'return (bilinear)_at\(dd, y, x\), _annulus_rms', None),
    ('pages', 'FRAME_CACHE', 'deep', 'ops', B14, 're', r'if len\(_FR\) > (\d+):', None),
    ('pages', 'FRAME_CACHE', 'sublst', 'ops', TEN, 're', r'if len\(_FR\) > (\d+): _FR\.popitem', None),
    ('pages', 'PREDICTED_PANELS', 'deep', 'ops', B14, 'flag', ("'Predicted V': (predicted_v(", True), None),
    ('pages', 'PREDICTED_PANELS', 'sublst', 'ops', TEN, 'flag', ("'Predicted V'", True), None),
    ('pages', 'PREDICTED_PANELS', 'snapshot', 'clv15', SNAP, 'flag', ('NO PREDICTED CELL IN THE SNAPSHOT TIER', False), None),
    ('pages', 'PRED_V_MATCH_ARCMIN', None, 'ops', B14, 're', r'if sep < ([\d.]+) and \(best is None', None),
    ('pages', 'PRED_V_MATCH_ARCMIN', None, 'ops', TEN, 're', r'if sep < ([\d.]+) and \(best is None', None),
    ('pages', 'PRED_I_PARENT_MIN_JY', None, 'ops', B14, 're', r'if F\[j\] < ([\d.]+) or sep\[j\] / beam <=', None),
    ('pages', 'PRED_I_PARENT_MIN_JY', None, 'ops', TEN, 're', r'if F\[j\] < ([\d.]+) or sep\[j\] / beam_deg <=', None),
    ('pages', 'PRED_I_MIN_SEP_BEAMS', None, 'ops', B14, 're', r'sep\[j\] / beam <= ([\d.]+):', None),
    ('pages', 'PRED_I_STAMPS', None, 'ops', B14, 're', r'if len\(stamps\) < (\d+):', lambda v: v[0]),
    ('pages', 'PRED_I_STAMPS', None, 'ops', B14, 're', r'if len\(stamps\) >= (\d+):\s*\n?\s*break', lambda v: v[1]),
    ('pages', 'PRED_I_STAMP_HALF_BEAMS', None, 'ops', B14, 're', r'hs = int\(([\d.]+) \* bpx\)', None),
    ('pages', 'PRED_I_QA_DISK_BEAMS', None, 'ops', B14, 're', r'disk = np\.hypot\(yy - hs, xx - hs\) <= ([\d.]+) \* bpx', None),
    ('pages', 'PRED_I_QA_R_MIN', None, 'ops', B14, 're', r"_qa_note = '' if r >= ([\d.]+)", None),
    ('pages', 'PRED_I_QA_R_MIN', None, 'ops', B14, 're', r'\n\s+if r >= ([\d.]+):\n', None),
    ('pages', 'PRED_I_PLACE_MIN_JY', None, 'ops', B14, 're', r'if F\[j\] < ([\d.]+) or sep\[j\] > ', None),
    ('pages', 'PRED_I_PLACE_MIN_JY', None, 'ops', B14, 're', r'if F\[j\] < ([\d.]+): continue', None),
    ('pages', 'I_TEMPLATE_NULLS', None, 'ops', B14, 're', r'while len\(vals\) < (\d+) and tries', lambda v: v[0]),
    ('pages', 'I_TEMPLATE_NULLS', None, 'ops', B14, 're', r'while len\(vals\) < (\d+) and bp and', lambda v: v[0] + v[1]),
    ('pages', 'I_TEMPLATE_NULL_PCTL', None, 'ops', B14, 're', r'np\.percentile\(vals, ([\d.]+)\)', None),
    ('pages', 'I_TEMPLATE_MIN_NULLS', None, 'ops', B14, 're', r'if len\(vals\) >= (\d+) else None', None),
    ('pages', 'I_TEMPLATE_MARGINAL_FACTOR', None, 'ops', B14, 're', r'<= ([\d.]+) \* p95', None),
    ('pages', 'I_TEMPLATE_SEED', None, 'ops', B14, 're', r'rng_seed=(\d+)', None),
    ('pages', 'XMATCH_RADIUS_ARCMIN', None, 'ops', B14, 're', r'np\.radians\(([\d.]+) / 60\.0\)', None),
    ('pages', 'XMATCH_RADIUS_ARCMIN', None, 'ops', TEN, 're', r'np\.where\(sep < ([\d.]+)\)', None),
    ('pages', 'XMATCH_SAMPLES', 'deep', 'ops', B14, 'const', 'CATS', tuple),
    ('pages', 'XMATCH_SAMPLES', 'sublst', 'ops', TEN, 'const', 'CATS', tuple),
    ('pages', 'XMATCH_SAMPLE_DIR', None, 'ops', B14, 're', r"open\(f'(/[\w/]+)/\{cat\}\.csv'", None),
    ('paths', 'CLV15_ROOT', None, 'ops', B14, 're', r"open\(f'(/[\w/]+)/\{cat\}\.csv'", None),
    ('pages', 'XMATCH_SAMPLE_DIR', None, 'ops', TEN, 'const', 'XM_DIR', None),
    ('pages', 'SKY_AREA_DEG2', None, 'ops', B14, 're', r'rho = len\(crows\) / ([\d.]+)', None),
    ('pages', 'SKY_AREA_DEG2', None, 'ops', TEN, 're', r'_RHO\[cat\] = n / ([\d.]+)', None),
    ('pages', 'VHF_ONLY', 'deep', 'ops', B14, 'const', '_VHF_ONLY_BANDS', lambda v: set(v[1])),
    ('pages', 'VHF_ONLY', 'sublst', 'ops', TEN, 'const', 'VHF', lambda v: tuple(v[1])),
    ('pages', 'SUBLST_FIT_FLUX_SNR', None, 'ops', TEN, 're', r"r\['_snr'\] >= ([\d.]+)", None),
    ('pages', 'PB_LABEL', 'deep', 'ops', STD7, 're', r"info\.get\('pb_label', '([^']+)'\)", None),
    ('pages', 'PB_LABEL', 'sublst', 'ops', TEN, 're', r"pb_label='([^']+)'", None),
    ('pages_deep', 'PASS_MS', None, 'ops', 'p3s3_tier_pages_input.py', 're', r"pass_ms='(\d+)'", None),
    ('pages_sublst', 'RUN_COLUMN', None, 'ops', 'p3t_pages_input.py', 're', r"pass_ms='\d+', run='(\w+)'", None),
    ('paths', 'SUBLST_ROOT', None, 'ops', 'sublst_pages_prep.py', 're', r"SL = '([^']+)'", None),
    ('pages_deep', 'TESS_ORBIT_REFRESH_DAYS', None, 'ops', 'tess/tess_coverage.py', 're', r'> (\d+) \* 86400', None),
    ('pages_deep', 'TESS_ORBIT_URL', None, 'ops', 'tess/tess_coverage.py', 'const', 'URL', None),
    ('pages_deep', 'TESS_INTEGRATION_S', None, 'ops', 'tess/tess_coverage.py', 're', r'INTEG_S = ([\d.]+)', None),
    ('pages_sublst', 'SERVED_CATALOG_TAG', None, 'ops', 'sublst_pages_prep.py', 're', r"source_catalogs_(\w+?)_\{S\}_\{T\}", None),
    ('pages_sublst', 'REQUIRED_FILES', None, 'ops', 'sublst_pages_prep.py', 're', r"exists\(f'\{CAT\}/(pertie/pertie_detections\.csv)'\)",
     lambda v: v[0]),
    ('pages_sublst', 'REQUIRED_FILES', None, 'ops', 'sublst_pages_prep.py', 're', r"for f in \(('rejected\.csv', 'rejected_index\.npz', 'detection_ts\.csv')\)",
     lambda v: ', '.join(repr(x) for x in v[1:])),
    ('pages_snapshot', 'PANELS_TOP_N', None, 'clv15', SNAP, 're', r'chosen = uniq\[:(\d+)\]', None),
    ('pages_snapshot', 'FIT_NO_SOURCE_SNR', None, 'clv15', SNAP, 're', r'abs\(float\(id_fit\.get\("snr"\) or 0\)\) >= ([\d.]+)', None),
    ('pages_snapshot', 'IMAP_CHUNK', None, 'clv15', SNAP, 're', r'imap_unordered\(_safe_render, todo, chunksize=(\d+)\)', None),
    ('paths', 'COADD_SB_ROOT', None, 'ops', 'snapshot_filter_catalog.py', 're',
     r'"coadd_root\.txt"\), "w"\)\.write\("([^"\\]+)\\n"\)', None),
    ('pages_deep', 'STRAT_EL_BINS_DEG', None, 'ops', 'deep_stratified_queue.py', 're',
     r"p\['el'\] < (\d+) else \('el\d+-\d+' if p\['el'\] < (\d+)", tuple),
    ('pages_deep', 'STRAT_PARENT_BINS_BEAMS', None, 'ops', 'deep_stratified_queue.py', 're',
     r"if d < (\d+) else \('par\d+-\d+bm' if d < (\d+)", tuple),
    ('pages_deep', 'STRAT_PARENT_SNR_MIN', None, 'ops', 'deep_stratified_queue.py', 're', r'P\[:, 2\] / g >= (\d+)\]', None),
)

# the modules each stage runs in place (their anchors are checked before the stage)
STAGE_FILES = {
    ('deep', 'tess'): ('tess/tess_coverage.py',),
    ('deep', 'input'): ('p3s3_tier_pages_input.py',),
    ('deep', 'build'): (B14, STD7, 'gate_inputs.py'),
    ('deep', 'strat'): ('deep_stratified_queue.py',),
    ('sublst', 'prep'): ('sublst_pages_prep.py',),
    ('sublst', 'input'): ('p3t_pages_input.py',),
    ('sublst', 'build'): (TEN, STD7, 'gate_inputs.py'),
    ('snapshot', 'build'): (SNAP, STD7),
    ('snapshot', 'filtercat'): ('snapshot_filter_catalog.py',),
}


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _same(found, want):
    """Compare a value read from source with the config value (numbers as floats, sequences element-wise, sets as sets)."""
    if isinstance(want, set) or isinstance(found, set):
        return set(map(str, found)) == set(map(str, want))
    if isinstance(want, (tuple, list)) and isinstance(found, (tuple, list)):
        return len(want) == len(found) and all(_same(f, w) for f, w in zip(found, want))
    if isinstance(want, dict) and isinstance(found, dict):
        return found == want
    fn, wn = _num(found), _num(want)
    if fn is not None and wn is not None and not isinstance(want, bool):
        return fn == wn
    return str(found) == str(want)


def _value(sec, key, entry):
    v = getattr(getattr(common.CFG(), sec), key)
    return v[entry] if entry is not None else v


def _site_paths(roots):
    """{overridden root: site root} - a frozen tree rewrites code locations inside its copies (sublst_freeze_code.sh
    REWRITES.tsv), so a path constant naming the frozen copy of a root equals the site root's value."""
    site = common.legacy_roots()
    return {v: site[k] for k, v in roots.items() if k in site and site[k] != v}


def _norm_found(v, remap):
    if isinstance(v, str):
        for a, b in remap.items():
            if v == a or v.startswith(a.rstrip('/') + '/'):
                return b + v[len(a):]
    return v


def check(roots, files=None):
    """[{key, file, ok, found, want, detail}] for every anchor in ``files`` (relative paths; None = every anchor)."""
    out = []
    remap = _site_paths(roots)
    for sec, key, entry, root, rel, kind, spec, tf in ANCHORS:
        if files is not None and rel not in files:
            continue
        want = _value(sec, key, entry)
        want = tf(want) if tf else want
        name = f'{sec}.{key}' + (f'[{entry}]' if entry is not None else '')
        path = os.path.join(roots.get(root, ''), rel)
        r = dict(key=name, file=path, want=want, found=None, ok=False, detail='')
        try:
            if not os.path.isfile(path):
                r['detail'] = 'module not found'
            elif kind == 'const':
                r['found'] = _norm_found(common.module_constant(path, spec), remap)
                r['ok'] = _same(r['found'], want)
            elif kind == 'flag':
                text, present_means = spec
                with open(path, encoding='utf-8', errors='replace') as f:
                    present = text in f.read()
                r['found'] = present_means if present else (not present_means)
                r['ok'] = r['found'] == want
            else:
                with open(path, encoding='utf-8', errors='replace') as f:
                    src = f.read()
                hits = [m.groups() for m in re.finditer(spec, src)]
                if not hits:
                    r['detail'] = f'anchor not found ({spec}) - the module changed; update the anchor and the key together'
                else:
                    vals = [_norm_found(h[0], remap) if len(h) == 1 else tuple(h) for h in hits]
                    r['found'] = vals[0] if len(set(map(str, vals))) == 1 else vals
                    r['ok'] = all(_same(v, want) for v in vals)
                    if not r['ok'] and len(vals) > 1:
                        r['detail'] = f'{len(vals)} occurrences: {vals[:4]}'
        except Exception as e:      # an unreadable module is a failed check, never a pass
            r['detail'] = f'{type(e).__name__}: {e}'
        out.append(r)
    return out


def gate(u, roots, tier, stage):
    """Gate P3 for one stage: record each anchor as a check of the unit and refuse on any difference."""
    if not common.CFG().pages.MIRROR_CHECK:
        u.note('MIRROR drift check off (pages.MIRROR_CHECK False)')
        return []
    files = STAGE_FILES.get((tier, stage))
    if not files:
        return []
    res = check(roots, files)
    bad = []
    for r in res:
        u.check(f'mirror:{r["key"]}', r['ok'], value=str(r['found'])[:80],
                detail=f'{os.path.basename(r["file"])}: want {r["want"]!r}' + (f'; {r["detail"]}' if r['detail'] else ''))
        if not r['ok']:
            bad.append(r)
    if bad:
        raise common.Refusal('P3 MIRROR drift: ' + '; '.join(
            f'{r["key"]} config {r["want"]!r} vs {os.path.basename(r["file"])} {r["found"]!r}'
            + (f' ({r["detail"]})' if r['detail'] else '') for r in bad[:4]))
    return res


def plan_lines(roots, tier, stage):
    """Dry-run text: the anchors of a stage and whether they agree."""
    files = STAGE_FILES.get((tier, stage))
    if not files or not common.CFG().pages.MIRROR_CHECK:
        return []
    res = check(roots, files)
    bad = [r for r in res if not r['ok']]
    if not bad:
        return [f'    gate P3 mirror: {len(res)} constants of {", ".join(files)} agree with config']
    return [f'    gate P3 mirror: WOULD REFUSE - {r["key"]} config {r["want"]!r} vs {os.path.basename(r["file"])} '
            f'{r["found"]!r} {r["detail"]}' for r in bad]
