"""Content tests of `exopipe3 dewarp` (steps/dewarp.py), one registry entry per product class (core/validate discovers this
module). Each test checks the file's CONTENT; a test that needs context and has none FAILS ('no context'), never passes.

  dewarp_table          rows               the 24 MD columns; a final row for each of the 19 scopes; valid outcomes; OK and
                                           REFERENCE ONLY rows re-derived from the written values ((a), N >= 10, (b'))
  dewarp_screen         record             SCREEN_X/SCREEN_Y equal shape = the measured master's; METHOD rbf, KERNEL, SUBBAND;
                                           NANCHOR = the scope's final table row; finite offsets
  dewarped_master       D1_D6              check_mdewarp_gen.py (CMG) run IN PLACE through core.legacy.run (site interpreter
                                           'default'; its child imports the production phase3_master_dewarp, pb_apply and
                                           cascade_dewarp, which gate G-D0b proves equal to the branch's and the child's
                                           fingerprint confirms; D7 with a prior generation); per-test lines kept for the
                                           step (cmg_result)
                        cubic              INSTALL_CONTENT_OK names the generation and 'files resampled cubic-spline'; every
                                           product's MDWINTP is cubic-spline (install G1 + R-MDEWARP-CUBIC-REAPPLY)
                        applied_screen     APPLIED_SCREEN.json names the generation; its screen exists with the recorded md5; its
                                           row fields equal the measurement's final row (install G1b)
                        products           membership.verify_products(gen, '_mdewarp', cell) is empty (install G-PRODUCTS)
                        D4_tested_pixels   every recomputed pbcorr sibling has >= D4_MIN_TESTED_PIXELS tested pixels (pure fix
                                           of pkg-masters-post defect D3: sibling_check passes at 0 tested pixels)
  (the archived screen record mdewarp_screens.csv is install's class 'screen_table' - crosscheck 4.2: one class owned by
   install (rows + install_cell_row in lib/install/tests.py); this step writes its applied=0 rows as that class and checks
   its own cell's row inline, check 'dewarp.record_row')

Context: the step passes it to add_output and stores it with the output in the unit record. Without context (`exopipe3
verify`, `--resume`) a context-dependent test takes it from the newest dewarp record listing that output (record directories
`_records/dewarp/` found above the product, and the production record root <GEN_ROOT>/_records/dewarp/); a context found that
way runs CMG in SCRATCH mode (a symlink to the generation in a temporary directory, so CMG's sidecars never overwrite the
generation's own).
"""
import ast
import csv
import glob
import json
import os
import re
import shutil
import tempfile

from exopipe.phase3.core.validate import content_test
from exopipe.phase3.lib.dewarp import steps_common as SC

STATUS_PREFIXES = ('REFERENCE ONLY', 'ALREADY DEWARPED', 'REFUSED', 'NOT ATTEMPTED', 'APPLY REFUSED')
_COLS = {}


def md_cols():
    """phase3_master_dewarp.COLS, read from its source (no import: the module loads PyBDSF)."""
    if 'md' not in _COLS:
        _COLS['md'] = list(SC.module_literals(SC.md_path(), ('COLS',))['COLS'])
    return _COLS['md']


def _f(x):
    return float(x)


# ---------------------------------------------------------------------------------------------------------------------
# context lookup (spec section 6 and blocker B1)
# ---------------------------------------------------------------------------------------------------------------------

def record_dirs(path):
    """_records/dewarp directories above ``path``, then the production record root."""
    out = []
    d = os.path.dirname(os.path.abspath(path))
    while True:
        r = os.path.join(d, '_records', 'dewarp')
        if os.path.isdir(r) and r not in out:
            out.append(r)
        nd = os.path.dirname(d)
        if nd == d:
            break
        d = nd
    try:
        r = os.path.join(SC.C().GEN_ROOT, '_records', 'dewarp')
        if os.path.isdir(r) and r not in out:
            out.append(r)
    except Exception:
        pass
    return out


def record_ctx(path, cls):
    """The context the newest dewarp unit record stored with this output (path + class), or None."""
    p = os.path.abspath(path).rstrip('/')
    best = None
    for r in record_dirs(p):
        for f in glob.glob(os.path.join(r, '*.json')):
            try:
                with open(f) as fh:
                    rec = json.load(fh)
            except (OSError, ValueError):
                continue
            for o in rec.get('outputs') or []:
                if o.get('path') == p and o.get('class') == cls and isinstance(o.get('ctx'), dict):
                    key = str(rec.get('finished') or '')
                    if best is None or key > best[0]:
                        best = (key, dict(o['ctx']), f)
    if best is None:
        return None
    ctx = best[1]
    ctx['_record'] = best[2]
    return ctx


# ---------------------------------------------------------------------------------------------------------------------
# dewarp_table
# ---------------------------------------------------------------------------------------------------------------------

@content_test('dewarp_table', name='rows')
def table_rows(path, lst=None, **ctx):
    from exopipe.phase3.core import cells
    if lst is None:
        m = re.match(r'^table_(\d\dh)\.csv$', os.path.basename(path))
        lst = m.group(1) if m else None
    if not lst:
        return False, None, 'no context: the LST (pass lst= or name the file table_<LST>.csv)'
    with open(path, newline='') as f:
        rd = csv.DictReader(f)
        cols = list(rd.fieldnames or [])
        rows = list(rd)
    if cols != md_cols():
        return False, len(rows), f'header {cols[:4]}... is not the {len(md_cols())} phase3_master_dewarp columns'
    c = SC.C()
    gmin, tol = float(c.HELDOUT_MIN_GAIN_AS), float(c.TABLE_GAIN_TOL_AS)
    final = {}
    for r in rows:
        if r['lst'] == lst:
            final[r['scope']] = r
    scopes = cells.scopes()
    problems = []
    missing = [s for s in scopes if s not in final]
    if missing:
        problems.append(f'no row for {len(missing)} scope(s) {missing[:4]}')
    extra = sorted(s for s in final if s not in scopes)
    if extra:
        problems.append(f'rows for unknown scopes {extra[:4]}')
    n_ok = n_ref = 0
    for s, r in sorted(final.items()):
        st = r['status'] or ''
        if st == 'OK' or st.startswith('REFERENCE ONLY'):
            try:
                mb, ma = _f(r['med_before']), _f(r['med_after'])
                na, nn = int(r['n_after']), int(r['n_anchors'])
                gain = _f(r['anchor_off_med_as']) - _f(r['cv_resid_med_as'])
            except (TypeError, ValueError):
                problems.append(f'{s}: {st[:14]} row with unparsable med/N/offset values')
                continue
            if not ma <= mb:          # (a) strict on the unrounded medians; the written values may tie at 3 decimals
                problems.append(f'{s}: {st[:14]} but med_after {ma} > med_before {mb}')
            if na < c.MIN_ANCHORS or nn < c.MIN_ANCHORS:
                problems.append(f'{s}: {st[:14]} with n_after {na} / n_anchors {nn} < {c.MIN_ANCHORS}')
            if st == 'OK':
                n_ok += 1
                if gain < gmin - tol - 1e-9:
                    problems.append(f"{s}: OK but written (b') gain {gain:.1f} < {gmin:g} - {tol:g} arcsec")
            else:
                n_ref += 1
                if gain >= gmin + tol - 1e-9:
                    problems.append(f"{s}: REFERENCE ONLY but written (b') gain {gain:.1f} >= {gmin:g} + {tol:g} arcsec")
        elif not st.startswith(STATUS_PREFIXES):
            problems.append(f'{s}: final status {st[:40]!r} is not a measurement outcome')
    value = f'{len(final)} scopes, {n_ok} OK, {n_ref} REFERENCE ONLY'
    if problems:
        return False, value, '; '.join(problems[:5])
    return True, value, f'{len(rows)} rows, final row per scope for all {len(scopes)} scopes of {lst}'


# ---------------------------------------------------------------------------------------------------------------------
# dewarp_screen
# ---------------------------------------------------------------------------------------------------------------------

@content_test('dewarp_screen', name='record')
def screen_record(path, table=None, master=None, **ctx):
    import numpy as np
    from astropy.io import fits
    from exopipe.phase3.core.record import fits_primary_cards
    p = os.path.abspath(path)
    m = re.match(r'^(.+)_warp_screen\.fits$', os.path.basename(p))
    if not m:
        return False, None, 'not a <scope>_warp_screen.fits file'
    scope = m.group(1)
    lst = os.path.basename(os.path.dirname(os.path.dirname(p)))
    how = 'given'
    if table is None:
        rc = record_ctx(p, 'dewarp_screen')
        if rc:
            table, master, how = rc.get('table'), master or rc.get('master'), 'from the run record'
    if table is None:     # MD's layout: <measure dir>/<LST>/maps/<scope>_warp_screen.fits and <measure dir>/tables/
        t = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(p))), 'tables', f'table_{lst}.csv')
        if os.path.isfile(t):
            table, how = t, 'from the measurement layout'
    if not table or not os.path.isfile(table):
        return False, None, f'no context: the measurement table of {lst} (table={table})'
    row = SC.final_rows(table, lst).get((lst, scope))
    if row is None:
        return False, None, f'{table} has no row for {lst} {scope}'
    with fits.open(p, memmap=False) as h:
        names = [x.name for x in h]
        if 'SCREEN_X' not in names or 'SCREEN_Y' not in names:
            return False, None, f'extensions {names}: SCREEN_X and SCREEN_Y required'
        sx, sy = h['SCREEN_X'].data, h['SCREEN_Y'].data
        hx = h['SCREEN_X'].header
    problems = []
    if sx is None or sy is None or sx.ndim != 2 or sx.shape != sy.shape:
        return False, None, f'SCREEN_X/SCREEN_Y shapes {getattr(sx, "shape", None)} / {getattr(sy, "shape", None)}'
    if not (np.isfinite(sx).all() and np.isfinite(sy).all()):
        problems.append(f'{int((~np.isfinite(sx)).sum() + (~np.isfinite(sy)).sum())} non-finite offsets')
    if hx.get('METHOD') != 'rbf':
        problems.append(f"METHOD {hx.get('METHOD')!r} != 'rbf'")
    if hx.get('KERNEL') != SC.C().WARP_RBF_KERNEL:
        problems.append(f"KERNEL {hx.get('KERNEL')!r} != {SC.C().WARP_RBF_KERNEL!r}")
    if hx.get('SUBBAND') != scope:
        problems.append(f"SUBBAND {hx.get('SUBBAND')!r} != {scope!r}")
    try:
        if int(hx.get('NANCHOR', -1)) != int(row['n_anchors']):
            problems.append(f"NANCHOR {hx.get('NANCHOR')} != the final row's n_anchors {row['n_anchors']}")
    except (TypeError, ValueError):
        problems.append(f"final row n_anchors {row.get('n_anchors')!r} unparsable (status {row['status'][:30]!r})")
    if master is None and row.get('input'):
        cand = os.path.join(row['input'], 'coadd_final', SC.C().MEASURE_FILES[0].format(scope=scope))
        if os.path.isfile(cand):
            master = cand
    shape_note = 'measured master not available (shape vs master not checked)'
    if master and os.path.isfile(master):
        mc = fits_primary_cards(master)
        mshape = (mc.get('NAXIS2'), mc.get('NAXIS1'))
        if mshape != sx.shape:
            problems.append(f'screen shape {sx.shape} != measured master {mshape} ({master})')
        shape_note = f'shape = master {os.path.basename(master)}'
    value = f'{sx.shape[0]}x{sx.shape[1]}, NANCHOR {hx.get("NANCHOR")}'
    if problems:
        return False, value, '; '.join(problems[:5]) + f' (table {how})'
    return True, value, f'{scope} vs final row {row["status"][:20]!r}; {shape_note}; table {how}'


# ---------------------------------------------------------------------------------------------------------------------
# dewarped_master
# ---------------------------------------------------------------------------------------------------------------------

_CMG = {}       # abs generation path -> the latest CMG result of this process


def cmg_result(gen):
    """The parsed result of the last CMG run on this generation in this process (for the step's per-test checks)."""
    return _CMG.get(os.path.abspath(gen).rstrip('/'))


def _list_lit(text):
    try:
        v = ast.literal_eval(text)
        return [str(x) for x in v] if isinstance(v, (list, tuple)) else []
    except (ValueError, SyntaxError):
        return []


def run_cmg(gen, cell, table, meas_root, working_root, rebuild_root, prior_gen=None, d6_waivers=None, mode='write',
            log_dir=None, **_ignored):
    """Run check_mdewarp_gen.py in place through core.legacy.run (dewarp_lst.sh:64-66: same arguments, site interpreter
    'default', exactly as production runs it). mode 'write': on the generation itself (CMG writes INSTALL_CONTENT_OK and
    APPLIED_SCREEN.json beside it, where the install reads them); 'scratch': on a symlink in a temporary directory (sidecars
    land there and are removed). Before the run, gate G-D0b (the production modules the child imports equal the branch's);
    after it, the child's code fingerprint is compared with what the gate checked. Returns the parsed result."""
    from exopipe.phase3.core import legacy
    script = SC.content_check_script()
    gen = os.path.abspath(gen).rstrip('/')
    if mode not in ('write', 'scratch'):
        raise ValueError(f'mode {mode!r}')
    gate_bad, gate_sums = SC.gate_child_modules(script)
    tmp = tempfile.mkdtemp(prefix='exopipe3_dewarp_cmg_')       # also the child's cwd: never a repository checkout
    gen_arg = gen
    if mode == 'scratch':
        os.makedirs(os.path.join(tmp, cell))
        gen_arg = os.path.join(tmp, cell, os.path.basename(gen))
        os.symlink(gen, gen_arg)
    args = [cell, '--gen', gen_arg, '--table', table, '--meas-root', meas_root, '--root', working_root,
            '--input-root', rebuild_root]
    if prior_gen:
        args += ['--prior-gen', prior_gen]
    waivers = dict(d6_waivers or {})
    if waivers:
        for f in sorted(waivers):
            args += ['--waive-d6-product', f]
        args += ['--waiver', ' | '.join(sorted(set(str(v) for v in waivers.values())))]
    log = os.path.join(log_dir, f'cmg_{cell}.log') if log_dir else os.path.join(tmp, f'cmg_{cell}.log')
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    try:
        if gate_bad:
            res = dict(rc=None, ok=False, lines={}, fingerprint={}, child_mismatch=gate_bad, d6_waived=[], d6_failed=[],
                       seconds=0.0, mode=mode, script=script, script_sha1=SC.sha1(script), log=None, argv=None, tail='')
            _CMG[gen] = res
            return res
        r = legacy.run(script, args, python='default', cwd=tmp, log=log, refusal_codes=())
        out = open(log, errors='replace').read() if os.path.isfile(log) else r.tail
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    lines = {}
    pat = re.compile(r'^' + re.escape(cell) + r' (D\d) (PASS|FAIL): (.*)$')
    for ln in out.splitlines():
        m = pat.match(ln)
        if m:
            lines[m.group(1)] = (m.group(2) == 'PASS', m.group(3))
    fp = dict(r.fingerprint or {})
    mismatch = [f'the child ran {p} with sha1 {fp[p]}, not the {h} checked by G-D0b' for p, h in gate_sums.items()
                if p in fp and fp[p] != h]
    br = SC.branch_root().rstrip('/') + '/'
    repo = (SC.child_repo(script) or '').rstrip('/') + '/'
    unchecked = sorted(p for p in fp if p.startswith(repo) and p not in gate_sums
                       and not p.endswith('__init__.py'))
    if not fp:
        mismatch.append('the child wrote no code fingerprint (core/parity_sitecustomize not loaded?)')
    d6_waived, d6_failed = [], []
    if 'D6' in lines:
        msg = lines['D6'][1]
        i = msg.rfind('): [')
        if 'WAIVED (' in msg and i >= 0:
            d6_waived = [x.split()[0] for x in _list_lit(msg[i + 3:]) if x.split()]
        m = re.search(r'3 s\.e\.\): (\[.*?\])(?:;|$)', msg)
        if m:
            d6_failed = [x.split()[0] for x in _list_lit(m.group(1)) if x.split()]
    res = dict(rc=r.rc, ok=(r.rc == 0 and f'{cell} MDEWARP-GEN-CONTENT-OK' in out and not mismatch),
               lines=lines, fingerprint=fp, child_mismatch=mismatch, child_unchecked=unchecked,
               child_branch_modules=sorted(p for p in fp if p.startswith(br)), d6_waived=d6_waived,
               d6_failed=d6_failed, seconds=r.wall_s, mode=mode, script=script, script_sha1=SC.sha1(script),
               log=log if log_dir else None, argv=r.argv, tail=out[-2000:])
    _CMG[gen] = res
    return res


def _gen_ctx(path, ctx, need=('cell', 'table', 'meas_root', 'working_root', 'rebuild_root')):
    """(gen, ctx, how): the given context, else the record's (mode forced to 'scratch'), else None."""
    gen = os.path.abspath(path).rstrip('/')
    if all(ctx.get(k) for k in need):
        return gen, dict(ctx), 'given'
    rc = record_ctx(gen, 'dewarped_master')
    if rc and all(rc.get(k) for k in need):
        rc['mode'] = 'scratch'
        return gen, rc, f'from the run record {rc.get("_record")}'
    return gen, None, None


def _cell_of(gen, ctx=None):
    from exopipe.phase3.core import cells
    cell = (ctx or {}).get('cell') or os.path.basename(os.path.dirname(gen))
    cells.parse_cell(cell)
    return cell


@content_test('dewarped_master', name='D1_D6')
def d1_d6(path, **ctx):
    gen, c, how = _gen_ctx(path, ctx)
    if c is None:
        return False, None, ('no context: the measurement table, measurement root and input roots of this generation '
                             '(no dewarp run record lists it)')
    res = run_cmg(gen, c['cell'], c['table'], c['meas_root'], c['working_root'], c['rebuild_root'],
                  prior_gen=c.get('prior_gen'), d6_waivers=c.get('d6_waivers'), mode=c.get('mode') or 'write',
                  log_dir=c.get('log_dir'))
    fails = sorted(t for t, (ok, _) in res['lines'].items() if not ok)
    tests = 'D1-D7' if c.get('prior_gen') else 'D1-D6'
    if res['ok']:
        value = f'{tests} PASS'
        detail = '; '.join(f'{t} {m[:90]}' for t, (_, m) in sorted(res['lines'].items()))
    else:
        value = 'FAIL ' + (','.join(fails) if fails else f'rc {res["rc"]}')
        if res['child_mismatch']:
            detail = '; '.join(res['child_mismatch'][:2])
        elif fails:
            detail = ' | '.join(f'{t}: {res["lines"][t][1]}' for t in fails[:2])[:300]
        else:
            detail = (f'CMG exit {res["rc"]}, no verdict line; output tail: ' + res['tail'][-240:].replace('\n', ' '))
    return res['ok'], value, f'{detail} [{res["mode"]} mode, {res["seconds"]} s, context {how}, cmg sha1 ' \
                             f'{res["script_sha1"][:12]}]'


def _fits_under(gen):
    out = []
    for sub in ('coadd_final', 'subtracted'):
        out += sorted(glob.glob(os.path.join(gen, sub, '**', '*.fits'), recursive=True))
    return out


@content_test('dewarped_master', name='cubic')
def cubic(path, **ctx):
    from exopipe.phase3.core.record import fits_primary_cards
    gen = os.path.abspath(path).rstrip('/')
    meth = SC.C().RESAMPLE_METHOD
    okf = os.path.join(os.path.dirname(gen), SC.CONTENT_OK)
    if not os.path.isfile(okf):
        return False, None, f'{okf} missing (written by the D1-D6 content test on success)'
    text = open(okf).read()
    if gen not in text:
        return False, None, f'{okf} does not name {gen}'
    if f'files resampled {meth}' not in text:
        return False, None, f"{okf} lacks 'files resampled {meth}' (not a {meth} content record)"
    files = _fits_under(gen)
    if not files:
        return False, 0, f'no FITS product under {gen}/coadd_final or subtracted'
    bad = [os.path.relpath(f, gen) for f in files if fits_primary_cards(f).get('MDWINTP') != meth]
    if bad:
        return False, len(bad), f'{len(bad)} of {len(files)} products lack MDWINTP={meth}, e.g. {bad[:2]}'
    line2 = (text.splitlines()[1:] or [''])[0]
    return True, len(files), f'{len(files)} products MDWINTP={meth}; {SC.CONTENT_OK}: {line2[:120]}'


@content_test('dewarped_master', name='applied_screen')
def applied_screen(path, table=None, **ctx):
    gen = os.path.abspath(path).rstrip('/')
    sj = os.path.join(os.path.dirname(gen), SC.APPLIED_SCREEN)
    if not os.path.isfile(sj):
        return False, None, f'{sj} missing (written by the D1-D6 content test on success)'
    try:
        meta = json.load(open(sj))
    except ValueError as e:
        return False, None, f'{sj} unreadable ({e})'
    if meta.get('gen') != gen:
        return False, None, f"{sj} names {meta.get('gen')}, not {gen}"
    scr = meta.get('screen') or ''
    if not os.path.isfile(scr):
        return False, None, f'screen {scr} missing'
    h = SC.md5(scr)
    if h != meta.get('screen_md5'):
        return False, h, f"screen {scr} md5 {h} != recorded {meta.get('screen_md5')}"
    cell = _cell_of(gen, ctx)
    lst, scope = cell.split('_', 1)
    if (meta.get('lst'), meta.get('scope')) != (lst, scope):
        return False, None, f"{sj} is for {meta.get('lst')} {meta.get('scope')}, not {cell}"
    how = 'given'
    if table is None:
        rc = record_ctx(gen, 'dewarped_master')
        if rc and rc.get('table'):
            table, how = rc['table'], 'from the run record'
    if table is None:      # MD layout of the screen: <measure dir>/<LST>/maps/<scope>_warp_screen.fits
        t = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(scr))), 'tables', f'table_{lst}.csv')
        if os.path.isfile(t):
            table, how = t, 'from the screen layout'
    if not table or not os.path.isfile(table):
        return False, None, f'no context: the measurement table (table={table})'
    row = SC.final_rows(table, lst).get((lst, scope))
    if row is None:
        return False, None, f'{table} has no row for {cell}'
    keys = ('status', 'n_anchors', 'med_before', 'med_after', 'anchor_off_med_as', 'cv_resid_med_as', 'warp_p50_as',
            'warp_p95_as', 'measured_at')
    diff = [k for k in keys if str(meta.get(k)) != str(row.get(k))]
    if diff:
        return False, len(diff), f'APPLIED_SCREEN.json differs from the final row of {table} in {diff[:4]}'
    return True, h, f'screen {os.path.basename(scr)} md5 verified; row fields equal the final row (table {how})'


@content_test('dewarped_master', name='products')
def products(path, **ctx):
    from exopipe.phase3.lib.masters import membership as MB
    gen = os.path.abspath(path).rstrip('/')
    cell = _cell_of(gen, ctx)
    if not os.path.isdir(os.path.join(gen, 'coadd_final')):
        return False, None, f'{gen} has no coadd_final/'
    v = MB.verify_products(gen, token=SC.C().GEN_TOKEN, cell=cell)
    if v:
        return False, len(v), ' | '.join(v[:3])[:300]
    return True, 0, f'membership.verify_products({cell}): flux scale, both PB maps, cubic dewarp where applied=1'


@content_test('dewarped_master', name='D4_tested_pixels')
def d4_tested_pixels(path, **ctx):
    import numpy as np
    from astropy.io import fits
    gen = os.path.abspath(path).rstrip('/')
    cell = _cell_of(gen, ctx)
    scope = cell.split('_', 1)[1]
    c = SC.C()
    cf = os.path.join(gen, 'coadd_final')
    sibs = sorted(f for f in glob.glob(os.path.join(cf, '*.fits')) if '_pbcorr_' in os.path.basename(f))
    if not sibs:
        return False, 0, f'no recomputed pbcorr sibling in {cf}'
    counts, bad = {}, []
    for s in sibs:
        b = os.path.basename(s)
        plain = os.path.join(cf, b.replace('_pbcorr_', '_', 1))
        pbn = os.path.join(cf, f'PB_deep_R075{c.GEN_TOKEN}_{scope}.fits' if '_R075_' in b else
                           f'PB_deep{c.GEN_TOKEN}_{scope}.fits')
        if not (os.path.isfile(plain) and os.path.isfile(pbn)):
            bad.append(f'{b}: plain or PB map missing')
            continue
        p = np.squeeze(fits.getdata(plain)).astype(np.float64)
        pb = np.squeeze(fits.getdata(pbn)).astype(np.float64)
        floor = float(fits.getheader(s).get('PBFLOOR', c.PB_FLOOR))
        tie = np.isfinite(pb) & (np.abs(pb - floor) <= 1e-6 * max(floor, 1.0))      # as pb_apply.sibling_check
        n = int((np.isfinite(pb) & (pb > floor) & np.isfinite(p) & ~tie).sum())
        counts[b] = n
        if n < c.D4_MIN_TESTED_PIXELS:
            bad.append(f'{b}: {n} tested pixels < {c.D4_MIN_TESTED_PIXELS}')
    if bad:
        return False, len(bad), '; '.join(bad[:3])
    return True, min(counts.values()), f'{len(sibs)} siblings, tested pixels min {min(counts.values())}'
