"""Absolute flux scale (per-LST VLSSr scale table) and band correction k x g (fluxcorr) of `exopipe3 calibrate`, deep tier.

scale     derive measure  ops/vlssr_flux_check.py IN PLACE (VLSSR_CAT_RUN, VLSSR_PB_ROBUST, VLSSR_OUT): ratios_73MHz.csv
                          (the cells skipped for a missing PB map are listed in the record: G14, B-60 default)
                 table    ops/pbstep2_20261001/scale_table_v4.py IN PLACE (<ratios> <table in force> <out>): s_new = s_old /
                          median ratio in the declination window; an LST of the table in force without matches refuses (G11).
                          The 'ref' text's literal date is replaced by the measure date (named amendment A3).
          diff            per-LST factor s_new / s_old; cells that would need a re-bake; served rows per LST
fluxcorr  derive stageA   lib/masters/shflux (the branch module = production phase3_master_shflux, git mv) --no-build --table,
                          one process per cell, STAGEA_WORKERS lists in parallel (run_stageA_20261001.sh)
                 bandvec  vlssr_flux_check.py IN PLACE with VLSSR_LOO_BAND=1 and VLSSR_BAND_VECTOR, then the ported heredoc of
                          run_band_vector_20261001.sh:16-28: k <- k / (1 + LOO median), normalised at 73 MHz, until every band
                          is within BANDVEC_TOL; not converged after BANDVEC_MAX_STEPS refuses (G13, "NOT adopted")
                 table    lib/masters/flux_correction.read_vector + sh_fit IN PROCESS; the CSV of build_table written with the
                          version from VERSION_TEMPLATE (build_table writes its module constant, so a new version could not
                          carry its own label); G12 (sh_fit RuntimeError) refuses
          diff            per-scope factor new/old; the masters' REQUIRED_FLUXVER change (G15); served rows per band
install  a NEW versioned file (+ provenance sidecar), the config patch (tables.* and every mirrored key, e.g.
         masters_fluxscale.*; fluxcorr: the version string everywhere - G15) and a draft ledger row; never production files
verify   scale: recompute the table from its ratios and the table in force; fluxcorr: recompute g, k x g, the version;
         band vector: 15 subbands, k(73 MHz) = 1, converged (its LOO medians with the final vector applied)
"""
import csv
import glob
import os
import shutil
import time

from exopipe.phase3.core.record import Refusal
from exopipe.phase3.lib.calibrate import common as K


def provenance_path(path):
    return path + '.provenance.json'


def read_provenance(path):
    p = provenance_path(path)
    return K.read_json(p) if os.path.exists(p) else None


def _yyyymmdd():
    return time.strftime('%Y%m%d', time.gmtime())


def split_version(v):
    """'v5' -> ('v5', today's YYYYMMDD); 'v2-20261001' -> ('v2', '20261001'): the version and the date that production
    writes into file names and the fluxcorr version string (a parity rebuild of v2 / v4 passes --version v2-20261001)."""
    import re
    m = re.match(r'^(v\d+)-(\d{8})$', v or '')
    return (m.group(1), m.group(2)) if m else (v, _yyyymmdd())


def next_version(cls):
    """'v5' after the installed scale v4; 'v3' after fluxcorr-v2 (the integer of the version in force + 1)."""
    import re
    T = K.CFG().tables
    cur = T.SCALE_TABLE_VERSION if cls == 'scale' else T.FLUXCORR_VERSION
    m = re.search(r'v(\d+)', cur)
    return f'v{int(m.group(1)) + 1}' if m else 'v1'


def _cells_by_lst():
    root = K.CFG().paths.COADD_SB_ROOT
    out = {}
    if os.path.isdir(root):
        for c in os.listdir(root):
            if len(c) > 4 and c[2] == 'h' and c[3] == '_':
                out.setdefault(c[:3], []).append(c)
    return out


def _served_rows(ctx, key):
    """{lst or band: n rows} of the served list (deep: master_catalog_asrun.csv or master_catalog.csv of the served run)."""
    run_dir = ctx.args.served_run or ctx.C.calibrate.SERVED_RUN.get('deep')
    p = ctx.args.list
    if not p and run_dir:
        for n in ctx.C.calibrate.DIFF_LIST_CSV:
            if os.path.exists(os.path.join(run_dir, n)):
                p = os.path.join(run_dir, n)
                break
    out = {}
    if p and os.path.exists(p):
        with open(p, newline='') as f:
            for r in csv.DictReader(f):
                out[r.get(key, '')] = out.get(r.get(key, ''), 0) + 1
    return p, out


# ---------------------------------------------------------------------------------------------------------------------
# scale
# ---------------------------------------------------------------------------------------------------------------------

def vlssr_env(ctx, out, extra=None):
    S = ctx.C.calibrate_scale
    env = {'VLSSR_CAT_RUN': ctx.args.cat_run or S.CAT_RUN, 'VLSSR_PB_ROBUST': S.PB_ROBUST, 'VLSSR_OUT': out}
    env.update(extra or {})
    return env


def _skipped(tail):
    for ln in reversed(tail.splitlines()):
        if ln.startswith('cells skipped for a missing PB map:'):
            return ln.split(':', 1)[1].strip()
    return None


def scale_measure(ctx):
    d = ctx.stage_dir('measure')
    script = K.ops_script('vlssr_check')
    env = vlssr_env(ctx, d)
    if ctx.dry:
        return {'script': script, 'env': env, 'out': os.path.join(d, 'ratios_73MHz.csv')}
    if ctx.skip_if_current('derive-measure'):
        return None
    K.fresh_stage_dir(d, ctx.args.fresh)
    with ctx.run.unit(ctx.unit_name('derive-measure'), cls=ctx.cls, tier=ctx.tier, version=ctx.version, stage='measure') as u:
        u.add_input(script)
        res = K.run_legacy(u, script, [], env=env, log=os.path.join(d, 'run.log'))
        sk = _skipped(res.tail)
        u.meta['skipped_cells'] = sk
        u.note(f'cells skipped for a missing PB map (listed, not refused: B-60 default): {sk}')
        rp = os.path.join(d, 'ratios_73MHz.csv')
        K.need_file(rp, 'vlssr_flux_check output')
        u.add_output(rp, 'vlssr_ratios')
    return None


def scale_table(ctx):
    C = ctx.C
    d = ctx.stage_dir('table')
    old = ctx.args.old_scale_table or C.tables.SCALE_TABLE
    ver, ymd = split_version(ctx.version)
    out = os.path.join(d, f'vlssr_scale_table_{ver}_{ymd}.csv')
    if ctx.dry:
        return {'ratios': '(the measure record of this version, or --ratios)', 'table_in_force': old, 'out': out}
    if ctx.skip_if_current('derive-table'):
        return None
    if ctx.args.ratios:
        ratios, mdate = K.need_file(ctx.args.ratios, '--ratios'), None
    else:
        rec = K.require_done(ctx.out, ctx.unit_name('derive-measure'), 'scale table')
        ratios = K.outputs_of(rec, 'vlssr_ratios')[0]
        mdate = (rec.get('finished') or '')[:10]
    K.fresh_stage_dir(d, ctx.args.fresh)
    script = K.ops_script('scale_table')
    with ctx.run.unit(ctx.unit_name('derive-table'), cls=ctx.cls, tier=ctx.tier, version=ctx.version, stage='table') as u:
        for p in (ratios, K.need_file(old, 'the scale table in force'), script):
            u.add_input(p)
        raw = os.path.join(d, 'scale_table_v4_output.csv')
        K.run_legacy(u, script, [ratios, old, raw], log=os.path.join(d, 'scale_table.log'))
        # named amendment A3: the script writes its own date literally into 'ref'
        S = C.calibrate_scale
        # the measure record's date; else the date of an explicit dated version (--version v4-20261001); else today
        date = mdate or (f'{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}' if '-' in ctx.version else time.strftime('%Y-%m-%d', time.gmtime()))
        rows = K.read_csv(raw)
        fields = list(rows[0].keys()) if rows else []
        for r in rows:
            r['ref'] = r['ref'].replace(f're-measured {S.REF_DATE_LITERAL}', f're-measured {date}')
        K.write_csv(out, fields, rows)
        K.write_json(provenance_path(out), {'class': 'scale_table', 'ratios': ratios, 'previous': old, 'measured': date,
                                            'version': ctx.version, 'script': script})
        u.add_output(out, 'scale_table')
    return None


def scale_rows(path):
    return {r['lst']: r for r in K.read_csv(path)}


def recompute_scale(ratios_path, old_path):
    """The scale table recomputed from config (DEC_WINDOW_DEG, ERR_MEDIAN_FACTOR, ROUND_DIGITS) - the arithmetic of
    scale_table_v4.py:11-32: {lst: (scale, err, nsrc, median_ratio, scale_old, rescale)}; raises on a missing LST."""
    import numpy as np
    S = K.CFG().calibrate_scale
    lo, hi = S.DEC_WINDOW_DEG
    old = {r['lst']: float(r['scale']) for r in K.read_csv(old_path)}
    by = {}
    for r in K.read_csv(ratios_path):
        try:
            dec, ratio = float(r['dec']), float(r['ratio'])
        except (KeyError, ValueError):
            continue
        if lo <= dec <= hi and np.isfinite(ratio) and ratio > 0:
            by.setdefault(r['lst'], []).append(ratio)
    missing = sorted(set(old) - set(by))
    if missing:
        raise ValueError(f'no VLSSr matches in the window for {missing}')
    nd = S.ROUND_DIGITS
    out = {}
    for lst in sorted(old):
        v = np.array(by[lst])
        med = float(np.median(v))
        n = len(v)
        err = S.ERR_MEDIAN_FACTOR * float(np.std(v)) / np.sqrt(n) / med
        out[lst] = (round(old[lst] / med, nd['scale']), round(err, nd['err']), n, round(med, nd['median_ratio']), old[lst],
                    round(1.0 / med, nd['rescale']))
    return out


def scale_diff(ctx, cand, inst_p, d, u):
    new, old = scale_rows(cand), scale_rows(inst_p)
    served_list, served = _served_rows(ctx, 'lst')
    cells = _cells_by_lst()
    rows = []
    for lst in sorted(set(new) | set(old)):
        a, b = old.get(lst), new.get(lst)
        fa = float(a['scale']) if a else None
        fb = float(b['scale']) if b else None
        rows.append({'lst': lst, 'scale_installed': fa, 'scale_candidate': fb,
                     'factor': (round(fb / fa, 6) if fa and fb else ''), 'n_cells': len(cells.get(lst, [])),
                     'n_served_rows': served.get(lst, 0)})
    rp = K.write_csv(os.path.join(d, 'scale_diff.csv'), ['lst', 'scale_installed', 'scale_candidate', 'factor', 'n_cells',
                                                         'n_served_rows'], rows)
    f = [abs(r['factor'] - 1) for r in rows if r['factor'] != '']
    return {'rows_csv': rp, 'n_lst': len(rows), 'max_abs_factor_minus_1': max(f) if f else None,
            'cells_needing_rebake': sum(r['n_cells'] for r in rows if r['factor'] not in ('', 1.0)),
            'served_rows_rescaled': sum(r['n_served_rows'] for r in rows if r['factor'] not in ('', 1.0)),
            'served_list': served_list}


# ---------------------------------------------------------------------------------------------------------------------
# fluxcorr
# ---------------------------------------------------------------------------------------------------------------------

def stagea(ctx):
    from exopipe.phase3.core import cells
    C = ctx.C
    d = ctx.stage_dir('stageA')
    if not ctx.args.cells:
        raise Refusal('fluxcorr stage A needs an explicit cell list (--cells @file): the 2026-10-01 list (444 cells, every '
                      'cell not master-dewarped then) is /fast/gh/main/phase3/shflux/stageA_pbstep2_20261001/cells_*.txt; '
                      'no rule chooses the cells by itself')
    sel = cells.expand_cells(ctx.args.cells)
    if ctx.args.exclude_cells:
        ex = set(cells.expand_cells(ctx.args.exclude_cells))
        sel = [c for c in sel if c not in ex]
    nw = ctx.workers(C.calibrate_fluxcorr.STAGEA_WORKERS)
    lists = [sel[k::nw] for k in range(nw)]
    root = C.paths.COADD_SB_ROOT
    if ctx.dry:
        return {'cells': len(sel), 'workers': nw, 'root': root,
                'tables': [os.path.join(d, f'shflux_measure_{k}.csv') for k in range(nw) if lists[k]]}
    if ctx.skip_if_current('derive-stageA'):
        return None
    K.fresh_stage_dir(d, ctx.args.fresh)
    with ctx.run.unit(ctx.unit_name('derive-stageA'), cls=ctx.cls, tier=ctx.tier, version=ctx.version, stage='stageA') as u:
        K.write_text(os.path.join(d, 'cells.txt'), ''.join(f'{c}\n' for c in sel))

        def worker(k):
            tab = os.path.join(d, f'shflux_measure_{k}.csv')
            for c in lists[k]:
                lst, sc = cells.parse_cell(c)
                K.run_legacy(u, 'exopipe.phase3.lib.masters.shflux', [lst, sc, '--root', root, '--no-build', '--table', tab],
                             module=True, cwd=K.repo_root(), log=os.path.join(d, f'measure_{k}.log'))
            return tab
        tabs = K.parallel([lambda k=k: worker(k) for k in range(nw) if lists[k]], nw)
        got = set()
        for t in tabs:
            for r in K.read_csv(t):
                got.add(f"{r['lst']}_{r['scope']}")
        miss = sorted(set(sel) - got)
        u.check('calibrate.stagea.every_cell_has_a_row', not miss, value=len(miss), detail=f'cells without a row: {miss[:10]}')
        for t in tabs:
            u.add_output(t, 'shflux_stagea')
    return None


def _loo_arith(V, T, NV, norm):
    """run_band_vector_20261001.sh:16-28 (heredoc), ported: k <- k / (1 + median_b), renormalised at `norm`; returns
    (worst |median|, {band: median})."""
    k = {r['subband']: float(r['factor']) for r in K.read_csv(V)}
    m = {r['subband']: float(r['raw_median_pct']) / 100.0 for r in K.read_csv(T)}
    k2 = {b: k[b] / (1.0 + m.get(b, 0.0)) for b in k}
    n = k2[norm]
    k2 = {b: v / n for b, v in k2.items()}
    # written exactly as the heredoc writes it (csv.writer default line ends, CRLF; :22-23): a vector adopted after step
    # 2+ is then byte-identical to production's recipe (band_vector_final_20261001.csv = band_vector_loo6.csv, CRLF)
    tmp = f'{NV}.tmp.{os.getpid()}'
    with open(tmp, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['subband', 'factor'])
        for b, v in k2.items():
            w.writerow([b, f'{v:.6f}'])
    os.replace(tmp, NV)
    worst = max(abs(v) for v in m.values())
    return worst, m


def bandvec(ctx):
    C = ctx.C
    F = C.calibrate_fluxcorr
    d = ctx.stage_dir('bandvec')
    start = ctx.args.band_vector_start or F.BANDVEC_START or C.tables.BAND_VECTOR
    script = K.ops_script('vlssr_check')
    if ctx.dry:
        return {'start_vector': start, 'max_steps': F.BANDVEC_MAX_STEPS, 'tol': F.BANDVEC_TOL, 'script': script,
                'env': vlssr_env(ctx, os.path.join(d, 'loo<k>'), {'VLSSR_LOO_BAND': '1' if F.BANDVEC_LOO else '0',
                                                                  'VLSSR_BAND_VECTOR': '<vector of the step>'})}
    if ctx.skip_if_current('derive-bandvec'):
        return None
    K.fresh_stage_dir(d, ctx.args.fresh)
    with ctx.run.unit(ctx.unit_name('derive-bandvec'), cls=ctx.cls, tier=ctx.tier, version=ctx.version, stage='bandvec') as u:
        u.add_input(K.need_file(start, 'start band vector'))
        u.add_input(script)
        V, steps = start, []
        for step in range(1, F.BANDVEC_MAX_STEPS + 1):
            out = os.path.join(d, f'loo{step}')
            env = vlssr_env(ctx, out, {'VLSSR_LOO_BAND': '1' if F.BANDVEC_LOO else '0', 'VLSSR_BAND_VECTOR': V})
            K.run_legacy(u, script, [], env=env, log=os.path.join(d, f'loo{step}.log'))
            T = os.path.join(out, 'smooth_by_subband_powerlaw.csv')
            NV = os.path.join(d, f'band_vector_loo{step}.csv')
            worst, m = _loo_arith(V, T, NV, F.BANDVEC_NORM_BAND)
            steps.append({'step': step, 'applied_vector': V, 'loo_medians': T, 'worst': worst, 'next_vector': NV})
            if worst < F.BANDVEC_TOL:
                final = os.path.join(d, f'band_vector_final_{_yyyymmdd()}.csv')
                shutil.copyfile(V, final)
                it = K.write_json(os.path.join(d, 'iterate.json'), {'steps': steps, 'converged': True, 'final': final,
                                                                    'tol': F.BANDVEC_TOL})
                K.write_json(provenance_path(final), {'class': 'band_vector', 'convergence': T, 'iterate': it,
                                                      'start': start, 'version': ctx.version})
                u.add_output(final, 'band_vector')
                return None
            V = NV
        K.write_json(os.path.join(d, 'iterate.json'), {'steps': steps, 'converged': False, 'tol': F.BANDVEC_TOL})
        raise Refusal(f'G13: band vector NOT converged after {F.BANDVEC_MAX_STEPS} steps (worst {100 * steps[-1]["worst"]:.2f} '
                      f'per cent > {100 * F.BANDVEC_TOL:.1f}) - NOT adopted; last vector {V} (continue with --from bandvec '
                      f'--band-vector-start {V} --fresh)')


def fit_table(stagea_files, vector, scale_table, version, yyyymmdd):
    """(rows of the table CSV, shcells rows, fit dict) - lib/masters/flux_correction.read_vector + sh_fit in process; the
    CSV of build_table (:65-70) with this version's label (VERSION_TEMPLATE)."""
    from exopipe.phase3.lib.masters import flux_correction as FC
    F = K.CFG().calibrate_fluxcorr
    k = FC.read_vector(vector)
    st = scale_rows(scale_table)
    resc = {lst: float(r['rescale']) for lst, r in st.items()}
    sdate = (next(iter(st.values())).get('measured_at') or '')[:10]
    try:
        g, ge, n, chi, rows, g_ivw, ge_ivw = FC.sh_fit(sorted(stagea_files), k, lst_rescale=resc)
    except RuntimeError as e:
        raise Refusal(f'S&H fit: {e} (gate G12)')
    label = F.VERSION_TEMPLATE.format(version=version, yyyymmdd=yyyymmdd, scale_date=sdate)
    out = []
    for sc in tuple(F.SUBBANDS) + tuple(F.WIDEBAND):
        out.append({'scope': sc, 'k_band': f'{k[sc]:.6f}', 'g_abs': f'{g:.6f}', 'g_err': f'{ge:.6f}', 'fluxcorr': f'{k[sc] * g:.6f}',
                    'n_sh_cells': n, 'sh_reduced_chi': f'{chi:.3f}', 'g_ivw_reference': f'{g_ivw:.6f}', 'version': label})
    return out, rows, dict(g=g, g_err=ge, n=n, chi=chi, g_ivw=g_ivw, label=label)


TABLE_COLS = ['scope', 'k_band', 'g_abs', 'g_err', 'fluxcorr', 'n_sh_cells', 'sh_reduced_chi', 'g_ivw_reference', 'version']


def fc_table(ctx):
    C = ctx.C
    d = ctx.stage_dir('table')
    ver, ymd = split_version(ctx.version)
    out = os.path.join(d, f'flux_correction_{ver}_{ymd}.csv')
    scale = ctx.args.scale_candidate or C.tables.SCALE_TABLE
    if ctx.dry:
        return {'stagea': ctx.args.stagea or '(the stageA record of this version)', 'band_vector':
                ctx.args.band_vector or '(the bandvec record of this version)', 'scale_table': scale, 'out': out}
    if ctx.skip_if_current('derive-table'):
        return None
    if ctx.args.stagea:
        files = sorted(glob.glob(ctx.args.stagea))
        if not files:
            raise Refusal(f'--stagea {ctx.args.stagea}: no file')
    else:
        files = K.outputs_of(K.require_done(ctx.out, ctx.unit_name('derive-stageA'), 'fluxcorr table'), 'shflux_stagea')
    if ctx.args.band_vector:
        vec = K.need_file(ctx.args.band_vector, '--band-vector')
    else:
        vec = K.outputs_of(K.require_done(ctx.out, ctx.unit_name('derive-bandvec'), 'fluxcorr table'), 'band_vector')[0]
    K.fresh_stage_dir(d, ctx.args.fresh)
    with ctx.run.unit(ctx.unit_name('derive-table'), cls=ctx.cls, tier=ctx.tier, version=ctx.version, stage='table') as u:
        for p in files + [vec, K.need_file(scale, 'scale table')]:
            u.add_input(p)
        rows, sh, fit = fit_table(files, vec, scale, ver, ymd)
        with open(out + '.tmp', 'w', newline='') as f:     # csv.writer default line ends, as flux_correction.build_table (:65-68)
            w = csv.writer(f)
            w.writerow(TABLE_COLS)
            for r in rows:
                w.writerow([r[c] for c in TABLE_COLS])
        os.replace(out + '.tmp', out)
        shp = out.replace('.csv', '_shcells.csv')
        with open(shp + '.tmp', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['lst', 'scope', 'sh_over_k', 'err'])
            w.writerows(sh)
        os.replace(shp + '.tmp', shp)
        K.write_json(provenance_path(out), {'class': 'fluxcorr_table', 'stagea': files, 'band_vector': vec, 'scale_table': scale,
                                            'shcells': shp, 'version': ver, 'yyyymmdd': ymd, 'fit': fit})
        u.note(f"g {fit['g']:.6f} +- {fit['g_err']:.6f} over {fit['n']} cells (reduced chi {fit['chi']:.3f}; ivw {fit['g_ivw']:.6f})")
        u.add_output(out, 'fluxcorr_table')
    return None


def fc_diff(ctx, cand, inst_p, d, u):
    new = {r['scope']: r for r in K.read_csv(cand)}
    old = {r['scope']: r for r in K.read_csv(inst_p)}
    served_list, served = _served_rows(ctx, 'band')
    rows = []
    for sc in sorted(set(new) | set(old)):
        a, b = old.get(sc), new.get(sc)
        fa = float(a['fluxcorr']) if a else None
        fb = float(b['fluxcorr']) if b else None
        rows.append({'scope': sc, 'fluxcorr_installed': fa, 'fluxcorr_candidate': fb,
                     'factor': (round(fb / fa, 6) if fa and fb else ''), 'n_served_rows': served.get(sc, 0)})
    rp = K.write_csv(os.path.join(d, 'fluxcorr_diff.csv'), ['scope', 'fluxcorr_installed', 'fluxcorr_candidate', 'factor',
                                                           'n_served_rows'], rows)
    vo = (next(iter(old.values()))['version'] if old else '').split(' ')[0]
    vn = (next(iter(new.values()))['version'] if new else '').split(' ')[0]
    cells = _cells_by_lst()
    return {'rows_csv': rp, 'version_installed': vo, 'version_candidate': vn, 'required_fluxver_changes': vo != vn,
            'cells_needing_rebake': sum(len(v) for v in cells.values()) if vo != vn else 0,
            'served_rows_rescaled': sum(r['n_served_rows'] for r in rows if r['factor'] not in ('', 1.0)),
            'served_list': served_list}


# ---------------------------------------------------------------------------------------------------------------------
# candidate, diff, install, verify (both classes)
# ---------------------------------------------------------------------------------------------------------------------

def candidate_path(ctx):
    if ctx.args.candidate:
        return K.need_file(ctx.args.candidate, '--candidate')
    rec = K.require_done(ctx.out, ctx.unit_name('derive-table'), ctx.action)
    cls = 'scale_table' if ctx.cls == 'scale' else 'fluxcorr_table'
    return K.outputs_of(rec, cls)[0]


def installed(ctx):
    T = ctx.C.tables
    return (T.SCALE_TABLE, T.SCALE_TABLE_SHA256) if ctx.cls == 'scale' else (T.FLUXCORR_TABLE, T.FLUXCORR_TABLE_SHA256)


def diff(ctx):
    d = ctx.stage_dir('diff')
    inst_p, pin = installed(ctx)
    if ctx.dry:
        try:
            cp = candidate_path(ctx)
        except Refusal as e:
            cp = f'(not yet: {e})'
        return {'candidate': cp, 'installed': inst_p}
    if ctx.skip_if_current('diff'):
        return None
    cp = candidate_path(ctx)
    K.fresh_stage_dir(d, ctx.args.fresh)
    with ctx.run.unit(ctx.unit_name('diff'), cls=ctx.cls, tier=ctx.tier, version=ctx.version, stage='diff') as u:
        u.add_input(cp)
        u.add_input(inst_p)
        sha = K.sha256(K.need_file(inst_p, 'installed table'))
        if sha != pin:
            raise Refusal(f'installed table {inst_p}: sha256 {sha} != pin {pin} (gate G19)')
        part = scale_diff(ctx, cp, inst_p, d, u) if ctx.cls == 'scale' else fc_diff(ctx, cp, inst_p, d, u)
        summ = {'class': ctx.cls, 'tier': ctx.tier, 'version': ctx.version, 'candidate': cp, 'candidate_sha256': K.sha256(cp),
                'installed': inst_p, 'installed_sha256': sha, 'served': part}
        sp = K.write_json(os.path.join(d, 'diff_summary.json'), summ)
        u.meta['candidate_sha256'] = summ['candidate_sha256']
        u.add_output(sp, 'calibration_diff')
    return None


def install(ctx):
    C = ctx.C
    T = C.tables
    d = ctx.stage_dir('install')
    inst_p, pin = installed(ctx)
    if ctx.dry:
        try:
            cp = candidate_path(ctx)
        except Refusal as e:
            cp = f'(not yet: {e})'
        return {'candidate': cp, 'installed': inst_p, 'patch': os.path.join(d, 'config_patch.diff'),
                'ledger_draft': os.path.join(d, 'ledger_row_draft.csv')}
    if ctx.skip_if_current('install'):
        return None
    rid = K.check_ruling(ctx.args)
    cp = candidate_path(ctx)
    cand_sha = K.sha256(cp)
    diff_rec = K.diff_gate(ctx.out, ctx.cls, ctx.tier, ctx.version, cand_sha, installed_sha=K.sha256(K.need_file(inst_p, 'installed table')))
    K.fresh_stage_dir(d, ctx.args.fresh)
    with ctx.run.unit(ctx.unit_name('install'), cls=ctx.cls, tier=ctx.tier, version=ctx.version, stage='install') as u:
        u.add_input(cp)
        base_sha = K.sha256(K.need_file(inst_p, 'installed table'))
        if base_sha != pin:
            raise Refusal(f'installed table {inst_p}: sha256 {base_sha} != pin {pin} (gate G19)')
        vroot = C.calibrate.VERSIONED_TABLES_ROOT if (C.calibrate.VERSIONED_TABLES_ROOT and C.paths.PRODUCTION_WRITES) else d
        versioned = os.path.join(vroot, os.path.basename(cp))
        shutil.copyfile(cp, versioned)
        prov = read_provenance(cp) or {}
        files = {'table': {'path': versioned, 'sha256': K.sha256(versioned)}}
        rep = K.pointer_replacements(inst_p, versioned, inst_p, versioned)
        scoped = []
        literals = [inst_p, base_sha]
        if ctx.cls == 'scale':
            ver, ymd = split_version(ctx.version)
            newv = f'{ver}-{os.path.splitext(os.path.basename(cp))[0].rsplit("_", 1)[-1]}'
            rep[T.SCALE_TABLE_VERSION] = newv
            scoped.append(('tables', 'SCALE_TABLE_PREVIOUS', {T.SCALE_TABLE_PREVIOUS: prov.get('previous') or inst_p}))
            scoped.append(('tables', 'SCALE_TABLE_INPUTS', {T.SCALE_TABLE_INPUTS['ratios']: prov.get('ratios'),
                                                            T.SCALE_TABLE_INPUTS['previous']: prov.get('previous')}))
        else:
            sh_src = cp.replace('.csv', '_shcells.csv')
            sh_v = os.path.join(vroot, os.path.basename(sh_src))
            shutil.copyfile(K.need_file(sh_src, 'shcells table'), sh_v)
            files['shcells'] = {'path': sh_v, 'sha256': K.sha256(sh_v)}
            rep.update(K.pointer_replacements(T.FLUXCORR_SHCELLS, sh_v, T.FLUXCORR_SHCELLS, sh_v))
            old_ver = T.FLUXCORR_VERSION
            new_ver = K.read_csv(cp)[0]['version'].split(' ')[0]
            rep[old_ver] = new_ver
            literals.append(old_ver)
            vec = prov.get('band_vector')
            if vec and os.path.exists(vec) and K.sha256(vec) != T.BAND_VECTOR_SHA256:
                vv = os.path.join(vroot, os.path.basename(vec))
                if os.path.abspath(vv) != os.path.abspath(vec):
                    shutil.copyfile(vec, vv)
                files['band_vector'] = {'path': vv, 'sha256': K.sha256(vv)}
                rep.update(K.pointer_replacements(T.BAND_VECTOR, vv, T.BAND_VECTOR, vv))
                vp = read_provenance(vec) or {}
                if vp.get('convergence'):
                    scoped.append(('tables', 'BAND_VECTOR_CONVERGENCE', {T.BAND_VECTOR_CONVERGENCE: vp['convergence']}))
            stage_glob = (os.path.join(os.path.dirname(prov['stagea'][0]), 'shflux_measure_*.csv')
                          if prov.get('stagea') else T.FLUXCORR_INPUTS['stagea_glob'])
            scoped.append(('tables', 'FLUXCORR_INPUTS', {T.FLUXCORR_INPUTS['stagea_glob']: stage_glob,
                                                         T.FLUXCORR_INPUTS['band_vector']: files.get('band_vector', {}).get('path', T.BAND_VECTOR),
                                                         T.FLUXCORR_INPUTS['scale_table']: prov.get('scale_table') or T.SCALE_TABLE}))
        patch, changed = K.config_patch(rep, scoped=scoped)
        if ctx.cls == 'fluxcorr' and rep.get(T.FLUXCORR_VERSION) != T.FLUXCORR_VERSION:
            # gate G15: the masters' product test (REQUIRED_FLUXVER) changes in the same patch as the table version
            req = [f'{s}.{k}' for s, k in changed if k == 'REQUIRED_FLUXVER']
            u.check('calibrate.install.G15_required_fluxver', bool(req), value=req or None,
                    detail='a fluxcorr install changes FLUXCORR_VERSION and the masters REQUIRED_FLUXVER key together (G15)')
        pp = K.write_text(os.path.join(d, 'config_patch.diff'), patch)
        ds = K.read_json(K.outputs_of(K.load_record(ctx.out, ctx.unit_name('diff')), 'calibration_diff')[0]) if diff_rec else {}
        desc = (f'calibrate {ctx.cls} install {ctx.version}: {cp} (sha256 {cand_sha}) -> {versioned}; replaces {inst_p} (sha256 '
                f'{base_sha}); diff {diff_rec} ({json_short(ds.get("served"))}); config patch {pp} changes '
                f'{", ".join(f"{s}.{k}" for s, k in changed)}')
        lp = os.path.join(d, 'ledger_row_draft.csv')
        K.ledger_draft(lp, ctx.cls, ctx.tier, ctx.version, [f['path'] for f in files.values()] + [pp], desc, rid,
                       ctx.args.install_words)
        man = {'class': ctx.cls, 'tier': ctx.tier, 'version': ctx.version, 'family_rule': ctx.cls,
               'candidate': {'path': cp, 'sha256': cand_sha}, 'base': {'path': inst_p, 'sha256': base_sha},
               'versioned': files['table'], 'files': files, 'patch': pp, 'patch_changes': list(dict.fromkeys(f'{s}.{k}' for s, k in changed)),
               'ledger_draft': lp, 'diff_record': diff_rec, 'ruling': {'id': rid, 'words': ctx.args.install_words},
               'code_literals': [f'{a} ({b})' for a, b in K.code_literals(literals)],
               'production_writes': bool(C.paths.PRODUCTION_WRITES)}
        mp = K.write_json(os.path.join(d, 'install_manifest.json'), man)
        u.add_output(mp, 'install_bundle')
    return None


def json_short(x):
    import json
    return json.dumps(x)[:300] if x else ''


def verify_targets(ctx):
    T = ctx.C.tables
    if ctx.args.candidate:
        return [(ctx.args.candidate, 'candidate', 'scale_table' if ctx.cls == 'scale' else 'fluxcorr_table')]
    if ctx.cls == 'scale':
        return [(T.SCALE_TABLE, 'installed', 'scale_table')]
    return [(T.FLUXCORR_TABLE, 'installed', 'fluxcorr_table'), (T.BAND_VECTOR, 'installed', 'band_vector')]
