"""Template-fit position error scale K per tier and Stokes (`exopipe3 calibrate template-k`).

K = the 68th percentile of |position error| x S/N / FWHM per beam axis (major, minor) over injections with a FORCED start
and the injected S/N (the deep definition; ledger 2285). The VALUES live in config fitting.K_TEMPLATE; this class measures
them and installs a new value as a config patch.

derive  inject          deep: deep_forced_kcal_20261013/kinj_deep.py IN PLACE (the re-measure harness after the template mirror
                        fix, exopipe_ops 554b721: production deep_forced_radio + clv15 44f0155) on the ratified plan (30 frames,
                        10 cells, the same seeds), --per-frame DEEP_PER_FRAME, one process per frame -> trials_<i>.csv;
                        sub-LST: staged_20261012_sublst_fitted_positions/tests/kcal_tenmin_inject.py IN PLACE on a sub-LST
                        catalogue directory (--catalog-dir) -> trials.csv (orientation calibrate_template_k.INJECT_ORIENTATION)
        summary         port of kinj_deep.summary (:91-130): K per Stokes and axis, 16-84 per cent bootstrap, status counts,
                        from this version's inject record, or --trials, or (deep) the stored trials of tables.K_RECORDS['deep']
                        -> k_summary.json (class template_k_record)
        snapshot        NOT available: the template_fit_validation trials are not on disk (spec B6)
diff    K_new / K_old per axis against fitting.K_TEMPLATE; template-fit rows of the served list (when it has pos_kind)
install gates G16 (0 < K < 5, both Stokes), G17 (the record's template orientation = fitting.TEMPLATE_BEAM_ORIENTATION),
        G18 (--install-ruling); the record as a versioned file, a patch of fitting.K_TEMPLATE, a draft ledger row
verify  deep: K recomputed from the stored trials equals fitting.K_TEMPLATE (3 decimals); sub-LST / snapshot: the config
        values only (bounds, both Stokes, ruled) - their trials are not on Lustre
"""
import glob
import os
import shutil

from exopipe.phase3.core.record import Refusal
from exopipe.phase3.lib.calibrate import common as K


def k_percentile(rows, ek, bk, sk, pct):
    import numpy as np
    v = [abs(float(r[ek])) * abs(float(r[sk])) / float(r[bk]) for r in rows]
    return float(np.percentile(v, pct)) if v else float('nan')


def summarise(trials_files, tier, orientation):
    """The K record of a set of trials files (port of kinj_deep.summary, :91-118; columns per tier from TRIALS_COLUMNS)."""
    import numpy as np
    Q = K.CFG().calibrate_template_k
    st_col, ok_val, emaj, emin = Q.TRIALS_COLUMNS[tier]
    rows = [r for p in sorted(trials_files) for r in K.read_csv(p)]
    rec = {'tier': tier, 'orientation': orientation, 'percentile': Q.K_PERCENTILE, 'snr_key': Q.SNR_KEY,
           'columns': list(Q.TRIALS_COLUMNS[tier]), 'trials': {p: K.digests(p)['sha1'] for p in sorted(trials_files)},
           'n_injections': len(rows), 'K': {}}
    for S in ('I', 'V'):
        b = [r for r in rows if r.get('stokes') == S]
        if not b:
            continue
        ok = [r for r in b if r.get(st_col) == ok_val]
        st = {}
        for r in b:
            key = r.get(st_col) or r.get('pos_reason') or 'none'
            st[key] = st.get(key, 0) + 1
        km = k_percentile(ok, emaj, 'bmaj_deg', Q.SNR_KEY, Q.K_PERCENTILE)
        kn = k_percentile(ok, emin, 'bmin_deg', Q.SNR_KEY, Q.K_PERCENTILE)
        rng = np.random.default_rng(Q.BOOTSTRAP_SEED)
        bs = []
        for _ in range(Q.BOOTSTRAP_N):
            sel = [ok[i] for i in rng.integers(len(ok), size=len(ok))]
            bs.append((k_percentile(sel, emaj, 'bmaj_deg', Q.SNR_KEY, Q.K_PERCENTILE),
                       k_percentile(sel, emin, 'bmin_deg', Q.SNR_KEY, Q.K_PERCENTILE)))
        bs = np.array(bs)
        rec['K'][S] = {'major': km, 'minor': kn, 'n_injected': len(b), 'n_ok': len(ok), 'statuses': st,
                       'boot_major_16_84': [float(np.percentile(bs[:, 0], 16)), float(np.percentile(bs[:, 0], 84))],
                       'boot_minor_16_84': [float(np.percentile(bs[:, 1], 16)), float(np.percentile(bs[:, 1], 84))]}
    return rec


def rounded(rec, S):
    n = K.CFG().calibrate_template_k.REPORT_DECIMALS
    return round(rec['K'][S]['major'], n), round(rec['K'][S]['minor'], n)


def _trials_for(ctx):
    """(trials files, template orientation, source label): --trials; else this version's inject record; else (deep) the
    stored trials of tables.K_RECORDS['deep']."""
    Q = ctx.C.calibrate_template_k
    if ctx.args.trials:
        files = sorted(glob.glob(os.path.join(ctx.args.trials, 'trials*.csv'))) if os.path.isdir(ctx.args.trials) \
            else sorted(glob.glob(ctx.args.trials))
        same = ctx.tier == 'deep' and _same(ctx.args.trials, ctx.C.tables.K_RECORDS.get('deep'))
        orient = Q.TRIALS_ORIENTATION.get(ctx.tier) if same else Q.INJECT_ORIENTATION.get(ctx.tier)
        return files, orient, ctx.args.trials
    rec = K.load_record(ctx.out, ctx.unit_name('derive-inject'))
    if rec and rec.get('status') == 'done':
        return K.outputs_of(rec, 'template_k_trials'), Q.INJECT_ORIENTATION.get(ctx.tier), 'inject record'
    if ctx.tier == 'deep':
        d = ctx.C.tables.K_RECORDS.get('deep')
        return sorted(glob.glob(os.path.join(d, 'trials_*.csv'))) if d else [], Q.TRIALS_ORIENTATION.get('deep'), d
    if ctx.tier == 'sublst':
        raise Refusal('template-k --tier sublst summary: no inject record of this version and no --trials (the 2026-10-13 '
                      'sub-LST trials are on lwacalim00 /fast, not on Lustre; spec B6)')
    raise Refusal('template-k --tier snapshot: no trials source (template_fit_validation run_20261004b not found; spec B6)')


def _same(a, b):
    return bool(a and b) and os.path.realpath(a) == os.path.realpath(b)


def inject(ctx):
    """deep: deep_forced_kcal_20261013/kinj_deep.py IN PLACE (the plan of the ratified runs copied in, one process per frame,
    --one i --per-frame DEEP_PER_FRAME); sub-LST: kcal_tenmin_inject.py IN PLACE on --catalog-dir."""
    Q = ctx.C.calibrate_template_k
    d = ctx.stage_dir('inject')
    if ctx.tier == 'deep':
        script = K.ops_script('kinj_deep')
        plan = Q.DEEP_PLAN
        if ctx.dry:
            n = len(K.read_csv(plan)) if os.path.exists(plan) else 0
            return {'script': script, 'plan': plan, 'frames': n, 'argv': [d, '--one', '<i>', '--per-frame', str(Q.DEEP_PER_FRAME)]}
        if ctx.skip_if_current('derive-inject'):
            return None
        K.need_file(plan, 'calibrate_template_k.DEEP_PLAN')
        K.fresh_stage_dir(d, ctx.args.fresh)
        with ctx.run.unit(ctx.unit_name('derive-inject'), cls=ctx.cls, tier=ctx.tier, version=ctx.version, stage='inject') as u:
            u.add_input(script)
            u.add_input(plan)
            shutil.copyfile(plan, os.path.join(d, 'plan.csv'))
            idx = [int(r['i']) for r in K.read_csv(plan)]

            def one(i):
                K.run_legacy(u, script, [d, '--one', str(i), '--per-frame', str(Q.DEEP_PER_FRAME)],
                             log=os.path.join(d, f'frame_{i}.log'))
                return os.path.join(d, f'trials_{i}.csv')
            outs = K.parallel([lambda i=i: one(i) for i in idx], ctx.workers(ctx.C.calibrate.DERIVE_PROCESSES))
            for o in outs:
                u.add_output(o, 'template_k_trials')
        return None
    if ctx.tier != 'sublst':
        raise Refusal(f'template-k --tier {ctx.tier}: no injection harness (spec B6)')
    cat = ctx.args.catalog_dir
    script = K.ops_script('kcal_tenmin')
    argv = [cat, d, '--n-i', str(Q.SUBLST_N_I), '--n-v', str(Q.SUBLST_N_V), '--per-frame', str(Q.SUBLST_PER_FRAME),
            '--nproc', str(ctx.workers(12)), '--seed', str(Q.SUBLST_SEED)]
    if ctx.dry:
        return {'script': script, 'argv': argv, 'catalog_dir': cat or '(missing: --catalog-dir is required)'}
    if not cat:
        raise Refusal('template-k --tier sublst inject needs --catalog-dir <the LST catalogue dir> (kcal_tenmin_inject.py argv 1)')
    K.need_file(cat, '--catalog-dir')
    if ctx.skip_if_current('derive-inject'):
        return None
    K.fresh_stage_dir(d, ctx.args.fresh)
    with ctx.run.unit(ctx.unit_name('derive-inject'), cls=ctx.cls, tier=ctx.tier, version=ctx.version, stage='inject') as u:
        u.add_input(script)
        K.run_legacy(u, script, argv, log=os.path.join(d, 'inject.log'))
        tp = os.path.join(d, Q.SUBLST_TRIALS_FILE)
        K.need_file(tp, 'kcal_tenmin_inject.py trials')
        u.add_output(tp, 'template_k_trials')
    return None


def summary(ctx):
    d = ctx.stage_dir('summary')
    if ctx.dry:
        try:
            files, orient, src = _trials_for(ctx)
        except Refusal as e:
            files, orient, src = [], None, f'(not yet: {e})'
        return {'trials_source': src, 'n_files': len(files), 'orientation': orient}
    if ctx.skip_if_current('derive-summary'):
        return None
    files, orient, src = _trials_for(ctx)
    if not files:
        raise Refusal(f'template-k summary: no trials files in {src}')
    K.fresh_stage_dir(d, ctx.args.fresh)
    with ctx.run.unit(ctx.unit_name('derive-summary'), cls=ctx.cls, tier=ctx.tier, version=ctx.version, stage='summary') as u:
        for p in files:
            u.add_input(p)
        rec = summarise(files, ctx.tier, orient)
        rec.update(version=ctx.version, trials_source=src)
        rp = K.write_json(os.path.join(d, 'k_summary.json'), rec)
        for S in rec['K']:
            km, kn = rounded(rec, S)
            u.note(f'K {ctx.tier} {S}: major {km} minor {kn} ({rec["K"][S]["n_ok"]} of {rec["K"][S]["n_injected"]} ok)')
        u.add_output(rp, 'template_k_record')
    return None


def candidate_path(ctx):
    if ctx.args.candidate:
        return K.need_file(ctx.args.candidate, '--candidate')
    rec = K.require_done(ctx.out, ctx.unit_name('derive-summary'), ctx.action)
    return K.outputs_of(rec, 'template_k_record')[0]


def diff(ctx):
    import csv
    d = ctx.stage_dir('diff')
    if ctx.dry:
        try:
            cp = candidate_path(ctx)
        except Refusal as e:
            cp = f'(not yet: {e})'
        return {'candidate': cp, 'installed': f'fitting.K_TEMPLATE[{ctx.tier}, I|V]'}
    if ctx.skip_if_current('diff'):
        return None
    cp = candidate_path(ctx)
    K.fresh_stage_dir(d, ctx.args.fresh)
    with ctx.run.unit(ctx.unit_name('diff'), cls=ctx.cls, tier=ctx.tier, version=ctx.version, stage='diff') as u:
        u.add_input(cp)
        rec = K.read_json(cp)
        cur = ctx.C.fitting.K_TEMPLATE
        rows = []
        for S in sorted(rec['K']):
            new = rounded(rec, S)
            old = cur.get((ctx.tier, S))
            rows.append({'tier': ctx.tier, 'stokes': S, 'k_major_installed': old[0] if old else '', 'k_minor_installed': old[1] if old else '',
                         'k_major_candidate': new[0], 'k_minor_candidate': new[1],
                         'factor_major': round(new[0] / old[0], 4) if old else '', 'factor_minor': round(new[1] / old[1], 4) if old else ''})
        rp = K.write_csv(os.path.join(d, 'k_diff.csv'), list(rows[0].keys()) if rows else ['tier'], rows)
        # template-fit rows of the served list: their statistical error scales by the factor (K changes no position)
        run_dir = ctx.args.served_run or ctx.C.calibrate.SERVED_RUN.get(ctx.tier)
        served = {'list': None, 'template_rows': None}
        for n in ctx.C.calibrate.DIFF_LIST_CSV:
            p = os.path.join(run_dir or '', n)
            if run_dir and os.path.exists(p):
                with open(p, newline='') as f:
                    rd = csv.DictReader(f)
                    if 'pos_kind' in (rd.fieldnames or []):
                        cnt = {}
                        for r in rd:
                            if r.get('pos_kind') == 'template':
                                cnt[r.get('stokes', '')] = cnt.get(r.get('stokes', ''), 0) + 1
                        served = {'list': p, 'template_rows': cnt}
                    else:
                        served = {'list': p, 'template_rows': None,
                                  'note': 'the list has no pos_kind column: template-fit rows not counted'}
                break
        summ = {'class': ctx.cls, 'tier': ctx.tier, 'version': ctx.version, 'candidate': cp, 'candidate_sha256': K.sha256(cp),
                'rows_csv': rp, 'k': rows, 'served': served}
        sp = K.write_json(os.path.join(d, 'diff_summary.json'), summ)
        u.meta['candidate_sha256'] = summ['candidate_sha256']
        u.add_output(sp, 'calibration_diff')
    return None


def install(ctx):
    C = ctx.C
    d = ctx.stage_dir('install')
    if ctx.dry:
        try:
            cp = candidate_path(ctx)
        except Refusal as e:
            cp = f'(not yet: {e})'
        return {'candidate': cp, 'patch': os.path.join(d, 'config_patch.diff'), 'config_key': 'fitting.K_TEMPLATE'}
    if ctx.skip_if_current('install'):
        return None
    rid = K.check_ruling(ctx.args)
    cp = candidate_path(ctx)
    cand_sha = K.sha256(cp)
    diff_rec = K.diff_gate(ctx.out, ctx.cls, ctx.tier, ctx.version, cand_sha)
    rec = K.read_json(cp)
    Q = C.calibrate_template_k
    lo, hi = Q.K_BOUNDS
    if set(rec.get('K', {})) != {'I', 'V'}:
        raise Refusal(f'template-k install: the record holds Stokes {sorted(rec.get("K", {}))}; both I and V are required (G16)')
    vals = {S: rounded(rec, S) for S in ('I', 'V')}
    bad = {S: v for S, v in vals.items() if not all(lo < x < hi for x in v)}
    if bad:
        raise Refusal(f'template-k install: K outside ({lo}, {hi}): {bad} (gate G16)')
    if rec.get('orientation') != C.fitting.TEMPLATE_BEAM_ORIENTATION:
        raise Refusal(f'template-k install: K measured with the {rec.get("orientation")!r} template, config fits with '
                      f'{C.fitting.TEMPLATE_BEAM_ORIENTATION!r} (gate G17; the orientation flip is one config commit with the '
                      f're-measured K of every tier - FOLD_IN D5 row)')
    K.fresh_stage_dir(d, ctx.args.fresh)
    with ctx.run.unit(ctx.unit_name('install'), cls=ctx.cls, tier=ctx.tier, version=ctx.version, stage='install') as u:
        u.add_input(cp)
        vroot = C.calibrate.VERSIONED_TABLES_ROOT if (C.calibrate.VERSIONED_TABLES_ROOT and C.paths.PRODUCTION_WRITES) else d
        versioned = os.path.join(vroot, f'k_record.{ctx.tier}.{ctx.version}.json')
        shutil.copyfile(cp, versioned)
        edits, old_arg, new_arg = [], [], []
        for S in ('I', 'V'):
            km, kn = vals[S]
            edits.append(('fitting', 'K_TEMPLATE', rf"(\(\s*'{ctx.tier}'\s*,\s*'{S}'\s*\)\s*:\s*\()([^)]*)(\))",
                          rf'\g<1>{km:.3f}, {kn:.3f}\g<3>'))
            o = C.fitting.K_TEMPLATE.get((ctx.tier, S))
            if o:
                old_arg.append(f'{S}={o[0]:.3f},{o[1]:.3f}')
                new_arg.append(f'{S}={km:.3f},{kn:.3f}')
        rep = {' '.join(old_arg): ' '.join(new_arg)} if old_arg else {}
        missing = [S for S in ('I', 'V') if (ctx.tier, S) not in C.fitting.K_TEMPLATE]
        if missing:
            raise Refusal(f'template-k install: fitting.K_TEMPLATE has no entry ({ctx.tier}, {missing}) to patch')
        patch, changed = K.config_patch(rep, entry_edits=edits)
        pp = K.write_text(os.path.join(d, 'config_patch.diff'), patch)
        desc = (f'calibrate template-k --tier {ctx.tier} install {ctx.version}: K I {vals["I"]}, V {vals["V"]} (record {cp}, sha256 '
                f'{cand_sha}; orientation {rec.get("orientation")}; trials {rec.get("trials_source")}); was '
                f'{ {S: C.fitting.K_TEMPLATE.get((ctx.tier, S)) for S in ("I", "V")} }; diff {diff_rec}; patch {pp}')
        lp = os.path.join(d, 'ledger_row_draft.csv')
        K.ledger_draft(lp, ctx.cls, ctx.tier, ctx.version, [versioned, pp], desc, rid, ctx.args.install_words)
        man = {'class': ctx.cls, 'tier': ctx.tier, 'version': ctx.version, 'family_rule': 'template-k',
               'candidate': {'path': cp, 'sha256': cand_sha}, 'versioned': {'path': versioned, 'sha256': K.sha256(versioned)},
               'k_new': {S: list(v) for S, v in vals.items()}, 'config_key': 'fitting.K_TEMPLATE', 'patch': pp,
               'patch_changes': list(dict.fromkeys(f'{s}.{k}' for s, k in changed)), 'ledger_draft': lp, 'diff_record': diff_rec,
               'ruling': {'id': rid, 'words': ctx.args.install_words},
               'code_literals': [f'{a} ({b})' for a, b in K.code_literals(old_arg)],
               'production_writes': bool(C.paths.PRODUCTION_WRITES)}
        mp = K.write_json(os.path.join(d, 'install_manifest.json'), man)
        u.add_output(mp, 'install_bundle')
    return None


def verify(ctx):
    """deep: recompute K from tables.K_RECORDS['deep'] and compare with fitting.K_TEMPLATE; every tier: bounds, both Stokes,
    the ruling of the key. A candidate record (--candidate) is checked by its content tests."""
    if ctx.args.candidate:
        return K.verify(ctx, [(ctx.args.candidate, 'candidate', 'template_k_record')])
    C = ctx.C
    Q = C.calibrate_template_k
    plan = {'tier': ctx.tier, 'config': {S: C.fitting.K_TEMPLATE.get((ctx.tier, S)) for S in ('I', 'V')}
            if ctx.tier != 'snapshot' else {'K_SNAPSHOT': C.fitting.K_SNAPSHOT},
            'trials': C.tables.K_RECORDS.get(ctx.tier)}
    if ctx.dry:
        return plan
    d = os.path.join(ctx.vdir, 'verify-installed')
    if os.path.exists(d):
        sup = os.path.join(ctx.vdir, '_superseded')
        os.makedirs(sup, exist_ok=True)
        shutil.move(d, os.path.join(sup, f'verify-installed.{K.utcstamp()}'))
    os.makedirs(d)
    with ctx.run.unit(ctx.unit_name('verify-installed'), cls=ctx.cls, tier=ctx.tier, version=ctx.version, stage='verify') as u:
        lo, hi = Q.K_BOUNDS
        rep = {'tier': ctx.tier, 'config': plan['config'], 'checks': []}
        if ctx.tier == 'snapshot':
            ks = (C.fitting.K_SNAPSHOT[0], C.fitting.K_SNAPSHOT[1], C.fitting.K_SNAPSHOT_MAJOR_HIGH)
            u.check('template_k.config_bounds', all(lo < x < hi for x in ks), value=ks, detail='fitting.K_SNAPSHOT, K_SNAPSHOT_MAJOR_HIGH')
            u.note('snapshot K not recomputed: template_fit_validation run_20261004b trials not found (spec B6)')
        else:
            vals = {S: C.fitting.K_TEMPLATE.get((ctx.tier, S)) for S in ('I', 'V')}
            u.check('template_k.config_both_stokes', all(vals.values()), value=vals, detail='fitting.K_TEMPLATE has I and V')
            u.check('template_k.config_bounds', all(v and all(lo < x < hi for x in v) for v in vals.values()), value=vals,
                    detail=f'0 < K < 5 (G16)')
            u.check('template_k.config_ruled', bool(C.ruled('fitting.K_TEMPLATE')), value=C.ruled('fitting.K_TEMPLATE'),
                    detail='fitting.RULED names the ruling of K_TEMPLATE')
            tdir = C.tables.K_RECORDS.get(ctx.tier)
            files = sorted(glob.glob(os.path.join(tdir, 'trials_*.csv'))) if tdir else []
            if files:
                for p in files:
                    u.add_input(p)
                rec = summarise(files, ctx.tier, Q.TRIALS_ORIENTATION.get(ctx.tier))
                rp = K.write_json(os.path.join(d, 'k_recomputed.json'), rec)
                for S in ('I', 'V'):
                    got = rounded(rec, S) if S in rec['K'] else None
                    u.check(f'template_k.recompute_{S}', got is not None and tuple(got) == tuple(vals[S]), value=got,
                            detail=f'K recomputed from {len(files)} trials files vs fitting.K_TEMPLATE[{ctx.tier},{S}] {vals[S]}')
                u.add_output(rp, 'template_k_record')
            else:
                u.note(f'{ctx.tier} K not recomputed: trials not on Lustre (tables.K_RECORDS[{ctx.tier!r}] = {tdir!r}; spec B6)')
        rp2 = K.write_json(os.path.join(d, 'verify_report.json'), dict(rep, checks=u.tests))
        u.add_output(rp2, 'calibration_verify')
    return None
