#!/usr/bin/env python3
"""Tests of `exopipe3 forced --tier snapshot --stages tier_site,tier_runs` (family T: the HWJ / 10 pc forced-photometry pages
of the 10-minute and deep tiers, tier_forced_page_20261027.py in place; plain asserts, runnable as
`python tests/test_forced_tier_pages.py`, under 2 minutes).

Synthetic data only, in a fresh $TMPDIR/exopipe3_tests_forced_tier_pages/ (removed at the end; /tmp when TMPDIR is unset).
READ-ONLY checks of the production script (config truth: every literal the config mirrors equals the file's) and of the four
evaluations (the dry-run plan counts their events). The in-place script itself never runs here: every CLI case below refuses
before the first in-place call (the parity build is the end-to-end test).
The FALSIFIERS: each content test passes on a synthetic product and FAILS on a broken copy of it.

Run with: NUMPY_MADVISE_HUGEPAGE=0 EXOPIPE3_SITE=/lustre/gh/main/exopipe_ops/exopipe3_site.py \
          /opt/devel/pipeline/envs/py38_orca_nkosogor/bin/python3 tests/test_forced_tier_pages.py
"""
import ast
import contextlib
import csv
import io
import json
import os
import shutil
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')

from exopipe.phase3.core import env as _env       # noqa: E402

_env.legacy()
from exopipe.phase3.config import CFG, ConfigRefusal   # noqa: E402
from exopipe.phase3.core import validate, record       # noqa: E402

TMP = os.path.join(os.environ.get('TMPDIR') or '/tmp', 'exopipe3_tests_forced_tier_pages')
OPS = '/lustre/gh/main/exopipe_ops'
SCRIPT = os.path.join(OPS, 'tier_forced_page_20261027.py')
PP_SCRIPT = os.path.join(OPS, 'planets_forced_page_20261026.py')
RESULTS = []
_LAST_OUT = ['']


def check(name, cond, detail=''):
    RESULTS.append((name, bool(cond), detail))
    print('%s  %s%s' % ('PASS' if cond else 'FAIL', name, ('  -- ' + detail) if detail and not cond else ''), flush=True)


def cli(argv):
    """exopipe3 in-process; returns (exit code, stderr text); stdout kept in _LAST_OUT."""
    from exopipe.phase3 import cli as C
    err, out = io.StringIO(), io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
        try:
            rc = C.main(argv)
        finally:
            CFG.reset()
    _LAST_OUT[0] = out.getvalue()
    return rc, err.getvalue()


def _assigns(path):
    out = {}
    for node in ast.parse(open(path).read(), path).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except Exception:
                pass
    return out


def _write_csv(p, rows):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _png(p, n=12000, fill=b'\0'):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n' + fill * n)


def _reason(out, unit):
    rec = record.load_record(out, 'forced', unit) or {}
    return rec.get('status'), rec.get('reason') or ''


# ---------------------------------------------------------------------------------------------------------------------
# 1. config: the section, its mirrors of the in-place literals, protected / ruled keys, G37
# ---------------------------------------------------------------------------------------------------------------------

def test_config():
    import importlib
    from exopipe.phase3 import config
    from exopipe.phase3.lib.forced import common as K
    t, f = CFG.forced_tier_pages, CFG.forced
    check('config section forced_tier_pages loads', 'forced_tier_pages' in CFG.sections())
    check('steps.forced CONFIG declares forced_tier_pages',
          'forced_tier_pages' in config.declared_sections(importlib.import_module('exopipe.phase3.steps.forced')))
    check('snapshot stages end with family T', f.STAGES['snapshot'][-2:] == ('tier_site', 'tier_runs'))
    check('family T never a default nor per hour', not {'tier_site', 'tier_runs'} & set(f.DEFAULT_STAGES['snapshot'] +
                                                                                       f.EPOCH_STAGES['snapshot']))
    check('INPLACE_SCRIPTS tier_pages is the dated driver', f.INPLACE_SCRIPTS['tier_pages'] == os.path.basename(SCRIPT))
    check('PYTHON_KEY tier_pages is a site key', f.PYTHON_KEY['tier_pages'] in CFG.site.PYTHON)
    a, pp = _assigns(SCRIPT), _assigns(PP_SCRIPT)
    src = open(SCRIPT).read()
    check('MIRROR FIT_WORK_PRODUCTION = TIER_FIT_WORK (:559)', a.get('TIER_FIT_WORK') == t.FIT_WORK_PRODUCTION,
          '%r' % a.get('TIER_FIT_WORK'))
    check('MIRROR SAMPLES / TIERS (:31, :33)', a.get('SAMPLES') == t.SAMPLES and a.get('TIERS') == t.TIERS)
    check('MIRROR SITE_ROOT = planets_forced_page SF', pp.get('SF') == t.SITE_ROOT, '%r' % pp.get('SF'))
    check('MIRROR the review root RUN_ROOT = paths.SNAPSHOT_REVIEW_ROOT', a.get('RUN_ROOT') == CFG.paths.SNAPSHOT_REVIEW_ROOT)
    check('MIRROR EVAL_DIRS = ev_dir() of the four (tier, kind)', "'eval_%s_%sgen5_hwj10pc' % (tier, 'wb_' if kind == 'wb' else '')"
          in src and t.EVAL_DIRS == {(ti, k): os.path.join(t.SITE_ROOT, 'eval_%s_%sgen5_hwj10pc' % (ti, 'wb_' if k == 'wb' else ''))
                                     for ti in t.TIERS for k in ('sb', 'wb')})
    check('MIRROR RUN_NAME (run_name :394-395)', "'Run_SNAPFORCED_%s%s_gen5' % (st, tier)" in src
          and t.RUN_NAME.format(S='V', tier='deep') == 'Run_SNAPFORCED_Vdeep_gen5')
    check('MIRROR NEW_SUFFIX (:945)', "D = D0 + '%s'" % t.NEW_SUFFIX in src)
    check('MIRROR PANEL_MIN_BYTES (cmd_verify :358)', 'os.path.getsize(png) > %d' % t.PANEL_MIN_BYTES in src)
    check('OK_LINES are the script\'s own lines', all(v.strip() in src for v in t.OK_LINES.values()))
    check('forced_tier_pages.EVAL_DIRS protected', CFG.protected('forced_tier_pages.EVAL_DIRS'))
    try:
        CFG.set('forced_tier_pages.EVAL_DIRS', "{}")
        ok = False
    except ConfigRefusal:
        ok = True
    CFG.reset()
    check('forced_tier_pages.EVAL_DIRS override refused', ok)
    check('ruled key forced_tier_pages.EXCLUDED_EVENTS names its ledger ID',
          CFG.ruled('forced_tier_pages.EXCLUDED_EVENTS') == ('R-GREGG-20261014-REFACTOR-DECISIONS-LIST-RECOMMENDATIONS',))
    par = open(os.path.join(REPO, 'exopipe', 'phase3', 'parity', 'production_20261013.cfg')).read()
    check('parity file: production has no exclusion gate here', "forced_tier_pages.EXCLUDED_EVENTS = 'build'" in par)
    for k, v, refused in (('forced_tier_pages.SAMPLES', "('HWJ',)", True), ('forced_tier_pages.FIT_WORK_PRODUCTION', "'/x'", True),
                          ('forced_tier_pages.NPROC[runs]', '4', False)):
        CFG.authorize([], out=TMP)
        CFG.set(k, v)
        try:
            K.check_overrides()
            got = False
        except record.Refusal:
            got = True
        CFG.reset()
        check('G37: override of %s %s' % (k, 'refused (an in-place literal)' if refused else 'allowed (steering)'), got == refused)


# ---------------------------------------------------------------------------------------------------------------------
# 2. families
# ---------------------------------------------------------------------------------------------------------------------

class A:
    def __init__(self, **kw):
        self.stages = kw.get('stages')
        self.stage_from = kw.get('stage_from')
        self.stage_until = kw.get('stage_until')
        self.epochs = kw.get('epochs')


def test_families():
    from exopipe.phase3.lib.forced import wide_stages as WS
    check('tier stages: family T, pipeline order', WS.family(A(stages='tier_runs,tier_site')) == ('T', ['tier_site', 'tier_runs']))
    for st in ('build,tier_site', 'tier_runs,activity', 'wide_jobs,tier_site'):
        try:
            WS.family(A(stages=st))
            ok = False
        except record.Refusal as e:
            ok = 'W-FAMILY' in str(e)
        check('W-FAMILY: %s refused (one family per invocation)' % st, ok)


# ---------------------------------------------------------------------------------------------------------------------
# 3. content tests and their falsifiers (synthetic site and run)
# ---------------------------------------------------------------------------------------------------------------------

def _site():
    """A synthetic site over two synthetic evaluations; returns (page_record path, evals ctx)."""
    evs = {}
    for kind in ('sb', 'wb'):
        ev = os.path.join(TMP, 'eval_deep_%s' % kind)
        th = [dict(sample=s, tier='deep', stokes=S, thr_survey_gauss='5.0', thr_survey='5.5', n_target_trials='100')
              for s in ('HWJ', '10pc') for S in ('I', 'V')]
        _write_csv(os.path.join(ev, 'thresholds.csv'), th)
        rows = [dict(sample='HWJ', target='T1', tier='deep', stokes='V', cell='04h_2025-01-0%d' % (1 if kind == 'sb' else 2),
                     band='41MHz' if kind == 'sb' else 'Red', ts='2025-01-01T04:00:00', screen='clean', snr='-6.1'),
                dict(sample='10pc', target='T2', tier='deep', stokes='I', cell='04h_2025-01-03', band='46MHz', ts='x',
                     screen='clean', snr='4.9'),            # below threshold: not an event
                dict(sample='10pc', target='T3', tier='deep', stokes='I', cell='04h_2025-01-03', band='46MHz', ts='y',
                     screen='zone', snr='9.0')]             # not clean: not an event
        _write_csv(os.path.join(ev, 'targets_ge4.csv'), rows)
        evs['deep|%s' % kind] = ev
    site = os.path.join(TMP, 'site')
    cards, thd = [], {}
    for k, ev in sorted(evs.items()):
        tier, kind = k.split('|')
        for r in csv.DictReader(open(os.path.join(ev, 'thresholds.csv'))):
            thd['|'.join((r['sample'], tier, kind, r['stokes']))] = r
        for i, r in enumerate(csv.DictReader(open(os.path.join(ev, 'targets_ge4.csv')))):
            if r['screen'] == 'clean' and abs(float(r['snr'])) >= 5.0:
                name = 'event_%04d_%s_deep_%s_%s' % (len(cards), r['sample'], kind, r['stokes'])
                cards.append(dict(name=name, label='EVENT', row=dict(r, kind=kind), lines=['x'], measured_V=None,
                                  measured_I=None))
    os.makedirs(os.path.join(site, 'yellow'))
    for c in cards:
        _png(os.path.join(site, 'panels', c['name'] + '.png'))
        open(os.path.join(site, 'yellow', c['name'] + '.npz'), 'wb').close()
    with open(os.path.join(site, 'index.html'), 'w') as f:
        f.write(''.join('<img src="panels/%s.png">' % c['name'] for c in cards))
    rec = dict(built_utc='2026-11-17 00:00 UT', thresholds=thd, counts={}, cards=cards, n_events=len(cards),
               inputs=dict(tiers='x', phot='y', evals=evs))
    with open(os.path.join(site, 'page_record.json'), 'w') as f:
        json.dump(rec, f)
    return os.path.join(site, 'page_record.json'), evs


def _passes(cls, path, **ctx):
    res = validate.run(cls, path, **ctx)
    return all(r['passed'] for r in res), '; '.join(str(r['detail'])[:300] for r in res)


def test_site_content():
    rp, evs = _site()
    site = os.path.dirname(rp)
    ok, d = _passes('forced_tier_site', rp, evals=evs)
    check('forced_tier_site passes on the synthetic site (2 events of 6 rows)', ok, d)
    rec = json.load(open(rp))
    # falsifiers, each on a copy of the site
    def broken(tag, fn):
        b = os.path.join(TMP, 'site_' + tag)
        shutil.copytree(site, b)
        fn(b)
        ok, d = _passes('forced_tier_site', os.path.join(b, 'page_record.json'), evals=evs)
        check('FALSIFIER forced_tier_site refuses %s' % tag, not ok, d)

    def drop_card(b):
        r = dict(rec, cards=rec['cards'][1:], n_events=len(rec['cards']) - 1)
        json.dump(r, open(os.path.join(b, 'page_record.json'), 'w'))
    broken('a missing event', drop_card)
    broken('a broken panel', lambda b: _png(os.path.join(b, 'panels', rec['cards'][0]['name'] + '.png'), n=100))
    broken('a panel off the page', lambda b: open(os.path.join(b, 'index.html'), 'w').write('<html></html>'))

    def thr(b):
        r = json.loads(json.dumps(rec))
        k = sorted(r['thresholds'])[0]
        r['thresholds'][k]['n_target_trials'] = '99'
        json.dump(r, open(os.path.join(b, 'page_record.json'), 'w'))
    broken('thresholds that are not the evaluation files\'', thr)
    broken('a missing yellow-number file', lambda b: os.remove(os.path.join(b, 'yellow', rec['cards'][0]['name'] + '.npz')))
    ok, d = _passes('forced_tier_site', rp, evals={'deep|sb': evs['deep|sb']})
    check('FALSIFIER forced_tier_site refuses a site that read other evaluations than named', not ok, d)
    return rp


def _run(site_rp):
    """A synthetic deep Stokes V run of the site's cards; returns its queue path."""
    rec = json.load(open(site_rp))
    cards = [c for c in rec['cards'] if c['row']['tier'] == 'deep' and c['row']['stokes'] == 'V']
    D = os.path.join(TMP, 'review', 'Run_SNAPFORCED_Vdeep_gen5.new')
    fits, dets = [], {}
    for i, c in enumerate(cards):
        r = c['row']
        _png(os.path.join(D, 'cards', c['name'] + '.png'), fill=bytes([i + 1]))
        cid = 'c%d' % i
        fits.append(dict(cand_id=cid, fit_ra_deg='10.12345' if i == 0 else '', fit_dec_deg='20.5' if i == 0 else '',
                         astrometric='1' if i == 0 else '0', fit_kind='gaussian', fit_status='ok' if i == 0 else 'failed'))
        rd = os.path.join(D, 'candidates', 'V', '%s_%s_deep' % (r['target'], r['band']))
        dets.setdefault(rd, []).append(dict(card=c['name'], cand_id=cid, lst='04h', date=r['cell'][4:], band=r['band'],
                                            snr=r['snr'], sample=r['sample'], target=r['target'],
                                            p_gauss=r.get('p_gauss'), expected_fa=r.get('expected_fa'),
                                            ra='10.12345' if i == 0 else '', dec='20.50000' if i == 0 else ''))
        os.makedirs(rd, exist_ok=True)
        shutil.copyfile(os.path.join(D, 'cards', c['name'] + '.png'),
                        os.path.join(rd, 'rank0001_04h_%s_%s.png' % (r['cell'][4:], r['band'])))
    for rd, rows in dets.items():
        for d in rows:
            d.update(p_gauss=d['p_gauss'] or '', expected_fa=d['expected_fa'] or '')   # csv writes None as ''
        _write_csv(os.path.join(rd, 'detections.csv'), rows)
    _write_csv(os.path.join(D, 'forced_fits.csv'), fits)
    _write_csv(os.path.join(D, 'source_catalogs_V', '41MHz_sources_survivors.csv'), [dict(identity_id='T1')])
    _write_csv(os.path.join(D, 'source_catalogs', 'localizations.csv'), [dict(identity_id='T1', n_det='2')])
    _png(os.path.join(D, 'localization_figs', 'T1.png'))
    _write_csv(os.path.join(D, 'forced_assoc_identity.csv'), [dict(identity_id='T1', label='target')])
    with open(os.path.join(D, 'queue_V.txt'), 'w') as f:
        f.write('# queue_name: forced photometry (HWJ + 10 pc), Stokes V, deep tier\n# run: x\n# selection: y\nT1\n')
    return os.path.join(D, 'queue_V.txt'), len(cards)


def test_run_content(site_rp):
    # the card rows carry no p_gauss / expected_fa in the synthetic site: the run copies them as '' (equal to the card's None
    # only after this fix-up), so give the cards the values the run serves
    rec = json.load(open(site_rp))
    for c in rec['cards']:
        c['row'].update(p_gauss='0.01', expected_fa='0.1')
    json.dump(rec, open(site_rp, 'w'))
    q, n = _run(site_rp)
    D = os.path.dirname(q)
    site = os.path.dirname(site_rp)
    ok, d = _passes('forced_tier_run', q, site=site, tier='deep', stokes='V')
    check('forced_tier_run passes on the synthetic run (%d served events)' % n, ok and n == 2, d)
    ok, d = _passes('forced_tier_run', q, site=site, tier='10min', stokes='V')
    check('FALSIFIER forced_tier_run refuses a queue of another tier than recorded', not ok, d)

    def broken(tag, fn):
        b = D.replace('.new', '.broken_' + tag.replace(' ', '_')[:20])
        shutil.copytree(D, b)
        fn(b)
        ok, d = _passes('forced_tier_run', os.path.join(b, 'queue_V.txt'), site=site, tier='deep', stokes='V')
        check('FALSIFIER forced_tier_run refuses %s' % tag, not ok, d)

    def first_panel(b):
        import glob
        return sorted(glob.glob(os.path.join(b, 'candidates', 'V', '*', '*.png')))[0]

    broken('a served panel that is not its card', lambda b: _png(first_panel(b), fill=b'\7'))

    def pos_without_fit(b):
        import glob
        for x in sorted(glob.glob(os.path.join(b, 'candidates', 'V', '*', 'detections.csv'))):
            rr = list(csv.DictReader(open(x)))
            for r in rr:
                if not r['ra']:
                    r.update(ra='11.00000', dec='21.00000')
            _write_csv(x, rr)
    broken('a position served without a radio position', pos_without_fit)

    def unserved(b):
        import glob
        p = sorted(glob.glob(os.path.join(b, 'candidates', 'V', '*', 'detections.csv')))[0]
        shutil.rmtree(os.path.dirname(p))
    broken('a card not served', unserved)
    broken('a multi-detection identity without its figure', lambda b: os.remove(os.path.join(b, 'localization_figs', 'T1.png')))
    broken('a queued identity not catalogued', lambda b: open(os.path.join(b, 'queue_V.txt'), 'a').write('T9\n'))
    broken('a queued identity without its association row',
           lambda b: _write_csv(os.path.join(b, 'forced_assoc_identity.csv'), [dict(identity_id='T0', label='x')]))


# ---------------------------------------------------------------------------------------------------------------------
# 4. CLI: dry run, flags and gates (every case refuses before the first in-place call)
# ---------------------------------------------------------------------------------------------------------------------

def test_cli():
    from exopipe.phase3.lib.forced import common as K
    from exopipe.phase3.lib.forced import tier_snapshot as TS
    out = os.path.join(TMP, 'o_dry')
    base = ['forced', '--tier', 'snapshot', '--cores', 'none']
    rc, err = cli(base + ['--stages', 'tier_site,tier_runs', '--dry-run', '--out', out])
    o = _LAST_OUT[0]
    check('cli: dry run of family T exit 0 with the plan (events counted, gates planned)',
          rc == 0 and '[tier_site]' in o and 'events selected by the evaluations' in o and 'gate T-EXCL' in o
          and '[tier_runs]' in o and 'TIER_FIT_WORK=' in o, err[-300:] + o[-600:])
    check('cli: the dry run writes nothing', not os.path.exists(out))
    for argv, what, word in ((['--stages', 'build,tier_site'], 'families C and T mixed', 'W-FAMILY'),
                             (['--stages', 'tier_site', '--epochs', '04h_2025-10-14'], 'family T per hour', '--epochs'),
                             (['--stages', 'tier_site', '--wide-only', 'I:X'], 'a family-W flag with T', '--wide-only'),
                             (['--stages', 'tier_site', '--stokes', 'V'], '--stokes with T', '--stokes'),
                             (['--stages', 'build', '--tier-site', '/x'], '--tier-site with family C', '--tier-site'),
                             (['--stages', 'activity', '--tier-site', '/x'], '--tier-site with family N', '--tier-site')):
        rc, err = cli(base + argv + ['--dry-run', '--out', os.path.join(TMP, 'o_flags')])
        check('cli: %s refused (exit 2)' % what, rc == 2 and word in err, err[-300:])
    # T-G35: the site directory exists
    out = os.path.join(TMP, 'o_g35')
    os.makedirs(os.path.join(out, CFG.forced_tier_pages.SITE_NAME))
    rc, err = cli(base + ['--stages', 'tier_site', '--out', out])
    st, why = _reason(out, 'T_site')
    check('cli: T-G35 an existing site refused (exit 2) before any in-place call',
          rc == 2 and st == 'refused' and 'T-G35' in why and not os.path.exists(os.path.join(out, '_records', 'forced', 'logs',
                                                                                             'T_site.log')), '%s %s' % (st, why))
    # T-EXCL: an event on an excluded cell-band (the exclusion read replaced in this process)
    out = os.path.join(TMP, 'o_excl')
    keep = TS.excluded_events
    TS.excluded_events = lambda rows, ex=None: [('04h_2025-01-01', '41MHz', 'x')]
    try:
        rc, err = cli(base + ['--stages', 'tier_site', '--out', out])
    finally:
        TS.excluded_events = keep
    st, why = _reason(out, 'T_site')
    check('cli: T-EXCL an event on an excluded cell-band refused (exit 2) before any in-place call',
          rc == 2 and 'T-EXCL' in why and not os.path.exists(os.path.join(out, '_records', 'forced', 'logs', 'T_site.log')),
          '%s %s' % (st, why))
    # T-HOST: tier_runs off the review host
    out = os.path.join(TMP, 'o_host')
    keep = K.hostname
    K.hostname = lambda: 'lwacalim09'
    try:
        rc, err = cli(base + ['--stages', 'tier_runs', '--out', out])
    finally:
        K.hostname = keep
    st, why = _reason(out, 'T_runs')
    check('cli: T-HOST tier_runs off the review host refused (exit 2)', rc == 2 and 'T-HOST' in why, '%s %s' % (st, why))
    # T-SITE: no site (and no tier_site record) - only where this host is the review host
    if K.host_of(CFG.paths.COADD_SB_ROOT) in (None, K.hostname()):
        out = os.path.join(TMP, 'o_site')
        rc, err = cli(base + ['--stages', 'tier_runs', '--out', out])
        st, why = _reason(out, 'T_runs')
        check('cli: T-SITE tier_runs without a site refused (exit 2)', rc == 2 and 'T-SITE' in why, '%s %s' % (st, why))
        # a site that fails its content test (the synthetic one, read against the configured evaluations)
        out = os.path.join(TMP, 'o_site2')
        rc, err = cli(base + ['--stages', 'tier_runs', '--out', out, '--tier-site', os.path.join(TMP, 'site')])
        st, why = _reason(out, 'T_runs')
        check('cli: T-SITE a site failing its content test refused (exit 2)', rc == 2 and 'T-SITE' in why and 'content test'
              in why, '%s %s' % (st, why))
    else:
        print('SKIP  T-SITE cases: this host is not the review host', flush=True)


def test_import_side_effects():
    sys.path.insert(0, os.path.join(REPO, 'tests'))
    import test_import_side_effects as T
    mine = [b for b in T.ast_violations() if 'tier_pages' in b[0]]
    check('no module-level side effects in the family T files (AST layer of the standing test)', not mine, str(mine))


def main():
    t0 = time.time()
    shutil.rmtree(TMP, ignore_errors=True)
    os.makedirs(TMP)
    try:
        test_config()
        test_families()
        rp = test_site_content()
        test_run_content(rp)
        test_cli()
        test_import_side_effects()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    bad = [r for r in RESULTS if not r[1]]
    print('\n%d checks, %d failed, %.0f s' % (len(RESULTS), len(bad), time.time() - t0))
    for n, _, d in bad:
        print('FAILED: %s  %s' % (n, d))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
