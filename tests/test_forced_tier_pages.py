#!/usr/bin/env python3
"""Tests of `exopipe3 forced --tier snapshot --stages tier_eval,tier_site,tier_runs` (family T: the HWJ / 10 pc / LPT
forced-photometry pages of the 10-minute and deep tiers, snapshot_forced_eval.py and tier_forced_page_20261027.py in place;
plain asserts, runnable as `python tests/test_forced_tier_pages.py`, under 2 minutes).

Synthetic data only, in a fresh $TMPDIR/exopipe3_tests_forced_tier_pages/ (removed at the end; /tmp when TMPDIR is unset).
READ-ONLY checks of the production script (config truth: every literal the config mirrors equals the file's) and of the four
evaluations (the dry-run plan counts their events). The in-place scripts themselves never run here: every CLI case below refuses
before the first in-place call, or replaces the call in this process (the parity build and the real-data content test of
2026-11-24 are the end-to-end tests).
The FALSIFIERS: each content test passes on a synthetic product and FAILS on a broken copy of it.

Run with: NUMPY_MADVISE_HUGEPAGE=0 EXOPIPE3_SITE=/lustre/gh/main/exopipe_ops/exopipe3_site.py \
          /opt/devel/pipeline/envs/py38_orca_nkosogor/bin/python3 tests/test_forced_tier_pages.py
"""
import ast
import contextlib
import csv
import glob
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
EVAL_SCRIPT = os.path.join(OPS, 'snapshot_forced_eval.py')
SAMPLES = ('HWJ', '10pc', 'LPT')            # the synthetic site's and evaluations' samples (forced_tier_pages.SAMPLES)
PROD = ['--set', "forced_tier_pages.SAMPLES=('HWJ', '10pc')"]   # production's page (the parity file's value)
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
    from exopipe.phase3.lib.forced import tier_pages as TP
    t, f = CFG.forced_tier_pages, CFG.forced
    check('config section forced_tier_pages loads', 'forced_tier_pages' in CFG.sections())
    check('steps.forced CONFIG declares forced_tier_pages',
          'forced_tier_pages' in config.declared_sections(importlib.import_module('exopipe.phase3.steps.forced')))
    check('snapshot stages end with family T', f.STAGES['snapshot'][-3:] == ('tier_eval', 'tier_site', 'tier_runs'))
    check('family T never a default nor per hour', not {'tier_eval', 'tier_site', 'tier_runs'} & set(
        f.DEFAULT_STAGES['snapshot'] + f.EPOCH_STAGES['snapshot']))
    check('INPLACE_SCRIPTS tier_pages is the dated driver', f.INPLACE_SCRIPTS['tier_pages'] == os.path.basename(SCRIPT))
    check('INPLACE_SCRIPTS tier_eval is the stage-2 evaluation', f.INPLACE_SCRIPTS['tier_eval'] == os.path.basename(EVAL_SCRIPT)
          and os.path.exists(EVAL_SCRIPT))
    check('PYTHON_KEY tier_pages / tier_eval are site keys', f.PYTHON_KEY['tier_pages'] in CFG.site.PYTHON
          and f.PYTHON_KEY['tier_eval'] in CFG.site.PYTHON)
    a, pp = _assigns(SCRIPT), _assigns(PP_SCRIPT)
    src = open(SCRIPT).read()
    check('MIRROR FIT_WORK_PRODUCTION = TIER_FIT_WORK (:559)', a.get('TIER_FIT_WORK') == t.FIT_WORK_PRODUCTION,
          '%r' % a.get('TIER_FIT_WORK'))
    check('MIRROR SCRIPT_SAMPLES / SCRIPT_SAMPLE_LABEL / TIERS (:31-33)', a.get('SAMPLES') == t.SCRIPT_SAMPLES
          and a.get('SAMPLE_LABEL') == t.SCRIPT_SAMPLE_LABEL and a.get('TIERS') == t.TIERS)
    check('SAMPLES: the script\'s samples and LPT, each with a label and a pinned list (photometry.SAMPLE_FILES / SAMPLE_SHA1)',
          set(t.SCRIPT_SAMPLES) <= set(t.SAMPLES) and 'LPT' in t.SAMPLES
          and all(s_ in t.SAMPLE_LABEL and s_ in CFG.photometry.SAMPLE_FILES and s_ in CFG.photometry.SAMPLE_SHA1
                  for s_ in t.SAMPLES)
          and all(t.SAMPLE_LABEL[s_] == t.SCRIPT_SAMPLE_LABEL[s_] for s_ in t.SCRIPT_SAMPLES), t.SAMPLES)
    check('MIRROR SCRIPT_EVAL_NAME / EVAL_ROOT_GLOBAL / SCRIPT_SF: ev_dir joins the module literal SF (:30, :37-38); '
          'EVAL_NAME of the script\'s samples is production\'s name (review F7)',
          "%s = PP.SF" % t.EVAL_ROOT_GLOBAL in src and "os.path.join(SF, 'eval_%s_%sgen5_hwj10pc'" in src
          and all(os.path.basename(v) == t.SCRIPT_EVAL_NAME.format(tier=ti, wb='wb_' if k == 'wb' else '')
                  == t.EVAL_NAME.format(tier=ti, wb='wb_' if k == 'wb' else '', samples=TP.eval_tag(t.SCRIPT_SAMPLES))
                  and os.path.dirname(v) == t.SCRIPT_SF for (ti, k), v in t.EVAL_DIRS.items())
          and TP.eval_tag(t.SAMPLES) == 'hwj10pclpt', TP.eval_tag(t.SAMPLES))
    ppsrc = open(PP_SCRIPT).read()
    check('MIRROR FT_PRODUCTION / PHOT_PRODUCTION = planets_forced_page FT and P3_PHOT_ROOT defaults (:54, :61)',
          "FT = os.environ.get('P3_FORCED_TIERS', '%s')" % t.FT_PRODUCTION in ppsrc
          and "os.environ.setdefault('P3_PHOT_ROOT', '%s')" % t.PHOT_PRODUCTION in ppsrc)
    esrc = open(EVAL_SCRIPT).read()
    import re
    m = re.match(t.EVAL_OK_RE, 'cells 24; thresholds 12 rows; targets >= 4: 7; drops {\'x\': 1} -> /a/b')
    check('EVAL_OK_RE reads the evaluation\'s last line (its format, main)', m is not None and m.group(1) == '/a/b'
          and "print('cells %d; thresholds %d rows; targets >= 4: %d; drops %s -> %s'" in esrc)
    check('tier_eval passes the evaluation\'s own flags (--tier, --survey-cells, --samples, --kind, --nproc) and P3_PHOT_ROOT',
          all("ap.add_argument('%s'" % x in esrc for x in ('--tier', '--survey-cells', '--samples', '--kind', '--nproc'))
          and "PHOT = os.environ.get('P3_PHOT_ROOT'" in esrc)
    check('MIRROR SITE_ROOT, SCRIPT_SF = planets_forced_page SF', pp.get('SF') == t.SITE_ROOT == t.SCRIPT_SF,
          '%r' % pp.get('SF'))
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
    check('ruled key forced_tier_pages.SAMPLES names the LPT rulings', set(CFG.ruled('forced_tier_pages.SAMPLES')) ==
          {'R-GREGG-20261118-GEN6-WITH-LPTS-SAMPLE-WIRED', 'R-GREGG-20261124-SIX-ANSWERS-RAMPUP-STATUS'})
    check('parity file: production\'s page shows HWJ / 10 pc', "forced_tier_pages.SAMPLES = ('HWJ', '10pc')" in par)
    for k, v, refused in (('forced_tier_pages.SCRIPT_SAMPLES', "('HWJ',)", True),
                          ('forced_tier_pages.FIT_WORK_PRODUCTION', "'/x'", True),
                          ('forced_tier_pages.EVAL_NAME', "'x'", True),
                          ('forced_tier_pages.SAMPLES', "('HWJ', '10pc')", False),
                          ('forced_tier_pages.NPROC[runs]', '4', False)):
        CFG.authorize([], out=TMP)
        CFG.set(k, v)
        try:
            K.check_overrides()
            got = False
        except record.Refusal:
            got = True
        CFG.reset()
        check('G37: override of %s %s' % (k, 'refused (an in-place literal)' if refused else 'allowed (steering: passed to the '
                                                                                         'in-place scripts)'), got == refused)


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
    check('tier stages: family T, pipeline order', WS.family(A(stages='tier_runs,tier_site')) == ('T', ['tier_site', 'tier_runs'])
          and WS.family(A(stages='tier_site,tier_eval')) == ('T', ['tier_eval', 'tier_site']))
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

def _site(samples=SAMPLES, tag=''):
    """A synthetic site over two synthetic evaluations of ``samples`` (directories named with ``tag``); returns (page_record
    path, evals ctx)."""
    evs = {}
    for kind in ('sb', 'wb'):
        ev = os.path.join(TMP, 'eval_deep_%s%s' % (tag, kind))
        th = [dict(sample=s, tier='deep', stokes=S, thr_survey_gauss='5.0', thr_survey='5.5', n_target_trials='100')
              for s in samples for S in ('I', 'V')]
        _write_csv(os.path.join(ev, 'thresholds.csv'), th)
        rows = [dict(sample='HWJ', target='T1', tier='deep', stokes='V', cell='04h_2025-01-0%d' % (1 if kind == 'sb' else 2),
                     band='41MHz' if kind == 'sb' else 'Red', ts='2025-01-01T04:00:00', screen='clean', snr='-6.1'),
                dict(sample='10pc', target='T2', tier='deep', stokes='I', cell='04h_2025-01-03', band='46MHz', ts='x',
                     screen='clean', snr='4.9'),            # below threshold: not an event
                dict(sample='10pc', target='T3', tier='deep', stokes='I', cell='04h_2025-01-03', band='46MHz', ts='y',
                     screen='zone', snr='9.0')]             # not clean: not an event
        if 'LPT' in samples:
            rows.append(dict(sample='LPT', target='GPM_J0715-1141', tier='deep', stokes='I', cell='06h_2025-01-04',
                             band='55MHz' if kind == 'sb' else 'Green', ts='z', screen='clean', snr='5.2'))
        _write_csv(os.path.join(ev, 'targets_ge4.csv'), rows)
        evs['deep|%s' % kind] = ev
    site = os.path.join(TMP, 'site' + tag)
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
    for smp in samples:                    # the script's exceedance plot per (sample, tier) and section per (tier, Stokes, sample)
        _png(os.path.join(site, 'plots', 'exceedance_%s_deep.png' % smp))
    with open(os.path.join(site, 'index.html'), 'w') as f:
        f.write(''.join('<img src="panels/%s.png">' % c['name'] for c in cards)
                + ''.join('<p><img src="plots/exceedance_%s_deep.png"></p>' % smp for smp in samples)
                + ''.join('<h3 id="deep-%s-%s">x</h3>' % (S, smp) for S in ('V', 'I') for smp in samples))
    ft, phot = os.path.join(TMP, 'ft'), os.path.join(TMP, 'cut')
    for d in (ft, phot):
        os.makedirs(d, exist_ok=True)
    rec = dict(built_utc='2026-11-17 00:00 UT', thresholds=thd, counts={}, cards=cards, n_events=len(cards),
               inputs=dict(tiers=ft, phot=phot, evals=evs))
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
    check('forced_tier_site passes on the synthetic site (4 events of 8 rows, LPT among them)', ok and
          sum(c['row']['sample'] == 'LPT' for c in json.load(open(rp))['cards']) == 2, d)
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

    def via_links(b, to=None):
        # the script records its literal ev_dir paths: links (tier_pages.TCtx.eval_links) to the evaluations, or elsewhere
        ln = os.path.join(b, 'links')
        os.makedirs(ln)
        r = json.loads(json.dumps(rec))
        for k, v in sorted(evs.items()):
            os.symlink(to or v, os.path.join(ln, k.replace('|', '_')))
            r['inputs']['evals'][k] = os.path.join(ln, k.replace('|', '_'))
        json.dump(r, open(os.path.join(b, 'page_record.json'), 'w'))
    b = os.path.join(TMP, 'site_links')
    shutil.copytree(site, b)
    via_links(b)
    ok, d = _passes('forced_tier_site', os.path.join(b, 'page_record.json'), evals=evs)
    check('forced_tier_site passes a site that read its evaluations through links to them (review F7)', ok, d)
    broken('evaluations read through links to another directory', lambda b: via_links(b, to=TMP))
    broken('a broken panel', lambda b: _png(os.path.join(b, 'panels', rec['cards'][0]['name'] + '.png'), n=100))
    broken('a panel off the page', lambda b: open(os.path.join(b, 'index.html'), 'w').write('<html></html>'))

    def thr(b):
        r = json.loads(json.dumps(rec))
        k = sorted(r['thresholds'])[0]
        r['thresholds'][k]['n_target_trials'] = '99'
        json.dump(r, open(os.path.join(b, 'page_record.json'), 'w'))
    broken('thresholds that are not the evaluation files\'', thr)
    broken('a missing yellow-number file', lambda b: os.remove(os.path.join(b, 'yellow', rec['cards'][0]['name'] + '.npz')))
    broken('a sample without its exceedance plot', lambda b: os.remove(os.path.join(b, 'plots', 'exceedance_LPT_deep.png')))

    def nosec(b):
        p = os.path.join(b, 'index.html')
        open(p, 'w').write(open(p).read().replace('id="deep-V-LPT"', 'id="x"'))
    broken('a sample without its page section', nosec)
    # a site built by production's literal (HWJ / 10 pc) over evaluations without LPT: refused under SAMPLES with LPT, and
    # passes when the stage names production's samples
    rp2, evs2 = _site(samples=('HWJ', '10pc'), tag='prod_')
    ok, d = _passes('forced_tier_site', rp2, evals=evs2)
    check('FALSIFIER forced_tier_site refuses a HWJ / 10 pc site when SAMPLES has LPT', not ok and 'LPT' in d, d)
    ok, d = _passes('forced_tier_site', rp2, evals=evs2, samples=['HWJ', '10pc'])
    check('forced_tier_site passes the HWJ / 10 pc site when the stage names those samples', ok, d)
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


def _eval_out(out, inp, tier, kind, samples=SAMPLES):
    """What snapshot_forced_eval.py --tier ``tier`` --kind ``kind`` --samples ``samples`` writes in ``out`` for the tier tree
    ``inp`` (its main, :536-576): thresholds, exceedance, cells, targets_ge4, drops; returns its success line."""
    cells = sorted(os.path.basename(d) for d in glob.glob(os.path.join(inp, '[0-2][0-9]h')))
    _write_csv(os.path.join(out, 'thresholds.csv'), [dict(sample=s_, stokes=S, tier=tier, n_controls=50, n_target_trials=10,
                                                          thr_here=5.0, thr_here_gauss=4.5, thr_survey=5.5,
                                                          thr_survey_gauss=5.0) for s_ in samples for S in ('I', 'V')])
    _write_csv(os.path.join(out, 'exceedance.csv'), [dict(sample=s_, stokes=S, tier=tier, screen=scr, set=st, n=7, ge4='0')
                                                     for s_ in samples for S in ('I', 'V') for scr in ('clean', 'zone')
                                                     for st in ('targets', 'controls')])
    _write_csv(os.path.join(out, 'cells.csv'), [dict(cell=c_, kind=kind) for c_ in cells])
    _write_csv(os.path.join(out, 'targets_ge4.csv'), [dict(sample=samples[-1], target='T9', tier=tier, stokes='I',
                                                           cell=cells[0] + '_2025-01-01', band='55MHz', ts='t', screen='clean',
                                                           snr='4.2')])
    open(os.path.join(out, 'drops.csv'), 'w').write('reason,n_files_or_groups\n')
    return 'cells %d; thresholds %d rows; targets >= 4: 1; drops {} -> %s' % (len(cells), 2 * len(samples), out)


def _tier_tree(ft, samples=SAMPLES, lsts=('06h', '07h')):
    """A tier forced root with every sample measured in a subband and a wideband of each tier and LST (each file holds only
    its nights: the gates glob names, T_eval's admitted set reads 'night')."""
    import numpy as np
    for tier in ('10min', 'deep'):
        for lst in lsts:
            for s_ in samples:
                for b in ('55MHz', 'Green'):
                    p = os.path.join(ft, tier, lst, 'forced_%s_%s_I.npz' % (s_, b))
                    os.makedirs(os.path.dirname(p), exist_ok=True)
                    np.savez(p, night=np.array(['2025-01-01', '2025-01-02']))
    return ft


def test_eval_content():
    inp = os.path.join(_tier_tree(os.path.join(TMP, 'ft_eval')), 'deep')
    ev = os.path.join(TMP, 'ev_content')
    _eval_out(ev, inp, 'deep', 'sb')
    ctx = dict(tier='deep', kind='sb', samples=list(SAMPLES), inp=inp)
    ok, d = _passes('forced_tier_eval', os.path.join(ev, 'thresholds.csv'), **ctx)
    check('forced_tier_eval passes on a synthetic evaluation of HWJ / 10pc / LPT', ok, d)

    def broken(tag, fn, **kw):
        b = os.path.join(TMP, 'ev_' + tag.replace(' ', '_'))
        shutil.copytree(ev, b)
        fn(b)
        ok, d = _passes('forced_tier_eval', os.path.join(b, 'thresholds.csv'), **dict(ctx, **kw))
        check('FALSIFIER forced_tier_eval refuses %s' % tag, not ok, d)

    def drop(name, keep):
        def fn(b):
            p = os.path.join(b, name)
            rows = [r for r in csv.DictReader(open(p)) if keep(r)]
            _write_csv(p, rows)
        return fn
    broken('a sample without its Stokes V threshold', drop('thresholds.csv', lambda r: (r['sample'], r['stokes']) != ('LPT', 'V')))
    broken('a sample without clean controls', drop('exceedance.csv', lambda r: not (r['sample'] == 'LPT' and r['set'] ==
                                                                                     'controls' and r['screen'] == 'clean')))
    broken('cells other than the LSTs of its input', drop('cells.csv', lambda r: r['cell'] != '07h'))
    broken('an evaluation without targets_ge4.csv', lambda b: os.remove(os.path.join(b, 'targets_ge4.csv')))
    broken('an evaluation of other tiers\' rows', lambda b: None, tier='10min')
    ok, d = _passes('forced_tier_eval', os.path.join(ev, 'thresholds.csv'), **dict(ctx, samples=['HWJ', '10pc']))
    check('FALSIFIER forced_tier_eval refuses thresholds of a sample it was not asked for', not ok and 'LPT' in d, d)


# ---------------------------------------------------------------------------------------------------------------------
# 4. CLI: dry run, flags and gates (every case refuses before the first in-place call)
# ---------------------------------------------------------------------------------------------------------------------

def test_cli():
    from exopipe.phase3.lib.forced import common as K
    from exopipe.phase3.lib.forced import tier_pages as TP
    from exopipe.phase3.lib.forced import tier_snapshot as TS
    out = os.path.join(TMP, 'o_dry')
    base = ['forced', '--tier', 'snapshot', '--cores', 'none'] + PROD    # production's samples: its evaluations serve them
    rc, err = cli(base + ['--stages', 'tier_site,tier_runs', '--dry-run', '--out', out])
    o = _LAST_OUT[0]
    check('cli: dry run of family T (HWJ / 10 pc, production\'s evaluations) exit 0 with the plan (events counted, gates '
          'planned)',
          rc == 0 and '[tier_site]' in o and 'events selected by the evaluations' in o and 'gate T-EXCL' in o
          and '[tier_runs]' in o and 'TIER_FIT_WORK=' in o, err[-300:] + o[-600:])
    check('cli: the dry run writes nothing', not os.path.exists(out))
    for argv, what, word in ((['--stages', 'build,tier_site'], 'families C and T mixed', 'W-FAMILY'),
                             (['--stages', 'tier_site', '--epochs', '04h_2025-10-14'], 'family T per hour', '--epochs'),
                             (['--stages', 'tier_site', '--wide-only', 'I:X'], 'a family-W flag with T', '--wide-only'),
                             (['--stages', 'tier_site', '--stokes', 'V'], '--stokes with T', '--stokes'),
                             (['--stages', 'build', '--tier-site', '/x'], '--tier-site with family C', '--tier-site'),
                             (['--stages', 'activity', '--tier-site', '/x'], '--tier-site with family N', '--tier-site'),
                             (['--stages', 'build', '--tier-forced-root', '/x'], '--tier-forced-root with family C',
                              '--tier-forced-root'),
                             (['--stages', 'wide_jobs', '--tier-cutout-root', '/x'], '--tier-cutout-root with family W',
                              '--tier-cutout-root')):
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
    # a dry run whose plan names a refusing gate exits 2 (as family W), here T-G35 on the same --out
    rc, err = cli(base + ['--stages', 'tier_site', '--dry-run', '--out', out])
    o = _LAST_OUT[0]
    check('cli: a dry run whose plan names a refusing gate (T-G35) exits 2', rc == 2 and 'would refuse' in o and 'T-G35' in o,
          '%s %s' % (rc, o[-400:]))
    # T-WORKERS: --workers above forced_tier_pages.MAX_WORKERS (the in-place pool reads Lustre)
    out = os.path.join(TMP, 'o_workers')
    n = CFG.forced_tier_pages.MAX_WORKERS + 2
    rc, err = cli(base + ['--stages', 'tier_site', '--workers', str(n), '--out', out])
    st, why = _reason(out, 'T_site')
    check('cli: T-WORKERS --workers %d refused (exit 2) before any in-place call' % n,
          rc == 2 and 'T-WORKERS' in why and not os.path.exists(os.path.join(out, '_records', 'forced', 'logs', 'T_site.log')),
          '%s %s' % (st, why))
    rc, err = cli(base + ['--stages', 'tier_site', '--workers', str(n), '--dry-run', '--out', os.path.join(TMP, 'o_workers2')])
    check('cli: T-WORKERS named by the dry run (exit 2)', rc == 2 and 'T-WORKERS' in _LAST_OUT[0], _LAST_OUT[0][-300:])
    # T-BUILD: the build printed its success line but exited non-zero (the in-place call replaced in this process)
    out = os.path.join(TMP, 'o_rc')

    class _Res:
        rc = 1

        def has(self, s_):
            return True

        def line(self, s_):
            return s_

        def why(self):
            return 'exit 1'

        def lines(self):
            return []
    keep = TP.TCtx.inplace
    TP.TCtx.inplace = lambda self, u, argv, log, globals_=None, test_stage=False, stage=None, env=None: _Res()
    try:
        rc, err = cli(base + ['--stages', 'tier_site', '--out', out])
    finally:
        TP.TCtx.inplace = keep
    st, why = _reason(out, 'T_site')
    check('cli: T-BUILD a build exiting non-zero refused although it printed its success line', rc == 2 and 'T-BUILD' in why
          and 'exit 1' in why, '%s %s' % (st, why))
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
        rc, err = cli(base + ['--stages', 'tier_runs', '--dry-run', '--out', os.path.join(TMP, 'o_site3'), '--tier-site',
                              os.path.join(TMP, 'site')])
        o = _LAST_OUT[0]
        check('cli: T-SITE named by a tier_runs dry run (exit 2)', rc == 2 and 'gate T-SITE' in o and 'would refuse' in o,
              '%s %s' % (rc, o[-400:]))
        # T-EXCL in tier_runs: a site (here accepted as an input) with a card on an excluded cell-band - a site production
        # built has no exclusion gate, and an exclusion can postdate any site
        out = os.path.join(TMP, 'o_excl_runs')
        site_rp = os.path.join(TMP, 'site', 'page_record.json')
        keep_sc, keep_ex = TP._site_check, TS.excluded_events
        TP._site_check = lambda c: (site_rp, None, 'test: accepted')
        TS.excluded_events = lambda rows, ex=None: [(r['cell'], r['band'], r['ts']) for r in rows][:1]
        try:
            rc, err = cli(base + ['--stages', 'tier_runs', '--out', out, '--tier-site', os.path.join(TMP, 'site')])
        finally:
            TP._site_check, TS.excluded_events = keep_sc, keep_ex
        st, why = _reason(out, 'T_runs')
        check('cli: T-EXCL tier_runs refuses a site card on an excluded cell-band (exit 2) before any in-place call',
              rc == 2 and 'T-EXCL' in why and 'site card' in why
              and not os.path.exists(os.path.join(out, '_records', 'forced', 'logs', 'T_runs.log')), '%s %s' % (st, why))
        # T-WORK: the fit's work directory on Lustre (one plane FITS per event): refused, never a fallback
        out = os.path.join(TMP, 'o_work')
        rc, err = cli(base + ['--stages', 'tier_runs', '--out', out, '--work', '/lustre/gh/main/exopipe_ops/no_such_work'])
        st, why = _reason(out, 'T_runs')
        check('cli: T-WORK a fit work directory on Lustre refused (exit 2)', rc == 2 and 'T-WORK' in why, '%s %s' % (st, why))
        rc, err = cli(base + ['--stages', 'tier_runs', '--dry-run', '--out', os.path.join(TMP, 'o_work2'), '--work',
                              '/lustre/gh/main/exopipe_ops/no_such_work'])
        check('cli: T-WORK named by the dry run (exit 2)', rc == 2 and 'T-WORK' in _LAST_OUT[0], _LAST_OUT[0][-300:])
    else:
        print('SKIP  T-SITE, T-EXCL (tier_runs) and T-WORK cases: this host is not the review host', flush=True)


class _Call:
    """A core.legacy.run result as forced.common.call returns it (rc, lines, has, line, why)."""

    def __init__(self, rc, lines):
        self.rc, self._lines = rc, list(lines)

    def lines(self):
        return list(self._lines)

    def has(self, s_):
        return any(s_ in l for l in self._lines)

    def line(self, s_):
        return next((l for l in self._lines if s_ in l), None)

    def why(self):
        return 'exit %d; last lines %s' % (self.rc, self._lines[-2:])


def test_cli_lpt():
    """Family T with forced_tier_pages.SAMPLES = HWJ / 10pc / LPT (the default): the gates of an LPT page, and tier_eval then
    tier_site with the evaluation and the build replaced in this process (the real-data run of 2026-11-24 is the end-to-end
    test of the in-place scripts)."""
    from exopipe.phase3.lib.forced import common as K
    from exopipe.phase3.lib.forced import tier_pages as TP
    from exopipe.phase3.lib.forced import tier_snapshot as TS
    t = CFG.forced_tier_pages
    base = ['forced', '--tier', 'snapshot', '--cores', 'none']
    # T-EVAL-SAMPLES: evaluations of HWJ / 10 pc only (production's kind; here the synthetic ones) cannot make an LPT page
    evs = {('deep', k): os.path.join(TMP, 'eval_deep_prod_%s' % k) for k in ('sb', 'wb')}
    keep = TP.TCtx.eval_source
    TP.TCtx.eval_source = lambda self: (dict(evs), TMP, 'test: HWJ / 10 pc evaluations', None)
    try:
        rc, err = cli(base + ['--stages', 'tier_site', '--dry-run', '--out', os.path.join(TMP, 'o_lpt_es')])
    finally:
        TP.TCtx.eval_source = keep
    o = _LAST_OUT[0]
    check('cli: T-EVAL-SAMPLES evaluations without LPT thresholds named by the dry run (exit 2)',
          rc == 2 and 'gate T-EVAL-SAMPLES: 4 (tier, kind, sample, Stokes)' in o and "'LPT'" in o, o[-500:])
    # review F1: tier_runs reads a site production built (HWJ / 10 pc) under the default SAMPLES (with LPT): T-SITE tests it
    # against the samples its evaluations hold, not SAMPLES

    class _Run:
        resume, step, out_root = False, 'forced', os.path.join(TMP, 'o_f1')

    class _C:
        pass
    cx = _C()
    cx.run, cx.W, cx.t, cx.site = _Run(), _Run.out_root, t, os.path.join(TMP, 'siteprod_')
    cx.eval_source = lambda: (dict(evs), TMP, 'test: HWJ / 10 pc evaluations', None)
    got, why, how = TP._site_check(cx)
    check('cli: T-SITE passes a site of HWJ / 10 pc with its HWJ / 10 pc evaluations under SAMPLES %s (review F1)'
          % (t.SAMPLES,), 'LPT' in t.SAMPLES and got == os.path.join(cx.site, 'page_record.json') and why is None
          and TP._eval_samples(t, evs) == ['HWJ', '10pc'], '%s %s' % (why, how))
    prod_gaps = TP._eval_sample_gaps(t, t.EVAL_DIRS)
    rc, err = cli(base + ['--stages', 'tier_site', '--dry-run', '--out', os.path.join(TMP, 'o_lpt_es2')])
    o = _LAST_OUT[0]
    check('cli: tier_site on production\'s evaluations: T-EVAL-SAMPLES named exactly when they lack a sample (%d gaps)'
          % len(prod_gaps), (rc == 2 and 'gate T-EVAL-SAMPLES: %d' % len(prod_gaps) in o) if prod_gaps
          else 'every sample has its thresholds' in o, o[-500:])
    # T-ROOTS: other tier files than production's evaluations read, without tier_eval
    ft = _tier_tree(os.path.join(TMP, 'ft_full'))
    cut = os.path.join(TMP, 'cut_full')
    for s_ in SAMPLES:
        os.makedirs(os.path.join(cut, s_))
    rc, err = cli(base + PROD + ['--stages', 'tier_site', '--dry-run', '--out', os.path.join(TMP, 'o_lpt_roots'),
                                 '--tier-forced-root', ft])
    o = _LAST_OUT[0]
    check('cli: T-ROOTS production\'s evaluations with another --tier-forced-root named by the dry run (exit 2)',
          rc == 2 and 'T-ROOTS production' in o, o[-400:])
    # T-INPUT: no LPT measurement in the tier forced root
    part = _tier_tree(os.path.join(TMP, 'ft_part'), samples=('HWJ', '10pc'))
    rc, err = cli(base + ['--stages', 'tier_eval', '--dry-run', '--out', os.path.join(TMP, 'o_lpt_in'), '--tier-forced-root',
                          part, '--tier-cutout-root', cut])
    o = _LAST_OUT[0]
    check('cli: T-INPUT tier_eval without LPT measurements named by the dry run (exit 2)',
          rc == 2 and 'gate T-INPUT LPT' in o and 'forced_LPT_*.npz' in o, o[-400:])
    rc, err = cli(base + ['--stages', 'tier_eval', '--dry-run', '--out', os.path.join(TMP, 'o_lpt_ok'), '--tier-forced-root',
                          ft, '--tier-cutout-root', cut])
    o = _LAST_OUT[0]
    check('cli: tier_eval dry run on complete inputs exit 0; each evaluation of HWJ,10pc,LPT with P3_PHOT_ROOT = the cutouts',
          rc == 0 and o.count('--samples HWJ,10pc,LPT') == 4 and o.count('P3_PHOT_ROOT=%s' % cut) == 4
          and 'gates T-ROOTS, T-SAMPLES, T-G35, T-INPUT, T-WORKERS: pass' in o, o[-600:])
    # T-G35: an evaluation directory of this --out exists
    out = os.path.join(TMP, 'o_lpt_g35')
    os.makedirs(os.path.join(out, t.EVAL_NAME.format(tier='deep', wb='', samples=TP.eval_tag(SAMPLES))))
    calls = []
    keep = K.call
    K.call = lambda *a, **kw: calls.append(a) or _Call(1, [])
    try:
        rc, err = cli(base + ['--stages', 'tier_eval', '--out', out, '--tier-forced-root', ft, '--tier-cutout-root', cut])
    finally:
        K.call = keep
    st, why = _reason(out, 'T_eval')
    check('cli: T-G35 an existing evaluation directory refused (exit 2) before any evaluation', rc == 2 and 'T-G35' in why
          and not calls, '%s %s' % (st, why))
    # T-EVAL-RUN: an evaluation exiting 0 without its success line
    out = os.path.join(TMP, 'o_lpt_run')
    K.call = lambda u, target, args, *a, **kw: _Call(0, ['cells 2; thresholds 6 rows; targets >= 4: 0; drops {} -> /else'])
    try:
        rc, err = cli(base + ['--stages', 'tier_eval', '--out', out, '--tier-forced-root', ft, '--tier-cutout-root', cut])
    finally:
        K.call = keep
    st, why = _reason(out, 'T_eval')
    check('cli: T-EVAL-RUN a success line naming another directory refused (exit 2)', rc == 2 and 'T-EVAL-RUN' in why,
          '%s %s' % (st, why))
    # tier_eval -> tier_site: the four evaluations of every sample, then the build reads them with the samples and roots
    out = os.path.join(TMP, 'o_lpt_e2e')
    evcalls, builds = [], []

    def fake_call(u, target, args, python, log, env=None, cwd=None, stage=None):
        evcalls.append((target, [str(a) for a in args], python, dict(env or {})))
        g = lambda f: str(args[args.index(f) + 1])
        return _Call(0, ['x done', _eval_out(args[1], args[0], g('--tier'), g('--kind'), tuple(g('--samples').split(',')))])

    def fake_inplace(self, u, argv, log, globals_=None, test_stage=False, stage=None, env=None):
        builds.append((list(argv), dict(globals_ or {}), dict(env or {})))
        return _Call(1, ['stopped here (test)'])
    keep_i, keep_x = TP.TCtx.inplace, TS.excluded_events
    K.call, TP.TCtx.inplace, TS.excluded_events = fake_call, fake_inplace, (lambda rows, ex=None: [])
    try:
        rc, err = cli(base + ['--stages', 'tier_eval,tier_site', '--out', out, '--tier-forced-root', ft,
                              '--tier-cutout-root', cut])
    finally:
        K.call, TP.TCtx.inplace, TS.excluded_events = keep, keep_i, keep_x
    st_e, why_e = _reason(out, 'T_eval')
    st_s, why_s = _reason(out, 'T_site')
    want_ev = sorted(os.path.join(out, 'eval_%s_%sgen5_hwj10pclpt' % (ti, 'wb_' if k == 'wb' else '')) for ti, k in t.EVAL_DIRS)
    check('cli: tier_eval runs the four evaluations of HWJ,10pc,LPT on the given roots, each passing forced_tier_eval',
          st_e == 'done' and sorted(a[1][1] for a in evcalls) == want_ev
          and all(a[1][a[1].index('--samples') + 1] == 'HWJ,10pc,LPT' and a[1][0].startswith(ft + '/')
                  and a[0] == EVAL_SCRIPT and a[3].get('P3_PHOT_ROOT') == cut and a[3].get('OMP_NUM_THREADS') == '1'
                  and a[2] == CFG.forced.PYTHON_KEY['tier_eval'] for a in evcalls), '%s %s %s' % (st_e, why_e, evcalls[:1]))
    links = os.path.join(out, 'eval_links_hwj10pclpt')
    lk = {os.path.join(links, t.SCRIPT_EVAL_NAME.format(tier=ti, wb='wb_' if k == 'wb' else '')):
          os.path.join(out, 'eval_%s_%sgen5_hwj10pclpt' % (ti, 'wb_' if k == 'wb' else '')) for ti, k in t.EVAL_DIRS}
    check('cli: the evaluations are named from their samples (hwj10pclpt) and read by the script through links with its '
          'literal names (review F7)', all(os.path.islink(a) and os.path.realpath(a) == os.path.realpath(b)
                                           for a, b in lk.items()), sorted(os.listdir(out)))
    gl = builds[0][1] if builds else {}
    check('cli: tier_site builds from those evaluations: SF = the links directory, SAMPLES with LPT and its label, the tier '
          'files and cutouts in the environment (the build replaced; T-BUILD then refuses)',
          st_s == 'refused' and 'T-BUILD' in why_s and builds and builds[0][0][0] == 'build'
          and gl.get(t.EVAL_ROOT_GLOBAL) == links and gl.get('SAMPLES') == list(SAMPLES)
          and gl.get('SAMPLE_LABEL', {}).get('LPT') == t.SAMPLE_LABEL['LPT']
          and builds[0][2] == {'P3_FORCED_TIERS': ft, 'P3_PHOT_ROOT': cut}, '%s %s %s' % (st_s, why_s, builds[:1]))
    # review F2 / F3: the resume identities. T_eval: the exclusion list an input, the admitted (tier, LST, band, night) set
    # recorded and compared; T_site: the evaluations, samples and roots compared
    rec = record.load_record(out, 'forced', 'T_eval') or {}
    adm = rec.get('admitted') or []
    check('cli: T_eval records the exclusion list as an input and its admitted set (review F2)',
          CFG.paths.DATA_DECISIONS in (rec.get('inputs') or {}) and adm and all(a.count('|') == 3 for a in adm)
          and {a.split('|')[0] for a in adm} == {'10min', 'deep'}, '%s %s' % (adm[:3], sorted(rec.get('inputs') or {})[:3]))
    seen = []
    keep_r = K.resumed
    K.resumed = lambda run, name, inputs, admitted=None, **meta: seen.append((name, admitted, meta)) or True
    try:
        cli(base + ['--stages', 'tier_eval', '--resume', '--out', out, '--tier-forced-root', ft, '--tier-cutout-root', cut])
        TP.TCtx.inplace, TS.excluded_events = fake_inplace, (lambda rows, ex=None: [])
        cli(base + ['--stages', 'tier_site', '--resume', '--out', out, '--tier-forced-root', ft, '--tier-cutout-root', cut])
    finally:
        K.resumed, TP.TCtx.inplace, TS.excluded_events = keep_r, keep_i, keep_x
    se = [x for x in seen if x[0] == 'T_eval']
    ss = [x for x in seen if x[0] == 'T_site']
    check('cli: --resume of T_eval passes the admitted set read now (review F2)', se and se[0][1] == adm, str(se)[:300])
    check('cli: --resume of T_site compares the evaluations, samples and both roots (review F3)',
          ss and {'site', 'evals', 'samples', 'tier_forced_root', 'tier_cutout_root'} <= set(ss[0][2]), str(ss)[:300])


def test_resume_identity():
    """--resume adopts a T_* record only for the same site (and review root, fit work) and the same admitted cell-bands;
    then K.resumed decides (inputs, code, config, outputs). The record read and K.resumed are replaced in this process."""
    from exopipe.phase3.core import record as R
    from exopipe.phase3.lib.forced import common as K
    from exopipe.phase3.lib.forced import tier_pages as TP
    from exopipe.phase3.lib.forced import tier_snapshot as TS

    class _Run:
        resume, step, out_root = True, 'forced', TMP

    class _C:
        pass
    c = _C()
    c.run, c.W, c.t = _Run(), TMP, CFG.forced_tier_pages
    rows = [dict(cell='04h_2025-01-01', band='41MHz', ts='a'), dict(cell='04h_2025-01-02', band='Red', ts='b')]
    calls, bad = [], []
    keep = R.load_record, K.resumed, TS.excluded_events
    R.load_record = lambda out, step, unit: {'meta': {'site': '/s/A', 'review_root': '/r', 'fit_work': '/f'}}
    K.resumed = lambda run, name, inputs, admitted=None, **meta: calls.append((name, admitted, meta)) or True
    TS.excluded_events = lambda rows_, ex=None: list(bad)
    try:
        other = TP._resumed(c, 'tier_site', rows, site='/s/B')
        other_root = TP._resumed(c, 'tier_runs', rows, site='/s/A', review_root='/r2', fit_work='/f')
        same = TP._resumed(c, 'tier_site', rows, site='/s/A')
        bad.append(('04h_2025-01-02', 'Red', 'b'))
        TP._resumed(c, 'tier_site', rows, site='/s/A')
        no_rows = TP._resumed(c, 'tier_runs', None, site='/s/A', review_root='/r', fit_work='/f')
    finally:
        R.load_record, K.resumed, TS.excluded_events = keep
    check('resume: a record of another site or review root is not adopted', other is False and other_root is False
          and all(x[2].get('site') == '/s/A' for x in calls), str(calls))
    check('resume: the same site goes to K.resumed with the admitted cell-bands',
          same is True and calls and calls[0][1] == ['04h_2025-01-01|41MHz', '04h_2025-01-02|Red'], str(calls[:1]))
    check('resume: an exclusion added since changes the admitted set (the unit re-runs; its gates then refuse)',
          len(calls) == 2 and calls[1][1] == ['04h_2025-01-01|41MHz'], str(calls[1:]))
    check('resume: no site events to identify the exclusion state - not skipped', no_rows is False and len(calls) == 2,
          str(calls))


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
        test_eval_content()
        test_cli()
        test_cli_lpt()
        test_resume_identity()
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
