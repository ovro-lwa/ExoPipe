"""Tests of the masters-deep unit (steps/masters.py dispatcher + deep tier; lib/masters/deep_*.py; lib/masters/tests.py;
config/steps/masters.py). Plain asserts, no pytest:

    NUMPY_MADVISE_HUGEPAGE=0 EXOPIPE3_SITE=/lustre/gh/main/exopipe_ops/exopipe3_site.py \
        /opt/devel/pipeline/envs/py38_orca_nkosogor/bin/python3 tests/test_masters-deep.py

Synthetic data only under /tmp/exopipe3_tests_masters-deep/ (removed at the end). READ-ONLY checks on real production
products (the 23h_23MHz rebuild scratch on lwacalim07 /fast, the installed 23h_Green cell and its parent catalogue): skipped
with a message when a product is not present on this node. Nothing is written under a production root; no processing runs
on real data (the deep build is exercised up to its first gate only).
"""
import csv
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')
PY = sys.executable
T = '/tmp/exopipe3_tests_masters-deep'
CELL = '05h_64MHz'          # in neither ref_override table (checked below)
REAL_GEN = '/fast/gh/main/phase3/rebuild_20261007/23h_23MHz/gen_shflux'
REAL_CHECK = '/lustre/gh/main/exopipe_ops/membership_rebuild_20261007/check_23h_23MHz.txt'
REAL_PARENT = '/lustre/gh/main/phase3_collect/p3s5_collect_lwacalim07/parent_catalogs/master_R075_23h_Green.gaul.csv'

import numpy as np                                        # noqa: E402
from astropy.io import fits                               # noqa: E402

from exopipe.phase3.core import env as _env               # noqa: E402
_env.apply()
_env.legacy()
from exopipe.phase3.config import CFG, ConfigRefusal      # noqa: E402
from exopipe.phase3.core import validate                  # noqa: E402
from exopipe.phase3.core.record import Refusal            # noqa: E402

RESULTS = []


def check(name, cond, detail=''):
    RESULTS.append((name, bool(cond), detail))
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f'  ({detail})' if detail else ''), flush=True)


def tests_of(cls, path, **ctx):
    return {r['name'].split('.', 1)[1]: r for r in validate.run(cls, path, **ctx)}


# ---------------------------------------------------------------------------------------------------------------------
# synthetic products
# ---------------------------------------------------------------------------------------------------------------------

def master_header(dates, extra=None):
    h = fits.Header()
    h['CTYPE1'], h['CTYPE2'], h['CDELT1'], h['CDELT2'] = 'RA---SIN', 'DEC--SIN', -0.0767, 0.0767
    h['CRPIX1'], h['CRPIX2'], h['CRVAL1'], h['CRVAL2'] = 32.0, 32.0, 75.0, 37.2
    h['BMAJ'], h['BMIN'], h['BPA'], h['RESTFRQ'] = 0.2, 0.18, 30.0, 64.0e6
    h['COADDOP'], h['FLUXFIT'], h['NCOADD'] = 'ivw-mean-v3', 'flat-dc', len(dates)
    h['CONSTDIG'] = hashlib.md5(','.join(sorted(set(dates))).encode()).hexdigest()
    for d in dates:
        h['HISTORY'] = f'COADD constituent {d}'
    for k, v in (extra or {}).items():
        h[k] = v
    return h


def write(path, data, hdr=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fits.writeto(path, np.asarray(data, dtype=np.float32), hdr, overwrite=True)
    return path


DATES = ['2025-01-02', '2025-01-03', '2025-01-05', '2025-01-07']
RNG = np.random.default_rng(7)
IMG = (RNG.normal(0, 0.1, (64, 64)) + 0).astype(np.float32)
IMG[20, 20] = 50.0
BAKED = {'SHSCALE': 1.05, 'FLUXBAKE': True, 'FLUXCORR': 0.98, 'FLUXBND': 0.99, 'FLUXABS': 0.99, 'FLUXVER': 'fluxcorr-v2-x',
         'SHERR': 0.01}


def make_working_cell(root, cell, mdwarp=False, factor_shscale=1.05):
    lst, sc = cell.split('_')
    fd = os.path.join(root, cell, 'coadd_final')
    extra = dict(BAKED, SHSCALE=factor_shscale)
    if mdwarp:
        extra['MDWARP'] = True
    write(f'{fd}/I_deep_Taper_R075_{sc}.fits', IMG * 1.05 * 0.98, master_header(DATES, extra))
    write(f'{fd}/V_deep_Taper_R0_{sc}.fits', IMG * 0.1, master_header(DATES, extra))
    with open(f'{fd}/epoch_gate.csv', 'w') as f:
        f.write('date,i_ratio,n_src,verdict,reason,master_md5,method\n')
        f.write('2025-01-02,0.99,40,pass,,abc,epoch content test\n2025-01-03,0.70,40,FAIL,low,abc,epoch content test\n')
    os.makedirs(os.path.join(root, cell, 'flow'), exist_ok=True)
    return fd


# ---------------------------------------------------------------------------------------------------------------------
# 1. config conventions (refactor decisions 2026-10-13 ~04:30)
# ---------------------------------------------------------------------------------------------------------------------

def test_config():
    from exopipe.phase3.lib.masters import deep_inplace as IP
    from exopipe.phase3.core import ledger
    mine = IP.MASTERS_SECTIONS
    for sec in mine:
        s = getattr(CFG, sec, None)
        check(f'config section {sec} loaded', s is not None)
        if s is None:
            continue
        check(f'{sec}: no lower-case markers', not hasattr(s, '_apply') and not hasattr(s, '_protected'))
        keys = {k for k, _ in s.items()}
        wiring = IP.WIRING.get(sec, {})
        check(f'{sec}: every key wired', keys == set(wiring), f'unwired {sorted(keys - set(wiring))}, '
                                                              f'stale {sorted(set(wiring) - keys)}')
        prot = set(getattr(s, 'PROTECTED_KEYS', ()) or ())
        mirrors = {k for k, v in wiring.items() if v == 'mirror'}
        check(f'{sec}: every mirror key protected', mirrors <= prot, f'unprotected mirrors {sorted(mirrors - prot)}')
        check(f'{sec}: PROTECTED_KEYS exist', prot <= keys, f'unknown {sorted(prot - keys)}')
        for k, ids in (getattr(s, 'RULED', {}) or {}).items():
            for lid in (ids if isinstance(ids, (tuple, list)) else (ids,)):
                check(f'{sec}.{k} ruling {lid} in the fix ledger', ledger.has_id(lid, CFG.paths.FIX_LEDGER))
    vocab = {'default', 'flow', 'py36', 'casa', 'tess'}
    vals = set(CFG.masters.PYTHON_SITE_KEY.values())
    check('PYTHON_SITE_KEY uses the site vocabulary only', vals <= vocab, f'{sorted(vals - vocab)}')
    check('PYTHON_SITE_KEY keys exist in the site file', vals <= set(CFG.site.PYTHON), f'site {sorted(CFG.site.PYTHON)}')
    check('sub-LST wideband is not a masters stage (frames owns it)', 'wideband' not in CFG.masters.STAGES['sublst'])
    check('deep default stages = the build part of deep-build-cell',
          tuple(CFG.masters.DEFAULT_STAGES['deep']) == ('build', 'pbmaps', 'vclean', 'fluxscale', 'verify'))
    check('no own key for shared roots', not hasattr(CFG.masters, 'DEEP_WORKING_TREE') and
          not hasattr(CFG.masters, 'ARCHIVE_RUN') and not hasattr(CFG.masters_pbmaps, 'PB_QA_ROOT'))
    # protected / mirror keys are refused by config.py; a wrapper key is accepted (and reset)
    for k in ('masters_build.BEAM_GATE_RATIO', 'masters_fluxscale.SCALE_TABLE', 'masters_sources.THRESH_PIX',
              'masters.STAGES', 'masters_build.FORCE_REBUILD'):
        try:
            CFG.set(k, '1.0')
            ok = False
        except (ConfigRefusal, ValueError):
            ok = True
        CFG.reset()
        check(f'override of {k} refused', ok)
    CFG.set('masters_verify.T1_TOL', '0.02')
    ok = CFG.masters_verify.T1_TOL == 0.02 and not IP.forbidden_overrides()
    CFG.reset()
    check('override of a wrapper key accepted (masters_verify.T1_TOL)', ok and CFG.masters_verify.T1_TOL == 0.01)
    # pinned tables (read-only): the sha1 in config equals the file
    pins = [(CFG.masters_build.SOLAR_AUDIT, CFG.masters_build.SOLAR_AUDIT_SHA1),
            (CFG.masters_build.NAMED_OUT_OF_MASTER, CFG.masters_build.NAMED_OUT_OF_MASTER_SHA1),
            (CFG.masters_build.VARIANT_BAND_MEDIANS, CFG.masters_build.VARIANT_BAND_MEDIANS_SHA1),
            (CFG.masters_fluxscale.SCALE_TABLE, CFG.masters_fluxscale.SCALE_TABLE_SHA1),
            (CFG.masters_fluxscale.FLUXCORR_TABLE, CFG.masters_fluxscale.FLUXCORR_TABLE_SHA1),
            (CFG.masters_stagein.REF_OVERRIDE_TABLE, CFG.masters_stagein.REF_OVERRIDE_TABLE_SHA1)]
    for p, want in pins:
        try:
            got = IP.sha1_pin(p, want, 'test')
            check(f'pin {os.path.basename(p)}', got == want)
        except Refusal as e:
            check(f'pin {os.path.basename(p)}', False, str(e)[:160])


# ---------------------------------------------------------------------------------------------------------------------
# 2. stage selection
# ---------------------------------------------------------------------------------------------------------------------

def test_resolve_stages():
    import argparse
    from exopipe.phase3.steps import masters as M

    def ns(**kw):
        d = dict(tier='deep', stages=None, from_stage=None, until_stage=None)
        d.update(kw)
        return argparse.Namespace(**d)
    check('default deep stages', M.resolve_stages(ns()) == ('build', 'pbmaps', 'vclean', 'fluxscale', 'verify'))
    check('--stages kept in tier order', M.resolve_stages(ns(stages='verify,stage-in')) == ('stage-in', 'verify'))
    check('--from/--until slice', M.resolve_stages(ns(from_stage='pbmaps', until_stage='fluxscale')) ==
          ('pbmaps', 'vclean', 'fluxscale'))
    check('--stages + --until slice', M.resolve_stages(ns(stages='stage-in', until_stage='build')) == ('stage-in', 'build'))
    try:
        M.resolve_stages(ns(stages='wideband'))
        ok = False
    except SystemExit:
        ok = True
    check('unknown stage refused (exit 1)', ok)
    check('sub-LST default stages', M.resolve_stages(ns(tier='sublst')) == ('prep', 'admit', 'stack', 'loo'))


# ---------------------------------------------------------------------------------------------------------------------
# 3. stage-in reference resolver (port of rebuild_cell.sh:47-117 + e5c218e as one rule indexed by CONSTDIG)
# ---------------------------------------------------------------------------------------------------------------------

def _roots(extra_local=None):
    """Test copy roots in production's order, under the scratch tree."""
    r = [('predewarp-quarantine', f'{T}/wq', 'cell'), ('superseded-archive', f'{T}/sup', 'archive'),
         ('unwarped-copy', f'{T}/uw_*', 'flat'), ('quarantine-copy', f'{T}/q_*', 'cell')]
    return repr(tuple(r))


def _set_roots():
    CFG.set('paths.COADD_SB_ROOT', f'{T}/wt')
    CFG.set('masters_stagein.REFERENCE_COPY_ROOTS', _roots())
    CFG.set('masters_stagein.PREDEWARP_QUARANTINE', f'{T}/wq')
    CFG.set('masters_stagein.SUPERSEDED_ARCHIVE', f'{T}/sup')
    CFG.set('masters_stagein.EPOCH_GATE_CARRY_DIR', f'{T}/egc')


def _clean(*names):
    for n in names:
        for d in glob.glob(f'{T}/{n}'):
            shutil.rmtree(d)


def _refuses(fn, needle):
    try:
        fn()
        return False, 'no refusal'
    except Refusal as ex:
        return needle in str(ex), str(ex)[:200]


def test_stagein_branches():
    from exopipe.phase3.lib.masters import deep_stagein as DS
    lst, sc = CELL.split('_')
    ov = DS.working_root_table()
    tabs = DS.reference_tables()
    check(f'{CELL} is in no hand table', CELL not in ov and CELL not in tabs)
    check('prevgen table precedes ref_override_I (11h_18MHz row from ref_prevgen.csv)',
          tabs.get('11h_18MHz', {}).get('table') == 'ref_prevgen.csv' and tabs['11h_18MHz']['V'] is None)
    _set_roots()
    _clean('wq', 'sup', 'uw_*', 'q_*')
    R = lambda **kw: DS.resolve_reference(CELL, C, tables=kw.pop('tables', {}), **kw)
    try:
        # current master / pixel factor (not dewarped)
        fd = make_working_cell(f'{T}/wt', CELL)
        C = DS.working_cell(CELL, ov)
        check('working cell = paths.COADD_SB_ROOT/<cell>', C == f'{T}/wt/{CELL}')
        ents = R()
        e = ents[0]
        f = 1.05 * 0.98
        d = fits.getdata(f'{fd}/I_deep_Taper_R075_{sc}.fits').astype(np.float64)
        check('current-master chosen (same constituents)', e['branch'] == 'current-master' and
              e['generation'] == 'same-constituents' and abs(e['factor'] - f) < 1e-12)
        check('current-master pixels = master / factor (float64 then float32)',
              np.array_equal(e['data'], (d / f).astype(np.float32)))
        check('SH*/FLUX* cards stripped', not [k for k in e['header'] if k.startswith(('SH', 'FLUX'))])
        check('current-master HISTORY text as production', e['history'] ==
              f'2026-10-04 rebuild reference: native un-warped I_deep_Taper_R075 (current master / pixel factor {f:.6f})')
        out_cell = f'{T}/refs/{CELL}'
        written = DS.write_reference(out_cell, sc, ents)
        check('both references written + coadd_final_ref copies', len(written) == 2 and
              all(os.path.exists(q) for _, _, q, _ in written))
        r = tests_of('reference_master', written[0][2], source=e['source'], factor=e['factor'],
                     previous=f'{fd}/I_deep_Taper_R075_{sc}.fits', generation=e['generation'])
        check('reference_master tests pass on the written reference', all(x['passed'] for x in r.values()),
              '; '.join(f"{k}: {x['detail']}" for k, x in r.items() if not x['passed']))
        make_working_cell(f'{T}/wt', CELL, factor_shscale=5.0)
        ok, why = _refuses(lambda: R(read_data=False), 'unusable')
        check('pixel factor outside 0.3-3 refuses', ok, why)
        # dewarped + pre-dewarp copy of the same CONSTDIG in the quarantine
        make_working_cell(f'{T}/wt', CELL, mdwarp=True)
        make_working_cell(f'{T}/wq', CELL)
        ents = R(read_data=False)
        check('predewarp-quarantine chosen (I and V)', [x['branch'] for x in ents] == ['predewarp-quarantine'] * 2)
        # dewarped, no quarantine copy: exactly one native superseded master with the CONSTDIG
        _clean('wq')
        sup = f'{T}/sup/{lst}/{sc}'
        write(f'{sup}/{sc}_I_deep_Taper_Robust-0.75_a.fits', IMG, master_header(DATES))
        write(f'{sup}/{sc}_V_deep_Taper_Robust+0.0_a.fits', IMG * 0.1, master_header(DATES))
        ents = R()
        check('superseded-archive chosen, pixels unchanged', ents[0]['branch'] == 'superseded-archive'
              and np.array_equal(ents[0]['data'], IMG) and ents[0]['factor'] is None)
        write(f'{sup}/{sc}_I_deep_Taper_Robust-0.75_b.fits', IMG, master_header(DATES))
        ok, why = _refuses(lambda: R(read_data=False), 'need exactly 1')
        check('two superseded masters with the CONSTDIG refuse', ok, why)
        os.remove(f'{sup}/{sc}_I_deep_Taper_Robust-0.75_b.fits')
        write(f'{sup}/{sc}_I_deep_Taper_Robust-0.75_a.fits', IMG, master_header(DATES, {'SHSCALE': 1.0}))
        ok, why = _refuses(lambda: R(read_data=False), 'not native')
        check('a non-native superseded master refuses', ok, why)
        _clean('sup')
        # the unwarped copy (ref_override_I's role): newest directory with the CONSTDIG wins; a stale-CONSTDIG copy is skipped
        write(f'{T}/uw_20261009/{CELL}/I_deep_Taper_R075_{sc}.fits', IMG * 1.05, master_header(DATES, dict(BAKED)))
        write(f'{T}/uw_20261013/{CELL}/I_deep_Taper_R075_{sc}.fits', IMG, master_header(DATES[:3], dict(BAKED)))
        ents = R(read_data=False)
        check('unwarped-copy chosen; stale-CONSTDIG newer copy skipped; no V reference',
              ents[0]['branch'] == 'unwarped-copy' and '/uw_20261009/' in ents[0]['source']
              and ents[1]['branch'] == 'none' and 'NONE' in ents[1]['none_reason'], str(DS.plan_summary(ents)))
        check('unwarped-copy HISTORY = production override text', ents[0]['history'].startswith(
              '2026-10-09 rebuild reference: native un-warped I_deep_Taper_R075 (override '))
        # the hand table as a cross-check: a row naming another file refuses; the same file passes
        row = {CELL: dict(table='ref_override_I.csv', constdig=None, I=f'{T}/uw_20261013/{CELL}/I_deep_Taper_R075_{sc}.fits', V=None)}
        ok, why = _refuses(lambda: R(read_data=False, tables=row), 'disagree')
        check('a hand-table row naming another file refuses', ok, why)
        row[CELL]['I'] = ents[0]['source']
        check('a hand-table row naming the chosen file passes', R(read_data=False, tables=row)[0]['source'] == row[CELL]['I'])
        _clean('uw_*')
        # previous generation (Gregg (A)): no copy of the current constituents -> the ALIGNREF master's pre-screen copy
        prev_dates = DATES + ['2025-01-09']
        prev_cd = master_header(prev_dates)['CONSTDIG']
        for st_, var_ in (('I', 'I_deep_Taper_R075'), ('V', 'V_deep_Taper_R0')):
            hh = fits.getheader(f'{fd}/{var_}_{sc}.fits')
            hh['ALIGNREF'] = f'MASTER:ivw-mean-v3:{prev_cd}'
            fits.writeto(f'{fd}/{var_}_{sc}.fits', fits.getdata(f'{fd}/{var_}_{sc}.fits'), hh, overwrite=True)
        ok, why = _refuses(lambda: R(read_data=False), 'nor of the previous generation')
        check('no copy of either generation refuses', ok, why)
        write(f'{T}/q_20261004/{CELL}/coadd_final/I_deep_Taper_R075_{sc}.fits', IMG * 1.05, master_header(prev_dates, dict(BAKED)))
        write(f'{T}/q_20261008b/{CELL}/coadd_final/I_deep_Taper_R075_{sc}.fits', IMG, master_header(prev_dates, dict(BAKED, MDWARP=True)))
        ents = R()
        e = ents[0]
        check('previous-generation chosen: pre-screen copy (dewarped copy skipped), pixels / factor',
              e['generation'] == 'previous-generation' and '/q_20261004/' in e['source'] and e['constdig'] == prev_cd
              and abs(e['factor'] - 1.05 * 0.98) < 1e-12
              and np.array_equal(e['data'], ((IMG * 1.05).astype(np.float32).astype(np.float64) / e['factor']).astype(np.float32)),
              str(DS.plan_summary(ents)))
        check('previous-generation HISTORY = production (e5c218e) text', e['history'] ==
              f'2026-10-13 rebuild reference: PREVIOUS-GENERATION pre-screen I_deep_Taper_R075 ({e["source"]} / pixel factor {1.05 * 0.98:.6f}; Gregg (A))')
        check('previous-generation without a V copy: no V reference', ents[1]['branch'] == 'none')
        write(f'{T}/q_20261004/{CELL}/coadd_final/V_deep_Taper_R0_{sc}.fits', IMG * 0.1, master_header(prev_dates, dict(BAKED)))
        ents = R()
        check('previous-generation V from its pre-screen copy', ents[1]['generation'] == 'previous-generation'
              and ents[1]['constdig'] == prev_cd and ents[1]['source'].endswith(f'V_deep_Taper_R0_{sc}.fits'))
        written = DS.write_reference(f'{T}/refs_pg/{CELL}', sc, ents)
        r = tests_of('reference_master', written[0][2], source=ents[0]['source'], factor=ents[0]['factor'],
                     previous=f'{fd}/I_deep_Taper_R075_{sc}.fits', generation='previous-generation')
        check('reference_master tests pass on a previous-generation reference (CONSTDIG = current ALIGNREF)',
              all(x['passed'] for x in r.values()), '; '.join(f"{k}: {x['detail']}" for k, x in r.items() if not x['passed']))
        r = tests_of('reference_master', written[0][2], source=ents[0]['source'], factor=ents[0]['factor'],
                     previous=f'{fd}/I_deep_Taper_R075_{sc}.fits')
        check('the same reference FAILS the same-constituents CONSTDIG test (the test discriminates)',
              not r['constdig_equals_previous']['passed'])
        row = {CELL: dict(table='ref_prevgen.csv', constdig='0' * 32, I=ents[0]['source'], V=ents[1]['source'])}
        ok, why = _refuses(lambda: R(read_data=False, tables=row), 'declares')
        check('a prevgen row declaring another CONSTDIG refuses', ok, why)
        row = {CELL: dict(table='ref_override_I.csv', constdig=None, I=ents[0]['source'], V=None, generation='same-constituents')}
        ok, why = _refuses(lambda: R(read_data=False, tables=row), 'stale')
        check('a ref_override_I row reached only as the previous generation refuses (stale row, as production)', ok, why)
        # previous generation found in the superseded archive (native): no factor, HISTORY says native
        _clean('q_*')
        sup = f'{T}/sup/{lst}/{sc}'
        write(f'{sup}/{sc}_I_deep_Taper_Robust-0.75_p.fits', IMG, master_header(prev_dates))
        ents = R()
        check('previous generation from the superseded archive: native, no factor',
              ents[0]['generation'] == 'previous-generation' and ents[0]['branch'] == 'superseded-archive'
              and ents[0]['factor'] is None and ents[0]['history'].endswith(', native; Gregg (A))'), str(ents[0]['history']))
        _clean('sup')
        # a root node-local to another host, reached before any copy: refuse (the copy may sit there)
        CFG.set('masters_stagein.REFERENCE_COPY_ROOTS', repr((('unwarped-copy', '/fast/gh/main/phase3/unwarped_ref_*', 'flat'),)))
        from exopipe.phase3.core import env as _env
        if _env.node_local_holder('/fast/gh/main/phase3/x') not in (None, _env.host()):
            ok, why = _refuses(lambda: R(read_data=False), 'node-local to')
            check('a node-local root of another host refuses', ok, why)
        else:
            check('a node-local root of another host refuses (skipped: run on a host other than its holder)', True)
    finally:
        CFG.reset()
        _clean('wq', 'sup', 'uw_*', 'q_*')


# ---------------------------------------------------------------------------------------------------------------------
# 4. the CLI: stage-in, refusal of a second stage-in, --fresh --supersede, build refused without flows
# ---------------------------------------------------------------------------------------------------------------------

def cli(*args):
    env = dict(os.environ)
    p = subprocess.run([PY, '-m', 'exopipe.phase3', 'masters'] + list(args), cwd=REPO, env=env, capture_output=True,
                       text=True, timeout=300)
    return p.returncode, p.stdout + p.stderr


def test_cli():
    lst, sc = CELL.split('_')
    if os.path.isdir(f'{T}/wt'):
        shutil.rmtree(f'{T}/wt')
    if os.path.isdir(f'{T}/sup'):
        shutil.rmtree(f'{T}/sup')
    make_working_cell(f'{T}/wt', CELL)
    out = f'{T}/out'
    common = ['--tier', 'deep', '--cells', CELL, '--out', out, '--cores', 'none',
              '--set', f'paths.COADD_SB_ROOT={T}/wt', '--set', f'masters_stagein.REFERENCE_COPY_ROOTS={_roots()}',
              '--set', f'masters_stagein.EPOCH_GATE_CARRY_DIR={T}/egc',
              '--set', 'masters.FREE_FLOOR_GB[deep]=0']
    rc, txt = cli('--help')
    check('masters --help works', rc == 0 and 'DEEP TIER' in txt and '--deep-installed-root' in txt, txt[-300:])
    rc, txt = cli(*common, '--stages', 'stage-in', '--dry-run')
    check('stage-in --dry-run plans, writes nothing', rc == 0 and 'branch current-master' in txt and not os.path.exists(out),
          txt[-400:])
    # production behaviour (the parity file's value): the carried epoch-gate record
    rc, txt = cli(*common, '--stages', 'stage-in', '--set', 'masters_stagein.EPOCH_GATE_CARRY=carry')
    rec = json.load(open(f'{out}/_records/masters/stage-in_{CELL}.json'))
    classes = sorted(o['class'] for o in rec['outputs'])
    check('stage-in done (exit 0)', rc == 0 and rec['status'] == 'done', txt[-400:])
    check('stage-in (EPOCH_GATE_CARRY carry = production) outputs: 2 reference masters + carried epoch-gate record',
          classes == ['csv_table', 'reference_master', 'reference_master'], str(classes))
    eg = f'{out}/{CELL}/coadd_final/epoch_gate.csv'
    check('epoch-gate record carried byte-identical', open(eg).read() == open(f'{T}/wt/{CELL}/coadd_final/epoch_gate.csv').read())
    # the ruled default (B-05 a): no record carried in
    out_r = f'{T}/out_b05'
    rc, txt = cli(*[x if x != out else out_r for x in common], '--stages', 'stage-in')
    rec_r = json.load(open(f'{out_r}/_records/masters/stage-in_{CELL}.json'))
    check('stage-in (default EPOCH_GATE_CARRY none, B-05 a): 2 reference masters, no epoch-gate record carried',
          rc == 0 and sorted(o['class'] for o in rec_r['outputs']) == ['reference_master', 'reference_master']
          and not os.path.exists(f'{out_r}/{CELL}/coadd_final/epoch_gate.csv')
          and any('no epoch-gate record carried' in n for n in rec_r.get('notes', [])), (txt[-300:], rec_r.get('notes')))
    check('stage-in copies no flows (the flow step does)', not os.path.exists(f'{out}/{CELL}/flow'))
    check('every content test of stage-in passed', all(t['passed'] or t.get('waived') for t in rec['tests']))
    rc, txt = cli(*common, '--stages', 'stage-in')
    rec2 = json.load(open(f'{out}/_records/masters/stage-in_{CELL}.json'))
    check('second stage-in refused (G-M25, exit 2)', rc == 2 and rec2['status'] == 'refused' and 'exists' in rec2['reason'],
          rec2.get('reason', '')[:200])
    rc, txt = cli(*common, '--stages', 'stage-in', '--fresh', '--supersede', 't1')
    check('--fresh --supersede moves the cell aside with a manifest',
          rc == 0 and os.path.isdir(f'{out}/_superseded/t1/{CELL}') and os.path.exists(f'{out}/_superseded/t1/{CELL}.manifest.csv'),
          txt[-300:])
    rc, txt = cli(*common)                                  # default stages: build first
    recb = json.load(open(f'{out}/_records/masters/build_{CELL}.json'))
    check('build refused without the flow step (G-M05, exit 2)', rc == 2 and recb['status'] == 'refused'
          and 'flow' in recb['reason'], recb.get('reason', '')[:200])
    check('later stages not attempted after a refusal', not os.path.exists(f'{out}/_records/masters/pbmaps_{CELL}.json'))
    rc, txt = cli(*common, '--set', 'masters_build.COADDOP=x')
    check('--set on a mirror key refused by config (exit 2)', rc == 2 and 'PROTECTED' in txt, txt[-200:])
    rc, txt = cli(*common, '--epochs', f'{lst}_2025-01-02')
    check('--epochs refused for the deep tier (exit 2)', rc == 2 and 'intermittent' in txt, txt[-200:])
    rc, txt = cli('--tier', 'deep', '--cells', CELL, '--stages', 'stage-in', '--cores', 'none')
    check('default --out (production generation root) refused while PRODUCTION_WRITES is False (exit 2)',
          rc == 2 and 'PRODUCTION_WRITES is False' in txt and not os.path.exists('/fast/gh/main/phase3/rebuild_20261007/_records/masters'),
          txt[-200:])


# ---------------------------------------------------------------------------------------------------------------------
# 5. in-place helpers
# ---------------------------------------------------------------------------------------------------------------------

def test_inplace():
    from exopipe.phase3.lib.masters import deep_inplace as IP
    os.environ['P3_COADD_ROOT'] = '/nowhere'
    gone = IP.clear_inherited_p3()
    check('inherited P3_* cleared', 'P3_COADD_ROOT' in gone and 'P3_COADD_ROOT' not in os.environ)
    with IP.call_env(['masters', 'masters_build']) as env:
        inside = (os.environ.get('P3_ARCHIVE_WRITE'), os.environ.get('P3_FORCE_REBUILD'), os.environ.get('P3_ARCHIVE_RUN'))
    check('call env = rebuild_cell.sh:26 exports', inside == ('0', '1', CFG.paths.ARCHIVE_RUN) and env['P3_REF_MODE'] == 'master')
    check('call env restored', 'P3_ARCHIVE_WRITE' not in os.environ)
    IP.set_import_env()
    applied = dict(IP.apply_attrs('masters_build'))
    from exopipe.phase3.lib.masters import coadd, night_image_gate
    check('attr keys applied to the in-place modules', coadd.MIN_DATES_FOR_VARIANT == 2 and coadd.LOW_OUTLIER_SIGMA == 4.0
          and night_image_gate.THRESHOLD == 0.5 and applied.get('coadd.REF_MODE') == 'master')
    try:
        IP.check_mirrors()
        IP.check_import_env()
        ok = True
    except Refusal as e:
        ok = str(e)
    check('mirror literals equal config; pb_apply imported with the config values', ok is True, str(ok)[:200])
    try:
        ex, _ = IP.exclusion_readers_agree('23h', '23MHz', ['2025-08-09', '2025-08-16', '2025-08-18', '2025-09-09'])
        check('exclusion readers agree (real data_decisions.csv, 23h 23MHz)', ex.admitted('23h', '23MHz', ['2025-08-16']) == [])
    except Refusal as e:
        check('exclusion readers agree', False, str(e)[:200])
    check('production paths recognised', IP.is_production_path('/lustre/gh/main/phase3_collect/x') and
          IP.is_production_path(os.path.join(CFG.paths.OPS_ROOT, 'x')) and not IP.is_production_path(T))


# ---------------------------------------------------------------------------------------------------------------------
# 6. content tests on synthetic files
# ---------------------------------------------------------------------------------------------------------------------

def test_content_synthetic():
    lst, sc = CELL.split('_')
    cd = f'{T}/cells/{CELL}'
    fd = f'{cd}/coadd_final'
    good = write(f'{fd}/I_deep_Taper_R075_{sc}.fits', IMG, master_header(DATES))
    r = tests_of('master', good)
    check('master: CONSTDIG/NCOADD consistent passes', r['constdig_matches_history']['passed'])
    check('master: integrity passes', r['integrity']['passed'])
    bad = write(f'{fd}/I_deep_Taper_R0_{sc}.fits', IMG, master_header(DATES, {'NCOADD': 3}))
    check('master: NCOADD != constituents fails', not tests_of('master', bad)['constdig_matches_history']['passed'])
    blown = IMG.copy()
    blown[0, 0] = 2e5
    bad = write(f'{fd}/I_deep_NoTaper_R0_{sc}.fits', blown, master_header(DATES))
    check('master: |pixel| >= 1e5 fails integrity', not tests_of('master', bad)['integrity']['passed'])
    # LOO
    parent_h = fits.getheader(good)
    lh = master_header(DATES, {'NCOADD': 3, 'LOOEXCL': '2025-01-02', 'PARENTDG': parent_h['CONSTDIG']})
    loo = write(f'{fd}/loo/2025-01-02_I_loo_Taper_R075_{sc}.fits', IMG, lh)
    check('loo_master: derivative of the parent passes', tests_of('loo_master', loo)['parentdg']['passed'])
    lh['PARENTDG'] = 'x' * 32
    loo2 = write(f'{fd}/loo/2025-01-03_I_loo_Taper_R075_{sc}.fits', IMG, lh)
    check('loo_master: wrong PARENTDG fails', not tests_of('loo_master', loo2)['parentdg']['passed'])
    check('loo_master: count per Stokes (2 of NCOADD 4) fails', not tests_of('loo_master', loo)['count_per_stokes']['passed'])
    # PB map + sibling
    yy, xx = np.mgrid[:64, :64]
    pb = np.clip(1 - ((yy - 32) ** 2 + (xx - 32) ** 2) / 900.0, 0, 1).astype(np.float32)
    ph = fits.Header({'PBLIBV': 'v1-2026-07-04', 'PBMODEL': 'v1-2026-07-04', 'PBCHROM': 'bs-sqbp-v1',
                      'PBRESID': 'residual_pb_v2-perband', 'PBROBUST': '0', 'PBBMAJ': 0.2})
    for name in ('PB_deep', 'PB_deep_R075'):
        write(f'{fd}/{name}_{sc}.fits', pb, ph)
    r = tests_of('pb_map', f'{fd}/PB_deep_{sc}.fits')
    check('pb_map: values and both maps pass', r['values']['passed'] and r['both_maps']['passed'],
          r['values']['detail'])
    zero = write(f'{T}/cells/zero/coadd_final/PB_deep_{sc}.fits', np.zeros((64, 64)), ph)
    check('pb_map: all-zero map fails (latent defect D2)', not tests_of('pb_map', zero)['values']['passed'])
    sib = np.where(pb > 0.1, IMG / np.where(pb > 0, pb, 1), np.nan).astype(np.float32)
    sp = write(f'{fd}/I_deep_Taper_R075_pbcorr_{sc}.fits', sib, fits.Header({'PBFLOOR': 0.1}))
    r = tests_of('pbcorr_sibling', sp)
    check('pbcorr_sibling: sibling x PB = plain passes', r['sibling_check']['passed'] and r['tested_pixels']['passed'],
          r['sibling_check']['detail'])
    sib2 = sib.copy()
    sib2[32, 32] *= 1.01
    sp2 = write(f'{T}/cells/{CELL}b/coadd_final/I_deep_Taper_R075_pbcorr_{sc}.fits', sib2, fits.Header({'PBFLOOR': 0.1}))
    shutil.copy(good, f'{T}/cells/{CELL}b/coadd_final/I_deep_Taper_R075_{sc}.fits')
    shutil.copy(f'{fd}/PB_deep_R075_{sc}.fits', f'{T}/cells/{CELL}b/coadd_final/PB_deep_R075_{sc}.fits')
    check('pbcorr_sibling: 1 per cent off at one pixel fails', not tests_of('pbcorr_sibling', sp2)['sibling_check']['passed'])
    # weights_table
    with open(f'{fd}/weights.csv', 'w') as f:
        f.write('date,weight,qa3_scale,qa3_tilt,qa3_nsrc,qa3_guard,qa3_rms_mjy,sci_dir,in_master,flux_dc,qa3_window\n')
        for d in DATES:
            f.write(f'{d},1,1.0,0,50,pass,10,/x,1,1.0,inside\n')
    with open(f'{cd}/residuals_provisional.csv', 'w') as f:
        f.write('lst,freq,date,i_rms_mjy,v_rms_mjy,weight,has_flow,is_reference,stage2_metric,i_outlier,v_outlier,'
                'beam_outlier,outlier,i_thresh_mjy,v_thresh_mjy\n')
        for d in DATES[:3]:
            f.write(f'{lst},{sc},{d},10,5,1,True,False,nonmatched-provisional-v1,False,False,False,False,20,10\n')
    check('weights_table: dates differ from residuals_provisional.csv fails (T7)',
          not tests_of('weights_table', f'{fd}/weights.csv')['columns_and_rows']['passed'])
    with open(f'{cd}/residuals_provisional.csv', 'a') as f:
        f.write(f'{lst},{sc},{DATES[3]},11,5,1,True,False,nonmatched-provisional-v1,False,False,False,False,20,10\n')
    check('weights_table: equal date sets pass',
          tests_of('weights_table', f'{fd}/weights.csv')['columns_and_rows']['passed'])
    with open(f'{fd}/stage2_verdict.csv', 'w') as f:
        f.write('date,i_rms_mjy,v_rms_mjy,i_outlier,v_outlier,beam_outlier,outlier,i_thresh_mjy,v_thresh_mjy,metric,'
                'measured_against\n2025-01-02,10,5,False,False,False,False,20,10,psfmatched-option2-v1,provisional\n')
    check('membership_record: a superseded metric fails',
          not tests_of('membership_record', f'{fd}/stage2_verdict.csv')['stage2_metric']['passed'])
    # master_catalog: sidecar digests
    pdir = f'{T}/parent_catalogs'
    os.makedirs(pdir, exist_ok=True)
    gaul = f'{pdir}/master_R075_{CELL}.gaul.csv'
    with open(gaul, 'w') as f:
        f.write('# Gaussian list\n# Gaus_id, RA\n   0, 1.0\n   1, 2.0\n')
    d = fits.getdata(good).astype(np.float32)
    meta = {'master': os.path.basename(good), 'thresh_isl': 4.0, 'thresh_pix': 5.0,
            'master_md5_data': hashlib.md5(np.ascontiguousarray(d).tobytes()).hexdigest()}
    json.dump(meta, open(gaul + '.gen.json', 'w'))
    r = tests_of('master_catalog', gaul, master=good)
    check('master_catalog: raw digest, rows, recipe pass', r['sidecar_md5']['passed'] and r['rows']['passed'] and
          r['recipe']['passed'], r['sidecar_md5']['detail'])
    meta['master_md5_data'] = 'f' * 32
    json.dump(meta, open(gaul + '.gen.json', 'w'))
    check('master_catalog: a digest of another master fails', not tests_of('master_catalog', gaul, master=good)['sidecar_md5']['passed'])
    with open(gaul, 'w') as f:
        f.write('# Gaussian list\n')
    check('master_catalog: 0 rows fails', not tests_of('master_catalog', gaul, master=good)['rows']['passed'])


# ---------------------------------------------------------------------------------------------------------------------
# 7. read-only checks on real production products
# ---------------------------------------------------------------------------------------------------------------------

def test_real_products():
    from exopipe.phase3.lib.masters import deep_verify as DV
    from exopipe.phase3.lib.masters import deep_stages as DSt
    row = DSt.scale_row('23h', '23MHz')
    prod = None
    tab = '/fast/gh/main/phase3/rebuild_20261007/shflux_table.csv'
    if os.path.exists(tab):
        prod = [r for r in csv.DictReader(open(tab)) if r['lst'] == '23h' and r['scope'] == '23MHz'][-1:]
    if prod:
        p = prod[0]
        check('shflux row = production row (23h 23MHz)', all(str(row[k]) == p[k] for k in ('status', 'scale', 'err', 'nsrc',
                                                                                          'srcs', 'freq_src', 'detail')),
              f"{row['scale']} vs {p['scale']}")
    else:
        print('SKIP  production shflux_table.csv not on this node')
    if os.path.isdir(REAL_GEN):
        t0 = time.time()
        res = {n: (ok, v, d) for n, ok, v, d in DV.run_all(REAL_GEN)}
        check('T1-T8 pass on the production 23h_23MHz generation', all(ok for ok, _, _ in res.values()),
              '; '.join(f'{n}: {d}' for n, (ok, v, d) in res.items() if not ok))
        txt = open(REAL_CHECK).read() if os.path.exists(REAL_CHECK) else ''
        check('T1 value = production check file (median 0.9969 at 468 peaks)',
              'at 468 peaks: median 0.9969' in res['T1'][2] and 'median 0.9969' in txt, res['T1'][2])
        check('T5 count = production (114 files)', res['T5'][1] == 114)
        print(f'      (T1-T8 in {time.time() - t0:.1f} s)')
        r = tests_of('master', f'{os.path.dirname(REAL_GEN)}/coadd_final/I_deep_Taper_R075_23MHz.fits')
        check('master tests pass on the production 23h_23MHz master', all(x['passed'] for x in r.values()))
    else:
        print(f'SKIP  {REAL_GEN} not on this node (installed and removed, or another node)')
    if os.path.exists(REAL_PARENT):
        r = tests_of('master_catalog', REAL_PARENT)
        check('master_catalog tests pass on the installed 23h_Green parent catalogue (NaN-border digest)',
              all(x['passed'] for x in r.values()), '; '.join(x['detail'] for x in r.values() if not x['passed']))
    else:
        print(f'SKIP  {REAL_PARENT} missing')


# ---------------------------------------------------------------------------------------------------------------------
# 8. review fixes 2026-10-13: build clean start, chain-of-records currency, installed-cell gate of sources, T-test failures
# ---------------------------------------------------------------------------------------------------------------------

def test_review_fixes():
    from exopipe.phase3.core import record as R
    from exopipe.phase3.lib.masters import deep_stages as DS, deep_verify as DV
    from exopipe.phase3.lib.masters import deep_inplace as IP
    lst, sc = CELL.split('_')
    # (1) build_clean_start: a stage-in cell is clean; any earlier build product or an overwritten reference is not
    cd = f'{T}/rf/{CELL}'
    write(f'{cd}/coadd_final/I_deep_Taper_R075_{sc}.fits', IMG, master_header(DATES))
    write(f'{cd}/coadd_final/V_deep_Taper_R0_{sc}.fits', IMG * 0.1, master_header(DATES))
    os.makedirs(f'{cd}/coadd_final_ref')
    for v in ('I_deep_Taper_R075', 'V_deep_Taper_R0'):
        shutil.copy(f'{cd}/coadd_final/{v}_{sc}.fits', f'{cd}/coadd_final_ref/')
    os.makedirs(f'{cd}/flow')
    os.makedirs(f'{cd}/coadd_provisional')               # empty: a provisional stage that refused before writing
    check('clean start: a stage-in cell (+ flow/, an empty coadd_provisional/) is clean', DS.build_clean_start(cd, sc) == [],
          str(DS.build_clean_start(cd, sc)))
    with open(f'{cd}/residuals_provisional.csv', 'w') as f:
        f.write('date\n')
    check('clean start: an earlier provisional record refuses', DS.build_clean_start(cd, sc) == ['residuals_provisional.csv'])
    os.remove(f'{cd}/residuals_provisional.csv')
    write(f'{cd}/coadd_final/I_deep_Taper_R075_{sc}.fits', IMG * 1.5, master_header(DATES))   # the new master overwrote it
    why = DS.build_clean_start(cd, sc)
    check('clean start: a reference overwritten by a build refuses', len(why) == 1 and 'I_deep_Taper_R075' in why[0], str(why))
    shutil.copy(f'{cd}/coadd_final_ref/I_deep_Taper_R075_{sc}.fits', f'{cd}/coadd_final/')
    # (2) the CLI refuses a second build on a built cell before any in-place call (G-M25 at build)
    out = f'{T}/rf'
    with open(f'{cd}/coadd_final/weights.csv', 'w') as f:
        f.write('date,in_master\n')
    R.atomic_write_json(R.record_path(out, 'masters', f'stage-in_{CELL}'), {'status': 'done', 'outputs': [], 'meta': {}})
    rc, txt = cli('--tier', 'deep', '--cells', CELL, '--out', out, '--cores', 'none', '--stages', 'build',
                  '--set', 'masters.FREE_FLOOR_GB[deep]=0')
    rec = json.load(open(f'{out}/_records/masters/build_{CELL}.json'))
    check('a build on a built cell is refused (exit 2) and names stage-in --fresh --supersede', rc == 2 and
          rec['status'] == 'refused' and 'supersede' in rec['reason'] and 'coadd_final/weights.csv' in rec['reason'],
          rec.get('reason', '')[:200])
    rc, txt = cli('--tier', 'deep', '--cells', CELL, '--out', out, '--cores', 'none', '--stages', 'build', '--dry-run')
    check('the dry run shows the clean-start gate', rc == 2 and 'not as stage-in left it' in txt, txt[-300:])
    # (3) chain of records: outputs of a done prerequisite record gone -> refuse; a night the build admitted excluded now
    o2 = f'{T}/rf2'
    R.atomic_write_json(R.record_path(o2, 'masters', 'build_23h_23MHz'),
                        {'status': 'done', 'admitted': ['2025-08-09', '2025-08-18'],
                         'outputs': [{'path': f'{o2}/23h_23MHz/coadd_final/I_deep_Taper_R075_23MHz.fits', 'class': 'master'}]})
    ok, why = _refuses(lambda: DS.require_stage(o2, 'pbmaps', '23h_23MHz', {}), 'are gone')
    check('pbmaps refused when the done build record\'s outputs are gone (superseded cell)', ok, why)
    write(f'{o2}/23h_23MHz/coadd_final/I_deep_Taper_R075_23MHz.fits', IMG, master_header(DATES))
    check('exclusion_stale: 23h 23MHz 2025-08-18 (excluded 2026-10-12) found in the build\'s admitted set',
          DS.exclusion_stale(o2, '23h_23MHz') == ['2025-08-18'], str(DS.exclusion_stale(o2, '23h_23MHz')))
    ok, why = _refuses(lambda: DS.require_stage(o2, 'pbmaps', '23h_23MHz', {}), 'excludes now')
    check('pbmaps refused: the build admitted a night excluded now (exclusions absolute)', ok, why)
    ok2 = DS.require_stage(o2, 'pbmaps', '23h_23MHz', {('build', '23h_23MHz'): 'done'}) is not None
    check('... but not when the build ran in this invocation', ok2)
    R.atomic_write_json(R.record_path(o2, 'masters', 'build_23h_23MHz'),
                        {'status': 'done', 'admitted': ['2025-08-09'], 'outputs': []})
    check('exclusion_stale: [] when every admitted night is still admitted', DS.exclusion_stale(o2, '23h_23MHz') == [])
    # (4) the co-add's 'Too few aligned dates' return is a refusal that names its reason
    tee = IP.Tee(None)
    tee.write('  Too few aligned dates (2)\n')
    ok, why = _refuses(lambda: DS._refuse_or_noinput(None, tee, 'co-add provisional'), 'Too few aligned dates')
    check('co-add None after "Too few aligned dates" -> refusal naming it', ok, why)
    # (5) pbmaps sibling expectation: 5, or 6 when the cleaned V exists (pbmaps re-run after vclean)
    fd = f'{T}/rf3/{CELL}/coadd_final'
    os.makedirs(fd)
    n5 = DS.siblings_expected(fd, sc)
    write(f'{fd}/V_deep_Taper_R0_clean_{sc}.fits', IMG * 0.1)
    check('pbmaps siblings expected: 5, 6 with the cleaned V', (n5, DS.siblings_expected(fd, sc)) == (5, 6))
    # (6) master_generation T2 / T7 FAIL on a broken record (the T-tests discriminate)
    gd = f'{T}/rf4/{CELL}/gen_shflux'
    os.makedirs(gd)
    os.makedirs(f'{T}/rf4/{CELL}/coadd_final')
    with open(f'{T}/rf4/{CELL}/coadd_final/weights.csv', 'w') as f:
        f.write('date,in_master,flux_dc\n2025-01-02,1,1.0\n2025-01-03,1,1.01\n')
    with open(f'{T}/rf4/{CELL}/coadd_final/night_image_gate.csv', 'w') as f:
        f.write('date,verdict\n2025-01-02,PASS\n2025-01-03,FAIL\n')
    with open(f'{T}/rf4/{CELL}/residuals_provisional.csv', 'w') as f:
        f.write('date\n2025-01-02\n')
    ok2, _, det2 = DV.t2(gd)
    ok7, _, det7 = DV.t7(gd)
    check('T2 fails: a night-image FAIL night in the master', not ok2 and '2025-01-03' in det2, det2)
    check('T7 fails: residuals_provisional.csv dates != weights.csv dates', not ok7 and '2025-01-03' in det7, det7)
    # (6b) node-local working cell of another host refuses (chain_cell.sh:17, via the site's NODE_LOCAL_ROOTS)
    old = CFG.site.NODE_LOCAL_ROOTS
    try:
        CFG.site.NODE_LOCAL_ROOTS = {'not-this-host': [f'{T}/rf']}
        ok, why = _refuses(lambda: DS.not_here(f'{T}/rf/{CELL}', 'the working cell', 'stage-in'), 'node-local to not-this-host')
    finally:
        CFG.site.NODE_LOCAL_ROOTS = old
    check('a working cell node-local to another host refuses', ok, why)
    # (7) sources: the installed cell's membership.verify (install_cell.sh:70-71) - real, read-only
    real = '/data07/gh/phase3/coadd_sb'
    if os.path.isdir(real):
        DS.inplace(f'{T}/rf5')
        # the installed cells change as production installs rebuilds (21h_18MHz held 2025-08-06 until round 20261013):
        # take the first installed cell whose master holds a now-excluded night, and a clean one for the positive check
        bad = clean = None
        for c in sorted(os.listdir(real)):
            if not os.path.isdir(f'{real}/{c}/coadd_final') or (bad and clean):
                continue
            try:
                v = DS.installed_cell_violations(real, c)
            except Exception:
                continue
            if not bad and any('excluded' in x for x in v):
                bad = (c, v)
            elif not clean and not v:
                clean = c
        check('an installed cell without violations passes membership.verify', clean is not None, 'no clean cell found')
        if clean:
            # the same check with one of the clean cell's master nights added to the exclusions: membership.verify flags it
            from exopipe.phase3.lib.masters import coadd as CO, membership as MB
            lst_c, sc_c = clean.split('_', 1)
            w = list(csv.DictReader(open(f'{real}/{clean}/coadd_final/weights.csv')))
            night = next(r['date'] for r in w if str(r.get('in_master', '1')) == '1')
            ex = set(CO.load_exclusions()) | {(lst_c, sc_c.replace('MHz', ''), night)}
            vv = list(MB.verify(f'{real}/{clean}', exclusions=ex))
            check(f'membership.verify flags a master night of {clean} once it is excluded ({night})',
                  any(night in x and 'excluded' in x for x in vv), str(vv[:2]))
        if bad is None:
            print('SKIP  no installed master holds an excluded night today: the sources refusal on a real cell is not '
                  'exercised (membership.verify itself is checked above with an injected exclusion)')
            return
        c, v = bad
        check(f'installed {c} fails membership.verify (an excluded night still in its master)', True, str(v[:2]))
        rc, txt = cli('--tier', 'deep', '--cells', c, '--out', f'{T}/rf5', '--work', f'{T}/w5', '--cores', 'none',
                      '--stages', 'sources', '--waive', 'masters.sources.install_record', '--ruling', 'test (no product)')
        rec = json.load(open(f'{T}/rf5/_records/masters/sources_{c}.json'))
        check('sources refused on an installed master holding an excluded night, before PyBDSF (exit 2)',
              rc == 2 and rec['status'] == 'refused' and 'membership' in rec['reason']
              and not os.path.exists(f'{T}/rf5/parent_catalogs'), rec.get('reason', '')[:200])
    else:
        print(f'SKIP  {real} missing (not lwacalim07)')


def main():
    t0 = time.time()
    if os.path.exists(T):
        shutil.rmtree(T)
    os.makedirs(T)
    try:
        for fn in (test_config, test_resolve_stages, test_stagein_branches, test_inplace, test_content_synthetic, test_cli,
                   test_review_fixes, test_real_products):
            print(f'---- {fn.__name__}', flush=True)
            try:
                fn()
            except Exception as e:
                import traceback
                traceback.print_exc()
                check(f'{fn.__name__} raised', False, f'{type(e).__name__}: {e}')
    finally:
        shutil.rmtree(T, ignore_errors=True)
    bad = [r for r in RESULTS if not r[1]]
    print(f'\n{len(RESULTS) - len(bad)} of {len(RESULTS)} checks passed in {time.time() - t0:.0f} s')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
