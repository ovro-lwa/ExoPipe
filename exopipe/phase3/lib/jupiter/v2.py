"""`exopipe3 jupiter v2` - the Jupiter V2 build of a cell (the V2 code tree /lustre/gh/main/phase3/jupiter/v2/code, in place;
B-87 default (b): outside git, identified per file by sha256 in the run record). One cell is supported (gate J20: the code
hard-codes 05h_2024-12-29; decision 8 - rebuild the other nights - is not started).

Stages and the production command lines they reproduce (masters/run_all.sh, integrate/run_integration.sh, the hand steps):
  geometry    nightgeom/hzfetch.py DATE G0 G1 C0 C1 (skipped when horizons_<date>/manifest.json exists - a fetch is immutable;
              only a NETWORK_HOSTS host fetches), nightgeom/build_cell.py CELL, nightgeom/test_cell.py CELL
  masters     masters/jupv2_masters.py select raw masters resid fit red10 metrics mastersum test figs (run_all.sh: nproc and
              interpreter per sub-stage, scratch crops/parts of the cell removed before resid, figs after a failed test with the
              unit refused), then masters/bright_ratio.py and masters/imprint_contrast.py (run by hand today)
  prepare     site/jsite_prepare_variant.py per variant (J22: the masters verdict ALL-PASS; an existing derived root refuses)
  provenance  site/jsite_crop_archive.py with every root (J27: on the node holding the crops)
  anchor      anchor/jblue_build.py per root where the product is absent (J23: never overwritten), anchor/jblue_test.py
  refcheck    v2/integrate/inputs/jup_refcheck_20241229.py SCRATCH, then the two tables copied to integrate/inputs (identical
              files kept; a different existing table refuses) - not a default stage (the inputs exist)
  metrics     J24 disclosure check, integrate/jint_disclose.py, jint_metrics.py, jint_compare.py, jint_test_metrics.py
  render      site/jsite_render.py per root (ruled notes, --occ-disclosed, --metrics-dir, nproc 12; J25 published only after its
              own test)
  summary     integrate/jint_content_summary.py (J26 freshness)
Units v2.<stage>.<cell>; the first stage not done stops the cell.

A production --out runs exactly those command lines. A non-production --out (parity) redirects every WRITE root (arguments,
or module globals set by the in-place driver) into --out / --work and reads the products this run did not make from the
production tree; every child runs under the write guard, so a missed redirect raises PermissionError instead of writing
production. masters, metrics and summary cannot be redirected in place (bright_ratio / imprint_contrast have no entry function;
jint_compare / jint_test_metrics read and write under one hard-coded V2 root; jint_content_summary re-runs every component test
in subprocesses that read the production tree) and refuse into a non-production --out.
"""
import csv
import glob
import os
import shutil
import sys
import time

from exopipe.phase3.core.record import NoInput, Refusal
from exopipe.phase3.lib.jupiter import common as K
from exopipe.phase3.lib.jupiter import mirrors

NOT_REDIRECTABLE = {
    'masters': 'masters/bright_ratio.py and imprint_contrast.py have no entry function (hard-coded roots at import)',
    'metrics': 'jint_compare.py / jint_test_metrics.py read and write under one hard-coded V2 root',
    'summary': 'jint_content_summary.py re-runs every component test in subprocesses that read the production V2 tree',
}
# run_all.sh:19-25: sub-stages given --nproc (and --cores) explicitly; the others run with the module defaults
MASTERS_NPROC_STAGES = ('raw', 'resid', 'fit', 'red10')
MASTERS_CORES_STAGES = ('fit', 'red10')


def _C():
    from exopipe.phase3.config import CFG
    return CFG


class Ctx:
    def __init__(self, args, run_, roots, cell):
        cfg = _C()
        self.args, self.run, self.roots, self.cell = args, run_, roots, cell
        self.lst, self.date = cell.split('_', 1)
        self.R = roots.v2
        self.R_prod = os.path.join(cfg.jupiter.ROOT, cfg.jupiter.V2)
        self.code = cfg.jupiter.V2_CODE_ROOT
        self.prod = roots.production
        self.scr = roots.work

    def src(self, rel):
        """A product this run may have made (under --out) or, if not, the production one (read only)."""
        p = os.path.join(self.R, rel)
        return p if os.path.exists(p) else os.path.join(self.R_prod, rel)

    def products(self):
        p = os.path.join(self.roots.products, self.cell)
        return self.roots.products if os.path.isdir(p) else os.path.join(_C().jupiter.ROOT, _C().jupiter.PRODUCTS)

    def root_of(self, label):
        if label == 'prod':
            return self.products()
        return self.src(os.path.join('masters', 'derived', label))

    def specs(self):
        out = []
        for lab in _C().jupiter_v2.ROOT_LABELS:
            out += ['--variant', f'{lab}={self.root_of(lab)}']
        return out

    def mod(self, rel):
        return os.path.join(self.code, rel)

    def call(self, u, rel, argv, stage_key, globals_=None, test_stage=False, log=None, cwd=None):
        res = K.in_place(u, self.mod(rel), 'main', argv=argv, globals_=globals_ if not self.prod else {},
                         python=K.interpreter_key(stage_key), log=log or os.path.join(self.roots.logs, f'v2_{self.cell}.log'),
                         cwd=cwd or os.path.join(self.scr, 'jupv2_inplace'), guard=self.roots.guard, test_stage=test_stage,
                         env=dict(_C().jupiter.ENV_V2_EXTRA))
        return res


def _redirect(value_key, mod_rel, name):
    """{name: config value} when the config value differs from the module literal (reads of pinned inputs)."""
    path = mirrors.module_path('v2:' + mod_rel)
    lit = mirrors.module_literals(path).get(name, (None,))[0]
    val = _C().get(value_key)
    return {name: val} if lit is not None and lit != val else {}


def _check(u, res, what):
    if res.kind == 'refused':
        u.refuse(f'{what}: {res.reason}')
    if res.kind == 'test-failed':
        u.refuse(f'{what}: content test failed ({res.log})')
    if res.kind != 'ok':
        raise RuntimeError(f'{what}: {res.reason} (log {res.log})')


def _fresh_output(u, path, cls, t0):
    if not os.path.exists(path):
        u.check(f'jupiter.v2.written[{os.path.basename(path)}]', False, None, f'{path} not written')
        return
    if os.path.getmtime(path) < t0:
        u.check(f'jupiter.v2.fresh[{os.path.basename(path)}]', False, None, f'{path} predates this stage')
        return
    u.add_output(path, cls)


def disclosure_gate(c):
    """J24: the live occurrence failing set and values = the disclosure record = the accepted set (jint_disclose.check)."""
    from exopipe.phase3.lib.jupiter.tests_v2 import _disclosure_check
    O = _C().jupiter_occurrence
    ok, n, detail = _disclosure_check(c.src(O.RESULTS), c.src(O.DISCLOSURE))
    return ok, detail


# ---- stages ---------------------------------------------------------------------------------------------------------
def st_geometry(u, c, t0):
    V = _C().jupiter_v2
    hz = os.path.join(c.R, 'nightgeom', f'horizons_{c.date}')
    hz_src = c.src(os.path.join('nightgeom', f'horizons_{c.date}'))
    if not os.path.exists(os.path.join(hz_src, 'manifest.json')):
        grid = (c.args.hz_grid.split(',') if c.args.hz_grid else list(V.HZ_GRID.get(c.date) or []))
        if len(grid) != 4:
            u.refuse(f'no Horizons grid for {c.date}: give --hz-grid G0,G1,C0,C1 (jupiter_v2.HZ_GRID)')
        hosts = getattr(_C().site, 'NETWORK_HOSTS', None) or []
        if K.host() not in hosts:
            u.refuse(f'the Horizons fetch needs a host that reaches JPL (site NETWORK_HOSTS {hosts or "unset"}); this is {K.host()}')
        res = c.call(u, 'nightgeom/hzfetch.py', [c.date] + grid, 'v2.geometry', {'OUTROOT': os.path.join(c.R, 'nightgeom')})
        _check(u, res, 'hzfetch')
        _fresh_output(u, os.path.join(hz, 'manifest.json'), 'jupiter_geom_raw', t0)
        hz_src = hz
    else:
        u.note(f'Horizons fetch {hz_src} exists (immutable) - not fetched again')
        u.add_input(os.path.join(hz_src, 'manifest.json'))
    gl = {'OUT': os.path.join(c.R, 'nightgeom'), 'PROD': c.products(), 'jgeom.HZROOT': os.path.dirname(hz_src)}
    gl.update(_redirect('jupiter.SUBLST_ROOT', 'nightgeom/build_cell.py', 'SL'))
    _check(u, c.call(u, 'nightgeom/build_cell.py', [c.cell], 'v2.geometry', gl), 'build_cell')
    gl2 = {'build_cell.OUT': os.path.join(c.R, 'nightgeom'), 'jgeom.HZROOT': os.path.dirname(hz_src)}
    _check(u, c.call(u, 'nightgeom/test_cell.py', [c.cell], 'v2.geometry', gl2, test_stage=True), 'test_cell')
    _fresh_output(u, os.path.join(c.R, 'nightgeom', c.cell, 'CONTENT_TEST.txt'), 'jupiter_geom', t0)


def st_masters(u, c, t0):
    V = _C().jupiter_v2
    from exopipe.phase3.core import legacy
    scr = os.path.join(c.scr, V.SCRATCH['masters'])
    cores = K.pin_free_cores(c.args) or ''
    test_failed = None
    for s in V.MASTERS_STAGES:
        if s in ('bright', 'imprints'):
            script = {'bright': 'masters/bright_ratio.py', 'imprints': 'masters/imprint_contrast.py'}[s]
            res = legacy.run(c.mod(script), [], python=K.interpreter_key('v2.masters'), env=dict(_C().jupiter.ENV), cwd=scr,
                             log=os.path.join(c.roots.logs, f'v2_masters_{s}.log'), unit=u)
            if not res.ok:
                u.refuse(f'masters {s}: {res.reason}')
            continue
        if s == 'resid':
            for d in glob.glob(os.path.join(scr, 'V*', c.cell, 'crops')) + glob.glob(os.path.join(scr, 'V*', c.cell, 'parts')):
                shutil.rmtree(d)
        argv = [s, '--cell', c.cell]
        if s in MASTERS_NPROC_STAGES:
            argv += ['--nproc', str(V.MASTERS_NPROC[s])]
        if s in MASTERS_CORES_STAGES and cores:
            argv += ['--cores', cores]
        res = c.call(u, 'masters/jupv2_masters.py', argv, f'v2.masters.{s}', test_stage=(s == 'test'),
                     log=os.path.join(c.roots.logs, f'v2_masters_{s}.log'), cwd=scr)
        if s == 'test' and res.kind == 'test-failed' and V.MASTERS_TEST_FAILURE_CONTINUES:
            test_failed = res
            continue
        _check(u, res, f'masters {s}')
    _fresh_output(u, os.path.join(c.R, 'masters', f'CONTENT_TEST_{c.cell}.txt'), 'jupiter_v2_masters', t0)
    if test_failed:
        u.refuse(f'masters test failed (figs were made; run_all.sh behaviour) - {test_failed.log}')


def st_prepare(u, c, t0):
    V = _C().jupiter_v2
    msrc = c.src('masters')
    verdict = os.path.join(msrc, f'CONTENT_TEST_{c.cell}.txt')
    if not os.path.exists(verdict) or open(verdict).read().strip().splitlines()[-1:] != ['ALL-PASS']:
        u.refuse(f'J22 the masters verdict {verdict} is not ALL-PASS')
    u.add_input(verdict)
    der = os.path.join(c.R, 'masters', 'derived')
    exist = [v for v in V.VARIANTS if os.path.exists(os.path.join(der, v, c.cell))]
    if exist and not c.args.fresh:
        u.refuse(f'J22 derived root(s) {", ".join(exist)} exist under {der} (jsite_prepare_variant refuses): --fresh --supersede TAG')
    if exist:
        if not c.args.supersede:
            u.refuse('--fresh needs --supersede TAG (derived roots are moved aside, never deleted)')
        dest = os.path.join(der, '_superseded', c.args.supersede)
        os.makedirs(dest)
        for v in exist:
            os.replace(os.path.join(der, v, c.cell), os.path.join(dest, f'{v}_{c.cell}'))
        u.note(f'{len(exist)} derived root(s) moved to {dest}')
    for v in V.VARIANTS:
        argv = ['--src', os.path.join(msrc, v), '--label', v, '--cell', c.cell, '--test-file', verdict, '--out', der]
        _check(u, c.call(u, 'site/jsite_prepare_variant.py', argv, 'v2.prepare', log=os.path.join(c.roots.logs, f'v2_prepare_{v}.log')),
               f'prepare {v}')
        _fresh_output(u, os.path.join(der, v, c.cell, 'CONTENT_TEST.txt'), 'jupiter_v2_root', t0)


def st_provenance(u, c, t0):
    V = _C().jupiter_v2
    argv = ['--cell', c.cell] + c.specs()
    if not c.prod:
        argv += ['--out', os.path.join(c.R, 'provenance'), '--stage', os.path.join(c.scr, V.SCRATCH['provenance'])]
    _check(u, c.call(u, 'site/jsite_crop_archive.py', argv, 'v2.provenance', log=os.path.join(c.roots.logs, 'v2_provenance.log')),
           'provenance (J27: refuses when a referenced crop is not on this node or an archive exists)')
    for lab in V.ROOT_LABELS:
        _fresh_output(u, os.path.join(c.R, 'provenance', c.cell, f'crops_{lab}.tar'), 'jupiter_crop_archive', t0)


def st_anchor(u, c, t0):
    V = _C().jupiter_v2
    K.check_casa(K.interpreter_key('v2.anchor'), c.scr)
    for lab in V.ROOT_LABELS:
        d = os.path.join(c.R, 'anchor', c.cell, lab)
        if not os.path.isdir(d):
            argv = ['--cell', c.cell, '--label', lab, '--root', c.root_of(lab), '--nproc', str(V.ANCHOR_NPROC)]
            if not c.prod:
                argv += ['--out', os.path.join(c.R, 'anchor'), '--scratch', os.path.join(c.scr, V.SCRATCH['anchor'])]
            gl = _redirect('jupiter.SUBLST_ROOT', 'anchor/jblue_build.py', 'SUBLST')
            gl.update(_redirect('jupiter.CUBES_ROOT', 'anchor/jblue_build.py', 'CUBES'))
            _check(u, c.call(u, 'anchor/jblue_build.py', argv, 'v2.anchor', gl, log=os.path.join(c.roots.logs, f'v2_blue_build_{lab}.log')),
                   f'blue build {lab}')
        else:
            u.note(f'J23 anchor {lab}: product exists ({d}) - not rebuilt')
        argv = ['--cell', c.cell, '--label', lab]
        if not c.prod:
            argv += ['--dir', d, '--report', os.path.join(c.R, 'anchor', c.cell, f'CONTENT_TEST_blue_{lab}.txt')]
        gl = _redirect('jupiter.SUBLST_ROOT', 'anchor/jblue_test.py', 'SUBLST')
        _check(u, c.call(u, 'anchor/jblue_test.py', argv, 'v2.anchor', gl, test_stage=True,
                         log=os.path.join(c.roots.logs, f'v2_blue_test_{lab}.log')), f'blue test {lab}')
        u.add_output(os.path.join(d, 'MANIFEST.sha256'), 'jupiter_blue_anchor')


def st_refcheck(u, c, t0):
    from exopipe.phase3.core import legacy
    V = _C().jupiter_v2
    if mirrors.production_value('jupiter.SUBLST_ROOT') != _C().jupiter.SUBLST_ROOT:
        u.refuse('the refcheck analysis hard-codes the sub-LST root (its line 12): --set jupiter.SUBLST_ROOT is not honoured there')
    scr = os.path.join(c.scr, V.SCRATCH['refcheck'])
    os.makedirs(scr, exist_ok=True)
    script = os.path.join(c.R_prod, V.REFCHECK_SCRIPT)
    res = legacy.run(script, [scr], python=K.interpreter_key('v2.refcheck'), env=dict(_C().jupiter.ENV), cwd=scr,
                     log=os.path.join(c.roots.logs, 'v2_refcheck.log'), parity_root=os.pathsep.join(c.roots.guard) if c.roots.guard else None,
                     unit=u)
    if not res.ok:
        raise RuntimeError(f'refcheck: {res.reason}')
    rel_ref, rel_win = V.REF_INPUTS[c.cell]
    for name, rel in (('ref_offsets.csv', rel_ref), ('window_summary.csv', rel_win)):
        src, dst = os.path.join(scr, name), os.path.join(c.R, rel)
        if os.path.exists(dst):
            if K.sha256(dst) == K.sha256(src):
                u.note(f'{dst} identical to the fresh {name} (kept)')
                continue
            u.refuse(f'{dst} exists and differs from the fresh {name}: the reference-tie inputs are never overwritten '
                     f'(move it aside by hand with a manifest)')
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst + '.part')
        if K.sha256(dst + '.part') != K.sha256(src):
            raise RuntimeError(f'copy of {src} differs')
        os.replace(dst + '.part', dst)
    u.add_output(os.path.join(c.R, rel_ref), 'jupiter_refcheck')


def st_metrics(u, c, t0):
    ok, detail = disclosure_gate(c)
    if not ok:
        u.refuse(f'J24 occurrence disclosure: {detail}')
    _check(u, c.call(u, 'integrate/jint_disclose.py', [], 'v2.metrics'), 'jint_disclose')
    _check(u, c.call(u, 'integrate/jint_metrics.py', ['--cell', c.cell] + c.specs(), 'v2.metrics',
                     log=os.path.join(c.roots.logs, 'v2_metrics.log')), 'jint_metrics')
    _check(u, c.call(u, 'integrate/jint_compare.py', ['--cell', c.cell], 'v2.metrics', log=os.path.join(c.roots.logs, 'v2_compare.log')),
           'jint_compare')
    _check(u, c.call(u, 'integrate/jint_test_metrics.py', ['--cell', c.cell] + c.specs(), 'v2.metrics', test_stage=True,
                     log=os.path.join(c.roots.logs, 'v2_metrics_test.log')), 'jint_test_metrics')
    _fresh_output(u, os.path.join(c.R, 'integrate', c.cell, 'CONTENT_TEST_metrics.txt'), 'jupiter_metrics', t0)


def st_render(u, c, t0):
    V, O = _C().jupiter_v2, _C().jupiter_occurrence
    ok, detail = disclosure_gate(c)
    if not ok:
        u.refuse(f'J24 occurrence disclosure: {detail}')
    labels = [x.strip() for x in (c.args.labels or ','.join(V.ROOT_LABELS)).split(',') if x.strip()]
    bad = [x for x in labels if x not in V.ROOT_LABELS]
    if bad:
        u.refuse(f'--labels {bad}: roots are {", ".join(V.ROOT_LABELS)}')
    site = os.path.join(c.R, 'site')
    gl = {}
    if not c.prod:
        for rel, name in (('nightgeom', 'jsite_lib.NIGHTGEOM_DIR'), ('anchor', 'jsite_lib.ANCHOR_DIR')):
            if os.path.exists(os.path.join(c.R, rel)):
                gl[name] = os.path.join(c.R, rel)
        rel_win = V.REF_INPUTS[c.cell][1]
        if os.path.exists(os.path.join(c.R, rel_win)):
            gl['jsite_lib.REFSUM'] = os.path.join(c.R, rel_win)
        for k, v in _redirect('jupiter.SUBLST_ROOT', 'site/jsite_lib.py', 'SUBLST').items():
            gl['jsite_lib.' + k] = v
    for lab in labels:
        argv = ['--cell', c.cell, '--variant', f'{lab}={c.root_of(lab)}']
        if lab != 'prod':
            argv += ['--note', f'{lab}={V.ROOT_NOTES[lab]}']
        argv += ['--nproc', str(V.RENDER_NPROC), '--occ-disclosed', c.src(O.DISCLOSURE),
                 '--metrics-dir', c.src(os.path.join('integrate', c.cell))]
        if not c.prod:
            argv += ['--site', site, '--scratch', os.path.join(c.scr, V.SCRATCH['site'])]
        res = c.call(u, 'site/jsite_render.py', argv, 'v2.render', gl, log=os.path.join(c.roots.logs, f'v2_render_{lab}.log'))
        _check(u, res, f'render {lab} (J25: published only after its own test)')
        link = os.path.join(site, c.cell, lab)
        _fresh_output(u, os.path.join(os.path.realpath(link), 'MANIFEST.sha256'), 'jupiter_site_render', t0)


def st_summary(u, c, t0):
    res = c.call(u, 'integrate/jint_content_summary.py', ['--cell', c.cell], 'v2.summary',
                 log=os.path.join(c.roots.logs, 'v2_content_summary.log'))
    if res.kind != 'ok':
        u.refuse(f'J26 content summary NOT VALID ({res.kind}: {res.reason}; log {res.log})')
    _fresh_output(u, os.path.join(c.R, 'site', c.cell, 'CONTENT_TEST.txt'), 'jupiter_site_summary', t0)


STAGES = {'geometry': st_geometry, 'masters': st_masters, 'prepare': st_prepare, 'provenance': st_provenance,
          'anchor': st_anchor, 'refcheck': st_refcheck, 'metrics': st_metrics, 'render': st_render, 'summary': st_summary}


# ---- the action -----------------------------------------------------------------------------------------------------
def run(args, run_, roots, stages):
    V = _C().jupiter_v2
    cells = K.select_cells(args, run_)
    unsupported = [c for c in cells if c not in V.CELLS_SUPPORTED]
    if unsupported:
        raise Refusal(f'J20 V2 single cell: {", ".join(unsupported)} not in jupiter_v2.CELLS_SUPPORTED {V.CELLS_SUPPORTED} - the '
                      f'V2 code hard-codes 05h_2024-12-29 (jupv2_masters.py:47, cell_lookup.py:23-24, jint_metrics.py:128, '
                      f'jsite_lib.py:25, jsite_test.py:73, jup_refcheck_20241229.py:12); decision 8 is not started')
    mirrors.check_literals(mirrors.ACTION_MODULES['v2'])
    blocked = [s for s in stages if s in NOT_REDIRECTABLE and not roots.production]
    if run_.dry_run:
        return dry_run(args, run_, roots, stages, cells, blocked)
    if blocked:
        raise Refusal('into a non-production --out these v2 stages cannot run in place: '
                      + '; '.join(f'{s} ({NOT_REDIRECTABLE[s]})' for s in blocked))
    roots.need_work()
    for cell in cells:
        c = Ctx(args, run_, roots, cell)
        for stage in stages:
            name = f'v2.{stage}.{cell}'
            if args.resume and not args.fresh:
                ok, why = run_.resume_check(name, inputs=None, admitted=None)
                if ok:
                    u = run_.unit(name, action='v2', stage=stage, cell=cell)
                    u.status, u.reason = 'skipped-valid', why
                    continue
            try:
                with run_.unit(name, action='v2', stage=stage, cell=cell, host=K.host()) as u:
                    K.pin_free_cores(args)
                    STAGES[stage](u, c, time.time() - 1)
            except Exception as e:      # recorded as error; the later stages of this cell do not run
                print(f'jupiter v2 {name}: error ({type(e).__name__}: {e})', file=sys.stderr)
                break
            print(f'jupiter v2 {name}: {u.status}' + (f' ({u.reason})' if u.reason else ''), file=sys.stderr)
            if u.status != 'done':
                break
    return None


def dry_run(args, run_, roots, stages, cells, blocked):
    V = _C().jupiter_v2
    print(f'jupiter v2: {len(cells)} cell(s), stages {",".join(stages)}; {roots.describe()}')
    print(f'  code tree {_C().jupiter.V2_CODE_ROOT} (outside git; sha256 per loaded file goes into each unit record)')
    for cell in cells:
        c = Ctx(args, run_, roots, cell)
        print(f'\n  cell {cell} (J20 supported); products read from {c.products()}')
        ok, detail = disclosure_gate(c)
        for st in stages:
            line = f'    {st}: '
            if st in blocked:
                line += f'REFUSE into a non-production --out ({NOT_REDIRECTABLE[st]})'
            elif st == 'geometry':
                hz = c.src(os.path.join('nightgeom', f'horizons_{c.date}', 'manifest.json'))
                line += (f'Horizons fetch {"present - not fetched again" if os.path.exists(hz) else "needed (network host)"} ({hz}); '
                         f'build_cell.py {cell}; test_cell.py {cell} -> {os.path.join(c.R, "nightgeom", cell)}/CONTENT_TEST.txt')
            elif st == 'masters':
                line += f'jupv2_masters.py {" ".join(V.MASTERS_STAGES[:10])} + bright_ratio.py, imprint_contrast.py -> {c.R}/masters'
            elif st == 'prepare':
                verdict = os.path.join(c.src('masters'), f'CONTENT_TEST_{cell}.txt')
                vok = os.path.exists(verdict) and open(verdict).read().strip().splitlines()[-1:] == ['ALL-PASS']
                exist = [v for v in V.VARIANTS if os.path.exists(os.path.join(c.R, 'masters', 'derived', v, cell))]
                line += (f'J22 masters verdict {"ALL-PASS" if vok else "NOT ALL-PASS - refuses"} ({verdict}); derived roots '
                         f'existing {exist or "none"}' + (' - refuses without --fresh --supersede TAG' if exist and not args.fresh else ''))
            elif st == 'provenance':
                line += f'jsite_crop_archive.py --cell {cell} {" ".join(c.specs()[:4])} ... (on the node holding the crops)'
            elif st == 'anchor':
                have = [l for l in V.ROOT_LABELS if os.path.isdir(os.path.join(c.R, 'anchor', cell, l))]
                line += f'jblue_build.py where absent (present, J23 not rebuilt: {have or "none"}); jblue_test.py per root'
            elif st == 'refcheck':
                line += f'{os.path.join(c.R_prod, V.REFCHECK_SCRIPT)} <scratch>; tables to {c.R}/integrate/inputs (never overwritten)'
            elif st in ('metrics', 'render'):
                line += f'J24 disclosure {"pass" if ok else "REFUSE"} ({detail[:120]}); '
                line += ('jint_disclose, jint_metrics, jint_compare, jint_test_metrics' if st == 'metrics' else
                         f'jsite_render.py per root ({args.labels or ",".join(V.ROOT_LABELS)}) -> {c.R}/site/{cell}')
            elif st == 'summary':
                line += f'jint_content_summary.py --cell {cell} -> {c.R}/site/{cell}/CONTENT_TEST.txt'
            print(line)
    return 2 if blocked else 0
