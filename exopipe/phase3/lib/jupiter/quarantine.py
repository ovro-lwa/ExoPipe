"""`exopipe3 jupiter quarantine` - move superseded V2 render directories of a site cell to a quarantine directory with a
checksum manifest (V2 code tree site/jsite_quarantine.py, then site/jsite_quarantine_links.py and its --test, in place; both take
every root as an argument, so a non-production --out needs no module-global redirect). Never deletes (deletion needs Gregg's
explicit 'delete', CLAUDE.md 7). Unit quarantine.<cell>; output <out>/v2/site/<cell>/<qdir>/MANIFEST.sha256
[jupiter_render_quarantine].
"""
import os
import sys
import time

from exopipe.phase3.core.record import NoInput
from exopipe.phase3.lib.jupiter import common as K
from exopipe.phase3.lib.jupiter import mirrors


def _C():
    from exopipe.phase3.config import CFG
    return CFG


def superseded(site_cell):
    """Render directories no product-root symlink points to (jsite_quarantine.py's definition)."""
    rd = os.path.join(site_cell, 'renders')
    if not os.path.isdir(rd):
        return []
    live = {os.path.realpath(os.path.join(site_cell, e)) for e in os.listdir(site_cell)
            if os.path.islink(os.path.join(site_cell, e))}
    return sorted(d for d in os.listdir(rd) if os.path.realpath(os.path.join(rd, d)) not in live)


def run(args, run_, roots, stages):
    cells = K.select_cells(args, run_)
    qdir = args.qdir or 'quarantine_superseded_' + time.strftime('%Y%m%d', time.gmtime())
    q1 = mirrors.module_path('v2:site/jsite_quarantine.py')
    q2 = mirrors.module_path('v2:site/jsite_quarantine_links.py')
    py = K.interpreter_key('v2.quarantine')
    if run_.dry_run:
        print(f'jupiter quarantine: {len(cells)} cell(s); qdir {qdir}; {roots.describe()}')
        for c in cells:
            sc = os.path.join(roots.v2, 'site', c)
            sup = superseded(sc)
            print(f'  {c}: {sc} - {len(sup)} superseded render dir(s): {", ".join(sup[:6]) or "none"}')
            print(f'    {q1} --site-cell {sc} --qdir {qdir}; {q2} --qdir {os.path.join(sc, qdir)} --v2 {roots.v2} (+ --test)')
        return 0
    for c in cells:
        sc = os.path.join(roots.v2, 'site', c)
        with run_.unit(f'quarantine.{c}', action='quarantine', cell=c) as u:
            sup = superseded(sc)
            if not sup:
                raise NoInput(f'no superseded render directory under {sc}/renders')
            u.meta['superseded'] = sup
            cwd = os.path.join(roots.work or '/tmp', 'quarantine')
            for argv, test in ((['--site-cell', sc, '--qdir', qdir], False),
                               (['--qdir', os.path.join(sc, qdir), '--v2', roots.v2], False),
                               (['--qdir', os.path.join(sc, qdir), '--v2', roots.v2, '--test'], True)):
                mod = q1 if '--site-cell' in argv else q2
                res = K.in_place(u, mod, 'main', argv=argv, python=py, log=os.path.join(roots.logs, f'quarantine_{c}.log'),
                                 cwd=cwd, guard=roots.guard, test_stage=True)
                if res.kind != 'ok':
                    u.refuse(f'{os.path.basename(mod)} {" ".join(argv[-1:])}: {res.kind} {res.reason}')
            u.add_output(os.path.join(sc, qdir, 'MANIFEST.sha256'), 'jupiter_render_quarantine')
        print(f'jupiter quarantine {c}: {u.status}' + (f' ({u.reason})' if u.reason else ''), file=sys.stderr)
    return None
