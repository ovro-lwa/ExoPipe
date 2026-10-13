"""`exopipe3 jupiter occurrence` - CML x Io-phase occurrence maps from the Nancay decametric catalogue (Marques et al. 2017),
the V2 code tree occurrence/ in place, in the order and with the arguments of occurrence/run_all.sh (fetch_horizons ovro,
fetch_horizons nancay, validate_geometry, digitize_literature, build_maps, make_figures, cell_lookup, content_test), the chain
stopping at the first failure (run_all.sh:16). One run-wide unit 'occurrence'.

The modules hard-code their output roots at import (OUT, SCR, FIG, CELL, V) and two of them (validate_geometry,
digitize_literature) run inline __main__ blocks, so the chain cannot be redirected to another root in place: the action runs
only into the production root (refused in the refactor window while paths.PRODUCTION_WRITES is False; refused for any other
--out). The parity check of this component is verify-only (spec P5): `exopipe3 verify jupiter_occurrence
/lustre/gh/main/phase3/jupiter/v2/occurrence/validation/content_test_results.json` and `verify jupiter_disclosure ...`.
The content test (content_test.py) exits 1 while the 3 accepted gating failures stand; the unit is done when the failing set
and values equal the disclosure record and the accepted set exactly (class jupiter_occurrence, gate J24).
"""
import os
import sys

from exopipe.phase3.core.record import Refusal
from exopipe.phase3.lib.jupiter import common as K
from exopipe.phase3.lib.jupiter import mirrors

# stage -> [(script, argv)] (run_all.sh:12-13)
CHAIN = {'fetch': [('fetch_horizons.py', ['ovro']), ('fetch_horizons.py', ['nancay'])],
         'validate': [('validate_geometry.py', [])], 'digitize': [('digitize_literature.py', [])],
         'maps': [('build_maps.py', [])], 'figures': [('make_figures.py', [])], 'lookup': [('cell_lookup.py', [])],
         'test': [('content_test.py', [])]}


def _C():
    from exopipe.phase3.config import CFG
    return CFG


def run(args, run_, roots, stages):
    from exopipe.phase3.core import legacy
    mirrors.check_literals(mirrors.ACTION_MODULES['occurrence'])
    code = os.path.join(_C().jupiter.V2_CODE_ROOT, 'occurrence')
    occ = os.path.join(roots.v2, 'occurrence')
    py = K.interpreter_key('occurrence')
    if run_.dry_run:
        print(f'jupiter occurrence: stages {",".join(stages)}; {roots.describe()}')
        if not roots.production:
            print('  REFUSE: the occurrence modules hard-code their output roots (OUT/SCR/FIG/CELL/V at import; inline __main__ '
                  'blocks) - runs only into the production root; parity = verify only (spec P5)')
        for st in stages:
            for script, argv in CHAIN[st]:
                print(f'  {st}: {legacy.interpreter(py)} {os.path.join(code, script)} {" ".join(argv)} (cwd {code})')
        print(f'  output {os.path.join(occ, "validation", "content_test_results.json")} [jupiter_occurrence]')
        return 0 if roots.production else 2
    if not roots.production:
        raise Refusal('occurrence runs only into the production root: its modules hard-code OUT/SCR/FIG/CELL/V at import and two '
                      'run inline __main__ blocks, so they cannot be redirected in place; check this component with '
                      '`exopipe3 verify jupiter_occurrence <results json>` (spec P5)')
    with run_.unit('occurrence', action='occurrence', stages=list(stages)) as u:
        u.add_input(os.path.join(_C().jupiter_occurrence.NANCAY_DIR, 'obs.dat'))
        for p in _C().jupiter_occurrence.LITERATURE:
            u.add_input(p)
        for st in stages:
            for script, argv in CHAIN[st]:
                K.pin_free_cores(args)
                res = legacy.run(os.path.join(code, script), argv, python=py, env=dict(_C().jupiter.ENV), cwd=code,
                                 log=os.path.join(roots.logs, f'occurrence_{script[:-3]}.log'), unit=u)
                if st == 'test' and res.rc == 1:
                    u.note('content_test.py exit 1 (gating failures) - judged against the disclosure record by the class test')
                    continue
                if not res.ok:
                    u.refuse(f'FAILED: {script} {" ".join(argv)} ({res.reason}; run_all.sh stops at the first failure)')
        u.add_output(os.path.join(occ, 'validation', 'content_test_results.json'), 'jupiter_occurrence')
    print(f'jupiter occurrence: {u.status}' + (f' ({u.reason})' if u.reason else ''), file=sys.stderr)
    return None
