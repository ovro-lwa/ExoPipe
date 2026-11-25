#!/usr/bin/env python3
"""CI pin of the dewarp content-test child's modules (gate G-D0b, lib/dewarp/steps_common.py). Data-free: reads only files
of this repository.

G-D0b runs on a node: it requires each production module the content-test child imports (CHILD_MODULES, production side) to
be exactly the production-equivalent text of its branch module (steps_common.child_production_text: the
CHILD_PERMITTED_LINE_DIFFS lines rewritten, the B-30 site lines removed). CI has no production repository, so it checks the
branch side against a pin instead, by the same function:
  1. every CHILD_MODULES branch file has a pin (CHILD_PINS), and no pin names another file;
  2. the sha256 of each branch file's production-equivalent text is its pin;
  3. the rule is live: an appended comment changes the hash; a B-30 site insertion and the permitted line keep it.
tests/test_dewarp.py (node) checks that each pin is the sha256 of production's module.

Run standalone:  python tests/test_dewarp_child_pins.py
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from exopipe.phase3.lib.dewarp import steps_common as SC  # noqa: E402

FIX = ('This module is pinned to production {prod} by dewarp gate G-D0b. Re-run G-D0b against production on a node '
       '(tests/test_dewarp.py) and, if it passes, set CHILD_PINS[{br!r}] = {got!r} in exopipe/phase3/lib/dewarp/steps_common.py '
       'in the same pull request.')


def _text(br_rel):
    with open(os.path.join(REPO, br_rel)) as f:
        return f.read()


def test_every_module_pinned():
    have, want = set(SC.CHILD_PINS), {br for _, br in SC.CHILD_MODULES}
    assert have == want, f'CHILD_PINS names {sorted(have - want)} not in CHILD_MODULES; unpinned {sorted(want - have)}'


def test_pins():
    bad = []
    for prod_rel, br_rel in SC.CHILD_MODULES:
        got = SC.child_text_sha256(SC.child_production_text(br_rel, _text(br_rel)))
        if got != SC.CHILD_PINS.get(br_rel):
            bad.append(f'{br_rel}: the sha256 of its production-equivalent text is {got}, not its pin '
                       f'{SC.CHILD_PINS.get(br_rel)}. ' + FIX.format(prod=prod_rel, br=br_rel, got=got))
    assert not bad, '\n'.join(bad)


def test_rule_is_live():
    br = 'exopipe/phase3/lib/masters/pb_apply.py'
    t = _text(br)
    pin = SC.child_text_sha256(SC.child_production_text(br, t))
    assert SC.child_text_sha256(SC.child_production_text(br, t + '# an edit\n')) != pin, 'an appended comment kept the hash'
    have, want = SC.CHILD_PERMITTED_LINE_DIFFS[br][0]
    assert have + '\n' in t and SC.child_production_text(br, t).count(want + '\n') == 1, 'the permitted line is not rewritten'
    site = ("    from exopipe.phase3.core import fallback      # WAVE2 B-30\n"
            "    fallback.site('fallbacks.PB_DEEP_MAP_FALLBACK', 'test site')\n")
    anchor = 'def subband_bin_pb(subband, k, pb_dir=PB_DIR):\n'
    assert t.count(anchor) == 1, anchor
    t2 = t.replace(anchor, anchor + site, 1)
    assert SC.child_text_sha256(SC.child_production_text(br, t2)) == pin, 'a B-30 site insertion changed the hash'


def main():
    fails = 0
    for name in ('test_every_module_pinned', 'test_pins', 'test_rule_is_live'):
        try:
            globals()[name]()
            print(f'PASS {name}')
        except AssertionError as e:
            fails += 1
            print(f'FAIL {name}: {e}')
    print(f'{3 - fails} of 3 passed')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
