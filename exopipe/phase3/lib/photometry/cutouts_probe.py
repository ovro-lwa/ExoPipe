#!/usr/bin/env python
"""The 10-s cutout engines' own reads, taken BEFORE an engine starts (steps_cutouts.engine_probe): their exclusion list and the
writer's target list.

Run as a child process under the engines' interpreter, with the engines' environment (an ad hoc target set included), this
imports the production modules of --code-root (the directory of the engine the unit would run, as production runs it:
`cd $OPS`) and prints two lines; it writes nothing:
  'ENGINE-EXCLUSIONS ' + JSON {path, module, excluded}: the file the engines read (phase3_snapshot_search.DATA_DECISIONS) and
      excluded_subbands(lst, date), as written - the reader the regeneration engines keep (spec photometry.md B-38 decision c):
      it refuses a subband in phase3_snapshot_search.main (line 'EXCLUDED:', exit 3) and drops the excluded member subbands of
      a wideband in the group search (regen_group_cutouts). A missing or unreadable file fails here (exit 1, the traceback):
      the step refuses the unit (PHOT-EXCL-FILE, ruling C-09: both readers);
  'ENGINE-TARGETS ' + JSON {module, n_targets, multiple}: the writer's fixed targets (phase3_snapshot_cutouts._targets, the
      lists or the ad hoc set) and, per '<sample>/<file object name>' that more than one entry shares, their number. Entries of
      one name within MERGE_ARCMIN are one object, one file, but write_cutouts counts each (its 'cubes written', the count in
      the engines' done lines): the step holds the staged files to the count through these multiplicities (check_counts)."""
import argparse
import json
import os
import sys

EXCLUSIONS = 'ENGINE-EXCLUSIONS '
TARGETS = 'ENGINE-TARGETS '


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--code-root', required=True)
    ap.add_argument('--lst', required=True)
    ap.add_argument('--date', required=True)
    a = ap.parse_args(argv)
    # the production modules of --code-root, never this package's directory (as stamp_engine.py)
    here = os.path.dirname(os.path.realpath(__file__))
    sys.path[:] = [a.code_root] + [p for p in sys.path if os.path.realpath(p or '.') != here]
    os.chdir(a.code_root)
    import phase3_snapshot_search as SS
    out = sorted(SS.excluded_subbands(a.lst, a.date))
    print(EXCLUSIONS + json.dumps({'path': SS.DATA_DECISIONS, 'module': os.path.abspath(SS.__file__), 'excluded': out},
                                  sort_keys=True), flush=True)
    import phase3_snapshot_cutouts as CUT
    n = {}
    targets = CUT._targets()
    for tag, nm, _, _ in targets:
        k = '%s/%s' % (tag, CUT.safe_name(nm))
        n[k] = n.get(k, 0) + 1
    print(TARGETS + json.dumps({'module': os.path.abspath(CUT.__file__), 'n_targets': len(targets),
                                'multiple': {k: v for k, v in sorted(n.items()) if v > 1}}, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
