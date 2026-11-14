#!/usr/bin/env python3
"""Run production's ops/sublst_pages_prep.py IN PLACE with ONE of its lines changed: the catalog directory it reads.

Production hardcodes the catalog directory of an LST hour (sublst_pages_prep.py:31,
`CAT = f'{SL}/compare_{LST}/new_{ARM}/catalog'`). The pages of a night (design 2.3a of PER_HOUR_GAP_DESIGNS_20261111.md,
stage 3) are prepared from the hour's catalog VIEW of the night pass (lib/catalog/delta_build.view_root) instead, through a
staging directory (lib/pages/night.stage_catalog). This runner reads the script, replaces the text of that one assignment by
the directory named in EXOPIPE3_PAGES_CATALOG, and executes the result as __main__ (its file name, its directory first on
sys.path, its argv), so every other line of production's script runs as written: what it copies, what it refuses, the
leave-one-out window Masters it writes.

PINS: the script's PATH-NORMALIZED sha1 (a frozen tree's own root replaced by /lustre/gh/main, as
lib/catalog/wraps/builder_grouping.py) must be one of EXOPIPE3_PAGES_PREP_SHA1 (config pages_sublst.PREP_SHA1), and the
patched text must occur exactly once; any other script REFUSES (exit 2) - re-port this file.

usage: prep_catalog.py <sublst_pages_prep.py> ARM OUT    (env EXOPIPE3_PAGES_CATALOG, EXOPIPE3_PAGES_TIE, EXOPIPE3_PAGES_PREP_SHA1, SUBLST_LST)
"""
import hashlib
import os
import sys
import types

LIVE_ROOT = '/lustre/gh/main'               # the live code root a frozen tree's rewrites point away from
OLD = "CAT = f'{SL}/compare_{LST}/new_{ARM}/catalog'; OLD = "
NEW = "CAT = os.environ['EXOPIPE3_PAGES_CATALOG']; OLD = "     # EXOPIPE3 NIGHT: the hour's staged catalog view
# the reference-tie table that travels with the pages (DECISION-20261114-PR50-VIEW-CARRIES-ITS-REFERENCE-TIE-TABLE): the one
# of the catalog read, never production's current one; scripts before 2026-11-13 copy none (the line is absent: 0 or 1 times)
OLD_TIE = "_RT = f'{SL}/compare_{LST}/new_{ARM}/tie/tie_members.csv'"
NEW_TIE = "_RT = os.environ['EXOPIPE3_PAGES_TIE']"                  # EXOPIPE3 NIGHT: the catalog's own tie table


class Refused(RuntimeError):
    pass


def normalized_sha1(script, src):
    """sha1 of the script's source with its own code root (a frozen tree's <root>/) replaced by the live root"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(script)))
    if root != LIVE_ROOT:
        src = src.replace(root + '/', LIVE_ROOT + '/')
    return hashlib.sha1(src.encode()).hexdigest()


def patched(script, accepted):
    """(patched source, sha1, path-normalized sha1) - REFUSES for a script that is not pinned or does not hold the line."""
    with open(script) as f:
        src = f.read()
    sha1, norm = hashlib.sha1(src.encode()).hexdigest(), normalized_sha1(script, src)
    accepted = [a for a in accepted if a]
    if not accepted:
        raise Refused('EXOPIPE3_PAGES_PREP_SHA1 is empty: the accepted scripts must be named (config pages_sublst.PREP_SHA1) - '
                      'REFUSING')
    if norm not in accepted:
        raise Refused(f'the script {script} (path-normalized sha1 {norm}) is not one the catalog-directory patch was written '
                      f'and checked for ({len(accepted)} accepted, config pages_sublst.PREP_SHA1) - re-port '
                      f'lib/pages/wraps/prep_catalog.py - REFUSING')
    n = src.count(OLD)
    if n != 1:
        raise Refused(f'the script {script} holds the catalog-directory assignment {n} times (1 expected) - re-port '
                      f'lib/pages/wraps/prep_catalog.py - REFUSING')
    nt = src.count(OLD_TIE)
    if nt > 1:
        raise Refused(f'the script {script} holds the tie-table path {nt} times (at most 1 expected) - re-port '
                      f'lib/pages/wraps/prep_catalog.py - REFUSING')
    return src.replace(OLD, NEW).replace(OLD_TIE, NEW_TIE), sha1, norm


def run(script, args, catalog, accepted, tie=None):
    script = os.path.abspath(script)
    if not catalog or not os.path.isdir(catalog):
        raise Refused(f'EXOPIPE3_PAGES_CATALOG={catalog!r} is not a directory - REFUSING')
    tie = tie if tie is not None else os.environ.get('EXOPIPE3_PAGES_TIE', '')
    if not tie:
        raise Refused('EXOPIPE3_PAGES_TIE is not set: the tie table of the catalog read must be named (absent is allowed, '
                      'unnamed is not) - REFUSING')
    os.environ['EXOPIPE3_PAGES_TIE'] = tie
    src, sha1, _norm = patched(script, accepted)
    mod = types.ModuleType('__main__')
    mod.__file__ = script
    mod.__dict__['__builtins__'] = __builtins__
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [os.path.dirname(script)] + [p for p in sys.path if os.path.abspath(p or '.') != here]
    sys.argv = [script] + list(args)
    sys.modules['__main__'] = mod
    print(f'PAGES-PREP start: script {script} sha1 {sha1}; catalog directory {catalog} (in place of the hour\'s all-nights '
          f'catalog directory); tie table {tie}' + ('' if os.path.exists(tie) else ' (absent: the blocks say the tie has not run)'),
          flush=True)
    exec(compile(src, script, 'exec'), mod.__dict__)
    print(f'PAGES-PREP done: script sha1 {sha1}', flush=True)


def main(argv):
    if len(argv) != 4:
        sys.exit('usage: prep_catalog.py <sublst_pages_prep.py> ARM OUT  (env EXOPIPE3_PAGES_CATALOG, EXOPIPE3_PAGES_PREP_SHA1)')
    try:
        run(argv[1], argv[2:], os.environ.get('EXOPIPE3_PAGES_CATALOG', ''),
            os.environ.get('EXOPIPE3_PAGES_PREP_SHA1', '').split(','))
    except Refused as e:
        print(f'PAGES-PREP REFUSING: {e}', flush=True)
        sys.exit(2)


if __name__ == '__main__':
    main(sys.argv)
