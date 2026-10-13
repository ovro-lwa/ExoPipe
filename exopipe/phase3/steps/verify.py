"""exopipe3 verify - run the content test of any product class, alone.

    exopipe3 verify master /path/to/I_deep_Taper_R075_Full.fits
    exopipe3 verify search_frame --glob '/lustre/.../04h/*/Run_*/Full/V/deep/*msub_ssub*.fits' --emit-failing bad.txt
    exopipe3 verify --list

Inputs:  product files of one class (paths, @file lists or --glob).
Outputs: a table of test results on stdout; with --emit-failing FILE, the failing paths (usable as an item list).
Failure modes: unknown class (exit 1); any file failing a test (exit 2, the test name and value printed).
Writes no products and no records (WRITES = False).
"""
import glob as _glob
import sys

SUMMARY = 'run the content test of any product class alone'
TIERS = None
SELECTORS = ()
WRITES = False
CONFIG = ()               # reads no step section (config.SHARED_SECTIONS only)


def add_arguments(p):
    p.add_argument('cls', nargs='?', help='product class (see --list)')
    p.add_argument('paths', nargs='*', help='files, or @FILE with one path per line')
    p.add_argument('--glob', action='append', default=[], help='shell glob of files (quote it)')
    p.add_argument('--list', action='store_true', help='list the product classes')
    p.add_argument('--emit-failing', metavar='FILE', help='write failing paths to FILE')


def default_out(args):
    return '/tmp'


def run(args, run):
    from exopipe.phase3.core import validate
    if args.list or not args.cls:
        for c in validate.classes():
            print(f'{c:20s} {validate.OWNERS.get(c, "")}')
        return 0
    paths = []
    for p in args.paths:
        if p.startswith('@'):
            paths += [l.strip() for l in open(p[1:]) if l.strip() and not l.startswith('#')]
        else:
            paths.append(p)
    for g in args.glob:
        paths += sorted(_glob.glob(g))
    if not paths:
        print('no files given', file=sys.stderr)
        return 3
    failing = []
    for p in paths:
        res = validate.run(args.cls, p)
        bad = [r for r in res if not r['passed']]
        for r in res:
            print(f"{'PASS' if r['passed'] else 'FAIL'}  {r['name']:40s} {r['value']!s:>14}  {p}  {r['detail']}")
        if bad:
            failing.append(p)
    if args.emit_failing:
        with open(args.emit_failing, 'w') as f:
            f.write(''.join(x + '\n' for x in failing))
    print(f'{len(paths) - len(failing)} of {len(paths)} files pass every {args.cls} test', file=sys.stderr)
    return 2 if failing else 0
