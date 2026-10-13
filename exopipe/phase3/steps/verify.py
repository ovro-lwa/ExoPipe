"""exopipe3 verify - run the content test of any product class, alone; or compare two product roots (parity).

    exopipe3 verify master /path/to/I_deep_Taper_R075_Full.fits
    exopipe3 verify search_frame --glob '/lustre/.../04h/*/Run_*/Full/V/deep/*msub_ssub*.fits' --emit-failing bad.txt
    exopipe3 verify --list
    exopipe3 verify compare ROOT_A ROOT_B --class fits [--permitted-diffs FILE] [--map FILE] [--report DIR]

Inputs:  product files of one class (paths, @file lists or --glob).
         compare: two roots (old side, new side); files paired by relative path or by a product map (CSV old,new);
         --class fits|csv|json|auto (auto = all three by extension; a product class name means auto).
Outputs: a table of test results on stdout; with --emit-failing FILE, the failing paths (usable as an item list).
         compare: parity_report.csv (one row per finding), parity_summary.json/.txt in --report DIR (default
         /tmp/exopipe3_compare_<UTC>), the summary on stdout. FITS: data md5, NaN mask, max|A-B| and max|A-B|/rms(A),
         header card by card (DATE, CHECKSUM, DATASUM, COMMENT ignored; HISTORY timestamps, run ids and roots
         normalised; GITTAG/ORIGIN/SCRIPT/PIPEVER permitted as PD-GITTAG). CSV: columns, row count, rows by --key,
         cells exact or within --rtol/--atol. JSON: deep equality ignoring run ids, timestamps, hosts, commits.
         Permitted differences: a CSV of signatures id,ledger_id,class,path,kind,key,tolerance,note
         (core/compare.py documents the kinds).
Failure modes: unknown class (exit 1); any file failing a test (exit 2, the test name and value printed).
         compare: exit 0 only when every difference is permitted; 2 when any is not (missing and extra files count);
         3 when there was nothing to compare; 1 on an unreadable permitted-differences or map file.
Writes no products and no records (WRITES = False).
"""
import glob as _glob
import os
import sys
import time

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
    g = p.add_argument_group('verify compare ROOT_A ROOT_B (the parity comparator, core/compare.py)')
    g.add_argument('--class', dest='compare_class', default='auto',
                   help='fits, csv, json or auto (by extension; default); a product class name compares by extension')
    g.add_argument('--permitted-diffs', metavar='FILE', help='CSV of permitted-difference signatures')
    g.add_argument('--map', metavar='FILE', help='product map: CSV rows old,new (paths relative to the roots)')
    g.add_argument('--report', metavar='DIR', help='report directory (default /tmp/exopipe3_compare_<UTC>)')
    g.add_argument('--key', default='', help='CSV key columns, comma-separated (rows paired by key)')
    g.add_argument('--rtol', type=float, default=0.0, help='CSV/JSON numeric relative tolerance (default exact)')
    g.add_argument('--atol', type=float, default=0.0, help='CSV/JSON numeric absolute tolerance (default exact)')
    g.add_argument('--fits-atol', type=float, default=0.0, help='FITS max|A-B| treated as equal (default 0)')
    g.add_argument('--fits-rel-rms', type=float, default=0.0, help='FITS max|A-B|/rms(A) treated as equal (default 0)')
    g.add_argument('--ignore-card', action='append', default=[], help='FITS header card to ignore (repeatable)')
    g.add_argument('--ignore-key', action='append', default=[], help='JSON key or CSV column to ignore (repeatable)')
    g.add_argument('--include-records', action='store_true', help='also compare files under _records/')


def default_out(args):
    return '/tmp'


def compare(args):
    from exopipe.phase3.core import compare as cmp, validate
    if len(args.paths) != 2:
        print('verify compare needs exactly two roots: ROOT_A ROOT_B', file=sys.stderr)
        return 1
    cls = args.compare_class
    if cls not in ('auto',) + tuple(cmp.KIND_SUFFIXES):
        if cls not in validate.classes():
            print(f'unknown --class {cls!r}: fits, csv, json, auto or a product class', file=sys.stderr)
            return 1
        cls = 'auto'
    opts = cmp.Options(csv_keys=[k for k in args.key.split(',') if k], rtol=args.rtol, atol=args.atol,
                       fits_atol=args.fits_atol, fits_rel_rms=args.fits_rel_rms, ignore_cards=args.ignore_card,
                       ignore_keys=args.ignore_key, include_records=args.include_records)
    try:
        rep = cmp.compare_roots(args.paths[0], args.paths[1], cls, permitted=args.permitted_diffs, mapping=args.map,
                                opts=opts)
    except (OSError, ValueError) as e:
        print(f'error: {e}', file=sys.stderr)
        return 1
    rep.cls = args.compare_class
    outdir = args.report or os.path.join('/tmp', time.strftime('exopipe3_compare_%Y%m%dT%H%M%SZ', time.gmtime())
                                         + f'_{os.getpid()}')
    _, lines = cmp.write_report(rep, outdir)
    for r in rep.rows:
        if r['level'] == 'DIFF':
            print(f"DIFF  {r['kind']:9s} {r['a'] or '-'} | {r['b'] or '-'}  {r['key']}  {r['detail']}")
    print('\n'.join(lines))
    return {'PASS': 0, 'FAIL': 2, 'NOTHING': 3}[rep.verdict]


def run(args, run):
    from exopipe.phase3.core import validate
    if args.cls == 'compare':
        return compare(args)
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
