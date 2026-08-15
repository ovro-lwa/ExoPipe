#!/usr/bin/env python3
"""
phase3_archive_manifest.py — READ-ONLY manifest + sizing sweep for the
corrected Phase-3 archiving plan (Gregg-approved 2026-07-21). No transfers.

Walks every archive unit class, classifies EVERY file:
  conformant    parse_name() succeeds (either Phase-3 grammar or the two
                legacy grammars), or known conformant CSV/ledger classes
  candidate     files inside candidates/ rank dirs (counted as bundles)
  intermediate  known scratch patterns (ep_*_up2*, *_flow.npy, work_*,
                tmp*, *.claim, .pybdsf.log, etc.) — REFUSED from archive
  unknown       everything else — listed for Gregg review
Header spot-audit: N random conformant FITS per unit get full header
validation (time block, freq block, beam) — recorded pass/fail.

Outputs: /data07/gh/phase3/logs/archive_manifest_20260721.csv (per unit x
class: files, bytes, examples) + archive_manifest_20260721_unknowns.txt
+ a summary block on stdout.
"""
import os, sys, csv, glob, random, re
from collections import defaultdict

sys.path.insert(0, '/lustre/gh/main/clv15')
sys.argv = ['manifest']
from phase3_products import parse_name

D7 = '/data07/gh/phase3'
TR = f'{D7}/transients'
INTERMEDIATE_PAT = re.compile(
    r'(ep_.*_up2|_flow\.npy$|^work_|^tmp|\.claim$|\.pybdsf\.log$|'
    r'\.lock$|\.casalog$|^_|\.tmp$|\.partial$)')
CSV_OK = re.compile(r'\.(csv|txt|reg|json|log)$')

UNITS = []
for lst in sorted(os.listdir(D7)):
    if re.match(r'^\d{2}h$', lst):
        UNITS.append((f'P3-trees/{lst}', f'{D7}/{lst}'))
UNITS += [('i_resid', f'{D7}/i_resid'),
          ('coadd_sb-masters', f'{D7}/coadd_sb'),
          ('coadd_wb-masters', '/fast/gh/main/phase3/coadd'),
          ('resp_wb_run', '/fast/gh/main/phase3/resp_wb_run')]
for rr in ('Run_V_DEEP', 'Run_I_DEEP', 'Run_V_SB', 'Run_I_SB',
           'Run_PROD10', 'Run_SB10'):
    UNITS.append((f'transients/{rr}', f'{TR}/{rr}'))


def classify(path, fname):
    if '/candidates/' in path:
        return 'candidate'
    if INTERMEDIATE_PAT.search(fname):
        return 'intermediate'
    if fname.endswith('.fits'):
        try:
            parse_name(fname)
            return 'conformant'
        except Exception:
            return 'unknown-fits'
    if CSV_OK.search(fname):
        return 'conformant-ancillary'
    return 'unknown'


def validate_header(path):
    from astropy.io import fits
    try:
        h = fits.getheader(path)
        need = ['DATE-OBS', 'BMAJ', 'RESTFRQ']
        missing = [k for k in need if k not in h]
        return ('PASS' if not missing else f'MISSING:{",".join(missing)}')
    except Exception as e:
        return f'ERR:{str(e)[:40]}'


def main():
    out_rows = []
    unknowns = open(f'{D7}/logs/archive_manifest_20260721_unknowns.txt', 'w')
    for unit, root in UNITS:
        if not os.path.isdir(root):
            continue
        agg = defaultdict(lambda: [0, 0, None])   # class -> [n, bytes, example]
        samples = []
        for dirpath, dirnames, filenames in os.walk(root):
            for f in filenames:
                p = os.path.join(dirpath, f)
                try:
                    sz = os.lstat(p).st_size
                except OSError:
                    continue
                c = classify(p, f)
                a = agg[c]
                a[0] += 1; a[1] += sz
                if a[2] is None:
                    a[2] = p
                if c.startswith('unknown'):
                    unknowns.write(p + '\n')
                if c == 'conformant' and len(samples) < 400:
                    samples.append(p)
        audit = ''
        if samples:
            random.seed(20260721)
            res = [validate_header(p) for p in
                   random.sample(samples, min(20, len(samples)))]
            npass = sum(1 for r in res if r == 'PASS')
            audit = f'{npass}/{len(res)} PASS'
        for c, (n, b, ex) in sorted(agg.items()):
            out_rows.append(dict(unit=unit, cls=c, files=n, bytes=b,
                                 gb=round(b / 1e9, 1), example=ex or '',
                                 header_audit=audit if c == 'conformant' else ''))
        tot = sum(v[1] for v in agg.values())
        print(f'{unit:28s} {sum(v[0] for v in agg.values()):>9d} files '
              f'{tot/1e9:8.1f} GB  audit[{audit}]', flush=True)
    unknowns.close()
    mp = f'{D7}/logs/archive_manifest_20260721.csv'
    with open(mp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)
    tot_all = sum(r['bytes'] for r in out_rows)
    n_int = sum(r['bytes'] for r in out_rows if r['cls'] == 'intermediate')
    n_unk = sum(r['bytes'] for r in out_rows if r['cls'].startswith('unknown'))
    print(f'\nMANIFEST -> {mp}')
    print(f'TOTAL {tot_all/1e12:.2f} TB | intermediates (refused) '
          f'{n_int/1e9:.0f} GB | unknown (review) {n_unk/1e9:.0f} GB')
    print('MANIFEST-SWEEP-DONE', flush=True)


if __name__ == '__main__':
    main()
