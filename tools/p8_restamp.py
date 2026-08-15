#!/usr/bin/env python3
"""P8 restamp tool (R-08, plan: tools/P8_RESTAMP_PLAN.md).

DRY-RUN FIRST: `python tools/p8_restamp.py --dry-run` writes
tools/p8_restamp_ledger.csv with zero FITS writes. The write pass
(`--apply`) is REFUSED unless the dry-run ledger exists and Gregg's go
marker tools/.p8_apply_approved is present (Gregg creates it).

Scope: /data07/gh/phase3/{lst}/{date}/P3_PROD/{sb}/{I,V}/deep/*_msub_*.fits
+ their *_pbcorr_* siblings. Lustre copies handled in a second pass after
/data07 verifies (--root lustre).

Time-source hierarchy per epoch (P8_RESTAMP_PLAN.md):
  1. P3_PROD pre-msub *_aligned_* sibling header — TRUSTED only if its
     MJD-MID differs from the (lst,sb) master's (else it carries P8 too).
  2. P2 raw deep filename midpoint (verified trustworthy 2026-08-06):
     DATE-OBS = midpoint - span/2; span/INTTIME from the P2 header.
  3. Neither -> row in ledger with action=UNRESOLVED, never stamped.
R-01(b) folded in: proc-token repair (PROC list, SSUB/ALIGNED keys) in the
same pass, recorded per row.
"""
import os, re, csv, glob, sys, argparse

D7 = '/data07/gh/phase3'
LU = '/lustre/pipeline/exopipe/phase3'
P2 = '/lustre/pipeline/exopipe/phase2'
SB_RE = re.compile(r'^\d{2}MHz$')
TS_RE = re.compile(r'-(\d{8}_\d{6})-image')  # P2 filename midpoint, e.g. 20241218_033402
LEDGER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      'p8_restamp_ledger.csv')

def master_midpoints(root):
    """(lst, sb) -> master MJD-MID from coadd_sb masters."""
    from astropy.io import fits
    out = {}
    for mdir in sorted(glob.glob(f'{root}/coadd_sb/*_*MHz/coadd_final')):
        cell = os.path.basename(os.path.dirname(mdir))
        lst, sb = cell.split('_', 1)
        for f in sorted(glob.glob(f'{mdir}/*_Taper_R0_*.fits'))[:1]:
            try:
                h = fits.getheader(f)
                out[(lst, sb)] = h.get('MJD-MID')
            except Exception:
                pass
    return out

def p2_deep_time(lst, date, sb):
    """(date_obs_isot, date_end_isot, inttime_s, basis) from the P2 raw deep
    dewarped filename midpoint + header span; None on miss."""
    from astropy.io import fits
    from astropy.time import Time
    import astropy.units as u
    hits = glob.glob(f'{P2}/{lst}/{date}/Science_*/{sb}/I/deep/'
                     f'*image_dewarped.fits')
    if not hits:
        # dewarp-failed epochs (searched via the deep raw-failsafe) keep raw
        # products; the filename midpoint is dewarp-independent and raw
        # filenames are the plan's stated basis.
        hits = [h for h in glob.glob(f'{P2}/{lst}/{date}/Science_*/{sb}/I/'
                                     f'deep/*-image.fits')
                if 'pbcorr' not in h]
    if not hits:
        return None
    # prefer the longest-duration (deep) file: parse DURATIONs field '3591s'
    def dur_of(fn):
        m = re.search(r'-(\d+)s-', os.path.basename(fn))
        return int(m.group(1)) if m else 0
    f = max(hits, key=dur_of)
    m = TS_RE.search(os.path.basename(f))
    if not m:
        return None
    t = m.group(1)  # YYYYMMDD_HHMMSS = interval MIDPOINT (Phase-1 convention)
    ts = (f'{t[0:4]}-{t[4:6]}-{t[6:8]}T{t[9:11]}:{t[11:13]}:{t[13:15]}')
    span = dur_of(f)
    try:
        h = fits.getheader(f)
        span = float(h.get('DURATION', span)) or span
        itime = float(h.get('INTTIME', span)) or span
    except Exception:
        itime = span
    mid = Time(ts, format='isot', scale='utc')
    t0 = mid - (span / 2.0) * u.s
    t1 = mid + (span / 2.0) * u.s
    return t0.isot, t1.isot, itime, f'p2-filename-mid({os.path.basename(f)})'

def apply_pass(root_key):
    """Write pass. Reads the dry-run ledger; stamps only RESTAMP*/PROC-ONLY
    rows. Idempotent via P8FIX header key. root='data07' stamps the local
    msub files; root='lustre' stamps the Lustre msub copies + the (Lustre-
    only) pbcorr siblings — run ONLY after the data07 pass verifies."""
    import csv as _csv
    from astropy.io import fits
    from astropy.time import Time
    rows = list(_csv.DictReader(open(LEDGER)))
    out, n_done, n_skip, n_refuse = [], 0, 0, 0
    for r in rows:
        if not r['action'].startswith(('RESTAMP', 'PROC')):
            continue
        targets = []
        if root_key == 'data07':
            targets = [r['file']]
        else:
            targets = [r['file'].replace(D7, LU)]
            targets += [s for s in r.get('pbcorr_siblings', '').split(';')
                        if s.startswith(LU)]
        for f in targets:
            if not os.path.exists(f):
                out.append((f, 'MISSING')); continue
            try:
                with fits.open(f, mode='update') as hdul:
                    h = hdul[0].header
                    if h.get('P8FIX'):
                        n_skip += 1; out.append((f, 'SKIP-P8FIX')); continue
                    if r['action'].startswith('RESTAMP'):
                        t0 = Time(r['new_date_obs'], format='isot',
                                  scale='utc')
                        t1 = Time(r['new_date_end'], format='isot',
                                  scale='utc')
                        new_mid = (t0.mjd + t1.mjd) / 2.0
                        mm = r.get('master_mid')
                        if mm and abs(new_mid - float(mm)) < 1e-9:
                            n_refuse += 1
                            out.append((f, 'REFUSE-STILL-MASTER-MID'))
                            continue
                        h['DATE-OBS'] = (t0.isot,
                                         'Start of first integration (UTC)')
                        h['MJD-OBS'] = (t0.mjd, 'Start MJD (UTC)')
                        h['DATE-END'] = (t1.isot,
                                         'End of last integration (UTC)')
                        h['DURATION'] = (float((t1 - t0).sec),
                                         '[s] Wall-clock span')
                        h['INTTIME'] = (float(r['new_inttime']),
                                        '[s] Summed integration time')
                        h['MJD-MID'] = (new_mid, 'Midpoint MJD (UTC)')
                        h['TIMESYS'] = ('UTC', 'Time system')
                        h['TIMEFIX'] = ('v2-p8restamp-p2mid',
                                        'Time correction version-basis')
                        h['NSTACK'] = (1, 'Single-epoch product (P8 fix)')
                    # R-01(b) proc-token repair
                    h['ALIGNED'] = (True, 'Flow-aligned to master (R-01b)')
                    h['SSUB'] = (True, 'Striation-subtracted (R-01b)')
                    h['P8FIX'] = ('20260811', 'P8 restamp applied')
                    h.add_history(
                        f'P8 restamp 2026-08-11: {r["action"]} basis='
                        f'{r.get("basis", "proc-only")[:50]}')
                n_done += 1; out.append((f, 'DONE'))
            except Exception as e:
                out.append((f, f'ERROR:{str(e)[:50]}'))
    rep = LEDGER.replace('.csv', f'_applied_{root_key}.csv')
    with open(rep, 'w', newline='') as fh:
        w = _csv.writer(fh); w.writerow(['file', 'result'])
        w.writerows(out)
    print(f'APPLY[{root_key}]: done={n_done} skip={n_skip} '
          f'refused={n_refuse} report={rep}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--root', default='data07', choices=['data07', 'lustre'])
    a = ap.parse_args()
    if a.apply:
        ok = os.path.exists(LEDGER) and os.path.exists(
            os.path.join(os.path.dirname(LEDGER), '.p8_apply_approved'))
        if not ok:
            sys.exit('REFUSED: --apply requires the dry-run ledger AND '
                     'tools/.p8_apply_approved (Gregg creates it after '
                     'reviewing the ledger).')
        return apply_pass(a.root)
    from astropy.io import fits
    root = D7 if a.root == 'data07' else LU
    masters = master_midpoints(D7)
    rows, n_need, n_ok, n_unres = [], 0, 0, 0
    lsts = sorted(d for d in os.listdir(root) if re.match(r'^\d{2}h$', d))
    for lst in lsts:
        for date in sorted(os.listdir(f'{root}/{lst}')):
            base = f'{root}/{lst}/{date}/P3_PROD'
            if not os.path.isdir(base):
                continue
            for sb in sorted(os.listdir(base)):
                if not SB_RE.match(sb):
                    continue
                for st in ('I', 'V'):
                    dd = f'{base}/{sb}/{st}/deep'
                    if not os.path.isdir(dd):
                        continue
                    for f in sorted(glob.glob(f'{dd}/*_msub_*.fits')):
                        try:
                            h = fits.getheader(f)
                        except Exception as e:
                            rows.append(dict(file=f, action='READ-ERROR',
                                             note=str(e)[:60])); continue
                        mid = h.get('MJD-MID')
                        mmid = masters.get((lst, sb))
                        p8 = (mid is not None and mmid is not None
                              and abs(mid - mmid) < 1e-9)
                        # proc-token under-labeling (R-01b) check
                        proc = str(h.get('PROC', ''))
                        need_proc = ('ssub' not in proc.lower()
                                     or 'aligned' not in proc.lower())
                        if not p8 and not need_proc:
                            n_ok += 1
                            rows.append(dict(file=f, action='OK',
                                mjd_mid=mid, master_mid=mmid)); continue
                        t = p2_deep_time(lst, date, sb) if p8 else None
                        if p8 and t is None:
                            n_unres += 1
                            rows.append(dict(file=f, action='UNRESOLVED',
                                mjd_mid=mid, master_mid=mmid,
                                note='no P2 source found')); continue
                        n_need += 1
                        r = dict(file=f,
                                 action=('RESTAMP+PROC' if p8 and need_proc
                                         else 'RESTAMP' if p8 else 'PROC-ONLY'),
                                 mjd_mid=mid, master_mid=mmid)
                        if t:
                            r.update(new_date_obs=t[0], new_date_end=t[1],
                                     new_inttime=t[2], basis=t[3])
                        # pbcorr siblings live on Lustre with the
                        # _pbcorr_dewarped_msub_ token (materialized pass)
                        cand = [f.replace('_dewarped_msub_',
                                          '_pbcorr_dewarped_msub_'),
                                f.replace(D7, LU).replace(
                                    '_dewarped_msub_',
                                    '_pbcorr_dewarped_msub_')]
                        r['pbcorr_siblings'] = ';'.join(
                            x for x in dict.fromkeys(cand)
                            if x != f and os.path.exists(x))
                        rows.append(r)
    keys = ['file', 'action', 'mjd_mid', 'master_mid', 'new_date_obs',
            'new_date_end', 'new_inttime', 'basis', 'pbcorr_siblings', 'note']
    with open(LEDGER, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'DRY-RUN complete: {len(rows)} files scanned | '
          f'restamp/proc needed: {n_need} | already-ok: {n_ok} | '
          f'unresolved: {n_unres}')
    print(f'ledger: {LEDGER}')

if __name__ == '__main__':
    main()
