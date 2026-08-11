"""
exopipe.phase3.assemble — seed module (refactor step 2, extracted 2026-08-11).

Origin: /lustre/gh/main/clv15/phase3_i_deep_live.py (VERBATIM extraction of
load_rows, epoch_mid, FP_CLASS, ATEAM and the ROOT/COADD/REPEAT_DEG constants,
plus internal deps LEDGERS and the _HDR cache that they reference).
The remaining assembly-capability content arrives at move time.
"""
import sys; sys.path.insert(0, '/lustre/gh/main/clv15')
import os, csv, glob
from astropy.io import fits
from astropy.time import Time
import astropy.units as u
import search_i_transients as sit

# TO-MOVE-TO-COMMON-CONFIG: hardcoded path constants (ROOT, LEDGERS, COADD)
# must migrate to the shared exopipe config module at move time.
ROOT = '/data07/gh/phase3/transients/Run_I_DEEP'

LEDGERS = ([f'{ROOT}/i_deep_ledger.csv']
           + sorted(glob.glob(f'{ROOT}/nodes/*/i_deep_ledger.csv')))
COADD = '/fast/gh/main/phase3/coadd'          # 07 holds ALL cells (stage source)
FP_CLASS = {'Blue': (2, 4), 'Green': (15, 30), 'Red': (47, 96),
            'Full': (65, 291)}                # pack §3 matched-funnel terciles
REPEAT_DEG = 0.5


ATEAM = [(n, i['coord']) for n, i in sit.CONFIG['A_TEAM'].items()]



def load_rows():
    rows, seen = [], set()
    for p in LEDGERS:
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p)):
            k = (r['lst'], r['date'], r['band'], r['pix_x'], r['pix_y'],
                 r['sign'])
            if k in seen:
                continue
            seen.add(k)
            rows.append(r)
    return rows



_HDR = {}



def epoch_mid(lst, band, date):
    key = (lst, band, date)
    if key not in _HDR:
        try:
            eh = fits.getheader(
                f'{COADD}/{lst}_{band}/subtracted/{date}_I_sub_{band}.fits')
            _HDR[key] = (Time(eh['DATE-OBS'])
                         + float(eh.get('INTTIME') or 3591.0) / 2 * u.s,
                         str(eh.get('DATE-OBS')))
        except Exception:
            _HDR[key] = (None, '')
    return _HDR[key]

