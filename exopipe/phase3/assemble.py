"""
exopipe.phase3.assemble — seed module (refactor step 2, extracted 2026-08-11).

Origin: /lustre/gh/main/clv15/phase3_i_deep_live.py (VERBATIM extraction of
load_rows, epoch_mid, FP_CLASS, ATEAM and the ROOT/COADD/REPEAT_DEG constants,
plus internal deps LEDGERS and the _HDR cache that they reference).
The remaining assembly-capability content arrives at move time.
"""
from exopipe.common.roots import coadd_root
import os, csv, glob
from astropy.io import fits
from astropy.time import Time
import astropy.units as u
from exopipe.phase3 import search_i_transients as sit

# TO-MOVE-TO-COMMON-CONFIG: hardcoded path constants (ROOT, LEDGERS, COADD)
# must migrate to the shared exopipe config module at move time.
ROOT = '/data07/gh/phase3/transients/Run_I_DEEP'

LEDGERS = ([f'{ROOT}/i_deep_ledger.csv']
           + sorted(glob.glob(f'{ROOT}/nodes/*/i_deep_ledger.csv')))
COADD = None   # NO default (Gregg 2026-09-04): coadd_root() at point of use
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
                f'{coadd_root()}/{lst}_{band}/subtracted/{date}_I_sub_{band}.fits')
            _HDR[key] = (Time(eh['DATE-OBS'])
                         + float(eh.get('INTTIME') or 3591.0) / 2 * u.s,
                         str(eh.get('DATE-OBS')))
        except Exception:
            _HDR[key] = (None, '')
    return _HDR[key]



# constituents (step-3 moves)
# Guarded star-imports: phase3_i_deep_final / _package_deep / _v5_build import
# this seed back (`import phase3_i_deep_live as live` -> this module). When a
# constituent-first import arrives here mid-cycle, the re-export is skipped
# (the seed names above are already defined and satisfy the cycle); a direct
# import of this module re-exports everything.
# NOTE: phase3_tenmin_assemble is NOT star-imported — its load_rows(tier,
# stokes, lst_filter) would shadow the extracted seed helper load_rows();
# import it directly as exopipe.phase3.phase3_tenmin_assemble.
try:
    from exopipe.phase3.phase3_catalog import *  # noqa
except (ImportError, AttributeError) as _e:  # circular-import guard only
    if 'partially initialized' not in str(_e):
        raise
try:
    from exopipe.phase3.phase3_final_csv import *  # noqa
except (ImportError, AttributeError) as _e:  # circular-import guard only
    if 'partially initialized' not in str(_e):
        raise
try:
    from exopipe.phase3.phase3_i_deep_final import *  # noqa
except (ImportError, AttributeError) as _e:  # circular-import guard only
    if 'partially initialized' not in str(_e):
        raise
try:
    from exopipe.phase3.phase3_i_deep_package_deep import *  # noqa
except (ImportError, AttributeError) as _e:  # circular-import guard only
    if 'partially initialized' not in str(_e):
        raise
# phase3_i_v5_build __main__-guarded 2026-08-12 (post-incident guard sweep);
# import is now side-effect-free. Run it as a script for the v5 build.
try:
    from exopipe.phase3.phase3_i_v5_build import *  # noqa
except (ImportError, AttributeError) as _e:  # circular-import guard only
    if 'partially initialized' not in str(_e):
        raise
try:
    from exopipe.phase3.phase3_sb_assemble import *  # noqa
except (ImportError, AttributeError) as _e:  # circular-import guard only
    if 'partially initialized' not in str(_e):
        raise
try:
    from exopipe.phase3.phase3_source_ids import *  # noqa
except (ImportError, AttributeError) as _e:  # circular-import guard only
    if 'partially initialized' not in str(_e):
        raise
# phase3_v_persource + phase3_xmatch __main__-guarded 2026-08-12
# (post-incident guard sweep); imports are now side-effect-free. Run them as
# scripts for the one-shot migration / cross-match passes.
try:
    from exopipe.phase3.phase3_v_persource import *  # noqa
except (ImportError, AttributeError) as _e:  # circular-import guard only
    if 'partially initialized' not in str(_e):
        raise
try:
    from exopipe.phase3.phase3_xmatch import *  # noqa
except (ImportError, AttributeError) as _e:  # circular-import guard only
    if 'partially initialized' not in str(_e):
        raise
