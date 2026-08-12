"""Aggregator for the exopipe.phase3.coadd capability (refactor step 3) — re-exports constituent modules."""
from exopipe.phase3.coadd_full_pipeline import *  # noqa
# EXCLUDED (import-time side effect): phase3_pbcorr_masters executes its full
# archive PB-correction pass at import (MODE defaults to 'all'; no __main__
# guard) — rewrites *_pbcorr/PB_deep masters. Import it explicitly to run it.
# from exopipe.phase3.phase3_pbcorr_masters import *  # noqa
