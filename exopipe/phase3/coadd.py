"""Aggregator for the exopipe.phase3.coadd capability (refactor step 3) — re-exports constituent modules."""
from exopipe.phase3.coadd_full_pipeline import *  # noqa
# phase3_pbcorr_masters __main__-guarded 2026-08-12 (post-incident guard
# sweep); import is now side-effect-free. Run it as a script for the archive
# PB-correction pass.
from exopipe.phase3.phase3_pbcorr_masters import *  # noqa
