"""Aggregator for the exopipe.common.config capability (refactor step 3) — re-exports constituent modules."""
from exopipe.common.phase2_config import *  # noqa
from exopipe.phase3.config import STAGING_ROOT, ARCHIVE_ROOT, epoch_dir, coadd_dir, coadd_run_root, transients_run, candidate_dir, sample_dir  # noqa (phase3_paths, absorbed)
from exopipe.common.pipeline_config import *  # noqa
