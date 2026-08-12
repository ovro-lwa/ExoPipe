"""Aggregator for the exopipe.common.astrometry capability (refactor step 3) — re-exports constituent modules."""
# EXCLUDED (import-time side effect): phase3_astrom_check reads sys.argv[1]
# at import (lst) and executes its check — import crashes without CLI args.
# Run it as a script instead. Aggregator kept as a placeholder so the
# capability import path exists.
# from exopipe.common.phase3_astrom_check import *  # noqa
