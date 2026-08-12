"""Aggregator for the exopipe.phase3.mosaic capability (refactor step 3) — re-exports constituent modules."""
from exopipe.phase3.add_ateam import *  # noqa
from exopipe.phase3.make_3color_mollweide import *  # noqa
from exopipe.phase3.make_band_products import *  # noqa
from exopipe.phase3.mosaic_coadd_jysr import *  # noqa
# EXCLUDED (import-time side effect): mosaic_conformance_pass reads
# sys.argv[1] at import (MODE) and executes its pass — import crashes without
# CLI args. Run it as a script instead.
# from exopipe.phase3.mosaic_conformance_pass import *  # noqa
from exopipe.phase3.mosaic_flux_vs_zd import *  # noqa
from exopipe.phase3.pb_apply import *  # noqa
from exopipe.phase3.pb_render import *  # noqa
from exopipe.phase3.verify_mosaic import *  # noqa
