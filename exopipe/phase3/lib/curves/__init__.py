"""Library of `exopipe3 curves` (the snapshot light-curve product).

- items.py    snapshot item kind (core.cells.parse_night_cell / parse_epoch, subbands only) and the search universe
- plan.py     per-cell plan: search-done evidence (G1), exclusions (G2), flow registration (G3), Master catalogues (G6),
              stage check (G4), input fingerprints
- lock.py     gate G13: the in-place extractor's module literals against config (ast, nothing imported)
- workers.py  clv15/phase3_snapshot_lightcurves.py run IN PLACE as production runs it, through core.legacy.run (site
              interpreter 'py36'), PROCS subbands at a time (I then V); outcome classification; facts read back
- publish.py  atomic publish of passing tables, byte-identical skip, --supersede (G12)
- tests.py    content tests of the product class 'light_curve' (LC1-LC6)
"""
