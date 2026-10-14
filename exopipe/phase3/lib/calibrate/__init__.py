"""lib/calibrate - the library of `exopipe3 calibrate` (versioned reference tables; DESIGN 11.2).

    common.py         table classes, versions, stage directories, records lookup, in-place helpers, sha256, the install
                      bundle (versioned file, mirror, config patch, draft ledger row)
    bright_parent.py  bright-parent tables (deep, sub-LST) and multiples tables (deep): derive inputs/curves/build, diff,
                      install, verify
    cuts.py           the conformance table step replayed on a detection list (ported from deep_tables_elev_20261006._cuts
                      and p3s2_conform.py:254-300) for the bright-parent diff: installed vs candidate tables
    fluxscale.py      per-LST VLSSr scale table and the band correction k x g (stage A, LOO band vector, table)
    template_k.py     template-fit error scale K per tier and Stokes (summary from trials; sub-LST injection in place)
    tests.py          content tests of every product class the step writes (discovered by core/validate)

Nothing here is a tunable: values come from config sections tables, calibrate and calibrate_* (config/steps/calibrate.py).
"""
