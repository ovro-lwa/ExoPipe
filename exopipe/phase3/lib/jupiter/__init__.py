"""`exopipe3 jupiter` - the Jupiter reflection program (steps/jupiter.py dispatches here).

  common.py       actions/stages, Jupiter-cell grammar and window, roots and the child write guard, step gates, in-place calls
  _inplace.py     the child driver: runs one function of a production module unchanged (no exopipe imports)
  mirrors.py      config keys that mirror in-place module literals (checked at start; overrides refused unless redirectable)
  measure.py      action measure (clv15/phase3_jupiter_measure.py: extract, fit, assign, test)
  v2.py           action v2 (the V2 code tree: geometry, masters, prepare, provenance, anchor, refcheck, metrics, render, summary)
  occurrence.py   action occurrence (V2 code tree occurrence/: fetch .. test)
  ephem.py        action ephem (phase0/jup_ephem_fetch.py)
  flag.py         action flag-backfill (exopipe/phase1/jupiter_flag_backfill.py: window, run, verify)
  serve.py        action serve (8800 Phase-1 page, 8801 V2 static site)
  quarantine.py   action quarantine (V2 code tree site/jsite_quarantine*.py)
  tests.py        content tests of the Phase-1, ephemeris, window, backfill and served classes
  tests_v2.py     content tests of the V2 and occurrence classes
"""
