"""Library of `exopipe3 notify` (the hour's one email: the Phase 2 report plus the Phase 3 highlights).

- sources.py  readers of the served tables: untargeted candidates of the hour that pass their page's default filters
              (members + survivors for the deep and 10-minute pages; the server's page index for the 10-s and 1-minute
              pages), the forced-photometry event lists and the catalogs' cross-match for the target samples
- block.py    the HTML block: chain outcome and limits first, target detections, then the candidates (Stokes V first)
"""
