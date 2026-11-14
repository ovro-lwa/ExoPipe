"""Library of `exopipe3 notify` (the hour's one email: the Phase 2 report plus the Phase 3 highlights).

- sources.py  the listed rows: untargeted candidates of the hour of the deep and 10-minute pages that pass the filters of
              the list (through page_rule.py), the forced-photometry event lists under the same value filters, and the
              catalogs' cross-match for the target samples
- page_rule.py  one served page's own rule, evaluated from the page code the registry names (readers only)
- qa.py       what the email adds to Phase 2's QA section: the Phase 3 QA percentile table (lib/qa/percentiles.py) and
              the ionosphere numbers as percentiles among the processed hours
- block.py    the HTML block: chain outcome and limits first, target detections, then the candidates (Stokes V first)
"""
