"""Configuration of `exopipe3 notify` - the hour's one email (the Phase 2 report plus the Phase 3 highlights).

Per run: --set notify.KEY=VALUE or --config FILE; `exopipe3 config --step notify` lists the section.
"""
from exopipe.phase3.config import _Section


class notify(_Section):
    """`exopipe3 notify`. Gregg 2026-11-12: "one email at the end of Phase 3 that is close to identical to the current one
    in what is included, just with the transient and forced photometry highlights added"; transients: "Ideally I want real
    transients not junk" (the candidates that pass every default filter of their served page, most significant first);
    targets of the Hot / Warm Jupiter and 10 pc samples: "Just detections!!". The recipients, the mail account and the Slack
    webhook are Phase 2's (exopipe/common/phase2_config.py): this section adds none."""
    SEND = False                                                 # False: the email is WRITTEN (<out>/notify/<item>.eml, .html), not sent.
                                                                 # True needs Gregg's go: the recipients are Phase 2's list
    TOP_N = 10                                                   # transient candidates listed per Stokes (V first, then I)
    OUT_ROOT = '/lustre/gh/main/exopipe_ops/status_reports'      # <out>/notify/<item>_highlights.html (one small file per hour)
    # The served review pages (two-port server): the registry gives each page's run directory, catalog directory and code.
    REGISTRY = '/lustre/gh/main/phase3/serve/registry.json'
    HUB_URL = 'http://localhost:8765'                            # links as the hub writes them (the reader tunnels to the review host)
    PAGES = {('deep', 'V'): 'p8766', ('deep', 'I'): 'p8768',     # (tier, Stokes) -> page id; hub layout of 2026-10-27
             ('10-minute', 'V'): 'p8787', ('10-minute', 'I'): 'p8788',
             ('1-minute', 'V'): 'p8988', ('1-minute', 'I'): 'p8989',
             ('10-s', 'V'): 'p8986', ('10-s', 'I'): 'p8987'}
    # Forced photometry: the served event lists (events at or above each sample's detection threshold, with the
    # association verdict of each) under paths.SNAPSHOT_REVIEW_ROOT
    FORCED_RUN = 'Run_SNAPFORCED_{stokes}{tier}_gen5'
    FORCED_TIERS = {'deep': 'deep', '10-minute': '10min', '1-minute': '1min', '10-s': '10s'}
    SAMPLES = {'HWJ': 'Hot / Warm Jupiter', '10pc': '10 pc'}     # sample tag of the forced tables -> name in the email
    XM_SAMPLES = {'Hot_Warm_Jupiters': 'HWJ', '10pc': '10pc'}    # prefix of a catalog's xm_best -> sample tag
