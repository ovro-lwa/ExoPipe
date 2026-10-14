"""B-30 (Gregg 2026-10-14, the refactor chat's proposal): the PB weights' nights without a Phase-2 noise file.

pb_apply.wideband_pb weights a scope's member subbands by the QA thermal-noise files of the beam master's constituent nights
(subband_weights); a night without its file is left out of the weights. Every such night is COUNTED
(fallbacks.PB_QA_MISSING_FALLBACK), and a WIDEBAND scope - whose member subbands those weights mix; a single subband's weights
cancel - refuses when more than masters_pbmaps.PB_QA_MISSING_MAX_FRAC of its nights lack the file.

Two callers, each checking the weights of the PB maps it outputs: the masters pbmaps stage (lib/masters/deep_stages.pbmaps:
pb_qa_inputs, the split do_cell is about to use, before it writes) and the master-dewarp apply (lib/dewarp/steps_cell.apply:
pb_qa_missing, the cards of the source generation's siblings, whose maps it resamples and whose cards it keeps).
Light at import (os only).
"""
import os


def pb_qa_missing(fd, sc, variants):
    """[(variant, nights with a Phase-2 noise file, nights without)] of the pbcorr siblings of ``variants`` in ``fd``, from the
    cards pb_apply.apply_pbcorr stamps (PBNDATE, PBMISS): a night without its QA thermal-noise file is left out of the PB
    weights (pb_apply.subband_weights)."""
    from astropy.io import fits
    out = []
    for v in variants:
        p = os.path.join(fd, f'{v}_pbcorr_{sc}.fits')
        if os.path.exists(p):
            h = fits.getheader(p)
            out.append((v, int(h.get('PBNDATE', -1)), int(h.get('PBMISS', 0) or 0)))
    return out


def pb_qa_inputs(fd, lst, sc):
    """[(beam master, nights with a Phase-2 noise file, nights without)] of the I masters the PB maps are made from
    (pb_apply.wideband_pb: subband_weights over constituent_dates of I_deep_Taper_<robust>), read BEFORE pbmaps writes anything
    (review 2026-10-14 m4: the refusal came after do_cell had overwritten the maps). The siblings' PBNDATE / PBMISS cards
    (pb_qa_missing) are the same split, compared after do_cell."""
    from exopipe.phase3.lib.masters import pb_apply as PB
    out = []
    for rob in ('R075', 'R0'):
        b = os.path.join(fd, f'I_deep_Taper_{rob}_{sc}.fits')
        if os.path.exists(b):
            _w, used, missing = PB.subband_weights(lst, PB.constituent_dates(b))
            out.append((f'I_deep_Taper_{rob}', len(used), len(missing)))
    return out


def pb_qa_gate(u, cell, sc, rows, max_frac):
    """B-30 (Gregg 2026-10-14): every night missing from the PB weights is COUNTED (fallbacks.PB_QA_MISSING_FALLBACK, the
    survey variant's count once per cell); a WIDEBAND scope - its member subbands mixed by those weights; a single subband's
    weights cancel - refuses when a sibling misses more than ``max_frac`` of its nights (masters_pbmaps.PB_QA_MISSING_MAX_FRAC,
    PROPOSED 0.25; production 1.0 = no limit)."""
    from exopipe.phase3.core import fallback
    u.meta['pb_qa_nights'] = {v: {'with_noise_file': nd, 'missing': nm} for v, nd, nm in rows}
    survey = next(((v, nd, nm) for v, nd, nm in rows if v == 'I_deep_Taper_R075'), rows[0] if rows else None)
    if survey and survey[2]:
        fallback.site('fallbacks.PB_QA_MISSING_FALLBACK', f'{cell}: {survey[2]} of {survey[1] + survey[2]} nights lack their '
                      f'Phase-2 noise file - left out of the PB weights', n=survey[2])
    if sc.endswith('MHz'):
        return
    over = [(v, nd, nm) for v, nd, nm in rows if nm and nm / max(1, nd + nm) > float(max_frac)]
    if over:
        v, nd, nm = over[0]
        u.refuse(f'PB weights: {v} misses the Phase-2 noise file of {nm} of {nd + nm} nights ({nm / (nd + nm):.0%}), above '
                 f'masters_pbmaps.PB_QA_MISSING_MAX_FRAC {float(max_frac):.0%} - a wideband PB mixes its subbands by those '
                 f'weights')
