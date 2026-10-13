"""Co-add masters for `exopipe3 masters` (steps/masters.py); install, frames and search call into it too.

coadd (the co-add pipeline: flux scale, flow apply, outlier QA, masters, LOO, archive publish), membership (stage-2
rule and the T8 verify), epoch_gate and night_image_gate (gate records), flux_correction (band correction table),
shflux (absolute flux scale and re-bake), pb_apply and pbcorr_masters (PB maps, pbcorr siblings), vclean_masters
(cleaned Stokes V masters; runs search_v by path)."""
