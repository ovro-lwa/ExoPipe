#!/usr/bin/env python3
"""Generate README_canon_parameters.txt in the Run_GATES root from the
canon gate constants IN CODE (§10.1 reproducibility; Run_GATES previously
carried no parameter record). Re-run after any ratified constant change."""
import os, sys, datetime

sys.path.insert(0, '/lustre/gh/main/clv15')
import phase3_gate_analysis as GA
import phase3_final_survivors as FS
import phase3_veto as PV

OUT = '/data07/gh/phase3/transients/Run_GATES/README_canon_parameters.txt'

txt = f"""CANON GATE PARAMETERS — generated from code constants
(tools/gen_run_gates_readme.py; do not hand-edit)
Canon per PHASE3_ARCHITECTURE.md §10.1 (G-05 ruling 2026-08-11):
offline chain = phase3_gate_analysis.py + phase3_final_survivors.py.

Image tag:            Low <= {GA.LOW_MAX}, High >= {GA.HIGH_MIN} (post-mstruct >=7sigma count)
mstruct:              reject if |local_snr| < 5 x max(mstruct, 1)
I/V 20x (rb=0):       IV_THRESH = {GA.IV_THRESH} (exact-pixel master-I / |V|)
bilobe (Int/High):    BILOBE_ON = {GA.BILOBE_ON}; SEP_HI = {PV.SEP_HI} beams,
                      LESSER_SIG = {PV.LESSER_SIG} sigma, LESSER_RATIO = {PV.LESSER_RATIO}
gaussfit:             SNR_GATE = {PV.SNR_GATE}, RESID_MAX = {PV.RESID_MAX},
                      ar_max = max(2.0, 1.5*bmaj/bmin); failed fit => KEEP; no V size cut
A-team veto:          ATEAM_DEG = {GA.ATEAM_DEG} (downstream of mstruct; unified
                      6-source list PROPOSED in section 10.1 card 6)
Jupiter veto:         JUP_DEG = {GA.JUP_DEG}
conditional I/V 10x:  BAD_N7 = {FS.BAD_N7}, BAD_IV_THRESH = {FS.BAD_IV_THRESH}
                      (fires on images with final >=7sigma count > BAD_N7)
MAX_V_PER_EPOCH:      flag-only, never a gate (G-06 ruling 2026-08-11)

RATIFIED AMENDMENTS (2026-08-11 round 3) NOT YET IN CODE (refactored path;
gate code frozen until then):
  A-team:   unified 6-source list (+ Hya A, Her A) both Stokes;
            V radius 3.0 deg (code above still implements 2.0/4-source);
            I radius 5.0 deg; strictly downstream of mstruct, no pre-blanks
  Moon:     NEW 15 arcmin removal veto, per-epoch evaluation MANDATORY
  Ephemeris: per-epoch obs_time for ALL evaluations (fixed-time retired)
  Image tag counter: ONE counter, two consumers (tag + section-3 QA
            percentile); may never diverge
  Size gate (I): empirical margin per section 10.1 Card 9 (covariance
            retained; spread_img recorded-only)

Generated: {datetime.datetime.utcnow().isoformat()}Z
"""
with open(OUT, 'w') as fh:
    fh.write(txt)
print(f'wrote {OUT}')
print(txt)
