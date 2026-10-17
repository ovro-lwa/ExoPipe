#!/usr/bin/env python3
"""STANDING TEST - night admission of wideband co-adds (refactor lib/masters/coadd.py, ported from exopipe main d8ad64a): the
incomplete-subband-set refusal and the excluded-18 MHz refusal, both read through wideband_stacked_subbands, and the
findings the membership check and D-ADMIT read (wideband_admission_findings).

Rulings: Gregg 2026-08-21 "No wideband product is ever made if any required sub-band is missing"; 2026-10-17 (incomplete-set
nights, Option 1) "Enforce the rule upstream in the pipeline and maintain it going forward"; the 18 MHz allowance (2026-06-29;
2026-10-17 "18 MHz data fails QA regularly, and we have the rule that Red can still be formed if 18 MHz data are missing");
2026-10-17 "Agreed, enforce going forward, no rebuild now" (18 MHz excluded and never readmitted).

Asserts, on constructed Phase 2 science directories (headers only; no archive inputs):
  (i)   a band image's subbands: NSTACK = 5 means all five; otherwise the constituent lines, which must number NSTACK;
        no NSTACK, or lines that do not account for it, cannot be shown (None); an unreadable file is None;
  (ii)  a Full image is read through its band constituents, never through a band's subband lines it inherited (00h
        2025-09-02: Blue's lines, then Red+Blue), and a miscounted Full NSTACK does not matter; a Full without band
        constituents is read like a band image;
  (iii) admit_night: a complete night, and one missing only 18 MHz, are admitted; a night missing any other subband, or
        whose headers cannot show its set, is refused; a Red / Full holding an excluded 18 MHz is refused, the same
        night without the exclusion admitted; subband scopes never consult the wideband headers.

Run standalone:  python tests/test_masters_wideband_admission.py
Under pytest:    pytest tests/test_masters_wideband_admission.py
"""
import os
import shutil
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

_IMPORT_ERROR = None
try:
    import numpy as np
    from astropy.io import fits
    from exopipe.phase3.lib.masters import coadd as cfp
except Exception as e:  # noqa: BLE001
    _IMPORT_ERROR = e

RED = ['18', '23', '27', '32', '36']
GREEN = ['41', '46', '50', '55', '59']
BLUE = ['64', '69', '73', '78', '82']
LST, DATE = '00h', '2025-09-02'


def _name(band, st):
    rob = 'Robust-0.75' if st == 'I' else 'Robust-0'
    return f'{band}_{st}_deep_Taper_{rob}_dewarped_20250902_LST00h.fits'


def _write(sci, band, st, nstack=None, subs=(), bands=(), inherited=(), garbage=False):
    """One Phase 2 deep image of `band` and Stokes `st`, headers only. subs: subband constituent lines; bands: band
    constituent lines (a Full); inherited: subband lines copied from a band's own history (a Full)."""
    d = os.path.join(sci, band, st, 'deep')
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, _name(band, st))
    if garbage:
        open(p, 'wb').write(b'not a fits file')
        return p
    h = fits.Header()
    if nstack is not None:
        h['NSTACK'] = nstack
    for s in inherited:
        h.add_history(f'  constituent: {s}MHz/{st}/deep/{s}MHz-{st}-Taper-3240s-Robust-0.75-20250902_0')
    for s in subs:
        h.add_history(f'  constituent: {s}MHz/{st}/deep/{s}MHz-{st}-Taper-3240s-Robust-0.75-20250902_0')
    if bands:
        h.add_history(f'Full band: inverse-variance weighted stack of {"+".join(bands)} ({nstack} subbands)')
        for b in bands:
            h.add_history(f'  constituent: {b}/{st}/deep/{_name(b, st)}'[:72])
    fits.PrimaryHDU(data=np.zeros((2, 2), np.float32), header=h).writeto(p, overwrite=True)
    return p


def _night(root, tag, **bands):
    """A science directory with the given images: bands[name] = dict of _write arguments, applied to I and V."""
    sci = os.path.join(root, tag)
    for band, kw in bands.items():
        for st in 'IV':
            _write(sci, band, st, **kw)
    return sci


def run():
    if _IMPORT_ERROR is not None:
        print(f'UNTESTED-IMPORT: {_IMPORT_ERROR}')
        return 2
    root = tempfile.mkdtemp(prefix='wbadmit_')
    fails = []

    def check(what, got, want):
        if got != want:
            fails.append(f'{what}: got {got!r}, want {want!r}')

    try:
        S = cfp.wideband_stacked_subbands
        # (i) band images
        sci = _night(root, 'bands',
                     Red=dict(nstack=5),
                     Green=dict(nstack=4, subs=GREEN[1:]),
                     Blue=dict(nstack=4, subs=BLUE[:3] + BLUE[4:]))
        check('Red NSTACK 5', S(os.path.join(sci, 'Red/I/deep', _name('Red', 'I'))), set(RED))
        check('Green NSTACK 4 + lines', S(os.path.join(sci, 'Green/I/deep', _name('Green', 'I'))), set(GREEN[1:]))
        check('Blue NSTACK 4 + lines', S(os.path.join(sci, 'Blue/I/deep', _name('Blue', 'I'))), set(BLUE) - {'78'})
        sci = _night(root, 'unshown', Red=dict(nstack=4), Green=dict(subs=GREEN), Blue=dict(nstack=4, subs=BLUE))
        check('NSTACK 4 without lines', S(os.path.join(sci, 'Red/I/deep', _name('Red', 'I'))), None)
        check('no NSTACK', S(os.path.join(sci, 'Green/I/deep', _name('Green', 'I'))), None)
        check('5 lines for NSTACK 4', S(os.path.join(sci, 'Blue/I/deep', _name('Blue', 'I'))), None)
        sci = _night(root, 'garbage', Red=dict(garbage=True))
        check('unreadable', S(os.path.join(sci, 'Red/I/deep', _name('Red', 'I'))), None)
        check('missing file', S(os.path.join(sci, 'Red/I/deep', 'nope_I_deep.fits')), None)

        # (ii) Full images
        sci = _night(root, 'full_rb',                                   # 00h 2025-09-02: inherited Blue lines, Red+Blue
                     Red=dict(nstack=5, subs=RED), Blue=dict(nstack=5, subs=BLUE),
                     Full=dict(nstack=10, inherited=BLUE, bands=('Red', 'Blue')))
        check('Full Red+Blue', S(os.path.join(sci, 'Full/I/deep', _name('Full', 'I'))), set(RED + BLUE))
        sci = _night(root, 'full_miscount',                             # complete, NSTACK miscounted
                     Red=dict(nstack=5), Green=dict(nstack=5), Blue=dict(nstack=5),
                     Full=dict(nstack=3, inherited=GREEN, bands=('Red', 'Green', 'Blue')))
        check('Full R+G+B, NSTACK 3', S(os.path.join(sci, 'Full/I/deep', _name('Full', 'I'))), set(RED + GREEN + BLUE))
        sci = _night(root, 'full_direct', Full=dict(nstack=15))
        check('Full NSTACK 15, no band lines', S(os.path.join(sci, 'Full/I/deep', _name('Full', 'I'))), set(RED + GREEN + BLUE))
        sci = _night(root, 'full_noband', Red=dict(nstack=5), Full=dict(nstack=10, bands=('Red', 'Blue')))
        check('Full whose Blue image is missing', S(os.path.join(sci, 'Full/I/deep', _name('Full', 'I'))), None)

        # (iii) admission
        A = cfp.admit_night
        none = set()
        ex18 = {(LST, '18', DATE)}
        sci = _night(root, 'complete', Red=dict(nstack=5), Green=dict(nstack=5), Blue=dict(nstack=5),
                     Full=dict(nstack=15, bands=('Red', 'Green', 'Blue')))
        for b in ('Red', 'Green', 'Blue', 'Full'):
            check(f'complete {b} admitted', A(LST, b, DATE, sci, none), True)
        check('complete Red, 18 excluded: holds 18, refused', A(LST, 'Red', DATE, sci, ex18), False)
        check('complete Full, 18 excluded: holds 18, refused', A(LST, 'Full', DATE, sci, ex18), False)
        check('Green, 18 excluded: not concerned', A(LST, 'Green', DATE, sci, ex18), True)
        check('subband scope ignores wideband headers', A(LST, '23MHz', DATE, sci, none), True)
        sci = _night(root, 'no18', Red=dict(nstack=4, subs=RED[1:]), Green=dict(nstack=5), Blue=dict(nstack=5),
                     Full=dict(nstack=14, inherited=RED[1:], bands=('Red', 'Green', 'Blue')))
        for ex in (none, ex18):
            check(f'Red without 18 admitted ({len(ex)} exclusions)', A(LST, 'Red', DATE, sci, ex), True)
            check(f'Full without 18 admitted ({len(ex)} exclusions)', A(LST, 'Full', DATE, sci, ex), True)
        sci = _night(root, 'no23', Red=dict(nstack=4, subs=[RED[0]] + RED[2:]), Green=dict(nstack=5), Blue=dict(nstack=5),
                     Full=dict(nstack=14, bands=('Red', 'Green', 'Blue')))
        check('Red missing 23 refused', A(LST, 'Red', DATE, sci, none), False)
        check('Full whose Red misses 23 refused', A(LST, 'Full', DATE, sci, none), False)
        check('Green of that night admitted', A(LST, 'Green', DATE, sci, none), True)
        sci = _night(root, 'rb', Red=dict(nstack=5), Blue=dict(nstack=5),
                     Full=dict(nstack=10, inherited=BLUE, bands=('Red', 'Blue')))
        check('Full of Red+Blue refused', A(LST, 'Full', DATE, sci, none), False)
        sci = _night(root, 'unknown', Red=dict(nstack=4))
        check('Red that cannot show its set refused', A(LST, 'Red', DATE, sci, none), False)
        sci = os.path.join(root, 'vonly')                              # only the V image is stacked short
        _write(sci, 'Red', 'I', nstack=5)
        _write(sci, 'Red', 'V', nstack=4, subs=RED[:3] + RED[4:])
        check('Red whose V image misses 32 refused', A(LST, 'Red', DATE, sci, none), False)
        check('excluded night still refused', A(LST, 'Red', DATE, _night(root, 'excl', Red=dict(nstack=5)),
                                                 {(LST, 'Red', DATE)}), False)

        # (iv) the findings, in image order, without log lines (what membership.verify and admit.nights_to_admit read)
        F = cfp.wideband_admission_findings
        sci = os.path.join(root, 'complete')
        check('findings: complete night', F(LST, 'Full', sci, DATE, none), [])
        check('findings: complete Red, 18 excluded', [k for k, _, _ in F(LST, 'Red', sci, DATE, ex18)], ['excluded-18', 'excluded-18'])
        check('findings: subband scope', F(LST, '23MHz', sci, DATE, ex18), [])
        sci = os.path.join(root, 'no23')
        got = F(LST, 'Red', sci, DATE, none)
        check('findings: Red missing 23 (I then V)', [(k, n_.split('_')[1]) for k, n_, _ in got], [('incomplete', 'I'), ('incomplete', 'V')])
        check('findings: text', got[0][2] if got else None, 'was stacked from an incomplete subband set (missing 23MHz)')
        sci = os.path.join(root, 'unknown')
        check('findings: cannot show, 18 excluded', [k for k, _, _ in F(LST, 'Red', sci, DATE, ex18)],
              ['excluded-18', 'incomplete', 'excluded-18', 'incomplete'])
    finally:
        shutil.rmtree(root, ignore_errors=True)
    for f in fails:
        print('FAIL', f)
    print('PASS' if not fails else f'{len(fails)} FAIL')
    return 1 if fails else 0


def test_masters_wideband_admission():
    assert run() == 0


if __name__ == '__main__':
    sys.exit(run())
