#!/usr/bin/env python3
"""STANDING TEST — R1 amendment #1: per-epoch beam+RESTFRQ stamping and
validate-on-write refusal for stage-B per-epoch residuals.

Ruling: fix_ledger R1-ratified scope (a) + R1-amendment1-GO (Gregg
2026-08-13); R-A123 (A3's fix = the write-time refusal); CLAUDE.md rule 7
(epoch products carry their OWN headers) + rule 3 (gates refuse, not warn).

Asserts, on a minimal constructed write (no archive inputs):
  (i)   own-epoch stamping: save_epoch_residual writes BMAJ/BMIN/BPA/RESTFRQ
        from the epoch's OWN Phase-2 source header (digit-exact), with
        BEAMSRC + HISTORY provenance;
  (ii)  refusal on the R1 signature: an outgoing header carrying the co-add
        TEMPLATE beam while the epoch's own header says otherwise is REFUSED
        (EpochHeaderRefusal), and no product is written;
  (iii) refusal, not fallback, when the four keys cannot be resolved from
        the epoch's own source (missing file / missing keys);
  (iv)  the single-night degenerate case (own == template) PASSES — the
        golden 11h_Green configuration.

Run standalone:  python tests/test_r1_epoch_header_gate.py
Under pytest:    pytest tests/test_r1_epoch_header_gate.py

Style follows tests/test_import_side_effects.py: harness-less standalone
run; missing third-party packages are reported UNTESTED-IMPORT (with
reason), never silently skipped.
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
    import exopipe.phase3.lib.masters.coadd as cfp
except Exception as e:                                    # pragma: no cover
    _IMPORT_ERROR = f'{e.__class__.__name__}: {e}'


# Distinct, recognizable key sets (template != epoch-own; V-own != I-own).
TEMPLATE_KEYS = dict(BMAJ=0.1414852996307893, BMIN=0.1336650227868275,
                     BPA=31.45531107254968, RESTFRQ=52159673.584604)
OWN_I_KEYS = dict(BMAJ=0.1458810709937129, BMIN=0.1339876977856135,
                  BPA=34.17104278661974, RESTFRQ=51914645.21067964)
OWN_V_KEYS = dict(BMAJ=0.2381164103196867, BMIN=0.1638488479838786,
                  BPA=46.60309601434563, RESTFRQ=51914645.21067964)


def _mk_source(path, keys, with_time=True):
    """Minimal Phase-2-like source image carrying beam+RESTFRQ (+time)."""
    hdr = fits.Header()
    for k, v in keys.items():
        hdr[k] = v
    if with_time:
        hdr['DATE-OBS'] = '2025-01-06T05:00:00.000'
        hdr['MJD-OBS'] = 60681.2083333
        hdr['DURATION'] = 600.0
    fits.writeto(path, np.zeros((4, 4), dtype=np.float32), hdr,
                 overwrite=True)
    return path


def _tmpl_hdr():
    hdr = fits.Header()
    hdr['NAXIS'] = 2
    hdr['NAXIS1'] = 4
    hdr['NAXIS2'] = 4
    for k, v in TEMPLATE_KEYS.items():
        hdr[k] = v
    hdr['DATE-OBS'] = '2024-12-27T05:00:00.000'   # stale template time
    return hdr


def _setup(tmp):
    src_i = _mk_source(os.path.join(tmp, 'epoch_I_R075.fits'), OWN_I_KEYS)
    src_v = _mk_source(os.path.join(tmp, 'epoch_V_R0.fits'), OWN_V_KEYS)
    return _tmpl_hdr(), src_i, src_v


def test_own_epoch_stamping():
    """(i) written residual carries the epoch's OWN keys, digit-exact,
    per Stokes, with provenance."""
    tmp = tempfile.mkdtemp(prefix='r1gate_')
    try:
        tmpl, src_i, src_v = _setup(tmp)
        data = np.zeros((4, 4), dtype=np.float32)
        for stokes, src, own in (('I', src_i, OWN_I_KEYS),
                                 ('V', src_v, OWN_V_KEYS)):
            epoch_hdr = cfp.epoch_time_header(tmpl, src)
            out = os.path.join(tmp, f'ep_{stokes}_sub.fits')
            cfp.save_epoch_residual(out, data, epoch_hdr, src, tmpl, stokes)
            got = fits.getheader(out)
            for k, v in own.items():
                assert float(got[k]) == v, (stokes, k, got[k], v)
            assert got['BEAMSRC'] == 'epoch', got.get('BEAMSRC')
            hist = '\n'.join(str(c) for c in got.get('HISTORY', []))
            assert os.path.basename(src) in hist, hist
            assert 'R1-amendment1' in hist, hist
            # time block is the epoch's own, not the template's
            assert got['DATE-OBS'] == '2025-01-06T05:00:00.000'
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_refusal_r1_signature():
    """(ii) a template-beamed outgoing header (the R1 signature) is REFUSED
    and nothing is written."""
    tmp = tempfile.mkdtemp(prefix='r1gate_')
    try:
        tmpl, src_i, _ = _setup(tmp)
        out = os.path.join(tmp, 'injected_I_sub.fits')
        injected = cfp.epoch_time_header(tmpl, src_i)  # old writer: template
        # beam+RESTFRQ survive, only the time block is per-epoch
        raised = False
        try:
            cfp.validate_epoch_residual_header(out, injected, src_i, tmpl)
        except cfp.EpochHeaderRefusal as e:
            raised = True
            msg = str(e)
            assert 'R1-amendment1' in msg
            assert 'write-failure signature' in msg
            assert repr(TEMPLATE_KEYS['BMAJ']) in msg      # expected vs found
            assert repr(OWN_I_KEYS['BMAJ']) in msg
        assert raised, 'R1-signature header was NOT refused'
        assert not os.path.exists(out), 'refusal must write nothing'
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_refusal_missing_source_and_keys():
    """(iii) unresolvable source or keys = refusal, never fallback."""
    tmp = tempfile.mkdtemp(prefix='r1gate_')
    try:
        tmpl, _, _ = _setup(tmp)
        data = np.zeros((4, 4), dtype=np.float32)
        out = os.path.join(tmp, 'nosrc_I_sub.fits')
        for bad_src in (None,
                        os.path.join(tmp, 'DOES_NOT_EXIST.fits'),
                        _mk_source(os.path.join(tmp, 'nobeam.fits'), {})):
            raised = False
            try:
                cfp.save_epoch_residual(out, data,
                                        cfp.epoch_time_header(tmpl, bad_src),
                                        bad_src, tmpl, 'I')
            except cfp.EpochHeaderRefusal as e:
                raised = True
                assert 'Fallback to template/master values is forbidden' \
                    in str(e)
            assert raised, f'missing source {bad_src!r} was NOT refused'
            assert not os.path.exists(out), 'refusal must write nothing'
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_single_night_template_passes():
    """(iv) own == template (single-night template IS the epoch) passes."""
    tmp = tempfile.mkdtemp(prefix='r1gate_')
    try:
        src = _mk_source(os.path.join(tmp, 'single.fits'), TEMPLATE_KEYS)
        tmpl = _tmpl_hdr()
        out = os.path.join(tmp, 'single_I_sub.fits')
        cfp.save_epoch_residual(out, np.zeros((4, 4), dtype=np.float32),
                                cfp.epoch_time_header(tmpl, src),
                                src, tmpl, 'I')
        assert os.path.exists(out)
        got = fits.getheader(out)
        assert float(got['BMAJ']) == TEMPLATE_KEYS['BMAJ']
        assert got['BEAMSRC'] == 'epoch'
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


TESTS = [test_own_epoch_stamping, test_refusal_r1_signature,
         test_refusal_missing_source_and_keys,
         test_single_night_template_passes]

if _IMPORT_ERROR is not None:                            # pragma: no cover
    # Mirror test_import_side_effects: loud UNTESTED, never silent skip.
    import unittest

    class _Untested(unittest.TestCase):
        def test_untested_import(self):
            self.skipTest(f'UNTESTED-IMPORT: {_IMPORT_ERROR}')

    for _t in TESTS:
        globals().pop(_t.__name__, None)


def main():
    if _IMPORT_ERROR is not None:
        print(f'UNTESTED-IMPORT: cannot import dependencies in this '
              f'interpreter ({_IMPORT_ERROR}) — run under the pipeline env')
        return 1
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f'PASS  {t.__name__}')
        except AssertionError as e:
            failed += 1
            print(f'FAIL  {t.__name__}: {e}')
    print(f'{len(TESTS) - failed}/{len(TESTS)} passed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
