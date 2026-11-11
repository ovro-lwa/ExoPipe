#!/usr/bin/env python3
"""Grammar sync with production (RELEASE1_PLAN_20261108.md, "Grammar code": the refactor grammar takes production's tokens, so
every name in the archive round-trips; the renames of PB maps and striation models wait for the next generation).

Production's grammar (exopipe/common/phase3_products.py, last grammar commit e2d3bf7) has four processing tokens the refactor
lacked until this sync: lsrecon (after coadd), cleanres (after lsfill), pkcorr and caveat (after pbmap, in that order). Production
has no '1min' cadence (CADENCES = deep, 10min, snap on main and on every branch that holds the file) and the archive holds no FITS
name with one, so the refactor does not add it either: a '1min' name is refused by both grammars (a mutant below).

Data-free: the names below are REAL archive names, copied as literal strings (sampled read-only on 2026-11-10 from
/lustre/pipeline/exopipe/phase3/Coadd/Run_NOTAPER_HYBRID_20261028/ and mosaic/Run_HYBRID_20241218-20260419/, with three
mosaic/Run_FEATHER3_20241218-20251029/ names as controls), so this file runs in CI with no data.

Checks:
  1. every real name parses as the Phase 3 grammar and build_name(**parse_name(n)) == n;
  2. build_name puts the new tokens where production does, whatever order they are given in (proc reversed and shuffled), and
     the vocabulary equals a literal copy of production's;
  3. mutants are refused: misspelled or tagged tokens, a token in a fixed field, between LST and the date, or after the time id,
     the '1min' cadence, legacy or negative-zero robust; out-of-order names parse but do not round-trip;
  4. cluster only (production's tree present): the live production vocabulary equals the refactor's, and every name and mutant
     above gets the same verdict from both grammars. Without production's tree this prints UNTESTED-PRODUCTION and does not fail.
Run: python tests/test_products_grammar_sync.py   (or under pytest)
"""
import importlib.util
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

_IMPORT_ERROR = None
try:
    from exopipe.phase3.core import products as P
except Exception as e:                                    # pragma: no cover
    _IMPORT_ERROR = f'{e.__class__.__name__}: {e}'

PRODUCTION = '/lustre/gh/main/exopipe/exopipe/common/phase3_products.py'

# Production's vocabulary on 2026-11-10 (exopipe/common/phase3_products.py at e2d3bf7).
PRODUCTION_PROC_ORDER = ['vclean', 'mstruct', 'loo', 'shflux', 'mdewarp', 'pbcorr', 'dewarped', 'aligned',
                         'psfmatch', 'coadd', 'lsrecon', 'msub', 'nmsub', 'ssub', 'sigstack',
                         'mosaic', 'jysr', 'jybeam', 'feather', 'tcorr', 'lsfill', 'tpfill', 'cleanres',
                         'psfmap', 'rgbcomp', 'molrender', 'carrender', 'pbmap', 'pkcorr', 'caveat']
PRODUCTION_CADENCES = ['deep', '10min', 'snap']

# ---- real archive names (literal; no data read) ----------------------------------------------------------------------------
# lsrecon: hybrid NoTaper masters (Coadd/Run_NOTAPER_HYBRID_20261028/<LST>/<scope>/)
LSRECON_MASTERS = [
    '55MHz_I_deep_NoTaper_Robust+0.0_shflux_dewarped_aligned_coadd_lsrecon_LST23h_20250823-20251014_N005.fits',
    '73MHz_I_deep_NoTaper_Robust-0.75_shflux_pbcorr_dewarped_aligned_coadd_lsrecon_LST21h_20250816-20251014_N012.fits',
    '27MHz_I_deep_NoTaper_Robust+0.0_shflux_mdewarp_pbcorr_dewarped_aligned_coadd_lsrecon_LST13h_20250309-20250507_N016.fits',
    '73MHz_I_deep_Taper_Robust+0.0_shflux_dewarped_aligned_coadd_lsrecon_lsfill_LST00h_20250901-20251029_N007.fits',
    '82MHz_I_deep_Taper_Robust-0.75_shflux_pbcorr_dewarped_aligned_coadd_lsrecon_ssub-rst8_lsfill_LST22h_20250804-20251014_N015.fits',
    '82MHz_I_deep_Taper_Robust-0.75_shflux_mdewarp_dewarped_aligned_coadd_lsrecon_ssub-rst8_lsfill_LST14h_20250309-20250508_N016.fits',
    'Red_I_deep_NoTaper_Robust-0.75_shflux_mdewarp_dewarped_aligned_coadd_lsrecon_LST22h_20250801-20251014_N019.fits',
    'Green_I_deep_Taper_Robust+0.0_shflux_mdewarp_pbcorr_dewarped_aligned_coadd_lsrecon_lsfill_LST18h_20250405-20251014_N016.fits',
    'Blue_I_deep_NoTaper_Robust+0.0_shflux_pbcorr_dewarped_aligned_coadd_lsrecon_LST10h_20241218-20250505_N026.fits',
    'Full_I_deep_Taper_Robust-0.75_shflux_pbcorr_dewarped_aligned_coadd_lsrecon_lsfill_LST05h_20241218-20251014_N022.fits',
]
# cleanres: CLEAN residuals of the hybrid masters (same Run)
CLEANRES_MASTERS = [
    '69MHz_I_deep_Taper_Robust+0.0_shflux_dewarped_aligned_coadd_lsrecon_lsfill_cleanres_LST22h_20250806-20251014_N015.fits',
    '73MHz_I_deep_Taper_Robust-0.75_shflux_dewarped_aligned_coadd_lsrecon_ssub-rst8_lsfill_cleanres_LST04h_20241218-20250111_N019.fits',
    '82MHz_I_deep_Taper_Robust+0.0_shflux_mdewarp_dewarped_aligned_coadd_lsrecon_ssub-rst8_lsfill_cleanres_LST18h_20250405-20251014_N016.fits',
    'Red_I_deep_Taper_Robust-0.75_shflux_mdewarp_dewarped_aligned_coadd_lsrecon_lsfill_cleanres_LST16h_20250309-20250508_N016.fits',
    'Green_I_deep_Taper_Robust+0.0_shflux_dewarped_aligned_coadd_lsrecon_lsfill_cleanres_LST20h_20250503-20251014_N015.fits',
    'Blue_I_deep_Taper_Robust-0.75_shflux_dewarped_aligned_coadd_lsrecon_ssub-rst8_lsfill_cleanres_LST15h_20250309-20250508_N016.fits',
]
# lsrecon: hybrid mosaics (mosaic/Run_HYBRID_20241218-20260419/<scope>/I/{healpix,radec}/)
LSRECON_MOSAICS = [
    'Green_I_deep_Taper_Robust+0.0_pbcorr_coadd_lsrecon_mosaic_jysr_feather_lsfill_tpfill_molrender_allsky_20241218-20260419_N024.fits',   # tpfill (exopipe 4b8fdd5; sampled 2026-11-11)
    '23MHz_I_deep_Taper_Robust-0.75_pbcorr_coadd_lsrecon_mosaic_jysr_feather_lsfill_tpfill_molrender_allsky_20241218-20251029_N024.fits',   # tpfill (exopipe 4b8fdd5; sampled 2026-11-11)
    '50MHz_I_deep_Taper_Robust-0.75_pbcorr_coadd_lsrecon_mosaic_jysr_feather_lsfill_tpfill_carrender_allsky_20241218-20260419_N024.fits',   # tpfill (exopipe 4b8fdd5; sampled 2026-11-11)
    '36MHz_I_deep_Taper_Robust-0.75_pbcorr_coadd_lsrecon_mosaic_jysr_feather_lsfill_tpfill_allsky_20241218-20251029_N024.fits',   # tpfill (exopipe 4b8fdd5; sampled 2026-11-11)
    '55MHz_I_deep_Taper_Robust+0.0_pbcorr_coadd_lsrecon_mosaic_jysr_feather_lsfill_allsky_20241218-20251029_N024.fits',
    '27MHz_I_deep_Taper_Robust-0.75_coadd_lsrecon_mosaic_jybeam_feather_lsfill_allsky_20241218-20251029_N024.fits',
    '41MHz_I_deep_Taper_Robust+0.0_coadd_lsrecon_mosaic_jybeam_feather_lsfill_carrender_allsky_20241218-20251029_N024.fits',
    'Red_I_deep_Taper_Robust-0.75_pbcorr_coadd_lsrecon_mosaic_jysr_feather_lsfill_molrender_allsky_20241218-20251029_N024.fits',
    'Green_I_deep_Taper_Robust+0.0_pbcorr_coadd_lsrecon_mosaic_feather_lsfill_psfmap_allsky_20241218-20260419_N024.fits',
    'Blue_I_deep_Taper_Robust-0.75_coadd_lsrecon_mosaic_feather_lsfill_pbmap_allsky_20241218-20260419_N024.fits',
    'Full_I_deep_Taper_Robust+0.0_coadd_lsrecon_mosaic_jybeam_feather_lsfill_molrender_allsky_20241218-20260419_N024.fits',
]
# pkcorr: peak-recovery maps of the hybrid mosaics (same Run)
PKCORR_MAPS = [
    '59MHz_I_deep_Taper_Robust+0.0_pbcorr_coadd_lsrecon_mosaic_feather_lsfill_pkcorr_allsky_20241218-20251029_N024.fits',
    '64MHz_I_deep_Taper_Robust-0.75_pbcorr_coadd_lsrecon_mosaic_feather_lsfill_pkcorr_allsky_20241218-20251029_N024.fits',
    'Red_I_deep_Taper_Robust+0.0_pbcorr_coadd_lsrecon_mosaic_feather_lsfill_pkcorr_allsky_20241218-20251029_N024.fits',
    'Green_I_deep_Taper_Robust-0.75_pbcorr_coadd_lsrecon_mosaic_feather_lsfill_pkcorr_allsky_20241218-20260419_N024.fits',
    'Blue_I_deep_Taper_Robust+0.0_pbcorr_coadd_lsrecon_mosaic_feather_lsfill_pkcorr_allsky_20241218-20260419_N024.fits',
    'Full_I_deep_Taper_Robust-0.75_pbcorr_coadd_lsrecon_mosaic_feather_lsfill_pkcorr_allsky_20241218-20260419_N024.fits',
]
# caveat: caveat maps of the high-band hybrid mosaics (same Run)
CAVEAT_MAPS = [
    '69MHz_I_deep_Taper_Robust+0.0_pbcorr_coadd_lsrecon_mosaic_feather_lsfill_caveat_allsky_20241218-20251029_N024.fits',
    '73MHz_I_deep_Taper_Robust-0.75_pbcorr_coadd_lsrecon_mosaic_feather_lsfill_caveat_allsky_20241218-20251029_N024.fits',
    'Blue_I_deep_Taper_Robust-0.75_pbcorr_coadd_lsrecon_mosaic_feather_lsfill_caveat_allsky_20241218-20260419_N024.fits',
    'Full_I_deep_Taper_Robust+0.0_pbcorr_coadd_lsrecon_mosaic_feather_lsfill_caveat_allsky_20241218-20260419_N024.fits',
]
# controls (no new token; round-tripped before the sync too): mosaic/Run_FEATHER3_20241218-20251029/
CONTROLS = [
    '23MHz_I_deep_Taper_Robust-0.75_coadd_mosaic_jybeam_feather_carrender_allsky_20241218-20251029_N024.fits',
    '59MHz_I_deep_Taper_Robust-0.75_coadd_ssub-rst8_mosaic_feather_pbmap_allsky_20241218-20251029_N024.fits',
    '78MHz_I_deep_Taper_Robust+0.0_coadd_ssub-rst8_mosaic_jybeam_feather_carrender_allsky_20241218-20251029_N024.fits',
]
BY_TOKEN = [('lsrecon', LSRECON_MASTERS), ('cleanres', CLEANRES_MASTERS), ('lsrecon', LSRECON_MOSAICS),
            ('pkcorr', PKCORR_MAPS), ('caveat', CAVEAT_MAPS)]
REAL = LSRECON_MASTERS + CLEANRES_MASTERS + LSRECON_MOSAICS + PKCORR_MAPS + CAVEAT_MAPS + CONTROLS

# ---- mutants ---------------------------------------------------------------------------------------------------------------
_M = CLEANRES_MASTERS[2]     # 82MHz ... coadd_lsrecon_ssub-rst8_lsfill_cleanres_LST18h_...
_K = PKCORR_MAPS[1]          # 64MHz ... lsfill_pkcorr_allsky_...
_C = CAVEAT_MAPS[3]          # Full ... lsfill_caveat_allsky_...
# (base name, substring, replacement, why): parse_name must raise ValueError
REFUSED = [
    (_M, '_lsrecon_', '_lsrecn_', 'misspelled lsrecon'),
    (_M, '_cleanres_', '_cleanre_', 'misspelled cleanres'),
    (_K, '_pkcorr_', '_pkcor_', 'misspelled pkcorr'),
    (_C, '_caveat_', '_caveats_', 'misspelled caveat'),
    (_M, '_lsrecon_', '_LSRECON_', 'upper-case lsrecon'),
    (_K, '_pkcorr_', '_PKCORR_', 'upper-case pkcorr'),
    (_M, '_lsrecon_', '_lsrecon-c01_', 'tagged lsrecon (only msub, nmsub, ssub, loo carry a tag)'),
    (_M, '_cleanres_', '_cleanres-rst8_', 'tagged cleanres'),
    (_K, '_pkcorr_', '_pkcorr-c01_', 'tagged pkcorr'),
    (_C, '_caveat_', '_caveat-c01_', 'tagged caveat'),
    (_M, '_coadd_lsrecon_', '_coadd-lsrecon_', 'hyphen in place of the underscore delimiter'),
    (_M, '_Taper_', '_lsrecon_', 'new token in the taper field'),
    (_K, '_I_deep_', '_I_pkcorr_', 'new token in the cadence field'),
    (_M, '_LST18h_', '_LST18h_cleanres_', 'new token between LST and the date range'),
    (_M, '_N016.fits', '_N016_cleanres.fits', 'new token after the time id'),
    (_K, '_N024.fits', '_N024_pkcorr.fits', 'new token after the time id (mosaic)'),
    (_C, '_allsky_', '_caveat_', 'new token in place of the allsky cell'),
    (_M, '_deep_', '_1min_', "'1min' cadence (production has none)"),
    (_K, '_deep_', '_1min_', "'1min' cadence on a mosaic name"),
    (_M, '_Robust+0.0_', '_Robust-0_', 'legacy implicit-sign robust with the new tokens'),
    (_C, '_Robust+0.0_', '_Robust-0.0_', 'negative-zero robust'),
]
# (base name, substring, replacement, why): parses, but the rebuild differs (canonical order), so the round-trip refuses it
OUT_OF_ORDER = [
    (_M, '_coadd_lsrecon_', '_lsrecon_coadd_', 'lsrecon before coadd'),
    (_M, '_lsrecon_ssub-rst8_', '_ssub-rst8_lsrecon_', 'lsrecon after ssub'),
    (_M, '_lsfill_cleanres_', '_cleanres_lsfill_', 'cleanres before lsfill'),
    (_K, '_lsfill_pkcorr_', '_pkcorr_lsfill_', 'pkcorr before lsfill'),
    (_C, '_lsfill_caveat_', '_caveat_lsfill_', 'caveat before lsfill'),
    (_K, '_lsrecon_mosaic_', '_mosaic_lsrecon_', 'lsrecon after mosaic'),
]
# A synthetic 1-min snapshot-tier name (none exists in the archive): refused like the mutants above.
ONE_MIN = 'Green_V_1min_Taper_Robust+0.0_pbcorr_dewarped_LST10h_20241229T081959.fits'


def _mutate(base, old, new):
    assert base.count(old) == 1, f'mutant pattern {old!r} must occur exactly once in {base}'
    return base.replace(old, new)


def _verdict(M, n):
    """'roundtrip', 'no-roundtrip' or 'refused' (ValueError) for one grammar module."""
    try:
        g = M.parse_name(n)
    except ValueError:
        return 'refused'
    return 'roundtrip' if g['grammar'] == 'phase3' and M.build_name(**g) == n else 'no-roundtrip'


def _production():
    if not os.path.isfile(PRODUCTION):
        return None
    spec = importlib.util.spec_from_file_location('production_phase3_products', PRODUCTION)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- tests -----------------------------------------------------------------------------------------------------------------
def test_vocabulary_equals_production_copy():
    assert P.PROC_ORDER == PRODUCTION_PROC_ORDER, f'PROC_ORDER {P.PROC_ORDER} != production {PRODUCTION_PROC_ORDER}'
    assert P.CADENCES == PRODUCTION_CADENCES, f'CADENCES {P.CADENCES} != production {PRODUCTION_CADENCES}'
    assert '1min' not in P.CADENCES


def test_each_list_carries_its_token():
    for tok, names in BY_TOKEN:
        for n in names:
            assert tok in P.parse_name(n)['proc'], f'{n} lacks {tok}'
    for n in CONTROLS:
        assert not {'lsrecon', 'cleanres', 'pkcorr', 'caveat'} & set(P.parse_name(n)['proc']), n


def test_real_names_round_trip():
    bad = []
    for n in REAL:
        try:
            g = P.parse_name(n)
            if g['grammar'] != 'phase3' or P.build_name(**g) != n:
                bad.append(n)
        except ValueError as e:
            bad.append(f'{n}: {e}')
    assert not bad, f'{len(bad)}/{len(REAL)} do not round-trip: {bad[:3]}'


def test_canonical_positions_whatever_the_input_order():
    rnd = random.Random(20261110)
    for n in REAL:
        g = P.parse_name(n)
        shuffled = list(g['proc'])
        rnd.shuffle(shuffled)
        for proc in (list(reversed(g['proc'])), shuffled):
            got = P.build_name(**dict(g, proc=proc))
            assert got == n, f'proc {proc} built {got}, expected {n}'


def test_build_refuses_tagged_new_tokens_and_1min():
    g = P.parse_name(CLEANRES_MASTERS[0])
    for tok in ('lsrecon-c01', 'cleanres-rst8', 'pkcorr-c01', 'caveat-c01', 'lsrecn'):
        try:
            P.build_name(**dict(g, proc=list(g['proc']) + [tok]))
        except ValueError:
            continue
        raise AssertionError(f'build_name accepted proc token {tok!r}')
    try:
        P.build_name(**dict(g, cadence='1min'))
    except ValueError:
        return
    raise AssertionError("build_name accepted cadence '1min'")


def test_mutants_refused():
    accepted = []
    for base, old, new, why in REFUSED:
        n = _mutate(base, old, new)
        if _verdict(P, n) != 'refused':
            accepted.append(f'{why}: {n}')
    if _verdict(P, ONE_MIN) != 'refused':
        accepted.append(f'1-min snapshot name: {ONE_MIN}')
    assert not accepted, f'{len(accepted)} mutants parsed: {accepted}'


def test_out_of_order_does_not_round_trip():
    bad = []
    for base, old, new, why in OUT_OF_ORDER:
        n = _mutate(base, old, new)
        v = _verdict(P, n)
        if v != 'no-roundtrip':
            bad.append(f'{why}: {v}: {n}')
    assert not bad, bad


def test_same_verdicts_as_production():
    """Cluster only: the live production grammar gives every name and mutant above the same verdict."""
    PM = _production()
    if PM is None:
        print(f'UNTESTED-PRODUCTION: {PRODUCTION} not present (CI); checked on the cluster')
        return
    for attr in ('PROC_ORDER', 'CADENCES', 'SCOPES', 'STOKES_VALUES', 'TAPERS'):
        assert getattr(P, attr) == getattr(PM, attr), f'{attr}: refactor {getattr(P, attr)} != production {getattr(PM, attr)}'
    names = list(REAL) + [ONE_MIN] + [_mutate(b, o, w) for b, o, w, _ in REFUSED + OUT_OF_ORDER]
    diff = [(n, _verdict(P, n), _verdict(PM, n)) for n in names if _verdict(P, n) != _verdict(PM, n)]
    assert not diff, f'{len(diff)} verdicts differ (name, refactor, production): {diff[:3]}'


TESTS = [test_vocabulary_equals_production_copy, test_each_list_carries_its_token, test_real_names_round_trip,
         test_canonical_positions_whatever_the_input_order, test_build_refuses_tagged_new_tokens_and_1min,
         test_mutants_refused, test_out_of_order_does_not_round_trip, test_same_verdicts_as_production]

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
        print(f'UNTESTED-IMPORT: cannot import dependencies in this interpreter ({_IMPORT_ERROR}) - run under the pipeline env')
        return 1
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f'PASS  {t.__name__}')
        except Exception as e:                           # a ValueError from the grammar is a failure too, not a crash
            failed += 1
            print(f'FAIL  {t.__name__}: {e.__class__.__name__}: {str(e)[:300]}')
    print(f'{len(TESTS) - failed}/{len(TESTS)} passed ({len(REAL)} real names, {len(REFUSED) + 1} refused mutants, '
          f'{len(OUT_OF_ORDER)} out-of-order mutants)')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
