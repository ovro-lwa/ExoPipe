"""PB maps carry their own product name (Gregg 2026-11-17, R-GREGG-20261117-PB-PRODNAME-AND-PBDIFFUSE-SH-CARDS-GO)."""
import pytest
from astropy.io import fits

from exopipe.phase3.lib.masters import pb_apply as P
from exopipe.phase3.core.products import parse_name

MASTER = '18MHz_I_deep_Taper_Robust+0.0_shflux_dewarped_aligned_coadd_LST00h_20250809-20251029_N008.fits'
PBMAP = '18MHz_I_deep_Taper_Robust+0.0_dewarped_aligned_coadd_pbmap_LST00h_20250809-20251029_N008.fits'


def _hdr(name=MASTER):
    h = fits.Header()
    h['PRODNAME'] = (name, 'product name')
    h['BUNIT'] = 'Jy/beam'
    h['SHSCALE'] = 1.1
    return h


def test_master_name_becomes_pbmap_name():
    h = P.pb_map_header(_hdr())
    assert h['PRODNAME'] == PBMAP
    assert h.comments['PRODNAME'] == 'product name'
    p = parse_name(h['PRODNAME'])
    assert p['proc'][-1] == 'pbmap' and 'shflux' not in p['proc'] and p['robust'] == 0.0
    assert P.pb_map_header_problems(h) == []
    assert P.PB_MAP_PRODNAME_RULING in ''.join(str(x) for x in h['HISTORY']).replace(' ', '')


def test_robust_075_kept():
    n = MASTER.replace('Robust+0.0', 'Robust-0.75')
    assert P.pb_map_prodname(n) == PBMAP.replace('Robust+0.0', 'Robust-0.75')


def test_idempotent():
    h = P.pb_map_header(_hdr())
    assert P.pb_map_header(h.copy()).tostring() == h.tostring()


def test_master_name_is_a_problem():
    h = _hdr(); del h['SHSCALE']; h['BUNIT'] = ''
    assert any('not a pbmap name' in x for x in P.pb_map_header_problems(h))


def test_unparsable_name_refused():
    with pytest.raises(RuntimeError):
        P.pb_map_header(_hdr('garbage'))


def test_no_prodname_stays_absent():
    h = _hdr(); del h['PRODNAME']
    h = P.pb_map_header(h)
    assert 'PRODNAME' not in h and P.pb_map_header_problems(h) == []
