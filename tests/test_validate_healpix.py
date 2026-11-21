#!/usr/bin/env python
"""validate_fits_headers.validate_header on HEALPix maps and on images (decision C-08 of the mosaic spec, Gregg 2026-11-21).

A HEALPix map keeps its standard header in HDU 0, which has no data, and its pixels in a binary table in HDU 1
(PIXTYPE = 'HEALPIX'). The validator checks that table (NSIDE a power of two, ORDERING RING or NESTED, 12 NSIDE^2 pixels
when INDXSCHM is IMPLICIT, data not truncated) instead of refusing the map with 'No data in primary HDU'. Every other
file keeps the old rule. The validator reads headers only (shape from NAXISn) but still refuses a truncated file.
Run: python tests/test_validate_healpix.py
"""
import os
import sys
import tempfile

import numpy as np
from astropy.io import fits

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from exopipe.common.validate_fits_headers import validate_header   # noqa: E402

FAILS = []
STD = dict(BMAJ=0.5, BMIN=0.4, BPA=10.0, RESTFRQ=6.0e7, OBSFREQ=6.0e7, DATE_OBS=None)


def check(label, cond, detail=''):
    print(('PASS ' if cond else 'FAIL ') + label + ('' if cond else '  ' + str(detail)))
    if not cond:
        FAILS.append(label)


def std_cards(h):
    for k, v in STD.items():
        if v is not None:
            h[k] = v
    h['DATE-OBS'] = '2025-01-01T00:00:00'
    h['DURATION'] = 600.0
    h['INTTIME'] = 600.0
    h['TELESCOP'] = 'OVRO-LWA'
    return h


def image(path):
    h = fits.Header()
    for k, v in (('CTYPE1', 'RA---SIN'), ('CTYPE2', 'DEC--SIN'), ('CRVAL1', 10.0), ('CRVAL2', 40.0), ('CRPIX1', 8.0),
                 ('CRPIX2', 8.0), ('CDELT1', -0.03), ('CDELT2', 0.03), ('CUNIT1', 'deg'), ('CUNIT2', 'deg')):
        h[k] = v
    fits.PrimaryHDU(np.ones((16, 16), np.float32), header=std_cards(h)).writeto(path, overwrite=True)


def healpix(path, nside=1, rows=None, tform='E', ordering='RING', indxschm=None, pixtype='HEALPIX'):
    rep = int(tform[:-1] or 1)
    n = 12 * nside ** 2 // rep if rows is None else rows
    col = fits.Column(name='SIGNAL', format=tform, array=np.ones((n, rep) if rep > 1 else n, np.float32))
    t = fits.BinTableHDU.from_columns([col])
    for k, v in (('PIXTYPE', pixtype), ('ORDERING', ordering), ('NSIDE', nside), ('INDXSCHM', indxschm), ('COORDSYS', 'C')):
        if v is not None:
            t.header[k] = v
    fits.HDUList([fits.PrimaryHDU(header=std_cards(fits.Header())), t]).writeto(path, overwrite=True)


def main():
    d = tempfile.mkdtemp(prefix='test_validate_healpix_')
    p = os.path.join(d, 'x.fits')

    image(p)
    r = validate_header(p)
    check('image: valid', r.ok, r.errors)
    check('image: shape from NAXISn', r.keywords_found.get('SHAPE') == '(16, 16)', r.keywords_found.get('SHAPE'))
    check('image: no HEALPix note', 'HEALPIX' not in r.keywords_found and not any('HEALPix' in i for i in r.info), r.info)
    with open(p, 'r+b') as f:
        f.truncate(os.path.getsize(p) - 2880)
    r = validate_header(p)
    check('image: truncated data refused', not r.ok and any('truncated' in e for e in r.errors), r.errors)

    healpix(p, nside=1)
    r = validate_header(p)
    check('HEALPix NSIDE 1 RING: valid', r.ok, r.errors)
    check('HEALPix: shape is the pixel count', r.keywords_found.get('SHAPE') == '(12,)', r.keywords_found.get('SHAPE'))
    check('HEALPix: no celestial-WCS warning', not any('celestial WCS' in w for w in r.warnings), r.warnings)
    check('HEALPix: HEALPix note', any('HEALPix' in i for i in r.info), r.info)
    healpix(p, nside=4, ordering='NESTED')
    r = validate_header(p)
    check('HEALPix NSIDE 4 NESTED: valid', r.ok and r.keywords_found.get('SHAPE') == '(192,)', (r.errors, r.keywords_found))
    healpix(p, nside=2, tform='16E')
    r = validate_header(p)
    check('HEALPix 16 pixels per row: valid, 48 pixels', r.ok and r.keywords_found.get('SHAPE') == '(48,)', (r.errors, r.keywords_found))
    healpix(p, nside=2, rows=5, indxschm='EXPLICIT')
    r = validate_header(p)
    check('HEALPix partial sky, INDXSCHM EXPLICIT: valid', r.ok, r.errors)

    healpix(p, nside=3, rows=108)
    r = validate_header(p)
    check('HEALPix NSIDE 3 refused (not a power of two)', not r.ok and any('power of two' in e for e in r.errors), r.errors)
    healpix(p, nside=1, ordering=None)
    r = validate_header(p)
    check('HEALPix without ORDERING refused', not r.ok and any('ORDERING' in e for e in r.errors), r.errors)
    healpix(p, nside=1, rows=11)
    r = validate_header(p)
    check('HEALPix with 11 of 12 pixels refused', not r.ok and any('12 NSIDE^2' in e for e in r.errors), r.errors)
    healpix(p, nside=1, rows=0)
    r = validate_header(p)
    check('HEALPix with no pixels refused', not r.ok and any('no pixels' in e for e in r.errors), r.errors)
    healpix(p, nside=4)
    with open(p, 'r+b') as f:
        f.truncate(os.path.getsize(p) - 2880)
    r = validate_header(p)
    check('HEALPix: truncated table refused', not r.ok and any('truncated' in e for e in r.errors), r.errors)

    healpix(p, nside=4)
    with fits.open(p, mode='update', memmap=False) as h:
        h[1].header['NSIDE'] = 4.0
    r = validate_header(p)
    check('HEALPix NSIDE 4.0 (an integral float card): valid', r.ok and r.keywords_found.get('SHAPE') == '(192,)', r.errors)
    image(p)
    with fits.open(p, mode='append', memmap=False) as h:
        h.append(fits.BinTableHDU.from_columns([fits.Column(name='X', format='D', array=np.arange(2000.0))], name='PROVENANCE'))
    with open(p, 'r+b') as f:
        f.truncate(os.path.getsize(p) - 2880 * 2)
    r = validate_header(p)
    check('image with a PROVENANCE table cut short: refused (every HDU is checked)', not r.ok
          and any('truncated' in e and 'HDU 1' in e for e in r.errors), r.errors)
    import gzip
    image(p)
    with open(p, 'rb') as f, gzip.open(p + '.gz', 'wb') as g:
        g.write(f.read())
    r = validate_header(p + '.gz')
    check('a gzip-compressed image: valid (its size on disk is not the FITS stream\'s)', r.ok, r.errors)
    os.remove(p + '.gz')

    healpix(p, nside=1, pixtype='OTHER')
    r = validate_header(p)
    check('empty primary + non-HEALPix table: No data in primary HDU', r.errors == ['No data in primary HDU'], r.errors)
    fits.PrimaryHDU(header=std_cards(fits.Header())).writeto(p, overwrite=True)
    r = validate_header(p)
    check('empty primary, no extension: No data in primary HDU', r.errors == ['No data in primary HDU'], r.errors)

    from exopipe.phase3.config import CFG
    check('mosaic_conform.HEADER_VALIDATE is True (decision C-08)', CFG.mosaic_conform.HEADER_VALIDATE is True)

    for f in os.listdir(d):
        os.remove(os.path.join(d, f))
    os.rmdir(d)
    print('ALL PASS' if not FAILS else 'FAILED: %d' % len(FAILS))
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
