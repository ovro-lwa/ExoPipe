"""Mean local sidereal time at the observatory and the UTC ranges of an LST hour, by the standard formula: pure arithmetic,
no astropy, no IERS tables, no network, no logging.

THE FORMULA. Greenwich mean sidereal time, in hours, GMST = 18.697374558 + 24.06570982441908 * D, with D the days since
2000-01-01 12:00 UT1 (Julian date 2451545.0): the U.S. Naval Observatory's linear form of the IAU 1982 GMST, which loses
0.1 s per century. The local mean sidereal time (LMST) adds the site's east longitude / 15 (``CFG.instrument.SITE_LON_DEG``,
the observatory location of Phases 1 and 2, common/pipeline_config.OVRO_LOC). UTC is taken as UT1: |UT1 - UTC| < 0.9 s by
the definition of UTC, so a time computed here is within 1 s of an IERS-based one (astropy's Time.sidereal_time('mean'),
which Phase 1's LSTScheduler.parse_range uses; tests/test_retention_do_not_delete.py compares the two).
THE LST HOUR OF A DATE. LMST runs 24 sidereal hours in 23 h 56 min 4 s of UTC, so one UTC date (24 h) holds every LST hour
once and, near its start or its end, one LST hour a second time: an hour h whose first start falls in the first ~4 min of
the date (about 00:02 UTC) starts again about 23:58 UTC; an hour under way at 00:00 UTC starts again late that day. Phase 1
dates the cell of LST h by its job's start date: a run with --range <h>h-<h+1>h --date D processes the first start at or
after 00:00 UTC of D. ``lst_hour_ranges`` returns EVERY range of LST h that overlaps the UTC date D - Phase 1's and any other
- each one whole (from its start, which may lie on the day before, to its end, which may lie on the day after).
"""
import calendar
import re
import time

J2000_UNIX_S = 946728000.0                 # s since 1970 of 2000-01-01T12:00:00 (the formula's epoch, UTC taken as UT1)
GMST_J2000_H = 18.697374558                # h, GMST at that epoch
GMST_RATE_H_PER_DAY = 24.06570982441908    # sidereal hours per day of 86400 s
SIDEREAL_DAY_S = 86400.0 * 24.0 / GMST_RATE_H_PER_DAY     # s, 86164.0905
SIDEREAL_HOUR_S = SIDEREAL_DAY_S / 24.0                   # s, 3590.1704
_LST_RE = re.compile(r'^(\d\d)h$')
_DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')


def lmst_hours(t, lon_deg):
    """Local mean sidereal time, hours in [0, 24), at ``t`` seconds since 1970 UTC (taken as UT1) and east longitude
    ``lon_deg`` (degrees; west negative)."""
    d = (float(t) - J2000_UNIX_S) / 86400.0
    return (GMST_J2000_H + GMST_RATE_H_PER_DAY * d + float(lon_deg) / 15.0) % 24.0


def _hour(lst):
    if isinstance(lst, str):
        m = _LST_RE.match(lst)
        h = int(m.group(1)) if m else -1
    elif isinstance(lst, int) and not isinstance(lst, bool):
        h = lst
    else:
        h = -1
    if not 0 <= h <= 23:
        raise ValueError('not an LST hour: %r (00h..23h, or 0..23)' % (lst,))
    return h


def lst_hour_ranges(date, lst, lon_deg=None, pad_s=0.0):
    """Every UTC range during which the local mean sidereal time is in [h, h + 1) and that overlaps the UTC date ``date``
    (YYYY-MM-DD): [(start, end)] in seconds since 1970 UTC, sorted, one or two of them, each one sidereal hour (3590.17 s)
    long and then widened by ``pad_s`` seconds on both sides (module docstring). ``lst``: '19h' or 19; ``lon_deg``: east
    longitude in degrees (None: ``CFG.instrument.SITE_LON_DEG``). Example: 19h of 2025-05-04 at the observatory is
    12:02:37-13:02:27 UTC. ValueError on an LST that is not 00h..23h, a date that is not YYYY-MM-DD or a negative pad."""
    h = _hour(lst)
    if not isinstance(date, str) or not _DATE_RE.match(date):
        raise ValueError('not a date YYYY-MM-DD: %r' % (date,))
    day0 = float(calendar.timegm(time.strptime(date, '%Y-%m-%d')))     # ValueError on a date that is no date
    pad = float(pad_s)
    if not pad >= 0:
        raise ValueError('a negative pad (%r s) would shrink the ranges' % (pad_s,))
    if lon_deg is None:
        from exopipe.phase3.config import CFG
        lon_deg = CFG.instrument.SITE_LON_DEG
    first = day0 + ((h - lmst_hours(day0, lon_deg)) % 24.0) * SIDEREAL_HOUR_S    # the first start at or after 00:00 UTC
    out = []
    for k in (-1, 0, 1):
        s = first + k * SIDEREAL_DAY_S
        if s < day0 + 86400.0 and s + SIDEREAL_HOUR_S > day0:
            out.append((s - pad, s + SIDEREAL_HOUR_S + pad))
    return out
