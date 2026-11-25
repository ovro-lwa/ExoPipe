"""The do-not-delete gate of the raw-data retention: Gregg's do-not-delete cells of data_decisions.csv are never deleted.

THE CELLS. A row of data_decisions.csv whose reason says "do not delete" (core/exclusions.do_not_delete_cells, the only
reader of the file; e.g. "ionospheric vortex event; do not delete (Gregg 2026-06-28)") marks its cell - one LST hour on one
date - as raw data that is never deleted. The cell's UTC RANGES are every range of that date during which the local mean
sidereal time is in the LST hour (core/sidereal.lst_hour_ranges: the standard GMST formula, UTC taken as UT1, the site
longitude CFG.instrument.SITE_LON_DEG; Phase 1's range of the cell - the first at or after 00:00 UTC - and, when the LST hour
recurs on that date (about 00:02 and 23:58 UTC), the other one too), each widened by retention.DO_NOT_DELETE_PAD_S (300 s)
on both sides: protects more, never less. 19h of 2025-05-04: 12:02:37-13:02:27 UTC, protected 11:57:37-13:07:27 UTC.
THE RULE. A raw item - one (subband, UT date, UT hour) directory of retention.RAW_ROOT - is PROTECTED when its data overlap
a protected range of any do-not-delete cell, WHATEVER ITS SUBBAND: a cell is a time range on the sky, and deleting one
subband of it breaks the cell. So the rows' own subbands do not narrow it: the 13 MHz subband, which ExoPipe does not image
and no row names, and any subband without a row are protected with the rest. An item's data run from its hour's start to
one integration (CFG.instrument.INTEGRATION_S) past the hour's end: its last 10-s set starts before the hour ends.
THE GATE (RT-DO-NOT-DELETE; it refuses, it never warns). The plan classifies a protected item 'protected' before anything
else (never listed, never in a manifest) and records the cells it honored in the manifest's sidecar; the delete refuses the
WHOLE manifest, deleting nothing, when any of its items is protected by the cells read at the delete (``Cells``), and
re-checks each item right before deleting it (re-reading the file only when its sha1 changed). Fail closed: a missing or
unreadable data_decisions.csv, a row core/exclusions refuses (an LST, date or subband it cannot read; deletion wording
without "do not delete"), or a cell whose UTC ranges cannot be computed refuses the plan and the delete. A do-not-delete row
read with an irregularity (more fields than the header, the text outside the reason column) protects its cell and is
reported (``irregular_note``).
"""
import hashlib
import time

from exopipe.phase3.core.record import Refusal

CODE = 'RT-DO-NOT-DELETE'


def ranges(path=None):
    """(cells, fingerprint): cells [{lst, date, ranges: [(start, end)], subbands, reasons, lines, irregular}] (start, end:
    seconds since 1970 UTC, padded), sorted by their first start; fingerprint: the data_decisions read's (path, sha1, ...)
    plus pad_s and site_lon_deg. Reads the file now. Refuses (RT-DO-NOT-DELETE) on anything it cannot read or convert."""
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import exclusions as E
    from exopipe.phase3.core import sidereal as S
    try:
        dnd, fp = E.do_not_delete_cells(path)
    except Refusal as e:
        raise Refusal('%s: the do-not-delete cells of data_decisions.csv cannot be read - nothing is planned or deleted '
                      '(fail closed): %s' % (CODE, e))
    pad, lon = CFG.retention.DO_NOT_DELETE_PAD_S, CFG.instrument.SITE_LON_DEG
    out = []
    for (lst, date), c in sorted(dnd.items()):
        try:
            rr = S.lst_hour_ranges(date, lst, lon, pad_s=pad)
            if not rr:
                raise ValueError('no range')
        except Exception as e:         # any failure of the conversion fails closed
            raise Refusal('%s: the UTC ranges of the do-not-delete cell %s %s (data_decisions.csv lines %s) cannot be '
                          'computed (%s: %s) - nothing is planned or deleted (fail closed)' % (
                              CODE, lst, date, _lines(c['lines']), e.__class__.__name__, e))
        out.append(dict(lst=lst, date=date, ranges=rr, subbands=sorted(c['subbands']), reasons=sorted(c['reasons']),
                        lines=list(c['lines']), irregular=list(c['irregular'])))
    out.sort(key=lambda c: (c['ranges'][0][0], c['lst']))
    return out, dict(fp, pad_s=float(pad), site_lon_deg=float(lon))


class Cells:
    """The do-not-delete cells as read once by ``ranges`` and re-read only when data_decisions.csv changed: ``current()``
    hashes the file's bytes (sha1; a few hundred kB) and re-reads the cells when the hash - or the configured path - differs
    from the read's. Never size and mtime alone: an edit within the same second of the same size leaves both unchanged.
    ``n_reads``: the reads of the cells so far."""

    def __init__(self, path=None):
        self.path_arg = path
        self.path = self._path()
        self.cells, self.fp = ranges(self.path)
        self.n_reads = 1

    def _path(self):
        from exopipe.phase3.config import CFG
        return self.path_arg or CFG.paths.DATA_DECISIONS

    def current(self):
        path = self._path()
        try:
            with open(path, 'rb') as f:
                sha1 = hashlib.sha1(f.read()).hexdigest()
        except OSError as e:
            raise Refusal('%s: data_decisions.csv %s cannot be read now (%s) - nothing deleted (fail closed)' % (
                CODE, path, e))
        if path != self.path or sha1 != self.fp.get('sha1'):
            self.path = path
            self.cells, self.fp = ranges(path)
            self.n_reads += 1
        return self.cells


def _lines(lines):
    lines = sorted(lines)
    return '%d-%d' % (lines[0], lines[-1]) if len(lines) > 1 and lines[-1] - lines[0] == len(lines) - 1 else \
        ','.join(str(n) for n in lines)


def item_span(t0, t1):
    """The time range of an item's data: its UT hour [t0, t1) and one integration past it (the last set's end)."""
    from exopipe.phase3.config import CFG
    return t0, t1 + float(CFG.instrument.INTEGRATION_S)


def cells_over(t0, t1, cells):
    """The do-not-delete cells with a protected range that overlaps the data of the item of UT hour [t0, t1) - of ANY
    subband (module docstring)."""
    a, b = item_span(t0, t1)
    return [c for c in cells if any(s < b and e > a for s, e in c['ranges'])]


def _span(s, e):
    a, b = time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(s)), time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(e))
    return '%s-%s' % (a, b[11:] if a[:10] == b[:10] else b)


def describe(cells):
    """One line for the cells: '19h 2025-05-04 (protected 2025-05-04T11:57:37-13:07:27 UTC, data_decisions.csv lines
    2085-2099: <reasons>)' per cell, '; ' between cells."""
    return '; '.join('%s %s (protected %s UTC, data_decisions.csv lines %s: %s)' % (
        c['lst'], c['date'], ' and '.join(_span(s, e) for s, e in c['ranges']), _lines(c['lines']),
        ' / '.join(c['reasons'])) for c in cells)


def irregular_note(cells):
    """'' or one line naming the do-not-delete rows read with an irregularity (they protect their cells; the row wants
    fixing)."""
    irr = sorted({(n, w, c['lst'], c['date']) for c in cells for n, w in c.get('irregular') or ()})
    if not irr:
        return ''
    return ('data_decisions.csv: %d do-not-delete row(s) read with an irregularity - their cells are protected; the rows '
            'want fixing: %s' % (len({x[0] for x in irr}), '; '.join('line %d (%s %s): %s' % (n, lst, d, w)
                                                                       for n, w, lst, d in irr[:10])))


def refuse(items, cells, what='the manifest'):
    """Refuses (RT-DO-NOT-DELETE) when any item of ``items`` [(item name, t0, t1)] lies in a do-not-delete cell: the reason
    names every such item and every cell. Returns None otherwise."""
    hit = [(name, cells_over(t0, t1, cells)) for name, t0, t1 in items]
    hit = [(n, cs) for n, cs in hit if cs]
    if not hit:
        return None
    named = {}
    for _n, cs in hit:
        for c in cs:
            named[(c['lst'], c['date'])] = c
    raise Refusal('%s: %s holds %d item(s) inside the do-not-delete cells of data_decisions.csv - NOTHING deleted (a '
                  'do-not-delete cell protects every subband of its time range): items %s%s; cells %s' % (
                      CODE, what, len(hit), ', '.join(n for n, _c in hit[:20]), ' ...' if len(hit) > 20 else '',
                      describe(sorted(named.values(), key=lambda c: c['ranges'][0][0]))))
