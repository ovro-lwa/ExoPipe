"""The do-not-delete gate of the raw-data retention: Gregg's do-not-delete cells of data_decisions.csv are never deleted.

THE CELLS. A row of data_decisions.csv whose reason says "do not delete" (core/exclusions.do_not_delete_cells, the only
reader of the file; e.g. "ionospheric vortex event; do not delete (Gregg 2026-06-28)") marks its cell - one LST hour on one
date - as raw data that is never deleted. The cell's UTC range is the one its Phase 1 run processed (core/cells.epoch_utc:
19h of 2025-05-04 is 12:02:37-13:02:27 UTC on 2025-05-04).
THE RULE. A raw item - one (subband, UT date, UT hour) directory of retention.RAW_ROOT - is PROTECTED when its data overlap
the UTC range of any do-not-delete cell, WHATEVER ITS SUBBAND: a cell is a time range on the sky, and deleting one subband of
it breaks the cell. So the rows' own subbands do not narrow it: the 13 MHz subband, which ExoPipe does not image and no row
names, and any subband without a row are protected with the rest. An item's data run from its hour's start to one
integration (CFG.instrument.INTEGRATION_S) past the hour's end: its last 10-s set starts before the hour ends.
THE GATE (RT-DO-NOT-DELETE; it refuses, it never warns). The plan classifies a protected item 'protected' before anything
else (never listed, never in a manifest); the delete refuses the WHOLE manifest, deleting nothing, when any of its items is
protected by the cells read at the delete, and re-checks each item right before deleting it. Fail closed: a missing or
unreadable data_decisions.csv, a do-not-delete row that cannot be read, or a cell whose UTC range cannot be computed refuses
the plan and the delete.
"""
from exopipe.phase3.core.record import Refusal

CODE = 'RT-DO-NOT-DELETE'


def ranges(path=None):
    """(cells, fingerprint): cells [{lst, date, start, end, subbands, reasons, lines}] (start, end: seconds since 1970 UTC),
    sorted by start; fingerprint: the data_decisions read's (path, sha1, ...). Reads the file now. Refuses (RT-DO-NOT-DELETE)
    on anything it cannot read or convert."""
    from exopipe.phase3.core import cells as C
    from exopipe.phase3.core import exclusions as E
    try:
        dnd, fp = E.do_not_delete_cells(path)
    except Refusal as e:
        raise Refusal('%s: the do-not-delete cells of data_decisions.csv cannot be read - nothing is planned or deleted '
                      '(fail closed): %s' % (CODE, e))
    out = []
    for (lst, date), c in sorted(dnd.items()):
        try:
            t0, t1 = C.epoch_utc(lst, date)
        except Exception as e:         # any failure of the conversion (IERS tables, a date Phase 1 refuses) fails closed
            raise Refusal('%s: the UTC range of the do-not-delete cell %s %s (data_decisions.csv lines %s) cannot be '
                          'computed (%s: %s) - nothing is planned or deleted (fail closed)' % (
                              CODE, lst, date, _lines(c['lines']), e.__class__.__name__, e))
        out.append(dict(lst=lst, date=date, start=t0, end=t1, subbands=sorted(c['subbands']),
                        reasons=sorted(c['reasons']), lines=list(c['lines'])))
    out.sort(key=lambda c: (c['start'], c['lst']))
    return out, fp


def _lines(lines):
    lines = sorted(lines)
    return '%d-%d' % (lines[0], lines[-1]) if len(lines) > 1 and lines[-1] - lines[0] == len(lines) - 1 else \
        ','.join(str(n) for n in lines)


def item_span(t0, t1):
    """The time range of an item's data: its UT hour [t0, t1) and one integration past it (the last set's end)."""
    from exopipe.phase3.config import CFG
    return t0, t1 + float(CFG.instrument.INTEGRATION_S)


def cells_over(t0, t1, cells):
    """The do-not-delete cells whose UTC range overlaps the data of the item of UT hour [t0, t1) - of ANY subband (module
    docstring)."""
    a, b = item_span(t0, t1)
    return [c for c in cells if c['start'] < b and c['end'] > a]


def describe(cells):
    """One line for the cells: '19h 2025-05-04 (2025-05-04T12:02:37-13:02:27 UTC, data_decisions.csv lines 2085-2099:
    <reasons>)' per cell, '; ' between cells."""
    import time
    return '; '.join('%s %s (%s-%s UTC, data_decisions.csv lines %s: %s)' % (
        c['lst'], c['date'], time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(c['start'])),
        time.strftime('%H:%M:%S', time.gmtime(c['end'])), _lines(c['lines']), ' / '.join(c['reasons'])) for c in cells)


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
                      describe(sorted(named.values(), key=lambda c: c['start']))))
