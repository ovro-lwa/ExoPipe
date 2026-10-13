"""Snapshot item kind for `exopipe3 curves` (core.cells.parse_night_cell / parse_epoch, refactor decision 5).

A snapshot CELL is a night cell ``<LST>_<YYYY-MM-DD>`` (one LST hour of one night, e.g. ``18h_2025-04-11``); an item may name
one subband, ``<LST>_<YYYY-MM-DD>_<sb>`` (``18h_2025-04-11_36MHz``, the same as ``--scopes 36MHz`` for that cell); a band
suffix (Red, ...) is refused here (the light-curve product is per subband). ``--epochs`` items are per-hour epochs
``<LST>_<date>`` (core.cells.parse_epoch; DESIGN 12.1): for this step the snapshot cell of that hour. The universe used to
expand patterns (``18h_*``, ``*_2025-04-*``) is every cell directory under paths.SNAPSHOT_FLEET_OUT that holds at least one
admission table (``<cell>/admission_<sb>.csv``: the cell was searched).
"""
import glob
import os
import re

_CELL_RE = re.compile(r'^(?P<lst>\d\dh)_(?P<date>\d{4}-\d{2}-\d{2})$')


def subbands():
    from exopipe.phase3.config import CFG
    return tuple(CFG.instrument.ALL_SUBBANDS)


def parse_item(item):
    """'18h_2025-04-11' -> ('18h', '2025-04-11', None); '18h_2025-04-11_36MHz' -> ('18h', '2025-04-11', '36MHz')."""
    from exopipe.phase3.core import cells
    lst, date, sb = cells.parse_night_cell(item or '')
    cells.parse_lst(lst)
    cells.parse_date(date)
    if sb is not None and sb not in subbands():
        raise ValueError(f'{item!r}: the light-curve product is per subband - {sb!r} is not one of {", ".join(subbands())}')
    return lst, date, sb


def parse_epoch_item(item):
    """A per-hour epoch '<LST>_<date>' (core.cells.parse_epoch) -> ('18h', '2025-04-11', None)."""
    from exopipe.phase3.core import cells
    lst, date = cells.parse_epoch(item or '')
    cells.parse_lst(lst)
    return lst, date, None


def universe(search_out_root):
    """Cells under the search output root holding at least one admission table (sorted)."""
    out = []
    try:
        names = os.listdir(search_out_root)
    except FileNotFoundError:
        return out
    for name in sorted(names):
        m = _CELL_RE.match(name)
        if m and glob.glob(os.path.join(search_out_root, name, 'admission_*.csv')):
            out.append(name)
    return out


def _needs_universe(spec):
    return spec is not None and any(c in str(spec) for c in '*?[')


def expand(spec, search_out_root, parse=parse_item, kind='snapshot cell'):
    """Expand an item spec (core/cells grammar: list, @file, patterns, record:STATUS:STEP@OUT, JOB:STATUS) to snapshot items.
    Patterns are matched against cells and against cell_subband items of the universe."""
    from exopipe.phase3.core import cells
    uni = None
    if _needs_universe(spec):
        base = universe(search_out_root)
        uni = base + ([f'{c}_{sb}' for c in base for sb in subbands()] if parse is parse_item else [])
    items = cells.expand(spec, universe=uni, kind=kind)
    for it in items:
        parse(it)
    return items


def select(cells_spec, epochs_spec, exclude_spec, scopes_spec, search_out_root):
    """The units to run: [(cell, subbands or None)] in order. ``subbands`` None = every subband; a set when the
    item names a subband and/or --scopes limits them. --exclude-cells removes whole cells (``<cell>``) or one subband of a
    cell (``<cell>_<sb>``)."""
    items = expand(cells_spec, search_out_root) + expand(epochs_spec, search_out_root, parse_epoch_item, 'epoch')
    excl = expand(exclude_spec, search_out_root) if exclude_spec else []
    scopes = None
    if scopes_spec:
        scopes = [s.strip() for s in str(scopes_spec).split(',') if s.strip()]
        bad = [s for s in scopes if s not in subbands()]
        if bad:
            raise ValueError(f'--scopes takes subband labels only ({", ".join(subbands())}); not {", ".join(bad)}')
        scopes = set(scopes)
    excl_cells = {e for e in excl if parse_item(e)[2] is None}
    excl_sb = {}
    for e in excl:
        lst, date, sb = parse_item(e)
        if sb:
            excl_sb.setdefault(f'{lst}_{date}', set()).add(sb)
    per_cell = {}
    order = []
    for it in items:
        lst, date, sb = parse_item(it)
        cell = f'{lst}_{date}'
        if cell in excl_cells:
            continue
        if cell not in per_cell:
            per_cell[cell] = set() if sb else None
            order.append(cell)
        elif per_cell[cell] is not None and not sb:
            per_cell[cell] = None          # the whole cell was also named
        if sb and per_cell[cell] is not None:
            per_cell[cell].add(sb)
    out = []
    for cell in order:
        sbs = per_cell[cell]
        if scopes is not None:
            sbs = set(scopes) if sbs is None else (sbs & scopes)
        if cell in excl_sb:
            sbs = (set(subbands()) if sbs is None else sbs) - excl_sb[cell]
        out.append((cell, None if sbs is None else sorted(sbs, key=subbands().index)))
    return out


def unit_name(cell, sbs, stokes):
    """Run-record unit name: the cell, plus the subband list when limited and '_I' when only Stokes I is run (a limited run
    never overwrites the whole cell's record)."""
    name = cell
    if sbs is not None:
        name += '_' + '+'.join(sbs)
    if tuple(stokes) == ('I',):
        name += '_I'
    return name
