"""Cell, LST, band and item-list grammar shared by every step.

A deep CELL is ``<LST>_<scope>``: LST ``00h``..``23h``; scope a subband label (``18MHz`` .. ``82MHz``, nominal - exact
frequencies in ``CFG.instrument.SUBBAND_FREQ_HZ``) or a band (``Red``, ``Green``, ``Blue``, ``Full``). Band labels are
nominal groupings of subbands, never physical frequency ranges (architecture 0.1).

Item specs (``--cells``, ``--lsts``, ``--nights``, ...): comma-separated tokens, each one of
- a literal (``04h_Full``), or a shell-style pattern matched against the known universe (``04h_*``, ``*_Red``);
- ``@FILE``: one item per line (``#`` comments and blank lines ignored; the first comma-separated field is the item);
- ``RECORD:STATUS:STEP[@OUT]``: the units of STEP whose record status is STATUS under OUT (e.g. ``record:refused:search``)
- ``JOB:STATUS``: items of an `exopipe3 run` job with that status (``failed:20261013T0100_search``).
"""
import fnmatch
import glob
import json
import os
import re

LSTS = tuple(f'{h:02d}h' for h in range(24))
BANDS = ('Red', 'Green', 'Blue', 'Full')
_CELL_RE = re.compile(r'^(?P<lst>\d\dh)_(?P<scope>\d+MHz|Red|Green|Blue|Full)$')
_DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')


def subbands():
    from exopipe.phase3.config import CFG
    return tuple(CFG.instrument.ALL_SUBBANDS)


def scopes():
    return tuple(subbands()) + BANDS


def all_cells():
    return tuple(f'{l}_{s}' for l in LSTS for s in scopes())


def parse_cell(cell):
    """'04h_Full' -> ('04h', 'Full'); refuses anything else."""
    m = _CELL_RE.match(cell)
    if not m or m['scope'] not in scopes():
        raise ValueError(f'not a cell: {cell!r} (expected <LST>_<subband|band>, e.g. 04h_Full, 13h_41MHz)')
    return m['lst'], m['scope']


def is_band(scope):
    return scope in BANDS


def band_members(scope):
    """Member subbands of a band scope (Full = every science subband); a subband is its own member."""
    from exopipe.phase3.config import CFG
    if scope == 'Full':
        return tuple(CFG.instrument.ALL_SUBBANDS)
    if scope in CFG.instrument.SUBBAND_BANDS:
        return tuple(CFG.instrument.SUBBAND_BANDS[scope])
    return (scope,)


def parse_date(s):
    if not _DATE_RE.match(s):
        raise ValueError(f'not a date YYYY-MM-DD: {s!r}')
    return s


def _read_list(path):
    items = []
    with open(path) as f:
        for line in f:
            line = line.split('#', 1)[0].strip()
            if line:
                items.append(line.split(',')[0].strip())
    return items


def _record_query(token):
    # record:STATUS:STEP[@OUT]
    _, status, rest = token.split(':', 2)
    step, _, out = rest.partition('@')
    if not out:
        raise ValueError(f'{token}: give the product root, record:{status}:{step}@/path/to/out')
    units = []
    for p in sorted(glob.glob(os.path.join(out, '_records', step, '*.json'))):
        with open(p) as f:
            rec = json.load(f)
        if rec.get('status') == status:
            units.append(rec['unit'])
    return units


def _job_query(token):
    from exopipe.phase3.core import runner
    job, status = token.rsplit(':', 1)
    return runner.job_items(job, status)


def expand(spec, universe=None, kind='item'):
    """Expand an item spec to a sorted, de-duplicated list. Patterns need a universe (e.g. all_cells())."""
    if spec is None:
        return []
    out = []
    for tok in [t.strip() for t in str(spec).split(',') if t.strip()]:
        if tok.startswith('@'):
            out.extend(_read_list(tok[1:]))
        elif tok.startswith('record:'):
            out.extend(_record_query(tok))
        elif re.match(r'^[A-Za-z0-9_.-]+:(failed|refused|done|queued|running|no-input|error|held)$', tok):
            out.extend(_job_query(tok))
        elif any(c in tok for c in '*?['):
            if universe is None:
                raise ValueError(f'pattern {tok!r} needs a universe of known {kind}s')
            hits = fnmatch.filter(universe, tok)
            if not hits:
                raise ValueError(f'pattern {tok!r} matches no {kind}')
            out.extend(hits)
        else:
            out.append(tok)
    seen, res = set(), []
    for x in out:
        if x not in seen:
            seen.add(x)
            res.append(x)
    return res


def expand_cells(spec):
    cells = expand(spec, universe=all_cells(), kind='cell')
    for c in cells:
        parse_cell(c)
    return cells


# ---------------------------------------------------------------------------------------------------------------------
# Named item kinds (crosscheck 4.4). A step declares its kind in --help; the runner chains fan items out per kind.
#   cell        deep  <LST>_<scope>            04h_Full, 13h_41MHz
#   lst         any   <LST>                    04h
#   epoch       all   <LST>_<date>             04h_2025-10-14   (one hour of one night: all bands, all tiers; DESIGN 12)
#   night_cell  snapshot / sub-LST <LST>_<date>[_<subband>]   18h_2025-04-11, 18h_2025-04-11_36MHz
# ---------------------------------------------------------------------------------------------------------------------
_EPOCH_RE = re.compile(r'^(?P<lst>\d\dh)_(?P<date>\d{4}-\d{2}-\d{2})(?:_(?P<sb>\d+MHz|Red|Green|Blue|Full))?$')


def parse_epoch(item):
    """'04h_2025-10-14' -> ('04h', '2025-10-14'); refuses anything else (a subband suffix is not an epoch)."""
    m = _EPOCH_RE.match(item)
    if not m or m['sb']:
        raise ValueError(f'not an epoch item: {item!r} (expected <LST>_<YYYY-MM-DD>, e.g. 04h_2025-10-14)')
    return m['lst'], m['date']


def parse_night_cell(item):
    """'18h_2025-04-11' or '18h_2025-04-11_36MHz' -> (lst, date, subband-or-None)."""
    m = _EPOCH_RE.match(item)
    if not m or (m['sb'] and m['sb'] not in scopes()):
        raise ValueError(f'not a night cell: {item!r} (expected <LST>_<YYYY-MM-DD>[_<subband|band>])')
    return m['lst'], m['date'], m['sb']


def parse_lst(item):
    if item not in LSTS:
        raise ValueError(f'not an LST: {item!r} (00h..23h)')
    return item


def expand_epochs(spec):
    items = expand(spec, kind='epoch')
    for it in items:
        parse_epoch(it)
    return items


def epoch_lsts(spec):
    """The LSTs the epochs of an --epochs value name (a cell of another LST is not selected by it)."""
    return {parse_epoch(e)[0] for e in expand_epochs(spec)}


# ---------------------------------------------------------------------------------------------------------------------
# The UTC time range of an epoch (one LST hour of one night)
# ---------------------------------------------------------------------------------------------------------------------
def _phase1_scheduler():
    """Phase 1's LSTScheduler (exopipe/phase1/pipeline_controller.py), imported without letting that module's import-time
    logging.basicConfig (its own command line's log format) take over this process's root logger."""
    import importlib
    import logging
    import sys
    name = 'exopipe.phase1.pipeline_controller'
    if name not in sys.modules:
        root = logging.getLogger()
        handlers, level = list(root.handlers), root.level
        try:
            importlib.import_module(name)
        finally:
            root.handlers[:] = handlers
            root.setLevel(level)
    return sys.modules[name].LSTScheduler()


_EPOCH_UTC = {}


def epoch_utc(lst, date):
    """(start, end), seconds since 1970 UTC, of the LST hour ``lst`` ('19h') of the epoch dated ``date`` (YYYY-MM-DD): Phase 1's
    own conversion, the UTC range a Phase 1 run with --range <h>h-<h+1>h --date <date> processes
    (exopipe/phase1/pipeline_controller.LSTScheduler.parse_range): from the first time at or after 00:00 UTC of ``date`` when
    the mean local sidereal time at the observatory (common/pipeline_config.OVRO_LOC) is h hours, for one sidereal hour
    (3590.17 s). Example: 19h of 2025-05-04 is 12:02:37-13:02:27 UTC on 2025-05-04. ValueError on an LST that is not
    00h..23h, a date that is not YYYY-MM-DD, or a range Phase 1 cannot compute."""
    parse_lst(lst)
    parse_date(date)
    key = (lst, date)
    if key not in _EPOCH_UTC:
        h = int(lst[:2])
        try:
            t0, t1 = _phase1_scheduler().parse_range('%dh-%dh' % (h, h + 1), date)
        except SystemExit:               # parse_range logs and exits on what it cannot parse (a command line's behavior)
            raise ValueError(f'Phase 1 cannot convert LST {lst} on {date} to UTC')
        _EPOCH_UTC[key] = (float(t0.unix), float(t1.unix))
    return _EPOCH_UTC[key]
