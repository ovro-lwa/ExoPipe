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
