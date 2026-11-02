"""Search attempt directories: their names, their order, and which one is current - one grammar for lib/search/deep_step,
lib/search/tests and lib/frames/deep_step.

Production writes an attempt as <collect root>/<label>_collect_<host>/attempts/<cell>_<S>_<TAG>, TAG one of:
  <UTC>        YYYYmmddTHHMMSS: a search attempt (exopipe_ops p3s5_search_fleet.sh:194 'date -u +%Y%m%dT%H%M%S'; the search step
               names its attempts the same way);
  <UTC>stdrop  the search-tier drop patch of 2026-10-30 (exopipe_ops p3s5_searchtier_drop_20261030.py:65): a hard-linked copy of the
               cell's published attempt in which the epochs the 2.0 x median gate newly refuses are dropped - their catalogue, ledger
               and candidate rows removed, their image rows in the search's refused form, refusal rows tagged <UTC>stdrop - and
               every other row is its source's; each of its logs names the source ('P3S5-STDROP <tag>: ...; source attempt <dir>;').
               142 attempts (108 I + 34 V), each the last register row of its (cell, Stokes) (census 2026-11-02). The tool
               rewrites only the catalogue, the two ledgers, the searchtier refusal file, the candidates and the logs; every other
               file is a hard link to its source's. A future patch made from a source holding search_table_record.csv (written
               since production exopipe 55b0263, 2026-10-18) would carry the source's verdicts of the dropped nights, whose
               gate-ledger rows it removed, and the search_table_record content test (lib/search/tests.py) would refuse it; none of
               the 142 holds one (their sources were searched 2026-10-04 to 10-13).
A '<...>.part' directory is a copy in progress (11 on disk) and never an attempt. No other form exists among the 3,446 attempt
directories of p3s4, p3s5 and p3t4 (census 2026-11-02). (The frames step's own attempts add '_<n>' when two start in one second:
lib/frames/deep_step._new_attempt.)
ORDER: attempts are ordered by their UTC - production's merge orders a cell's attempts by att[:15] (p3s3_merge.py) - and a drop
patch's UTC is later than its source's.
CURRENT: the last row of the register published_<S>.csv for (cell, Stokes) - production's rule (p3s3_sources.published_attempts:
'last entry wins'; the drop tool appends its row, :125-126). A drop patch therefore supersedes the attempt it patched and a later
re-search supersedes the patch; one attempt is current at a time (lib/search/deep_step.resolve_published reads the registers so).
"""
import os
import re

UTC_RX = r'\d{8}T\d{6}'
SUFFIXES = ('stdrop',)
TAG_RX = UTC_RX + '(?:' + '|'.join(SUFFIXES) + ')?'
NAME = re.compile(r'^(?P<cell>\d\dh_(?:\d+MHz|Red|Green|Blue|Full))_(?P<s>[IV])_(?P<att>(?P<utc>' + UTC_RX + r')(?P<suffix>'
                  + '|'.join(SUFFIXES) + r')?)$')
STDROP_LOG = re.compile(r'^P3S5-STDROP (?P<tag>\S+): .*; source attempt (?P<src>\S+);')


def parse(path):
    """The NAME match of an attempt directory (its basename), or None."""
    return NAME.match(os.path.basename(str(path).rstrip('/')))


def utc(tag):
    """The UTC (YYYYmmddTHHMMSS) of an attempt tag, or None when the tag is not of the grammar."""
    m = re.match('^(' + UTC_RX + ')(?:' + '|'.join(SUFFIXES) + ')?$', tag or '')
    return m.group(1) if m else None


def stdrop_source(attempt_dir):
    """The directory a drop-patched attempt was made from, as its logs name it (the line of its own tag), or None (not a drop patch,
    or no such line)."""
    m = parse(attempt_dir)
    if not m or m['suffix'] != 'stdrop' or not os.path.isdir(attempt_dir):
        return None
    for fn in sorted(os.listdir(attempt_dir)):
        if fn.endswith('.log'):
            with open(os.path.join(attempt_dir, fn), errors='replace') as fh:
                for ln in fh:
                    x = STDROP_LOG.match(ln)
                    if x and x['tag'] == m['att']:
                        return x['src']
    return None


def searched_attempt(attempt_dir, _depth=0):
    """The attempt whose search wrote an attempt's rows: a search attempt itself; for a drop patch its source (its rows are the
    source's; the dropped epochs' image rows keep the source's values), followed through a patch of a patch - the path the logs name,
    which may be gone from disk; None when it cannot be told (no attempt name, or a drop patch whose logs name no source)."""
    m = parse(attempt_dir)
    if not m:
        return None
    if not m['suffix']:
        return attempt_dir
    src = stdrop_source(attempt_dir) if _depth < 8 else None
    return searched_attempt(src, _depth + 1) if src else None


def rows_utc(attempt_dir):
    """The UTC of the search whose rows an attempt holds (searched_attempt) - what an expectation by code generation must use - or
    None."""
    s = searched_attempt(attempt_dir)
    return parse(s)['utc'] if s else None


def register_rows(path, cell):
    """[(host, tag)] of ``cell`` in a register published_<S>.csv (lines cell,host,tag), in file order; [] when there is none."""
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for line in fh:
            parts = line.strip().split(',')
            if len(parts) >= 3 and parts[0] == cell:
                out.append((parts[1], parts[2]))
    return out
