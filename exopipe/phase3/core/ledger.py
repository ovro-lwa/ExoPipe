"""Read-only lookup of ledger IDs in Gregg's fix ledger (``CFG.paths.FIX_LEDGER``, clv15/fix_ledger.csv).

Used by config.py to check ``--ruling-for SECTION.KEY=LEDGER_ID``: an override of a ruled key is accepted with a
ruling only when that ID is a row of the ledger. The ledger is append-only and written by hand, so the reader is
tolerant: CRLF line ends, a UTF-8 byte-order mark, rows with fewer or more fields than the header, and SHIFTED rows
whose ID sits in the date column (``R-XREF-REJECTED-TIME-FREQ,2026-09-20,...``; 151 such rows on 2026-10-13) are all
read. The file is never written here.
"""
import csv
import os
import re

_DATE = re.compile(r'^\d{4}-\d{2}-\d{2}')
_ID = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._:+/-]*$')     # an ID is one token (no spaces)
_CACHE = {}


def ids(path):
    """The set of ledger IDs in ``path`` (the fix_id column, plus the first field of shifted rows)."""
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_mtime_ns, st.st_size)
    if key in _CACHE:
        return _CACHE[key]
    out = set()
    with open(path, newline='', encoding='utf-8', errors='replace') as f:
        rd = csv.reader(f)
        header = [h.strip().lstrip('﻿').lower() for h in next(rd, [])]
        col = header.index('fix_id') if 'fix_id' in header else 1
        for row in rd:
            row = [c.strip() for c in row]
            if len(row) > col and _ID.match(row[col]) and not _DATE.match(row[col]):
                out.add(row[col])
            if row and row[0] and not _DATE.match(row[0]) and _ID.match(row[0]):
                out.add(row[0])           # shifted row: the ID is in the date column
    _CACHE.clear()
    _CACHE[key] = out
    return out


def has_id(ledger_id, path):
    """True when ``ledger_id`` is a row of the ledger at ``path``. A missing or unreadable ledger raises OSError."""
    lid = (ledger_id or '').strip()
    return bool(lid) and lid in ids(path)
