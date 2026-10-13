"""The parity comparator behind `exopipe3 verify compare ROOT_A ROOT_B` (DESIGN 11.1, 11.2; critique_coverage gap 9).

Walks both roots, pairs files by relative path (or through a product map ``old,new``), compares each pair by class and
classifies every difference as permitted (a named signature in the permitted-differences file, or a built-in
provenance rule) or not. The verdict is PASS only when every difference is permitted.

Classes:
- FITS (.fits .fit .fts .fits.gz), every HDU: data identity by md5 of the contiguous array; otherwise shape, the NaN
  mask (pixels finite on one side only), max |A-B| over pixels finite in both, and max |A-B| / rms(A) (the migration
  critique's measure, not maxrel with a floor); header card by card with an ignore list (DATE, DATE-MAP, CHECKSUM,
  DATASUM, COMMENT, blank) and HISTORY compared after normalising timestamps, run ids and the two roots; provenance
  cards (GITTAG, ORIGIN, SCRIPT, PIPEVER) are differences permitted by the built-in rule PD-GITTAG.
- CSV: columns, row count, rows paired by key columns (--key; otherwise rows sorted and paired in order), every cell
  compared exactly or with a numeric tolerance (atol + rtol*|A|); NaN equals NaN; root paths normalised.
- JSON: deep equality with ignore keys (run ids, timestamps, hosts, pids, commits - see JSON_IGNORE_KEYS); numbers
  with the same tolerance.

Permitted-differences file (CSV; '#' comment lines ignored), one signature per row:

    id,ledger_id,class,path,kind,key,tolerance,note
    PD-K-TENMIN,R-GREGG-20261013-TENMIN-TEMPLATE-K-MEASURED,csv,*/forced_*.csv,cell,ra_err*,,measured K

- class: fits | csv | json | * ;  path: shell pattern on either relative path;  key: shell pattern on the header card,
  CSV column or JSON key path;  tolerance (optional): the difference is permitted only when its value (max |A-B|, or
  a count) is at or below it.
- kind: data, nanmask, shape, dtype, hdus, header, history (FITS); column, rows, cell (CSV); json (JSON); missing (in
  A, not in B), extra (in B, not in A); * for any.

Rules reused from the golden-cell harness exopipe_ops/golden_cells/bin/diff_classify.py (2026-08-13, sha256
2d3ab8ea71279900ada6132f993f902f67461485b7daf69a69d13c1b9272351d), COPIED here, not imported (exopipe_ops is ops):
md5_data (data identity), cmp_fits (header loop, IGNORE_KEYS, PERMITTED_HDR -> PD-GITTAG, exact float compare with NaN
equal to NaN), keyed / cmp_ledger (keyed CSV rows, reference-only / run-only rows, numeric cell tolerance). Changed
from the source: HISTORY is compared after normalisation instead of ignored; the NaN mask is compared instead of
nan_to_num; the relative measure is max|A-B|/rms(A); the default CSV tolerance is exact (the harness used rtol 2e-3);
every HDU is compared, not only the primary.
"""
import csv
import fnmatch
import hashlib
import json
import math
import os
import re
import time

SOURCE = 'exopipe_ops/golden_cells/bin/diff_classify.py'
SOURCE_SHA256 = '2d3ab8ea71279900ada6132f993f902f67461485b7daf69a69d13c1b9272351d'

FITS_SUFFIXES = ('.fits', '.fit', '.fts', '.fits.gz')
KIND_SUFFIXES = {'fits': FITS_SUFFIXES, 'csv': ('.csv',), 'json': ('.json',)}
# diff_classify.IGNORE_KEYS minus HISTORY (HISTORY is normalised and compared below)
IGNORE_CARDS = frozenset({'DATE', 'DATE-MAP', 'CHECKSUM', 'DATASUM', 'COMMENT', ''})
# diff_classify.PERMITTED_HDR: provenance cards of the writing tree (permitted_diffs.md PD-GITTAG)
PROVENANCE_CARDS = {'GITTAG': 'PD-GITTAG', 'ORIGIN': 'PD-GITTAG', 'SCRIPT': 'PD-GITTAG', 'PIPEVER': 'PD-GITTAG'}
JSON_IGNORE_KEYS = frozenset({'run_id', 'started', 'finished', 'created', 'wall_s', 'pid', 'host', 'user', 'argv',
                              'commit', 'site_sha1', 'code', 'launch_host', 'jobs_root', 'pkg_root', 'log',
                              'config_hash_all', 'mtime_ns'})
_ISO = re.compile(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?')
_COMPACT = re.compile(r'(?<!\d)\d{8}T\d{6}Z?(?!\d)')     # also inside run ids (masters_20261013T031000Z_...)
_RUN_ID = re.compile(r'\b\w+_<UTC>_[\w.-]+_\d+\b')
REPORT_COLUMNS = ('level', 'cls', 'kind', 'a', 'b', 'key', 'detail', 'value', 'permitted_by')


class Options:
    """Comparison settings (defaults: exact)."""

    def __init__(self, csv_keys=(), rtol=0.0, atol=0.0, fits_atol=0.0, fits_rel_rms=0.0, ignore_cards=(),
                 ignore_keys=(), include_records=False, max_rows_per_key=20):
        self.csv_keys = tuple(csv_keys or ())
        self.rtol, self.atol = float(rtol), float(atol)
        self.fits_atol, self.fits_rel_rms = float(fits_atol), float(fits_rel_rms)
        self.ignore_cards = frozenset(c.upper() for c in ignore_cards or ())
        self.ignore_keys = frozenset(ignore_keys or ())
        self.include_records = bool(include_records)
        self.max_rows_per_key = int(max_rows_per_key)


def kind_of(path):
    p = path.lower()
    for k, suf in KIND_SUFFIXES.items():
        if p.endswith(suf):
            return k
    return None


# ---- permitted differences and product map --------------------------------------------------------------------------

BUILTIN_RULES = tuple({'id': pid, 'ledger_id': '', 'class': 'fits', 'path': '*', 'kind': 'header', 'key': card,
                       'tolerance': '', 'note': f'built-in: provenance card ({SOURCE} PERMITTED_HDR)'}
                      for card, pid in PROVENANCE_CARDS.items())


def _data_lines(path):
    with open(path, newline='') as f:
        return [line for line in f if line.strip() and not line.lstrip().startswith('#')]


def load_permitted(path):
    """Signatures of the permitted-differences CSV (see the module docstring)."""
    if not path:
        return []
    rows = list(csv.DictReader(_data_lines(path)))
    if rows and not {'id', 'kind'} <= set(rows[0]):
        raise ValueError(f'{path}: a permitted-differences file is a CSV with columns '
                         f'id,ledger_id,class,path,kind,key,tolerance,note (got {list(rows[0])})')
    out = []
    for r in rows:
        r = {k: (v or '').strip() for k, v in r.items() if k}
        if not r.get('id'):
            raise ValueError(f'{path}: every signature needs an id ({r})')
        for k in ('ledger_id', 'class', 'path', 'kind', 'key', 'tolerance', 'note'):
            r.setdefault(k, '')
        if r['tolerance']:
            float(r['tolerance'])
        out.append(r)
    return out


def load_map(path):
    """Product map: CSV rows 'old,new' (a header row old,new optional) -> {old relative path: new relative path}."""
    if not path:
        return {}
    m = {}
    for row in csv.reader(_data_lines(path)):
        if len(row) < 2:
            continue
        a, b = row[0].strip(), row[1].strip()
        if (a, b) == ('old', 'new'):
            continue
        m[a] = b
    return m


def _match(rule, f):
    if rule['class'] not in ('', '*') and rule['class'] != f['cls']:
        return False
    if rule['kind'] not in ('', '*') and rule['kind'] != f['kind']:
        return False
    if rule['path'] not in ('', '*') and not (fnmatch.fnmatch(f['a'] or '', rule['path'])
                                              or fnmatch.fnmatch(f['b'] or '', rule['path'])):
        return False
    if rule['key'] not in ('', '*') and not fnmatch.fnmatch(str(f['key'] or ''), rule['key']):
        return False
    if rule['tolerance']:
        v = f.get('value')
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not abs(v) <= float(rule['tolerance']):
            return False
    return True


# ---- walking and pairing --------------------------------------------------------------------------------------------

def _walk(root, kinds, include_records):
    out = {}
    for dp, dn, fn in os.walk(root):
        dn[:] = sorted(d for d in dn if d != '__pycache__' and (include_records or d != '_records'))
        for f in sorted(fn):
            k = kind_of(f)
            if k in kinds:
                p = os.path.join(dp, f)
                out[os.path.relpath(p, root)] = k
    return out


def pair_files(root_a, root_b, kinds, mapping=None, include_records=False):
    """(pairs [(rel_a, rel_b, kind)], only_a [rel], only_b [rel]). A map entry pairs a renamed product."""
    fa = _walk(root_a, kinds, include_records)
    fb = _walk(root_b, kinds, include_records)
    mapping = {(os.path.relpath(a, root_a) if os.path.isabs(a) else os.path.normpath(a)):
               (os.path.relpath(b, root_b) if os.path.isabs(b) else os.path.normpath(b))
               for a, b in (mapping or {}).items()}
    pairs, only_a, used = [], [], set()
    for ra, k in sorted(fa.items()):
        rb = mapping.get(ra, ra)
        if rb in fb:
            pairs.append((ra, rb, k))
            used.add(rb)
        else:
            only_a.append(ra)
    only_b = sorted(rb for rb in fb if rb not in used)
    return pairs, only_a, only_b


# ---- the comparison -------------------------------------------------------------------------------------------------

class Report:
    """Findings of one comparison. ``rows`` keeps up to Options.max_rows_per_key rows per (file pair, kind, key) plus a
    summary row; the counts and the verdict cover every finding."""

    def __init__(self, root_a, root_b, cls, rules, opts):
        self.root_a, self.root_b, self.cls, self.rules, self.opts = root_a, root_b, cls, rules, opts
        self.rows = []
        self.counts = {'OK': 0, 'DIFF': 0, 'PERM': 0, 'INFO': 0}
        self.by_kind = {}
        self.rules_used = {}
        self.pairs = 0
        self.files_with_diffs = set()
        self._per_key = {}
        self._roots = sorted({os.path.abspath(root_a), os.path.abspath(root_b), os.path.realpath(root_a),
                              os.path.realpath(root_b)}, key=len, reverse=True)

    def norm(self, s):
        """A string with both roots replaced by <ROOT> (products name their own root)."""
        for r in self._roots:
            s = s.replace(r, '<ROOT>')
        return s

    def emit(self, level, cls, a, b, kind, key='', detail='', value=None):
        f = {'level': level, 'cls': cls, 'kind': kind, 'a': a, 'b': b, 'key': key, 'detail': detail,
             'value': value, 'permitted_by': ''}
        if level == 'DIFF':
            for rule in self.rules:
                if _match(rule, f):
                    f['level'] = 'PERM'
                    f['permitted_by'] = rule['id'] + (f" ({rule['ledger_id']})" if rule.get('ledger_id') else '')
                    self.rules_used[rule['id']] = self.rules_used.get(rule['id'], 0) + 1
                    break
        self.counts[f['level']] += 1
        if f['level'] in ('DIFF', 'PERM'):
            self.by_kind.setdefault(kind, {'DIFF': 0, 'PERM': 0})[f['level']] += 1
            self.files_with_diffs.add((a, b))
            k = (a, b, kind, key, f['level'])
            n = self._per_key.get(k, 0) + 1
            self._per_key[k] = n
            if n > self.opts.max_rows_per_key:
                return f
        self.rows.append(f)
        return f

    def finish(self):
        for (a, b, kind, key, level), n in self._per_key.items():
            if n > self.opts.max_rows_per_key:
                self.rows.append({'level': 'INFO', 'cls': '', 'kind': kind, 'a': a, 'b': b, 'key': key,
                                  'detail': f'{n - self.opts.max_rows_per_key} more {level} rows not listed '
                                            f'({n} in all)', 'value': n, 'permitted_by': ''})

    @property
    def verdict(self):
        if self.counts['DIFF']:
            return 'FAIL'
        if not self.pairs and not self.counts['PERM']:
            return 'NOTHING'
        return 'PASS'

    def summary(self):
        return {'verdict': self.verdict, 'root_a': os.path.abspath(self.root_a), 'root_b': os.path.abspath(self.root_b),
                'class': self.cls, 'pairs_compared': self.pairs, 'pairs_with_differences': len(self.files_with_diffs),
                'unpermitted_differences': self.counts['DIFF'], 'permitted_differences': self.counts['PERM'],
                'identical_or_within_tolerance': self.counts['OK'], 'by_kind': self.by_kind,
                'permitted_rules_used': self.rules_used, 'rules_source': SOURCE, 'rules_source_sha256': SOURCE_SHA256,
                'options': {k: (sorted(v) if isinstance(v, frozenset) else v) for k, v in vars(self.opts).items()},
                'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}


def compare_roots(root_a, root_b, cls='auto', permitted=None, mapping=None, opts=None):
    """Compare two product roots; returns a Report (``report.verdict``: PASS / FAIL / NOTHING)."""
    opts = opts or Options()
    kinds = tuple(KIND_SUFFIXES) if cls == 'auto' else (cls,)
    rep = Report(root_a, root_b, cls, list(load_permitted(permitted)) + list(BUILTIN_RULES), opts)
    for root in (root_a, root_b):
        if not os.path.isdir(root):
            raise ValueError(f'not a directory: {root}')
    pairs, only_a, only_b = pair_files(root_a, root_b, kinds, load_map(mapping) if isinstance(mapping, str)
                                       else mapping, opts.include_records)
    for ra in only_a:
        rep.emit('DIFF', kind_of(ra), ra, '', 'missing', '', 'in ROOT_A only (no counterpart in ROOT_B)')
    for rb in only_b:
        rep.emit('DIFF', kind_of(rb), '', rb, 'extra', '', 'in ROOT_B only (no counterpart in ROOT_A)')
    for ra, rb, k in pairs:
        rep.pairs += 1
        pa, pb = os.path.join(root_a, ra), os.path.join(root_b, rb)
        before = rep.counts['DIFF'] + rep.counts['PERM']
        try:
            {'fits': _cmp_fits, 'csv': _cmp_csv, 'json': _cmp_json}[k](pa, pb, ra, rb, rep)
        except Exception as e:                              # an unreadable file is a difference, never a skip
            rep.emit('DIFF', k, ra, rb, 'unreadable', '', f'{type(e).__name__}: {e}')
        if rep.counts['DIFF'] + rep.counts['PERM'] == before:
            rep.emit('OK', k, ra, rb, 'file', '', 'identical' if k != 'fits' else 'pixel-identical, headers match')
    rep.finish()
    return rep


# ---- FITS -----------------------------------------------------------------------------------------------------------

def _md5_data(arr):
    """diff_classify.md5_data: md5 of the contiguous data array."""
    import numpy as np
    return hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _same_value(va, vb):
    """diff_classify.cmp_fits: floats compared exactly, NaN equal to NaN; other values by equality."""
    if isinstance(va, float) and isinstance(vb, float):
        return va == vb or (math.isnan(va) and math.isnan(vb))
    return va == vb


def _norm_history(line, rep):
    s = _ISO.sub('<UTC>', rep.norm(line))
    s = _COMPACT.sub('<UTC>', s)
    return _RUN_ID.sub('<RUN_ID>', s)


def _cmp_header(ha, hb, pre, a, b, rep):
    ignore = IGNORE_CARDS | rep.opts.ignore_cards | {'HISTORY'}
    for k in sorted((set(ha.keys()) | set(hb.keys())) - ignore):
        if k not in ha or k not in hb:
            rep.emit('DIFF', 'fits', a, b, 'header', pre + k,
                     f'card only in {"B" if k not in ha else "A"}: {hb.get(k) if k not in ha else ha.get(k)!r}')
            continue
        va, vb = ha.get(k), hb.get(k)
        if isinstance(va, str) and isinstance(vb, str):
            va, vb = rep.norm(va), rep.norm(vb)
        if not _same_value(va, vb):
            rep.emit('DIFF', 'fits', a, b, 'header', pre + k, f'{va!r} -> {vb!r}')
    # astropy wraps a long HISTORY value over several cards (a root path can be split between two): the cards are
    # joined before normalising, and compared as one text
    raw_a = [str(x) for x in (ha['HISTORY'] if 'HISTORY' in ha else [])]
    raw_b = [str(x) for x in (hb['HISTORY'] if 'HISTORY' in hb else [])]
    ta, tb = _norm_history(''.join(raw_a), rep), _norm_history(''.join(raw_b), rep)
    if ta != tb:
        i = next((j for j, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
        n = abs(len(raw_a) - len(raw_b)) + sum(1 for x, y in zip(raw_a, raw_b)
                                               if _norm_history(x, rep) != _norm_history(y, rep))
        rep.emit('DIFF', 'fits', a, b, 'history', pre + 'HISTORY',
                 f'HISTORY differs after normalising timestamps/run ids/roots ({len(raw_a)} vs {len(raw_b)} cards); '
                 f'first at character {i}: {ta[max(0, i - 30):i + 40]!r} -> {tb[max(0, i - 30):i + 40]!r}',
                 value=max(n, 1))


def _cmp_data(xa, xb, pre, a, b, rep):
    import numpy as np
    da, db = xa.data, xb.data
    if da is None and db is None:
        return
    if (da is None) != (db is None):
        rep.emit('DIFF', 'fits', a, b, 'data', pre, f'data in {"A" if db is None else "B"} only')
        return
    image = getattr(da, 'dtype', None) is not None and da.dtype.names is None
    if not image:                                       # tables: byte identity only
        if np.asarray(da).tobytes() != np.asarray(db).tobytes():
            rep.emit('DIFF', 'fits', a, b, 'data', pre, 'table data differ')
        return
    if da.shape != db.shape:
        rep.emit('DIFF', 'fits', a, b, 'shape', pre, f'{da.shape} vs {db.shape}')
        return
    if da.dtype == db.dtype and _md5_data(da) == _md5_data(db):
        return
    x, y = np.asarray(da, dtype=np.float64), np.asarray(db, dtype=np.float64)
    fx, fy = np.isfinite(x), np.isfinite(y)
    nmask = int(np.count_nonzero(fx != fy))
    if nmask:
        rep.emit('DIFF', 'fits', a, b, 'nanmask', pre, f'{nmask} pixel(s) finite on one side only', value=nmask)
    both = fx & fy
    d = np.abs(x[both] - y[both])
    nd = int(np.count_nonzero(d))
    if nd:
        mx = float(d.max())
        rms = float(np.std(x[fx])) if fx.any() else 0.0
        rel = mx / rms if rms > 0 else float('inf')
        detail = f'n={nd} maxabs={mx:.3e} maxabs/rms={rel:.3e} (rms of A {rms:.3e})'
        o = rep.opts
        if mx <= o.fits_atol or (o.fits_rel_rms > 0 and rel <= o.fits_rel_rms):
            rep.emit('OK', 'fits', a, b, 'data', pre, 'within the comparison tolerance: ' + detail, value=mx)
        else:
            rep.emit('DIFF', 'fits', a, b, 'data', pre, detail, value=mx)
    elif not nmask and da.dtype != db.dtype:
        rep.emit('DIFF', 'fits', a, b, 'dtype', pre, f'same values, dtype {da.dtype} vs {db.dtype}')


def _cmp_fits(pa, pb, a, b, rep):
    from astropy.io import fits
    with fits.open(pa, memmap=True) as ha, fits.open(pb, memmap=True) as hb:
        if len(ha) != len(hb):
            rep.emit('DIFF', 'fits', a, b, 'hdus', '', f'{len(ha)} vs {len(hb)} HDUs', value=abs(len(ha) - len(hb)))
        for i in range(min(len(ha), len(hb))):
            pre = f'[{i}]' if i else ''
            _cmp_header(ha[i].header, hb[i].header, pre, a, b, rep)
            _cmp_data(ha[i], hb[i], pre, a, b, rep)


# ---- CSV ------------------------------------------------------------------------------------------------------------

def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _same_cell(x, y, rep):
    if x == y:
        return True, None
    fx, fy = _num(x), _num(y)
    if fx is not None and fy is not None:
        if math.isnan(fx) and math.isnan(fy):
            return True, None
        d = abs(fx - fy)
        return d <= rep.opts.atol + rep.opts.rtol * abs(fx), d
    return rep.norm(x or '') == rep.norm(y or ''), None


def _cmp_csv(pa, pb, a, b, rep):
    with open(pa, newline='') as f:
        ra = csv.DictReader(f)
        cols_a, rows_a = list(ra.fieldnames or []), list(ra)
    with open(pb, newline='') as f:
        rb = csv.DictReader(f)
        cols_b, rows_b = list(rb.fieldnames or []), list(rb)
    for c in cols_a:
        if c not in cols_b:
            rep.emit('DIFF', 'csv', a, b, 'column', c, 'column in A only')
    for c in cols_b:
        if c not in cols_a:
            rep.emit('DIFF', 'csv', a, b, 'column', c, 'column in B only')
    if len(rows_a) != len(rows_b):
        rep.emit('DIFF', 'csv', a, b, 'rows', '', f'row count A={len(rows_a)} B={len(rows_b)}',
                 value=abs(len(rows_a) - len(rows_b)))
    common = [c for c in cols_a if c in cols_b and c not in rep.opts.ignore_keys]
    keys = [k for k in rep.opts.csv_keys if k in cols_a and k in cols_b]
    if rep.opts.csv_keys and len(keys) != len(rep.opts.csv_keys):
        rep.emit('INFO', 'csv', a, b, 'rows', '', f'key columns {list(rep.opts.csv_keys)} not in both files: rows '
                                                  f'paired after sorting')
        keys = []
    if keys:                                             # diff_classify.keyed / cmp_ledger
        def keyed(rows, side):
            out = {}
            for r in rows:
                k = tuple(r.get(c, '') for c in keys)
                if k in out:
                    rep.emit('DIFF', 'csv', a, b, 'rows', ','.join(keys), f'duplicate key {k} in {side}')
                out[k] = r
            return out
        ka, kb = keyed(rows_a, 'A'), keyed(rows_b, 'B')
        for k in ka:
            if k not in kb:
                rep.emit('DIFF', 'csv', a, b, 'rows', ','.join(keys), f'row in A only: {k}')
        for k in kb:
            if k not in ka:
                rep.emit('DIFF', 'csv', a, b, 'rows', ','.join(keys), f'row in B only: {k}')
        pairs = [(k, ka[k], kb[k]) for k in ka if k in kb]
    else:
        sa = sorted(rows_a, key=lambda r: tuple(r.get(c) or '' for c in common))
        sb = sorted(rows_b, key=lambda r: tuple(r.get(c) or '' for c in common))
        pairs = [(i, x, y) for i, (x, y) in enumerate(zip(sa, sb))]
    for k, x, y in pairs:
        for c in common:
            ok, d = _same_cell(x.get(c), y.get(c), rep)
            if not ok:
                rep.emit('DIFF', 'csv', a, b, 'cell', c, f'row {k}: {x.get(c)!r} vs {y.get(c)!r}', value=d)


# ---- JSON -----------------------------------------------------------------------------------------------------------

def _cmp_json(pa, pb, a, b, rep):
    with open(pa) as f:
        ja = json.load(f)
    with open(pb) as f:
        jb = json.load(f)
    ignore = JSON_IGNORE_KEYS | rep.opts.ignore_keys

    def walk(x, y, path):
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y), key=str):
                if k in ignore:
                    continue
                p = f'{path}.{k}' if path else str(k)
                if k not in x or k not in y:
                    rep.emit('DIFF', 'json', a, b, 'json', p, f'key in {"A" if k in x else "B"} only')
                else:
                    walk(x[k], y[k], p)
        elif isinstance(x, list) and isinstance(y, list):
            if len(x) != len(y):
                rep.emit('DIFF', 'json', a, b, 'json', path, f'list length {len(x)} vs {len(y)}',
                         value=abs(len(x) - len(y)))
            for i, (u, v) in enumerate(zip(x, y)):
                walk(u, v, f'{path}[{i}]')
        elif (isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool)
              and not isinstance(y, bool)):
            if x != y and not (isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y)):
                d = abs(x - y)
                if d > rep.opts.atol + rep.opts.rtol * abs(x):
                    rep.emit('DIFF', 'json', a, b, 'json', path, f'{x!r} vs {y!r}', value=d)
        elif isinstance(x, str) and isinstance(y, str):
            if rep.norm(x) != rep.norm(y):
                rep.emit('DIFF', 'json', a, b, 'json', path, f'{x!r} vs {y!r}')
        elif x != y or type(x) is not type(y):
            rep.emit('DIFF', 'json', a, b, 'json', path, f'{x!r} vs {y!r}')

    walk(ja, jb, '')


# ---- report ---------------------------------------------------------------------------------------------------------

def write_report(rep, outdir):
    """parity_report.csv (one row per finding), parity_summary.json and parity_summary.txt in ``outdir``."""
    os.makedirs(outdir, exist_ok=True)
    csv_p = os.path.join(outdir, 'parity_report.csv')
    with open(csv_p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=REPORT_COLUMNS)
        w.writeheader()
        for r in rep.rows:
            w.writerow({k: ('' if r.get(k) is None else r.get(k)) for k in REPORT_COLUMNS})
    s = rep.summary()
    with open(os.path.join(outdir, 'parity_summary.json'), 'w') as f:
        json.dump(s, f, indent=1, sort_keys=True)
    lines = [f"verdict: {s['verdict']}", f"ROOT_A: {s['root_a']}", f"ROOT_B: {s['root_b']}", f"class: {s['class']}",
             f"file pairs compared: {s['pairs_compared']} ({s['pairs_with_differences']} with differences)",
             f"unpermitted differences: {s['unpermitted_differences']}",
             f"permitted differences: {s['permitted_differences']} "
             + (f"(rules: {', '.join(f'{k} x{v}' for k, v in sorted(s['permitted_rules_used'].items()))})"
                if s['permitted_rules_used'] else ''),
             'by kind: ' + ', '.join(f"{k} {v['DIFF']} unpermitted / {v['PERM']} permitted"
                                     for k, v in sorted(s['by_kind'].items())),
             f"rules from {SOURCE} (sha256 {SOURCE_SHA256[:16]}...)", f'report: {csv_p}']
    with open(os.path.join(outdir, 'parity_summary.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return csv_p, lines
