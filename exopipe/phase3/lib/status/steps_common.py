"""Shared helpers of `exopipe3 status` (refactor window): output placement, the status write gates (S-G2, S-G3, S-G5,
S-G11), item lists, atomic CSV writes, header reads, the hub registry and run-directory resolution.

Nothing here decides a science product. Every path comes from config (sections status*, paths, serve, site).
"""
import csv
import fnmatch
import glob
import hashlib
import json
import os
import socket
import time

from exopipe.phase3.core.record import Refusal, NoInput

csv.field_size_limit(1 << 30)


def cfg():
    from exopipe.phase3.config import CFG
    return CFG


def utc_stamp():
    return time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())


# ---------------------------------------------------------------------------------------------------------------------
# placeholders and placement
# ---------------------------------------------------------------------------------------------------------------------

def expand(template, **kw):
    """Expand {qa} {hub} {p2} {ops} {root} {sublst} plus caller keys in a config template."""
    C = cfg()
    base = {'qa': C.status.QA_DIR, 'hub': hub_dir(), 'p2': C.paths.PHASE2_ROOT, 'ops': C.paths.OPS_ROOT,
            'root': C.paths.COADD_SB_ROOT, 'sublst': C.paths.SUBLST_ROOT}
    base.update(kw)
    return template.format(**base)


def hub_dir():
    C = cfg()
    serve = getattr(C, 'serve', None)
    h = getattr(serve, 'HUB_DIR', None) if serve is not None else None
    if not h:
        raise Refusal('serve.HUB_DIR is not defined (config section serve): the hub folder status reads/writes has no '
                      'root - no fallback root (R-COADD-ROOT-NO-DEFAULTS class, S-G11)')
    return h


def is_production_qa(out):
    """True when --out IS the production QA directory (after cutover only: production names are used directly)."""
    return os.path.realpath(out) == os.path.realpath(cfg().status.QA_DIR)


def place(out, kind, name):
    """Where a status product goes. kind: 'QA', 'review_hub', 'clv15', 'status', 'snapshot', 'items'.
    Production placement (out == status.QA_DIR, after cutover) puts QA -> QA_DIR, review_hub -> serve.HUB_DIR,
    clv15 -> paths.CLV15_ROOT; every other --out keeps one relative layout <out>/<kind>/<name> (the parity product map)."""
    C = cfg()
    if is_production_qa(out):
        if kind == 'QA':
            return os.path.join(C.status.QA_DIR, name)
        if kind == 'review_hub':
            return os.path.join(hub_dir(), name)
        if kind == 'clv15':
            return os.path.join(C.paths.CLV15_ROOT, name)
    return os.path.join(out, kind, name)


def unit_name(action, stage, tier, items=None):
    """<action>_<stage>_<tier>[_<8 hex of sha1(selected items)>] (a narrowed run never overwrites a full run's record)."""
    base = f"{action}_{stage.replace('-', '')}_{tier or 'none'}"
    if items:
        base += '_' + hashlib.sha1('\n'.join(sorted(items)).encode()).hexdigest()[:8]
    return base


# ---------------------------------------------------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------------------------------------------------

def _under(path, root):
    p, r = os.path.realpath(os.path.abspath(path)), os.path.realpath(os.path.abspath(root))
    return p == r or p.startswith(r.rstrip('/') + '/')


def check_write(path):
    """S-G3 (any time) + S-G2 (window): refuse a planned output in status.NEVER_WRITE, or - while
    paths.PRODUCTION_WRITES is False - under paths.PRODUCTION_ROOTS or equal to status.PROTECTED_OUTPUTS."""
    C = cfg()
    for p in C.status.NEVER_WRITE:
        if os.path.realpath(path) == os.path.realpath(p):
            raise Refusal(f'S-G3: {path} is never written by status (status.NEVER_WRITE: only Gregg writes verdicts; '
                          f'the installed solar audit is a pinned gate input)')
    if C.paths.PRODUCTION_WRITES:
        return path
    for p in C.status.PROTECTED_OUTPUTS:
        if os.path.realpath(path) == os.path.realpath(p):
            raise Refusal(f'S-G2: {path} is a protected production output (status.PROTECTED_OUTPUTS) and '
                          f'paths.PRODUCTION_WRITES is False (refactor window)')
    root = C.production_root_of(path)
    if root is not None:
        raise Refusal(f'S-G2: {path} is under the production root {root} and paths.PRODUCTION_WRITES is False '
                      f'(refactor window): pass --out to a parity/scratch root')
    return path


def node_local_owner(path):
    """(host, root) when ``path`` lies under a node-local root of the site file (NODE_LOCAL_ROOTS), else None."""
    for host, roots in (cfg().site.NODE_LOCAL_ROOTS or {}).items():
        for r in roots:
            if _under(path, r):
                return host, r
    return None


def need_inputs(paths, what):
    """S-G5: a stage whose input is node-local and absent on this host refuses with the host named; S-G11: an empty root
    key refuses (no fallback root)."""
    me = socket.gethostname().split('.')[0]
    for p in paths:
        if not p:
            raise Refusal(f'S-G11: {what}: an input root key is empty - no fallback root (R-COADD-ROOT-NO-DEFAULTS)')
        if os.path.exists(p):
            continue
        own = node_local_owner(p)
        if own and own[0] != me:
            raise Refusal(f'S-G5: {what} reads {p}, node-local to {own[0]} ({own[1]}); this host is {me}. Run this stage '
                          f'on {own[0]}')
        raise Refusal(f'{what}: input {p} not found')


def exclusions():
    """S-G1: the exclusion list (core/exclusions; a missing or unreadable file refuses, ruling C-09)."""
    from exopipe.phase3.core import exclusions as E
    return E.load()


# ---------------------------------------------------------------------------------------------------------------------
# tables and item lists
# ---------------------------------------------------------------------------------------------------------------------

def write_csv(path, columns, rows):
    """Atomic CSV write (tmp + rename). ``rows``: dicts; missing keys written empty."""
    check_write(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f'{path}.part.{os.getpid()}'
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(columns), extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: ('' if r.get(k) is None else r.get(k)) for k in columns})
    os.replace(tmp, path)
    return path


def write_text(path, text):
    check_write(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f'{path}.part.{os.getpid()}'
    with open(tmp, 'w') as f:
        f.write(text)
    os.replace(tmp, path)
    return path


def read_csv(path):
    with open(path, newline='', encoding='utf-8', errors='replace') as f:
        return list(csv.DictReader(f))


ITEM_KINDS = ('cell', 'epoch', 'lst', 'night_cell', 'unit')


def item_of(kind, lst=None, scope=None, date=None, unit=None):
    """The item of a row in the given kind (core/cells grammar)."""
    if kind == 'cell':
        return f'{lst}_{scope}' if lst and scope else None
    if kind == 'epoch':
        return f'{lst}_{date}' if lst and date else None
    if kind == 'night_cell':
        if not (lst and date):
            return None
        return f'{lst}_{date}' + (f'_{scope}' if scope and scope.endswith('MHz') else '')
    if kind == 'lst':
        return lst
    if kind == 'unit':
        return unit
    raise ValueError(f'unknown item kind {kind!r} ({", ".join(ITEM_KINDS)})')


def row_item(kind, r):
    """The item a table row names in the given kind: a stale-table row (column 'class') its 'item'; a deep row
    <lst>_<scope|band>; a snapshot row its night 'cell'; else item_of over lst / scope|band / date."""
    if 'class' in r and r.get('item'):
        return r['item']
    if kind == 'night_cell' and r.get('cell') and r['cell'][3:4] == '_' and r['cell'][4:5].isdigit():
        return r['cell']
    return item_of(kind, lst=r.get('lst'), scope=r.get('scope') or r.get('band'), date=r.get('date'),
                   unit=r.get('item') or r.get('unit'))


def parse_item(kind, item):
    """Validate one item of its declared kind with core/cells (raises ValueError)."""
    from exopipe.phase3.core import cells
    if kind == 'cell':
        cells.parse_cell(item)
    elif kind == 'epoch':
        cells.parse_epoch(item)
    elif kind == 'night_cell':
        cells.parse_night_cell(item)
    elif kind == 'lst':
        cells.parse_lst(item)
    elif kind == 'unit':
        if not item or any(c.isspace() for c in item):
            raise ValueError(f'not a unit name: {item!r}')
    else:
        raise ValueError(f'unknown item kind {kind!r}')
    return item


def write_items(path, kind, items, source_table, qualifies, stage):
    """An item list usable as --cells/--epochs/--lsts @FILE: '#' header lines (kind, producing table, qualifying
    status), then one item per line, sorted and unique. core/cells._read_list ignores the '#' lines."""
    its = sorted({i for i in items if i})
    for i in its:
        parse_item(kind, i)
    head = [f'# exopipe3 status item list: stage {stage}; kind {kind}; {len(its)} items; written {utc_stamp()}',
            f'# kind: {kind}', f'# table: {source_table}', f'# qualifies: {qualifies}']
    write_text(path, '\n'.join(head) + '\n' + ''.join(f'{i}\n' for i in its))
    return its


def read_items(path):
    """(header dict, items) of an item list written by write_items."""
    head, items = {}, []
    with open(path) as f:
        for line in f:
            s = line.rstrip('\n')
            if s.startswith('#'):
                if ':' in s[1:]:
                    k, v = s[1:].split(':', 1)
                    head[k.strip()] = v.strip()
                continue
            s = s.split('#', 1)[0].strip()
            if s:
                items.append(s.split(',')[0].strip())
    return head, items


# ---------------------------------------------------------------------------------------------------------------------
# FITS headers (header-only reads; no astropy)
# ---------------------------------------------------------------------------------------------------------------------

def fits_history(path, max_blocks=400):
    """HISTORY card texts of the primary header (order kept), read block by block up to END."""
    out = []
    with open(path, 'rb') as f:
        for n in range(max_blocks):
            blk = f.read(2880)
            if len(blk) < 2880 or (n == 0 and not blk.startswith(b'SIMPLE')):
                return out
            for i in range(0, 2880, 80):
                card = blk[i:i + 80].decode('ascii', 'replace')
                key = card[:8].strip()
                if key == 'END':
                    return out
                if key == 'HISTORY':
                    out.append(card[8:].rstrip())
    return out


def coadd_constituents(path):
    """Dates of the 'COADD constituent YYYY-MM-DD' HISTORY lines of a master (coadd_exclusion_conformance.py:66 rule)."""
    return [h.split('constituent')[-1].strip() for h in fits_history(path) if 'COADD constituent' in h]


def fits_cards(path):
    from exopipe.phase3.core.record import fits_primary_cards
    return fits_primary_cards(path)


# ---------------------------------------------------------------------------------------------------------------------
# hub registry and run directories
# ---------------------------------------------------------------------------------------------------------------------

def registry():
    p = expand(cfg().status.REGISTRY_FILE)
    with open(p) as f:
        return p, json.load(f)


def served_runs(tier):
    """{run name: [registry rows]} served for the tier according to the hub registry (read only, B-56)."""
    _, reg = registry()
    keys = cfg().status.REGISTRY_KEYS.get(tier, ())
    out = {}
    for S in ('V', 'I'):
        for k in keys:
            row = (reg.get(S) or {}).get(k)
            if isinstance(row, dict) and row.get('run'):
                out.setdefault(row['run'], []).append(dict(row, stokes=S, key=k))
    return out


def runs_root(tier):
    C = cfg()
    serve = getattr(C, 'serve', None)
    key = (getattr(serve, 'RUN_ROOT_KEY', {}) or {}).get(tier) if serve is not None else None
    if not key:
        key = {'deep': 'TRANSIENTS_ROOT', 'sublst': 'TRANSIENTS_ROOT', 'snapshot': 'SNAPSHOT_REVIEW_ROOT'}[tier]
    return getattr(C.paths, key)


def run_tier(name):
    """Tier of a run directory name: Run_SLST* = sublst, Run_SNAP* = snapshot, any other Run_* = deep."""
    b = os.path.basename(os.path.normpath(name))
    if b.startswith('Run_SLST'):
        return 'sublst'
    if b.startswith('Run_SNAP'):
        return 'snapshot'
    return 'deep'


def resolve_run(spec, tier):
    """A run label (P3S5), a run name (Run_P3S5) or a directory -> absolute run directory."""
    if os.path.isdir(spec):
        return os.path.abspath(spec)
    name = spec if spec.startswith('Run_') else f'Run_{spec}'
    for t in (tier, 'deep', 'sublst', 'snapshot'):
        try:
            p = os.path.join(runs_root(t), name)
        except Exception:
            continue
        if os.path.isdir(p):
            return p
    raise NoInput(f'run {spec!r}: no directory {name} under the run roots of tier {tier}')


def glob_cells(universe, spec):
    """Filter a universe of item strings by a --cells spec (literals, patterns, @file)."""
    from exopipe.phase3.core import cells
    if not spec:
        return list(universe)
    want = cells.expand(spec, universe=list(universe), kind='item')
    pats = [w for w in want if any(c in w for c in '*?[')]
    lit = set(w for w in want if w not in pats)
    return [u for u in universe if u in lit or any(fnmatch.fnmatch(u, p) for p in pats)]


def thread_map(fn, items, workers=None):
    from concurrent.futures import ThreadPoolExecutor
    items = list(items)
    if not items:
        return []
    n = max(1, min(workers or cfg().status.WORKERS, len(items)))
    with ThreadPoolExecutor(max_workers=n) as ex:
        return list(ex.map(fn, items))


def listdir(path):
    try:
        return sorted(os.listdir(path))
    except OSError:
        return []


def newest(paths):
    return sorted(paths)[-1] if paths else None


def glob1(pattern):
    return sorted(glob.glob(pattern))
