"""Shell logic of the deep publish pass, ported to Python (DESIGN 11.1: launchers are ported, modules are called in place).

Each function names the lines it replaces. The two heredocs (PYS split, PYU0 union) are reproduced byte for byte: same
reads, same csv module calls, same row order and line endings (tests/test_catalog-deep.py runs the production heredoc
text and this port on the same synthetic run and compares the files). The differences are the ones the refactor asks
for and that change no product: the run directory is a parameter (the heredocs hard-code
/data07/gh/phase3/transients/Run_<label>, ops/p3s3_pass.sh:68, :105), the union reads only the files the associate
records list (no glob, DESIGN 11.5 / D9), and every file is written atomically (tmp + rename).
"""
import csv
import glob
import json
import os


def _atomic_dictwriter_rows(path, fieldnames, rows, extrasaction='raise'):
    tmp = f'{path}.tmp.{os.getpid()}'
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction=extrasaction)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def split_selection(run_dir, stokes=('V', 'I'), tiers=('wb', 'sb')):
    """ops/p3s3_pass.sh:66-76 (heredoc PYS): selected_all.csv -> selected_<S>_<tier>.csv, tier sb = 'MHz' in band.
    Returns {(S, tier): rows}. A split that cannot be written raises (D10: the pass aborts)."""
    rows = list(csv.DictReader(open(f'{run_dir}/selected_all.csv')))
    out = {}
    for st in stokes:
        for tier in tiers:
            sub = [r for r in rows if r['stokes'] == st and (('MHz' in r['band']) == (tier == 'sb'))]
            _atomic_dictwriter_rows(f'{run_dir}/selected_{st}_{tier}.csv',
                                    list(rows[0].keys()) if rows else ['cand_id'], sub)
            out[(st, tier)] = len(sub)
    return out


def union_bands(files_by_tier, tiers):
    """The bands of the union: every <band> with a <band>_sources_survivors.csv in a primary tier (PYU0 line 112)."""
    return sorted({os.path.basename(p).split('_sources_')[0] for K in tiers for p in files_by_tier.get(K, ())
                   if p.endswith('_sources_survivors.csv')})


def union_tiers(run_dir, files_by_tier, tiers, kinds, out_dir=None):
    """ops/p3s3_pass.sh:103-129 (heredoc PYU0, D9 fix 09995b5): the FOUR PRIMARY TIERS only, one row per (Stokes,
    source_id), each row carrying its 'stokes'; the header is every tier's columns in first-seen order (+ 'stokes').
    ``files_by_tier`` = {K: [paths]} - the files the associate record of tier K lists (a file not listed does not
    exist for the union). Returns [written paths]."""
    out_dir = out_dir or f'{run_dir}/source_catalogs'
    listed = {K: {os.path.abspath(p) for p in files_by_tier.get(K, ())} for K in tiers}
    written = []
    for b in union_bands(files_by_tier, tiers):
        for kind in kinds:
            rows = []
            hdr = []
            seen = set()
            for K in tiers:
                p = f'{run_dir}/source_catalogs_{K}/{b}_{kind}.csv'
                if os.path.abspath(p) not in listed[K]:
                    continue
                for r in csv.DictReader(open(p)):
                    for c in r:
                        if c not in hdr:
                            hdr.append(c)
                    k = (K[0], r.get('source_id') or tuple(r.values()))
                    if k in seen:
                        continue
                    seen.add(k)
                    r['stokes'] = K[0]
                    rows.append(r)
            if hdr:
                if 'stokes' not in hdr:
                    hdr.append('stokes')
                os.makedirs(out_dir, exist_ok=True)
                path = f'{out_dir}/{b}_{kind}.csv'
                _atomic_dictwriter_rows(path, hdr, rows, extrasaction='ignore')
                written.append(path)
    return written


# ---- what the in-place modules will read (compared with the records before they run: no stale file may enter) -------

def identity_reads(run_dir, skip_dirs):
    """The survivors files ops/p3s3_identity.py reads (:22-31) and the union copies it rewrites (:92)."""
    tier = [f for f in sorted(glob.glob(f'{run_dir}/source_catalogs_*/*_sources_survivors.csv'))
            if os.path.basename(os.path.dirname(f)) not in skip_dirs]
    union = sorted(glob.glob(f'{run_dir}/source_catalogs/*_sources_survivors.csv'))
    return {os.path.abspath(f) for f in tier}, {os.path.abspath(f) for f in union}


def localize_reads(run_dir, pattern):
    """The survivors files ops/build_localizations.py (:21) and source_localization.py (:112) read."""
    return {os.path.abspath(p) for p in glob.glob(f'{run_dir}/{pattern}/*_sources_survivors.csv')}


def list_reads(run_dir, tiers):
    """The survivors files ops/p3s2_transient_list.py reads (:33-40)."""
    out = set()
    for K in tiers:
        out |= {os.path.abspath(p) for p in glob.glob(f'{run_dir}/source_catalogs_{K}/*_sources_survivors.csv')
                if 'ALL_' not in p}
    return out


# ---- harvest (ops/p3s3_harvest_published.sh) -------------------------------------------------------------------------

def attempt_dir(label, node, cell, st, att, collect_roots):
    """ops/p3s3_sources.attempt_dir (:30-40): the first collect root that holds the attempt, else the last root's path."""
    rel = f'{label.lower()}_collect_{node}/attempts/{cell}_{st}_{att}'
    for r in collect_roots:
        if os.path.isdir(f'{r}/{rel}'):
            return f'{r}/{rel}'
    return f'{collect_roots[-1]}/{rel}'


def parse_sources_output(text, label, collect_roots):
    """The lines `ops/p3s3_sources.py <S...>` prints (its __main__: 'cell stokes node att harvested|NOT-HARVESTED', one per
    published (cell, Stokes), last entry winning - exactly what ops/p3s3_harvest_published.sh:24 reads) ->
    [{cell, stokes, node, att, dir, harvested}]. The module is run in place (core.legacy.run); its directory rule is
    re-applied here only to name the directory (the harvested flag is the module's)."""
    out = []
    for line in text.splitlines():
        q = line.split()
        if len(q) != 5 or q[4] not in ('harvested', 'NOT-HARVESTED') or q[1] not in ('V', 'I'):
            continue
        cell, st, node, att, state = q
        out.append(dict(cell=cell, stokes=st, node=node, att=att, harvested=state == 'harvested',
                        dir=attempt_dir(label, node, cell, st, att, collect_roots)))
    return out


def harvest_commands(a, label, host, dest_root, staging):
    """The rsync of one published, unharvested attempt (ops/p3s3_harvest_published.sh:16-23): local when the attempt's
    node is this host, else over ssh with SSH_AUTH_SOCK cleared (CLAUDE.md 8; the shell lacked the prefix).
    Returns (destination dir, argv, env additions, remote?)."""
    low = label.lower()
    cell, st, node, att = a['cell'], a['stokes'], a['node'], a['att']
    dest = f'{dest_root}/{low}_collect_{node}/attempts/{cell}_{st}_{att}'
    src = f'{staging}/{cell}_{st}_{att}'
    if node == host:
        argv = ['rsync', '-a', f'{src}/Run_{label}', f'{src}/transients', f'{src}/{cell}.log', f'{dest}/']
        return dest, argv, {}, False
    argv = ['rsync', '-a', f'{node}:{src}/Run_{label}', f'{node}:{src}/transients', f'{node}:{src}/{cell}.log', f'{dest}/']
    return dest, argv, {'SSH_AUTH_SOCK': ''}, True


def copy_state(src, dst, keep_cell=None):
    """The state the merge reads and the staleness gate edits (published_<S>.csv, markers_<S>/), copied for a run outside
    the production roots so the gate drops markers only in the copy (spec 1.3 item 3). ``keep_cell(cell)`` filters the
    published lists (the --cells test filter). Returns {'published_V': n, ...}."""
    import shutil
    os.makedirs(dst, exist_ok=True)
    n = {}
    for st in ('V', 'I'):
        p = os.path.join(src, f'published_{st}.csv')
        if os.path.exists(p):
            keep = []
            for line in open(p):
                if not line.strip():
                    continue
                cell = line.strip().split(',')[0]
                if keep_cell is None or keep_cell(cell):
                    keep.append(line if line.endswith('\n') else line + '\n')
            with open(os.path.join(dst, f'published_{st}.csv'), 'w') as f:
                f.writelines(keep)
            n[f'published_{st}'] = len(keep)
        m = os.path.join(src, f'markers_{st}')
        if os.path.isdir(m):
            md = os.path.join(dst, f'markers_{st}')
            if os.path.isdir(md):
                shutil.rmtree(md)
            shutil.copytree(m, md, symlinks=True)
            n[f'markers_{st}'] = len(os.listdir(md))
    return n


# ---- preflight (ops/fork_serve_p3s5_20261008.sh:20-30) ---------------------------------------------------------------

def parents_missing(par, coadd_root):
    """The installed cells (a master under <coadd_root>/<cell>/coadd_final/, as ops/p3s5_parent_union.py:16-20 scans them)
    whose catalogue or sidecar is not in the snapshot ``par``. Machine-readable replacement of the regex over the printed
    list (fork_serve:24); the digest currency of each catalogue is the parent_snapshot content test and gate_inputs."""
    cells, miss = [], []
    for cdir in sorted(glob.glob(f'{coadd_root}/[0-2][0-9]h_*')):
        cell = os.path.basename(cdir)
        lst, sc = cell.split('_', 1)
        if not os.path.exists(f'{cdir}/coadd_final/I_deep_Taper_R075_{sc}.fits'):
            continue
        cells.append(cell)
        name = f'{par}/master_R075_{lst}_{sc}.gaul.csv'
        if not (os.path.exists(name) and os.path.exists(name + '.gen.json')):
            miss.append(cell)
    return cells, miss


def parents_manifest(par, path):
    """Write the snapshot's manifest (cell, catalogue, sha1, master, master_md5_data): the record of what conformance and
    the fork read through P3_PARENT_DIR."""
    import hashlib
    rows = []
    for c in sorted(glob.glob(f'{par}/master_R075_*.gaul.csv')):
        cell = os.path.basename(c)[len('master_R075_'):-len('.gaul.csv')]
        side = c + '.gen.json'
        g = json.load(open(side)) if os.path.exists(side) else {}
        rows.append({'cell': cell, 'catalogue': c, 'sha1': hashlib.sha1(open(c, 'rb').read()).hexdigest(),
                     'master': g.get('master', ''), 'master_md5_data': g.get('master_md5_data', '')})
    _atomic_dictwriter_rows(path, ['cell', 'catalogue', 'sha1', 'master', 'master_md5_data'], rows)
    return len(rows)
