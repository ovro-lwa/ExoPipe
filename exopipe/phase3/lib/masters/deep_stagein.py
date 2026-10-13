"""Deep stage-in of `exopipe3 masters`: port of ops/membership_rebuild_20261007/rebuild_cell.sh:23, 34-43, 47-120 (the shell
logic and its heredoc Python, line by line; DESIGN 11.1 - shell is ported, Python modules are called in place). The copy of
the stored flows (:44, rsync -a) is the `flow` step's stage-in (refactor decision 6 / crosscheck 1.1 group 1), run after
this one; it reads the record of this stage (<out>/_records/masters/stage-in_<cell>.json).

What production does for one cell before the co-add stages (rebuild_cell.sh), and what this module does:
  :23      the working cell C = paths.COADD_SB_ROOT/<cell>, or the root ref_override.csv names for the cell
           -> ``working_cell()`` (table sha1-pinned, masters_stagein.REF_OVERRIDE_TABLE_SHA1)
  :34      refuse when the scratch cell exists                     -> G-M25 (``--fresh --supersede TAG`` moves it aside)
  :41-43   epoch-gate record carried in: epoch_gate_carry/<cell>/epoch_gate.csv, else C/coadd_final/epoch_gate.csv
           -> ``carry_epoch_gate()`` (B-05 default (c))
  :44      rsync -a C/flow/ -> R/<cell>/flow/                      -> NOT here: `exopipe3 flow --tier deep --stages stage-in`
  :47-117  the native, un-warped reference (I survey master + V master) written under the canonical names, four branches:
           (1) ref_override_I.csv: un-warped I of the same CONSTDIG, no V reference;
           (2) RBF-dewarped cell: the pre-dewarp master in working_quarantine_20261004 / its pixel factor;
           (3) flow-dewarped cell: exactly one superseded archive master with the CONSTDIG (native, un-warped);
           (4) otherwise the current master / its pixel factor (SHSCALE x FLUXCORR) - window 0.3-3, SH*/FLUX* stripped
           -> ``resolve_reference()``: one rule indexed by CONSTDIG/MDWARP - same constituents, else the previous
           generation (ALIGNREF; Gregg (A), e5c218e), else refuse; the hand tables are per-row cross-checks
           (same arithmetic: float64 / f -> float32; same HISTORY text)
  :118-120 copies into coadd_final_ref/ (kept for T1)              -> ``write_reference()``
--reference path:P (ruled waiver 'masters.reference') takes P (and its V sibling, if present) as the reference instead;
--reference night:YYYY-MM-DD writes no reference (the co-add then takes the fixed-table night; the build applies it).
"""
import csv
import glob
import json
import os
import shutil

import numpy as np

from exopipe.phase3.core.record import Refusal
from exopipe.phase3.lib.masters import deep_inplace as IP

REF_VARS = (('I', 'I_deep_Taper_R075', 'Robust-0.75'), ('V', 'V_deep_Taper_R0', 'Robust+0.0'))


def _cfg():
    return IP.cfg()


def working_root_table(pin=True):
    """ref_override.csv {cell: root}: the working cell's root when it is not paths.COADD_SB_ROOT/<cell>
    (rebuild_cell.sh:23), sha1-pinned (G-M25)."""
    s = _cfg().masters_stagein
    ov = {}
    if os.path.exists(s.REF_OVERRIDE_TABLE):
        if pin:
            IP.sha1_pin(s.REF_OVERRIDE_TABLE, s.REF_OVERRIDE_TABLE_SHA1, 'ref_override.csv')
        # awk -F, -v c=$cell '$1==c{print $2}' (rebuild_cell.sh:23): first column = cell, second = root; header row ignored
        with open(s.REF_OVERRIDE_TABLE) as f:
            for line in f:
                parts = line.rstrip('\n').split(',')
                if len(parts) >= 2 and parts[0] != 'cell':
                    ov.setdefault(parts[0], parts[1])
    return ov


def reference_tables():
    """Production's hand tables of the 2026-10 rounds, read as PER-ROW CROSS-CHECKS of resolve_reference (never as its
    input, never sha1-pinned - production edits them row by row): {cell: dict(table, constdig, I, V)}, V None = the table
    gives the cell no V reference. ref_prevgen.csv (cell,constdig,I,V; exopipe_ops e5c218e) takes precedence over
    ref_override_I.csv (cell,path; never a V), as rebuild_cell.sh:60-89. An empty table path switches that check off."""
    s = _cfg().masters_stagein
    rows = {}
    if s.REF_OVERRIDE_I_TABLE and os.path.exists(s.REF_OVERRIDE_I_TABLE):
        for r in csv.DictReader(open(s.REF_OVERRIDE_I_TABLE)):
            rows[r['cell']] = dict(table=os.path.basename(s.REF_OVERRIDE_I_TABLE), constdig=None, I=r['path'], V=None,
                                   generation='same-constituents')
    if s.REF_PREVGEN_TABLE and os.path.exists(s.REF_PREVGEN_TABLE):
        for r in csv.DictReader(open(s.REF_PREVGEN_TABLE)):
            rows[r['cell']] = dict(table=os.path.basename(s.REF_PREVGEN_TABLE), constdig=r['constdig'], I=r['I'],
                                   V=r['V'] or None, generation='previous-generation')
    return rows


def working_cell(cell, ov=None):
    """C of rebuild_cell.sh:19/23: the cell's working (installed) directory, or its ref_override root."""
    if ov is None:
        ov = working_root_table()
    return ov.get(cell) or os.path.join(_cfg().paths.COADD_SB_ROOT, cell)


def epoch_gate_source(cell, C):
    """rebuild_cell.sh:41: the carried record if epoch_gate_carry/<cell>/epoch_gate.csv exists, else the cell's own."""
    egc = os.path.join(_cfg().masters_stagein.EPOCH_GATE_CARRY_DIR, cell, 'epoch_gate.csv')
    if os.path.isfile(egc):
        return egc
    egc = os.path.join(C, 'coadd_final', 'epoch_gate.csv')
    return egc if os.path.isfile(egc) else None


def _strip(hh):
    pre = tuple(_cfg().masters_stagein.STRIP_CARD_PREFIXES)
    for k in [k for k in hh.keys() if k.startswith(pre)]:
        hh.remove(k, remove_all=True)


def _factor_ok(f):
    lo, hi = _cfg().masters_stagein.REF_PIXEL_FACTOR_RANGE
    return bool(f and np.isfinite(f) and lo < f < hi)


def alignref_digest(h):
    """The CONSTDIG in a master's ALIGNREF card ('MASTER:<generation>:<CONSTDIG>'): the previous-generation master it was
    flow-aligned to, i.e. the master it replaced (R-ALIGNMENT-REFERENCE-IS-PREVIOUS-GENERATION-MASTER). None when the
    master was aligned to a night (no previous generation)."""
    a = str(h.get('ALIGNREF') or '')
    return a.rsplit(':', 1)[-1] if a.startswith('MASTER:') and a.count(':') >= 2 else None


def _copy_paths(spec, cell, st, var, rob):
    """(directory, [candidate files]) of one masters_stagein.REFERENCE_COPY_ROOTS entry, newest directory first."""
    label, pattern, layout = spec
    lst, sc = cell.split('_', 1)
    dirs = sorted(glob.glob(pattern), reverse=True) if any(c in pattern for c in '*?[') else [pattern]
    for d in dirs:
        if layout == 'cell':
            yield d, [f'{d}/{cell}/coadd_final/{var}_{sc}.fits']
        elif layout == 'flat':
            yield d, [f'{d}/{cell}/{var}_{sc}.fits']
        elif layout == 'archive':
            yield d, sorted(glob.glob(f'{d}/{lst}/{sc}/{sc}_{st}_deep_Taper_{rob}_*.fits'))
        else:
            raise ValueError(f'REFERENCE_COPY_ROOTS {label}: unknown layout {layout!r} (cell, flat, archive)')


def find_prescreen_copy(cell, st, var, rob, target):
    """The first pre-screen (un-warped) master with CONSTDIG == target over masters_stagein.REFERENCE_COPY_ROOTS, in their
    order, newest directory first within a glob: (label, path, header), or None. Refuses: a root node-local to another
    host reached before any hit (a copy may sit there: run on that host); an archive directory with more than one master
    of the CONSTDIG (rebuild_cell.sh:96 'need exactly 1') or whose one master is not native (:97-98)."""
    from astropy.io import fits
    from exopipe.phase3.core import env
    for spec in _cfg().masters_stagein.REFERENCE_COPY_ROOTS:
        label, pattern, layout = spec
        if not env.readable_here(pattern.split('*')[0]):
            raise Refusal(f'{var}: no pre-screen copy with CONSTDIG {target} before {label} ({pattern}), which is '
                          f'node-local to {env.node_local_holder(pattern.split("*")[0])} - run the stage-in there')
        for d, paths in _copy_paths(spec, cell, st, var, rob):
            hdrs = [(p, fits.getheader(p)) for p in paths if os.path.exists(p)]
            if layout == 'archive':
                hits = [(p, h) for p, h in hdrs if h.get('CONSTDIG') == target]
                if len(hits) > 1:
                    raise Refusal(f'{var}: {len(hits)} superseded masters with CONSTDIG {target} - need exactly 1')
                if hits and (hits[0][1].get('SHSCALE') is not None or hits[0][1].get('MDWARP')
                             or hits[0][1].get('FLUXBAKE')):
                    raise Refusal(f'{hits[0][0]}: not native and un-warped')
            else:
                hits = [(p, h) for p, h in hdrs if h.get('CONSTDIG') == target and not h.get('MDWARP')]
            if hits:
                return label, hits[0][0], hits[0][1]
    return None


def _history(var, generation, label, path, f):
    """The HISTORY line production writes for each branch (rebuild_cell.sh:72, 89, 109, 113 and e5c218e)."""
    if generation == 'previous-generation':
        how = f'{path} / pixel factor {f:.6f}' if f is not None else f'{path}, native'
        return f'2026-10-13 rebuild reference: PREVIOUS-GENERATION pre-screen {var} ({how}; Gregg (A))'
    if label == 'unwarped-copy':
        return f'2026-10-09 rebuild reference: native un-warped {var} (override {path} / pixel factor {f:.6f})'
    if label == 'predewarp-quarantine':
        return (f'2026-10-04 rebuild reference: native un-warped {var} (pre-dewarp master (working quarantine 20261004) / '
                f'pixel factor {f:.6f})')
    if label == 'superseded-archive':
        return f'2026-10-04 rebuild reference: native un-warped {var} (superseded {path.split("/")[-1]})'
    if label == 'current-master':
        return f'2026-10-04 rebuild reference: native un-warped {var} (current master / pixel factor {f:.6f})'
    return f'rebuild reference: native un-warped {var} (pre-screen copy {path} / pixel factor {f:.6f}; same constituents)'


def resolve_reference(cell, C, read_data=True, tables=None):
    """The cell's native, un-warped reference: ONE rule over what is on disk, indexed by CONSTDIG and MDWARP (replaces
    the four branches and the hand tables of rebuild_cell.sh:47-117 + e5c218e; deep chat 2026-10-13):
      1 same constituents: the current master itself when it is not dewarped (current-master), else the first pre-screen
        copy of its CONSTDIG over REFERENCE_COPY_ROOTS (predewarp-quarantine, superseded-archive, unwarped-copy,
        quarantine-copy);
      2 previous generation (Gregg 2026-10-13 '(A) is approved', IMPL-REBUILD-REFERENCE-A-PREVIOUS-GENERATION-PRESCREEN-
        20261013): no copy of the current constituents -> the first pre-screen copy of the master the current one
        replaced, whose CONSTDIG is the current I master's ALIGNREF; the rebuilt master is re-screened as usual;
      3 otherwise refuse.
    V takes I's generation. A missing V copy means no V reference (the flow stage's diagnostic V rms is not measured;
    membership is unaffected) only where production allowed it (masters_stagein.V_OPTIONAL_BRANCHES); else refuse.
    A row of the hand tables (reference_tables(); pass {} to skip) must name the file chosen, else refuse.
    Returns one dict per Stokes: stokes, var, generation, branch, target, src, source, factor, constdig, ncoadd,
    history, data, header, none_reason. Raises Refusal on every refusal. read_data=False plans only (headers)."""
    from astropy.io import fits
    from exopipe.phase3.lib.masters import flux_correction as FC
    s = _cfg().masters_stagein
    if tables is None:
        tables = reference_tables()
    lst, sc = cell.split('_', 1)
    out, gen_I, branch_I = [], None, None
    for st, var, rob in REF_VARS:
        cur = f'{C}/coadd_final/{var}_{sc}.fits'
        if not os.path.exists(cur):
            raise Refusal(f'{var}: current master {cur} missing - no reference resolvable (G-M04)')
        hc = fits.getheader(cur)
        e = dict(stokes=st, var=var, generation=None, branch=None, target=None, src='', source=None, factor=None,
                 constdig=None, ncoadd=None, history='', data=None, header=None, none_reason='')
        hit, gen = None, None
        if st == 'I' or gen_I == 'same-constituents':
            gen, target = 'same-constituents', hc.get('CONSTDIG')
            if not hc.get('MDWARP'):
                hit = ('current-master', cur, hc)
            else:
                hit = find_prescreen_copy(cell, st, var, rob, target)
        if hit is None and st == 'I' and s.PREVIOUS_GENERATION_REFERENCE:
            gen, target = 'previous-generation', alignref_digest(hc)
            if target is None:
                raise Refusal(f'{var}: no pre-screen master of the current constituents (CONSTDIG {hc.get("CONSTDIG")}) and '
                              f'no previous generation recorded (ALIGNREF {hc.get("ALIGNREF")!r}) - refusing')
            hit = find_prescreen_copy(cell, st, var, rob, target)
        elif st == 'V' and gen_I == 'previous-generation':
            gen, target = gen_I, out[0]['target']
            hit = find_prescreen_copy(cell, st, var, rob, target)
        e.update(generation=gen, target=target)
        if hit is None:
            if st == 'I':
                raise Refusal(f'{var}: no pre-screen master of the current constituents (CONSTDIG {hc.get("CONSTDIG")})'
                              + (f' nor of the previous generation (CONSTDIG {target})' if gen == 'previous-generation'
                                 else '') + ' in masters_stagein.REFERENCE_COPY_ROOTS - refusing')
            if branch_I not in s.V_OPTIONAL_BRANCHES and gen_I not in s.V_OPTIONAL_BRANCHES:
                raise Refusal(f'{var}: no pre-screen V master with CONSTDIG {target} (I reference: {branch_I}) - refusing')
            e.update(branch='none', none_reason=(
                f'reference {var}: NONE - no pre-screen V master with CONSTDIG {target} ({gen}); the flow stage '
                f'diagnostic V rms is not measured; membership is unaffected'))
            out.append(e)
            continue
        label, path, hs = hit
        if label == 'superseded-archive':
            f, strip = None, False
            src = (f'previous-generation pre-screen {path}, native' if gen == 'previous-generation' else
                   f'superseded {path.split("/")[-1]}')
        else:
            f = FC.pixel_factor(hs) if (label == 'current-master' or hs.get('FLUXBAKE')) else 1.0
            if not _factor_ok(f):
                raise Refusal(f'{var}: pixel factor {f} of {path} unusable')
            strip = True
            src = (f'previous-generation pre-screen {path}' if gen == 'previous-generation' else
                   'current master' if label == 'current-master' else f'{label} {path}') + f' / pixel factor {f:.6f}'
        e.update(branch=label, source=path, factor=f, src=src, history=_history(var, gen, label, path, f),
                 constdig=hs.get('CONSTDIG'), ncoadd=hs.get('NCOADD'))
        if read_data:
            with fits.open(path) as h:
                d = h[0].data.astype(np.float32) if f is None else (h[0].data.astype(np.float64) / f).astype(np.float32)
                hh = h[0].header.copy()
            if strip:
                _strip(hh)
            e.update(data=d, header=hh)
        if st == 'I':
            gen_I, branch_I = gen, label
        out.append(e)
    _crosscheck_tables(cell, out, tables)
    return out


def _crosscheck_tables(cell, entries, tables):
    """A hand-table row for the cell must name exactly the file the resolver chose, Stokes by Stokes (V None = none), as
    the same generation: a ref_override_I.csv row claims the current constituents - reached only as the previous
    generation, the row is stale and production refuses it (rebuild_cell.sh:62-64), so this refuses too."""
    row = tables.get(cell)
    if not row:
        return
    if row.get('generation') and entries and entries[0]['generation'] != row['generation']:
        raise Refusal(f'{cell}: {row["table"]} gives a {row["generation"]} reference but the current master '
                      f'({entries[0]["target"]} sought) resolves to the {entries[0]["generation"]} '
                      f'({entries[0]["source"]}) - the table row is stale; correct it (production refuses this cell too)')
    for e in entries:
        want = row[e['stokes']]
        got = e['source']
        same = (want is None and got is None) or (want and got and os.path.realpath(want) == os.path.realpath(got))
        if not same:
            raise Refusal(f'{e["var"]}: the reference resolver chose {got or "no reference"} ({e["generation"]}, '
                          f'{e["branch"]}) but {row["table"]} names {want or "no reference"} for {cell} - the hand table '
                          f'and the CONSTDIG index disagree; correct the table row or masters_stagein.REFERENCE_COPY_ROOTS')
        if row.get('constdig') and e['source'] and e['constdig'] != row['constdig']:
            raise Refusal(f'{e["var"]}: {got} carries CONSTDIG {e["constdig"]}, {row["table"]} declares {row["constdig"]}')


def reference_from_path(path, read_data=True):
    """--reference path:P - the I reference at P (and V_deep_Taper_R0_<scope>.fits beside it, when present). A baked
    master (FLUXBAKE) is divided by its pixel factor and its SH*/FLUX* cards stripped, as branch (4); MDWARP refuses."""
    from astropy.io import fits
    from exopipe.phase3.lib.masters import flux_correction as FC
    base = os.path.basename(path)
    if not base.startswith('I_deep_Taper_R075_') or not base.endswith('.fits'):
        raise Refusal(f'--reference path:{path}: expected an I_deep_Taper_R075_<scope>.fits file')
    sc = base[len('I_deep_Taper_R075_'):-len('.fits')]
    out = []
    for st, var, rob in REF_VARS:
        p = os.path.join(os.path.dirname(path), f'{var}_{sc}.fits')
        e = dict(stokes=st, var=var, generation='given', branch='path', target=None, src='', source=None, factor=None, constdig=None, ncoadd=None,
                 history='', data=None, header=None, none_reason='')
        if not os.path.exists(p):
            if st == 'I':
                raise Refusal(f'--reference path:{path}: file missing')
            e['none_reason'] = f'reference {var}: NONE - no {os.path.basename(p)} beside the given I reference'
            out.append(e)
            continue
        h0 = fits.getheader(p)
        if h0.get('MDWARP'):
            raise Refusal(f'--reference path: {p} carries MDWARP (dewarped) - a reference must be un-warped')
        f = FC.pixel_factor(h0) if h0.get('FLUXBAKE') or h0.get('SHSCALE') is not None else None
        if f is not None and not _factor_ok(f):
            raise Refusal(f'{var}: --reference pixel factor {f} unusable')
        src = f'given reference {p}' + (f' / pixel factor {f:.6f}' if f is not None else ' (native)')
        e.update(source=p, factor=f, src=src, constdig=h0.get('CONSTDIG'), ncoadd=h0.get('NCOADD'),
                 history=f'rebuild reference (--reference path, ruled): native un-warped {var} ({src})')
        if read_data:
            with fits.open(p) as h:
                d = h[0].data.astype(np.float32) if f is None else (h[0].data.astype(np.float64) / f).astype(np.float32)
                hh = h[0].header.copy()
            if f is not None:
                _strip(hh)
            e.update(data=d, header=hh)
        out.append(e)
    return out


def write_reference(cell_dir, sc, entries):
    """Write each reference under the canonical name in coadd_final/ (overwrite=False, as production) and copy it into
    coadd_final_ref/ (rebuild_cell.sh:118-120). Returns [(stokes, canonical path, ref copy path, history line)]."""
    from astropy.io import fits
    os.makedirs(f'{cell_dir}/coadd_final', exist_ok=True)
    os.makedirs(f'{cell_dir}/coadd_final_ref', exist_ok=True)
    out = []
    for e in entries:
        if e['data'] is None:
            continue
        p = f'{cell_dir}/coadd_final/{e["var"]}_{sc}.fits'
        hh = e['header']
        hh['HISTORY'] = e['history']
        fits.writeto(p, e['data'], hh, overwrite=False)
        q = f'{cell_dir}/coadd_final_ref/{e["var"]}_{sc}.fits'
        shutil.copy(p, q)                      # `cp` (rebuild_cell.sh:118-120): mtime of the copy is now
        out.append((e['stokes'], p, q, e['history']))
    return out


def supersede(cell_dir, out_root, tag):
    """--fresh --supersede TAG: move an earlier cell directory to <out>/_superseded/<tag>/<cell>/ with a sha1 manifest
    (replaces rebuild_cell.sh:34 'scratch cell exists'). Returns the new location."""
    dst = os.path.join(out_root, '_superseded', tag, os.path.basename(cell_dir))
    if os.path.exists(dst):
        raise Refusal(f'--supersede {tag}: {dst} exists already - choose another tag')
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    rows = []
    for root, _, fs in os.walk(cell_dir):
        for f in sorted(fs):
            p = os.path.join(root, f)
            if os.path.isfile(p) and not os.path.islink(p):
                rows.append((os.path.relpath(p, cell_dir), os.path.getsize(p), IP.sha1_file(p)))
    shutil.move(cell_dir, dst)
    with open(dst + '.manifest.csv', 'w') as fh:
        fh.write('path,size,sha1\n')
        fh.writelines(f'{a},{b},{c}\n' for a, b, c in rows)
    return dst


def plan_summary(entries):
    """The reference choice as recorded (run record notes / --dry-run): one dict per Stokes, no pixel data."""
    return [dict(stokes=e['stokes'], generation=e.get('generation'), branch=e['branch'], source=e['source'], factor=e['factor'],
                 constdig=e['constdig'], ncoadd=e['ncoadd'], none=e['none_reason']) for e in entries]


def describe(entries):
    return json.dumps(plan_summary(entries), sort_keys=True, default=str)
