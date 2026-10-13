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
           -> ``native_reference()`` (same arithmetic: float64 / f -> float32; same HISTORY text; same refusals)
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


def override_tables(pin=True):
    """(ref_override {cell: root}, ref_override_I {cell: path}) - each table sha1-pinned (G-M25)."""
    s = _cfg().masters_stagein
    ov, ovi = {}, {}
    if os.path.exists(s.REF_OVERRIDE_TABLE):
        if pin:
            IP.sha1_pin(s.REF_OVERRIDE_TABLE, s.REF_OVERRIDE_TABLE_SHA1, 'ref_override.csv')
        # awk -F, -v c=$cell '$1==c{print $2}' (rebuild_cell.sh:23): first column = cell, second = root; header row ignored
        with open(s.REF_OVERRIDE_TABLE) as f:
            for line in f:
                parts = line.rstrip('\n').split(',')
                if len(parts) >= 2 and parts[0] != 'cell':
                    ov.setdefault(parts[0], parts[1])
    if os.path.exists(s.REF_OVERRIDE_I_TABLE):
        if pin:
            IP.sha1_pin(s.REF_OVERRIDE_I_TABLE, s.REF_OVERRIDE_I_TABLE_SHA1, 'ref_override_I.csv')
        ovi = {r['cell']: r['path'] for r in csv.DictReader(open(s.REF_OVERRIDE_I_TABLE))}
    return ov, ovi


def working_cell(cell, ov=None):
    """C of rebuild_cell.sh:19/23: the cell's working (installed) directory, or its ref_override root."""
    if ov is None:
        ov, _ = override_tables()
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


def native_reference(cell, C, ovi, read_data=True):
    """The four-branch reference reconstruction of rebuild_cell.sh:47-117. Returns a list, one entry per Stokes:
    dict(stokes, var, branch, src (text as production logs it), source (path read), factor, constdig, ncoadd,
    history (the HISTORY line), data (float32 array or None), header (astropy Header or None), none_reason).
    Raises Refusal with production's own SystemExit text on every refusal. read_data=False plans only (headers)."""
    from astropy.io import fits
    from exopipe.phase3.lib.masters import flux_correction as FC
    s = _cfg().masters_stagein
    lst, sc = cell.split('_', 1)
    out = []
    for st, var, rob in REF_VARS:
        cur = f'{C}/coadd_final/{var}_{sc}.fits'
        if not os.path.exists(cur):
            raise Refusal(f'{var}: current master {cur} missing - no reference resolvable (G-M04)')
        hc = fits.getheader(cur)
        e = dict(stokes=st, var=var, branch=None, src='', source=None, factor=None, constdig=None, ncoadd=None,
                 history='', data=None, header=None, none_reason='')
        if cell in ovi and st == 'V':
            e.update(branch='override-I-no-V', none_reason=(
                f'reference {var}: NONE - no un-warped V master of these constituents survives (deleted 2026-10-08); the '
                f'flow stage diagnostic V rms is not measured; membership is unaffected'))
            out.append(e)
            continue
        if cell in ovi:
            src_p = ovi[cell]
            if not os.path.exists(src_p):
                raise Refusal(f'{var}: override {src_p} not readable on this node (ref_override_I.csv; lwacalim07-local)')
            _ho = fits.getheader(src_p)
            if _ho.get('CONSTDIG') != hc.get('CONSTDIG') or _ho.get('MDWARP'):
                raise Refusal(f'{var}: override {src_p} is not the un-warped master of the current constituents')
            f = FC.pixel_factor(_ho) if _ho.get('FLUXBAKE') else 1.0
            if not _factor_ok(f):
                raise Refusal(f'{var}: override pixel factor {f} unusable')
            e.update(branch='override-I', source=src_p, factor=f,
                     src=f'un-warped override {src_p} / pixel factor {f:.6f}',
                     history=f'2026-10-09 rebuild reference: native un-warped {var} (override {src_p} / pixel factor {f:.6f})')
            if read_data:
                with fits.open(src_p) as h:
                    d = (h[0].data.astype(np.float64) / f).astype(np.float32)
                    hh = h[0].header.copy()
                _strip(hh)
                e.update(data=d, header=hh)
            e.update(constdig=_ho.get('CONSTDIG'), ncoadd=_ho.get('NCOADD'))
            out.append(e)
            continue
        _wq = f'{s.PREDEWARP_QUARANTINE}/{cell}/coadd_final/{var}_{sc}.fits'
        _hw = fits.getheader(_wq) if (hc.get('MDWARP') and os.path.exists(_wq)) else None
        if _hw is not None and _hw.get('CONSTDIG') == hc.get('CONSTDIG') and not _hw.get('MDWARP'):
            f = FC.pixel_factor(_hw) if _hw.get('FLUXBAKE') else 1.0
            if not _factor_ok(f):
                raise Refusal(f'{var}: pre-dewarp master pixel factor {f} unusable')
            source, strip, branch = _wq, True, 'predewarp-quarantine'
            src = f'pre-dewarp master (working quarantine 20261004) / pixel factor {f:.6f}'
            ident = _hw
        elif hc.get('MDWARP'):
            cands = [p for p in sorted(glob.glob(f'{s.SUPERSEDED_ARCHIVE}/{lst}/{sc}/{sc}_{st}_deep_Taper_{rob}_*.fits'))
                     if fits.getheader(p).get('CONSTDIG') == hc.get('CONSTDIG')]
            if len(cands) != 1:
                raise Refusal(f'{var}: {len(cands)} superseded masters with CONSTDIG {hc.get("CONSTDIG")} - need exactly 1')
            hs = fits.getheader(cands[0])
            if hs.get('SHSCALE') is not None or hs.get('MDWARP') or hs.get('FLUXBAKE'):
                raise Refusal(f'{cands[0]}: not native and un-warped')
            f, source, strip, branch = None, cands[0], False, 'superseded-archive'
            src = f'superseded {cands[0].split("/")[-1]}'
            ident = hs
        else:
            f = FC.pixel_factor(hc)
            if not _factor_ok(f):
                raise Refusal(f'{var}: pixel factor {f} unusable')
            source, strip, branch = cur, True, 'current-master'
            src = f'current master / pixel factor {f:.6f}'
            ident = hc
        e.update(branch=branch, source=source, factor=f, src=src,
                 history=f'2026-10-04 rebuild reference: native un-warped {var} ({src})')
        if read_data:
            with fits.open(source) as h:
                if f is None:
                    d = h[0].data.astype(np.float32)
                else:
                    d = (h[0].data.astype(np.float64) / f).astype(np.float32)
                hh = h[0].header.copy()
            if strip:
                _strip(hh)
            e.update(data=d, header=hh)
        e.update(constdig=ident.get('CONSTDIG'), ncoadd=ident.get('NCOADD'))
        out.append(e)
    return out


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
        e = dict(stokes=st, var=var, branch='path', src='', source=None, factor=None, constdig=None, ncoadd=None,
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
    return [dict(stokes=e['stokes'], branch=e['branch'], source=e['source'], factor=e['factor'],
                 constdig=e['constdig'], ncoadd=e['ncoadd'], none=e['none_reason']) for e in entries]


def describe(entries):
    return json.dumps(plan_summary(entries), sort_keys=True, default=str)
