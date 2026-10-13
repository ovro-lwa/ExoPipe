"""Content tests of the `exopipe3 masters --tier snapshot` product classes (discovered by core/validate: lib/<step>/tests*.py).

  snapshot_master_registry  <out>/snapshot/registry/<slot>.csv or <out>/snapshot/snapshot_master_registry_gen4.csv: every row's
                            frozen file has today the size recorded and (rehash, the default) the md5 recomputed now = the
                            row's md5 = the MANIFEST.csv md5_copy (manifest sha1 pinned); every link row resolves now to its
                            slot's frozen copy inside FROZEN_ROOT, never into the deep archive Coadd/ tree; every slot has its
                            4 files and 2 links; the combined registry also carries the full counts (456 slots, 1,824 files,
                            912 links). The step passes rehash=False right after computing the md5 itself.
  snapshot_pb_library       <out>/snapshot/pb_library_registry.csv: the 105 model-beam maps the snapshot tier reads (15 subbands
                            x 6 bins + deep), each finite, inside masters_snapshot.PB_LIBRARY_VALUE_RANGE with some pixel above
                            0.1, PBBEAM = the model beam, PBSCOPE / PBLSTBIN = its name; rehash re-reads every map and compares
                            md5, range and cards with the row.
  snapshot_parent_catalog   master_R075_<LST>_<scope>.gaul.csv (+ .gen.json) of the snapshot parent set (M21, exopipe_ops
                            a670970 / 3ff6565): the sidecar names the slot's frozen gen-4 R075 master and its master_md5_data
                            equals the md5 of that master's float32 data after the builder's zap (non-finite and |x| >
                            PARENT_ZAP_JYB -> 0, phase3_deep._load), re-read now; CONSTDIG = the master's; thresh_pix /
                            thresh_isl / atrous_do = the recipe; Gaussian rows = rows_pybdsf - rows dropped > 0, every RA/Dec
                            finite (snapshot_parentcat_sanitize).
All tests work on a production file without context (`exopipe3 verify <class> <file>`).
"""
import csv
import hashlib
import json
import os
import re

from exopipe.phase3.core.validate import content_test


def _T():
    from exopipe.phase3.lib.masters import tier_snapshot as T
    return T


def _verdict(problems, value, ok_detail):
    if problems:
        return False, value, '; '.join(problems[:6]) + (f' (+{len(problems) - 6} more)' if len(problems) > 6 else '')
    return True, value, ok_detail


@content_test('snapshot_master_registry')
def registry_matches_tree(path, rehash=True, **ctx):
    T = _T()
    s = T._sec()
    with open(path, newline='') as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames or []
        rows = list(rd)
    problems = []
    if cols != T.REGISTRY_COLUMNS:
        return False, None, f'columns {cols} != {T.REGISTRY_COLUMNS}'
    why = T.manifest_problem()
    if why:
        return False, None, why
    _, by, n_man = T.manifest()
    slots = {}
    for r in rows:
        slots.setdefault(r['slot'], []).append(r)
        mr = by.get(r['slot'], {}).get(r['file'])
        want = (mr or {}).get('md5_copy', '')
        if not want or r['md5_manifest'] != want or r['md5_now'] != want or r['md5_ok'] != '1':
            problems.append(f'{r["slot"]} {r["file"]}: md5 {r["md5_now"][:8]} / row manifest {r["md5_manifest"][:8]} / '
                            f'MANIFEST.csv {want[:8]}')
            continue
        p = r['path']
        if not os.path.isfile(p) or str(os.path.getsize(p)) != r['size']:
            problems.append(f'{r["slot"]} {r["file"]}: frozen copy missing or its size changed since the registry')
        elif rehash:
            now = T.TS.md5_file(p)
            if now != want:
                problems.append(f'{r["slot"]} {r["file"]}: md5 now {now} != manifest {want}')
        if r['link']:
            target, ok, lwhy = T.link_state(r['link'], p)
            if not ok or r['link_ok'] != '1':
                problems.append(f'{r["slot"]} link {r["file"]}: {lwhy or "recorded as failing"}')
    for slot, rs in slots.items():
        want_files = [f for f, _, _ in T.slot_files(slot)]
        if sorted(r['file'] for r in rs) != sorted(want_files):
            problems.append(f'{slot}: files {sorted(r["file"] for r in rs)} != {sorted(want_files)}')
        if sum(1 for r in rs if r['link']) != len(s.LINKED_FILES):
            problems.append(f'{slot}: {sum(1 for r in rs if r["link"])} link rows != {len(s.LINKED_FILES)}')
    if os.path.basename(path) == T.COMBINED:
        nl = sum(1 for r in rows if r['link'] and r['link_ok'] == '1')
        if len(slots) != s.N_SLOTS or len(rows) != s.N_FROZEN_FILES or n_man != s.N_FROZEN_FILES or \
                nl != s.N_SLOTS * len(s.LINKED_FILES):
            problems.append(f'counts: {len(slots)} slots, {len(rows)} files, {nl} good links, manifest {n_man} rows (want '
                            f'{s.N_SLOTS}, {s.N_FROZEN_FILES}, {s.N_SLOTS * len(s.LINKED_FILES)})')
    return _verdict(problems, len(rows), f'{len(rows)} frozen files of {len(slots)} slot(s) = manifest md5'
                                         f'{" (re-hashed now)" if rehash else ""}; links into the frozen tree')


@content_test('snapshot_pb_library')
def library_maps(path, rehash=True, **ctx):
    T = _T()
    s = T._sec()
    lo, hi = s.PB_LIBRARY_VALUE_RANGE
    with open(path, newline='') as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames or []
        rows = {r['file']: r for r in rd}
    if cols != T.PBLIB_COLUMNS:
        return False, None, f'columns {cols} != {T.PBLIB_COLUMNS}'
    exp = T.pblib_expected()
    problems = []
    missing = [f for f, _, _ in exp if f not in rows]
    if missing:
        problems.append(f'{len(missing)} of {len(exp)} library maps not in the registry, e.g. {missing[:3]}')
    for f, sb, k in exp:
        r = rows.get(f)
        if r is None:
            continue
        if rehash:
            p = os.path.join(T.pb_library(), f)
            if not os.path.isfile(p):
                problems.append(f'{f} missing now')
                continue
            now = T.pblib_row(p)
            diff = [c for c in T.PBLIB_COLUMNS if str(now[c]) != str(r[c])]
            if diff:
                problems.append(f'{f}: {diff} changed since the registry')
                continue
        if r['finite'] != '1':
            problems.append(f'{f}: non-finite pixels')
        elif not (float(r['min']) >= lo and float(r['max']) <= hi and float(r['max']) > 0.1):
            problems.append(f'{f}: values {r["min"]}..{r["max"]} outside {lo}..{hi} or never above 0.1')
        if r['pbbeam'] != s.PB_BEAM_MODEL:
            problems.append(f'{f}: PBBEAM {r["pbbeam"]!r} != model beam {s.PB_BEAM_MODEL}')
        if r['pbscope'] != sb or str(r['pblstbin']) != str(k):
            problems.append(f'{f}: PBSCOPE/PBLSTBIN {r["pbscope"]}/{r["pblstbin"]} != {sb}/{k}')
    return _verdict(problems, len(rows), f'{len(rows)} model-beam maps finite, in range, beam {s.PB_BEAM_MODEL}'
                                         f'{" (re-read now)" if rehash else ""}')


_PARENT_RE = re.compile(r'^master_R075_(\d\dh)_(\d+MHz|Red|Green|Blue|Full)\.gaul\.csv$')


def parent_master_path(slot):
    """The frozen gen-4 parent master of a slot (snapshot_parentcat_contract.py:49)."""
    s = _T()._sec()
    band = slot.split('_', 1)[1]
    return os.path.join(s.FROZEN_ROOT, slot, 'coadd_final', s.PARENT_MASTER_FILE.format(b=band))


def parent_data_md5(master):
    """md5 of the master's float32 data after phase3_deep._load's zap (non-finite and |x| > PARENT_ZAP_JYB set to 0), as
    the builder digests it (snapshot_parentcat_contract.py:54, 66)."""
    import numpy as np
    from astropy.io import fits
    a = np.squeeze(fits.getdata(master)).astype(float)
    a[~np.isfinite(a)] = 0.0
    a[np.abs(a) > _T()._sec().PARENT_ZAP_JYB] = 0.0
    return hashlib.md5(np.ascontiguousarray(a.astype(np.float32)).tobytes()).hexdigest()


@content_test('snapshot_parent_catalog')
def parent_catalog_contract(path, **ctx):
    import numpy as np
    s = _T()._sec()
    m = _PARENT_RE.match(os.path.basename(path))
    if not m:
        return False, None, f'not a snapshot parent catalogue name: {os.path.basename(path)}'
    slot = f'{m.group(1)}_{m.group(2)}'
    side = path + '.gen.json'
    if not os.path.isfile(side):
        return False, None, 'no .gen.json sidecar'
    g = json.load(open(side))
    master = parent_master_path(slot)
    problems = []
    if g.get('master') != master:
        problems.append(f'sidecar master {g.get("master")} != {master}')
    if not os.path.isfile(master):
        return False, None, f'parent master {master} missing'
    from astropy.io import fits
    hdr = fits.getheader(master)
    md5 = parent_data_md5(master)
    if g.get('master_md5_data') != md5:
        problems.append(f'sidecar md5 {str(g.get("master_md5_data"))[:8]} != master data now {md5[:8]}')
    if g.get('CONSTDIG') != hdr.get('CONSTDIG'):
        problems.append(f'sidecar CONSTDIG {g.get("CONSTDIG")} != master {hdr.get("CONSTDIG")}')
    if (g.get('thresh_pix'), g.get('thresh_isl'), bool(g.get('atrous_do'))) != \
            (s.PARENT_THRESH_PIX, s.PARENT_THRESH_ISL, bool(s.PARENT_ATROUS)):
        problems.append(f'recipe thresh_pix {g.get("thresh_pix")} thresh_isl {g.get("thresh_isl")} atrous '
                        f'{g.get("atrous_do")} != {s.PARENT_THRESH_PIX} / {s.PARENT_THRESH_ISL} / {s.PARENT_ATROUS}')
    ra, dec = [], []
    with open(path) as f:
        for ln in f:
            if not ln.strip() or ln.startswith('#'):
                continue
            q = ln.split(',')
            try:
                ra.append(float(q[4]))
                dec.append(float(q[6]))
            except (ValueError, IndexError):
                ra.append(float('nan'))
                dec.append(float('nan'))
    n = len(ra)
    want = int(g.get('rows_pybdsf', -1)) - int(g.get('rows_dropped_nonfinite_position', 0))
    if n <= 0 or n != want:
        problems.append(f'{n} Gaussian rows, sidecar says {want}')
    nbad = int(np.sum(~(np.isfinite(ra) & np.isfinite(dec)))) if n else 0
    if nbad:
        problems.append(f'{nbad} rows with a non-finite RA/Dec')
    return _verdict(problems, n, f'{n} Gaussian rows on {os.path.basename(master)} (data md5 {md5[:8]} = sidecar, '
                                 f'thresh_pix {s.PARENT_THRESH_PIX}, thresh_isl {s.PARENT_THRESH_ISL})')
