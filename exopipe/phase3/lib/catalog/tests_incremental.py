"""Content tests of the incremental deep catalogue (lib/catalog/incremental.py); discovered by core/validate.

    exopipe3 verify incremental_partition /data07/gh/phase3/transients/Run_P3S5/source_catalogs_V_wb
    exopipe3 verify incremental_state <out>/_incremental
    exopipe3 verify epoch_block <out>/_incremental/epochs/04h_2025-10-14
    exopipe3 verify incremental_delta <out>/_incremental/delta_V_wb.csv

Classes (each test reads CONTENT, never only presence):
  epoch_block           one epoch's stored inputs: every row, ledger row and image row is of that (LST, date); cand_ids
                        unique; every (Stokes, cell) with a detection has its searched frame and every frame link resolves;
                        the fit cache holds one admissible fit (gaussian / template / peak) per detection; the sha1s of
                        epoch.json match the files.
  incremental_state     the association state equals a FULL rebuild: the stored detections (builder input order) are
                        re-associated with the LIVE builder's own assoc_groups and the partition must equal the stored one
                        (this is the "incremental == full" test on any state); sources_<K>.csv = the components (members +
                        sigma-stack-removed); the partition sha1 and every epoch block's sha1 equal state.json.
  incremental_partition a deep catalogue directory (source_catalogs_<K>, primary or multiples-fork tier) against its input
                        selection: the builder's member partition must equal the order-independent association of that
                        selection (pairs tested as the builder tests them, sigma-stack singleton gate on Stokes I primary
                        tiers), and every source_id must equal the builder's naming of that group. A catalogue built from a
                        different selection (a mixed-state run dir) fails.
  incremental_delta     a delta file: known change classes; every current source of sources_<K>.csv appears exactly once
                        (never 'removed'), with its n_det; removed rows name only previous sources.
"""
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict

from exopipe.phase3.core.validate import content_test


def _I():
    from exopipe.phase3.lib.catalog import incremental
    return incremental


def _sha1(path):
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


@content_test('epoch_block')
def epoch_rows_frames_and_fits(path, **ctx):
    I = _I()
    meta_p = os.path.join(path, 'epoch.json')
    if not os.path.exists(meta_p):
        return False, None, 'no epoch.json'
    meta = json.load(open(meta_p))
    lst, date = I.parse_epoch(meta['epoch'])
    bad = []
    for n, h in (meta.get('files') or {}).items():
        p = os.path.join(path, n)
        if not os.path.exists(p) or _sha1(p) != h:
            bad.append(f'{n} differs from epoch.json')
    hdr, rows = I._read_csv(os.path.join(path, 'master_catalog.csv'))
    ix = {k: i for i, k in enumerate(hdr)}
    for need in ('cand_id', 'lst', 'date', 'band', 'stokes', 'ra', 'dec', 'snr'):
        if need not in ix:
            return False, None, f'master_catalog.csv lacks column {need}'
    off = [r[ix['cand_id']] for r in rows if r[ix['lst']] != lst or r[ix['date']][:10] != date]
    if off:
        bad.append(f'{len(off)} rows not of {meta["epoch"]}, e.g. {off[:2]}')
    dup = [k for k, v in Counter(r[ix['cand_id']] for r in rows).items() if v > 1]
    if dup:
        bad.append(f'{len(dup)} duplicate cand_ids')
    if len(rows) != meta.get('n_rows'):
        bad.append(f'{len(rows)} rows but epoch.json says {meta.get("n_rows")}')
    for name in ('ledger.csv', 'images.csv', 'refused.csv'):
        h2, r2 = I._read_csv(os.path.join(path, name))
        if 'lst' in h2 and 'date' in h2:
            il, idt = h2.index('lst'), h2.index('date')
            n_off = sum(1 for r in r2 if r[il] != lst or r[idt][:10] != date)
            if n_off:
                bad.append(f'{name}: {n_off} rows of another epoch')
    _, fr = I._read_csv(os.path.join(path, 'frames.csv'))
    have = {(f[0], f[1], f[2]) for f in fr}
    for f in fr:
        if not f[2].startswith(date + '_'):
            bad.append(f'frame {f[2]} is not of {date}')
            break
        link = os.path.join(path, 'vresid' if f[0] == 'V' else 'iresid', f[1], f[2])
        if not os.path.exists(link):
            bad.append(f'frame link {link} does not resolve')
            break
    miss = sum(1 for r in rows if (r[ix['stokes']], f"{lst}_{r[ix['band']]}",
                                   f"{date}_{r[ix['stokes']]}_redo_resid_{r[ix['band']]}.fits") not in have)
    if miss:
        bad.append(f'{miss} detections without their searched frame')
    if rows:
        ch, cr = I._read_csv(os.path.join(path, 'fitcache.csv'))
        if 'key' not in ch or 'pos_kind' not in ch:
            bad.append('fitcache.csv is not a fitted-position cache')
        else:
            ik, ip = ch.index('key'), ch.index('pos_kind')
            fitted = {I.cache_cand_id(r[ik]): r[ip] for r in cr}
            nofit = [r[ix['cand_id']] for r in rows if r[ix['cand_id']] not in fitted]
            badk = [c for c, k in fitted.items() if k not in I.FIT_KINDS]
            if nofit:
                bad.append(f'{len(nofit)} detections without a fit, e.g. {nofit[:2]}')
            if badk:
                bad.append(f'{len(badk)} fits without an admissible kind')
    return (not bad), len(rows), ('; '.join(bad[:5]) if bad else
                                  f'{len(rows)} detections of {meta["epoch"]}, {len(fr)} frames, every detection fitted')


@content_test('incremental_state')
def state_equals_full_rebuild(path, **ctx):
    I = _I()
    sp = os.path.join(path, 'state.json')
    if not os.path.exists(sp):
        return False, None, 'no state.json'
    st = json.load(open(sp))
    if st.get('version') != I.STATE_VERSION:
        return False, None, f'state version {st.get("version")} != {I.STATE_VERSION}'
    ns = I.builder_namespace()
    if ns['__sha1__'] != st['engine']['builder_sha1']:
        return False, None, (f'the builder changed since the state was written ({st["engine"]["builder_sha1"][:12]} -> '
                             f'{ns["__sha1__"][:12]}): the next associate run rebuilds the state')
    K_assoc = float(st['engine']['K_ASSOC'])
    bad, n_all = [], 0
    for e, files in (st.get('epochs') or {}).items():
        for n, h in files.items():
            p = os.path.join(path, 'epochs', e, n)
            if not os.path.exists(p) or _sha1(p) != h:
                bad.append(f'epoch block {e}/{n} changed after the state was written')
                break
    for K, kst in (st.get('keys') or {}).items():
        dets = I.read_engine_dets(path, K) or {}
        n = sum(len(v) for v in dets.values())
        n_all += n
        if n != kst.get('n_dets'):
            bad.append(f'{K}: {n} detections stored, state.json says {kst.get("n_dets")}')
        if I.partition_digest(dets) != kst.get('partition_sha1'):
            bad.append(f'{K}: partition sha1 differs from state.json')
        for band, dl in dets.items():
            full = I.associate_band_builder(ns, dl, K_assoc)
            stored = defaultdict(set)
            ref = defaultdict(set)
            for d, lab in zip(dl, full):
                stored[d['comp']].add(d['cand_id'])
                ref[lab].add(d['cand_id'])
            if {frozenset(v) for v in stored.values()} != {frozenset(v) for v in ref.values()}:
                bad.append(f'{K} {band}: the incremental partition differs from the full rebuild')
        srcs = I.read_engine_sources(path, K)
        comps = defaultdict(set)
        for band, dl in dets.items():
            for d in dl:
                comps[(band, d['comp'])].add(d['cand_id'])
        seen = set()
        for band, sl in srcs.items():
            for s in sl:
                k = (band, s['comp'])
                seen.add(k)
                if set(s['members']) | set(s['sstack_removed']) != comps.get(k, set()) or s['n_det'] != len(s['members']):
                    bad.append(f'{K} {band}: source {s["source_id"]} members differ from its component')
                    break
        dropped = [k for k in comps if k not in seen]
        for k in dropped:      # a component may vanish only through the sigma-stack singleton gate (every member removed)
            if K.startswith('V') or '_f' in K:
                bad.append(f'{K} {k[0]}: component {k[1]} has no source and this key has no sigma-stack gate')
                break
    return (not bad), n_all, ('; '.join(bad[:6]) if bad else
                              f'{len(st.get("keys") or {})} keys, {n_all} detections: incremental partition == full '
                              f'rebuild (builder assoc_groups); {len(st.get("epochs") or {})} epoch blocks unchanged')


def _selection_for(cat_dir):
    """(K, selection path, is_fork) of a deep catalogue directory source_catalogs_<K>."""
    I = _I()
    base = os.path.basename(os.path.normpath(cat_dir))
    if not base.startswith('source_catalogs_'):
        return None, None, None
    K = base[len('source_catalogs_'):]
    U = os.path.dirname(os.path.normpath(cat_dir))
    forks = dict(I.C().FORK_KEYS)
    if K in forks:
        return K, os.path.join(U, forks[K]), True
    if K in ('V_wb', 'V_sb', 'I_wb', 'I_sb'):
        return K, os.path.join(U, f'selected_{K}.csv'), False
    return K, None, None


@content_test('incremental_partition')
def builder_partition_equals_order_independent_association(path, **ctx):
    I = _I()
    K, sel, is_fork = _selection_for(path)
    if sel is None:
        return False, None, f'{path}: not a deep catalogue key directory (source_catalogs_<V|I>_<wb|sb|fwb|fsb>)'
    if not os.path.exists(sel):
        return False, None, f'its selection {sel} does not exist'
    ns = I.builder_namespace()
    I.binding_check(ns)
    h, rows = I._read_csv(sel)
    beam = {}
    ep = os.path.join(path, 'E_params.json')
    if os.path.exists(ep):
        beam = json.load(open(ep)).get('beam_as', {})
    stokes = {r[h.index('stokes')] for r in rows} if rows else set()
    thr = I.sstack_threshold()[0] if (stokes == {'I'} and not is_fork) else None
    res = I.associate_key(ns, h, rows, beam, thr, prev_dets=None)
    ok, detail = I.crosscheck(res, path)
    n = sum(len(v) for v in res['sources'].values())
    fitted = sum(1 for r in rows if 'pos_kind' in h and r[h.index('pos_kind')] in I.FIT_KINDS)
    return ok, n, (f'{K}: {detail}; {len(rows)} selected detections ({fitted} fitted)' if ok else f'{K}: {detail}')


@content_test('incremental_delta')
def delta_names_every_current_source_once(path, **ctx):
    I = _I()
    h, rows = I._read_csv(path)
    if tuple(h) != I.DELTA_FIELDS:
        return False, None, f'columns {h} != {list(I.DELTA_FIELDS)}'
    ix = {k: i for i, k in enumerate(h)}
    bad = []
    unk = {r[ix['change']] for r in rows} - set(I.CHANGES)
    if unk:
        bad.append(f'unknown change classes {sorted(unk)}')
    K = os.path.basename(path)[len('delta_'):-len('.csv')]
    srcs = I.read_engine_sources(os.path.dirname(path), K)
    cur = Counter((r[ix['band']], r[ix['source_id']]) for r in rows if r[ix['change']] != 'removed')
    want = {(s['band'], s['source_id']): s['n_det'] for sl in srcs.values() for s in sl}
    if set(cur) != set(want):
        bad.append(f'{len(set(want) - set(cur))} current sources missing from the delta, {len(set(cur) - set(want))} extra')
    if any(v > 1 for v in cur.values()):
        bad.append('a current source appears more than once')
    nd = sum(1 for r in rows if r[ix['change']] != 'removed' and want.get((r[ix['band']], r[ix['source_id']])) != int(r[ix['n_det']]))
    if nd:
        bad.append(f'{nd} rows with an n_det different from sources_{K}.csv')
    if any(r[ix['change']] == 'removed' and (r[ix['source_id']] or not r[ix['prev_source_ids']]) for r in rows):
        bad.append('a removed row names a current source or no previous source')
    cnt = Counter(r[ix['change']] for r in rows)
    return (not bad), len(rows), ('; '.join(bad) if bad else ', '.join(f'{k} {v}' for k, v in sorted(cnt.items())))
