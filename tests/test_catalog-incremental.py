#!/usr/bin/env python3
"""Tests of unit catalog-incremental (lib/catalog/incremental.py, lib/catalog/tests_incremental.py). Plain asserts, no pytest.

    NUMPY_MADVISE_HUGEPAGE=0 EXOPIPE3_SITE=/lustre/gh/main/exopipe_ops/exopipe3_site.py \
        /opt/devel/pipeline/envs/py38_orca_nkosogor/bin/python3 tests/test_catalog-incremental.py [--no-real] [--slow]

Synthetic part (fresh directory /tmp/exopipe3_tests_catalog-incremental/, removed at the end):
  1. incremental == full: a synthetic multi-epoch sky (true sources seen in several epochs, chains that link two sources only
     once an intermediate epoch arrives, singles, fitted and unfitted rows, V and I) is added epoch by epoch in 4 different
     orders; after EVERY addition the incremental association equals the full association and the builder's own
     assoc_groups (partition), and the derived sources (names, positions, best detection, members in order, sigma-stack
     removals) equal the full derivation of the canonical input;
  2. an epoch excluded after it was added, and a refitted (moved) detection: incremental == full (dissolve path);
  3. the canonical assembly of epoch blocks is byte-identical whatever order the blocks were written in, and drops rows
     of a scope excluded now;
  4. classify_delta: new, extended, unchanged, changed, merged, split, removed, renamed;
  5. the four content tests PASS on good synthetic products and FAIL on a corrupted copy;
  6. the epoch phase end to end on a synthetic run with real FITS frames: the in-place deep_fit_positions.py through
     core.legacy.run, epoch_block passes, a second run takes every fit from the cache and writes the same cache, an
     excluded scope is left out at assembly.
Real part (read-only, Run_P3S5 on lwacalim07; skipped with --no-real or when the run is absent):
  7. content test incremental_partition on production catalogue directories (expected: V_wb, I_wb, I_fwb PASS; --slow adds
     V_sb, expected FAIL because Run_P3S5 is mixed-state - spec catalog.md K6);
  8. the engine adds Run_P3S5 V_wb in 6 random epoch batches: final partition == the builder's full partition and the
     production members file;
  9. a dry run of run_incremental on one real epoch (plan printed, nothing written).
"""
import argparse
import csv
import json
import os
import random
import shutil
import sys
import time
from collections import OrderedDict, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')

from exopipe.phase3.core import env as _env  # noqa: E402
_env.legacy()
from exopipe.phase3.core import validate  # noqa: E402
from exopipe.phase3.lib.catalog import incremental as I  # noqa: E402

TMP = '/tmp/exopipe3_tests_catalog-incremental'
RUN = '/data07/gh/phase3/transients/Run_P3S5'
HDR = ['cand_id', 'lst', 'date', 'band', 'stokes', 'ra', 'dec', 'snr', 'sstack', 'pos_kind', 'pos_stat_maj_arcsec',
       'pos_stat_min_arcsec']
RESULTS = []


def ok(name, cond, detail=''):
    RESULTS.append((name, bool(cond), detail))
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}", flush=True)
    assert cond, f'{name}: {detail}'


# ---------------------------------------------------------------------------------------------------------------------
# synthetic sky
# ---------------------------------------------------------------------------------------------------------------------

def synth_rows(seed=7, stokes='V'):
    """Rows of a synthetic multi-epoch detection set (merge-output columns used by the association)."""
    rng = random.Random(seed)
    lsts = ('04h', '05h', '06h')
    dates = ('2025-01-10', '2025-01-11', '2025-01-12', '2025-01-13', '2025-01-14', '2025-01-15')
    epochs = [(l, d) for d in dates for l in lsts]
    bands = ('Red', 'Green', '41MHz')
    rows = []
    n = [0]

    def add(l, d, band, ra, dec, snr, fitted=True, sstack=''):
        n[0] += 1
        cid = f'X{stokes}_{l}_{d}_{band}_{n[0]:04d}'
        st = rng.uniform(20.0, 120.0)
        rows.append([cid, l, d, band, stokes, f'{ra % 360.0:.5f}', f'{dec:.5f}', f'{snr:.2f}', sstack,
                     'template' if fitted else '', f'{st:.1f}' if fitted else '', f'{st * 0.9:.1f}' if fitted else ''])
    # true sources near each LST's meridian (dec 37 +- 20), seen in some epochs with arcminute scatter
    for k in range(18):
        l = lsts[k % 3]
        ra0 = 15.0 * (int(l[:2]) + 0.5) + rng.uniform(-8, 8)
        dec0 = 37.0 + rng.uniform(-20, 20)
        band = bands[k % 3]
        for (el, d) in epochs:
            if el == l and rng.random() < 0.6:
                add(el, d, band, ra0 + rng.gauss(0, 1.5 / 60), dec0 + rng.gauss(0, 1.5 / 60), rng.uniform(5, 12) * rng.choice((1, -1)),
                    fitted=rng.random() < 0.8)
    # chains: A and B 9 arcmin apart; a detection half-way only on one date links them (merged / split cases)
    for j, l in enumerate(lsts):
        ra0, dec0 = 15.0 * (int(l[:2]) + 0.5), 30.0 + 5 * j
        add(l, dates[0], 'Red', ra0, dec0, 9.0)
        add(l, dates[1], 'Red', ra0, dec0 + 9.0 / 60, 8.0)
        add(l, dates[3], 'Red', ra0, dec0 + 4.5 / 60, 7.5)
    # singles
    for (l, d) in epochs:
        for _ in range(4):
            band = rng.choice(bands)
            add(l, d, band, 15.0 * (int(l[:2]) + 0.5) + rng.uniform(-15, 15), 37.0 + rng.uniform(-30, 30),
                rng.uniform(5, 9), fitted=rng.random() < 0.7,
                sstack=(f'{rng.uniform(0.5, 2.0):.2f}' if stokes == 'I' else ''))
    return epochs, rows


def canonical(rows, epochs_in):
    keep = set(epochs_in)
    ix = {k: i for i, k in enumerate(HDR)}
    sel = [r for r in rows if (r[1], r[2]) in keep]
    return sorted(sel, key=lambda r: tuple(r[ix[c]] for c in I.C().CANONICAL_ROW_ORDER))


def sources_sig(res):
    return {band: [(s['source_id'], s['ra'], s['dec'], s['best_cand_id'], tuple(s['members']), tuple(s['sstack_removed']))
                   for s in sl] for band, sl in res['sources'].items()}


def partitions_equal(res, ns, K):
    for band, dl in res['dets'].items():
        lab_b = I.associate_band_builder(ns, dl, K)
        a = defaultdict(set)
        b = defaultdict(set)
        for d, l in zip(dl, lab_b):
            a[d['comp']].add(d['cand_id'])
            b[l].add(d['cand_id'])
        if {frozenset(v) for v in a.values()} != {frozenset(v) for v in b.values()}:
            return False, band
    return True, ''


BEAM = {'Red': 787.0, 'Green': 518.0, '41MHz': 600.0}


def test_engine_incremental_equals_full(ns):
    K = float(ns['K_ASSOC'])
    for stokes in ('V', 'I'):
        epochs, rows = synth_rows(seed=11 if stokes == 'V' else 12, stokes=stokes)
        thr = I.sstack_threshold()[0] if stokes == 'I' else None
        n_multi = 0
        for trial in range(4):
            order = list(epochs)
            random.Random(100 + trial).shuffle(order)
            state, added = None, []
            for e in order:
                added.append(e)
                sel = canonical(rows, added)
                res = I.associate_key(ns, HDR, sel, BEAM, thr, prev_dets=state, K=K)
                full = I.associate_key(ns, HDR, sel, BEAM, thr, prev_dets=None, K=K)
                assert I.partition_sets(res['dets']) == I.partition_sets(full['dets']), (stokes, trial, e)
                assert sources_sig(res) == sources_sig(full), (stokes, trial, e, 'sources differ')
                pe, band = partitions_equal(res, ns, K)
                assert pe, (stokes, trial, e, band, 'builder assoc_groups differs')
                state = res['dets']
            n_multi = sum(1 for sl in res['sources'].values() for s in sl if s['n_det'] > 1)
        ok(f'engine.incremental==full.{stokes}', True,
           f'{len(rows)} detections, {len(epochs)} epochs x 4 orders, checked after every addition; '
           f'{sum(len(v) for v in res["sources"].values())} sources ({n_multi} multi-detection)')


def test_engine_removal_and_change(ns):
    K = float(ns['K_ASSOC'])
    epochs, rows = synth_rows(seed=21)
    sel = canonical(rows, epochs)
    full = I.associate_key(ns, HDR, sel, BEAM, None, K=K)
    # exclusion of the date that holds the chain links (dates[3]) after it was added: split
    drop = {e for e in epochs if e[1] == '2025-01-13'}
    sel2 = canonical(rows, [e for e in epochs if e not in drop])
    inc = I.associate_key(ns, HDR, sel2, BEAM, None, prev_dets=full['dets'], K=K)
    ref = I.associate_key(ns, HDR, sel2, BEAM, None, K=K)
    ok('engine.removal==full', I.partition_sets(inc['dets']) == I.partition_sets(ref['dets'])
       and sources_sig(inc) == sources_sig(ref),
       f"stats {json.dumps({b: s for b, s in inc['stats'].items()}, sort_keys=True)[:200]}")
    delta = []
    for band in sorted(set(full['sources']) | set(inc['sources'])):
        delta += I.classify_delta(full['sources'].get(band, []), inc['sources'].get(band, []))
    ch = {d['change'] for d in delta}
    ok('engine.removal.delta_has_split_or_changed', bool(ch & {'split', 'changed', 'removed'}), str(sorted(ch)))
    # a refitted detection moves (its fit changed): it and its old component are re-associated
    ix = HDR.index('ra')
    sel3 = [list(r) for r in sel]
    sel3[5][ix] = f'{float(sel3[5][ix]) + 0.5:.5f}'
    inc3 = I.associate_key(ns, HDR, sel3, BEAM, None, prev_dets=full['dets'], K=K)
    ref3 = I.associate_key(ns, HDR, sel3, BEAM, None, K=K)
    st = next(iter(inc3['stats'].values()))
    ok('engine.changed_detection==full', I.partition_sets(inc3['dets']) == I.partition_sets(ref3['dets'])
       and sources_sig(inc3) == sources_sig(ref3) and any(s.get('changed') for s in inc3['stats'].values()),
       f'{st}')


def test_classify_delta():
    def S(sid, mem, band='Red'):
        return {'band': band, 'comp': mem[0], 'source_id': sid, 'members': list(mem), 'n_det': len(mem)}
    prev = [S('J0001+0001', ['a']), S('J0002+0002', ['b', 'c']), S('J0003+0003', ['d']), S('J0004+0004', ['e']),
            S('J0005+0005', ['f', 'g']), S('J0006+0006', ['h', 'i']), S('J0007+0007', ['z'])]
    new = [S('J0001+0001', ['a']),                 # unchanged
           S('J0002+0003', ['b', 'c', 'x']),       # extended (renamed)
           S('J0003+0004', ['d', 'e']),            # merged
           S('J0005+0005', ['f']), S('J0005+0006', ['g']),   # split
           S('J0006+0006', ['h', 'y']),            # changed (i removed, y added)
           S('J0009+0009', ['w'])]                 # new; z removed
    d = I.classify_delta(prev, new)
    by = {(r['source_id'] or r['prev_source_ids']): r for r in d}
    want = {'J0001+0001': 'unchanged', 'J0002+0003': 'extended', 'J0003+0004': 'merged', 'J0005+0005': 'split',
            'J0005+0006': 'split', 'J0006+0006': 'changed', 'J0009+0009': 'new', 'J0007+0007': 'removed'}
    got = {k: by[k]['change'] for k in want}
    ok('delta.classes', got == want and by['J0002+0003']['renamed'] == 1 and by['J0001+0001']['renamed'] == 0, str(got))


# ---------------------------------------------------------------------------------------------------------------------
# canonical assembly
# ---------------------------------------------------------------------------------------------------------------------

class _FakeExcl:
    def __init__(self, rows):
        self.rows = set(rows)

    def is_excluded(self, lst, scope, date):
        return (lst, scope, date) in self.rows

    def fingerprint(self):
        return {'sha1': 'test'}


class _FakeSrc:
    def __init__(self, d):
        self.dir = d
        self.spec = f'run:{d}'


class _FakeCtx:
    def __init__(self, out, src):
        self.out = out
        self.state_dir = os.path.join(out, '_incremental')
        self.dry = False
        self.source = _FakeSrc(src)


def write_block(state_dir, epoch, rows, frames_root):
    b = os.path.join(state_dir, 'epochs', epoch)
    os.makedirs(b, exist_ok=True)
    lst, date = I.parse_epoch(epoch)
    I._write_csv(os.path.join(b, 'master_catalog.csv'), HDR, rows)
    I._write_csv(os.path.join(b, 'refused.csv'), HDR + ['refusal_reason'], [])
    I._write_csv(os.path.join(b, 'ledger.csv'), ['lst', 'scope', 'stokes', 'date', 'ra', 'dec', 'final'],
                 [[r[1], r[3], r[4], r[2], r[5], r[6], '1'] for r in rows])
    I._write_csv(os.path.join(b, 'images.csv'), ['lst', 'scope', 'stokes', 'date', 'n_final'],
                 sorted({(r[1], r[3], r[4], r[2], '1') for r in rows}))
    frames = []
    for r in rows:
        st, cell, fn = r[4], f'{lst}_{r[3]}', f'{date}_{r[4]}_redo_resid_{r[3]}.fits'
        tgt = os.path.join(frames_root, cell, fn)
        if not os.path.exists(tgt):
            os.makedirs(os.path.dirname(tgt), exist_ok=True)
            open(tgt, 'w').write('frame')
        if (st, cell, fn, tgt) not in frames:
            frames.append((st, cell, fn, tgt))
    I._write_csv(os.path.join(b, 'frames.csv'), ['stokes', 'cell', 'file', 'target'], [list(f) for f in frames])
    for st, cell, fn, tgt in frames:
        I._link(tgt, os.path.join(b, 'vresid' if st == 'V' else 'iresid', cell, fn))
    I._write_csv(os.path.join(b, 'fitcache.csv'), ['key', '_fracpx', 'pos_kind'],
                 [[f'fv0123456789|{r[0]}|x|1|{r[5]}|{r[6]}|{r[7]}', '', 'template'] for r in rows])
    meta = {'epoch': epoch, 'lst': lst, 'date': date, 'n_rows': len(rows), 'n_frames': len(frames),
            'scopes': sorted({r[3] for r in rows}),
            'files': {n: I._sha1_file(os.path.join(b, n)) for n in I.BLOCK_FILES if n != 'epoch.json'}}
    I._write_json(os.path.join(b, 'epoch.json'), meta)
    return b


def test_assembly_order_independent():
    epochs, rows = synth_rows(seed=31)
    by = defaultdict(list)
    for r in rows:
        by[f'{r[1]}_{r[2]}'].append(r)
    frames_root = os.path.join(TMP, 'frames')
    digests = []
    for trial in range(3):
        out = os.path.join(TMP, f'asm_{trial}')
        ctx = _FakeCtx(out, os.path.join(TMP, 'srcrun'))
        order = sorted(by)
        random.Random(trial).shuffle(order)
        for e in order:
            rr = list(by[e])
            random.Random(trial + 7).shuffle(rr)          # rows inside a block in any order too
            write_block(ctx.state_dir, e, rr, frames_root)
        ex = _FakeExcl({('05h', 'Green', '2025-01-12')})
        summ = I.assemble(ctx, I.epochs_in_state(ctx.state_dir), ex)
        digests.append((summ['merge_output_sha1'], I._sha1_file(os.path.join(out, 'v_deep_ledger.csv')),
                        I._sha1_file(os.path.join(out, 'v_deep_images.csv'))))
        hdr, mc = I._read_csv(os.path.join(out, 'master_catalog.csv'))
        assert not any(r[1] == '05h' and r[3] == 'Green' and r[2] == '2025-01-12' for r in mc)
        assert summ['rows_excluded_now'] > 0
        nlinks = sum(len(fs) for _, _, fs in os.walk(os.path.join(out, 'vresid')))
    ok('assembly.order_independent', len(set(digests)) == 1,
       f'3 block orders -> merge output sha1 {digests[0][0][:12]}; {summ["rows"]} rows, {summ["rows_excluded_now"]} excluded '
       f'now; {nlinks} frame links')
    # a block with a cand_id already present in another block refuses
    out = os.path.join(TMP, 'asm_0')
    ctx = _FakeCtx(out, os.path.join(TMP, 'srcrun'))
    e0 = I.epochs_in_state(ctx.state_dir)[0]
    dup = list(by[e0][:1])
    write_block(ctx.state_dir, '07h_2025-02-01', [[dup[0][0], '07h', '2025-02-01'] + dup[0][3:]], frames_root)
    try:
        I.assemble(ctx, I.epochs_in_state(ctx.state_dir), _FakeExcl(()))
        refused = False
    except I.Refusal:
        refused = True
    ok('assembly.duplicate_cand_id_refuses', refused)
    shutil.rmtree(os.path.join(ctx.state_dir, 'epochs', '07h_2025-02-01'))
    return out


def test_frame_caches_and_union_fresh():
    """review 2026-10-13: the conformance frame caches of the workspace keep only rows whose frame is unchanged since the
    previous assembly (a re-extracted epoch's new frame is read again, as in a full rebuild into an empty --out), and the
    union moves the previous union files aside (a band that left every primary tier keeps no stale union file)."""
    out = os.path.join(TMP, 'asm_1')
    ctx = _FakeCtx(out, os.path.join(TMP, 'srcrun'))
    links = sorted(os.path.join(dp, f) for sub in ('vresid', 'iresid') for dp, _, fs in os.walk(os.path.join(out, sub))
                   for f in fs)
    keys = [I._frame_key(p).split('|') for p in links]
    I._write_csv(os.path.join(out, 'frame_times.csv'), ['stokes', 'lst', 'band', 'date', 'date_obs', 'date_end'],
                 [k + ['x', 'y'] for k in keys] + [['V', '23h', 'Gone', '2030-01-01', 'x', 'y']])
    I.assemble(ctx, I.epochs_in_state(ctx.state_dir), _FakeExcl(()))
    kept = I._read_csv(os.path.join(out, 'frame_times.csv'))[1]
    first = len(kept) == len(keys) and not any(r[2] == 'Gone' for r in kept)
    os.utime(os.path.realpath(links[0]), (1e9, 1e9))      # the frame of one key changed (a re-extracted epoch)
    I.assemble(ctx, I.epochs_in_state(ctx.state_dir), _FakeExcl(()))
    kept2 = {tuple(r[:4]) for r in I._read_csv(os.path.join(out, 'frame_times.csv'))[1]}
    second = tuple(keys[0]) not in kept2 and len(kept2) == len(keys) - 1
    ok('assembly.frame_caches_follow_the_frames', first and second,
       f'{len(keys)} keys cached; a key without a frame dropped: {first}; the key whose frame changed dropped: {second}')
    sc = os.path.join(out, 'source_catalogs')
    os.makedirs(sc, exist_ok=True)
    I._write_csv(os.path.join(sc, 'Gone_sources_survivors.csv'), ['source_id', 'band'], [['J1', 'Gone']])
    tdir = os.path.join(out, 'source_catalogs_V_wb')
    os.makedirs(tdir, exist_ok=True)
    I._write_csv(os.path.join(tdir, 'Full_sources_survivors.csv'), ['source_id', 'band'], [['J2', 'Full']])
    written = I.union_tiers(out)
    ok('union.previous_files_moved_aside', not os.path.exists(os.path.join(sc, 'Gone_sources_survivors.csv'))
       and os.path.join(sc, 'Full_sources_survivors.csv') in written,
       f'written {[os.path.basename(w) for w in written]}; Gone_sources_survivors.csv superseded')


# ---------------------------------------------------------------------------------------------------------------------
# content tests on synthetic products
# ---------------------------------------------------------------------------------------------------------------------

def test_content_tests(ns, asm_out):
    K = float(ns['K_ASSOC'])
    epochs, rows = synth_rows(seed=41)
    sel = canonical(rows, epochs)
    run = os.path.join(TMP, 'ctrun')
    os.makedirs(run, exist_ok=True)
    I._write_csv(os.path.join(run, 'selected_V_wb.csv'), HDR, sel)
    cat = os.path.join(run, 'source_catalogs_V_wb')
    os.makedirs(cat, exist_ok=True)
    json.dump({'beam_as': BEAM}, open(os.path.join(cat, 'E_params.json'), 'w'))
    res = I.associate_key(ns, HDR, sel, BEAM, None, K=K)
    for band, sl in res['sources'].items():         # the members file the builder would write (:819-828)
        with open(os.path.join(cat, f'{band}_source_members.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['source_id', 'band', 'cand_id', 'det_ra', 'det_dec', 'det_snr', 'det_date', 'det_epoch_qa',
                        'det_stat_err_arcsec'])
            for s in sl:
                for m in s['members']:
                    w.writerow([s['source_id'], band, m, '', '', '', '', '', ''])
    r = validate.run('incremental_partition', cat)
    ok('content.incremental_partition.good', all(x['passed'] for x in r), r[0]['detail'])
    band = next(b for b, sl in res['sources'].items() if sum(1 for s in sl if s['n_det'] > 1) >= 1 and len(sl) > 1)
    p = os.path.join(cat, f'{band}_source_members.csv')
    lines = open(p).read().splitlines()
    rw = [l.split(',') for l in lines[1:]]
    a = next(i for i, x in enumerate(rw) if x[0] != rw[0][0])
    rw[0][0], rw[a][0] = rw[a][0], rw[0][0]
    open(p, 'w').write('\n'.join([lines[0]] + [','.join(x) for x in rw]) + '\n')
    r = validate.run('incremental_partition', cat)
    ok('content.incremental_partition.corrupt_fails', not all(x['passed'] for x in r), r[0]['detail'])

    # incremental_state + incremental_delta on a state written by the engine
    sd = os.path.join(TMP, 'ctstate')
    os.makedirs(sd, exist_ok=True)
    I.write_engine_state(sd, 'V_wb', res)
    delta = []
    for b in res['sources']:
        delta += I.classify_delta([], res['sources'][b])
    I._write_csv(os.path.join(sd, 'delta_V_wb.csv'), list(I.DELTA_FIELDS), [[str(x[f]) for f in I.DELTA_FIELDS] for x in delta])
    st = {'version': I.STATE_VERSION, 'engine': {'builder_sha1': ns['__sha1__'], 'K_ASSOC': K}, 'epochs': {},
          'keys': {'V_wb': {'n_dets': sum(len(v) for v in res['dets'].values()),
                            'partition_sha1': I.partition_digest(res['dets'])}}}
    I._write_json(os.path.join(sd, 'state.json'), st)
    r = validate.run('incremental_state', sd)
    ok('content.incremental_state.good', all(x['passed'] for x in r), r[0]['detail'])
    r = validate.run('incremental_delta', os.path.join(sd, 'delta_V_wb.csv'))
    ok('content.incremental_delta.good', all(x['passed'] for x in r), r[0]['detail'])
    # corrupt: merge two stored components without the pair passing the criterion
    hdr, er = I._read_csv(os.path.join(sd, 'engine_V_wb.csv'))
    ic = hdr.index('comp')
    comps = sorted({x[ic] for x in er})
    for x in er:
        if x[ic] == comps[1]:
            x[ic] = comps[0]
    I._write_csv(os.path.join(sd, 'engine_V_wb.csv'), hdr, er)
    st['keys']['V_wb']['partition_sha1'] = I.partition_digest(I.read_engine_dets(sd, 'V_wb'))
    I._write_json(os.path.join(sd, 'state.json'), st)
    r = validate.run('incremental_state', sd)
    ok('content.incremental_state.corrupt_fails', not all(x['passed'] for x in r), r[0]['detail'][:160])
    hdr, dr = I._read_csv(os.path.join(sd, 'delta_V_wb.csv'))
    I._write_csv(os.path.join(sd, 'delta_V_wb.csv'), hdr, dr[1:])
    r = validate.run('incremental_delta', os.path.join(sd, 'delta_V_wb.csv'))
    ok('content.incremental_delta.corrupt_fails', not all(x['passed'] for x in r), r[0]['detail'])

    # epoch_block on an assembled block
    b = os.path.join(asm_out, '_incremental', 'epochs', I.epochs_in_state(os.path.join(asm_out, '_incremental'))[0])
    r = validate.run('epoch_block', b)
    ok('content.epoch_block.good', all(x['passed'] for x in r), r[0]['detail'])
    _, fr = I._read_csv(os.path.join(b, 'frames.csv'))
    os.unlink(os.path.join(b, 'vresid' if fr[0][0] == 'V' else 'iresid', fr[0][1], fr[0][2]))
    r = validate.run('epoch_block', b)
    ok('content.epoch_block.corrupt_fails', not all(x['passed'] for x in r), r[0]['detail'])


# ---------------------------------------------------------------------------------------------------------------------
# real products (read-only)
# ---------------------------------------------------------------------------------------------------------------------

def test_real_partitions(slow):
    keys = [('V_wb', True), ('I_wb', True), ('I_fwb', True)] + ([('V_sb', False)] if slow else [])
    for K, expect in keys:
        d = os.path.join(RUN, f'source_catalogs_{K}')
        t0 = time.time()
        r = validate.run('incremental_partition', d)
        passed = all(x['passed'] for x in r)
        print(f"  real incremental_partition {K}: {'PASS' if passed else 'FAIL'} ({time.time() - t0:.0f} s) {r[0]['detail'][:300]}")
        ok(f'real.incremental_partition.{K}.expected_{"pass" if expect else "fail"}', passed == expect, r[0]['detail'][:200])


def test_real_engine_batches(ns):
    K = float(ns['K_ASSOC'])
    sel_p = os.path.join(RUN, 'selected_V_wb.csv')
    cat = os.path.join(RUN, 'source_catalogs_V_wb')
    hdr, rows = I._read_csv(sel_p)
    beam = json.load(open(os.path.join(cat, 'E_params.json'))).get('beam_as', {})
    il, idt = hdr.index('lst'), hdr.index('date')
    epochs = sorted({(r[il], r[idt]) for r in rows})
    rng = random.Random(5)
    order = list(epochs)
    rng.shuffle(order)
    batches = [order[i::6] for i in range(6)]
    pos = {tuple(r): i for i, r in enumerate(rows)}       # the production input order of the full set
    state, added = None, set()
    t0 = time.time()
    for bt in batches:
        added |= set(bt)
        sel = sorted([r for r in rows if (r[il], r[idt]) in added], key=lambda r: pos[tuple(r)])
        res = I.associate_key(ns, hdr, sel, beam, None, prev_dets=state, K=K)
        state = res['dets']
    good, det = I.crosscheck(res, cat)
    ok('real.engine.V_wb.6_batches==production', good,
       f'{len(rows)} detections in {len(epochs)} epochs added in 6 random batches ({time.time() - t0:.0f} s): {det}')


def test_real_dry_run():
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import record
    epoch = '04h_2024-12-18'
    par = sorted(p for p in (os.listdir('/fast/gh/main/phase3') if os.path.isdir('/fast/gh/main/phase3') else [])
                 if p.startswith('p3s5_parents_union_') and os.path.isdir(os.path.join('/fast/gh/main/phase3', p)))
    out = os.path.join(TMP, 'dryrun_out')
    args = argparse.Namespace(tier='deep', run_label='P3S5', epochs=epoch, inc_phase=None, inc_source=None,
                              inc_parents=os.path.join('/fast/gh/main/phase3', par[-1]) if par else None,
                              inc_rebuild=False, parents=None)
    run = record.Run('catalog', ['test'], out, CFG, dry_run=True)
    rc = I.run_incremental(args, run)
    wrote = os.path.exists(out) and any(files for _, _, files in os.walk(out))
    ok('real.dry_run', rc == 0 and not wrote and [u.status for u in run.units] == ['no-input', 'no-input'],
       f'units {[(u.name, u.status, u.reason) for u in run.units]}; nothing written: {not wrote}')


def make_synth_run(root, epoch, n_det=6, seed=3):
    """A synthetic deep run with one epoch: merge output (master_catalog_asrun.csv), ledger, images, refused rows and REAL
    FITS search frames (SIN, 0.03 deg pixels, 0.15 deg beam, unit noise, Gaussian sources at the detections) under
    vresid/<cell>/ - enough for the in-place deep_fit_positions.py to fit every detection."""
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS
    rng = np.random.default_rng(seed)
    lst, date = I.parse_epoch(epoch)
    ra0, dec0 = 15.0 * (int(lst[:2]) + 0.5), 40.0
    rows, ledger = [], []
    for band in ('Red', 'Green'):
        n, cd, bm = 200, 0.03, 0.15
        w = WCS(naxis=2)
        w.wcs.ctype = ['RA---SIN', 'DEC--SIN']
        w.wcs.crval = [ra0, dec0]
        w.wcs.crpix = [n / 2 + 0.5, n / 2 + 0.5]
        w.wcs.cdelt = [-cd, cd]
        d = rng.normal(0, 1.0, (n, n)).astype('float32')
        yy, xx = np.mgrid[0:n, 0:n]
        sig = bm / cd / 2.3548
        for k in range(n_det):
            x, y = rng.uniform(40, n - 40), rng.uniform(40, n - 40)
            snr = float(rng.uniform(8, 20)) if k % 2 == 0 else float(rng.uniform(5.5, 6.9))
            d += (snr * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sig ** 2))).astype('float32')
            ra, dec = [float(v) for v in w.pixel_to_world_values(round(x), round(y))]
            cid = f'VD_{lst}_{date}_{band}_{k:04d}'
            rows.append([cid, lst, date, band, 'V', 'deep', f'{ra % 360:.5f}', f'{dec:.5f}', f'{snr:.2f}'])
            ledger.append([lst, band, 'V', date, f'{ra % 360:.5f}', f'{dec:.5f}', '1'])
        h = w.to_header()
        h['BMAJ'], h['BMIN'], h['BPA'] = bm, bm, 0.0
        h['MJD-MID'] = 60690.3
        h['DATE-OBS'], h['DATE-END'] = f'{date}T08:00:00', f'{date}T09:00:00'
        p = os.path.join(root, 'vresid', f'{lst}_{band}', f'{date}_V_redo_resid_{band}.fits')
        os.makedirs(os.path.dirname(p), exist_ok=True)
        fits.PrimaryHDU(d, h).writeto(p, overwrite=True)
    I._write_csv(os.path.join(root, 'master_catalog_asrun.csv'),
                 ['cand_id', 'lst', 'date', 'band', 'stokes', 'cadence', 'ra', 'dec', 'snr'], rows)
    I._write_csv(os.path.join(root, 'master_catalog_refused.csv'),
                 ['cand_id', 'lst', 'date', 'band', 'stokes', 'cadence', 'ra', 'dec', 'snr', 'refusal_reason'], [])
    I._write_csv(os.path.join(root, 'v_deep_ledger.csv'), ['lst', 'scope', 'stokes', 'date', 'ra', 'dec', 'final'], ledger)
    I._write_csv(os.path.join(root, 'v_deep_images.csv'), ['lst', 'scope', 'stokes', 'date', 'n_final'],
                 [[lst, b, 'V', date, str(n_det)] for b in ('Red', 'Green')])
    return rows


def test_b30_sstack_threshold():
    """B-30 (a): the sigma-stack threshold read from the production search_i_transients.py; an unreadable source refuses
    under the ruled value (fallbacks.CATALOG_SSTACK_THRESHOLD_FALLBACK) and gives the builder's 7.0, counted, under 'fallback'."""
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import fallback
    k = 'fallbacks.CATALOG_SSTACK_THRESHOLD_FALLBACK'
    m = fallback.mark()
    thr, src = I.sstack_threshold()
    ok('B-30: the readable production source gives its threshold, no fallback', thr == 7.0 and 'CONFIG[SIGMA_THRESHOLD]' in src
       and fallback.since(m) == ([], {}), (thr, src, fallback.since(m)))
    sec = CFG.catalog_incremental
    real = sec.SSTACK_SOURCE              # a PROTECTED key: patched on the section for this check, restored below
    for pol in ('refuse', 'fallback'):
        CFG.reset()
        CFG.authorize([], out=TMP)
        CFG.set(k, pol)
        sec.SSTACK_SOURCE = os.path.join(TMP, 'no_such_search_i.py')
        m = fallback.mark()
        try:
            got = I.sstack_threshold()
        except fallback.FallbackRefused as e:
            got = str(e)
        finally:
            sec.SSTACK_SOURCE = real
        st = fallback.since(m)
        CFG.reset()
        if pol == 'refuse':
            ok('B-30: an unreadable threshold source refuses under the ruled value', isinstance(got, str) and k in got, got)
        else:
            ok('B-30: ... and gives the builder fallback 7.0 under fallback, counted', got[0] == 7.0 and st[1] == {k: 1},
               (got, st))


def test_epoch_phase_end_to_end():
    """The epoch phase on a synthetic run: the in-place deep_fit_positions.py runs through core.legacy.run, the block passes
    epoch_block; a second run of the same epoch takes every fit from the cache (deep_fit_positions' own cache key) and
    writes a byte-identical block; an excluded scope is left out at assembly."""
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import record
    src = os.path.join(TMP, 'synthrun')
    epoch = '05h_2030-01-12'   # a date with no data_decisions row (05h 2025-01-12 is excluded: the gate refuses it)
    rows = make_synth_run(src, epoch)
    out = os.path.join(TMP, 'e2e_out')
    args = argparse.Namespace(tier='deep', run_label='P3S5', epochs=epoch, inc_phase='epoch', inc_source=f'run:{src}',
                              inc_parents=None, inc_rebuild=False, parents=None)
    sums = []
    for attempt in range(2):
        run = record.Run('catalog', ['test'], out, CFG)
        I.run_incremental(args, run)
        u = run.units[0]
        rec = json.load(open(record.record_path(out, 'catalog', u.name)))
        b = os.path.join(out, '_incremental', 'epochs', epoch)
        summ = json.load(open(os.path.join(b, 'fit_summary.json')))
        sums.append((I._sha1_file(os.path.join(b, 'fitcache.csv')), summ['cached'], summ['fitted_now']))
        ok(f'e2e.epoch_phase.run{attempt + 1}', u.status == 'done' and rec['status'] == 'done'
           and all(t['passed'] for t in rec['tests']) and rec['meta'].get('legacy_calls'),
           f"{u.status} {u.reason}; fits: {summ['kinds']} cached {summ['cached']} fitted now {summ['fitted_now']}; "
           f"legacy calls {len(rec['meta'].get('legacy_calls') or [])}, child code files {len(rec['meta'].get('legacy_code') or {})}")
    ok('e2e.epoch_phase.second_run_from_cache', sums[0][0] == sums[1][0] and sums[1][1] == len(rows) and sums[1][2] == 0,
       f'run 1 fitted {sums[0][2]}, run 2 cached {sums[1][1]} / fitted {sums[1][2]}; fit cache identical: {sums[0][0] == sums[1][0]}')
    ctx = _FakeCtx(os.path.join(TMP, 'e2e_asm'), src)
    os.makedirs(ctx.state_dir, exist_ok=True)
    shutil.copytree(os.path.join(out, '_incremental', 'epochs'), os.path.join(ctx.state_dir, 'epochs'), symlinks=True)
    summ = I.assemble(ctx, [epoch], _FakeExcl({('05h', 'Green', '2030-01-12')}))
    hdr, mc = I._read_csv(os.path.join(ctx.out, 'master_catalog.csv'))
    fc = I._read_csv(os.path.join(ctx.out, 'fit_positions_cache.csv'))[1]
    ok('e2e.assemble_excludes_scope', summ['rows'] == len(rows) // 2 and all(r[3] == 'Red' for r in mc) and len(fc) == len(rows)
       and not os.path.exists(os.path.join(ctx.out, 'vresid', '05h_Green')),
       f"{summ['rows']} rows (Green excluded: {summ['rows_excluded_now']}), {len(fc)} fit-cache rows carried")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-real', action='store_true')
    ap.add_argument('--slow', action='store_true')
    a = ap.parse_args()
    t0 = time.time()
    shutil.rmtree(TMP, ignore_errors=True)
    os.makedirs(TMP)
    try:
        ns = I.builder_namespace()
        I.binding_check(ns)
        print(f'builder {ns["__path__"]} sha1 {ns["__sha1__"][:12]}; K_ASSOC {ns["K_ASSOC"]}; sigma-stack {I.sstack_threshold()}')
        test_engine_incremental_equals_full(ns)
        test_engine_removal_and_change(ns)
        test_classify_delta()
        asm = test_assembly_order_independent()
        test_frame_caches_and_union_fresh()
        test_content_tests(ns, asm)
        test_epoch_phase_end_to_end()
        test_b30_sstack_threshold()
        if not a.no_real and os.path.isdir(RUN):
            test_real_partitions(a.slow)
            test_real_engine_batches(ns)
            test_real_dry_run()
        elif not a.no_real:
            print(f'SKIP real checks: {RUN} not readable from this host')
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    n_fail = sum(1 for r in RESULTS if not r[1])
    print(f'\n{len(RESULTS) - n_fail} of {len(RESULTS)} checks passed in {time.time() - t0:.0f} s')
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
