#!/usr/bin/env python3
"""Data-free test of lib/pages/night.py and lib/pages/wraps/prep_catalog.py: the pages of a night (design 2.3a, stage 3).

Synthetic page runs of one set (V_sb) over a synthetic search tree, with a STUB renderer written from
build_tenmin_pages.py's own lines (the rows it keeps, its page order, its image names, its skip of a frame without a
searched image). Checks: the mirror of the renderer's rules and that production's renderer and prep still hold the mirrored
lines; an all-nights page run as the base and a night's run on it (which rows are carried, which rendered and why; the
reduced selection; the rename to the full selection's ranks; hard links; the content test) against a full render of the
night's selection (same image names, same bytes); every reason a whole set or a single row is rendered; the leave-one-out
window Masters carried or not; the staged catalog directory and its refusals; the code digest across two frozen trees; the
prep runner's pins, patch and refusals, and production's script against the configured pins.

Scratch: /tmp/exopipe3_tests_pages_night (removed at the end unless EXOPIPE3_TEST_KEEP is set).
"""
import csv
import errno
import glob
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')
from exopipe.phase3.lib.pages import night as NV                    # noqa: E402
from exopipe.phase3.lib.pages.wraps import prep_catalog as PC       # noqa: E402

T = '/tmp/exopipe3_tests_pages_night'
OPS = '/lustre/gh/main/exopipe_ops'
K = 'V_sb'
SUB = f'{T}/sub'
SR = f'{SUB}/search_B'
N1, N2, N3 = '2025-04-05', '2025-04-06', '2025-05-08'
HEAD = 'cand_id,member_cand_id,lst,date,band,stokes,snr,peak_jy,ra,dec'
N = [0]
FAILS = []


def check(name, ok, detail=''):
    N[0] += 1
    print(('PASS  ' if ok else 'FAIL  ') + name + (f'  ({detail})' if (detail and not ok) else ''))
    if not ok:
        FAILS.append(name)


def line(cid, mid, night, k, snr, peak='1.0'):
    return f'{cid},{mid},17h,{night}T{k:02d},41MHz,V,{snr},{peak},250.0,40.0'


def put(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        f.write(text)


def resid(night, k):
    return f'{SR}/17h_41MHz_V/Run_X/vresid/17h_41MHz/{night}T{k:02d}_V_redo_resid_41MHz.fits'


def stack(night, k):
    return f'{SUB}/17h/17h_{night}/41MHz/V/sublst_17h_{night}_41MHz_V_w{k}.fits'


def world():
    """the search tree: residual images and window stacks of every frame but (N2, window 2)"""
    for night in (N1, N2, N3):
        for k in range(4):
            if (night, k) == (N2, 2):
                continue
            put(resid(night, k), f'resid {night} {k}')
            put(stack(night, k), f'stack {night} {k}')
            put(stack(night, k)[:-5] + '_resid_loo.fits', f'loo {night} {k}')
    put(f'{SR}/17h_41MHz_V/v110_out/VD_P3_17h_41MHz_{N1}T00_0000_17h_{N1}T00_41MHz_predicted_cutout.fits', 'pred')
    put(f'{T}/parents/master_R075_17h_41MHz.gaul.csv', 'parent\n')
    put(f'{T}/parents2/master_R075_17h_41MHz.gaul.csv', 'parent\n')


def stub_prep(P, lines):
    """the leave-one-out window Masters prep writes: those the run does not hold (sublst_pages_prep.py:150-161)"""
    n = 0
    for r in csv.DictReader([HEAD] + lines):
        if 'T' not in r['date'] or r['snr'] == 'nan':
            continue
        night, b = r['date'].split('T')
        o = f'{P}/loo_masters/17h_{night}/41MHz/V/loo_master_w{int(b)}.fits'
        if os.path.exists(o) or not os.path.exists(stack(night, int(b))):
            continue
        put(o, f'master {night} {b}')
        n += 1
    return n


def mk_run(P, lines, frozen, pertie, tie=None, view_of=None, base=None, ident=None):
    """a page run before its build: selection, catalog directory, tables, links, Masters (carried from ``base`` first)"""
    put(f'{P}/source_catalogs/tenmin_selection_{K}.csv', '\n'.join([HEAD] + lines) + '\n')
    if view_of:
        put(f'{P}/source_catalogs_{K}/catalog_view.json',
            json.dumps(dict(frozen_record_sha1=NV._sha1(f'{view_of}/source_catalogs_{K}/builder_frozen.json'))))
    else:
        put(f'{P}/source_catalogs_{K}/builder_frozen.json', frozen)
    put(f'{P}/source_catalogs_{K}/E_params.json', '{"E": 1}')
    put(f'{P}/pertie_detections.csv', 'cand_id,sys_arcsec\n' + ''.join(f'{c},{v}\n' for c, v in pertie.items()))
    if tie is not None:
        put(f'{P}/tie_members.csv', 'cand_id,ref\n' + ''.join(f'{c},{v}\n' for c, v in tie.items()))
    os.symlink(f'{T}/parents', f'{P}/parent_catalogs')
    os.makedirs(f'{P}/v110_products')
    os.symlink(f'{SR}/17h_41MHz_V/v110_out', f'{P}/v110_products/17h_41MHz')
    lm = NV.link_masters(base, P, ident, SUB) if base else None
    return lm, stub_prep(P, lines)


def stub_render(sel, run):
    """build_tenmin_pages.py in miniature (its lines 329-340, 352-356, 362, 387, 422-425): one file per kept row whose frame
    has a searched image, named by the row's rank in THIS selection file; the content is the row"""
    rows = []
    for r in csv.DictReader(open(sel)):
        d = r.get('date', '')
        if 'T' not in d:
            continue
        night, b = d.split('T', 1)
        if not b.isdigit():
            continue
        try:
            s = abs(float(r.get('snr')))
        except (TypeError, ValueError):
            s = float('nan')
        if not math.isfinite(s):
            continue
        r['_night'], r['_bin'], r['_snr'] = night, int(b), s
        rows.append(r)
    rows.sort(key=lambda r: -r['_snr'])
    n = 0
    for i, r in enumerate(rows):
        r['_rank'] = i + 1
    for r in rows:
        if not os.path.exists(resid(r['_night'], r['_bin'])):
            continue
        S = r.get('stokes') or 'V'
        sid = r.get('cand_id') or f"{r['lst']}_{r['date']}_{r['band']}_{r['_rank']:05d}"
        ddir = f'{run}/candidates/{S}/{sid}'
        os.makedirs(ddir, exist_ok=True)
        with open(f'{ddir}/rank{r["_rank"]:05d}_{r["lst"]}_{r["_night"]}_bin{r["_bin"]}_{r["band"]}.png', 'w') as f:
            json.dump({k: v for k, v in r.items() if not k.startswith('_')}, f, sort_keys=True)
        n += 1
    return n


def build(P, base, ident):
    """the build stage of lib/pages/tier_sublst.py in miniature"""
    frames = NV.run_frames(P, [K], ident['search_root'], SUB)
    pl = NV.plan(K, P, base, ident, frames)
    sel, moves = NV.reduced(pl, P)
    rendered = stub_render(sel, P) if sel else 0
    moved, none = NV.rename(P, moves)
    got = NV.link_blocks(base, P, pl) if pl['link'] else dict(linked=0, copied=0)
    ok, detail, counts = NV.test(P, [K], frames)
    NV.write_record(P, dict(version=NV.VERSION, complete=bool(ok), frames=frames, **ident))
    return dict(pl=pl, sel=sel, moves=moves, rendered=rendered, moved=moved, none=none, got=got, ok=ok, detail=detail,
                counts=counts, frames=frames)


def images(P):
    return sorted(os.path.relpath(p, P) for p in glob.glob(f'{P}/candidates/*/*/*'))


BASE_LINES = [line('A', 'A1', N1, 0, 9), line('B', 'B1', N1, 1, 8), line('C', 'C1', N2, 0, -7.5), line('D', 'D1', N2, 2, 7),
              line('E', 'E1', N1, 3, 6), line('F', 'F1', N2, 3, 5), line('A', 'A2', N2, 0, 4.5),
              'X,X1,17h,nodate,41MHz,V,50,1.0,250.0,40.0', line('Y', 'Y1', N1, 2, 'nan')]
NEW_LINES = [line('A', 'A1', N1, 0, 9), line('B', 'B1', N1, 1, 8), line('D', 'D1', N2, 2, 7),
             line('E', 'E1', N1, 3, 6, peak='1.1'), line('F', 'F1', N2, 3, 5), line('A', 'A2', N2, 0, 4.5),
             'X,X1,17h,nodate,41MHz,V,50,1.0,250.0,40.0', line('Y', 'Y1', N1, 2, 'nan'),
             line('H', 'H1', N3, 1, 8.5), line('A', 'A3', N3, 2, 3)]
PERTIE = dict(A1='1', A2='1', B1='1', C1='1', D1='1', E1='1', F1='1')
PERTIE_NEW = dict(PERTIE, F1='2', H1='1', A3='1')


def test_mirror():
    rows = list(csv.DictReader([HEAD] + BASE_LINES))
    rk = NV.ranked(rows)
    check('ranked: a row without <night>T<digits> and a row without a finite S/N are dropped', len(rk) == 7, str(rk))
    check('ranked: falling |S/N| (a negative S/N by its size), ranks from 1',
          [(r, rows[i]['member_cand_id']) for r, i, _n, _k in rk]
          == [(1, 'A1'), (2, 'B1'), (3, 'C1'), (4, 'D1'), (5, 'E1'), (6, 'F1'), (7, 'A2')], str(rk))
    tie = list(csv.DictReader([HEAD, line('P', 'P1', N1, 0, 5), line('Q', 'Q1', N1, 1, -5), line('R', 'R1', N1, 2, 5)]))
    check('ranked: equal |S/N| keeps the file order', [tie[i]['cand_id'] for _r, i, _n, _k in NV.ranked(tie)] == ['P', 'Q', 'R'])
    r, i, night, k = rk[2]
    check('block_rel: candidates/<Stokes>/<page id>/rank<5 digits>_<LST>_<night>_bin<k>_<band>.png',
          NV.block_rel(rows[i], r, night, k) == f'candidates/V/C/rank00003_17h_{N2}_bin0_41MHz.png', NV.block_rel(rows[i], r, night, k))
    nocid = dict(rows[0], cand_id='', stokes='')
    check('block_rel: a row without page id is named by its rank; no Stokes reads V',
          NV.block_rel(nocid, 4, N1, 0) == f'candidates/V/17h_{N1}T00_41MHz_00004/rank00004_17h_{N1}_bin0_41MHz.png')
    check('frames_of: the frame keys of the kept rows', len(NV.frames_of(rows)) == 6 and ('17h', N2, 0, '41MHz') in NV.frames_of(rows))
    src = open(f'{OPS}/build_tenmin_pages.py').read()
    for what, text in (
            ('page order', "rows.sort(key=lambda r: -r['_snr'])"),
            ('rank', "for i, r in enumerate(rows): r['_rank'] = i + 1"),
            ('kept rows', "if not np.isfinite(r['_snr']): continue"),
            ('page id', '''sid = r.get('cand_id') or f"{lst}_{r['date']}_{band}_{r['_rank']:05d}"'''),
            ('block directory', "ddir = f'{RUN}/candidates/{S}/{sid}'"),
            ('image name', '''fig.savefig(f'{ddir}/rank{r["_rank"]:05d}_{lst}_{night}_bin{k}_{band}.png', dpi=110)'''),
            ('searched image', '''f"{night}T{k:02d}_{S}_redo_resid_{'pbcorr_' if pbcorr else ''}{band}.fits")'''),
            ('window Master', "p = f'{_SUBLST_M}/{lst}_{night}/{band}/{S}/loo_master_w{k}.fits'"),
            ('predicted V', "glob.glob(f'{RUN}/v110_products/{lst}_{band}/*_predicted_cutout.fits')"),
            ('parent catalog', "pth = f'{RUN}/parent_catalogs/master_R075_{lst}_{band}.gaul.csv'")):
        check(f'production build_tenmin_pages.py still holds the mirrored line: {what}', src.count(text) == 1, text)
    src = open(f'{OPS}/sublst_pages_prep.py').read()
    for what, text in (('window stack', "a = f'{SL}/{LST}/{LST}_{night}/{band}/{S}/sublst_{LST}_{night}_{band}_{S}_w{k}.fits'"),
                       ('leave-one-out residual', "r = a[:-5] + '_resid_loo.fits'"),
                       ('written only when absent', 'if os.path.exists(o) or not (os.path.exists(a) and os.path.exists(r)):')):
        check(f'production sublst_pages_prep.py still holds the mirrored line: {what}', src.count(text) == 1, text)


def test_flow():
    ident = dict(code='c1', samples={'10pc': [10, 20]}, search_root=SR)
    B, P, F = f'{T}/base', f'{T}/night', f'{T}/full'
    mk_run(B, BASE_LINES, 'frozen-1', PERTIE)
    b = build(B, None, ident)
    check('base run: no earlier run, the whole set is rendered from the full selection',
          b['pl']['whole'] and b['sel'] == b['pl']['selection'] and b['rendered'] == 6 and not b['moves'], str(b['pl']['whole']))
    check('base run: content test passes; 6 images, 1 row without (its frame has no searched image)',
          b['ok'] and b['counts'][K] == dict(rows=7, images=6, without_image=1), b['detail'])
    lm, wrote = mk_run(P, NEW_LINES, None, PERTIE_NEW, view_of=B, base=B, ident=ident)
    check('night run: the earlier run\'s window Masters are carried, the new night\'s are written',
          lm == dict(carried=5, not_carried=0, why='') and wrote == 2, f'{lm} {wrote}')
    n = build(P, B, ident)
    pl = n['pl']
    mids = lambda poss: sorted(pl['rows'][pl['blocks'][p][1]]['member_cand_id'] for p in poss)    # noqa: E731
    check('night run: unchanged rows are carried', mids([p for p, _b in pl['link']]) == ['A1', 'A2', 'B1'], str(mids([p for p, _b in pl['link']])))
    check('night run: changed and new rows are rendered', mids(pl['render']) == ['A3', 'D1', 'E1', 'F1', 'H1'], str(mids(pl['render'])))
    check('night run: the reasons', pl['why'] == {'the row changed': 1, 'the window-tie row changed': 1,
                                                  'the row is not in the earlier page run': 2,
                                                  'the earlier page run holds no image of the row': 1}, str(pl['why']))
    red = open(n['sel']).read().split('\n')
    check('night run: the reduced selection holds the 5 rows, in the full file\'s order, under the set\'s name',
          os.path.basename(n['sel']) == f'tenmin_selection_{K}.csv' and n['sel'] != pl['selection']
          and red[0] == HEAD and [x.split(',')[1] for x in red[1:-1]] == ['D1', 'E1', 'F1', 'H1', 'A3'], str(red))
    check('night run: 4 images rendered and renamed to the full ranks, 1 row without image; 3 hard links',
          n['rendered'] == 4 and n['moved'] == 4 and len(n['none']) == 1 and n['got'] == dict(linked=3, copied=0), str(n))
    check('night run: content test passes (7 images of 8 rows)', n['ok'] and n['counts'][K] == dict(rows=8, images=7, without_image=1), n['detail'])
    want = [f'candidates/V/A/rank00001_17h_{N1}_bin0_41MHz.png', f'candidates/V/A/rank00007_17h_{N2}_bin0_41MHz.png',
            f'candidates/V/A/rank00008_17h_{N3}_bin2_41MHz.png', f'candidates/V/B/rank00003_17h_{N1}_bin1_41MHz.png',
            f'candidates/V/E/rank00005_17h_{N1}_bin3_41MHz.png', f'candidates/V/F/rank00006_17h_{N2}_bin3_41MHz.png',
            f'candidates/V/H/rank00002_17h_{N3}_bin1_41MHz.png']
    check('night run: every image under its rank in the full selection', images(P) == want, str(images(P)))
    check('night run: a carried image is the earlier run\'s file (B: rank 2 there, rank 3 here)',
          os.stat(f'{P}/{want[3]}').st_ino == os.stat(f'{B}/candidates/V/B/rank00002_17h_{N1}_bin1_41MHz.png').st_ino)
    mk_run(F, NEW_LINES, None, PERTIE_NEW, view_of=B)
    f = build(F, None, ident)
    check('night run = a full render of the night\'s selection: same image names, same bytes',
          f['ok'] and images(F) == images(P) and all(open(f'{F}/{r}').read() == open(f'{P}/{r}').read() for r in images(P)))
    check('the night run\'s record: complete, with the frames\' files', (NV.read_record(P) or {}).get('complete') is True
          and len(NV.read_record(P)['frames']) == 8)
    # a second night pass on the night's run: nothing changed but D (never an image) -> one row rendered, none drawn
    P2 = f'{T}/night2'
    mk_run(P2, NEW_LINES, None, PERTIE_NEW, view_of=B, base=P, ident=ident)
    n2 = build(P2, P, ident)
    check('a run on the night\'s run: 7 carried, the row without image tried again, same images',
          n2['got']['linked'] == 7 and n2['rendered'] == 0 and n2['ok'] and images(P2) == images(P), str(n2['got']))
    nod = [x for x in NEW_LINES if ',D1,' not in x]
    P3 = f'{T}/night3'
    mk_run(P3, nod, None, PERTIE_NEW, view_of=B, base=P, ident=ident)
    fr = NV.run_frames(P3, [K], SR, SUB)
    pl3 = NV.plan(K, P3, P, ident, fr)
    check('nothing to render: no reduced selection, no renderer', NV.reduced(pl3, P3) == (None, []) and len(pl3['link']) == 7, str(pl3['why']))
    return ident, B, P


def variant(name, src, lines=None, mutate=None):
    P = f'{T}/v_{name}'
    shutil.copytree(src, P, symlinks=True)
    shutil.rmtree(f'{P}/candidates', ignore_errors=True)
    if os.path.exists(f'{P}/{NV.RECORD}'):
        os.remove(f'{P}/{NV.RECORD}')
    if lines is not None:
        put(f'{P}/source_catalogs/tenmin_selection_{K}.csv', '\n'.join(lines) + '\n')
    if mutate:
        mutate(P)
    return P


def test_reasons(ident, B, P):
    full = [HEAD] + NEW_LINES

    def whole(name, expect, mutate=None, lines=None, base=P, idn=None, frames=True):
        V = variant(name, P, lines, mutate)
        fr = NV.run_frames(V, [K], SR, SUB) if frames else None
        pl = NV.plan(K, V, base, idn or ident, fr)
        check(f'whole set rendered: {name}', expect in pl['whole'] and len(pl['render']) == len(pl['blocks']) and not pl['link'], pl['whole'])

    def base_with(name, change):
        b = f'{T}/b_{name}'
        shutil.copytree(P, b, symlinks=True)
        change(b)
        return b

    def rec(b, **kw):
        r = NV.read_record(b)
        r.update(kw)
        NV.write_record(b, r)
    whole('no earlier run named', 'no earlier page run', base=None)
    whole('earlier run gone', 'does not exist', base=f'{T}/nowhere')
    whole('no record', f'holds no {NV.RECORD}', base=base_with('norec', lambda b: os.remove(f'{b}/{NV.RECORD}')))
    whole('record not complete', 'not complete', base=base_with('inc', lambda b: rec(b, complete=False)))
    whole('record of another version', 'another version', base=base_with('ver', lambda b: rec(b, version=0)))
    whole('code not pinned', 'not a frozen tree', idn=dict(ident, code=''))
    whole('another code', 'another code', idn=dict(ident, code='c2'))
    whole('other samples', 'other cross-match sample files', idn=dict(ident, samples={'10pc': [10, 21]}))
    whole('another search root', 'another search root', idn=dict(ident, search_root=SR + '_x'))
    whole('frames not recorded by this run', 'were not recorded', frames=False)
    whole('frames not recorded by the earlier run', 'were not recorded', base=base_with('nofr', lambda b: rec(b, frames={})))
    whole('no key columns', 'columns to name a block by', lines=[x.split(',', 2)[0] + ',' + x.split(',', 2)[2] for x in full])
    whole('a field spans lines', 'cannot be cut by lines', lines=[HEAD] + [NEW_LINES[0].replace(',9,', ',9,"1\n0",', 1).replace(',1.0,250', ',250', 1)] + NEW_LINES[1:])
    whole('a row without page id', 'no page id', lines=[HEAD] + [',' + NEW_LINES[1].split(',', 1)[1]] + NEW_LINES[2:])
    whole('no selection in the earlier run', 'holds no selection',
          base=base_with('nosel', lambda b: os.remove(f'{b}/source_catalogs/tenmin_selection_{K}.csv')))
    whole('other columns', 'columns differ', lines=[HEAD + ',extra'] + [x + ',' for x in NEW_LINES])
    whole('another all-nights build', 'all-nights build', mutate=lambda V: put(f'{V}/source_catalogs_{K}/catalog_view.json', '{"frozen_record_sha1": "x"}'))
    whole('no all-nights build named', 'all-nights build', mutate=lambda V: put(f'{V}/source_catalogs_{K}/catalog_view.json', '{}'))
    whole('E_params.json differs', 'E_params.json', mutate=lambda V: put(f'{V}/source_catalogs_{K}/E_params.json', '{"E": 2}'))

    def relink(V):
        os.remove(f'{V}/parent_catalogs')
        os.symlink(f'{T}/parents2', f'{V}/parent_catalogs')
    whole('another parent catalog directory', 'parent catalog directory', mutate=relink)
    whole('no window-tie table', 'window-tie table', mutate=lambda V: os.remove(f'{V}/pertie_detections.csv'))
    whole('reference-tie table in one run only', 'reference-tie table', mutate=lambda V: put(f'{V}/tie_members.csv', 'cand_id,ref\n'))

    def rows_why(name, expect, n, mutate=None, lines=None, base=P):
        V = variant(name, P, lines, mutate)
        pl = NV.plan(K, V, base, ident, NV.run_frames(V, [K], SR, SUB))
        check(f'row rendered: {name}', not pl['whole'] and pl['why'].get(expect) == n, f'{pl["whole"]} {pl["why"]}')
        return pl
    rows_why('the key names two rows', 'the page and member detection name more than one row', 2, lines=full + [line('B', 'B1', N1, 1, 2)])
    tb = base_with('tie', lambda b: put(f'{b}/tie_members.csv', 'cand_id,ref\nA1,r1\nB1,r1\n'))
    rows_why('reference-tie row changed', 'the reference-tie row changed', 1, base=tb,
             mutate=lambda V: put(f'{V}/tie_members.csv', 'cand_id,ref\nA1,r1\nB1,r2\n'))

    def cell(V):
        os.remove(f'{V}/v110_products/17h_41MHz')
        os.symlink(f'{T}/parents', f'{V}/v110_products/17h_41MHz')
    rows_why('another predicted-V directory', 'the predicted-V directory of the cell differs', 8, mutate=cell)
    FR = 'a file of the frame changed (searched image, predicted image, parent catalog or window Master)'
    st = os.stat(resid(N1, 1))
    os.utime(resid(N1, 1), ns=(st.st_atime_ns, st.st_mtime_ns + 1000))
    pl = rows_why('a searched image changed', FR, 1)
    check('row rendered: a searched image changed - the row of that frame only',
          [pl['rows'][pl['blocks'][p][1]]['member_cand_id'] for p in pl['render'] if pl['fkeys'][p] == NV.frame_key('17h', N1, 1, '41MHz')] == ['B1'])
    os.utime(resid(N1, 1), ns=(st.st_atime_ns, st.st_mtime_ns))
    newpred = f'{SR}/17h_41MHz_V/v110_out/VD_P3_17h_41MHz_{N2}T00_0001_17h_{N2}T00_41MHz_predicted_cutout.fits'
    put(newpred, 'pred2')
    rows_why('a predicted image of the frame appeared', FR, 1)          # A2's frame (C is no longer selected)
    os.remove(newpred)
    rows_why('a window Master of the run changed', FR, 1,
             mutate=lambda V: put(f'{V}/loo_masters/17h_{N1}/41MHz/V/loo_master_w0.fits', 'another master'))
    par = f'{T}/parents/master_R075_17h_41MHz.gaul.csv'
    st = os.stat(par)
    os.utime(par, ns=(st.st_atime_ns, st.st_mtime_ns + 1000))
    rows_why('the parent catalog changed', FR, 8)
    os.utime(par, ns=(st.st_atime_ns, st.st_mtime_ns))
    # the Masters: a changed window stack is not carried
    st = os.stat(stack(N1, 3))
    os.utime(stack(N1, 3), ns=(st.st_atime_ns, st.st_mtime_ns + 1000))
    V = f'{T}/v_masters'
    lm = NV.link_masters(P, V, ident, SUB)
    check('window Masters: the Master of a changed window stack is not carried', lm == dict(carried=6, not_carried=1, why=''), str(lm))
    check('window Masters: not carried from a run that cannot vouch',
          NV.link_masters(P, f'{T}/v_masters2', dict(ident, code='c2'), SUB)['why'] != '' and not os.path.exists(f'{T}/v_masters2'))
    os.utime(stack(N1, 3), ns=(st.st_atime_ns, st.st_mtime_ns))


def test_files(ident, B, P):
    # rename, link, the content test
    V = variant('rn', P)
    put(f'{V}/candidates/V/Z/a.png', 'a')
    put(f'{V}/candidates/V/Z/b.png', 'b')
    try:
        NV.rename(V, [(1, 'candidates/V/Z/a.png', 'candidates/V/Z/b.png')])
        ok = False
    except NV.NightRefusal as e:
        ok = 'exists already' in str(e) and str(e).endswith('REFUSING')
    check('rename: REFUSES to name an image over an existing one', ok)
    os.remove(f'{V}/candidates/V/Z/b.png')
    put(f'{V}/candidates/V/Z/rank00002.png', 'two')
    mv = NV.rename(V, [(1, 'candidates/V/Z/a.png', 'candidates/V/Z/rank00002.png'), (2, 'candidates/V/Z/rank00002.png', 'candidates/V/Z/rank00005.png'),
                       (3, 'candidates/V/Z/none.png', 'candidates/V/Z/rank00009.png')])
    check('rename: from the highest reduced rank down (rank 2 moves before rank 1 takes its name); a row without image is listed',
          mv == (2, ['candidates/V/Z/rank00009.png']) and open(f'{V}/candidates/V/Z/rank00002.png').read() == 'a'
          and open(f'{V}/candidates/V/Z/rank00005.png').read() == 'two')
    shutil.rmtree(f'{V}/candidates')
    fr = NV.run_frames(V, [K], SR, SUB)
    pl = NV.plan(K, V, P, ident, fr)
    put(os.path.join(V, pl['blocks'][pl['link'][0][0]][2]), 'in the way')
    try:
        NV.link_blocks(P, V, pl)
        ok = False
    except NV.NightRefusal as e:
        ok = 'exists already' in str(e)
    check('link_blocks: REFUSES to carry an image over an existing one', ok)
    shutil.rmtree(f'{V}/candidates')
    real = os.link

    def nolink(a, b):
        raise OSError(errno.EXDEV, 'cross-device')
    os.link = nolink
    try:
        got = NV.link_blocks(P, V, pl)
    finally:
        os.link = real
    check('link_blocks: another file system - copies with the file\'s times', got == dict(linked=0, copied=len(pl['link']))
          and all(os.stat(os.path.join(V, pl['blocks'][p][2])).st_mtime_ns == os.stat(os.path.join(P, b)).st_mtime_ns for p, b in pl['link']))
    ok, detail, counts = NV.test(V, [K], fr)
    check('content test: rows without image are counted, not failed', ok and counts[K]['without_image'] == 8 - len(pl['link']), detail)
    put(f'{V}/candidates/V/A/rank00099_17h_{N1}_bin0_41MHz.png', 'stale rank')
    ok, detail, _c = NV.test(V, [K], fr)
    check('content test FAILS: an image no selection row names', not ok and '1 image(s) no selection row names' in detail, detail)
    os.remove(f'{V}/candidates/V/A/rank00099_17h_{N1}_bin0_41MHz.png')
    blind = {k: dict(v, n_searched=0) for k, v in fr.items()}
    ok, detail, _c = NV.test(V, [K], blind)
    check('content test FAILS: an image of a frame without a searched image this module finds', not ok and 'without a searched image' in detail, detail)
    dup = variant('dup', P, [HEAD, line('A', 'A1', N1, 0, 9)])
    put(f'{dup}/source_catalogs/tenmin_selection_I_sb.csv', '\n'.join([HEAD, line('A', 'A9', N1, 0, 9)]) + '\n')
    ok, detail, _c = NV.test(dup, [K, 'I_sb'])
    check('content test FAILS: two rows share an image name', not ok and 'shared by two rows' in detail, detail)
    # identities
    check('frozen_id: the sha1 of builder_frozen.json = the frozen_record_sha1 a view names; \'\' for neither',
          NV.frozen_id(f'{B}/source_catalogs_{K}') == NV.frozen_id(f'{P}/source_catalogs_{K}') == hashlib.sha1(b'frozen-1').hexdigest()
          and NV.frozen_id(f'{T}/parents') == '')
    put(f'{T}/xm/10pc.csv', 'a\n')
    sid = NV.samples_id(f'{T}/xm', ('10pc', 'absent'))
    check('samples_id: size and time of each sample file; a sample without file is omitted',
          list(sid) == ['10pc'] and sid['10pc'][0] == 2 and sid['10pc'][1] == os.stat(f'{T}/xm/10pc.csv').st_mtime_ns)
    for t in ('t1', 't2', 't3'):
        tree = f'{T}/trees/{t}'
        put(f'{tree}/exopipe_ops/a.py', f"ROOT = '{tree}/clv15'\n" + ('X = 2\n' if t == 't3' else 'X = 1\n'))
        put(f'{tree}/clv15/b.py', 'pass\n')
        put(f'{tree}/CLOSURE.txt', f'{tree} written {t}\n')
        with open(f'{tree}/MANIFEST.sha256', 'w') as f:
            for rel in ('CLOSURE.txt', 'clv15/b.py', 'exopipe_ops/a.py'):
                f.write(hashlib.sha256(open(f'{tree}/{rel}', 'rb').read()).hexdigest() + f'  {rel}\n')
    d1, d2, d3 = (NV.code_digest(f'{T}/trees/{t}') for t in ('t1', 't2', 't3'))
    check('code_digest: two frozen trees of one code agree (their own root normalized; the tree\'s bookkeeping left out)', d1 and d1 == d2)
    check('code_digest: another code differs; no tree gives \'\'', d3 and d3 != d1 and NV.code_digest(None) == '')


def test_stage_catalog():
    AN, VR, ST = f'{T}/an', f'{T}/an_night/views/17h', f'{T}/out/_night/17h/catalog'
    for n in ('rejected.csv', 'detection_ts.csv', 'selected_V_sb.csv', 'selected_I_sb.csv', 'master_catalog.csv', 'catalogs.log'):
        put(f'{AN}/{n}', f'all nights {n}')
    put(f'{AN}/pertie/pertie_detections.csv', 'all nights')
    put(f'{AN}/pertie/conform_inputs/SHA256SUMS', 'sums')
    put(f'{VR}/hour_view.json', json.dumps(dict(complete=True, all_nights=AN)))
    for n in ('master_catalog.csv', 'v_deep_images.csv', 'selected_V_sb.csv', 'catalogs_noveto_loc.log'):
        put(f'{VR}/{n}', f'view {n}')
    put(f'{VR}/pertie/pertie_detections.csv', 'view')
    CONF = ('pertie/conform_inputs/pertie_detections.csv', 'pertie/conform_inputs/pertie_frames.csv',
            'pertie/conform_inputs/SHA256SUMS', 'catalogs.log')            # config pages_sublst.NIGHT_CONFORM_FILES
    for n in CONF[:3]:
        put(f'{VR}/{n}', f'view read {os.path.basename(n)}')
    put(f'{VR}/catalogs.log', 'x CATALOGS B conform\nconform: window-tie table read: view-sha\n')
    put(f'{VR}/tie_members.csv', 'view tie')
    put(f'{AN}/pertie/conform_inputs/pertie_frames.csv', 'all nights frames')
    put(f'{VR}/source_catalogs_noveto_V_sb/x.csv', 'view')
    put(f'{VR}/localizations_noveto/localizations.csv', 'view')
    put(f'{VR}/_loc/side.csv', 'side')
    put(f'{ST}/stale.csv', 'from an earlier pass')
    files = ('rejected.csv', 'rejected_index.npz', 'detection_ts.csv')
    st = NV.stage_catalog(ST, VR, files, CONF)
    rd = lambda n: open(f'{ST}/{n}').read()                                    # noqa: E731
    check('staged catalog: the view root\'s tables, catalogs and window-tie table',
          rd('master_catalog.csv') == 'view master_catalog.csv' and rd('selected_V_sb.csv') == 'view selected_V_sb.csv'
          and rd('pertie/pertie_detections.csv') == 'view' and rd('source_catalogs_noveto_V_sb/x.csv') == 'view'
          and rd('localizations_noveto/localizations.csv') == 'view')
    check('staged catalog: from the all-nights directory, the records a night does not write and the keys it did not view',
          rd('rejected.csv') == 'all nights rejected.csv' and rd('selected_I_sb.csv') == 'all nights selected_I_sb.csv')
    check('staged catalog: the conformance read and its checksum log are the VIEW root\'s own, never the all-nights read '
          '(DECISION-20261114-PR50-VIEW-ROOT-RECORDS-ITS-OWN-CONFORMANCE-READ); the tie table is the view\'s '
          '(DECISION-20261114-PR50-VIEW-CARRIES-ITS-REFERENCE-TIE-TABLE)',
          rd('pertie/conform_inputs/SHA256SUMS') == 'view read SHA256SUMS'
          and rd('pertie/conform_inputs/pertie_frames.csv') == 'view read pertie_frames.csv'
          and 'view-sha' in rd('catalogs.log') and rd('tie_members.csv') == 'view tie'
          and os.path.realpath(f'{ST}/pertie/conform_inputs/SHA256SUMS') == os.path.realpath(f'{VR}/pertie/conform_inputs/SHA256SUMS'))
    check('staged catalog: the record, side directories and logs of the view stage are left out; an earlier staging is removed',
          not any(os.path.lexists(f'{ST}/{n}') for n in ('hour_view.json', '_loc', 'catalogs_noveto_loc.log', 'stale.csv')))
    check('staged catalog: what it returns', st['all_nights'] == AN and st['absent'] == ['rejected_index.npz']
          and not any(n.startswith('pertie/conform_inputs') for n in st['carried']) and 'catalogs.log' in st['view']
          and 'selected_I_sb.csv' in st['carried'] and 'pertie/pertie_detections.csv' in st['view'], str(st))

    def refuses(name, expect, prepare, restore):
        prepare()
        try:
            NV.stage_catalog(ST, VR, files, CONF)
            ok = False
        except NV.NightRefusal as e:
            ok = expect in str(e) and str(e).endswith('REFUSING')
        restore()
        check(f'staged catalog REFUSES: {name}', ok)
    rec = f'{VR}/hour_view.json'
    refuses('a view root that is not complete', 'not complete', lambda: put(rec, json.dumps(dict(complete=False, all_nights=AN))),
            lambda: put(rec, json.dumps(dict(complete=True, all_nights=AN))))
    refuses('an all-nights directory that is gone', 'is gone', lambda: put(rec, json.dumps(dict(complete=True, all_nights=AN + '_x'))),
            lambda: put(rec, json.dumps(dict(complete=True, all_nights=AN))))
    refuses('no window-tie table', 'window-tie table', lambda: os.rename(f'{VR}/pertie', f'{VR}/pertie_x'),
            lambda: os.rename(f'{VR}/pertie_x', f'{VR}/pertie'))
    refuses('not a view root', f'holds no {NV.VIEW_ROOT_RECORD}', lambda: os.rename(rec, rec + '_x'), lambda: os.rename(rec + '_x', rec))
    sums = f'{VR}/pertie/conform_inputs/SHA256SUMS'
    refuses('a view root without its own conformance read (the all-nights read is there)', 'no conformance read',
            lambda: os.rename(sums, sums + '_x'), lambda: os.rename(sums + '_x', sums))
    refuses('a view root whose log has no conform checksum line', 'checksum line',
            lambda: put(f'{VR}/catalogs.log', 'x CATALOGS B conform\n'),
            lambda: put(f'{VR}/catalogs.log', 'x CATALOGS B conform\nconform: window-tie table read: view-sha\n'))
    os.remove(f'{VR}/tie_members.csv')
    st = NV.stage_catalog(ST, VR, files, CONF)
    check('staged catalog: a view without a tie table is staged (an annotation, never refused) and never takes another one',
          not os.path.lexists(f'{ST}/tie_members.csv'))


STUB_PREP = '''import os, sys
LST = os.environ.get('SUBLST_LST', ''); ARM, OUT = sys.argv[1], sys.argv[2]
SL = '/nowhere'; CAT = f'{SL}/compare_{LST}/new_{ARM}/catalog'; OLD = f'{SL}/compare_{LST}/old'
print('STUB', __name__, os.path.basename(__file__), sys.argv[1:], CAT, OLD, sys.path[0] == os.path.dirname(os.path.abspath(__file__)))
'''

STUB_TIE = '''_RT = f'{SL}/compare_{LST}/new_{ARM}/tie/tie_members.csv'
print('TIE', _RT)
'''


def test_wrap():
    from exopipe.phase3.lib.pages import common
    S = common.CFG().pages_sublst
    wrap = os.path.join(REPO, 'exopipe', 'phase3', 'lib', 'pages', 'wraps', 'prep_catalog.py')
    script = f'{T}/code/exopipe_ops/sublst_pages_prep.py'
    put(script, STUB_PREP)
    pin = PC.normalized_sha1(script, STUB_PREP)
    cat = f'{T}/an'

    def go(script=script, pins=pin, catalog=cat, tie=f'{T}/an/tie_members.csv'):
        env = dict(os.environ, EXOPIPE3_PAGES_CATALOG=catalog, EXOPIPE3_PAGES_PREP_SHA1=pins, SUBLST_LST='17h',
                   EXOPIPE3_PAGES_TIE=tie)
        r = subprocess.run([sys.executable, wrap, script, 'B', f'{T}/outrun'], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True)
        return r.returncode, r.stdout
    rc, out = go()
    check('prep runner: the script runs as __main__ in place with the catalog directory replaced and every other line as written',
          rc == 0 and f"STUB __main__ sublst_pages_prep.py ['B', '{T}/outrun'] {cat} /nowhere/compare_17h/old True" in out
          and 'PAGES-PREP done' in out, out)
    rc, out = go(pins='0' * 40)
    check('prep runner REFUSES a script that is not pinned (exit 2)', rc == 2 and 'PAGES-PREP REFUSING' in out and 'STUB' not in out, out)
    rc, out = go(pins='')
    check('prep runner REFUSES without pins', rc == 2 and 'is empty' in out, out)
    rc, out = go(catalog=f'{T}/nowhere')
    check('prep runner REFUSES a catalog directory that does not exist', rc == 2 and 'is not a directory' in out, out)
    rc, out = go(tie='')
    check('prep runner REFUSES when the tie table of the catalog is not named', rc == 2 and 'EXOPIPE3_PAGES_TIE' in out, out)
    tied = f'{T}/code4/exopipe_ops/sublst_pages_prep.py'
    put(tied, STUB_PREP + STUB_TIE)
    rc, out = go(script=tied, pins=PC.normalized_sha1(tied, open(tied).read()))
    check('prep runner: the tie table path is the one named (EXOPIPE3_PAGES_TIE), never production\'s current one',
          rc == 0 and f'TIE {T}/an/tie_members.csv' in out and '/nowhere/' not in out.split('TIE ')[-1].split('\n')[0], out)
    put(tied, STUB_PREP + STUB_TIE + STUB_TIE)
    rc, out = go(script=tied, pins=PC.normalized_sha1(tied, open(tied).read()))
    check('prep runner REFUSES a script that holds the tie-table path twice', rc == 2 and 'tie-table path 2 times' in out, out)
    twice = f'{T}/code2/exopipe_ops/sublst_pages_prep.py'
    put(twice, STUB_PREP + STUB_PREP.split('\n')[2] + '\n')
    rc, out = go(script=twice, pins=PC.normalized_sha1(twice, open(twice).read()))
    check('prep runner REFUSES a script that holds the assignment twice', rc == 2 and '2 times' in out, out)
    tree = f'{T}/code3'
    put(f'{tree}/exopipe_ops/sublst_pages_prep.py', STUB_PREP.replace("'/nowhere'", "'/lustre/gh/main/clv15'").replace('/lustre/gh/main/', tree + '/'))
    live = hashlib.sha1(STUB_PREP.replace("'/nowhere'", "'/lustre/gh/main/clv15'").encode()).hexdigest()
    check('prep runner: a frozen tree\'s copy has the live script\'s path-normalized sha1',
          PC.normalized_sha1(f'{tree}/exopipe_ops/sublst_pages_prep.py', open(f'{tree}/exopipe_ops/sublst_pages_prep.py').read()) == live)
    try:
        _src, sha1, norm = PC.patched(f'{OPS}/sublst_pages_prep.py', S.PREP_SHA1)
        ok = (norm in S.PREP_SHA1 and _src.count(PC.NEW) == 1 and PC.OLD not in _src
              and _src.count(PC.NEW_TIE) == 1 and PC.OLD_TIE not in _src)
    except PC.Refused as e:
        ok, sha1 = False, str(e)
    check('production sublst_pages_prep.py is one of pages_sublst.PREP_SHA1; its catalog assignment and tie-table path are '
          'each replaced once', ok, str(sha1))
    trees = sorted(glob.glob('/lustre/gh/main/phase3/sublst/code_frozen/[0-2][0-9]h_*/exopipe_ops/sublst_pages_prep.py'),
                   key=lambda p: p.split('/')[-3].split('_', 1)[1])          # the hours' trees, by their freeze time
    bad = []
    for p in trees[-3:]:
        try:
            PC.patched(p, S.PREP_SHA1)
        except PC.Refused as e:
            bad.append(str(e)[:200])
    check(f'the newest frozen code trees\' copies are accepted ({len(trees[-3:])} read)', not bad, str(bad))
    check('config: the night keys', S.NIGHT_INDEX == 'hour_views.json' and '{lst}' in S.NIGHT_STAGING
          and set(S.REQUIRED_FILES) - {'pertie/pertie_detections.csv'} <= set(S.NIGHT_ALL_NIGHTS_FILES)
          and 'PREP_SHA1' in S.PROTECTED_KEYS and 'catalogs.log' not in S.NIGHT_ALL_NIGHTS_FILES
          and 'catalogs.log' in S.NIGHT_CONFORM_FILES and S.NIGHT_TIE == 'tie_members.csv'
          and all(f.startswith('pertie/conform_inputs/') for f in S.NIGHT_CONFORM_FILES if f != 'catalogs.log'))


STUB_GUARD = '''import os, sys
class FrozenCodeGuard(object):
    def find_spec(self, name, path=None, target=None):
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, 'find_spec'):
                continue
            spec = finder.find_spec(name, path, target)
            if spec is not None:
                return spec
        return None
sys.meta_path.insert(0, FrozenCodeGuard())
try:                                   # the chain of exopipe_ops/sublst_frozen_guard/sitecustomize.py, as written there
    import importlib.machinery as _m
    _here = os.path.dirname(os.path.abspath(__file__))
    _spec = _m.PathFinder.find_spec('sitecustomize', [p for p in sys.path if p and os.path.abspath(p) != _here])
    if _spec is not None and _spec.origin and os.path.abspath(_spec.origin) != os.path.abspath(__file__):
        _code = compile(open(_spec.origin).read(), _spec.origin, 'exec')
        exec(_code, {'__name__': 'sitecustomize', '__file__': _spec.origin})
except SystemExit:
    raise
except Exception:
    pass
'''


def test_review50():
    """review of PR 50: a view root naming no all-nights directory refuses; a Master older than its inputs is vouched for by
    no record; a night run's frozen build comes from the pass index; the wrap's path normalization is builder_grouping's;
    the place stage does not publish the reduced selections"""
    VR = f'{T}/r50/views/17h'
    put(f'{VR}/hour_view.json', json.dumps(dict(complete=True)))
    put(f'{VR}/pertie/pertie_detections.csv', 'view')
    try:
        NV.stage_catalog(f'{T}/r50/st', VR, ())
        ok, msg = False, ''
    except NV.NightRefusal as e:
        ok, msg = 'names no all-nights' in str(e) and str(e).endswith('REFUSING'), str(e)
    check('stage_catalog: a view record without all_nights REFUSES (a NightRefusal, not a KeyError)', ok, msg)
    R, SB = f'{T}/r50/run', f'{T}/r50/sub'
    a = f'{SB}/17h/17h_2026-10-01/sb3/I/sublst_17h_2026-10-01_sb3_I_w2.fits'
    put(a, 'stack')
    put(a[:-5] + '_resid_loo.fits', 'resid')
    m = f'{R}/loo_masters/17h_2026-10-01/sb3/I/loo_master_w2.fits'
    put(m, 'master')
    t0 = os.stat(a).st_mtime_ns
    os.utime(m, ns=(t0 + 10 ** 9, t0 + 10 ** 9))
    fresh, want = NV.masters_made_from(R, SB, '17h', '2026-10-01', 2, 'sb3'), NV.masters_in(SB, '17h', '2026-10-01', 2, 'sb3')
    os.utime(a, ns=(t0 + 2 * 10 ** 9, t0 + 2 * 10 ** 9))                   # the stack rebuilt after its Master was written
    stale = NV.masters_made_from(R, SB, '17h', '2026-10-01', 2, 'sb3')
    check('masters_made_from: the inputs\' digest for a Master newer than its inputs; None once an input is newer',
          fresh == want and fresh is not None and stale is None, (fresh, want, stale))
    from exopipe.phase3.lib.pages import tier_sublst as TS
    from exopipe.phase3.lib.catalog.wraps import builder_grouping as BG
    tag = TS.common.CFG().pages_sublst.SERVED_CATALOG_TAG
    P = f'{T}/r50/P'
    put(f'{P}/source_catalogs_V_sb/catalog_view.json', json.dumps(dict(frozen_record_sha1='from-the-carried-root')))
    c = type('C', (), {})()
    c.night_dir, c.views = f'{T}/r50', {'17h': dict(kind='view', root=VR, keys={f'{tag}_V_sb': dict(record_sha1='from-the-index')})}
    c2 = type('C', (), {})()
    c2.night_dir, c2.views = None, {}
    check('a night run\'s frozen build: the pass index entry\'s record_sha1, never the carried root\'s catalog_view.json '
          '(DECISION-20261114-PR47-CARRIED-ROOT-READERS-TAKE-TABLE-FROM-INDEX); an all-nights run reads its copy',
          TS.frozen_of(c, '17h', P, 'V_sb') == 'from-the-index' and TS.frozen_of(c, '17h', P, 'I_sb') == ''
          and TS.frozen_of(c2, '17h', P, 'V_sb') == 'from-the-carried-root')
    fz = f'{T}/r50/frozen_tree/exopipe_ops/sublst_pages_prep.py'
    src = "X = '/lustre/gh/main/exopipe_ops/a'; Y = '" + f'{T}/r50/frozen_tree' + "/exopipe_ops/b'\n"
    check('the wrap\'s path normalization is builder_grouping\'s (a frozen tree\'s root and the live root)',
          PC.normalized_sha1(fz, src) == BG.normalized_sha1(fz, src) and PC.LIVE_ROOT == BG.LIVE_ROOT)
    tsrc = open(TS.__file__).read()
    check('place: the reduced selections (source_catalogs/_night/) are excluded from the rsync to the review host',
          "['--exclude', night.REDUCED_DIR + '/', P + '/', tgt + '/']" in tsrc and NV.REDUCED_DIR == os.path.join('source_catalogs', '_night'))


def test_site_hooks():
    """the refactor's start-up hook first on PYTHONPATH and a sitecustomize that chains to the next one (the frozen-code
    guard) second: each is installed once, and imports work"""
    put(f'{T}/guard/sitecustomize.py', STUB_GUARD)
    boot = os.path.join(REPO, 'exopipe', 'phase3', 'core', 'parity_sitecustomize')
    env = {k: v for k, v in os.environ.items() if not k.startswith('EXOPIPE3_')}
    env['PYTHONPATH'] = boot + os.pathsep + f'{T}/guard'
    r = subprocess.run([sys.executable, '-c', 'import sys, csv, colorsys; print([type(f).__name__ for f in sys.meta_path])'],
                       env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, cwd=T)
    names = r.stdout.strip().split('\n')[-1]
    check('start-up hooks: the refactor\'s hook and a chaining guard are each installed once; imports work',
          r.returncode == 0 and names.count("'FrozenCodeGuard'") == 1 and names.count("'_PatchImportGuard'") == 1
          and 'RecursionError' not in r.stdout, r.stdout[-400:])
    env['PYTHONPATH'] = f'{T}/guard' + os.pathsep + boot          # review of PR 50: the guard FIRST (sublst_code_root.sh
    r = subprocess.run([sys.executable, '-c', 'import sys, csv, colorsys; print([type(f).__name__ for f in sys.meta_path])'],
                       env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, cwd=T)   # prepends it)
    names = r.stdout.strip().split('\n')[-1]
    check('start-up hooks, the guard first: the refactor\'s hook never chains back to it, each hook installed once, imports work',
          r.returncode == 0 and names.count("'FrozenCodeGuard'") == 1 and names.count("'_PatchImportGuard'") == 1
          and 'RecursionError' not in r.stdout, r.stdout[-400:])
    src = open(f'{OPS}/sublst_frozen_guard/sitecustomize.py').read()
    check('production\'s frozen-code guard still chains as the stub does',
          "_spec = _m.PathFinder.find_spec('sitecustomize', [p for p in sys.path if p and os.path.abspath(p) != _here])" in src
          and "exec(_code, {'__name__': 'sitecustomize', '__file__': _spec.origin})" in src)


def main():
    shutil.rmtree(T, ignore_errors=True)
    os.makedirs(T)
    try:
        world()
        test_mirror()
        ident, B, P = test_flow()
        test_reasons(ident, B, P)
        test_files(ident, B, P)
        test_stage_catalog()
        test_wrap()
        test_review50()
        test_site_hooks()
    finally:
        if not os.environ.get('EXOPIPE3_TEST_KEEP'):
            shutil.rmtree(T, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
