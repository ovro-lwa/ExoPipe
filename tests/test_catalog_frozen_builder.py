#!/usr/bin/env python3
"""The frozen statistics of the per-hour 10-minute catalog build (lib/catalog/wraps/builder_grouping.py, FROZEN STATISTICS;
Gregg 2026-11-13: "the builder runs only on the sources the new night touches, with its statistics frozen from the last
all-nights build, and writes a delta rather than a whole new catalog"), on a stub builder that carries the patch points
(tests/frozen_stub_builder.py), plus the two stage helpers the delta build uses (grouping_stage.builder_env,
grouping_stage.assemble_tenmin_input with night directories).

  patches     the eleven frozen patch texts occur once in the live builder and the patched text compiles; without one the
              runner refuses; the config pins the live builder for the frozen statistics (a protected key)
  modes       on one input: pass-through writes byte-identical files; the record build changes no file and writes
              builder_frozen.json; a delta build with every identity touched equals the all-nights build in every table row
              except the rank and the review directory, runs no footprint test and removes the two population files
  night       a night appended: only the touched identities are built (extended, new); the delta record names what each
              replaces; a source seen again holds its earlier detection that the all-nights build's test for sources seen
              once had removed (the present behavior; the question is open with Gregg, 2026-11-13); the frozen forms give
              the values computed here by hand, which differ from an all-nights build of the same rows where they should
  order       two nights in both orders: a source's row depends only on the all-nights record and its own detections (the
              other night's detections, in the same band or another, never enter); an identity whose multiples-level group
              changed without a detection of its own is built again ("regrouped")
  refusals    no record; a record of another version; an input that lacks a row of the all-nights build; a builder that is
              not the pinned one; the grouping in pass-through; a weighted null density; another image table; a Master image
              that changed; an unknown mode or touch value; builder_env's two refusals
  night_dirs  assemble_tenmin_input: a night directory's rows follow its LST's run; the same row twice is kept once; another
              content, an LST outside --lsts and a directory twice refuse
Run: NUMPY_MADVISE_HUGEPAGE=0 EXOPIPE3_SITE=/lustre/gh/main/exopipe_ops/exopipe3_site.py taskset -c $(bash
     /lustre/gh/main/exopipe_ops/node_free_cores.sh) /opt/devel/pipeline/envs/py38_orca_nkosogor/bin/python3
     tests/test_catalog_frozen_builder.py [names]
Writes only under /tmp/exopipe3_tests_frozen_builder (removed at the end); reads the live builder read-only.
"""
import collections
import csv
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

from exopipe.phase3.lib.catalog import grouping_stage as GS  # noqa: E402
from frozen_stub_builder import FROZEN_BUILDER  # noqa: E402

T = '/tmp/exopipe3_tests_frozen_builder'
LIVE_BUILDER = '/lustre/gh/main/exopipe_ops/build_source_catalogs_v3.py'
PY = sys.executable
FAILS, N = [], [0]
STUBCAT = r'''
import csv, os
def rcsv(p):
    with open(p, newline='') as f:
        rd = csv.DictReader(f)
        return list(rd.fieldnames or []), list(rd)
def wcsv(p, cols, rows):
    os.makedirs(os.path.dirname(p) or '.', exist_ok=True)
    with open(p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(cols), extrasaction='ignore'); w.writeheader(); w.writerows(rows)
'''


def check(name, ok, detail=''):
    N[0] += 1
    print(('PASS  ' if ok else 'FAIL  ') + name + ('' if ok else f'  ({str(detail)[:900]})'), flush=True)
    if not ok:
        FAILS.append(name)
    return ok


def rcsv(p):
    with open(p, newline='') as f:
        rd = csv.DictReader(f)
        return list(rd.fieldnames or []), list(rd)


def wcsv(p, cols, rows):
    os.makedirs(os.path.dirname(p) or '.', exist_ok=True)
    with open(p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(cols), extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def _bg():
    import importlib.util
    spec = importlib.util.spec_from_file_location('bg', GS.builder_runner())
    bg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bg)
    return bg


# ---- the synthetic catalog: one LST (05h), two bands, three nights in the all-nights build, two nights appended ----------
D1, D2, D3, NT1, NT2 = '2030-01-01', '2030-01-02', '2030-01-03', '2030-01-04', '2030-01-05'
# (cand_id, band, date, ra, dec, S/N, sigma-stack ratio, matched sample); the identity is the letter in the cand_id
BASE = [('I_05h_A1', '46MHz', D1, 80.0, 30.0, 6.2, '0.5', ''),      # A: two windows, below 7 sigma: a multiple
        ('I_05h_A2', '46MHz', D2, 80.001, 30.0, 6.4, '0.5', ''),
        ('I_05h_B1', '46MHz', D3, 81.0, 31.0, 9.5, '0.5', 'S1'),    # B: seen once, 9.5 sigma, matched
        ('I_05h_C1', '46MHz', D1, 82.0, 32.0, 7.5, '2.0', ''),      # C: seen once, fails the sigma-stack test
        ('I_05h_D1', '46MHz', D2, 83.0, 33.0, 9.0, '0.5', 'S1'),    # D: seen once, 9 sigma, matched
        ('I_05h_E1', '46MHz', D2, 84.0, 34.0, 8.0, '0.5', ''),      # E: one window, two bands
        ('I_05h_E2', '50MHz', D2, 84.0005, 34.0, 6.0, '', ''),
        ('I_05h_K1', '46MHz', D1, 89.0, 39.0, 6.8, '0.5', ''),      # K: seen once below 7 sigma: fails the sigma-stack test
        ('I_05h_G1', '50MHz', D1, 87.0, 37.0, 10.0, '0.5', 'S1'),   # G: seen once, 10 sigma, matched
        ('I_05h_H1', '50MHz', D3, 88.0, 38.0, 6.3, '', '')]         # H: seen once, 6.3 sigma: below the floor
NIGHT1 = [('I_05h_A3', '46MHz', NT1, 80.0005, 30.0, 6.6, '0.5', ''),    # A seen a third time
          ('I_05h_N1', '46MHz', NT1, 85.0, 35.0, 9.2, '0.5', 'S1'),     # N: new, 9.2 sigma, matched
          ('I_05h_K2', '46MHz', NT1, 89.0004, 39.0, 6.9, '0.5', ''),    # K seen again
          ('I_05h_H2', '50MHz', NT1, 88.0004, 38.0, 6.4, '', '')]       # H seen again
NIGHT2 = [('I_05h_D2', '46MHz', NT2, 83.0003, 33.0, 7.2, '0.5', ''),    # D seen again
          ('I_05h_P1', '50MHz', NT2, 86.0, 36.0, 8.5, '0.5', 'S1'),     # P: new, matched
          ('I_05h_S1', '50MHz', NT2, 85.01, 35.0, 9.6, '0.5', ''),      # S: new, another band, 36 arcsec from N
          ('I_05h_R1', '50MHz', NT2, 87.02, 37.0, 7.9, '0.5', '')]      # R: new identity in G's multiples-level group
MULT = {'R': 'G'}
XSEP_ARCMIN = 600.0


def L(cand):
    return cand.split('_')[2][0]


def name(k):
    return f'J05{ord(k) - 65:02d}+0000{k.lower()}'


def write_world(d, tag, spec, image_dates=(D1, D2, D3)):
    """selected_<tag>.csv, table_<tag>.csv (the grouping of exactly these rows) and images_<tag>.csv; returns their names"""
    rows = [dict(cand_id=c, band=b, lst='05h', date=dt, stokes='I', ra=repr(ra), dec=repr(de), snr=repr(s), sstack=ss, xsmp=x,
                 xsep=(repr(XSEP_ARCMIN) if x else ''), sys='') for c, b, dt, ra, de, s, ss, x in spec]
    wcsv(f'{d}/selected_{tag}.csv', list(rows[0]), rows)
    wins = collections.defaultdict(set)
    for r in rows:
        k = L(r['cand_id'])
        wins[MULT.get(k, k)].add((r['lst'], r['date']))
    tab = []
    for r in rows:
        k = L(r['cand_id'])
        m = MULT.get(k, k)
        tab.append(dict(cand_id=r['cand_id'], stokes='I', band=r['band'], lst=r['lst'], date=r['date'],
                        key=f"{r['lst']}|{r['date']}", night=r['date'], ra=r['ra'], dec=r['dec'], snr=r['snr'],
                        identity_id=f'GTI-{ord(k):07d}', identity_name=name(k), multiples_id=f'MTI-{ord(m):07d}',
                        multiples_name=name(m), multiplicity=str(len(wins[m])), multiple=str(int(len(wins[m]) >= 2))))
    wcsv(f'{d}/table_{tag}.csv', GS.DET_COLS, tab)
    wcsv(f'{d}/images_{tag}.csv', ['lst', 'scope', 'stokes', 'date'],
         [dict(lst='05h', scope=b, stokes='I', date=dt) for b in ('46MHz', '50MHz') for dt in image_dates])
    return f'selected_{tag}.csv', f'{d}/table_{tag}.csv', f'images_{tag}.csv'


def setup(d):
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    open(f'{d}/_stubcat.py', 'w').write(STUBCAT)
    b = f'{d}/build_source_catalogs_v3.py'
    open(b, 'w').write(FROZEN_BUILDER.lstrip())
    return b, _bg().normalized_sha1(b, open(b).read())


def run(d, b, sha, sel, out, table, images, frozen='', record=None, touch='', mode='design-b', frozen_accepted=None, extra=None):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('P3_', 'EXOPIPE3_FROZEN', 'EXOPIPE3_GROUPING', 'STUB_'))}
    env.update(EXOPIPE3_GROUPING_MODE=mode, EXOPIPE3_GROUPING_TIER='10min', EXOPIPE3_GROUPING_RUNGS='multiples-level',
               EXOPIPE3_GROUPING_BUILDER_SHA1=sha, EXOPIPE3_GROUPING_TABLE=table, EXOPIPE3_FROZEN_MODE=frozen,
               EXOPIPE3_FROZEN_RECORD=record or '', EXOPIPE3_FROZEN_TOUCH=touch,
               EXOPIPE3_FROZEN_BUILDER_SHA1=sha if frozen_accepted is None else frozen_accepted)
    env.update(extra or {})
    p = subprocess.run([PY, GS.builder_runner(), b, sel, out, images], cwd=d, env=env, capture_output=True, text=True,
                       timeout=300)
    return p.returncode, p.stdout + p.stderr


def digest(d, skip=()):
    return {f: hashlib.sha1(open(os.path.join(d, f), 'rb').read()).hexdigest() for f in sorted(os.listdir(d)) if f not in skip}


def sources(d, kind='master'):
    return {(r['band'], r['source_id']): r for p in glob.glob(f'{d}/*_sources_{kind}.csv') for r in rcsv(p)[1]}


def members(d):
    m = collections.defaultdict(set)
    for p in glob.glob(f'{d}/*_source_members.csv'):
        for r in rcsv(p)[1]:
            m[(r['band'], r['source_id'])].add(r['cand_id'])
    return dict(m)


def tables(d, ignore=()):
    """every CSV of a build directory as {file: sorted rows without the ignored columns}"""
    out = {}
    for p in sorted(glob.glob(f'{d}/*.csv')):
        cols, rows = rcsv(p)
        out[os.path.basename(p)] = sorted(tuple((c, r[c]) for c in cols if c not in ignore) for r in rows)
    return out


def base_builds(d):
    """the all-nights build of BASE with its record; returns what a later run needs"""
    b, sha = setup(d)
    sel, tab, img = write_world(d, 'base', BASE)
    rc, txt = run(d, b, sha, sel, 'out_r', tab, img, frozen='record')
    return b, sha, sel, tab, img, rc, txt


# =====================================================================================================================
def test_patches():
    bg = _bg()
    src = open(LIVE_BUILDER).read()
    try:
        out, names = bg.patch(src, frozen=True)
        compile(out, LIVE_BUILDER, 'exec')
        got = names
    except Exception as e:                      # noqa: BLE001 - the failure reported
        got = [f'{type(e).__name__}: {e}']
    check('patches: the eleven frozen patch texts and the twelve grouping ones apply exactly once to the live builder and '
          'the patched text compiles', len(got) == 23 and [n for n in got if n.startswith('F')] == [p[0] for p in bg.FROZEN_PATCHES]
          and len(bg.FROZEN_PATCHES) == 11 and len(bg.patch(src)[1]) == 12, got[-3:])
    for nm, old, _new in bg.FROZEN_PATCHES:
        if old.count('\n') == 0:
            continue
        try:
            bg.patch(src.replace(old, old.replace('\n', ' \n', 1)), frozen=True)
            r = 'patched'
        except bg.Refused as e:
            r = str(e)
        if not ('REFUSING' in r and nm in r and 're-port' in r):
            check(f'patches: the builder without the text of {nm} REFUSES (re-port)', False, r[:300])
            break
    else:
        check('patches: the builder without any one frozen patch text REFUSES, naming the patch (re-port)', True)
    try:
        out, names = bg.patch(FROZEN_BUILDER, frozen=True)
        compile(out, 'stub', 'exec')
        ok = len(names) == 23
    except Exception as e:                      # noqa: BLE001
        ok, names = False, str(e)
    check('patches: the stub builder of these tests carries all 23 patch points and the census', ok, names)
    from exopipe.phase3.config import CFG
    live = bg.normalized_sha1(LIVE_BUILDER, src)
    g = CFG.catalog_grouping
    check('config: catalog_grouping.FROZEN_BUILDER_SHA1 pins the live builder, is among the grouping\'s accepted builders and '
          'is a protected key', live in g.FROZEN_BUILDER_SHA1 and set(g.FROZEN_BUILDER_SHA1) <= set(g.BUILDER_SHA1)
          and 'FROZEN_BUILDER_SHA1' in g.PROTECTED_KEYS, (live, g.FROZEN_BUILDER_SHA1))


def test_modes():
    d = f'{T}/modes'
    b, sha, sel, tab, img, rc_r, txt_r = base_builds(d)
    rc_b, txt_b = run(d, b, sha, sel, 'out_b', tab, img)
    rc_p, txt_p = run(d, b, sha, sel, 'out_p', tab, img, frozen='passthrough')
    check('pass-through (eleven patches, every hook returning the builder\'s own value): byte-identical files to the build '
          'without them', rc_b == 0 and rc_p == 0 and digest(f'{d}/out_b') == digest(f'{d}/out_p') and 'patches 23' in txt_p
          and 'patches 12' in txt_b and 'footprint tests: 88' in txt_p, (rc_b, rc_p, txt_p[-400:]))   # 2 bands x (36 sky + 8 list)
    rec = json.load(open(f'{d}/out_r/builder_frozen.json')) if os.path.exists(f'{d}/out_r/builder_frozen.json') else {}
    check('record: the all-nights build writes the same files plus builder_frozen.json, with no frozen patch applied',
          rc_r == 0 and digest(f'{d}/out_r', skip=('builder_frozen.json',)) == digest(f'{d}/out_b') and 'patches 12' in txt_r
          and 'FROZEN-BUILDER record' in txt_r, (rc_r, txt_r[-400:]))
    want_src = {('46MHz', name('A')), ('46MHz', name('B')), ('46MHz', name('D')), ('46MHz', name('E')), ('50MHz', name('E')),
                ('50MHz', name('G')), ('50MHz', name('H'))}
    area = 41253.0 * 25 / 36
    check('record content: version 2; every input detection with its identity and a digest of its multiples-level group; the '
          'seven sources before the gates (C and K, removed as sources seen once, are not among them) with best detection '
          'and matched sample; the band systematics, footprint areas, densities and what the footprint stands on',
          rec.get('version') == 2 and set(rec.get('rows', {})) == {c[0] for c in BASE}
          and all(len(v) == 2 and v[0].startswith('GTI-') and len(v[1]) == 16 for v in rec['rows'].values())
          and {(s[0], s[4]) for s in rec['sources']} == want_src and len(rec['sources']) == 7
          and sorted(s[2] for s in rec['sources'] if s[3] == 'S1') == ['I_05h_B1', 'I_05h_D1', 'I_05h_G1']
          and rec['sys_as'] == {'46MHz': 52.0, '50MHz': 52.0} and rec['vetoed'] == []
          and all(abs(rec['area'][b_] - area) < 1e-6 and abs(rec['rho'][f'S1|{b_}'] - 8 / area) < 1e-12 for b_ in ('46MHz', '50MHz'))
          and rec['footprint_inputs']['lsts'] == ['05h'] and rec['footprint_inputs']['xcat']['n'] == 8
          and rec['inputs']['selected']['rows'] == 10 and len(rec['inputs']['images']['sha1']) == 40, rec)
    rc_s, txt_s = run(d, b, sha, sel, 'out_s', tab, img, frozen='delta', record=f'{d}/out_r/builder_frozen.json', touch='all')
    ign = ('rank', 'review_dir')
    ta, tb = tables(f'{d}/out_s', ign) if rc_s == 0 else {}, tables(f'{d}/out_r', ign)
    check('delta with every identity touched and no appended night: every row of every table equals the all-nights build\'s '
          'except the rank and the review directory', rc_s == 0 and ta == tb and len(tb) >= 8,
          (rc_s, txt_s[-500:], [k for k in tb if ta.get(k) != tb[k]]))
    sv = sources(f'{d}/out_s', 'survivors') if rc_s == 0 else {}
    dl = json.load(open(f'{d}/out_s/builder_delta.json')) if os.path.exists(f'{d}/out_s/builder_delta.json') else {}
    check('delta: no rank and no review directory; no footprint test ran (areas and densities are the record\'s); the two '
          'population files are removed; builder_delta.json names the eight identities and no new detection',
          bool(sv) and all(r['rank'] == '' and r['review_dir'] == '' for r in sv.values()) and 'footprint tests: 0' in txt_s
          and not os.path.exists(f'{d}/out_s/E_params.json') and not os.path.exists(f'{d}/out_s/README_source_catalogs.txt')
          and os.path.exists(f'{d}/out_r/E_params.json') and dl.get('touch_all') is True and dl.get('new_detections') == 0
          and len(dl.get('touched', [])) == 8 and {t['kind'] for t in dl['touched']} == {'regrouped'}
          and 'FROZEN-BUILDER delta' in txt_s, (txt_s[-400:], dl))


def _night_worlds(d):
    out = {}
    for tag, spec in (('n1', BASE + NIGHT1), ('n2', BASE + NIGHT2), ('n12', BASE + NIGHT1 + NIGHT2), ('n21', BASE + NIGHT2 + NIGHT1)):
        out[tag] = write_world(d, tag, spec)
    return out


def test_night():
    d = f'{T}/night'
    b, sha, sel, tab, img, rc_r, txt_r = base_builds(d)
    rec = f'{d}/out_r/builder_frozen.json'
    w = _night_worlds(d)
    s1, t1, _i = w['n1']
    rc, txt = run(d, b, sha, s1, 'out_d1', t1, img, frozen='delta', record=rec)
    src, mem, sv = sources(f'{d}/out_d1'), members(f'{d}/out_d1'), sources(f'{d}/out_d1', 'survivors')
    A, N_, K, H = ('46MHz', name('A')), ('46MHz', name('N')), ('46MHz', name('K')), ('50MHz', name('H'))
    check('night: the delta builds the four touched identities and no other source (4 new detections of 14)',
          rc_r == 0 and rc == 0 and set(src) == {A, N_, K, H} and set(mem) == {A, N_, K, H}
          and '4 detection(s) the all-nights build did not have, 4 touched identities' in txt and 'footprint tests: 0' in txt,
          (rc, txt[-500:], sorted(src)))
    dl = {t['identity_name']: t for t in json.load(open(f'{d}/out_d1/builder_delta.json'))['touched']} if rc == 0 else {}
    check('night: builder_delta.json - A, K and H extended, N new; A and H replace their all-nights sources, K and N replace '
          'none', {k: (t['kind'], t['new_detections'], t['detections'], t['replaces']) for k, t in dl.items()} ==
          {name('A'): ('extended', 1, 3, [['46MHz', name('A')]]), name('N'): ('new', 1, 1, []),
           name('K'): ('extended', 1, 2, []), name('H'): ('extended', 1, 2, [['50MHz', name('H')]])}, dl)
    check('night: a source seen again holds its earlier detections - A three, H two, and K the detection the all-nights '
          'build had removed as seen once (present behavior, open with Gregg); all four survive, without rank',
          mem.get(A) == {'I_05h_A1', 'I_05h_A2', 'I_05h_A3'} and mem.get(H) == {'I_05h_H1', 'I_05h_H2'}
          and mem.get(K) == {'I_05h_K1', 'I_05h_K2'} and set(sv) == {A, N_, K, H}
          and all(r['rank'] == '' for r in sv.values()), (mem, sorted(sv)))
    rho_pi_r2 = 8 / (41253.0 * 25 / 36) * np.pi * (XSEP_ARCMIN / 60.0) ** 2
    a, n, h = src.get(A, {}), src.get(N_, {}), src.get(H, {})
    check('night, frozen forms by hand: A - 3 searched epochs of the all-nights build + its own new window = 4, and the '
          'repeat term from the 7 all-nights detections of the band + its own new one (8 x 0.01 x 4 = 0.32); H 0.12',
          (a.get('n_nights_searched'), a.get('n_cover'), a.get('e_rep'), h.get('e_rep'), h.get('n_cover')) ==
          ('4', '4', '0.32', '0.12', '4'), (a, h))
    check('night, frozen forms by hand: N (9.2 sigma, matched) - 2 all-nights sources at or above it + itself = 3; matched '
          'ones 3; chance load from 2 sources in its band and 3 in all bands',
          (n.get('xm_N'), n.get('xm_nobs')) == ('3', '3') and abs(float(n.get('xm_E_at_snr') or 0) - 2 * rho_pi_r2) < 2e-6
          and abs(float(n.get('xm_E_all') or 0) - 3 * rho_pi_r2) < 2e-6 and n.get('other_bands') == '', n)
    # an all-nights build of the same rows (its image table holds the night): same members and survivors; its repeat term
    # counts the night's other detections, the delta's does not
    sf, tf, imf = write_world(d, 'full1', BASE + NIGHT1, image_dates=(D1, D2, D3, NT1))
    rcf, txtf = run(d, b, sha, sf, 'out_f1', tf, imf)
    fsrc, fmem, fsv = sources(f'{d}/out_f1'), members(f'{d}/out_f1'), sources(f'{d}/out_f1', 'survivors')
    check('night: against an all-nights build of the same rows - the four sources have the same members and survive in both; '
          'A\'s repeat term there counts the night\'s other detections (0.4), the delta\'s only A\'s own (0.32)',
          rcf == 0 and all(fmem.get(k) == mem.get(k) for k in (A, N_, K, H)) and {A, N_, K, H} <= set(fsv)
          and fsrc[A]['e_rep'] == '0.4' and fsrc[A]['n_cover'] == '4', (rcf, txtf[-300:], fsrc.get(A)))


def test_order():
    d = f'{T}/order'
    b, sha, sel, tab, img, rc_r, txt_r = base_builds(d)
    rec = f'{d}/out_r/builder_frozen.json'
    w = _night_worlds(d)
    rcs, src, mem = {}, {}, {}
    for tag in ('n1', 'n2', 'n12', 'n21'):
        s_, t_, _i = w[tag]
        rcs[tag], txt = run(d, b, sha, s_, f'out_{tag}', t_, img, frozen='delta', record=rec)
        src[tag], mem[tag] = sources(f'{d}/out_{tag}'), members(f'{d}/out_{tag}')
    first = [('46MHz', name('A')), ('46MHz', name('N')), ('46MHz', name('K')), ('50MHz', name('H'))]
    second = [('46MHz', name('D')), ('50MHz', name('P')), ('50MHz', name('S')), ('50MHz', name('R')), ('50MHz', name('G'))]
    check('order: the four delta builds ran; one night alone builds its own sources, both nights build the nine',
          rc_r == 0 and set(rcs.values()) == {0} and set(src['n1']) == set(first) and set(src['n2']) == set(second)
          and set(src['n12']) == set(first + second) == set(src['n21']), (rcs, sorted(src['n12'])))
    bad = [(k, t) for k in first for t in ('n12', 'n21') if src[t].get(k) != src['n1'].get(k) or mem[t].get(k) != mem['n1'].get(k)]
    bad += [(k, t) for k in second for t in ('n12', 'n21') if src[t].get(k) != src['n2'].get(k) or mem[t].get(k) != mem['n2'].get(k)]
    check('order: every column of a source\'s row and its members are the same whether the other night arrived before, after '
          'or not at all', not bad, bad)
    n12 = src['n12'].get(('46MHz', name('N')), {})
    s12 = src['n12'].get(('50MHz', name('S')), {})
    g2 = src['n2'].get(('50MHz', name('G')), {})
    dl = {t['identity_name']: t['kind'] for t in json.load(open(f'{d}/out_n2/builder_delta.json'))['touched']}
    check('order: the second night\'s detection 36 arcsec from N in the other band and above N\'s S/N enters neither N\'s '
          'other-band match nor its count of sources above (3); nor does N enter S\'s',
          n12.get('other_bands') == '' and n12.get('xm_N') == '3' and s12.get('other_bands') == '', (n12, s12))
    check('order: G, whose multiples-level group gained the new identity R without a detection of its own, is built again '
          '(regrouped) with the group\'s two identities', dl.get(name('G')) == 'regrouped' and dl.get(name('R')) == 'new'
          and g2.get('multiples_n_identities') == '2' and g2.get('multiple') == '1', (dl, g2))
    # the all-nights build of both nights: there the two do see each other
    sf, tf, imf = write_world(d, 'full12', BASE + NIGHT1 + NIGHT2, image_dates=(D1, D2, D3, NT1, NT2))
    rcf, txtf = run(d, b, sha, sf, 'out_f12', tf, imf)
    f = sources(f'{d}/out_f12').get(('46MHz', name('N')), {})
    check('order: an all-nights build of the same rows does match N to that detection (other band 9.6) and counts 4 sources '
          'above - the difference the frozen forms are for', rcf == 0 and f.get('other_bands') == '50MHz:9.6'
          and f.get('xm_N') == '4', (rcf, txtf[-300:], f))


def test_refusals():
    d = f'{T}/refusals'
    b, sha, sel, tab, img, rc_r, txt_r = base_builds(d)
    rec = f'{d}/out_r/builder_frozen.json'
    w = _night_worlds(d)
    s1, t1, _i = w['n1']

    def refuses(what, text, **kw):
        a = dict(sel=s1, out='out_x', table=t1, images=img, frozen='delta', record=rec)
        a.update(kw)
        shutil.rmtree(f'{d}/out_x', ignore_errors=True)
        rc, txt = run(d, b, sha, a.pop('sel'), a.pop('out'), a.pop('table'), a.pop('images'), **a)
        check(f'refusal: {what} REFUSES (exit 2)', rc == 2 and 'REFUSING' in txt and text in txt, (rc, txt[-400:]))
    refuses('a delta build without a record', 'no record of the all-nights build', record=None)
    json.dump(dict(json.load(open(rec)), version=1), open(f'{d}/rec_v1.json', 'w'))
    refuses('a record of another version', 'record version 1', record=f'{d}/rec_v1.json')
    sx, tx, _ix = write_world(d, 'lack', [r for r in BASE + NIGHT1 if r[0] != 'I_05h_B1'])
    refuses('an input without a detection of the all-nights build', 'are not in this build\'s input', sel=sx, table=tx)
    refuses('a builder that is not the pinned one for the frozen statistics', 'population-wide quantities', frozen_accepted='0' * 40)
    refuses('the record mode with the grouping in pass-through', 'needs the design-B grouping', frozen='record', record=None,
            mode='passthrough')
    refuses('a delta build with a weighted null density (P3_REP_DENSITY_CSV)', 'P3_REP_DENSITY_CSV',
            extra=dict(P3_REP_DENSITY_CSV=f'{d}/selected_base.csv'))
    wcsv(f'{d}/images_other.csv', ['lst', 'scope', 'stokes', 'date'],
         [dict(lst='05h', scope=b_, stokes='I', date=dt) for b_ in ('46MHz', '50MHz') for dt in (D1, D2, D3, NT1)])
    refuses('a delta build on another image table than the all-nights build\'s', 'image table', images='images_other.csv')
    refuses('an unknown frozen mode', 'EXOPIPE3_FROZEN_MODE', frozen='bogus', record=None)
    refuses('an unknown touch value', 'EXOPIPE3_FROZEN_TOUCH', touch='some')
    # a Master image: the record names it (path, size, modification time); the delta stands only on the same file
    m = f'{d}/master.fits'
    open(m, 'w').write('x' * 100)
    rc0, _t = run(d, b, sha, sel, 'out_rm', tab, img, frozen='record', extra=dict(STUB_MASTER=m))
    recm = f'{d}/out_rm/builder_frozen.json'
    rc1, txt1 = run(d, b, sha, s1, 'out_dm', t1, img, frozen='delta', record=recm, extra=dict(STUB_MASTER=m))
    check('a delta build with the Master images the record names runs', rc0 == 0 and rc1 == 0, (rc0, rc1, txt1[-300:]))
    open(m, 'w').write('x' * 101)
    refuses('a delta build after a Master image changed', 'do not stand for this build', record=recm, extra=dict(STUB_MASTER=m))
    env = GS.builder_env(t1, 'sublst')
    envd = GS.builder_env(t1, 'sublst', frozen='delta', frozen_record=rec, frozen_touch='all')
    names = ('EXOPIPE3_FROZEN_MODE', 'EXOPIPE3_FROZEN_RECORD', 'EXOPIPE3_FROZEN_TOUCH', 'EXOPIPE3_FROZEN_BUILDER_SHA1')
    check('builder_env: the four frozen names are always set (empty mode and record when not asked for), so an inherited '
          'value never reaches a build', all(k in env for k in names) and env['EXOPIPE3_FROZEN_MODE'] == ''
          and env['EXOPIPE3_FROZEN_RECORD'] == '' and envd['EXOPIPE3_FROZEN_MODE'] == 'delta'
          and envd['EXOPIPE3_FROZEN_RECORD'] == os.path.abspath(rec) and envd['EXOPIPE3_FROZEN_TOUCH'] == 'all'
          and env['EXOPIPE3_FROZEN_BUILDER_SHA1'], (env, envd))
    for what, kw in (('delta without a record', dict(frozen='delta')), ('a record without delta', dict(frozen='record', frozen_record=rec)),
                     ('an unknown mode', dict(frozen='bogus'))):
        try:
            GS.builder_env(t1, 'sublst', **kw)
            got = 'no refusal'
        except GS.StageRefusal as e:
            got = str(e)
        check(f'builder_env: {what} REFUSES', 'REFUSING' in got, got)


def test_night_dirs():
    d = f'{T}/night_dirs'
    shutil.rmtree(d, ignore_errors=True)
    mc = ['cand_id', 'lst', 'stokes', 'snr']
    pc = ['cand_id', 'sys_arcmin']
    wcsv(f'{d}/run05/master_catalog.csv', mc, [dict(cand_id='I1_05h_a', lst='05h', stokes='I', snr='8')])
    wcsv(f'{d}/run05/pertie_detections.csv', pc, [dict(cand_id='I1_05h_a', sys_arcmin='0.5')])
    wcsv(f'{d}/run06/master_catalog.csv', mc, [dict(cand_id='I1_06h_b', lst='06h', stokes='I', snr='9')])
    wcsv(f'{d}/run06/pertie_detections.csv', pc, [dict(cand_id='I1_06h_b', sys_arcmin='0.4')])
    for i, (cid, extra) in enumerate((('I1_05h_n1', []), ('I1_05h_n2', [dict(cand_id='I1_05h_n1', lst='05h', stokes='I', snr='7')]))):
        wcsv(f'{d}/night{i + 1}/master_catalog.csv', mc, extra + [dict(cand_id=cid, lst='05h', stokes='I', snr='7')])
        wcsv(f'{d}/night{i + 1}/pertie/pertie_detections.csv', pc, [dict(cand_id=cid, sys_arcmin='0.3')])
    runs = {'05h': f'{d}/run05', '06h': f'{d}/run06'}
    m, p, _r = GS.assemble_tenmin_input(['05h', '06h'], f'{d}/out', lst_dirs=runs,
                                        night_dirs=[('05h', f'{d}/night1'), ('05h', f'{d}/night2')])
    check('night directories: each night\'s rows follow its LST\'s run in the order given, before the next LST; a row a later '
          'night repeats with the same content is kept once',
          [r['cand_id'] for r in rcsv(m)[1]] == ['I1_05h_a', 'I1_05h_n1', 'I1_05h_n2', 'I1_06h_b']
          and [r['cand_id'] for r in rcsv(p)[1]] == ['I1_05h_a', 'I1_05h_n1', 'I1_05h_n2', 'I1_06h_b'], (rcsv(m)[1], rcsv(p)[1]))
    wcsv(f'{d}/night3/master_catalog.csv', mc, [dict(cand_id='I1_05h_a', lst='05h', stokes='I', snr='8.5')])
    wcsv(f'{d}/night3/pertie/pertie_detections.csv', pc, [dict(cand_id='I1_05h_a', sys_arcmin='0.5')])
    for what, lsts, nd in (('a night row with the cand_id of a run row and another content', ['05h', '06h'], [('05h', f'{d}/night3')]),
                           ('a night directory of an LST outside --lsts', ['06h'], [('05h', f'{d}/night1')]),
                           ('a night directory named twice', ['05h', '06h'], [('05h', f'{d}/night1'), ('05h', f'{d}/night1')]),
                           ('a night directory without a catalog', ['05h', '06h'], [('05h', f'{d}/nowhere')])):
        try:
            GS.assemble_tenmin_input(lsts, f'{d}/out2', lst_dirs={k: v for k, v in runs.items() if k in lsts}, night_dirs=nd)
            got = 'assembled'
        except GS.StageRefusal as e:
            got = str(e)
        check(f'night directories: {what} REFUSES', 'REFUSING' in got or 'night directory' in got, got)


ORDER = ['test_patches', 'test_modes', 'test_night', 'test_order', 'test_refusals', 'test_night_dirs']


def main(names):
    t0 = time.time()
    shutil.rmtree(T, ignore_errors=True)
    os.makedirs(T)
    try:
        for n in names or ORDER:
            t = time.time()
            try:
                globals()[n]()
            except Exception as e:              # noqa: BLE001 - a crash is a failure of that test
                import traceback
                traceback.print_exc()
                check(f'{n} crashed: {type(e).__name__}: {e}', False)
            print(f'  ({n}: {time.time() - t:.1f} s)', flush=True)
    finally:
        shutil.rmtree(T, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed in {time.time() - t0:.0f} s'
          + (f'; FAILED: {FAILS}' if FAILS else ''))
    return 1 if FAILS else 0


if __name__ == '__main__':
    os.chdir(REPO)
    sys.exit(main(sys.argv[1:]))
