"""B-30 (a) fallback sites of the search engine (lib/search/deep_search.py, candidates.py, search_v.py, deep_step.py,
tier_sublst.py; coadd.build_striation_model; core/fallback.py) and the ruled ephemeris time ("ephemerides at frame mid-time":
search_detect.EPHEMERIS_TIME, search_sublst.EPHEMERIS_TIME).

Per helper site, on synthetic inputs: under the ruled policy the site refuses naming its key (a count-only site counts);
under the production policy 'fallback' the function returns what production returns, counted. The sites inside
run_deep_cell (PRODNAME, pbcorr sibling, beam fit, epochs without a flow, weights.csv, sanity gates) need a real cell: they
are covered by the registry test (every key named by a site), _unmeasured_searched's own check and the search parity
replay; the DURATION block is compared here with production's own text (phase3_deep.py) over every value case.
Run: python tests/test_search_b30.py
"""
import ast
import json
import math
import os
import shutil
import sys
import textwrap

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')

from astropy.io import fits                                          # noqa: E402
from exopipe.phase3.config import CFG                                # noqa: E402
from exopipe.phase3.core import fallback                             # noqa: E402
from exopipe.phase3.core.record import Refusal                       # noqa: E402

T = '/tmp/exopipe3_tests_search_b30'
PROD_DEEP = '/lustre/gh/main/exopipe/exopipe/phase3/phase3_deep.py'      # production's deep search (read only)
PARITY = os.path.join(REPO, 'exopipe', 'phase3', 'parity', 'production_20261013.cfg')
FAILS, N = [], [0]


def check(name, ok, detail=''):
    N[0] += 1
    print(('PASS  ' if ok else 'FAIL  ') + name + ('' if ok else f'  ({detail})'))
    if not ok:
        FAILS.append(name)


def outcome(fn):
    try:
        return fn(), None
    except fallback.FallbackRefused as e:
        return None, str(e)


def under(key, pol, fn):
    """Run fn() with ``key`` at policy ``pol``; returns (result, refusal text, (refusals, counts) recorded)."""
    CFG.reset()
    CFG.authorize([], out=T)
    CFG.set(key, pol)
    m = fallback.mark()
    r, ref = outcome(fn)
    st = fallback.since(m)
    CFG.reset()
    return r, ref, st


def raises(fn, cls):
    try:
        fn()
    except cls as e:
        return str(e)
    return None


def duration_blocks():
    """(production block, branch block): the epoch frame's DURATION, from production phase3_deep.py and the branch."""
    def cut(lines, i, end):
        j = next(k for k in range(i, len(lines)) if end in lines[k])
        return textwrap.dedent('\n'.join(lines[i:j]))
    pl = open(PROD_DEEP).read().split('\n')
    i = next(k for k, ln in enumerate(pl) if "hdr['DURATION'] = float(fits.getheader(" in ln) - 1
    assert pl[i].strip() == 'try:', pl[i]
    bl = open(os.path.join(REPO, 'exopipe', 'phase3', 'lib', 'search', 'deep_search.py')).read().split('\n')
    j = next(k for k, ln in enumerate(bl) if '_dur_fb = None' in ln)
    return cut(pl, i, 'wcs = WCS(hdr).celestial'), cut(bl, j, 'wcs = WCS(hdr).celestial')


def main():
    shutil.rmtree(T, ignore_errors=True)
    os.makedirs(T)
    from exopipe.phase3.lib.search import deep_search as DS, search_v as SV, deep_step as DST, tier_sublst as TS
    from exopipe.phase3.lib.search import candidates as CA
    from exopipe.phase3.lib.masters import coadd as CO
    try:
        # ---- garbage pixels: a per-item drop, counted under the ruled default, once per file -----------------------
        a = np.zeros((8, 8), np.float32)
        a[1, 1], a[2, 2], a[3, 3] = 1e9, np.nan, -1e8
        p = f'{T}/zap.fits'
        fits.writeto(p, a)
        CFG.reset()
        DS._ZAP_COUNTED.clear()
        m = fallback.mark()
        r = DS._load(p)
        check('zapped garbage pixels: counted (n pixels) under the ruled default, the array as production',
              fallback.since(m) == ([], {'fallbacks.SEARCH_ZAP_PIXELS_FALLBACK': 2}) and r[1, 1] == 0 and r[2, 2] == 0
              and r[3, 3] == 0 and CFG.fallbacks.SEARCH_ZAP_PIXELS_FALLBACK == 'fallback', fallback.since(m))
        m = fallback.mark()
        r2 = DS._load(p)
        check('a file read again (a streamed epoch at its next use): the same array, not counted again',
              fallback.since(m) == ([], {}) and np.array_equal(r, r2), fallback.since(m))

        # ---- ephemeris time: the 00:00 default refuses; the ruled frame mid-time ------------------------------------
        k = 'fallbacks.SEARCH_EPHEMERIS_TIME_FALLBACK'
        r, ref, st = under(k, 'refuse', lambda: DS._body_skycoord('moon', None, '2025-08-16'))
        check('ephemeris without an epoch time refuses under the ruled policy', ref and k in ref, ref)
        r, ref, st = under(k, 'refuse', lambda: DS._body_skycoord('moon', 'not-a-time', '2025-08-16'))
        check('ephemeris with an unparsable epoch time refuses', ref and 'unparsable' in ref, ref)
        r, ref, st = under(k, 'fallback', lambda: DS._body_skycoord('moon', None, '2025-08-16'))
        want = DS._body_skycoord('moon', '2025-08-16T00:00:00', '2025-08-16')
        check('ephemeris fallback: 00:00 UTC of the date, counted', ref is None and st[1] == {k: 1}
              and r.separation(want).arcsec < 1e-6, (ref, st))
        sub = f'{T}/base/subtracted'
        os.makedirs(sub)
        h = fits.Header()
        h['DATE-OBS'], h['DURATION'] = '2025-08-16T05:00:00', 3600.0
        fits.writeto(f'{sub}/2025-08-16_I_sub_41MHz.fits', np.zeros((4, 4), np.float32), h)
        h2 = fits.Header()          # every real epoch carries MJD-MID (the first branch of frame_mid_time)
        h2['DATE-OBS'], h2['DURATION'], h2['MJD-MID'] = '2025-08-18T05:00:00', 3600.0, 60905.25     # 2025-08-18T06:00
        fits.writeto(f'{sub}/2025-08-18_I_sub_41MHz.fits', np.zeros((4, 4), np.float32), h2)
        b, ot = f'{T}/base', '2025-08-16T05:00:00'
        os.environ['EXOPIPE3_EPHEMERIS_TIME'] = 'date-obs'
        check('production ephemeris time (date-obs): the epoch DATE-OBS, unchanged',
              DS._eph_time(b, '2025-08-16', '41MHz', ot) == ot)
        os.environ['EXOPIPE3_EPHEMERIS_TIME'] = 'frame-mid'
        te = DS._eph_time(b, '2025-08-16', '41MHz', ot)
        te2 = DS._eph_time(b, '2025-08-18', '41MHz', '2025-08-18T05:00:00')
        check('ruled ephemeris time: the frame mid-time (DATE-OBS 05:00 + DURATION 3600 s -> 05:30)',
              te.startswith('2025-08-16T05:30:00'), te)
        check('ruled ephemeris time on a frame with MJD-MID: MJD-MID (06:00), not DATE-OBS + DURATION/2',
              te2.startswith('2025-08-18T06:00:00'), te2)
        os.environ.pop('EXOPIPE3_EPHEMERIS_TIME', None)
        te3 = DS._eph_time(b, '2025-08-16', '41MHz', ot)
        check('no EXOPIPE3_EPHEMERIS_TIME (a process no step started): the ruled default frame-mid',
              te3 == te and CFG.search_detect.EPHEMERIS_TIME == 'frame-mid', te3)
        os.environ['EXOPIPE3_EPHEMERIS_TIME'] = 'frame-midd'
        e = raises(lambda: DS._eph_time(b, '2025-08-16', '41MHz', ot), RuntimeError)
        check('an unknown EXOPIPE3_EPHEMERIS_TIME refuses (a typo never selects DATE-OBS)', e and 'REFUSING' in e, e)
        os.environ['EXOPIPE3_EPHEMERIS_TIME'] = 'frame-mid'
        r, ref, st = under(k, 'refuse', lambda: DS._eph_time(b, '2025-08-17', '41MHz', '2025-08-17T05:00:00'))
        check('frame-mid without a readable frame refuses under the ruled policy (the reason named)', ref and k in ref
              and 'FileNotFoundError' in ref, ref)
        r, ref, st = under(k, 'fallback', lambda: DS._eph_time(b, '2025-08-17', '41MHz', '2025-08-17T05:00:00'))
        check('frame-mid without a readable frame under fallback: its DATE-OBS, counted', r == '2025-08-17T05:00:00'
              and st[1] == {k: 1}, (r, st))
        os.environ.pop('EXOPIPE3_EPHEMERIS_TIME', None)
        env = DST.engine_env(f'{T}/c', f'{T}/a', f'{T}/f', 'X', f'{T}/ar', f'{T}/tmp')
        plan = {'label': 'SLST09B', 'arm': 'B', 'lst': '09h'}
        env10 = TS.engine_env(plan, f'{T}/att')
        check('engine_env (deep and 10-minute) passes the ruled frame-mid to the engine child',
              env.get('EXOPIPE3_EPHEMERIS_TIME') == 'frame-mid' == env10.get('EXOPIPE3_EPHEMERIS_TIME')
              == CFG.search_sublst.EPHEMERIS_TIME, (env.get('EXOPIPE3_EPHEMERIS_TIME'), env10.get('EXOPIPE3_EPHEMERIS_TIME')))
        CFG.authorize([], out=T)
        CFG.apply_override_file(PARITY)
        env = DST.engine_env(f'{T}/c', f'{T}/a', f'{T}/f', 'X', f'{T}/ar', f'{T}/tmp')
        env10 = TS.engine_env(plan, f'{T}/att')
        CFG.reset()
        check('the parity file gives both tiers production\'s date-obs', env['EXOPIPE3_EPHEMERIS_TIME'] == 'date-obs'
              == env10['EXOPIPE3_EPHEMERIS_TIME'], (env['EXOPIPE3_EPHEMERIS_TIME'], env10['EXOPIPE3_EPHEMERIS_TIME']))
        CFG.authorize([], out=T)
        CFG.set('search_detect.EPHEMERIS_TIME', 'frame_mid')
        CFG.set('search_sublst.EPHEMERIS_TIME', 'mid')
        e1 = raises(lambda: DST.engine_env(f'{T}/c', f'{T}/a', f'{T}/f', 'X', f'{T}/ar', f'{T}/tmp'), Refusal)
        e2 = raises(lambda: TS.engine_env(plan, f'{T}/att'), Refusal)
        CFG.reset()
        check('an unknown EPHEMERIS_TIME refuses in either tier before any child starts', e1 and 'search_detect' in e1
              and e2 and 'search_sublst' in e2, (e1, e2))

        # ---- candidates' body labels: at the vetoes' time; a body that fails is a site --------------------------------
        src = open(os.path.join(REPO, 'exopipe', 'phase3', 'lib', 'search', 'deep_search.py')).read()
        rdc = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == 'run_deep_cell')
        wc = [n for n in ast.walk(rdc) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
              and n.func.attr == 'write_candidates']
        ots = sorted(ast.get_source_segment(src, n.args[13]) for n in wc)
        check('run_deep_cell writes candidates with the vetoes\' ephemeris time (never the bare DATE-OBS)', len(wc) == 2
              and ots[0] == '_te_' and ots[1].startswith('_te_v[d]') and '_eph_time(' in ots[1], ots)
        k = 'fallbacks.SEARCH_EPHEMERIS_LABEL_FALLBACK'
        real_gb = CA.get_body

        def gb(body, t, loc):
            if body == 'mars':
                raise ValueError('no ephemeris')
            return real_gb(body, t, loc)
        CA.get_body = gb
        lab = lambda: CA.ephemeris_label(10.0, -60.0, '2025-08-16T05:30:00')
        try:
            r, ref, st = under(k, 'refuse', lab)
            check('a body whose position fails refuses the label under the ruled policy', ref and k in ref and 'mars' in ref,
                  ref)
            r, ref, st = under(k, 'fallback', lab)
            check('the same under fallback: production\'s label (\'\' here), counted', r == '' and st[1] == {k: 1}, (r, st))
        finally:
            CA.get_body = real_gb
        m = fallback.mark()
        check('a label with every body computed is not a fallback',
              CA.ephemeris_label(10.0, -60.0, '2025-08-16T05:30:00') == '' and fallback.since(m) == ([], {}))

        # ---- epochs the sanity / search-tier gates could not measure: named only where they will be searched ---------
        un = DS._unmeasured_searched(['a', 'b', 'c', 'd'], {'I': {'b'}, 'V': set()},
                                     {'I': {'a': 1, 'b': 1, 'c': 1}, 'V': {'a': 1, 'd': 1}}, {'b', 'c'},
                                     {('a', 'V'), ('d', 'V'), ('c', 'I'), ('e', 'I'), ('c', 'V')})
        check('unmeasured epochs: named per Stokes only when searched (not refused, not skipped, a frame for that Stokes) - '
              'e.g. an epoch with no V frame and no V rms (10 installed cells) is never named', un == {'I': ['c'], 'V': ['a', 'd']},
              un)

        # ---- master structure map: beam <= 0, destripe failure -----------------------------------------------------
        mv = np.random.default_rng(3).normal(0, 1, (32, 32)).astype(np.float64)
        k = 'fallbacks.SEARCH_BEAM_FALLBACK'
        r, ref, st = under(k, 'refuse', lambda: DS._mstruct_maps(mv, f'{T}/nomaster.fits', 0.0))
        check('structure map with no beam refuses under the ruled policy', ref and k in ref, ref)
        k2 = 'fallbacks.SEARCH_MSTRUCT_DESTRIPE_FALLBACK'
        r, ref, st = under(k2, 'refuse', lambda: DS._mstruct_maps(mv, f'{T}/nomaster.fits', 3.0))
        check('structure map whose destripe fails refuses under the ruled policy', ref and k2 in ref, ref)
        r, ref, st = under(k2, 'fallback', lambda: DS._mstruct_maps(mv, f'{T}/nomaster.fits', 3.0))
        check('destripe failure under fallback: the raw map twice, counted', ref is None and st[1] == {k2: 1}
              and r[0] is r[1], (ref, st))

        # ---- residuals that fail in the fork workers: the site in the parent, n = epochs -----------------------------
        saved = DS._residual
        DS._residual = lambda epoch, master, flow, hdr, cd: (_ for _ in ()).throw(ValueError('bad')) if epoch is None \
            else np.asarray(epoch) - master
        ep = {'d1': np.ones((4, 4)), 'd2': None, 'd3': np.ones((4, 4)), 'd4': None}
        k = 'fallbacks.SEARCH_RESIDUAL_FALLBACK'
        try:
            for spill in (None, f'{T}/spill'):
                if spill:
                    os.makedirs(spill)
                call = lambda: DS._build_residuals_mp(ep, np.zeros((4, 4)), {d: None for d in ep}, None, 1.0, list(ep), 2,
                                                      spill=spill, tag='t')
                how = 'streaming (spill)' if spill else 'in memory'
                r, ref, st = under(k, 'refuse', call)
                check(f'{how}: residuals that failed in workers refuse in the parent under the ruled policy', ref
                      and k in ref and 'd2' in ref and 'd4' in ref, ref)
                r, ref, st = under(k, 'fallback', call)
                check(f'{how}: under fallback the epochs are dropped as production, counted per epoch (2)', ref is None
                      and sorted(r) == ['d1', 'd3'] and st[1] == {k: 2} and float(np.asarray(r['d1']).sum()) == 16.0,
                      (ref, st))
        finally:
            DS._residual = saved

        # ---- striation blocks whose world coordinates fail: counted (n rows), in both copies ---------------------------
        class BadWCS:
            class wcs:
                cdelt = np.array([0.0625, 0.0625])

            def all_pix2world(self, *a):
                raise ValueError('no celestial solution')
        k = 'fallbacks.STRIATION_WCS_BLOCK_FALLBACK'
        data, mask = np.ones((600, 20)), np.zeros((600, 20), bool)
        for name, fn in (('coadd (the search)', CO.build_striation_model), ('search_v (V-clean)', SV.build_striation_model)):
            CFG.reset()
            m = fallback.mark()
            r = fn(data, BadWCS(), mask)
            check(f'{name}: every block failing - counted under the ruled default (n = 600 rows), production\'s zero model',
                  fallback.since(m) == ([], {k: 600}) and not np.any(r[0]) and r[1] == 0, fallback.since(m))
            r, ref, st = under(k, 'refuse', lambda: fn(data, BadWCS(), mask))
            check(f'{name}: under \'refuse\' the block refuses (the error named)', ref and k in ref
                  and 'no celestial solution' in ref, ref)

        # ---- search_v.main's primary-beam correction is dead, so not a site (a refusal there would refuse a V-clean
        #      master over a value no product reads): this check fails when a reader of it appears --------------------
        def key_reads(tree, keys):
            """Line numbers reading one of ``keys`` from a dict: x['k'] in load context or x.get('k', ...)."""
            out = []
            for n in ast.walk(tree):
                if isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Load):
                    sl = n.slice.value if isinstance(n.slice, ast.Index) else n.slice      # Python 3.8 / 3.9+
                    if isinstance(sl, ast.Constant) and sl.value in keys:
                        out.append(n.lineno)
                if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'get' and n.args
                        and isinstance(n.args[0], ast.Constant) and n.args[0].value in keys):
                    out.append(n.lineno)
            return out
        readers = []
        for dp, dn, fn_ in os.walk(os.path.join(REPO, 'exopipe')):
            dn[:] = [x for x in dn if x != '__pycache__']
            for f in fn_:
                if f.endswith('.py'):
                    pth = os.path.join(dp, f)
                    readers += [f'{os.path.relpath(pth, REPO)}:{ln}' for ln in
                                key_reads(ast.parse(open(pth).read()), ('true_flux', 'true_total_flux'))]
        svs = open(os.path.join(REPO, 'exopipe', 'phase3', 'lib', 'search', 'search_v.py')).read()
        readers += [f'search_v.py:{ln}' for ln in key_reads(ast.parse(svs), ('pb',))]    # the source dicts never leave it
        mn = next(n for n in ast.walk(ast.parse(svs)) if isinstance(n, ast.FunctionDef) and n.name == 'main')
        fm_ifs = [ast.get_source_segment(svs, n) for n in ast.walk(mn)
                  if isinstance(n, ast.If) and any(isinstance(x, ast.Name) and x.id == 'fm' for x in ast.walk(n.test))]
        loads = sum(1 for n in ast.walk(mn) if isinstance(n, ast.Name) and n.id == 'fm' and isinstance(n.ctx, ast.Load))
        check('search_v: pb / true_flux are read by nothing and fm feeds only apply_pb_correction (no PB / frequency site)',
              not readers and fm_ifs == ['if fm: apply_pb_correction(isrc,obs_time,fm)'] and loads == 2,
              (readers, fm_ifs, loads))
        check('search_v: the beam model and the frequency card are no longer sites',
              'fallback.site' not in svs[svs.index('def _get_beam_model'):svs.index('def load_image')])

        # ---- the frame DURATION: the branch block = production's, value and type, in every case ----------------------
        old, new = duration_blocks()
        MISSING = object()

        class Stub:
            def __init__(self, case):
                self.case = case

            def getheader(self, path):
                if self.case == 'unreadable':
                    raise OSError('cannot read')
                hh = fits.Header()
                if self.case is not MISSING:
                    hh['DURATION'] = self.case
                return hh

        class FB:
            def __init__(self):
                self.calls = []

            def site(self, key, detail, n=1):
                self.calls.append(key)
        diffs, fired = [], set()
        for case in (3600.0, 411.2, 0, 0.0, '', 'abc', '3600', float('nan'), True, False, MISSING, 'unreadable'):
            for cadence in ('deep', '10min'):
                res = {}
                for nm, code in (('old', old), ('new', new)):
                    fb, hd = FB(), {}
                    ns = dict(fits=Stub(case), hdr=hd, ebase='/b', d='2025-01-01', scope='41MHz', cadence=cadence, _FB=fb)
                    exec(code.replace('from exopipe.phase3.core import fallback', 'fallback = _FB'), ns)
                    res[nm] = (hd.get('DURATION'), len(fb.calls))
                o, nw = res['old'][0], res['new'][0]
                same = type(o) is type(nw) and (o == nw or (isinstance(o, float) and math.isnan(o) and math.isnan(nw)))
                if not same:
                    diffs.append((case, cadence, o, nw))
                if res['new'][1]:
                    fired.add('MISSING' if case is MISSING else repr(case))
                if res['old'][1]:
                    diffs.append(('production block called a site', case))
        check('DURATION block: production\'s value and type in all 24 cases (12 values x 2 cadences)', not diffs, diffs)
        print('      DURATION site fires for:', sorted(fired))
        # nan: astropy refuses a NaN card, so the stub's header read fails - production's default, the site fires
        check('DURATION block: the site fires exactly where the value falls back',
              fired == {'0', '0.0', "''", "'abc'", 'nan', 'False', 'MISSING', "'unreadable'"}, sorted(fired))

        # ---- the engine children report a refusal (a site) as refused, exit 2 ----------------------------------------
        spec = f'{T}/spec.json'
        json.dump({'label': 'X', 'stokes': 'V', 'cadence': 'deep', 'lst': '23h', 'scope': '23MHz', 'cell': '23h_23MHz',
                   'markers': [], 'result': f'{T}/result.json'}, open(spec, 'w'))
        spec10 = f'{T}/spec10.json'
        json.dump({'label': 'X', 'stokes': 'V', 'cadence': '10min', 'lst': '09h', 'band': '18MHz', 'cell': '09h_18MHz',
                   'included_dates': None, 'markers': ['REFUSING'], 'result': f'{T}/result10.json'}, open(spec10, 'w'))
        from exopipe.phase3.core import env as E
        saved, saved_legacy, saved_argv = DS.run_deep_cell, E.legacy, list(sys.argv)
        DS.run_deep_cell = lambda *a, **kw: (_ for _ in ()).throw(fallback.FallbackRefused('B-30 fallback site x: y'))
        try:
            rc = DST._child(spec)
            E.legacy = lambda *a, **kw: None
            rc10 = TS.engine_child(spec10)
        finally:
            DS.run_deep_cell, E.legacy, sys.argv[:] = saved, saved_legacy, saved_argv
        res = json.load(open(f'{T}/result.json'))
        res10 = json.load(open(f'{T}/result10.json'))
        check('deep engine child: a Refusal (a fallback site) is refused, exit 2', rc == 2 and res['status'] == 'refused', res)
        check('10-minute engine child: a Refusal (a fallback site) is refused, exit 2', rc10 == 2 and res10['refusal'] is True,
              (rc10, res10))
    finally:
        CFG.reset()
        os.environ.pop('EXOPIPE3_EPHEMERIS_TIME', None)
        shutil.rmtree(T, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
