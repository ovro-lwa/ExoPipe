"""The stub catalogue builder of tests/grouping_stub_builder.py WITH the eleven further patch points of the frozen statistics
(exopipe/phase3/lib/catalog/wraps/builder_grouping.py FROZEN_PATCHES): the other-band match, the nights searched, the repeat
statistic's S/N list and covering epochs, the counts of sources at or above a threshold behind a cross-match, the rank, and
the footprint area and sample density loops (which print how many footprint tests ran: 0 in a delta build).
Made from PATCHABLE_BUILDER by exact-text insertions (each anchor must occur once), so the twelve grouping patch points and
the census stay as the grouping tests have them. Not a test file itself: imported by tests/test_catalog_frozen_builder.py."""
from grouping_stub_builder import PATCHABLE_BUILDER


def _ins(src, anchor, text, before=False, replace=False):
    if src.count(anchor) != 1:
        raise AssertionError(f'frozen stub: the anchor {anchor!r} occurs {src.count(anchor)} times in the grouping stub')
    return src.replace(anchor, text if replace else (text + anchor if before else anchor + text))


def _make():
    s = PATCHABLE_BUILDER
    s = _ins(s, 'import csv, json, math, os, sys\n', 'import numpy as np\n')
    s = _ins(s, '# census: identity_id x4 (COLS above, the source dict, the merge below twice)\n',
             "COLS += ['n_nights_searched', 'other_bands', 'n_cover', 'xm_best', 'xm_N', 'xm_E_at_snr', 'xm_E_all', 'xm_nobs']\n")
    s = _ins(s, 'def _sys_det(r):\n    return 52.0\n',
             "def _sys_det(r):\n    return 52.0 + float(r.get('sys') or 0.0)\n", replace=True)
    # the population-wide inputs: the image table's epochs, the footprint and density, the band systematic, the other bands
    s = _ins(s, 'allout, allmem = {}, {}\n', r'''BANDS = sorted(byband)
nsearched = defaultdict(set)
if os.path.exists(imgcsv):
    for r in csv.DictReader(open(imgcsv)):
        nsearched[(r['lst'], r['scope'])].add(r['date'])
EFLOOR = 20.0


def _elev(lst, ra, dec):
    return 90.0 - abs(dec - 37.0)


LSTS = sorted({r['lst'] for r in rows})
BEAM_AS = {b: 600.0 for b in BANDS}


def master_path(lst, band):
    return os.environ.get('STUB_MASTER') or None


class _Angle:
    def __init__(self, deg):
        self.deg = deg


class _XC:
    def __init__(self, ra, dec):
        self.ra, self.dec = _Angle(ra), _Angle(dec)


xc = [_XC(10.0 * i, 5.0 * i) for i in range(1, 9)]
xsamples = np.array(['S1'] * 8)
mc_ra = np.linspace(0.0, 350.0, 36)
mc_dec = np.linspace(-80.0, 80.0, 36)
N_FOOTPRINT = [0]


def in_footprint(band, ra, dec):
    N_FOOTPRINT[0] += 1
    return dec > -30.0


AREA = {}
for band in BANDS:
    frac = np.mean([in_footprint(band, r_, d_)
                    for r_, d_ in zip(mc_ra, mc_dec)])
    AREA[band] = 41253.0 * frac

RHO = {}
for band in BANDS:
    inb = np.array([in_footprint(band, c.ra.deg, c.dec.deg) for c in xc])
    for smp in set(xsamples):
        n = int(((xsamples == smp) & inb).sum())
        RHO[(smp, band)] = n / max(AREA[band], 1.0)
print(f'footprint tests: {N_FOOTPRINT[0]}', flush=True)
SYS_AS = {b: float(np.median([_sys_det(r) for r in byband[b]])) for b in BANDS}
allband_positions = {b: (np.array([(r['_ra'], r['_dec']) for r in byband[b]]), np.array([r['_snr'] for r in byband[b]]),
                         np.array([0.0 for r in byband[b]]), [None for r in byband[b]]) for b in BANDS}
''', before=True)
    # F3 (nights searched), F1 and F2 (the other-band match) in the group loop
    s = _ins(s, "        cells = sorted({m['lst'] for m in mem})\n", r'''        nsr = len(set().union(*[nsearched.get((c_, band), set())
                                for c_ in cells]))
        ob = []
        for b2 in BANDS:
            if b2 == band or b2 not in allband_positions:
                continue
            c2, s2, f2, k2 = allband_positions[b2]
            sep2 = np.hypot(c2[:, 0] - ra, c2[:, 1] - dec) * 3600.0
            m2 = sep2 < 3.0 * np.sqrt(60.0 ** 2 + SYS_AS.get(b2, 0.0) ** 2 + 0.0 * f2)
            if m2.any():
                ob.append(f'{b2}:{s2[m2][np.argmax(np.abs(s2[m2]))]:.1f}')
''')
    s = _ins(s, "            review_dir='', _rpt_pct=(50.0 if len(mem) > 1 else ''), e_rep=('0.1' if len(mem) > 1 else ''), _matches=[],\n"
                "            xm_E_at_snr=''))\n",
             "            review_dir='', _rpt_pct=(50.0 if len(mem) > 1 else ''), e_rep='', n_nights_searched=nsr,\n"
             "            other_bands=';'.join(ob), _dets=[(m['cand_id'], m['_ra'], m['_dec'], m['_snr']) for m in mem],\n"
             "            _matches=([(0.01, float(best.get('xsep') or 1.0), best['xsmp'], 'star', '')] if best.get('xsmp') else []),\n"
             "            xm_E_at_snr=''))\n", replace=True)
    # F4 and F5: the repeat statistic of the band (its S/N list, the covering epochs)
    s = _ins(s, '    allout[band] = out\n', r'''    _snrs = np.sort(np.array([abs(r['_snr']) for r in br]))
    _wts = None

    def _n_ge(s):
        return int(len(_snrs) - int(np.searchsorted(_snrs, s)))
    for s_ in out:
        dets = sorted((abs(float(d[3])) for d in s_['_dets']), reverse=True) if s_.get('_dets') else [abs(float(s_['best_snr']))]
        N = len(dets); ra_, dec_ = float(s_['ra']), float(s_['dec'])
        M = sum(len(v) for (l_, sc_), v in nsearched.items() if sc_ == band and _elev(l_, ra_, dec_) >= EFLOOR)
        s_['n_cover'] = M
        s_['e_rep'] = (f'{_n_ge(dets[-1]) * 0.01 * max(M, 1):.4g}' if N >= 2 else '')
''', before=True)
    # F6 to F8: the counts behind a cross-match
    s = _ins(s, "for band in allout:\n    for s in allout[band]:\n        a = abs(s['best_snr'])\n        lab = []\n", r'''allsrc = [s for band in allout for s in allout[band]]


def n_ge_band(band, thr):
    return sum(1 for s in allout.get(band, []) if abs(s['best_snr']) >= thr)


def E_band(smp, band, thr, r_deg):
    return n_ge_band(band, thr) * RHO.get((smp, band), 0.0) * np.pi * r_deg ** 2


def E_allbands(smp, thr, r_deg):
    return sum(n_ge_band(b, thr) * RHO.get((smp, b), 0.0) for b in allout) * np.pi * r_deg ** 2


def n_ge(thr):
    return sum(1 for s in allsrc if abs(s['best_snr']) >= thr)


_obs = defaultdict(list)
for band in BANDS:
    for s in allout.get(band, []):
        if s['_matches']:
            _obs[s['_matches'][0][2]].append(abs(s['best_snr']))
for v in _obs.values():
    v.sort()


def n_obs_ge(smp, thr):
    import bisect
    v = _obs.get(smp, [])
    return len(v) - bisect.bisect_left(v, thr)


for band in allout:
    for s in allout[band]:
        a = abs(s['best_snr'])
        lab = []
        if s['_matches']:
            pch, d_, smp, n_, dd_ = s['_matches'][0]
            N = n_ge(a)
            _r_obs_deg = float(d_) / 60.0
            e_here = E_band(smp, band, a, _r_obs_deg); e_all = E_allbands(smp, a, _r_obs_deg)
            nobs = n_obs_ge(smp, a)
            s.update(xm_best=f'{smp}:{n_}', xm_N=N, xm_E_at_snr=round(e_here, 6), xm_E_all=round(e_all, 6), xm_nobs=nobs)
            if a < THR and e_here < 0.5:
                lab.append('E:' + smp)
''', replace=True)
    s = _ins(s, "json.dump({}, open(f'{outdir}/E_params.json', 'w'))\n",
             "json.dump(dict(best_snr={b: sorted(abs(s['best_snr']) for s in allsrc if s['band'] == b) for b in BANDS}),\n"
             "          open(f'{outdir}/E_params.json', 'w'))\n"
             "open(f'{outdir}/README_source_catalogs.txt', 'w').write(f'{len(allsrc)} sources before the gates\\n')\n", replace=True)
    return s


FROZEN_BUILDER = _make()
