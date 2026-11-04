"""Content tests of families W (the forced targets' wide-area burst test) and N (the night-activity table) of `exopipe3 forced`
(core.validate registry; each runs on every file a unit writes, on every file --resume would keep, and alone through
`exopipe3 verify <class> <file>`). Each re-derives what the file states from its own rows - never only its presence:
  forced_wide_jobs        the jobs: unique (Stokes, identity), bands sorted subbands; a base job's background separation
                          recomputed from its two positions; an extension job's bright sources within its radius, sorted
  forced_wide_events      the forced events: unique (Stokes, tier, identity, cand_id), the frame id naming its band and time
  forced_wide_bursts      a burst file: the analysis' members, consistent lengths; every peak in a searched (ok) file / tier,
                          |S/N| >= SNR_MIN, S/N = value / rms exactly, position inside the cutout
  forced_wide_results     the verdict of every row from its own numbers (the centroid's z, the bursts, the candidates)
  forced_wide_candidates  the explains / excess / shown flags from the row's own ratio, p and centering
  forced_wide_published   a run's file = the results rows of its (Stokes, cadence) + one position per named source, each at the
                          distance the name states from the target
  night_activity_bands    every median and ratio recomputed from the table's own counts
  night_activity_nights   every night row recomputed from the bands table (activity, rank, counts)"""
import csv
import json
import math
import os
import re
import statistics

import numpy as np

from exopipe.phase3.core.validate import content_test


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def _rows(path):
    with open(path, newline='') as f:
        rd = csv.DictReader(f)
        return list(rd), list(rd.fieldnames or [])


def _band_ok(b):
    return isinstance(b, str) and b.endswith('MHz') and b[:-3].isdigit()


# ---------------------------------------------------------------------------------------------------------------------
# jobs and events
# ---------------------------------------------------------------------------------------------------------------------

@content_test('forced_wide_jobs')
def jobs_consistent(path, **ctx):
    from exopipe.phase3.lib.forced import wide_jobs as WJ
    w = _cfg().forced_wide
    with open(path) as f:
        doc = json.load(f)
    base, ext = doc.get('base'), doc.get('ext')
    if not isinstance(base, list) or not isinstance(ext, list):
        return False, None, 'no base / ext job lists'
    probs, seen = [], set()
    fa = WJ.FA()
    for kind, jobs, keys in (('base', base, WJ.BASE_KEYS), ('ext', ext, WJ.EXT_KEYS)):
        for j in jobs:
            miss = [k for k in keys if k not in j]
            if miss:
                probs.append('%s job %s lacks %s' % (kind, j.get('identity'), miss))
                continue
            k = (j['stokes'], j['identity'])
            if j['stokes'] not in ('I', 'V') or not j['identity'] or k in seen:
                probs.append('job %s: Stokes not I/V, no identity, or listed twice' % (k,))
            seen.add(k)
            ra, dec = j['T']
            if not (0.0 <= ra < 360.0 and -90.0 <= dec <= 90.0):
                probs.append('%s: target position %s out of range' % (k, j['T']))
            b = j['bands']
            if not b or not all(_band_ok(x) for x in b) or b != sorted(b, key=lambda x: float(x[:-3])) or len(set(b)) != len(b):
                probs.append('%s: bands %s not distinct subbands in frequency order' % (k, b))
            if kind == 'base':
                sep = float(fa.sep_arcsec(j['S'][0], j['S'][1], ra, dec)) / 60.0
                if abs(sep - j['sep_arcmin']) > 1e-9 or j['s74'] < w.BASE_BG_MIN_S74_JY or not j['bg_name']:
                    probs.append('%s: background %s %.3f Jy at %.4f arcmin (recomputed %.4f)' % (k, j['bg_name'], j['s74'],
                                                                                            j['sep_arcmin'], sep))
            else:
                br = j['bright']
                rmax = w.EXT_RADIUS_FWHM * WJ.beam_fwhm_arcmin(b[0])
                if not br or [x[2] for x in br] != sorted((x[2] for x in br), reverse=True):
                    probs.append('%s: bright sources missing or not by flux' % (k,))
                for x in br:
                    d = float(fa.sep_arcsec(x[0], x[1], ra, dec)) / 60.0
                    if x[2] < w.EXT_MIN_S74_JY or d >= rmax + 1e-6:
                        probs.append('%s: bright source %s %.2f Jy at %.2f arcmin (limit %.1f Jy, %.2f arcmin)'
                                     % (k, x[3], x[2], d, w.EXT_MIN_S74_JY, rmax))
    n = len(base) + len(ext)
    return (not probs and n > 0), n, ('%d base + %d extension jobs' % (len(base), len(ext)) if not probs
                                      else '; '.join(probs[:5]) + (' ...' if len(probs) > 5 else ''))


@content_test('forced_wide_events')
def events_consistent(path, **ctx):
    from exopipe.phase3.lib.forced.wide_jobs import EVENT_COLUMNS
    rows, cols = _rows(path)
    if tuple(cols) != EVENT_COLUMNS:
        return False, len(rows), 'columns %s != %s' % (cols, list(EVENT_COLUMNS))
    probs, seen = [], set()
    for r in rows:
        k = (r['stokes'], r['tier'], r['identity'], r['cand_id'])
        if k in seen:
            probs.append('event listed twice: %s' % (k,))
        seen.add(k)
        try:
            snr, sig = float(r['snr']), float(r['sigma_frame_jyb'])
        except ValueError:
            probs.append('%s: S/N or sigma not a number' % (k,))
            continue
        if r['stokes'] not in ('I', 'V') or r['tier'] not in ('10s', '1min') or not _band_ok(r['band']):
            probs.append('%s: Stokes / tier / band %s %s %s' % (k, r['stokes'], r['tier'], r['band']))
        if not (sig > 0 and math.isfinite(snr)):
            probs.append('%s: sigma %s, S/N %s' % (k, sig, snr))
        if '_%s_' % r['band'] not in r['cand_id'] or r['ts'][:8] not in r['cand_id']:
            probs.append('%s: the frame id does not name its band %s and date %s' % (k, r['band'], r['ts'][:8]))
    return not probs, len(rows), ('%d events' % len(rows) if not probs else '; '.join(probs[:5]))


# ---------------------------------------------------------------------------------------------------------------------
# burst files
# ---------------------------------------------------------------------------------------------------------------------

@content_test('forced_wide_bursts')
def bursts_consistent(path, stokes=None, identity=None, band=None, **ctx):
    from exopipe.phase3.lib.forced.wide_search import NPZ_KEYS, STATUSES
    c = _cfg().forced_wide_search
    z = np.load(path)
    # the analysis' own files (forced_wide_search.py) carry one more member, 'seconds' (their wall-clock time): read as inputs
    # with --wide-bursts, they are tested on the same members
    if tuple(z.files) not in (NPZ_KEYS, NPZ_KEYS + ('seconds',)):
        return False, None, 'members %s != %s' % (z.files, list(NPZ_KEYS))
    if band is not None and os.path.basename(path) != '%s_%s_%s.npz' % (stokes, identity, band):
        return False, None, 'file name does not match (%s, %s, %s)' % (stokes, identity, band)
    n = len(z['snr'])
    nf = len(z['f_status'])
    probs = []
    if any(len(z[k]) != n for k in NPZ_KEYS[:10]) or any(len(z[k]) != nf for k in NPZ_KEYS[10:15]):
        return False, n, 'member lengths differ'
    ok_files = {}
    for cell, tier, npl, zn, st in zip(z['f_cell'], z['f_tier'], z['f_planes'], z['f_zone'], z['f_status']):
        st = str(st)
        if st not in STATUSES and not st.startswith('ERROR '):
            probs.append('file status %r' % st)
        if int(zn) not in (-1, 0, 1) or str(tier) not in ('10s', '1min', '-') or int(npl) < 0 or (st != 'ok' and int(npl) != 0):
            probs.append('file row %s %s: tier %s planes %s zone %s' % (cell, st, tier, npl, zn))
        if st == 'ok':
            ok_files[(str(cell), str(tier))] = int(npl)
    if len({str(x) for x in z['f_cell']}) > int(z['n_files']):
        probs.append('%d files in the table > n_files %d' % (len({str(x) for x in z['f_cell']}), int(z['n_files'])))
    if n:
        snr, val, rms = z['snr'], z['value'], z['rms']
        if not np.all(np.abs(snr) >= c.SNR_MIN):
            probs.append('%d peaks below |S/N| %.1f' % (int(np.sum(np.abs(snr) < c.SNR_MIN)), c.SNR_MIN))
        if not np.all(rms > 0) or not np.array_equal(snr, val / rms):
            probs.append('S/N != value / rms (or rms <= 0)')
        if not (np.all(np.isfinite(z['x'])) and np.all(np.isfinite(z['y'])) and np.all(z['x'] >= c.EDGE_PX - 1)
                and np.all(z['y'] >= c.EDGE_PX - 1)):
            probs.append('a peak position outside the cutout')
        if not (np.all((z['ra'] >= 0) & (z['ra'] < 360)) and np.all(np.abs(z['dec']) <= 90)):
            probs.append('a sky position out of range')
        bad = [(str(a), str(b)) for a, b in set(zip(z['cell'].tolist(), z['tier'].tolist())) if ok_files.get((str(a), str(b)), 0) <= 0]
        if bad:
            probs.append('peaks of %d (cell, tier) without a searched file, e.g. %s' % (len(bad), bad[0]))
        if not all(len(str(t)) == 15 and str(t)[8] == '_' for t in z['ts']):
            probs.append('a frame time not YYYYMMDD_HHMMSS')
        if not set(z['tier'].tolist()) <= {'10s', '1min'}:
            probs.append('tiers %s' % sorted(set(z['tier'].tolist())))
    return not probs, n, ('%d peaks, %d file rows' % (n, nf) if not probs else '; '.join(probs[:5]))


# ---------------------------------------------------------------------------------------------------------------------
# the v3 results
# ---------------------------------------------------------------------------------------------------------------------

RESULT_COLUMNS = ('stokes', 'identity', 'target', 'tier', 'n_events', 'event_jy_median', 'bursts_at_target', 'bursts_by_band',
                  'n_candidates', 'n_capable', 'centroid_fwhm', 'centroid_arcmin', 'centroid_pa', 'centroid_err_fwhm',
                  'centroid_z', 'centroid_nights', 'v2_background', 'verdict', 'named', 'coverage', 'why')
# with the knife-edge flag (forced_wide_test.KNIFE_EDGE_SE set): the bootstrap's resamples, the centroid z's Monte Carlo error,
# the flag, and a flagged row's Monte Carlo draws (KNIFE_EDGE_DRAWS: the majority's fraction, every outcome's count, where the
# row's numbers come from)
MC_RESULT_COLUMNS = ('bootstrap_n', 'centroid_z_se', 'knife_edge', 'named_knife_edge', 'mc_draws', 'mc_n', 'mc_outcomes',
                     'mc_row')
NAMING = ('BACKGROUND SOURCE', 'CONSISTENT WITH A BACKGROUND SOURCE')
_KE = re.compile(r'^(?P<stat>.+) (?P<v>-?[0-9.]+(?:e[-+]?[0-9]+)?) vs (?P<t>[0-9.]+(?:e[-+]?[0-9]+)?(?: and [0-9.]+(?:e[-+]?[0-9]+)?)?), '
                 r'Monte Carlo error (?P<se>[0-9.]+(?:e[-+]?[0-9]+)?)$')


def knife_edge_problems(r, k):
    """[problem] of one results row's Monte Carlo columns: the knife_edge text names known statistics, each at one of its
    thresholds within k (KNIFE_EDGE_SE) of its Monte Carlo errors (the printed values' rounding allowed); the centroid z's entry
    agrees with the row's centroid_z and centroid_z_se."""
    t = _cfg().forced_wide_test
    out = []
    if r['bootstrap_n'] not in ('',) and not r['bootstrap_n'].isdigit():
        out.append('bootstrap_n %r' % r['bootstrap_n'])
    if r['centroid_z_se'] not in ('',) and not float(r['centroid_z_se']) >= 0:
        out.append('centroid_z_se %r' % r['centroid_z_se'])
    txt = r['knife_edge']
    if not txt:
        return out
    if not txt.startswith('knife-edge: '):
        return out + ['knife_edge %r' % txt[:60]]
    if r['centroid_z'] == '':
        out.append('knife-edge without a centroid')
    kinds = (('centroid z to ', {t.Z_C_ON, t.Z_C_NEAR}), ('centroid z', {t.Z_ON, t.Z_OFF}),
             ('taper p of ', {t.P_CONSISTENT, t.P_EXCESS}), ('taper ratio of ', {t.RATIO_CONSISTENT, t.R_EXCESS}),
             ('matched control lines (within ', {float(t.MATCH_MIN)}))
    for ent in txt[len('knife-edge: '):].split('; '):
        m = _KE.match(ent)
        if not m:
            out.append('knife-edge entry %r' % ent[:60])
            continue
        st, v, se = m.group('stat'), float(m.group('v')), float(m.group('se'))
        ths = [float(x) for x in m.group('t').split(' and ')]
        kind = next((ths_ for pre, ths_ in kinds if st.startswith(pre)), None)
        if kind is None or not all(any(abs(th - x) < 1e-9 for x in kind) for th in ths):
            out.append('knife-edge entry %r: an unknown statistic or threshold' % ent[:60])
        elif not all(se > 0 and abs(v - th) <= k * se * 1.05 + 0.006 for th in ths):
            out.append('knife-edge entry %r: not within %g errors of its threshold' % (ent[:60], k))
        if st == 'centroid z' and r['centroid_z'] != '' and r.get('mc_row', '') in ('', 'default', 'split') and (
                abs(v - float(r['centroid_z'])) > 0.051 or abs(se - float(r['centroid_z_se'])) > 0.051 * se + 1e-4):
            out.append('knife-edge entry %r: the centroid z / error of the row are %s / %s' % (ent[:60], r['centroid_z'],
                                                                                              r['centroid_z_se']))
    return out


def verdict_problems(r, tol=0.05):
    """[problem] of one results row: its verdict against its own numbers (the rule of forced_wide_v3.verdict_row; the centroid
    z is written to 0.1, so a z within ``tol`` of a threshold may sit on either side)."""
    from exopipe.phase3.lib.forced.wide_v3 import VERDICTS, SPLIT_PREFIX, verdict_class
    t = _cfg().forced_wide_test
    v_pub = r['verdict']
    out = []
    split = r.get('mc_row') == 'split'             # its draws split: UNDECIDED (knife-edge ...) whatever this draw's z
    if split and not (v_pub.startswith(SPLIT_PREFIX) and v_pub.endswith(')')):
        return ['a split row with the verdict %r' % v_pub[:60]]
    v = verdict_class(v_pub)                       # a source split ("<class> (source split: ...)") reads as its class
    if v != v_pub and not split and not (r.get('mc_row') or '').startswith(('default', 'draw ')):
        return ['the verdict %r on a row without Monte Carlo draws' % v_pub[:60]]
    if v not in VERDICTS and not split:
        return ['verdict %r unknown' % v_pub]
    by_band = sum(int(x.split(':')[1]) for x in r['bursts_by_band'].split()) if r['bursts_by_band'] else 0
    if by_band != int(r['bursts_at_target']):
        out.append('bursts_by_band sum %d != bursts_at_target %s' % (by_band, r['bursts_at_target']))
    if int(r['n_capable']) > int(r['n_candidates']):
        out.append('n_capable > n_candidates')
    if bool(r['named']) != (v in NAMING and v == v_pub):
        out.append('named %r with verdict %s' % (r['named'][:60], v_pub[:60]))
    if int(r['bursts_at_target']) < t.MIN_NOBS or r['centroid_z'] == '':
        if not v.startswith('UNDECIDED (fewer than 3'):
            out.append('fewer than %d bursts / no centroid, verdict %s' % (t.MIN_NOBS, v))
        return out

    z = float(r['centroid_z'])
    hi = z >= t.Z_OFF - tol
    lo = z < t.Z_ON + tol
    mid = (t.Z_ON - tol <= z < t.Z_OFF + tol)
    if int(r['n_capable']) == 0:
        allowed = set()
        if hi:
            allowed.add('NOT CENTERED ON THE TARGET (no cataloged source can make the events)')
        if lo:
            allowed.add('NOT EXPLAINED BY A CATALOGED SOURCE (no cataloged source can make the events)')
        if mid:
            allowed.add('UNDECIDED')
    else:
        allowed = set()
        if hi:
            allowed |= {'BACKGROUND SOURCE', 'NOT CENTERED ON THE TARGET (source not identified)'}
        if lo:
            allowed |= {'UNDECIDED (centered on the target and on a cataloged source alike)',
                        'NOT EXPLAINED BY A CATALOGED SOURCE (centered on the target)', 'UNDECIDED'}
        if mid:
            allowed |= {'BACKGROUND SOURCE', 'CONSISTENT WITH A BACKGROUND SOURCE', 'UNDECIDED'}
    if split:
        allowed = {v}
    if v not in allowed:
        out.append('centroid z %.1f with %d capable candidates allows %s, not %s' % (z, int(r['n_capable']), sorted(allowed), v))
    f, e = float(r['centroid_fwhm']), float(r['centroid_err_fwhm'])
    if e <= 0 or abs(f / e - z) > 0.05 + 0.0006 * (1 + f / e) / max(e, 1e-3):
        out.append('centroid z %.1f != offset %.3f / error %.3f' % (z, f, e))
    return out


def _parse_outcome(txt):
    """(verdict, set of source names) of an outcome's text (wide_v3.outcome_text)"""
    for v in NAMING:
        if txt.startswith(v + ' (') and txt.endswith(')'):
            return v, frozenset(txt[len(v) + 2:-1].split(', '))
    return txt, frozenset()


def draws_problems(r, K, KMAX=None):
    """[problem] of a row's Monte Carlo draws (K = the run's KNIFE_EDGE_DRAWS, KMAX its KNIFE_EDGE_DRAWS_MAX): exactly the
    knife-edge rows have them; their outcome counts (mc_outcomes) add up to the row's draws (mc_n: a multiple of K, K to KMAX),
    most frequent first; the row stopped drawing before KMAX only when settled; and the row is what those counts DECIDE
    (wide_v3.draws_decision, re-derived here from mc_outcomes): the decided verdict class and decidedly named sources with
    mc_draws; the settled case's class and source split; or the split verdict with no named source; named_knife_edge = the
    sources named in an undecided fraction."""
    from exopipe.phase3.lib.forced import wide_v3 as V3
    md, mo, mr, nk = r['mc_draws'], r['mc_outcomes'], r['mc_row'], r.get('named_knife_edge', '')
    mn = r.get('mc_n', '')
    if not K or not r['knife_edge']:
        return ['Monte Carlo draws on a row without them (no flag, or KNIFE_EDGE_DRAWS None)'] if (md or mo or mr or nk or mn) else []
    if not (mo and mr):
        return ['a knife-edge row without its Monte Carlo draws']
    KMAX = max(KMAX or K, K)
    out, outs = [], []
    for ent in mo.split(' | '):
        n, _sep, txt = ent.partition(': ')
        if not n.isdigit() or not txt:
            out.append('mc_outcomes entry %r' % ent[:60])
            continue
        v_, names = _parse_outcome(txt)
        outs += [(v_, sorted(names))] * int(n)
    ns = [int(e.partition(': ')[0]) for e in mo.split(' | ') if e.partition(': ')[0].isdigit()]
    Kr = len(outs)
    if str(Kr) != str(mn) or Kr % K or not K <= Kr <= KMAX or ns != sorted(ns, reverse=True):
        return out + ['mc_outcomes count %d draws (mc_n %r; batches of %d to %d) or not most frequent first: %s'
                      % (Kr, mn, K, KMAX, ns)]
    conf = _cfg().forced_wide_test.KNIFE_EDGE_DECIDE
    dec = V3.draws_decision(outs, Kr, conf)
    if Kr < KMAX and not V3.settled(dec):
        out.append('drawing stopped at %d of %d with the row not settled' % (Kr, KMAX))
    if nk != V3.knife_named_text(dec, Kr):
        out.append('named_knife_edge %r, the draws give %r' % (nk[:60], V3.knife_named_text(dec, Kr)[:60]))
    names_row = sorted(V3.named_names(r['named']))
    if dec['cls'] is None:
        if mr != 'split' or r['verdict'] != V3.split_verdict(dec, Kr) or r['named'] or md:
            out.append('the draws split (%s) but the row reads %s / %r / %r / %r' % (
                ', '.join('%s %d' % kv for kv in dec['classes'])[:80], mr, r['verdict'][:50], r['named'][:30], md[:30]))
    elif dec['no_source']:
        if r['verdict'] != V3.source_split_verdict(dec, Kr) or r['named'] or md != '%s in %d of %d Monte Carlo draws' % (
                dec['cls'], dec['n'], Kr):
            out.append('the draws decide %s with no source, the row reads %r / %r / %r' % (dec['cls'], r['verdict'][:70],
                                                                                         r['named'][:30], md[:40]))
    else:
        if r['verdict'] != dec['cls'] or names_row != sorted(dec['named']):
            out.append('the draws decide %s naming %s, the row reads %s naming %s' % (dec['cls'], sorted(dec['named']),
                                                                                   r['verdict'], names_row))
        if md != V3.decided_text(dec, Kr):
            out.append('mc_draws %r, the draws give %r' % (md[:80], V3.decided_text(dec, Kr)[:80]))
    if dec['cls'] is not None and not (mr == 'default' or (mr.startswith('draw ') and mr[5:].isdigit() and int(mr[5:]) < Kr)):
        out.append('mc_row %r' % mr)
    return out


_ABSENT = object()


@content_test('forced_wide_results')
def results_consistent(path, knife_edge_se=_ABSENT, knife_edge_draws=_ABSENT, knife_edge_draws_max=_ABSENT, **ctx):
    """one row per (Stokes, tier, target), each verdict consistent with its numbers; the Monte Carlo columns exactly when the
    run flagged knife-edges (knife_edge_se: the run's KNIFE_EDGE_SE; absent - an older record - either column set)"""
    rows, cols = _rows(path)
    mc = tuple(cols) == RESULT_COLUMNS + MC_RESULT_COLUMNS
    given = knife_edge_se is not _ABSENT
    if not (mc or tuple(cols) == RESULT_COLUMNS) or (given and mc != (knife_edge_se is not None)):
        return False, len(rows), 'columns %s != %s%s' % (cols, list(RESULT_COLUMNS), ' + %s' % list(MC_RESULT_COLUMNS)
                                                          if given and knife_edge_se is not None else '')
    k = knife_edge_se if given and knife_edge_se is not None else _cfg().forced_wide_test.KNIFE_EDGE_SE
    probs, seen = [], set()
    for r in rows:
        k_ = (r['stokes'], r['tier'], r['identity'])
        if k_ in seen or r['stokes'] not in ('I', 'V') or r['tier'] not in ('10s', '1min'):
            probs.append('%s listed twice or Stokes / tier unknown' % (k_,))
        seen.add(k_)
        probs += ['%s: %s' % (k_, p) for p in verdict_problems(r)]
        if mc:
            probs += ['%s: %s' % (k_, p) for p in knife_edge_problems(r, k if k is not None else 2.0)]
            K = knife_edge_draws if knife_edge_draws is not _ABSENT else _cfg().forced_wide_test.KNIFE_EDGE_DRAWS
            KM = knife_edge_draws_max if knife_edge_draws_max is not _ABSENT else _cfg().forced_wide_test.KNIFE_EDGE_DRAWS_MAX
            probs += ['%s: %s' % (k_, p) for p in draws_problems(r, K, KM)]
    return (not probs and len(rows) > 0), len(rows), ('%d rows' % len(rows) if not probs else '; '.join(probs[:5]))


def _num(x):
    return float(x) if x not in ('', None) else float('nan')


@content_test('forced_wide_candidates')
def candidates_consistent(path, results=None, **ctx):
    t = _cfg().forced_wide_test
    rows, cols = _rows(path)
    probs = []
    keys = None
    if results and os.path.exists(results):
        keys = {(r['stokes'], r['tier'], r['identity']) for r in _rows(results)[0]}
    tol = 0.006
    for r in rows:
        k = (r['stokes'], r['tier'], r['identity'], r['candidate'])
        if keys is not None and k[:3] not in keys:
            probs.append('%s: no results row' % (k[:3],))
        for f in ('shown', 'explains', 'excess'):
            if r[f] not in ('True', 'False'):
                probs.append('%s: %s %r' % (k, f, r[f]))
        p, ratio = _num(r['p_emp']), _num(r['ratio'])
        if p == p and not (0.0 < p <= 1.0):
            probs.append('%s: p_emp %s' % (k, r['p_emp']))
        if r['centering'] == 'BACKGROUND SOURCE' and r['explains'] != 'True':
            probs.append('%s: centering BACKGROUND SOURCE but explains False' % (k,))
        n_obs = int(r['n_obs']) if r['n_obs'] not in ('', None) else -1
        if r['centering'] != 'BACKGROUND SOURCE' and p == p and n_obs >= t.MIN_NOBS:
            clear_yes = p >= t.P_CONSISTENT + tol and ratio <= t.RATIO_CONSISTENT - tol
            clear_no = p < t.P_CONSISTENT - tol or ratio > t.RATIO_CONSISTENT + tol
            if (clear_yes and r['explains'] != 'True') or (clear_no and r['explains'] != 'False'):
                probs.append('%s: explains %s with p %s ratio %s' % (k, r['explains'], r['p_emp'], r['ratio']))
        if r['excess'] == 'True' and r['bands_far'] and not (p == p and p <= t.P_EXCESS + tol and ratio >= t.R_EXCESS - tol):
            probs.append('%s: excess True with p %s ratio %s' % (k, r['p_emp'], r['ratio']))
        if r['excess'] == 'True' and r['bands_near'] and r['centering'] != 'NOT THE BACKGROUND SOURCE':
            probs.append('%s: excess True with the centering %s' % (k, r['centering']))
        if r['shown'] == 'True' and r['centering'] != 'BACKGROUND SOURCE' and r['explains'] != 'True':
            probs.append('%s: shown without explains' % (k,))
        for f in ('p_emp_se', 'ratio_se', 'n_ctl_se', 'centroid_to_candidate_se'):
            if f in r and r[f] not in ('', None) and not float(r[f]) >= 0:
                probs.append('%s: %s %s' % (k, f, r[f]))
        if 'p_emp_se' in r and (p == p) != (r['p_emp_se'] not in ('', None)):
            probs.append('%s: p_emp %r with the error %r' % (k, r['p_emp'], r['p_emp_se']))
    return not probs, len(rows), ('%d candidate rows' % len(rows) if not probs else '; '.join(probs[:5]))


@content_test('forced_wide_published')
def published_matches_results(path, results=None, jobs=None, stokes=None, tier=None, **ctx):
    """the run's file = the results rows of its (Stokes, cadence), each named source at the stated distance from the target"""
    from exopipe.phase3.lib.forced import wide_jobs as WJ
    rows, cols = _rows(path)
    probs = []
    if cols[-1:] != ['named_radec'] or tuple(cols[:-1]) not in (RESULT_COLUMNS, RESULT_COLUMNS + MC_RESULT_COLUMNS):
        return False, len(rows), 'columns %s' % cols
    if results:
        want = [r for r in _rows(results)[0] if r['stokes'] == stokes and r['tier'] == tier]
        got = [{k: v for k, v in r.items() if k != 'named_radec'} for r in rows]
        if got != want:
            return False, len(rows), '%d rows != the %d results rows of %s %s' % (len(got), len(want), stokes, tier)
    pos = {}
    if jobs:
        pos = {(j['stokes'], j['identity']): j['T'] for j in WJ.all_jobs(WJ.load_jobs(jobs))}
    fa = WJ.FA() if pos else None
    for r in rows:
        names = r['named'].split('; ') if r['named'] else []
        radec = r['named_radec'].split(';') if r['named_radec'] else []
        if len(names) != len(radec):
            probs.append('%s: %d named, %d positions' % (r['identity'], len(names), len(radec)))
            continue
        for nm, rd in zip(names, radec):
            a, d = (float(x) for x in rd.split(','))
            if not (0 <= a < 360 and -90 <= d <= 90):
                probs.append('%s: position %s' % (r['identity'], rd))
            T = pos.get((r['stokes'], r['identity']))
            if T is not None:
                stated = float(nm.rsplit(' at ', 1)[1].rstrip("'"))
                sep = float(fa.sep_arcsec(a, d, T[0], T[1])) / 60.0
                if abs(sep - stated) > 0.06 + 1e-4 * stated:
                    probs.append("%s: %s at %.2f' from the target, the name says %.1f'" % (r['identity'], nm, sep, stated))
    return not probs, len(rows), ('%d rows' % len(rows) if not probs else '; '.join(probs[:5]))


# ---------------------------------------------------------------------------------------------------------------------
# the night-activity tables
# ---------------------------------------------------------------------------------------------------------------------

def _med_text(v):
    """the text csv writes for statistics.median's value (an int, or a float for an even count)"""
    return str(v)


@content_test('night_activity_bands')
def bands_recomputed(path, n_detections=None, **ctx):
    from exopipe.phase3.lib.forced.night_activity import BAND_COLUMNS
    fmt = _cfg().night_activity.RATIO_FORMAT
    rows, cols = _rows(path)
    if tuple(cols) != BAND_COLUMNS:
        return False, len(rows), 'columns %s != %s' % (cols, list(BAND_COLUMNS))
    by = {}
    for r in rows:
        by.setdefault((r['lst'], r['band']), []).append(r)
    probs = []
    total = 0
    for (l_, b), rr in by.items():
        nights = [r['night'] for r in rr]
        if len(set(nights)) != len(nights):
            probs.append('%s %s: a night twice' % (l_, b))
        for r in rr:
            n = int(r['n_det'])
            total += n
            if n <= 0:
                probs.append('%s %s %s: n_det %d (a band row is a searched night: >= 1)' % (l_, r['night'], b, n))
            oth = [int(x['n_det']) for x in rr if x is not r]
            med = statistics.median(oth) if oth else float('nan')
            ratio = n / med if oth and med > 0 else float('nan')
            if _med_text(med) != r['median_other'] or fmt % ratio != r['ratio']:
                probs.append('%s %s %s: median %s ratio %s, recomputed %s %s' % (l_, r['night'], b, r['median_other'], r['ratio'],
                                                                                 _med_text(med), fmt % ratio))
    if n_detections is not None and total != int(n_detections):
        probs.append('%d detections counted, %d read' % (total, n_detections))
    return (not probs and len(rows) > 0), len(rows), ('%d band rows, %d detections' % (len(rows), total) if not probs
                                                      else '; '.join(probs[:5]))


@content_test('night_activity_nights')
def nights_recomputed(path, bands=None, **ctx):
    from exopipe.phase3.lib.forced.night_activity import NIGHT_COLUMNS
    fmt = _cfg().night_activity.RATIO_FORMAT
    rows, cols = _rows(path)
    if tuple(cols) != NIGHT_COLUMNS:
        return False, len(rows), 'columns %s != %s' % (cols, list(NIGHT_COLUMNS))
    bands = bands or os.path.join(os.path.dirname(os.path.abspath(path)), _cfg().night_activity.FILES[0])   # alone: beside it
    if not os.path.exists(bands):
        return False, len(rows), 'the bands table %s (ctx bands) is needed to recompute the nights' % bands
    brow = _rows(bands)[0]
    per = {}
    for r in brow:
        n = int(r['n_det'])
        med = float(r['median_other'])
        ratio = n / med if med == med and med > 0 else float('nan')
        per.setdefault((r['lst'], r['night']), []).append((n, ratio))
    act = {}
    for k, v in per.items():
        rr = [x[1] for x in v if x[1] == x[1]]
        act[k] = (statistics.median(rr) if rr else float('nan'), len(rr), sum(x[0] for x in v))
    want = []
    for l_ in sorted({k[0] for k in per}):
        nights = sorted(k[1] for k in per if k[0] == l_)
        ok = [n for n in nights if act[(l_, n)][0] == act[(l_, n)][0]]
        for n in nights:
            a, nb, nd = act[(l_, n)]
            want.append(dict(lst=l_, night=n, n_bands_searched=str(nb), n_det=str(nd), activity=fmt % a,
                             rank=str(1 + sum(1 for m in ok if act[(l_, m)][0] > a)) if a == a else '', n_nights=str(len(ok))))
    probs = []
    if len(want) != len(rows):
        probs.append('%d night rows, %d (LST, night) in the bands table' % (len(rows), len(want)))
    for r, w in zip(rows, want):
        if r != w:
            probs.append('%s %s: %s, recomputed %s' % (r['lst'], r['night'], {k: r[k] for k in r if r[k] != w.get(k)},
                                                       {k: w[k] for k in w if r.get(k) != w[k]}))
    return (not probs and len(rows) > 0), len(rows), ('%d night rows' % len(rows) if not probs else '; '.join(probs[:5]))
