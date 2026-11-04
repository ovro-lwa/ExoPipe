"""THE NIGHT-ACTIVITY TABLE of family N (stage activity): EACH NIGHT'S IONOSPHERIC ACTIVITY at an LST, for the astrometry figure
v2's variability line (astrometry_assoc.night_act, every tier's figure). Port of phase3/refactor_analysis/review_items_20261031/
figure_v2/night_activity_build.py (2026-10-31; Gregg 2026-10-31: "Perhaps it is just necessary to see if the detections are
clustered by night? ... the HWJ forced photometry would cluster by cell/night for scintillation right?"); values in config
(night_activity).
Input: the snapshot 10-s Stokes I search's detections of every LISTED source of the served view (the flood rule's input,
flood_nights_20261031/flood_collect.py), de-duplicated by (frame id, position to DEDUP_CHARS characters).
Per (LST, night, band): the count over the median count of that band's OTHER SEARCHED nights - a band with no detection at all
that night is taken as NOT SEARCHED there (excluded or omitted; the 13h 36 MHz band of 2025-04-11 has none while that night's
other bands hold 19,944), never as quiet. Per (LST, night): the night's activity = the median of its searched bands' ratios
(ionospheric scintillation is broadband), and its rank among the LST's nights (1 = most active).
Writes the bands table (lst, night, band, n_det, median_other, ratio) and the nights table (lst, night, n_bands_searched, n_det,
activity, rank, n_nights) - the names the figure reads (FILES). Every set the script iterated is iterated in sorted order: the
bands table's rows of one (LST, band) are in night order (the script's came in string-hash order, new in every process); no
value depends on it (medians, sums, counts), so the nights table is the script's byte for byte."""
import collections
import csv
import os
import statistics

BAND_COLUMNS = ('lst', 'night', 'band', 'n_det', 'median_other', 'ratio')
NIGHT_COLUMNS = ('lst', 'night', 'n_bands_searched', 'n_det', 'activity', 'rank', 'n_nights')
INPUT_COLUMNS = ('stokes', 'tier', 'lst', 'night', 'band', 'cand_id', 'ra', 'dec')


def ncfg():
    from exopipe.phase3.config import CFG
    return CFG.night_activity


def read_detections(path, c=None):
    """(counts {(lst, night, band): n}, nights {lst: {night}}, n rows, n unique, wrong [(row number, stokes, tier)]): the input
    de-duplicated by (cand_id, ra[:DEDUP_CHARS], dec[:DEDUP_CHARS]) (night_activity_build.py:12-20); rows of another Stokes or
    tier are listed (the stage refuses them, N-INPUT)."""
    c = c or ncfg()
    k_ = c.DEDUP_CHARS
    cnt = collections.Counter()
    nights = collections.defaultdict(set)
    seen = set()
    wrong = []
    n = 0
    with open(path, newline='') as f:
        rd = csv.DictReader(f)
        missing = [x for x in INPUT_COLUMNS if x not in (rd.fieldnames or [])]
        if missing:
            raise KeyError(f'{path}: columns {missing} missing (the flood rule input has {", ".join(INPUT_COLUMNS)})')
        for r in rd:
            n += 1
            if r['stokes'] != c.STOKES or r['tier'] != c.TIER:
                if len(wrong) < 20:
                    wrong.append((n, r['stokes'], r['tier']))
                continue
            k = (r['cand_id'], r['ra'][:k_], r['dec'][:k_])
            if k in seen:
                continue
            seen.add(k)
            cnt[(r['lst'], r['night'], r['band'])] += 1
            nights[r['lst']].add(r['night'])
    return cnt, nights, n, len(seen), wrong


def build(cnt, nights, c=None):
    """(band rows, night rows) of night_activity_build.py from the counts (its computation and its formats)."""
    c = c or ncfg()
    fmt = c.RATIO_FORMAT
    bands = collections.defaultdict(set)
    for (l_, n_, b) in cnt:
        bands[l_].add(b)
    rows, ratio = [], {}
    for l_ in sorted(nights):
        for b in sorted(bands[l_]):
            # nights this band was searched (any detection), in date order: the script iterated a set here, so its rows of one
            # (LST, band) came in the process's string-hash order (a new order every run); the values do not depend on it
            srch = [n_ for n_ in sorted(nights[l_]) if cnt.get((l_, n_, b), 0) > 0]
            for n_ in srch:
                oth = [cnt[(l_, m, b)] for m in srch if m != n_]
                med = statistics.median(oth) if oth else float('nan')
                r_ = cnt[(l_, n_, b)] / med if oth and med > 0 else float('nan')
                ratio[(l_, n_, b)] = r_
                rows.append(dict(lst=l_, night=n_, band=b, n_det=cnt[(l_, n_, b)], median_other=med, ratio=fmt % r_))
    nrows = []
    for l_ in sorted(nights):
        act = {}
        for n_ in sorted(nights[l_]):
            rr = [ratio[(l_, n_, b)] for b in sorted(bands[l_]) if (l_, n_, b) in ratio and ratio[(l_, n_, b)] == ratio[(l_, n_, b)]]
            act[n_] = (statistics.median(rr) if rr else float('nan'), len(rr), sum(cnt.get((l_, n_, b), 0) for b in bands[l_]))
        ok = [n_ for n_ in act if act[n_][0] == act[n_][0]]
        for n_ in sorted(nights[l_]):
            a, nb, nd = act[n_]
            nrows.append(dict(lst=l_, night=n_, n_bands_searched=nb, n_det=nd, activity=fmt % a,
                              rank=(1 + sum(1 for m in ok if act[m][0] > a)) if a == a else '', n_nights=len(ok)))
    return rows, nrows


def write_table(path, rows, columns):
    """csv.DictWriter as the script wrote (its header = the first row's keys = ``columns``), tmp + rename."""
    tmp = '%s.tmp.%d' % (path, os.getpid())
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(columns))
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)
