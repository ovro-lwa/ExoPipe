"""Content tests of the snapshot batch chain (catalog --tier snapshot), registered with core/validate.

Classes: snapshot_run (a chain run directory: R1-R5 and R7 of ops/snapshot_rechain_content_test.py recomputed from the run's
own files, plus no refusal marker, the per-Stokes views agreeing with the union catalogues, Card 2 per frame (card2) and the
restore-only 3-D gate (restore) read back from the per-detection record), restore_wave_diff (the restore-only wave's per-batch
test, restore_wave_served_diff.json + .csv, read back against both runs' records), tie_cache (one cell's tie
cache file), snapshot_tie_summary (the chain_summary.json a per-cell --stop-after-tie writes). The run content test's
operating points are config (catalog_snapshot.TEST_GATE3D_PCT, TEST_MEMBER_ORDER, GHOST_DATES). R6 (missing panels) is the
pages step's class snapshot_page_run. Every test reads content and returns (passed, value, detail).
"""
import collections
import csv
import glob
import json
import math
import os

from exopipe.phase3.core.validate import content_test

csv.field_size_limit(10 ** 9)


def _f(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def run_checks(run):
    """R1-R5, R7 and the extras on a run directory: [(name, passed, detail)] (the order of the production test)."""
    c = _cfg().catalog_snapshot
    S = json.load(open(os.path.join(run, 'chain_summary.json')))
    out = []
    pf = S.get('prefit') or {}
    r1 = (pf.get('n_without_fit'), S.get('n_served_with_unfitted_frames'), S.get('n_unfitted_frames_under_served'))
    out.append(('R1', r1 == (0, 0, 0), 'unfitted at the fit step %s of %s; served with unfitted frames %s; unfitted frames '
                'under served %s' % (r1[0], pf.get('n_at_per_frame_fit_step'), r1[1], r1[2])))
    dfiles = sorted(glob.glob(os.path.join(run, 'candidates', '*', '*', 'detections.csv')))
    n_rows = n_bad = n_nocol = 0
    served_pos = {}                     # cand_id -> [(ra, dec)] of every served row (R3)
    for p in dfiles:
        with open(p, newline='') as f:
            rd = csv.DictReader(f)
            cols = rd.fieldnames or []
            if 'pos_err_stat_arcsec' not in cols or 'pos_err_arcsec' not in cols:
                n_nocol += 1
                continue
            for r in rd:
                n_rows += 1
                try:
                    served_pos.setdefault(r.get('cand_id', ''), []).append((float(r['ra']), float(r['dec'])))
                except (KeyError, TypeError, ValueError):
                    served_pos.setdefault(r.get('cand_id', ''), []).append((float('nan'), float('nan')))
                a, b = _f(r.get('pos_err_stat_arcsec')), _f(r.get('pos_err_arcsec'))
                if a is None or b is None or a <= 0 or b <= 0:
                    n_bad += 1
    out.append(('R2', bool(dfiles) and n_nocol == 0 and n_bad == 0, '%d detections.csv, %d rows: %d files without the own-'
                'error columns, %d rows without a finite positive error' % (len(dfiles), n_rows, n_nocol, n_bad)))
    gp = os.path.join(run, 'ghost_removed.csv')
    grows = list(csv.DictReader(open(gp))) if os.path.exists(gp) else None
    n_g = (S.get('domain_dropped') or {}).get('ghost')
    has_g = any(x.split('_', 1)[1] in tuple(c.GHOST_DATES) for x in S.get('cells', []))
    # A LEAK IS THE SAME DETECTION SERVED, not the same cand_id (exopipe_ops 56810b9: a cand_id names a FRAME, so other
    # sources of a ghost's frame read as served ghosts): a served row of the removed cand_id within TEST_GHOST_LEAK_DEG of its
    # position; an unreadable served position counts as a leak (the production expression, kept: min(1.0, nan) is 1.0)
    def _sep(ra1, de1, ra2, de2):
        x = (math.sin(math.radians(de1)) * math.sin(math.radians(de2))
             + math.cos(math.radians(de1)) * math.cos(math.radians(de2)) * math.cos(math.radians(ra1 - ra2)))
        return math.degrees(math.acos(max(-1.0, min(1.0, x))))
    leak = sorted({g['cand_id'] for g in (grows or []) for (ra_s, de_s) in served_pos.get(g['cand_id'], ())
                   if _sep(float(g['ra_deg']), float(g['dec_deg']), ra_s, de_s) < float(c.TEST_GHOST_LEAK_DEG)})
    out.append(('R3', grows is not None and len(grows) == n_g and (len(grows) > 0 or not has_g) and not leak,
                'ghost_removed.csv rows %s, summary %s, ghost-date cells %s, removed detections served %d%s' % (
                    None if grows is None else len(grows), n_g, has_g, len(leak), (' e.g. ' + leak[0]) if leak else '')))
    tc = S.get('tie_cache') or {}
    out.append(('R4', tc.get('tied_now') == 0, 'tie cache: %s from the cache, %s tied in this run' % (
        tc.get('from_cache'), tc.get('tied_now'))))
    dd = S.get('domain_dropped') or {}
    r5 = (S.get('gate3d_match_percentile'), S.get('member_order'), (S.get('ghost_gate') or {}).get('dates'),
          any('moon' in k.lower() for k in dd))
    out.append(('R5', r5 == (float(c.TEST_GATE3D_PCT), c.TEST_MEMBER_ORDER, list(c.GHOST_DATES), False),
                '3-D gate percentile %s; member order %s; ghost dates %s; Moon rule %s' % r5))
    n_ra = n_out = 0
    for p in sorted(glob.glob(os.path.join(run, 'source_catalogs*', '*.csv'))) + dfiles:
        with open(p, newline='') as f:
            rd = csv.DictReader(f)
            cols = [x for x in (rd.fieldnames or []) if x == 'ra' or x.startswith('ra_') or x.endswith('_ra')]
            for r in rd if cols else ():
                for x in cols:
                    v = (r.get(x) or '').strip()
                    if not v:
                        continue
                    n_ra += 1
                    fv = _f(v)
                    if fv is None or not (0.0 <= fv < 360.0):
                        n_out += 1
    out.append(('R7', n_ra > 0 and n_out == 0, '%d RA values, %d outside [0, 360)' % (n_ra, n_out)))
    marks = [m for m in ('CHECKSOURCE_MISSING.txt', 'FIT_INCOMPLETE.txt') if os.path.exists(os.path.join(run, m))]
    out.append(('markers', not marks, 'refusal markers present: %s' % (marks or 'none')))
    bad_views = []
    union = {}
    for p in glob.glob(os.path.join(run, 'source_catalogs', '*_sources_survivors.csv')):
        for r in csv.DictReader(open(p)):
            union.setdefault(r.get('stokes', ''), set()).add(r['source_id'])
    # PER-TIER PAGES (clv15 82792d5, 2026-10-18, under Gregg 2026-10-17 'four independent tiers'): an identity the chain serves
    # that QUALIFIES IN NO TIER (served_tiers.csv: chain_outcome served, qualifies 0 in every tier row - its seed is in the other
    # tier) is dropped from the per-Stokes views and the tier pages, recorded there; it is the only union row a view may lack
    notier = set()
    stp = os.path.join(run, 'served_tiers.csv')
    if os.path.exists(stp):
        q = {}
        for r in csv.DictReader(open(stp, newline='')):
            if r.get('chain_outcome') == 'served':
                q.setdefault(r.get('identity_jname', ''), []).append(r.get('qualifies'))
        notier = {j for j, v in q.items() if v and all(x == '0' for x in v)}
    n_notier = 0
    for st in ('V', 'I'):
        view = set()
        for p in glob.glob(os.path.join(run, 'source_catalogs_%s' % st, '*_sources_survivors.csv')):
            # a view row is one (identity, band): source_id '<identity>_<band>', identity_id = the union row's source_id
            view |= {(r.get('identity_id') or '').strip() or r['source_id'] for r in csv.DictReader(open(p))}
        # an identity with detections in both Stokes has ONE union row (the Stokes of its best detection) and rows in both
        # views: so every union row of Stokes S is in the S view, and every view identity is a union source (measured on
        # Run_SNAPB1R: 139 of 396 V-view identities have an I union row)
        allu = set().union(*union.values()) if union else set()
        miss, orphan = union.get(st, set()) - view, view - allu
        n_notier += len(miss & notier)
        miss -= notier
        if miss or orphan:
            bad_views.append('%s: %d union rows not in the view, %d view identities not in the union' % (st, len(miss),
                                                                                                        len(orphan)))
    out.append(('views', not bad_views, 'per-Stokes views cover the union rows of their Stokes and nothing else: %s%s' % (
        '; '.join(bad_views) or 'yes', '; %d union rows of identities that qualify in no tier (served_tiers.csv), not viewed as '
        'the per-tier rule drops them' % n_notier if n_notier else '')))
    out.append(card2_check(run, S))
    out.append(restore_check(run, S, dfiles))
    return out


def _record_cols(path, want):
    """Rows of the per-detection record (detection_record.csv.gz) as dicts of the wanted columns ('' for a column it lacks),
    and whether it has each: (iterator, set of present columns)."""
    import gzip
    f = gzip.open(path, 'rt', newline='')
    rd = csv.reader(f)
    h = next(rd)
    ix = [(c, h.index(c)) for c in want if c in h]
    miss = [c for c in want if c not in h]

    def gen():
        try:
            for r in rd:
                d = {c: r[i] for c, i in ix}
                for c in miss:
                    d[c] = ''
                yield d
        finally:
            f.close()
    return gen(), {c for c, _ in ix}


def card2_check(run, S):
    """CARD 2 PER FRAME in the chain (clv15 1279bd4; catalog_snapshot.CARD2): card2_removed.csv holds the summary's 'removed'
    rows, each a Stokes V detection of a Card 2 tier (its cand_id) with |V| / (I Master pixel x scale) <= the ratio; the summary's
    counts add up (evaluated + not evaluated = the V rows of the tiers); the record's rows first removed by 'card2' are the same
    detections (cand_id multiset) and none is served. A run without the summary key (a chain before 1279bd4) holds no
    card2_removed.csv."""
    c = _cfg().catalog_snapshot.CARD2
    sm = S.get('card2')
    p = os.path.join(run, 'card2_removed.csv')
    if sm is None:
        ok = not os.path.exists(p)
        return ('card2', ok, 'no Card 2 in this run (a chain before clv15 1279bd4)' if ok else
                'card2_removed.csv without the chain summary\'s card2 counts')
    bad = []
    if not os.path.exists(p):
        return ('card2', False, 'the chain summary has card2 counts and the run no card2_removed.csv')
    rows = list(csv.DictReader(open(p, newline='')))
    if float(sm.get('ratio', -1)) != float(c['ratio']) or list(sm.get('tiers') or []) != list(c['tiers']):
        bad.append('summary ratio %s tiers %s, config %s %s' % (sm.get('ratio'), sm.get('tiers'), c['ratio'], list(c['tiers'])))
    n_ev, n_ne, n_rm, n_v = (int(sm.get(k, -1)) for k in ('evaluated', 'not_evaluated', 'removed', 'v_10s_rows'))
    if n_ev + n_ne != n_v or n_rm > n_ev or n_rm != len(rows):
        bad.append('counts: evaluated %d + not evaluated %d vs V rows %d; removed %d, file rows %d' % (n_ev, n_ne, n_v, n_rm,
                                                                                                    len(rows)))
    off = []
    for r in rows:
        parts = (r.get('cand_id') or '').split('_')
        st, tier = (parts[3], parts[4]) if len(parts) > 4 else ('', '')
        pk, ih, vi = _f(r.get('peak_jyb')), _f(r.get('i_here_jy')), _f(r.get('v_over_i_here'))
        pix, sc = _f(r.get('i_master_pix_jyb')), _f(r.get('scale'))
        if st != 'V' or tier not in c['tiers'] or vi is None or vi > float(c['ratio']) + 5e-6 or pk is None or not ih or \
                ih <= 0 or abs(abs(pk) / ih - vi) > 2e-3 * max(vi, 1e-3) + 5e-6 or pix is None or sc is None or \
                abs(pix * sc - ih) > 2e-3 * abs(ih):
            off.append(r.get('cand_id'))
    if off:
        bad.append('%d rows off the rule (not V of a Card 2 tier, |V|/I above the ratio, or I x scale inconsistent; e.g. %s)'
                   % (len(off), off[:2]))
    rp = os.path.join(run, 'detection_record.csv.gz')
    nrec, rec_ok = None, False
    if os.path.exists(rp):
        it, have = _record_cols(rp, ['cand_id', 'first_removed_by', 'served'])
        rec_c2, served_c2 = {}, 0
        for d in it:
            if d['first_removed_by'] == 'card2':
                rec_c2[d['cand_id']] = rec_c2.get(d['cand_id'], 0) + 1
                served_c2 += d['served'] == '1'
        csv_c2 = {}
        for r in rows:
            csv_c2[r['cand_id']] = csv_c2.get(r['cand_id'], 0) + 1
        nrec = sum(rec_c2.values())
        rec_ok = rec_c2 == csv_c2 and not served_c2
        if not rec_ok:
            bad.append('record: %d rows first removed by card2 (%d served), card2_removed.csv %d (%d cand_ids differ)'
                       % (nrec, served_c2, len(rows), len(set(rec_c2.items()) ^ set(csv_c2.items()))))
    return ('card2', not bad, 'Card 2 per frame: %d removed of %d evaluated (%d V rows of %s, %d not evaluated)%s%s' % (
        n_rm, n_ev, n_v, '/'.join(c['tiers']), n_ne,
        '; no per-detection record' if nrec is None else ('; the record agrees (%d rows)' % nrec if rec_ok else ''),
        ('; ' + '; '.join(bad)) if bad else ''))


def _k5(cand, ra, dec):
    try:
        return (cand, '%.5f' % (float(ra) % 360.0), '%.5f' % float(dec))
    except (TypeError, ValueError):
        return (cand, ra, dec)


def restore_check(run, S, dfiles):
    """THE RESTORE-ONLY 3-D GATE (clv15 b8d1989 + bf7dce8; catalog_snapshot.RESTORED_BY), read back from the run's own files as
    the wave's per-batch test does (exopipe_ops snapshot_restore_wave_diff.py, its checks within the new run): in a run with the
    summary's gate3d_restore_only, the record's rows tagged restored_by are RESTORED_BY exactly where gate_3d_installed is
    'remove' and gate_3d_restore is not, as many as the summary's restored; the summary adds up (removed + restored = the installed
    pass's removals; restore pass only = the restore pass's removals - removed); every served detections.csv has the column and
    the chain's tagged rows (candidates/<band>/) are the record's served tagged rows. A run without the key carries no tag."""
    c = _cfg().catalog_snapshot
    ro = S.get('gate3d_restore_only')
    rp = os.path.join(run, 'detection_record.csv.gz')
    if not os.path.exists(rp):
        return ('restore', not ro, 'no per-detection record' + (' - a restore-only run cannot be read back' if ro else ''))
    it, have = _record_cols(rp, ['cand_id', 'ra_peak', 'dec_peak', 'served', 'gate_3d_installed', 'gate_3d_restore',
                                 'restored_by'])
    tagged, pair_bad, tag_bad, served_tagged, n_rows = 0, 0, 0, set(), 0
    for d in it:
        n_rows += 1
        pair = d['gate_3d_installed'] == 'remove' and d['gate_3d_restore'] != 'remove'
        if d['restored_by']:
            tagged += 1
            tag_bad += d['restored_by'] != c.RESTORED_BY
            pair_bad += not pair
            if d['served'] == '1':
                served_tagged.add(_k5(d['cand_id'], d['ra_peak'], d['dec_peak']))
        elif pair:
            pair_bad += 1
    if not ro:
        ok = tagged == 0 and pair_bad == 0
        return ('restore', ok, 'no restore-only 3-D gate in this run' + ('' if ok else ': %d rows tagged restored_by, %d with '
                                                                         'the installed-remove / restore-keep pair' % (tagged, pair_bad)))
    bad = []
    try:
        rem, rest = int(ro['removed']), int(ro['restored'])
        if rem + rest != int(ro['removed_installed_pass']) or int(ro['restore_pass_only']) != int(ro['removed_restore_pass']) - rem:
            bad.append('summary does not add up: %s' % dict(ro))
    except (KeyError, TypeError, ValueError) as e:
        return ('restore', False, 'gate3d_restore_only unreadable: %s %s' % (type(e).__name__, e))
    if not {'restored_by', 'gate_3d_installed', 'gate_3d_restore'} <= have:
        bad.append('the record lacks the restore-only columns (version < 5)')
    if tagged != rest:
        bad.append('record restored rows %d, chain summary restored %d' % (tagged, rest))
    if pair_bad or tag_bad:
        bad.append('%d rows whose tag disagrees with gate_3d_installed / gate_3d_restore, %d with a tag other than %r'
                   % (pair_bad, tag_bad, c.RESTORED_BY))
    nocol, tagged_csv = [], set()
    for p in dfiles:
        sub = os.path.basename(os.path.dirname(os.path.dirname(p)))
        with open(p, newline='') as f:
            rd = csv.DictReader(f)
            if 'restored_by' not in (rd.fieldnames or []):
                nocol.append(p)
                continue
            if sub in ('Red', 'Green', 'Blue', 'Full'):
                for r in rd:
                    if r['restored_by']:
                        tagged_csv.add(_k5(r['cand_id'], r.get('ra_peak'), r.get('dec_peak')))
    if nocol:
        bad.append('%d served detections.csv without the restored_by column (e.g. %s)' % (len(nocol), nocol[:1]))
    if tagged_csv != served_tagged:
        bad.append('tagged detections.csv rows %d, record served tagged rows %d (%d only in the files, %d only in the record)'
                   % (len(tagged_csv), len(served_tagged), len(tagged_csv - served_tagged), len(served_tagged - tagged_csv)))
    return ('restore', not bad, 'restore-only 3-D gate (%s): installed pass removes %s, restore pass %s; removed %d, restored %d '
            '(%d served); %d record rows%s' % (os.path.basename(str(ro.get('table'))), ro.get('removed_installed_pass'),
                                               ro.get('removed_restore_pass'), rem, rest, len(served_tagged), n_rows,
                                               ('; ' + '; '.join(bad)) if bad else ''))


@content_test('snapshot_run')
def snapshot_run_content(path, **ctx):
    if not os.path.exists(os.path.join(path, 'chain_summary.json')):
        return False, None, 'no chain_summary.json'
    res = run_checks(path)
    bad = [n for n, ok, _ in res if not ok]
    return not bad, len(bad), '; '.join('%s %s %s' % (n, 'PASS' if ok else 'FAIL', d) for n, ok, d in res)[:1500]


@content_test('tie_cache')
def tie_cache_rows(path, errmodel=None, **ctx):
    """tie_<cell>.csv: the cache columns, every row of one cell, finite tied positions; rows of the installed model (ctx
    errmodel, from the preflight record) present; repeated keys counted (the module keeps the last)."""
    with open(path, newline='') as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames or []
        rows = list(rd)
    need = ('cell', 'sb', 'st', 'tier', 'ts', 'ra_pre', 'dec_pre', 'tie_status', 'ra', 'dec', 'errmodel')
    miss = [x for x in need if x not in cols]
    if miss:
        return False, len(rows), 'missing columns %s' % miss
    cell = os.path.basename(path)[4:-4]
    other = sum(1 for r in rows if r['cell'] != cell)
    keys = [(r['sb'], r['st'], r['tier'], r['ts'], r['ra_pre'], r['dec_pre'], r['errmodel']) for r in rows]
    dup = len(keys) - len(set(keys))
    nonfin = sum(1 for r in rows if _f(r['ra']) is None or _f(r['dec']) is None)
    models = {}
    for r in rows:
        models[r['errmodel']] = models.get(r['errmodel'], 0) + 1
    cur = '' if errmodel is None else ', %d rows of the installed model %s' % (models.get(errmodel, 0), errmodel)
    # a repeated key is a re-tie appended later: phase3_snapshot_tiecache.apply_cached keeps the LAST row (counted only)
    ok = not (other or nonfin) and (errmodel is None or models.get(errmodel, 0) > 0)
    return ok, len(rows), '%s: %d rows %s; %d of another cell, %d repeated keys (last wins), %d non-finite positions%s' % (
        cell, len(rows), models, other, dup, nonfin, cur)


@content_test('snapshot_tie_summary')
def tie_summary(path, **ctx):
    """chain_summary.json of a --stop-after-tie run: every detection at the per-frame fit step fitted (prefit.n_without_fit
    = 0, the precondition of the tie pass) and a tie_cache record with integer counts."""
    S = json.load(open(path))
    pf, tc = S.get('prefit') or {}, S.get('tie_cache') or {}
    ok = pf.get('n_without_fit') == 0 and isinstance(tc.get('from_cache'), int) and isinstance(tc.get('tied_now'), int)
    return ok, tc.get('tied_now'), 'prefit without fit %s of %s; tie cache %s from cache, %s tied now' % (
        pf.get('n_without_fit'), pf.get('n_at_per_frame_fit_step'), tc.get('from_cache'), tc.get('tied_now'))


@content_test('restore_wave_diff')
def restore_wave_rows(path, **ctx):
    """restore_wave_served_diff.json and the .csv beside it (lib/catalog/restore_wave, port of exopipe_ops
    snapshot_restore_wave_diff.py; the file G27 reads) read back against BOTH runs' records, on its own reading (not the port's):
    the csv's columns; one row per previously served detection not served now (the old record's served rows less the new
    record's served ones), none twice; each row's cause from the new record - absent (not in it), card2 / 3d-gate / not-served /
    other-earlier by the removing stage, and for a stage after the 3-D gate: secondary only with restored detections listed,
    each tagged in the new record and (for a grouping column) in the row's group; other-later / secondary-pool with none; the
    json's counts = the rows' and the records' (previously served, served again, restored, restored served); its status PASS
    exactly when no detection is absent and the restored rows agree with their pair and the chain summary."""
    from exopipe.phase3.lib.catalog import restore_wave as RW
    try:
        j = json.load(open(path))
        old, new = j['old'], j['new']
        counts = j['counts']
    except (OSError, ValueError, KeyError) as e:
        return False, None, f'unreadable wave test {path}: {type(e).__name__}: {e}'
    cp = os.path.join(os.path.dirname(os.path.abspath(path)), RW.CSV_NAME)
    if not os.path.exists(cp):
        return False, None, f'no {RW.CSV_NAME} beside {path}'
    with open(cp, newline='') as f:
        rd = csv.DictReader(f)
        cols, rows = rd.fieldnames or [], list(rd)
    msgs = []
    if cols != RW.COLS:
        msgs.append(f'columns {cols} are not the wave test\'s')
    for p in (os.path.join(old, 'detection_record.csv.gz'), os.path.join(new, 'detection_record.csv.gz')):
        if not os.path.exists(p):
            return False, None, f'{p} is not readable: the wave test cannot be read back'
    it, _h = _record_cols(os.path.join(old, 'detection_record.csv.gz'), ['det_uid', 'served'])
    prev = {d['det_uid'] for d in it if d['served'] == '1'}
    want = ['det_uid', 'cand_id', 'ra_peak', 'dec_peak', 'served', 'first_removed_by', 'restored_by', 'gate_3d_installed',
            'gate_3d_restore'] + list(RW.FALLBACK)
    it, _h = _record_cols(os.path.join(new, 'detection_record.csv.gz'), want)
    now, n_tag, served_tag, pair_bad = {}, 0, 0, 0
    for d in it:
        now[d['det_uid']] = d
        pair = d['gate_3d_installed'] == 'remove' and d['gate_3d_restore'] != 'remove'
        if d['restored_by']:
            n_tag += 1
            served_tag += d['served'] == '1'
            pair_bad += (not pair) or d['restored_by'] != RW.RESTORED_BY
        elif pair:
            pair_bad += 1
    listed = [r['det_uid'] for r in rows]
    if len(set(listed)) != len(listed):
        msgs.append(f'{len(listed) - len(set(listed))} detections listed twice')
    want_listed = {u for u in prev if not (u in now and now[u]['served'] == '1')}
    if set(listed) != want_listed:
        msgs.append(f'rows {len(set(listed))}, previously served not served now {len(want_listed)} ({len(set(listed) - want_listed)} '
                    f'listed wrongly, {len(want_listed - set(listed))} missing)')
    bad_cause = []
    for r in rows:
        u, c = r['det_uid'], r['cause']
        d = now.get(u)
        st = d['first_removed_by'] if d else None
        if d is None:
            ok_ = c == 'absent'
        elif st == 'card2':
            ok_ = c == 'card2'
        elif st == 'bright-parent-3d':
            ok_ = c == '3d-gate'
        elif st in RW.POST3D:
            nr = int(r['n_restored_in_group'] or 0)
            rs = [x for x in (r['restored_in_group'] or '').split(';') if x]
            if c == 'secondary':
                gc = r['group_column']
                ok_ = nr > 0 and rs and all(now.get(x, {}).get('restored_by') == RW.RESTORED_BY for x in rs)
                if ok_ and gc in RW.FALLBACK:
                    ok_ = all(now[x].get(gc) == r['group_index'] for x in rs) and d.get(gc) == r['group_index']
            else:
                ok_ = c in ('other-later', 'secondary-pool') and nr == 0 and not rs
        elif st:
            ok_ = c == 'other-earlier'
        else:
            ok_ = c == 'not-served'
        if not ok_:
            bad_cause.append(f'{u} {c} (stage {st!r})')
    if bad_cause:
        msgs.append(f'{len(bad_cause)} rows whose cause the new record does not give, e.g. {bad_cause[:2]}')
    by = collections.Counter(r['cause'] for r in rows)
    for c in RW.CAUSES[1:]:
        if counts.get(c) != by.get(c, 0):
            msgs.append(f'counts {c} {counts.get(c)} != {by.get(c, 0)} rows')
    n_served = sum(1 for u in prev if u in now and now[u]['served'] == '1')
    for k, v in (('prev_served', len(prev)), ('served', n_served), ('restored', n_tag), ('restored_served', served_tag)):
        if counts.get(k) != v:
            msgs.append(f'counts {k} {counts.get(k)} != {v} read from the records')
    try:
        ro = json.load(open(os.path.join(new, 'chain_summary.json'))).get('gate3d_restore_only') or {}
    except (OSError, ValueError):
        ro = {}
    rest_ok = bool(ro) and int(ro.get('restored', -1)) == n_tag and not pair_bad
    expect_pass = not by.get('absent') and rest_ok
    if (j.get('status') == 'PASS') and not expect_pass:
        msgs.append(f'status PASS while {by.get("absent", 0)} detections are absent / the restored rows disagree ({n_tag} tagged, '
                    f'summary {ro.get("restored")}, {pair_bad} off their pair)')
    if j.get('status') != 'PASS' and expect_pass and not str(j.get('status', '')).startswith('FAIL'):
        msgs.append(f'status {j.get("status")!r} is neither PASS nor FAIL')
    return (not msgs and j.get('status') == 'PASS'), len(rows), (
        '; '.join(msgs) if msgs else
        f'{j.get("tag_new")} against {j.get("tag_old")}: {len(prev)} previously served, {n_served} served again, {len(rows)} listed '
        f'({dict(sorted(by.items()))}); restored {n_tag} ({served_tag} served); status {str(j.get("status"))[:200]}')
