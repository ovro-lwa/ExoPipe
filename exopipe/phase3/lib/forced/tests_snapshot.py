"""Content tests of the snapshot forced event-panel classes (family C; spec forced.md 6). Each test reads the run's files
and recomputes with code of its own (no import of snapshot_forced_panels); forced_assoc additionally runs the in-place
association content test (snapshot_forced_assoc.py verify, read-only) through core.legacy.run.

Classes and the file each is registered on:
  forced_event_build   <RUN>/queue.txt                     the build tree against the evaluations it names
  forced_fits          <RUN>/forced_fits.csv               one fit record per event; astrometric rule; size/axis
  forced_ties          <RUN>/forced_ties.csv               one tie row per positioned fit; used-reference reconstruction
  forced_localization  <run>/source_catalogs/localizations.csv   one row per identity with astrometric fits; figures
  forced_assoc         <RUN>/forced_assoc_identity.csv     labels declared, AD Leo unlabelled; in-place verify A1-A6
  forced_event_panels  <run>/missing_panels.csv            every served event panelled with its light-curve strip;
                                                           radio position only
"""
import csv
import glob
import math
import os
import re
import struct

from exopipe.phase3.core.validate import content_test


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def _rows(p):
    if not os.path.exists(p):
        return []
    with open(p, newline='') as f:
        return list(csv.DictReader(f))


def _num(s):
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _detail(bad, n=5):
    return '; '.join(str(b) for b in bad[:n]) + (' ... (%d)' % len(bad) if len(bad) > n else '')


def sep_arcmin(ra1, dec1, ra2, dec2):
    r1, d1, r2, d2 = map(math.radians, (ra1, dec1, ra2, dec2))
    c = math.sin(d1) * math.sin(d2) + math.cos(d1) * math.cos(d2) * math.cos(r1 - r2)
    return math.degrees(math.acos(max(-1.0, min(1.0, c)))) * 60.0


_QHEAD = re.compile(r'^# forced photometry \(HWJ \+ 10 pc\), Stokes ([IV]), (10s|1min) tier only, .*; eval (\S+) \+ (\S+)\s*$')


def queue_header(run):
    """(Stokes, cadence, eval_sb, eval_wb, identities) from <run>/queue.txt (snapshot_forced_panels.py:698-700)."""
    with open(os.path.join(run, 'queue.txt')) as f:
        lines = f.read().splitlines()
    m = _QHEAD.match(lines[0]) if lines else None
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3), m.group(4), [l for l in lines[1:] if l.strip() and not l.startswith('#')]


def build_events(run):
    """Rows of the BUILD tree (candidates/<band>/rank*/detections.csv; the served tree candidates/<Stokes>/ excluded)."""
    out = []
    for p in sorted(glob.glob(os.path.join(run, 'candidates', '*', 'rank*', 'detections.csv'))):
        if os.path.basename(os.path.dirname(os.path.dirname(p))) in ('I', 'V'):
            continue
        out += _rows(p)
    return out


def has_position(fr):
    """A fit record carries a radio position: a Gaussian fit at >= 7 sigma ('astrometric') or an ok template fit
    (snapshot_forced_panels.has_position :712-716, rewritten)."""
    return bool(fr) and bool(fr.get('fit_ra_deg')) and (fr.get('astrometric') == '1' or (
        fr.get('fit_kind') == 'template' and fr.get('fit_status') == 'ok'))


def radio_of(fr, tr):
    if not has_position(fr):
        return None
    if tr and tr.get('tie_status') == 'tied' and tr.get('ra_tied'):
        return float(tr['ra_tied']), float(tr['dec_tied'])
    return float(fr['fit_ra_deg']), float(fr['fit_dec_deg'])


# ---------------------------------------------------------------------------------------------------------------------

@content_test('forced_event_build')
def snapshot_build(path, eval_sb=None, eval_wb=None, stokes=None, cadence=None, **ctx):
    """queue.txt names one Stokes and one cadence and the two evaluations; every build event has that Stokes and cadence, a
    unique cand_id and a queued identity; every queued identity has events; no two identities within MERGE_ARCMIN;
    skipped_events.csv has the declared columns and a reason per row; recomputed from the evaluations (selection of
    snapshot_forced_panels.event_rows): every build event is a selected row, and every selected row is built, listed as
    skipped, or absorbed into its identity's stronger event of the same frame (merged_events)."""
    from exopipe.phase3.lib.forced.tier_snapshot import selected_events
    s = _cfg().forced_snapshot
    run = os.path.dirname(os.path.abspath(path))
    qh = queue_header(run)
    if qh is None:
        return False, None, 'queue.txt has no build header line (Stokes, tier, evaluations)'
    S, cad, ev_sb, ev_wb, idents = qh
    eval_sb, eval_wb = eval_sb or ev_sb, eval_wb or ev_wb
    bad = []
    if stokes and stokes != S or cadence and cadence != cad:
        bad.append('queue says Stokes %s %s, the run is %s %s' % (S, cad, stokes, cadence))
    evs = build_events(run)
    if not evs:
        bad.append('no build events')
    ids = [e.get('cand_id') for e in evs]
    if len(set(ids)) != len(ids):
        bad.append('%d duplicate cand_ids' % (len(ids) - len(set(ids))))
    for e in evs:
        if e.get('stokes') != S or e.get('tier') != cad:
            bad.append('event %s is Stokes %s %s' % (e.get('cand_id'), e.get('stokes'), e.get('tier')))
            break
    qset, eid = set(idents), {e.get('identity') for e in evs}
    if qset != eid:
        bad.append('queued identities %d vs identities with events %d (only queued %s, only events %s)'
                   % (len(qset), len(eid), sorted(qset - eid)[:3], sorted(eid - qset)[:3]))
    pos = {}
    for p in glob.glob(os.path.join(run, 'source_catalogs', '*_sources_survivors.csv')):
        for r in _rows(p):
            if _num(r.get('ra')) is not None and _num(r.get('dec')) is not None:
                pos.setdefault(r['source_id'], (float(r['ra']), float(r['dec'])))
    keys = sorted(pos)
    close = [(a, b) for i, a in enumerate(keys) for b in keys[i + 1:]
             if sep_arcmin(pos[a][0], pos[a][1], pos[b][0], pos[b][1]) < s.MERGE_ARCMIN]
    if close:
        bad.append('%d identity pairs within %g arcmin, e.g. %s' % (len(close), s.MERGE_ARCMIN, close[0]))
    skf = os.path.join(run, 'skipped_events.csv')
    if not os.path.exists(skf):
        bad.append('no skipped_events.csv (the build lists every event it cannot panel, even none)')
        skipped = []
    else:
        with open(skf, newline='') as f:
            rd = csv.DictReader(f)
            skipped = list(rd)
            if tuple(rd.fieldnames or ()) != tuple(s.SKIP_FIELDS):
                bad.append('skipped_events.csv columns %s' % rd.fieldnames)
        if any(not r.get('skip_reason') for r in skipped):
            bad.append('a skipped event without its reason')
    n_sel = n_abs = 0
    if all(os.path.exists(os.path.join(e, 'thresholds.csv')) for e in (eval_sb, eval_wb)):
        sel = selected_events((eval_sb, eval_wb), S, cad)
        n_sel = len(sel)
        k = lambda r, cell: (cell, r['target'], r['band'], r['tier'], r['ts'])     # noqa: E731
        built = {k(e, '%s_%s' % (e['lst'], e['date'])): e for e in evs}
        frame = {}
        for e in evs:
            fk = ('%s_%s' % (e['lst'], e['date']), e['band'], e['tier'], e['ts'])
            frame[fk] = max(frame.get(fk, 0.0), abs(float(e['snr'])))
        skset = {k(r, r['cell']) for r in skipped}
        selset = set()
        for r in sel:
            kk = k(r, r['cell'])
            selset.add(kk)
            if kk in built or kk in skset:
                continue
            if frame.get((r['cell'], r['band'], r['tier'], r['ts']), -1.0) >= abs(float(r['snr'])):
                n_abs += 1
                continue
            bad.append('selected row %s neither built, skipped nor absorbed' % (kk,))
        extra = [kk for kk in built if kk not in selset]
        if extra:
            bad.append('%d build events are not selected rows of the evaluations, e.g. %s' % (len(extra), extra[0]))
    else:
        bad.append('evaluations %s / %s not readable (no thresholds.csv)' % (eval_sb, eval_wb))
    if bad:
        return False, len(bad), _detail(bad)
    return True, len(evs), 'Stokes %s %s: %d events, %d identities, %d skipped; %d selected rows (%d absorbed by merge)' % (
        S, cad, len(evs), len(qset), len(skipped), n_sel, n_abs)


@content_test('forced_fits')
def snapshot_fits(path, **ctx):
    """One fit record per build event (ids unique and equal); 'astrometric' = 1 exactly for an ok Gaussian fit at
    |S/N| >= ASTROM_SNR (snapshot_forced_panels.py:838); every ok Gaussian fit has size_beams and axis_ratio in
    master_catalog.csv; every positioned fit has a finite position."""
    s = _cfg().forced_snapshot
    run = os.path.dirname(os.path.abspath(path))
    fits = _rows(path)
    ids = [r.get('cand_id') for r in fits]
    bad = []
    if len(set(ids)) != len(ids):
        bad.append('%d duplicate cand_ids' % (len(ids) - len(set(ids))))
    ev = {e['cand_id'] for e in build_events(run)}
    if set(ids) != ev:
        bad.append('fit records %d vs build events %d (missing %d, extra %d)' % (len(set(ids)), len(ev), len(ev - set(ids)),
                                                                              len(set(ids) - ev)))
    mc = {r['cand_id']: r for r in _rows(os.path.join(run, 'master_catalog.csv'))}
    n_pos = 0
    for r in fits:
        if r.get('fit_kind') != 'template':
            want = '1' if (r.get('fit_status') == 'ok' and abs(float(r['snr'])) >= s.ASTROM_SNR) else '0'
            if r.get('astrometric') != want:
                bad.append('%s: astrometric %r, rule says %s' % (r['cand_id'], r.get('astrometric'), want))
            if r.get('fit_status') == 'ok' and not (mc.get(r['cand_id'], {}).get('size_beams')
                                                   and mc[r['cand_id']].get('axis_ratio')):
                bad.append('%s: ok fit without size/axis in master_catalog.csv' % r['cand_id'])
        if has_position(r):
            n_pos += 1
            ra, dec = _num(r['fit_ra_deg']), _num(r['fit_dec_deg'])
            if ra is None or dec is None or not (0.0 <= ra < 360.0 and -90.0 <= dec <= 90.0):
                bad.append('%s: position %s %s' % (r['cand_id'], r['fit_ra_deg'], r['fit_dec_deg']))
    if bad:
        return False, len(bad), _detail(bad)
    return True, len(fits), '%d fit records, %d with a radio position' % (len(fits), n_pos)


@content_test('forced_ties')
def snapshot_ties(path, **ctx):
    """One tie row per positioned fit record (cmd_tie: dets = has_position); a tied row has a finite tied position; where
    both the applied correction (tie_dx/dy) and the used-reference reconstruction (used_dx/dy) are recorded they agree
    within TIE_RECON_TOL_ARCMIN (add_used_refs, :996-997)."""
    s = _cfg().forced_snapshot
    run = os.path.dirname(os.path.abspath(path))
    ties = _rows(path)
    want = {r['cand_id'] for r in _rows(os.path.join(run, 'forced_fits.csv')) if has_position(r)}
    have = [r.get('cand_id') for r in ties]
    bad = []
    if set(have) != want or len(set(have)) != len(have):
        bad.append('tie rows %d (%d distinct) vs positioned fits %d' % (len(have), len(set(have)), len(want)))
    n_tied = 0
    for r in ties:
        if r.get('tie_status') == 'tied':
            n_tied += 1
            if _num(r.get('ra_tied')) is None or _num(r.get('dec_tied')) is None:
                bad.append('%s: tied without a tied position' % r['cand_id'])
            for a, b in (('used_dx_arcmin', 'tie_dx_arcmin'), ('used_dy_arcmin', 'tie_dy_arcmin')):
                x, y = _num(r.get(a)), _num(r.get(b))
                if x is not None and y is not None and abs(x - y) > s.TIE_RECON_TOL_ARCMIN:
                    bad.append('%s: %s %s vs %s %s' % (r['cand_id'], a, x, b, y))
        elif r.get('ra_tied'):
            bad.append('%s: %s with a tied position' % (r['cand_id'], r.get('tie_status')))
    if bad:
        return False, len(bad), _detail(bad)
    return True, len(ties), '%d tie rows, %d tied' % (len(ties), n_tied)


@content_test('forced_localization')
def snapshot_loc(path, **ctx):
    """Every identity with astrometric fits has ONE localization row whose n_det equals its astrometric fits in the tier the
    localization uses (10-s when it has any, else 1-min: multi-integration rule 1; snapshot_forced_panels.py:1398-1409), no
    other identity has a row, and every identity with n_det > 1 has its astrometry figure."""
    run = os.path.dirname(os.path.dirname(os.path.abspath(path)))
    fits = _rows(os.path.join(run, 'forced_fits.csv'))
    if not fits:
        return False, None, 'no forced_fits.csv beside %s' % path
    t10 = {r['identity'] for r in fits if r.get('astrometric') == '1' and r.get('tier') == '10s'}
    nast = {}
    for r in fits:
        if r.get('astrometric') == '1' and r.get('tier') == ('10s' if r['identity'] in t10 else '1min'):
            nast[r['identity']] = nast.get(r['identity'], 0) + 1
    loc = _rows(path)
    by = {}
    for r in loc:
        by.setdefault(r['identity_id'], []).append(r)
    bad = []
    for i, n in nast.items():
        rr = by.get(i, [])
        if len(rr) != 1 or _num(rr[0].get('n_det')) != n:
            bad.append('identity %s: %d astrometric fits, localization rows %s' % (i, n, [x.get('n_det') for x in rr]))
        elif n > 1 and not os.path.exists(os.path.join(run, 'localization_figs', '%s.png' % i)):
            bad.append('identity %s: no astrometry figure' % i)
    extra = sorted(set(by) - set(nast))
    if extra:
        bad.append('%d localization rows of identities without astrometric fits, e.g. %s' % (len(extra), extra[0]))
    if bad:
        return False, len(bad), _detail(bad)
    return True, len(loc), '%d identities localized, %d with figures' % (len(loc), sum(1 for n in nast.values() if n > 1))


@content_test('forced_assoc')
def snapshot_assoc_verify(path, run_dir=None, inplace=True, **ctx):
    """One association row per identity of the run, labels from the declared set, 'no radio position' exactly where the
    localization has no row, the NO_LABEL identities (AD Leo) never labelled; then snapshot_forced_assoc.py verify RUN
    [--served] in place must print FORCED-ASSOC-VERIFY-OK (A1-A5; A6 on served rows)."""
    c = _cfg()
    a = c.forced_assoc
    run = run_dir or os.path.dirname(os.path.abspath(path))
    rows = _rows(path)
    ids = {r['identity_id']: r for r in rows}
    run_ids = {r['identity'] for r in _rows(os.path.join(run, 'forced_fits.csv'))}
    loc = {r['identity_id'] for r in _rows(os.path.join(run, 'source_catalogs', 'localizations.csv'))}
    bad = []
    if set(ids) != run_ids or len(ids) != len(rows):
        bad.append('association rows %d (%d distinct) vs identities %d' % (len(rows), len(ids), len(run_ids)))
    for i, r in ids.items():
        if r.get('label') not in a.LABELS:
            bad.append('%s: label %r' % (i, r.get('label')))
        if (i in loc) == (r.get('label') == 'no radio position'):
            bad.append('%s: label %r vs localization row %s' % (i, r.get('label'), i in loc))
        if i in a.NO_LABEL and i in loc and r.get('label') != 'not labelled':
            bad.append('%s (ruled: never labelled) has label %r' % (i, r.get('label')))
    detail = ''
    if inplace and not bad:
        from exopipe.phase3.core import legacy
        from exopipe.phase3.lib.forced import common as K
        served = any(os.path.isdir(os.path.join(run, 'candidates', S)) for S in ('I', 'V'))
        res = legacy.run(K.script('assoc'), ['verify', run] + (['--served'] if served else []),
                         python=c.forced.PYTHON_KEY['snapshot'], env={'P3_RADIO_CATDIR': a.RADIO_CATDIR})
        line = next((l for l in reversed(res.tail.splitlines()) if 'FORCED-ASSOC-VERIFY-OK' in l or 'REFUS' in l), '')
        if not (res.rc == 0 and 'FORCED-ASSOC-VERIFY-OK' in res.tail):
            bad.append('in-place verify: %s' % (line or res.reason)[:300])
        detail = line[:200]
    if bad:
        return False, len(bad), _detail(bad)
    return True, len(rows), '%d identities; %s' % (len(rows), detail or 'in-place verify not run')


def png_size(p):
    """(width, height) from the PNG IHDR chunk (no decode)."""
    with open(p, 'rb') as f:
        h = f.read(24)
    if h[:8] != b'\x89PNG\r\n\x1a\n' or h[12:16] != b'IHDR':
        return None
    return struct.unpack('>II', h[16:24])


@content_test('forced_event_panels')
def snapshot_panels(path, stokes=None, **ctx):
    """missing_panels.csv is empty; the served tree candidates/<Stokes>/ holds one panel per served event, each at least
    VERIFY_PANEL_MIN_HEIGHT_PX high (the light-curve strip); one cadence among the served events; every served event's
    position is its radio position (tied where tied) or blank where it has none (RULING-RADIO-POSITION-ONLY)."""
    s = _cfg().forced_snapshot
    run = os.path.dirname(os.path.abspath(path))
    bad = []
    miss = [l for l in open(path).read().split('\n') if l.strip()]
    if miss:
        bad.append('%d panels missing' % len(miss))
    sts = [S for S in ('I', 'V') if os.path.isdir(os.path.join(run, 'candidates', S))]
    S = stokes or (sts[0] if len(sts) == 1 else None)
    if S is None:
        return False, None, 'served tree candidates/<Stokes>/: found %s' % sts
    sev = [r for p in glob.glob(os.path.join(run, 'candidates', S, '*', 'detections.csv')) for r in _rows(p)]
    pngs = glob.glob(os.path.join(run, 'candidates', S, '*', '*.png'))
    if not sev:
        bad.append('no served events')
    if len(pngs) != len(sev):
        bad.append('panels %d vs served events %d' % (len(pngs), len(sev)))
    short = [p for p in pngs if (png_size(p) or (0, 0))[1] < s.VERIFY_PANEL_MIN_HEIGHT_PX]
    if short:
        bad.append('%d panels without the light-curve strip (height < %d px), e.g. %s'
                   % (len(short), s.VERIFY_PANEL_MIN_HEIGHT_PX, os.path.basename(short[0])))
    tiers = {r.get('tier') for r in sev}
    if len(tiers) > 1:
        bad.append('served events of %s - one cadence per run' % sorted(tiers))
    fits = {r['cand_id']: r for r in _rows(os.path.join(run, 'forced_fits.csv'))}
    ties = {r['cand_id']: r for r in _rows(os.path.join(run, 'forced_ties.csv'))}
    badp = 0
    for e in sev:
        p = radio_of(fits.get(e['cand_id']), ties.get(e['cand_id']))
        if (p is None) != (not e.get('ra')) or (p and (abs(float(e['ra']) - p[0]) > 1e-4 or abs(float(e['dec']) - p[1]) > 1e-4)):
            badp += 1
    if badp:
        bad.append('%d served events whose position is not their radio position (or not blank without one)' % badp)
    if bad:
        return False, len(bad), _detail(bad)
    return True, len(sev), 'Stokes %s %s: %d served events, %d panels with the light-curve strip, radio positions only' % (
        S, ','.join(sorted(t for t in tiers if t)), len(sev), len(pngs))
