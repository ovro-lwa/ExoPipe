"""Readers of the tables the hour's highlights come from. Every value is read from a table that a search, a catalog,
the forced photometry or the page build wrote, or is given by the served page's own code; a table, column or rule that is
missing becomes a sentence in the block's limits, never a guess.

Untargeted candidates - the served deep and 10-minute review pages, Stokes V and I (config notify.PAGES; Gregg
2026-11-14). The page registry of the two-port server names each page's code, arguments and environment; page_rule.py
builds the page's own object from them. A detection of the hour is LISTED when its candidate is listed on its page and the
detection itself passes
  - the five value filters of config notify (S/N, elevation, count of detections in its epoch image, fitted size, fitted
    axis ratio), with the page's own values;
  - every default filter of its page;
  - the page's bright-source filter at the allowed mean sky loss of config notify;
and its candidate is not credibly matched to a known pulsar. One row per candidate, tier and Stokes: the hour's most
significant passing detection. <run>/master_catalog.csv gives the formal S/N (`global_snr`) where the tier records one.

Targets - the served forced-photometry event lists (config notify.FORCED_RUN: events at or above each sample's detection
threshold, each with its association verdict from the astrometry module), under the same value filters, and the
cross-match columns of the catalogs above (xm_best, offset percentile, chance-coincidence probability).
"""
import csv
import json
import os
import re

_CACHE = {}
_UTC = re.compile(r'_(\d{8})_(\d{6})(?:_|$)')
_WIN = re.compile(r'T(\d\d)(?:_|$)')


def _rows(path, needle=None):
    """DictReader rows of a CSV; with a needle only the lines that contain it are parsed (the tables are large)."""
    with open(path, newline='') as f:
        head = f.readline()
        while head.startswith('#'):
            head = f.readline()
        cols = next(csv.reader([head]))
        lines = (l for l in f if not l.startswith('#') and (needle is None or needle in l))
        for r in csv.reader(lines):
            yield dict(zip(cols, r))


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v else None


def _registry(CFG):
    with open(CFG.notify.REGISTRY) as f:
        pages = json.load(f)['pages']
    out = {}
    for p in pages:
        a = p.get('argv') or []
        g = lambda k: a[a.index(k) + 1] if k in a and a.index(k) + 1 < len(a) else None   # noqa: E731
        out[p['id']] = {'run': g('--run'), 'cat': g('--catalog-dir'), 'code': p.get('code'), 'entry': p}
    return out


_RULES = {}


def _rule(pid, p, CFG):
    """The page's own rule (page_rule.PageRule: its filter values, default filters, pulsar match and bright-source
    staircase at notify.BRIGHT_SKY_LOSS), built once per process and page."""
    if pid not in _RULES:
        from exopipe.phase3.lib.notify import page_rule
        _RULES[pid] = page_rule.PageRule(p['entry'], CFG.notify.BRIGHT_SKY_LOSS)
    return _RULES[pid]


HONED = ('snr', 'elev', 'nve', 'size', 'axis')       # the five value filters, by the page's own keys


def honed(m, n_max, c):
    """{filter: passes} of one detection's values for the five value filters of config notify (Gregg 2026-11-14). A
    missing value fails its filter, as on the pages. n_max: the count at the page's percentile breakpoint."""
    v = m.get
    return {'snr': v('snr') is not None and v('snr') > c.SNR_MIN,
            'elev': v('elev') is not None and v('elev') > c.ELEVATION_MIN_DEG,
            'nve': v('nve') is not None and n_max is not None and v('nve') <= n_max,
            'size': v('size') is not None and v('size') < c.SIZE_MAX_BEAMS,
            'axis': v('axis') is not None and v('axis') < c.AXIS_RATIO_MAX}


def elevation_deg(lst_hours, ra_deg, dec_deg, lat_deg):
    """Elevation from geometry (the pages' formula), degrees."""
    import math
    ha = math.radians(lst_hours * 15.0 - ra_deg)
    return math.degrees(math.asin(math.sin(math.radians(dec_deg)) * math.sin(math.radians(lat_deg))
                                  + math.cos(math.radians(dec_deg)) * math.cos(math.radians(lat_deg)) * math.cos(ha)))


def _when(cid, det_date, lst):
    """The time of a detection as its tables record it: a UTC stamp in the frame name (10-s, 1-minute), the 10-minute
    window of the hour, or the night and LST hour (deep)."""
    m = _UTC.search(cid)
    if m:
        d, t = m.groups()
        return f'{d[:4]}-{d[4:6]}-{d[6:]} {t[:2]}:{t[2:4]}:{t[4:]}'
    w = _WIN.search(det_date or '') or _WIN.search(cid)
    if w:
        return f'{(det_date or "")[:10]}, 10-minute window {int(w.group(1)) + 1} of 6 of LST {lst}'
    return f'{(det_date or "")[:10]}, LST {lst} hour'


_XM = re.compile(r'^(?P<sample>[^:]+):(?P<name>[^(]+?)\s*(?:\((?P<stats>.*)\))?$')


def _xm(s):
    """(sample prefix, target name, offset percentile, chance-coincidence probability, ambiguous) of a served row's
    cross-match: its own columns where the catalog has them (deep, 10-minute), else the statistics written inside the
    xm_best text (10-s, 1-minute: '... pct 11, E=0.360, p=3.9e-03')."""
    m = _XM.match((s.get('xm_best') or '').strip())
    if not m:
        return None
    pct, pch = _f(s.get('xm_offset_pct')), _f(s.get('xm_p_chance'))
    st = m.group('stats') or ''
    if pct is None:
        q = re.search(r'pct\s+([0-9.]+)', st)
        pct = _f(q.group(1)) if q else None
    if pch is None:
        q = re.search(r'\bp=([0-9.eE+-]+)', st)
        pch = _f(q.group(1)) if q else None
    amb = str(s.get('xm_ambiguous', '')).strip() not in ('', '0', 'False')
    return m.group('sample'), m.group('name').strip(), pct, pch, amb


def collect(item, CFG):
    """Everything the block needs for one hour: (transient rows sorted by significance, per-page counts, target rows from
    the catalogs' cross-match, limits, {'images': per page (count at the percentile breakpoint, {epoch image: count}),
    'xm_hidden': cross-matched detections that do not pass})."""
    if item in _CACHE:
        return _CACHE[item]
    lst, date = item.split('_', 1)
    needle = f'{lst}_{date}'
    c = CFG.notify
    limits, per_tier, rows, xm_targets, empty, images, xm_hidden = [], {}, [], [], [], {}, 0
    try:
        reg = _registry(CFG)
    except (OSError, ValueError, KeyError) as e:
        reg = None
        limits.append(f'the page registry {c.REGISTRY} cannot be read ({type(e).__name__}): no untargeted '
                      f'candidate of any tier is listed')
    for (tier, stokes), pid in (c.PAGES.items() if reg is not None else ()):
        name = f'{tier} Stokes {stokes}'
        p = reg.get(pid)
        if not p or not p['run'] or not p.get('code'):
            limits.append(f'{name}: page {pid} is not in the registry - not listed')
            continue
        try:
            rule = _rule(pid, p, CFG)
            hour = rule.hour(lst, date)
        except Exception as e:                      # noqa: BLE001 - a rule that cannot be evaluated is said, not guessed
            limits.append(f'{name}: the rule of page {pid} cannot be evaluated from the page\'s own code '
                          f'({type(e).__name__}: {e}) - not listed; the forced-photometry events of this tier and Stokes '
                          f'fail the epoch-image count filter (no count)')
            continue
        if rule.stokes != stokes:
            limits.append(f'{name}: page {pid} is a Stokes {rule.stokes} page by its own code, not Stokes {stokes} as '
                          f'config notify.PAGES names it - not listed; the forced-photometry events of this tier and Stokes '
                          f'fail the epoch-image count filter (no count)')
            continue
        for w in rule.warnings:
            limits.append(f'{name}, page {pid}: {w}')
        bp = rule.ref.get('nve') or []
        n_max = bp[c.N_EPOCH_MAX_PERCENTILE] if len(bp) == 101 else None
        # the hour's rows of the run's per-detection table: the formal S/N where the tier records one, and the count of
        # each epoch image as the page counts it (Stokes V: the search's recorded n_v_epoch of the image; Stokes I: the
        # table's Stokes I rows of the band and night), for the forced-photometry events of the same images. The table is
        # streamed and only the hour's lines are parsed.
        formal, count = {}, {}
        mc = os.path.join(p['run'], 'master_catalog.csv')
        if not os.path.exists(mc):
            count = None
            limits.append(f'{name}: page {pid}\'s run has no {mc}: no formal S/N, and the forced-photometry events of '
                          f'this tier and Stokes fail the epoch-image count filter (no count)')
        else:
            for r in _rows(mc, needle):
                if r.get('lst') != lst or r.get('date', '')[:10] != date or r.get('stokes', stokes) != stokes:
                    continue
                if 'global_snr' in r:
                    formal[r['cand_id']] = _f(r.get('global_snr'))
                if stokes == 'V':
                    if _f(r.get('n_v_epoch')) is not None:
                        count[(r.get('band'), r.get('date'))] = _f(r.get('n_v_epoch'))
                else:
                    k = (r.get('band'), r.get('date', '')[:10])
                    count[k] = count.get(k, 0) + 1
        images[(tier, stokes)] = (n_max, count)
        cnt = dict.fromkeys(('served',) + HONED + ('defaults', 'bright_pass', 'bright_no_data', 'pulsar', 'pass'), 0)
        best, bands, seen = {}, {}, set()            # identity -> best passing detection; its passing bands; served identities
        for sid, pos, d, m, s in hour:
            snr = _f(d.get('snr'))
            band = d.get('band')
            seen.add(sid)
            cnt['served'] += 1
            h = honed(m, n_max, c)
            dflt = rule.passes_defaults(m)
            bright = (sid, pos) not in rule.removed
            psr = bool(c.REMOVE_KNOWN_PULSARS and rule.pulsar(sid))
            for k in HONED:
                cnt[k] += h[k]
            cnt['defaults'] += dflt
            cnt['bright_pass'] += bright
            cnt['bright_no_data'] += (sid, pos) in rule.no_data
            cnt['pulsar'] += psr
            ok = snr is not None and all(h.values()) and dflt and bright and not psr
            url = f'{c.HUB_URL}/p/{pid}/source/{sid}?b={band}'
            when = _when(d.get('cand_id') or '', d.get('date'), lst)
            xm = _xm(s)
            if xm and xm[0] in c.XM_SAMPLES:
                if not ok:
                    xm_hidden += 1
                else:
                    xm_targets.append({
                        'target': xm[1], 'sample': c.SAMPLES[c.XM_SAMPLES[xm[0]]],
                        'found_by': 'untargeted search', 'tier': tier, 'band': band, 'stokes': stokes,
                        'time_utc': when, 'flux_jy': None,
                        'snr_formal': formal.get(d.get('cand_id')), 'snr_local': snr, 'offset_percentile': xm[2],
                        'p_chance': xm[3], 'note': 'AMBIGUOUS association' if xm[4] else '',
                        'link': url, '_abs': abs(snr), '_key': (xm[1], tier, stokes)})
            if not ok:
                continue
            cnt['pass'] += 1
            bands.setdefault(sid, set()).add(band)
            cur = best.get(sid)
            if cur is None or abs(snr) > cur['_abs']:
                cls = ', '.join(x for x in (s.get('promotion') or s.get('source_class') or '',
                                            'multi-integration' if str(s.get('multi_integration', '')).strip() in ('1', 'True')
                                            else '') if x)
                ra, dec = _f(s.get('ra')), _f(s.get('dec'))
                best[sid] = {'tier': tier, 'band': band, 'stokes': stokes, '_abs': abs(snr), 'time_utc': when,
                             'ra_deg': ra if ra is not None else _f(d.get('ra')),
                             'dec_deg': dec if dec is not None else _f(d.get('dec')),
                             'gal_lat_deg': m.get('glat'), 'elevation_deg': m.get('elev'),
                             'snr_formal': formal.get(d.get('cand_id')), 'snr_local': snr, 'cls': cls,
                             'link': url, 'link_text': sid}
        for sid, b in best.items():
            if len(bands[sid]) > 1:
                b['cls'] = (b['cls'] + '; ' if b['cls'] else '') + f'{len(bands[sid])} bands this hour'
        rows += list(best.values())
        if seen:
            per_tier[name] = dict(cnt, candidates=len(seen), listed=len(best), n_max=n_max, bright=rule.bright)
        else:
            empty.append(f'{name} (page {pid})')
    if empty:
        limits.append('no detection of this hour among the candidates listed on: ' + ', '.join(empty) + ' (the hour is not '
                      'in that served run, or nothing was detected)')
    # one row per target, tier and Stokes among the cross-matched untargeted detections that pass: the most significant
    one = {}
    for r in xm_targets:
        if r['_key'] not in one or r['_abs'] > one[r['_key']]['_abs']:
            r['_n'] = one[r['_key']]['_n'] + 1 if r['_key'] in one else 1
            one[r['_key']] = r
        else:
            one[r['_key']]['_n'] += 1
    xm_targets = list(one.values())
    for r in xm_targets:
        if r['_n'] > 1:
            r['note'] = (r['note'] + '; ' if r['note'] else '') + (f'{r["_n"]} detections of this hour cross-matched to this '
                                                                   f'target in this tier and Stokes (the most significant is shown)')
    rows.sort(key=lambda r: -r['_abs'])
    for i, r in enumerate(rows):
        r['rank'] = i + 1
    _CACHE[item] = (rows, per_tier, xm_targets, limits, {'images': images, 'xm_hidden': xm_hidden})
    return _CACHE[item]


def at_least_deep(sel, top_n, fraction):
    """The listed rows of one Stokes: the top_n most significant of `sel` (sorted by significance), changed until at
    least `fraction` of the listed rows are of the deep tier: while they are fewer, the least significant listed row of
    another tier leaves and the next most significant deep row not yet listed, if there is one, enters. With no passing
    deep candidate nothing is listed. The rows keep their rank among all passing candidates of the Stokes."""
    chosen = list(sel[:top_n])
    rest = [r for r in sel[top_n:] if r['tier'] == 'deep']
    while chosen and sum(r['tier'] == 'deep' for r in chosen) < fraction * len(chosen):
        del chosen[max(k for k, r in enumerate(chosen) if r['tier'] != 'deep')]
        if rest:
            chosen.append(rest.pop(0))
    return chosen


def transients(item, CFG, top_n):
    """([(Stokes, the listed rows of that Stokes (at_least_deep), number passing in that Stokes)] with Stokes V first (the
    hub's order), number passing in all, per-page counts, limits)."""
    rows, per_tier, _xm_rows, limits, _extra = collect(item, CFG)
    parts = []
    for st in ('V', 'I'):
        sel = [dict(r) for r in rows if r['stokes'] == st]
        for i, r in enumerate(sel):
            r['rank'] = i + 1
        parts.append((st, at_least_deep(sel, top_n, CFG.notify.DEEP_MIN_FRACTION), len(sel)))
    return parts, len(rows), per_tier, list(limits)


def targets(item, CFG):
    """(rows, the sentence under the table, limits): the forced-photometry events of the two samples in this hour, then
    the served untargeted candidates of this hour that a catalog's cross-match gives to a target of the two samples."""
    lst, date = item.split('_', 1)
    prefix = f'F_{lst}_{date}'
    limits, rows, n_runs = [], [], 0
    c = CFG.notify
    _rows_t, _pt, xm, _lim, extra = collect(item, CFG)
    try:
        lst_h = float(lst[:2])
    except ValueError:
        lst_h = None
    fcnt = dict.fromkeys(('events',) + HONED + ('pass', 'no_fit', 'no_fit_low_snr', 'no_page', 'no_image_count',
                                                 'no_count_fail'), 0)
    listed_tiers = {t for t, _s in c.PAGES}
    try:
        reg = _registry(CFG)
    except (OSError, ValueError, KeyError):
        reg = {}
    page_of = {os.path.normpath(v['run']): k for k, v in reg.items() if v.get('run')}
    for tier, tag in CFG.notify.FORCED_TIERS.items():
        for stokes in ('V', 'I'):
            run = os.path.join(CFG.paths.SNAPSHOT_REVIEW_ROOT, CFG.notify.FORCED_RUN.format(stokes=stokes, tier=tag))
            ev = os.path.join(run, 'forced_assoc_events.csv')
            if not os.path.exists(ev):
                limits.append(f'forced photometry, {tier} Stokes {stokes}: no served event list ({ev}) - not listed')
                continue
            n_runs += 1
            hour = [r for r in _rows(ev, prefix) if r.get('cand_id', '').startswith(prefix)
                    and r.get('sample') in CFG.notify.SAMPLES]
            if not hour:
                continue
            ident, fits = {}, {}
            ip, fp = os.path.join(run, 'forced_assoc_identity.csv'), os.path.join(run, 'forced_fits.csv')
            if os.path.exists(ip):
                ident = {r['identity_id']: r for r in _rows(ip)}
            if os.path.exists(fp):
                fits = {r['cand_id']: r for r in _rows(fp, prefix)}
            pid = page_of.get(os.path.normpath(run))
            n_max, count = extra['images'].get((tier, stokes), (None, None))
            # a tier with a listed page (deep, 10-minute) whose page rule or per-detection table could not be read has no
            # count: its events FAIL the count filter (collect() states why in the limits); a tier without a listed page
            # (10-s, 1-minute) has no count by design and its events pass it
            paged = tier in listed_tiers
            for r in hour:
                # THE SAME FILTERS AS THE TRANSIENT LIST (Gregg 2026-11-14: "Apply the following filters to *all* of
                # these"), from the event's own tables: S/N, elevation at the middle of the LST hour from the event's
                # position, the fitted size and axis ratio of forced_fits.csv, and - for a deep or 10-minute event - the
                # count of its epoch image as the same tier's page counts it. A missing value fails, except the count of
                # an image without a catalog detection and of a tier without a listed page. The bright-source filter and
                # the pulsar match are the pages' and do not exist for a forced event.
                fit = fits.get(r['cand_id'], {})
                ra = _f(r.get('ra')) if _f(r.get('ra')) is not None else _f(fit.get('fit_ra_deg'))
                dec = _f(r.get('dec')) if _f(r.get('dec')) is not None else _f(fit.get('fit_dec_deg'))
                m = {'snr': abs(_f(r.get('snr'))) if _f(r.get('snr')) is not None else None,
                     'elev': elevation_deg(lst_h + 0.5, ra, dec, c.SITE_LAT_DEG) if None not in (ra, dec, lst_h) else None,
                     'size': _f(fit.get('size_beams')), 'axis': _f(fit.get('axis_ratio')), 'nve': None}
                h = honed(m, n_max, c)
                if not paged:
                    h['nve'] = True                              # a tier without a listed page: no count exists
                    fcnt['no_page'] += 1
                elif count is None or n_max is None:
                    h['nve'] = False                             # its page's count could not be read: fails
                    fcnt['no_count_fail'] += 1
                else:
                    night = (r.get('cand_id') or '').split('_')[2] if (r.get('cand_id') or '').count('_') >= 2 else ''
                    n = count.get((r.get('band'), night)) if stokes == 'V' else count.get((r.get('band'), night[:10]), 0)
                    h['nve'] = n is None or n <= n_max
                    fcnt['no_image_count'] += n is None
                fcnt['events'] += 1
                for k in HONED:
                    fcnt[k] += h[k]
                if m['size'] is None or m['axis'] is None:
                    fcnt['no_fit'] += 1
                    fcnt['no_fit_low_snr'] += not h['snr']
                if not all(h.values()):
                    continue
                fcnt['pass'] += 1
                i_ = ident.get(r.get('identity_id'), {})
                label = r.get('identity_label') or i_.get('label') or 'not labelled'
                note = f'association: {label.upper() if label == "ambiguous" else label}'
                if r.get('bg_name'):
                    note += (f' (nearest catalogued background source {r["bg_name"]}, the event\'s offset percentile from '
                             f'it {r.get("pct_bg") or "not recorded"})')
                d, t = (r.get('ts') or '_').split('_')[0], (r.get('ts') or '_').split('_')[-1]
                rows.append({'target': (r.get('target') or '').replace('_', ' '), 'sample': CFG.notify.SAMPLES[r['sample']],
                             'found_by': 'forced photometry', 'tier': tier, 'band': r.get('band'), 'stokes': stokes,
                             'time_utc': f'{d[:4]}-{d[4:6]}-{d[6:]} {t[:2]}:{t[2:4]}:{t[4:]}' if len(d) == 8 and len(t) == 6
                             else r.get('ts'),
                             'flux_jy': _f(fit.get('fit_peak_jyb')), 'snr_formal': _f(r.get('snr')),
                             'snr_local': None, 'offset_percentile': _f(r.get('pct_target')),
                             'p_chance': _f(i_.get('target_pcc')), 'note': note, '_label': label,
                             'link': f'{CFG.notify.HUB_URL}/p/{pid}/source/{r.get("identity_id")}' if pid else None,
                             '_abs': abs(_f(r.get('snr')) or 0.0)})
    # one row per target, tier, Stokes and association verdict: the hour's most significant event, with the count.
    # Events the association gives to a catalogued background source, or to no source at the target, are not detections
    # of the target: they are counted in the sentence under the table and stay on the forced pages.
    LISTED = ('target', 'ambiguous', 'no radio position', 'not labelled')
    groups, other = {}, {}
    for r in rows:
        if r['_label'] not in LISTED:
            other[r['_label']] = other.get(r['_label'], 0) + 1
            continue
        k = (r['target'], r['tier'], r['stokes'], r['_label'])
        g = groups.get(k)
        if g is None or r['_abs'] > g['_abs']:
            r['_n'] = (g['_n'] if g else 0) + 1
            groups[k] = r
        else:
            g['_n'] += 1
    n_events = fcnt['events']
    rows = list(groups.values())
    for r in rows:
        if r['_n'] > 1:
            r['note'] += f'; {r["_n"]} events of this target, tier and Stokes this hour (the most significant is shown)'
    order = {k: i for i, k in enumerate(LISTED)}
    rows.sort(key=lambda r: (order[r['_label']], -r['_abs']))
    xm.sort(key=lambda r: -r['_abs'])
    rows += xm
    others = '; '.join(f'{n} given to {"a catalogued " + k if k == "background source" else k}' for k, n in sorted(other.items()))
    passing = (f'{fcnt["pass"]} pass every filter (S/N above {c.SNR_MIN:g}: {fcnt["snr"]}; elevation above '
               f'{c.ELEVATION_MIN_DEG:g} deg: {fcnt["elev"]}; fitted size below {c.SIZE_MAX_BEAMS:g} beams: {fcnt["size"]}; '
               f'fitted axis ratio below {c.AXIS_RATIO_MAX:g}: {fcnt["axis"]}; count of the epoch image at or below the '
               f'{c.N_EPOCH_MAX_PERCENTILE}th percentile: {fcnt["nve"]}, of which {fcnt["no_page"]} are 10-second or '
               f'1-minute events (no listed page, no count) and {fcnt["no_image_count"]} are of an epoch image without a '
               f'catalog detection, and pass'
               + (f'; {fcnt["no_count_fail"]} deep or 10-minute events fail it because their page\'s count could not be '
                  f'read (see the limits)' if fcnt['no_count_fail'] else '')
               + f'; {fcnt["no_fit"]} events have no fitted size or axis ratio and fail, of which {fcnt["no_fit_low_snr"]} '
               f'have |S/N| at or below {c.SNR_MIN:g} and fail that filter too: the forced fit is a template fit, without '
               f'a size, below 7 sigma)')
    note = (f'{n_events} forced-photometry events of the two samples at or above their sample\'s detection threshold in this '
            f'hour ({n_runs} served event lists read: four tiers, Stokes V and I); {passing}'
            + (f'. Of the passing events the association step gives to no target: {others} (not listed; they are on the '
               f'forced pages)' if other else '')
            + f'. {len(xm)} served untargeted detections of this hour that pass the filters of the transient list are '
            f'cross-matched to a target by their catalog ({extra["xm_hidden"]} more are cross-matched and do not pass). The '
            f'forced S/N is the formal one (flux density over the frame noise); the event lists hold no local-empirical '
            f'S/N per event. Association verdicts are the astrometry module\'s (offset percentile, chance-coincidence '
            f'probability; no radius). The bright-source filter and the pulsar match are the pages\' and are not applied '
            f'to forced events.')
    if not n_events and not xm:
        note = (f'None: no forced-photometry event of the two samples at or above its detection threshold in this hour '
                f'({n_runs} served event lists read: four tiers, Stokes V and I), and no served untargeted detection of this '
                f'hour that passes the filters is cross-matched to a target of the two samples ({extra["xm_hidden"]} are '
                f'cross-matched and do not pass).')
    return rows, note, limits
