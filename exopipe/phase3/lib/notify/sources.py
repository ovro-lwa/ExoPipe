"""Readers of the tables the hour's highlights come from. Nothing is computed here: every value is read from a table that a
search, a catalog, the forced photometry or the page build wrote, and a table or column that is missing becomes a sentence
in the block's limits, never a guess.

Untargeted candidates - one reader for the eight served review pages (deep, 10-minute, 1-minute, 10-s; Stokes V and I). The
page registry of the two-port server names each page's run directory, catalog directory and code. Per page:
  <catalog-dir>/<band>_source_members.csv     one row per detection (source_id, cand_id, det_snr, det_date, band[, det_ra, det_dec])
  <catalog-dir>/<band>_sources_survivors.csv  one row per SERVED identity and band (position, class, cross-match, review_dir)
  <run>/master_catalog.csv                    per detection; `global_snr` where the tier records a formal S/N (deep, 10-minute)
  <run>/fp_filters.json + fp_filters.csv, scintillation_index.json + .csv     the page's default filters and their values
A detection of the hour is LISTED when its identity is served and the detection itself passes every default filter of its
page. The rule is the page's own (review_filters.passes and the constants of the page's review_filters.py, loaded from the
directory of the page's code); the page shows a candidate when ANY of its detections passes, so every listed detection is
of a candidate the page shows by default. One row per identity, tier and Stokes: the hour's most significant detection.

Targets - the served forced-photometry event lists (config notify.FORCED_RUN: events at or above each sample's detection
threshold, each with its association verdict from the astrometry module) and the cross-match columns of the catalogs above
(xm_best, offset percentile, chance-coincidence probability).
"""
import csv
import importlib.util
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
        out[p['id']] = {'run': g('--run'), 'cat': g('--catalog-dir'), 'code': p.get('code')}
    return out


def _page_filters_module(code):
    """review_filters.py beside the page's code, loaded by its path (the page's own rule and constants)."""
    path = os.path.join(os.path.dirname(code), 'review_filters.py')
    spec = importlib.util.spec_from_file_location('exopipe3_notify_review_filters', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class PageFilters:
    """The default filters of one page, assembled as review_filters.build assembles them: the categories of
    fp_filters.json with a `default_off` list for the page's Stokes (not those the page code exempts), and the
    scintillation group where the run's scintillation_index.json says default_hidden."""

    def __init__(self, rf, run, stokes):
        self.rf, self.defaults, self.cats, self.rows, self.n, self.scint = rf, {}, [], {}, {}, {}
        self.by_cid = {}
        fpp, fpm = os.path.join(run, rf.FP_FILE), os.path.join(run, rf.FP_META)
        if os.path.exists(fpp) and os.path.exists(fpm):
            with open(fpm) as f:
                meta = json.load(f)
            own = lambda x: x.get('stokes') in (None, '', stokes)                         # noqa: E731
            self.cats = [c for c in meta.get('cats', []) if str(c.get('key', '')).startswith('fp_c_') and c.get('states')
                         and own(c)]
            for c in self.cats:
                off = [s for s in (c.get('default_off') or []) if s in c['states'] and c['key'] not in rf.NO_DEFAULT_CATS]
                if off:
                    self.defaults[c['key']] = set(s for s in c['states'] if s not in off)
            for r in _rows(fpp):
                cid = r.get('cand_id')
                if not cid:
                    continue
                try:
                    k = (cid, round(float(r['ra']), 4), round(float(r['dec']), 4))
                except (KeyError, TypeError, ValueError):
                    k = (cid, None, None)
                self.rows[k] = r
                self.by_cid.setdefault(cid, []).append(r)
        scp = os.path.join(run, rf.SCINT_FILE)
        if os.path.exists(scp):
            with open(os.path.join(run, rf.SCINT_META)) as f:
                hide = bool(json.load(f)['default_hidden'])
            for r in _rows(scp):
                self.scint[(r['lst'], r['date'], r['band'])] = rf.SCINT_FROM[r['state']]
            if self.scint and hide:
                self.defaults[rf.SCINT_CAT['key']] = set(s for s in rf.SCINT_STATES if s not in rf.SCINT_CAT['default_off'])
        self.undecided = 0       # detections whose frame has several filter rows that disagree and no position to choose by

    def values(self, cid, ra, dec, lst, date, band):
        """The detection's value of every default category, as the page gives it (no row: the category's last state)."""
        m = {}
        if self.rows:
            fr = None
            if ra is not None and dec is not None:
                fr = self.rows.get((cid, round(ra, 4), round(dec, 4)))
            fr = fr or self.rows.get((cid, None, None))
            rows = self.by_cid.get(cid) or []
            if fr is None and len(rows) == 1:
                fr = rows[0]
            open_ = False
            for c in self.cats:
                if fr is None and len(rows) > 1 and ra is None:
                    vals = set(r.get(c['key']) for r in rows)        # the members table of this tier holds no position:
                    v = vals.pop() if len(vals) == 1 else None       # rows of one frame that agree decide; else as unmatched
                    open_ = open_ or (c['key'] in self.defaults and v is None)
                else:
                    v = (fr or {}).get(c['key'])
                m[c['key']] = v if v in c['states'] else c['states'][-1]
            self.undecided += open_                                  # counted once per detection
        if self.scint:
            m[self.rf.SCINT_CAT['key']] = self.scint.get((lst, date[:10], band), self.rf.SCINT_STATES[2])
        return m

    def passes(self, m):
        return self.rf.passes([m], self.defaults)


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
_INDEX_CACHE = {}


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


def _hour_from_index(run, lst, date):
    """The served candidates of a run that have a detection in the hour, from the page index the server itself reads
    (<run>/page_index/shard_*.tsv.gz: one line per served candidate directory with the text of its detections.csv):
    [(review_dir, [detection rows of the hour])]. One pass per run and hour."""
    key = (run, lst, date)
    if key in _INDEX_CACHE:
        return _INDEX_CACHE[key]
    import glob
    import gzip
    needle = f'{lst}_{date}'
    out = []
    for sh in sorted(glob.glob(os.path.join(run, 'page_index', 'shard_*.tsv.gz'))):
        with gzip.open(sh, 'rt', newline='') as f:
            for line in f:
                if needle not in line:
                    continue
                part = line.rstrip('\n').split('\t')
                try:
                    text = json.loads(part[-1]).get('t') or ''
                except ValueError:
                    continue
                det = [r for r in csv.DictReader(text.splitlines()) if r.get('lst') == lst and r.get('date', '')[:10] == date]
                if det:
                    out.append((part[0], det))
    _INDEX_CACHE[key] = out
    return out


def _survivors_by_review_dir(path, wanted):
    """The rows of a (large) survivors table whose review_dir is wanted; review_dir is located by the header."""
    out = {}
    if not os.path.exists(path) or not wanted:
        return out
    with open(path, newline='') as f:
        cols = next(csv.reader([f.readline()]))
        k = cols.index('review_dir') if 'review_dir' in cols else None
        if k is None:
            return out
        for line in f:
            cell = line.split(',', k + 1)[k] if k < 6 else None      # the columns before review_dir hold no comma
            if cell is not None and cell not in wanted:
                continue
            r = dict(zip(cols, next(csv.reader([line]))))
            if r.get('review_dir') in wanted:
                out[r['review_dir']] = r
    return out


def _detections(tier, stokes, p, lst, date, CFG):
    """The served detections of the hour on one page, as dicts (ident, band, cand_id, snr, ra, dec, det_date, row of the
    served identity). (detections, number of detections of the hour in the tables whether served or not)."""
    needle = f'{lst}_{date}'
    out, n_det = [], 0
    if os.path.exists(os.path.join(p['run'], 'page_index', 'manifest.json')):
        tag = CFG.notify.FORCED_TIERS[tier]                      # the tier's tag in the tables: 10s, 1min
        hour = [(rd, [r for r in det if r.get('tier') == tag and r.get('stokes') == stokes])
                for rd, det in _hour_from_index(p['run'], lst, date) if f'/{stokes}/' in f'/{rd}/']
        hour = [(rd, det) for rd, det in hour if det]
        by_band = {}                         # the served row of a candidate directory is looked up in the table of the
        for rd, det in hour:                 # directory's own band suffix and of every band its detections name (the
            for band in {rd.rsplit('_', 1)[-1]} | {r.get('band') for r in det} | {r.get('subband') for r in det}:
                if band:                     # 1-minute directories are <identity>_<band>_1min: their suffix is no band)
                    by_band.setdefault(band, set()).add(rd)
        surv = {}
        for band, wanted in by_band.items():
            surv.update(_survivors_by_review_dir(os.path.join(p['cat'], f'{band}_sources_survivors.csv'), wanted))
        for rd, det in hour:
            n_det += len(det)
            s = surv.get(rd)
            if s is None:
                continue                                         # the candidate is not on this tier's page
            for r in det:
                out.append({'ident': s.get('identity_id') or s.get('source_id'), 'band': r.get('band'), 'cand_id': r['cand_id'],
                            'snr': _f(r.get('snr')), 'ra': _f(r.get('ra')), 'dec': _f(r.get('dec')),
                            'det_date': r.get('date'), 's': s, 'pos': (_f(r.get('ra')), _f(r.get('dec')))})
        return out, n_det
    for mf in sorted(f for f in os.listdir(p['cat']) if f.endswith('_source_members.csv')):
        band = mf[:-len('_source_members.csv')]
        members = [r for r in _rows(os.path.join(p['cat'], mf), needle) if needle in r.get('cand_id', '')]
        if not members:
            continue
        n_det += len(members)
        sp = os.path.join(p['cat'], f'{band}_sources_survivors.csv')
        surv = {r['source_id']: r for r in _rows(sp)} if os.path.exists(sp) else {}
        for r in members:
            s = surv.get(r['source_id'])
            if s is None:
                continue                                         # a detection whose identity is not served
            out.append({'ident': s.get('identity_id') or s['source_id'], 'band': band, 'cand_id': r['cand_id'],
                        'snr': _f(r.get('det_snr')), 'ra': _f(r.get('det_ra')), 'dec': _f(r.get('det_dec')),
                        'det_date': r.get('det_date') or date, 's': s, 'pos': (_f(s.get('ra')), _f(s.get('dec')))})
    return out, n_det


def collect(item, CFG):
    """Everything the block needs for one hour: (transient rows sorted by significance, per-page counts, target rows from
    the catalogs' cross-match, limits)."""
    if item in _CACHE:
        return _CACHE[item]
    lst, date = item.split('_', 1)
    needle = f'{lst}_{date}'
    limits, per_tier, rows, xm_targets, empty = [], {}, [], [], []
    try:
        reg = _registry(CFG)
    except (OSError, ValueError, KeyError) as e:
        reg = None
        limits.append(f'the page registry {CFG.notify.REGISTRY} cannot be read ({type(e).__name__}): no untargeted '
                      f'candidate of any tier is listed')
    for (tier, stokes), pid in (CFG.notify.PAGES.items() if reg is not None else ()):
        name = f'{tier} Stokes {stokes}'
        p = reg.get(pid)
        if not p or not p['run'] or not p['cat'] or not os.path.isdir(p['cat']):
            limits.append(f'{name}: page {pid} is not in the registry or its catalog directory is missing - not listed')
            continue
        try:
            flt = PageFilters(_page_filters_module(p['code']), p['run'], stokes)
            dets, n_det = _detections(tier, stokes, p, lst, date, CFG)
        except Exception as e:                      # noqa: BLE001 - a table or filter that cannot be read is said, not guessed
            limits.append(f'{name}: the tables or default filters of page {pid} cannot be read ({type(e).__name__}: {e}) - '
                          f'not listed')
            continue
        formal = {}
        mc = os.path.join(p['run'], 'master_catalog.csv')
        if dets and os.path.exists(mc) and os.path.getsize(mc) < 2 ** 28:      # the tiers whose table records a formal S/N
            for r in _rows(mc, needle):
                if 'global_snr' in r and r.get('stokes', stokes) == stokes:
                    formal[r['cand_id']] = _f(r.get('global_snr'))
        best, bands, seen = {}, {}, set()            # identity -> best passing detection; its passing bands; served identities
        for d in dets:
            s, ident, band, snr = d['s'], d['ident'], d['band'], d['snr']
            seen.add(ident)
            if snr is None:
                continue
            ok = flt.passes(flt.values(d['cand_id'], d['ra'], d['dec'], lst, d['det_date'] or date, band))
            url = f'{CFG.notify.HUB_URL}/p/{pid}/source/{ident}?b={band}'
            xm = _xm(s)
            if xm and xm[0] in CFG.notify.XM_SAMPLES:
                xm_targets.append({
                    'target': xm[1], 'sample': CFG.notify.SAMPLES[CFG.notify.XM_SAMPLES[xm[0]]],
                    'found_by': 'untargeted search', 'tier': tier, 'band': band, 'stokes': stokes,
                    'time_utc': _when(d['cand_id'], d['det_date'], lst), 'flux_jy': None,
                    'snr_formal': formal.get(d['cand_id']), 'snr_local': snr, 'offset_percentile': xm[2], 'p_chance': xm[3],
                    'note': '; '.join(x for x in ('AMBIGUOUS association' if xm[4] else '',
                                                  '' if ok else 'hidden by a default filter of its page') if x),
                    'link': url, '_abs': abs(snr), '_ok': ok, '_key': (xm[1], tier, stokes)})
            if not ok:
                continue
            bands.setdefault(ident, set()).add(band)
            cur = best.get(ident)
            if cur is None or abs(snr) > cur['_abs']:
                cls = ', '.join(x for x in (s.get('promotion') or s.get('source_class') or '',
                                            'multi-integration' if str(s.get('multi_integration', '')).strip() in ('1', 'True')
                                            else '') if x)
                best[ident] = {'tier': tier, 'band': band, 'stokes': stokes, '_abs': abs(snr),
                               'time_utc': _when(d['cand_id'], d['det_date'], lst),
                               'ra_deg': d['pos'][0], 'dec_deg': d['pos'][1],
                               'snr_formal': formal.get(d['cand_id']), 'snr_local': snr, 'cls': cls,
                               'link': url, 'link_text': ident}
        for ident, b in best.items():
            if len(bands[ident]) > 1:
                b['cls'] = (b['cls'] + '; ' if b['cls'] else '') + f'{len(bands[ident])} bands this hour'
        rows += list(best.values())
        if seen:
            per_tier[name] = (len(seen), len(best))
        if n_det == 0:
            empty.append(f'{name} (page {pid})')
        if flt.undecided:
            limits.append(f'{name}: {flt.undecided} detections lie in frames whose filter rows disagree and the tier\'s '
                          f'members table holds no position to choose by: treated as the page treats an unmatched row')
    if empty:
        limits.append('no detection of this hour in the served catalog of: ' + ', '.join(empty) + ' (the hour is not in that '
                      'served run, or nothing was detected)')
    # one row per target, tier and Stokes among the cross-matched untargeted detections: the most significant, a
    # detection that passes its page's default filters before one that does not
    one = {}
    for r in xm_targets:
        if r['_key'] not in one or (r['_ok'], r['_abs']) > (one[r['_key']]['_ok'], one[r['_key']]['_abs']):
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
    _CACHE[item] = (rows, per_tier, xm_targets, limits)
    return _CACHE[item]


def transients(item, CFG, top_n):
    """([(Stokes, the top_n rows of that Stokes, number passing in that Stokes)] with Stokes V first (the hub's order),
    number passing in all, per-page counts, limits)."""
    rows, per_tier, _xm, limits = collect(item, CFG)
    parts = []
    for st in ('V', 'I'):
        sel = [dict(r) for r in rows if r['stokes'] == st]
        for i, r in enumerate(sel):
            r['rank'] = i + 1
        parts.append((st, sel[:top_n], len(sel)))
    return parts, len(rows), per_tier, list(limits)


def targets(item, CFG):
    """(rows, the sentence under the table, limits): the forced-photometry events of the two samples in this hour, then
    the served untargeted candidates of this hour that a catalog's cross-match gives to a target of the two samples."""
    lst, date = item.split('_', 1)
    prefix = f'F_{lst}_{date}'
    limits, rows, n_runs = [], [], 0
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
            for r in hour:
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
                             'flux_jy': _f(fits.get(r['cand_id'], {}).get('fit_peak_jyb')), 'snr_formal': _f(r.get('snr')),
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
    n_events = len(rows)
    rows = list(groups.values())
    for r in rows:
        if r['_n'] > 1:
            r['note'] += f'; {r["_n"]} events of this target, tier and Stokes this hour (the most significant is shown)'
    _rows_t, _pt, xm, _lim = collect(item, CFG)
    order = {k: i for i, k in enumerate(LISTED)}
    rows.sort(key=lambda r: (order[r['_label']], -r['_abs']))
    xm.sort(key=lambda r: -r['_abs'])
    rows += xm
    others = '; '.join(f'{n} given to {"a catalogued " + k if k == "background source" else k}' for k, n in sorted(other.items()))
    note = (f'{n_events} forced-photometry events of the two samples at or above their sample\'s detection threshold in this '
            f'hour ({n_runs} served event lists read: four tiers, Stokes V and I)'
            + (f', of which the association step gives to no target: {others} (not listed; they are on the forced pages)'
               if other else '')
            + f'. {len(xm)} served untargeted detections of this hour are cross-matched to a target by their catalog. The '
            f'forced S/N is the formal one (flux density over the frame noise); the event lists hold no local-empirical '
            f'S/N per event. Association verdicts are the astrometry module\'s (offset percentile, chance-coincidence '
            f'probability; no radius).')
    if not n_events and not xm:
        note = (f'None: no forced-photometry event of the two samples at or above its detection threshold in this hour '
                f'({n_runs} served event lists read: four tiers, Stokes V and I), and no served untargeted detection of this '
                f'hour is cross-matched to a target of the two samples.')
    return rows, note, limits
