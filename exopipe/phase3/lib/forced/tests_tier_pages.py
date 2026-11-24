"""Content tests of snapshot family T (the HWJ / 10 pc / LPT forced-photometry pages of the 10-minute and deep tiers;
lib/forced/tier_pages.py). Code of their own: no import of tier_forced_page_20261027.py or planets_forced_page_20261026.py
(the in-place verify / verify-chain are the stages' own content tests, run before these).

Classes and the file each is registered on:
  forced_tier_eval  <EVAL>/thresholds.csv      a stage-2 evaluation of every sample: thresholds per (sample, Stokes) of its
                                               tier, clean target and control counts per sample, the cells = every LST of
                                               its input, the event table the site reads
  forced_tier_site  <SITE>/page_record.json    the site lists exactly the clean events at or above their Gaussian thresholds
                                               in the four evaluations it names (and those are the configured ones), with the
                                               thresholds of those files - every sample's, each with its exceedance plot per
                                               tier and its page section per tier and Stokes; one panel above the size floor,
                                               on the page, and one yellow-number file per card
  forced_tier_run   <RUN>.new/queue_<S>.txt    every card of the site's (tier, Stokes) served exactly once, as a byte copy of
                                               the run's fitted card, with the card's numbers; radio position only; queue
                                               identities = catalogue identities; an astrometry figure for every multi-
                                               detection identity; one association row per queued identity
"""
import csv
import glob
import hashlib
import json
import os
import re

from exopipe.phase3.core.validate import content_test


def _t():
    from exopipe.phase3.config import CFG
    return CFG.forced_tier_pages


def _rows(p):
    if not os.path.exists(p):
        return []
    with open(p, newline='') as f:
        return list(csv.DictReader(f))


def _detail(bad, n=5):
    return '; '.join(str(b) for b in bad[:n]) + (' ... (%d)' % len(bad) if len(bad) > n else '')


def _sha(p):
    with open(p, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def _key(r):
    return (r['sample'], r['target'], r['tier'], r['kind'], r['stokes'], r['cell'], r['band'], r['ts'])


@content_test('forced_tier_eval')
def tier_eval(path, tier=None, kind=None, samples=None, inp=None, **ctx):
    """``path`` <EVAL>/thresholds.csv of snapshot_forced_eval.py --tier ``tier`` --kind ``kind`` --samples ``samples`` on the
    tier tree ``inp`` (<FT>/<tier>)."""
    t = _t()
    samples = list(samples or t.SAMPLES)
    D = os.path.dirname(os.path.abspath(path))
    bad = []
    th = [r for r in _rows(path) if r.get('tier') == tier]
    for s in samples:
        for S in ('I', 'V'):
            n = sum(1 for r in th if r['sample'] == s and r['stokes'] == S)
            if n != 1:
                bad.append('%d threshold rows of %s Stokes %s %s' % (n, s, S, tier))
    ex = _rows(os.path.join(D, 'exceedance.csv'))
    for s in samples:
        for S in ('I', 'V'):
            for st in ('targets', 'controls'):
                n = sum(int(r['n']) for r in ex if (r['sample'], r['stokes'], r['tier'], r['screen'], r['set'])
                        == (s, S, tier, 'clean', st))
                if n <= 0:
                    bad.append('no clean %s measurement of %s Stokes %s %s' % (st, s, S, tier))
    cells = sorted(r['cell'] for r in _rows(os.path.join(D, 'cells.csv')) if r.get('kind') == kind)
    want = sorted(os.path.basename(d) for d in glob.glob(os.path.join(inp or '', '[0-2][0-9]h')) if os.path.isdir(d))
    if not want or cells != want:
        bad.append('cells %s, the input %s holds %s' % (cells[:6], inp, want[:6]))
    if not os.path.exists(os.path.join(D, 'targets_ge4.csv')):
        bad.append('no targets_ge4.csv (the site lists its events from it)')
    other = sorted({r['sample'] for r in th} - set(samples))
    if other:
        bad.append('thresholds of samples not evaluated: %s' % other)
    if bad:
        return False, len(bad), _detail(bad)
    return True, len(th), '%s %s: thresholds of %s, both Stokes; clean targets and controls of each; cells = the %d LSTs of %s' \
                          % (tier, kind, '/'.join(samples), len(want), inp)


@content_test('forced_tier_site')
def tier_site(path, evals=None, samples=None, **ctx):
    """``evals`` {'<tier>|<kind>': directory}: the evaluations the site must have read (the stage passes those it chose:
    tier_eval's, or forced_tier_pages.EVAL_DIRS; default the latter). ``samples``: the samples the page shows (default
    forced_tier_pages.SAMPLES)."""
    from exopipe.phase3.lib.forced.tier_pages import selected_events
    t = _t()
    samples = tuple(samples or t.SAMPLES)
    want_evs = {tuple(k.split('|')): v for k, v in evals.items()} if evals else dict(t.EVAL_DIRS)
    site = os.path.dirname(os.path.abspath(path))
    with open(path) as f:
        rec = json.load(f)
    bad = []
    cards = rec.get('cards') or []
    if rec.get('n_events') != len(cards):
        bad.append('n_events %s vs %d cards' % (rec.get('n_events'), len(cards)))
    names = [c.get('name') for c in cards]
    if len(set(names)) != len(names):
        bad.append('%d repeated card names' % (len(names) - len(set(names))))
    # the evaluations the build read are the configured ones (EVAL_DIRS mirrors the script's ev_dir)
    evs = {tuple(k.split('|')): v for k, v in ((rec.get('inputs') or {}).get('evals') or {}).items()}
    if evs != want_evs:
        bad.append('the site read evaluations %s, configured %s' % (sorted(evs.items()), sorted(want_evs.items())))
    # the events: exactly the clean rows at or above their Gaussian thresholds in those files, recomputed here
    want = {_key(r) for r in selected_events(t, want_evs, samples)}
    have = {_key(c['row']) for c in cards if 'row' in c}
    if want != have:
        bad.append('events: %d listed, %d at or above threshold in the evaluations (%d listed not selected, %d selected not '
                   'listed)' % (len(have), len(want), len(have - want), len(want - have)))
    # the thresholds on the page are those of the files
    th = {}
    for (tier, kind), ev in sorted(want_evs.items()):
        for r in _rows(os.path.join(ev, 'thresholds.csv')):
            if r['tier'] == tier and r['sample'] in samples:
                th['|'.join((r['sample'], tier, kind, r['stokes']))] = r
    if th != rec.get('thresholds'):
        bad.append('thresholds differ from the evaluation files')
    # every sample is on the page: its thresholds per (tier, kind, Stokes), its exceedance plot per tier (plots/exceedance_
    # <sample>_<tier>.png, :204) and its section per tier and Stokes (<h3 id="<tier>-<S>-<sample>">, :298)
    tiers = sorted({k[0] for k in want_evs})
    nth = ['%s %s %s %s' % (s, ti, k, S) for s in samples for (ti, k) in sorted(want_evs) for S in ('I', 'V')
           if '|'.join((s, ti, k, S)) not in (rec.get('thresholds') or {})]
    if nth:
        bad.append('%d (sample, tier, kind, Stokes) without thresholds on the page, e.g. %s' % (len(nth), nth[0]))
    try:
        with open(os.path.join(site, 'index.html')) as f:
            page = f.read()
    except OSError:
        page = ''
        bad.append('no index.html')
    noplot = [(s, ti) for s in samples for ti in tiers
              if not os.path.exists(os.path.join(site, 'plots', 'exceedance_%s_%s.png' % (s, ti)))
              or 'plots/exceedance_%s_%s.png' % (s, ti) not in page]
    nosec = [(s, ti, S) for s in samples for ti in tiers for S in ('I', 'V') if 'id="%s-%s-%s"' % (ti, S, s) not in page]
    if noplot:
        bad.append('%d (sample, tier) without their exceedance plot on the page, e.g. %s' % (len(noplot), noplot[0]))
    if nosec:
        bad.append('%d (sample, tier, Stokes) without their page section, e.g. %s' % (len(nosec), nosec[0]))
    small, off, noy = [], [], []
    for n in names:
        png = os.path.join(site, 'panels', '%s.png' % n)
        if not os.path.exists(png) or os.path.getsize(png) <= t.PANEL_MIN_BYTES:
            small.append(n)
        if 'panels/%s.png' % n not in page:
            off.append(n)
        if not os.path.exists(os.path.join(site, 'yellow', '%s.npz' % n)):
            noy.append(n)
    for what, L in (('panels missing or at most %d bytes' % t.PANEL_MIN_BYTES, small), ('panels not on the page', off),
                    ('cards without a yellow-number file', noy)):
        if L:
            bad.append('%d %s, e.g. %s' % (len(L), what, L[0]))
    if bad:
        return False, len(bad), _detail(bad)
    return True, len(cards), '%d events = every clean measurement at or above its Gaussian threshold in %d evaluations; ' \
                             'samples %s each with thresholds, exceedance plots and sections; one panel (> %d bytes, on the ' \
                             'page) and one yellow file per card' % (len(cards), len(evs), '/'.join(samples), t.PANEL_MIN_BYTES)


_QH = re.compile(r'^# queue_name: forced photometry \(HWJ \+ 10 pc\), Stokes ([IV]), (\S+) tier\s*$')


@content_test('forced_tier_run')
def tier_run(path, site=None, tier=None, stokes=None, **ctx):
    from exopipe.phase3.lib.forced.tests_snapshot import radio_of
    D = os.path.dirname(os.path.abspath(path))
    with open(path) as f:
        lines = f.read().splitlines()
    m = _QH.match(lines[0]) if lines else None
    if not m:
        return False, None, 'queue header not the family T header: %r' % (lines[:1],)
    S, T = m.group(1), m.group(2)
    bad = []
    if os.path.basename(path) != 'queue_%s.txt' % S or (stokes and stokes != S) or (tier and tier != T):
        bad.append('queue %s says Stokes %s, %s tier (expected %s, %s)' % (os.path.basename(path), S, T, stokes, tier))
    if not site:
        return False, None, 'no site named (the record passes site=)'
    with open(os.path.join(site, 'page_record.json')) as f:
        cards = {c['name']: c for c in json.load(f)['cards'] if c['row']['tier'] == T and c['row']['stokes'] == S}
    fits = {r['cand_id']: r for r in _rows(os.path.join(D, 'forced_fits.csv'))}
    dets = [(os.path.dirname(p), d) for p in sorted(glob.glob(os.path.join(D, 'candidates', S, '*', 'detections.csv')))
            for d in _rows(p)]
    pngs = glob.glob(os.path.join(D, 'candidates', S, '*', '*.png'))
    if len(pngs) != len(dets):
        bad.append('%d served panels vs %d served events' % (len(pngs), len(dets)))
    seen, nrad = set(), 0
    for rd, d in dets:
        c = cards.get(d.get('card'))
        if c is None or d['card'] in seen:
            bad.append('%s: card %s not of the site or served twice' % (d.get('cand_id'), d.get('card')))
            continue
        seen.add(d['card'])
        r = c['row']
        if tuple(d.get(k) for k in ('snr', 'sample', 'target', 'band', 'p_gauss', 'expected_fa')) != \
                tuple(r.get(k) for k in ('snr', 'sample', 'target', 'band', 'p_gauss', 'expected_fa')):
            bad.append('%s: numbers differ from the card' % d['card'])
        p = radio_of(fits.get(d.get('cand_id')))
        if p is None:
            if d.get('ra') or d.get('dec'):
                bad.append('%s: a position served without a radio position' % d['card'])
        else:
            nrad += 1
            if (d.get('ra'), d.get('dec')) != ('%.5f' % p[0], '%.5f' % p[1]):
                bad.append('%s: served position is not its radio position' % d['card'])
        mine = [f for f in os.listdir(rd) if f.endswith('.png') and f.endswith('_%s.png' % d['band'])
                and ('_%s_' % d['lst']) in f]
        card_png = os.path.join(D, 'cards', d['card'] + '.png')
        if not os.path.exists(card_png) or not any(_sha(os.path.join(rd, f)) == _sha(card_png) for f in mine):
            bad.append('%s: no served panel is a byte copy of its card' % d['card'])
    if seen != set(cards):
        bad.append('%d of the site\'s %d %s Stokes %s cards served' % (len(seen), len(cards), T, S))
    q = [l.strip() for l in lines if l.strip() and not l.startswith('#')]
    cat = {r['identity_id'] for p in glob.glob(os.path.join(D, 'source_catalogs_%s' % S, '*_sources_survivors.csv'))
           for r in _rows(p)}
    if len(set(q)) != len(q) or set(q) != cat:
        bad.append('queue identities %d (%d distinct) vs catalogue identities %d' % (len(q), len(set(q)), len(cat)))
    nofig = [L['identity_id'] for L in _rows(os.path.join(D, 'source_catalogs', 'localizations.csv'))
             if int(L.get('n_det') or 0) > 1 and not os.path.exists(os.path.join(D, 'localization_figs',
                                                                                     '%s.png' % L['identity_id']))]
    if nofig:
        bad.append('%d multi-detection identities without an astrometry figure, e.g. %s' % (len(nofig), nofig[0]))
    ai = [r['identity_id'] for r in _rows(os.path.join(D, 'forced_assoc_identity.csv'))]
    if sorted(ai) != sorted(set(q)):
        bad.append('association table %d rows vs %d queued identities' % (len(ai), len(set(q))))
    if bad:
        return False, len(bad), _detail(bad)
    return True, len(dets), 'Stokes %s %s: %d events served (every card once, byte copies), %d with a radio position, %d ' \
                            'identities queued = catalogued = associated' % (S, T, len(dets), nrad, len(q))
