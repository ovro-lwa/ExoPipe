"""Content tests of the combined snapshot view (lib/serve/snapshot_view.py), registered with core/validate.

Classes:
  snapshot_view          the view directory (ctx: kept = the omit product's deep_omission_kept.csv, runs = the batch runs in
                         order, final = after the filter catalogue and the localization). Production's in-build tests T1-T6 run
                         in the build before the rename; these read the written files back:
                         V1 files: the summary json (production's tests line, its name, the batches = the runs' tags), the
                            omission files, the general catalogue, every per-tier view and queue, no <view>.part left; final: the
                            filter catalogue, card2 union, coadd_root.txt and both tiers' localizations.
                         V2 queues (T4): each queue's header (its name and run lines, the VIEW LIMITS line, the DEEP OMISSION line
                            with the summary's counts) and its names = the identities of its tier view, once each, by best |S/N|.
                         V3 pages (T5, T6, the omission): every combined source's page directory holds detections.csv with
                            n_det rows, each of the directory's tier, on no OMIT key and in no omitted image (the per-image rule,
                            exopipe_ops dde1760, read back here: the cells' member tables), none a Card 2 one (the batches' own
                            card2_vleak.csv flags, dbc40a6); its panels carry its tier and no panel's own key and image (parsed
                            here, independently of the build's pattern) is omitted (F1); every member row's detection is in its page
                            directory (serve.SNAPVIEW_TEST_SAMPLE_DIRS > 0: a seeded sample of page directories).
                         V7 panels (F1): each page directory's files are exactly its member batch sources' files (less the
                            sources the kept product drops or re-judges out) less the kept product's 'panel dropped' and 'card2
                            panel dropped' rows (ctx kept, runs) - a panel linked that the omission or Card 2 removed, a kept panel
                            missing (the same page directories as V3).
                         V4 identities (T2 against the kept product): the detections of candidates/_identity/* are exactly the
                            kept product's kept detections (batch, detection key, multiset); the counts of the summary.
                         V5 record: deep_omission.csv = the omit product's, byte for byte, rejudged.csv = its re-judging table;
                            the record's removals = the omit stage's (detections, Card 2 detections, dropped identities; per-tier
                            rows kind for kind - re-judged sources and member rows, Card 2 member rows and panels included), the
                            summary text's page line with the Card 2 / re-judging counts, the summary json's card2 and rejudge
                            counts, its per-instant test (tests_per_instant.check_omission, production's record format).
                         V6 (final) the filter catalogue's per-detection contract (pages' filter_catalog), the
                            localizations' identities are view identities, and the page figures: every figured identity's
                            page directories hold astrometry.png of the figure's size (production left truncated copies,
                            2026-11-04), none without a figure.
  snapshot_card2_union   card2_filter.csv of a view (ctx runs): the union of the member batches' files, recomputed, byte for byte.
Every test reads content and returns (passed, value, detail).
"""
import collections
import csv
import glob
import json
import os
import random
import re

from exopipe.phase3.core.validate import content_test

csv.field_size_limit(10 ** 9)


def _SV():
    from exopipe.phase3.lib.serve import snapshot_view as SV
    return SV


def _DO():
    from exopipe.phase3.lib.catalog import deep_omission as DO
    return DO


def _S():
    from exopipe.phase3.config import CFG
    return CFG.serve


def _fail(msgs, value, what):
    if msgs:
        return False, len(msgs), '; '.join(msgs[:6]) + (f' (+{len(msgs) - 6} more)' if len(msgs) > 6 else '') + ' | ' + what
    return True, value, what


def _summary(view):
    p = os.path.join(view, 'combined_view_summary.json')
    with open(p) as f:
        return json.load(f)


def _tiers():
    S = _S()
    return [(st, t) for st in S.SNAPVIEW_STOKES for t in S.SNAPVIEW_TIERS]


@content_test('snapshot_view', name='files')
def view_files(path, runs=None, final=False, **ctx):
    """V1: the view's files (module docstring)."""
    SV, DO = _SV(), _DO()
    msgs = []
    if not os.path.isdir(path):
        return False, None, f'no view directory {path}'
    if os.path.exists(path.rstrip('/') + '.part'):
        msgs.append(f'{path}.part is left beside the view')
    try:
        summ = _summary(path)
    except (OSError, ValueError) as e:
        return False, None, f'combined_view_summary.json unreadable: {e}'
    if summ.get('tests') != SV.TESTS_PASSED:
        msgs.append(f'summary tests {summ.get("tests")!r} != {SV.TESTS_PASSED!r}')
    if summ.get('out') != os.path.basename(path.rstrip('/')):
        msgs.append(f'summary out {summ.get("out")!r} is not the view\'s name')
    if runs is not None and summ.get('batches') != [DO.batch_tag(r) for r in runs]:
        msgs.append(f'summary batches {summ.get("batches")} != the runs\' tags')
    need = ['deep_omission.csv', 'deep_omission_dropped.csv', 'deep_omission_summary.txt', 'rejudged.csv', 'judgements']
    need += [f'queue_{s}_{t}.txt' for s, t in _tiers()] + [f'source_catalogs_{s}_{t}' for s, t in _tiers()]
    if final:
        need += ['master_catalog.csv', 'card2_filter.csv', 'coadd_root.txt']
        need += [os.path.join(f'source_catalogs_loc_{t}', 'localizations.csv') for t in _S().SNAPVIEW_TIERS]
    msgs += [f'{x} missing' for x in need if not os.path.exists(os.path.join(path, x))]
    if not glob.glob(os.path.join(path, 'source_catalogs', '*_sources_survivors.csv')):
        msgs.append('no general catalogue source_catalogs/*_sources_survivors.csv')
    return _fail(msgs, len(need), f'{len(need)} required files and directories present; summary {summ.get("combined_identities")} '
                                  f'combined identities of {summ.get("batches") and len(summ["batches"])} batches')


def _cat_rows(view, S, T, kind):
    out = []
    for p in sorted(glob.glob(os.path.join(view, f'source_catalogs_{S}_{T}', f'*_{kind}.csv'))):
        band = os.path.basename(p)[: -len(f'_{kind}.csv')]
        with open(p, newline='') as f:
            out += [(band, r) for r in csv.DictReader(f)]
    return out


@content_test('snapshot_view', name='queues')
def view_queues(path, **ctx):
    """V2: each queue's header and names (T4 read back)."""
    SV = _SV()
    summ = _summary(path)
    dom = summ.get('deep_omission') or {}
    msgs, n = [], 0
    for S, T in _tiers():
        q = os.path.join(path, f'queue_{S}_{T}.txt')
        with open(q) as f:
            lines = f.read().splitlines()
        head = [ln for ln in lines if ln.startswith('#')]
        names = [ln.strip() for ln in lines if ln.strip() and not ln.startswith('#')]
        nwb = dom.get('omit_wb_image_keys', 0)
        want = [SV.QUEUE_NAME % (S, T, len(summ.get('batches') or [])),
                SV.QUEUE_RUN % (summ.get('out'), ' '.join(summ.get('batches') or [])), SV.VIEW_LIMITS,
                SV.DEEP_OMISSION_LINE % (dom.get('omitted', -1), dom.get('served', -1), dom.get('omit_keys', -1),
                                         SV.DEEP_OMISSION_WB % nwb if nwb else ''),
                SV.CARD2_REJUDGE_LINE % ((summ.get('card2') or {}).get('removed_from_view', -1),
                                         (summ.get('rejudge') or {}).get('rejudged_out', -1))]
        if [h + '\n' for h in head[:5]] != want:
            msgs.append(f'queue_{S}_{T}: header lines {head[:5]} are not the view\'s (name, run, VIEW LIMITS, DEEP OMISSION, CARD 2 '
                        f'AND RE-JUDGING)')
        best = collections.defaultdict(float)
        for _band, r in _cat_rows(path, S, T, 'sources_survivors'):
            best[r['identity_id']] = max(best[r['identity_id']], abs(float(r.get('best_snr') or 0)))
        if len(names) != len(set(names)):
            msgs.append(f'queue_{S}_{T}: {len(names) - len(set(names))} names listed twice')
        if set(names) != set(best):
            msgs.append(f'queue_{S}_{T}: {len(set(names) - set(best))} names of no source of the tier view, '
                        f'{len(set(best) - set(names))} identities of the view not queued')
        elif names != sorted(best, key=lambda x: (-best[x], x)):
            msgs.append(f'queue_{S}_{T}: names not in best |S/N| order')
        if summ.get('queues', {}).get(f'{S}_{T}', 0) != len(names):
            msgs.append(f'queue_{S}_{T}: {len(names)} names, the summary says {summ.get("queues", {}).get(f"{S}_{T}")}')
        n += len(names)
    return _fail(msgs, n, f'{n} queued identities in {len(_tiers())} queues; headers, coverage and order')


# THE PANEL'S OWN KEY (V_combview review F1), parsed HERE - not with deep_omission.SNAP_PANEL, the pattern the kept product and the
# build both use: a mistake in that shared helper would make them agree on the wrong panels, unseen. A panel file is
# [<batch tag>_]rank<N>_<LST>_<night>_<band><14-digit frame time>[1min].(png|page.json) (the page code's PANEL_TOKEN, with the
# view's clash prefix); its key is (LST, band, night) with its page directory's Stokes and tier.
PANEL_FILE = re.compile(r'^(?:[A-Za-z0-9]+_)?rank\d+_(\d\dh)_(\d{4}-\d\d-\d\d)_([A-Za-z0-9]+)\.(?:png|page\.json)$')
PANEL_BAND = re.compile(r'^([A-Za-z0-9]*?[A-Za-z])(\d{14})(1min)?$')


def panel_image(fn):
    """The image id (YYYYMMDD_HHMMSS) of a panel file name, or None."""
    m = PANEL_FILE.match(fn)
    t = PANEL_BAND.match(m.group(3)) if m else None
    return (t.group(2)[:8] + '_' + t.group(2)[8:]) if t else None


def panel_key(fn, S, T):
    """(lst, band, night, Stokes, tier) of a panel file name in a page directory of Stokes S, tier T; None for a file that is no
    panel name (counted, never judged)."""
    m = PANEL_FILE.match(fn)
    t = PANEL_BAND.match(m.group(3)) if m else None
    return (m.group(1), t.group(1), m.group(2), S, T) if t else None


class _Removed:
    """The omission (whole-night and per image) and Card 2 READ BACK HERE (V_combview review T4 / T8 - not deep_omission's
    Omission / Card2, which the kept product and the build share): the decisions of the view, the cells' member tables
    (tests_deep_omission.ImageCheck; the fleet root of the omit product's summary beside ctx kept), the batches' Card 2 flags
    (ctx runs). image: a detection's ts, a panel's 14 digits as YYYYMMDD_HHMMSS."""

    def __init__(self, path, kept=None, runs=None):
        DO = _DO()
        from exopipe.phase3.config import CFG
        from exopipe.phase3.lib.catalog import tests_deep_omission as TD
        with open(os.path.join(path, 'deep_omission.csv'), newline='') as f:
            dec = list(csv.DictReader(f))
        self.OM, self.OMWB = DO.omitted_keys(dec), DO.omitted_wb(dec)
        proto = None
        if kept and os.path.isfile(os.path.join(os.path.dirname(os.path.abspath(kept)), 'deep_omission_summary.json')):
            proto = json.load(open(os.path.join(os.path.dirname(os.path.abspath(kept)), 'deep_omission_summary.json'))).get('proto')
        self.ic = TD.ImageCheck(proto or CFG.paths.SNAPSHOT_FLEET_OUT, CFG.catalog_omission.SNAPSHOT_MEMBERS)
        self.flagged = set()
        for _tag, keys in TD.card2_flagged(runs or [], CFG.catalog_omission.SNAPSHOT_CARD2, DO.batch_tag).items():
            self.flagged |= keys
        self.has_card2 = bool(runs)

    def key(self, k, image):
        if k in self.OM:
            return True
        return k in self.OMWB and self.ic.holds(k, image, self.OMWB[k]) is not False

    def det(self, d):
        DO = _DO()
        return self.key(DO.okey(d), d['ts'])

    def card2(self, d):
        return d.get('tier') == '10s' and d.get('stokes') == 'V' and (
            d['cand_id'], '%.5f' % float(d.get('ra_peak') or d['ra']), '%.5f' % float(d.get('dec_peak') or d['dec'])) in self.flagged


@content_test('snapshot_view', name='pages')
def view_pages(path, kept=None, runs=None, **ctx):
    """V3: page directories (T5, T6 read back), no omitted (per image, dde1760) or Card 2 (dbc40a6) detection on a page, and
    no panel of an omitted instant or image on a page (F1: each panel's own key, parsed independently of the build)."""
    SV, DO = _SV(), _DO()
    S_ = _S()
    RM = _Removed(path, kept, runs)
    srcs = []
    for S, T in _tiers():
        srcs += [(S, T, band, r) for band, r in _cat_rows(path, S, T, 'sources_survivors')]
    sample = S_.SNAPVIEW_TEST_SAMPLE_DIRS
    if sample and len(srcs) > sample:
        srcs = random.Random(20261103).sample(srcs, sample)
    msgs, n_rows, members_pd = [], 0, collections.defaultdict(set)
    n_panels = n_unparsed = 0
    want_pd = {r['review_dir'] for _S, _T, _b, r in srcs}
    for S, T in _tiers():
        for band, r in _cat_rows(path, S, T, 'source_members'):
            mm = SV.MEMBER_ROW.match(r['source_id'])
            if not mm:
                msgs.append(f'{S} {T} member row {r["source_id"]}: no <identity>_<cell>_<band> source_id')
                continue
            pd = SV.page_dir(S, T, f'{mm.group(1)}_{band}', S_.SNAPVIEW_PAGE_SUFFIX)
            if pd in want_pd:
                members_pd[pd].add(r['cand_id'])
    for S, T, band, r in srcs:
        pd = r['review_dir']
        if pd != SV.page_dir(S, T, r['source_id'], S_.SNAPVIEW_PAGE_SUFFIX):
            msgs.append(f'{S} {T} {r["source_id"]}: review_dir {pd} is not its page directory')
        d = os.path.join(path, pd)
        try:
            with open(os.path.join(d, 'detections.csv'), newline='') as f:
                dets = list(csv.DictReader(f))
            names = os.listdir(d)
        except OSError as e:
            msgs.append(f'{pd}: {e}')
            continue
        n_rows += len(dets)
        if str(len(dets)) != str(r.get('n_det')):
            msgs.append(f'{pd}: {len(dets)} detections, n_det {r.get("n_det")}')
        tier = SV.dir_tier(pd, S_.SNAPVIEW_PAGE_SUFFIX)
        if tier != T:
            msgs.append(f'{pd}: a {T} source in a {tier} directory')
        bad_t = [x['cand_id'] for x in dets if x.get('tier') != tier]
        bad_o = [x['cand_id'] for x in dets if RM.det(x)]
        bad_c2 = [x['cand_id'] for x in dets if RM.card2(x)]
        bad_p = [fn for fn in names if SV.panel_tier(fn) not in (None, tier)]
        if bad_t or bad_p:
            msgs.append(f'T6 {pd}: {len(bad_t)} rows and {len(bad_p)} panels of another tier (e.g. {(bad_t + bad_p)[:1]})')
        if bad_o:
            msgs.append(f'{pd}: {len(bad_o)} detections on an OMIT key or image (e.g. {bad_o[0]})')
        if bad_c2:
            msgs.append(f'T7 {pd}: {len(bad_c2)} detections Card 2 removes (e.g. {bad_c2[0]})')
        keys = [(fn, panel_key(fn, S, T)) for fn in names if fn not in ('detections.csv', 'astrometry.png')]
        n_panels += sum(1 for _fn, k in keys if k)
        n_unparsed += sum(1 for _fn, k in keys if not k)
        bad_pk = [fn for fn, k in keys if k and RM.key(k, panel_image(fn))]
        if bad_pk:
            msgs.append(f'F1 {pd}: {len(bad_pk)} panels of an omitted instant are served (e.g. {bad_pk[0]}) - the omission removed '
                        f'their detections')
        miss = members_pd.get(pd, set()) - {x['cand_id'] for x in dets}
        if miss:
            msgs.append(f'T5 {pd}: {len(miss)} member detections not in its detections.csv (e.g. {sorted(miss)[0]})')
    return _fail(msgs, len(srcs), f'{len(srcs)} page directories ({"a seeded sample" if sample else "every one"}), {n_rows} '
                                  f'detections, {n_panels} panels ({n_unparsed} other files): one tier each, n_det, no detection '
                                  f'or panel on an OMIT key or image, no Card 2 detection{"" if RM.has_card2 else " (no runs given)"}, '
                                  f'every member detection present')


@content_test('snapshot_view', name='panels')
def view_panels(path, kept=None, runs=None, **ctx):
    """V7 (V_combview review F1): every page directory's files are exactly its member batch sources' files minus the kept
    product's 'panel dropped' rows - no panel the omission removed is linked, no kept panel is missing (production's T3 counts
    links in the loop that decides them, so it cannot see a wrong decision). A page's member batch sources: the general
    catalogue's batch_members ('<batch>:<identity>'), each such batch identity's per-tier survivors row of the page's band (its
    batch's source_catalogs_<S>_<T>), less the sources the kept product drops; their page directories are listed now. A name
    clash ('<batch>_<file>', production's rule) counts as the file. The same page directories as V3 (all, or the seeded sample)."""
    DO, SV = _DO(), _SV()
    S_ = _S()
    if not kept or not os.path.isfile(kept) or not runs:
        return False, None, f'no kept product or batch runs given (ctx kept = {kept!r}, runs = {runs!r}): the panels cannot be traced'
    tag2run = {DO.batch_tag(r): os.path.abspath(r) for r in runs}
    # batch_members = '+'.join('<batch>:<identity>'): a J-name holds '+' itself, so split only at a '+' that starts '<batch>:'
    split_bm = re.compile(r'\+(?=(?:%s):)' % '|'.join(re.escape(t) for t in sorted(tag2run, key=len, reverse=True)))
    members = {}
    for p in glob.glob(os.path.join(path, 'source_catalogs', '*_sources_survivors.csv')):
        with open(p, newline='') as f:
            for r in csv.DictReader(f):
                members[r['source_id']] = [tuple(x.split(':', 1)) for x in split_bm.split(r.get('batch_members') or '')
                                           if ':' in x]
    src_dropped, pan_dropped = set(), collections.Counter()
    with open(kept, newline='') as f:
        for r in csv.DictReader(f):
            if r['kind'] in ('source dropped', 'source re-judged'):
                src_dropped.add((r['run'], r['catalog'], r['source_id']))
            elif r['kind'] in ('panel dropped', 'card2 panel dropped'):
                pan_dropped[(r['run'], r['catalog'], r['source_id'], r['stokes'], r['tier'], r['note'])] += 1
    srcs = []
    for S, T in _tiers():
        srcs += [(S, T, band, r) for band, r in _cat_rows(path, S, T, 'sources_survivors')]
    sample = S_.SNAPVIEW_TEST_SAMPLE_DIRS
    if sample and len(srcs) > sample:
        srcs = random.Random(20261103).sample(srcs, sample)
    bcache, msgs, n_files = {}, [], 0
    for S, T, band, r in srcs:
        cdir = f'source_catalogs_{S}_{T}'
        catalog = f'{cdir}/{band}_sources_survivors.csv'
        want = collections.Counter()
        for tag, J in members.get(r['identity_id'], []):
            run = tag2run.get(tag)
            if run is None:
                msgs.append(f'{r["review_dir"]}: member batch {tag} is no run given')
                continue
            if (run, cdir, band) not in bcache:
                by = collections.defaultdict(list)
                bp = os.path.join(run, cdir, f'{band}_sources_survivors.csv')
                if os.path.exists(bp):
                    with open(bp, newline='') as f:
                        for x in csv.DictReader(f):
                            by[x['identity_id']].append(x)
                bcache[(run, cdir, band)] = by
            for x in bcache[(run, cdir, band)].get(J, []):
                if (run, catalog, x['source_id']) in src_dropped:
                    continue
                for fn in os.listdir(os.path.join(run, x['review_dir'])):
                    if fn in ('detections.csv', 'astrometry.png'):
                        continue
                    if pan_dropped[(run, x['review_dir'], x['source_id'], S, T, fn)] > 0:
                        continue
                    want[fn] += 1
        have = [fn for fn in os.listdir(os.path.join(path, r['review_dir'])) if fn not in ('detections.csv', 'astrometry.png')]
        n_files += len(have)
        extra = []
        for fn in sorted(have, key=lambda x: (not x.startswith('rank'), x)):     # the plain names first, then the clash names
            pre, _, rest = fn.partition('_')
            if want[fn] > 0:
                want[fn] -= 1
            elif pre in tag2run and want[rest] > 0:
                want[rest] -= 1
            else:
                extra.append(fn)
        miss = sorted(k for k, v in want.items() if v > 0)
        if extra or miss:
            msgs.append(f'F1 {r["review_dir"]}: {len(extra)} files no member batch source keeps (e.g. {extra[:1]}), {len(miss)} of its '
                        f'member batch sources\' kept panels missing (e.g. {miss[:1]})')
    return _fail(msgs, n_files, f'{len(srcs)} page directories, {n_files} files: exactly their member batch sources\' files less '
                                f'the kept product\'s panel drops')


def _kept_rows(kept):
    with open(kept, newline='') as f:
        for r in csv.DictReader(f):
            if r['kind'] == 'detection kept':
                yield r


@content_test('snapshot_view', name='identities')
def view_identities(path, kept=None, **ctx):
    """V4: the identities' detections are the kept product's kept detections, exactly (T2 read back against the kept product)."""
    DO = _DO()
    if not kept or not os.path.isfile(kept):
        return False, None, f'no kept product given (ctx kept = {kept!r}): the view\'s detections cannot be traced to the drop'
    summ = _summary(path)
    have = collections.Counter()
    n_dirs = 0
    for p in glob.glob(os.path.join(path, 'candidates', '_identity', '*', 'detections.csv')):
        n_dirs += 1
        with open(p, newline='') as f:
            for d in csv.DictReader(f):
                have[(d.get('batch_run', ''),) + DO.dkey(d)] += 1
    want = collections.Counter()
    ids = set()
    for r in _kept_rows(kept):
        want[(r['batch'], r['cand_id'], r['ra_peak'], r['dec_peak'])] += 1
        ids.add((r['batch'], r['identity']))
    msgs = []
    if have != want:
        msgs.append(f'T2: {sum((have - want).values())} identity detections not kept by the kept product, '
                    f'{sum((want - have).values())} kept detections in no identity')
    if sum(have.values()) != summ.get('detections'):
        msgs.append(f'{sum(have.values())} identity detections, the summary says {summ.get("detections")}')
    if n_dirs != summ.get('combined_identities'):
        msgs.append(f'{n_dirs} identity directories, the summary says {summ.get("combined_identities")} combined identities')
    if len(ids) != summ.get('batch_identities'):
        msgs.append(f'{len(ids)} batch identities kept, the summary says {summ.get("batch_identities")}')
    gen = set()
    for p in glob.glob(os.path.join(path, 'source_catalogs', '*_sources_survivors.csv')):
        with open(p, newline='') as f:
            gen |= {r['review_dir'] for r in csv.DictReader(f)}
    dirs = {os.path.relpath(os.path.dirname(p), path) for p in glob.glob(os.path.join(path, 'candidates', '_identity', '*',
                                                                                            'detections.csv'))}
    if gen != dirs:
        msgs.append(f'{len(gen - dirs)} general catalogue rows without their identity directory, {len(dirs - gen)} directories '
                    f'of no row')
    return _fail(msgs, sum(have.values()), f'{sum(have.values())} detections in {n_dirs} combined identities = the kept product\'s '
                                           f'kept detections of {len(ids)} batch identities')


@content_test('snapshot_view', name='record')
def view_record(path, kept=None, **ctx):
    """V5: the view's omission files against the omit stage's (module docstring)."""
    import filecmp
    DO = _DO()
    from exopipe.phase3.lib.catalog import tests_per_instant as PI
    if not kept or not os.path.isfile(kept):
        return False, None, f'no kept product given (ctx kept = {kept!r}): the record cannot be traced to the omit stage'
    od = os.path.dirname(os.path.abspath(kept))
    msgs = []
    if not filecmp.cmp(os.path.join(path, 'deep_omission.csv'), os.path.join(od, 'deep_omission.csv'), shallow=False):
        msgs.append('deep_omission.csv differs from the omit stage\'s decision table')
    with open(os.path.join(path, 'deep_omission_dropped.csv'), newline='') as f:
        rec = list(csv.DictReader(f))
    with open(os.path.join(od, 'deep_omission_dropped.csv'), newline='') as f:
        orec = list(csv.DictReader(f))
    kp = collections.Counter()
    with open(kept, newline='') as f:
        for r in csv.DictReader(f):
            kp[r['kind']] += 1
    det = lambda rr, kd='detection': collections.Counter((r['batch'], r['identity'], r['cand_id']) for r in rr if r['kind'] == kd)
    if det(rec) != det(orec):
        msgs.append('the record\'s omitted detections are not the omit stage\'s')
    if det(rec, 'card2 detection') != det(orec, 'card2 detection'):
        msgs.append('the record\'s Card 2 detections are not the omit stage\'s')
    ids = lambda rr: {(r['batch'], r['identity']) for r in rr if r['kind'] == 'identity dropped'}
    if ids(rec) != ids(orec):
        msgs.append('the record\'s dropped identities are not the omit stage\'s')
    k = collections.Counter(r['kind'] for r in rec)
    for mine, theirs in (('source dropped', 'source dropped'), ('member row', 'member row dropped'), ('panel', 'panel dropped'),
                         ('source partial', 'source re-pointed'), ('source re-judged', 'source re-judged'),
                         ('member row re-judged', 'member row re-judged'), ('card2 member row', 'card2 member row dropped'),
                         ('card2 panel', 'card2 panel dropped')):
        if k.get(mine, 0) != kp.get(theirs, 0):
            msgs.append(f'{k.get(mine, 0)} {mine!r} rows, the kept product has {kp.get(theirs, 0)} {theirs!r}')
    summ = _summary(path)
    n_det = k.get('detection', 0)
    with open(os.path.join(path, 'deep_omission_summary.txt')) as f:
        lines = f.read().splitlines()
    SV = _SV()
    want1 = DO.page_line(n_det) + SV.SUMMARY_LINE1_NEW % ((summ.get('card2') or {}).get('removed_from_view', -1),
                                                          (summ.get('rejudge') or {}).get('rejudged_out', -1))
    if not lines or lines[0] != want1:
        msgs.append(f'summary text line 1 {lines[:1]} is not the page line of {n_det} detections with the Card 2 / re-judging '
                    f'counts')
    if (summ.get('card2') or {}).get('removed_from_view') != k.get('card2 detection', 0):
        msgs.append(f'summary json card2 {summ.get("card2")} does not count the record\'s {k.get("card2 detection", 0)} Card 2 '
                    f'detections')
    if (summ.get('rejudge') or {}).get('per_tier_rows_dropped') != k.get('source re-judged', 0):
        msgs.append(f'summary json rejudge {summ.get("rejudge")} does not count the record\'s {k.get("source re-judged", 0)} '
                    f're-judged source rows')
    if not filecmp.cmp(os.path.join(path, 'rejudged.csv'), os.path.join(od, 'deep_omission_rejudged.csv'), shallow=False):
        msgs.append('rejudged.csv differs from the omit stage\'s re-judging table')
    dom = summ.get('deep_omission') or {}
    if (dom.get('omitted'), dom.get('record')) != (n_det, dict(k)):
        msgs.append(f'summary json deep_omission {dom} does not count the record ({dict(k)})')
    ok, val, detail = PI.check_omission(os.path.join(path, 'deep_omission_dropped.csv'), 'snapshot',
                                        os.path.join(path, 'deep_omission.csv'))
    if not ok:
        msgs.append(f'per-instant: {detail[:300]}')
    return _fail(msgs, n_det, f'{len(rec)} record rows ({", ".join(f"{a} {b}" for a, b in sorted(k.items()))}) = the omit stage\'s '
                              f'drop; decisions byte-identical; per-instant {val}')


@content_test('snapshot_view', name='final')
def view_final(path, final=False, **ctx):
    """V6: after the filter catalogue and the localization (final=True): the per-detection filter catalogue, the localized
    identities are view identities; THE PAGE FIGURES - every identity localized with more than one detection in a tier has its
    figure (source_catalogs_loc_<T>/figs/<identity>.png, snapshot_localize_tiers.sh), each of its page directories of that tier
    holds astrometry.png of the figure's size, and a page of an identity without a figure holds none. Found by the V2 follow-up's
    real-data parity (2026-11-04): production's localizer left 7 of 11,245 page copies TRUNCATED at 64 KiB multiples (0, 65,536,
    131,072 bytes; shutil.copyfile in its figure pool) without an error, the figures themselves complete. Sizes, not bytes: a
    truncated copy is shorter; reading every copy of a full view would cost more than the build."""
    if not final:
        return True, 0, 'not the final view (the filter catalogue and the localization come later)'
    from exopipe.phase3.lib.pages import tests as PT
    msgs = []
    ok, n, detail = PT.filter_rows(os.path.join(path, 'master_catalog.csv'), per_detection=True)
    if not ok:
        msgs.append(f'filter catalogue: {detail[:300]}')
    gen = set()
    for p in glob.glob(os.path.join(path, 'source_catalogs', '*_sources_survivors.csv')):
        with open(p, newline='') as f:
            gen |= {r['source_id'] for r in csv.DictReader(f)}
    nloc = 0
    for T in _S().SNAPVIEW_TIERS:
        with open(os.path.join(path, f'source_catalogs_loc_{T}', 'localizations.csv'), newline='') as f:
            loc = {r['identity_id'] for r in csv.DictReader(f)}
        nloc += len(loc)
        if loc - gen:
            msgs.append(f'{T}: {len(loc - gen)} localized identities are no view identity (e.g. {sorted(loc - gen)[0]})')
    nfig, npage = page_figures(path, msgs)
    return _fail(msgs, n, f'filter catalogue {detail}; {nloc} localizations of view identities; {nfig} figures, {npage} page '
                          f'copies of their size')


def page_figures(path, msgs):
    """V6's page-figure check (view_final): (figures, page copies checked); problems appended to ``msgs``."""
    nfig = npage = 0
    for T in _S().SNAPVIEW_TIERS:
        loc = os.path.join(path, f'source_catalogs_loc_{T}')
        figs = os.path.join(loc, 'figs')
        sizes = {fn[:-4]: os.path.getsize(os.path.join(figs, fn)) for fn in os.listdir(figs) if fn.endswith('.png')} \
            if os.path.isdir(figs) else {}
        nfig += len(sizes)
        with open(os.path.join(loc, 'localizations.csv'), newline='') as f:
            multi = {r['identity_id'] for r in csv.DictReader(f) if int(r.get('n_det') or 0) > 1}
        if multi - set(sizes):
            msgs.append(f'{T}: {len(multi - set(sizes))} identities localized with more than one detection have no figure '
                        f'(e.g. {sorted(multi - set(sizes))[0]})')
        short, stale, missing = [], [], []
        for p in sorted(glob.glob(os.path.join(path, f'source_catalogs_[IV]_{T}', '*_sources_survivors.csv'))):
            with open(p, newline='') as f:
                for r in csv.DictReader(f):
                    i = (r.get('identity_id') or '').strip() or r['source_id']
                    a = os.path.join(path, r['review_dir'], 'astrometry.png')
                    try:
                        sz = os.path.getsize(a)
                    except OSError:
                        sz = None
                    if i in sizes:
                        npage += 1
                        if sz is None:
                            missing.append(r['review_dir'])
                        elif sz != sizes[i]:
                            short.append(f"{r['review_dir']} ({sz} of {sizes[i]} bytes)")
                    elif sz is not None:
                        stale.append(r['review_dir'])
        for what, xs in (('page copies of another size than their figure (truncated)', short),
                         ('pages of a figured identity without astrometry.png', missing),
                         ('pages of an identity without a figure holding an astrometry.png', stale)):
            if xs:
                msgs.append(f'{T}: {len(xs)} {what}, e.g. {xs[:3]}')
    return nfig, npage


@content_test('snapshot_card2_union')
def card2_union_exact(path, runs=None, **ctx):
    """card2_filter.csv of a combined view: the union of its member batches' files (comb_build_20261102.sh:53-69), recomputed
    from ``runs`` and equal byte for byte."""
    if not runs:
        return False, None, 'no member runs given (ctx runs): the union cannot be recomputed'
    first, hdr, seen, out = None, None, set(), []
    for r in runs:
        with open(os.path.join(r, 'card2_filter.csv')) as f:
            lines = f.read().splitlines()
        com = [ln for ln in lines if ln.startswith('#')]
        body = [ln for ln in lines if not ln.startswith('#')]
        if first is None:
            first, hdr = com, body[0]
        elif body[0] != hdr:
            return False, None, f'{r}: card2_filter.csv header differs'
        for ln in body[1:]:
            if ln not in seen:
                seen.add(ln)
                out.append(ln)
    view = os.path.dirname(os.path.abspath(path))
    want = '\n'.join(first + ["# combined view %s: the union of its %d member batches' card2_filter.csv"
                              % (os.path.basename(view), len(runs)), hdr] + out) + '\n'
    with open(path) as f:
        have = f.read()
    return (have == want, len(out), f'{len(out)} rows: the union of {len(runs)} member batches'
            + ('' if have == want else ' - the file differs from the recomputed union'))
