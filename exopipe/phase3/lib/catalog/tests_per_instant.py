"""The per-instant veto rule as a content test, every tier (FOLD_IN 2026-10-18 05:0x; plan/CUTOVER_READINESS.md "Gregg's
per-instant veto rule ... every tier's removal steps are audited against it").

Gregg 2026-10-18, verbatim (R-GREGG-20261018-VETOES-PER-FRAME-INDEPENDENTLY): "All frames should pass through vetoes
independently. Otherwise, for example, a single VHF reflection could take out a 200x detected transient."; refined
(R-GREGG-20261018-VETOES-PER-FRAME-REFINED-FREQUENCY-YES-TIME-NO): "Vetoing across frequency for a single frame is probably
okay, but not across time". A veto measured at one INSTANT removes that instant's detections in any band, subband, wideband
and Stokes - never another instant's, never another tier's. The instant of a detection, per tier:
    10-s        (cell, '10s', integration label)      the snapshot record's instant_key (clv15 phase3_snapshot_chain.instant_of)
    1-minute    (cell, '1min', integration minute)     the same record
    10-minute   (LST, night, window)                  the row's date '<night>T<kk>' (cadence 10min)
    deep        (LST, night)                          the row's lst and date (cadence deep)
The content test (FOLD_IN: "no removal reason names another instant") is registered as 'per_instant' on the class that holds
each tier's removal record, and refuses the product on any finding:

  conformant_catalog (deep and 10-minute: master_catalog.csv with master_catalog_removed.csv and master_catalog_asrun.csv
  beside, as ops/p3s2_conform.py and the B-02 check write them), fork_catalog (the deep multiples fork's own conformance
  record, <run>/fork_<S>/) and rejected_record (rejected.csv: the record the 10-minute pages read, emit_rejected_record.py -
  its reason is the removal_reason cut to 160 characters):
    T1 every removed row is of the record's tier (cadence deep / 10min, the date's form) and carries its own instant;
    T2 every removal reason is of a DECLARED kind (REASON_KINDS below, each with the instant it names and the tiers it is
       declared for) - an undeclared reason refuses (a new removal step declares its instant first), and so does the
       conformance's night-keyed noise-sanity removal in the 10-minute tier (inert there today; keyed by night it would
       remove every window of the night - FOLD_IN 2026-10-18: per window only);
    T3 a Jupiter-track removal names a co-added night (jupiter_track_night, or the reason's words): another instant's
       Jupiter. It stands only for the MASTER's imprint - the master co-adds that night's Jupiter, so this night's own
       residual carries minus Jupiter there (ops/p3s2_conform.py, the 2026-10-18 deep audit: "measured in the master, not on
       another night's detection") - i.e. only in a tier whose masters hold every night's Jupiter: deep while
       masters_build.JUPITER_MASK_DEEP is False (R-DEEP-MASTER-JUPITER-MASK-DEFERRED; plan/specs/masters.md amendment
       2026-10-18 "the deep tier keeps the track until its masters are rebuilt this way"; DECISIONS_PENDING "two cross-night
       removals kept"). In the 10-minute tier the window masters carry the moving mask (RULING-MOVING-JUPITER-MASK-IN-MASTERS):
       a track removal naming another night refuses;
    T4 a pulsar-group removal ('multi-detection source inconsistent with <PSR> ... (n=N)') is decided by ONE instant's group
       (a2d37b9: across-instant groups may only rescue). With the pulsar stage's group record beside (B-02 check:
       pulsar_groups.csv) exactly: the row is a member of an inconsistent 'per-epoch' group of that pulsar whose members are
       all of its instant. Without it (production's p3s2_conform), the necessary condition: N <= the detections of the row's
       own instant within catalog_conform.PULSAR_PRECUT_DEG of the pulsar in the as-run record (less the rows the daytime gate
       and the exclusions take, which never join a group) - a group of more members than its instant holds spans instants.
       Where the record cut the reason before '(n=N)' (rejected.csv), N >= 2 (a group has two members).
  snapshot_run (10-s and 1-minute; the chain run directory):
    S1 vetoes applied per SOURCE refuse: a run whose vetoed_sources.csv holds negative-lobe / reflection-asymmetry removals
       without the per-instant record (chain_summary per_instant_vetoes, vetoed_detections.csv) - the rule before clv15
       99fe3ea, where one member's measurement removed every instant of its source;
    S2 every vetoed_detections.csv row's instant is its own cell and tier, and the member the veto measured (neg_member /
       asym_member) is of the same cell and tier and - by the per-detection record (detection_record.csv.gz) - of the same
       instant;
    S3 every detection the record says a veto removed (first_removed_by neg-lobe / asymmetry) carries that veto's 'remove'
       verdict for ITS OWN instant (gate_neg_lobe / gate_asymmetry), and its instant's measured member is at its instant;
       the record and vetoed_detections.csv agree on the number each veto removed. A run whose vetoes removed detections
       without the record refuses (nothing places them at their instants).
    The 3-D bright-source gate, the seed and floor rules and the routes are SOURCE-level selection rules (the source stands or
    falls on its remaining detections, the ruling's own words), not vetoes measured at an instant: not judged here (their
    across-instant reading is DECISIONS_PENDING "the snapshot bright-source gate's offset rule").
Every test reads the files' content and returns (passed, value, detail); imports are inside the tests.
"""
import csv
import gzip
import json
import math
import os
import re

from exopipe.phase3.core.validate import content_test

csv.field_size_limit(10 ** 9)

TIER_CADENCE = {'deep': 'deep', 'sublst': '10min'}
# the pulsar-group words (ops/p3s2_conform.py; removal_rules.pulsar_consistency): 'table: multi-detection source inconsistent
# with <PSR> - <sep> arcsec from its combined position, outside the 99% radius <r99> arcsec (n=<N>): background ...'. The kind
# is the phrase alone and the pulsar its first field: rejected.csv cuts the reason at 160 characters, before '(n=N)'.
PULSAR_GROUP_RE = re.compile(r'multi-detection source inconsistent with (?P<psr>.+?) - \d+ arcsec')
PULSAR_GROUP_N_RE = re.compile(r'outside the 99% radius \d+ arcsec \(n=(?P<n>\d+)\)')
JUPITER_NIGHT_RE = re.compile(r'co-added night (\d{4}-\d\d-\d\d)')
# Every removal reason the conformance writes (ops/p3s2_conform.py main loop and its pulsar groups; lib/search/removal_rules.
# pulsar_consistency for the B-02 pulsar stage) with the instant it names: 'own' = measured on the row's own frame, position
# or epoch, names no other instant; 'master-night' = a co-added night (T3); 'pulsar-group' = one instant's group (T4); and
# the tiers it is declared for. First match wins (the pulsar group before the table prefix it shares).
ALL_TIERS = ('deep', 'sublst')
REASON_KINDS = (
    ('pulsar-group', 'pulsar-group', lambda r: 'multi-detection source inconsistent with' in r, ALL_TIERS),
    ('duplicate', 'own', lambda r: r.startswith('duplicate (cell executed twice'), ALL_TIERS),
    ('excluded epoch', 'own', lambda r: r.startswith('excluded epoch'), ALL_TIERS),      # data_decisions: its own night
    ('elevation floor', 'own', lambda r: r.startswith('elevation '), ALL_TIERS),        # its own position at its epoch
    ('daytime', 'own', lambda r: r.startswith('daytime'), ALL_TIERS),                  # its own frame's DATE-OBS..DATE-END
    ('twilight', 'own', lambda r: r.startswith('twilight in a solar-audit QUARANTINE band'), ALL_TIERS),
    ('jupiter-track', 'master-night', lambda r: r.startswith('jupiter-track:'), ALL_TIERS),
    ('table', 'own', lambda r: r.startswith('per-band per-Stokes table:'), ALL_TIERS),  # the parent table at its position
    # the noise-sanity test is keyed by NIGHT (weights.csv dates): the deep instant; at 10 minutes it never matches today and
    # a key that did would remove every window of the night (FOLD_IN 2026-10-18: per window only) - not declared there
    ('noise sanity', 'own', lambda r: r.startswith('epoch refused:'), ('deep',)),
    ('row-line', 'own', lambda r: r.startswith('row-line:'), ALL_TIERS),               # finals of its own epoch
    ('no ledger verdict', 'own', lambda r: r.startswith('no final gate-ledger verdict'), ALL_TIERS),
    ('size', 'own', lambda r: r.startswith('gaussfit size'), ALL_TIERS),               # its own fit
    ('axis ratio', 'own', lambda r: r.startswith('axis ratio'), ALL_TIERS),
)
NEVER_GROUPED = ('daytime', 'twilight', 'excluded epoch')   # rows the pulsar step leaves out of every group (p3s2_conform)
# snapshot chain (clv15 phase3_snapshot_chain.py 99fe3ea; phase3_snapshot_detrecord.py): the two frame-reading vetoes, their
# names in vetoed_detections.csv / vetoed_sources.csv, in the record's first_removed_by, its verdict column and the member column
SNAP_VETOES = {'negative-lobe': ('neg-lobe', 'gate_neg_lobe', 'neg_member'),
               'reflection-asymmetry': ('asymmetry', 'gate_asymmetry', 'asym_member')}
SNAP_REEVAL = ('refl-floor', 'seed')       # survivors of a vetoed instant removed when the source is re-built (not vetoes)
SNAP_TIERS = ('10s', '1min')
RECORD_COLS = ('cell', 'subband', 'stokes', 'tier', 'ts', 'instant_key', 'gate_neg_lobe', 'gate_asymmetry', 'neg_member',
               'asym_member', 'first_removed_by')


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def _rows(path):
    with open(path, newline='') as f:
        rd = csv.DictReader(f)
        return list(rd.fieldnames or []), list(rd)


def _f(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _sep_deg(ra1, de1, ra2, de2):
    a, b, c, d = (math.radians(x) for x in (ra1, de1, ra2, de2))
    x = math.sin(b) * math.sin(d) + math.cos(b) * math.cos(d) * math.cos(a - c)
    return math.degrees(math.acos(max(-1.0, min(1.0, x))))


def _fail(msgs, value, what):
    if msgs:
        return False, len(msgs), '; '.join(msgs[:6]) + (f' (+{len(msgs) - 6} more)' if len(msgs) > 6 else '') + ' | ' + what
    return True, value, what


def reason_kind(reason, tier=None):
    """(kind, instant) of a removal reason (REASON_KINDS), or (None, None) when undeclared - for ``tier`` when given: a kind
    not declared for that tier is (kind, None)."""
    r = (reason or '').strip()
    for kind, inst, match, tiers in REASON_KINDS:
        if match(r):
            return (kind, inst) if tier is None or tier in tiers else (kind, None)
    return None, None


def own_instant(tier, lst, date):
    """The instant key of a deep / 10-minute row: (lst, night) / (lst, '<night>T<kk>'); None when the date is not of the
    tier's form (a row of another tier)."""
    d = (date or '').strip()
    if tier == 'deep' and re.match(r'^\d{4}-\d\d-\d\d$', d):
        return (lst, d)
    if tier == 'sublst' and re.match(r'^\d{4}-\d\d-\d\dT\d\d$', d):
        return (lst, d)
    return None


def pulsar_positions():
    """{ATNF common name: (RA, Dec)} from catalog.PULSARS_CSV, as the conformance reads it (ops/p3s2_conform.py)."""
    pos = {}
    with open(_cfg().catalog.PULSARS_CSV, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            try:
                pos[(r.get('common_name') or '').strip()] = (float(r['ra_current']), float(r['dec_current']))
            except (KeyError, TypeError, ValueError):
                pass
    return pos


def _instant_of_id(cand_id):
    """(lst, date) from a cand_id <prefix>_<lst>_<date>_<band>_<n> (catalog rows, restored <S>R_ rows); None otherwise."""
    p = (cand_id or '').split('_')
    return (p[1], p[2]) if len(p) >= 5 else None


def master_imprint_tier(tier):
    """True when the tier's masters hold every co-added night's Jupiter, so a Jupiter-track removal naming another night
    stands for the master's imprint in the row's own residual (T3): deep while masters_build.JUPITER_MASK_DEEP is False
    (R-DEEP-MASTER-JUPITER-MASK-DEFERRED); never the 10-minute tier (its window masters carry the moving mask)."""
    return tier == 'deep' and not bool(_cfg().masters_build.JUPITER_MASK_DEEP)


def check_removals(tier, removed, population=None, groups=None, imprint=None):
    """T1-T4 on removal records. ``removed``: [{'id', 'lst', 'date', 'cadence', 'reason', 'jupiter_night'}]; ``population``:
    [(instant, ra, dec, id)] of the as-run detections (T4 bound), None = not available; ``groups``: pulsar_groups.csv rows
    (exact T4), or None; ``imprint``: whether this tier's masters hold Jupiter (default master_imprint_tier(tier)).
    Returns (passed, value, detail)."""
    C = _cfg()
    imprint_ok = master_imprint_tier(tier) if imprint is None else bool(imprint)
    msgs, counts, n_imprint, pgroups = [], {}, 0, []
    for x in removed:
        inst = own_instant(tier, x['lst'], x['date'])
        if x.get('cadence') and x['cadence'] != TIER_CADENCE[tier]:
            msgs.append(f"T1 {x['id']}: a {x['cadence']} row in the {tier} record (vetoes never cross tiers)")
            continue
        if inst is None:
            msgs.append(f"T1 {x['id']}: date {x['date']!r} is not a {tier} instant (another tier's row, or no instant)")
            continue
        kind, what = reason_kind(x['reason'], tier)
        if kind is None:
            msgs.append(f"T2 {x['id']}: undeclared removal reason {x['reason'][:80]!r} (its instant must be declared)")
            continue
        if what is None:
            msgs.append(f"T2 {x['id']}: a {kind} removal is not declared for the {tier} tier ({x['reason'][:60]!r}): it is "
                        f"not measured at this tier's instant")
            continue
        counts[kind] = counts.get(kind, 0) + 1
        if what == 'master-night':
            m = JUPITER_NIGHT_RE.search(x['reason'] or '')
            night = (x.get('jupiter_night') or '').strip() or (m.group(1) if m else '')
            if not night:
                msgs.append(f"T3 {x['id']}: a Jupiter-track removal that names no night")
            elif night != inst[1][:10]:
                if imprint_ok:
                    n_imprint += 1
                else:
                    msgs.append(f"T3 {x['id']} ({inst[0]} {inst[1]}): removed for Jupiter on night {night} - another instant; "
                                f"the {tier} masters carry the moving Jupiter mask, so no master imprint stands for it")
        elif what == 'pulsar-group':
            m = PULSAR_GROUP_RE.search(x['reason'] or '')
            if m is None:
                msgs.append(f"T4 {x['id']}: a pulsar-group removal whose pulsar cannot be read: {x['reason'][:80]!r}")
                continue
            nm = PULSAR_GROUP_N_RE.search(x['reason'] or '')
            pgroups.append((x, inst, m.group('psr').strip(), int(nm.group('n')) if nm else None))
    n_pg = 0
    if pgroups:
        if groups is not None:
            by_member = {}
            for g in groups:
                for mid in (g.get('members') or '').split('|'):
                    by_member.setdefault(mid, []).append(g)
            insts = {}
            for _i, _r, _d, mid in (population or ()):
                insts[mid] = _i
            for x, inst, name, _n in pgroups:
                n_pg += 1
                gs = [g for g in by_member.get(x['id'], ()) if g.get('pulsar') == name and str(g.get('consistent')) == '0']
                ep = [g for g in gs if g.get('grouping') == 'per-epoch']
                if not ep:
                    msgs.append(f"T4 {x['id']}: removed by the {name} group but no inconsistent per-epoch group of it holds the "
                                f"row ({len(gs)} across-epoch) - only one instant's group may remove")
                    continue
                other = sorted({mid for g in ep for mid in g['members'].split('|')
                                if (insts.get(mid) or _instant_of_id(mid)) != inst})
                if other:
                    msgs.append(f"T4 {x['id']}: its per-epoch {name} group holds {len(other)} member(s) of another instant, "
                                f"e.g. {other[0]}")
        elif population is None:
            msgs.append(f'T4: {len(pgroups)} pulsar-group removals and no as-run record to bound their groups')
        else:
            psr = pulsar_positions()
            rad = float(C.catalog_conform.PULSAR_PRECUT_DEG)
            byinst = {}
            for i_, ra, dec, _id in population:
                byinst.setdefault(i_, []).append((ra, dec))
            for x, inst, name, n in pgroups:
                n_pg += 1
                if name not in psr:
                    msgs.append(f"T4 {x['id']}: pulsar {name} is not in {C.catalog.PULSARS_CSV}")
                    continue
                pra, pdec = psr[name]
                n_here = sum(1 for ra, dec in byinst.get(inst, ()) if _sep_deg(ra, dec, pra, pdec) <= rad)
                if (n or 2) > n_here:
                    msgs.append(f"T4 {x['id']} ({inst[0]} {inst[1]}): its {name} group has {n or 2} members"
                                f"{'' if n else ' at least (n cut from the record)'} but its instant holds {n_here} "
                                f"detection(s) within {rad:g} deg of the pulsar - the group spans instants")
    what = (f'{len(removed)} removals of the {tier} record: '
            + ', '.join(f'{k} {v}' for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
            + f'; jupiter-track naming another night: {n_imprint} as the master\'s imprint'
            + (' (deep masters hold Jupiter: masters_build.JUPITER_MASK_DEEP False)' if n_imprint else '')
            + f'; pulsar-group removals checked against one instant: {n_pg}'
            + (' (group record)' if groups is not None and n_pg else ' (as-run bound)' if n_pg else ''))
    return _fail(msgs, len(removed), what)


def _tier_of(rows, ctx, key='cadence'):
    t = ctx.get('tier')
    if t in TIER_CADENCE:
        return t
    cads = {(r.get(key) or '').strip() for r in rows} - {''}
    return 'sublst' if cads == {'10min'} else 'deep'


@content_test('conformant_catalog', name='per_instant')
def conform_per_instant(path, **ctx):
    """T1-T4 on master_catalog_removed.csv beside the conformant catalogue (module docstring)."""
    from exopipe.phase3.lib.catalog import tests as CT
    from exopipe.phase3.lib.catalog import conform_check as CC
    rmp = CT._sibling(path, 'master_catalog_removed.csv', ctx)
    asp = CT._sibling(path, 'master_catalog_asrun.csv', ctx)
    if not rmp:
        return False, None, 'no master_catalog_removed.csv beside: the removals cannot be placed at their instants'
    _, rem = _rows(rmp)
    if not rem:
        return True, 0, 'no removal in master_catalog_removed.csv: nothing to judge'
    tier = _tier_of(rem, ctx)
    removed = [dict(id=r.get('cand_id', ''), lst=r.get('lst', ''), date=r.get('date', ''), cadence=r.get('cadence', ''),
                    reason=r.get('removal_reason', ''), jupiter_night=r.get('jupiter_track_night', '')) for r in rem]
    population = None
    if asp and any(reason_kind(x['reason'])[0] == 'pulsar-group' for x in removed):
        skip = {x['id'] for x in removed if reason_kind(x['reason'])[0] in NEVER_GROUPED}
        population = []
        for r in _rows(asp)[1]:
            i_ = own_instant(tier, r.get('lst', ''), r.get('date', ''))
            ra, dec = _f(r.get('ra')), _f(r.get('dec'))
            if i_ and ra is not None and dec is not None and r.get('cand_id') not in skip:
                population.append((i_, ra, dec, r.get('cand_id', '')))
    gp = CT._sibling(path, CC.PULSAR_GROUPS, ctx)
    groups = _rows(gp)[1] if gp else None
    if groups is not None and population is None:
        population = [(own_instant(tier, r.get('lst', ''), r.get('date', '')), None, None, r.get('cand_id', ''))
                      for r in (_rows(asp)[1] if asp else [])]
    return check_removals(tier, removed, population, groups)


@content_test('fork_catalog', name='per_instant')
def fork_per_instant(path, **ctx):
    """T1-T4 on the deep multiples fork's own conformance record (<run>/fork_<S>/master_catalog_removed.csv beside its
    master_catalog.csv and as-run copy: the fork's two p3s2_conform calls, ops/multiples_fork.py) for a fork catalogue
    directory source_catalogs_<S>_f<wb|sb>."""
    from exopipe.phase3.lib.catalog import tests as CT
    key = CT._catalogue_key(path)
    if not key or not key.split('_')[1].startswith('f'):
        return False, None, f'{os.path.basename(os.path.normpath(path))} is not a fork catalogue (source_catalogs_<S>_f<wb|sb>)'
    fm = os.path.join(os.path.dirname(os.path.abspath(os.path.normpath(path))), f'fork_{key[0]}', 'master_catalog.csv')
    if not os.path.exists(fm):
        return False, None, f'{fm} not found: the fork\'s removals cannot be placed at their instants'
    return conform_per_instant(fm, tier='deep')


@content_test('rejected_record', name='per_instant')
def rejected_per_instant(path, **ctx):
    """T1-T4 on rejected.csv (the 10-minute / deep pages' record; module docstring): the row's instant from its detail
    '<cand_id> <date> peak ...' and cell '<lst>_<band>'; the as-run population = master_catalog.csv beside + the record."""
    _, rows = _rows(path)
    removed = []
    for r in rows:
        det = (r.get('detail') or '').split()
        removed.append(dict(id=r.get('cand_id', ''), lst=(r.get('cell') or '').split('_')[0],
                            date=det[1] if len(det) > 1 else '', cadence=r.get('tier', ''), reason=r.get('veto', ''),
                            jupiter_night=''))
    tier = ctx.get('tier') if ctx.get('tier') in TIER_CADENCE else (
        'sublst' if {x['cadence'] for x in removed} == {'10min'} else 'deep')
    population = None
    if any(reason_kind(x['reason'])[0] == 'pulsar-group' for x in removed):
        mc = os.path.join(os.path.dirname(os.path.abspath(path)), 'master_catalog.csv')
        if os.path.exists(mc):
            population = []
            for r in _rows(mc)[1]:
                i_ = own_instant(tier, r.get('lst', ''), r.get('date', ''))
                ra, dec = _f(r.get('ra')), _f(r.get('dec'))
                if i_ and ra is not None and dec is not None:
                    population.append((i_, ra, dec, r.get('cand_id', '')))
            for x, r in zip(removed, rows):
                i_ = own_instant(tier, x['lst'], x['date'])
                ra, dec = _f(r.get('ra_deg')), _f(r.get('dec_deg'))
                if i_ and ra is not None and dec is not None and reason_kind(x['reason'])[0] not in NEVER_GROUPED:
                    population.append((i_, ra, dec, x['id']))
    return check_removals(tier, removed, population, None)


# ---------------------------------------------------------------------------------------------------------------------
# snapshot runs (10-s and 1-minute)
# ---------------------------------------------------------------------------------------------------------------------

def _parse_member(s):
    """(cell, subband, tier, ts) of a measured member: '<cell> <sb> <ts>' (a 10-s subband member, negative_lobe_veto),
    '<cell> <sb> <tier> <ts>' (a block or group image, negative_lobe_block_veto), '<cell> <sb> <S> <tier> <ts>'
    (reflection_asymmetry_veto); None when unparsable."""
    t = (s or '').split()
    if len(t) == 3:
        return t[0], t[1], '10s', t[2]
    if len(t) in (4, 5) and t[-2] in SNAP_TIERS:
        return t[0], t[1], t[-2], t[-1]
    return None


def read_record(path):
    """The per-detection record's columns RECORD_COLS, row by row (a generator of dicts)."""
    with gzip.open(path, 'rt', newline='') as f:
        rd = csv.reader(f)
        h = next(rd)
        ix = {k: h.index(k) for k in RECORD_COLS if k in h}
        miss = [k for k in RECORD_COLS if k not in ix]
        if miss:
            raise ValueError(f'{os.path.basename(path)} lacks the columns {miss}')
        for r in rd:
            yield {k: r[i] if i < len(r) else '' for k, i in ix.items()}


def check_snapshot_run(run):
    """S1-S3 on a snapshot chain run directory (module docstring): (passed, value, detail)."""
    msgs = []
    sp = os.path.join(run, 'chain_summary.json')
    if not os.path.exists(sp):
        return False, None, 'no chain_summary.json'
    S = json.load(open(sp))
    piv = S.get('per_instant_vetoes')
    vsp, vdp = os.path.join(run, 'vetoed_sources.csv'), os.path.join(run, 'vetoed_detections.csv')
    if not os.path.exists(vsp):
        return False, None, 'no vetoed_sources.csv: the run\'s veto removals cannot be judged'
    src_veto = [r for r in _rows(vsp)[1] if r.get('veto') in SNAP_VETOES]
    if piv is None or not os.path.exists(vdp):
        if src_veto:
            return False, len(src_veto), (f'S1: {len(src_veto)} sources removed by the negative-lobe / reflection-asymmetry veto '
                                          f'with no per-instant record (chain_summary per_instant_vetoes '
                                          f'{"absent" if piv is None else "present"}, vetoed_detections.csv '
                                          f'{"present" if os.path.exists(vdp) else "absent"}): the vetoes were applied per '
                                          f'SOURCE - one member\'s measurement removed every instant (before clv15 99fe3ea)')
        if piv is None and not os.path.exists(vdp):
            return True, 0, 'no veto removed a source and the run predates the per-instant record: nothing to judge'
        return False, None, 'S1: the per-instant summary and vetoed_detections.csv must exist together'
    vd = _rows(vdp)[1]
    n_vd = {k: 0 for k in SNAP_VETOES}
    want_member = []                     # (row id, member frame key, the row's instant label, row's frame key)
    for r in vd:
        rid = f"{r.get('cell')} {r.get('subband')} {r.get('stokes')} {r.get('tier')} {r.get('ts')}"
        by = r.get('removed_by', '')
        if by not in SNAP_VETOES and by not in SNAP_REEVAL:
            msgs.append(f'S2 {rid}: removed_by {by!r} is neither a veto nor a re-evaluation')
            continue
        it = (r.get('instant') or '').split()
        if len(it) != 3 or it[0] != r.get('cell') or it[1] != r.get('tier') or it[1] not in SNAP_TIERS:
            msgs.append(f"S2 {rid}: its instant {r.get('instant')!r} is not of its own cell and tier")
            continue
        if by in SNAP_VETOES:
            n_vd[by] += 1
            mem = _parse_member(r.get(SNAP_VETOES[by][2]))
            if mem is None:
                msgs.append(f'S2 {rid}: {by} without its measured member')
            elif mem[0] != r.get('cell') or mem[2] != r.get('tier'):
                msgs.append(f"S2 {rid}: {by} measured on {r.get(SNAP_VETOES[by][2])!r}, another "
                            f"{'cell' if mem[0] != r.get('cell') else 'tier'} (vetoes never cross tiers or instants)")
            else:
                want_member.append((rid, mem, it[2], (r.get('cell'), r.get('subband'), r.get('tier'), r.get('ts'))))
    rp = os.path.join(run, 'detection_record.csv.gz')
    n_removed_by_veto = sum(n_vd.values())
    if not os.path.exists(rp):
        if n_removed_by_veto:
            msgs.append(f'S3: {n_removed_by_veto} detections removed by a veto and no per-detection record '
                        f'(detection_record.csv.gz) to place them at their instants')
        return _fail(msgs, n_removed_by_veto, f'{len(vd)} vetoed_detections rows; no per-detection record')
    frame_inst = {}                      # (cell, subband, tier, ts) -> {instant_key}
    pending = []                         # (detection id, member frame key, its instant key)
    n_rec = {k: 0 for k in SNAP_VETOES}
    n_rows = 0
    rec_of = {v[0]: k for k, v in SNAP_VETOES.items()}
    try:
        for d in read_record(rp):
            n_rows += 1
            key = (d['cell'], d['subband'], d['tier'], d['ts'])
            frame_inst.setdefault(key, set()).add(d['instant_key'])
            veto = rec_of.get(d['first_removed_by'])
            if not veto:
                continue
            n_rec[veto] += 1
            _rec, gate, mcol = SNAP_VETOES[veto]
            did = f"{d['cell']} {d['subband']} {d['stokes']} {d['tier']} {d['ts']}"
            if d[gate] != 'remove':
                msgs.append(f"S3 {did}: removed by the {veto} veto while its own instant's verdict is {d[gate]!r} - "
                            f"another instant's measurement")
                continue
            mem = _parse_member(d[mcol])
            if mem is None or mem[0] != d['cell'] or mem[2] != d['tier']:
                msgs.append(f"S3 {did}: its {veto} member {d[mcol]!r} is not of its own cell and tier")
                continue
            pending.append((did, mem, d['instant_key']))
    except (OSError, ValueError, EOFError) as e:
        return False, None, f'per-detection record unreadable: {type(e).__name__}: {e}'
    for did, mem, ik in pending:
        got = frame_inst.get(mem, set())
        if ik not in got:
            msgs.append(f"S3 {did}: measured at {' '.join(mem)} (instant {sorted(got) or 'not in the record'}), not at its own "
                        f"instant {ik}")
    for rid, mem, lab, own in want_member:
        if lab not in frame_inst.get(own, set()):
            msgs.append(f'S2 {rid}: the record places its frame at {sorted(frame_inst.get(own, set()))}, not at {lab}')
        elif lab not in frame_inst.get(mem, set()):
            msgs.append(f"S2 {rid}: its measured member {' '.join(mem)} is at instant {sorted(frame_inst.get(mem, set()))}, "
                        f"not {lab}")
    for k in SNAP_VETOES:
        if n_rec[k] != n_vd[k]:
            msgs.append(f'S3: the record says {n_rec[k]} detections were removed by {k}, vetoed_detections.csv lists {n_vd[k]}')
    tiers = {}
    for r in vd:
        if r.get('removed_by') in SNAP_VETOES:
            tiers[r.get('tier')] = tiers.get(r.get('tier'), 0) + 1
    what = (f"per instant: {n_removed_by_veto} detections removed by a veto (negative lobe {n_vd['negative-lobe']}, reflection "
            f"asymmetry {n_vd['reflection-asymmetry']}; " + ', '.join(f'{t} {n}' for t, n in sorted(tiers.items()))
            + f"), each on its own instant's measurement of its own cell and tier ({n_rows} record rows); "
            f"{len(vd) - n_removed_by_veto} survivors removed on re-evaluation (floor / seed); 3-D gate, seed and routes are "
            f"source-level rules, not judged here")
    return _fail(msgs, n_removed_by_veto, what)


@content_test('snapshot_run', name='per_instant')
def snapshot_per_instant(path, **ctx):
    """S1-S3 on a snapshot chain run directory (module docstring)."""
    return check_snapshot_run(path)
