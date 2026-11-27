"""`exopipe3 catalog --tier sublst --incremental --epochs <LST>_<YYYY-MM-DD>`: the 10-minute tier of ONE night of one LST hour
in the per-hour chain (runner.CHAINS['epoch']), judged and built on the frozen state of the all-nights pass.

Rulings (fix ledger R-GREGG-20261112-PER-HOUR-STEPS-SIX-RULINGS): (1) a new night may change an earlier source's identity (a
source seen again becomes a repeat and the page says when) and never an earlier verdict; (2) frozen statistics from earlier
nights judge a new night everywhere, so the result does not depend on the order in which nights arrive; (3) a night that is not
in a Master is searched as a non-member against the frozen Masters (the frames and search steps before this one). The night's
delta build is answer 3 of R-GREGG-20261113-SPEEDUP-CHECKIN-TEN-ANSWERS. Design: sections 2.2a and 2.3a of
plan/PER_HOUR_GAP_DESIGNS_20261111.md.

One invocation takes ONE item; --out is the night's own directory (default catalog_sublst.NIGHT_DIR). Units, in order (the
first that does not end done stops the item):
  <LST>_<arm>.frozen-state   THE GATE. It refuses (exit 2) unless the frozen state the night stands on exists, and writes
        what it found to <night directory>/frozen_state.json:
        I-G1  catalog_grouping.GROUPING is 'design-b' (the builder's frozen statistics exist only under it);
        I-G2  the latest grouping run of each Stokes in the 10-minute grouping directory (--sublst-grouping, default
              catalog_grouping.SUBLST_DIR) is done and its table unchanged (grouping_stage.recorded_inputs);
        I-G3  both Stokes were grouped from the same directories, and the grouping holds an all-nights catalog directory F of
              this hour (its --group-lst-dir);
        I-G4  this night of this hour was not appended before from another directory;
        I-G5  F holds its image table v_deep_images.csv;
        I-G6  F's conformance record <LST>_<arm>.conform is done and lists conform_table_noise.csv and
              conform_pulsar_evidence.csv as they are now (tier_sublst.frozen_records_refusal: the frozen statistics the
              night is judged by);
        I-G7  every hour in scope (every hour of the grouping, or --sublst-view-hours) has its image table and, for every
              build (--sublst-builds) and key with rows (as delta_build.key_state), the all-nights build's frozen record
              source_catalogs_<build>_<key>/builder_frozen.json (what each delta build reads);
        I-G8  the deep omission: the latest grouping run of each Stokes recorded a deep-omission decision table, the same
              one for both Stokes (path and sha256 of the table and of its coverage record), both unchanged now
              (grouping_stage.grouping_omission, read once per item), and every frozen record of I-G7 names that table or
              one it appends to (wraps/builder_grouping.Omission.extends: the frozen record's table, unchanged, a byte
              prefix of it, its cells unchanged, the rows after it deciding only new cells - the verdicts of nights appended
              since). Any other table REFUSES: the frozen state stands on one omission and the nights appended to it, made
              again only by a new all-nights pass (FIRST_RUN).
        It ends no-input (exit 3; the chain goes on) when F's image table already holds the night: the night's 10-minute
        rows are then the all-nights pass's (a night in the window Masters, catalogued by the batch chain 'sublst-lst'),
        never appended a second time.
  <LST>_<arm>.preflight .. .veto   the NIGHT PASS (lib/catalog/tier_sublst.py with --sublst-night <date>
        --sublst-frozen-from F): the LST chain's stages on the night's rows only, read through a night view of the search
        jobs, its conformance judged by F's records. A merge that finds no row of the night ends the item no-input (exit 3).
  <LST>_<arm>.omit   the night's deep-omission verdicts (deep_omission.append_night), under the items' lock: the table the
        latest grouping run recorded now (the gate's, or an append to it by another item since: I-G8) is the parent; the
        night's refused cells are rated as production rates a night (sublst_deep_omission.decide, :134-225; its rate among
        the all-nights directory's, the nights appended before and this night's detections; every threshold over every night)
        and APPENDED to it as a new table version <night directory>/deep_omission/deep_omission.csv with its coverage record
        (the parent's cells plus every cell of the night's detections and decisions). Earlier verdicts are never recomputed
        (ruling (1) of R-GREGG-20261112-PER-HOUR-STEPS-SIX-RULINGS: "no for earlier verdicts (a detection removed by a gate
        stays removed until the periodic audit pass)"). A parent that already rated the night's detections appends nothing.
        I-G10: the latest grouping runs of I and V recorded different nights appended before (the rate's population) -
        REFUSES. The night's deep record is read from the attempts the deep search published on any node (G18).
  <LST>_<arm>.group   the ONE tier-wide 10-minute grouping, incremental, filtered by the omit unit's table: I-G9, the night
        has verdicts (that table's coverage record holds the night: a night without them REFUSES, named, never kept for
        want of a verdict), and the latest grouping runs still name the omit unit's parent; its input is the latest grouping run's own (every
        hour's all-nights directory, every night appended since, read again here) plus this night's directory, last; the run
        writes its record in the grouping directory and this unit records it in <night directory>/night_grouping.json
        (with the table the night was grouped under; the delta builds and the cumulative read the same one).
        Earlier detections keep their identities; only the night's are assigned.
  <LST>_<arm>.build, .buildv, .view   the night pass's delta builds (this hour and every other hour a touched identity
        reaches) and the catalog view of every hour with the index hour_views.json (tier_sublst._build_delta, st_view).
  The verdicts, grouping, builds and views of one item hold <grouping directory>/.night_items.lock (night_lock): an item of another
  hour waits up to catalog_sublst.NIGHT_LOCK_WAIT_S, then refuses (the grouping unit's own lock would refuse it at once).
AN ITEM RUN AGAIN (a chain retry; the chain line passes --resume) runs the gate, then, with --resume, resumes at its first
unit that does not stand (resume_point): before the grouping holds the night, at the first night-pass stage not done (the
tier's own --stages gap check, tier_sublst.check_from, vouches for the stage before and restores the files later stages
rewrite in place), else at the omit unit (omit and group run together, one hold of the lock); once the grouping holds the
night from this directory, the night pass and the omit unit are never run again and group, build, buildv, view are judged by core/record.resume_check (record done, clean tree, recorded inputs,
code and config unchanged, outputs passing their content tests now - night_grouping_record, catalog_view and
hour_views_index fail once a later grouping run replaced the table they stand on). The units before the resume point are
recorded skipped-valid; the gate records it (meta `resume`). Without --resume the gate REFUSES a night directory holding
an earlier attempt's done night-pass units (a night pass run again over its own outputs reads the earlier attempt's
later-stage files) and a night the grouping already holds from this directory (re-running its night pass would change rows the grouping holds; a grouped night is
judged again only by the audit pass).
A night is never appended twice: the grouping run lists the night directory once (I-G4, _group).
What the chain runs next for the tier: `pages --tier sublst --epochs {item}` (the pages of the night from this directory's
index, lib/pages/tier_sublst.py) and `serve --tier sublst --epochs {item}` (the cumulative Run_SLSTB with its reuses and the
one-table gate on the views' certificates, lib/serve/sublst_build.py).

Not here: the first all-nights pass (the frozen state the gate asks for, FIRST_RUN), the periodic audit pass, and the
re-admission question of ruling (1) the design leaves open (a touched identity rebuilt from all its members may re-admit a
detection a singles gate removed earlier).
"""
import argparse
import contextlib
import copy
import csv
import json
import os
import sys

STAGE_GATE = 'frozen-state'
STAGE_OMIT = 'omit'
STAGE_GROUP = 'group'
GATE_RECORD = 'frozen_state.json'        # in the night directory: the frozen state the night stood on
GROUP_RECORD = 'night_grouping.json'     # in the night directory: the grouping run that took the night
NIGHT_LOCK = '.night_items.lock'         # in the grouping directory: one item at a time groups, builds and views (night_lock)
BUILD_STAGES = 'build,buildv,view'
# what writes the frozen state (named in every refusal of the gate)
FIRST_RUN = ('The frozen state is written by the all-nights 10-minute pass under design B, in this order: (1) for every LST '
             'hour L, `exopipe3 catalog --tier sublst --lsts L --out F_L --until veto` over all nights (preflight..veto; its '
             'conformance writes conform_table_noise.csv and conform_pulsar_evidence.csv); (2) the deep omission of all '
             'hours on the same detections, `exopipe3 catalog --tier sublst --stages omit --lsts <every hour> --omit-lst-dir '
             'L=F_L (every hour) --out O` (O/deep_omission.csv, the decision table, rated on the F_L catalogs as production '
             'rates the runs it merges, and O/deep_omission_cells.csv, the cells it decides); (3) ONE tier-wide grouping (one state per Stokes for the whole tier, not one per LST) '
             '`exopipe3 catalog --tier sublst --stages group --group-new-state --group-omission O --lsts <every hour> '
             '--group-lst-dir L=F_L` (every hour) into catalog_grouping.SUBLST_DIR, which drops the OMIT cells of O before '
             'grouping; (4) the builds of every hour `--lsts L --out F_L --stages build,buildv` (each drops the same OMIT '
             'cells from its input and writes source_catalogs_<build>_<key>/builder_frozen.json naming O\'s sha256)')
# arguments of a batch catalog run that the per-hour item sets itself
OWN_ARGS = (('stages', '--stages'), ('from_stage', '--from'), ('until_stage', '--until'), ('sublst_night', '--sublst-night'),
            ('sublst_frozen_from', '--sublst-frozen-from'), ('group_input', '--group-input'), ('group_omission', '--group-omission'),
            ('group_new_state', '--group-new-state'), ('group_mode', '--group-mode'), ('group_lst_dir', '--group-lst-dir'),
            ('group_night_dir', '--group-night-dir'))


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def _rec():
    from exopipe.phase3.core import record
    return record


def _ts():
    from exopipe.phase3.lib.catalog import tier_sublst
    return tier_sublst


def _gs():
    from exopipe.phase3.lib.catalog import grouping_stage
    return grouping_stage


def night_dir(lst, night, arm=None, cfg=None):
    """The night pass's own directory of one night of one LST hour (catalog_sublst.NIGHT_DIR)."""
    cfg = cfg or _cfg()
    c = cfg.catalog_sublst
    return c.NIGHT_DIR.format(transients_root=cfg.paths.TRANSIENTS_ROOT, lst=lst, arm=arm or c.ARM, night=night)


def default_out(args):
    """The night directory of the ONE --epochs item; with none or several, the directories' common root (run refuses)."""
    from exopipe.phase3.core import cells
    cfg = _cfg()
    items = cells.expand_epochs(args.epochs) if getattr(args, 'epochs', None) else []
    if len(items) == 1:
        lst, night = cells.parse_epoch(items[0])
        return night_dir(lst, night, getattr(args, 'sublst_arm', None), cfg)
    return cfg.catalog_sublst.NIGHT_DIR.split('{lst}')[0].format(transients_root=cfg.paths.TRANSIENTS_ROOT).rstrip('/')


def _has_rows(path):
    if not os.path.isfile(path):
        return False
    with open(path, newline='') as f:
        return sum(1 for _ in zip(range(2), f)) > 1


def night_rows(images, night):
    """Rows of the image table ``images`` (v_deep_images.csv) on ``night`` (a window key YYYY-MM-DDTnn or the night itself)."""
    with open(images, newline='') as f:
        rd = csv.DictReader(f)
        if 'date' not in (rd.fieldnames or []):
            raise _rec().Refusal(f'I-G5 {images} has no date column: whether it holds the night {night} cannot be told')
        return sum(1 for r in rd if (r.get('date') or '').split('T')[0] == night)


def _real(p):
    return os.path.realpath(p)


def frozen_state(lst, night, arm, out, grouping_dir, builds, view_hours=None, cfg=None):
    """THE GATE of the per-hour 10-minute item (gates I-G1..I-G7 of the module text). Returns the state the night stands on;
    raises record.Refusal when a part of it is absent, record.NoInput when the all-nights directory already holds the night."""
    cfg = cfg or _cfg()
    rec, TS, GS = _rec(), _ts(), _gs()
    from exopipe.phase3.lib.catalog import delta_build as DB
    unit = f'{lst}_{arm}'
    try:
        mode = GS.mode(cfg)
    except GS.StageRefusal as e:
        raise rec.Refusal(f'I-G1 {e}')
    if not mode.design_b:
        raise rec.Refusal(f'I-G1 catalog_grouping.GROUPING is {mode.grouping!r}: the per-hour 10-minute path stands on the '
                          f'design-B grouping (a night\'s delta builds read the builder\'s frozen statistics, which only a '
                          f'design-B all-nights build writes)')
    got, latest = {}, {}
    for S in ('I', 'V'):
        try:
            fin, path, r = GS._latest_group_record(grouping_dir, S)
            runs, nights, (table, sha) = GS.recorded_inputs(grouping_dir, S, with_table=True)
        except GS.StageRefusal as e:
            raise rec.Refusal(f'I-G2 no frozen 10-minute grouping state for the night {night} of {lst}: {e}. {FIRST_RUN}')
        if (((r.get('meta') or {}).get('tables') or {}).get(S) or {}).get('sha1') != sha:
            raise rec.Refusal(f'I-G2 a grouping run of {grouping_dir} finished while the gate read it (Stokes {S}): run the '
                              f'item again')
        got[S] = ({L: _real(D) for L, D in runs.items() if D}, [(L, _real(D)) for L, D in nights])
        latest[S] = dict(record=path, finished=fin, table=table, sha1=sha)
    if got['I'] != got['V']:
        raise rec.Refusal(f'I-G3 the Stokes I and V groupings in {grouping_dir} were assembled from different directories: '
                          f'group both Stokes again (one run, --group-stokes I,V)')
    runs, nights = got['I']
    if not runs.get(lst):
        raise rec.Refusal(f'I-G3 the grouping {grouping_dir} holds no all-nights catalog directory of {lst} (its hours: '
                          f'{", ".join(sorted(runs)) or "none"}). {FIRST_RUN}')
    F = runs[lst]
    for L, D in nights:
        if L != lst or D == _real(out):
            continue
        try:
            with open(os.path.join(D, GATE_RECORD)) as f:
                other = json.load(f).get('night')
        except (OSError, ValueError):
            other = None
        if other == night:
            raise rec.Refusal(f'I-G4 the night {night} of {lst} was appended to the grouping from {D}, not from {out}: one '
                              f'night of an hour has one directory')
    images = os.path.join(F, DB.IMAGES)
    if not os.path.isfile(images):
        raise rec.Refusal(f'I-G5 the all-nights directory {F} of {lst} holds no {DB.IMAGES} (every delta build reads it). '
                          f'{FIRST_RUN}')
    n = night_rows(images, night)
    if n:
        raise rec.NoInput(f'the night {night} is in the all-nights catalog directory {F} of {lst} ({n} rows of its image table '
                          f'{DB.IMAGES}): its 10-minute rows are the all-nights pass\'s, never appended a second time')
    noise, evid = TS.conform_noise_record(F), TS.conform_pulsar_evidence(F)
    why, conform_finished = TS.frozen_records_refusal(F, unit, (noise, evid))
    if why:
        raise rec.Refusal(f'I-G6 the frozen statistics the night is judged by: {why}. {FIRST_RUN}')
    scope = sorted(runs)
    if view_hours:
        bad = [H for H in view_hours if H not in runs]
        if bad or lst not in view_hours:
            raise rec.Refusal(f'I-G7 --sublst-view-hours {",".join(view_hours)}: {bad or "none"} not hours of the grouping, '
                              f'or the night\'s own hour {lst} is not named')
        scope = sorted(view_hours)
    frozen, missing = {}, []
    for H in scope:
        FH = runs[H]
        if not os.path.isfile(os.path.join(FH, DB.IMAGES)):
            missing.append(f'{H} ({FH} holds no {DB.IMAGES})')
            continue
        appended = [D for L, D in nights if L == H]
        for tag in builds:
            for K in TS.SPLIT_KEYS:
                cat = os.path.join(FH, DB.CAT.format(tag=tag, key=K))
                sel = DB.SELECTED.format(key=K)
                if not (os.path.isdir(cat) or any(_has_rows(os.path.join(d, sel)) for d in [FH] + appended)):
                    continue                     # as delta_build.key_state 'empty': no row of the key, no catalog, nothing frozen
                frz = os.path.join(cat, DB.FROZEN_RECORD)
                if os.path.isfile(frz):              # presence only: each delta build hashes the records it reads
                    frozen.setdefault(H, {})[f'{tag}_{K}'] = frz
                else:
                    missing.append(f'{H} {tag} {K}')
    if missing:
        hrs = sorted({m.split()[0] for m in missing})
        raise rec.Refusal(f'I-G7 {len(hrs)} hour(s) in scope lack what a delta build stands on ({len(missing)} part(s): '
                          f'{", ".join(missing[:8])}{" ..." if len(missing) > 8 else ""}): each needs its image table '
                          f'{DB.IMAGES} and, for every build and key with rows, '
                          f'{os.path.join("<all-nights directory>", DB.CAT.format(tag="<build>", key="<key>"), DB.FROZEN_RECORD)}. '
                          f'{FIRST_RUN}' +
                          ('' if view_hours else '; or name the hours to view with --sublst-view-hours (an hour left out is '
                                                 'never certified for the cumulative)'))
    om = omission_state(grouping_dir, frozen)    # I-G9 is the group unit's: the omit unit rates the night after its pass
    omission = om.ref()
    return dict(version=1, lst=lst, arm=arm, night=night, night_dir=_real(out), grouping_dir=_real(grouping_dir),
                grouping=latest, frozen_from=F, hours=runs, nights_before=[list(x) for x in nights], scope=scope,
                builds=list(builds), conform=dict(record=_rec().record_path(F, 'catalog', f'{unit}.conform'),
                                                  finished=conform_finished, noise=noise, evidence=evid),
                frozen_records=frozen, omission=omission)


def omission_state(grouping_dir, frozen):
    """I-G8 (the module text): the Omission (read once) of the deep-omission decision table the frozen state stands on - the
    one the latest grouping run of both Stokes recorded (grouping_stage.grouping_omission), unchanged now; every frozen record
    of ``frozen`` ({hour: {build_key: builder_frozen.json}}) names it or a table it appends to (Omission.extends: the nights
    appended since the all-nights builds). Anything else raises record.Refusal."""
    rec, GS = _rec(), _gs()
    try:
        om = GS.grouping_omission(grouping_dir)
    except GS.StageRefusal as e:
        raise rec.Refusal(f'I-G8 the deep omission of the frozen state: {e}. {FIRST_RUN}')
    bad = []
    for H, recs in sorted(frozen.items()):
        for name, p in sorted(recs.items()):
            try:
                with open(p) as f:
                    have = json.load(f).get('omission') or {}
            except (OSError, ValueError) as e:
                have = {'sha256': f'unreadable ({type(e).__name__})'}
            why = om.extends(have)
            if why:
                bad.append(f'{H} {name}: {str(have.get("sha256") or "no table")[:12]} ({why})')
    if bad:
        raise rec.Refusal(f'I-G8 {len(bad)} frozen record(s) were built under a deep-omission decision table that is neither '
                          f'the grouping\'s ({om.path}, sha256 {om.sha256[:12]}, coverage record {om.cells_sha256[:12]}) nor one '
                          f'it appends to: '
                          f'{", ".join(bad[:8])}{" ..." if len(bad) > 8 else ""} - the frozen state stands on one omission. '
                          f'{FIRST_RUN}')
    return om


def _night_args(args, lst, night, frozen_from, gdir, stages=None):
    a = copy.copy(args)
    a.lsts, a.sublst_night, a.sublst_frozen_from, a.sublst_grouping = lst, night, frozen_from, gdir
    a.stages, a.from_stage, a.until_stage = stages, None, None
    a.incremental, a.epochs = False, None
    return a


def _night_pass(run, args, unit, stages):
    """tier_sublst.run on this run; returns {stage: unit record} of the stages it ran. A usage refusal of the tier's context
    (SystemExit) is a refusal of the item."""
    n0 = len(run.units)
    try:
        _ts().run(args, run)
    except SystemExit as e:
        raise _rec().Refusal(f'the night pass refused its arguments: {e}')
    return {x.name[len(unit) + 1:]: x for x in run.units[n0:] if x.name.startswith(unit + '.')}


def night_lock(gdir, wait_s, what):
    """The items' lock <grouping directory>/NIGHT_LOCK (grouping_stage._Lock with a wait): a second item polls for up to
    ``wait_s`` seconds (catalog_sublst.NIGHT_LOCK_WAIT_S), then REFUSES naming the holder. Held while the gate reads the
    grouping and while the item groups, builds and views (the grouping unit's own state lock refuses a second run at once;
    without this one, two hours' items would refuse each other, or read a grouping another item is replacing)."""
    return _gs()._Lock(os.path.join(gdir, NIGHT_LOCK), f'per-hour 10-minute item {what}', wait_s=wait_s)


def night_keys_refusal(st, out, builds):
    """I-G7 for the night's own rows, checked after the night pass and BEFORE the grouping takes the night (else an item
    whose night has rows of a key the all-nights build never catalogued would be grouped and then refused by its delta build
    for good: delta_build.key_state 'no-record'). None, or the reason."""
    from exopipe.phase3.lib.catalog import delta_build as DB
    miss = []
    for K in _ts().SPLIT_KEYS:
        if not _has_rows(os.path.join(out, DB.SELECTED.format(key=K))):
            continue
        for tag in builds:
            frz = os.path.join(st['frozen_from'], DB.CAT.format(tag=tag, key=K), DB.FROZEN_RECORD)
            if not os.path.isfile(frz):
                miss.append(f'{tag} {K}')
    if miss:
        return (f'I-G7 the night has rows of key(s) whose all-nights build of {st["lst"]} wrote no {DB.FROZEN_RECORD} in '
                f'{st["frozen_from"]} ({", ".join(miss)}): its delta builds could not be made, so the grouping does not take '
                f'the night. {FIRST_RUN}')
    return None


def _keep_last_done(gdir, gu, saved):
    """After a grouping run that did not end done: move its unit record out of the records the gate reads
    (<grouping directory>/_records/catalog/failed/) and put back the last done record it overwrote, so the gate and the
    next item stand on the last done state (grouping_stage leaves the tables and state unchanged on a refusal; a table that
    changed anyway fails the record's sha1 check and refuses). The run's own summary stays under _records/catalog/_runs/.
    Returns the path the failed record was moved to, or None."""
    import time
    rec = _rec()
    p = rec.record_path(gdir, 'catalog', gu.name) if gu is not None else None
    moved = None
    if p and os.path.isfile(p):
        d = os.path.join(os.path.dirname(p), 'failed')
        os.makedirs(d, exist_ok=True)
        moved = os.path.join(d, f'{os.path.basename(p)[:-5]}.{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}.json')
        os.replace(p, moved)
    sp, data = saved
    if not os.path.exists(sp):
        tmp = f'{sp}.restore.{os.getpid()}'
        with open(tmp, 'wb') as f:
            f.write(data)
        os.replace(tmp, sp)
    return moved


def _deep_search_roots():
    """The out roots whose registers may name the attempts the chain's `search --tier deep --epochs {item}` published: it is
    given no --out and writes to the default root of the node that ran it (lib/search/deep_step.default_out), which no record
    of the chain names and which may be another node than this item's - so every site root of the search
    (deep_omission.search_roots); the night's deep record the omit unit reads (deep_omission.night_deep_record) takes the
    newest published attempt over all of them and production's register, and refuses where they do not settle it."""
    from exopipe.phase3.lib.catalog import deep_omission as DO
    return DO.search_roots(None)


def _rate_night(u, st, out):
    """The omit unit (module text): the night's deep-omission verdicts, under night_lock. The table the latest grouping runs
    of both Stokes recorded NOW (another item may have appended its night since the gate) must be the gate's or an append to
    it (I-G8); the night is rated - on the deep search's record of the night, read from the attempts the chain's deep search
    published, on whichever node (_deep_search_roots; deep_omission.night_deep_record), and refused (G18) for a cell of its
    detections, or a member subband of a band group with detections, the search has no record of running on - and appended to
    it as a new version in <night directory>/deep_omission (deep_omission.APPEND_DIR, deep_omission.append_night), or that
    table, when it already decides the night. The nights appended before (the rate population) come from the latest grouping
    runs of both Stokes, which must agree (I-G10). Returns (the parent's ref, the ref of the table the night is grouped
    under)."""
    GS, rec = _gs(), _rec()
    from exopipe.phase3.lib.catalog import deep_omission as DO
    gdir, lst, night = st['grouping_dir'], st['lst'], st['night']
    try:
        cur = GS.grouping_omission(gdir)
        nds = {S: GS.recorded_inputs(gdir, S)[1] for S in ('I', 'V')}
    except GS.StageRefusal as e:
        u.refuse(f'I-G8 the grouping {gdir} before the night\'s verdicts: {e}')
    if nds['I'] != nds['V']:                          # the nights appended before: one list, the same for both Stokes
        u.refuse(f'I-G10 the latest grouping runs of Stokes I and V in {gdir} recorded different nights appended before '
                 f'(I: {len(nds["I"])}, V: {len(nds["V"])}; only in I: {sorted(set(nds["I"]) - set(nds["V"]))[:3]}, only in V: '
                 f'{sorted(set(nds["V"]) - set(nds["I"]))[:3]}) - the night\'s rate would be judged among one Stokes\' nights '
                 f'for both: group again')
    nd = nds['I']
    why = cur.extends(st['omission'])
    if why:
        u.refuse(f'I-G8 the deep-omission decision table of the grouping {gdir} ({cur.path}) is neither the one the gate read '
                 f'({st["omission"]["path"]}) nor an append to it: {why}')
    before = [_real(D) for L, D in nd if L == lst and _real(D) != _real(out)]
    try:
        summ = DO.append_night(cur.ref(), lst, night, out, os.path.join(out, DO.APPEND_DIR), search_out=_deep_search_roots(),
                               lst_dirs={lst: st['frozen_from']}, night_dirs={lst: before})
    except (rec.Refusal, DO.OmissionRefusal) as e:
        u.refuse(f'the night\'s deep-omission verdicts: {e}')
    u.add_input(cur.path)
    u.add_input(cur.cells_path)
    for p in summ.get('deep_record_files') or ():      # the deep search's record of the night: image ledgers, refusal files
        u.add_input(p)
    u.meta['omission_append'] = summ
    t = summ['table']
    if summ['status'] == 'appended':
        u.add_output(t['path'], 'omission_decisions', tier='sublst')
        u.add_output(t['cells_path'], 'omission_cells', tier='sublst', decisions=t['path'])
        u.add_output(os.path.join(os.path.dirname(t['path']), DO.APPEND_RECORD), 'omission_append', tier='sublst')
        u.note(f'night {night} of {lst}: {summ["appended_rows"]} decision row(s) ({summ["verdicts"]}) and '
               f'{summ["appended_cells"]} cell(s) appended to {cur.path} (sha256 {cur.sha256[:12]}) as {t["path"]} (sha256 '
               f'{t["sha256"][:12]}); earlier rows byte-identical')
    else:
        u.note(f'night {night} of {lst}: the table {cur.path} (sha256 {cur.sha256[:12]}) already decides it - nothing appended')
    return cur.ref(), t


def _group(run, u, st, out, parent, om):
    """The tier-wide grouping, incremental, with this night's directory appended last, filtered by ``om`` (the omit unit's
    table: I-G9, it must decide the night); its own run record in the grouping directory. Records the run in out/GROUP_RECORD.
    Runs under night_lock, right after the omit unit: the inputs are read again here (another item may have appended its
    night since the gate), the all-nights directories must be the gate's, and the latest runs must still name ``parent``."""
    rec, GS = _rec(), _gs()
    lst, gdir = st['lst'], st['grouping_dir']
    now = {}
    for S in ('I', 'V'):
        try:
            runs_s, nights_s = GS.recorded_inputs(gdir, S)
        except GS.StageRefusal as e:
            u.refuse(f'the grouping {gdir} before the night\'s run: {e}')
        now[S] = ({L: _real(D) for L, D in runs_s.items() if D}, [(L, _real(D)) for L, D in nights_s])
    if now['I'] != now['V']:
        u.refuse(f'the Stokes I and V groupings in {gdir} were assembled from different directories (changed since the gate)')
    runs, nd = now['I'][0], list(now['I'][1])
    if runs != st['hours']:
        u.refuse(f'the all-nights directories of the grouping {gdir} changed since the gate read them (a new all-nights pass): '
                 f'run the item again')
    if len(nd) != len(st['nights_before']):
        u.note(f'{len(nd) - len(st["nights_before"])} night director(ies) appended by another item since the gate: kept')
    why = night_keys_refusal(st, out, st['builds'])
    if why:
        u.refuse(why)
    if not any(L == lst and D == _real(out) for L, D in nd):
        nd.append((lst, _real(out)))
    hours = sorted(runs)
    for S in ('I', 'V'):                  # the latest runs still name the omit unit's parent (records only)
        try:
            ref = GS.recorded_omission_ref(gdir, S)
        except GS.StageRefusal as e:
            u.refuse(f'I-G8 {e}')
        if ref != parent:
            u.refuse(f'I-G8 the deep-omission decision table of the Stokes {S} grouping {gdir} changed since the omit unit read '
                     f'it: run the item again')
    try:                                  # I-G9: the night has verdicts in the table it is grouped under
        why = GS._bg().Omission(om['path'], om['sha256'], om['cells_sha256']).night_refusal(lst, st['night'])
    except GS._bg().Refused as e:
        why = str(e)
    if why:
        u.refuse(f'I-G9 {why}')
    gargs = argparse.Namespace(tier='sublst', stages=STAGE_GROUP, lsts=','.join(hours), run_label=None, group_input=None,
                               group_new_state=False, group_mode='incremental', group_stokes='I,V', group_omission=om['path'],
                               group_lst_dir=[f'{H}={runs[H]}' for H in hours], group_night_dir=[f'{L}={D}' for L, D in nd])
    argv = (['exopipe3 catalog', '--tier', 'sublst', '--stages', STAGE_GROUP, '--out', gdir, '--group-mode', 'incremental',
             '--group-omission', om['path'], '--lsts', gargs.lsts]
            + [x for H in hours for x in ('--group-lst-dir', f'{H}={runs[H]}')]
            + [x for L, D in nd for x in ('--group-night-dir', f'{L}={D}')])
    u.note(f'grouping run: {" ".join(argv)} (from `{" ".join(run.argv)}`)')
    _fin, last_p, _r = GS._latest_group_record(gdir, 'I')       # the last done record (the gate stood on it)
    with open(last_p, 'rb') as f:
        saved = (last_p, f.read())
    grun = rec.Run('catalog', argv, gdir, run.cfg, dry_run=False, config_sections=run.config_sections)
    grun.cores = getattr(run, 'cores', None)
    try:
        GS.run(gargs, grun)
    finally:
        summ = grun.write_summary()
    gu = [x for x in grun.units if x.name.endswith('.' + STAGE_GROUP)]
    if not gu or gu[-1].status != 'done':
        moved = _keep_last_done(gdir, gu[-1] if gu else None, saved)
        u.refuse(f'the incremental 10-minute grouping run with the night ({summ}; its unit record moved to {moved}) ended '
                 f'{gu[-1].status if gu else "without a unit"}: {(gu[-1].reason if gu else "")[:600]}. The grouping stands on '
                 f'its last done run ({last_p})')
    tables = {}
    for S in ('I', 'V'):
        try:
            runs2, nights2, (table, sha) = GS.recorded_inputs(gdir, S, with_table=True)
        except GS.StageRefusal as e:
            u.refuse(f'after the grouping run: {e}')
        held = [D for L, D in nights2 if L == lst and _real(D) == _real(out)]
        u.check(f'group.night-held.{S}', bool(held), value=len(nights2),
                detail=f'the latest Stokes {S} grouping run lists {len(nights2)} night director(ies)'
                       + ('' if held else f', not {out}'), path=table)
        tables[S] = dict(path=table, sha1=sha)
        u.add_input(table)
        try:                              # the run read the gate's table (the grouping run hashed it again)
            ref = GS.recorded_omission_ref(gdir, S)
        except GS.StageRefusal as e:
            u.refuse(f'after the grouping run: I-G8 {e}')
        if ref != om:
            u.refuse(f'I-G8 the grouping run with the night recorded another deep-omission decision table ({ref["path"]} sha256 '
                     f'{ref["sha256"][:12]}) than the omit unit\'s ({om["path"]} sha256 {om["sha256"][:12]})')
    doc = dict(version=1, lst=lst, night=st['night'], grouping_dir=gdir, run_summary=summ, unit=gu[-1].name, omission=om,
               record=rec.record_path(gdir, 'catalog', gu[-1].name), mode='incremental', tables=tables,
               hours=runs, nights=[list(x) for x in nd], counts=gu[-1].meta.get('summary'))
    p = os.path.join(out, GROUP_RECORD)
    rec.atomic_write_json(p, doc)
    u.meta['grouping'] = doc
    u.add_output(p, 'night_grouping_record')


def item_units():
    """The units of an item after the gate, in order: the night pass's stages (tier_sublst.NIGHT_DEFAULT), `omit`, `group`,
    then BUILD_STAGES."""
    return list(_ts().NIGHT_DEFAULT) + [STAGE_OMIT, STAGE_GROUP] + BUILD_STAGES.split(',')


def resume_point(run, unit, out, grouped):
    """Where an item run again with --resume resumes: (the first unit of item_units() that runs again (None: none), why,
    [(unit stage, why) of the units before it, recorded skipped-valid]).
    - The night pass's stages are judged by their records' status (done), as the tier's own --from / --stages gap check
      does (tier_sublst.check_from, which also restores the files later stages rewrite in place, master_catalog.csv among
      them: a stage's recorded input changes by design, so core/record.resume_check cannot judge these stages). Once the
      grouping holds the night from this directory (``grouped``), the night pass is never judged again: its stages must
      all be done (else the item REFUSES) and stand for the rows the grouping took, and `group` stands too (the tier-wide
      grouping is never run again for no new night).
    - build, buildv, view stand when core/record.resume_check passes (record done, clean tree, recorded inputs,
      code and config unchanged, outputs passing their content tests now: a later grouping run fails them)."""
    rec = _rec()
    stood = []
    for s in _ts().NIGHT_DEFAULT:
        r = rec.load_record(out, run.step, f'{unit}.{s}')
        st = (r or {}).get('status')
        if st != 'done':
            if grouped:
                raise rec.Refusal(f'--resume: the grouping holds the night from {out}, but its night-pass unit '
                                  f'{unit}.{s} is {st or "unrecorded"}: the directory does not hold the night pass the '
                                  f'grouping took')
            return s, f'{unit}.{s} is {st or "unrecorded"}', stood
        stood.append((s, 'the grouping holds the night from this directory: the night pass is not run again' if grouped
                      else 'record done (the next stage\'s --stages gap check vouches for its outputs)'))
    if not grouped:                       # omit and group run together, under one hold of the items' lock
        return STAGE_OMIT, 'the grouping does not hold the night yet', stood
    stood.append((STAGE_OMIT, 'the grouping holds the night from this directory under the table it recorded'))
    stood.append((STAGE_GROUP, 'the grouping holds the night from this directory: never grouped again for no new night'))
    for s in BUILD_STAGES.split(','):
        ok, why = run.resume_check(f'{unit}.{s}', inputs=None, admitted=None)
        if not ok:
            return s, why, stood
        stood.append((s, why))
    return None, 'every unit of the item stands', stood


def _skip(run, unit, stood, **meta):
    """Record the units that stand as skipped-valid in this run (their done records are kept; deep_chain's form)."""
    for s, why in stood:
        x = run.unit(f'{unit}.{s}', stage=s, **meta)
        x.status, x.reason = 'skipped-valid', f'resume: {why}'
        print(f'{unit}.{s}: skipped (--resume: {why})', file=sys.stderr)


def _plan_lines(st, lst, night, out):
    yield f'night pass: catalog --tier sublst --lsts {lst} --sublst-night {night} --sublst-frozen-from {st["frozen_from"]} --out {out}'
    yield (f'verdicts: the night rated and appended to the grouping\'s deep-omission decision table '
           f'{st["omission"]["path"]} (or one appended to it since) as {os.path.join(out, "deep_omission", "deep_omission.csv")}')
    yield (f'grouping: catalog --tier sublst --stages group --group-mode incremental --out {st["grouping_dir"]} over the '
           f'{len(st["hours"])} hour(s) of the latest run + {len(st["nights_before"])} night(s) appended before + this one')
    yield f'builds and views: the night pass with --stages {BUILD_STAGES} (scope {", ".join(st["scope"])})'


def run(args, run, epochs):
    """The per-hour 10-minute item (module text). Returns an exit code or None (the units decide)."""
    cfg = _cfg()
    rec, TS, GS = _rec(), _ts(), _gs()
    from exopipe.phase3.core import cells
    c = cfg.catalog_sublst
    if 'sublst' not in cfg.catalog_incremental.TIERS:
        raise rec.Refusal(f'catalog --incremental: tier sublst has no incremental path (catalog_incremental.TIERS = '
                          f'{cfg.catalog_incremental.TIERS})')
    if not epochs or len(epochs) != 1:
        raise rec.Refusal(f'catalog --tier sublst --incremental takes ONE item <LST>_<YYYY-MM-DD> per invocation (got '
                          f'{len(epochs or [])}; the per-hour chain passes one): --out is the night\'s own directory')
    lst, night = cells.parse_epoch(epochs[0])
    named = [flag for dest, flag in OWN_ARGS if getattr(args, dest, None) not in (None, '', [], False)]
    if named:
        raise rec.Refusal(f'catalog --tier sublst --incremental runs its own stages and grouping: do not give {", ".join(named)}')
    if (getattr(args, 'group_stokes', None) or 'I,V').replace(' ', '') != 'I,V':
        raise rec.Refusal(f'--group-stokes {args.group_stokes}: the per-hour item groups both Stokes (I,V) in one run')
    if getattr(args, 'lsts', None) and getattr(args, 'lsts') != lst:
        raise rec.Refusal(f'--lsts {args.lsts} is not the item\'s hour {lst}')
    arm = getattr(args, 'sublst_arm', None) or c.ARM
    unit = f'{lst}_{arm}'
    out = os.path.abspath(run.out_root)
    gdir = os.path.abspath(getattr(args, 'sublst_grouping', None) or GS.sublst_dir(cfg))
    raw = getattr(args, 'sublst_builds', None)
    builds = tuple(b.strip() for b in raw.split(',') if b.strip()) if raw else tuple(c.BUILDS)
    vh = getattr(args, 'sublst_view_hours', None)
    try:
        view_hours = sorted({cells.parse_lst(x.strip()) for x in str(vh).split(',') if x.strip()}) if vh else None
    except ValueError as e:
        raise rec.Refusal(f'--sublst-view-hours: {e}')
    print(f'catalog --tier sublst --incremental {lst}_{night}: night directory {out}; grouping {gdir}; builds '
          f'{",".join(builds)}', file=sys.stderr)
    if run.dry_run:
        try:
            st = frozen_state(lst, night, arm, out, gdir, builds, view_hours, cfg)
        except rec.Refusal as e:
            print(f'[{unit}.{STAGE_GATE}] REFUSES: {e}')
            return rec.EXIT_REFUSED
        except rec.NoInput as e:
            print(f'[{unit}.{STAGE_GATE}] no input: {e}')
            return rec.EXIT_NO_INPUT
        print(f'[{unit}.{STAGE_GATE}] frozen state found: all-nights directory {st["frozen_from"]}; grouping '
              f'{st["grouping"]["I"]["record"]}')
        for ln in _plan_lines(st, lst, night, out):
            print(f'  {ln}')
        _night_pass(run, _night_args(args, lst, night, st['frozen_from'], gdir), unit, None)
        return 0
    os.makedirs(out, exist_ok=True)
    # 1. THE GATE: the frozen state must exist (refuses), and the night must not be in it already (no-input)
    with run.unit(f'{unit}.{STAGE_GATE}', tier='sublst', lst=lst, arm=arm, night=night, stage=STAGE_GATE) as u:
        try:                                      # read under the items' lock: never while another item's grouping runs
            with night_lock(gdir, c.NIGHT_LOCK_WAIT_S, f'{lst}_{night} (gate)'):
                st = frozen_state(lst, night, arm, out, gdir, builds, view_hours, cfg)
        except GS.StageRefusal as e:
            u.refuse(f'the gate could not read the grouping: {e}')
        for S, g in st['grouping'].items():
            u.add_input(g['record'])
            u.add_input(g['table'])
        u.add_input(st['omission']['path'])
        u.add_input(st['omission']['cells_path'])
        u.add_input(st['conform']['record'])
        u.add_input(st['conform']['noise'])
        u.add_input(st['conform']['evidence'])
        u.check('frozen-state.records', True, value=sum(len(v) for v in st['frozen_records'].values()),
                detail=f'{len(st["scope"])} hour(s) in scope, frozen records of every build and key with rows; all-nights '
                       f'directory of {lst}: {st["frozen_from"]}')
        # an item run again: the grouping may already hold the night from this directory (an earlier attempt grouped it)
        grouped = any(L == lst and D == _real(out) for L, D in st['nights_before'])
        if grouped and not run.resume:
            u.refuse(f'the 10-minute grouping {gdir} already holds the night {night} of {lst} from this directory {out} '
                     f'(an earlier attempt of the item grouped it): run the item with --resume (the chain line does), which '
                     f'continues at its first unit that does not stand; a night pass run again over a grouped night would '
                     f'change rows the grouping holds (a grouped night is judged again only by the audit pass, ruling 4 of '
                     f'R-GREGG-20261112-PER-HOUR-STEPS-SIX-RULINGS, not built)')
        done_before = [s for s in TS.NIGHT_DEFAULT
                       if (rec.load_record(out, run.step, f'{unit}.{s}') or {}).get('status') == 'done']
        if done_before and not run.resume:      # a night pass run again over its own outputs reads stale later-stage files
            u.refuse(f'the night directory {out} holds an earlier attempt of the item (night-pass unit(s) done: '
                     f'{", ".join(done_before)}): run the item with --resume (the chain line does), which continues at its '
                     f'first unit that does not stand, or give it an empty night directory')
        start, why, stood = resume_point(run, unit, out, grouped) if run.resume else (TS.NIGHT_DEFAULT[0], 'no --resume', [])
        u.meta['resume'] = dict(start=start, why=why, standing=[s for s, _w in stood], grouped=grouped)
        p = os.path.join(out, GATE_RECORD)
        rec.atomic_write_json(p, st)
        u.meta['frozen_state'] = dict(frozen_from=st['frozen_from'], grouping_dir=st['grouping_dir'], scope=st['scope'],
                                      nights_before=len(st['nights_before']))
        u.add_output(p, 'night_frozen_state')
    if u.status != 'done':
        return None
    meta = dict(tier='sublst', lst=lst, arm=arm, night=night)
    units = item_units()
    if start is None:
        _skip(run, unit, stood, **meta)
        print(f'catalog --tier sublst --incremental {lst}_{night}: every unit stands (--resume)', file=sys.stderr)
        return None
    _skip(run, unit, stood, **meta)
    i0 = units.index(start)
    # 2. THE NIGHT PASS, preflight..veto (from the resume point)
    if start in TS.NIGHT_DEFAULT:
        ran = list(TS.NIGHT_DEFAULT[TS.NIGHT_DEFAULT.index(start):])
        done = _night_pass(run, _night_args(args, lst, night, st['frozen_from'], gdir,
                                            None if ran == list(TS.NIGHT_DEFAULT) else ','.join(ran)), unit, None)
        m = done.get('merge')
        if m is not None and m.status == 'no-input':
            print(f'catalog --tier sublst --incremental {lst}_{night}: no row of the night in the search jobs ({m.reason})',
                  file=sys.stderr)
            return rec.EXIT_NO_INPUT
        if [s for s in ran if s not in done or done[s].status != 'done']:
            return None
    bstages = [s for s in BUILD_STAGES.split(',') if units.index(s) >= i0]
    first = STAGE_OMIT if units.index(STAGE_OMIT) >= i0 else bstages[0]
    # 3. THE GROUPING, incremental, with the night, and 4. THE DELTA BUILDS AND THE VIEWS: one item at a time (night_lock)
    with contextlib.ExitStack() as held:
        try:
            held.enter_context(night_lock(gdir, c.NIGHT_LOCK_WAIT_S, f'{lst}_{night}'))
        except GS.StageRefusal as e:              # on the first unit that would run: a standing group record is kept
            with run.unit(f'{unit}.{first}', tier='sublst', lst=lst, arm=arm, night=night, stage=first) as u:
                u.refuse(str(e))
            return None
        if first == STAGE_OMIT:          # 3a. the night's verdicts, then 3b. the grouping under them (one hold of the lock)
            with run.unit(f'{unit}.{STAGE_OMIT}', tier='sublst', lst=lst, arm=arm, night=night, stage=STAGE_OMIT) as u:
                parent, om = _rate_night(u, st, out)
            if u.status != 'done':
                return None
            with run.unit(f'{unit}.{STAGE_GROUP}', tier='sublst', lst=lst, arm=arm, night=night, stage=STAGE_GROUP) as u:
                _group(run, u, st, out, parent, om)
            if u.status != 'done':
                return None
        _night_pass(run, _night_args(args, lst, night, st['frozen_from'], gdir, ','.join(bstages)), unit, ','.join(bstages))
    return None
