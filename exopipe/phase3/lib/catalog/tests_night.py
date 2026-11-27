"""Content tests of the 10-minute night pass's views and of the per-hour item's records (catalog --tier sublst
--sublst-night ... --stages view; catalog --tier sublst --incremental). Registered with core.validate.content_test; run by
UnitRecord.add_output when the stage writes the product, and again by --resume (core/record.resume_check retest).

  catalog_view          a view root <night directory>/views/<LST> (lib/catalog/delta_build.view_root + finish_root)
  hour_views_index      the night pass's index <night directory>/hour_views.json (tier_sublst.st_view)
  night_frozen_state    the item's gate record <night directory>/frozen_state.json (incremental_sublst.frozen_state)
  night_grouping_record the item's grouping record <night directory>/night_grouping.json (incremental_sublst._group)

Each test reads the files again; none trusts the writer's own lists. A view and the index FAIL once a later grouping run
replaced the table they were made under (they stand for an earlier grouping; --resume builds and views again). The item's
two records are records of what the gate saw and what the grouping took: they are tested for their own claims, never
against the grouping table's present bytes (a later night's run replaces it by design).
"""
import glob
import json
import os

from exopipe.phase3.core.validate import content_test


def _json(p):
    with open(p) as f:
        return json.load(f)


@content_test('catalog_view')
def catalog_view_root(path, **ctx):
    """A view root of one LST hour: its record hour_view.json is complete; every catalog it lists exists and passes
    delta_build.view_test again from its files (the all-nights catalog, the delta and the grouping table its own view
    record catalog_view.json names; the table must still be that table by sha1, and the delta the one the root records)
    and carries its own view rows (builder_view.json, named in catalog_view.json); the conformance read passes delta_build.conform_test; the
    reference-tie table is the one recorded (or absent when none was); a build whose catalogs hold survivors has its
    localization table, not empty."""
    from exopipe.phase3.lib.catalog import delta_build as DB
    p = os.path.join(path, DB.ROOT_RECORD)
    if not os.path.isfile(p):
        return False, None, f'{path} holds no {DB.ROOT_RECORD}'
    rec = _json(p)
    if rec.get('complete') is not True:
        return False, None, f'{p}: the root is not complete (no localizations recorded)'
    msgs, grp, n = [], {}, 0
    for name, c in sorted((rec.get('catalogs') or {}).items()):
        tag, key = name.split('_', 1)
        out = os.path.join(path, DB.CAT.format(tag=tag, key=key))
        if not os.path.isdir(out):
            msgs.append(f'{name}: no directory {out}')
            continue
        try:
            vr = _json(os.path.join(out, DB.VIEW_RECORD))
            table, sha = vr['grouping_table']['path'], vr['grouping_table']['sha1']
        except (OSError, ValueError, KeyError, TypeError) as e:
            msgs.append(f'{name}: no readable view record {DB.VIEW_RECORD} ({type(e).__name__})')
            continue
        if not ((vr.get('view_record') or {}).get('sha256') and os.path.isfile(os.path.join(out, DB.VIEW_ROWS))):
            msgs.append(f'{name}: no own view rows ({DB.VIEW_RECORD} naming {DB.VIEW_ROWS})')
        if (vr.get('delta') or None) != (c.get('delta') or None):
            msgs.append(f'{name}: the view record names the delta {vr.get("delta")}, the root {c.get("delta")}')
        if not os.path.isfile(table):
            msgs.append(f'{name}: the grouping table {table} is gone')
            continue
        if table not in grp:
            grp[table] = DB.read_grouping(table)
        if grp[table][2] != sha:
            msgs.append(f'{name}: the grouping table {table} is no longer the one the view was made under (sha1 '
                        f'{grp[table][2][:12]}, recorded {sha[:12]}): the view stands for an earlier grouping')
            continue
        ok, detail, _c = DB.view_test(out, vr['all_nights'], vr.get('delta'), grp[table])
        if not ok:
            msgs.append(f'{name}: {detail}')
        n += 1
    if not n and not msgs:
        msgs.append('the root lists no catalog')
    ok, why = DB.conform_test(path, rec['all_nights'], rec['nights'])
    if not ok:
        msgs.append(f'conformance read: {why}')
    tp, want = os.path.join(path, DB.TIE), (rec.get('tie') or {}).get('sha256')
    if (want is None) != (not os.path.exists(tp)) or (want and want != DB._sha256(tp)):
        msgs.append(f'{DB.TIE} is not the table the root recorded')
    for tag in rec.get('tags') or []:
        if glob.glob(os.path.join(path, DB.CAT.format(tag=tag, key='*'), '*_sources_survivors.csv')):
            lp = os.path.join(path, DB.LOC_DIR.format(tag=tag), DB.LOC_TABLES[0])
            if not (os.path.isfile(lp) and os.path.getsize(lp) > 0):
                msgs.append(f'{tag}: survivors but no localization table {lp}')
    return (not msgs, n, '; '.join(msgs) if msgs else
            f'{n} catalog(s) pass view_test again under their grouping tables; conformance read, tie table and localizations '
            f'as recorded')


@content_test('hour_views_index')
def hour_views_index(path, **ctx):
    """The night pass's index: it names the night and at least one hour; every hour is viewed as its all-nights directory
    (root = all_nights, which exists) or as a view root that stands for exactly the hour's recorded state digest
    (delta_build.root_stands); every key of a hour carries a certificate; the grouping tables are named with their sha1."""
    from exopipe.phase3.lib.catalog import delta_build as DB
    try:
        ix = _json(path)
    except (OSError, ValueError) as e:
        return False, None, f'{path}: not readable JSON ({type(e).__name__})'
    msgs = []
    hours = ix.get('hours') or {}
    if not ix.get('night'):
        msgs.append('no night named')
    if not hours:
        msgs.append('no hour viewed')
    for S, t in sorted((ix.get('tables') or {}).items()):
        if not (t.get('path') and t.get('sha1')):
            msgs.append(f'the Stokes {S} grouping table is not named with its sha1')
    if not ix.get('tables'):
        msgs.append('no grouping table named')
    for L, e in sorted(hours.items()):
        kind, root = e.get('kind'), e.get('root')
        if kind == 'all-nights':
            if not (root and os.path.realpath(root) == os.path.realpath(e.get('all_nights') or '') and os.path.isdir(root)):
                msgs.append(f'{L}: an all-nights view whose root is not its existing all-nights directory')
        elif kind == 'view':
            if not (root and DB.root_stands(root, e.get('state_digest'))):
                msgs.append(f'{L}: the view root {root} does not stand for the recorded state')
        else:
            msgs.append(f'{L}: unknown kind {kind!r}')
        bad = [k for k, v in (e.get('keys') or {}).items() if not v.get('certificate')]
        if bad:
            msgs.append(f'{L}: key(s) without a certificate: {", ".join(sorted(bad))}')
    return (not msgs, len(hours), '; '.join(msgs) if msgs else
            f'{len(hours)} hour(s): ' + ', '.join(f'{L} {e.get("kind")}' for L, e in sorted(hours.items())))


@content_test('night_frozen_state')
def night_frozen_state(path, **ctx):
    """The per-hour item's gate record, tested for its own claims: it names the hour, the arm, the night, the all-nights
    directory (which holds the image table) and the done grouping run of each of Stokes I and V with its table and sha1; the
    night's hour is in scope; every builder's frozen record it found is still there (the all-nights build's, never
    rewritten) and names the deep-omission decision table the record names, which is unchanged (gate I-G8: the frozen state
    stands on one omission, never replaced by a night); and gate I-G6 holds again (tier_sublst.frozen_records_refusal: the
    conformance records the night was judged by, as the all-nights directory's own conformance record lists them)."""
    try:
        st = _json(path)
    except (OSError, ValueError) as e:
        return False, None, f'{path}: not readable JSON ({type(e).__name__})'
    msgs = []
    for k in ('lst', 'arm', 'night', 'frozen_from', 'grouping', 'conform', 'scope'):
        if not st.get(k):
            msgs.append(f'no {k}')
    if msgs:
        return False, None, '; '.join(msgs)
    if sorted(st['grouping']) != ['I', 'V']:
        msgs.append(f'grouping runs of {sorted(st["grouping"])}, not I and V')
    for S, g in sorted(st['grouping'].items()):
        if not all(g.get(k) for k in ('record', 'table', 'sha1', 'finished')):
            msgs.append(f'the Stokes {S} grouping run is not named with its record, table, sha1 and finish time')
    if st['lst'] not in st['scope']:
        msgs.append(f'the night\'s own hour {st["lst"]} is not in scope')
    if not os.path.isfile(os.path.join(st['frozen_from'], 'v_deep_images.csv')):
        msgs.append(f'{st["frozen_from"]} holds no v_deep_images.csv')
    om = st.get('omission') or {}
    omission = None
    if not (om.get('path') and om.get('sha256') and om.get('cells_sha256')):
        msgs.append('no deep-omission decision table named with its coverage record (I-G8)')
    else:
        from exopipe.phase3.lib.catalog.wraps import builder_grouping as BG
        try:                              # the one reading of the table (wraps/builder_grouping.Omission)
            omission = BG.Omission(om['path'], om['sha256'], om['cells_sha256'])
        except BG.Refused as e:
            msgs.append(f'the deep-omission decision table the gate read: {e} (I-G8)')
        else:
            why = omission.night_refusal(st['lst'], st['night'])
            if why:
                msgs.append(f'{why} (I-G9)')
    n = 0
    for H, keys in sorted((st.get('frozen_records') or {}).items()):
        for k, fp in sorted(keys.items()):
            n += 1
            if not os.path.isfile(fp):
                msgs.append(f'{H} {k}: the frozen record {fp} is gone')
            elif omission is not None:
                try:
                    have = _json(fp).get('omission') or {}
                except (OSError, ValueError):
                    have = {}
                if not omission.same(have):
                    msgs.append(f'{H} {k}: the frozen record {fp} names another deep-omission decision table (I-G8)')
    c = st['conform']
    paths = [c.get('noise'), c.get('evidence')]
    if not all(paths):
        msgs.append('the conformance records are not named')
    else:
        from exopipe.phase3.lib.catalog import tier_sublst as TS
        why, _fin = TS.frozen_records_refusal(st['frozen_from'], f'{st["lst"]}_{st["arm"]}', paths)
        if why:
            msgs.append(f'the conformance records the night is judged by: {why}')
    return (not msgs, n, '; '.join(msgs) if msgs else
            f'{st["lst"]} {st["night"]}: frozen from {st["frozen_from"]}; {n} frozen builder record(s) present, each '
            f'naming the unchanged deep-omission decision table; both grouping runs named; I-G6 holds')


@content_test('night_grouping_record')
def night_grouping_record(path, **ctx):
    """The per-hour item's grouping record, tested for its own claims: an incremental run whose night list holds this night
    directory (the record's own directory) once; both Stokes' tables named with their sha1; the run's record named (a later
    run of the same unit name rewrites that file, so its present content is not this record's claim)."""
    try:
        g = _json(path)
    except (OSError, ValueError) as e:
        return False, None, f'{path}: not readable JSON ({type(e).__name__})'
    msgs = []
    if g.get('mode') != 'incremental':
        msgs.append(f'mode {g.get("mode")!r}, not incremental')
    here = os.path.realpath(os.path.dirname(os.path.abspath(path)))
    held = [d for L, d in (g.get('nights') or []) if os.path.realpath(d) == here]
    if len(held) != 1:
        msgs.append(f'the night list holds this directory {len(held)} time(s)')
    tables = g.get('tables') or {}
    if sorted(tables) != ['I', 'V']:
        msgs.append(f'tables of {sorted(tables)}, not I and V')
    for S, t in sorted(tables.items()):
        if not (t.get('path') and t.get('sha1')):
            msgs.append(f'the Stokes {S} table is not named with its sha1')
    if not g.get('record'):
        msgs.append('the grouping run\'s record is not named')
    return (not msgs, len(g.get('nights') or []), '; '.join(msgs) if msgs else
            f'incremental grouping with {len(g.get("nights") or [])} night director(ies), this one once; both tables named')
