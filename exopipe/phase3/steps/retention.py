"""exopipe3 retention - two-week retention of the raw night data: the retention table, extraction holds, the dry run (default) and
the ruled delete. Plan: plan/ORCHESTRATION_DESIGN_20261117.md requirement 4, 5.4 and section 6 step 5; Gregg 2026-11-17
(R-GREGG-20261117-ORCHESTRATION-DESIGN-COMMENTS item 5, verbatim): "The night time observing will ony be ~10 -12 hours a night
and the data are averaged by 4 channels, so it's <100 Tb for 2 weeks retention."

    exopipe3 retention                                    # = plan: the dry run over retention.RAW_ROOT (writes the plan table
                                                          # and, when items are expired, a DRYRUN_ manifest; deletes nothing)
    exopipe3 retention plan --nights 2026-11-16 --scopes 73MHz,78MHz
    exopipe3 retention record --start 2026-11-16T02:00:00Z --end 2026-11-16T03:00:00Z --produced 2026-11-16T05:10:00Z \\
        --source "epoch chain 07h_2026-11-16 job <id>"   # a production record (the per-hour driver writes these)
    exopipe3 retention hold --start 2026-11-16T02:10:00Z --end 2026-11-16T02:20:00Z --scopes 23MHz --reason "<candidate, ID>"
    exopipe3 retention release --hold-id H-... --reason "extracted: <manifest>, <ledger ID>"
    exopipe3 retention delete --manifest <out>/manifests/DRYRUN_RETENTION_<UTC>.csv --ruling-for retention.DELETE=R-GREGG-...

Actions and units (records <out>/_records/retention/<unit>.json):
  plan     unit plan_<UTC>. Lists the raw tree by name (subband, UT date and UT hour directories; never a walk of the tree) and
           classifies each (subband, date, hour) item from the do-not-delete cells of data_decisions.csv, the production
           records and the active holds (lib/retention/plan.py: protected, held, unprocessed, unrecognized, empty, partial,
           retained, expired). PROTECTED (lib/retention/protect.py): the item's data overlap a UTC range of a cell (LST hour
           + date: every range of that date in the LST hour, core/sidereal.py, widened by retention.DO_NOT_DELETE_PAD_S on
           both sides) that a data_decisions.csv row marks "do not delete" - in ANY subband, 13 MHz included, since a cell is
           a time range on the sky; never listed, never in a manifest. A do-not-delete row read with an irregularity (more
           fields than the header, the text outside the reason column) protects its cell and is named in the plan's note.
           Writes <out>/plans/RETENTION_TABLE_<UTC>.csv
           (every item; class retention_plan) and, when items are expired, <out>/manifests/DRYRUN_RETENTION_<UTC>.csv + .json
           (class retention_manifest): the oldest-expiring retention.MAX_ITEMS_PER_MANIFEST expired items and every path a
           delete would remove - each file, and each file and directory of an MS directory (listed inside that MS only) - with
           its size, mtime (ns) and inode from os.lstat (and its sha256 with --checksum content), per item an identity sha256. No
           expired item: no manifest (the note says so). The sidecar records the records the plan read (the real path of
           --out, the size and sha256 of the table and the holds file). --now UTC evaluates the expiry at another time (a
           what-if; the delete always uses the clock). With --dry-run: listed and classified only - no lstat, no checksum,
           nothing written; the counts are printed.
  record   unit record_<UTC>: appends a production record to <out>/retention.TABLE_NAME (class retention_records): the raw
           sets of --scopes (default every subband) that start in [--start, --end) went through the pipeline at --produced
           (default now); --source names the chain, job or ledger ID. Expiry = the latest covering production + RETENTION_DAYS.
  hold     unit hold_<UTC>: appends a hold of [--start, --end) on --scopes to <out>/retention.HOLDS_NAME (retention_holds).
  release  unit release_<UTC>: appends the release of --hold-id.
  delete   one unit per item, delete_<subband>_<date>_<hour>. REFUSES, deleting nothing, when any item of --manifest lies in a
           do-not-delete cell of data_decisions.csv as read at the delete (RT-DO-NOT-DELETE, the first check of the
           preflight; the reason names the items and the cells; Gregg's go for the manifest does not lift it: the cell's row
           does), and re-checks each item right before deleting it (re-reading the cells only when the file's sha1 changed).
           REFUSES unless: --out is retention.ROOT, so
           the records and holds are the production ones (RT-ROOT); --manifest is a dry-run manifest of this tree whose sidecar
           sha256 still matches, planned from the records of this --out (RT-MANIFEST); --ruling-for retention.DELETE=<ledger ID>
           names a row of paths.FIX_LEDGER whose fix_id starts with retention.RULING_PREFIX (R-GREGG-), whose authorized_by
           begins with the word Gregg, whose status begins with RULED (any case) and whose text quotes this manifest's sha256 -
           in full or its first 12 hex digits or more; its file name alone does not count (RT-RULING: Gregg's go is for one
           manifest; code cannot prove who wrote a row - the guard is procedural, no session writes an R-GREGG row without
           Gregg's words, plus these checks); the table and the holds file only grew since the plan (their planned bytes still
           hash to the planned digests: RT-RECORDS); and, checked for EVERY item before anything is deleted, the item directory
           holds exactly the paths listed with the
           same kind, size, mtime and inode (and sha256 for a content manifest) (RT-MATCH), is still expired by the records now
           (RT-NOT-EXPIRED) and is not held now (RT-HELD). Then, item by item, it unlinks the listed files (each re-checked right
           before), removes the listed MS directories deepest first with os.rmdir (empty directories only; never a recursive
           removal), writes each removed path to <out>/manifests/APPLIED_RETENTION_<UTC>.csv (class retention_applied: every
           listed path gone) and removes the emptied hour directory (and the date directory when it is empty). The first item
           that does not end done stops the delete. Deleting raw data always needs Gregg's go: there is no other path to a
           delete in this step.
Inputs:  the raw tree retention.RAW_ROOT (names; os.lstat of the paths of manifest items only); <out>/retention.TABLE_NAME and
         <out>/retention.HOLDS_NAME; paths.FIX_LEDGER (delete); paths.DATA_DECISIONS through core/exclusions (its
         do-not-delete rows; plan and delete). Every key: `exopipe3 config --section retention`.
Failure modes (exit 2 = refused): RT-ROOT the raw root is not a directory, or a delete's --out is not retention.ROOT;
         RT-TABLE a table or holds row that cannot be read;
         RT-RECORD / RT-HOLD an invalid record, hold or release; RT-CHECKSUM an unknown checksum kind; RT-MANIFEST, RT-RULING,
         RT-RECORDS, RT-MATCH, RT-NOT-EXPIRED, RT-HELD (delete, above); RT-DO-NOT-DELETE (plan and delete) a manifest item in
         a do-not-delete cell, or - fail closed - data_decisions.csv missing or unreadable, a do-not-delete row whose LST,
         date or subband cannot be read, a row with deletion wording (delet, keep raw, retain, preserv, don't) but not
         "do not delete", or a cell whose UTC ranges cannot be computed. No space gate (SPACE_GATE below). Exit 3:
         the tree holds no item in the selection.
Selectors: --nights (UT dates of the tree) and --scopes (subbands) restrict plan; record and hold take --scopes as the
         subbands they cover.
NOT YET: holds from a review-page action (the command only); extraction of a held range to a kept area with a checksum
         manifest (5.4's last line); the per-hour driver's call of record_production (plan row 37).
"""
import os
import sys

SUMMARY = 'raw night data kept 2 weeks: retention table, extraction holds, expiry plan (dry run) and the ruled delete'
TIERS = ()
SELECTORS = ('nights', 'scopes')
CONFIG = ('retention', 'paths', 'exclusions', 'instrument')
WRITES = True
PRODUCTS = False          # writes tables and manifests, no data product (no GITTAG to stamp)
SPACE_GATE = False        # the one writer step without the space gate (cli.main; core/space.py): its delete FREES room, and
                          # the plan and the holds a delete needs write kilobytes to a few megabytes - refusing them when the
                          # quota is low would keep it full. The raw data's own writer passes the gate
                          # (`python -m exopipe.phase3.core.space --path <raw root>`, plan/ORCHESTRATION_DESIGN_20261117.md 5.4)
ACTIONS = ('plan', 'record', 'hold', 'release', 'delete')


def add_arguments(p):
    p.add_argument('action', nargs='?', default='plan', choices=ACTIONS, help='what to do (default plan: the dry run)')
    g = p.add_argument_group('retention')
    g.add_argument('--start', metavar='UTC', help='record / hold: start of the range (YYYY-MM-DDTHH:MM:SSZ)')
    g.add_argument('--end', metavar='UTC', help='record / hold: end of the range (exclusive)')
    g.add_argument('--produced', metavar='UTC', help='record: when the range went through the pipeline (default now)')
    g.add_argument('--source', help='record: the chain, job or ledger ID that produced the range')
    g.add_argument('--reason', help='hold / release: why (the candidate, the extraction, a ledger ID)')
    g.add_argument('--hold-id', help='release: the hold released')
    g.add_argument('--manifest', metavar='CSV', help='delete: the dry-run manifest Gregg approved')
    g.add_argument('--checksum', choices=('identity', 'content'),
                   help='plan: identity (size, mtime, inode; reads no data) or content (also sha256 of every byte) '
                        '(default retention.CHECKSUM)')
    g.add_argument('--max-items', type=int, help='plan: items in one manifest (default retention.MAX_ITEMS_PER_MANIFEST)')
    g.add_argument('--now', metavar='UTC', help='plan: evaluate the expiry at this time instead of now (a what-if)')


def default_out(args):
    from exopipe.phase3.config import CFG
    return CFG.retention.ROOT


def _set(text):
    return {s.strip() for s in (text or '').replace(';', ',').split(',') if s.strip()} or None


def _nights(args):
    if not args.nights:
        return None
    if args.nights.startswith('@'):
        with open(args.nights[1:]) as f:
            return {ln.strip() for ln in f if ln.strip() and not ln.startswith('#')}
    return _set(args.nights)


def _time(text, what):
    from exopipe.phase3.core.record import Refusal
    from exopipe.phase3.lib.retention import table as T
    if not text:
        raise Refusal('RT-ARGS: %s is required' % what)
    try:
        return T.parse_utc(text)
    except ValueError as e:
        raise Refusal('RT-ARGS: %s: %s' % (what, e))


def _delete_ruling(args, CFG):
    """The ledger ID of --ruling-for retention.DELETE=<ID> (cli.main has checked that it is a row of paths.FIX_LEDGER)."""
    key = CFG.retention.DELETE_RULING_KEY
    ids = [r.split('=', 1)[1].strip() for r in (args.ruling_for or []) if '=' in r and r.split('=', 1)[0].strip() == key]
    return ids[-1] if ids else None


def run(args, run):
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import ledger, record
    from exopipe.phase3.core.record import Refusal
    from exopipe.phase3.lib.retention import plan as P
    from exopipe.phase3.lib.retention import protect as D
    from exopipe.phase3.lib.retention import table as T
    c = CFG.retention
    out = run.out_root
    st = P.stamp()
    if args.action != 'delete' and _delete_ruling(args, CFG):
        raise Refusal('RT-ARGS: --ruling-for %s is for delete only' % c.DELETE_RULING_KEY)
    if args.now and args.action != 'plan':
        raise Refusal('RT-ARGS: --now is a what-if of plan; %s always uses the clock' % args.action)

    if args.action == 'plan':
        now = _time(args.now, '--now') if args.now else None
        pl = P.make_plan(c.RAW_ROOT, out, now=now, nights=_nights(args), subbands=_set(args.scopes),
                         checksum=args.checksum, max_items=args.max_items, stat=not run.dry_run)
        by = pl.counts()
        irr = D.irregular_note(pl.protect)   # do-not-delete rows that protect their cells but want fixing
        if irr:
            print('retention plan: %s' % irr, file=sys.stderr)
        if run.dry_run:                  # names and statuses only: no lstat, no checksum, nothing written
            print('retention plan (dry run: listed and classified only - no stat, no checksum, nothing written): %d items '
                  '(%s); %d expired, %d would be in the manifest%s' % (
                      len(pl.items), ', '.join('%s %d' % (k, by[k]) for k in P.STATUSES if k in by) or 'none',
                      pl.n_expired, len(pl.batch), '; %d names outside the layout' % len(pl.odd) if pl.odd else ''),
                  file=sys.stderr)
            return record.EXIT_DONE
        nb = sum(r['bytes'] for _i, _c, rs in pl.batch for r in rs)
        line = ('%d items (%s); %d expired, %d in the manifest (%d files, %d directories, %.3f TB)%s' % (
            len(pl.items), ', '.join('%s %d' % (k, by[k]) for k in P.STATUSES if k in by) or 'none', pl.n_expired,
            len(pl.batch), sum(r['kind'] != 'dir' for _i, _c, rs in pl.batch for r in rs),
            sum(r['kind'] == 'dir' for _i, _c, rs in pl.batch for r in rs), nb / 1e12,
            '; %d names outside the layout' % len(pl.odd) if pl.odd else ''))
        if not pl.items and not pl.odd:
            with run.unit('plan_%s' % st, action='plan'):
                raise record.NoInput('no item under %s in this selection' % c.RAW_ROOT)
            return run.exit_code()
        with run.unit('plan_%s' % st, action='plan', raw_root=c.RAW_ROOT) as u:
            plan_csv, man_csv, man_json = P.write_plan(pl, out)
            u.add_output(plan_csv, 'retention_plan')
            if man_csv:
                u.add_output(man_csv, 'retention_manifest')
                u.note('dry-run manifest %s (sha256 %s): Gregg\'s go for it is a ledger row quoting that sha256; then '
                       'exopipe3 retention delete --out %s --manifest %s --ruling-for %s=<ID>' % (
                           man_csv, P.sha256_file(man_csv), c.ROOT, man_csv, c.DELETE_RULING_KEY))
            else:
                u.note('no expired item: no manifest written')
            u.note(line)
            if irr:
                u.note(irr)
        print('retention plan: %s; table %s%s' % (line, plan_csv, '; manifest %s' % man_csv if man_csv else ''),
              file=sys.stderr)
        return run.exit_code()

    if args.action in ('record', 'hold'):
        start, end = _time(args.start, '--start'), _time(args.end, '--end')
        sbs = args.scopes or '*'
        if args.action == 'record':
            produced = _time(args.produced, '--produced') if args.produced else None
            if run.dry_run:
                print('would record production of %s .. %s (%s)' % (T.iso(start), T.iso(end), sbs), file=sys.stderr)
                return record.EXIT_DONE
            with run.unit('record_%s' % st, action='record') as u:
                import time as _t
                rid = T.record_production(out, start, end, sbs, _t.time() if produced is None else produced, args.source)
                u.add_output(T.table_path(out), 'retention_records')
                u.note('production record %s: %s .. %s, %s' % (rid, T.iso(start), T.iso(end), sbs))
        else:
            if run.dry_run:
                print('would hold %s .. %s (%s)' % (T.iso(start), T.iso(end), sbs), file=sys.stderr)
                return record.EXIT_DONE
            with run.unit('hold_%s' % st, action='hold') as u:
                hid = T.hold(out, start, end, sbs, args.reason)
                u.add_output(T.holds_path(out), 'retention_holds')
                u.note('hold %s: %s .. %s, %s: %s' % (hid, T.iso(start), T.iso(end), sbs, args.reason))
                print('hold %s' % hid, file=sys.stderr)
        return run.exit_code()

    if args.action == 'release':
        if not args.hold_id:
            raise Refusal('RT-ARGS: release needs --hold-id')
        if run.dry_run:
            print('would release %s' % args.hold_id, file=sys.stderr)
            return record.EXIT_DONE
        with run.unit('release_%s' % st, action='release') as u:
            T.release(out, args.hold_id, args.reason)
            u.add_output(T.holds_path(out), 'retention_holds')
            u.note('released %s: %s' % (args.hold_id, args.reason))
        return run.exit_code()

    # ---- delete: Gregg's go for this manifest, then every item re-checked before anything is deleted
    if not args.manifest:
        raise Refusal('RT-MANIFEST: delete needs --manifest <the dry-run manifest Gregg approved>')
    man = os.path.abspath(args.manifest)
    if os.path.realpath(out) != os.path.realpath(c.ROOT):
        raise Refusal('RT-ROOT: a delete reads the production records and holds: --out must be retention.ROOT (%s), not %s'
                      % (c.ROOT, out))
    lid = _delete_ruling(args, CFG)
    if not lid:
        raise Refusal('RT-RULING: a delete of raw data needs Gregg\'s go: --ruling-for %s=<ledger ID of his ruling naming '
                      'this manifest>' % c.DELETE_RULING_KEY)
    side, items = P.read_manifest(man)
    rows = ledger.rows(lid, CFG.paths.FIX_LEDGER)
    if not lid.startswith(c.RULING_PREFIX):
        raise Refusal('RT-RULING: %s is not one of Gregg\'s rulings (fix_id %s*)' % (lid, c.RULING_PREFIX))
    why = [P.ruling_problem(r, side) for r in rows] or ['no row']
    if all(why):
        raise Refusal('RT-RULING: the ledger row %s is not Gregg\'s go for this manifest (%s): it needs authorized_by %s, a '
                      'status %s... and the manifest\'s sha256 %s (at least %d hex digits) in its text' % (
                          lid, '; '.join(why), c.RULING_AUTHORIZED_BY, c.RULING_STATUS_PREFIX,
                          side['csv_sha256'][:c.RULING_SHA_MIN_HEX], c.RULING_SHA_MIN_HEX))
    side, items = P.preflight(man, c.RAW_ROOT, out)
    if run.dry_run:
        print('delete (dry run): the manifest holds - %d items, %d files, %.3f TB would be deleted under ruling %s' % (
            side['n_items'], side['n_files'], side['bytes'] / 1e12, lid), file=sys.stderr)
        return record.EXIT_DONE
    applied = os.path.join(out, c.MANIFEST_DIR, 'APPLIED_RETENTION_%s.csv' % st)
    n_all = b_all = n_items = 0
    for key in sorted(items):
        e = items[key]
        with run.unit('delete_%s' % key.replace('/', '_'), action='delete', ruling=lid, manifest=man) as u:
            u.add_input(man)
            n, b, why = P.delete_item(e, applied)
            n_all, b_all = n_all + n, b_all + b
            u.add_output(applied, 'retention_applied', item=key, n_paths=len(e['now']))
            u.note('%d files, %d bytes deleted under ruling %s%s' % (n, b, lid, '; ' + why if why else ''))
        if u.status == 'done':
            n_items += 1
        else:                            # anything unexpected (a path changed after the preflight) stops the delete
            print('retention delete stopped at %s: %s %s' % (key, u.status, u.reason), file=sys.stderr)
            break
    print('retention delete: %d of %d items, %d files, %.3f TB deleted under ruling %s; applied manifest %s' % (
        n_items, len(items), n_all, b_all / 1e12, lid, applied), file=sys.stderr)
    return run.exit_code()
