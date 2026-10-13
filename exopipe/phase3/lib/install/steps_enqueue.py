"""exopipe3 install --stages enqueue,cleanup (unit <cell>.enqueue): hand an installed deep cell to the re-search and emit
the per-hour items to re-run against its rebuilt master.

Ported from membership_rebuild_20261007/install_cell.sh:27-44 (shell, no in-place callee):
  - hold rule (install_cell.sh:27): install_enqueue.HOLD_LIST (None: the twopass hold list was retired, ruling 2122);
  - retire the cell's search markers, policy-skips and parks with a UTC stamp (:32-35), so the fleet does not skip the
    cell as already searched against the old master;
  - assign both Stokes to the node of the FIRST line of assign_<S>.csv (:36);
  - the flow-holder map row <cell>,<FLOW_HOLDER>,<FLOW_WORK_FMT> (:40; host from the site key FLOW_HOLDER);
  - append the cell to ready.txt (:41);
  - remove this round's node-local rebuild scratch (:44) - only node-local paths of this host, only after a passing
    verify record, only when paths.PRODUCTION_WRITES and no --out (inputs are never written under --out).
With --out the live state (install_enqueue.SEARCH_STATE_ROOT, read every loop by p3s5_search_fleet.sh) is NOT written:
the step writes a mirror <out>/p3s5_state seeded once from the live files (SEEDED.csv lists every seeded file).

NEW (DESIGN 12.2, the intermittent mode): the per-hour items <LST>_<date> of every night of the cell to re-run through
the per-hour chain against the rebuilt master, as an item file (`--items @FILE`, `--epochs @FILE`): one per cell
(<records>/<cell>.rerun_epochs.txt, class rerun_epochs) and one combined, de-duplicated file per invocation
(<records>/rerun_epochs/<run id>.txt).
"""
import csv
import os
import shutil
import time

from exopipe.phase3.core.record import Refusal, NoInput
from exopipe.phase3.lib.install import steps_common as SC

RERUN_MODES = ('searchable', 'all')


def _cfg():
    return SC.cfg()


# ---------------------------------------------------------------------------------------------------------------------
# per-hour re-run list
# ---------------------------------------------------------------------------------------------------------------------

def rerun_nights(cdir, lst, scope, mode):
    """Dates of the cell's nights to re-run per install_enqueue.RERUN_NIGHTS (see config). Excluded nights
    (core.exclusions) are always left out. 'searchable' needs the night-image and epoch-gate records (the search refuses a
    cell without them), else Refusal."""
    from exopipe.phase3.core import exclusions
    from exopipe.phase3.lib.masters import coadd as Cfp
    if mode not in RERUN_MODES:
        raise Refusal(f'install_enqueue.RERUN_NIGHTS {mode!r}: one of {RERUN_MODES}')
    fd, sd = os.path.join(cdir, 'coadd_final'), os.path.join(cdir, 'subtracted')
    with open(os.path.join(fd, 'weights.csv')) as f:
        w = {r['date'] for r in csv.DictReader(f)}
    res = set()
    if os.path.isdir(sd):
        for fn in os.listdir(sd):
            m = Cfp._EPOCH_RE.match(fn)
            if m and m['kind'] == 'sub' and m['fl'] == scope:
                res.add(m['date'])
    if mode == 'all':
        cand = w | res
    else:
        cand = w & res
        for rec, what in (('night_image_gate.csv', 'night-image'), ('epoch_gate.csv', 'epoch')):
            p = os.path.join(fd, rec)
            if not os.path.exists(p):
                raise Refusal(f'{os.path.basename(cdir)}: {p} missing - the {what} gate record decides which nights the '
                              f'search admits (install_enqueue.RERUN_NIGHTS = searchable)')
            with open(p) as f:
                cand -= {r['date'] for r in csv.DictReader(f) if r.get('verdict') == 'FAIL'}
    ex = exclusions.load()
    return sorted(d for d in cand if not ex.is_excluded(lst, scope, d))


def write_rerun_file(path, per_cell, working_root, mode):
    """Item file: '# key: value' header lines, then one '<LST>_<date>,<cell>[;<cell>...]' line per item (the item grammar
    of core/cells takes the first comma-separated field)."""
    from exopipe.phase3.core import exclusions
    items = {}
    for cell, dates in per_cell.items():
        lst = cell.split('_', 1)[0]
        for d in dates:
            items.setdefault(f'{lst}_{d}', []).append(cell)
    ex = exclusions.load().fingerprint()
    head = ['# exopipe3 install: per-hour items <LST>_<date> to re-run against the rebuilt masters (DESIGN 12.2)',
            f'# cells: {",".join(sorted(per_cell))}', f'# working_root: {working_root}', f'# nights: {mode}',
            f'# data_decisions_sha1: {ex["sha1"]}', f'# written: {SC.utcnow()}', f'# items: {len(items)}']
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f'{path}.tmp.{os.getpid()}'
    with open(tmp, 'w') as f:
        f.write('\n'.join(head) + '\n')
        for it in sorted(items):
            f.write(f'{it},{";".join(sorted(items[it]))}\n')
    os.replace(tmp, path)
    return path


def read_rerun_file(path):
    """(header dict, {item: [cells]})."""
    head, items = {}, {}
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('#'):
                k, sep, v = line[1:].partition(':')
                if sep:
                    head[k.strip()] = v.strip()
            elif line.strip():
                it, _, cs = line.partition(',')
                items[it.strip()] = [c for c in cs.split(';') if c]
    return head, items


def write_combined_rerun(u, cells_, roots, run_obj):
    mode = _cfg().install_enqueue.RERUN_NIGHTS
    per = {}
    for cell in cells_:
        lst, scope = cell.split('_', 1)
        per[cell] = rerun_nights(roots.cdir(cell), lst, scope, mode)
    p = os.path.join(roots.records, 'rerun_epochs', f'{run_obj.run_id}.txt')
    write_rerun_file(p, per, roots.working, mode)
    u.add_output(p, 'rerun_epochs')
    return p


# ---------------------------------------------------------------------------------------------------------------------
# the search state (live or mirror)
# ---------------------------------------------------------------------------------------------------------------------

def _lines(path):
    try:
        with open(path) as f:
            return [l.rstrip('\n') for l in f]
    except FileNotFoundError:
        return []


def seed_mirror(live, mirror, cell, stokes):
    """Copy the live list files once, and this cell's markers / parks once, into the mirror (SEEDED.csv records them)."""
    C = _cfg().install_enqueue
    os.makedirs(mirror, exist_ok=True)
    log = os.path.join(mirror, 'SEEDED.csv')
    seeded = list(csv.DictReader(open(log))) if os.path.exists(log) else []
    done_paths = {r['dst'] for r in seeded}
    done_cells = {r['cell'] for r in seeded if r['cell']}
    new = []
    names = [C.READY_LIST, C.FLOW_HOLDER_MAP] + [C.ASSIGN_FMT.format(stokes=s) for s in stokes]
    for n in names:
        src, dst = os.path.join(live, n), os.path.join(mirror, n)
        if dst not in done_paths and not os.path.exists(dst) and os.path.exists(src):
            shutil.copy2(src, dst)
            new.append(dict(utc=SC.utcnow(), cell='', src=src, dst=dst, md5=SC.md5(dst)))
    if cell not in done_cells:
        for s in stokes:
            for sub, nm in ((C.MARKERS_FMT, cell), (C.MARKERS_FMT, f'{cell}.policy-skip'), (C.PARKED_FMT, cell)):
                src = os.path.join(live, sub.format(stokes=s), nm)
                dst = os.path.join(mirror, sub.format(stokes=s), nm)
                if os.path.exists(src) and not os.path.exists(dst):
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    (shutil.copytree if os.path.isdir(src) else shutil.copy2)(src, dst)
                    new.append(dict(utc=SC.utcnow(), cell=cell, src=src, dst=dst, md5=''))
        new.append(dict(utc=SC.utcnow(), cell=cell, src='', dst='', md5=''))     # the cell is seeded (even with nothing)
    if new:
        exists = os.path.exists(log)
        with open(log, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['utc', 'cell', 'src', 'dst', 'md5'])
            if not exists:
                w.writeheader()
            w.writerows(new)
    return new


def plan_edits(state, cell, holder, stamp):
    """The edits install_cell.sh:32-41 makes, as a list of (kind, a, b), computed read-only."""
    C = _cfg().install_enqueue
    edits = []
    for s in C.STOKES:
        ret = os.path.join(state, C.RETIRED_FMT.format(stokes=s))
        for nm in (cell, f'{cell}.policy-skip'):
            f = os.path.join(state, C.MARKERS_FMT.format(stokes=s), nm)
            if os.path.exists(f):
                edits.append(('retire', f, os.path.join(ret, f'{nm}.marker.{stamp}')))
        pk = os.path.join(state, C.PARKED_FMT.format(stokes=s), cell)
        if os.path.exists(pk):
            edits.append(('retire', pk, os.path.join(ret, f'{cell}.park.{stamp}')))
        af = os.path.join(state, C.ASSIGN_FMT.format(stokes=s))
        lines = _lines(af)
        if not any(l.startswith(f'{cell},') for l in lines):
            if C.ASSIGN_NODE_RULE != 'first-line':
                raise Refusal(f'install_enqueue.ASSIGN_NODE_RULE {C.ASSIGN_NODE_RULE!r}: only "first-line" is implemented')
            if not lines or ',' not in lines[0]:
                raise Refusal(f'{af}: no first line to take the node from (install_cell.sh:36)')
            edits.append(('append', af, f'{cell},{lines[0].split(",")[1]}'))
    mp = os.path.join(state, C.FLOW_HOLDER_MAP)
    if not any(l.startswith(f'{cell},') for l in _lines(mp)):
        edits.append(('append', mp, f'{cell},{holder},{C.FLOW_WORK_FMT.format(cell=cell)}'))
    rd = os.path.join(state, C.READY_LIST)
    if cell not in _lines(rd):
        edits.append(('append', rd, cell))
    return edits


def apply_edits(edits):
    for kind, a, b in edits:
        if kind == 'retire':
            if os.path.exists(b):
                raise Refusal(f'retired destination exists: {b}')
            os.makedirs(os.path.dirname(b), exist_ok=True)
            os.rename(a, b)
        else:
            with open(a, 'a') as f:
                f.write(b + '\n')


# ---------------------------------------------------------------------------------------------------------------------
# the unit
# ---------------------------------------------------------------------------------------------------------------------

def enqueue_cell(u, cell, roots, run_obj, stages, echo, verified_now=False):
    """Returns the per-cell re-run item file (enqueue) or None. ``verified_now``: this invocation's unit of the cell ran
    the verify stage and passed (its record is written before this unit starts)."""
    from exopipe.phase3.core import cells
    C, CE = _cfg().install, _cfg().install_enqueue
    lst, scope = cells.parse_cell(cell)
    dry = run_obj.dry_run
    gates = SC.Gates(u, run_obj, collect=dry, echo=echo)
    cdir = roots.cdir(cell)
    master = os.path.join(cdir, 'coadd_final', f'{C.EPOCH_MASTER_STEM["I"]}_{scope}.fits')
    gates.check('CELL', os.path.isfile(master), f'installed I master {master} ' + ('present' if os.path.isfile(master)
                                                                                  else 'ABSENT'))
    if CE.HOLD_LIST:
        held = cell in [l.split(',')[0].strip() for l in _lines(CE.HOLD_LIST)]
        gates.check('Q', not held, f'{cell} {"HELD" if held else "not held"} by {CE.HOLD_LIST}')
    else:
        gates.check('Q', True, 'install_enqueue.HOLD_LIST None (twopass hold retired, ruling 2122); queued without a '
                               'parent-catalogue precondition (B-30 default (c), install_cell.sh:26)')
    # V: a cell is queued only after its installed tree passed the membership + product verify (install_cell.sh:23-24 stops
    # the chain before the enqueue on a failed verify) - a done install/verify record of this out root with install.V passed
    ok, name = _verify_passed(roots, cell)
    if dry and not ok and verified_now:
        gates.check('V', True, f'verify runs in this invocation before the enqueue (dry run: not run)')
    else:
        gates.check('V', ok, (f'record {name} (the newest install/recover/verify record) holds a passing install.V'
                              if ok else
                              f'the newest install/recover/verify record of {cell} under {roots.records} is not done '
                              f'with a passing install.V (run --stages verify first)'))
    gates.finish()
    out_path = None
    if 'enqueue' in stages:
        holder = getattr(_cfg().site, 'FLOW_HOLDER', None)
        if not holder:
            raise Refusal('the site file names no FLOW_HOLDER (the host whose /fast holds the V-search flows, today '
                          'lwacalim07, install_cell.sh:40): add FLOW_HOLDER = "<host>" to $EXOPIPE3_SITE')
        live = CE.SEARCH_STATE_ROOT
        state = roots.state
        stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
        # the edits are planned against the live state when the mirror is not seeded yet (what seeding will copy)
        basis = state if (not roots.rerooted or os.path.exists(os.path.join(state, 'SEEDED.csv'))) else live
        edits = plan_edits(basis, cell, holder, stamp)
        if basis != state:
            edits = [(k, a.replace(live, state, 1), b.replace(live, state, 1) if k == 'retire' else b)
                     for k, a, b in edits]
        echo(f'  enqueue {cell} into {state}' + (' (mirror of ' + live + ')' if roots.rerooted else ' (LIVE search state)'))
        for k, a, b in edits:
            echo(f'    {k:7s} {a} {"->" if k == "retire" else "+="} {b}')
        mode = CE.RERUN_NIGHTS
        nights = rerun_nights(cdir, lst, scope, mode)
        echo(f'  per-hour items to re-run ({mode}): {len(nights)}' + (f' ({lst}_{nights[0]} .. {lst}_{nights[-1]})'
                                                                      if nights else ''))
        if not dry:
            if roots.rerooted:
                seeded = seed_mirror(live, state, cell, CE.STOKES)
                if seeded:
                    u.note(f'mirror {state}: seeded {len([s for s in seeded if s["dst"]])} file(s) from {live}')
                edits = plan_edits(state, cell, holder, stamp)
            apply_edits(edits)
            from exopipe.phase3.lib.masters import epoch_gate as EG
            req = dict(cell=cell, utc=SC.utcnow(), state_root=state, live=not roots.rerooted, stokes=list(CE.STOKES),
                       holder=holder, edits=[list(e) for e in edits],
                       retired=[[a, b] for k, a, b in edits if k == 'retire'],
                       i_master=master, i_master_md5=EG.master_md5(master),
                       install_record=os.path.join(roots.records, f'{cell}.json'))
            rp = os.path.join(roots.records, f'{cell}.request.json')
            SC.keep_previous(rp)
            from exopipe.phase3.core.record import atomic_write_json
            atomic_write_json(rp, req)
            u.add_input(master)
            u.add_input(os.path.join(cdir, 'coadd_final', 'weights.csv'))
            u.set_admitted(SC.admitted_in_master(os.path.join(cdir, 'coadd_final', 'weights.csv'), lst, scope))
            u.add_output(rp, 'research_request')
            out_path = write_rerun_file(os.path.join(roots.records, f'{cell}.rerun_epochs.txt'), {cell: nights},
                                        roots.working, mode)
            u.add_output(out_path, 'rerun_epochs')
    if 'cleanup' in stages:
        cleanup_cell(u, cell, roots, run_obj, echo, enqueue_too='enqueue' in stages)
    return out_path


def _verify_passed(roots, cell):
    """(True, record) when the NEWEST record of this out root that installed, recovered or verified the cell (<cell>,
    <cell>.recover, <cell>.verify) is done and holds a passing verify gate install.V - production runs the verify right
    before the enqueue every time (install_cell.sh:19-24, also for kind 'installed'), so a verify older than a later
    install attempt (refused, failed mid-way) or a recover never vouches for the tree as it is now."""
    from exopipe.phase3.core import record
    recs = [(n, record.load_record(roots.out, 'install', n)) for n in (cell, f'{cell}.recover', f'{cell}.verify')]
    recs = [(n, r) for n, r in recs if r]
    if not recs:
        return False, None
    name, rec = max(recs, key=lambda x: (x[1].get('finished') or '', x[1].get('started') or '', x[0].endswith('.verify')))
    if rec.get('status') == 'done' and any(t.get('name') == 'install.V' and t.get('passed') for t in rec.get('tests') or []):
        return True, name
    return False, None


def cleanup_cell(u, cell, roots, run_obj, echo, enqueue_too=False):
    C, CE = _cfg().install, _cfg().install_enqueue
    targets = [C.REBUILD_ROOT_FMT.format(cell=cell).rstrip('/'), C.PROMOTED_GEN_FMT.format(cell=cell).rstrip('/')]
    existing = [t for t in targets if os.path.exists(t)]
    why_not = None
    if not CE.REMOVE_ROUND_SCRATCH:
        why_not = 'install_enqueue.REMOVE_ROUND_SCRATCH is False'
    elif roots.rerooted:
        why_not = 'under --out inputs are read in place and never written'
    elif not _cfg().paths.PRODUCTION_WRITES:
        why_not = 'paths.PRODUCTION_WRITES is False (refactor window)'
    echo(f'  cleanup {cell}: ' + (f'would remove {existing}' if existing else 'no rebuild scratch present')
         + (f' - NOT performed: {why_not}' if why_not and existing else ''))
    if why_not or not existing:
        u.note(f'cleanup: {"nothing to remove" if not existing else "not performed: " + why_not}; targets {targets}')
        if not enqueue_too:
            raise NoInput(f'cleanup of {cell}: ' + ('nothing to remove' if not existing else f'not performed ({why_not}); '
                                                    f'would remove {existing}'))
        return
    ok, name = _verify_passed(roots, cell)
    if not ok:
        raise Refusal(f'V: cleanup of {cell} refused: the newest install/recover/verify record under {roots.records} is '
                      f'not done with a passing gate install.V (install_cell.sh removes the scratch only after the verify '
                      f'passed)')
    notlocal = [t for t in existing if not SC.local_to_host(t)]
    if notlocal:
        raise Refusal(f'H: cleanup of {cell} refused: {notlocal} not node-local to {SC.host()} (site NODE_LOCAL_ROOTS)')
    if run_obj.dry_run:
        return
    for t in existing:
        shutil.rmtree(t)
    rec = os.path.join(roots.records, f'{cell}.cleanup.json')
    from exopipe.phase3.core.record import atomic_write_json
    atomic_write_json(rec, dict(cell=cell, utc=SC.utcnow(), host=SC.host(), removed=existing, verify_record=name))
    u.add_output(rec, 'cleanup_record')
