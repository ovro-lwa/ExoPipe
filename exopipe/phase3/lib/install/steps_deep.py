"""exopipe3 install --tier deep: stages plan, install, verify (unit <cell>), enqueue/cleanup (unit <cell>.enqueue, in
steps_enqueue.py) and the exclusive modes recover, screen-unapplied, epochs-repair.

The install transaction is production's phase3_install.install called IN PLACE (exopipe/phase3/lib/install/install.py,
"PI"; DESIGN 11.1). This module adds what production checked too late or not at all, before anything is staged:
  - G-DEST: every working-quarantine, archive-quarantine and screen-quarantine destination of the cell is checked BEFORE
    staging, and the quarantine round is chosen by the install (--round auto: '<UTC date><suffix>', the first round whose
    every destination is free; deep requirement (b), ledger 2283). Production checked one destination at a time inside
    the move loops and left .install_staging behind on a collision (12h Red, 2026-10-12 23:10).
  - G1-EXACT: the generation's content record must name the generation on a line of its own (PI: substring).
  - RB: the rebuild record names its own root; its verdict line (T1-T8 passed, or Gregg's waiver) is recorded.
  - RB-FILES / CELL: the files PI reads or copies after staging exist (else PI would stop mid-way or leave the staging).
  - G0-GEN: the exclusion-conformance gate on the generation itself (read-only; PI repeats it on the staging).
  - the plan (every destination, existed = no) is written to the attempt manifest before staging; a refusal before the
    first move removes the staging (a refusal leaves nothing behind).
"""
import contextlib
import fcntl
import glob
import json
import os
import shutil
import time

from exopipe.phase3.core.record import Refusal, NoInput
from exopipe.phase3.lib.install import steps_common as SC

DEEP_STAGES = ('plan', 'install', 'verify', 'enqueue', 'cleanup')
MODES = ('recover', 'screen-unapplied', 'epochs-repair')

_G1B = {}


def _cfg():
    return SC.cfg()


def snapshot_link_scan(archive_roots):
    """G-1b once per invocation (spec section 11, speed-up 2): the links under install.SNAPSHOT_MASTERS_LINK_TREE that
    resolve into <archive>/Coadd/ of each archive root. 912 links, ~40 s on Lustre (2026-10-13)."""
    key = tuple(dict.fromkeys(archive_roots))
    if key not in _G1B:
        tree = _cfg().install.SNAPSHOT_MASTERS_LINK_TREE
        links = glob.glob(f'{tree}/**/*.fits', recursive=True)
        real = {l: os.path.realpath(l) for l in links}
        live = {a: sorted(l for l, r in real.items() if r.startswith(a.rstrip('/') + '/Coadd/')) for a in key}
        _G1B[key] = (len(links), live)
    return _G1B[key]


# ---------------------------------------------------------------------------------------------------------------------
# inputs of one cell
# ---------------------------------------------------------------------------------------------------------------------

def resolve(cell, args):
    """(kind, gen, rebuild_root) of a cell from --kind/--gen/--rebuild-root/--not-rebuilt and config templates."""
    C = _cfg().install
    kind = args.kind
    if kind not in ('mdewarp', 'rebuild'):
        raise Refusal('--kind mdewarp|rebuild is required for this stage (the caller decides it from the dewarp result; '
                      'there is no guessing)')
    gen = (args.gen or (C.MDEWARP_GEN_FMT if kind == 'mdewarp' else C.PROMOTED_GEN_FMT)).format(cell=cell).rstrip('/')
    if os.path.basename(gen) == 'gen_shflux' or C.SHFLUX_TOKEN in os.path.basename(gen):
        raise Refusal(f'{cell}: --gen {gen} is a gen_shflux generation: during the refactor window the in-place '
                      f'phase3_install cannot take it (its G1 reads <gen>/../INSTALL_CONTENT_OK and canon() strips only '
                      f'"_mdewarp"); install the promoted generation (promote_rebuild.py) - accepted after absorption')
    if args.not_rebuilt and args.rebuild_root:
        raise Refusal('--not-rebuilt and --rebuild-root contradict each other')
    if kind == 'rebuild':
        if args.not_rebuilt:
            raise Refusal('--kind rebuild installs a rebuilt cell: --not-rebuilt does not apply')
        rr = (args.rebuild_root or C.REBUILD_ROOT_FMT).format(cell=cell).rstrip('/')
    else:
        rr = None if args.not_rebuilt else (args.rebuild_root or C.REBUILD_ROOT_FMT).format(cell=cell).rstrip('/')
    return kind, gen, rr


class CellPlan:
    """Everything the install of one cell will do, computed read-only (the plan stage)."""

    def __init__(self, cell, args, roots):
        from exopipe.phase3.core import cells
        self.cell = cell
        self.lst, self.scope = cells.parse_cell(cell)
        self.kind, self.gen, self.rebuild_root = resolve(cell, args)
        self.roots = roots
        self.cdir = roots.cdir(cell)
        self.omit_vclean = bool(args.omit_vclean)
        self.round = None
        self.rounds_tried = []
        self.new, self.cur, self.tgt, self.orphans, self.omitted = [], [], {}, [], []
        self.need = self.free = self.floor = None
        self.dests = []
        self.rb_text = ''
        self.smeta = None
        self.content_record = os.path.join(os.path.dirname(self.gen), _cfg().install.CONTENT_RECORD)
        self.screen_json = os.path.join(os.path.dirname(self.gen), _cfg().install.SCREEN_RECORD_JSON)

    @property
    def wfin(self):
        """The coadd_final/ whose weights.csv and night-image record PI reads (phase3_install.py:306)."""
        return os.path.join(self.rebuild_root or self.cdir, 'coadd_final')

    @property
    def staging(self):
        return os.path.join(self.cdir, _cfg().install.STAGING_DIRNAME)

    # ---- file sets (PI's own functions, so the plan is the set PI will use)
    def compute_sets(self, PI):
        C = _cfg().install
        new = PI._fits_under(self.gen)
        if self.omit_vclean:
            self.omitted = [r for r in new if C.OMIT_VCLEAN_PATTERN in os.path.basename(r)]
            new = [r for r in new if r not in self.omitted]
        self.new = new
        self.cur = PI._fits_under(self.cdir) if os.path.isdir(self.cdir) else []
        tgt, dup = {}, []
        for r in new:
            c_ = os.path.join(os.path.dirname(r), PI.canon(os.path.basename(r)))
            if c_ in tgt.values():
                dup.append(c_)
            tgt[r] = c_
        self.tgt = tgt
        self.orphans = sorted(set(self.cur) - set(tgt.values()))
        return dup

    def destinations(self, rnd):
        """Every quarantine destination the install would write in round ``rnd``: (kind, source, destination). The same
        globs PI uses at the move sites (phase3_install.py:344-364, :384-385, :392-393, :152-155)."""
        C = _cfg().install
        wq, aq, arch = self.roots.wq(rnd), self.roots.aq(rnd), self.roots.archive
        cd, cell = self.cdir, self.cell
        out = [('working-quarantine', f'{cd}/{r}', f'{wq}/{cell}/{r}') for r in sorted(set(self.cur))]
        if self.rebuild_root:
            for f in C.CELL_FILES:
                if os.path.exists(f'{cd}/coadd_final/{f}'):
                    out.append(('working-quarantine', f'{cd}/coadd_final/{f}', f'{wq}/{cell}/coadd_final/{f}'))
            for f in C.CELL_TOP + CARRIED_RECORDS:        # B-03 (a), D-ADMIT: the branch's records move with the cell's
                if os.path.exists(f'{cd}/{f}'):
                    out.append(('working-quarantine', f'{cd}/{f}', f'{wq}/{cell}/{f}'))
            for d in C.CELL_DIRS:
                if os.path.isdir(f'{cd}/{d}'):
                    for p in sorted(glob.glob(f'{cd}/{d}/**/*', recursive=True) + glob.glob(f'{cd}/{d}/.*')):
                        if os.path.isfile(p):
                            out.append(('working-quarantine', p, f'{wq}/{cell}/{os.path.relpath(p, cd)}'))
        eg = f'{cd}/coadd_final/epoch_gate.csv'
        if os.path.exists(eg):
            out.append(('working-quarantine', eg, f'{wq}/{cell}/coadd_final/epoch_gate.csv'))
        mdir = self.roots.masters_dir(self.lst, self.scope)
        old = [p for p in sorted(glob.glob(f'{mdir}/*')) if os.path.isfile(p)]
        old += [p for p in sorted(glob.glob(self.roots.night_glob(self.lst, self.scope)))
                if not any(t in os.path.basename(p) for t in C.ARCHIVE_SKIP_TOKENS)]
        out += [('archive-quarantine', p, f'{aq}/{os.path.relpath(p, arch)}') for p in old]
        if self.kind == 'mdewarp':
            sp = self.roots.screen_path(self.lst, self.scope)
            if os.path.exists(sp):
                out.append(('screen-quarantine', sp, f'{aq}/{os.path.relpath(sp, arch)}'))
        return out

    def choose_round(self, requested):
        """(round, taken destinations of the requested round or []) - auto: the first free round of today (UTC)."""
        if requested and requested != 'auto':
            if not SC.valid_round(requested):
                raise Refusal(f'--round {requested!r}: letters, digits, "_" and "-" only')
            d = self.destinations(requested)
            self.rounds_tried = [(requested, sum(os.path.exists(x[2]) for x in d))]
            return requested, d, [x for x in d if os.path.exists(x[2])]
        day = time.strftime('%Y%m%d', time.gmtime())
        for sfx in _cfg().install.ROUND_SUFFIXES:
            r = day + sfx
            d = self.destinations(r)
            taken = [x for x in d if os.path.exists(x[2])]
            self.rounds_tried.append((r, len(taken)))
            if not taken:
                return r, d, []
        return None, [], []

    def rows(self, gate_summary=''):
        """The plan rows of the attempt manifest (written before staging)."""
        t = SC.utcnow()
        C = _cfg().install
        meta = dict(kind=self.kind, gen=self.gen, rebuild_root=self.rebuild_root, working_cell=self.cdir,
                    archive_root=self.roots.archive, archive_run=self.roots.run, record_csv=self.roots.record_csv,
                    omit_vclean=self.omit_vclean, staging_bytes_needed=self.need, free_bytes=self.free,
                    floor_bytes=self.floor, host=SC.host())
        rows = [dict(time=t, cell=self.cell, step='plan-meta', detail=json.dumps(meta, sort_keys=True)),
                dict(time=t, cell=self.cell, step='plan-round', path=self.roots.wq(self.round),
                     dest=self.roots.aq(self.round),
                     detail=f'round {self.round} ({"auto" if C.ROUND == "auto" else "named"}); tried '
                            + ', '.join(f'{r}:{n} taken' for r, n in self.rounds_tried)),
                dict(time=t, cell=self.cell, step='plan-gates', detail=gate_summary[:2000])]
        for r, c_ in sorted(self.tgt.items()):
            rows.append(dict(time=t, cell=self.cell, step='plan-map', path=f'{self.gen}/{r}', dest=f'{self.cdir}/{c_}'))
        for r in self.omitted:
            rows.append(dict(time=t, cell=self.cell, step='plan-omit', path=f'{self.gen}/{r}', detail='--omit-vclean'))
        for r in self.orphans:
            rows.append(dict(time=t, cell=self.cell, step='plan-orphan', path=f'{self.cdir}/{r}',
                             detail='current file without a new counterpart: quarantined, not replaced'))
        for kind, src, dst in self.dests:
            rows.append(dict(time=t, cell=self.cell, step='plan-dest', path=src, dest=dst, detail=f'{kind}; existed: no'))
        return rows


# ---------------------------------------------------------------------------------------------------------------------
# the plan stage
# ---------------------------------------------------------------------------------------------------------------------

def plan_cell(u, plan, run_obj, gates, call_pi_gates, round_req, echo):
    """Evaluate the read-only gates of one cell (spec section 5, order of evaluation). ``call_pi_gates``: run PI._gates
    now (dry run / plan only); in an install the same gates run inside PI.install just before staging."""
    C = _cfg().install
    roots = plan.roots
    PI = SC.pi_module()
    cell = plan.cell
    # H: the working tree is local to this host (production) / the out root is writable here (--out)
    if roots.rerooted:
        ne = SC.nearest_existing(roots.out)
        gates.check('H', os.access(ne, os.W_OK), f'--out {roots.out} writable on {SC.host()} (nearest existing {ne})')
    else:
        gates.check('H', SC.local_to_host(roots.working),
                    f'working root {roots.working} local to {SC.host()} per site NODE_LOCAL_ROOTS '
                    f'{(_cfg().site.NODE_LOCAL_ROOTS or {}).get(SC.host(), [])}')
    gates.check('R', bool(roots.run), f'archive run paths.ARCHIVE_RUN = {roots.run!r} (contract step 51)')
    # G-1: freeze marker (PI repeats it, unconditionally)
    fm = C.FREEZE_MARKER
    gates.check('G-1', os.path.exists(fm) or not C.REQUIRE_FREEZE_MARKER,
                f'freeze marker {fm} ' + ('present' if os.path.exists(fm) else 'ABSENT - the snapshot / sub-LST freeze '
                                          'is not confirmed'))
    if os.path.exists(fm):
        u.add_input(fm)
    # PI's own read-only gates (G-1, G-1b, G1 substring, CUB, RB, G-MEM, G-PROD, G1b), in place
    g1b_by_pi = False
    if call_pi_gates:
        try:
            with SC.pi_context(PI, roots), SC.captured_stdout() as buf:
                plan.smeta = PI._gates(cell, plan.gen, plan.kind, plan.rebuild_root)
            u.check('install.PI-GATES', True, detail='phase3_install._gates passed: G-1, G-1b, G1, CUB, RB, G-MEM, G-PROD, '
                                                     'G1b' + (f' ({buf.getvalue().strip()[:200]})' if buf.getvalue().strip() else ''))
            if echo:
                echo('  gate PI-GATES  PASS  phase3_install._gates: G-1, G-1b, G1, CUB, RB, G-MEM, G-PROD, G1b')
            g1b_by_pi = True
        except RuntimeError as e:
            g = SC.pi_gate_of(str(e))
            g1b_by_pi = g not in ('R', 'G-1', 'G-1b', 'PI')
            gates.check(g, False, f'phase3_install._gates: {e}')
    else:
        u.note('G1 (substring), CUB, RB, G-MEM, G-PROD, G1b run inside phase3_install.install just before staging')
    # G-1b: no snapshot_masters link resolves into the archive masters this install moves - the production archive always,
    # and the re-rooted archive under --out (where PI's own scan looks); once per invocation; an archive root PI's _gates
    # just scanned is not scanned twice
    prod = _cfg().paths.PHASE3_ARCHIVE_ROOT
    archs = [a for a in dict.fromkeys([prod, roots.archive]) if not (g1b_by_pi and a == roots.archive)]
    if not call_pi_gates and not roots.rerooted:
        # an install on the production roots: phase3_install._gates scans the production archive itself, first thing in
        # install(), before anything is staged - one scan, not two (spec section 11, speed-up 2)
        archs = []
        u.note('G-1b (production archive) runs inside phase3_install.install before staging')
    if archs:
        nlinks, live = snapshot_link_scan(archs)
        bad = {a: v for a, v in live.items() if v}
        gates.check('G-1b', not bad, f'{nlinks} links under {C.SNAPSHOT_MASTERS_LINK_TREE}; resolving into '
                    f'<archive>/Coadd/: ' + ', '.join(f'{a}: {len(v)}' for a, v in live.items())
                    + (f' (e.g. {next(iter(bad.values()))[0]})' if bad else '')
                    + (f'; {roots.archive}: 0 (phase3_install._gates)' if g1b_by_pi else ''),
                    value=sum(len(v) for v in live.values()))
    elif g1b_by_pi:
        gates.check('G-1b', True, f'{roots.archive}: no snapshot_masters link resolves into <archive>/Coadd/ '
                                  f'(scanned by phase3_install._gates)')
    # G1-EXACT: the generation's content record names it on a line of its own
    okf = plan.content_record
    lines = open(okf).read().splitlines() if os.path.exists(okf) else []
    gates.check('G1-EXACT', plan.gen in [l.strip().rstrip('/') for l in lines],
                f'{okf} ' + ('names ' + plan.gen + ' on a line of its own' if plan.gen in [l.strip().rstrip('/') for l in lines]
                             else ('ABSENT' if not lines else f'does not name {plan.gen} on a line of its own')))
    if os.path.exists(okf):
        u.add_input(okf)
        if len(lines) > 1:
            u.note(f'content record line 2: {lines[1][:300]}')
    # RB: a rebuilt cell's record (T1-T8 of check_rebuild.py, or Gregg's waiver) names its own root
    if plan.rebuild_root:
        rbf = os.path.join(plan.rebuild_root, C.REBUILD_RECORD)
        rl = open(rbf).read().splitlines() if os.path.exists(rbf) else []
        plan.rb_text = ' | '.join(rl[1:])[:500]
        ok = bool(rl) and rl[0].strip().rstrip('/') == plan.rebuild_root
        gates.check('RB', ok, (f'{rbf}: names its root; verdict "{plan.rb_text}"' if ok else
                               (f'rebuild root {plan.rebuild_root} absent (pass --not-rebuilt for an mdewarp generation of a '
                                f'cell that was not rebuilt)' if not os.path.isdir(plan.rebuild_root) else
                                (f'{rbf} absent: T1-T8 not passed' if not rl else f'{rbf} line 1 {rl[0]!r} does not name '
                                 f'{plan.rebuild_root}'))))
        if rl:
            u.add_input(rbf)
            u.meta['rebuild_verdict'] = plan.rb_text
        need = [os.path.join(plan.rebuild_root, 'coadd_final', f) for f in C.CELL_FILES]
        miss = [p for p in need if not os.path.isfile(p)]
        gates.check('RB-FILES', not miss, (f'rebuild root holds {", ".join(C.CELL_FILES)}' if not miss else
                                           f'rebuild root lacks {miss} (phase3_install copies them after the moves began)'))
        for p in need:
            if os.path.isfile(p):
                u.add_input(p)
    # CELL: the working cell exists with the records PI reads after staging (weights.csv, night_image_gate.csv)
    w_need = [os.path.join(plan.wfin, 'weights.csv'), os.path.join(plan.wfin, 'night_image_gate.csv')]
    w_miss = [p for p in w_need if not os.path.isfile(p)]
    gates.check('CELL', os.path.isdir(plan.cdir) and not w_miss,
                (f'working cell {plan.cdir} present; {plan.wfin} holds weights.csv and night_image_gate.csv'
                 if os.path.isdir(plan.cdir) and not w_miss else
                 (f'working cell {plan.cdir} absent' if not os.path.isdir(plan.cdir) else f'missing {w_miss}')))
    # Z, G2: the generation's file set
    if not os.path.isdir(plan.gen):
        gates.check('Z', False, f'generation {plan.gen} absent')
        return
    dup = plan.compute_sets(PI)
    gates.check('Z', bool(plan.new), f'{len(plan.new)} generation FITS under coadd_final/ and subtracted/'
                + (f' ({len(plan.omitted)} cleaned-V products omitted)' if plan.omitted else ''), value=len(plan.new))
    gates.check('G2', not dup, (f'{len(plan.tgt)} files map to distinct canonical names; {len(plan.cur)} current, '
                                f'{len(plan.orphans)} without a new counterpart (quarantined)') if not dup else
                f'two generation files map to {dup[0]}')
    # G3: space on the working-tree filesystem (PI's arithmetic, phase3_install.py:279-283)
    if os.path.isdir(plan.cdir):
        plan.need = sum(os.path.getsize(f'{plan.gen}/{r}') for r in plan.new) * C.STAGING_SPACE_FACTOR
        plan.floor = float(C.FREE_FLOOR_GB) * 1e9
        plan.free = shutil.disk_usage(plan.cdir).free
        gates.check('G3', plan.free >= plan.need + plan.floor,
                    f'{plan.need / 1e9:.1f} GB needed + {plan.floor / 1e9:.0f} GB floor, {plan.free / 1e9:.1f} GB free',
                    value=round(plan.free / 1e9, 1))
        gates.check('STG', not os.path.exists(plan.staging),
                    f'{plan.staging} ' + ('absent' if not os.path.exists(plan.staging) else
                                          'EXISTS (an earlier install did not finish) - use --stages recover'))
    # G0-GEN: exclusion conformance of the generation itself (PI repeats it on the staging, and b on the installed tree)
    if plan.omit_vclean:
        u.note('G0-GEN skipped (--omit-vclean filters the generation): G0 runs on the staging inside phase3_install')
    else:
        from exopipe.phase3.lib.masters import coadd as Cfp
        try:
            with SC.captured_stdout() as buf:
                Cfp._refuse_excluded_constituents(os.path.join(plan.gen, 'coadd_final'), plan.lst, plan.scope)
            gates.check('G0-GEN', True, buf.getvalue().strip()[:300] or 'no excluded constituent')
        except RuntimeError as e:
            gates.check('G0-GEN', False, str(e))
    # G-DEST + the round
    if os.path.isdir(plan.cdir):
        rnd, dests, taken = plan.choose_round(round_req)
        if rnd is None:
            gates.check('G-DEST', False, 'no free quarantine round today: tried ' +
                        ', '.join(f'{r} ({n} destinations taken)' for r, n in plan.rounds_tried))
            return
        dd = [x[2] for x in dests]
        dupd = sorted({x for x in dd if dd.count(x) > 1})
        if taken:
            first = taken[0]
            word = {'working-quarantine': 'working quarantine', 'archive-quarantine': 'archive quarantine',
                    'screen-quarantine': 'screen quarantine'}[first[0]]
            gates.check('G-DEST', False, f'{word} destination exists: {first[2]} ({len(taken)} of {len(dests)} '
                                         f'destinations of round {rnd} exist)')
            return
        gates.check('G-DEST', not dupd, (f'round {rnd}: {len(dests)} destinations free ('
                                         f'{sum(1 for x in dests if x[0] == "working-quarantine")} working, '
                                         f'{sum(1 for x in dests if x[0] != "working-quarantine")} archive); tried '
                                         + ', '.join(f'{r}:{n}' for r, n in plan.rounds_tried)) if not dupd else
                    f'two moves share a destination: {dupd[0]}')
        plan.round, plan.dests = rnd, dests
        u.meta['round'] = rnd


def print_plan(plan, echo):
    echo(f'  kind {plan.kind}; generation {plan.gen}')
    echo(f'  rebuild root {plan.rebuild_root or "(none: --not-rebuilt)"}; working cell {plan.cdir}')
    echo(f'  generation files {len(plan.new)} (omitted {len(plan.omitted)}); current {len(plan.cur)}; orphans '
         f'{len(plan.orphans)}')
    if plan.need is not None:
        echo(f'  staging {plan.need / 1e9:.2f} GB + floor {plan.floor / 1e9:.0f} GB; free {plan.free / 1e9:.1f} GB')
    if plan.round:
        n_w = sum(1 for x in plan.dests if x[0] == 'working-quarantine')
        echo(f'  round {plan.round}: working quarantine {plan.roots.wq(plan.round)}/{plan.cell} ({n_w} files), archive '
             f'quarantine {plan.roots.aq(plan.round)} ({len(plan.dests) - n_w} files)')
    echo(f'  writes: {plan.cdir}, {plan.roots.masters_dir(plan.lst, plan.scope)}, per-night deep files, '
         f'{plan.roots.record_csv} (append)' + (f', {plan.roots.screen_path(plan.lst, plan.scope)}'
                                                if plan.kind == 'mdewarp' else ''))
    echo('  outputs: install_manifest, archive_cell, installed_cell' + (', screen_table' if plan.kind == 'mdewarp'
                                                                         else ''))


# ---------------------------------------------------------------------------------------------------------------------
# the install stage
# ---------------------------------------------------------------------------------------------------------------------

@contextlib.contextmanager
def lane_lock(records_dir, lanes):
    """One of install.LANES lanes (exclusive flock), waiting while all are held (install_cell.sh:18-20 flock)."""
    d = os.path.join(records_dir, '_locks')
    os.makedirs(d, exist_ok=True)
    while True:
        for k in range(max(1, int(lanes))):
            f = open(os.path.join(d, f'lane{k}.lock'), 'w')
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                f.close()
                continue
            try:
                yield k
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
                f.close()
            return
        time.sleep(5)


@contextlib.contextmanager
def production_lock(roots):
    """install_cell.sh:18-20 (`exec 9> $D/.install.lock; flock 9`, around the phase3_install call): an install on the
    PRODUCTION roots also holds the production launchers' lock (install.PRODUCTION_LOCK), blocking, so it never overlaps
    an install_cell.sh install of the same working tree (chain_cell.sh and dewarp_lst.sh both go through it; /data07 ran
    full under concurrent installs, ledger 1845). Under --out nothing production is written: not taken."""
    p = _cfg().install.PRODUCTION_LOCK
    if roots.rerooted or not p:
        yield None
        return
    f = open(p, 'w')                     # production opens it with '>' too
    try:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield p
    finally:
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()


def manifest_path(roots, cell):
    return os.path.join(roots.records, f'{cell}.manifest.csv')


def install_cell(u, plan, gates, echo):
    """The transaction: PI.install in place, with the plan written first. Returns the action rows of the attempt."""
    PI = SC.pi_module()
    roots, cell = plan.roots, plan.cell
    C = _cfg().install
    mpath = manifest_path(roots, cell)
    prev = SC.keep_previous(mpath)
    if prev:
        u.note(f'earlier manifest kept as {prev}')
    plan_rows = plan.rows(gate_summary='; '.join(f"{t['name']}={'pass' if t['passed'] else 'FAIL'}" for t in u.tests))
    SC.write_manifest(mpath, plan_rows)
    os.makedirs(os.path.dirname(roots.record_csv), exist_ok=True)
    with lane_lock(roots.records, C.LANES) as lane, production_lock(roots) as plock:
        if plock:
            u.note(f'holding the production install lock {plock} (install_cell.sh:18-20)')
        off = SC.log_offset(roots.record_csv)
        with SC.pi_context(PI, roots, plan.round) as settings:
            u.note(f'lane {lane}; in-place phase3_install.install(cell={cell}, gen={plan.gen}, kind={plan.kind}, '
                   f'rebuild_root={plan.rebuild_root}, root={roots.working}, omit_vclean={plan.omit_vclean}) with '
                   f'{json.dumps(settings, sort_keys=True)}')
            buf = None
            try:
                with SC.captured_stdout() as buf:
                    res = PI.install(cell, plan.gen, plan.kind, plan.rebuild_root, root=roots.working,
                                     omit_vclean=plan.omit_vclean)
            except Exception as e:
                text = buf.getvalue() if buf is not None else ''
                if text.strip():
                    u.note('phase3_install output: ' + text.strip()[-3000:])
                moved = [r for r in SC.action_rows(roots.record_csv, off, cell) if r['step'] in SC.MOVE_STEPS]
                PI.rec(cell, 'INSTALL-FAILED', plan.gen, '', '', '', str(e)[:400])     # as phase3_install.main
                rows = SC.action_rows(roots.record_csv, off, cell)
                SC.write_manifest(mpath, plan_rows, rows)
                u.add_output(mpath, 'install_manifest')
                gate = SC.pi_gate_of(str(e))
                if not moved:
                    if os.path.isdir(plan.staging):
                        shutil.rmtree(plan.staging)       # a refusal leaves nothing behind (ledger 2283)
                        u.note(f'removed {plan.staging} (refused before the first move)')
                    if isinstance(e, RuntimeError):
                        u.check(f'install.{gate}', False, detail=str(e)[:2000])
                        raise Refusal(f'{gate}: {e}')
                    raise
                raise RuntimeError(f'{cell}: install STOPPED after {len(moved)} move(s) ({gate}: {e}) - the cell is '
                                   f'half-moved; `exopipe3 install --cells {cell} --kind {plan.kind} --gen {plan.gen} '
                                   f'--round {plan.round} --stages recover` reports its state (R1/R2; rebuilt cells: '
                                   f'manual repair, phase3_install.py:442-443)') from e
            text = buf.getvalue()
            carried = carry_cell_records(PI, plan)        # B-03 (a), D-ADMIT: inside the same round's quarantine settings
            if carried:
                u.note(f'cell records carried into the working cell (B-03 input refusals, D-ADMIT build statistics): {carried}')
            add_carried_outputs(u, carried)
        if text.strip():
            u.note('phase3_install output: ' + text.strip()[-3000:])
        rows = SC.action_rows(roots.record_csv, off, cell)
    SC.write_manifest(mpath, plan_rows, rows)
    n_work, n_aq, n_pub = res
    u.check('install.PI-GATES', True, detail='inside phase3_install.install: G-1, G-1b, G1, CUB, RB, G-MEM, G-PROD, G1b, '
                                             'G2, G3, STG, a0, EG, G4, G0, SET, b, AQ, d' + (', e' if plan.kind == 'mdewarp' else '')
                                             + ' passed')
    u.check('install.INSTALL-OK', True, value=n_work,
            detail=f'INSTALL-OK {cell}: {n_work} working files, {n_aq} archive files quarantined, {n_pub} published')
    u.meta.update(dict(kind=plan.kind, gen=plan.gen, rebuild_root=plan.rebuild_root, round=plan.round,
                       n_working=n_work, n_archive_quarantined=n_aq, n_published=n_pub))
    echo(f'  INSTALL-OK {cell}: {n_work} working files, {n_aq} archive files quarantined, {n_pub} published')
    return mpath, rows


INPUT_REFUSALS = 'refusals_input.csv'      # B-03 (a), lib/masters/input_resolver.REFUSALS_FILE (a cell-level record)
BUILD_STATS = os.path.join('coadd_final', 'build_stats.json')   # D-ADMIT (a), lib/masters/build_stats.FILE
CARRIED_RECORDS = (INPUT_REFUSALS, BUILD_STATS)    # cell records of the branch's builds (paths relative to the cell)


def carry_record(PI, plan, rel):
    """One cell record of the rebuilt generation (``rel``, relative to the cell) goes into the working cell with it, as
    phase3_install carries the cell's other records (CELL_FILES, CELL_TOP): the working cell's earlier record is
    quarantined first, the rebuild root's copied and logged. No production rebuild writes these files: nothing happens
    under production values. Returns the installed path or None."""
    if not plan.rebuild_root:
        return None
    src = os.path.join(plan.rebuild_root, rel)
    dst = os.path.join(plan.cdir, rel)
    if os.path.exists(dst):
        PI._move_to_wq(plan.cell, dst, rel)
    if not os.path.exists(src):
        return None
    shutil.copy2(src, dst)
    PI.rec(plan.cell, 'cell-file', src, dst, PI.md5(dst))
    return dst


def carry_input_refusals(PI, plan):
    """B-03 (a), same-state: the input-refusal record - without it the installed cell keeps no trace of why a night has
    no V products (the per-night product test, T6 and the search read it)."""
    return carry_record(PI, plan, INPUT_REFUSALS)


def add_carried_outputs(u, carried):
    """The carried build statistics is an output of the install unit: its content test (the record re-derived from the
    installed cell's own records) runs on the copy where it landed."""
    for p in carried:
        if os.path.basename(p) == os.path.basename(BUILD_STATS):
            u.add_output(p, 'build_stats')


def carry_cell_records(PI, plan):
    """Every record of CARRIED_RECORDS: the B-03 input refusals and the D-ADMIT build statistics (build_stats.json -
    sources relative to the cell, so the copy stays valid where it lands; the population a later admitted night is judged
    by). Returns the installed paths."""
    return [p for p in (carry_record(PI, plan, rel) for rel in CARRIED_RECORDS) if p]


def record_install_io(u, plan, rows):
    """Inputs (every generation file with its md5 from the action log; the records), admitted nights, outputs."""
    for r in rows:
        if r['step'] == 'installed' and os.path.isabs(r['path']) and os.path.exists(r['path']):
            u.add_input(r['path'], extra={'md5': r['md5']})
    if plan.kind == 'mdewarp' and os.path.exists(plan.screen_json):
        u.add_input(plan.screen_json)
    fd = os.path.join(plan.cdir, 'coadd_final')
    u.set_admitted(SC.admitted_in_master(os.path.join(fd, 'weights.csv'), plan.lst, plan.scope))


def add_install_outputs(u, plan, mpath):
    roots = plan.roots
    u.add_output(mpath, 'install_manifest')
    u.add_output(roots.masters_dir(plan.lst, plan.scope), 'archive_cell')
    if plan.kind == 'mdewarp':
        add_screen_output(u, roots, plan.cell, plan.screen_json)


def add_screen_output(u, roots, cell, applied_json):
    """Step e's product: the archived screen record (class screen_table, owned by install - crosscheck 4.2; dewarp writes its applied=0 rows as the same class) with
    the context of this cell (install_cell, applied_json: the cell's applied=1 row and its published screen = the
    generation's APPLIED_SCREEN.json). The context is kept on the output entry for later checks."""
    ctx = {'install_cell': cell, 'applied_json': applied_json}
    e = u.add_output(roots.screen_table(), 'screen_table', **ctx)
    e['ctx'] = ctx


def verify_cell(u, cdir, gates, echo):
    """Stage verify: the installed cell against every membership and product ruling (install_cell.sh:23-24 =
    `python -m exopipe.phase3.membership verify <cell>`, membership.py:317-329)."""
    from exopipe.phase3.lib.masters import membership as MB
    v = MB.verify(cdir) + MB.verify_products(cdir)
    text = f'MEMBERSHIP+PRODUCTS {"PASS" if not v else "FAIL"} {os.path.basename(cdir)}' + ''.join('\n  ' + x for x in v)
    echo('  ' + text.replace('\n', '\n  '))
    gates.check('V', not v, text, value=len(v))


# ---------------------------------------------------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------------------------------------------------

def _round_for_recover(cell, args, roots, run_obj):
    from exopipe.phase3.core import record
    if args.round and args.round != 'auto':
        return args.round
    rec = record.load_record(roots.out, 'install', cell)
    rnd = ((rec or {}).get('meta') or {}).get('round')
    if not rnd:
        raise Refusal(f'{cell}: no install record naming the round of the interrupted attempt under {roots.records}: '
                      f'name it with --round (the round of its working-quarantine rows in the action log)')
    return rnd


def recover_cell(u, cell, args, roots, run_obj, echo):
    PI = SC.pi_module()
    kind, gen, rebuild_root = resolve(cell, args)
    rnd = _round_for_recover(cell, args, roots, run_obj)
    execute = bool(args.execute) and not run_obj.dry_run
    u.meta.update(kind=kind, gen=gen, round=rnd, execute=execute)
    off = SC.log_offset(roots.record_csv)
    with SC.pi_context(PI, roots, rnd) as settings:
        u.note(f'in-place phase3_install.resume(execute={execute}) with {json.dumps(settings, sort_keys=True)}')
        buf = None
        try:
            with SC.captured_stdout() as buf:
                res = PI.resume(cell, gen, kind, roots.working, omit_vclean=args.omit_vclean, execute=execute)
        except Exception as e:
            text = buf.getvalue() if buf is not None else ''
            if text.strip():
                echo(text.rstrip())
                u.note(text.strip()[-3000:])
            moved = [r for r in SC.action_rows(roots.record_csv, off, cell) if r['step'] in SC.MOVE_STEPS]
            if execute:
                PI.rec(cell, 'RESUME-FAILED', gen, '', '', '', str(e)[:400])       # as phase3_install.main
            if isinstance(e, RuntimeError) and not moved:
                raise Refusal(f'recover: {e}')
            raise
        text = buf.getvalue()
        carried = []
        if res is not None:                    # as the normal path: the branch's cell records move with the cell's records
            import types
            carried = carry_cell_records(PI, types.SimpleNamespace(rebuild_root=rebuild_root, cdir=roots.cdir(cell), cell=cell))
            if carried:
                u.note(f'cell records carried into the working cell (recover): {carried}')
    echo(text.rstrip())
    u.note(text.strip()[-3000:])
    if res is None:
        raise NoInput('recover reported only (pass --execute to act): ' + ' / '.join(text.strip().splitlines()[-2:]))
    rows = SC.action_rows(roots.record_csv, off, cell)
    mpath = os.path.join(roots.records, f'{cell}.recover.manifest.csv')
    SC.keep_previous(mpath)
    SC.write_manifest(mpath, [dict(time=SC.utcnow(), cell=cell, step='plan-round', path=roots.wq(rnd), dest=roots.aq(rnd),
                                   detail=f'round {rnd} of the interrupted attempt (recover)')], rows)
    lst, scope = cell.split('_', 1)
    u.check('install.RECOVER-OK', True, value=res[0], detail=f'INSTALL-OK {cell} (resumed): {res[0]} working files, '
                                                             f'{res[1]} archive files quarantined, {res[2]} published')
    u.add_output(mpath, 'install_manifest')
    u.add_output(roots.masters_dir(lst, scope), 'archive_cell')
    if kind == 'mdewarp':
        add_screen_output(u, roots, cell, os.path.join(os.path.dirname(gen), _cfg().install.SCREEN_RECORD_JSON))
    u.add_output(roots.cdir(cell), 'installed_cell')
    add_carried_outputs(u, carried)


def epochs_repair_cell(u, cell, roots, run_obj, echo):
    PI = SC.pi_module()
    lst, scope = cell.split('_', 1)
    cdir = roots.cdir(cell)
    if run_obj.dry_run:
        sd = os.path.join(cdir, 'subtracted')
        subs = sorted(glob.glob(f'{sd}/*_sub_{scope}.fits'))
        eps = sorted(glob.glob(f'{sd}/*_epoch_{scope}.fits'))
        echo(f'  epochs-repair {cell}: {len(subs)} residuals, {len(eps)} epoch images in {sd}; would derive the missing '
             f'epochs (residual + master), test them, publish the cell to {roots.masters_dir(lst, scope)} and audit')
        return
    off = SC.log_offset(roots.record_csv)
    with SC.pi_context(PI, roots) as settings:
        u.note(f'in-place phase3_install.epochs_only with {json.dumps(settings, sort_keys=True)}')
        buf = None
        try:
            with SC.captured_stdout() as buf:
                res = PI.epochs_only(cell, roots.working)
        except Exception as e:
            text = buf.getvalue() if buf is not None else ''
            if text.strip():
                u.note(text.strip()[-3000:])
            PI.rec(cell, 'EPOCHS-FAILED', '', '', '', '', str(e)[:400])            # as phase3_install.main
            wrote = [r for r in SC.action_rows(roots.record_csv, off, cell) if r['step'] in SC.MOVE_STEPS]
            if isinstance(e, RuntimeError) and not wrote:
                raise Refusal(f'{SC.pi_gate_of(str(e))}: {e}')
            raise
        text = buf.getvalue()
    u.note(text.strip()[-3000:])
    rows = SC.action_rows(roots.record_csv, off, cell)
    mpath = os.path.join(roots.records, f'{cell}.epochs.manifest.csv')
    SC.keep_previous(mpath)
    SC.write_manifest(mpath, [], rows)
    u.check('install.EPOCHS-OK', True, value=res[0], detail=f'EPOCHS-OK {cell}: {res[0]} epochs derived, {res[1]} '
                                                            f'products published and audited')
    echo(f'  EPOCHS-OK {cell}: {res[0]} epochs derived, {res[1]} products published and audited')
    u.add_output(mpath, 'install_manifest')
    u.add_output(roots.masters_dir(lst, scope), 'archive_cell')
    u.add_output(cdir, 'installed_cell')


def screen_unapplied(u, args, roots, run_obj, echo):
    import csv
    PI = SC.pi_module()
    table = args.table
    if not table or not os.path.isfile(table):
        raise Refusal(f'--stages screen-unapplied needs --table <merged dewarp measurement table> (got {table!r})')
    sdir = roots.screen_dir()
    p = os.path.join(sdir, _cfg().install.SCREEN_TABLE)
    have = {(r['lst'], r['scope']) for r in csv.DictReader(open(p)) if r.get('applied') == '1'} if os.path.exists(p) else set()
    final = {(r['lst'], r['scope']) for r in csv.DictReader(open(table))}
    todo = sorted(final - have)
    echo(f'  screen table {p}: {len(have)} applied rows; {len(final)} measured cells in {table}; {len(todo)} applied=0 rows '
         f'to write' + (f' (e.g. {", ".join("_".join(x) for x in todo[:5])})' if todo else ''))
    u.add_input(table)
    if run_obj.dry_run:
        return
    with SC.pi_context(PI, roots) as settings:
        u.note(f'in-place phase3_install.screen_table_unapplied with {json.dumps(settings, sort_keys=True)}')
        n1, n0 = PI.screen_table_unapplied(table)
    u.check('install.SCREEN-TABLE', True, value=n0, detail=f'SCREEN-TABLE {n1} applied rows kept, {n0} applied=0 rows written')
    u.add_output(p, 'screen_table')


# ---------------------------------------------------------------------------------------------------------------------
# the dispatcher of the deep tier
# ---------------------------------------------------------------------------------------------------------------------

def _echo_fn(dry):
    def echo(text):
        print(text, flush=True)
    return echo


def ensure_unit_status_printed(u, echo):
    echo(f'  unit {u.name}: {u.status}' + (f' ({u.reason[:300]})' if u.reason else ''))


def run(args, run_obj, stages, out):
    from exopipe.phase3.core import cells, record
    from exopipe.phase3.lib.install import steps_enqueue as SE
    roots = SC.Roots(out, SC.is_rerooted(args.out))
    PI = SC.pi_module()
    SC.check_protected_overrides()
    SC.check_window_consistency(PI)
    SC.check_waivers(run_obj)
    echo = _echo_fn(run_obj.dry_run)
    dry = run_obj.dry_run
    echo(f'exopipe3 install --tier deep: stages {",".join(stages)}; out {roots.out} ('
         + ('re-rooted under --out' if roots.rerooted else 'production roots') + ')' + ('; DRY RUN (writes nothing)'
                                                                                         if dry else ''))
    mode = [s for s in stages if s in MODES]
    if mode == ['screen-unapplied']:
        with run_obj.unit(f'screen-unapplied.{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}') as u:
            screen_unapplied(u, args, roots, run_obj, echo)
        return _dry_rc(run_obj) if dry else None
    sel = cells.expand_cells(args.cells) if args.cells else []
    excl = set(cells.expand_cells(args.exclude_cells)) if args.exclude_cells else set()
    sel = [c for c in sel if c not in excl]
    if not sel:
        raise NoInput('no cells selected (--cells; --exclude-cells removed the rest)' if args.cells else
                      'no cells selected: give --cells (e.g. 12h_Red, @list)')
    enqueued = []
    for cell in sel:
        echo(f'{cell}:')
        if mode == ['recover']:
            with run_obj.unit(f'{cell}.recover') as u:
                recover_cell(u, cell, args, roots, run_obj, echo)
            continue
        if mode == ['epochs-repair']:
            with run_obj.unit(f'{cell}.epochs') as u:
                epochs_repair_cell(u, cell, roots, run_obj, echo)
            continue
        main_ok = True
        if 'plan' in stages or 'install' in stages or 'verify' in stages:
            if 'install' in stages:
                name = cell
            elif 'plan' in stages:
                name = f'{cell}.plan'
            else:
                name = f'{cell}.verify'
            with run_obj.unit(name, stages=list(stages)) as u:
                _main_unit(u, cell, args, roots, run_obj, stages, echo)
                if dry:
                    u.reason = 'dry run: plan printed, nothing written'
            main_ok = u.status in ('done',) or (dry and u.status in ('no-input', 'done'))
            echo(f'  unit {name}: {u.status}' + (f' ({u.reason[:300]})' if u.reason else ''))
        path = None
        if ('enqueue' in stages or 'cleanup' in stages):
            if not main_ok:
                echo(f'  enqueue/cleanup of {cell} not attempted: unit {name} {u.status}')
                continue
            with run_obj.unit(f'{cell}.enqueue', stages=[s for s in stages if s in ('enqueue', 'cleanup')]) as ue:
                path = SE.enqueue_cell(ue, cell, roots, run_obj, stages, echo, verified_now='verify' in stages)
                if dry:
                    ue.reason = 'dry run: plan printed, nothing written'
            echo(f'  unit {cell}.enqueue: {ue.status}' + (f' ({ue.reason[:300]})' if ue.reason else ''))
            if path and ue.status == 'done':
                enqueued.append((cell, path))
    if enqueued and not dry:
        with run_obj.unit(f'rerun.{run_obj.run_id}') as ur:
            p = SE.write_combined_rerun(ur, [c for c, _ in enqueued], roots, run_obj)
        echo(f'per-hour items to re-run: {p}\n  exopipe3 run submit --chain epoch --items @{p}')
    return _dry_rc(run_obj) if dry else None


def _main_unit(u, cell, args, roots, run_obj, stages, echo):
    from exopipe.phase3.core import cells
    lst, scope = cells.parse_cell(cell)
    dry = run_obj.dry_run
    gates = SC.Gates(u, run_obj, collect=dry, echo=echo)
    if 'plan' in stages or 'install' in stages:
        plan = CellPlan(cell, args, roots)
        u.meta.update(kind=plan.kind, gen=plan.gen, rebuild_root=plan.rebuild_root)
        # --resume: a done install record whose inputs, admitted nights, code, config are unchanged and whose outputs pass
        if run_obj.resume and 'install' in stages and not dry:
            adm = SC.admitted_in_master(os.path.join(plan.cdir, 'coadd_final', 'weights.csv'), lst, scope)
            ok, why = run_obj.resume_check(cell, inputs=None, admitted=adm)
            if ok:
                from exopipe.phase3.core import record
                SC.adopt_record(u, record.load_record(roots.out, 'install', cell), why)
                echo(f'  resumed: {why}')
                return
            u.note(f'--resume: not skipped ({why})')
        call_pi = dry or 'install' not in stages
        plan_cell(u, plan, run_obj, gates, call_pi, args.round or _cfg().install.ROUND, echo)
        print_plan(plan, echo)
        gates.finish()
        if 'install' not in stages:
            if not dry:
                mpath = os.path.join(roots.records, f'{cell}.plan.manifest.csv')
                SC.keep_previous(mpath)
                SC.write_manifest(mpath, plan.rows(gate_summary='; '.join(
                    f"{t['name']}={'pass' if t['passed'] else 'FAIL'}" for t in u.tests)))
                u.add_output(mpath, 'install_manifest')
            if 'verify' not in stages:
                return
        elif dry:
            echo('  install: would call phase3_install.install (stages a0-e) with the round above')
            if 'verify' in stages:
                echo(f'  verify: would run membership.verify + verify_products on {plan.cdir}')
            return
        else:
            mpath, rows = install_cell(u, plan, gates, echo)
            record_install_io(u, plan, rows)
            add_install_outputs(u, plan, mpath)
    # verify (also alone: unit <cell>.verify, today's kind 'installed')
    cdir = roots.cdir(cell)
    if 'verify' in stages:
        if dry and 'install' not in stages:
            if os.path.isdir(cdir):
                verify_cell(u, cdir, gates, echo)
                gates.finish()
            else:
                gates.check('CELL', False, f'installed cell {cdir} absent')
                gates.finish()
            return
        if not os.path.isdir(cdir):
            raise Refusal(f'CELL: installed cell {cdir} absent')
        if 'install' not in stages:
            u.add_input(os.path.join(cdir, 'coadd_final', 'weights.csv'))
            u.set_admitted(SC.admitted_in_master(os.path.join(cdir, 'coadd_final', 'weights.csv'), lst, scope))
        verify_cell(u, cdir, gates, echo)
    if 'install' in stages or 'verify' in stages:
        u.add_output(cdir, 'installed_cell')


def _dry_rc(run_obj):
    """Exit code of a dry run: 2 when any unit's gates refused, 1 on an error, else 0 (a plan writes no output, which the
    record would call no-input)."""
    st = [u.status for u in run_obj.units]
    if 'error' in st:
        return 1
    if 'refused' in st:
        return 2
    return 0
