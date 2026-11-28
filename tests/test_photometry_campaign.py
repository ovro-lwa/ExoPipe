#!/usr/bin/env python
"""Cutout-generation campaigns (lib/photometry/campaign.py, config photometry_campaign; plan GEN6_PORT_PLAN_20261123.md pull
request 4, the port of the generation-6 drivers). Data free: a synthetic cube root (empty files named as flow cubes), a
synthetic exclusion list, synthetic unit records, stub engine files and a stub `lfs`. Asserts:
  - the generation: both roots derive from photometry_campaign.GENERATION; the served generation or a bad name refuses
    (CP-GEN); --campaign routes the cutout root (steps_common.cutout_root) and the stage-1 root (steps_snapshot.stage1_root);
    the flags a campaign does not take refuse (CP-SCOPE), so do stages other than cutouts and measure (CP-STAGES); the
    step parses --campaign; the production cutout root refuses in the refactor window (C-ROOT), reported by a dry run;
  - the enumeration: cube directories one level down, subbands with both Stokes cubes; excluded by core/exclusions OR the
    subband engine's reader; held on a disagreement of the two, one Stokes cube, no MANIFEST COMPLETE line, no rescale
    record; the widebands of every cell with a subband unit, none for a cell without;
  - the code pin: a pinned engine changed refuses the unit (CP-PIN) and every later one, even after the file is restored; a
    unit's child fingerprint joins the pin when first seen, another version or 'mixed:' stops the campaign;
  - the space gate: `lfs quota -q -u <the run user>` and the filesystem_summary line of `lfs df` parsed; each floor (quota
    bytes, quota files, free bytes) refuses below it (CP-SPACE), sticky; a limit of 0 is no limit; an unreadable read
    refuses; at the start a low read refuses the campaign and writes no start record;
  - the I/O guard: the host's stop file with an io_guard line, or the alert file, makes a unit wait; other stop lines do not;
  - stage-1 readiness: done / C-EXCLUDED / C-NOCUT with no-file counts are terminal; a no-input gap (C-NOCUBE) holds
    nothing; a refused gap (C-FLUXSCALE before the engine) holds its cell's kind as a gap (SM-HELD, stage 1 not run); any
    other refusal (e.g. the generation check), an error, no record, a C-NOCUT whose counts promise files, a held subband
    hold ONLY their cell's kind (SM-HELD); all units without a cutout -> 'no wideband' (SM-NOBAND, no input, nothing
    measured); a ready kind is measured into its kind's manifest with the campaign gate; the campaign's cutouts outcome
    never stops its measure stage (run_tier);
  - the wiring: steps_cutouts lists the campaign's units, runs the campaign gate first in each unit (before C-NOCUBE) and
    the pin after each unit; its dry run lists a unit without a cube as C-NOCUBE before any flux-scale verdict;
  - measure_cell: the gate runs first in the unit; a kind manifest is rewritten unless it holds rows of another band
    (SM-SUBSET);
  - the complete-when check: a done record passes only with its required checks (by band kind: GP, GA + fs1 + the
    flux-scale record, GC, GB for the generation-5 sample, GW), its manifest's files present with their sizes and a file
    per Stokes with a positive engine count; stage 1 per kind done (MM1 + MM2) or 'no <kind>'; gaps (input-gate refusals,
    ruled) are listed and complete ("COMPLETE with N gaps", exit 0); output or content refusals, errors and missing records
    are incomplete (exit 1).
  - the resume hash: the step's CONFIG holds every section of config/steps/photometry.py but photometry_campaign
    (photometry_fluxscale among them): a flux-scale tolerance change changes the hash; a campaign knob change does not
    (operational: the roots its GENERATION moves are each unit's recorded inputs and outputs).
  - review of PR 154: a record of another run under the same --out (a gen5 --in-place record) is not done; the pin persists
    in the start record, a later invocation with other engine code refuses CP-PIN unless --new-campaign, whose check then
    calls the old units incomplete; a unit whose children ran another version than the pin is recorded refused CP-PIN and
    holds its cell; check() enumerates with both readers (a disagreement is incomplete); a re-cut after stage 1 makes the
    stage 1 incomplete and readiness measures it again; --cells naming a cell without a cube directory holds that cell; an
    unreadable rescale record is C-FLUXSCALE-READ (incomplete), a verdict C-FLUXSCALE (a gap); a manifest that cannot be
    parsed refuses SM-SUBSET; a stopped campaign refuses before the I/O guard; the summary line counts the units without a
    cutout by why.
Scratch: a fresh directory under $EXOPIPE3_TEST_SCRATCH, else the system temp directory, removed at exit.
Run: python tests/test_photometry_campaign.py
"""
import argparse
import contextlib
import csv
import io
import json
import os
import pwd
import shutil
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')

from exopipe.phase3.config import CFG                                # noqa: E402
from exopipe.phase3.core import exclusions, record                  # noqa: E402
from exopipe.phase3.core.record import Refusal                      # noqa: E402
from exopipe.phase3.lib.photometry import campaign as CP            # noqa: E402
from exopipe.phase3.lib.photometry import steps_common as K         # noqa: E402
from exopipe.phase3.lib.photometry import steps_snapshot as SS      # noqa: E402

FAILS = []
SB = [b for b in CFG.photometry.SNAPSHOT_BANDS if 'MHz' in b]
WB = [b for b in CFG.photometry.SNAPSHOT_BANDS if 'MHz' not in b]


def check(label, cond, detail=''):
    print(('PASS ' if cond else 'FAIL ') + label + ('' if cond else '  ' + str(detail)[-1500:]), flush=True)
    if not cond:
        FAILS.append(label)


def raises(fn, *a, **k):
    try:
        fn(*a, **k)
    except Refusal as e:
        return str(e)
    return None


class Patch:
    def __init__(self):
        self.saved = []

    def set(self, obj, key, val):
        # the raw attribute (a config.Derived key's descriptor, not its value), so undo() restores it as it was
        self.saved.append((obj, key, vars(obj)[key] if key in vars(obj) else getattr(obj, key)))
        setattr(obj, key, val)

    def undo(self):
        while self.saved:
            obj, key, val = self.saved.pop()
            setattr(obj, key, val)


TIDENT = ('gen6', '/r/photometry_gen6')     # (generation, cutout root) of the synthetic campaign the records belong to


def write_rec(out, name, status, reason='', meta=None, tests=(), outputs=(), ident=TIDENT):
    """A unit record; ``ident``: the campaign it belongs to (meta campaign + cutout_root), None: another run's (no keys)."""
    m = dict(campaign=ident[0], cutout_root=ident[1]) if ident else {}
    m.update(meta or {})
    record.atomic_write_json(record.record_path(out, 'photometry', name),
                             dict(unit=name, status=status, reason=reason, meta=m, tests=list(tests), outputs=list(outputs)))


def start_rec(out, root, run_id='photometry_20261123T000000Z_h_1', pin=None, e=None):
    """A campaign start record under ``out`` (what check() reads: generation, cutout root, pin)."""
    d = os.path.join(out, '_records', 'photometry', CFG.photometry_campaign.RECORD_SUBDIR)
    os.makedirs(d, exist_ok=True)
    rec = dict(run_id=run_id, generation='gen6', cutout_root=root, pin=pin or {}, started='2026-11-23T00:00:00Z', scope=None,
               enumeration=(e or CP.Enumeration()).to_dict())
    record.atomic_write_json(os.path.join(d, 'gen6_%s.json' % run_id), rec)
    return rec


def st1_rec(out, root, cell, kind, wr=None):
    """A done stage-1 record of (cell, kind) measured from the cutouts ``root`` holds now (its listing digest)."""
    bands = list(CP.kind_bands(kind))
    files = [p for b in bands for p in SS.cell_files(cell, root, None, b)]
    (wr or write_rec)(out, SS.unit_name(cell, bands), 'done', meta={
        'bands_done': bands, 'cutout_root': root, 'cutout_listing_sha1': SS.listing_sha1(files) if files else ''})


def run_of(out, **kw):
    os.makedirs(out, exist_ok=True)
    return record.Run('photometry', ['test'], out, CFG, **kw)


# ---------------------------------------------------------------------------------------------------------------------

def served_gen5(P, d):
    """The served state the gen6 campaign ran in (2026-11-17..27): snapshot generation 5 - photometry.GENERATIONS['snapshot']
    = gen5 (CUTOUT_ROOTS['snapshot'] photometry_gen5, STAGE1_DIR stage1_gen5 and PLANETS_PAGE_GENERATION follow: Derived)
    and the served registry's fixture of then (the fixture with the 10-s / 1-min runs of gen5 and the planets page record's
    gen5 per-night inputs, in ``d``; SERVED-GEN compares the two), in memory only. Since 2026-11-27 generation 6 is served
    (the planets page since PLANETS-GEN6-DONE, 2026-11-28), so the configured campaign generation gen6 refuses CP-GEN
    (test_generation); the campaign's own tests run in its own state."""
    ph = CFG.photometry
    with open(K.served_fixture_path()) as f:
        fx = json.load(f)
    for v in fx['registry']['forced'].values():
        for cad in ('10s', '1min'):
            v['run'] = v['run'].replace('%s_gen6' % cad, '%s_gen5' % cad)
    fl = fx['page_records']['planets']['fields']
    fl.update({'inputs.stage1': os.path.join(ph.OUT_ROOT['snapshot'], ph.PLANETS_STAGE1_FMT.format(generation='gen5')),
               'inputs.phot': CFG.photometry_campaign.CUTOUT_ROOT_FMT.format(generation='gen5')})
    path = os.path.join(d, 'served_generations_gen5.json')
    with open(path, 'w') as f:
        json.dump(fx, f)
    P.set(ph, 'SERVED_GENERATIONS_FILE', path)
    P.set(ph, 'GENERATIONS', dict(ph.GENERATIONS, snapshot='gen5'))


def test_generation(d):
    P = Patch()
    served_file = CFG.photometry.SERVED_GENERATIONS_FILE
    try:
        e = raises(CP.generation)
        check('generation 6 is served (photometry.GENERATIONS[snapshot] = gen6, 2026-11-27), so the configured campaign '
              'generation gen6 refuses CP-GEN (a campaign writes a new generation)', CFG.photometry.GENERATIONS['snapshot']
              == 'gen6' and CFG.photometry_campaign.GENERATION == 'gen6' and e is not None and 'CP-GEN' in e, e)
        e2 = raises(CP.generation, write=False)
        check('... CP-GEN\'s served-generation refusal is for writes: the read-only check reads gen6 (generation(write=False) = '
              'gen6, stage1_dir(write=False) = stage1_gen6)', e2 is None and CP.generation(write=False) == 'gen6'
              and CP.stage1_dir(write=False) == 'stage1_gen6' and CP.cutout_root(write=False).endswith('photometry_gen6'), e2)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = CP.main(['check', '--generation', 'gen6', '--out', os.path.join(d, 'out_gen_check_none')])
        check('`campaign check --generation gen6` (the served generation) runs: past CP-GEN to the content test, which says '
              'there is no gen6 campaign start record under that --out (exit 2)', rc == 2 and 'CP-GEN' not in buf.getvalue()
              and 'no campaign start record of generation gen6' in buf.getvalue(), (rc, buf.getvalue()[-300:]))
        served_gen5(P, d)
        check('the generation written is photometry_campaign.GENERATION (gen6), the served one photometry.GENERATIONS'
              '[snapshot] (gen5 while the campaign ran; CUTOUT_ROOTS and STAGE1_DIR follow it)', CP.generation() == 'gen6'
              and CFG.photometry.GENERATIONS['snapshot'] == 'gen5' and CFG.photometry.STAGE1_DIR == 'stage1_gen5'
              and CFG.photometry.CUTOUT_ROOTS['snapshot'].endswith('photometry_gen5'), CP.generation())
        check('both roots derive from it: %s, <out>/%s' % (CP.cutout_root(), CP.stage1_dir()),
              CP.cutout_root() == '/lustre/pipeline/exopipe/phase3/photometry_gen6' and CP.stage1_dir() == 'stage1_gen6')
        P.set(CFG.photometry_campaign, 'GENERATION', 'gen7')
        check('... another generation value moves both roots', CP.cutout_root().endswith('photometry_gen7')
              and CP.stage1_dir() == 'stage1_gen7')
        args = argparse.Namespace(campaign=True, cutout_root=None)
        run = argparse.Namespace(out_root=os.path.join(d, 'out_gen'))
        check('--campaign: the cutout root the stages read and write is the campaign\'s (steps_common.cutout_root), stage 1 '
              'is <out>/<its directory> (steps_snapshot.stage1_root)', K.cutout_root(args) == CP.cutout_root()
              and SS.stage1_root(run, args) == os.path.join(run.out_root, 'stage1_gen7'))
        plain = argparse.Namespace(campaign=False, cutout_root=None, tier='snapshot')
        check('... without --campaign the served roots as before', K.cutout_root(plain) == CFG.photometry.CUTOUT_ROOTS['snapshot']
              and SS.stage1_root(run, plain) == SS.stage1_root(run) == os.path.join(run.out_root, CFG.photometry.STAGE1_DIR))
        P.set(CFG.photometry_campaign, 'GENERATION', CFG.photometry.GENERATIONS['snapshot'])
        e = raises(CP.generation)
        check('CP-GEN: the served generation as the campaign\'s refuses (a campaign writes a new generation)',
              e is not None and 'CP-GEN' in e and 'served' in e, e)
        P.set(CFG.photometry_campaign, 'GENERATION', 'photometry_gen6')
        e = raises(CP.generation)
        check('CP-GEN: a value that does not name a generation refuses', e is not None and 'CP-GEN' in e, e)
        P.set(CFG.photometry_campaign, 'GENERATION', 'gen7')
        P.set(CFG.photometry, 'SERVED_GENERATIONS_FILE', served_file)
        e = raises(CP.generation)
        check('SERVED-GEN: photometry.GENERATIONS[snapshot] gen5 against the served registry\'s fixture (gen6) refuses: the '
              'served-root gate protects what the hub serves (a served generation changes in one commit with the fixture)',
              e is not None and 'SERVED-GEN' in e and 'GENERATIONS[snapshot] = gen5' in e and 'serves gen6' in e, e)
    finally:
        P.undo()


def campaign_args(**kw):
    a = dict(tier='snapshot', cutout_root=None, stage1=None, scopes=None, index=None, manifest_name=None, targets=None,
             planets=False, in_place=False, campaign=True, cells=None, epochs=None, exclude_cells=None)
    a.update(kw)
    return argparse.Namespace(**a)


def test_resume_hash():
    """The step's resume hash (record.Run.config_hash over config.step_sections of the step's CONFIG) covers every section of
    config/steps/photometry.py but the campaign's - the flux-scale gate's (#153) among them: a changed tolerance makes --resume
    redo the units; a changed campaign knob (a space floor) does not."""
    import importlib
    import inspect
    from exopipe.phase3 import config
    from exopipe.phase3.config.steps import photometry as PC
    step = importlib.import_module('exopipe.phase3.steps.photometry')
    secs = config.step_sections(step)
    mine = sorted(n for n, o in vars(PC).items() if inspect.isclass(o) and o.__module__ == PC.__name__
                  and issubclass(o, config._Section) and o is not config._Section)
    check('the photometry step declares every section of config/steps/photometry.py but photometry_campaign (its resume hash '
          'covers them)', mine and all(n in secs for n in mine if n != 'photometry_campaign')
          and 'photometry_campaign' not in secs and 'photometry_fluxscale' in secs, (mine, secs))
    P = Patch()
    try:
        for sec, key, val, moves in (('photometry_fluxscale', 'FSAPPLD_TOL', 3e-6, True),
                                     ('photometry_campaign', 'QUOTA_MIN_FREE_FILES', 7, False),
                                     ('photometry_campaign', 'LFS_TIMEOUT_S', 11, False)):
            h0 = CFG.hash(secs)
            P.set(getattr(CFG, sec), key, val)
            h1 = CFG.hash(secs)
            P.undo()
            check('a change to %s.%s %s the photometry step\'s resume hash (%s -> %s), restored after' % (
                sec, key, 'changes' if moves else 'does NOT change (operational: --resume stays valid)', h0, h1),
                (h1 != h0) == moves and CFG.hash(secs) == h0)
    finally:
        P.undo()


def test_args():
    check('a campaign with its own flags only passes the gate', raises(CP.gate_args, campaign_args(), ['cutouts', 'measure'])
          is None)
    for flag, kw in (('--cutout-root', dict(cutout_root='/x')), ('--stage1', dict(stage1='/x')), ('--scopes', dict(scopes='Red')),
                     ('--index', dict(index='/x')), ('--manifest-name', dict(manifest_name='m.csv')),
                     ('--targets', dict(targets='A=/x')), ('--planets', dict(planets=True)), ('--in-place', dict(in_place=True))):
        e = raises(CP.gate_args, campaign_args(**kw), ['measure'])
        check('CP-SCOPE: --campaign with %s refuses' % flag, e is not None and 'CP-SCOPE' in e and flag in e, e)
    e = raises(CP.gate_args, campaign_args(tier='deep'), ['measure'])
    check('CP-SCOPE: --campaign in the deep tier refuses', e is not None and 'CP-SCOPE' in e, e)
    e = raises(CP.gate_args, campaign_args(), ['measure', 'evaluate'])
    check('CP-STAGES: a campaign with the evaluation (stage 2) refuses', e is not None and 'CP-STAGES' in e, e)
    e = raises(CP.scope_cells, campaign_args(cells='00h_2025-01-05_55MHz'))
    check('CP-SCOPE: a band-suffixed cell refuses (a campaign covers whole cells)', e is not None and 'CP-SCOPE' in e, e)
    from exopipe.phase3.steps import photometry as SP
    ap = argparse.ArgumentParser()
    SP.add_arguments(ap)
    check('the step parses --campaign and --new-campaign; photometry_campaign is not in its resume hash',
          ap.parse_args(['--campaign']).campaign is True and ap.parse_args(['--campaign']).new_campaign is False
          and ap.parse_args(['--campaign', '--new-campaign']).new_campaign is True and 'photometry_campaign' not in SP.CONFIG)


# ---------------------------------------------------------------------------------------------------------------------

class World:
    """A synthetic cube root and exclusion list."""

    def __init__(self, d):
        self.d = d
        self.root = os.path.join(d, 'cubes')
        self.rec = {}
        self.dd = os.path.join(d, 'data_decisions.csv')
        with open(self.dd, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['lst', 'subband', 'date', 'verdict', 'reason'])
            w.writerows([('00h', '23MHz', '2025-01-05', 'exclude', 'synthetic'),
                         ] + [('02h', sb, '2025-01-07', 'exclude', 'synthetic') for sb in SB])
        self.ex = exclusions.PerNight(self.dd, gpu_cells=set(), fresh=True)
        # A: units 41, 55 MHz; 64 MHz V only (held); 23 MHz excluded (core); 27 MHz excluded by the engine only (held, the
        # readers disagree); 32 MHz no I record (held). B: no COMPLETE line (its subbands held). C: every subband excluded.
        self.cube('00h_2025-01-05', ['41MHz', '55MHz', '23MHz', '27MHz', '32MHz'], complete=True, norec={'32MHz': 'I'})
        self.cube('00h_2025-01-05', ['64MHz'], stokes='V')
        self.cube('01h_2025-01-06', ['41MHz'], complete=False)
        self.cube('02h_2025-01-07', SB, complete=True)
        os.makedirs(os.path.join(self.root, 'not_a_cell'))
        open(os.path.join(self.root, '03h_2025-01-08'), 'w').close()       # a file, not a cell directory
        self.engine = {'00h_2025-01-05': {'23MHz', '27MHz'}, '01h_2025-01-06': set(), '02h_2025-01-07': set(SB)}
        self.enumerate = CP.enumerate_units      # the real one (test_start replaces the module's)

    def cube(self, cell, sbs, complete=True, stokes='VI', norec=None):
        cd = os.path.join(self.root, cell)
        os.makedirs(cd, exist_ok=True)
        man = os.path.join(cd, 'MANIFEST.txt')
        if not os.path.exists(man):
            open(man, 'w').write('cubes of %s\n%s' % (cell, 'COMPLETE 2026-11-17\n' if complete else 'building\n'))
        for sb in sbs:
            for S in stokes:
                p = os.path.join(cd, CP.cube_name(cell, sb, S))
                open(p, 'w').close()
                if (norec or {}).get(sb) != S:
                    self.rec[os.path.normpath(p)] = {'source': 'rescale-results', 'factor': 1.0}

    def complete(self, cell):
        p = os.path.join(self.root, cell, 'MANIFEST.txt')
        return os.path.isfile(p) and any(ln.startswith('COMPLETE') for ln in open(p))

    def enum(self, engine=True, cells=None):
        return self.enumerate(cells, self.root, self.rec, self.ex, self.engine if engine else None, self.complete)


def test_enumeration(W):
    e = W.enum()
    A = '00h_2025-01-05'
    want = [(A, '41MHz'), (A, '55MHz')] + [(A, b) for b in WB]
    check('units: the subbands with both Stokes cubes, a COMPLETE cube MANIFEST, both rescale records and no exclusion, then '
          'the widebands of the cell; no unit of a cell without COMPLETE or with every subband excluded',
          e.units == want, e.units)
    held = {(h['cell'], h['subband']): h['reason'] for h in e.held}
    check('held: one Stokes cube only (64 MHz V)', 'only the Stokes V cube' in held.get((A, '64MHz'), ''), held)
    check('held: the readers disagree (27 MHz: the subband engine excludes, core/exclusions not) - not cut, and named',
          'PHOT-EXCL-READERS' in held.get((A, '27MHz'), ''), held)
    check('held: no rescale record of the I cube (32 MHz)', 'no rescale record of the I cube' in held.get((A, '32MHz'), ''),
          held)
    check('held: a cell without a COMPLETE line in its cube MANIFEST', 'COMPLETE' in held.get(('01h_2025-01-06', '41MHz'), ''),
          held)
    exc = {(x['cell'], x['subband']) for x in e.excluded}
    check('excluded: the rows both readers name (23 MHz of 00h; every subband of 02h), none of them a unit',
          exc == {(A, '23MHz')} | {('02h_2025-01-07', sb) for sb in SB}, sorted(exc))
    check('cells: those with a unit or a held subband (not the all-excluded 02h, not a directory or file not named as a '
          'night cell)', e.cells == [A, '01h_2025-01-06'] and e.n_cube_dirs == 3, (e.cells, e.n_cube_dirs))
    e2 = W.enum(engine=False)
    check('without the engine\'s reader (a dry run\'s listing): core/exclusions alone decides (27 MHz a unit)',
          (A, '27MHz') in e2.units and not e2.engine_read, e2.units)
    e3 = W.enum(cells=[A, '05h_2025-01-09'])
    check('--cells narrows the campaign; a named cell without a cube directory is held; the plan counts the cube directories '
          'in scope (1) apart from those under the root (3)',
          {c for c, _ in e3.units} == {A} and any(h['cell'] == '05h_2025-01-09' for h in e3.held)
          and (e3.n_scoped, e3.n_cube_dirs, e.n_scoped) == (1, 3, 3), (e3.units, e3.held, e3.n_scoped, e3.n_cube_dirs))
    eng = dict(W.engine)
    del eng['01h_2025-01-06']
    e4 = raises(CP.enumerate_units, None, W.root, W.rec, W.ex, eng, W.complete)
    check('CP-ENUM: the engine\'s reader without a cell refuses', e4 is not None and 'CP-ENUM' in e4, e4)
    back = CP.Enumeration.from_dict(json.loads(json.dumps(e.to_dict())))
    check('the enumeration survives its start record (JSON)', back.units == e.units and back.held == e.held)


# ---------------------------------------------------------------------------------------------------------------------

def fake_lfs(kb=100, kq=0, kl=10 * 2 ** 30, fl=10, fq=0, fll=10 ** 7, avail=500 * 10 ** 9, rq=0, rd=0, star=''):
    calls = []

    def lfs(argv):
        calls.append(argv)
        if argv[0] == 'quota':
            return rq, 'Disk quotas for usr x (uid 1):\n     /lustre %d%s %d %d - %d %d %d -\n' % (kb, star, kq, kl, fl, fq, fll), ''
        return rd, ('UUID 1K-blocks Used Available Use%% Mounted on\nlustre-OST0000_UUID 1 1 1 1%% /lustre[OST:0]\n\n'
                    'filesystem_summary:  3822424680448 3619706371072 %d  95%% /lustre\n' % avail), ''
    lfs.calls = calls
    return lfs


def test_space():
    q = CP.parse_quota('/lustre 231098215438  237296943104 237296943104 - 36161145 40000000 40000000 -', '/lustre')
    check('lfs quota parsed (the line of 2026-11-23: KiB and files, used and limits)',
          q == dict(kbytes=231098215438, kbytes_limit=237296943104, files=36161145, files_limit=40000000), q)
    q2 = CP.parse_quota('Disk quotas for usr gh:\n     /lustre\n 237296943200* 0 237296943104 6d 5 0 0 -', '/lustre')
    check('... a value over its limit (\'*\'), the name on its own line, a files limit of 0 (none set)',
          q2 == dict(kbytes=237296943200, kbytes_limit=237296943104, files=5, files_limit=None), q2)
    check('lfs df: the filesystem_summary line\'s available KiB, not its first line',
          CP.parse_df('UUID x\nlustre-MDT0000_UUID 9 9 9 1% /lustre[MDT:0]\nfilesystem_summary:  3 2 202712948736  95% /lustre\n')
          == 202712948736)
    P = Patch()
    try:
        cc = CFG.photometry_campaign
        P.set(cc, 'QUOTA_MIN_FREE_BYTES', 2 * 2 ** 40)
        P.set(cc, 'QUOTA_MIN_FREE_FILES', 200000)
        P.set(cc, 'FS_MIN_FREE_BYTES', 120 * 10 ** 12)
        T = 2 ** 30                                    # KiB per TiB
        ok_args = dict(kb=1 * T, kl=4 * T, fl=10, fll=10 ** 6, avail=int(200e12 / 1024))
        lfs = fake_lfs(**ok_args)
        ok, text, vals = CP.space(lfs)
        check('space ok: 3 TiB and 999990 files of headroom, 200 TB free (floors 2 TiB, 200000 files, 120 TB)', ok
              and vals['quota_free_bytes'] == 3 * 2 ** 40 and vals['quota_free_files'] == 999990, (ok, text, vals))
        user = pwd.getpwuid(os.geteuid()).pw_name
        check('the quota read is the RUN USER\'s (%s; never a fixed name) on photometry_campaign.SPACE_FILESYSTEM' % user,
              lfs.calls[0] == ['quota', '-q', '-u', user, '/lustre'] and lfs.calls[1] == ['df', '/lustre'], lfs.calls)
        for what, kw, word in (('quota bytes (1.9 TiB headroom)', dict(kb=int(2.1 * T)), 'TiB <'),
                               ('quota files (199999 headroom)', dict(fl=10 ** 6 - 199999), 'files <'),
                               ('free bytes (119 TB)', dict(avail=int(119e12 / 1024)), 'TB <')):
            ok, text, _ = CP.space(fake_lfs(**dict(ok_args, **kw)))
            check('CP-SPACE floor %s: not ok' % what, not ok and word in text, text)
        ok, text, vals = CP.space(fake_lfs(**dict(ok_args, kl=0, kq=0, fll=0)))
        check('limits of 0 are no limit (not a headroom of minus the usage)', ok and vals['quota_free_bytes'] is None
              and vals['quota_free_files'] is None, (ok, text))
        ok, text, vals = CP.space(fake_lfs(**dict(ok_args, kl=0, kq=3 * T)))
        check('the soft quota counts when it is the only limit', ok and vals['quota_free_bytes'] == 2 * 2 ** 40, (ok, vals))
        for what, kw in (('quota exit 1 (Permission denied)', dict(rq=1)), ('df exit 1', dict(rd=1))):
            ok, text, _ = CP.space(fake_lfs(**dict(ok_args, **kw)))
            check('unreadable (%s): not ok' % what, not ok and 'unreadable' in text, text)

        def boom(argv):
            raise OSError('no lfs')
        ok, text, _ = CP.space(boom)
        check('lfs that cannot run: not ok', not ok and 'could not run' in text, text)
        state = {'ok': True}
        camp = CP.Campaign('gen6', CP.Enumeration(), None, space_fn=lambda: (state['ok'], 'probe', {}), sleep=lambda s: None)
        P.set(CP, 'io_guard_raised', lambda: None)
        r1 = camp.refusal()
        state['ok'] = False
        r2 = camp.refusal()
        state['ok'] = True
        r3 = camp.refusal()
        check('Campaign: space ok -> the unit runs; low -> CP-SPACE; later units refuse too after it recovers (gen6: the STOP '
              'file stops every claim)', r1 is None and r2 and 'CP-SPACE' in r2 and r3 and 'CP-SPACE' in r3, (r1, r2, r3))
    finally:
        P.undo()


def test_pin(d):
    code = os.path.join(d, 'code')
    os.makedirs(code)
    files = []
    for n in ('engine_a.py', 'engine_b.py', 'helper.py'):
        p = os.path.join(code, n)
        open(p, 'w').write('# %s\n' % n)
        files.append(p)
    pin = CP.CodePin(files[:2])
    P = Patch()
    try:
        P.set(CP, 'io_guard_raised', lambda: None)
        camp = CP.Campaign('gen6', CP.Enumeration(), pin, space_fn=lambda: (True, 'ok', {}), sleep=lambda s: None)
        r0 = camp.refusal()
        open(files[0], 'a').write('# edited mid-campaign\n')
        r1 = camp.refusal()
        open(files[0], 'w').write('# engine_a.py\n')
        r2 = camp.refusal()
        check('CP-PIN: an engine edited after the campaign start refuses the next unit, and every later one even once restored',
              r0 is None and r1 and 'CP-PIN' in r1 and 'engine_a.py' in r1 and r2 and 'CP-PIN' in r2, (r0, r1, r2))
        pin = CP.CodePin(files[:2])
        h = K.code_sha12(files[2])
        bad = pin.absorb({files[2]: h})
        check('a unit\'s child fingerprint: a module first seen joins the pin', not bad and pin.files.get(files[2]) == h,
              pin.files)
        open(files[2], 'a').write('# edited\n')
        check('... and is checked from then on', any('helper.py' in x for x in pin.changed()), pin.changed())
        # review of PR 154, finding 3: a unit whose children ran another version than the pin does not stay done
        pin = CP.CodePin(files[:2])
        cell = '00h_2025-01-05'
        e = CP.Enumeration()
        e.units = [(cell, b) for b in ('41MHz', '55MHz')]
        camp = CP.Campaign('gen6', e, pin, space_fn=lambda: (True, 'ok', {}), sleep=lambda s: None, root=TIDENT[1])
        out = os.path.join(d, 'out_pin')
        run = run_of(out)
        write_rec(out, 'cutouts.%s.41MHz' % cell, 'done')
        name = 'cutouts.%s.55MHz' % cell
        write_rec(out, name, 'done', meta={'legacy_code': {files[1]: 'ffffffffffff'}})
        u = argparse.Namespace(name=name, run=run, status='done', reason='', meta={'legacy_code': {files[1]: 'ffffffffffff'}})
        check('the unit\'s record is stale for --resume (its engine code is not the pin\'s): cut again, never resumed',
              camp.stale(run, name) and not camp.stale(run, 'cutouts.%s.41MHz' % cell))
        camp.after(u)
        r = camp.refusal()
        rec = record.load_record(out, 'photometry', name)
        rd = CP.readiness(out, e, cell, 'sb', TIDENT)
        check('a unit that ran another version of a pinned engine is recorded refused CP-PIN (its done outcome kept in its '
              'notes), stops the campaign (CP-PIN on the next unit), and holds its cell as incomplete - not a gap',
              r and 'CP-PIN' in r and name in r and u.status == 'refused' and rec['status'] == 'refused'
              and rec['reason'].startswith('CP-PIN: %s ran another engine version' % name) and 'done' in rec['notes'][-1]
              and CP.gap_kind(rec['status'], rec['reason']) is None and rd[0] == 'held' and 'CP-PIN' in rd[1], (r, rec, rd))
        u2 = argparse.Namespace(name='cutouts.%s.41MHz' % cell, run=run, status='done', reason='',
                                meta={'legacy_code': {files[2]: 'mixed:aaaaaaaaaaaa,bbbbbbbbbbbb'}})
        camp.after(u2)
        check('... a unit whose children ran a module in two versions (\'mixed:\') too, though the campaign stopped already',
              record.load_record(out, 'photometry', u2.name)['status'] == 'refused' and u2.reason.startswith('CP-PIN'),
              u2.reason)
        pin = CP.CodePin(files[:2])
        check('a module a unit ran in two versions (\'mixed:\') is a pin break',
              pin.absorb({files[2]: 'mixed:aaaaaaaaaaaa,bbbbbbbbbbbb'}) != [])
        e = raises(CP.CodePin, [os.path.join(code, 'absent.py')])
        check('CP-PIN: an engine file unreadable at the start refuses the campaign', e is not None and 'CP-PIN' in e, e)
        K.code_sha12(files[0])
        check('steps_common.legacy_code_changed and the pin share one fingerprint (code_sha12)',
              K.legacy_code_changed({'meta': {'legacy_code': {files[0]: K.code_sha12(files[0])}}}) == ''
              and K.legacy_code_changed({'meta': {'legacy_code': {files[0]: 'x'}}}) == 'engine_a.py')
    finally:
        P.undo()


def test_io_guard(d):
    ops = os.path.join(d, 'ops')
    os.makedirs(ops)
    P = Patch()
    try:
        P.set(CFG.paths, 'OPS_ROOT', ops)
        stop = os.path.join(ops, 'sublst_stack_STOP_%s' % K.hostname())
        alert = os.path.join(ops, 'io_guard_ALERT')
        check('I/O guard down: no stop file, no alert', CP.io_guard_raised() is None)
        open(stop, 'w').write('manual stop for maintenance\n')
        check('a stop line not starting with io_guard does not raise it', CP.io_guard_raised() is None)
        open(stop, 'w').write('manual\nio_guard 2026-11-23T10:00Z slow\n')
        check('the host\'s stop file with an io_guard line raises it', CP.io_guard_raised() == stop)
        os.remove(stop)
        open(alert, 'w').close()
        check('the alert file raises it', CP.io_guard_raised() == alert)
        slept = []

        def sleep(s):
            slept.append(s)
            if len(slept) == 2:
                os.remove(alert)
        CP.io_guard_wait(sleep)
        check('io_guard_wait polls every IO_GUARD_POLL_S until it is lowered (2 polls)', slept == [60, 60], slept)
        open(alert, 'w').close()
        slept.clear()
        camp = CP.Campaign('gen6', CP.Enumeration(), None, space_fn=lambda: (True, 'ok', {}), sleep=sleep)
        r = camp.refusal()
        check('a campaign unit waits for the guard, then runs (a yield, no refusal)', r is None and len(slept) == 2, (r, slept))
        open(alert, 'w').close()
        slept.clear()
        camp.stopped = 'CP-PIN: synthetic'
        r = camp.refusal()
        check('a stopped campaign refuses at once, before the I/O guard (no wait for a raised guard)',
              r and r.startswith('CP-PIN: synthetic') and slept == [], (r, slept))
        if os.path.exists(alert):
            os.remove(alert)
    finally:
        P.undo()


# ---------------------------------------------------------------------------------------------------------------------

def test_readiness(d):
    out = os.path.join(d, 'out_ready')
    X, Y, Z = '00h_2025-01-05', '01h_2025-01-06', '02h_2025-01-07'
    e = CP.Enumeration()
    e.units = [(c, b) for c in (X, Y, Z) for b in ['41MHz', '55MHz'] + WB]
    e.cells = [X, Y, Z]
    for b in ('41MHz', '55MHz'):
        write_rec(out, 'cutouts.%s.%s' % (X, b), 'done')
    write_rec(out, 'cutouts.%s.Red' % X, 'done')
    write_rec(out, 'cutouts.%s.Green' % X, 'no-input', 'C-EXCLUDED: excluded')
    write_rec(out, 'cutouts.%s.Blue' % X, 'no-input', 'C-NOCUT: none', {'engine_outcome': {'V': '0', 'I': 'refused: night'}})
    write_rec(out, 'cutouts.%s.41MHz' % Y, 'done')
    write_rec(out, 'cutouts.%s.55MHz' % Y, 'refused', 'C-GENCHECK: check (b) failed')
    for b in WB:
        write_rec(out, 'cutouts.%s.%s' % (Y, b), 'no-input', 'C-EXCLUDED: %s excluded under the deep rule' % b)
    write_rec(out, 'cutouts.%s.41MHz' % Z, 'no-input', 'C-NOCUT: none', {'engine_outcome': {'V': 'None', 'I': 'None'}})
    write_rec(out, 'cutouts.%s.Red' % Z, 'no-input', 'C-NOCUT: x', {'engine_outcome': {'V': 'None', 'I': '0'}})
    write_rec(out, 'cutouts.%s.Green' % Z, 'no-input', 'C-NOCUBE: no cube')
    write_rec(out, 'cutouts.%s.Blue' % Z, 'error', 'RuntimeError: crash')
    R = {(c, k): CP.readiness(out, e, c, k, TIDENT) for c in (X, Y, Z) for k in ('sb', 'wb')}
    check('ready: every unit done, C-EXCLUDED, or C-NOCUT whose counts promise no file (sb of %s; wb with done, excluded, '
          'night-rule)' % X, R[(X, 'sb')][0] == 'ready' and R[(X, 'wb')][0] == 'ready', R)
    check('held: a refused unit (the generation check) holds that cell\'s subbands only', R[(Y, 'sb')][0] == 'held'
          and 'C-GENCHECK' in R[(Y, 'sb')][1], R[(Y, 'sb')])
    check('\'no wideband\': every wideband group excluded - an outcome, not held, not ready', R[(Y, 'wb')][0] == 'no wideband',
          R[(Y, 'wb')])
    check('held: a missing record (55 MHz of %s)' % Z, R[(Z, 'sb')][0] == 'held' and 'no record' in R[(Z, 'sb')][1], R[(Z, 'sb')])
    zw = R[(Z, 'wb')][1]
    check('held: an error and a wideband C-NOCUT whose count \'None\' (group not built) promises no file count; a C-NOCUBE '
          '(a no-input gap) holds nothing', R[(Z, 'wb')][0] == 'held' and 'C-NOCUBE' not in zw and 'error' in zw
          and 'Red' in zw, zw)
    # review of PR 154, finding 1: a record of another run under the same --out is not this campaign's
    V = '04h_2025-01-09'
    ev = CP.Enumeration()
    ev.units = [(V, b) for b in ('41MHz', '55MHz')]
    ev.cells = [V]
    write_rec(out, 'cutouts.%s.41MHz' % V, 'done', meta={'cutout_root': CFG.photometry.CUTOUT_ROOTS['snapshot']}, ident=None)  # gen5 --in-place
    write_rec(out, 'cutouts.%s.55MHz' % V, 'done', ident=('gen6', '/elsewhere/photometry_gen6'))
    sv = [CP.unit_state(out, V, b, TIDENT) for b in ('41MHz', '55MHz')]
    rv = CP.readiness(out, ev, V, 'sb', TIDENT)
    check('a done record of another run under the same --out (a gen5 --in-place record; a gen6 record of another cutout root) '
          'is not done: held (incomplete), naming it, and it holds its cell\'s stage 1',
          [x[0] for x in sv] == ['held', 'held'] and all('a record of another run' in x[1] for x in sv)
          and CFG.photometry.CUTOUT_ROOTS['snapshot'] in sv[0][1] and rv[0] == 'held', (sv, rv))
    write_rec(out, 'cutouts.%s.41MHz' % V, 'refused', 'C-FLUXSCALE: %s 41MHz: V: no FSAPPLD - the engine not started' % V,
              meta={'cutout_root': CFG.photometry.CUTOUT_ROOTS['snapshot']}, ident=None)
    write_rec(out, 'cutouts.%s.55MHz' % V, 'no-input', 'C-EXCLUDED: x', ident=None)
    sv = [CP.unit_state(out, V, b, TIDENT)[0] for b in ('41MHz', '55MHz')]
    check('... another run\'s refused gap or exclusion is no gap or exclusion of this campaign (held)', sv == ['held', 'held'], sv)
    # gaps (R-GREGG-20261027-...-LEVEL-REFUSALS-STAY-GAPS-...): a gate that refused the INPUTS
    W = '03h_2025-01-08'
    eg = CP.Enumeration()
    eg.units = [(W, b) for b in ['41MHz', '55MHz'] + WB]
    eg.cells = [W]
    write_rec(out, 'cutouts.%s.41MHz' % W, 'refused', 'C-FLUXSCALE: %s 41MHz: V: no FSAPPLD - the engine not started' % W)
    write_rec(out, 'cutouts.%s.55MHz' % W, 'done')
    write_rec(out, 'cutouts.%s.Red' % W, 'done')
    write_rec(out, 'cutouts.%s.Green' % W, 'no-input', 'C-NOCUBE: no consolidated flow cube of %s Green (V)' % W)
    write_rec(out, 'cutouts.%s.Blue' % W, 'refused', 'C-FLUXSCALE: the member cubes of %s Blue fail the gate or changed since '
              'the check before the engine - nothing placed (staging kept: /s): x' % W)
    sts = {b: CP.unit_state(out, W, b, TIDENT)[0] for b in ['41MHz', 'Green', 'Blue']}
    check('gap classes: C-FLUXSCALE before the engine = a refused gap; C-NOCUBE = a no-input gap; C-FLUXSCALE after the cut '
          '(nothing placed) = incomplete', sts == {'41MHz': 'gap', 'Green': 'gap-noinput', 'Blue': 'held'}, sts)
    rg = (CP.readiness(out, eg, W, 'sb', TIDENT), CP.readiness(out, eg, W, 'wb', TIDENT))
    check('readiness: a refused gap holds its cell\'s kind as a gap (stage 1 not run); a non-gap refusal holds it as '
          'incomplete', rg[0][0] == 'gap' and 'C-FLUXSCALE' in rg[0][1] and rg[1][0] == 'held' and 'Blue' in rg[1][1], rg)
    write_rec(out, 'cutouts.%s.Blue' % W, 'done')
    check('readiness: a no-input gap (C-NOCUBE) does not hold - the kind is measured from the bands with cutouts',
          CP.readiness(out, eg, W, 'wb', TIDENT)[0] == 'ready', CP.readiness(out, eg, W, 'wb', TIDENT))
    for b in ('Red', 'Blue'):
        write_rec(out, 'cutouts.%s.%s' % (W, b), 'no-input', 'C-EXCLUDED: x')
    check('readiness: every wideband excluded or a no-input gap -> \'no wideband\'',
          CP.readiness(out, eg, W, 'wb', TIDENT)[0] == 'no wideband', CP.readiness(out, eg, W, 'wb', TIDENT))
    rf = 'C-FLUXSCALE: %s 41MHz: V: no FSAPPLD - the engine not started' % W
    P = Patch()
    try:
        P.set(CFG.photometry_campaign, 'GAP_REFUSED', {})
        check('the gap classes are config (photometry_campaign.GAP_REFUSED, ruled): without them the C-FLUXSCALE unit is '
              'incomplete', CP.gap_kind('refused', rf) is None and CP.unit_state(out, W, '41MHz', TIDENT)[0] == 'held')
    finally:
        P.undo()
    check('gap_kind: only the status the class names (a no-input C-FLUXSCALE, a refused C-NOCUBE are no gaps)',
          CP.gap_kind('refused', rf) == 'refused' and CP.gap_kind('no-input', rf) is None
          and CP.gap_kind('refused', 'C-NOCUBE: x') is None and CP.gap_kind('no-input', 'C-NOCUBE: x') == 'no-input')
    e.held = [dict(cell=X, subband='64MHz', reason='only the Stokes V cube')]
    check('held: a subband held at enumeration holds the cell\'s subbands, not its widebands',
          CP.readiness(out, e, X, 'sb', TIDENT)[0] == 'held' and CP.readiness(out, e, X, 'wb', TIDENT)[0] == 'ready')
    e.held = []
    e2 = CP.Enumeration()
    e2.units = [(X, b) for b in ('41MHz', '55MHz')]
    check('\'empty\': no unit of the kind in the cell', CP.readiness(out, e2, X, 'wb', TIDENT)[0] == 'empty')
    e3 = CP.Enumeration()
    e3.units = [(Z, '41MHz')]
    check('\'no subband\': the same rule for subbands (41 MHz of %s, night rule on both Stokes)' % Z,
          CP.readiness(out, e3, Z, 'sb', TIDENT)[0] == 'no subband')
    # stage 1 per cell
    calls = []
    P = Patch()
    try:
        def fake_measure(run, args, cell, bands, files, root, stage1, man, targets=(), kind_manifest=False, gate=None):
            calls.append((cell, tuple(bands), man, kind_manifest, gate))
            return []
        P.set(SS, 'measure_cell', fake_measure)
        run = run_of(out)
        camp = CP.Campaign('gen6', e, None, space_fn=lambda: (True, 'ok', {}), sleep=lambda s: None)
        root, st1 = TIDENT[1], os.path.join(out, 'stage1_gen6')
        for c in (X, Y, Z):
            for k in ('sb', 'wb'):
                CP.stage1_cell(run, argparse.Namespace(), camp, c, k, root, st1)
        got = {(c, b, m, km) for c, b, m, km, g in calls}
        check('ready kinds measured, into the kind\'s manifest (subbands forced_manifest.csv, widebands forced_manifest_wb.csv) '
              'with the campaign gate; nothing measured for a held or a no-wideband kind',
              got == {(X, tuple(SB), 'forced_manifest.csv', True), (X, tuple(WB), 'forced_manifest_wb.csv', True)}
              and all(g == camp.gate for *_, g in calls), calls)
        ry = record.load_record(out, 'photometry', SS.unit_name(Y, SB))
        rw = record.load_record(out, 'photometry', SS.unit_name(Y, WB))
        rz = record.load_record(out, 'photometry', SS.unit_name(Z, WB))
        check('SM-HELD: the held kind\'s stage-1 record refuses and names the blocking unit', ry['status'] == 'refused'
              and ry['reason'].startswith('SM-HELD') and 'cutouts.%s.55MHz' % Y in ry['reason'], ry['reason'])
        check('SM-NOBAND: the no-wideband record is no input with the outcome \'no wideband\' (not done, not failed)',
              rw['status'] == 'no-input' and rw['meta']['outcome'] == 'no wideband' and rw['reason'].startswith('SM-NOBAND'),
              (rw['status'], rw['meta'].get('outcome'), rw['reason']))
        check('a held cell does not hold its neighbor (%s measured; %s and %s held)' % (X, Y, Z),
              rz['status'] == 'refused' and len(calls) == 2, (rz['status'], calls))
        camp_g = CP.Campaign('gen6', eg, None, space_fn=lambda: (True, 'ok', {}), sleep=lambda s: None)
        CP.stage1_cell(run, argparse.Namespace(), camp_g, W, 'sb', root, st1)
        rgw = record.load_record(out, 'photometry', SS.unit_name(W, SB))
        check('SM-HELD for a gap: the subband stage 1 of %s is not run, its record refused naming the gap' % W,
              len(calls) == 2 and rgw['status'] == 'refused' and rgw['reason'].startswith('SM-HELD') and 'gaps' in rgw['reason']
              and 'C-FLUXSCALE' in rgw['reason'], rgw['reason'])
    finally:
        P.undo()
    # run_tier: the campaign's cutouts never stop its measure stage
    P = Patch()
    try:
        seen = []
        refused = argparse.Namespace(status='refused')
        P.set(SS, 'STAGE_FUNCS', {'cutouts': lambda r, a: [refused], 'measure': lambda r, a: seen.append('measure') or [],
                                  'evaluate': lambda r, a: []})
        P.set(CP, 'start', lambda run, args: None)
        run = run_of(os.path.join(d, 'out_tier'))
        SS.run_tier(run, campaign_args(scopes=None, cells=None), ['cutouts', 'measure'])
        SS.run_tier(run, argparse.Namespace(scopes=None, cells=None, in_place=False), ['cutouts', 'measure'])
        check('run_tier: with --campaign a refused cutouts unit does not stop the measure stage (readiness holds its cell); '
              'without, it does (stop_after)', seen == ['measure'], seen)
    finally:
        P.undo()


def test_wiring(W, d):
    """steps_cutouts under --campaign: the campaign's unit list, its gate first in each unit, its pin after each unit."""
    from exopipe.phase3.core import exclusions
    from exopipe.phase3.lib.photometry import steps_cutouts as SC
    out = os.path.join(d, 'out_wire')
    e = W.enum()
    P = Patch()
    try:
        P.set(CP, 'io_guard_raised', lambda: None)
        args = campaign_args(photometry_campaign=CP.Campaign('gen6', e, None, space_fn=lambda: (True, 'ok', {})))
        check('snapshot_items: the started campaign\'s own units, subbands before widebands within a cell',
              SC.snapshot_items(args) == sorted(e.units) and SC.snapshot_items(args)[-1][1] in WB
              and all(not (a[0] == b[0] and a[1] in WB and b[1] not in WB)
                      for a, b in zip(SC.snapshot_items(args), SC.snapshot_items(args)[1:])), SC.snapshot_items(args))
        # the real unit: the gate is its first act (an empty cube list would otherwise end it C-NOCUBE)
        P.set(K, 'sample_paths', lambda *a, **k: {'HWJ': '/x/HWJ.csv'})
        P.set(SC, 'rescale_identity', lambda cubes: ({}, ''))
        P.set(SC, 'listing_sha1', lambda *a, **k: 'x')
        P.set(SC, 'cutout_admitted', lambda *a, **k: [])
        P.set(exclusions, 'for_tier', lambda *a, **k: W.ex)
        run = run_of(out)
        A = '00h_2025-01-05'
        nocube = {'V': [], 'I': []}
        low = CP.Campaign('gen6', e, None, space_fn=lambda: (False, 'quota low', {}))
        SC.snapshot_unit(run, campaign_args(photometry_campaign=low), A, '41MHz', '/r', [], False, ['HWJ'], nocube)
        r1 = record.load_record(out, 'photometry', 'cutouts.%s.41MHz' % A)
        ok_camp = CP.Campaign('gen6', e, None, space_fn=lambda: (True, 'ok', {}))
        SC.snapshot_unit(run, campaign_args(photometry_campaign=ok_camp), A, '27MHz', '/r', [], False, ['HWJ'], nocube)
        r2 = record.load_record(out, 'photometry', 'cutouts.%s.27MHz' % A)
        check('snapshot_unit under a campaign: the campaign gate runs first (CP-SPACE refuses before the cube check); a passing '
              'gate goes on (C-NOCUBE), the record names the campaign', r1['status'] == 'refused'
              and r1['reason'].startswith('CP-SPACE: quota low') and r2['status'] == 'no-input'
              and r2['reason'].startswith('C-NOCUBE') and r2['meta'].get('campaign') == 'gen6', (r1['reason'], r2['reason']))
        # --resume over a record of another run (finding 1): never resumed, the unit runs for this campaign
        write_rec(out, 'cutouts.%s.55MHz' % A, 'done', meta={'cutout_root': '/r', 'inputs_listing_sha1': 'x'}, ident=None)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            SC.snapshot_unit(run_of(out, resume=True), campaign_args(photometry_campaign=CP.Campaign(
                'gen6', e, None, space_fn=lambda: (True, 'ok', {}), root='/r')), A, '55MHz', '/r', [], False, ['HWJ'], nocube)
        r3 = record.load_record(out, 'photometry', 'cutouts.%s.55MHz' % A)
        check('--resume: a done record of another run (no campaign name) is not resumed - the unit runs for this campaign',
              'not skipped (its record is not this campaign' in buf.getvalue() and r3['reason'].startswith('C-NOCUBE')
              and r3['meta'].get('campaign') == 'gen6', (buf.getvalue()[-300:], r3['reason']))
        # the stage loop: the campaign's units in order, the pin after each unit
        code = os.path.join(d, 'code_wire')
        os.makedirs(code, exist_ok=True)
        eng = os.path.join(code, 'engine.py')
        open(eng, 'w').write('x = 1\n')
        camp = CP.Campaign('gen6', e, CP.CodePin([eng]), space_fn=lambda: (True, 'ok', {}))
        seen = []

        def stub_unit(run_, args_, cell, band, *a):
            with run_.unit('cutouts.%s.%s' % (cell, band)) as u:
                args_.photometry_campaign.gate(u)
                seen.append((cell, band))
                u.meta['legacy_code'] = {eng: 'f' * 12 if len(seen) == 2 else K.code_sha12(eng)}
            return run_.units[-1:]
        P.set(CP, 'start', lambda run_, args_: setattr(args_, 'photometry_campaign', camp) or camp)
        P.set(SC, 'gate_write_roots', lambda *a, **k: None)
        P.set(SC, 'run_samples', lambda *a: ['HWJ'])
        P.set(SC, 'snapshot_cubes', lambda c, b: nocube)
        P.set(SC, 'EngineReads', lambda *a, **k: None)
        P.set(SC, 'snapshot_unit', stub_unit)
        run2 = run_of(os.path.join(d, 'out_wire2'))
        units = SC.stage_snapshot_cutouts(run2, campaign_args())
        st = [(u.name, u.status, (u.reason or '')[:6]) for u in units]
        r2nd = record.load_record(run2.out_root, 'photometry', st[1][0])
        check('the cutouts stage under --campaign: every enumerated unit in order; a unit that ran another engine version '
              'is recorded refused CP-PIN itself and stops the campaign - every later unit refuses before it starts',
              len(units) == len(e.units) and seen == sorted(e.units)[:2] and st[0][2] != 'CP-PIN'
              and all(x[1] == 'refused' and x[2] == 'CP-PIN' for x in st[1:]) and r2nd['status'] == 'refused'
              and camp.stopped.startswith('CP-PIN'), st)
        # the dry run in snapshot_unit's order: a unit without a cube is no input (C-NOCUBE) before any flux-scale verdict
        cube = {'V': ['/c/v.fits'], 'I': ['/c/i.fits']}
        P.set(SC, 'snapshot_cubes', lambda c, b: nocube if b == WB[0] else cube)
        P.set(SC, 'fluxscale_dry', lambda cell, band, ex: 'would refuse C-FLUXSCALE: stub')
        camp_d = CP.Campaign('gen6', e, None, space_fn=lambda: (True, 'ok', {}))
        P.set(CP, 'start', lambda run_, args_: setattr(args_, 'photometry_campaign', camp_d) or camp_d)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            dry_units = SC.stage_snapshot_cutouts(run_of(os.path.join(d, 'out_wire3'), dry_run=True), campaign_args())
        txt = buf.getvalue()
        nwb = sum(1 for _, b in e.units if b == WB[0])
        lines = [x for x in txt.splitlines() if ' unit cutouts.' in x]
        check('the dry run: a unit without a cube is listed no input C-NOCUBE, not by its flux-scale verdict (%d of %d units); '
              'the summary counts them apart' % (nwb, len(e.units)), dry_units == [] and len(lines) == len(e.units)
              and all(('no input C-NOCUBE' in x) == ('.%s:' % WB[0] in x) for x in lines)
              and ('%d of %d unit(s) would refuse; %d without a cube (C-NOCUBE)' % (len(e.units) - nwb, len(e.units), nwb)) in txt,
              txt[-700:])
    finally:
        P.undo()


def test_measure_cell(d):
    out = os.path.join(d, 'out_mc')
    root = os.path.join(d, 'root_mc')
    cell = '00h_2025-01-05'
    st1 = os.path.join(out, 'stage1_gen6')
    os.makedirs(os.path.join(st1, cell))
    files = {}
    for b in WB:
        p = os.path.join(root, 'HWJ', b, 'T0', 'T0_00h_20250105T010203_I.fits')
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, 'w').close()
        files[b] = [p]
    man = os.path.join(st1, cell, 'forced_manifest.csv')
    P = Patch()
    try:
        def sentinel(*a, **k):
            raise Refusal('SENTINEL: past SM-SUBSET')
        P.set(K, 'sample_paths', sentinel)
        run = run_of(out)

        def go(gate=None, kind=True):
            SS.measure_cell(run, argparse.Namespace(), cell, list(WB), files, root, st1, 'forced_manifest.csv',
                            kind_manifest=kind, gate=gate)
            return run.units[-1]
        open(man, 'w').write('file,status,ring_beams,n_ctl\n%s,ok,4.00,4\n' % files['Red'][0].replace('_I', '_V'))
        u = go()
        check('a kind manifest holding only the kind\'s bands is rewritten (no SM-SUBSET)', 'SENTINEL' in u.reason, u.reason)
        u = go(kind=False)
        check('... the same manifest name without kind_manifest: SM-SUBSET as before', u.reason.startswith('SM-SUBSET'),
              u.reason)
        open(man, 'a').write('%s,ok,4.00,4\n' % os.path.join(root, 'HWJ', '55MHz', 'T0', 'T0_00h_20250105T010203_V.fits'))
        u = go()
        check('SM-SUBSET: a kind manifest that holds rows of another band (55 MHz) is not overwritten by the widebands',
              u.reason.startswith('SM-SUBSET'), u.reason)

        for what, text in (('another header, its rows of the kind\'s band', 'file,status\n%s,ok\n' % files['Red'][0]),
                           ('a row naming no snapshot band', 'file,status,ring_beams,n_ctl\n/x/HWJ/T0/T0_I.fits,ok,4.00,4\n'),
                           ('bytes that are no text', None)):
            if text is None:
                open(man, 'wb').write(b'\xff\xfe\x00garbage\x80\n')
            else:
                open(man, 'w').write(text)
            u = go()
            check('SM-SUBSET: a kind manifest that cannot be parsed for its bands (%s) is never read as an empty band set - '
                  'not overwritten' % what, SS.manifest_bands(man) is None and u.reason.startswith('SM-SUBSET')
                  and 'cannot be parsed' in u.reason, (SS.manifest_bands(man), u.reason))
        open(man, 'w').write('file,status,ring_beams,n_ctl\n')
        check('a kind manifest with the header and no row holds no band (rewritten)', SS.manifest_bands(man) == set()
              and 'SENTINEL' in go().reason)

        def gate(u):
            u.refuse('CP-SPACE: synthetic')
        u = go(gate=gate)
        check('the campaign gate runs first in the stage-1 unit: its refusal is the unit\'s', u.status == 'refused'
              and u.reason == 'CP-SPACE: synthetic', (u.status, u.reason))
    finally:
        P.undo()


# ---------------------------------------------------------------------------------------------------------------------

def done_rec(out, root, cell, band, files, counts, drop=(), gen5=False, no_fs=False, campaign='gen6', legacy=None):
    """A done cutouts record (the checks a placed unit records), its manifest listing ``files`` (relative to root); a subband
    also its flux-scale record (``no_fs``: not recorded); ``gen5``: a unit of the generation-5 comparison sample;
    ``campaign``: the campaign it names (None: none); ``legacy``: its children's code fingerprint."""
    from exopipe.phase3.lib.photometry import tests_cutouts as TC
    cc = CFG.photometry_campaign
    kind = 'wb' if band in WB else 'sb'
    mp = os.path.join(root, '_manifest', '10s', '%s_%s.csv' % (cell, band))
    rows = []
    for rel in files:
        p = os.path.join(root, rel)
        rows.append(dict(file=rel, sha256='0' * 64, bytes=os.path.getsize(p), sample=rel.split('/')[0],
                         object=rel.split('/')[2], stokes=rel[-6]))
    TC.write_manifest(mp, rows)
    cut = [S for S in 'VI' if counts[S] not in ('0', 'None')]
    names = (list(cc.COMPLETE_CHECKS) + list(cc.COMPLETE_KIND_CHECKS[kind])
             + [n.format(stokes=S) for S in cut for n in cc.COMPLETE_STOKES_CHECKS[kind]]
             + [n.format(stokes=S) for S in cut for n in cc.COMPLETE_GEN5_CHECKS if gen5 and kind == 'sb'])
    tests = [dict(name=n, passed=True) for n in names if n not in drop]
    outputs = [dict(path=mp, **{'class': 'photometry_cutout_manifest'}, ctx={'root': root})]
    if kind == 'sb' and not no_fs:
        fp = os.path.join(root, '_fluxscale', band, '%s_%s_fluxscale.json' % (cell, band))
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        open(fp, 'w').write('{}')
        outputs.append(dict(path=fp, **{'class': 'photometry_fluxscale_record'}))
    meta = {'engine_outcome': counts, 'cutout_root': root, 'generation5_sample': gen5}
    if campaign:
        meta['campaign'] = campaign
    if legacy:
        meta['legacy_code'] = legacy
    write_rec(out, 'cutouts.%s.%s' % (cell, band), 'done', meta=meta, tests=tests, outputs=outputs, ident=None)


def test_check(d):
    out = os.path.join(d, 'out_chk')
    root = os.path.join(d, 'root_chk')
    cell = '00h_2025-01-05'
    rels = {}
    for b in ['41MHz', '55MHz'] + WB:
        for S in 'VI':
            rel = 'HWJ/%s/T0/T0_00h_20250105T010203_%s.fits' % (b, S)
            os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
            open(os.path.join(root, rel), 'w').write('x' * 10)
            rels.setdefault(b, []).append(rel)
    done_rec(out, root, cell, '41MHz', rels['41MHz'], {'V': '1', 'I': '1'})
    probs = CP.check_done(record.load_record(out, 'photometry', 'cutouts.%s.41MHz' % cell), '41MHz')
    check('check_done: a done record with its checks and files passes', probs == [], probs)
    done_rec(out, root, cell, '55MHz', rels['55MHz'][1:], {'V': '1', 'I': '1'})
    probs = CP.check_done(record.load_record(out, 'photometry', 'cutouts.%s.55MHz' % cell), '55MHz')
    check('check_done: a Stokes with a positive engine count and no file in the manifest is a problem (the count check '
          'again)', any('Stokes V: engine count 1, no file' in p for p in probs), probs)
    done_rec(out, root, cell, 'Red', rels['Red'], {'V': '1', 'I': '1'}, drop=('photometry_cutout_manifest.I1_identity_I',))
    probs = CP.check_done(record.load_record(out, 'photometry', 'cutouts.%s.Red' % cell), 'Red')
    check('check_done: a required per-Stokes check not recorded is a problem', any('I1_identity_I not recorded' in p
                                                                                 for p in probs), probs)
    done_rec(out, root, cell, 'Green', rels['Green'], {'V': '1', 'I': '1'})
    open(os.path.join(root, rels['Green'][0]), 'a').write('more')
    probs = CP.check_done(record.load_record(out, 'photometry', 'cutouts.%s.Green' % cell), 'Green')
    check('check_done: a listed file of another size is a problem', any('of another size' in p for p in probs), probs)
    done_rec(out, root, cell, 'Blue', rels['Blue'], {'V': '1', 'I': 'None'})
    probs = CP.check_done(record.load_record(out, 'photometry', 'cutouts.%s.Blue' % cell), 'Blue')
    check('check_done: a wideband count \'None\' (group not built) promises no file count: a problem',
          any('promises no file count' in p for p in probs), probs)
    rec = record.load_record(out, 'photometry', 'cutouts.%s.41MHz' % cell)
    rec['tests'].append(dict(name='photometry_cutout_manifest.GB_generation5_V', passed=False))
    check('check_done: a failed recorded test is a problem', any('failed tests' in p for p in CP.check_done(rec, '41MHz')))
    # the whole campaign: one good cell, stage 1 done and no wideband
    out2 = os.path.join(d, 'out_chk2')
    c1, c2 = '00h_2025-01-05', '01h_2025-01-06'
    e = CP.Enumeration()
    e.units = [(c1, b) for b in ['41MHz'] + WB] + [(c2, b) for b in ['41MHz'] + WB]
    e.cells = [c1, c2]

    def wr(*a, **k):                    # the records of this campaign: generation gen6, cutout root ``root``
        return write_rec(*a, ident=('gen6', root), **k)
    started = start_rec(out2, root, e=e)
    for b in ['41MHz', 'Red']:
        done_rec(out2, root, c1, b, rels[b], {'V': '1', 'I': '1'})
    for b in ('Green', 'Blue'):
        wr(out2, 'cutouts.%s.%s' % (c1, b), 'no-input', 'C-EXCLUDED: x')
    done_rec(out2, root, c2, '41MHz', rels['41MHz'], {'V': '1', 'I': '1'})
    wr(out2, 'cutouts.%s.Red' % c2, 'no-input', 'C-NOCUT: no file', meta={'engine_outcome': {'V': '0', 'I': '0'}})
    wr(out2, 'cutouts.%s.Green' % c2, 'no-input', 'C-EXCLUDED: x')
    wr(out2, 'cutouts.%s.Blue' % c2, 'no-input', 'C-NOCUT: night rule', meta={'engine_outcome': {
        'V': 'refused: below the night rule', 'I': 'refused: below the night rule'}})
    st1 = os.path.join(out2, 'stage1_gen6')
    for c, k, man in ((c1, 'sb', 'forced_manifest.csv'), (c1, 'wb', 'forced_manifest_wb.csv'),
                      (c2, 'sb', 'forced_manifest.csv')):
        os.makedirs(os.path.join(st1, c), exist_ok=True)
        open(os.path.join(st1, c, man), 'w').write('file,status,ring_beams,n_ctl\n/x/HWJ/Red/T0/T0_I.fits,ok,4.00,4\n')
        st1_rec(out2, root, c, k, wr)
    wr(out2, SS.unit_name(c2, WB), 'no-input', 'SM-NOBAND: no wideband', meta={'outcome': 'no wideband'})
    ok, counts, probs, gaps = CP.check(out2, e, started, st1)
    check('check: every unit terminal, stage 1 done per kind or \'no wideband\' -> complete; the units without a cutout '
          'counted by why (3 excluded, 1 night rule, 1 with a zero count)', ok and counts['done'] == 3
          and counts['none'] == 5 and counts.get('stage1_no_wideband') == 1 and counts.get('stage1_done') == 3
          and (counts['none_excluded'], counts['none_night_rule'], counts['none_no_cutout']) == (3, 1, 1), (counts, probs))
    # review of PR 154, finding 5: a re-cut after stage 1 (the same bytes, a new file time) - the stage 1 is stale
    cut = os.path.join(root, rels['41MHz'][0])
    st = os.stat(cut)
    os.utime(cut, ns=(st.st_atime_ns, st.st_mtime_ns + 10 ** 9))
    ok5, _, probs5, _ = CP.check(out2, e, started, st1)
    now = [p for b in CP.kind_bands('sb') for p in SS.cell_files(c1, root, None, b)]
    again = not K.run_resume_ok(run_of(out2, resume=True), SS.unit_name(c1, list(CP.kind_bands('sb'))), [],
                                {'cutout_listing_sha1': SS.listing_sha1(now)})
    check('a re-cut after stage 1: the stage 1 is incomplete (measured from other cutouts than the cell holds now) and its '
          'resume identity no longer matches, so readiness measures it again; the cutouts unit itself still passes',
          not ok5 and any(p.startswith('stage 1 %s sb: measured from other cutouts' % c1) for p in probs5)
          and not any(p.startswith('cutouts.') for p in probs5) and again, probs5)
    st1_rec(out2, root, c1, 'sb', wr)
    # review of PR 154, finding 1: a gen5 --in-place done record under the same --out is not this campaign's
    done_rec(out2, root, c1, 'Red', rels['Red'], {'V': '1', 'I': '1'}, campaign=None)
    ok1, _, probs1, _ = CP.check(out2, e, started, st1)
    check('check: a done record without this campaign\'s name (a gen5 --in-place record under the same --out) is incomplete',
          not ok1 and any(p.startswith('cutouts.%s.Red: a record of another run' % c1) for p in probs1), probs1)
    done_rec(out2, root, c1, 'Red', rels['Red'], {'V': '1', 'I': '1'})
    wr(out2, 'cutouts.%s.Green' % c1, 'refused', 'C-WRITE: short')
    ok, counts, probs, gaps = CP.check(out2, e, started, st1)
    check('check: a refused unit is incomplete (unlike gen6\'s GATE-REFUSED) and its cell\'s wideband stage 1 is held',
          not ok and counts['incomplete'] == 1 and any(p.startswith('cutouts.%s.Green' % c1) and 'C-WRITE' in p for p in probs)
          and any('stage 1 %s wb held' % c1 in p for p in probs), (counts, probs))
    wr(out2, 'cutouts.%s.Green' % c1, 'no-input', 'C-EXCLUDED: x')
    os.remove(os.path.join(st1, c1, 'forced_manifest.csv'))
    ok, _, probs, gaps = CP.check(out2, e, started, st1)
    check('check: a done stage-1 record whose manifest is gone is incomplete', not ok and any('no manifest' in p for p in probs),
          probs)
    open(os.path.join(st1, c1, 'forced_manifest.csv'), 'w').write('file,status,ring_beams,n_ctl\n/x/HWJ/41MHz/T0/T0_I.fits,'
                                                                  'REFUSED: x,4.00,4\n')
    ok, _, probs, gaps = CP.check(out2, e, started, st1)
    check('check: a stage-1 manifest with a REFUSED row fails MM2: incomplete', not ok and any('REFUSED' in p for p in probs),
          probs)
    open(os.path.join(st1, c1, 'forced_manifest.csv'), 'w').write('file,status,ring_beams,n_ctl\n/x/HWJ/41MHz/T0/T0_I.fits,ok,'
                                                                  '4.00,4\n')
    e.held = [dict(cell=c1, subband='64MHz', reason='only the Stokes V cube')]
    ok, _, probs, gaps = CP.check(out2, e, started, st1)
    check('check: a held (cell, subband) is incomplete', not ok and any('held at enumeration' in p for p in probs), probs)
    e.held = []
    P = Patch()
    try:
        P.set(CP, 'enumerate_units', lambda cells=None, *a, **k: e)
        P.set(CP, 'cube_cells', lambda r: [c1, c2])
        P.set(CP, 'engine_exclusions', lambda run, cells: {c: set() for c in cells})
        rc = CP.main(['check', '--out', out2])
        wr(out2, 'cutouts.%s.Red' % c1, 'error', 'RuntimeError: x')
        rc2 = CP.main(['check', '--out', out2, '--no-stage1'])
        rc3 = CP.main(['check', '--out', os.path.join(d, 'out_chk_none')])
        check('the check command exits 0 complete, 1 incomplete, 2 without a campaign start record under --out',
              (rc, rc2, rc3) == (0, 1, 2), (rc, rc2, rc3))
    finally:
        P.undo()
    st_uc = dict(scope=[c1], enumeration=dict(units=[[c1, '41MHz'], [c1, 'Red']]))
    now = [(c1, '41MHz'), (c1, '46MHz'), (c2, '41MHz')]
    ch = (CP.units_changed(st_uc, now), CP.units_changed(dict(st_uc, scope=None), now, cells=[c1]))
    check('units changed: compared over the cells both scopes cover (a cell outside the start record\'s scope is no change)',
          ch == (([(c1, '46MHz')], [(c1, 'Red')]), ([(c1, '46MHz')], [(c1, 'Red')])), ch)
    # the generation checks (#153) a done record must carry, by band kind
    out4 = os.path.join(d, 'out_chk4')
    cases = [('GP', '41MHz', dict(drop=('photometry_cutout_manifest.GP_cubes_unchanged',)), 'GP_cubes_unchanged not recorded'),
             ('GA', '41MHz', dict(drop=('photometry_cutout_manifest.GA_fluxscale_record_I',)), 'GA_fluxscale_record_I not'),
             ('GC', '41MHz', dict(drop=('photometry_cutout_manifest.GC_targets_V',)), 'GC_targets_V not recorded'),
             ('fs1', '41MHz', dict(drop=('photometry_fluxscale_record.fs1_record',)), 'fs1_record not recorded'),
             ('fs output', '41MHz', dict(no_fs=True), '0 photometry_fluxscale_record outputs'),
             ('GB', '41MHz', dict(gen5=True, drop=('photometry_cutout_manifest.GB_generation5_V',)), 'GB_generation5_V not'),
             ('GW', 'Red', dict(drop=('photometry_cutout_manifest.GW_weights_I',)), 'GW_weights_I not recorded')]
    for tag, b, kw, want in cases:
        done_rec(out4, root, cell, b, rels[b], {'V': '1', 'I': '1'}, **kw)
        probs = CP.check_done(record.load_record(out4, 'photometry', 'cutouts.%s.%s' % (cell, b)), b)
        check('check_done: %s missing from a done %s record is a problem' % (tag, 'wideband' if b in WB else 'subband'),
              any(want in p for p in probs), probs)
    done_rec(out4, root, cell, '41MHz', rels['41MHz'], {'V': '1', 'I': '1'}, gen5=True)
    done_rec(out4, root, cell, 'Red', rels['Red'], {'V': '1', 'I': '1'})
    probs = [CP.check_done(record.load_record(out4, 'photometry', 'cutouts.%s.%s' % (cell, b)), b) for b in ('41MHz', 'Red')]
    check('check_done: a subband of the generation-5 sample with GA/GB/GC, fs1 and its flux-scale record, a wideband with GW, '
          'pass (no flux-scale record asked of a wideband)', probs == [[], []], probs)
    # GAPS: refused by an input gate -> COMPLETE with N gaps (exit 0); an output or content refusal -> INCOMPLETE (exit 1)
    done_rec(out2, root, c1, 'Red', rels['Red'], {'V': '1', 'I': '1'})
    fsr = 'C-FLUXSCALE: %s Red: Blue member 64MHz V: FSAPPLD 1.02 off its record 0.98 - the engine not started' % c1
    wr(out2, 'cutouts.%s.Red' % c1, 'refused', fsr)
    ok, counts, probs, gaps = CP.check(out2, e, started, st1)
    check('check: a unit refused by the flux-scale gate before the engine is a GAP - complete, listed with its code and reason; '
          'its cell\'s wideband stage 1 is not run, a gap too', ok and counts['gap'] == 1 and probs == []
          and any(g.startswith('cutouts.%s.Red: refused C-FLUXSCALE' % c1) for g in gaps)
          and any(g.startswith('stage 1 %s wb not run' % c1) for g in gaps), (counts, probs, gaps))
    wr(out2, 'cutouts.%s.Green' % c2, 'no-input', 'C-NOCUBE: no consolidated flow cube of %s Green (V)' % c2)
    ok, counts, probs, gaps = CP.check(out2, e, started, st1)
    check('check: a C-NOCUBE unit is a gap that holds nothing (%s\'s widebands stay \'no wideband\')' % c2,
          ok and counts['gap'] == 2 and counts.get('stage1_no_wideband') == 1, (counts, probs, gaps))
    P = Patch()
    try:
        P.set(CP, 'enumerate_units', lambda cells=None, *a, **k: e)
        P.set(CP, 'cube_cells', lambda r: [c1, c2])
        P.set(CP, 'engine_exclusions', lambda run, cells: {c: set() for c in cells})
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = CP.main(['check', '--out', out2])
        txt = buf.getvalue()
        check('the check command: COMPLETE with N gaps, the units without a cutout COUNTED by why (not listed, not gaps), exit '
              '0, a GAP line per gap', rc == 0
              and 'COMPLETE with 3 gaps (2 excluded, 1 night-rule refused, 1 with no cutout; not gaps)' in txt
              and txt.count('\nGAP ') + txt.startswith('GAP ') == 3 and 'excluded (' not in txt, (rc, txt[-600:]))
        for code in ('C-GENCHECK: (b) failed - nothing placed', 'C-WRITE: short', 'C-COUNT: x', 'C-IDENTITY: x', 'C-PLACE: x',
                     'C-CONTENT: x', 'C-ENGINE: x', 'C-STALE: x', 'CP-SPACE: low - not started',
                     'C-FLUXSCALE: the member cubes of x fail the gate or changed - nothing placed (staging kept: s)'):
            wr(out2, 'cutouts.%s.Red' % c1, 'refused', code)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = CP.main(['check', '--out', out2])
            check('the check command: a unit refused %s is INCOMPLETE (exit 1), not a gap' % code.split(':')[0]
                  + (' (after the cut)' if 'member cubes' in code else ''),
                  rc == 1 and 'INCOMPLETE' in buf.getvalue() and 'PROBLEM cutouts.%s.Red' % c1 in buf.getvalue()
                  and '(2 excluded, 1 night-rule refused, 1 with no cutout; not gaps)' in buf.getvalue(),
                  (rc, buf.getvalue()[-400:]))
    finally:
        P.undo()


def test_check_readers(W, d):
    """Review of PR 154, finding 4: the check enumerates with BOTH exclusion readers, as the campaign does - a disagreement
    is held, incomplete; it probes only the cells with a cube directory and writes nothing under --out."""
    out = os.path.join(d, 'out_chk_readers')
    start_rec(out, TIDENT[1])
    probes = []
    P = Patch()
    try:
        P.set(CFG.paths, 'SNAPSHOT_CUBES_ROOT', W.root)
        P.set(CP, 'enumerate_units', lambda cells=None, engine_excluded=None, present=None: W.enumerate(
            cells, W.root, W.rec, W.ex, engine_excluded, W.complete, present))
        runs = []

        def reader(engine):
            def f(run, cells):
                probes.append(list(cells))
                runs.append(run.out_root)
                return {c: engine.get(c, set()) for c in cells}
            return f
        outs = []
        for engine in (W.engine, dict(W.engine, **{'00h_2025-01-05': {'23MHz'}})):
            P.set(CP, 'engine_exclusions', reader(engine))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = CP.main(['check', '--out', out, '--no-stage1', '--max-problems', '1000'])
            outs.append((rc, buf.getvalue()))
        bad = 'PROBLEM held at enumeration: 00h_2025-01-05 27MHz (PHOT-EXCL-READERS'
        check('check: the subband engine\'s exclusion reader is read (27 MHz: it excludes, core/exclusions not) - the '
              'disagreement is held, INCOMPLETE; with the readers agreeing no such problem',
              outs[0][0] == 1 and bad in outs[0][1] and bad not in outs[1][1], [o[1][-500:] for o in outs])
        check('... the reader probes only the cells with a cube directory, its child log outside --out (removed after)',
              probes[0] == ['00h_2025-01-05', '01h_2025-01-06', '02h_2025-01-07'] and not runs[0].startswith(out)
              and not os.path.exists(runs[0]) and sorted(os.listdir(os.path.join(out, '_records', 'photometry')))
              == [CFG.photometry_campaign.RECORD_SUBDIR], (probes, runs))
    finally:
        P.undo()


def test_fluxscale_classes(W, d):
    """Review of PR 154, finding 8: only a VERDICT of the flux-scale gate on the cubes is a gap (C-FLUXSCALE); rescale
    records or a header that cannot be read refuse C-FLUXSCALE-READ - operational, retryable, incomplete."""
    from exopipe.phase3.lib.photometry import fluxscale as FS
    from exopipe.phase3.lib.photometry import steps_cutouts as SC

    class U:
        def __init__(self):
            self.meta = {}

        def refuse(self, why):
            raise Refusal(why)
    ex = argparse.Namespace(is_excluded=lambda *a: False)
    cell = '00h_2025-01-05'
    out = os.path.join(d, 'out_fsclass')
    got = {}
    P = Patch()
    try:
        def boom(exc):
            def f(*a, **k):
                raise exc
            return f
        for tag, fn in (('records', boom(FS.RecordsError('no rescale record readable: /x/RESULTS.csv:3 (synthetic)'))),
                        ('file', boom(OSError(5, 'Input/output error'))),
                        ('header', lambda *a, **k: (False, ['V: header unreadable (OSError: synthetic)'], {'V': {}, 'I': {}})),
                        ('verdict', lambda *a, **k: (False, ['V: FSAPPLD 1.000000 != recorded 1.500000 (rescale-results)'],
                                                     {'V': {}, 'I': {}}))):
            P.set(FS, 'check', fn)
            r = raises(SC.fluxscale_gate, U(), cell, '41MHz', ex)
            write_rec(out, 'cutouts.%s.41MHz' % cell, 'refused', r)
            got[tag] = (r, SC.fluxscale_dry(cell, '41MHz', ex), CP.gap_kind('refused', r),
                        CP.unit_state(out, cell, '41MHz', TIDENT)[0])
            P.undo()
        check('records unreadable (RecordsError, OSError) or a header unreadable: C-FLUXSCALE-READ, operational and '
              'retryable - not a gap, the unit incomplete (held); the dry run says the same',
              all(got[t][0].startswith('C-FLUXSCALE-READ: ') and 'not a gap' in got[t][0] and got[t][2] is None
                  and got[t][3] == 'held' and got[t][1].startswith('would refuse C-FLUXSCALE-READ: ')
                  for t in ('records', 'file', 'header')), got)
        check('a verdict on the cubes (FSAPPLD off its record): C-FLUXSCALE, a gap', got['verdict'][0].startswith(
            'C-FLUXSCALE: ') and got['verdict'][2] == 'refused' and got['verdict'][3] == 'gap'
            and got['verdict'][1].startswith('would refuse C-FLUXSCALE: '), got['verdict'])
    finally:
        P.undo()


def test_start(W, d):
    out = os.path.join(d, 'out_start')
    code = os.path.join(d, 'code_start')
    os.makedirs(code)
    pins = []
    for n in ('a.py', 'b.py'):
        pins.append(os.path.join(code, n))
        open(pins[-1], 'w').write('# %s\n' % n)
    P = Patch()
    try:
        probes = []
        P.set(CP, 'engine_exclusions', lambda run, cells: probes.append(list(cells)) or {c: W.engine.get(c, set())
                                                                                       for c in cells})
        P.set(CP, 'enumerate_units', lambda cells=None, engine_excluded=None, present=None: W.enumerate(
            cells, W.root, W.rec, W.ex, engine_excluded, W.complete, present))
        P.set(CP, 'pin_files', lambda: pins)
        P.set(CP, 'space', lambda lfs=None: (True, 'synthetic ok', {'fs_free_bytes': 1}))
        P.set(CFG.paths, 'SNAPSHOT_CUBES_ROOT', W.root)
        args = campaign_args()
        e = raises(CP.start, run_of(out), args)
        check('C-ROOT: in the refactor window (paths.PRODUCTION_WRITES False) the campaign refuses its production cutout root',
              e is not None and 'C-ROOT' in e, e)
        Q = Patch()
        try:
            Q.set(CP, 'enumerate_units', lambda cells=None, engine_excluded=None, present=None: W.enum(engine=False))
            run_d = run_of(os.path.join(d, 'out_start_dry'), dry_run=True)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                camp_d = CP.start(run_d, campaign_args())
            check('C-ROOT in a dry run: reported as the refusal the run would raise (exit 2), the campaign still listed, '
                  'nothing written', camp_d is not None and any('C-ROOT' in w for w in run_d.photometry_would_refuse)
                  and '[campaign] generation gen6' in buf.getvalue() and camp_d.record_path is None
                  and not os.path.exists(os.path.join(d, 'out_start_dry', '_records', 'photometry', '_campaign')),
                  (getattr(run_d, 'photometry_would_refuse', None),
                                                                              buf.getvalue()[-300:]))
        finally:
            Q.undo()
        P.set(CFG.photometry_campaign, 'CUTOUT_ROOT_FMT', os.path.join(d, 'photometry_{generation}'))
        run = run_of(out)
        camp = CP.start(run, args)
        rec = json.load(open(camp.record_path))
        check('start: the start record holds the generation, both roots, the unit list, the pin and the space read',
              rec['generation'] == 'gen6' and rec['cutout_root'] == os.path.join(d, 'photometry_gen6')
              and rec['stage1'] == os.path.join(out, 'stage1_gen6') and len(rec['enumeration']['units']) == 5
              and sorted(rec['pin']) == sorted(pins) and rec['space'] == {'fs_free_bytes': 1}, rec)
        check('start: once per invocation (the measure stage takes the same campaign)', CP.start(run, args) is camp)
        check('the start record is found by the check (latest_record)', CP.latest_record(out, 'gen6')['run_id'] == run.run_id)
        # review of PR 154, finding 2: the pin persists in the start record; a later invocation continues it or refuses
        extra = os.path.join(code, 'c.py')
        open(extra, 'w').write('# c\n')
        h_a, h_c = K.code_sha12(pins[0]), K.code_sha12(extra)
        camp.after(argparse.Namespace(name='cutouts.x.41MHz', run=run, status='done', reason='',
                                      meta={'legacy_code': {extra: h_c}}))
        rec1 = json.load(open(camp.record_path))
        check('a child module first seen joins the pin AND the start record (a later invocation and the check see it)',
              rec1['pin'].get(extra) == h_c and len(rec1['pin']) == 3, rec1['pin'])

        def day(n, **kw):
            r = run_of(out, dry_run=kw.pop('dry_run', False))
            r.run_id = 'photometry_29991231T00000%dZ_h_%d' % (n, n)
            return r, campaign_args(**kw)
        run2, a2 = day(2)
        camp2 = CP.start(run2, a2)
        rec2 = json.load(open(camp2.record_path))
        check('day 2, the same engine code: the invocation (a --resume among them) continues the campaign and its pin (the '
              'absorbed module too)', rec2['continues'] == run.run_id and rec2['pin'] == rec1['pin']
              and camp2.pin.files == rec1['pin'], (rec2['continues'], rec2['pin']))
        open(extra, 'a').write('# day 3\n')
        e3a = raises(CP.start, *day(3))
        open(extra, 'w').write('# c\n')
        open(pins[0], 'a').write('# day 3\n')
        e3b = raises(CP.start, *day(3))
        check('day 3, an engine file changed since the pin (an absorbed child module; a pinned engine): CP-PIN, naming the '
              'file and --new-campaign; no start record written', e3a and e3a.startswith('CP-PIN') and 'c.py' in e3a
              and e3b and e3b.startswith('CP-PIN') and 'a.py' in e3b and '--new-campaign' in e3b
              and CP.latest_record(out, 'gen6')['run_id'] == run2.run_id, (e3a, e3b))
        rd, ad = day(3, dry_run=True)
        with contextlib.redirect_stdout(io.StringIO()):
            CP.start(rd, ad)
        check('... a dry run reports it as the refusal the run would raise', any(
            'CP-PIN' in w for w in getattr(rd, 'photometry_would_refuse', [])), getattr(rd, 'photometry_would_refuse', None))
        Q = Patch()
        try:
            Q.set(CFG.photometry_campaign, 'CUTOUT_ROOT_FMT', os.path.join(d, 'other', 'photometry_{generation}'))
            e3c = raises(CP.start, *day(3))
        finally:
            Q.undo()
        check('... another cutout root under the same --out: CP-GEN, naming --new-campaign', e3c and e3c.startswith('CP-GEN')
              and '--new-campaign' in e3c, e3c)
        cell = '00h_2025-01-05'
        rootc = os.path.join(d, 'photometry_gen6')
        rel = 'HWJ/41MHz/T0/T0_00h_20250105T010203_I.fits'
        os.makedirs(os.path.dirname(os.path.join(rootc, rel)), exist_ok=True)
        open(os.path.join(rootc, rel), 'w').write('x' * 10)
        done_rec(out, rootc, cell, '41MHz', [rel], {'V': '0', 'I': '1'}, legacy={pins[0]: h_a, extra: h_c})
        e1 = CP.Enumeration()
        e1.units = [(cell, '41MHz')]
        ok2 = CP.check(out, e1, CP.latest_record(out, 'gen6'))[0]
        run4, a4 = day(4, new_campaign=True)
        camp4 = CP.start(run4, a4)
        rec4 = json.load(open(camp4.record_path))
        ok4, _, probs4, _ = CP.check(out, e1, CP.latest_record(out, 'gen6'))
        check('--new-campaign: starts a new campaign under the same --out with the engine code of now; its check calls the '
              'day-1 unit (done under the old pin) INCOMPLETE, as the old campaign\'s check called it done',
              ok2 and rec4['new_campaign_over'] == run2.run_id and rec4['continues'] is None
              and rec4['pin'][pins[0]] == K.code_sha12(pins[0]) != h_a and not ok4
              and any('engine code' in p and 'a.py' in p for p in probs4), (ok2, rec4.get('new_campaign_over'), probs4))
        run5, a5 = day(5)
        check('... and the next invocation continues the new campaign', CP.start(run5, a5).pin.files == rec4['pin'])
        # finding 6: --cells naming a cell without a cube directory holds that cell, the others run
        probes.clear()
        camp6 = CP.start(run_of(os.path.join(d, 'out_start_cells')), campaign_args(cells='00h_2025-01-05,05h_2025-01-09'))
        check('--cells naming a cell without a cube directory: the campaign starts, that cell held (no cube directory), '
              'the engine\'s reader probes only the cells with one', probes == [['00h_2025-01-05']]
              and any(h['cell'] == '05h_2025-01-09' and 'no cube directory' in h['reason'] for h in camp6.enum.held)
              and {c for c, _ in camp6.enum.units} == {'00h_2025-01-05'}, (probes, camp6.enum.held))
        P.set(CP, 'space', lambda lfs=None: (False, 'synthetic low', {}))
        out3 = os.path.join(d, 'out_start_low')
        e = raises(CP.start, run_of(out3), campaign_args())
        check('CP-SPACE at the start: the campaign does not start and writes no start record', e is not None
              and 'CP-SPACE' in e and CP.latest_record(out3, 'gen6') is None, e)
    finally:
        P.undo()


def main():
    base = os.environ.get('EXOPIPE3_TEST_SCRATCH') or None
    d = tempfile.mkdtemp(prefix='test_photometry_campaign_', dir=base)
    try:
        test_generation(d)
        P = Patch()
        served_gen5(P, d)       # the rest in the state the campaign ran in (served generation 5); undone below
        test_resume_hash()
        test_args()
        W = World(d)
        test_enumeration(W)
        test_space()
        test_pin(d)
        test_io_guard(d)
        test_readiness(d)
        test_wiring(W, d)
        test_measure_cell(d)
        test_check(d)
        test_check_readers(W, d)
        test_fluxscale_classes(W, d)
        test_start(W, d)
        P.undo()
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print('%d check(s) failed: %s' % (len(FAILS), FAILS) if FAILS else 'all checks passed', flush=True)
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
