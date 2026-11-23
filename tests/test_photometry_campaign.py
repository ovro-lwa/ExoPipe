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
    the pin after each unit;
  - measure_cell: the gate runs first in the unit; a kind manifest is rewritten unless it holds rows of another band
    (SM-SUBSET);
  - the complete-when check: a done record passes only with its required checks (by band kind: GP, GA + fs1 + the
    flux-scale record, GC, GB for the generation-5 sample, GW), its manifest's files present with their sizes and a file
    per Stokes with a positive engine count; stage 1 per kind done (MM1 + MM2) or 'no <kind>'; gaps (input-gate refusals,
    ruled) are listed and complete ("COMPLETE with N gaps", exit 0); output or content refusals, errors and missing records
    are incomplete (exit 1).
  - the resume hash: the step's CONFIG holds every section of config/steps/photometry.py (photometry_fluxscale and
    photometry_campaign among them); a value change in either changes the hash.
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
        self.saved.append((obj, key, getattr(obj, key)))
        setattr(obj, key, val)

    def undo(self):
        while self.saved:
            obj, key, val = self.saved.pop()
            setattr(obj, key, val)


def write_rec(out, name, status, reason='', meta=None, tests=(), outputs=()):
    record.atomic_write_json(record.record_path(out, 'photometry', name),
                             dict(unit=name, status=status, reason=reason, meta=meta or {}, tests=list(tests),
                                  outputs=list(outputs)))


def run_of(out, **kw):
    os.makedirs(out, exist_ok=True)
    return record.Run('photometry', ['test'], out, CFG, **kw)


# ---------------------------------------------------------------------------------------------------------------------

def test_generation(d):
    P = Patch()
    try:
        check('the generation written is photometry_campaign.GENERATION (gen6), the served one stays photometry.GENERATION '
              '(gen5)', CP.generation() == 'gen6' and CFG.photometry.GENERATION == 'gen5', CP.generation())
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
        plain = argparse.Namespace(campaign=False, cutout_root=None)
        check('... without --campaign the served roots as before', K.cutout_root(plain) == CFG.photometry.CUTOUT_ROOT
              and SS.stage1_root(run, plain) == SS.stage1_root(run) == os.path.join(run.out_root, CFG.photometry.STAGE1_DIR))
        P.set(CFG.photometry_campaign, 'GENERATION', CFG.photometry.GENERATION)
        e = raises(CP.generation)
        check('CP-GEN: the served generation as the campaign\'s refuses (a campaign writes a new generation)',
              e is not None and 'CP-GEN' in e and 'served' in e, e)
        P.set(CFG.photometry_campaign, 'GENERATION', 'photometry_gen6')
        e = raises(CP.generation)
        check('CP-GEN: a value that does not name a generation refuses', e is not None and 'CP-GEN' in e, e)
    finally:
        P.undo()


def campaign_args(**kw):
    a = dict(tier='snapshot', cutout_root=None, stage1=None, scopes=None, index=None, manifest_name=None, targets=None,
             planets=False, in_place=False, campaign=True, cells=None, epochs=None, exclude_cells=None)
    a.update(kw)
    return argparse.Namespace(**a)


def test_resume_hash():
    """The step's resume hash (record.Run.config_hash over config.step_sections of the step's CONFIG) covers every section of
    config/steps/photometry.py - the flux-scale gate's (#153) and the campaign's among them: a changed tolerance or floor makes
    --resume redo the units."""
    import importlib
    import inspect
    from exopipe.phase3 import config
    from exopipe.phase3.config.steps import photometry as PC
    step = importlib.import_module('exopipe.phase3.steps.photometry')
    secs = config.step_sections(step)
    mine = sorted(n for n, o in vars(PC).items() if inspect.isclass(o) and o.__module__ == PC.__name__
                  and issubclass(o, config._Section) and o is not config._Section)
    check('the photometry step declares every section of config/steps/photometry.py (its resume hash covers them)',
          mine and all(n in secs for n in mine), (mine, secs))
    P = Patch()
    try:
        for sec, key, val in (('photometry_fluxscale', 'FSAPPLD_TOL', 3e-6), ('photometry_campaign', 'QUOTA_MIN_FREE_FILES', 7)):
            h0 = CFG.hash(secs)
            P.set(getattr(CFG, sec), key, val)
            h1 = CFG.hash(secs)
            P.undo()
            check('a change to %s.%s changes the photometry step\'s resume hash (%s -> %s), restored after' % (sec, key, h0, h1),
                  h1 != h0 and CFG.hash(secs) == h0)
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
    check('the step parses --campaign and declares the photometry_campaign section (its config hash)',
          ap.parse_args(['--campaign']).campaign is True and 'photometry_campaign' in SP.CONFIG)


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
    check('without the engine\'s reader (the read-only listing of the check): core/exclusions alone decides (27 MHz a unit)',
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
        pin = CP.CodePin(files[:2])
        camp = CP.Campaign('gen6', CP.Enumeration(), pin, space_fn=lambda: (True, 'ok', {}), sleep=lambda s: None)
        camp.after(argparse.Namespace(name='cutouts.x.55MHz', meta={'legacy_code': {files[1]: 'ffffffffffff'}}))
        r = camp.refusal()
        check('a unit that ran another version of a pinned engine stops the campaign (CP-PIN on the next unit)',
              r and 'CP-PIN' in r and 'cutouts.x.55MHz' in r, r)
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
    R = {(c, k): CP.readiness(out, e, c, k) for c in (X, Y, Z) for k in ('sb', 'wb')}
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
    sts = {b: CP.unit_state(out, W, b)[0] for b in ['41MHz', 'Green', 'Blue']}
    check('gap classes: C-FLUXSCALE before the engine = a refused gap; C-NOCUBE = a no-input gap; C-FLUXSCALE after the cut '
          '(nothing placed) = incomplete', sts == {'41MHz': 'gap', 'Green': 'gap-noinput', 'Blue': 'held'}, sts)
    rg = (CP.readiness(out, eg, W, 'sb'), CP.readiness(out, eg, W, 'wb'))
    check('readiness: a refused gap holds its cell\'s kind as a gap (stage 1 not run); a non-gap refusal holds it as '
          'incomplete', rg[0][0] == 'gap' and 'C-FLUXSCALE' in rg[0][1] and rg[1][0] == 'held' and 'Blue' in rg[1][1], rg)
    write_rec(out, 'cutouts.%s.Blue' % W, 'done')
    check('readiness: a no-input gap (C-NOCUBE) does not hold - the kind is measured from the bands with cutouts',
          CP.readiness(out, eg, W, 'wb')[0] == 'ready', CP.readiness(out, eg, W, 'wb'))
    for b in ('Red', 'Blue'):
        write_rec(out, 'cutouts.%s.%s' % (W, b), 'no-input', 'C-EXCLUDED: x')
    check('readiness: every wideband excluded or a no-input gap -> \'no wideband\'',
          CP.readiness(out, eg, W, 'wb')[0] == 'no wideband', CP.readiness(out, eg, W, 'wb'))
    rf = 'C-FLUXSCALE: %s 41MHz: V: no FSAPPLD - the engine not started' % W
    P = Patch()
    try:
        P.set(CFG.photometry_campaign, 'GAP_REFUSED', {})
        check('the gap classes are config (photometry_campaign.GAP_REFUSED, ruled): without them the C-FLUXSCALE unit is '
              'incomplete', CP.gap_kind('refused', rf) is None and CP.unit_state(out, W, '41MHz')[0] == 'held')
    finally:
        P.undo()
    check('gap_kind: only the status the class names (a no-input C-FLUXSCALE, a refused C-NOCUBE are no gaps)',
          CP.gap_kind('refused', rf) == 'refused' and CP.gap_kind('no-input', rf) is None
          and CP.gap_kind('refused', 'C-NOCUBE: x') is None and CP.gap_kind('no-input', 'C-NOCUBE: x') == 'no-input')
    e.held = [dict(cell=X, subband='64MHz', reason='only the Stokes V cube')]
    check('held: a subband held at enumeration holds the cell\'s subbands, not its widebands',
          CP.readiness(out, e, X, 'sb')[0] == 'held' and CP.readiness(out, e, X, 'wb')[0] == 'ready')
    e.held = []
    e2 = CP.Enumeration()
    e2.units = [(X, b) for b in ('41MHz', '55MHz')]
    check('\'empty\': no unit of the kind in the cell', CP.readiness(out, e2, X, 'wb')[0] == 'empty')
    e3 = CP.Enumeration()
    e3.units = [(Z, '41MHz')]
    check('\'no subband\': the same rule for subbands (41 MHz of %s, night rule on both Stokes)' % Z,
          CP.readiness(out, e3, Z, 'sb')[0] == 'no subband')
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
        root, st1 = os.path.join(d, 'root_ready'), os.path.join(out, 'stage1_gen6')
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
        check('the cutouts stage under --campaign: every enumerated unit in order; a unit that ran another engine version '
              'stops the campaign (CP-PIN after it) - every later unit refuses before it starts', len(units) == len(e.units)
              and seen == sorted(e.units)[:2] and all(x[1] == 'refused' and x[2] == 'CP-PIN' for x in st[2:])
              and camp.stopped.startswith('CP-PIN'), st)
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

        def gate(u):
            u.refuse('CP-SPACE: synthetic')
        u = go(gate=gate)
        check('the campaign gate runs first in the stage-1 unit: its refusal is the unit\'s', u.status == 'refused'
              and u.reason == 'CP-SPACE: synthetic', (u.status, u.reason))
    finally:
        P.undo()


# ---------------------------------------------------------------------------------------------------------------------

def done_rec(out, root, cell, band, files, counts, drop=(), gen5=False, no_fs=False):
    """A done cutouts record (the checks a placed unit records), its manifest listing ``files`` (relative to root); a subband
    also its flux-scale record (``no_fs``: not recorded); ``gen5``: a unit of the generation-5 comparison sample."""
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
    write_rec(out, 'cutouts.%s.%s' % (cell, band), 'done', meta={'engine_outcome': counts, 'cutout_root': root,
                                                                 'generation5_sample': gen5}, tests=tests, outputs=outputs)


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
    for b in ['41MHz', 'Red']:
        done_rec(out2, root, c1, b, rels[b], {'V': '1', 'I': '1'})
    for b in ('Green', 'Blue'):
        write_rec(out2, 'cutouts.%s.%s' % (c1, b), 'no-input', 'C-EXCLUDED: x')
    done_rec(out2, root, c2, '41MHz', rels['41MHz'], {'V': '1', 'I': '1'})
    for b in WB:
        write_rec(out2, 'cutouts.%s.%s' % (c2, b), 'no-input', 'C-EXCLUDED: x')
    st1 = os.path.join(out2, 'stage1_gen6')
    for c, k, man in ((c1, 'sb', 'forced_manifest.csv'), (c1, 'wb', 'forced_manifest_wb.csv'),
                      (c2, 'sb', 'forced_manifest.csv')):
        os.makedirs(os.path.join(st1, c), exist_ok=True)
        open(os.path.join(st1, c, man), 'w').write('file,status,ring_beams,n_ctl\n/x/HWJ/Red/T0/T0_I.fits,ok,4.00,4\n')
        write_rec(out2, SS.unit_name(c, list(CP.kind_bands(k))), 'done', meta={'bands_done': list(CP.kind_bands(k))})
    write_rec(out2, SS.unit_name(c2, WB), 'no-input', 'SM-NOBAND: no wideband', meta={'outcome': 'no wideband'})
    ok, counts, probs, gaps = CP.check(out2, e, st1)
    check('check: every unit terminal, stage 1 done per kind or \'no wideband\' -> complete', ok and counts['done'] == 3
          and counts['none'] == 5 and counts.get('stage1_no_wideband') == 1 and counts.get('stage1_done') == 3,
          (counts, probs))
    write_rec(out2, 'cutouts.%s.Green' % c1, 'refused', 'C-WRITE: short')
    ok, counts, probs, gaps = CP.check(out2, e, st1)
    check('check: a refused unit is incomplete (unlike gen6\'s GATE-REFUSED) and its cell\'s wideband stage 1 is held',
          not ok and counts['incomplete'] == 1 and any(p.startswith('cutouts.%s.Green' % c1) and 'C-WRITE' in p for p in probs)
          and any('stage 1 %s wb held' % c1 in p for p in probs), (counts, probs))
    write_rec(out2, 'cutouts.%s.Green' % c1, 'no-input', 'C-EXCLUDED: x')
    os.remove(os.path.join(st1, c1, 'forced_manifest.csv'))
    ok, _, probs, gaps = CP.check(out2, e, st1)
    check('check: a done stage-1 record whose manifest is gone is incomplete', not ok and any('no manifest' in p for p in probs),
          probs)
    open(os.path.join(st1, c1, 'forced_manifest.csv'), 'w').write('file,status,ring_beams,n_ctl\n/x/HWJ/41MHz/T0/T0_I.fits,'
                                                                  'REFUSED: x,4.00,4\n')
    ok, _, probs, gaps = CP.check(out2, e, st1)
    check('check: a stage-1 manifest with a REFUSED row fails MM2: incomplete', not ok and any('REFUSED' in p for p in probs),
          probs)
    open(os.path.join(st1, c1, 'forced_manifest.csv'), 'w').write('file,status,ring_beams,n_ctl\n/x/HWJ/41MHz/T0/T0_I.fits,ok,'
                                                                  '4.00,4\n')
    e.held = [dict(cell=c1, subband='64MHz', reason='only the Stokes V cube')]
    ok, _, probs, gaps = CP.check(out2, e, st1)
    check('check: a held (cell, subband) is incomplete', not ok and any('held at enumeration' in p for p in probs), probs)
    e.held = []
    P = Patch()
    try:
        P.set(CP, 'enumerate_units', lambda cells=None, *a, **k: e)
        rc = CP.main(['check', '--out', out2])
        write_rec(out2, 'cutouts.%s.Red' % c1, 'error', 'RuntimeError: x')
        rc2 = CP.main(['check', '--out', out2, '--no-stage1'])
        check('the check command exits 0 complete, 1 incomplete', rc == 0 and rc2 == 1, (rc, rc2))
    finally:
        P.undo()
    started = dict(scope=[c1], enumeration=dict(units=[[c1, '41MHz'], [c1, 'Red']]))
    now = [(c1, '41MHz'), (c1, '46MHz'), (c2, '41MHz')]
    ch = (CP.units_changed(started, now), CP.units_changed(dict(started, scope=None), now, cells=[c1]))
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
    write_rec(out2, 'cutouts.%s.Red' % c1, 'done')
    done_rec(out2, root, c1, 'Red', rels['Red'], {'V': '1', 'I': '1'})
    fsr = 'C-FLUXSCALE: %s Red: Blue member 64MHz V: FSAPPLD 1.02 off its record 0.98 - the engine not started' % c1
    write_rec(out2, 'cutouts.%s.Red' % c1, 'refused', fsr)
    ok, counts, probs, gaps = CP.check(out2, e, st1)
    check('check: a unit refused by the flux-scale gate before the engine is a GAP - complete, listed with its code and reason; '
          'its cell\'s wideband stage 1 is not run, a gap too', ok and counts['gap'] == 1 and probs == []
          and any(g.startswith('cutouts.%s.Red: refused C-FLUXSCALE' % c1) for g in gaps)
          and any(g.startswith('stage 1 %s wb not run' % c1) for g in gaps), (counts, probs, gaps))
    write_rec(out2, 'cutouts.%s.Green' % c2, 'no-input', 'C-NOCUBE: no consolidated flow cube of %s Green (V)' % c2)
    ok, counts, probs, gaps = CP.check(out2, e, st1)
    check('check: a C-NOCUBE unit is a gap that holds nothing (%s\'s widebands stay \'no wideband\')' % c2,
          ok and counts['gap'] == 2 and counts.get('stage1_no_wideband') == 1, (counts, probs, gaps))
    P = Patch()
    try:
        P.set(CP, 'enumerate_units', lambda cells=None, *a, **k: e)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = CP.main(['check', '--out', out2])
        txt = buf.getvalue()
        check('the check command: COMPLETE with N gaps, exit 0, a GAP line per gap', rc == 0
              and 'COMPLETE with 3 gaps' in txt and txt.count('\nGAP ') + txt.startswith('GAP ') == 3, (rc, txt[-600:]))
        for code in ('C-GENCHECK: (b) failed - nothing placed', 'C-WRITE: short', 'C-COUNT: x', 'C-IDENTITY: x', 'C-PLACE: x',
                     'C-CONTENT: x', 'C-ENGINE: x', 'C-STALE: x', 'CP-SPACE: low - not started',
                     'C-FLUXSCALE: the member cubes of x fail the gate or changed - nothing placed (staging kept: s)'):
            write_rec(out2, 'cutouts.%s.Red' % c1, 'refused', code)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = CP.main(['check', '--out', out2])
            check('the check command: a unit refused %s is INCOMPLETE (exit 1), not a gap' % code.split(':')[0]
                  + (' (after the cut)' if 'member cubes' in code else ''),
                  rc == 1 and 'INCOMPLETE' in buf.getvalue() and 'PROBLEM cutouts.%s.Red' % c1 in buf.getvalue(),
                  (rc, buf.getvalue()[-400:]))
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
        P.set(CP, 'engine_exclusions', lambda run, cells: {c: W.engine.get(c, set()) for c in cells})
        P.set(CP, 'enumerate_units', lambda cells=None, engine_excluded=None: CP.Enumeration.from_dict(
            W.enum().to_dict()) if engine_excluded is not None else None)
        P.set(CP, 'pin_files', lambda: pins)
        P.set(CP, 'space', lambda lfs=None: (True, 'synthetic ok', {'fs_free_bytes': 1}))
        P.set(CFG.paths, 'SNAPSHOT_CUBES_ROOT', W.root)
        args = campaign_args()
        e = raises(CP.start, run_of(out), args)
        check('C-ROOT: in the refactor window (paths.PRODUCTION_WRITES False) the campaign refuses its production cutout root',
              e is not None and 'C-ROOT' in e, e)
        Q = Patch()
        try:
            Q.set(CP, 'enumerate_units', lambda cells=None, engine_excluded=None: W.enum(engine=False))
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
        test_start(W, d)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print('%d check(s) failed: %s' % (len(FAILS), FAILS) if FAILS else 'all checks passed', flush=True)
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
