#!/usr/bin/env python3
"""Data-free tests of the wideband stage's arm-A evidence in a per-hour run of ONE night (lib/frames/tier_sublst.py:
night_arm_a_complete, wideband_arm_a, stage_wideband).

A night of the per-hour path that is not in the window Masters gets its arm-A residuals from the unit
nonmember_<LST>_<night>_<sb> (stage nonmember), not from the LST-wide stage `masters --tier sublst --stages loo`. The
wideband stage of that one night therefore reads the record of the night's own unit where this root holds one (done
passes, any other status refuses) and keeps the LST-wide evidence for a subband without one. An LST-wide wideband run
reads the LST-wide evidence only.

The engine (clv15/phase3_sublst_wideband.py, production code) is replaced by a stub that refuses: a unit whose reason is
the stub's line passed every gate before the engine. The engine's own run on real data is in the record
exopipe_ops refactor_20261012/plan/stage_map_tools/perhour_wideband_nonmember_20261113/RECORD.md.

Run: python tests/test_sublst_wideband_perhour.py [test names]   (exit 0 = all passed, 1 = a failure, 2 = imports missing)
"""
import argparse
import contextlib
import csv
import inspect
import io
import json
import os
import shutil
import sys
import tempfile
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')

_IMPORT_ERROR = None
try:
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import cells, record
    from exopipe.phase3.lib.frames import tier_sublst as FT
    from exopipe.phase3.lib.masters import sublst_nonmember as NM
except Exception as e:  # noqa: BLE001
    _IMPORT_ERROR = e
    if __name__ != '__main__':
        raise

LST = '09h'
NIGHT = '2024-02-01'
G = 'Red'
STUB_LINE = 'REFUSED: stub engine (the gates before the engine passed)'


class Patch:
    """Set attributes for the duration of a with-block (module functions, config section values)."""

    def __init__(self, *triples):
        self.triples, self.old = triples, []

    def __enter__(self):
        for obj, name, val in self.triples:
            self.old.append((obj, name, getattr(obj, name)))
            setattr(obj, name, val)
        return self

    def __exit__(self, *exc):
        for obj, name, val in reversed(self.old):
            setattr(obj, name, val)


class Root:
    """A synthetic sub-LST root: a complete refine log for every member of the group, the night's directory, an exclusion
    list of its own, and a stub in place of the wideband engine (its calls are kept in `calls`)."""

    def __init__(self):
        self.dir = tempfile.mkdtemp(prefix='sublst_wideband_perhour_')
        self.root = os.path.join(self.dir, 'sublst')
        self.members = list(cells.band_members(G))
        self.calls = []
        os.makedirs(os.path.join(self.root, LST, f'{LST}_{NIGHT}'))
        logs = os.path.join(self.root, f'{LST}_x2', 'logs')
        os.makedirs(logs)
        for sb in self.members:
            with open(os.path.join(logs, f'{LST}_{sb}.log'), 'w') as f:
                f.write(f'SUBLST-REFINE {LST} {sb} aligned=12 refused=0 refused_rule=0 refused_night=0\n')
        self.decisions = os.path.join(self.dir, 'data_decisions.csv')
        with open(self.decisions, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['lst', 'subband', 'date', 'verdict', 'reason'])
            w.writerow(['23h', '18', '2020-01-01', 'exclude', 'unrelated'])
        self.script = os.path.join(self.dir, 'phase3_sublst_wideband.py')
        open(self.script, 'w').close()

        def engine(script, args_, python, log, unit=None, env=None, refusal_codes=(2,), refusal_text=None):
            self.calls.append(list(args_))
            res = types.SimpleNamespace(ok=False, refused=True, rc=2, reason='stub', tail=STUB_LINE, fallback_refusals=[])
            return res, [STUB_LINE]
        self._patch = Patch((CFG.paths, 'DATA_DECISIONS', self.decisions), (FT, 'legacy_script', lambda _r, _f: self.script),
                            (FT, 'engine', engine))

    def __enter__(self):
        self._patch.__enter__()
        return self

    def __exit__(self, *exc):
        self._patch.__exit__(*exc)
        shutil.rmtree(self.dir, ignore_errors=True)

    def write_record(self, step, name, status, reason=None, **meta):
        p = record.record_path(self.root, step, name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, 'w') as f:
            json.dump({'unit': name, 'status': status, 'reason': reason if reason is not None else
                       ('' if status == 'done' else 'N-9 test'), 'meta': meta}, f)
        return p

    def night_record(self, sb, status='done', stokes=('I', 'V'), reason=None):
        return self.write_record('frames', NM.unit_name(LST, NIGHT, sb), status, reason, stokes=list(stokes))

    def stack(self, sb, S='I', k=0, night=NIGHT):
        """One window stack of a night at a subband (an empty file: only its modification time is read)."""
        p = os.path.join(self.root, LST, f'{LST}_{night}', sb, S, f'sublst_{LST}_{night}_{sb}_{S}_w{k}.fits')
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, 'w').close()
        return p

    def batch_record(self, sb, status='done'):
        return self.write_record('masters', f'loo_{LST}_{sb}', status)

    def forget(self, step, name):
        os.remove(record.record_path(self.root, step, name))

    def args(self, **kw):
        a = dict(epochs=f'{LST}_{NIGHT}', cells=None, lsts=None, nights=None, scopes=G, stokes=None, stages='wideband',
                 from_stage=None, until_stage=None, dry_run=False, workers=1, fresh=False, supersede=None, resume=False,
                 sublst_groups=None, sublst_mirror=False, sublst_loo_evidence=None)
        a.update(kw)
        return argparse.Namespace(**a)

    def ctx(self, **kw):
        return FT._Ctx(self.args(**kw), record.Run('frames', ['test'], self.root, CFG))

    def run(self, **kw):
        """Run the stage through the frames tier; returns (the units of the run, what it printed)."""
        del self.calls[:]
        r = record.Run('frames', ['test'], self.root, CFG, dry_run=bool(kw.get('dry_run')))
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            FT.run(self.args(**kw), r)
        return r.units, out.getvalue()


def only(units, name):
    assert [u.name for u in units] == [name], [(u.name, u.status, u.reason) for u in units]
    return units[0]


# =====================================================================================================================
# the evidence
# =====================================================================================================================

def test_the_nights_own_record_decides_where_this_root_holds_one():
    with Root() as t:
        c, sb = t.ctx(), t.members[1]
        name = NM.unit_name(LST, NIGHT, sb)
        assert name == f'nonmember_{LST}_{NIGHT}_{sb}'
        # no record of the night's unit: nothing to say, and the LST-wide evidence decides (missing here)
        assert FT.night_arm_a_complete(c, LST, sb, NIGHT) is None
        ok, why, path, how = FT.wideband_arm_a(c, LST, sb, NIGHT)
        assert not ok and 'masters loo record' in why and why.endswith(': missing') and '--stages loo' in how, (why, how)
        assert path == record.record_path(t.root, 'masters', f'loo_{LST}_{sb}')
        # the night's record, done
        p = t.night_record(sb)
        assert FT.night_arm_a_complete(c, LST, sb, NIGHT) == (True, f"record of the night's own arm-A unit {name} ({p}): done", p)
        ok, why, path, how = FT.wideband_arm_a(c, LST, sb, NIGHT)
        assert ok and path == p and f'--stages nonmember --epochs {LST}_{NIGHT}' in how and sb in how, (why, how)
        # another night's record says nothing about this night
        assert FT.night_arm_a_complete(c, LST, sb, '2024-02-02') is None
        # any status but done refuses, also when the LST-wide record is done
        t.batch_record(sb)
        for status in ('refused', 'error', 'no-input', 'running', 'skipped-valid'):
            t.night_record(sb, status)
            ok, why, path, how = FT.wideband_arm_a(c, LST, sb, NIGHT)
            assert not ok and why.endswith(f': {status}') and path == p and '--stages nonmember' in how, (status, why)
        # an LST-wide run (no single night) reads the LST-wide evidence only
        ok, why, path, how = FT.wideband_arm_a(c, LST, sb)
        assert ok and 'masters loo record' in why and '--stages loo' in how
        t.forget('masters', f'loo_{LST}_{sb}')
        t.night_record(sb)
        assert not FT.wideband_arm_a(c, LST, sb)[0] and FT.wideband_arm_a(c, LST, sb, NIGHT)[0]


def plan_of(sbs, stokes=('I', 'V')):
    """A night_plan (the stack's admit record) with one stack job of the night per subband, and one of another night."""
    return lambda run, opts, lst, night, planned_now=False: (
        [(f'{LST}_{NIGHT}', sb, list(stokes), '/cubes') for sb in sbs]
        + [(f'{LST}_2024-01-02', '73MHz', ['I', 'V'], '/cubes')], ['plan.txt'], 'record')


def test_the_record_is_read_against_the_nights_job_plan():
    """Where this root holds the night's job plan: a subband with a stack job needs the night's own record (the LST-wide
    record does not stand in for it), done for every Stokes the plan owes; a record of a subband the plan does not hold
    says nothing; a plan that cannot be read refuses. The plan is read once per run."""
    with Root() as t:
        a, b, other = t.members[0], t.members[1], t.members[2]
        reads = []

        def plan(run, opts, lst, night, planned_now=False):
            reads.append(night)
            return plan_of([a, b])(run, opts, lst, night)

        with Patch((NM, 'night_plan', plan)):
            c = t.ctx()
            assert NM.night_owed(c.run, None, LST, NIGHT) == {a: ['I', 'V'], b: ['I', 'V']}   # the other night's job apart
            # a stack job in the plan and no record of the night's unit: not complete, also with the LST-wide record done
            t.batch_record(a)
            ok, why, path, how = FT.wideband_arm_a(c, LST, a, NIGHT)
            assert not ok and 'holds a stack job' in why and why.endswith(': missing') and '--stages nonmember' in how, why
            assert path == record.record_path(t.root, 'frames', NM.unit_name(LST, NIGHT, a))
            t.night_record(a)                                   # done for both Stokes: complete
            assert FT.wideband_arm_a(c, LST, a, NIGHT)[0]
            t.night_record(b, stokes=('I',))                    # done for Stokes I only: the plan owes V too
            ok, why, _p, how = FT.wideband_arm_a(c, LST, b, NIGHT)
            assert not ok and 'for Stokes I only' in why and 'owes Stokes V too' in why and '--stages nonmember' in how, why
            assert FT.wideband_arm_a(t.ctx(stokes='I'), LST, b, NIGHT)[0]     # ... which a run of Stokes I alone does not need
            # no stack job of the subband in the plan: a record of an earlier plan says nothing, the LST-wide evidence decides
            t.night_record(other, 'refused')
            assert FT.night_arm_a_complete(c, LST, other, NIGHT) is None
            ok, why, _p, how = FT.wideband_arm_a(c, LST, other, NIGHT)
            assert not ok and 'masters loo record' in why and '--stages loo' in how, why
            del reads[:]
            c = t.ctx()
            for sb in t.members:
                FT.wideband_arm_a(c, LST, sb, NIGHT)
            assert reads == [NIGHT], reads                      # one read of the plan per run
            # the stage: the group refuses on the subband whose record is missing
            for sb in t.members:
                t.night_record(sb)
                t.batch_record(sb)
            t.forget('frames', NM.unit_name(LST, NIGHT, b))
            u = only(t.run()[0], f'{LST}_wideband_{G}_{NIGHT}')
            assert u.status == 'refused' and f'arm A of {LST} {b} not complete' in u.reason and 'holds a stack job' in u.reason \
                and not t.calls, (u.status, u.reason)

        def broken(run, opts, lst, night, planned_now=False):
            raise ValueError('bad plan line')

        with Patch((NM, 'night_plan', broken)):                 # never taken for "no plan"
            ok, why, _p, _how = FT.wideband_arm_a(t.ctx(), LST, a, NIGHT)
            assert not ok and 'could not be read' in why and 'ValueError: bad plan line' in why, why
            u = only(t.run()[0], f'{LST}_wideband_{G}_{NIGHT}')
            assert u.status == 'refused' and 'could not be read' in u.reason and not t.calls, (u.status, u.reason)


def test_a_record_older_than_the_nights_newest_stack_is_not_complete():
    """The night's residuals are of the stack the unit read: a done record older than the night's newest window stack of
    the subband refuses (S-G8's rule); another night's stack, or another subband's, does not count."""
    with Root() as t:
        sb, name = t.members[1], f'{LST}_wideband_{G}_{NIGHT}'
        for m in t.members:
            t.night_record(m)
        p = record.record_path(t.root, 'frames', NM.unit_name(LST, NIGHT, sb))
        s = t.stack(sb, 'V', 2)
        os.utime(p, (1000, 1000))
        os.utime(s, (2000, 2000))
        c = t.ctx()
        ok, why, path, how = FT.wideband_arm_a(c, LST, sb, NIGHT)
        assert not ok and "older than the night's newest" in why and path == p and '--stages nonmember' in how, why
        u = only(t.run()[0], name)
        assert u.status == 'refused' and f'arm A of {LST} {sb} not complete' in u.reason and 'older than' in u.reason \
            and not t.calls, (u.status, u.reason)
        os.utime(s, (500, 500))                               # the stack is older than the record: complete
        os.utime(t.stack(sb, 'I', 0, night='2024-01-02'), (3000, 3000))
        os.utime(t.stack(t.members[2], 'I', 0), (900, 900))
        assert FT.wideband_arm_a(c, LST, sb, NIGHT)[0]
        u = only(t.run()[0], name)
        assert u.status == 'refused' and STUB_LINE in u.reason and len(t.calls) == 1, (u.status, u.reason)


# =====================================================================================================================
# the stage
# =====================================================================================================================

def test_per_hour_wideband_of_one_night_passes_on_the_nights_records():
    with Root() as t:
        name = f'{LST}_wideband_{G}_{NIGHT}'
        # nothing recorded: refused on arm A (the LST-wide evidence, missing), before the engine
        u = only(t.run()[0], name)
        assert u.status == 'refused' and u.reason.startswith(f'arm A of {LST} {t.members[0]} not complete (masters loo record') \
            and '`masters --tier sublst --stages loo`' in u.reason and not t.calls, (u.status, u.reason, t.calls)
        # the night's unit done at every member: every gate passes and the engine is called for that night only
        recs = [t.night_record(sb) for sb in t.members]
        u = only(t.run()[0], name)
        assert u.status == 'refused' and STUB_LINE in u.reason, (u.status, u.reason)
        assert len(t.calls) == 1 and t.calls[0][t.calls[0].index('--nights') + 1] == NIGHT \
            and t.calls[0][t.calls[0].index('--groups') + 1] == G, t.calls
        ins = [(i['path'] if isinstance(i, dict) else i) for i in u.inputs]
        assert all(p in ins for p in recs), (recs, ins)                  # the records are inputs: a rerun of one re-opens it
        assert not [p for p in ins if os.sep + 'masters' + os.sep in p], ins


def test_a_member_whose_unit_of_the_night_is_not_done_refuses():
    with Root() as t:
        name = f'{LST}_wideband_{G}_{NIGHT}'
        for sb in t.members:
            t.night_record(sb)
        bad = t.members[2]
        t.night_record(bad, 'refused')
        u = only(t.run()[0], name)
        assert u.status == 'refused' and u.reason.startswith(f'arm A of {LST} {bad} not complete (record of the night\'s own '
                                                             f'arm-A unit nonmember_{LST}_{NIGHT}_{bad}') \
            and u.reason.rstrip().endswith(f'run `frames --tier sublst --stages nonmember --epochs {LST}_{NIGHT}` for {bad} '
                                           f'first') and not t.calls, (u.status, u.reason, t.calls)
        # ... also when the LST-wide record of that subband is done: the night's own unit wrote (or refused) its residuals
        t.batch_record(bad)
        u = only(t.run()[0], name)
        assert u.status == 'refused' and f'nonmember_{LST}_{NIGHT}_{bad}' in u.reason and not t.calls, (u.status, u.reason)


def test_a_member_without_a_record_of_the_night_keeps_the_lst_wide_evidence():
    with Root() as t:
        name = f'{LST}_wideband_{G}_{NIGHT}'
        other = t.members[-1]                      # the night is in the window Masters there: no unit of the night
        for sb in t.members[:-1]:
            t.night_record(sb)
        u = only(t.run()[0], name)
        assert u.status == 'refused' and u.reason.startswith(f'arm A of {LST} {other} not complete (masters loo record') \
            and not t.calls, (u.status, u.reason)
        p = t.batch_record(other)
        u = only(t.run()[0], name)
        assert u.status == 'refused' and STUB_LINE in u.reason and len(t.calls) == 1, (u.status, u.reason, t.calls)
        assert p in [(i['path'] if isinstance(i, dict) else i) for i in u.inputs]


def test_an_lst_wide_run_reads_the_lst_wide_evidence_only():
    with Root() as t:
        for sb in t.members:
            t.night_record(sb)
        u = only(t.run(epochs=None, lsts=LST)[0], f'{LST}_wideband_{G}')
        assert u.status == 'refused' and 'masters loo record' in u.reason and '--stages loo' in u.reason and not t.calls, \
            (u.status, u.reason)
        # ... and so does an LST-wide run narrowed to the one night with --nights (intermittent mode: never a per-hour unit)
        u = only(t.run(epochs=None, lsts=LST, nights=NIGHT)[0], f'{LST}_wideband_{G}')
        assert u.status == 'refused' and 'masters loo record' in u.reason and not t.calls, (u.status, u.reason)


def test_the_dry_run_reads_the_same_evidence():
    with Root() as t:
        for sb in t.members[:-1]:
            t.night_record(sb)
        units, out = t.run(dry_run=True)
        lines = [l.strip() for l in out.splitlines() if 'gate arm A complete' in l]
        assert len(lines) == len(t.members), out
        for sb, l in zip(t.members[:-1], lines):
            assert l.startswith(f'gate arm A complete {sb}: pass - record of the night\'s own arm-A unit '
                                f'nonmember_{LST}_{NIGHT}_{sb}'), l
        assert lines[-1].startswith(f'gate arm A complete {t.members[-1]}: WOULD REFUSE - masters loo record'), lines[-1]
        assert f'UNIT {LST}_wideband_{G}_{NIGHT}:' in out and not t.calls
        assert not os.path.exists(os.path.join(t.root, '_records', 'frames', f'{LST}_wideband_{G}_{NIGHT}.json'))


# =====================================================================================================================
# 18 MHz without a window Master (Gregg 2026-10-17: "Red can be built without 18 MHz, i.e. using just the 4 other
# subbands"): its refused unit of the night leaves it out of the group instead of refusing the group
# =====================================================================================================================

N6 = 'N-6: no installed window Master of 09h 18MHz Stokes I under /x: the batch stage writes them'
N7 = "N-7: 09h_2024-02-01 18MHz: no window of the night has an installed window Master (6 window stack(s)): ..."


def resid(t, sb, x2=False, S='V', k=1):
    """One residual of the night at a subband: arm A under <root>/<LST>/, the x2-aligned one under <root>/<LST>_x2/."""
    d = os.path.join(t.root, f'{LST}_x2' if x2 else LST, f'{LST}_{NIGHT}', sb, S)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f'sublst_{LST}_{NIGHT}_{sb}_{S}_w{k}_resid_loo.fits')
    open(p, 'w').close()
    return p


def test_eighteen_mhz_refused_for_want_of_a_master_is_left_out():
    with Root() as t:
        name, sb = f'{LST}_wideband_{G}_{NIGHT}', '18MHz'
        assert t.members[0] == sb and tuple(CFG.frames_sublst.WIDEBAND_OPTIONAL) == (sb,)
        for m in t.members:
            t.night_record(m)
        for reason in (N6, N7):
            p = t.night_record(sb, 'refused', reason=reason)
            lo = FT.optional_member_left_out(t.ctx(), LST, sb, NIGHT)
            assert lo[0] and f'nonmember_{LST}_{NIGHT}_{sb}' in lo[1] and reason[:40] in lo[1], lo
            u = only(t.run()[0], name)
            assert u.status == 'refused' and STUB_LINE in u.reason and len(t.calls) == 1, (u.status, u.reason)
            assert list(u.meta['left_out']) == [sb] and any(n.startswith(f'{sb} left out') for n in u.notes), (u.meta, u.notes)
            assert p in [(i['path'] if isinstance(i, dict) else i) for i in u.inputs]      # a new record re-opens the unit
        # S-G8 is not read for it either: the refine-nonmember stage writes no 18 MHz residual of the night
        os.remove(os.path.join(t.root, f'{LST}_x2', 'logs', f'{LST}_{sb}.log'))
        u = only(t.run()[0], name)
        assert u.status == 'refused' and STUB_LINE in u.reason and len(t.calls) == 1, (u.status, u.reason)
        units, out = t.run(dry_run=True)
        assert f'gate arm A complete {sb}: not required - {sb} left out' in out and f'gate S-G8 x2 complete {sb}: not ' \
            f'required' in out and f'{sb}: WOULD REFUSE' not in out and not units and not t.calls, out
        # an LST-wide run is untouched: it reads the LST-wide evidence
        u = only(t.run(epochs=None, lsts=LST)[0], f'{LST}_wideband_{G}')
        assert u.status == 'refused' and f'arm A of {LST} {sb} not complete (masters loo record' in u.reason, u.reason


def test_eighteen_mhz_is_not_left_out_on_any_other_refusal():
    with Root() as t:
        name, sb = f'{LST}_wideband_{G}_{NIGHT}', '18MHz'
        for m in t.members:
            t.night_record(m)
        # another refusal of the unit (the stack incomplete, a Master that cannot be subtracted), or another status
        for status, reason in (('refused', 'N-4: the stack is incomplete'), ('refused', 'N-8 test'), ('error', N6),
                               ('running', '')):
            t.night_record(sb, status, reason=reason)
            assert FT.optional_member_left_out(t.ctx(), LST, sb, NIGHT) is None, (status, reason)
            u = only(t.run()[0], name)
            assert u.status == 'refused' and u.reason.startswith(f'arm A of {LST} {sb} not complete') and not t.calls, \
                (status, u.reason)
        # another subband refused for want of a Master refuses the group: 18 MHz is the one exception
        t.night_record(sb)
        other = t.members[1]
        t.night_record(other, 'refused', reason=N6.replace(sb, other))
        assert FT.optional_member_left_out(t.ctx(), LST, other, NIGHT) is None
        u = only(t.run()[0], name)
        assert u.status == 'refused' and u.reason.startswith(f'arm A of {LST} {other} not complete') and not t.calls, u.reason


def test_eighteen_mhz_with_a_residual_of_the_night_on_disk_refuses():
    """A residual of the night left by an earlier run would be stacked by the engine: the group refuses, naming it."""
    with Root() as t:
        name, sb = f'{LST}_wideband_{G}_{NIGHT}', '18MHz'
        for m in t.members:
            t.night_record(m)
        t.night_record(sb, 'refused', reason=N7)
        for x2 in (False, True):
            r = resid(t, sb, x2=x2)
            lo = FT.optional_member_left_out(t.ctx(), LST, sb, NIGHT)
            assert lo[0] is False and r in lo[1] and 'move them aside' in lo[1], lo
            u = only(t.run()[0], name)
            assert u.status == 'refused' and u.reason.startswith(f'arm A of {LST} {sb} not complete') and r in u.reason \
                and u.reason.rstrip().endswith('first') and not t.calls, u.reason
            units, out = t.run(dry_run=True)
            assert f'gate arm A complete {sb}: WOULD REFUSE' in out and r in out, out
            os.remove(r)
        resid(t, t.members[1])                    # another subband's residual does not count
        assert FT.optional_member_left_out(t.ctx(), LST, sb, NIGHT)[0]


def test_eighteen_mhz_refusal_older_than_a_stack_or_a_master_refuses():
    """The refusal holds for the stacks and Masters it read: a newer window stack of the night, or a window Master
    installed after it, refuses until the unit runs again."""
    with Root() as t:
        name, sb = f'{LST}_wideband_{G}_{NIGHT}', '18MHz'
        for m in t.members:
            t.night_record(m)
        p = t.night_record(sb, 'refused', reason=N6)
        os.utime(p, (2000, 2000))
        s = t.stack(sb, 'I', 0)
        md = os.path.join(t.root, f'{LST}_masters', sb, 'V')
        os.makedirs(md)
        mp = os.path.join(md, f'sublst_master_{LST}_{sb}_V_w4.fits')
        open(mp, 'w').close()
        for newer, older in ((s, mp), (mp, s)):
            os.utime(newer, (3000, 3000))
            os.utime(older, (1000, 1000))
            lo = FT.optional_member_left_out(t.ctx(), LST, sb, NIGHT)
            assert lo[0] is False and newer in lo[1] and 'run the unit again' in lo[1], lo
            u = only(t.run()[0], name)
            assert u.status == 'refused' and newer in u.reason and not t.calls, u.reason
        os.utime(mp, (1000, 1000))
        os.utime(t.stack(sb, 'V', 2, night='2024-01-02'), (3000, 3000))     # another night's stack does not count
        assert FT.optional_member_left_out(t.ctx(), LST, sb, NIGHT)[0]
        u = only(t.run()[0], name)
        assert u.status == 'refused' and STUB_LINE in u.reason and len(t.calls) == 1, u.reason


# =====================================================================================================================
# a night IN the window Masters (Gregg 2026-11-13): the per-hour 10-minute steps end no-input
# =====================================================================================================================

def window_weights(t, sb, w, S='I', k=3):
    """One installed window Master's weights record of (LST, sb) giving the night the weight w."""
    p = os.path.join(t.root, f'{LST}_masters', sb, S, f'sublst_master_{LST}_{sb}_{S}_w{k}_weights.csv')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w') as f:
        f.write(f'night,weight\n2024-01-02,0.8\n{NIGHT},{w}\n')
    open(p[:-len('_weights.csv')] + '.fits', 'w').close()             # membership counts an installed Master only


def test_a_group_with_a_member_subband_ends_no_input():
    with Root() as t:
        name = f'{LST}_wideband_{G}_{NIGHT}'
        for sb in t.members:
            t.night_record(sb)
        sb = t.members[1]
        window_weights(t, sb, 0.0)                    # listed at weight 0: not a constituent, the gates and engine run
        u = only(t.run()[0], name)
        assert u.status == 'refused' and STUB_LINE in u.reason and len(t.calls) == 1, (u.status, u.reason)
        window_weights(t, sb, 0.6)                    # a constituent of ONE member subband's window Master
        u = only(t.run()[0], name)
        assert u.status == 'no-input' and not t.calls, (u.status, u.reason, t.calls)
        assert f'Masters of 1 member subband(s) ({sb})' in u.reason and "'sublst-lst'" in u.reason, u.reason
        assert u.meta['master_member'] == {sb: [['I', 3, 0.6]]}, u.meta
        units, out = t.run(dry_run=True)
        assert not units and f'UNIT {name}: NO-INPUT - ' in out and not t.calls, out
        # an LST-wide run is the batch chain's: membership does not apply
        u = only(t.run(epochs=None, lsts=LST, nights=f'{NIGHT},2024-01-02')[0], f'{LST}_wideband_{G}')
        assert u.status != 'no-input' or 'IS in the installed window' not in u.reason, (u.status, u.reason)


def test_the_stack_of_a_member_subband_ends_no_input():
    got = []

    def stack_night(run, lst, night, subbands=None, **kw):
        got.append(list(subbands or []))
        return {'stack_units': [], 'jobs': [], 'left_out': []}

    from exopipe.phase3.lib.masters import tier_sublst as MT
    with Root() as t, Patch((MT, 'stack_night', stack_night)):
        a, b = t.members[1], t.members[2]
        window_weights(t, a, 0.9, S='V', k=0)
        units, _out = t.run(stages='stack', scopes=f'{a},{b}')
        from exopipe.phase3.lib.masters import sublst_nonmember as NM
        u = only(units, NM.member_stack_unit_name(LST, NIGHT, a))   # its own name: never the night's stack unit
        assert u.name != MT.stack_unit_name(LST, NIGHT, a), u.name
        assert u.status == 'no-input' and 'w0 Stokes V weight 0.9' in u.reason, (u.status, u.reason)
        assert u.meta['master_member'] == [['V', 0, 0.9]] and got == [[b]], (u.meta, got)
        window_weights(t, b, 0.2)                     # every selected subband a member: no stack call at all
        del got[:]
        units, _out = t.run(stages='stack', scopes=f'{a},{b}')
        assert sorted(x.status for x in units) == ['no-input', 'no-input'] and not got, ([(x.name, x.status) for x in units], got)


def test_the_per_hour_chain_names_the_ten_minute_stages_in_order():
    """runner.CHAINS['epoch']: the 10-minute tier of ONE night is the list line + stack + arm A, the x2-aligned residual,
    the wideband groups + the tree entry, then the search and the catalog - each stage named (the tier's defaults are the
    stack alone and the LST-wide refine), each a stage the step knows, in the tier's own order."""
    from exopipe.phase3.core import runner
    from exopipe.phase3.steps import flow as FLOW
    steps = runner.chain_steps('epoch')
    ten = [(i, a) for i, a in enumerate(steps) if '--tier' in a and a[a.index('--tier') + 1] == 'sublst']
    assert [a[0] for _, a in ten] == ['frames', 'flow', 'frames', 'search', 'catalog'], [a[:3] for _, a in ten]
    assert [i for i, _ in ten] == list(range(ten[0][0], ten[0][0] + 5)), 'the 10-minute steps are consecutive'
    assert all(a[a.index('--epochs') + 1] == '{item}' for _, a in ten)

    def stages(a):
        return a[a.index('--stages') + 1].split(',') if '--stages' in a else None

    got = [stages(a) for _, a in ten]
    assert got == [['nonmember-list', 'stack', 'nonmember'], ['refine-nonmember'], ['wideband', 'tree-nonmember'], None,
                   None], got
    S = CFG.frames_sublst
    for st in (got[0], got[2]):
        args = argparse.Namespace(stages=','.join(st), from_stage=None, until_stage=None)
        assert FT.requested_stages(args, S.ORDER, S.STAGES['per-hour']) == st, st          # known, and in pipeline order
    assert got[1][0] in FLOW.STAGES['sublst'] and got[1][0] not in CFG.flow.DEFAULT_STAGES['sublst']
    # the defaults would not make the night's residuals or its tree entry: the stages must be named
    assert tuple(S.STAGES['per-hour']) == ('stack',)


def test_the_per_hour_chain_makes_the_night_cubes_before_the_ten_minute_tier():
    """Gregg 2026-11-13: the 10-minute stack reads the night's 10-second cubes (G-S02), so CHAINS['epoch'] runs snapshot
    frames, snapshot flow and `frames --tier snapshot --stages cube,movies` before the 10-minute steps, the rest of the
    snapshot steps after them, and pages / serve / archive last."""
    from exopipe.phase3.core import runner
    steps = [a[:a.index('--epochs')] if '--epochs' in a else a for a in runner.chain_steps('epoch')]

    def tier(a):
        return a[a.index('--tier') + 1] if '--tier' in a else None

    ten = [i for i, a in enumerate(steps) if tier(a) == 'sublst']
    snap = [(i, a) for i, a in enumerate(steps) if tier(a) == 'snapshot']
    before = [a for i, a in snap if i < ten[0]]
    after = [a for i, a in snap if i > ten[-1]]
    assert before == [['frames', '--tier', 'snapshot'], ['flow', '--tier', 'snapshot'],
                      ['frames', '--tier', 'snapshot', '--stages', 'cube,movies']], before
    assert [a[0] for a in after] == ['search', 'curves', 'catalog', 'forced'], after
    assert all(i < ten[0] for i, a in enumerate(steps) if tier(a) == 'deep')
    assert [a[0] for a in steps[snap[-1][0] + 1:]] == ['pages', 'serve', 'archive'], steps[snap[-1][0] + 1:]
    T = CFG.frames_snapshot
    args = argparse.Namespace(stages='cube,movies', from_stage=None, until_stage=None)
    assert FT.requested_stages(args, T.ORDER, T.DEFAULT_STAGES) == ['cube', 'movies']    # known, in pipeline order
    assert 'cube' not in T.DEFAULT_STAGES             # the default frames step makes no cube: the stage must be named
    # every catalog step of the per-hour chain is incremental: steps/catalog.py takes --epochs only with --incremental (a
    # usage error, exit 1, otherwise); a tier with no incremental path then refuses (exit 2) with its reason recorded
    cats = [a for a in runner.chain_steps('epoch') if a[0] == 'catalog']
    assert [tier(a) for a in cats] == ['deep', 'sublst', 'snapshot'] and all('--incremental' in a for a in cats), cats


def main(argv):
    if _IMPORT_ERROR is not None:
        print(f'imports missing: {type(_IMPORT_ERROR).__name__}: {_IMPORT_ERROR}')
        return 2
    tests = [(n, f) for n, f in sorted(globals().items(), key=lambda kv: inspect.getsourcelines(kv[1])[1]
                                       if callable(kv[1]) and kv[0].startswith('test_') else 0)
             if callable(f) and n.startswith('test_') and (not argv or n in argv)]
    failed = 0
    for n, f in tests:
        try:
            f()
            print(f'PASS {n}')
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback
            traceback.print_exc()
            print(f'FAIL {n}: {type(e).__name__}: {str(e)[:600]}')
    print(f'{len(tests) - failed} of {len(tests)} passed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
