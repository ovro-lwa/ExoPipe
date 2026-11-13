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

    def write_record(self, step, name, status):
        p = record.record_path(self.root, step, name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, 'w') as f:
            json.dump({'unit': name, 'status': status, 'reason': '' if status == 'done' else 'N-9 test'}, f)
        return p

    def night_record(self, sb, status='done'):
        return self.write_record('frames', NM.unit_name(LST, NIGHT, sb), status)

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
