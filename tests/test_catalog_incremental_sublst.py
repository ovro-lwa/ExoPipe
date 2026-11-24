#!/usr/bin/env python3
"""Data-free tests of the per-hour 10-minute item: `exopipe3 catalog --tier sublst --incremental --epochs <LST>_<date>`
(lib/catalog/incremental_sublst.py), its place in the per-hour chain, the night directory's guard in the night pass
(lib/catalog/tier_sublst.night_out_refusal) and the pages of the night taken from the item (lib/pages/tier_sublst.context).

The gate on the frozen all-nights state is tested part by part on a synthetic grouping directory and all-nights directory:
each missing part REFUSES with its gate's name and the text of what the first all-nights pass must write; a night already
in the all-nights directory ends no-input. The night pass, grouping and builds themselves are production code tested on real
data (the content test of the pull request); here only the gate and the orchestration's refusals before them run.

Run: python tests/test_catalog_incremental_sublst.py [test names]   (exit 0 = all passed, 1 = a failure, 2 = imports missing)
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

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')

_IMPORT_ERROR = None
try:
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import record
    from exopipe.phase3.lib.catalog import delta_build as DB
    from exopipe.phase3.lib.catalog import grouping_stage as GS
    from exopipe.phase3.lib.catalog import incremental_sublst as INC
    from exopipe.phase3.lib.catalog import tier_sublst as TS
except Exception as e:  # noqa: BLE001
    _IMPORT_ERROR = e
    if __name__ != '__main__':
        raise

LST, NIGHT, ARM = '17h', '2025-05-08', 'B'
TMP = None


class Patch:
    """Set attributes for the duration of a with-block (config section values)."""

    def __init__(self, *triples):
        self.triples, self.old = triples, []

    def __enter__(self):
        for obj, name, val in self.triples:
            self.old.append((obj, name, getattr(obj, name)))
            setattr(obj, name, val)
        return self

    def __exit__(self, *a):
        for obj, name, val in reversed(self.old):
            setattr(obj, name, val)


def tmp(name):
    global TMP
    if TMP is None:
        TMP = tempfile.mkdtemp(prefix='catalog_incremental_sublst_')
    d = os.path.join(TMP, name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    return d


def wcsv(path, hdr, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(rows)


def grouping(gdir, runs, nights=(), status='done', stokes=('I', 'V'), name='Run_SLSTB.group', finished='2026-11-24T00:00:00Z'):
    """A grouping run record of stage 'group' with its tables, as grouping_stage writes it."""
    tables = {}
    for S in stokes:
        t = os.path.join(gdir, f'grouping_{S}', GS.DETECTIONS)
        wcsv(t, ['det', 'identity'], [['a', '1']])
        tables[S] = dict(path=t, sha1=GS.table_digest(t))
    r = dict(unit=name, status=status, reason='', finished=finished,
             meta=dict(stage=GS.STAGE, stokes=list(stokes), tables=tables, input=dict(runs),
                       input_nights=[[L, D] for L, D in nights]))
    p = record.record_path(gdir, 'catalog', name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w') as f:
        json.dump(r, f)
    return p


def all_nights(F, lst=LST, night_rows=0, conform=True, frozen=('noveto', 'veto'), keys=('V_sb', 'I_sb')):
    """An all-nights catalog directory: its image table (night_rows rows of NIGHT), the two frozen conformance records with
    the done conformance record that lists them, and the builds' frozen records of ``keys`` (with selected rows)."""
    os.makedirs(F, exist_ok=True)
    rows = [[lst, 'Red', 'I', '2025-04-06T00']] + [[lst, 'Red', 'I', f'{NIGHT}T0{i}'] for i in range(night_rows)]
    wcsv(os.path.join(F, DB.IMAGES), ['lst', 'scope', 'stokes', 'date'], rows)
    noise, evid = TS.conform_noise_record(F), TS.conform_pulsar_evidence(F)
    for p in (noise, evid):
        wcsv(p, ['x'], [['1']])
    if conform:
        r = dict(unit=f'{lst}_{ARM}.conform', status='done', finished='2026-11-23T00:00:00Z',
                 meta=dict(output_fingerprints={noise: record.fingerprint(noise), evid: record.fingerprint(evid)}))
        p = record.record_path(F, 'catalog', f'{lst}_{ARM}.conform')
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, 'w') as f:
            json.dump(r, f)
    for K in keys:
        wcsv(os.path.join(F, DB.SELECTED.format(key=K)), ['det'], [['a']])
        for tag in frozen:
            cat = os.path.join(F, DB.CAT.format(tag=tag, key=K))
            os.makedirs(cat, exist_ok=True)
            with open(os.path.join(cat, DB.FROZEN_RECORD), 'w') as f:
                json.dump({'rows': {}}, f)
    return F


def gate(gdir, out, view_hours=None, lst=LST, night=NIGHT):
    return INC.frozen_state(lst, night, ARM, out, gdir, ('noveto', 'veto'), view_hours)


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return str(e)
    raise AssertionError(f'{fn.__name__} did not raise {exc.__name__}')


def parsed(step, argv):
    """argparse Namespace of `exopipe3 <step> <argv>` (the step's own and the common arguments, as cli.main builds them)."""
    import importlib
    from exopipe.phase3 import cli
    mod = importlib.import_module(f'exopipe.phase3.steps.{step}')
    sp = argparse.ArgumentParser(prog=f'exopipe3 {step}')
    mod.add_arguments(sp)
    cli._common(sp, mod)
    return sp.parse_args(argv)


# ---------------------------------------------------------------------------------------------------------------------

def test_the_tier_has_an_incremental_path_and_its_night_directory():
    """catalog_incremental.TIERS names sublst; the step's default --out of one item is the night's own directory
    (catalog_sublst.NIGHT_DIR), beside the grouping, never an LST-wide catalog directory."""
    from exopipe.phase3.steps import catalog as STEP
    assert 'sublst' in CFG.catalog_incremental.TIERS and 'deep' in CFG.catalog_incremental.TIERS
    a = parsed('catalog', ['--tier', 'sublst', '--incremental', '--epochs', f'{LST}_{NIGHT}'])
    want = CFG.catalog_sublst.NIGHT_DIR.format(transients_root=CFG.paths.TRANSIENTS_ROOT, lst=LST, arm=ARM, night=NIGHT)
    assert STEP.default_out(a) == want == INC.night_dir(LST, NIGHT), (STEP.default_out(a), want)
    assert want.endswith(f'/{LST}_{ARM}/{NIGHT}') and not want.startswith(CFG.catalog_sublst.CAT_ROOT)
    assert TS.night_out_refusal(want, LST, ARM, f'{LST}_{ARM}') == ''
    # what the first all-nights pass must write is named in every refusal of the gate
    for part in ('--group-new-state', 'conform_table_noise.csv', 'conform_pulsar_evidence.csv', DB.FROZEN_RECORD,
                 '--stages build,buildv', 'catalog_grouping.SUBLST_DIR'):
        assert part in INC.FIRST_RUN, part


def test_the_gate_refuses_without_the_frozen_grouping_state():
    """I-G2: no grouping run recorded (row 6 not run) -> REFUSED, naming what the first all-nights pass writes."""
    g, out = tmp('g_empty'), tmp('n_empty')
    msg = raises(record.Refusal, gate, g, out)
    assert msg.startswith('I-G2') and INC.FIRST_RUN in msg, msg
    grouping(g, {LST: tmp('F_ref')}, status='refused')                      # the latest run refused: no state either
    msg = raises(record.Refusal, gate, g, out)
    assert msg.startswith('I-G2') and 'ended refused' in msg, msg


def test_the_gate_checks_each_part_of_the_frozen_state():
    g, out = tmp('g_parts'), tmp('n_parts')
    F, F18 = os.path.join(TMP, 'F17'), os.path.join(TMP, 'F18')
    shutil.rmtree(F, ignore_errors=True)
    shutil.rmtree(F18, ignore_errors=True)
    grouping(g, {LST: F, '18h': F18})
    # I-G3: the night's hour is not in the grouping
    msg = raises(record.Refusal, gate, g, out, lst='19h')
    assert msg.startswith('I-G3') and '17h, 18h' in msg, msg
    # I-G5: no image table in the all-nights directory
    os.makedirs(F)
    msg = raises(record.Refusal, gate, g, out)
    assert msg.startswith('I-G5') and DB.IMAGES in msg, msg
    # the night IS in the all-nights directory: no-input (the batch chain's night), never appended twice
    all_nights(F, night_rows=2, conform=False)
    msg = raises(record.NoInput, gate, g, out)
    assert '2 rows' in msg and F in msg, msg
    # I-G6: no done conformance record listing the frozen statistics
    all_nights(F, conform=False)
    msg = raises(record.Refusal, gate, g, out)
    assert msg.startswith('I-G6') and f'{LST}_{ARM}.conform' in msg, msg
    all_nights(F)
    with open(TS.conform_noise_record(F), 'a') as f:                         # changed after its conformance recorded it
        f.write('2\n')
    msg = raises(record.Refusal, gate, g, out)
    assert msg.startswith('I-G6') and 'changed since' in msg, msg
    # I-G7: a build of this hour wrote no frozen record (a key with selected rows); another hour of the grouping has none
    shutil.rmtree(F)
    all_nights(F, frozen=('noveto',))
    msg = raises(record.Refusal, gate, g, out, view_hours=[LST])
    assert msg.startswith('I-G7') and f'{LST} veto V_sb' in msg and f'{LST} veto I_sb' in msg, msg
    all_nights(F)
    st = gate(g, out, view_hours=[LST])
    assert st['frozen_from'] == os.path.realpath(F) and st['scope'] == [LST] and len(st['frozen_records'][LST]) == 4, st
    assert set(st['grouping']) == {'I', 'V'} and st['nights_before'] == []
    msg = raises(record.Refusal, gate, g, out)                               # every hour in scope: 18h has no build
    assert msg.startswith('I-G7 1 hour(s) in scope lack') and f'18h ({F18} holds no {DB.IMAGES})' in msg \
        and '--sublst-view-hours' in msg, msg
    msg = raises(record.Refusal, gate, g, out, view_hours=['18h'])            # the night's own hour must be in scope
    assert msg.startswith('I-G7') and 'not named' in msg, msg
    all_nights(F18, lst='18h', keys=('V_wb',))
    st = gate(g, out)
    assert st['scope'] == [LST, '18h'] and set(st['frozen_records']) == {LST, '18h'}, st['scope']
    # an hour with no row of a key needs no frozen record of it (no catalog, nothing frozen)
    assert not os.path.exists(os.path.join(F18, DB.CAT.format(tag='veto', key='I_sb')))
    # I-G2: the table changed after its run recorded it
    with open(os.path.join(g, 'grouping_V', GS.DETECTIONS), 'a') as f:
        f.write('b,2\n')
    msg = raises(record.Refusal, gate, g, out)
    assert msg.startswith('I-G2') and 'changed since' in msg, msg


def test_the_gate_refuses_mixed_stokes_a_second_directory_and_production_grouping():
    g, out = tmp('g_mix'), tmp('n_mix')
    F = all_nights(os.path.join(tmp('F_mix'), 'F'))
    # I-G3: Stokes I and V grouped from different directories (two runs)
    grouping(g, {LST: F}, stokes=('I',), name='Run_SLSTB_I.group', finished='2026-11-24T00:00:01Z')
    grouping(g, {LST: F, '18h': F}, stokes=('V',), name='Run_SLSTB_V.group', finished='2026-11-24T00:00:02Z')
    msg = raises(record.Refusal, gate, g, out)
    assert msg.startswith('I-G3') and 'different directories' in msg, msg
    # I-G4: the same night of the hour was appended before from another directory
    g2, other = tmp('g_twice'), tmp('n_other')
    with open(os.path.join(other, INC.GATE_RECORD), 'w') as f:
        json.dump({'night': NIGHT, 'lst': LST}, f)
    grouping(g2, {LST: F}, nights=[(LST, other)])
    msg = raises(record.Refusal, gate, g2, out, view_hours=[LST])
    assert msg.startswith('I-G4') and other in msg, msg
    assert gate(g2, other, view_hours=[LST])['nights_before'] == [[LST, os.path.realpath(other)]]   # its own dir: allowed
    # I-G1: the per-hour path stands on design B
    with Patch((CFG.catalog_grouping, 'GROUPING', 'production'), (CFG.catalog_grouping, 'MULTIPLES_RULE', 'production')):
        msg = raises(record.Refusal, gate, g2, out, view_hours=[LST])
    assert msg.startswith('I-G1') and 'design-B' in msg, msg


def _run(out, argv, dry=False):
    a = parsed('catalog', argv + ['--out', out])
    from exopipe.phase3.core import cells
    r = record.Run('catalog', ['exopipe3', 'catalog'] + argv, out, CFG, dry_run=dry)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        rc = INC.run(a, r, cells.expand_epochs(a.epochs))
    return a, r, rc, buf.getvalue()


def test_the_item_refuses_before_any_unit_and_records_the_gate():
    """ONE item per invocation, the item's own stages and grouping (no --stages etc.); the gate is a unit with its record:
    refused (exit 2) without the frozen state, no-input (exit 3) for a night of the all-nights pass; no unit of the night
    pass runs after either."""
    out, g = tmp('item_out'), tmp('item_g')
    msg = raises(record.Refusal, _run, out, ['--tier', 'sublst', '--incremental', '--epochs',
                                             f'{LST}_{NIGHT},{LST}_2025-05-09', '--sublst-grouping', g])
    assert 'ONE item' in msg, msg
    for extra in (['--stages', 'merge'], ['--sublst-night', NIGHT], ['--group-new-state'], ['--group-lst-dir', f'{LST}=/x']):
        msg = raises(record.Refusal, _run, out, ['--tier', 'sublst', '--incremental', '--epochs', f'{LST}_{NIGHT}',
                                                 '--sublst-grouping', g] + extra)
        assert 'runs its own stages and grouping' in msg and extra[0] in msg, (extra, msg)
    argv = ['--tier', 'sublst', '--incremental', '--epochs', f'{LST}_{NIGHT}', '--sublst-grouping', g]
    _a, r, rc, txt = _run(out, argv, dry=True)                               # dry run: the gate's verdict, nothing written
    assert rc == record.EXIT_REFUSED and 'REFUSES: I-G2' in txt and not r.units, txt
    assert not os.path.exists(os.path.join(out, '_records'))
    _a, r, rc, txt = _run(out, argv)
    assert rc is None and [u.name for u in r.units] == [f'{LST}_{ARM}.{INC.STAGE_GATE}'], [u.name for u in r.units]
    assert r.exit_code() == record.EXIT_REFUSED and r.units[0].reason.startswith('I-G2')
    with open(record.record_path(out, 'catalog', r.units[0].name)) as f:
        assert INC.FIRST_RUN in json.load(f)['reason']
    assert not os.path.exists(os.path.join(out, INC.GATE_RECORD))
    F = all_nights(os.path.join(tmp('item_F'), 'F'), night_rows=1)
    grouping(g, {LST: F})
    _a, r, rc, txt = _run(out, argv + ['--sublst-view-hours', LST])
    assert [u.status for u in r.units] == ['no-input'] and r.exit_code() == record.EXIT_NO_INPUT, \
        [(u.name, u.status, u.reason) for u in r.units]
    assert 'never appended a second time' in r.units[0].reason


def test_the_night_lock_waits_then_refuses():
    g = tmp('lock')
    with INC.NightLock(g, 0, 'first'):
        msg = raises(record.Refusal, INC.NightLock(g, 0.2, 'second').__enter__)
        assert 'first: pid' in msg and 'after 0 s' in msg, msg
    with INC.NightLock(g, 0, 'third'):                                       # released: taken again at once
        pass


def test_a_night_pass_never_writes_an_lst_wide_directory():
    """tier_sublst.night_out_refusal: the night pass may run in its own directory under a production root (the per-hour
    chain's NIGHT_DIR), never under catalog_sublst.CAT_ROOT, in the published run of the hour, or in a directory holding
    records of the LST-wide stages."""
    unit = f'{LST}_{ARM}'
    assert 'CAT_ROOT' in TS.night_out_refusal(os.path.join(CFG.catalog_sublst.CAT_ROOT, 'catalog_B_17h'), LST, ARM, unit)
    pub = os.path.join(CFG.paths.TRANSIENTS_ROOT, CFG.catalog_omission.TENMIN_RUN.format(hh='17', arm=ARM))
    assert 'published run' in TS.night_out_refusal(pub, LST, ARM, unit)
    d = tmp('lstwide')
    assert TS.night_out_refusal(d, LST, ARM, unit) == ''
    p = record.record_path(d, 'catalog', f'{unit}.fitpos')
    os.makedirs(os.path.dirname(p))
    with open(p, 'w') as f:                                                  # a night pass's stage record: allowed
        json.dump({'unit': f'{unit}.fitpos', 'status': 'done', 'meta': {'stage': 'fitpos', 'night': NIGHT}}, f)
    assert TS.night_out_refusal(d, LST, ARM, unit) == ''
    with open(p, 'w') as f:                                                  # the LST-wide pass's (no night): refused
        json.dump({'unit': f'{unit}.fitpos', 'status': 'done', 'meta': {'stage': 'fitpos'}}, f)
    assert 'LST-wide stage(s) fitpos' in TS.night_out_refusal(d, LST, ARM, unit)
    os.remove(p)
    p = record.record_path(d, 'catalog', f'{unit}.tablefilter')               # an LST-wide-only stage: refused
    with open(p, 'w') as f:
        json.dump({'unit': f'{unit}.tablefilter', 'status': 'done', 'meta': {'stage': 'tablefilter', 'night': NIGHT}}, f)
    assert 'tablefilter' in TS.night_out_refusal(d, LST, ARM, unit)
    assert TS.night_meta(type('C', (), {'night': NIGHT})()) == {'night': NIGHT}
    assert TS.night_meta(type('C', (), {'night': None})()) == {}


def test_the_per_hour_chain_runs_the_ten_minute_catalog_pages_and_serve():
    from exopipe.phase3.core import runner
    steps = runner.chain_steps('epoch')
    ten = [a for a in steps if '--tier' in a and a[a.index('--tier') + 1] == 'sublst']
    assert ['catalog', '--tier', 'sublst', '--incremental', '--epochs', '{item}'] in ten
    assert ['pages', '--tier', 'sublst', '--epochs', '{item}'] in ten and ['serve', '--tier', 'sublst', '--epochs', '{item}'] in ten
    i = steps.index(['catalog', '--tier', 'sublst', '--incremental', '--epochs', '{item}'])
    assert i < steps.index(['pages', '--tier', 'sublst', '--epochs', '{item}']) \
        < steps.index(['serve', '--tier', 'sublst', '--epochs', '{item}']) < len(steps) - 1
    for argv in (['catalog', '--tier', 'sublst', '--incremental', '--epochs', f'{LST}_{NIGHT}'],
                 ['pages', '--tier', 'sublst', '--epochs', f'{LST}_{NIGHT}'], ['serve', '--tier', 'sublst', '--epochs',
                                                                              f'{LST}_{NIGHT}']):
        parsed(argv[0], argv[1:])                                             # each line parses


def test_the_pages_of_the_item_come_from_its_night_directory():
    """`pages --tier sublst --epochs <item>`: the night directory of the item (pages_sublst.NIGHT_DIR, the mirror of
    catalog_sublst.NIGHT_DIR); no-input when the
    item's catalog pass ended no-input, REFUSED when the directory has no index otherwise; one item per invocation; the
    earlier page runs default to pages_sublst.NIGHT_BASE_RUNS."""
    from exopipe.phase3.lib.pages import tier_sublst as PT
    root = tmp('pages_nights')
    nd_fmt = os.path.join(root, '{lst}_{arm}', '{night}')
    out = tmp('pages_out')
    # the pages step's mirror of the catalog's night directory (it does not declare the catalog's config sections)
    assert CFG.pages_sublst.NIGHT_DIR == CFG.catalog_sublst.NIGHT_DIR and CFG.pages_sublst.ARM == CFG.catalog_sublst.ARM
    with Patch((CFG.catalog_sublst, 'NIGHT_DIR', nd_fmt), (CFG.pages_sublst, 'NIGHT_DIR', nd_fmt)):
        nd = PT.night_item_dir(LST, NIGHT)
        assert nd == os.path.join(root, f'{LST}_{ARM}', NIGHT) == INC.night_dir(LST, NIGHT)
        r = record.Run('pages', ['pages'], out, CFG)
        a = parsed('pages', ['--tier', 'sublst', '--epochs', f'{LST}_{NIGHT}', '--out', out])
        msg = raises(record.Refusal, PT.context, a, r)
        assert 'P18' in msg and nd in msg, msg
        p = record.record_path(nd, 'catalog', f'{LST}_{ARM}.{INC.STAGE_GATE}')
        os.makedirs(os.path.dirname(p))
        with open(p, 'w') as f:
            json.dump({'status': 'no-input', 'reason': 'the night is in the all-nights catalog directory'}, f)
        msg = raises(record.NoInput, PT.context, a, r)
        assert f'{INC.STAGE_GATE}: the night is in the all-nights' in msg, msg
        a2 = parsed('pages', ['--tier', 'sublst', '--epochs', f'{LST}_{NIGHT},18h_{NIGHT}', '--out', out])
        assert 'ONE --epochs item' in raises(record.Refusal, PT.context, a2, r)
        with open(os.path.join(nd, CFG.pages_sublst.NIGHT_INDEX), 'w') as f:
            json.dump({'hours': {LST: {'kind': 'view', 'written_here': True, 'root': nd},
                                 '18h': {'kind': 'all-nights', 'all_nights': '/x'}}, 'not_viewed': []}, f)
        c = PT.context(a, r)
        assert c.night_dir == nd and c.lsts == [LST] and c.item == (LST, NIGHT), (c.night_dir, c.lsts)
        assert c.base_runs == os.path.abspath(CFG.pages_sublst.NIGHT_BASE_RUNS.format(
            transients_root=CFG.paths.TRANSIENTS_ROOT))
        # an explicit --lsts keeps the LST-wide meaning of --epochs (its LSTs); --sublst-base-runs wins
        with Patch((CFG.pages_sublst, 'NIGHT_DIR', nd_fmt + '/x')):          # the copy differs from the catalog's: refused
            assert 'differ from the catalog' in raises(record.Refusal, PT.context, a, r)
        a3 = parsed('pages', ['--tier', 'sublst', '--epochs', f'{LST}_{NIGHT}', '--sublst-night-dir', nd,
                              '--sublst-base-runs', out + '/b', '--out', out])
        c = PT.context(a3, r)
        assert c.item is None and c.lsts == [LST] and c.base_runs == out + '/b'


def main(argv):
    if _IMPORT_ERROR is not None:
        print(f'imports missing: {type(_IMPORT_ERROR).__name__}: {_IMPORT_ERROR}')
        return 2
    tests = [(n, f) for n, f in sorted(globals().items(), key=lambda kv: inspect.getsourcelines(kv[1])[1]
                                       if callable(kv[1]) and kv[0].startswith('test_') else 0)
             if callable(f) and n.startswith('test_') and (not argv or n in argv)]
    failed = 0
    try:
        for n, f in tests:
            try:
                f()
                print(f'PASS {n}')
            except Exception as e:  # noqa: BLE001
                failed += 1
                import traceback
                traceback.print_exc()
                print(f'FAIL {n}: {type(e).__name__}: {str(e)[:600]}')
    finally:
        if TMP:
            shutil.rmtree(TMP, ignore_errors=True)
    print(f'{len(tests) - failed} of {len(tests)} passed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
