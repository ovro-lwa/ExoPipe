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
import hashlib
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


def omission(name='omission', nights=(NIGHT,)):
    """A deep-omission decision table (the omit stage's columns that are read) under TMP and its coverage record (the cells
    it decides: its decision rows, one per LST and night of ``nights``, each rated on one detection - a night is decided by
    a table that rated its detections, Omission.night_refusal): {path, sha256, cells_path, cells_sha256}."""
    from exopipe.phase3.lib.catalog.wraps import builder_grouping as BG
    p = os.path.join(TMP or tmp('_tmp'), name, 'deep_omission.csv')
    if not os.path.exists(p):
        wcsv(p, ['lst', 'band', 'night', 'stokes', 'verdict', 'reason'],          # the reason: tables of other names differ
             [[f'{h:02d}h', 'Red', n, 'I', 'KEEP', name] for h in range(24) for n in nights])
        BG.write_cells(p, {(f'{h:02d}h', 'Red', n, 'I'): 1 for h in range(24) for n in nights})
    with open(p, 'rb') as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    with open(BG.cells_path(p), 'rb') as f:
        return dict(path=p, sha256=sha, cells_path=BG.cells_path(p), cells_sha256=hashlib.sha256(f.read()).hexdigest())


def grouping(gdir, runs, nights=(), status='done', stokes=('I', 'V'), name='Run_SLSTB.group', finished='2026-11-24T00:00:00Z',
             om='omission', omission_ref=None):
    """A grouping run record of stage 'group' with its tables and the deep-omission decision table ``om`` (a name for
    omission(); None: none recorded), as grouping_stage writes it."""
    tables = {}
    for S in stokes:
        t = os.path.join(gdir, f'grouping_{S}', GS.DETECTIONS)
        wcsv(t, ['det', 'identity'], [['a', '1']])
        tables[S] = dict(path=t, sha1=GS.table_digest(t))
    r = dict(unit=name, status=status, reason='', finished=finished,
             meta=dict(stage=GS.STAGE, stokes=list(stokes), tables=tables, input=dict(runs),
                       input_nights=[[L, D] for L, D in nights]))
    if om or omission_ref:
        r['meta']['omission'] = dict(omission_ref or omission(om), decisions=24, omit_keys=0, lsts=[], dropped={})
    p = record.record_path(gdir, 'catalog', name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w') as f:
        json.dump(r, f)
    return p


def all_nights(F, lst=LST, night_rows=0, conform=True, frozen=('noveto', 'veto'), keys=('V_sb', 'I_sb'), om='omission'):
    """An all-nights catalog directory: its image table (night_rows rows of NIGHT), the two frozen conformance records with
    the done conformance record that lists them, and the builds' frozen records of ``keys`` (with selected rows), each
    naming the deep-omission decision table ``om`` (a name for omission())."""
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
                json.dump({'rows': {}, 'omission': dict(omission(om), dropped=0)}, f)
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
                 '--stages build,buildv', 'catalog_grouping.SUBLST_DIR', '--until veto', '--stages omit', '--group-omission',
                 'not one per LST'):
        assert part in INC.FIRST_RUN, part
    # the pass order: every hour to the veto, then the omission, then ONE grouping, then the builds
    order = [INC.FIRST_RUN.index(x) for x in ('--until veto', '--stages omit', '--stages group', '--stages build,buildv')]
    assert order == sorted(order), order


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


def test_the_gate_refuses_another_deep_omission():
    """I-G8: the frozen state stands on ONE deep-omission decision table - the one the latest grouping run of both Stokes
    recorded, unchanged, and named by every frozen record in scope; anything else refuses."""
    g, out = tmp('g_om'), tmp('n_om')
    F = all_nights(os.path.join(tmp('F_om'), 'F'))
    grouping(g, {LST: F})
    st = gate(g, out, view_hours=[LST])
    assert st['omission'] == omission(), st.get('omission')
    grouping(g, {LST: F}, om=None)                                          # a grouping that followed no omission
    msg = raises(record.Refusal, gate, g, out, view_hours=[LST])
    assert msg.startswith('I-G8') and 'recorded no deep-omission decision table' in msg and INC.FIRST_RUN in msg, msg
    grouping(g, {LST: F}, stokes=('I',), name='Run_SLSTB_I.group', finished='2026-11-24T00:00:01Z')
    grouping(g, {LST: F}, stokes=('V',), name='Run_SLSTB_V.group', finished='2026-11-24T00:00:02Z', om='other')
    msg = raises(record.Refusal, gate, g, out, view_hours=[LST])            # I and V filtered by different tables
    assert msg.startswith('I-G8') and 'different deep-omission decision tables' in msg, msg
    g2 = tmp('g_om2')
    grouping(g2, {LST: F}, om='other')                                       # the frozen records name another table
    msg = raises(record.Refusal, gate, g2, out, view_hours=[LST])
    assert msg.startswith('I-G8 4 frozen record(s) were built under a deep-omission decision table that is neither') and \
        'the frozen state stands on one omission' in msg, msg
    g3 = tmp('g_om3')
    grouping(g3, {LST: F})
    with open(omission()['path'], 'a') as f:                                 # the table changed after the grouping
        f.write('17h,Red,2025-05-01,I,OMIT\n')
    msg = raises(record.Refusal, gate, g3, out, view_hours=[LST])
    assert msg.startswith('I-G8') and 'changed since' in msg, msg
    shutil.rmtree(os.path.dirname(omission()['path']))                      # the other tests' table, restored


def test_a_night_the_table_predates_is_rated_before_its_grouping_and_never_grouped_without_verdicts():
    """The gate takes a night the table predates (the omit unit rates it after the night pass); I-G9 is the group unit's:
    grouped under a table whose coverage record holds no cell of the night, the night REFUSES, named ("predates it"), never
    kept for want of a verdict; under a table that decides it, I-G9 passes. A frozen record on the parent of an appended
    table stands (I-G8, Omission.extends); one on a table the grouping's does not append to refuses."""
    from exopipe.phase3.lib.catalog.wraps import builder_grouping as BG
    g, out = tmp('g_ig9'), tmp('n_ig9')
    before = omission('before', nights=('2025-05-01',))                      # rated before the night 2025-05-08
    F = all_nights(os.path.join(tmp('F_ig9'), 'F'), om='before')
    grouping(g, {LST: F}, om='before')
    st = gate(g, out, view_hours=[LST])                                      # the gate takes the night
    assert st['omission'] == before, st['omission']
    os.makedirs(out, exist_ok=True)
    # the omit unit: the latest grouping's table is the parent; rated among the all-nights directory, the nights appended
    # before (the grouping's night list without this directory) and this night; the returned table is the one grouped under
    from exopipe.phase3.lib.catalog import deep_omission as DO
    seen = {}

    def fake_append(parent, lst, night, night_dir, o, **k):
        seen.update(parent=parent, lst=lst, night=night, night_dir=night_dir, out=o, **k)
        return dict(status='covered', table=dict(parent), appended_rows=0)
    earlier = tmp('ig9_earlier')
    grouping(g, {LST: F}, nights=[(LST, earlier), (LST, out)], om='before', finished='2026-11-24T00:00:05Z')
    ou = _UnitRefusal()
    site = os.path.join(tmp('ig9_search_out'), 'p3s5_collect_{host}')       # every site root of the search, whatever node
    for h in ('nodeA', 'nodeB'):
        os.makedirs(site.format(host=h), exist_ok=True)
    with Patch((DO, 'append_night', fake_append), (DO._cfg().search_run, 'COLLECT_ROOT', site)):
        parent, om = INC._rate_night(ou, st, out)
    assert parent == om == before and seen['parent'] == before and (seen['lst'], seen['night']) == (LST, NIGHT), seen
    assert seen['search_out'] == [site.format(host=h) for h in ('nodeA', 'nodeB')], seen   # G18 reads every node's register
    assert seen['night_dir'] == out and seen['out'] == os.path.join(out, DO.APPEND_DIR), seen
    assert seen['lst_dirs'] == {LST: st['frozen_from']} and seen['night_dirs'] == {LST: [os.path.realpath(earlier)]}, seen
    grouping(g, {LST: F}, om='other', finished='2026-11-24T00:00:06Z')       # the latest grouping's table: not an append
    with Patch((DO, 'append_night', fake_append)):
        msg = raises(_Refused, INC._rate_night, _UnitRefusal(), st, out)
    assert msg.startswith('I-G8') and 'nor an append to it' in msg, msg
    grouping(g, {LST: F}, om='before', finished='2026-11-24T00:00:07Z')
    gr = _UnitRefusal()
    msg = raises(_Refused, INC._group, _ResumeRun({}), gr, st, out, before, before)
    assert msg.startswith('I-G9') and f'night {NIGHT} of {LST} has no verdict' in msg and 'predates it' in msg, msg
    # the night appended to the table (a byte-prefix extension with the night's cell): I-G9 passes, the frozen records stand
    app = os.path.join(tmp('ig9_app'), 'deep_omission.csv')
    with open(before['path'], 'rb') as f:
        data = f.read()
    with open(app, 'wb') as f:
        f.write(data + f'{LST},Red,{NIGHT},I,KEEP,before\r\n'.encode())
    rated = {k: int(n) for k, n in BG.Omission(before['path']).detections.items()}
    BG.write_cells(app, {(LST, 'Green', NIGHT, 'V'): 3, **rated})
    V = BG.Omission(app)
    assert V.extends(before) == '' and V.night_refusal(LST, NIGHT) == ''
    def reached(*a, **k):
        raise _Refused('the grouping run')
    fr = argparse.Namespace(cfg=CFG, config_sections=None, argv=['test'], cores=None)
    with Patch((GS, 'run', reached)):
        msg = raises(_Refused, INC._group, fr, _UnitRefusal(), st, out, before, V.ref())
    assert msg == 'the grouping run', msg                                    # past I-G9: the grouping run is reached
    grouping(g, {LST: F}, om=None, omission_ref=V.ref(), finished='2026-11-25T00:00:00Z')
    st2 = gate(g, out, view_hours=[LST])                                     # the frozen records on its parent stand (I-G8)
    assert st2['omission'] == V.ref(), st2['omission']
    with open(app, 'wb') as f:                                               # an earlier row changed: not an append
        f.write(data.replace(b'KEEP', b'OMIT', 1) + f'{LST},Red,{NIGHT},I,KEEP,before\r\n'.encode())
    rated = {k: int(n) for k, n in BG.Omission(before['path']).detections.items()}
    BG.write_cells(app, {(LST, 'Green', NIGHT, 'V'): 3, **rated})
    grouping(g, {LST: F}, om=None, omission_ref=BG.Omission(app).ref(), finished='2026-11-25T00:00:01Z')
    msg = raises(record.Refusal, gate, g, out, view_hours=[LST])
    assert msg.startswith('I-G8 4 frozen record(s)') and 'does not begin with the recorded table' in msg, msg


class _Refused(Exception):
    pass


class _UnitRefusal:
    """A unit whose refuse raises _Refused (the item's units raise on refuse)."""

    def __init__(self):
        self.meta, self.notes = {}, []

    def refuse(self, why):
        raise _Refused(why)

    def note(self, m):
        self.notes.append(m)

    def check(self, *a, **k):
        pass

    def add_input(self, p):
        pass

    def add_output(self, *a, **k):
        pass


def _run(out, argv, dry=False):
    a = parsed('catalog', argv + ['--out', out])
    from exopipe.phase3.core import cells
    r = record.Run('catalog', ['exopipe3', 'catalog'] + argv, out, CFG, dry_run=dry, resume=a.resume)
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
    with INC.night_lock(g, 0, 'first'):                                      # grouping_stage._Lock with a wait
        msg = raises(GS.StageRefusal, INC.night_lock(g, 0.2, 'second').__enter__)
        assert 'held by another grouping run (pid' in msg and 'after 0 s of waiting' in msg, msg
    with INC.night_lock(g, 0, 'third'):                                       # released: taken again at once
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


def test_a_delta_catalog_is_tested_against_the_selection_its_build_read():
    """A night's delta build rebuilds a touched identity from every member it has, so its catalog directory holds earlier
    nights' members that the night's own selected_<key>.csv (the directory's sibling) lacks. The content test
    source_catalog takes the build's selection (ctx selection, which _build_delta passes); a named selection that is absent
    fails. Seen on real data 2026-11-24 (17h, night 2025-05-08): 3 Stokes I subband members and 6 Stokes I wideband
    members of earlier nights."""
    from exopipe.phase3.lib.catalog import tests as CT
    U = tmp('delta_selection')
    d = os.path.join(U, 'source_catalogs_noveto_I_sb')
    early, new = f'I1_{LST}_2025-04-14T01_23MHz_0680', f'I1_{LST}_{NIGHT}T03_23MHz_0001'
    wcsv(os.path.join(d, '23MHz_sources_survivors.csv'), ['source_id', 'best_snr', 'ra', 'dec', 'stokes', 'promotion'],
         [['J1700+3000', '9.0', '255.0', '30.0', 'I', '']])
    wcsv(os.path.join(d, '23MHz_source_members.csv'), ['source_id', 'band', 'cand_id'],
         [['J1700+3000', '23MHz', early], ['J1700+3000', '23MHz', new]])
    hdr = ['cand_id', 'stokes', 'lst', 'band', 'date']
    wcsv(os.path.join(U, 'selected_I_sb.csv'), hdr, [[new, 'I', LST, '23MHz', f'{NIGHT}T03']])          # the night's own rows
    B = os.path.join(U, '_side', 'delta_noveto')
    wcsv(os.path.join(B, 'selected_I_sb.csv'), hdr, [[early, 'I', LST, '23MHz', '2025-04-14T01'],         # the build's input
                                                    [new, 'I', LST, '23MHz', f'{NIGHT}T03']])
    none = type('X', (), {'is_excluded': staticmethod(lambda *a: False)})()
    with Patch((CT, '_excl', lambda tier=None: none)):
        ok, _, why = CT.source_catalogue_dir(d)
        assert not ok and '1 members not in selected_I_sb.csv' in why, why
        ok, n, why = CT.source_catalogue_dir(d, selection=os.path.join(B, 'selected_I_sb.csv'))
        assert ok and n == 1, why
        ok, _, why = CT.source_catalogue_dir(d, selection=os.path.join(B, 'absent.csv'))
        assert not ok and 'absent' in why, why


def test_the_per_hour_chain_runs_the_ten_minute_catalog_pages_and_serve():
    from exopipe.phase3.core import runner
    steps = runner.chain_steps('epoch')
    cat = ['catalog', '--tier', 'sublst', '--incremental', '--resume', '--epochs', '{item}']   # a retry resumes the item
    ten = [a for a in steps if '--tier' in a and a[a.index('--tier') + 1] == 'sublst']
    assert cat in ten, ten
    assert ['pages', '--tier', 'sublst', '--epochs', '{item}'] in ten and ['serve', '--tier', 'sublst', '--epochs', '{item}'] in ten
    i = steps.index(cat)
    assert i < steps.index(['pages', '--tier', 'sublst', '--epochs', '{item}']) \
        < steps.index(['serve', '--tier', 'sublst', '--epochs', '{item}']) < len(steps) - 1
    for argv in ([x.replace('{item}', f'{LST}_{NIGHT}') for x in cat],
                 ['pages', '--tier', 'sublst', '--epochs', f'{LST}_{NIGHT}'], ['serve', '--tier', 'sublst', '--epochs',
                                                                              f'{LST}_{NIGHT}']):
        parsed(argv[0], argv[1:])                                             # each line parses


def test_every_class_the_item_and_the_night_pass_write_has_a_content_test():
    """A class without a registered content test crashes the unit that writes it (KeyError in validate.run) and every
    --resume re-test: the night pass's view crashed so on real data before tests_night.py. Every literal class of an
    output call in the item and the 10-minute tier modules must resolve."""
    import re
    from exopipe.phase3.core import validate
    pat = re.compile(r"(?:\bs\.out|\.add_output|_add_output)\(\s*[^,()]+(?:\([^()]*\))?[^,()]*,\s*'([a-z0-9_]+)'")
    missing, seen = [], set()
    for mod in ('lib/catalog/incremental_sublst.py', 'lib/catalog/tier_sublst.py'):
        with open(os.path.join(REPO, 'exopipe', 'phase3', mod)) as f:
            for cls in pat.findall(f.read()):
                seen.add(cls)
                try:
                    validate.tests_for(cls)
                except KeyError:
                    missing.append(f'{mod}: {cls}')
    for cls in ('catalog_view', 'hour_views_index', 'night_frozen_state', 'night_grouping_record'):
        assert cls in seen, (cls, sorted(seen))
    assert not missing, missing


def _night_records(name):
    """A night directory with a grouping record of an incremental run that took it, its tables and the run's record."""
    g, out = tmp(name + '_g'), tmp(name + '_out')
    F = all_nights(os.path.join(tmp(name + '_F'), 'F'))
    rp = grouping(g, {LST: F}, nights=[(LST, out)])
    tables = {S: dict(path=os.path.join(g, f'grouping_{S}', GS.DETECTIONS),
                      sha1=GS.table_digest(os.path.join(g, f'grouping_{S}', GS.DETECTIONS))) for S in ('I', 'V')}
    doc = dict(version=1, lst=LST, night=NIGHT, grouping_dir=g, mode='incremental', tables=tables, hours={LST: F},
               nights=[[LST, out]], record=rp, omission=omission())
    p = os.path.join(out, INC.GROUP_RECORD)
    with open(p, 'w') as f:
        json.dump(doc, f)
    return g, out, F, p


def test_the_night_records_are_tested_for_their_own_claims():
    """night_grouping_record and night_frozen_state are records of what the grouping took and what the gate saw: they still
    pass once a later grouping run replaced the table (review of PR 161, finding 7), and FAIL when their own claims break (a
    night list without this directory, a frozen record gone)."""
    from exopipe.phase3.core import validate
    g, out, F, p = _night_records('rec')
    st = gate(g, out)
    sp = os.path.join(out, INC.GATE_RECORD)
    with open(sp, 'w') as f:
        json.dump(st, f)
    with open(os.path.join(g, 'grouping_I', GS.DETECTIONS), 'a') as f:      # a later grouping run's table
        f.write('b,2\n')
    for cls, path in (('night_grouping_record', p), ('night_frozen_state', sp)):
        r = validate.run(cls, path)
        assert all(x['passed'] for x in r), (cls, r)
    with open(omission()['path'], 'a') as f:                                 # I-G8: the table the state stands on changed
        f.write('17h,Red,2025-05-01,I,OMIT,late\n')
    det = ' '.join(x['detail'] for x in validate.run('night_frozen_state', sp))
    assert 'changed since' in det and '(I-G8)' in det, det
    shutil.rmtree(os.path.dirname(omission()['path']))
    omission()                                                               # restored (the same bytes)
    assert all(x['passed'] for x in validate.run('night_frozen_state', sp))
    frz = st['frozen_records'][LST]['noveto_V_sb']
    os.remove(frz)
    assert 'is gone' in ' '.join(x['detail'] for x in validate.run('night_frozen_state', sp))
    with open(p) as f:                                                       # a record of another night directory
        doc = json.load(f)
    doc['nights'] = [[LST, '/elsewhere']]
    with open(p, 'w') as f:
        json.dump(doc, f)
    assert 'holds this directory 0 time' in ' '.join(x['detail'] for x in validate.run('night_grouping_record', p))
    doc.pop('omission', None)                                                # a record from before the omission table
    with open(p, 'w') as f:
        json.dump(doc, f)
    r = validate.run('night_grouping_record', p)
    det = ' '.join(x['detail'] for x in r)
    assert not all(x['passed'] for x in r) and 'predates the deep-omission decision table' in det and 'group the night again' \
        in det, det


def test_the_night_keys_are_checked_before_the_grouping_takes_the_night():
    """I-G7 for the night's own rows (finding 6): a key with rows in the night whose all-nights build wrote no frozen record
    refuses BEFORE the grouping (else the delta build would refuse a grouped night for good)."""
    out = tmp('keys_out')
    F = all_nights(os.path.join(tmp('keys_F'), 'F'), keys=('V_sb',))
    st = dict(lst=LST, frozen_from=F)
    wcsv(os.path.join(out, DB.SELECTED.format(key='V_sb')), ['det'], [['a']])
    assert INC.night_keys_refusal(st, out, ('noveto', 'veto')) is None
    wcsv(os.path.join(out, DB.SELECTED.format(key='I_wb')), ['det'], [['b']])
    why = INC.night_keys_refusal(st, out, ('noveto', 'veto'))
    assert why.startswith('I-G7') and 'noveto I_wb' in why and 'veto I_wb' in why and INC.FIRST_RUN in why, why


def test_a_failed_grouping_run_leaves_the_last_done_state_standing():
    """Finding 3: a grouping run that did not end done has its unit record moved to _records/catalog/failed/, and the last
    done record it overwrote is put back, so the gate and the next item stand on it (grouping_stage leaves the tables and
    state unchanged on a refusal)."""
    g = tmp('failed_g')
    F = all_nights(os.path.join(tmp('failed_F'), 'F'))
    rp = grouping(g, {LST: F})
    with open(rp, 'rb') as f:
        saved = (rp, f.read())
    with open(rp, 'w') as f:                                                 # the refused run's record, same unit name
        json.dump({'unit': 'Run_SLSTB.group', 'status': 'refused', 'finished': '2026-11-25T00:00:00Z',
                   'meta': {'stage': GS.STAGE, 'stokes': ['I', 'V']}}, f)
    raises(GS.StageRefusal, GS.recorded_inputs, g, 'I')                      # the tier would refuse everywhere
    moved = INC._keep_last_done(g, argparse.Namespace(name='Run_SLSTB.group'), saved)
    assert os.path.dirname(moved) == os.path.join(os.path.dirname(rp), 'failed')
    with open(moved) as f:
        assert json.load(f)['status'] == 'refused'
    runs, _n = GS.recorded_inputs(g, 'I')
    assert runs == {LST: F}, runs


def test_the_omit_unit_refuses_two_stokes_that_recorded_different_nights():
    """I-G10: the omit unit takes the nights appended before (the night's rate population) from the latest grouping runs of
    BOTH Stokes; when they differ it REFUSES, named, and rates nothing - one Stokes' list is never applied to the other."""
    g, out = tmp('g_ig10'), tmp('n_ig10')
    omission('before', nights=('2025-05-01',))
    F = all_nights(os.path.join(tmp('F_ig10'), 'F'), om='before')
    grouping(g, {LST: F}, om='before')
    st = gate(g, out, view_hours=[LST])
    os.makedirs(out, exist_ok=True)
    from exopipe.phase3.lib.catalog import deep_omission as DO
    earlier = tmp('ig10_earlier')
    grouping(g, {LST: F}, nights=[(LST, earlier), (LST, out)], stokes=('I',), name='Run_SLSTB.group_i', om='before',
             finished='2026-11-24T00:00:05Z')
    grouping(g, {LST: F}, nights=[(LST, out)], stokes=('V',), name='Run_SLSTB.group_v', om='before',
             finished='2026-11-24T00:00:06Z')
    called = []
    with Patch((DO, 'append_night', lambda *a, **k: called.append(1))):
        msg = raises(_Refused, INC._rate_night, _UnitRefusal(), st, out)
    assert msg.startswith('I-G10') and 'only in I' in msg and earlier in msg and not called, msg


def test_a_view_index_and_a_view_root_are_tested_from_their_files():
    """hour_views_index: an all-nights hour whose root is its all-nights directory passes; a view root that does not stand
    for the recorded state, a key without a certificate, an index without tables FAIL. catalog_view: a root without its
    record, or not complete, FAILS."""
    from exopipe.phase3.core import validate
    d = tmp('index')
    F = all_nights(os.path.join(d, 'F'))
    ix = dict(version=1, lst=LST, night=NIGHT, tags=['noveto'], tables={'I': dict(path='/t', sha1='x', sha256='y')},
              hours={LST: dict(kind='all-nights', root=F, all_nights=F, state_digest='s',
                               keys={'noveto_I_sb': dict(kind='all-nights', certificate={'ok': True})})})
    p = os.path.join(d, DB.INDEX)
    with open(p, 'w') as f:
        json.dump(ix, f)
    assert all(x['passed'] for x in validate.run('hour_views_index', p))
    ix['hours']['18h'] = dict(kind='view', root=os.path.join(d, 'views', '18h'), state_digest='s',
                              keys={'noveto_V_sb': dict(kind='delta', certificate=None)})
    ix['tables'] = {}
    with open(p, 'w') as f:
        json.dump(ix, f)
    why = ' '.join(x['detail'] for x in validate.run('hour_views_index', p))
    assert 'does not stand' in why and 'without a certificate: noveto_V_sb' in why and 'no grouping table' in why, why
    root = os.path.join(d, 'views', '17h')
    os.makedirs(root)
    assert 'holds no' in ' '.join(x['detail'] for x in validate.run('catalog_view', root))
    with open(os.path.join(root, DB.ROOT_RECORD), 'w') as f:
        json.dump({'complete': False}, f)
    assert 'not complete' in ' '.join(x['detail'] for x in validate.run('catalog_view', root))


class _ResumeRun:
    """A run whose resume_check answers from a table {unit: (ok, why)} (units absent: no record)."""

    def __init__(self, table):
        self.table, self.units, self.step = table, [], 'catalog'
        self.resume = True

    def resume_check(self, unit, inputs=None, admitted=None):
        return self.table.get(unit, (False, 'no record'))

    def unit(self, name, **meta):
        u = argparse.Namespace(name=name, meta=meta, status=None, reason=None)
        self.units.append(u)
        return u


def _stage_records(out, unit, stages, status='done'):
    for s in stages:
        p = record.record_path(out, 'catalog', f'{unit}.{s}')
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, 'w') as f:
            json.dump({'unit': f'{unit}.{s}', 'status': status}, f)


def test_a_retried_item_resumes_at_its_first_unit_that_does_not_stand():
    """resume_point. Before the grouping holds the night: the first night-pass stage not done (the tier's --stages gap
    check vouches for the one before and restores in-place files; resume_check cannot judge these stages, whose recorded
    master_catalog.csv later stages rewrite), else the grouping. Once it holds the night: the night pass is never run
    again (its stages must all be done, else REFUSED), and group, build, buildv, view are judged by resume_check. The units
    before the resume point are recorded skipped-valid. Without --resume the gate refuses a grouped night."""
    u = f'{LST}_{ARM}'
    units = INC.item_units()
    nd = list(TS.NIGHT_DEFAULT)
    assert units[:len(nd)] == nd and units[len(nd):] == ['omit', 'group', 'build', 'buildv', 'view']
    out = tmp('resume_out')
    r = _ResumeRun({})
    assert INC.resume_point(r, u, out, False)[:2] == (nd[0], f'{u}.{nd[0]} is unrecorded')
    _stage_records(out, u, nd[:3])
    start, why, stood = INC.resume_point(r, u, out, False)
    assert start == nd[3] and [s for s, _ in stood] == nd[:3], (start, why, stood)
    _stage_records(out, u, nd)
    assert INC.resume_point(r, u, out, False)[:2] == ('omit', 'the grouping does not hold the night yet')
    stand = {f'{u}.{s}': (True, 'current') for s in ('build', 'buildv')}
    r = _ResumeRun(stand)                                                    # grouped: resume_check from the grouping on
    start, why, stood = INC.resume_point(r, u, out, True)
    assert start == 'view' and why == 'no record' and [s for s, _ in stood] == units[:-1], (start, why, stood)
    assert 'not run again' in stood[0][1]
    INC._skip(r, u, stood, lst=LST)
    assert [x.status for x in r.units] == ['skipped-valid'] * (len(units) - 1)
    assert INC.resume_point(_ResumeRun({**stand, f'{u}.view': (True, 'current')}), u, out, True)[0] is None
    # finding 4: a grouped night's group stands whatever resume_check would say (never grouped again for no new night)
    start, why, stood = INC.resume_point(_ResumeRun({f'{u}.group': (False, 'output changed')}), u, out, True)
    assert start == 'build' and ('group', 'the grouping holds the night from this directory: never grouped again for no new '
                                          'night') in stood, (start, why, stood)
    _stage_records(out, u, nd[5:6], status='refused')
    msg = raises(record.Refusal, INC.resume_point, _ResumeRun(stand), u, out, True)
    assert f'{u}.{nd[5]} is refused' in msg, msg
    # without --resume, a night directory with an earlier attempt's done night-pass units is refused (not grouped yet)
    g2, out2 = tmp('attempt_g'), tmp('attempt_out')
    grouping(g2, {LST: all_nights(os.path.join(tmp('attempt_F'), 'F'))})
    _stage_records(out2, u, nd[:2])
    _a, r, rc, txt = _run(out2, ['--tier', 'sublst', '--incremental', '--epochs', f'{LST}_{NIGHT}', '--sublst-grouping', g2,
                                 '--sublst-view-hours', LST])
    assert [x.status for x in r.units] == ['refused'] and 'earlier attempt of the item' in r.units[0].reason, \
        [(x.name, x.status, x.reason) for x in r.units]
    # the gate on a night the grouping already holds from this directory
    g, out, F, _p = _night_records('regate')
    argv = ['--tier', 'sublst', '--incremental', '--epochs', f'{LST}_{NIGHT}', '--sublst-grouping', g,
            '--sublst-view-hours', LST]
    _a, r, rc, txt = _run(out, argv)
    assert [x.status for x in r.units] == ['refused'] and 'run the item with --resume' in r.units[0].reason \
        and 'audit pass' in r.units[0].reason, [(x.name, x.status, x.reason) for x in r.units]
    _a, r, rc, txt = _run(out, argv + ['--resume'])
    assert [x.status for x in r.units] == ['refused'] and f'{u}.{nd[0]} is unrecorded' in r.units[0].reason, \
        [(x.name, x.status, x.reason) for x in r.units]
    # finding 5: the items' lock times out after the gate; the refusal is the first unit that would run (build), and the
    # done group record of the night directory is kept
    _stage_records(out, u, nd + ['group'])

    class _Held:
        def __enter__(self):
            raise GS.StageRefusal('held by another item after 0 s of waiting')

        def __exit__(self, *a):
            return False

    def fake(gdir, wait_s, what):
        return contextlib.nullcontext() if '(gate)' in what else _Held()
    with Patch((INC, 'night_lock', fake)):
        _a, r, rc, txt = _run(out, argv + ['--resume'])
    st = [(x.name.split('.')[-1], x.status) for x in r.units]
    assert st[0] == ('frozen-state', 'done') and st[-1] == ('build', 'refused') and ('group', 'skipped-valid') in st, st
    assert record.load_record(out, 'catalog', f'{u}.group')['status'] == 'done'


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
