#!/usr/bin/env python3
"""Tests of exopipe/phase3/lib/catalog/delta_build.py: the file work of a night's delta build of the 10-minute catalog (the
delta build directory and the stage's content test). Synthetic tables only; the builder itself is tested in
tests/test_catalog_frozen_builder.py. Run: python3 tests/test_catalog_delta_build.py"""
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from exopipe.phase3.lib.catalog import delta_build as DB   # noqa: E402

T = '/tmp/exopipe3_tests_catalog_delta_build'
N = [0, 0]
HEAD = 'cand_id,lst,band,date,snr,note'


def check(name, ok, detail=''):
    N[0] += 1
    N[1] += bool(ok)
    print(('PASS  ' if ok else 'FAIL  ') + name + ('' if ok else f'  [{detail}]'), flush=True)


def put(path, lines, term='\n'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        f.write(term.join(lines) + term)


def raw(path):
    with open(path, 'rb') as f:
        return f.read()


def refusal(fn, *a, **k):
    try:
        fn(*a, **k)
    except DB.DeltaRefusal as e:
        return str(e)
    return ''


def row(c, note='x'):
    return f'{c},05h,46MHz,{c.split("_")[1]},7.50,{note}'


def world(root):
    """An all-nights catalog directory A and two night directories N1, N2 (keys V_sb and I_sb; I_wb only in a night)."""
    A, N1, N2 = f'{root}/A', f'{root}/N1', f'{root}/N2'
    put(f'{A}/selected_V_sb.csv', [HEAD, row('V1_2030-01-02T00_b'), row('V1_2030-01-01T00_a', '"q,1"')])
    put(f'{A}/selected_I_sb.csv', [HEAD, row('I1_2030-01-01T00_a')])
    put(f'{A}/v_deep_images.csv', ['lst,scope,date', '05h,46MHz,2030-01-01T00'])
    put(f'{A}/pertie/pertie_detections.csv', ['cand_id,sys', 'V1_2030-01-02T00_b,1.0', 'V1_2030-01-01T00_a,1.1',
                                              'I1_2030-01-01T00_a,1.2'])
    put(f'{A}/leakveto/leakveto_05h_46MHz_V.csv', ['cand_id,veto', 'V1_2030-01-02T00_b,', 'V1_2030-01-01T00_a,'])
    put(f'{N1}/selected_V_sb.csv', [HEAD, row('V1_2030-01-05T03_z'), row('V1_2030-01-05T01_y')])
    put(f'{N1}/pertie/pertie_detections.csv', ['cand_id,sys', 'V1_2030-01-05T03_z,2.0', 'V1_2030-01-05T01_y,2.1'])
    put(f'{N1}/leakveto/leakveto_05h_46MHz_V.csv', ['cand_id,veto', 'V1_2030-01-05T03_z,leak', 'V1_2030-01-05T01_y,'])
    put(f'{N2}/selected_V_sb.csv', [HEAD, row('V1_2030-01-04T02_w')])
    put(f'{N2}/selected_I_sb.csv', [HEAD, row('I1_2030-01-04T02_w')])
    put(f'{N2}/pertie_detections.csv', ['cand_id,sys', 'V1_2030-01-04T02_w,3.0', 'I1_2030-01-04T02_w,3.1'])
    put(f'{N2}/leakveto/leakveto_05h_50MHz_V.csv', ['cand_id,veto', 'V1_2030-01-04T02_w,'])
    return A, N1, N2


def test_merge():
    A, N1, N2 = world(f'{T}/m')
    r = DB.merge(f'{A}/selected_V_sb.csv', [f'{N1}/selected_V_sb.csv', f'{N2}/selected_V_sb.csv'], f'{T}/m/o1.csv', 'selected')
    DB.merge(f'{A}/selected_V_sb.csv', [f'{N2}/selected_V_sb.csv', f'{N1}/selected_V_sb.csv'], f'{T}/m/o2.csv', 'selected')
    lines = raw(f'{T}/m/o1.csv').decode().split('\n')
    check('merge: the all-nights rows keep their bytes and their order, the header is written once',
          '\n'.join(lines[:3]) + '\n' == raw(f'{A}/selected_V_sb.csv').decode(), lines[:3])
    check('merge: the appended rows follow sorted by cand_id',
          [ln.split(',')[0] for ln in lines[3:6]] == ['V1_2030-01-04T02_w', 'V1_2030-01-05T01_y', 'V1_2030-01-05T03_z'], lines[3:])
    check('merge: the result does not depend on the order of the night tables', raw(f'{T}/m/o1.csv') == raw(f'{T}/m/o2.csv'))
    check('merge: it reports the rows per table and the appended detections',
          r['base'] == 2 and sorted(r['nights'].values()) == [1, 2] and len(r['appended']) == 3, r)
    check('merge: no partial file is left', not os.path.exists(f'{T}/m/o1.csv.part'))
    put(f'{T}/m/crlf.csv', [HEAD, row('V1_2030-01-01T00_a')], term='\r\n')
    DB.merge(f'{T}/m/crlf.csv', [f'{N2}/selected_V_sb.csv'], f'{T}/m/o3.csv', 'selected')
    check('merge: a table with CRLF line ends stays CRLF', raw(f'{T}/m/o3.csv').count(b'\r\n') == 3
          and raw(f'{T}/m/o3.csv').count(b'\n') == 3, raw(f'{T}/m/o3.csv'))
    put(f'{T}/m/other_head.csv', [HEAD + ',more', row('V1_2030-01-09T00_h') + ',1'])
    msg = refusal(DB.merge, f'{A}/selected_V_sb.csv', [f'{T}/m/other_head.csv'], f'{T}/m/o4.csv', 'selected')
    check('merge: a night table with another header refuses', 'header' in msg and msg.endswith('REFUSING'), msg)
    check('merge: a refusal writes nothing', not os.path.exists(f'{T}/m/o4.csv'))
    put(f'{T}/m/again.csv', [HEAD, row('V1_2030-01-02T00_b')])
    msg = refusal(DB.merge, f'{A}/selected_V_sb.csv', [f'{T}/m/again.csv'], f'{T}/m/o5.csv', 'selected')
    check('merge: a detection that is already in the all-nights table refuses (a night is appended once)',
          'V1_2030-01-02T00_b' in msg and 'appended once' in msg, msg)
    msg = refusal(DB.merge, f'{A}/selected_V_sb.csv', [f'{N1}/selected_V_sb.csv', f'{N1}/selected_V_sb.csv'], f'{T}/m/o6.csv', 's')
    check('merge: the same night table twice refuses', 'appended once' in msg, msg)
    put(f'{T}/m/multi.csv', [HEAD, 'V1_2030-01-08T00_m,05h,46MHz,d,7.5,"two', 'lines"'])
    msg = refusal(DB.merge, f'{A}/selected_V_sb.csv', [f'{T}/m/multi.csv'], f'{T}/m/o7.csv', 'selected')
    check('merge: a field that spans lines refuses', 'spans lines' in msg, msg)
    put(f'{T}/m/short.csv', [HEAD, 'V1_2030-01-08T00_s,05h'])
    msg = refusal(DB.merge, f'{A}/selected_V_sb.csv', [f'{T}/m/short.csv'], f'{T}/m/o8.csv', 'selected')
    check('merge: a line with another number of fields refuses', 'number of fields' in msg, msg)
    put(f'{T}/m/noid.csv', ['a,b', '1,2'])
    check('merge: a table without a cand_id column refuses', 'cand_id' in refusal(DB.merge, f'{T}/m/noid.csv', [], f'{T}/m/o9.csv', 's'))
    open(f'{T}/m/empty.csv', 'w').close()
    check('merge: an empty table refuses', 'empty' in refusal(DB.merge, f'{T}/m/empty.csv', [], f'{T}/m/o10.csv', 's'))


def test_assemble():
    A, N1, N2 = world(f'{T}/a')
    B = f'{T}/a/build'
    os.makedirs(f'{B}/stale')
    r = DB.assemble(B, A, [N1, N2], ('V_sb', 'I_sb', 'V_wb', 'I_wb'), 'noveto')
    check('assemble: an existing build directory is replaced', not os.path.exists(f'{B}/stale'))
    check('assemble: one selected file per key of the all-nights directory; a key no directory holds is left out',
          sorted(r['selected']) == ['I_sb', 'V_sb'] and sorted(os.listdir(B)) == ['pertie', 'selected_I_sb.csv',
                                                                                 'selected_V_sb.csv', 'v_deep_images.csv'],
          os.listdir(B))
    check('assemble: the selected rows are the all-nights rows plus every night\'s',
          r['selected']['V_sb']['base'] == 2 and len(r['selected']['V_sb']['appended']) == 3
          and r['selected']['I_sb']['appended'] == ['I1_2030-01-04T02_w'], r['selected'])
    check('assemble: the image table is the all-nights table, byte for byte',
          raw(f'{B}/v_deep_images.csv') == raw(f'{A}/v_deep_images.csv'))
    check('assemble: the window-tie table holds the all-nights rows and the nights\' (either layout of a directory)',
          r['pertie']['base'] == 3 and len(r['pertie']['appended']) == 4, r['pertie'])
    check('assemble: the build without the leak veto gets no leak-veto tables', not os.path.exists(f'{B}/leakveto')
          and r['leakveto'] == {} and r['vetoed'] == [])
    B2 = f'{T}/a/build2'
    r2 = DB.assemble(B2, A, [N2, N1], ('V_sb', 'I_sb'), 'noveto')
    same = all(raw(f'{B}/{x}') == raw(f'{B2}/{x}') for x in ('selected_V_sb.csv', 'selected_I_sb.csv',
                                                            'pertie/pertie_detections.csv'))
    check('assemble: the build directory does not depend on the order of the night directories', same and r2['pertie']['base'] == 3)
    Bv = f'{T}/a/buildv'
    rv = DB.assemble(Bv, A, [N1, N2], ('V_sb', 'I_sb'), 'veto')
    check('assemble (veto): the leak-veto tables are merged per file, a cell only a night has included',
          sorted(os.listdir(f'{Bv}/leakveto')) == ['leakveto_05h_46MHz_V.csv', 'leakveto_05h_50MHz_V.csv']
          and rv['leakveto']['leakveto_05h_46MHz_V.csv']['base'] == 2
          and rv['leakveto']['leakveto_05h_46MHz_V.csv']['appended'] == ['V1_2030-01-05T01_y', 'V1_2030-01-05T03_z'], rv['leakveto'])
    check('assemble (veto): it reports the detections the merged tables veto', rv['vetoed'] == ['V1_2030-01-05T03_z'], rv['vetoed'])
    shutil.rmtree(f'{A}/leakveto')
    msg = refusal(DB.assemble, f'{T}/a/b3', A, [N1], ('V_sb',), 'veto')
    check('assemble (veto): an all-nights directory without leak-veto tables refuses', 'leak-veto' in msg, msg)
    msg = refusal(DB.assemble, f'{T}/a/b4', A, [N1, N1], ('V_sb',), 'noveto')
    check('assemble: a night directory named twice refuses', 'twice' in msg, msg)
    msg = refusal(DB.assemble, f'{T}/a/b5', A, [A], ('V_sb',), 'noveto')
    check('assemble: the all-nights directory as a night directory refuses', 'named as a night' in msg, msg)
    put(f'{N1}/selected_I_wb.csv', [HEAD, row('I1_2030-01-05T03_wb')])
    msg = refusal(DB.assemble, f'{T}/a/b6', A, [N1], ('V_sb', 'I_wb'), 'noveto')
    check('assemble: night rows of a key the all-nights build made no catalog for refuse',
          'selected_I_wb.csv' in msg and 'build all nights again' in msg, msg)
    put(f'{N1}/selected_I_wb.csv', [HEAD])
    check('assemble: a night table of such a key with no row is passed over',
          sorted(DB.assemble(f'{T}/a/b7', A, [N1], ('V_sb', 'I_wb'), 'noveto')['selected']) == ['V_sb'])
    os.remove(f'{A}/v_deep_images.csv')
    check('assemble: a missing all-nights image table refuses',
          'image table' in refusal(DB.assemble, f'{T}/a/b8', A, [N1], ('V_sb',), 'noveto'))
    put(f'{A}/v_deep_images.csv', ['lst,scope,date'])
    shutil.rmtree(f'{N1}/pertie')
    check('assemble: a night directory without a window-tie table refuses',
          'window-tie' in refusal(DB.assemble, f'{T}/a/b9', A, [N1], ('V_sb',), 'noveto'))


def catalog(d, touched, masters, members, removed=(), record=True):
    """A delta catalog directory: touched = [(band, source)], masters = [(band, source)], members = [(band, source, cand)]."""
    os.makedirs(d, exist_ok=True)
    if record:
        with open(f'{d}/builder_delta.json', 'w') as f:
            json.dump(dict(new_detections=2, touched=[dict(identity_id=f'GTI-{i}', kind='new', sources=[[b, s, 1]])
                                                      for i, (b, s) in enumerate(touched)]), f)
    for band in sorted({b for b, _s in masters} | {b for b, _s, _c in members}):
        put(f'{d}/{band}_sources_master.csv', ['source_id,best_snr'] + [f'{s},8.0' for b, s in masters if b == band])
        put(f'{d}/{band}_source_members.csv', ['source_id,cand_id'] + [f'{s},{c}' for b, s, c in members if b == band])
    if removed:
        put(f'{d}/46MHz_sigma_stack_singleton_removed.csv', ['cand_id,why'] + [f'{c},single' for c in removed])


def test_content():
    D = f'{T}/c'
    catalog(f'{D}/ok', [('46MHz', 'J1'), ('50MHz', 'J2')], [('46MHz', 'J1'), ('50MHz', 'J2')],
            [('46MHz', 'J1', 'n1'), ('46MHz', 'J1', 'old1'), ('50MHz', 'J2', 'n2'), ('46MHz', 'J9', 'n4')], removed=['n3'])
    ok, detail, c = DB.content_test(f'{D}/ok', ['n1', 'n2', 'n3', 'n4'])
    check('content test: members of a delta source, a removed table and a source removed before the master table all count',
          ok and c['night_members'] == 3 and c['night_removed'] == 1 and c['sources'] == 2 and c['in_neither'] == 0, detail)
    ok, detail, c = DB.content_test(f'{D}/ok', ['n1', 'n2', 'n3', 'n4', 'n5'])
    check('content test: a selected detection of the night in neither a source nor a removed table fails',
          not ok and c['in_neither'] == 1 and 'n5' in detail, detail)
    ok, detail, c = DB.content_test(f'{D}/ok', ['n1', 'n2', 'n3', 'n4', 'n5'], vetoed=['n5'])
    check('content test (veto): a leak-vetoed detection of the night is a member of no source and passes',
          ok and c['night_vetoed'] == 1, detail)
    ok, detail, c = DB.content_test(f'{D}/ok', ['n1', 'n2'], vetoed=['n2'])
    check('content test (veto): a leak-vetoed detection that is a member fails', not ok and c['vetoed_but_member'] == 1, detail)
    catalog(f'{D}/untouched', [('46MHz', 'J1')], [('46MHz', 'J1'), ('46MHz', 'J7')], [('46MHz', 'J1', 'n1')])
    ok, detail, c = DB.content_test(f'{D}/untouched', ['n1'])
    check('content test: a source of no touched identity in a delta fails', not ok and c['untouched_sources'] == 1
          and '46MHz J7' in detail, detail)
    catalog(f'{D}/two', [('46MHz', 'J1'), ('46MHz', 'J2')], [('46MHz', 'J1'), ('46MHz', 'J2')],
            [('46MHz', 'J1', 'n1'), ('46MHz', 'J2', 'n1')])
    ok, detail, c = DB.content_test(f'{D}/two', ['n1'])
    check('content test: a detection that is a member of two sources fails', not ok and c['in_two_sources'] == 1, detail)
    catalog(f'{D}/norec', [], [('46MHz', 'J1')], [('46MHz', 'J1', 'n1')], record=False)
    ok, detail, c = DB.content_test(f'{D}/norec', ['n1'])
    check('content test: a catalog directory without the delta record fails', not ok and 'builder_delta.json' in detail, detail)
    catalog(f'{D}/empty', [], [], [])
    ok, detail, c = DB.content_test(f'{D}/empty', [])
    check('content test: a delta without a touched identity and a night without a selected detection pass', ok and c['sources'] == 0, detail)


def test_own_delta():
    """tier_sublst._own_delta: a night's delta directory of a key is replaced (or removed) only when empty, finished
    (builder_delta.json) or left by a failed pass (the started marker); anything else refuses."""
    from exopipe.phase3.lib.catalog import tier_sublst as TS
    from exopipe.phase3.core.record import Refusal
    base = f'{T}/own'
    def verdict(files):
        d = f'{base}/{len(files)}_{"_".join(f.strip(".") for f in files)}'
        os.makedirs(d)
        for f in files:
            put(f'{d}/{f}', ['x'])
        try:
            TS._own_delta(d, 'I_wb', 'noveto')
        except Refusal as e:
            return str(e)
        return ''
    check('own delta: an empty directory, a finished delta and a failed pass\'s are this stage\'s own',
          verdict([]) == '' and verdict([DB.DELTA_RECORD, 'Red_source_members.csv']) == ''
          and verdict([TS.DELTA_STARTED, 'Red_source_members.csv']) == '')
    check('own delta: a directory with other files only refuses', 'not replaced' in verdict(['Red_source_members.csv']))


def test_recorded_inputs():
    """grouping_stage.recorded_inputs: what the latest done grouping run was assembled from (the nights a delta build appends)."""
    from exopipe.phase3.lib.catalog import grouping_stage as GS
    G = f'{T}/g'
    table = f'{G}/grouping_V/{GS.DETECTIONS}'
    put(table, ['cand_id,identity_id', 'c1,GTI-1'])

    def record(name, finished, status='done', nights=(('05h', '/n/1'), ('06h', '/n/2')), inp=None, sha=None, stokes=('V',)):
        os.makedirs(f'{G}/_records/catalog', exist_ok=True)
        meta = dict(stage=GS.STAGE, stokes=list(stokes), input={'05h': '/all/05h'} if inp is None else inp,
                    tables={'V': dict(path=table, sha1=sha or GS.table_digest(table))})
        if nights:
            meta['input_nights'] = [list(x) for x in nights]
        with open(f'{G}/_records/catalog/{name}.json', 'w') as f:
            json.dump(dict(status=status, finished=finished, meta=meta), f)
    record('a', '2030-01-01T00:00:00Z', nights=())
    runs, nd = GS.recorded_inputs(G, 'V')
    check('recorded inputs: an all-nights grouping run names its all-nights directories and no night',
          runs == {'05h': '/all/05h'} and nd == [], (runs, nd))
    record('b', '2030-01-02T00:00:00Z')
    runs, nd = GS.recorded_inputs(G, 'V')
    check('recorded inputs: the LATEST run counts, with the nights it was run with', nd == [('05h', '/n/1'), ('06h', '/n/2')], nd)
    check('recorded inputs: the table and sha1 are those of the same record', GS.recorded_table(G, 'V') == (table, GS.table_digest(table)))
    r3 = GS.recorded_inputs(G, 'V', with_table=True)
    check('recorded inputs: with_table gives the nights and the table of one read of the record',
          r3 == (runs, nd, (table, GS.table_digest(table))), r3)
    record('c', '2030-01-03T00:00:00Z', inp='/a/direct/input.csv')
    check('recorded inputs: a run on a direct input names no LST directory', GS.recorded_inputs(G, 'V')[0] == {})

    def refused(fn):
        try:
            fn(G, 'V')
        except GS.StageRefusal as e:
            return str(e)
        return ''
    record('d', '2030-01-04T00:00:00Z', status='refused')
    check('recorded inputs: a latest run that did not end done refuses', 'REFUSING' in refused(GS.recorded_inputs))
    record('e', '2030-01-05T00:00:00Z', sha='0' * 40)
    check('recorded inputs: a table changed since its run refuses', 'changed since' in refused(GS.recorded_inputs))
    check('recorded inputs: no record of the Stokes refuses',
          'no record' in (lambda: refused(lambda g, s_: GS.recorded_inputs(g, 'I')))())


# ---- the catalog view of an LST hour (design 2.3a, stages 1 and 2) -----------------------------------------------------
GHEAD = 'cand_id,identity_id,identity_name,multiples_id,multiples_name,lst,date,multiple,multiplicity'
SV = 'rank,source_id,identity_id,band,best_cand_id,best_snr,review_dir,note'
MS = 'source_id,identity_id,band,best_cand_id,best_snr,review_dir,note'


def grouping(path, rows):
    """rows: (cand_id, identity_id, identity_name, multiples_id)"""
    put(path, [GHEAD] + [f'{c},{i},{n},{m},{m}n,05h,2030-01-01T00,0,1' for c, i, n, m in rows])
    return DB.read_grouping(path)


def digests(g):
    from exopipe.phase3.lib.catalog.wraps import builder_grouping as BG
    T, mg = g[0], g[1]
    return {c: [T[c]['identity_id'], BG.group_digest_of(mg[T[c]['multiples_id']])] for c in T}


def src(name, ident, cand, snr, rank='', note='x'):
    rev = f'candidates/46MHz/{rank}_46MHz_{name}' if rank else ''
    return ((f'{rank},' if rank is not None else '') + f'{name},{name},46MHz,{cand},{snr},{rev},{note}')


def view_world(root):
    """An all-nights catalog directory A of one key under the table OLD, and a delta D of a night under the table NEW: the
    night adds n1 (a new identity G6, tied in S/N with two earlier sources) and n2 (joins G3, which is renamed J0003 -> J0033);
    G2 shares G3's multiples-level group, so it is regrouped. G1, G4, G5 (a removed singleton) and G7 are untouched."""
    A, D = f'{root}/A/source_catalogs_noveto_V_sb', f'{root}/N/source_catalogs_noveto_V_sb'
    old = [('c1', 'G1', 'J0001', 'M1'), ('c2', 'G2', 'J0002', 'M2'), ('c3', 'G3', 'J0003', 'M2'), ('c4', 'G4', 'J0004', 'M4'),
           ('c5', 'G5', 'J0005', 'M5'), ('c7', 'G7', 'J0007', 'M7')]
    new = [r if r[0] != 'c3' else ('c3', 'G3', 'J0033', 'M2') for r in old] + [('n1', 'G6', 'J0006', 'M6'), ('n2', 'G3', 'J0033', 'M2')]
    gold, gnew = grouping(f'{root}/old/detections.csv', old), grouping(f'{root}/new/detections.csv', new)
    put(f'{A}/46MHz_sources_survivors.csv', [SV, src('J0001', 'G1', 'c1', '12.00', 'rank0001', '"q,1"'),
                                             src('J0002', 'G2', 'c2', '9.50', 'rank0002'), src('J0007', 'G7', 'c7', '9.50', 'rank0003'),
                                             src('J0003', 'G3', 'c3', '8.00', 'rank0004')], term='\r\n')
    put(f'{A}/46MHz_sources_master.csv', [MS, src('J0001', 'G1', 'c1', '12.00', 'rank0001', '"q,1"')[9:],
                                          src('J0002', 'G2', 'c2', '9.50', 'rank0002')[9:], src('J0007', 'G7', 'c7', '9.50', 'rank0003')[9:],
                                          src('J0003', 'G3', 'c3', '8.00', 'rank0004')[9:], src('J0004', 'G4', 'c4', '6.00', None)],
        term='\r\n')
    put(f'{A}/46MHz_source_members.csv', ['source_id,band,cand_id', 'J0001,46MHz,c1', 'J0002,46MHz,c2', 'J0007,46MHz,c7',
                                          'J0003,46MHz,c3', 'J0004,46MHz,c4'], term='\r\n')
    put(f'{A}/46MHz_sigma_stack_singleton_removed.csv', ['cand_id,lst', 'c5,05h'], term='\r\n')
    put(f'{A}/E_params.json', ['{"frozen": 1}'])
    put(f'{A}/grouping_table.json', [json.dumps(dict(table=f'{root}/old/detections.csv', table_sha1=gold[2], tier='10min'))])
    put(f'{A}/builder_frozen.json', [json.dumps(dict(version=2, grouping_table=dict(path=f'{root}/old/detections.csv', sha1=gold[2]),
                                                     rows=digests(gold)))])
    put(f'{D}/46MHz_sources_survivors.csv', [SV, src('J0002', 'G2', 'c2', '9.50', ''), src('J0033', 'G3', 'n2', '10.00', ''),
                                             src('J0006', 'G6', 'n1', '9.50', '')], term='\r\n')
    put(f'{D}/46MHz_sources_master.csv', [MS, src('J0002', 'G2', 'c2', '9.50', None), src('J0033', 'G3', 'n2', '10.00', None),
                                          src('J0006', 'G6', 'n1', '9.50', None)], term='\r\n')
    put(f'{D}/46MHz_source_members.csv', ['source_id,band,cand_id', 'J0002,46MHz,c2', 'J0033,46MHz,c3', 'J0033,46MHz,n2',
                                          'J0006,46MHz,n1'], term='\r\n')
    put(f'{D}/46MHz_sigma_stack_singleton_removed.csv', ['cand_id,lst'], term='\r\n')
    put(f'{D}/grouping_table.json', [json.dumps(dict(table=f'{root}/new/detections.csv', table_sha1=gnew[2], tier='10min'))])
    touched = [dict(identity_id='G2', identity_name='J0002', replaces=[['46MHz', 'J0002']], sources=[['46MHz', 'J0002', 1]]),
               dict(identity_id='G3', identity_name='J0033', replaces=[['46MHz', 'J0003']], sources=[['46MHz', 'J0033', 1]]),
               dict(identity_id='G6', identity_name='J0006', replaces=[], sources=[['46MHz', 'J0006', 1]])]
    dl = dict(version=2, grouping_table=dict(path=f'{root}/new/detections.csv', sha1=gnew[2]), touched=touched,
              frozen_record_sha1=DB._sha1(f'{A}/builder_frozen.json'),
              rows={c: v for c, v in digests(gnew).items() if c not in ('c5',)})
    put(f'{D}/builder_delta.json', [json.dumps(dl)])
    return A, D, gold, gnew, dl


def csvrows(path):
    import csv
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def test_view():
    R = f'{T}/view'
    A, D, gold, gnew, dl = view_world(R)
    told, tnew = f'{R}/old/detections.csv', f'{R}/new/detections.csv'
    frz = f'{A}/builder_frozen.json'
    # the certificate
    c_own = DB.certificate(frz, gold, told, lst='05h', key='V_sb')
    check('certificate: a build under its own table is untouched', c_own['ok'] is True and c_own['differing'] == 0
          and c_own['detections'] == 6 and c_own['table_to']['sha1'] == gold[2] and len(c_own['table_to']['sha256']) == 64
          and c_own['lst'] == '05h' and c_own['key'] == 'V_sb' and c_own['record_sha1'] == DB._sha1(frz), c_own)
    c_new = DB.certificate(frz, gnew, tnew)
    check('certificate: a regroup of the hour\'s detections is not ok (the renamed identity and its group\'s other identity)',
          c_new['ok'] is False and c_new['differing'] == 2 and {e[0] for e in c_new['examples']} == {'c2', 'c3'}, c_new)
    gone = grouping(f'{R}/gone/detections.csv', [('c1', 'G1', 'J0001', 'M1')])
    c = DB.certificate(frz, gone, f'{R}/gone/detections.csv')
    check('certificate: a detection missing from the table is not ok', c['ok'] is False and c['differing'] == 5, c)
    moved = grouping(f'{R}/moved/detections.csv', [('c1', 'G9', 'J0001', 'M1')] + [r for r in
                     [('c2', 'G2', 'J0002', 'M2'), ('c3', 'G3', 'J0003', 'M2'), ('c4', 'G4', 'J0004', 'M4'),
                      ('c5', 'G5', 'J0005', 'M5'), ('c7', 'G7', 'J0007', 'M7')]])
    c = DB.certificate(frz, moved, f'{R}/moved/detections.csv')
    check('certificate: a detection under another identity is not ok', c['ok'] is False and c['examples'][0][0] == 'c1'
          and 'G1 -> G9' in c['examples'][0][1], c)
    put(f'{R}/norows.json', [json.dumps(dict(version=2))])
    c = DB.certificate(f'{R}/norows.json', gold, told)
    check('certificate: a record without per-detection grouping is not ok', c['ok'] is False and 'rows' in c.get('why', ''), c)
    c_dl = DB.certificate(f'{D}/builder_delta.json', gnew, tnew)
    check('certificate: a delta record under the table it was built with is untouched', c_dl['ok'] is True and c_dl['detections'] == 7, c_dl)
    # a view without a delta
    V0 = f'{R}/V0'
    rec = DB.view_catalog(V0, A, None, gold, told, cert=c_own)
    frz = DB._sha1(f'{A}/builder_frozen.json')
    check('view without a delta: the record names the all-nights build it stands on (the frozen record\'s sha1)',
          rec['frozen_record_sha1'] == frz and json.load(open(f'{V0}/catalog_view.json'))['frozen_record_sha1'] == frz, rec)
    names = sorted(n for n in os.listdir(A) if n.endswith('.csv'))
    check('view without a delta: every table byte-identical to the all-nights table (the builder\'s order and numbering)',
          all(raw(f'{V0}/{n}') == raw(f'{A}/{n}') for n in names) and len(names) == 4)
    prov = json.load(open(f'{V0}/grouping_table.json'))
    check('view without a delta: the provenance names the table and carries the certificate; the record says certificate',
          prov['table_sha1'] == gold[2] and prov['view']['kind'] == 'certificate' and prov['view']['certificate']['ok'] is True
          and rec['kind'] == 'certificate' and rec['rows_replaced'] == 0 and raw(f'{V0}/E_params.json') == raw(f'{A}/E_params.json')
          and prov['tier'] == '10min' and not os.path.exists(f'{V0}/builder_frozen.json'), prov)
    ok, detail, counts = DB.view_test(V0, A, None, gold)
    check('view without a delta: content test passes', ok and counts['survivors'] == 4, detail)
    for what, cert in (('no certificate', None), ('a certificate that is not ok', c_new), ('a certificate for another table', c_own),
                       ('a certificate of another record', dict(c_own, table_to=dict(sha1=gnew[2]), record_sha1='0' * 40))):
        why = refusal(DB.view_catalog, f'{R}/Vx', A, None, gnew, tnew, cert=cert)
        check(f'view without a delta: {what} refuses', 'REFUSING' in why and not os.path.exists(f'{R}/Vx'), why)
    # the view with the night's delta
    V1 = f'{R}/V1'
    rec = DB.view_catalog(V1, A, D, gnew, tnew)
    check('view: the record names the all-nights build the delta stands on (the frozen record\'s sha1)',
          rec['frozen_record_sha1'] == frz and json.load(open(f'{V1}/catalog_view.json'))['frozen_record_sha1'] == frz, rec)
    sv = csvrows(f'{V1}/46MHz_sources_survivors.csv')
    check('view: survivors in falling |best S/N|; equal values keep the all-nights order, delta rows after them',
          [r['source_id'] for r in sv] == ['J0001', 'J0033', 'J0007', 'J0002', 'J0006'], [r['source_id'] for r in sv])
    check('view: ranks count from 1 and review_dir is the builder\'s form',
          [r['rank'] for r in sv] == [f'rank000{n}' for n in range(1, 6)]
          and all(r['review_dir'] == f'candidates/46MHz/{r["rank"]}_46MHz_{r["source_id"]}' for r in sv))
    check('view: an untouched row keeps every other value (a quoted field too)', sv[0]['note'] == 'q,1' and sv[2]['best_cand_id'] == 'c7'
          and b'"q,1"' in raw(f'{V1}/46MHz_sources_survivors.csv') and raw(f'{V1}/46MHz_sources_survivors.csv').count(b'\r\n') == 6)
    ms = csvrows(f'{V1}/46MHz_sources_master.csv')
    rev = {r['source_id']: r['review_dir'] for r in sv}
    check('view: the master table holds every source once, with its survivors row\'s review_dir or none',
          [r['source_id'] for r in ms] == ['J0001', 'J0033', 'J0007', 'J0002', 'J0006', 'J0004']
          and all(r['review_dir'] == rev.get(r['source_id'], '') for r in ms), ms)
    mem = sorted((r['source_id'], r['cand_id']) for r in csvrows(f'{V1}/46MHz_source_members.csv'))
    check('view: member rows of the replaced sources are gone, the delta\'s are in',
          mem == [('J0001', 'c1'), ('J0002', 'c2'), ('J0004', 'c4'), ('J0006', 'n1'), ('J0007', 'c7'), ('J0033', 'c3'), ('J0033', 'n2')], mem)
    check('view: a detection-keyed removed table keeps the rows of untouched identities',
          [r['cand_id'] for r in csvrows(f'{V1}/46MHz_sigma_stack_singleton_removed.csv')] == ['c5'])
    prov = json.load(open(f'{V1}/grouping_table.json'))
    check('view: the record and the provenance name the present table and the delta',
          rec['kind'] == 'delta' and rec['touched_identities'] == 3 and rec['sources_replaced'] == 2 and rec['rows_delta'] == 10
          and prov['table_sha1'] == gnew[2] and prov['view']['kind'] == 'delta'
          and prov['view']['delta_record_sha1'] == DB._sha1(f'{D}/builder_delta.json')
          and json.load(open(f'{V1}/catalog_view.json'))['grouping_table']['sha256'] == gnew[3], rec)
    ok, detail, counts = DB.view_test(V1, A, D, gnew)
    check('view: content test passes', ok and counts['survivors'] == 5 and counts['rows_from_delta'] == 10, detail)
    # the content test catches a view that is not the merge
    def tamper(name, fn, what):
        W = f'{R}/W'
        shutil.rmtree(W, ignore_errors=True)
        shutil.copytree(V1, W)
        lines = raw(f'{W}/{name}').decode().split('\r\n')
        with open(f'{W}/{name}', 'w', newline='') as f:
            f.write('\r\n'.join(fn(lines)))
        ok, detail, _c = DB.view_test(W, A, D, gnew)
        check(f'view content test: {what} fails', not ok and 'FAILED' in detail, detail)
    tamper('46MHz_sources_survivors.csv', lambda L: [L[0], L[1].replace('12.00', '13.00')] + L[2:], 'a changed value of an untouched row')
    tamper('46MHz_sources_survivors.csv', lambda L: L[:2] + L[3:], 'a missing row')
    tamper('46MHz_sources_survivors.csv', lambda L: [L[0], L[2], L[1]] + L[3:], 'rows out of S/N order')
    tamper('46MHz_sources_master.csv', lambda L: [L[0], L[1].replace('rank0001', 'rank0009')] + L[2:], 'a master review_dir that is not the survivors row\'s')
    tamper('46MHz_source_members.csv', lambda L: L[:-1] + ['J0001,46MHz,n2', ''], 'a member detection in two rows')
    # refusals of the merge
    def world2(edit):
        R2 = f'{T}/view2'
        shutil.rmtree(R2, ignore_errors=True)
        A2, D2, go, gn, dl2 = view_world(R2)
        edit(A2, D2, dl2, R2)
        return refusal(DB.view_catalog, f'{R2}/V', A2, D2, DB.read_grouping(f'{R2}/new/detections.csv'), f'{R2}/new/detections.csv')

    def redelta(D2, dl2):
        put(f'{D2}/builder_delta.json', [json.dumps(dl2)])
    why = world2(lambda A2, D2, dl2, R2: redelta(D2, dict(dl2, grouping_table=dict(path='x', sha1='0' * 40))))
    check('view: a delta built with another grouping table refuses', 'build the delta again' in why, why)
    why = world2(lambda A2, D2, dl2, R2: put(f'{A2}/builder_frozen.json', ['{"version": 2, "rows": {}}']))
    check('view: a delta that does not stand on this all-nights build refuses', 'does not stand on' in why, why)
    why = world2(lambda A2, D2, dl2, R2: put(f'{D2}/46MHz_source_members.csv', ['source_id,band,cand_id', 'J0001,46MHz,c1'], term='\r\n'))
    check('view: a delta row of an untouched identity refuses', 'of no touched identity' in why, why)
    why = world2(lambda A2, D2, dl2, R2: put(f'{D2}/46MHz_source_members.csv', ['source_id,band,cand_id,extra'], term='\r\n'))
    check('view: a delta table with another header refuses', 'header differs' in why, why)
    why = world2(lambda A2, D2, dl2, R2: redelta(D2, dict(dl2, touched=[dict(t, replaces=[]) for t in dl2['touched']])))
    check('view: replaced sources that are not the ones the delta record names refuse', 'not the ones the delta' in why, why)
    why = world2(lambda A2, D2, dl2, R2: put(f'{A2}/46MHz_sigma_stack_singleton_removed.csv', ['cand_id,lst', 'c5,05h', 'zz,05h'], term='\r\n'))
    check('view: an all-nights detection that the grouping table does not hold refuses', 'stale' in why, why)
    why = world2(lambda A2, D2, dl2, R2: put(f'{A2}/46MHz_sigma_stack_singleton_removed.csv', ['cand_id,lst', '"c5",05h'], term='\r\n'))
    check('view: a table that would not be written back to the same bytes refuses', 'same bytes' in why, why)
    why = world2(lambda A2, D2, dl2, R2: put(f'{A2}/46MHz_notes.csv', ['source,text', 'a,b']))
    check('view: a table without a detection column refuses', 'cannot place by detection' in why, why)
    why = world2(lambda A2, D2, dl2, R2: os.remove(f'{D2}/builder_delta.json'))
    check('view: a delta directory without its record refuses', 'not a delta build' in why, why)


# ---- the state of an hour, its view root and the pass's index (design 2.3a, stages 1 and 2) ------------------------------
SEL = 'cand_id,lst,band,snr'
IMG = 'lst,scope,stokes,date,n_final'
OLD_ROWS = [('c1', 'G1', 'J0001', 'M1'), ('c2', 'G2', 'J0002', 'M2'), ('c3', 'G3', 'J0003', 'M2'), ('c4', 'G4', 'J0004', 'M4'),
            ('c5', 'G5', 'J0005', 'M5'), ('c7', 'G7', 'J0007', 'M7')]


def hour_world(root):
    """view_world plus the directory-level tables of the hour: the all-nights directory <root>/A (selected rows, window-tie
    table, image table, detection table, localizations) and the night directory <root>/N (its two selected rows)."""
    A, D, gold, gnew, dl = view_world(root)
    F, Nd = f'{root}/A', f'{root}/N'
    put(f'{F}/selected_V_sb.csv', [SEL] + [f'{c},05h,46MHz,7.5' for c in ('c1', 'c2', 'c3', 'c4', 'c5', 'c7')])
    put(f'{F}/selected_V_wb.csv', [SEL])
    put(f'{F}/pertie/pertie_detections.csv', ['cand_id,sys'] + [f'{c},1.0' for c in ('c1', 'c2', 'c3', 'c4', 'c5', 'c7')])
    put(f'{F}/v_deep_images.csv', [IMG, '05h,46MHz,V,2030-01-01T00,3', '05h,46MHz,V,2030-01-02T00,3'])
    put(f'{F}/master_catalog.csv', [SEL] + [f'{c},05h,46MHz,7.5' for c in ('c1', 'c2', 'c3', 'c4', 'c5', 'c7')])
    put(f'{F}/localizations_noveto/localizations.csv', ['identity_id,ra,dec', 'J0001,1.0,1.0', 'J0002,2.0,2.0', 'J0003,3.0,3.0',
                                                        'J0007,7.0,7.0'], term='\r\n')
    put(f'{F}/localizations_noveto/localization_offsets.csv', ['cand_id,identity_id,sep', 'c1,J0001,0.1', 'c2,J0002,0.2',
                                                               'c3,J0003,0.3', 'c7,J0007,0.7'], term='\r\n')
    put(f'{Nd}/selected_V_sb.csv', [SEL, 'n2,05h,46MHz,10.0', 'n1,05h,46MHz,9.5'])
    put(f'{Nd}/pertie/pertie_detections.csv', ['cand_id,sys', 'n2,2.0', 'n1,2.1'])
    put(f'{Nd}/v_deep_images.csv', [IMG, '05h,46MHz,V,2030-01-05T01,2', '05h,46MHz,V,2030-01-05T00,2'])
    put(f'{Nd}/master_catalog.csv', [SEL, 'n2,05h,46MHz,10.0', 'n1,05h,46MHz,9.5'])
    return F, Nd, D, gold, gnew


def test_hour():
    R = f'{T}/hour'
    F, Nd, D, gold, gnew = hour_world(R)
    told, tnew = f'{R}/old/detections.csv', f'{R}/new/detections.csv'
    keys = ('V_sb', 'V_wb', 'I_sb')
    # what a build reads, without writing
    inp0, inp = DB.inputs(F, [], keys, 'noveto'), DB.inputs(F, [Nd], keys, 'noveto')
    asm = DB.assemble(f'{R}/B', F, [Nd], keys, 'noveto')
    check('inputs: the digest of what a build would read is the digest of the build directory assemble writes, nothing written',
          inp['V_sb']['digest'] == asm['digest']['V_sb'] and sorted(inp) == ['V_sb', 'V_wb'] and not os.path.exists(f'{F}/_side')
          and inp['V_sb']['appended'] == ['n1', 'n2'] and inp['V_sb']['base'] == 6 and inp['V_wb']['appended'] == [], inp)
    check('inputs: the digest changes with the appended nights, and with the build tag',
          inp0['V_sb']['digest'] != inp['V_sb']['digest'] and len(inp['V_sb']['digest']) == 40
          and inp['V_sb']['digest'] != inp['V_wb']['digest'])
    put(f'{R}/N3/selected_V_sb.csv', [SEL])
    put(f'{R}/N3/pertie/pertie_detections.csv', ['cand_id,sys', 'x9,1.0'])
    check('inputs: a night that adds only a window-tie row changes the digest (the build reads the whole table)',
          DB.inputs(F, [Nd, f'{R}/N3'], keys, 'noveto')['V_sb']['digest'] != inp['V_sb']['digest'])
    # the state of a key
    st = DB.key_state('05h', 'V_sb', 'noveto', F, inp0['V_sb'], [], gold, told)
    check('state: no appended row and the all-nights build certified untouched = all-nights',
          st['kind'] == 'all-nights' and st['certificate']['ok'] is True and st['record'].endswith('builder_frozen.json')
          and st['record_sha1'] == DB._sha1(st['record']) and st['appended'] == 0, st)
    st = DB.key_state('05h', 'V_sb', 'noveto', F, inp0['V_sb'], [], gnew, tnew)
    check('state: no appended row but a regrouped identity = a build is owed', st['kind'] == 'build' and 'not untouched' in st['why'], st)
    st = DB.key_state('05h', 'V_sb', 'noveto', F, inp['V_sb'], [D], gnew, tnew)
    check('state: appended rows and a delta without the record of what it read = a build is owed',
          st['kind'] == 'build' and '2 appended selected row(s)' in st['why'], st)
    DB.write_inputs(D, '05h', 'V_sb', 'noveto', F, [Nd], inp['V_sb']['digest'])
    st = DB.key_state('05h', 'V_sb', 'noveto', F, inp['V_sb'], [f'{R}/nowhere', D], gnew, tnew)
    check('state: a delta that read exactly this input, stands on this all-nights build and is certified = delta',
          st['kind'] == 'delta' and st['dir'] == D and st['certificate']['ok'] is True
          and st['record_sha1'] == DB._sha1(f'{D}/builder_delta.json'), st)
    check('state: a delta of another input, key or tag does not stand',
          DB.key_state('05h', 'V_sb', 'noveto', F, inp0['V_sb'], [D], gnew, tnew)['kind'] == 'build'
          and DB.key_state('05h', 'V_sb', 'veto', F, inp['V_sb'], [D], gnew, tnew)['kind'] == 'no-record'
          and DB.key_state('06h', 'V_sb', 'noveto', F, inp['V_sb'], [D], gnew, tnew)['kind'] == 'build')
    later = grouping(f'{R}/later/detections.csv', [r if r[0] != 'c3' else ('c3', 'G3', 'J0033', 'M2') for r in OLD_ROWS]
                     + [('n1', 'G6', 'J0006', 'M6'), ('n2', 'G3', 'J0033', 'M2'), ('z9', 'G99', 'J0099', 'M99')])
    moved = grouping(f'{R}/moved2/detections.csv', [r if r[0] != 'c3' else ('c3', 'G3', 'J0033', 'M2') for r in OLD_ROWS]
                     + [('n1', 'G6', 'J0006', 'M6'), ('n2', 'G7', 'J0007', 'M7')])
    stl = DB.key_state('05h', 'V_sb', 'noveto', F, inp['V_sb'], [D], later, f'{R}/later/detections.csv')
    check('state: under a later table that leaves the hour\'s detections as they were, the delta still stands',
          stl['kind'] == 'delta' and stl['certificate']['table_to']['sha1'] == later[2] != gnew[2], stl)
    check('state: under a later table that regroups one of its detections, a build is owed',
          DB.key_state('05h', 'V_sb', 'noveto', F, inp['V_sb'], [D], moved, f'{R}/moved2/detections.csv')['kind'] == 'build')
    check('state: a key without rows and without a catalog is empty; a key with a catalog but no frozen record cannot be certified',
          DB.key_state('05h', 'V_wb', 'noveto', F, inp['V_wb'], [], gnew, tnew)['kind'] == 'empty'
          and 'build all nights' in DB.key_state('05h', 'V_sb', 'veto', F, inp['V_sb'], [], gnew, tnew)['why'])
    why = refusal(DB.view_catalog, f'{R}/Vl', f'{F}/source_catalogs_noveto_V_sb', D, later, f'{R}/later/detections.csv')
    check('view: a delta built under an earlier table refuses without a certificate', 'no certificate' in why, why)
    rec = DB.view_catalog(f'{R}/Vl', f'{F}/source_catalogs_noveto_V_sb', D, later, f'{R}/later/detections.csv',
                          cert=stl['certificate'])
    DB.view_catalog(f'{R}/Vn', f'{F}/source_catalogs_noveto_V_sb', D, gnew, tnew)
    same = sorted(n for n in os.listdir(f'{R}/Vn') if n.endswith('.csv'))
    check('view: with the certificate it is, table for table, the view under the delta\'s own table',
          rec['kind'] == 'delta' and len(same) >= 4 and sorted(n for n in os.listdir(f'{R}/Vl') if n.endswith('.csv')) == same
          and all(raw(f'{R}/Vl/{n}') == raw(f'{R}/Vn/{n}') for n in same)
          and rec['certificate']['table_to']['sha1'] == later[2]
          and DB.view_test(f'{R}/Vl', f'{F}/source_catalogs_noveto_V_sb', D, later)[0], same)
    # the view root
    st = DB.key_state('05h', 'V_sb', 'noveto', F, inp['V_sb'], [D], gnew, tnew)
    states = {('noveto', 'V_sb'): st}
    root = f'{R}/views/05h'
    os.makedirs(f'{root}/stale')
    why = refusal(DB.view_root, root, '05h', F, [Nd], {('noveto', 'V_sb'): dict(st, kind='build')}, {'V': (gnew, tnew)},
                  ('noveto',), keys)
    check('view root: a state that does not stand refuses and leaves the directory as it was',
          'build is owed' in why and os.path.isdir(f'{root}/stale'), why)
    rec = DB.view_root(root, '05h', F, [Nd], states, {'V': (gnew, tnew)}, ('noveto',), keys)
    check('view root: the catalogs of every standing key, content-tested; an earlier root is replaced',
          rec['catalogs']['noveto_V_sb']['kind'] == 'delta' and rec['catalogs']['noveto_V_sb']['survivors'] == 5
          and os.path.exists(f'{root}/source_catalogs_noveto_V_sb/catalog_view.json') and not os.path.exists(f'{root}/stale'), rec)
    sel = raw(f'{root}/selected_V_sb.csv').decode().split('\n')
    check('view root: the selected rows, the window-tie table and the detection table are the all-nights rows then the nights\'',
          [ln.split(',')[0] for ln in sel[1:9]] == ['c1', 'c2', 'c3', 'c4', 'c5', 'c7', 'n1', 'n2']
          and raw(f'{root}/pertie/pertie_detections.csv').count(b'\n') == 9 and raw(f'{root}/master_catalog.csv').count(b'\n') == 9
          and rec['detections'] == dict(base=6, appended=2) and rec['selected']['V_sb'] == dict(base=6, appended=2), sel)
    img = raw(f'{root}/v_deep_images.csv').decode().split('\n')
    check('view root: the image table gains the nights\' windows, sorted, after the all-nights rows',
          [ln.split(',')[3] for ln in img[1:5]] == ['2030-01-01T00', '2030-01-02T00', '2030-01-05T00', '2030-01-05T01']
          and rec['images'] == dict(base=2, appended=2), img)
    # the selection before its split by key (the cumulative product reads each hour's)
    check('view root: no directory holds selected_all.csv = none in the root',
          rec['selected_all'] is None and not os.path.exists(f'{root}/selected_all.csv'), rec['selected_all'])
    put(f'{F}/selected_all.csv', [SEL] + [f'{c},05h,46MHz,7.5' for c in ('c1', 'c2', 'c3', 'c4', 'c5', 'c7')])
    r2, two, gr = f'{R}/views_sa/05h', ('V_sb', 'V_wb'), {'V': (gnew, tnew)}
    why = refusal(DB.view_root, r2, '05h', F, [Nd], states, gr, ('noveto',), two)
    check('view root: a night directory without selected_all.csv refuses when the all-nights directory holds one',
          'holds no selected_all.csv (the all-nights directory' in why, why)
    put(f'{Nd}/selected_all.csv', [SEL, 'n2,05h,46MHz,10.0', 'n1,05h,46MHz,9.5'])
    rec2 = DB.view_root(r2, '05h', F, [Nd], states, gr, ('noveto',), two)
    sa = raw(f'{r2}/selected_all.csv').decode().split('\n')
    check('view root: selected_all.csv = the all-nights rows then the nights\', the detections of the keyed tables',
          [ln.split(',')[0] for ln in sa[1:9]] == ['c1', 'c2', 'c3', 'c4', 'c5', 'c7', 'n1', 'n2'] and sa[9:] == ['']
          and rec2['selected_all'] == dict(base=6, appended=2, outside_keys=0)
          and json.load(open(f'{r2}/hour_view.json'))['selected_all'] == rec2['selected_all'], rec2['selected_all'])
    put(f'{Nd}/selected_all.csv', [SEL, 'n2,05h,46MHz,10.0'])
    why = refusal(DB.view_root, r2, '05h', F, [Nd], states, gr, ('noveto',), two)
    check('view root: a keyed selected row that selected_all.csv lacks refuses', "1 keyed row(s) not in it, e.g. ['n1']" in why, why)
    put(f'{Nd}/selected_all.csv', [SEL, 'n2,05h,46MHz,10.0', 'n1,05h,46MHz,9.5', 'n9,05h,46MHz,8.0'])
    why = refusal(DB.view_root, r2, '05h', F, [Nd], states, gr, ('noveto',), two)
    check('view root: a selected_all.csv row in no keyed table refuses when every key has a table',
          "1 of its rows in no keyed table, e.g. ['n9']" in why, why)
    rec3 = DB.view_root(r2, '05h', F, [Nd], states, gr, ('noveto',), keys)
    check('view root: with a key that has no table the rows outside the keyed tables are counted',
          rec3['selected_all'] == dict(base=6, appended=3, outside_keys=1), rec3['selected_all'])
    os.remove(f'{F}/selected_all.csv')
    why = refusal(DB.view_root, r2, '05h', F, [Nd], states, gr, ('noveto',), two)
    check('view root: a night directory with selected_all.csv on an all-nights directory without one refuses',
          'holds no selected_all.csv but the night directory' in why, why)
    os.remove(f'{Nd}/selected_all.csv')
    keep_det = raw(f'{Nd}/master_catalog.csv')
    os.remove(f'{Nd}/master_catalog.csv')
    why = refusal(DB.view_root, r2, '05h', F, [Nd], states, gr, ('noveto',), two)
    check('view root: a night directory without the detection (or image) table the all-nights directory holds refuses',
          'holds no master_catalog.csv (the all-nights directory' in why, why)
    open(f'{Nd}/master_catalog.csv', 'wb').write(keep_det)
    shutil.rmtree(f'{R}/views_sa')
    dg = DB.state_digest(states)
    check('view root: it is not complete before its localizations', rec['complete'] is False and not DB.root_stands(root, dg)
          and rec['state_digest'] == dg)
    # the localizations of the touched names
    side, names, n = DB.loc_side(root, 'noveto', 'source_catalogs_noveto_[IV]_[sw]b')
    sv = csvrows(f'{side}/source_catalogs_noveto_V_sb/46MHz_sources_survivors.csv')
    check('localizations: the side directory holds the survivors rows of the touched and replaced names and their member table',
          names == ['J0002', 'J0003', 'J0006', 'J0033'] and n == 3 and sorted(r['source_id'] for r in sv) == ['J0002', 'J0006', 'J0033']
          and raw(f'{side}/source_catalogs_noveto_V_sb/46MHz_source_members.csv') == raw(f'{root}/source_catalogs_noveto_V_sb/46MHz_source_members.csv'), names)
    put(f'{side}/out/localizations.csv', ['identity_id,ra,dec', 'J0002,2.5,2.5', 'J0006,6.0,6.0', 'J0033,3.3,3.3'], term='\r\n')
    put(f'{side}/out/localization_offsets.csv', ['cand_id,identity_id,sep', 'c2,J0002,0.25', 'n1,J0006,0.6', 'n2,J0033,0.9',
                                                 'c3,J0033,0.8'], term='\r\n')
    lm = DB.loc_merge(root, 'noveto', names, f'{F}/localizations_noveto', f'{side}/out', 'source_catalogs_noveto_[IV]_[sw]b')
    lo = raw(f'{root}/localizations_noveto/localizations.csv').decode().split('\r\n')
    of = raw(f'{root}/localizations_noveto/localization_offsets.csv').decode().split('\r\n')
    check('localizations: untouched names keep their all-nights rows byte for byte, touched names get the new rows, a replaced '
          'name is gone, in the script\'s order',
          lo[1:6] == ['J0001,1.0,1.0', 'J0002,2.5,2.5', 'J0006,6.0,6.0', 'J0007,7.0,7.0', 'J0033,3.3,3.3']
          and of[1:7] == ['c1,J0001,0.1', 'c2,J0002,0.25', 'n1,J0006,0.6', 'c7,J0007,0.7', 'n2,J0033,0.9', 'c3,J0033,0.8']
          and lm['localizations.csv'] == dict(kept=2, new=3, dropped=2, identities=5), (lo, of, lm))
    put(f'{R}/stray/localizations.csv', ['identity_id,ra,dec', 'J0001,9,9'], term='\r\n')
    why = refusal(DB.loc_merge, root, 'noveto', names, f'{F}/localizations_noveto', f'{R}/stray', 'source_catalogs_noveto_[IV]_[sw]b')
    check('localizations: a computed row of a name that was not asked for refuses', 'not asked for' in why, why)
    put(f'{R}/deadloc/localizations.csv', ['identity_id,ra,dec', 'J0001,1.0,1.0', 'J0555,5,5'], term='\r\n')
    why = refusal(DB.loc_merge, root, 'noveto', names, f'{R}/deadloc', f'{side}/out', 'source_catalogs_noveto_[IV]_[sw]b')
    check('localizations: an all-nights row of a name that is no survivor of the view refuses', 'no survivors of the view' in why, why)
    put(f'{R}/headloc/localizations.csv', ['identity_id,ra,dec,more', 'J0002,2,2,2'], term='\r\n')
    why = refusal(DB.loc_merge, root, 'noveto', names, f'{F}/localizations_noveto', f'{R}/headloc', 'source_catalogs_noveto_[IV]_[sw]b')
    check('localizations: a header that differs from the all-nights table\'s refuses', 'header' in why, why)
    DB.finish_root(root, {'noveto': lm})
    check('view root: complete after its localizations, and standing for exactly its states',
          DB.root_stands(root, dg) and not DB.root_stands(root, '0' * 40) and not DB.root_stands(f'{R}/nowhere', dg)
          and json.load(open(f'{root}/hour_view.json'))['localizations']['noveto']['localizations.csv']['new'] == 3)
    # carrying an earlier root: it must stand, be made from the same image / detection / selection tables, and pass view_test
    idg, gv = DB.view_inputs_digest(F, [Nd]), {'V': gnew}
    ok0, why0 = DB.root_carries(root, dg, idg, states, gv)
    check('carry: a complete root of these states, made from the same tables, whose catalogs pass view_test again, is carried',
          ok0 and rec['inputs_digest'] == idg, why0)
    ok1, why1 = DB.root_carries(root, dg, '0' * 40, states, gv)
    check('carry: a root made from another image, detection or selection table is not carried', not ok1 and 'other image' in why1,
          why1)
    img_n = f'{Nd}/v_deep_images.csv'
    keep_img = raw(img_n)
    open(img_n, 'wb').write(keep_img + keep_img.split(b'\n')[1] + b'\n')
    check('carry: a night image table that changed changes the inputs digest', DB.view_inputs_digest(F, [Nd]) != idg)
    open(img_n, 'wb').write(keep_img)
    vp = f'{root}/source_catalogs_noveto_V_sb/46MHz_sources_survivors.csv'
    keep_v = raw(vp)
    lines = keep_v.split(b'\n')
    open(vp, 'wb').write(b'\n'.join(lines[:-2] + lines[-1:]))      # one row lost after the root was made complete
    ok2, why2 = DB.root_carries(root, dg, idg, states, gv)
    check('carry: a complete root whose catalog lost a row is not carried (its flag alone never carries it)',
          not ok2 and DB.root_stands(root, dg) and 'noveto_V_sb' in why2, why2)
    open(vp, 'wb').write(keep_v)
    # the index of a pass
    e = dict(kind='view', root=root, keys={'noveto_V_sb': dict(kind='delta', dir=D)})
    DB._dump(f'{R}/p1/hour_views.json', dict(written='2030-01-05T00:00:00Z', hours={'05h': e}))
    DB._dump(f'{R}/p2/hour_views.json', dict(written='2030-01-06T00:00:00Z', hours={
        '05h': dict(kind='view', root=f'{R}/views2/05h', keys={'noveto_V_sb': dict(kind='delta', dir=f'{R}/D2')}),
        '06h': dict(kind='all-nights', root=f'{R}/F06', keys={'noveto_V_sb': dict(kind='all-nights', dir=None)})}))
    ix = DB.read_indexes([f'{R}/p1', f'{R}/p2', f'{R}/p1', f'{R}/p0'])
    check('index: the night directories that hold an index, newest written first, each once',
          [d for d, _i in ix] == [f'{R}/p2', f'{R}/p1'])
    check('index: the delta directories and view roots earlier passes listed for an hour, newest first; an all-nights hour '
          'lists none',
          DB.index_candidates(ix, '05h', 'noveto', 'V_sb') == [f'{R}/D2', D] and DB.index_roots(ix, '05h') == [f'{R}/views2/05h', root]
          and DB.index_candidates(ix, '06h', 'noveto', 'V_sb') == [] and DB.index_roots(ix, '06h') == []
          and DB.index_candidates(ix, '05h', 'veto', 'V_sb') == [])
    # the image-table merge
    put(f'{R}/img_dup.csv', [IMG, '05h,46MHz,V,2030-01-02T00,9'])
    why = refusal(DB.merge_keyed, f'{F}/v_deep_images.csv', [f'{R}/img_dup.csv'], f'{R}/img_o.csv', DB.IMAGE_KEY, 'image table')
    check('image table: a window that an earlier table lists refuses', 'already in' in why and not os.path.exists(f'{R}/img_o.csv'), why)
    put(f'{R}/img_head.csv', [IMG + ',more', '05h,46MHz,V,2030-01-09T00,9,1'])
    why = refusal(DB.merge_keyed, f'{F}/v_deep_images.csv', [f'{R}/img_head.csv'], f'{R}/img_o.csv', DB.IMAGE_KEY, 'image table')
    check('image table: a night table with another header refuses', 'header' in why, why)


def main():
    t0 = time.time()
    shutil.rmtree(T, ignore_errors=True)
    os.makedirs(T)
    test_merge()
    test_assemble()
    test_content()
    test_own_delta()
    test_recorded_inputs()
    test_view()
    test_hour()
    print(f'\n{N[1]} of {N[0]} checks passed in {time.time() - t0:.0f} s')
    if N[1] == N[0]:
        shutil.rmtree(T, ignore_errors=True)
    sys.exit(0 if N[1] == N[0] else 1)


if __name__ == '__main__':
    main()
