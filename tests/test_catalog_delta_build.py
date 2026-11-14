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


def main():
    t0 = time.time()
    shutil.rmtree(T, ignore_errors=True)
    os.makedirs(T)
    test_merge()
    test_assemble()
    test_content()
    test_recorded_inputs()
    print(f'\n{N[1]} of {N[0]} checks passed in {time.time() - t0:.0f} s')
    if N[1] == N[0]:
        shutil.rmtree(T, ignore_errors=True)
    sys.exit(0 if N[1] == N[0] else 1)


if __name__ == '__main__':
    main()
