#!/usr/bin/env python3
"""Data-free test of lib/catalog/night_view.py: the view of one LST's 10-minute search jobs that holds one night's rows,
or every row except one night's (the per-hour 10-minute catalog job, Gregg 2026-11-12,
R-GREGG-20261112-PER-HOUR-STEPS-SIX-RULINGS (1), (2)).

A synthetic search directory (two completed jobs of one cell, one job without a ledger, a job of another LST) with three
nights of rows. Checks: the kept rows byte for byte and in the job's order; the two modes partition every record; the
candidate directories follow the windows; everything else is linked and lock files are left out; the content test
passes on a fresh view and fails on a changed one; every refusal named in the module.

Scratch: /tmp/exopipe3_tests_catalog_night_view (removed at the end unless EXOPIPE3_TEST_KEEP is set).
"""
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exopipe.phase3.lib.catalog import night_view as NV   # noqa: E402

T = '/tmp/exopipe3_tests_catalog_night_view'
LABEL = 'SLST17B'
NIGHT = '2025-05-08'
N = [0]
FAILS = []


def check(name, ok, detail=''):
    N[0] += 1
    print(('PASS  ' if ok else 'FAIL  ') + name + (f'  ({detail})' if (detail and not ok) else ''))
    if not ok:
        FAILS.append(name)


CAT = ('cand_id,lst,date,band,stokes,peak_jy,global_snr\n'
       'V1_17h_2025-04-05T00_73MHz_0000,17h,2025-04-05T00,73MHz,V,-0.551,-5.09\n'
       'V1_17h_2025-05-08T02_73MHz_0001,17h,2025-05-08T02,73MHz,V,"0.700",6.10\n'
       'V1_17h_2025-05-08T04_73MHz_0002,17h,2025-05-08T04,73MHz,V,0.900,7.00\n'
       'V1_17h_2025-05-09T00_73MHz_0003,17h,2025-05-09T00,73MHz,V,1.100,8.00')          # no line feed at the end
LEDGER = ('lst,scope,stokes,date,ra,dec,final\r\n'
          '17h,73MHz,V,2025-04-05T00,252.7,4.9,1\r\n'
          '17h,73MHz,V,2025-05-08T02,252.8,5.0,1\r\n'
          '17h,73MHz,V,2025-05-08T02,252.9,5.1,0\r\n'
          '17h,73MHz,V,2025-05-08T04,253.0,5.2,1\r\n'
          '17h,73MHz,V,2025-05-09T00,253.1,5.3,1\r\n')
IMAGES = ('lst,scope,stokes,date,n_final\n'
          '17h,73MHz,V,2025-04-05T00,1\n17h,73MHz,V,2025-05-08T02,1\n17h,73MHz,V,2025-05-08T03,0\n'
          '17h,73MHz,V,2025-05-08T04,1\n17h,73MHz,V,2025-05-09T00,1\n')
TABLE = 'lst,scope,stokes,date,verdict\n17h,73MHz,V,2025-04-05T00,0\n17h,73MHz,V,2025-05-08T02,1\n'
CANDS = ['V1_17h_2025-04-05T00_73MHz_0000', 'V1_17h_2025-05-08T02_73MHz_0001', 'V1_17h_2025-05-08T04_73MHz_0002',
         'V1_17h_2025-05-09T00_73MHz_0003']


def put(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        f.write(text)


def search_dir(root, cat=CAT, ledger=LEDGER, cands=CANDS, extra=None):
    """A synthetic search directory; returns its path."""
    for st in ('V', 'I'):
        rs = f'{root}/17h_73MHz_{st}/transients/Run_{LABEL}'
        put(f'{rs}/master_catalog.csv', cat)
        put(f'{rs}/v_deep_ledger.csv', ledger)
        put(f'{rs}/v_deep_images.csv', IMAGES)
        put(f'{rs}/search_table_record.csv', TABLE)
        put(f'{rs}/v_deep_ledger.csv.lock', '')
        put(f'{rs}/refusals/17h_73MHz_searchtier_excluded.csv', '17h,73MHz,2025-04-06T01,V,3.1,tag\n')
        put(f'{rs}/sigstack/17h_73MHz_{st}_sigstack.fits', 'x')
        for c in cands:
            put(f'{rs}/candidates/{c}/resid.fits', c)
        put(f'{root}/17h_73MHz_{st}/Run_{LABEL}/vresid/17h_73MHz/2025-05-08T02_V_redo_resid_73MHz.fits', 'f')
        put(f'{root}/17h_73MHz_{st}/catdir/master_R075_17h_73MHz.gaul.csv', 'p')
        put(f'{root}/17h_73MHz_{st}/17h_73MHz.log', 'log')
        put(f'{root}/17h_73MHz_{st}/transients/other/keep.txt', 'k')
        for name, text in (extra or {}).items():
            put(f'{rs}/{name}', text)
    put(f'{root}/17h_82MHz_V/17h_82MHz.log', 'no ledger: not a completed search')
    put(f'{root}/18h_73MHz_V/transients/Run_{LABEL}/v_deep_ledger.csv', LEDGER)
    return root


def raw(path):
    with open(path, newline='') as f:
        return f.read()


def test_views():
    src = search_dir(f'{T}/search')
    vn, vw = f'{T}/view_night', f'{T}/view_without'
    rec = NV.build(src, vn, '17h', LABEL, NIGHT, 'night')
    NV.build(src, vw, '17h', LABEL, NIGHT, 'without')
    check('the view holds the LST\'s three jobs and its record, not the other LST\'s',
          sorted(os.listdir(vn)) == ['17h_73MHz_I', '17h_73MHz_V', '17h_82MHz_V', NV.MANIFEST], os.listdir(vn))
    rn, rw = f'{vn}/17h_73MHz_V/transients/Run_{LABEL}', f'{vw}/17h_73MHz_V/transients/Run_{LABEL}'
    cl = CAT.split('\n')
    check('night: master_catalog.csv is the header and the night\'s two rows, byte for byte (the quoted field kept)',
          raw(f'{rn}/master_catalog.csv') == '\n'.join(cl[:1] + cl[2:4]) + '\n', repr(raw(f'{rn}/master_catalog.csv')))
    check('without: the header and the other nights\' rows, in the job\'s order; the last row keeps its missing line feed',
          raw(f'{rw}/master_catalog.csv') == '\n'.join(cl[:2] + cl[4:]), repr(raw(f'{rw}/master_catalog.csv')))
    ll = LEDGER.split('\r\n')
    check('night: the ledger keeps its CRLF line ends and the night\'s three rows (final and not final)',
          raw(f'{rn}/v_deep_ledger.csv') == '\r\n'.join(ll[:1] + ll[2:5]) + '\r\n', repr(raw(f'{rn}/v_deep_ledger.csv')))
    check('night: the image ledger keeps the night\'s window without a detection',
          raw(f'{rn}/v_deep_images.csv').count('\n') == 4 and '2025-05-08T03,0' in raw(f'{rn}/v_deep_images.csv'))
    part = all(sorted(raw(f'{rn}/{n}').splitlines()[1:] + raw(f'{rw}/{n}').splitlines()[1:])
               == sorted(raw(f'{src}/17h_73MHz_V/transients/Run_{LABEL}/{n}').splitlines()[1:]) for n in NV.RECORDS)
    check('the two modes partition every record file: their rows together are the job\'s rows', part)
    check('night: candidates/ links the night\'s two candidate directories only',
          sorted(os.listdir(f'{rn}/candidates')) == CANDS[1:3]
          and os.readlink(f'{rn}/candidates/{CANDS[1]}') == f'{src}/17h_73MHz_V/transients/Run_{LABEL}/candidates/{CANDS[1]}')
    check('without: candidates/ links the other nights\' two', sorted(os.listdir(f'{rw}/candidates')) == [CANDS[0], CANDS[3]])
    check('lock files are left out', not os.path.lexists(f'{rn}/v_deep_ledger.csv.lock'))
    links = [f'{rn}/refusals', f'{rn}/sigstack', f'{vn}/17h_73MHz_V/Run_{LABEL}', f'{vn}/17h_73MHz_V/catdir',
             f'{vn}/17h_73MHz_V/17h_73MHz.log', f'{vn}/17h_73MHz_V/transients/other']
    check('refusals, sigma stack, residual frames, parent catalog, log and the other entries are links to the job\'s',
          all(os.path.islink(p) and os.readlink(p).startswith(src + '/') and os.path.exists(p) for p in links),
          [p for p in links if not os.path.islink(p)])
    check('a job without a ledger is linked whole (the merge skips it by name, as before)',
          os.path.islink(f'{vn}/17h_82MHz_V') and os.readlink(f'{vn}/17h_82MHz_V') == f'{src}/17h_82MHz_V')
    check('the record counts rows and candidates: kept of all',
          rec['rows']['master_catalog.csv'] == [8, 4] and rec['rows']['v_deep_ledger.csv'] == [10, 6]
          and rec['candidates'] == [8, 4], (rec['rows'], rec['candidates']))
    check('the job directory is unchanged by the two builds',
          raw(f'{src}/17h_73MHz_V/transients/Run_{LABEL}/master_catalog.csv') == CAT
          and sorted(os.listdir(f'{src}/17h_73MHz_V/transients/Run_{LABEL}/candidates')) == CANDS
          and os.path.exists(f'{src}/17h_73MHz_V/transients/Run_{LABEL}/v_deep_ledger.csv.lock'))
    ok, n, det = NV.verify(vn)
    check('content test: PASS on the night view (2 completed jobs x (2 + 3 + 3 + 1) rows)', ok and n == 18, det)
    ok, n, det = NV.verify(vw)
    check('content test: PASS on the view without the night', ok and n == 2 * (2 + 2 + 2 + 1), det)
    return src, vn


def test_content_test_fails(src, vn):
    def fresh(tag):
        d = f'{T}/tamper_{tag}'
        NV.build(src, d, '17h', LABEL, NIGHT, 'night')
        return d, f'{d}/17h_73MHz_V/transients/Run_{LABEL}'
    d, r = fresh('row')
    with open(f'{r}/master_catalog.csv', 'a', newline='') as f:
        f.write(CAT.split('\n')[1] + '\n')
    ok, _, det = NV.verify(d)
    check('content test: a row of another night in the view fails', not ok and 'of another night' in det, det)
    d, r = fresh('missing')
    put(f'{r}/v_deep_ledger.csv', '\r\n'.join(LEDGER.split('\r\n')[:3]) + '\r\n')
    ok, _, det = NV.verify(d)
    check('content test: a missing row of the night fails', not ok and 'v_deep_ledger.csv' in det, det)
    d, r = fresh('cand')
    os.unlink(f'{r}/candidates/{CANDS[1]}')
    ok, _, det = NV.verify(d)
    check('content test: a missing candidate directory fails', not ok and 'candidates' in det, det)
    d, r = fresh('copy')
    os.unlink(f'{r}/sigstack')
    shutil.copytree(f'{src}/17h_73MHz_V/transients/Run_{LABEL}/sigstack', f'{r}/sigstack')
    ok, _, det = NV.verify(d)
    check('content test: a copy where a link belongs fails', not ok and 'not a link' in det, det)
    d, r = fresh('job')
    os.unlink(f'{d}/17h_82MHz_V')
    ok, _, det = NV.verify(d)
    check('content test: a job missing from the view fails', not ok and 'exactly the search directory' in det, det)
    ok, _, det = NV.verify(f'{T}/no_such_view')
    check('content test: a directory without the view\'s record fails', not ok and 'no readable' in det, det)


def refusal(fn, *a):
    try:
        fn(*a)
    except NV.ViewRefusal as e:
        return str(e)
    return ''


def test_refusals(src, vn):
    check('an existing view directory refuses (built once, in a new directory)',
          'exists' in refusal(NV.build, src, vn, '17h', LABEL, NIGHT, 'night'))
    check('a night that is not YYYY-MM-DD refuses', 'is not YYYY-MM-DD' in refusal(NV.build, src, f'{T}/r1', '17h', LABEL,
                                                                                   '2025-05-08T02', 'night'))
    check('an unknown mode refuses', 'is not one of' in refusal(NV.build, src, f'{T}/r2', '17h', LABEL, NIGHT, 'only'))
    check('an LST without a job refuses', 'no search job' in refusal(NV.build, src, f'{T}/r3', '09h', LABEL, NIGHT, 'night'))
    s = search_dir(f'{T}/search_baddate', cat=CAT.replace('2025-05-09T00,73MHz', '20250509,73MHz'))
    check('a row whose date is not a night or a window refuses, naming the line',
          'line 5: date' in refusal(NV.build, s, f'{T}/r4', '17h', LABEL, NIGHT, 'night'))
    s = search_dir(f'{T}/search_nodate', ledger=LEDGER.replace('stokes,date,ra', 'stokes,epoch,ra'))
    check('a record file without a `date` column refuses', 'no `date` column' in refusal(NV.build, s, f'{T}/r5', '17h', LABEL,
                                                                                        NIGHT, 'night'))
    s = search_dir(f'{T}/search_multiline', cat=CAT.replace('"0.700"', '"0.7\n00"'))
    check('a record that spans two lines refuses', 'spans 2 lines' in refusal(NV.build, s, f'{T}/r6', '17h', LABEL, NIGHT,
                                                                              'night'))
    s = search_dir(f'{T}/search_badcand', cands=CANDS + ['notes'])
    check('a candidate directory whose name carries no window refuses',
          'carries no window' in refusal(NV.build, s, f'{T}/r7', '17h', LABEL, NIGHT, 'night'))
    s = search_dir(f'{T}/search_blank', ledger=LEDGER.replace('5.0,1\r\n', '5.0,1\r\n\r\n'))
    check('an empty line inside a record file refuses', 'empty line' in refusal(NV.build, s, f'{T}/r8', '17h', LABEL, NIGHT,
                                                                               'night'))
    s = search_dir(f'{T}/search_ff', cat=CAT.replace('"0.700"', '"0.7\x0c00"'))
    rec = NV.build(s, f'{T}/r9', '17h', LABEL, NIGHT, 'night')
    check('a form feed inside a field is not a line end (the row is kept whole)',
          rec['rows']['master_catalog.csv'] == [8, 4]
          and '"0.7\x0c00"' in raw(f'{T}/r9/17h_73MHz_V/transients/Run_{LABEL}/master_catalog.csv'))
    s = search_dir(f'{T}/search_empty_night')
    rec = NV.build(s, f'{T}/r10', '17h', LABEL, '2025-06-01', 'night')
    hdr_only = raw(f'{T}/r10/17h_73MHz_V/transients/Run_{LABEL}/master_catalog.csv')
    check('a night without a row gives record files of their header alone, and the content test passes',
          hdr_only == CAT.split('\n')[0] + '\n' and rec['rows']['master_catalog.csv'] == [8, 0] and NV.verify(f'{T}/r10')[0],
          repr(hdr_only))


def test_cli(src):
    me = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'exopipe', 'phase3', 'lib', 'catalog',
                      'night_view.py')
    p = subprocess.run([sys.executable, me, 'build', src, f'{T}/cli', '17h', LABEL, NIGHT, 'night'], stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True)
    check('command line: build prints the counts and the content test, rc 0',
          p.returncode == 0 and 'NIGHT-VIEW built' in p.stdout and 'NIGHT-VIEW content test PASS' in p.stdout, p.stdout[-300:])
    p = subprocess.run([sys.executable, me, 'build', src, f'{T}/cli', '17h', LABEL, NIGHT, 'night'], stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, universal_newlines=True)
    check('command line: a second build into the same directory refuses, rc 2',
          p.returncode == 2 and 'NIGHT-VIEW REFUSING' in p.stdout, p.stdout[-300:])
    src_txt = open(me).read()
    check('the module imports the standard library only',
          not [ln for ln in src_txt.splitlines() if ln.startswith(('import ', 'from '))
               and ln.split()[1].split('.')[0] not in ('csv', 'glob', 'json', 'os', 're', 'sys', 'time')])


def test_wiring():
    """The chain's night pass (`catalog --tier sublst --sublst-night`), by the text of its stages and by its command line."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(repo, 'exopipe', 'phase3', 'lib', 'catalog', 'tier_sublst.py')) as f:
        src = f.read()

    def body(name):
        i = src.index(f'def {name}(')
        return src[i:src.index('\ndef ', i + 10)]
    from exopipe.phase3.lib.catalog import tier_sublst as TS
    check('wiring: a night pass offers the chain\'s stages from preflight to the leak veto, in the chain\'s order',
          TS.NIGHT_STAGES == TS.STAGES[:len(TS.NIGHT_STAGES)] and TS.NIGHT_STAGES[-1] == 'veto'
          and not set(TS.NIGHT_STAGES) & {'build', 'buildv', 'test', 'compare', 'reftie'}, TS.NIGHT_STAGES)
    nv, merge, conf = body('night_view'), body('st_merge'), body('st_conform')
    check('wiring: the view is built in mode night from the jobs, then content-tested; a failed test refuses',
          "NV.build(ctx.jobs_src, ctx.src, ctx.lst, ctx.label, ctx.night, 'night')" in nv and 'NV.verify(ctx.src)' in nv
          and nv.index('NV.build(') < nv.index('NV.verify(') < nv.index('content test failed'))
    check('wiring: a night without a catalog row ends the merge stage no-input, after the content test',
          "rec['rows']['master_catalog.csv'][1] == 0" in nv and 'NoInput(' in nv and nv.index('NV.verify(') < nv.index('NoInput('))
    check('wiring: the merge stage builds the view before it lists the jobs it links',
          'night_view(ctx, s)' in merge and merge.index('night_view(ctx, s)') < merge.index('jobs = sorted(glob.glob('))
    check('wiring: the night\'s view is the search directory the chain reads; the size rule is read from the jobs',
          "self.src = os.path.join(self.U, '_side', 'night_view')" in src
          and "glob.glob(f'{self.jobs_src}/{self.lst}_*_[IV]')" in src)
    check('wiring: a night\'s conformance runs in frozen mode on the LST-wide record with the accepted module versions',
          "EXOPIPE3_CONFORM_NOISE_MODE='frozen'" in conf and 'EXOPIPE3_CONFORM_NOISE_RECORD=frozen' in conf
          and "','.join(shared('catalog_conform', 'NOISE_MODULE_SHA1'))" in conf)
    check('wiring: a night writes no record of its own, and a missing LST-wide record refuses before the module runs',
          'frozen, rec_on = conform_noise_record(ctx.frozen_from), False' in conf and 'no table-noise record' in conf
          and conf.index('no table-noise record') < conf.index("EXOPIPE3_CONFORM_NOISE_MODE='frozen'"))
    if not os.environ.get('EXOPIPE3_SITE'):
        print('SKIP  command line of the night pass: no site file on this machine (EXOPIPE3_SITE)')
        return

    def cli(*extra):
        p = subprocess.run([sys.executable, '-m', 'exopipe.phase3', 'catalog', '--tier', 'sublst', '--lsts', '17h', '--out',
                            f'{T}/cli_night', '--cores', 'none', '--dry-run'] + list(extra), cwd=repo, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, universal_newlines=True)
        return p.returncode, p.stdout
    rc, out = cli('--sublst-night', NIGHT)
    check('command line: a night without --sublst-frozen-from is an error', rc != 0 and '--sublst-frozen-from' in out, out[-300:])
    rc, out = cli('--sublst-frozen-from', f'{T}/lstwide')
    check('command line: --sublst-frozen-from without a night is an error', rc != 0 and '--sublst-night' in out, out[-300:])
    rc, out = cli('--sublst-night', '2025-05-08T02', '--sublst-frozen-from', f'{T}/lstwide')
    check('command line: a night that is not YYYY-MM-DD is an error', rc != 0 and 'is not YYYY-MM-DD' in out, out[-300:])
    rc, out = cli('--sublst-night', NIGHT, '--sublst-frozen-from', f'{T}/lstwide', '--stages', 'veto,build')
    check('command line: a build stage is not offered for a night', rc != 0 and 'offers the stages' in out, out[-300:])
    rc, out = cli('--sublst-night', NIGHT, '--sublst-frozen-from', f'{T}/cli_night')
    check('command line: the frozen record cannot come from the night\'s own directory', rc != 0 and 'own --out' in out, out[-300:])
    rc, out = cli('--sublst-night', NIGHT, '--sublst-frozen-from', f'{T}/lstwide')
    check('command line: the dry run plans the stages preflight to veto, the view and the frozen conformance',
          rc == 0 and 'NIGHT PASS 2025-05-08' in out and '.veto]' in out and '.build]' not in out
          and 'night view of 2025-05-08' in out and 'FROZEN' in out, out[-600:])


def main():
    shutil.rmtree(T, ignore_errors=True)
    os.makedirs(T)
    try:
        src, vn = test_views()
        test_content_test_fails(src, vn)
        test_refusals(src, vn)
        test_cli(src)
        test_wiring()
    finally:
        if not os.environ.get('EXOPIPE3_TEST_KEEP'):
            shutil.rmtree(T, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
