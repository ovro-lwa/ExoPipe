"""Placement batch P5 (core/placement in the writers of tess_orbits, qa/percentiles, masters/vclean_masters, the co-add final
stage, the sub-LST job lists, the deep search's parent catalogue and the frames-only engine child): what a placement refusal
does at each handler that would otherwise crash on it or swallow it, and the sweep of a killed writer's temporary files.
Data-free: scratch directories under TMPDIR; a refusal is forced by giving the placement a group this user is not in
(UNTESTED is printed when there is none, e.g. as root). sweep ages a temporary file by its ctime, which utime cannot set
back: a planted leftover is aged by its mtime through placement's seam _changed (test_placement_p6 tests the ctime).
Checks:
  sweep    placement.sweep removes a product's temporary file not changed for an hour and keeps a young one, another product's,
           a <dst>.1 product's, a name that is not a temporary name and the product; a tag's sweep only that tag's;
  writers  eight converted writers callable without data (flow write_json_atomic, epoch_gate.write_record,
           release1_cards._write_text, photometry write_zone_flags, coadd._write_scale2_record, deep_stages.own_note, the
           sub-LST search's verify report, followup._atomic_text) remove their product's planted stale temporary file before
           writing, and nothing else;
  tess     refresh: a table that cannot be placed is recorded 'refused: install not placed: placement: ...' (not a crash of
           main, not 'installed'); the current table's content is unchanged and its time touched ("checked now", so the
           reader does not install the fetched table itself), no temporary file left; main's exit status is still the
           current table's: 0 when it reaches far enough ahead, 2 when it does not;
  qa       write_for_chain: a table that cannot be placed refuses the chain's unit (record.Refusal 'not placed: placement:
           ...', recorded 'refused'); the table before is unchanged, no temporary file left;
  vclean   build_vclean_cell: the cleaned master is copied from the cleaning engine's scratch directory to a temporary file
           beside the product, its size and md5 checked, the PB sibling written from that copy, and the pair placed together
           (the directory's rule for new files; the replaced pair's mode). A refusal of either file places neither ('REFUSED
           placement: ... none of the 2 products placed'); a short copy refuses ('REFUSED placement: the copy ... is not
           verified ... nothing placed', placement.copy_verified) with neither placed; a failed sibling rename leaves the
           master placed and the status says so ('already placed: [<master>] (those stay live)'). No temporary file and no
           scratch directory left. Across two file systems when EXOPIPE_TEST_OTHER_FS names a writable directory on another
           file system than TMPDIR (UNTESTED otherwise);
  coadd    the final stage's worker: a PlacementRefusal of a (LST, freq) is REFUSED (returned with its text, listed apart from
           the combinations processed), not ERROR; any other exception is still ERROR;
  admit    the sub-LST job lists: a failed rename of the second list leaves the first placed; the unit's refusal names the
           list that is live and the one that is not;
  gaul     the deep search's parent catalogue: a PyBDSF extraction with no sources (write_catalog returns False) refuses
           before any placement, with a catalogue reason (not 'placement'), and leaves no catalogue and no temporary file;
  child    deep_child: an engine that raises a PlacementRefusal leaves result status 'refused', gate 'placement', exit 2; any
           other record.Refusal status 'refused', gate 'engine', exit 2; any other exception is still status 'error', exit 1."""
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')

from exopipe.phase3.core import placement as PL  # noqa: E402
from exopipe.phase3.core import record  # noqa: E402

TMP = os.path.join(os.environ.get('TMPDIR', tempfile.gettempdir()), 'test_placement_p5')
FOREIGN = next((g for g in (0, 1, 2, 3, 4, 65534) if g not in os.getgroups() and g != os.getegid()), None)
CAN_REFUSE = FOREIGN is not None and os.geteuid() != 0


def _fresh(name):
    d = os.path.join(TMP, name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    return d


class _refuse_for:
    """Within the block, core/placement gives any destination whose name ends with ``suffix`` a group this user is not in, so
    its placement refuses (PlacementRefusal) after the temporary file is written."""

    def __init__(self, suffix):
        self.suffix, self.real = suffix, PL.target_attrs

    def __enter__(self):
        real, suffix = self.real, self.suffix
        PL.target_attrs = lambda dst, like=None: (dict(real(dst, like), gid=FOREIGN) if str(dst).endswith(suffix)
                                                  else real(dst, like))
        return self

    def __exit__(self, *a):
        PL.target_attrs = self.real
        return False


def _plant(dst):
    """Beside ``dst``: a temporary file of it not modified for two hours (a killed writer's: the writer's sweep removes it), and
    four names a sweep keeps - a young temporary file of it, a stale temporary file of another product, a stale temporary file
    of the product <dst>.1 and a stale name that is not a temporary name. Returns (the stale one, the kept ones)."""
    d, b = os.path.dirname(dst), os.path.basename(dst)
    old = time.time() - 7200
    stale = os.path.join(d, f'{b}.4194301.0badf00d.tmp')
    keep = [os.path.join(d, f'{b}.4194302.0badf00e.tmp'), os.path.join(d, 'zz_other.csv.4194303.0badf00f.tmp'),
            os.path.join(d, f'{b}.1.4194304.0badf010.tmp'), os.path.join(d, f'{b}.4194305.old.tmp')]
    os.makedirs(d, exist_ok=True)
    for p in [stale] + keep:
        with open(p, 'w') as f:
            f.write('planted\n')
        if p != keep[0]:
            os.utime(p, (old, old))
    return stale, keep


def test_sweep():
    d = _fresh('sweep')
    dst = os.path.join(d, 'a.csv')
    open(dst, 'w').write('product\n')
    stale, keep = _plant(dst)
    tagged = os.path.join(d, 'a.csv.v1.4194306.0badf011.tmp')
    open(tagged, 'w').write('planted\n')
    os.utime(tagged, (time.time() - 7200,) * 2)
    msgs = []
    got = PL.sweep([dst], log=msgs.append)
    assert got == [stale] and not os.path.exists(stale), (got, os.listdir(d))
    assert all(os.path.exists(p) for p in keep + [dst, tagged]), os.listdir(d)
    assert len(msgs) == 1 and stale in msgs[0] and 'did not finish' in msgs[0], msgs
    assert PL.sweep(dst, tag='v1', log=msgs.append) == [tagged] and PL.sweep(dst, log=msgs.append) == [], os.listdir(d)
    print('ok sweep: a product\'s temporary file not modified for an hour is removed; a young one, another product\'s, the '
          '<dst>.1 product\'s, a non-temporary name and the product stay; a tag\'s sweep removes only that tag\'s')


def test_writers_sweep():
    import contextlib
    import io
    from exopipe.phase3.lib.flow import steps_common as FC
    from exopipe.phase3.lib.masters import admit as A, coadd as CO, deep_stages as DS, epoch_gate as EG, release1_cards as R1
    from exopipe.phase3.lib.photometry import tests as PT
    from exopipe.phase3.lib.search import tier_sublst as ST
    from exopipe.phase3.steps import followup as FU
    d = _fresh('writers')
    side = os.path.join(d, 'ds', 'master.gaul.csv.gen.json')
    os.makedirs(os.path.dirname(side))
    json.dump({'note': 'x'}, open(side, 'w'))
    cases = [
        ('flow write_json_atomic', os.path.join(d, 'flow', 'm.json'), lambda p: FC.write_json_atomic(p, {'a': 1})),
        ('epoch_gate write_record', os.path.join(d, 'eg', EG.GATE_FILE),
         lambda p: EG.write_record(os.path.dirname(p), {'2025-01-01': {'verdict': 'PASS'}})),
        ('release1_cards _write_text', os.path.join(d, 'r1', 'cards.csv'), lambda p: R1._write_text(p, 'a\n')),
        ('photometry write_zone_flags', os.path.join(d, 'ph', 'zone_flags.csv'),
         lambda p: PT.write_zone_flags(p, [(('t1', 'Red'), 'ok', 'ok')])),
        ('coadd _write_scale2_record', os.path.join(d, 'co', A.SCALE2),
         lambda p: CO._write_scale2_record(os.path.dirname(p), {'2025-01-01': ((1.0,), 9)}, 1.0)),
        ('deep_stages own_note', side, lambda p: DS.own_note(p, types.SimpleNamespace(run_id='r1'))),
        ('sub-LST search verify report', os.path.join(d, 'vr', 'verify_report.json'),
         lambda p: ST._write_report(types.SimpleNamespace(name='u', tests=[]), os.path.dirname(p), '/root', {})),
        ('followup _atomic_text', os.path.join(d, 'fu', 'summary.csv'), lambda p: FU._atomic_text(p, 'a\n')),
    ]
    for name, dst, call in cases:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        stale, keep = _plant(dst)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            call(dst)
        names = sorted(os.listdir(os.path.dirname(dst)))
        assert os.path.exists(dst) and not os.path.exists(stale), (name, names)
        assert all(os.path.exists(p) for p in keep), (name, names)
        assert len(names) == len(keep) + 1, (name, names)                      # nothing else left beside the product
        assert err.getvalue().count('placement: removed ') == 1 and stale in err.getvalue(), (name, err.getvalue())
    print(f'ok writers: {len(cases)} converted writers remove their product\'s stale temporary file before writing, and '
          f'nothing else')


def _leftovers(d):
    return sorted(f for f in os.listdir(d) if '.tmp' in f or f.endswith('.part'))


def _sha(p):
    import hashlib
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


class _Cfg:                                     # the run record needs only a config hash here (data-free)
    site_sha1 = None
    overrides = ()

    def hash(self, sections=None):
        return 'test'


def test_tess_install_refused():
    if not CAN_REFUSE:
        print('UNTESTED tess install refused: no group this user cannot set (running as root?)')
        return
    import contextlib
    import io
    from exopipe.phase3.lib.pages import tess_orbits as T
    d = _fresh('tess')
    p = os.path.join(d, 'TESS_orbit_times.csv')

    def table(rows):
        return ('Sector,Orbit,Start of Orbit,End of Orbit\n' + ''.join(f'{a},{b},{c},{e}\n' for a, b, c, e in rows)).encode()
    ended = [(97, 201, '2025-10-01 00:00:00', '2025-10-14 00:00:00')]
    for far, want_rc in ((True, 0), (False, 2)):
        last = [(300, 600, '2099-01-01 00:00:00', '2099-01-14 00:00:00')] if far else []
        cur = table(ended + last)
        new = table(ended + last + [(301, 601, '2099-02-01 00:00:00', '2099-02-14 00:00:00')])
        open(p, 'wb').write(cur)
        os.utime(p, (1700000000, 1700000000))
        real_fetch = T._fetch
        T._fetch = lambda url, timeout: new
        out = io.StringIO()
        try:
            with _refuse_for('TESS_orbit_times.csv'), contextlib.redirect_stdout(out):
                rc = T.main(['--table', p])
        finally:
            T._fetch = real_fetch
        rec = json.loads(out.getvalue())
        assert rc == want_rc, (rc, rec)
        assert rec['fetch'].startswith('refused: install not placed: placement:') and 'TESS_orbit_times.csv' in rec['fetch'], rec
        assert rec.get('mtime_touched') is True and os.stat(p).st_mtime > 1700000000, rec     # "checked now"
        assert open(p, 'rb').read() == cur and rec['sha256_after'] == rec['sha256_before'], rec
        assert os.listdir(d) == ['TESS_orbit_times.csv'], os.listdir(d)
        if not far:
            assert 'install not placed' in rec['refusal'], rec
    print('ok tess: a table that cannot be placed is "refused: install not placed: placement: ..."; the current table\'s '
          'content unchanged and its time touched, no temporary file; exit 0 when the current table reaches far enough ahead, '
          '2 when it does not')


def test_percentiles_not_placed():
    if not CAN_REFUSE:
        print('UNTESTED percentiles not placed: no group this user cannot set (running as root?)')
        return
    from exopipe.phase3.lib.qa import percentiles as QP
    d = _fresh('qa')
    img = os.path.join(d, QP.IMAGES)
    open(img, 'w').write('lst,scope,stokes,date,n_after_morph,n_after_mstruct\n'
                         '15h,41MHz,V,2025-05-01T00,3,1\n15h,41MHz,V,2025-05-01T01,6,2\n')
    path = QP.write_for_chain(d, QP.TENMIN)
    before = _sha(path)
    open(img, 'w').write('lst,scope,stokes,date,n_after_morph,n_after_mstruct\n'
                         '15h,41MHz,V,2025-05-01T00,3,1\n15h,41MHz,V,2025-05-01T01,6,2\n15h,41MHz,V,2025-05-01T02,6,4\n')
    run = record.Run('tplacement', ['test'], os.path.join(d, 'out'), _Cfg(), config_sections=())
    u = run.unit('qa_percentiles')
    with _refuse_for(QP.TABLE):
        with u:
            QP.write_for_chain(d, QP.TENMIN)
    assert u.status == 'refused' and 'not placed: placement:' in u.reason and QP.TABLE in u.reason, (u.status, u.reason)
    assert _sha(path) == before and not _leftovers(d), (_leftovers(d),)
    assert sorted(f for f in os.listdir(d) if f != 'out') == sorted([QP.IMAGES, QP.TABLE]), os.listdir(d)
    print('ok qa: a percentile table that cannot be placed refuses the chain\'s unit ("not placed: placement: ..."); the '
          'table before is unchanged, no temporary file')


def _vclean_cell(root, cell):
    """A cell with a sane V master, a sane I Taper_R0 master and a PB_deep map (32 x 32 pixels)."""
    import numpy as np
    from astropy.io import fits
    lst, scope = cell.split('_', 1)
    cf = os.path.join(root, cell, 'coadd_final')
    os.makedirs(cf)
    n = 32
    rng = np.random.default_rng(5)
    h = fits.Header()
    for k, v in (('CTYPE1', 'RA---SIN'), ('CTYPE2', 'DEC--SIN'), ('CRVAL1', 180.0), ('CRVAL2', 37.0), ('CRPIX1', 16.5),
                 ('CRPIX2', 16.5), ('CDELT1', -0.1), ('CDELT2', 0.1), ('NCOADD', 3), ('DATE-OBS', '2025-01-01T12:00:00'),
                 ('DATE-END', '2025-01-03T13:00:00')):
        h[k] = v
    vp = os.path.join(cf, f'V_deep_Taper_R0_{scope}.fits')
    fits.writeto(vp, rng.normal(0, 1, (n, n)).astype(np.float32), h)
    fits.writeto(os.path.join(cf, f'I_deep_Taper_R0_{scope}.fits'), rng.normal(0, 1, (n, n)).astype(np.float32), h)
    yy, xx = np.mgrid[:n, :n]
    pb = np.clip(1 - 0.8 * (np.hypot(xx - n / 2, yy - n / 2) / (n / 2)) ** 2, 0.02, 1).astype(np.float32)
    fits.writeto(os.path.join(cf, f'PB_deep_{scope}.fits'), pb, h)
    return cf, vp


_REAL_RUN = subprocess.run


def _fake_sv2(scale):
    """subprocess.run (the module's, so every caller's) in place of the cleaning engine: a call with --v_image writes
    <out_dir>/V_residual_final.fits = scale x the V master; any other call is run."""
    from astropy.io import fits

    def run(argv, *a, **kw):
        if '--v_image' not in list(argv):
            return _REAL_RUN(argv, *a, **kw)
        v, out = argv[argv.index('--v_image') + 1], argv[argv.index('--out_dir') + 1]
        with fits.open(v, memmap=False) as h:
            fits.writeto(os.path.join(out, 'V_residual_final.fits'), h[0].data * scale, h[0].header)
        return subprocess.CompletedProcess(argv, 0, '', '')
    return run


def test_vclean_cross_filesystem():
    from astropy.io import fits
    import numpy as np
    from exopipe.phase3.lib.masters import vclean_masters as VM
    d = _fresh('vclean')
    root, cell = os.path.join(d, 'coadd'), '12h_Red'
    cf, vp = _vclean_cell(root, cell)
    os.chmod(cf, 0o2755)                     # the directory's rule: group read, no group write, other read
    other = os.environ.get('EXOPIPE_TEST_OTHER_FS')
    scratch = None
    if other and os.path.isdir(other) and os.stat(other).st_dev != os.stat(d).st_dev:
        scratch = tempfile.mkdtemp(prefix='test_placement_p5_', dir=other)
        tmp_root = scratch
    else:
        tmp_root = os.path.join(d, 'scratch')
        print('UNTESTED vclean across two file systems: set EXOPIPE_TEST_OTHER_FS to a writable directory on another file '
              'system than TMPDIR (the engine\'s scratch is on the product\'s file system here)')
    ap = os.path.join(cf, 'V_deep_Taper_R0_clean_Red.fits')
    pbp = os.path.join(cf, 'V_deep_Taper_R0_clean_pbcorr_Red.fits')
    old = time.time() - 7200
    orphans = [f'{ap}.4194301.0badf00d.tmp', f'{pbp}.4194301.0badf00d.tmp']    # a killed build's: swept first
    for p in orphans:
        open(p, 'w').write('planted\n')
        os.utime(p, (old, old))
    try:
        VM.subprocess.run = _fake_sv2(0.5)
        cell_, st = VM.build_vclean_cell(root, cell, tmp_root=tmp_root, skip_existing=False)
        assert st == 'OK', st
        assert not any(os.path.exists(p) for p in orphans), os.listdir(cf)
        s = os.stat(ap)
        want = PL.target_attrs(os.path.join(cf, 'new_product.fits'))
        assert s.st_gid == want['gid'] and stat.S_IMODE(s.st_mode) == want['mode'] == 0o644, (oct(s.st_mode), want)
        assert np.array_equal(fits.getdata(ap, memmap=False), fits.getdata(vp, memmap=False) * 0.5)
        assert fits.getheader(ap, memmap=False)['VCSRC'] and os.path.exists(pbp)
        assert not _leftovers(cf) and not os.listdir(tmp_root), (_leftovers(cf), os.listdir(tmp_root))
        # a rebuild over the existing master keeps its mode (0600 here), and replaces its content
        os.chmod(ap, 0o600)
        os.remove(pbp)
        VM.subprocess.run = _fake_sv2(0.25)
        st = VM.build_vclean_cell(root, cell, tmp_root=tmp_root, skip_existing=False)[1]
        assert st == 'OK' and stat.S_IMODE(os.stat(ap).st_mode) == 0o600, (st, oct(os.stat(ap).st_mode))
        assert np.array_equal(fits.getdata(ap, memmap=False), fits.getdata(vp, memmap=False) * 0.25)
        pair = lambda: (_sha(ap), _sha(pbp))
        before = pair()
        # a short copy of the cleaned master beside the product (core/placement writes half of each block it reads: size
        # and md5 differ from the bytes read): refused before any placement, neither file placed, the pair before unchanged
        class _Half:
            def __init__(self, f):
                self.f = f

            def __enter__(self):
                return self

            def __exit__(self, *a):
                self.f.close()
                return False

            def write(self, b):
                return self.f.write(bytes(b[:max(1, len(b) // 2)]))
        VM.subprocess.run = _fake_sv2(0.125)
        PL.open = lambda p, mode='r', *a, **kw: (_Half(open(p, mode, *a, **kw)) if mode == 'xb' and
                                                 os.path.dirname(os.path.abspath(p)) == cf else open(p, mode, *a, **kw))
        try:
            st = VM.build_vclean_cell(root, cell, tmp_root=tmp_root, skip_existing=False)[1]
        finally:
            del PL.open
        assert st.startswith('REFUSED placement: the copy ') and 'is not verified' in st and 'nothing placed' in st, st
        assert pair() == before and not _leftovers(cf) and not os.listdir(tmp_root), (_leftovers(cf), os.listdir(tmp_root))
        # the sibling cannot be placed (an attribute refusal): neither file placed
        if CAN_REFUSE:
            with _refuse_for('V_deep_Taper_R0_clean_pbcorr_Red.fits'):
                st = VM.build_vclean_cell(root, cell, tmp_root=tmp_root, skip_existing=False)[1]
            assert st.startswith('REFUSED placement:') and 'none of the 2 products placed' in st, st
            assert pair() == before and not _leftovers(cf) and not os.listdir(tmp_root), _leftovers(cf)
        else:
            print('UNTESTED vclean sibling refused: no group this user cannot set (running as root?)')
        # the sibling's rename fails after the master's: the status says the master is placed and the sibling is not
        real_replace = os.replace

        def fail_sibling(a, b, *x, **kw):
            if os.fspath(b) == pbp:
                raise OSError(5, 'simulated rename failure')
            return real_replace(a, b, *x, **kw)
        PL.os.replace = fail_sibling
        try:
            st = VM.build_vclean_cell(root, cell, tmp_root=tmp_root, skip_existing=False)[1]
        finally:
            PL.os.replace = real_replace
        assert st.startswith('REFUSED placement: renaming') and 'simulated rename failure' in st, st
        assert f"already placed: ['{ap}'] (those stay live)" in st and f"not placed: ['{pbp}']" in st, st
        assert _sha(ap) != before[0] and _sha(pbp) == before[1], 'the master is the new one, the sibling the one before'
        assert np.array_equal(fits.getdata(ap, memmap=False), fits.getdata(vp, memmap=False) * 0.125)
        assert not _leftovers(cf) and not os.listdir(tmp_root), (_leftovers(cf), os.listdir(tmp_root))
        if CAN_REFUSE:
            os.remove(ap)
            os.remove(pbp)
            with _refuse_for('V_deep_Taper_R0_clean_Red.fits'):
                st = VM.build_vclean_cell(root, cell, tmp_root=tmp_root, skip_existing=False)[1]
            assert st.startswith('REFUSED placement:') and 'V_deep_Taper_R0_clean_Red.fits' in st, st
            assert not os.path.exists(ap) and not os.path.exists(pbp), os.listdir(cf)
            assert not _leftovers(cf) and not os.listdir(tmp_root), (_leftovers(cf), os.listdir(tmp_root))
        else:
            print('UNTESTED vclean refused: no group this user cannot set (running as root?)')
    finally:
        VM.subprocess.run = _REAL_RUN
        if scratch:
            shutil.rmtree(scratch, ignore_errors=True)
    print('ok vclean: the cleaned master and its sibling placed together from the engine\'s scratch '
          + ('on another file system ' if scratch else '') + '(the directory\'s rule for new files, the replaced master\'s '
          'mode; a killed build\'s temporary files swept first); a short copy ("REFUSED the copy of the cleaned master is not '
          'verified"), a sibling or master refusal ("REFUSED placement: ... none of the 2 products placed") leave the pair '
          'before; a failed sibling rename leaves the new master placed and says so; no temporary file, no scratch directory')


def test_deep_child_placement_refusal():
    d = _fresh('child')
    spec = {'tier': 'deep', 'cadence': 'deep', 'lst': '09h', 'scope': 'Red', 'stokes': 'I', 'label': 'X',
            'included_dates': [], 'result_path': os.path.join(d, 'result.json')}
    sp = os.path.join(d, 'spec.json')
    json.dump(spec, open(sp, 'w'))
    code = '\n'.join([
        'import os, sys, types',
        'import exopipe.phase3.lib.search as pkg',
        'from exopipe.phase3.lib.frames import deep_child as c',
        'm = types.ModuleType("exopipe.phase3.lib.search.deep_search")',
        'm.__file__ = "fake_engine.py"',
        'def run_deep_cell(lst, scope, label=None, included_dates=None, stokes=None, cadence=None, frames_only=False):',
        '    if os.environ["P5_ENGINE_RAISES"] == "placement":',
        '        from exopipe.phase3.core import placement',
        '        raise placement.PlacementRefusal("placement: /x/frame.fits cannot be given the group test - nothing renamed")',
        '    if os.environ["P5_ENGINE_RAISES"] == "refusal":',
        '        from exopipe.phase3.core import record',
        '        raise record.Refusal("09h Red: a gate of the engine refused")',
        '    raise ValueError("an engine failure")',
        'm.run_deep_cell = run_deep_cell',
        'sys.modules[m.__name__] = m',
        'pkg.deep_search = m',
        'sys.exit(c.main(["deep_child", %r]))'])
    reasons = {'placement': '/x/frame.fits cannot be given the group test - nothing renamed',     # run_child adds 'placement: '
               'refusal': '09h Red: a gate of the engine refused', 'error': 'ValueError: an engine failure'}
    for kind, want in (('placement', ('refused', 'placement', 2)), ('refusal', ('refused', 'engine', 2)),
                       ('error', ('error', None, 1))):
        p = subprocess.run([sys.executable, '-c', code % sp], cwd=REPO, capture_output=True, text=True, timeout=300,
                           env=dict(os.environ, PYTHONPATH=REPO, P5_ENGINE_RAISES=kind))
        res = json.load(open(spec['result_path']))
        assert (res['status'], res.get('gate'), p.returncode) == want, (res, p.returncode, p.stderr[-2000:])
        assert res['reason'] == reasons[kind], res
    print('ok child: an engine PlacementRefusal is status refused, gate placement, exit 2; any other record.Refusal status '
          'refused, gate engine, exit 2; another exception is still an error, exit 1')


def test_coadd_final_refused():
    import contextlib
    import io
    from exopipe.phase3.lib.masters import coadd as CO
    real = CO.run_final_lst_freq
    text = 'placement: /x/I_deep_Taper_R0_Red.fits cannot be given the group test - nothing renamed'

    def fake(lst, freq_str, exclusions):
        if lst == '09h':
            raise PL.PlacementRefusal(text)
        if lst == '10h':
            raise ValueError('an engine failure')
        return {'lst': lst, 'freq': CO.freq_label(freq_str), 'n_good': 3, 'n_total': 3}
    CO.run_final_lst_freq = fake
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            r9 = CO._run_final_worker(('09h', 'Red', {}))
            r10 = CO._run_final_worker(('10h', 'Red', {}))
            r11 = CO._run_final_worker(('11h', 'Red', {}))
            n0 = len(out.getvalue())
            CO.run_final(['09h', '11h'], ['Red'], {}, types.SimpleNamespace(parallel=1))
    finally:
        CO.run_final_lst_freq = real
    o = out.getvalue()
    assert r9 == {'lst': '09h', 'freq': 'Red', 'refused': text} and r10 is None and r11['n_good'] == 3, (r9, r10, r11)
    assert f'REFUSED 09h Red: {text}' in o[:n0] and 'ERROR 10h Red: an engine failure' in o[:n0], o[:n0]
    summ = o[n0:].split('FINAL STAGE COMPLETE', 1)[1]
    assert '1 (LST, freq) combinations processed' in summ and f'REFUSED 09h Red: {text}' in summ, summ
    print('ok coadd: a combination the final stage could not place is REFUSED (its text returned and listed apart from the '
          'combinations processed); another exception is still ERROR')


def test_admit_partial_rename():
    from exopipe.phase3.lib.masters import tier_sublst as TS
    d = _fresh('admit')
    cubes = os.path.join(d, 'cubes')
    os.makedirs(os.path.join(cubes, '09h_2025-01-01'))
    open(os.path.join(cubes, '09h_2025-01-01', 'MANIFEST.txt'), 'w').write('COMPLETE synthetic\n')
    run = record.Run('masters', ['test'], os.path.join(d, 'out'), _Cfg(), config_sections=())
    jf, jdf = TS.plan_paths(run, '09h')
    os.makedirs(os.path.dirname(jf))
    open(jf, 'w').write('old main\n')
    open(jdf, 'w').write('old deep\n')
    nope = os.path.join(d, 'absent')
    ex = types.SimpleNamespace(is_excluded=lambda lst, sb, n: False)
    patches = dict(p3_switches=lambda: [], cube_root=lambda: cubes, pins_ok=lambda: ([], []),
                   exclusion_readers_agree=lambda lst, sb, nights: (ex, None), frozen_weights=lambda lst, sb: nope,
                   cfg=lambda: types.SimpleNamespace(paths=types.SimpleNamespace(DATA_DECISIONS=nope)),
                   solar_audit_path=lambda: nope, nonmember_file=lambda: nope,
                   admit_plan=lambda lst, cells, sbs, nm: (['job main 1'], ['job deep 1'], []),
                   _compare_plan=lambda *a, **kw: None, say=lambda *a, **kw: None)
    real = {k: getattr(TS, k) for k in patches}
    sel = types.SimpleNamespace(lsts=['09h'], admit_key=lambda lst: lst, cells_of=lambda lst: ['09h_2025-01-01'],
                                subbands=['41MHz'])
    real_replace = os.replace

    def fail_deep(a, b, *x, **kw):
        if os.fspath(b) == jdf:
            raise OSError(5, 'simulated rename failure')
        return real_replace(a, b, *x, **kw)
    try:
        for k, v in patches.items():
            setattr(TS, k, v)
        PL.os.replace = fail_deep
        TS.stage_admit(run, types.SimpleNamespace(), sel)
    finally:
        PL.os.replace = real_replace
        for k, v in real.items():
            setattr(TS, k, v)
    u = run.units[-1]
    assert u.status == 'refused' and 'simulated rename failure' in u.reason, (u.status, u.reason)
    assert f"already placed: ['{jf}'] (those stay live)" in u.reason and f"not placed: ['{jdf}']" in u.reason, u.reason
    assert open(jf).read() == 'job main 1\n' and open(jdf).read() == 'old deep\n'
    assert not _leftovers(os.path.dirname(jf)), _leftovers(os.path.dirname(jf))
    print('ok admit: the second job list\'s rename fails after the first\'s: the unit is refused and its record names the '
          'list that is live (the new main list) and the one that is not (the deep list before); no temporary file')


def _gaul_block():
    """The deep search's parent-catalogue statement (``if not os.path.exists(_gaulp): ...`` in run_deep_cell), compiled on
    its own: the module needs the cluster's data to import."""
    import ast
    import textwrap
    src = open(os.path.join(REPO, 'exopipe', 'phase3', 'lib', 'search', 'deep_search.py')).read()
    hits = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.If)
            and ast.get_source_segment(src, n.test) == 'not os.path.exists(_gaulp)']
    assert len(hits) == 1, len(hits)
    n = hits[0]
    return compile(textwrap.dedent(' ' * n.col_offset + ast.get_source_segment(src, n)), 'deep_search.py:gaul', 'exec')


def test_gaul_empty_extraction():
    import numpy as np
    from astropy.io import fits
    code = _gaul_block()
    d = _fresh('gaul')
    catdir = os.path.join(d, 'catdir')
    os.makedirs(catdir)
    gaulp = os.path.join(catdir, 'master_R075_09h_Red.gaul.csv')
    mI = os.path.join(d, 'I_deep_Taper_R075_Red.fits')

    class Img:
        def __init__(self, n):
            self.sources = [object()] * n

        def write_catalog(self, outfile, **kw):
            if not self.sources:
                print('No sources were found in the image. Output file not written.')
                return False
            open(outfile, 'w').write('# Gaussian list\n')
            return True
    bdsf = types.ModuleType('bdsf')
    real_bdsf = sys.modules.get('bdsf')
    sys.modules['bdsf'] = bdsf

    def ns():
        return dict(os=os, np=np, fits=fits, placement=PL, time=time, _gaulp=gaulp, cell_work=d, lst='09h', scope='Red',
                    _pdata=np.zeros((8, 8)), _det_header=lambda *a: fits.Header(), _phdr={'NCOADD': 3, 'CDELT1': -0.01}, em=3.0, _pm=mI,
                    mI=mI)
    try:
        bdsf.process_image = lambda path, **kw: Img(0)
        try:
            exec(code, ns())
            raise AssertionError('an extraction with no sources was not refused')
        except RuntimeError as e:
            msg = str(e)
        assert msg.startswith('09h Red: parent catalogue not written - PyBDSF found no sources in I_deep_Taper_R075_Red.fits')
        assert msg.endswith('REFUSING') and 'placement' not in msg, msg     # deep_child: refused, gate 'engine'
        assert os.listdir(catdir) == [] and sorted(os.listdir(d)) == ['catdir'], (os.listdir(catdir), os.listdir(d))
        stale = f'{gaulp}.4194301.0badf00d.tmp'                   # a killed earlier build's temporary catalogue: swept
        open(stale, 'w').write('planted\n')
        os.utime(stale, (time.time() - 7200,) * 2)
        bdsf.process_image = lambda path, **kw: Img(4)
        exec(code, ns())
        assert sorted(os.listdir(catdir)) == sorted([os.path.basename(gaulp), os.path.basename(gaulp) + '.gen.json'])
        assert open(gaulp).read() == '# Gaussian list\n' and sorted(os.listdir(d)) == ['catdir'], os.listdir(d)
    finally:
        if real_bdsf is None:
            sys.modules.pop('bdsf', None)
        else:
            sys.modules['bdsf'] = real_bdsf
    print('ok gaul: an extraction with no sources refuses before any placement ("parent catalogue not written - PyBDSF '
          'found no sources ...; REFUSING", not a placement reason), no catalogue, no temporary file; a killed build\'s '
          'temporary catalogue is swept before the next')


def main():
    shutil.rmtree(TMP, ignore_errors=True)
    os.makedirs(TMP)
    real_changed, PL._changed = PL._changed, lambda st: st.st_mtime    # planted leftovers are aged by utime (mtime)
    try:
        test_sweep()
        test_writers_sweep()
        test_tess_install_refused()
        test_percentiles_not_placed()
        test_vclean_cross_filesystem()
        test_coadd_final_refused()
        test_admit_partial_rename()
        test_gaul_empty_extraction()
        test_deep_child_placement_refusal()
    finally:
        PL._changed = real_changed
        shutil.rmtree(TMP, ignore_errors=True)
    print('PASS test_placement_p5')


if __name__ == '__main__':
    main()
