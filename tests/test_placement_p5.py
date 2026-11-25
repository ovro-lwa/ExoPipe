"""Placement batch P5 (core/placement in the writers of tess_orbits, qa/percentiles, masters/vclean_masters and the frames-only
engine child): what a placement refusal does at each handler that would otherwise crash on it or swallow it. Data-free:
scratch directories under TMPDIR; a refusal is forced by giving the placement a group this user is not in (UNTESTED is
printed when there is none, e.g. as root). Checks:
  tess     refresh: a table that cannot be placed is recorded 'install refused: placement: ...' (not a crash of main, not
           'installed'); the current table is unchanged, its time not touched, no temporary file left; main's exit status is
           still the current table's: 0 when it reaches far enough ahead, 2 when it does not;
  qa       write_for_chain: a table that cannot be placed refuses the chain's unit (record.Refusal 'not placed: placement:
           ...', recorded 'refused'); the table before is unchanged, no temporary file left;
  vclean   build_vclean_cell: the cleaned master is copied from the cleaning engine's scratch directory to a temporary file
           beside the product and placed (the directory's rule for a new one; the replaced one's group and mode); a refusal
           returns 'REFUSED placement: ...' (the vclean stage refuses on any status but 'OK') with no product, no sibling, no
           temporary file and no scratch directory left. Across two file systems when EXOPIPE_TEST_OTHER_FS names a writable
           directory on another file system than TMPDIR (UNTESTED otherwise);
  child    deep_child: an engine that raises a PlacementRefusal leaves result status 'refused', gate 'placement', exit 2; any
           other exception is still status 'error', exit 1."""
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile

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
        assert rec['fetch'].startswith('install refused: placement:') and 'TESS_orbit_times.csv' in rec['fetch'], rec
        assert 'mtime_touched' not in rec and os.stat(p).st_mtime == 1700000000, rec
        assert open(p, 'rb').read() == cur and rec['sha256_after'] == rec['sha256_before'], rec
        assert os.listdir(d) == ['TESS_orbit_times.csv'], os.listdir(d)
        if not far:
            assert 'install refused' in rec['refusal'], rec
    print('ok tess: a table that cannot be placed is "install refused: placement: ..."; the current table unchanged, its time '
          'not touched, no temporary file; exit 0 when the current table reaches far enough ahead, 2 when it does not')


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
    try:
        VM.subprocess.run = _fake_sv2(0.5)
        cell_, st = VM.build_vclean_cell(root, cell, tmp_root=tmp_root, skip_existing=False)
        assert st == 'OK', st
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
    print('ok vclean: the cleaned master placed from the engine\'s scratch ' + ('on another file system ' if scratch else '')
          + '(the directory\'s rule for a new one, the replaced one\'s mode); a refusal is "REFUSED placement: ..." with no '
          'product, no sibling, no temporary file, no scratch directory')


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
        '    raise ValueError("an engine failure")',
        'm.run_deep_cell = run_deep_cell',
        'sys.modules[m.__name__] = m',
        'pkg.deep_search = m',
        'sys.exit(c.main(["deep_child", %r]))'])
    for kind, want in (('placement', ('refused', 'placement', 2)), ('error', ('error', None, 1))):
        p = subprocess.run([sys.executable, '-c', code % sp], cwd=REPO, capture_output=True, text=True, timeout=300,
                           env=dict(os.environ, PYTHONPATH=REPO, P5_ENGINE_RAISES=kind))
        res = json.load(open(spec['result_path']))
        assert (res['status'], res.get('gate'), p.returncode) == want, (res, p.returncode, p.stderr[-2000:])
        assert ('placement: /x/frame.fits' in res['reason']) == (kind == 'placement'), res
    print('ok child: an engine PlacementRefusal is status refused, gate placement, exit 2; another exception is still an '
          'error, exit 1')


def main():
    shutil.rmtree(TMP, ignore_errors=True)
    os.makedirs(TMP)
    try:
        test_tess_install_refused()
        test_percentiles_not_placed()
        test_vclean_cross_filesystem()
        test_deep_child_placement_refusal()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print('PASS test_placement_p5')


if __name__ == '__main__':
    main()
