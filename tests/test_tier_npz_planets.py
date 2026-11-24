#!/usr/bin/env python3
"""Data-free tests of the tier npz content test TN1 on the planet sample (lib/photometry/tests_cutouts.py tn1).

The tier measurement (photometry_tier_forced_20261026.py, run in place) writes a planet file (sample photometry.PLANET_SAMPLE,
'SolarSystem') with the planet's own track (ra10, dec10: one position per plane) and ORBIT controls: ctl_kind 'orbit' and
ctl_ra / ctl_dec, nc control positions per plane along the planet's track (:148). TN1 refuses a planet file without them - a
regression to ring controls - and names every planet target that lacks controls at a plane where it has a position.
A fixed-target sample (HWJ) needs none of these arrays.

Run: python tests/test_tier_npz_planets.py [test names]   (exit 0 = all passed, 1 = a failure, 2 = imports missing)
"""
import inspect
import os
import shutil
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')

_IMPORT_ERROR = None
try:
    import numpy as np

    from exopipe.phase3.config import CFG
    from exopipe.phase3.lib.photometry import tests_cutouts as TC
except Exception as e:  # noqa: BLE001
    _IMPORT_ERROR = e
    if __name__ != '__main__':
        raise

T, N = 3, 4                      # planet targets, planes
NAMES = ('Jupiter', 'Saturn', 'Uranus')


def npz(tmp, sample, drop=(), **over):
    """A tier npz of ``sample`` (T targets x N planes); planets with a position at every plane but the last of Saturn, and
    complete orbit controls there. ``over`` replaces arrays, ``drop`` leaves keys out. Returns its path."""
    nc = CFG.photometry_measure.N_CONTROLS
    d = dict(names=np.array(NAMES), ra=np.arange(T, dtype=float), dec=np.arange(T, dtype=float), ring_beams=np.full(T, 3.0),
             ts=np.arange(N), night=np.array(['2024-02-0%d' % (i + 1) for i in range(N)]),
             key=np.array(['k%d' % i for i in range(N)]), admitted=np.ones(N, bool),
             val10=np.zeros((T, nc + 1, N)), rms10=np.ones((T, nc + 1, N)))
    if sample == CFG.photometry.PLANET_SAMPLE:
        ra10 = np.full((T, N), 10.0)
        ra10[1, -1] = np.nan                                     # Saturn out of the field at the last plane
        cra = np.repeat(ra10[:, None, :], nc, axis=1) + 0.5
        d.update(ra10=ra10, dec10=ra10.copy(), ctl_kind=np.array('orbit'), ctl_ra=cra, ctl_dec=cra.copy())
    d.update(over)
    for k in drop:
        d.pop(k, None)
    p = os.path.join(tmp, 'forced_%s_Red_I.npz' % sample)
    np.savez(p, **d)
    return p


class Tmp:
    def __enter__(self):
        self.dir = tempfile.mkdtemp(prefix='tier_npz_planets_')
        return self.dir

    def __exit__(self, *exc):
        shutil.rmtree(self.dir, ignore_errors=True)


def test_a_planet_file_with_orbit_controls_passes():
    with Tmp() as tmp:
        ok, n, why = TC.tn1(npz(tmp, CFG.photometry.PLANET_SAMPLE))
        assert ok and n == T, (ok, n, why)


def test_a_fixed_target_sample_needs_no_track_or_controls():
    with Tmp() as tmp:
        ok, n, why = TC.tn1(npz(tmp, 'HWJ'))
        assert ok and n == T, (ok, n, why)


def test_a_planet_file_without_orbit_controls_is_refused():
    P = CFG.photometry.PLANET_SAMPLE
    with Tmp() as tmp:
        ok, _n, why = TC.tn1(npz(tmp, P, drop=('ctl_kind', 'ctl_ra', 'ctl_dec')))       # a file of ring controls
        assert not ok and 'ctl_ra None' in why and "ctl_kind '(none)' != 'orbit'" in why, why
        ok, _n, why = TC.tn1(npz(tmp, P, drop=('ctl_kind',)))                            # the kind not recorded
        assert not ok and "ctl_kind '(none)' != 'orbit'" in why, why
        ok, _n, why = TC.tn1(npz(tmp, P, ctl_kind=np.array('ring')))
        assert not ok and "ctl_kind 'ring' != 'orbit'" in why, why
        nc = CFG.photometry_measure.N_CONTROLS
        ok, _n, why = TC.tn1(npz(tmp, P, ctl_ra=np.zeros((T, nc - 1, N))))              # a control short
        assert not ok and 'ctl_ra (%d, %d, %d) != (%d, %d, %d)' % (T, nc - 1, N, T, nc, N) in why, why


def test_a_planet_target_without_controls_is_refused_by_name():
    P = CFG.photometry.PLANET_SAMPLE
    nc = CFG.photometry_measure.N_CONTROLS
    with Tmp() as tmp:
        ra10 = np.full((T, N), 10.0)
        ra10[1, -1] = np.nan
        cra = np.repeat(ra10[:, None, :], nc, axis=1) + 0.5
        cra[2, 5, 1] = np.nan                                 # Uranus: one control missing at a plane with its position
        ok, n, why = TC.tn1(npz(tmp, P, ctl_ra=cra))
        assert not ok and n == 1 and why == 'planet target(s) without orbit controls: Uranus (1 of 4 planes with its position ' \
                                            'lack orbit controls)', why
        cdec = np.repeat(ra10[:, None, :], nc, axis=1)
        cdec[:, :, 0] = np.nan                               # every planet: no control at the first plane
        ok, n, why = TC.tn1(npz(tmp, P, ctl_dec=cdec))
        assert not ok and n == 3 and all(x in why for x in NAMES), why
        r = ra10.copy()
        r[0] = np.nan                                        # Jupiter: no plane with its own position
        ok, n, why = TC.tn1(npz(tmp, P, ra10=r))
        assert not ok and n == 1 and 'Jupiter (no plane with its own position)' in why, why
        # a control missing only where the planet has no position (Saturn, last plane) is not a lack
        c = np.repeat(ra10[:, None, :], nc, axis=1)
        assert np.isnan(c[1, :, -1]).all()
        ok, n, why = TC.tn1(npz(tmp, P, ctl_ra=c, ctl_dec=c.copy()))
        assert ok and n == T, why


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
