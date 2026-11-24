"""The parent-catalog union: one config value for the current union and the digest gate on its files (core/manifest.py;
FINDING-20261118-PARENT-UNION-OTHER-STEPS-AND-NO-MANIFEST). Data-free: scratch unions under TMPDIR. With /lustre reachable, also
the pinned manifests of the real unions."""
import hashlib
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exopipe.phase3.config import CFG, ConfigRefusal  # noqa: E402
from exopipe.phase3.core import manifest as MF  # noqa: E402

TMP = os.path.join(os.environ.get('TMPDIR', tempfile.gettempdir()), 'test_parent_union_gate')


def _raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc as e:
        return str(e)
    raise AssertionError(f'{getattr(fn, "__name__", fn)} did not raise {exc.__name__}')


def _union(d, n=3):
    os.makedirs(d, exist_ok=True)
    for i in range(n):
        with open(os.path.join(d, f'master_R075_{i:02d}h_Blue.gaul.csv'), 'w') as f:
            f.write(f'# header\n0,0,0,0,{10 * i},0,20,0,0,0,{i + 1}\n')
        with open(os.path.join(d, f'master_R075_{i:02d}h_Blue.gaul.csv.gen.json'), 'w') as f:
            f.write('{"master_md5_data": "%032d"}\n' % i)
    return d


def test_config_one_value():
    P = CFG.paths
    assert P.PARENT_UNION == '/lustre/gh/main/phase3/deep_parent_catalogs_union_20261118T0130'
    assert P.PARENT_UNION_20261014 == '/lustre/gh/main/phase3/deep_parent_catalogs_union_20261014T0203'
    assert set(P.PARENT_UNION_MANIFEST_MD5) == {os.path.basename(P.PARENT_UNION), os.path.basename(P.PARENT_UNION_20261014)}
    # the readers of the current union hold None (= paths.PARENT_UNION); the binding mirror names it
    assert CFG.hybrid_recon.CATALOGS is None and CFG.mosaic_content.PARENTS is None
    assert MF.parent_union(CFG.hybrid_recon.CATALOGS, CFG) == MF.parent_union(CFG.mosaic_content.PARENTS, CFG) == P.PARENT_UNION
    assert CFG.catalog.DEEP_PARENT_DIR_DEFAULT == P.PARENT_UNION
    # the production-script locks name what production's p3s5_parent_union.py CANDS names (the 10-14 union today); they move
    # with that script (Gregg 2026-11-24): the lock refuses the step once the script names another source
    assert CFG.catalog.DEEP_PARENT_CANDIDATES[-1] == CFG.calibrate_bright_parent.PARENT_UNION_SOURCES[-1] == P.PARENT_UNION_20261014
    assert CFG.masters_striation.PARENTS is None       # the 11-18 union (Gregg 2026-11-24)
    for key in ('paths.PARENT_UNION', 'paths.PARENT_UNION_20261014', 'paths.PARENT_UNION_MANIFEST_MD5'):
        _raises((ConfigRefusal, KeyError, ValueError), CFG.set, key, "'x'", source='test')
    assert MF.parent_union('/elsewhere', CFG) == '/elsewhere'                     # a step's pin wins


def test_gate():
    d = _union(os.path.join(TMP, 'u'))
    try:
        assert _raises(MF.ManifestRefusal, MF.verify_union, d, {}).startswith('M1')            # no manifest
        assert _raises(MF.ManifestRefusal, MF.verify_union, os.path.join(TMP, 'none'), {}).startswith('M1')
        n, mmd5 = MF.write_manifest(d, os.path.join(d, MF.MANIFEST))
        assert n == 6 and mmd5 == hashlib.md5(open(os.path.join(d, MF.MANIFEST), 'rb').read()).hexdigest()
        lines = open(os.path.join(d, MF.MANIFEST)).read().splitlines()
        assert lines == sorted(lines, key=lambda x: x[34:]) and all(len(x.split('  ')[0]) == 32 for x in lines)
        pins = {'u': mmd5}
        r = MF.verify_union(d, pins)
        assert r == {'dir': d, 'union': 'u', 'manifest_md5': mmd5, 'n_files': 6, 'checked': 6}, r
        assert MF.verify_union(d, pins, names=['master_R075_01h_Blue.gaul.csv'])['checked'] == 1
        assert _raises(MF.ManifestRefusal, MF.verify_union, d, {'other': '0' * 32}).startswith('M2')
        assert _raises(MF.ManifestRefusal, MF.verify_union, d, pins, names=['master_R075_09h_Blue.gaul.csv']).startswith('M4')
        open(os.path.join(d, 'MASTER_MD5_DATA.tsv'), 'w').write('x\n')                         # a sidecar: not listed, admitted
        assert MF.verify_union(d, pins)['checked'] == 6
        # a copy elsewhere with the same manifest passes (the pin is by value)
        c = os.path.join(TMP, 'copy')
        shutil.copytree(d, c)
        assert MF.verify_union(c, pins)['union'] == 'u'
        # one catalog changed: refused when read (M4), not when another is read
        with open(os.path.join(c, 'master_R075_01h_Blue.gaul.csv'), 'a') as f:
            f.write('0,0,0,0,1,0,2,0,0,0,3\n')
        e = _raises(MF.ManifestRefusal, MF.verify_union, c, pins)
        assert e.startswith('M4: 1 of 6') and 'master_R075_01h_Blue.gaul.csv' in e, e
        assert MF.verify_union(c, pins, names=['master_R075_02h_Blue.gaul.csv'])['checked'] == 1
        # a file added or removed: M3
        open(os.path.join(d, 'master_R075_05h_Blue.gaul.csv'), 'w').write('# new\n')
        assert _raises(MF.ManifestRefusal, MF.verify_union, d, pins, names=[]).startswith('M3')
        os.remove(os.path.join(d, 'master_R075_05h_Blue.gaul.csv'))
        os.remove(os.path.join(d, 'master_R075_00h_Blue.gaul.csv'))
        assert _raises(MF.ManifestRefusal, MF.verify_union, d, pins, names=[]).startswith('M3')
        # a malformed manifest line
        with open(os.path.join(c, MF.MANIFEST), 'a') as f:
            f.write('nonsense\n')
        assert _raises(MF.ManifestRefusal, MF.read_manifest, os.path.join(c, MF.MANIFEST)).startswith('M1')
    finally:
        shutil.rmtree(TMP, ignore_errors=True)


def test_gate_cache():
    """gate_union: once per process per directory (cached on the manifest md5); a changed file is hashed again and refused;
    gate_pinned gates only pinned unions."""
    d = _union(os.path.join(TMP, 'uc'))
    try:
        n, mmd5 = MF.write_manifest(d, os.path.join(d, MF.MANIFEST))
        pins = {'uc': mmd5}
        assert MF.gate_union(d, None, pins)['hashed'] == 6
        r = MF.gate_union(d, None, pins)
        assert r['hashed'] == 0 and r['checked'] == 6, r                       # cached: nothing hashed again
        assert MF.gate_union(d, ['master_R075_01h_Blue.gaul.csv'], pins)['hashed'] == 0
        f = os.path.join(d, 'master_R075_01h_Blue.gaul.csv')
        open(f, 'a').write('0,0,0,0,0,0,0,0,0,0,9\n')                          # changed after its check
        assert _raises(MF.ManifestRefusal, MF.gate_union, d, None, pins).startswith('M4')
        assert _raises(MF.ManifestRefusal, MF.gate_union, d, None, {'uc': '0' * 32}).startswith('M2')
        assert MF.gate_pinned([os.path.join(TMP, 'not_a_union')], cfg=CFG) == {os.path.join(TMP, 'not_a_union'): None}
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_hybrid_compact_ratio_gate():
    """recon.compact_ratio reads one catalog; the gate refuses it (ReconRefusal) before any measurement."""
    from exopipe.phase3.lib.hybrid import recon as REC
    d = _union(os.path.join(TMP, 'h'))
    try:
        REC.bind()
        assert REC.CATALOGS == CFG.paths.PARENT_UNION and REC.CATALOG_PINS == CFG.paths.PARENT_UNION_MANIFEST_MD5
        old = REC.CATALOGS, REC.CATALOG_PINS
        try:
            REC.CATALOGS, REC.CATALOG_PINS = d, {}
            hdr = {'CDELT1': -0.1, 'BMAJ': 0.5}
            img = np.zeros((8, 8))
            e = _raises(REC.ReconRefusal, REC.compact_ratio, img, img, hdr, '01h', 'Blue')
            assert 'M1' in e and 'master_R075_01h_Blue.gaul.csv' in e, e
            _n, mmd5 = MF.write_manifest(d, os.path.join(d, MF.MANIFEST))
            REC.CATALOG_PINS = {'h': mmd5}
            with open(os.path.join(d, 'master_R075_01h_Blue.gaul.csv'), 'a') as f:
                f.write('# changed\n')
            e = _raises(REC.ReconRefusal, REC.compact_ratio, img, img, hdr, '01h', 'Blue')
            assert 'M4' in e, e
        finally:
            REC.CATALOGS, REC.CATALOG_PINS = old
    finally:
        shutil.rmtree(TMP, ignore_errors=True)


def test_real_unions():
    """The pins equal the md5 of each real union's MANIFEST.md5 where it is placed (the 11-18 one: once the refactor chat places it)."""
    for name, pin in sorted(CFG.paths.PARENT_UNION_MANIFEST_MD5.items()):
        m = os.path.join(os.path.dirname(CFG.paths.PARENT_UNION), name, MF.MANIFEST)
        if not os.path.isfile(m):
            print(f'  UNTESTED-CLUSTER: {m} absent')
            continue
        got = hashlib.md5(open(m, 'rb').read()).hexdigest()
        assert got == pin, (name, got, pin)
        assert len(MF.read_manifest(m)) == 912, name


TESTS = [test_config_one_value, test_gate, test_gate_cache, test_hybrid_compact_ratio_gate, test_real_unions]


def main():
    for t in TESTS:
        t()
        print(f'PASS {t.__name__}')
    print(f'ALL {len(TESTS)} PASS')


if __name__ == '__main__':
    main()
