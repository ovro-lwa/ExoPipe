#!/usr/bin/env python3
"""Data-free tests of `frames --tier sublst --stages tree-nonmember` (lib/frames/sublst_tree_nonmember.py): the entry of
ONE night of co-add weight 0 in the tree the 10-minute search reads.

The reference is `builder` below: the loop of clv15/phase3_sublst_tenmin_tree.py (lines 95-145 at clv15 5b1fc5b and
later), transcribed. Every test builds a synthetic sub-LST root (window primary-beam maps, x2 residuals, frozen deep
weights, the non-member nights list, a refine log), builds the tree of the member nights with `builder`, runs the unit for
the new night, and compares the result with `builder` over all the nights: frames.csv byte for byte, the links, the PB_bin
files byte for byte. On real data the same comparison is made against the production script itself (the record
exopipe_ops refactor_20261012/plan/stage_map_tools/perhour_tree_nonmember_20261113/RECORD.md).

Run: python tests/test_sublst_tree_nonmember.py [test names]   (exit 0 = all passed, 1 = a failure, 2 = imports missing)
"""
import argparse
import csv
import glob
import inspect
import os
import re
import shutil
import sys
import tempfile
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')

_IMPORT_ERROR = None
try:
    import numpy as np
    from astropy.io import fits

    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import record, validate
    from exopipe.phase3.lib.frames import sublst_tree_nonmember as TN
    from exopipe.phase3.lib.frames import tests_sublst as XS
    from exopipe.phase3.lib.frames import tier_sublst as FT
    from exopipe.phase3.lib.masters import tier_sublst as TS
except Exception as e:  # noqa: BLE001
    _IMPORT_ERROR = e
    if __name__ != '__main__':
        raise

LST = '09h'
SB = '55MHz'
MEMBERS = {'2024-01-02': 0.8, '2024-01-03': 1.3, '2024-01-05': 0.55}   # the weighted nights and their co-add weights
MEMBER_WINDOWS = {'2024-01-02': (0, 1, 2), '2024-01-03': (0, 1, 2), '2024-01-05': (0, 1)}
NEW = '2024-02-01'                      # the night that is not in the Masters: later than every member
EARLY = '2023-12-30'                    # another one: earlier than every member
SHAPE = (12, 10)
COLS = list(CFG.frames_sublst.TREE_COLUMNS) if _IMPORT_ERROR is None else []


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


def builder(root, lst, sb, wts, listed=(), excluded=()):
    """The tree of one band as clv15/phase3_sublst_tenmin_tree.py builds it from every night under <root>/<lst>_x2 (the
    script's loop, transcribed; `wts` = the frozen weights rows, `listed` = the non-member nights list). Returns the
    final line's counts (windows, frames, skipped)."""
    x2, sroot, tree = os.path.join(root, f'{lst}_x2'), os.path.join(root, lst), os.path.join(root, 'tenmin_root_x2')
    pat = re.compile(r"sublst_%s_(\d{4}-\d\d-\d\d)_(\d+MHz|Red|Green|Blue|Full)_([IV])_w(\d+)_resid_loo\.fits$" % lst)
    found = {}
    for p in glob.glob(os.path.join(x2, "%s_*" % lst, sb, "?", "sublst_*_resid_loo.fits")):
        m = pat.search(os.path.basename(p))
        if m and m.group(2) == sb:
            found.setdefault((m.group(1), int(m.group(4))), {})[m.group(3)] = p
    cdir = os.path.join(tree, "%s_%s" % (lst, sb))
    sdir, bdir = os.path.join(cdir, "subtracted"), os.path.join(cdir, "bin_masters")
    for d in (sdir, bdir):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
    wts = dict(wts)
    for n in set(listed) - set(wts):
        wts[n] = 0.0
    pbsum, pbw, pbh, pbu, pbn = {}, {}, {}, {}, {}
    rows, skipped = [], []
    for (night, k), st in sorted(found.items()):
        if night in excluded:
            skipped.append("%s w%d (excluded in data_decisions.csv)" % (night, k))
            continue
        if set(st) != {"I", "V"}:
            skipped.append("%s w%d (only %s)" % (night, k, "".join(sorted(st))))
            continue
        key = "%sT%02d" % (night, k)
        pbp = os.path.join(sroot, "%s_%s" % (lst, night), sb, "I", "sublst_%s_%s_%s_I_w%d_pb.fits" % (lst, night, sb, k))
        assert os.path.exists(pbp) and night in wts, 'REFUSED: no window PB %s or no co-add weight for %s' % (pbp, night)
        with fits.open(pbp, memmap=False) as hp:
            pbd = np.nan_to_num(np.asarray(hp[0].data, dtype=np.float64).squeeze())
            pbh.setdefault(k, hp[0].header.copy())
        pbsum[k] = pbsum.get(k, 0.0) + wts[night] * pbd
        pbw[k] = pbw.get(k, 0.0) + wts[night]
        pbu[k] = pbu.get(k, 0.0) + pbd
        pbn[k] = pbn.get(k, 0) + 1
        for s in ("I", "V"):
            h = fits.getheader(st[s])
            assert not ("BINMSTR" in h or "LSTBIN" not in h or "DURATION" not in h), 'REFUSED: %s' % st[s]
            link = os.path.join(sdir, "%s_%s_sub_%s.fits" % (key, s, sb))
            os.symlink(st[s], link)
            rows.append({"key": key, "date": night, "bin": k, "stokes": s, "src": st[s], "dc": h.get("FLUXDC", ""),
                         "has_flow": 1, "src_dewarped": "", "pb_blank_frac": "", "pb_ratio_median": "",
                         "bmaj_deg": h.get("BMAJ", ""), "bmin_deg": h.get("BMIN", ""), "bpa_deg": h.get("BPA", ""),
                         "n_inject": 0, "sub_path": link, "fluxbake": int(bool(h.get("FLUXBAKE", False)))})
    for k in sorted(pbsum):
        hh = pbh[k]
        hh["PBSRC"] = ("sub-LST window PB, night-weighted mean", "phase3_sublst_tenmin_tree.py")
        hh["PBNNIGHT"] = (sum(1 for r in rows if r["bin"] == k and r["stokes"] == "I"), "nights averaged")
        pbm = pbsum[k] / pbw[k] if pbw[k] > 0 else pbu[k] / pbn[k]
        if not pbw[k] > 0:
            hh["PBSRC"] = ("sub-LST window PB, unweighted mean (non-member nights only)", "phase3_sublst_tenmin_tree.py")
        fits.writeto(os.path.join(bdir, "PB_bin%d_%s.fits" % (k, sb)), pbm.astype(np.float32), hh, overwrite=True)
    with open(os.path.join(cdir, "frames.csv"), "w") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    return len(rows) // 2, len(rows), skipped


class Tree:
    """One synthetic sub-LST root in a temporary directory, with its own exclusion list, frozen weights and non-member
    nights list (the readers are pointed at them inside the with-block)."""

    def __init__(self, new=(NEW,), new_windows=(0, 1, 2), seed=0, pb_axes=2):
        self.dir = tempfile.mkdtemp(prefix='sublst_tree_nonmember_')
        self.root = os.path.join(self.dir, 'sublst')
        self.tree = os.path.join(self.root, 'tenmin_root_x2')
        self.cdir = os.path.join(self.tree, f'{LST}_{SB}')
        self.fcsv = os.path.join(self.cdir, 'frames.csv')
        self.rng = np.random.RandomState(seed)
        self.new = tuple(new)
        self.listed = list(new)
        self.decisions = os.path.join(self.dir, 'data_decisions.csv')
        self.listfile = os.path.join(self.dir, 'sublst_nonmember_nights.txt')
        self.frozen = os.path.join(self.dir, 'frozen')
        self.write_decisions(())
        self.write_list()
        self.write_weights(MEMBERS)
        for night, ks in MEMBER_WINDOWS.items():
            self.night(night, ks)
        self.write_log()
        self._patch = Patch((CFG.paths, 'DATA_DECISIONS', self.decisions), (TS, 'frozen_records', lambda: self.frozen),
                            (TS, 'nonmember_file', lambda: self.listfile), (TS, 'mirror_violations', lambda: []))
        self.before = None
        self.new_windows, self.pb_axes = tuple(new_windows), pb_axes

    def __enter__(self):
        self._patch.__enter__()
        builder(self.root, LST, SB, MEMBERS)                    # the tree of the member nights
        for n in self.new:
            self.night(n, self.new_windows, axes=self.pb_axes)
        self.write_log()
        self.before = self.snapshot()
        return self

    def __exit__(self, *exc):
        self._patch.__exit__(*exc)
        shutil.rmtree(self.dir, ignore_errors=True)

    # ---- inputs ----------------------------------------------------------------------------------------------------
    def write_decisions(self, excluded):
        with open(self.decisions, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['lst', 'subband', 'date', 'verdict', 'reason'])
            w.writerow(['23h', '18', '2020-01-01', 'exclude', 'unrelated'])
            for lst, sb, night in excluded:
                w.writerow([lst, sb.replace('MHz', ''), night, 'exclude', 'test'])

    def write_list(self):
        with open(self.listfile, 'w') as f:
            f.write('# LST NIGHT\n' + ''.join(f'{LST} {n}\n' for n in self.listed))

    def write_weights(self, wts):
        p = os.path.join(self.frozen, LST, SB, 'weights.csv')
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['date', 'weight', 'in_master'])
            for n, x in wts.items():
                w.writerow([n, repr(x), 1])
        assert TS.frozen_weights.__module__      # the resolver the unit and the content test read through

    def pb(self, night, k):
        return os.path.join(self.root, LST, f'{LST}_{night}', SB, 'I', f'sublst_{LST}_{night}_{SB}_I_w{k}_pb.fits')

    def resid(self, night, S, k):
        return os.path.join(self.root, f'{LST}_x2', f'{LST}_{night}', SB, S, f'sublst_{LST}_{night}_{SB}_{S}_w{k}_resid_loo.fits')

    def night(self, night, ks, stokes=('I', 'V'), axes=2):
        for k in ks:
            pb = self.rng.uniform(0.2, 0.9, SHAPE)
            pb[0, 0] = np.nan
            h = fits.Header()
            h['BUNIT'] = 'beam'
            h['DATE-OBS'] = (f'{night}T10:{k}0:00.000', 'first frame of the window')
            h['LSTWIN'] = k
            h['HISTORY'] = f'window primary beam of {night}'
            os.makedirs(os.path.dirname(self.pb(night, k)), exist_ok=True)
            # the night's map as 64-bit, or with two degenerate axes: the writer of PB_bin sets its own BITPIX and NAXIS
            fits.writeto(self.pb(night, k), pb.reshape((1, 1) + SHAPE) if axes == 4 else pb.astype(np.float32), h,
                         overwrite=True)
            for S in stokes:
                rh = fits.Header()
                rh['LSTBIN'], rh['DURATION'] = k, 600.0
                rh['FLUXDC'] = float(self.rng.uniform(0.9, 1.1))
                rh['BMAJ'], rh['BMIN'], rh['BPA'] = (float(self.rng.uniform(0.2, 0.3)), float(self.rng.uniform(0.1, 0.2)),
                                                     float(self.rng.uniform(0, 90)))
                rh['FLUXBAKE'] = True
                os.makedirs(os.path.dirname(self.resid(night, S, k)), exist_ok=True)
                fits.writeto(self.resid(night, S, k), self.rng.normal(0, 1, SHAPE).astype(np.float32), rh, overwrite=True)

    def write_log(self, final=True, refused=0):
        p = os.path.join(self.root, f'{LST}_x2', 'logs', f'{LST}_{SB}.log')
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, 'w') as f:
            f.write(f'ALIGNED {LST} {SB} w0 {NEW} median_flow=0.100 px\n')
            if final:
                f.write(f'SUBLST-REFINE {LST} {SB} aligned=9 refused={refused} refused_rule=0 refused_night=0\n')
        return p

    # ---- the tree as it is ---------------------------------------------------------------------------------------------
    def snapshot(self):
        """{relative path: link target, or file bytes} of the band's tree (the lease directory apart)."""
        out = {}
        for d, _dirs, files in os.walk(self.cdir):
            for fn in files:
                p = os.path.join(d, fn)
                out[os.path.relpath(p, self.cdir)] = ('link', os.readlink(p)) if os.path.islink(p) else open(p, 'rb').read()
        return out

    def reference(self, listed=None, excluded=()):
        """The tree `builder` writes from every night now under the root, in a second root that shares the inputs; the
        snapshot is spelled for THIS root's tree (the builder writes the tree path into the sub_path column and the
        links)."""
        ref = os.path.join(self.dir, 'ref')
        shutil.rmtree(ref, ignore_errors=True)
        os.makedirs(ref)
        for sub in (LST, f'{LST}_x2'):
            os.symlink(os.path.join(self.root, sub), os.path.join(ref, sub))
        n = builder(ref, LST, SB, MEMBERS, self.listed if listed is None else listed, excluded)
        out = {}
        rc = os.path.join(ref, 'tenmin_root_x2', f'{LST}_{SB}')
        for d, _dirs, files in os.walk(rc):
            for fn in files:
                p = os.path.join(d, fn)
                if os.path.islink(p):
                    out[os.path.relpath(p, rc)] = ('link', os.readlink(p).replace(ref + os.sep, self.root + os.sep))
                else:
                    b = open(p, 'rb').read()
                    out[os.path.relpath(p, rc)] = b.replace(ref.encode() + b'/', self.root.encode() + b'/') \
                        if fn == 'frames.csv' else b
        return out, n

    def leases_held(self):
        return glob.glob(os.path.join(self.tree, TN.LEASE_SUBDIR, '*', 'holders', '*.json'))

    def args(self, night=NEW, **kw):
        a = dict(epochs=f'{LST}_{night}', cells=None, lsts=None, nights=None, scopes=SB, stokes=None,
                 stages='tree-nonmember', from_stage=None, until_stage=None, dry_run=False, workers=1, fresh=False,
                 supersede=None, resume=False, sublst_groups=None, sublst_mirror=False, sublst_loo_evidence=None)
        a.update(kw)
        return argparse.Namespace(**a)

    def run(self, night=NEW, dry=False, **kw):
        """Run the stage through the frames tier for one night; returns (the record.Run, exit code of the tier)."""
        r = record.Run('frames', ['test'], self.root, CFG, dry_run=dry)
        rc = FT.run(self.args(night, dry_run=dry, **kw), r)
        return r, rc


def rec_inputs(rec):
    return [(i['path'] if isinstance(i, dict) else i) for i in rec['inputs']]


def unit(run, night=NEW):
    name = TN.unit_name(LST, SB, night)
    hits = [u for u in run.units if u.name == name]
    assert len(hits) == 1, (name, [(u.name, u.status) for u in run.units])
    return hits[0]


def same(got, want):
    """Compare two tree snapshots; an AssertionError names the first difference."""
    assert sorted(got) == sorted(want), (sorted(set(got) ^ set(want))[:6], len(got), len(want))
    for k in sorted(want):
        if got[k] != want[k]:
            if k == 'frames.csv':
                g, w = got[k].split(b'\r\n'), want[k].split(b'\r\n')
                i = next((i for i, (x, y) in enumerate(zip(g, w)) if x != y), min(len(g), len(w)))
                raise AssertionError(f'frames.csv line {i}: {g[i:i + 1]} != {w[i:i + 1]} ({len(g)} vs {len(w)} lines)')
            raise AssertionError(f'{k} differs' + (f': {got[k]} != {want[k]}' if isinstance(want[k], tuple) else
                                                   f' ({fits.Header.fromstring(got[k][:5760]).tostring(sep=chr(10))[:1200]})'))


def refused(t, needle, night=NEW, **kw):
    """The unit refuses with `needle` in its reason, the tree is what it was and no lease is held."""
    r, _ = t.run(night, **kw)
    u = unit(r, night)
    assert u.status == 'refused' and needle in u.reason, (u.status, u.reason)
    same(t.snapshot(), t.before)
    assert not t.leases_held()
    return u


# =====================================================================================================================
# the entry
# =====================================================================================================================

def test_entry_equals_the_builders_tree():
    with Tree() as t:
        ino = {k: os.stat(os.path.join(t.cdir, 'bin_masters', f'PB_bin{k}_{SB}.fits')).st_ino for k in (0, 1, 2)}
        r, rc = t.run()
        u = unit(r)
        assert rc is None and u.status == 'done' and r.exit_code() == record.EXIT_DONE, (u.status, u.reason)
        want, (windows, frames, skipped) = t.reference()
        same(t.snapshot(), want)
        assert (windows, frames, skipped) == (11, 22, [])
        m = u.meta[TN.STAGE]
        assert m['action'] == 'present' and m['windows'] == [0, 1, 2] and m['links_made'] == 6 and m['frames_rows'] == 22
        assert m['frames_written'] and m['pb_headers_written'] == [0, 1, 2] and m['weight'] == 0.0
        # the other nights' lines are the bytes they were; the pixels of the window primary-beam maps did not change
        old = t.before['frames.csv'].split(b'\r\n')
        new = t.snapshot()['frames.csv'].split(b'\r\n')
        assert [x for x in new if NEW.encode() not in x] == old and len(new) == len(old) + 6
        for k in (0, 1, 2):
            p = os.path.join('bin_masters', f'PB_bin{k}_{SB}.fits')
            a, b = fits.open(os.path.join(t.cdir, p))[0], fits.HDUList.fromstring(t.before[p])[0]
            assert a.data.tobytes() == b.data.tobytes() and a.header['PBNNIGHT'] == b.header['PBNNIGHT'] + 1
            assert a.header['DATE-OBS'].startswith('2024-01-02')         # the earliest presented night's header, kept
            assert os.stat(os.path.join(t.cdir, p)).st_ino != ino[k]     # replaced by a rename, never written in place
        # the record: frames.csv and the three maps, content-tested as the batch stage's products
        rec = record.load_record(t.root, 'frames', u.name)
        assert rec['status'] == 'done' and {o['class'] for o in rec['outputs']} == {'tenmin_tree'} and len(rec['outputs']) == 4
        assert len(rec['tests']) == 4 and all(x['passed'] for x in rec['tests']), rec['tests']
        assert rec['admitted'] == [NEW] and t.decisions in rec_inputs(rec) and t.listfile in rec_inputs(rec)
        assert t.resid(NEW, 'V', 2) in rec_inputs(rec) and t.pb(NEW, 0) in rec_inputs(rec)
        # 6 residuals, 3 window primary-beam maps, the weights, the two lists, the refine log (and, where it exists, the
        # list of GPU-frame cells that sets the night's exclusion read)
        assert len(set(rec_inputs(rec)) - {CFG.paths.SNAPSHOT_GPU_FRAME_CELLS}) == 13, rec_inputs(rec)
        assert all(all(x['passed'] for x in validate.run('tenmin_tree', os.path.join(t.cdir, p), lst=LST, band=SB))
                   for p in ['frames.csv'] + [os.path.join('bin_masters', f'PB_bin{k}_{SB}.fits') for k in (0, 1, 2)])
        assert not t.leases_held() and os.path.isdir(os.path.join(t.tree, '.leases', f'{LST}_{SB}'))
        # a second run finds the night presented and writes nothing
        snap = t.snapshot()
        st = {p: os.stat(os.path.join(t.cdir, p)).st_ino for p in snap}
        r2, _ = t.run()
        u2 = unit(r2)
        assert u2.status == 'done' and u2.meta[TN.STAGE]['action'] == 'already presented', (u2.status, u2.reason)
        assert not u2.meta[TN.STAGE]['frames_written'] and u2.meta[TN.STAGE]['pb_headers_written'] == []
        same(t.snapshot(), snap)
        assert st == {p: os.stat(os.path.join(t.cdir, p)).st_ino for p in snap}


def test_the_earliest_night_gives_the_map_its_header():
    # the builder writes PB_bin<k> with the header of the earliest presented night's window primary-beam map: a night
    # earlier than every presented night of a window replaces the header (here a 4-axis, 64-bit map: the writer sets
    # BITPIX and NAXIS from the pixels); a window the night lacks is not touched
    with Tree(new=(EARLY,), new_windows=(0, 1), pb_axes=4) as t:
        p2 = os.path.join(t.cdir, 'bin_masters', f'PB_bin2_{SB}.fits')
        ino2 = os.stat(p2).st_ino
        r, _ = t.run(EARLY)
        u = unit(r, EARLY)
        assert u.status == 'done', (u.status, u.reason)
        want, (windows, frames, _skipped) = t.reference()
        same(t.snapshot(), want)
        assert (windows, frames) == (10, 20)
        for k in (0, 1):
            h = fits.getheader(os.path.join(t.cdir, 'bin_masters', f'PB_bin{k}_{SB}.fits'))
            assert h['DATE-OBS'].startswith(EARLY) and h['NAXIS'] == 2 and h['BITPIX'] == -32 and h['PBNNIGHT'] == 4
            assert h['PBSRC'] == TN.PBSRC_WEIGHTED[0]
        assert os.stat(p2).st_ino == ino2 and t.snapshot()['frames.csv'].startswith(b'key,date,bin')
        assert t.snapshot()['frames.csv'].split(b'\r\n')[1].startswith(EARLY.encode())      # its rows come first
        st = {p: os.stat(os.path.join(t.cdir, p)).st_ino for p in t.snapshot()}
        r2, _ = t.run(EARLY)
        assert unit(r2, EARLY).meta[TN.STAGE]['action'] == 'already presented'
        assert st == {p: os.stat(os.path.join(t.cdir, p)).st_ino for p in t.snapshot()}


def test_two_nights_one_after_the_other():
    with Tree(new=(EARLY, NEW)) as t:
        for n in (NEW, EARLY):
            r, _ = t.run(n)
            assert unit(r, n).status == 'done', unit(r, n).reason
        want, (windows, frames, _s) = t.reference()
        same(t.snapshot(), want)
        assert (windows, frames) == (14, 28)


def test_a_window_with_one_stokes_is_skipped():
    with Tree() as t:
        os.remove(t.resid(NEW, 'V', 1))
        r, _ = t.run()
        u = unit(r)
        assert u.status == 'done' and u.meta[TN.STAGE]['windows'] == [0, 2], (u.status, u.reason)
        assert u.meta[TN.STAGE]['skipped'] == [f'{NEW} w1 (only I)']
        want, (_w, _f, skipped) = t.reference()
        same(t.snapshot(), want)
        assert skipped == [f'{NEW} w1 (only I)']


# =====================================================================================================================
# refusals: nothing is written
# =====================================================================================================================

def test_refusals_write_nothing():
    with Tree() as t:
        # T-1: excluded in data_decisions.csv
        t.write_decisions([(LST, SB, NEW)])
        u = refused(t, 'T-1')
        assert 'excluded' in u.reason
        t.write_decisions(())
        # T-2: a member night (the batch stage's), and a night with no weight at all
        t.write_weights(dict(MEMBERS, **{NEW: 0.7}))
        u = refused(t, 'T-2')
        assert 'co-add weight 0.7' in u.reason and TN.BATCH_STAGE in u.reason
        t.write_weights(MEMBERS)
        t.listed = []
        t.write_list()
        refused(t, 'T-2: ' + f'{LST}_{NEW} {SB}: no co-add weight')
        t.listed = [NEW]
        t.write_list()
        # T-4: a residual the builder refuses
        with fits.open(t.resid(NEW, 'V', 2), mode='update') as hl:
            hl[0].header['BINMSTR'] = 'x'
        refused(t, 'T-4')
        with fits.open(t.resid(NEW, 'V', 2), mode='update') as hl:
            del hl[0].header['BINMSTR']
            del hl[0].header['DURATION']
        refused(t, 'T-4')
        with fits.open(t.resid(NEW, 'V', 2), mode='update') as hl:
            hl[0].header['DURATION'] = 600.0
        # T-5: no window primary-beam map of the night
        os.rename(t.pb(NEW, 1), t.pb(NEW, 1) + '.hidden')
        refused(t, 'T-5')
        os.rename(t.pb(NEW, 1) + '.hidden', t.pb(NEW, 1))
        # S-G8: the refine of the band is not complete (no final line; no log)
        log = t.write_log(final=False)
        refused(t, 'S-G8')
        os.remove(log)
        refused(t, 'S-G8')
        t.write_log()
        # T-3: one Stokes only in every window; no residual of the night at all (the unit refuses: never left out)
        hidden = os.path.dirname(t.resid(NEW, 'V', 0)) + '.hidden'
        os.rename(os.path.dirname(t.resid(NEW, 'V', 0)), hidden)
        refused(t, 'T-3')
        os.rename(hidden, os.path.dirname(t.resid(NEW, 'V', 0)))
        night_dir = os.path.join(t.root, f'{LST}_x2', f'{LST}_{NEW}')
        os.rename(night_dir, night_dir + '.hidden')
        u = refused(t, 'T-3:')
        assert 'refine-nonmember' in u.reason
        os.rename(night_dir + '.hidden', night_dir)
        # T-2: no frozen deep weights file of the band (the builder fails on it)
        wp = TS.frozen_weights(LST, SB)
        os.rename(wp, wp + '.hidden')
        assert 'no frozen deep weights file' in refused(t, 'T-2').reason
        os.rename(wp + '.hidden', wp)
        # and after all of them the unit is done
        r, _ = t.run()
        assert unit(r).status == 'done', unit(r).reason
        same(t.snapshot(), t.reference()[0])


def test_refusals_of_the_tree_it_adds_to():
    with Tree() as t:
        # T-6: no tree, an empty one, one recorded as having no window, other columns
        os.rename(t.cdir, t.cdir + '.hidden')
        r, _ = t.run()
        assert unit(r).status == 'refused' and 'T-6' in unit(r).reason and not os.path.exists(t.cdir)
        assert not os.path.exists(os.path.join(t.tree, TN.LEASE_SUBDIR))        # not even a lease directory
        os.rename(t.cdir + '.hidden', t.cdir)
        full = open(t.fcsv, 'rb').read()
        for text, mark in ((b','.join(c.encode() for c in COLS) + b'\r\n', 'no rows'),
                           (full.replace(b'key,date', b'key,night', 1), 'columns')):
            open(t.fcsv, 'wb').write(text)
            t.before = t.snapshot()
            assert mark in refused(t, 'T-6').reason
        open(t.fcsv, 'wb').write(full)
        open(os.path.join(t.cdir, 'NO_WINDOWS.txt'), 'w').write('no window\n')
        t.before = t.snapshot()
        refused(t, 'T-6')
        os.remove(os.path.join(t.cdir, 'NO_WINDOWS.txt'))
        t.before = t.snapshot()
        # T-7: a link of the night's name that points elsewhere
        link = os.path.join(t.cdir, 'subtracted', f'{NEW}T00_I_sub_{SB}.fits')
        os.symlink('/nowhere', link)
        t.before = t.snapshot()
        refused(t, 'T-7')
        os.remove(link)
        t.before = t.snapshot()
        # T-9: a night the tree presents was excluded after the tree was built (the builder would drop it)
        gone = sorted(MEMBERS)[1]
        t.write_decisions([(LST, SB, gone)])
        u = refused(t, 'T-9')
        assert gone in u.reason and TN.BATCH_STAGE in u.reason and 'excluded' in u.reason
        t.write_decisions(())
        # T-8: a window of the night without a map; a map that is not a weighted mean; another shape; a count that is
        # not the rows of frames.csv
        pb0 = os.path.join(t.cdir, 'bin_masters', f'PB_bin0_{SB}.fits')
        os.rename(pb0, pb0 + '.hidden')
        t.before = t.snapshot()
        refused(t, 'T-8')
        os.rename(pb0 + '.hidden', pb0)
        keep = open(pb0, 'rb').read()
        for change, mark in ((lambda h, d: (h.__setitem__('PBSRC', 'sub-LST window PB, unweighted mean (non-member nights only)'), d)[1],
                              'not a weighted mean'),
                             (lambda h, d: d[:, :-1], 'pixels'),
                             (lambda h, d: (h.__setitem__('PBNNIGHT', h['PBNNIGHT'] - 1), d)[1], 'PBNNIGHT'),
                             (lambda h, d: (h.__setitem__('PBNNIGHT', h['PBNNIGHT'] + 1), d)[1], 'PBNNIGHT')):
            with fits.open(pb0, memmap=False) as hl:
                h, d = hl[0].header.copy(), np.array(hl[0].data)
            d = change(h, d)
            fits.writeto(pb0, d, h, overwrite=True)
            t.before = t.snapshot()
            assert mark in refused(t, 'T-8').reason
            open(pb0, 'wb').write(keep)
        t.before = t.snapshot()
        r, _ = t.run()
        assert unit(r).status == 'done', unit(r).reason
        same(t.snapshot(), t.reference()[0])
        # T-7: the night is presented, then its residual changes a card the frame record holds
        with fits.open(t.resid(NEW, 'I', 0), mode='update') as hl:
            hl[0].header['FLUXDC'] = 0.5
        t.before = t.snapshot()
        u = refused(t, 'T-7')
        assert 'already with other rows' in u.reason and TN.BATCH_STAGE in u.reason


# =====================================================================================================================
# failures and the lease
# =====================================================================================================================

def test_an_interrupted_run_is_completed():
    with Tree() as t:
        # a failure before frames.csv is replaced: nothing was written (no link without its row), the tree is what it was
        real = os.replace

        def no_csv(src, dst):
            if os.path.basename(dst) == 'frames.csv':
                raise OSError('injected: frames.csv not replaced')
            return real(src, dst)
        with Patch((os, 'replace', no_csv)):
            r, _ = t.run()
        u = unit(r)
        assert u.status == 'error' and 'injected' in u.reason, (u.status, u.reason)
        same(t.snapshot(), t.before)
        assert not t.leases_held()
        # a failure after frames.csv, at the third link: the rows are in, two links, no map changed; no link without
        # its row at any time (the search lists its epochs from the links)
        made, real_link = [], os.symlink

        def third_fails(src, dst):
            if len(made) == 2:
                raise OSError('injected: link not made')
            made.append(dst)
            return real_link(src, dst)
        with Patch((os, 'symlink', third_fails)):
            r, _ = t.run()
        assert unit(r).status == 'error' and 'injected' in unit(r).reason and not t.leases_held()
        snap = t.snapshot()
        rows = [x for x in snap['frames.csv'].split(b'\r\n') if NEW.encode() in x]
        links = sorted(os.path.basename(x) for x in glob.glob(os.path.join(t.cdir, 'subtracted', f'{NEW}T*')))
        assert len(rows) == 6 and links == [f'{NEW}T00_I_sub_{SB}.fits', f'{NEW}T00_V_sub_{SB}.fits'], (len(rows), links)
        assert all(os.path.basename(x).encode() in snap['frames.csv'] for x in made)
        n = [fits.getheader(os.path.join(t.cdir, 'bin_masters', f'PB_bin{k}_{SB}.fits'))['PBNNIGHT'] for k in (0, 1, 2)]
        assert n == [3, 3, 2], n
        # a failure at the second map: the rows and the links are in, one map counts the night, two are one short
        calls = []

        def second_fails(path, header):
            calls.append(path)
            if len(calls) == 2:
                raise OSError('injected: map not written')
            return real_pb(path, header)
        real_pb = TN._replace_pb
        with Patch((TN, '_replace_pb', second_fails)):
            r, _ = t.run()
        assert unit(r).status == 'error' and not t.leases_held()
        n = [fits.getheader(os.path.join(t.cdir, 'bin_masters', f'PB_bin{k}_{SB}.fits'))['PBNNIGHT'] for k in (0, 1, 2)]
        assert n == [4, 3, 2] and NEW.encode() in t.snapshot()['frames.csv'], n
        assert not glob.glob(os.path.join(t.cdir, '.*tmp*')) and not glob.glob(os.path.join(t.cdir, 'bin_masters', '.*'))
        # the next run of the unit completes it
        r, _ = t.run()
        u = unit(r)
        assert u.status == 'done' and u.meta[TN.STAGE]['pb_headers_written'] == [1, 2], (u.status, u.reason, u.meta)
        assert not u.meta[TN.STAGE]['frames_written'] and u.meta[TN.STAGE]['links_made'] == 0
        assert len(glob.glob(os.path.join(t.cdir, 'subtracted', f'{NEW}T*'))) == 6
        same(t.snapshot(), t.reference()[0])


def test_resume_skips_a_presented_night_with_a_valid_record():
    with Tree() as t:
        # the record's own check (fingerprints, code, config, outputs) by a stand-in: this checkout may have local edits
        with Patch((record, 'resume_check', lambda *a, **k: (True, 'stand-in'))):
            r, _ = t.run(resume=True)                 # the night is not presented: never skipped, whatever the record says
            assert unit(r).status == 'done' and unit(r).meta[TN.STAGE]['action'] == 'present'
            snap = t.snapshot()
            r, _ = t.run(resume=True)
            assert unit(r).status == 'skipped-valid', (unit(r).status, unit(r).reason)
            r, rc = t.run(resume=True, dry=True)
            assert rc == 0 and not r.units
            same(t.snapshot(), snap)
            # the record of the run that wrote is still the unit's record
            rec = record.load_record(t.root, 'frames', TN.unit_name(LST, SB, NEW))
            assert rec['status'] == 'done' and rec['meta'][TN.STAGE]['action'] == 'present'
        with Patch((record, 'resume_check', lambda *a, **k: (False, 'stand-in: an input changed'))):
            r, _ = t.run(resume=True)                 # the record's check says no: the unit runs (and writes nothing)
            assert unit(r).status == 'done' and unit(r).meta[TN.STAGE]['action'] == 'already presented'
            same(t.snapshot(), snap)


def test_one_writer_per_tree():
    with Tree() as t:
        other = TN.tree_lease(t.tree, LST, SB, 'another unit')
        assert other.dir == os.path.join(t.tree, '.leases', f'{LST}_{SB}')
        other.acquire(say=lambda m: None)
        try:
            with Patch((TN, 'LEASE_WAIT_S', 1), (TN, 'LEASE_POLL_S', 0.2)):
                r, _ = t.run()
            u = unit(r)
            assert u.status == 'error' and 'lease' in u.reason.lower(), (u.status, u.reason)
            same(t.snapshot(), t.before)
        finally:
            other.release()
        r, _ = t.run()
        assert unit(r).status == 'done' and not t.leases_held()
        # the batch stage takes the same lease around the builder
        src = inspect.getsource(FT.stage_tree)
        assert 'tree_lease(tree, lst, b, name)' in src and src.index('lease.acquire') < src.index('engine(script') \
            < src.index('lease.release()')
        # the lease directory is outside every listing of '<LST>_*' cells
        assert not [x for x in glob.glob(os.path.join(t.tree, f'{LST}_*')) if '.leases' in x]


# =====================================================================================================================
# the plan, the wiring, the content tests' weights
# =====================================================================================================================

def test_dry_run_plans_and_writes_nothing():
    with Tree() as t:
        r, rc = t.run(dry=True)
        assert rc == 0 and not r.units, (rc, [(u.name, u.status) for u in r.units])
        same(t.snapshot(), t.before)
        assert not os.path.exists(os.path.join(t.tree, TN.LEASE_SUBDIR))
        assert not os.path.exists(os.path.join(t.root, '_records'))
        t.write_decisions([(LST, SB, NEW)])
        r, rc = t.run(dry=True)
        assert rc == 2
        same(t.snapshot(), t.before)
        t.write_decisions(())
        # --sublst-mirror does not apply: the plan and the run read the night's inputs under this root only (a root
        # of production that holds other residuals is never looked at)
        other = os.path.join(t.dir, 'production')
        os.makedirs(os.path.join(other, f'{LST}_x2', f'{LST}_{NEW}', SB, 'I'))
        with Patch((CFG.paths, 'SUBLST_ROOT', other)):
            night_dir = os.path.join(t.root, f'{LST}_x2', f'{LST}_{NEW}')
            os.rename(night_dir, night_dir + '.hidden')
            r, rc = t.run(dry=True, sublst_mirror=True)
            assert rc == 2
            r, _ = t.run(sublst_mirror=True)
            assert unit(r).status == 'refused' and 'T-3' in unit(r).reason
            os.rename(night_dir + '.hidden', night_dir)
            r, rc = t.run(dry=True, sublst_mirror=True)
            assert rc == 0
            r, _ = t.run(sublst_mirror=True)
            assert unit(r).status == 'done', unit(r).reason
            same(t.snapshot(), t.reference()[0])


def test_stage_wiring():
    S = CFG.frames_sublst
    order = list(S.ORDER)
    assert order.index('tree-nonmember') == order.index('tree') + 1 < order.index('verify')
    assert all(TN.STAGE not in v for v in S.STAGES.values())
    with Tree() as t:
        # without a named night the stage refuses (it never runs over every night of an LST)
        r = record.Run('frames', ['test'], t.root, CFG)
        FT.run(t.args(epochs=None, lsts=LST), r)
        assert [u.status for u in r.units] == ['refused'] and 'ONE named night' in r.units[0].reason
        same(t.snapshot(), t.before)
        # every selected band has a unit: a band without a residual of the night refuses (T-2 without its frozen
        # weights, T-3 with them), it is never left out; the other band is done
        r = record.Run('frames', ['test'], t.root, CFG)
        rc = FT.run(t.args(scopes=f'{SB},73MHz'), r)
        got = {u.name: (u.status, u.reason[:4]) for u in r.units}
        assert got == {TN.unit_name(LST, SB, NEW): ('done', got[TN.unit_name(LST, SB, NEW)][1]),
                       TN.unit_name(LST, '73MHz', NEW): ('refused', 'T-2:')}, got
        os.makedirs(os.path.dirname(TS.frozen_weights(LST, '73MHz')))
        shutil.copy(TS.frozen_weights(LST, SB), TS.frozen_weights(LST, '73MHz'))
        r = record.Run('frames', ['test'], t.root, CFG)
        FT.run(t.args(scopes='73MHz'), r)
        assert [(u.name, u.status, u.reason[:4]) for u in r.units] == [(TN.unit_name(LST, '73MHz', NEW), 'refused', 'T-3:')]
        assert not os.path.exists(os.path.join(t.tree, f'{LST}_73MHz')) and not t.leases_held()
        assert not os.path.exists(os.path.join(t.tree, TN.LEASE_SUBDIR, f'{LST}_73MHz'))     # refused before any lease


def test_content_tests_count_listed_nights_at_weight_zero():
    with Tree() as t:
        r, _ = t.run()
        assert unit(r).status == 'done'
        pb0 = os.path.join(t.cdir, 'bin_masters', f'PB_bin0_{SB}.fits')
        assert all(x['passed'] for x in validate.run('tenmin_tree', pb0, lst=LST, band=SB))
        # off the list, the night has no weight: the map's test names it (the rule before this change, for every night)
        t.listed = []
        t.write_list()
        res = validate.run('tenmin_tree', pb0, lst=LST, band=SB)
        assert not all(x['passed'] for x in res) and 'no weight' in str(res), res
        t.listed = [NEW]
        t.write_list()
        # a map of non-member nights only is their unweighted mean (tree:139)
        t.write_weights({})
        t.listed = list(MEMBERS) + [NEW]
        t.write_list()
        builder(t.root, LST, SB, {}, t.listed)
        assert fits.getheader(pb0)['PBSRC'].startswith('sub-LST window PB, unweighted mean')
        assert all(x['passed'] for x in validate.run('tenmin_tree', pb0, lst=LST, band=SB))
        # ... and this unit does not add to it (one more night would change its pixels)
        t.night(EARLY, (0,))
        t.listed.append(EARLY)
        t.write_list()
        t.write_log()
        t.before = t.snapshot()
        assert 'not a weighted mean' in refused(t, 'T-8', night=EARLY).reason
    # the residual-set model (content test (b)): a listed night without a frozen row is a weight-0 night of the window
    with Tree() as t:
        for night in list(MEMBERS) + [NEW]:
            for S in ('I', 'V'):
                for k in dict(MEMBER_WINDOWS, **{NEW: (0, 1, 2)})[night]:
                    stack = os.path.join(t.root, LST, f'{LST}_{night}', SB, S, f'sublst_{LST}_{night}_{SB}_{S}_w{k}.fits')
                    os.makedirs(os.path.dirname(stack), exist_ok=True)
                    fits.writeto(stack, np.zeros(SHAPE, np.float32), overwrite=True)
        stub = types.ModuleType('phase3_tier_rules')
        stub.membership_vetoed = lambda lst, sb, night: False
        old = sys.modules.get('phase3_tier_rules')
        sys.modules['phase3_tier_rules'] = stub
        try:
            # windows 0 and 1 have three weighted nights: every residual is allowed, the listed night's too. Window 2 has
            # two: the two members' residuals fall under the per-night rule (one weighted night left), the listed
            # night's do not (both weighted nights are in its Master) - 4 of 22 not allowed
            _fail, summary = XS.residual_set_report(os.path.join(t.root, LST), lst=LST, subbands=[SB])
            assert 'arm x2: 22 subband residuals; not allowed 4;' in summary, summary
            t.listed = []                 # off the list the night has no weight: its 6 residuals are not allowed either
            t.write_list()
            _fail, summary = XS.residual_set_report(os.path.join(t.root, LST), lst=LST, subbands=[SB])
            assert 'arm x2: 22 subband residuals; not allowed 10;' in summary, summary
        finally:
            if old is None:
                del sys.modules['phase3_tier_rules']
            else:
                sys.modules['phase3_tier_rules'] = old


def test_the_builders_literals():
    # where production's builder is on disk, its text holds the literals this unit writes and the columns of frames.csv
    assert TN.PBSRC_WEIGHTED == ('sub-LST window PB, night-weighted mean', 'phase3_sublst_tenmin_tree.py')
    try:
        script = FT.legacy_script('clv15', 'phase3_sublst_tenmin_tree.py')
    except Exception:  # noqa: BLE001
        script = None
    if not script or not os.path.exists(script):
        print('    (production\'s builder is not on this machine: its text not compared)')
        return
    text = open(script).read()
    assert f'hh["PBSRC"] = ("{TN.PBSRC_WEIGHTED[0]}", "{TN.PBSRC_WEIGHTED[1]}")' in text
    assert f'"{TN.PBNNIGHT_COMMENT}")' in text and 'hh["PBNNIGHT"] = (sum(1 for r in rows if r["bin"] == k and r["stokes"] == "I")' in text
    assert 'pbh.setdefault(k, hp[0].header.copy())' in text and 'for (night, k), st in sorted(found.items()):' in text
    assert 'for n in nonmember_nights(a.lst) - set(wts):' in text
    m = re.search(r'^COLS = \[(.*?)\]', text, re.S | re.M)
    assert m and re.findall(r'"(\w+)"', m.group(1)) == COLS
    row = re.search(r'rows\.append\(\{(.*?)\}\)', text, re.S)
    assert row and re.findall(r'"(\w+)":', row.group(1)) == COLS


TESTS = [test_entry_equals_the_builders_tree, test_the_earliest_night_gives_the_map_its_header,
         test_two_nights_one_after_the_other, test_a_window_with_one_stokes_is_skipped, test_refusals_write_nothing,
         test_refusals_of_the_tree_it_adds_to, test_an_interrupted_run_is_completed,
         test_resume_skips_a_presented_night_with_a_valid_record, test_one_writer_per_tree,
         test_dry_run_plans_and_writes_nothing, test_stage_wiring, test_content_tests_count_listed_nights_at_weight_zero,
         test_the_builders_literals]


def main(argv):
    if _IMPORT_ERROR is not None:
        print(f'UNTESTED-IMPORT: {type(_IMPORT_ERROR).__name__}: {_IMPORT_ERROR}')
        return 2
    want = set(argv)
    failed = 0
    for t in TESTS:
        if want and t.__name__ not in want:
            continue
        try:
            t()
            print(f'PASS {t.__name__}')
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback
            print(f'FAIL {t.__name__}: {type(e).__name__}: {e}')
            traceback.print_exc()
    print(f'{len([t for t in TESTS if not want or t.__name__ in want]) - failed} passed, {failed} failed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
