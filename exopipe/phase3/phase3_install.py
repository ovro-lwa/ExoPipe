#!/usr/bin/env python3
"""GATED INSTALL of a deep cell's new generation into the working tree and the archive (Gregg 2026-10-04, preamble "RBF
master-dewarp install and full deep re-search": "gated install + archive with quarantine").

A new generation is a directory GEN holding coadd_final/ (incl. loo/) and subtracted/:
  kind 'mdewarp' - the RBF master-dewarp generation (phase3_master_dewarp --apply: <MDEWARP_OUT_ROOT>/<cell>/gen_mdewarp_rbf,
                   file names carry the 'mdewarp' token, stripped here: the MDWARP cards carry the provenance and give the
                   archive PRODNAME its token);
  kind 'rebuild' - a rebuilt cell's generation in canonical names (the scaled rebuild promoted), installed with the cell files
                   the rebuild rewrote (CELL_FILES, CELL_DIRS from the rebuild root).
A rebuilt cell that is also master-dewarped is installed from its 'mdewarp' generation with the rebuild's cell files.

GATES (refuse, never warn; nothing is moved until every gate passes):
  G1 the generation's content test passed: GEN/../INSTALL_CONTENT_OK exists and names this GEN (written only by the content
     test of the generation's kind: check_mdewarp_gen for 'mdewarp'; for 'rebuild', promote_rebuild.py, which presents the
     files check_rebuild.py tested (T1-T7, REBUILD_CONTENT_OK) under canonical names as hard links - the same inodes);
  G2 every new file maps to one canonical name; every current FITS of coadd_final/** and subtracted/ without a new
     counterpart is listed (it is quarantined, not left behind: the working tree mirrors the generation, as the flux apply);
  G3 free space for the working-tree copy;
  G0 the exclusion-conformance gate on the cell as it stands (and again after the install).
INSTALL (each step verified before the next):
  a0 every new file is STAGED on the working filesystem (<cell>/.install_staging) and md5-verified against the generation
     before anything moves (a failure here leaves the cell untouched);
  a working tree: each current file (and, for a rebuild, each listed cell file / directory) is COPIED to the working
    quarantine WQ/<cell>/<rel> and md5-verified, then removed; each staged file is renamed to its canonical name (same
    filesystem); manifest rows in RECORD;
  b stamp + exclusion conformance: coadd_full_pipeline._stamp_all_masters and _refuse_excluded_constituents (the co-add's
    own completion steps);
  c archive: every top-level file of the cell's masters directory (Coadd/<run>/<lst>/<scope>/; its 10min/ subdirectory
    belongs to the 10-minute tier and is never touched) and every per-night deep file that is not a search frame ('_msub')
    is moved to QROOT (same filesystem: rename, inode and size verified, md5 manifest), then
    coadd_full_pipeline._archive_publish_cell writes the cell (epochs of gate-refused nights are not archived, every other
    epoch must pass the epoch content check);
  d audit: the cell's archive deep files (search frames excluded) are exactly the files the publish wrote, each md5 = its
    working source;
  e screen (kind 'mdewarp'): the RBF screen the generation was resampled with (check_mdewarp_gen D3 proves it exactly and
    records it in GEN/../APPLIED_SCREEN.json; gate G1b checks the record and the screen's md5 before anything moves) is
    published to <archive>/mdewarp_screens/<run>/<lst>_<scope>_warp_screen.fits, md5-verified, and its row written to
    mdewarp_screens.csv there (S-2 answer to the snapshot chat, CROSS_CHAT_COORDINATION.md 2026-10-04); a screen it
    replaces is quarantined first.
CLI: python -m exopipe.phase3.phase3_install <cell> --gen GEN --kind mdewarp|rebuild [--rebuild-root RR] [--root R]
     python -m exopipe.phase3.phase3_install --screen-table-unapplied TABLE   (after all installs: one applied=0 row per
     measured cell that was not applied, with its status, so a cell without a screen is known not to have been applied)"""
import argparse, csv, fcntl, glob, hashlib, json, os, re, shutil, sys, time

ROOT = '/data07/gh/phase3/coadd_sb'
WQ = '/lustre/gh/main/phase3/working_quarantine_20261004'
QROOT_NAME = 'Run_INSTALL_QUARANTINE_20261004'
RECORD = '/lustre/gh/main/exopipe_ops/mdewarp_rbf_20261004/install_record.csv'
CELL_FILES = ('weights.csv', 'night_image_gate.csv')                                 # in coadd_final/
CELL_TOP = ('residuals_provisional.csv', 'outliers.csv', 'residuals.csv')            # in the cell directory
CELL_DIRS = ('flow', 'coadd_provisional')                                            # in the cell directory
RCOLS = ['time', 'cell', 'step', 'path', 'dest', 'md5', 'bytes', 'detail']
SCREEN_COLS = ['lst', 'scope', 'applied', 'status', 'n_anchors', 'vlssr_med_before_arcmin', 'vlssr_med_after_arcmin',
               'heldout_resid_med_arcsec', 'anchor_off_med_arcsec', 'screen_p50_arcsec', 'screen_p95_arcsec', 'screen_file',
               'screen_md5', 'measured_at', 'written_utc']
# SNAPSHOT / SUB-LST FREEZE (2026-10-04): those tiers read the archive deep masters LIVE through 912 links in
# /lustre/pipeline/exopipe/phase3/snapshot_masters/; moving a master to quarantine breaks them (16 already broken by the
# 2026-09-27..10-01 archive replacements). No install until the other chat's frozen copy is in place and its checks passed,
# as relayed by Gregg: this file holds his words and the time (written by the deep session, CLAUDE.md 3).
FREEZE_MARKER = '/lustre/gh/main/exopipe_ops/mdewarp_rbf_20261004/SNAPSHOT_FREEZE_CONFIRMED'


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def rec(cell, step, path='', dest='', m='', nbytes='', detail=''):
    new = not os.path.exists(RECORD)
    with open(RECORD, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=RCOLS)
        if new:
            w.writeheader()
        w.writerow(dict(time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), cell=cell, step=step, path=path, dest=dest,
                        md5=m, bytes=nbytes, detail=detail))


def canon(name):
    return name.replace('_mdewarp', '', 1)


def _fits_under(base):
    out = []
    for sub in ('coadd_final', 'subtracted'):
        out += [os.path.relpath(p, base) for p in glob.glob(f'{base}/{sub}/**/*.fits', recursive=True)]
    return sorted(out)


def _move_to_wq(cell, src, rel):
    """Copy to the working quarantine, verify md5, then remove the source (cross-filesystem move, verified)."""
    dst = f'{WQ}/{cell}/{rel}'
    if os.path.exists(dst):
        raise RuntimeError(f'working quarantine destination exists: {dst}')
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    m0 = md5(src); shutil.copy2(src, dst + '.part')
    if md5(dst + '.part') != m0:
        os.remove(dst + '.part'); raise RuntimeError(f'working quarantine copy of {src} not verified (md5)')
    os.replace(dst + '.part', dst); n = os.path.getsize(src); os.remove(src)
    rec(cell, 'working-quarantine', src, dst, m0, n)


def _screen_dir(arch, run):
    return f'{arch}/mdewarp_screens/{run}'


def _screen_row(sdir, row):
    """Write (replace) one (lst, scope) row of mdewarp_screens.csv under an exclusive lock (installs run in parallel)."""
    os.makedirs(sdir, exist_ok=True); p = f'{sdir}/mdewarp_screens.csv'
    with open(f'{sdir}/.mdewarp_screens.lock', 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        rows = list(csv.DictReader(open(p))) if os.path.exists(p) else []
        rows = [r for r in rows if (r['lst'], r['scope']) != (row['lst'], row['scope'])] + [row]
        with open(p + '.part', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=SCREEN_COLS); w.writeheader()
            w.writerows(sorted(rows, key=lambda r: (r['lst'], r['scope'])))
        os.replace(p + '.part', p)


def _publish_screen(cell, arch, run, qroot, meta):
    """Step e: the applied screen beside the archive masters, md5-verified; a screen it replaces is quarantined first."""
    lst, scope = cell.split('_', 1); sdir = _screen_dir(arch, run); os.makedirs(sdir, exist_ok=True)
    dst = f'{sdir}/{lst}_{scope}_warp_screen.fits'
    if os.path.exists(dst):
        q = f'{qroot}/{os.path.relpath(dst, arch)}'
        if os.path.exists(q):
            raise RuntimeError(f'{cell}: screen quarantine destination exists: {q} - STOPPED')
        os.makedirs(os.path.dirname(q), exist_ok=True); m = md5(dst); os.rename(dst, q); rec(cell, 'archive-quarantine', dst, q, m)
    shutil.copy2(meta['screen'], dst + '.part')
    if md5(dst + '.part') != meta['screen_md5']:
        os.remove(dst + '.part'); raise RuntimeError(f'{cell}: published screen copy not verified (md5) - STOPPED')
    os.replace(dst + '.part', dst); rec(cell, 'screen-published', meta['screen'], dst, meta['screen_md5'], os.path.getsize(dst))
    _screen_row(sdir, dict(lst=lst, scope=scope, applied=1, status=meta['status'], n_anchors=meta['n_anchors'],
                           vlssr_med_before_arcmin=meta['med_before'], vlssr_med_after_arcmin=meta['med_after'],
                           heldout_resid_med_arcsec=meta['cv_resid_med_as'], anchor_off_med_arcsec=meta['anchor_off_med_as'],
                           screen_p50_arcsec=meta['warp_p50_as'], screen_p95_arcsec=meta['warp_p95_as'],
                           screen_file=os.path.basename(dst), screen_md5=meta['screen_md5'], measured_at=meta['measured_at'],
                           written_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))


def screen_table_unapplied(table):
    """After all installs: one applied=0 row for every measured (lst, scope) of TABLE (the merged measurement table, final row
    per cell) that has no applied row, with its status - so a missing screen reads as 'measured, not applied'."""
    from exopipe.phase3 import coadd_full_pipeline as C
    run = os.environ.get('P3_ARCHIVE_RUN')
    if not run:
        raise RuntimeError('P3_ARCHIVE_RUN unset - REFUSING')
    sdir = _screen_dir(C._archive_root(), run); p = f'{sdir}/mdewarp_screens.csv'
    have = {(r['lst'], r['scope']) for r in csv.DictReader(open(p)) if r['applied'] == '1'} if os.path.exists(p) else set()
    final = {}
    for r in csv.DictReader(open(table)):
        final[(r['lst'], r['scope'])] = r
    n = 0
    for (lst, scope), r in sorted(final.items()):
        if (lst, scope) in have:
            continue
        _screen_row(sdir, dict(lst=lst, scope=scope, applied=0, status=r['status'][:120], n_anchors=r.get('n_anchors', ''),
                               vlssr_med_before_arcmin=r.get('med_before', ''), vlssr_med_after_arcmin=r.get('med_after', ''),
                               heldout_resid_med_arcsec=r.get('cv_resid_med_as', ''), anchor_off_med_arcsec=r.get('anchor_off_med_as', ''),
                               screen_p50_arcsec='', screen_p95_arcsec='', screen_file='', screen_md5='',
                               measured_at=r.get('measured_at', ''), written_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
        n += 1
    return len(have), n


def install(cell, gen, kind, rebuild_root=None, root=ROOT):
    lst, scope = cell.split('_', 1); cdir = f'{root}/{cell}'
    os.environ.setdefault('P3_COADD_ROOT', root)
    from exopipe.phase3 import coadd_full_pipeline as C
    run = os.environ.get('P3_ARCHIVE_RUN')
    if not run:
        raise RuntimeError('P3_ARCHIVE_RUN unset - REFUSING')
    # G-1: the snapshot / sub-LST tiers no longer read the archive masters live
    if not os.path.exists(FREEZE_MARKER):
        raise RuntimeError(f'{cell}: {FREEZE_MARKER} absent - the snapshot / sub-LST freeze is not confirmed; REFUSING')
    # G1
    okf = f'{os.path.dirname(gen.rstrip("/"))}/INSTALL_CONTENT_OK'
    if not (os.path.exists(okf) and gen.rstrip('/') in open(okf).read()):
        raise RuntimeError(f'{cell}: no content-test record {okf} naming {gen} - REFUSING')
    if kind == 'rebuild' or rebuild_root:
        if not rebuild_root or not os.path.exists(f'{rebuild_root}/REBUILD_CONTENT_OK'):
            raise RuntimeError(f'{cell}: a rebuilt cell needs --rebuild-root with REBUILD_CONTENT_OK - REFUSING')
    # G1b (kind 'mdewarp'): the applied screen is on record for this generation and unchanged since the content test
    smeta = None
    if kind == 'mdewarp':
        sj = f'{os.path.dirname(gen.rstrip("/"))}/APPLIED_SCREEN.json'
        smeta = json.load(open(sj)) if os.path.exists(sj) else None
        if not smeta or smeta.get('gen') != gen.rstrip('/') or not os.path.exists(smeta['screen']) or md5(smeta['screen']) != smeta['screen_md5']:
            raise RuntimeError(f'{cell}: no applied-screen record {sj} naming {gen} with an unchanged screen - REFUSING')
    # G2
    new = _fits_under(gen); cur = _fits_under(cdir)
    tgt = {}
    for r in new:
        c_ = os.path.join(os.path.dirname(r), canon(os.path.basename(r)))
        if c_ in tgt.values():
            raise RuntimeError(f'{cell}: two generation files map to {c_} - REFUSING')
        tgt[r] = c_
    orphans = sorted(set(cur) - set(tgt.values()))
    # G3
    need = sum(os.path.getsize(f'{gen}/{r}') for r in new) * 1.05
    if shutil.disk_usage(cdir).free < need:
        raise RuntimeError(f'{cell}: {need / 1e9:.1f} GB needed on the working tree - REFUSING')
    # G0 (also run after the install, as the co-add does): exclusion conformance of the cell as it stands
    C._refuse_excluded_constituents(f'{cdir}/coadd_final', lst, scope)
    # a0: STAGE every new file on the working filesystem and verify it BEFORE anything is moved, so the replacement below is
    # a sequence of same-filesystem renames (a failure before it leaves the cell untouched)
    stg = f'{cdir}/.install_staging'
    if os.path.exists(stg):
        raise RuntimeError(f'{cell}: {stg} exists (an earlier install did not finish) - REFUSING')
    staged = {}
    for r, c_ in sorted(tgt.items()):
        s_ = f'{gen}/{r}'; d = f'{stg}/{c_}'; os.makedirs(os.path.dirname(d), exist_ok=True)
        m0 = md5(s_); shutil.copy2(s_, d)
        if md5(d) != m0:
            raise RuntimeError(f'{cell}: staging copy of {r} not verified (md5) - REFUSING (cell untouched)')
        staged[c_] = (d, m0, s_)
    rec(cell, 'gates-passed', gen, '', '', '', f'kind {kind}; {len(new)} new (staged, md5-verified), {len(cur)} current, '
                                               f'{len(orphans)} without a new counterpart')
    # a: working tree
    for r in sorted(set(cur)):
        _move_to_wq(cell, f'{cdir}/{r}', r)
    if rebuild_root:
        for f in CELL_FILES:
            if os.path.exists(f'{cdir}/coadd_final/{f}'):
                _move_to_wq(cell, f'{cdir}/coadd_final/{f}', f'coadd_final/{f}')
            shutil.copy2(f'{rebuild_root}/coadd_final/{f}', f'{cdir}/coadd_final/{f}')
            rec(cell, 'cell-file', f'{rebuild_root}/coadd_final/{f}', f'{cdir}/coadd_final/{f}', md5(f'{cdir}/coadd_final/{f}'))
        for f in CELL_TOP:
            if os.path.exists(f'{cdir}/{f}'):
                _move_to_wq(cell, f'{cdir}/{f}', f)
            if os.path.exists(f'{rebuild_root}/{f}'):
                shutil.copy2(f'{rebuild_root}/{f}', f'{cdir}/{f}'); rec(cell, 'cell-file', f'{rebuild_root}/{f}', f'{cdir}/{f}', md5(f'{cdir}/{f}'))
        for d in CELL_DIRS:
            if os.path.isdir(f'{cdir}/{d}'):
                for p in sorted(glob.glob(f'{cdir}/{d}/**/*', recursive=True) + glob.glob(f'{cdir}/{d}/.*')):
                    if os.path.isfile(p):
                        _move_to_wq(cell, p, os.path.relpath(p, cdir))
                shutil.rmtree(f'{cdir}/{d}')
            if os.path.isdir(f'{rebuild_root}/{d}'):
                shutil.copytree(f'{rebuild_root}/{d}', f'{cdir}/{d}'); rec(cell, 'cell-dir', f'{rebuild_root}/{d}', f'{cdir}/{d}')
    for c_, (d0, m0, s_) in sorted(staged.items()):
        d = f'{cdir}/{c_}'; os.makedirs(os.path.dirname(d), exist_ok=True)
        os.replace(d0, d); rec(cell, 'installed', s_, d, m0, os.path.getsize(d))
    shutil.rmtree(stg)
    if sorted(_fits_under(cdir)) != sorted(tgt.values()):
        raise RuntimeError(f'{cell}: working tree after install != the generation file set - STOPPED')
    # b: stamp + exclusion conformance
    fd, sd = f'{cdir}/coadd_final', f'{cdir}/subtracted'
    C._stamp_all_masters(fd, lst, scope)
    C._refuse_excluded_constituents(fd, lst, scope)
    # c: archive quarantine, then publish
    arch = C._archive_root(); qroot = f'{arch}/{QROOT_NAME}'
    mdir = f'{arch}/Coadd/{run}/{lst}/{scope}'
    old = [p for p in sorted(glob.glob(f'{mdir}/*')) if os.path.isfile(p)]
    old += [p for p in sorted(glob.glob(f'{arch}/{lst}/*/{run}/{scope}/[IV]/deep/*.fits')) if '_msub' not in os.path.basename(p)]
    for p in old:
        q = f'{qroot}/{os.path.relpath(p, arch)}'
        if os.path.exists(q):
            raise RuntimeError(f'{cell}: archive quarantine destination exists: {q} - STOPPED')
        os.makedirs(os.path.dirname(q), exist_ok=True); m = md5(p)
        s0 = os.stat(p); os.rename(p, q); s1 = os.stat(q)
        if (s1.st_ino, s1.st_size) != (s0.st_ino, s0.st_size):
            os.rename(q, p); raise RuntimeError(f'{cell}: archive quarantine move of {p} not verified; moved back - STOPPED')
        rec(cell, 'archive-quarantine', p, q, m, s0.st_size)
    t0 = time.time()
    n = C._archive_publish_cell(fd, sd, lst, scope)
    # d: audit
    now = [p for p in sorted(glob.glob(f'{mdir}/*')) if os.path.isfile(p)]
    now += [p for p in sorted(glob.glob(f'{arch}/{lst}/*/{run}/{scope}/[IV]/deep/*.fits')) if '_msub' not in os.path.basename(p)]
    if len(now) != n:   # every old file was quarantined first, so what is present is what the publish wrote
        raise RuntimeError(f'{cell}: archive audit - {len(now)} files present, {n} published - FAILED')
    by_name = {}
    for p in glob.glob(f'{fd}/*.fits') + glob.glob(f'{fd}/loo/*.fits') + glob.glob(f'{sd}/*_epoch_*.fits'):
        nm = None
        try:
            from astropy.io import fits as _f
            nm = _f.getheader(p).get('PRODNAME')
        except Exception:
            pass
        if nm:
            by_name[nm if nm.endswith('.fits') else nm + '.fits'] = p
    for f in ('PB_deep_%s.fits' % scope, 'PB_deep_R075_%s.fits' % scope, 'weights.csv', 'night_image_gate.csv'):
        by_name[f] = f'{fd}/{f}'
    bad = [p for p in now if os.path.basename(p) not in by_name or md5(p) != md5(by_name[os.path.basename(p)])]
    if bad:
        raise RuntimeError(f'{cell}: archive audit - {len(bad)} files whose md5 differs from the working source, e.g. {bad[:2]} - FAILED')
    # e: the applied screen beside the archive masters
    if smeta:
        _publish_screen(cell, arch, run, qroot, smeta)
    rec(cell, 'installed-and-audited', gen, '', '', '', f'{len(tgt)} working files, {len(old)} archive files quarantined, {n} published'
                                                        + ('; screen published' if smeta else ''))
    return len(tgt), len(old), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cell', nargs='?'); ap.add_argument('--gen'); ap.add_argument('--kind', choices=('mdewarp', 'rebuild'))
    ap.add_argument('--rebuild-root', default=None); ap.add_argument('--root', default=ROOT)
    ap.add_argument('--screen-table-unapplied', default=None, metavar='TABLE')
    a = ap.parse_args()
    if a.screen_table_unapplied:
        n1, n0 = screen_table_unapplied(a.screen_table_unapplied)
        print(f'SCREEN-TABLE {n1} applied rows kept, {n0} applied=0 rows written', flush=True); return
    if not (a.cell and a.gen and a.kind):
        ap.error('cell, --gen and --kind are required for an install')
    try:
        r = install(a.cell, a.gen.rstrip('/'), a.kind, a.rebuild_root, a.root)
        print(f'INSTALL-OK {a.cell}: {r[0]} working files, {r[1]} archive files quarantined, {r[2]} published', flush=True)
    except Exception as e:
        rec(a.cell, 'INSTALL-FAILED', a.gen, '', '', '', str(e)[:400])
        print(f'INSTALL-FAILED {a.cell}: {e}', flush=True); sys.exit(1)


if __name__ == '__main__':
    main()
