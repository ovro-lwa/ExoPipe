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
  G0 the exclusion-conformance gate on the STAGED new generation (and again on the installed tree).
INSTALL (each step verified before the next):
  a0 every new file is STAGED on the working filesystem (<cell>/.install_staging) and md5-verified against the generation
     before anything moves (a failure here leaves the cell untouched);
  a1 the per-night EPOCH IMAGE of every night with a residual is staged too: the generation's own, or (a dewarped cell that
     was not rebuilt: its working tree never held them) derived as residual + master; G4: every epoch the archive step will
     publish passes coadd_full_pipeline._epoch_content_check in staging, or the install is refused with the cell untouched;
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
     python -m exopipe.phase3.phase3_install <cell> --epochs-only   (repair: an installed cell's missing archive epochs)
     python -m exopipe.phase3.phase3_install --screen-table-unapplied TABLE   (after all installs: one applied=0 row per
     measured cell that was not applied, with its status, so a cell without a screen is known not to have been applied)"""
import argparse, csv, fcntl, glob, hashlib, json, os, re, shutil, sys, time

ROOT = '/data07/gh/phase3/coadd_sb'
# ONE QUARANTINE PAIR PER INSTALL ROUND (2026-10-05): a second install of an installed cell (the cubic re-apply) would find
# the first round's destinations and stop; the round names both (working tree and archive), default the first round
INSTALL_ROUND = os.environ.get('P3_INSTALL_ROUND', '20261004')
WQ = f'/lustre/gh/main/phase3/working_quarantine_{INSTALL_ROUND}'
QROOT_NAME = f'Run_INSTALL_QUARANTINE_{INSTALL_ROUND}'
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


def _derive_epoch(sub_path, master_path, out_path):
    """The night's epoch image = its master-subtracted residual + the cell master (the co-add's i_sub = i_corrected - master),
    float32, with the residual's own header (its time block, beam, flux and dewarp cards) minus PRODNAME, so the archive step
    names it as an epoch image (build_name from DATE-OBS) and stamps the standard header."""
    import numpy as np
    from astropy.io import fits
    with fits.open(sub_path) as h:
        r = h[0].data.astype(np.float32); hh = h[0].header.copy()
    m = np.squeeze(fits.getdata(master_path)).astype(np.float32)
    if np.squeeze(r).shape != m.shape:
        raise RuntimeError(f'{sub_path}: shape {np.squeeze(r).shape} != master {m.shape}')
    for k in ('PRODNAME', 'CHECKSUM', 'DATASUM'):
        hh.remove(k, ignore_missing=True, remove_all=True)
    hh['HISTORY'] = (f'epoch image = {os.path.basename(sub_path)} + {os.path.basename(master_path)} (derived at install, '
                     f'2026-10-04; the co-add residual is epoch - master)')
    fits.writeto(out_path, (np.squeeze(r) + m).reshape(r.shape).astype(np.float32), hh, overwrite=False)


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


def _gates(cell, gen, kind, rebuild_root=None):
    """G-1, G-1b, G1 (+ the cubic ruling), the rebuild record and G1b - nothing has moved yet. Shared by install() and
    resume() (2026-10-05). Returns the applied-screen record (kind 'mdewarp') or None."""
    from exopipe.phase3 import coadd_full_pipeline as C
    # G-1: the snapshot / sub-LST tiers no longer read the archive masters live
    if not os.path.exists(FREEZE_MARKER):
        raise RuntimeError(f'{cell}: {FREEZE_MARKER} absent - the snapshot / sub-LST freeze is not confirmed; REFUSING')
    # G-1b (2026-10-05): the freeze is VERIFIED, not only recorded - no snapshot_masters link may resolve into the deep archive
    # masters this install moves (the frozen copy is the snapshot_masters_gen4 tree)
    _live = [l for l in glob.glob('/lustre/pipeline/exopipe/phase3/snapshot_masters/**/*.fits', recursive=True)
             if os.path.realpath(l).startswith(f'{C._archive_root()}/Coadd/')]
    if _live:
        raise RuntimeError(f'{cell}: {len(_live)} snapshot_masters links resolve into the deep archive masters (e.g. {_live[0]}) - '
                           f'an install would move what the snapshot tier reads; REFUSING')
    # G1
    okf = f'{os.path.dirname(gen.rstrip("/"))}/INSTALL_CONTENT_OK'
    if not (os.path.exists(okf) and gen.rstrip('/') in open(okf).read()):
        raise RuntimeError(f'{cell}: no content-test record {okf} naming {gen} - REFUSING')
    if kind == 'mdewarp':
        # THE CUBIC RE-APPLY RULING (Gregg 2026-10-05, "Okay approved"): a dewarp generation resampled any other way is REFUSED
        # (the 2026-10-04 bilinear generations lowered compact-source peaks 1-5 per cent): its content record (D6, the peak test)
        # names the method and every product carries it
        from astropy.io import fits as _f
        from exopipe.phase3 import phase3_master_dewarp as _MD
        if f'files resampled {_MD.RESAMPLE_METHOD}' not in open(okf).read():
            raise RuntimeError(f'{cell}: {okf} is not a {_MD.RESAMPLE_METHOD} content record (D1-D6) - REFUSING')
        _bm = [r for r in _fits_under(gen) if _f.getheader(f'{gen}/{r}').get('MDWINTP') != _MD.RESAMPLE_METHOD]
        if _bm:
            raise RuntimeError(f'{cell}: {len(_bm)} generation products lack MDWINTP={_MD.RESAMPLE_METHOD}, e.g. {_bm[0]} - REFUSING')
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
    return smeta


def install(cell, gen, kind, rebuild_root=None, root=ROOT, omit_vclean=False):
    lst, scope = cell.split('_', 1); cdir = f'{root}/{cell}'
    os.environ.setdefault('P3_COADD_ROOT', root)
    from exopipe.phase3 import coadd_full_pipeline as C
    run = os.environ.get('P3_ARCHIVE_RUN')
    if not run:
        raise RuntimeError('P3_ARCHIVE_RUN unset - REFUSING')
    smeta = _gates(cell, gen, kind, rebuild_root)
    # G2
    new = _fits_under(gen); cur = _fits_under(cdir)
    # --omit-vclean (2026-10-04; Gregg deferred the cleaned-V repair, FINDING-VCLEAN-MASTERS-LIST-EXCLUDED-NIGHTS-63-CELLS, while the
    # final deep transient list is the priority): the generation's cleaned-V products (resampled from stale cleaned-V masters
    # that list since-excluded nights) are NOT installed; the current ones have no new counterpart, so they are quarantined
    # (working tree and archive) - the cell has no cleaned-V product until the deferred repair rebuilds it from its V master
    if omit_vclean:
        om = [r for r in new if 'V_deep_Taper_R0_clean' in os.path.basename(r)]
        new = [r for r in new if r not in om]
        rec(cell, 'omit-vclean', gen, '', '', '', f'{len(om)} cleaned-V products of the generation not installed: {om}')
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
    # G0 is applied to the STAGED new generation below (2026-10-04: checking the cell as it stands refused rebuilt cells whose
    # CURRENT cleaned-V masters list since-excluded nights - the breach the install replaces - while their new generation is clean)
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
    # a1 EPOCH IMAGES + the EPOCH GATE (Gregg 2026-10-04 option B: "Dont force products for crappy nights. We want to find
    # those nights and exclude them via approved QA."). The new generation's every night with a residual is tested by
    # epoch_gate (the archive's own epoch content test; Stokes I decides for the night): a FAIL night gets no epoch product
    # and is refused from both searches through the record written below; a FAIL night INSIDE the master refuses the install
    # (the master must be rebuilt without it). Every other night of the generation record that the night-image gate did not
    # refuse must have its epoch image: the generation's own (rebuilt) or derived here as residual + master (a dewarped cell
    # that was not rebuilt; its working tree never held epochs - found on the first install, 01h 18MHz), and each passes
    # _epoch_content_check before anything moves.
    wfin = f'{rebuild_root}/coadd_final' if rebuild_root else f'{cdir}/coadd_final'
    wrows = {r['date']: r for r in csv.DictReader(open(f'{wfin}/weights.csv'))}
    from exopipe.phase3 import night_image_gate as NG, epoch_gate as EG
    gfail = NG.failed_dates(wfin)
    erows = EG.evaluate_cell(f'{stg}/coadd_final', f'{stg}/subtracted', scope)
    efail = {d for d, r in erows.items() if r['verdict'] == 'FAIL'}
    inm = sorted(d for d in efail if wrows.get(d, {}).get('in_master') == '1' and d not in gfail)
    if inm:
        shutil.rmtree(stg)
        raise RuntimeError(f'{cell}: night(s) {inm} IN THE MASTER fail the epoch gate - the master must be rebuilt without them; '
                           f'REFUSING (cell untouched)')
    nderiv = nchk = 0
    for p in sorted(glob.glob(f'{stg}/subtracted/*_sub_{scope}.fits')):
        e_ = C._EPOCH_RE.match(os.path.basename(p))
        if not e_ or e_['kind'] != 'sub':
            continue
        d_ = e_['date']
        if d_ not in wrows or d_ in gfail or d_ in efail:   # refused or outside the generation record: no epoch product
            continue
        ep = f"{stg}/subtracted/{d_}_{e_['st']}_epoch_{scope}.fits"
        mp = f"{stg}/coadd_final/" + ('I_deep_Taper_R075' if e_['st'] == 'I' else 'V_deep_Taper_R0') + f'_{scope}.fits'
        if not os.path.exists(ep):
            _derive_epoch(p, mp, ep); rel = os.path.relpath(ep, stg)
            staged[rel] = (ep, md5(ep), f'derived at install: {os.path.basename(p)} + {os.path.basename(mp)}'); tgt[f'derived:{rel}'] = rel
            nderiv += 1
        ok_, why = C._epoch_content_check(ep, mp, p, e_['st']); nchk += 1
        if not ok_:
            shutil.rmtree(stg)
            raise RuntimeError(f"{cell}: epoch {os.path.basename(ep)} fails the epoch content check ({why}) - REFUSING (cell untouched)")
    # G0: exclusion conformance of the generation being installed (every product's constituents; also run after the install)
    try:
        C._refuse_excluded_constituents(f'{stg}/coadd_final', lst, scope)
    except Exception:
        shutil.rmtree(stg); raise
    rec(cell, 'gates-passed', gen, '', '', '', f'kind {kind}; {len(new)} new (staged, md5-verified), {len(cur)} current, '
                                               f'{len(orphans)} without a new counterpart; {nderiv} epochs derived, {nchk} '
                                               f'publishable epochs pass the epoch content check; epoch gate FAIL nights {sorted(efail)}')
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
    return _finish(cell, gen, cdir, stg, staged, tgt, erows, efail, lst, scope, run, smeta)


def _finish(cell, gen, cdir, stg, staged, tgt, erows, efail, lst, scope, run, smeta, old=None):
    """Everything after the working-quarantine moves: install step a's renames, the set check, the epoch-gate record, b,
    c, d, e. Shared by install() and resume() (2026-10-05). old (resume state R2) = the archive files this attempt already
    quarantined: renames, record, b and the archive quarantine are done; the publish, audit and screen run."""
    from exopipe.phase3 import coadd_full_pipeline as C, epoch_gate as EG
    fd, sd = f'{cdir}/coadd_final', f'{cdir}/subtracted'
    arch = C._archive_root(); qroot = f'{arch}/{QROOT_NAME}'
    mdir = f'{arch}/Coadd/{run}/{lst}/{scope}'
    if old is None:
        for c_, (d0, m0, s_) in sorted(staged.items()):
            d = f'{cdir}/{c_}'; os.makedirs(os.path.dirname(d), exist_ok=True)
            os.replace(d0, d); rec(cell, 'installed', s_, d, m0, os.path.getsize(d))
        shutil.rmtree(stg)
        if sorted(_fits_under(cdir)) != sorted(tgt.values()):
            raise RuntimeError(f'{cell}: working tree after install != the generation file set - STOPPED')
        # the epoch-gate record of the installed generation (made against the staged master = the installed one)
        if os.path.exists(f'{fd}/{EG.GATE_FILE}'):
            _move_to_wq(cell, f'{fd}/{EG.GATE_FILE}', f'coadd_final/{EG.GATE_FILE}')
        EG.write_record(fd, erows); rec(cell, 'epoch-gate-record', f'{fd}/{EG.GATE_FILE}', '', md5(f'{fd}/{EG.GATE_FILE}'), '',
                                        f'{len(erows)} nights, FAIL {sorted(efail)}')
        # b: stamp + exclusion conformance
        C._stamp_all_masters(fd, lst, scope)
        C._refuse_excluded_constituents(fd, lst, scope)
        # c: archive quarantine, then publish
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
    n = C._archive_publish_cell(fd, sd, lst, scope)
    # d: audit
    _archive_audit(cell, fd, sd, mdir, arch, run, lst, scope, n)
    # e: the applied screen beside the archive masters
    if smeta:
        _publish_screen(cell, arch, run, qroot, smeta)
    rec(cell, 'installed-and-audited', gen, '', '', '', f'{len(tgt)} working files, {len(old)} archive files quarantined, {n} published'
                                                        + ('; screen published' if smeta else ''))
    return len(tgt), len(old), n


def resume(cell, gen, kind, root=ROOT, omit_vclean=False, execute=False):
    """RESUME AN INSTALL INTERRUPTED MID-WAY (2026-10-05, INCIDENT-DATA07-FULL-HALF-INSTALLED-CELLS-ROUND-20261005: /data07 ran out
    of space during round 20261005). The gates of install() run again; then the state of the cell's latest attempt with this
    generation (its record rows from 'gates-passed' on, and the files on disk) must be one of two, else REFUSED:
      R1 renames not started: the attempt moved the cell's working files to the working quarantine and installed none; the
         working tree holds no coadd_final/subtracted FITS; .install_staging holds every generation file under its canonical
         name (md5 = its generation source) plus epochs derived at install (each passes the epoch content check again); the
         epoch gate and G0 are evaluated again on the staging -> _finish: renames, record, b, c, d, e.
      R2 archive publish interrupted: the attempt has 'installed' and 'archive-quarantine' rows, no 'installed-and-audited', no
         staging; the working tree is exactly the installed set and every epoch reads with the master's shape -> _finish from
         the publish (atomic rewrites of what it wrote before), audit, screen.
    Not for rebuilt cells (their cell files move before the renames). DRY RUN unless execute=True: prints the state and the
    checks, moves nothing."""
    import collections
    import numpy as _np
    from astropy.io import fits as _f
    lst, scope = cell.split('_', 1); cdir = f'{root}/{cell}'; stg = f'{cdir}/.install_staging'; gen = gen.rstrip('/')
    os.environ.setdefault('P3_COADD_ROOT', root)
    from exopipe.phase3 import coadd_full_pipeline as C, epoch_gate as EG
    run = os.environ.get('P3_ARCHIVE_RUN')
    if not run:
        raise RuntimeError('P3_ARCHIVE_RUN unset - REFUSING')
    smeta = _gates(cell, gen, kind)
    allr = [r for r in csv.DictReader(open(RECORD)) if r['cell'] == cell]
    gi = max((i for i, r in enumerate(allr) if r['step'] == 'gates-passed' and r['path'].rstrip('/') == gen), default=None)
    if gi is None:
        raise RuntimeError(f'{cell}: no gates-passed record for {gen} - nothing to resume; REFUSING')
    att = allr[gi:]; steps = collections.Counter(r['step'] for r in att)
    if steps['installed-and-audited'] or steps['cell-file'] or steps['cell-dir']:
        raise RuntimeError(f'{cell}: the attempt finished, or is a rebuilt cell ({dict(steps)}) - REFUSING')
    cur = _fits_under(cdir)
    new = _fits_under(gen)
    if omit_vclean:
        new = [r for r in new if 'V_deep_Taper_R0_clean' not in os.path.basename(r)]
    tgt = {r: os.path.join(os.path.dirname(r), canon(os.path.basename(r))) for r in new}
    if steps['working-quarantine'] and not steps['installed'] and not steps['archive-quarantine'] and not cur and os.path.isdir(stg):
        have = set(_fits_under(stg))
        miss = sorted(set(tgt.values()) - have)
        if miss:
            raise RuntimeError(f'{cell}: R1 staging lacks {len(miss)} generation files, e.g. {miss[:2]} - REFUSING')
        staged = {}
        for r, c_ in sorted(tgt.items()):
            m0 = md5(f'{gen}/{r}')
            if md5(f'{stg}/{c_}') != m0:
                raise RuntimeError(f'{cell}: staged {c_} differs from its generation source (md5) - REFUSING')
            staged[c_] = (f'{stg}/{c_}', m0, f'{gen}/{r}')
        extra = sorted(have - set(tgt.values()))
        odd = [x for x in extra if not (C._EPOCH_RE.match(os.path.basename(x)) and C._EPOCH_RE.match(os.path.basename(x))['kind'] == 'epoch')]
        if odd:
            raise RuntimeError(f'{cell}: R1 staging holds {len(odd)} files that are neither generation files nor derived epochs, e.g. {odd[:2]} - REFUSING')
        erows = EG.evaluate_cell(f'{stg}/coadd_final', f'{stg}/subtracted', scope)
        efail = {d for d, r_ in erows.items() if r_['verdict'] == 'FAIL'}
        for x in extra:
            e_ = C._EPOCH_RE.match(os.path.basename(x))
            mp = f"{stg}/coadd_final/" + ('I_deep_Taper_R075' if e_['st'] == 'I' else 'V_deep_Taper_R0') + f'_{scope}.fits'
            ok_, why = C._epoch_content_check(f'{stg}/{x}', mp, f'{stg}/{x}'.replace('_epoch_', '_sub_'), e_['st'])
            if not ok_:
                raise RuntimeError(f'{cell}: staged epoch {x} fails the epoch content check ({why}) - REFUSING')
            staged[x] = (f'{stg}/{x}', md5(f'{stg}/{x}'), 'derived at install (resumed 2026-10-05)'); tgt[f'derived:{x}'] = x
        C._refuse_excluded_constituents(f'{stg}/coadd_final', lst, scope)
        print(f'{cell} R1 (renames not started): {len(staged)} staged files verified, {len(extra)} of them derived epochs (content check '
              f'passed); working tree empty; {steps["working-quarantine"]} files in the working quarantine this attempt; epoch gate FAIL '
              f'nights {sorted(efail)}', flush=True)
        if not execute:
            print(f'{cell} DRY RUN - would rename {len(staged)} files into the working tree, then the epoch-gate record, stamp, '
                  f'archive quarantine, publish, audit, screen', flush=True)
            return None
        return _finish(cell, gen, cdir, stg, staged, tgt, erows, efail, lst, scope, run, smeta)
    if steps['installed'] and steps['archive-quarantine'] and not os.path.exists(stg):
        inst = {os.path.relpath(r['dest'], cdir) for r in att if r['step'] == 'installed'}
        if set(cur) != inst:
            raise RuntimeError(f'{cell}: R2 working tree differs from the installed set ({len(set(cur) ^ inst)} files) - REFUSING')
        shp = _np.squeeze(_f.getdata(f'{cdir}/coadd_final/I_deep_Taper_R075_{scope}.fits')).shape; bad = []
        for x in sorted(cur):
            if '_epoch_' in os.path.basename(x):
                try:
                    if _np.squeeze(_f.getdata(f'{cdir}/{x}')).shape != shp:
                        bad.append(x)
                except Exception:
                    bad.append(x)
        if bad:
            raise RuntimeError(f'{cell}: R2 epoch file(s) unreadable or of another shape {bad[:3]} - REFUSING')
        old = [r['path'] for r in att if r['step'] == 'archive-quarantine']
        print(f'{cell} R2 (publish interrupted): working tree = the {len(inst)} installed files, every epoch readable; {len(old)} archive '
              f'files quarantined this attempt', flush=True)
        if not execute:
            print(f'{cell} DRY RUN - would publish, audit and publish the screen', flush=True)
            return None
        return _finish(cell, gen, cdir, stg, {}, {x: x for x in sorted(inst)}, None, None, lst, scope, run, smeta, old=old)
    raise RuntimeError(f'{cell}: not a resumable state (attempt steps {dict(steps)}, {len(cur)} working FITS, staging '
                       f'{"present" if os.path.isdir(stg) else "absent"}) - REFUSING')


def _archive_audit(cell, fd, sd, mdir, arch, run, lst, scope, n):
    """Step d. (1) the cell's archive deep files (search frames excluded) are exactly the files the publish wrote (every old one
    was quarantined first), each with the md5 of its working source; (2) EPOCH COMPLETENESS (2026-10-04, after the first install
    published no epochs and (1) could not see it): every night of the generation's weights.csv that the night-image gate did not
    refuse and that has a residual has its epoch image in the archive, for each Stokes with a residual."""
    from astropy.io import fits as _f
    from exopipe.phase3 import coadd_full_pipeline as C, night_image_gate as NG
    now = [p for p in sorted(glob.glob(f'{mdir}/*')) if os.path.isfile(p)]
    now += [p for p in sorted(glob.glob(f'{arch}/{lst}/*/{run}/{scope}/[IV]/deep/*.fits')) if '_msub' not in os.path.basename(p)]
    if len(now) != n:
        raise RuntimeError(f'{cell}: archive audit - {len(now)} files present, {n} published - FAILED')
    by_name = {}
    for p in glob.glob(f'{fd}/*.fits') + glob.glob(f'{fd}/loo/*.fits') + glob.glob(f'{sd}/*_epoch_*.fits'):
        try:
            nm = _f.getheader(p).get('PRODNAME')
        except Exception:
            nm = None
        if nm:
            by_name[nm if nm.endswith('.fits') else nm + '.fits'] = p
    for f in ('PB_deep_%s.fits' % scope, 'PB_deep_R075_%s.fits' % scope, 'weights.csv', 'night_image_gate.csv', 'epoch_gate.csv'):
        by_name[f] = f'{fd}/{f}'
    bad = [p for p in now if os.path.basename(p) not in by_name or md5(p) != md5(by_name[os.path.basename(p)])]
    if bad:
        raise RuntimeError(f'{cell}: archive audit - {len(bad)} files whose md5 differs from the working source, e.g. {bad[:2]} - FAILED')
    wdates = {r['date'] for r in csv.DictReader(open(f'{fd}/weights.csv'))}; gfail = NG.failed_dates(fd)
    from exopipe.phase3 import epoch_gate as EG
    try:
        gfail = gfail | EG.failed_dates(fd, scope)
    except FileNotFoundError:
        pass
    miss = []
    for p in sorted(glob.glob(f'{sd}/*_sub_{scope}.fits')):
        e_ = C._EPOCH_RE.match(os.path.basename(p))
        if not e_ or e_['date'] not in wdates or e_['date'] in gfail:
            continue
        ep = [q for q in glob.glob(f"{arch}/{lst}/{e_['date']}/{run}/{scope}/{e_['st']}/deep/*.fits")
              if not any(t in os.path.basename(q) for t in ('_msub', '_pbcorr', '_loo'))]
        if not ep:
            miss.append(f"{e_['date']} {e_['st']}")
    if miss:
        raise RuntimeError(f'{cell}: archive audit - {len(miss)} searched nights without an archived epoch image, e.g. {miss[:4]} - FAILED')


def epochs_only(cell, root=ROOT):
    """REPAIR of an installed cell whose archive lacks its epoch images (Gregg 2026-10-04, "1. Approved": the first install,
    01h 18MHz, quarantined its archived epochs and published none). As a1 + G4 of the install, on the WORKING tree: derive each
    night's epoch (residual + master) where none exists, refuse unless every publishable epoch passes _epoch_content_check
    (derived files removed again), then publish the cell (_archive_publish_cell rewrites its products byte-identically and adds
    the epochs) and run the step-d audit including epoch completeness."""
    lst, scope = cell.split('_', 1); cdir = f'{root}/{cell}'
    os.environ.setdefault('P3_COADD_ROOT', root)
    from exopipe.phase3 import coadd_full_pipeline as C, night_image_gate as NG
    run = os.environ.get('P3_ARCHIVE_RUN')
    if not run:
        raise RuntimeError('P3_ARCHIVE_RUN unset - REFUSING')
    fd, sd = f'{cdir}/coadd_final', f'{cdir}/subtracted'
    wrows = {r['date']: r for r in csv.DictReader(open(f'{fd}/weights.csv'))}; gfail = NG.failed_dates(fd)
    # EPOCH GATE (Gregg 2026-10-04 option B): the record of the installed generation; FAIL nights get no epoch product; a FAIL
    # night inside the master refuses (the master must be rebuilt without it)
    from exopipe.phase3 import epoch_gate as EG
    erows = EG.evaluate_cell(fd, sd, scope); efail = {d for d, r in erows.items() if r['verdict'] == 'FAIL'}
    inm = sorted(d for d in efail if wrows.get(d, {}).get('in_master') == '1' and d not in gfail)
    if inm:
        raise RuntimeError(f'{cell}: night(s) {inm} IN THE MASTER fail the epoch gate - the master must be rebuilt without them; REFUSING')
    new, bad = [], []
    for p in sorted(glob.glob(f'{sd}/*_sub_{scope}.fits')):
        e_ = C._EPOCH_RE.match(os.path.basename(p))
        if not e_ or e_['kind'] != 'sub':
            continue
        if e_['date'] not in wrows or e_['date'] in gfail or e_['date'] in efail:
            continue
        ep = f"{sd}/{e_['date']}_{e_['st']}_epoch_{scope}.fits"
        mp = f'{fd}/' + ('I_deep_Taper_R075' if e_['st'] == 'I' else 'V_deep_Taper_R0') + f'_{scope}.fits'
        if not os.path.exists(ep):
            _derive_epoch(p, mp, ep); new.append(ep)
        ok_, why = C._epoch_content_check(ep, mp, p, e_['st'])
        if not ok_:
            bad.append(f'{os.path.basename(ep)}: {why}')
    if bad:
        for ep in new:
            os.remove(ep)
        raise RuntimeError(f'{cell}: {len(bad)} epochs fail the epoch content check, e.g. {bad[:2]} - REFUSING (derived epochs removed)')
    EG.write_record(fd, erows); rec(cell, 'epoch-gate-record', f'{fd}/{EG.GATE_FILE}', '', md5(f'{fd}/{EG.GATE_FILE}'), '',
                                    f'{len(erows)} nights, FAIL {sorted(efail)}')
    for ep in new:
        rec(cell, 'epoch-derived', ep, '', md5(ep), os.path.getsize(ep), 'residual + master (epochs-only repair)')
    arch = C._archive_root(); mdir = f'{arch}/Coadd/{run}/{lst}/{scope}'
    n = C._archive_publish_cell(fd, sd, lst, scope)
    _archive_audit(cell, fd, sd, mdir, arch, run, lst, scope, n)
    rec(cell, 'epochs-published-and-audited', '', '', '', '', f'{len(new)} epochs derived; {n} products published and audited')
    return len(new), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cell', nargs='?'); ap.add_argument('--gen'); ap.add_argument('--kind', choices=('mdewarp', 'rebuild'))
    ap.add_argument('--rebuild-root', default=None); ap.add_argument('--root', default=ROOT)
    ap.add_argument('--screen-table-unapplied', default=None, metavar='TABLE')
    ap.add_argument('--epochs-only', action='store_true', help='repair an installed cell whose archive lacks its epoch images')
    ap.add_argument('--omit-vclean', action='store_true', help='do not install the generation\'s cleaned-V products (deferred repair)')
    ap.add_argument('--resume', action='store_true', help='continue an install interrupted mid-way (dry run unless --execute)')
    ap.add_argument('--execute', action='store_true', help='with --resume: act, not only report')
    a = ap.parse_args()
    if a.resume:
        if not (a.cell and a.gen and a.kind):
            ap.error('cell, --gen and --kind are required for --resume')
        try:
            r = resume(a.cell, a.gen, a.kind, a.root, omit_vclean=a.omit_vclean, execute=a.execute)
            if r:
                print(f'INSTALL-OK {a.cell} (resumed): {r[0]} working files, {r[1]} archive files quarantined, {r[2]} published', flush=True)
        except Exception as e:
            if a.execute:
                rec(a.cell, 'RESUME-FAILED', a.gen, '', '', '', str(e)[:400])
            print(f'RESUME-REFUSED {a.cell}: {e}', flush=True); sys.exit(1)
        return
    if a.epochs_only:
        try:
            r = epochs_only(a.cell, a.root)
            print(f'EPOCHS-OK {a.cell}: {r[0]} epochs derived, {r[1]} products published and audited', flush=True)
        except Exception as e:
            rec(a.cell, 'EPOCHS-FAILED', '', '', '', '', str(e)[:400])
            print(f'EPOCHS-FAILED {a.cell}: {e}', flush=True); sys.exit(1)
        return
    if a.screen_table_unapplied:
        n1, n0 = screen_table_unapplied(a.screen_table_unapplied)
        print(f'SCREEN-TABLE {n1} applied rows kept, {n0} applied=0 rows written', flush=True); return
    if not (a.cell and a.gen and a.kind):
        ap.error('cell, --gen and --kind are required for an install')
    try:
        r = install(a.cell, a.gen.rstrip('/'), a.kind, a.rebuild_root, a.root, omit_vclean=a.omit_vclean)
        print(f'INSTALL-OK {a.cell}: {r[0]} working files, {r[1]} archive files quarantined, {r[2]} published', flush=True)
    except Exception as e:
        rec(a.cell, 'INSTALL-FAILED', a.gen, '', '', '', str(e)[:400])
        print(f'INSTALL-FAILED {a.cell}: {e}', flush=True); sys.exit(1)


if __name__ == '__main__':
    main()
