#!/usr/bin/env python3
"""Cleaned Stokes-V deep masters — mosaic-input production chain.

PRODUCER -> CONSUMER CONTRACT (GC-PRE-4, Gregg ruling 2026-08-12):
produces V_deep_Taper_R0_clean(+pbcorr) masters — the critical input to the
Stokes V mosaicked products (PHASE3_ARCHITECTURE.md §7bis, ruling
2026-08-12d).

Per cell (Gregg via mosaic chat, 2026-07-11): raw V master ->
search_v_transients_v2 --clean_only (sidelobe + bilobe + striation
subtraction, APPARENT domain, using the matching I master for source
positions) -> divide by the cell's stored PB_deep map -> intrinsic cleaned-V
master.

Outputs per cell (coadd_final/):
  V_deep_Taper_R0_clean_{scope}.fits          (apparent, SKYSUB headers)
  V_deep_Taper_R0_clean_pbcorr_{scope}.fits   (intrinsic, PBMETHOD/PBWINDOW)

I master selection: Taper_R0 for wideband scopes, Taper_R075 for subbands
(the search convention); falls back to the other robust if the primary is
corrupt. Idempotent per cell: an existing pair is skipped only when the sibling passes its content test against the
cell's current PB_deep map (pb_apply.sibling_check); a failing sibling alone is regenerated (2026-10-03). skip_existing=False
rebuilds the pair.

ENGINE: the --clean_only machinery of exopipe/phase3/search_v_transients_v2
(the GC-PRE-1 re-absorbed PRODUCTION copy; main() Phases 2-4 + SKYSUB header
stamp), shelled as a subprocess exactly as the frozen production driver
(clv15/phase3_vclean_masters.py, now superseded by this module) did.
History: engine ran the 2026-07-11 cleaned-V pass 330/330 subband +
wideband cells; the frozen driver's cell-sweep semantics are preserved in
run_vclean().

Callable contract for the golden-cell harness
(exopipe_ops/golden_cells/bin/drive_vclean.py):
    build_vclean_cell(coadd_root, cell) -> (cell, status)
status 'OK' / 'skip (verified: ...)' / 'sibling regenerated (...)' on success (2026-10-03: the old 'skip (exists)' is
retired - an existing sibling is skipped only after its content test); any other string is a refusal
reason (caller decides exit semantics). No import-time side effects.

SOURCE IDENTITY AND REFRESH (production exopipe fc5c164, 2026-11-10; R-GREGG-20261110-VCLEAN-REBUILD-ALL-456-V-MOSAICS-IN-FEATHER3):
the cleaned master carries VCSRC / VCISRC (md5 of the V and I masters it was made from); vclean_source_check calls a pair
STALE when VCSRC is not the cell's current V master or its time block differs; build_vclean_cell refuses a stale pair in its
skip path; refresh_vclean_cell re-cleans, quarantines, installs, stamps and archives one installed cell. The install step runs
the refresh after its verify and refuses unless a second call reports CURRENT (production install_cell.sh:28-34, exopipe_ops
532ea6b; FIX-20261113-DEEP-REBUILD-CLEANS-V-BEFORE-FLUX-APPLY: the rebuild cleans V before the flux apply, so every rebuilt
cell's pair is stale until refreshed).

Usage (batch, mirrors the frozen driver):
    python vclean_masters.py [wideband|subband|all] [NPAR]
    python vclean_masters.py refresh <cell list file> [NPAR] [tmp root]
"""
from exopipe.phase3.config import coadd_root
import os
import re
import sys
import glob
import shutil
import tempfile
import subprocess

import numpy as np
from astropy.io import fits

WB_ROOT = None   # NO default (Gregg 2026-09-04): coadd_root() at point of use
SB_ROOT = '/data07/gh/phase3/coadd_sb'
TMP_ROOT = '/fast/gh/main/tmp'
SV2_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'search', 'search_v.py')
SV2_TIMEOUT_S = 7200
# SOURCE IDENTITY (Gregg 2026-11-10, R-GREGG-20261110-VCLEAN-REBUILD-ALL-456-V-MOSAICS-IN-FEATHER3): the cleaned master records the
# md5 of the V master it was cleaned from (VCSRC) and of the I master that gave the source positions (VCISRC). A pair whose VCSRC is
# not the cell's current V master is STALE (175 of 456 cells held older night sets, 63 since-excluded nights, 2026-11-10).
# (production vclean_masters.py:61-92, exopipe fc5c164)
VCSRC, VCISRC = 'VCSRC', 'VCISRC'
TIME_CARDS = ('NCOADD', 'DATE-OBS', 'DATE-END')


def vclean_source_check(clean_path, v_path):
    """(ok, why): the cleaned master was cleaned from THIS V master - its VCSRC equals md5(V master), and its time block
    (NCOADD, DATE-OBS, DATE-END) and COADD constituent nights equal the V master's."""
    from exopipe.phase3.lib.install.install import md5
    if not os.path.exists(clean_path):
        return False, 'no cleaned master'
    hc, hv = fits.getheader(clean_path, memmap=False), fits.getheader(v_path, memmap=False)
    src = hc.get(VCSRC)
    if not src:
        return False, 'no VCSRC card (cleaned before 2026-11-10: source unknown)'
    if src != md5(v_path):
        return False, f'VCSRC {src} is not the current V master'
    bad = [k for k in TIME_CARDS if str(hc.get(k)) != str(hv.get(k))]
    cons = lambda h: [str(x) for x in h.get('HISTORY', []) if str(x).startswith('COADD constituent')]
    if bad or cons(hc) != cons(hv):
        return False, f'time block differs from the V master ({bad or "COADD constituents"})'
    return True, f'cleaned from the current V master ({src[:8]}), time block and {len(cons(hv))} constituents equal'


def sane(path):
    """A master is sane if it exists, loads, and |max| is finite and < 1e6 Jy."""
    if not os.path.exists(path):
        return False
    try:
        a = fits.getdata(path, memmap=False)
        m = float(np.nanmax(np.abs(a)))
        return np.isfinite(m) and m < 1e6
    except Exception:
        return False


def pick_i_master(coadd_final_dir, scope):
    """Matching I master for the clean pass: R0 first for wideband scopes,
    R075 first for subbands (search convention); fall back if corrupt."""
    order = (['Taper_R0', 'Taper_R075'] if not scope.endswith('MHz')
             else ['Taper_R075', 'Taper_R0'])
    for var in order:
        p = f'{coadd_final_dir}/I_deep_{var}_{scope}.fits'
        if sane(p):
            if var != order[0]:
                from exopipe.phase3.core import fallback      # WAVE2 B-30
                fallback.site('fallbacks.VCLEAN_I_MASTER_FALLBACK', f'{coadd_final_dir}/I_deep_{order[0]}_{scope}.fits '
                              f'missing or corrupt - the V clean would use I_deep_{var}')
            return p
    return None


def write_vclean_sibling(cf, scope):
    """Write V_deep_Taper_R0_clean_pbcorr_{scope}.fits = the cleaned master / the cell's STORED PB_deep map (PB > floor), with
    the PB provenance of the cell's V_deep_Taper_R0 sibling (same map, same run: PBMODEL/PBCHROM/PBRESID/PBNDATE) and the
    PBMETHOD/PBWINDOW/BEAMMOD cards phase3_pbcorr_masters.do_cell writes. The cleaning engine is not run."""
    from exopipe.phase3.lib.masters import pb_apply as PB
    from exopipe.phase3.lib.masters.pbcorr_masters import BEAMMOD
    out_ap = f'{cf}/V_deep_Taper_R0_clean_{scope}.fits'
    out_pb = f'{cf}/V_deep_Taper_R0_clean_pbcorr_{scope}.fits'
    ref = f'{cf}/V_deep_Taper_R0_pbcorr_{scope}.fits'
    meta = None
    if os.path.exists(ref):
        rh = fits.getheader(ref, memmap=False)
        meta = dict(pb_model=rh.get('PBMODEL', 'v1-2026-07-04'), chromatic=rh.get('PBCHROM'), residual_pb=rh.get('PBRESID'),
                    n_dates=rh.get('PBNDATE', -1), n_missing=rh.get('PBMISS'))
    pb_map = np.squeeze(fits.getdata(f'{cf}/PB_deep_{scope}.fits', memmap=False)).astype(np.float64)
    PB.apply_pbcorr(out_ap, pb_map, out_pb, meta=meta)          # pb_apply reads with the package's switch (use_memmap False)
    with fits.open(out_pb, mode='update', memmap=False) as h:
        h[0].header['PBMETHOD'] = ('time-integrated-library', 'canonical (2026-07-11)')
        h[0].header['PBWINDOW'] = ('60min-deep', 'PB integration window')
        h[0].header['BEAMMOD'] = (BEAMMOD, 'beam model')
    ok, why = PB.sibling_check(out_pb, out_ap, pb_map)
    if not ok:
        raise RuntimeError(f'{out_pb}: written sibling fails its content test ({why})')
    return why


def build_vclean_cell(coadd_root, cell, tmp_root=None, skip_existing=True):
    """Build V_deep_Taper_R0_clean(+pbcorr) for one cell. Returns (cell, status).

    Refuses (status string, no product) when the raw V master, a sane I
    master, or the PB_deep map is missing — the PB map must exist first
    (phase3_pbcorr_masters / drive_pbcorr).
    """
    from exopipe.phase3.lib.masters import pb_apply as PB

    lst, scope = cell.split('_', 1)
    cf = f'{coadd_root}/{cell}/coadd_final'
    vp = f'{cf}/V_deep_Taper_R0_{scope}.fits'
    out_ap = f'{cf}/V_deep_Taper_R0_clean_{scope}.fits'
    out_pb = f'{cf}/V_deep_Taper_R0_clean_pbcorr_{scope}.fits'
    # CONTENT GUARD, not existence (CLAUDE.md 3; Gregg 2026-10-03 "preamble approved"): an existing pair is skipped only when
    # the sibling equals the cleaned master / the cell's CURRENT PB_deep map; otherwise only the sibling is regenerated. The
    # existence-only skip let PB step 2's map rewrite (2026-10-01) leave 19 siblings stale by up to 5.6 per cent.
    if skip_existing and os.path.exists(out_pb) and os.path.exists(out_ap):
        pbp = f'{cf}/PB_deep_{scope}.fits'
        if not os.path.exists(pbp):
            return cell, 'NO PB MAP'
        # 2026-11-10: a self-consistent pair is skipped only when it was cleaned from the CURRENT V master (VCSRC); a stale pair
        # is refused here and rebuilt, quarantined and archived by refresh_vclean_cell
        cur, cwhy = vclean_source_check(out_ap, vp)
        if not cur:
            return cell, f'REFUSED stale cleaned-V pair ({cwhy}) - refresh_vclean_cell rebuilds it'
        ok, why = PB.sibling_check(out_pb, out_ap, np.squeeze(fits.getdata(pbp, memmap=False)))
        if ok:
            return cell, 'skip (verified: sibling x PB == cleaned master)'
        if fits.getheader(f'{cf}/I_deep_Taper_R075_{scope}.fits', memmap=False).get('MDWARP'):
            return cell, (f'REFUSED stale sibling ({why}) in a master-dewarped cell: its PB maps are made by the dewarp '
                          f'install, not here')
        try:
            return cell, f'sibling regenerated (was: {why}; now: {write_vclean_sibling(cf, scope)})'
        except Exception as e:
            return cell, f'ERROR regenerating the sibling: {e}'
    if not sane(vp):
        return cell, 'NO SANE V MASTER'
    ip = pick_i_master(cf, scope)
    if ip is None:
        return cell, 'NO SANE I MASTER'
    pbp = f'{cf}/PB_deep_{scope}.fits'
    if not os.path.exists(pbp):
        return cell, 'NO PB MAP'
    tmp_root = tmp_root or TMP_ROOT
    os.makedirs(tmp_root, exist_ok=True)
    td = tempfile.mkdtemp(prefix=f'vclean_{cell}_', dir=tmp_root)
    try:
        env = {**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
               'OPENBLAS_NUM_THREADS': '2'}
        # sv2 imports exopipe.* (GC-PRE-1 rewrite): repo root on PYTHONPATH.
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))))
        env['PYTHONPATH'] = repo_root + (
            os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
        r = subprocess.run(
            [sys.executable, SV2_SCRIPT,
             '--v_image', vp, '--i_image', ip,
             '--clean_only', '--no_cutouts', '--out_dir', td],
            capture_output=True, text=True, timeout=SV2_TIMEOUT_S, env=env)
        res = glob.glob(f'{td}/*_residual_final.fits')
        if r.returncode != 0 or not res:
            return cell, f'sv2 FAILED rc={r.returncode}: {r.stderr[-200:]}'
        from exopipe.phase3.lib.install.install import md5
        with fits.open(res[0], mode='update', memmap=False) as h:
            h[0].header[VCSRC] = (md5(vp), 'md5 of the V master this was cleaned from')
            h[0].header[VCISRC] = (md5(ip), 'md5 of the I master (source positions)')
        shutil.move(res[0], out_ap)
        write_vclean_sibling(cf, scope)     # one writer for build and repair (2026-10-03)
        return cell, 'OK'
    except Exception as e:
        return cell, f'ERROR {e}'
    finally:
        shutil.rmtree(td, ignore_errors=True)


def refresh_vclean_cell(cell, root=None, tmp_root=None, mdw_table=None, check_only=False):
    """Release 1 cleaned-V refresh of ONE installed cell (Gregg 2026-11-10, R-GREGG-20261110-VCLEAN-REBUILD-ALL-456-V-MOSAICS-IN-
    FEATHER3; production vclean_masters.py:220-324, exopipe fc5c164). (1) A pair already cleaned from the current V master (VCSRC)
    with a passing sibling and an archive copy equal to it is reported CURRENT and left alone. Otherwise: (2) sv2 --clean_only into
    scratch, VCSRC / VCISRC stamped, and the content test vclean_source_check (VCSRC = md5 of the current V master; NCOADD,
    DATE-OBS, DATE-END and COADD constituents equal the V master's) passes BEFORE anything moves; (3) the old pair -> the install
    round's working quarantine (install._move_to_wq, md5 verified); (4) the new master installed, its PB sibling written (sibling
    content test), both stamped (header standard + PRODNAME), both tests re-run on the installed files; (5) the cell's exclusion
    conformance gate over every product - a refusal leaves the new pair installed and NOT archived; (6) the cell's archived vclean
    files -> the round's archive quarantine (md5 re-checked), the new pair published (size-checked copy), the archive audited: its
    vclean files are exactly the pair, md5 equal. Env: P3_ARCHIVE_RUN (required), P3_INSTALL_ROUND (the quarantine pair; the
    install step sets both). Returns (cell, status): 'OK ...' or 'CURRENT ...' on success, anything else is a refusal.
    mdw_table (v2, CUTOVER_READINESS "INSTALL-CELL HEADER REPAIR"): the MDWSTAT table path - the Release 1 cards are written on the
    new pair after its stamp, before it is published (release1_cards.write_cell), as install step b writes them on the cell's
    other co-adds; production instead repairs the archived pair afterwards (install_cell.sh:35-54). None = production's pair.
    With mdw_table, CURRENT also requires the cards: a pair that is current but still lacks a sourced card takes the same path
    as a refresh without the sv2 run - (2') md5-verified working copies of the installed pair in scratch, then (3)-(6) with the
    copies as the new pair, carded before the publish - so an archived file is never edited in place.
    check_only: report CURRENT or 'NOT CURRENT (<why>)' and write nothing (the install step's second call: a refused or failed
    attempt never runs the two-hour sv2 job again in the same invocation).
    Every FITS read here and in the helpers it calls is made with astropy's memory map off (memmap=False, and the switch held
    off for the call: R-GREGG-20261117-NO-MMAP-OF-LUSTRE-FILES-READ-OR-WRITE)."""
    with fits.conf.set_temp('use_memmap', False):
        return _refresh(cell, root, tmp_root, mdw_table, check_only)


def _refresh(cell, root, tmp_root, mdw_table, check_only):
    from exopipe.phase3.lib.masters import pb_apply as PB, coadd as C
    from exopipe.phase3.lib.install import install as PI
    root = root or SB_ROOT
    lst, scope = cell.split('_', 1)
    cf = f'{root}/{cell}/coadd_final'
    vp = f'{cf}/V_deep_Taper_R0_{scope}.fits'
    out_ap, out_pb = f'{cf}/V_deep_Taper_R0_clean_{scope}.fits', f'{cf}/V_deep_Taper_R0_clean_pbcorr_{scope}.fits'
    run = os.environ.get('P3_ARCHIVE_RUN')
    if not run:
        return cell, 'REFUSED: P3_ARCHIVE_RUN unset'
    arch = C._archive_root(); mdir = f'{arch}/Coadd/{run}/{lst}/{scope}'; qroot = f'{arch}/{PI.QROOT_NAME}'
    patt = f'{mdir}/{scope}_V_deep_Taper_Robust+0.0_vclean_*.fits'

    def dest(p):
        nm = fits.getheader(p, memmap=False).get('PRODNAME') or ''
        return f'{mdir}/{nm if nm.endswith(".fits") else nm + ".fits"}' if nm else None

    def audit():
        want = {dest(p): p for p in (out_ap, out_pb)}
        if None in want:
            return 'an installed file has no PRODNAME'
        now = sorted(glob.glob(patt))
        if sorted(want) != now:
            return f'archive vclean files {[os.path.basename(x) for x in now]} are not the installed pair'
        bad = [os.path.basename(d) for d, s in want.items() if PI.md5(d) != PI.md5(s)]
        return f'archive md5 differs: {bad}' if bad else ''

    def uncarded():
        """{file: Release 1 cards a write would still add} of the installed pair; {} without mdw_table (production's pair). A
        file the repair refuses whole has nothing to add (the cards stage leaves it to the sibling card-key gate)."""
        if not mdw_table:
            return {}
        from exopipe.phase3.lib.masters import release1_cards as R1
        adds, _ = R1.planned_adds(cf, lst, scope, mdw_table=mdw_table, name_dir=mdir,
                                  files=[os.path.basename(out_ap), os.path.basename(out_pb)])
        return {f: a for f, a in adds.items() if a and not a[0].startswith('(refused whole')}

    if not sane(vp):
        return cell, 'NO SANE V MASTER'
    ip = pick_i_master(cf, scope)
    if ip is None:
        return cell, 'NO SANE I MASTER'
    pbp = f'{cf}/PB_deep_{scope}.fits'
    if not os.path.exists(pbp):
        return cell, 'NO PB MAP'
    pb_map = np.squeeze(fits.getdata(pbp, memmap=False))
    cur, cwhy = vclean_source_check(out_ap, vp)
    recard = {}
    if not cur:
        state = cwhy
    elif not os.path.exists(out_pb):
        state = 'no pbcorr sibling'
    else:
        sok, swhy = PB.sibling_check(out_pb, out_ap, pb_map)
        if not sok:
            state = f'sibling: {swhy}'
        else:
            a, recard = audit(), uncarded()
            if not a and not recard:
                return cell, (f'CURRENT ({cwhy}; sibling and archive copies verified'
                              + ('; every sourced Release 1 card present' if mdw_table else '') + ')')
            state = '; '.join(x for x in (a, f'Release 1 cards still to add {recard}' if recard else '') if x)
    if check_only:
        return cell, f'NOT CURRENT ({state})'
    tmp_root = tmp_root or TMP_ROOT
    os.makedirs(tmp_root, exist_ok=True)
    td = tempfile.mkdtemp(prefix=f'vclean_{cell}_', dir=tmp_root)
    try:
        if recard:                                                   # (2') current but uncarded: working copies, no sv2
            src_ap, src_pb = f'{td}/{os.path.basename(out_ap)}', f'{td}/{os.path.basename(out_pb)}'
            for s_, d_ in ((out_ap, src_ap), (out_pb, src_pb)):
                m_ = PI.md5(s_); shutil.copy2(s_, d_)
                if PI.md5(d_) != m_:
                    return cell, f'REFUSED before install: the working copy of {os.path.basename(s_)} is not verified (md5)'
        else:                                                        # (2)
            env = {**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'}
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))))))
            env['PYTHONPATH'] = repo_root + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
            r = subprocess.run([sys.executable, SV2_SCRIPT, '--v_image', vp, '--i_image', ip, '--clean_only', '--no_cutouts',
                                '--out_dir', td], capture_output=True, text=True, timeout=SV2_TIMEOUT_S, env=env)
            res = glob.glob(f'{td}/*_residual_final.fits')
            if r.returncode != 0 or not res:
                return cell, f'sv2 FAILED rc={r.returncode}: {r.stderr[-200:]}'
            src_ap, src_pb = res[0], None
            with fits.open(src_ap, mode='update', memmap=False) as h:
                h[0].header[VCSRC] = (PI.md5(vp), 'md5 of the V master this was cleaned from')
                h[0].header[VCISRC] = (PI.md5(ip), 'md5 of the I master (source positions)')
        ok, why = vclean_source_check(src_ap, vp)
        if not ok:
            return cell, f'REFUSED before install: new cleaned master fails its content test ({why})'
        for p in (out_ap, out_pb):                                   # (3)
            if os.path.exists(p):
                PI._move_to_wq(cell, p, f'coadd_final/{os.path.basename(p)}')
        shutil.move(src_ap, out_ap)                                  # (4)
        if src_pb:
            shutil.move(src_pb, out_pb)
        else:
            write_vclean_sibling(cf, scope)
        _d, _ = C.load_image(vp)
        thermnoi = C.get_uniform_rms(_d, ~np.isfinite(_d) | (_d == 0))
        for p in (out_ap, out_pb):
            if not C._stamp_master_standard(p, lst, scope, thermnoi):
                return cell, f'REFUSED: {os.path.basename(p)} could not be stamped (installed, NOT archived)'
        cards = ''
        if mdw_table:                                                # v2: the Release 1 cards before the publish
            from exopipe.phase3.lib.masters import release1_cards as R1
            rep = R1.write_cell(cf, lst, scope, mdw_table=mdw_table, name_dir=mdir,
                                files=[os.path.basename(out_ap), os.path.basename(out_pb)])
            cards = '; Release 1 cards: ' + ' | '.join(rep['lines'])
            left = uncarded()
            if left:
                return cell, f'REFUSED: Release 1 cards still to add after the write {left} (installed, NOT archived)'
        ok1, why1 = vclean_source_check(out_ap, vp)
        ok2, why2 = PB.sibling_check(out_pb, out_ap, pb_map)
        if not (ok1 and ok2):
            return cell, f'REFUSED: installed pair fails its content tests ({why1}; {why2}) - NOT archived'
        try:                                                         # (5)
            C._refuse_excluded_constituents(cf, lst, scope)
        except RuntimeError as e:
            return cell, f'REFUSED (installed, NOT archived): {e}'
        for p in sorted(glob.glob(patt)):                            # (6)
            q = f'{qroot}/{os.path.relpath(p, arch)}'
            if os.path.exists(q):
                return cell, f'REFUSED (installed, NOT archived): archive quarantine destination exists {q}'
            os.makedirs(os.path.dirname(q), exist_ok=True); m = PI.md5(p); n = os.path.getsize(p); os.rename(p, q)
            if PI.md5(q) != m:
                raise RuntimeError(f'{cell}: archive quarantine move of {p} not verified (md5)')
            PI.rec(cell, 'vclean-archive-quarantine', p, q, m, n)
        for p in (out_ap, out_pb):
            d = dest(p); C._atomic_copy(p, d); PI.rec(cell, 'vclean-published', p, d, PI.md5(d), os.path.getsize(d))
        a = audit()
        if a:
            return cell, f'REFUSED: archive audit - {a}'
        PI.rec(cell, 'vclean-refreshed', out_ap, '', '', '', (f'recarded ({state}); ' if recard else '') + f'{why1}; sibling {why2}')
        return cell, (f'OK ({"recarded, no sv2 run: the pair was current but " + state + "; " if recard else ""}'
                      f'{why1}; sibling {why2}; 2 archive files published and audited{cards})')
    except Exception as e:
        return cell, f'ERROR {e}'
    finally:
        shutil.rmtree(td, ignore_errors=True)


def run_refresh(cells, npar=4, tmp_root=None):
    """Refresh the given cells (refresh_vclean_cell) NPAR at a time; prints one status line per cell and a summary
    (production vclean_masters.py:327-339)."""
    import multiprocessing as mp_
    from functools import partial
    print(f'{len(cells)} cells, NPAR={npar}', flush=True)
    nok = nbad = 0
    with mp_.get_context('fork').Pool(npar) as pool:
        for cell, st in pool.imap_unordered(partial(refresh_vclean_cell, tmp_root=tmp_root), cells):
            ok = st.startswith('OK') or st.startswith('CURRENT')
            nok += ok; nbad += (not ok)
            print(f'  {cell}: {st}', flush=True)
    print(f'VCLEAN-REFRESH-DONE ok={nok} problems={nbad}', flush=True)
    return nok, nbad


def enumerate_cells(mode='all', wb_root=None, sb_root=None):
    """(root, cell) pairs for the batch sweep, frozen-driver semantics."""
    wb_root = wb_root or coadd_root()
    sb_root = sb_root or SB_ROOT
    cells = []
    if mode in ('wideband', 'all') and os.path.isdir(wb_root):
        cells += [(wb_root, c) for c in sorted(os.listdir(wb_root))
                  if re.match(r'^\d\dh_(Red|Green|Blue|Full)$', c)]
    if mode in ('subband', 'all') and os.path.isdir(sb_root):
        cells += [(sb_root, c) for c in sorted(os.listdir(sb_root))
                  if re.match(r'^\d\dh_\d+MHz$', c)]
    return cells


def _do_cell(args):
    root, cell = args
    return build_vclean_cell(root, cell)


def run_vclean(mode='all', npar=2, wb_root=None, sb_root=None):
    """Batch sweep across cells (mirror of the frozen driver's __main__ body)."""
    import multiprocessing as mp_
    cells = enumerate_cells(mode, wb_root, sb_root)
    print(f'{len(cells)} cells ({mode}), NPAR={npar}', flush=True)
    nok = nbad = 0
    with mp_.get_context('fork').Pool(npar) as pool:
        for cell, st in pool.imap_unordered(_do_cell, cells):
            ok = st == 'OK' or st.startswith('skip (verified') or st.startswith('sibling regenerated')
            nok += ok
            nbad += (not ok)
            print(f'  {cell}: {st}', flush=True)
    print(f'VCLEAN-MASTERS-DONE ok={nok} problems={nbad}', flush=True)
    return nok, nbad


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'refresh':     # refresh <cell list file> [NPAR] [tmp root]
        cells = [ln.strip() for ln in open(sys.argv[2]) if ln.strip() and not ln.startswith('#')]
        _, bad = run_refresh(cells, int(sys.argv[3]) if len(sys.argv) > 3 else 4, sys.argv[4] if len(sys.argv) > 4 else None)
        sys.exit(1 if bad else 0)
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
    npar = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    run_vclean(mode, npar)


if __name__ == '__main__':
    main()
