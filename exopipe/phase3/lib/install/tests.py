"""Content tests of the product classes `exopipe3 install` writes (discovered by core/validate from lib/install/tests.py).

Every test reads the product's CONTENT and derives what it needs from the path alone (record.resume_check re-runs tests
without context): the working root and the archive root of a product follow the re-rooting convention of
steps_common.Roots (production roots, or <out>/coadd_sb and <out>/archive under --out).

  install_manifest           rows_match_files          the attempt manifest against the files it names (md5, sizes, sets)
  installed_cell             membership, products, epoch_gate_current, exclusion_conformance, prodname,
                             T2_gate_records, T3_loo, T4_pb_siblings, T7_date_sets, epochs
  archive_cell               audit                     the archive cell = what the publish wrote (md5) + epoch completeness
  screen_table               rows                      the archived screen record mdewarp_screens.csv: 15 columns, one
                                                       row per (lst, scope), applied rows' screens beside it with their
                                                       md5, applied=0 rows without a screen (one class for install's
                                                       applied=1 rows and dewarp's applied=0 rows - crosscheck 4.2)
                             install_cell_row          with an install context: the installed cell's one applied=1 row,
                                                       its published screen md5 = row = APPLIED_SCREEN.json, the row's
                                                       numbers = the JSON
  research_request           queued_once               the re-search state edits of an enqueue, still in force
  rerun_epochs               items_match_cells         the per-hour re-run items = the installed cells' nights now
  cleanup_record             removed                   the removed rebuild scratch is gone
"""
import csv
import glob
import io
import json
import os
from contextlib import redirect_stdout

from exopipe.phase3.core.validate import content_test


def _SC():
    from exopipe.phase3.lib.install import steps_common as SC
    return SC


def _cell_of(path):
    from exopipe.phase3.core import cells
    cell = os.path.basename(os.path.abspath(path).rstrip('/'))
    lst, scope = cells.parse_cell(cell)
    return cell, lst, scope


def _fits_data_equal(a, b):
    """True when two FITS files hold the same primary pixels (bit for bit, NaN where NaN). The install stamps the header
    standard on installed masters, LOO masters and epochs IN PLACE after their 'installed' row (steps b and c), so their
    file md5 legitimately differs from the generation's while their pixels must not."""
    import numpy as np
    from astropy.io import fits
    with fits.open(a, memmap=False) as ha, fits.open(b, memmap=False) as hb:
        da, db = ha[0].data, hb[0].data
        if da is None or db is None or da.shape != db.shape or da.dtype != db.dtype:
            return False
        return np.ascontiguousarray(da).tobytes() == np.ascontiguousarray(db).tobytes()


def _short(items, n=4):
    items = list(items)
    return '; '.join(str(x) for x in items[:n]) + (f' ... (+{len(items) - n})' if len(items) > n else '')


# ---------------------------------------------------------------------------------------------------------------------
# install_manifest
# ---------------------------------------------------------------------------------------------------------------------

@content_test('install_manifest')
def rows_match_files(path, **ctx):
    """Every action row against the file it names: installed (destination md5 = row md5 = generation source md5),
    cell-file / epoch-gate-record / screen-published (md5), working-quarantine (copy md5 = row; source gone or replaced),
    archive-quarantine (moved file: size and md5 = row); plan rows: every destination 'existed: no', written before the
    first action row, and every quarantine move of the attempt was planned. A plan without action rows (plan only) is
    valid while its destinations are still free."""
    SC = _SC()
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return False, 0, 'empty manifest'
    if set(SC.MCOLS) - set(rows[0]):
        return False, 0, f'columns {sorted(set(SC.MCOLS) - set(rows[0]))} missing'
    plan = [r for r in rows if r['section'] == 'plan']
    act = [r for r in rows if r['section'] == 'action']
    bad = []
    pdest = {r['dest'] for r in plan if r['step'] == 'plan-dest'}
    for r in plan:
        if r['step'] == 'plan-dest' and 'existed: no' not in r['detail']:
            bad.append(f'plan-dest {r["dest"]}: {r["detail"]}')
    if plan and act and max(r['time'] for r in plan) > min(r['time'] for r in act):
        bad.append(f'plan written at {max(r["time"] for r in plan)} after the first action row {min(r["time"] for r in act)}')
    if not act:
        taken = [d for d in pdest if os.path.exists(d)]
        if taken:
            bad.append(f'plan only, but {len(taken)} planned destinations exist now: {_short(taken, 2)}')
        return not bad, len(plan), (f'plan only: {len(pdest)} destinations, free' if not bad else _short(bad))
    # md5 of everything the rows name, in parallel
    want = []
    for r in act:
        if r['step'] in ('installed', 'cell-file', 'screen-published'):
            want.append(r['dest'])
            if r['step'] == 'installed' and os.path.isabs(r['path']):
                want.append(r['path'])
        elif r['step'] in ('working-quarantine', 'archive-quarantine'):
            want.append(r['dest'])
        elif r['step'] == 'epoch-gate-record':
            want.append(r['path'])
    sums = SC.md5_many(want)
    installed_dest = {r['dest'] for r in act if r['step'] in ('installed', 'cell-file')}
    celldirs = [r['dest'] for r in act if r['step'] == 'cell-dir']
    n = 0
    restamped = 0
    later_egr = {}
    for r in act:
        st = r['step']
        if st == 'installed':
            n += 1
            src = r['path'] if os.path.isabs(r['path']) else None      # 'derived at install: ...' has no source file
            if src and sums.get(src) not in (None, r['md5']):
                bad.append(f'installed {r["dest"]}: generation source {src} md5 {sums.get(src)} != row (changed since)')
            if sums.get(r['dest']) != r['md5']:
                # stamped in place after the move (header standard + PRODNAME): the pixels must still be the source's
                if not (sums.get(r['dest']) and src and sums.get(src) == r['md5']
                        and r['dest'].lower().endswith('.fits') and _fits_data_equal(r['dest'], src)):
                    bad.append(f'installed {r["dest"]}: md5 {sums.get(r["dest"])} != row {r["md5"]}'
                               + (' and its pixels differ from the generation source' if src and sums.get(r['dest'])
                                  else ''))
                else:
                    restamped += 1
        elif st in ('cell-file', 'screen-published'):
            n += 1
            if sums.get(r['dest']) != r['md5']:
                bad.append(f'{st} {r["dest"]}: md5 {sums.get(r["dest"])} != row {r["md5"]}')
        elif st == 'epoch-gate-record':
            later_egr[r['path']] = r['md5']
        elif st == 'working-quarantine':
            n += 1
            if sums.get(r['dest']) != r['md5']:
                bad.append(f'working-quarantine {r["dest"]}: md5 {sums.get(r["dest"])} != row {r["md5"]}')
            if os.path.exists(r['path']) and r['path'] not in installed_dest and r['path'] not in later_egr \
                    and not any(r['path'].startswith(d.rstrip('/') + '/') for d in celldirs) \
                    and not r['path'].endswith('/epoch_gate.csv'):
                bad.append(f'working-quarantine source {r["path"]} still present and not replaced')
            if pdest and r['dest'] not in pdest:
                bad.append(f'working-quarantine {r["dest"]} was not in the plan (G-DEST incomplete)')
        elif st == 'archive-quarantine':
            n += 1
            if not os.path.exists(r['dest']):
                bad.append(f'archive-quarantine {r["dest"]} missing')
                continue
            if r['bytes'] and int(r['bytes']) != os.path.getsize(r['dest']):
                bad.append(f'archive-quarantine {r["dest"]}: size {os.path.getsize(r["dest"])} != row {r["bytes"]}')
            if r['md5'] and sums.get(r['dest']) != r['md5']:
                bad.append(f'archive-quarantine {r["dest"]}: md5 {sums.get(r["dest"])} != row {r["md5"]}')
            if pdest and r['dest'] not in pdest:
                bad.append(f'archive-quarantine {r["dest"]} was not in the plan (G-DEST incomplete)')
        elif st == 'cell-dir':
            if not os.path.isdir(r['dest']):
                bad.append(f'cell-dir {r["dest"]} missing')
    for p, m in later_egr.items():
        if sums.get(p) != m:
            bad.append(f'epoch-gate-record {p}: md5 {sums.get(p)} != row {m}')
    failed = [r for r in act if r['step'] in ('INSTALL-FAILED', 'RESUME-FAILED', 'EPOCHS-FAILED')]
    ok_row = [r for r in act if r['step'] in ('installed-and-audited', 'epochs-published-and-audited')]
    detail = (f'{n} rows verified ({sum(1 for r in act if r["step"] == "installed")} installed, '
              f'{sum(1 for r in act if r["step"] == "working-quarantine")} working-quarantine, '
              f'{sum(1 for r in act if r["step"] == "archive-quarantine")} archive-quarantine; {restamped} installed '
              f'files re-stamped in place, pixels = generation); '
              + ('audited' if ok_row else ('attempt FAILED: ' + failed[-1]['detail'][:200] if failed else 'not audited')))
    return not bad, n, (detail if not bad else f'{len(bad)} problem(s): {_short(bad)}')


# ---------------------------------------------------------------------------------------------------------------------
# installed_cell (T8 = membership + products; T2, T3, T4, T6 (epochs), T7 of check_rebuild.py on the installed tree)
# ---------------------------------------------------------------------------------------------------------------------

@content_test('installed_cell', name='membership')
def ic_membership(path, **ctx):
    """membership.verify (T8, part 1): every master night obeys the membership rulings."""
    from exopipe.phase3.lib.masters import membership as MB
    v = MB.verify(os.path.abspath(path).rstrip('/'))
    return not v, len(v), 'membership rulings hold' if not v else _short(v, 3)


@content_test('installed_cell', name='products')
def ic_products(path, **ctx):
    """membership.verify_products (T8, part 2): ruled flux scale, both PB maps, cubic dewarp on an applied cell."""
    from exopipe.phase3.lib.masters import membership as MB
    v = MB.verify_products(os.path.abspath(path).rstrip('/'))
    return not v, len(v), 'products carry the rulings' if not v else _short(v, 3)


@content_test('installed_cell', name='epoch_gate_current')
def ic_epoch_gate(path, **ctx):
    """The epoch-gate record was made against the installed I master (md5 of its float32 data; epoch_gate.py:79-89)."""
    from exopipe.phase3.lib.masters import epoch_gate as EG
    cell, lst, scope = _cell_of(path)
    fd = os.path.join(path, 'coadd_final')
    fails = EG.failed_dates(fd, scope, check_master=True)
    return True, len(fails), f'record current for the installed master; FAIL nights {sorted(fails)}'


@content_test('installed_cell', name='exclusion_conformance')
def ic_exclusions(path, **ctx):
    """No master, variant or LOO master lists a night data_decisions excludes now (coadd _refuse_excluded_constituents)."""
    from exopipe.phase3.lib.masters import coadd as Cfp
    cell, lst, scope = _cell_of(path)
    buf = io.StringIO()
    with redirect_stdout(buf):
        Cfp._refuse_excluded_constituents(os.path.join(path, 'coadd_final'), lst, scope)
    return True, 0, buf.getvalue().strip()[:200] or 'no excluded constituent'


def _masters_and_loo(fd):
    from exopipe.phase3.lib.masters import coadd as Cfp
    ps = [os.path.join(fd, f) for f in sorted(os.listdir(fd)) if f.endswith('.fits') and Cfp._MASTER_RE.match(f)]
    loo = os.path.join(fd, 'loo')
    if os.path.isdir(loo):
        ps += [os.path.join(loo, f) for f in sorted(os.listdir(loo)) if Cfp._LOO_RE.match(f)]
    return ps


@content_test('installed_cell', name='prodname')
def ic_prodname(path, **ctx):
    """Every master, variant and LOO master carries its grammar name (PRODNAME), as the archive publish requires."""
    from exopipe.phase3.core.record import fits_primary_cards
    ps = _masters_and_loo(os.path.join(path, 'coadd_final'))
    miss = [os.path.basename(p) for p in ps if not fits_primary_cards(p).get('PRODNAME')]
    return bool(ps) and not miss, len(ps), (f'{len(ps)} products named' if ps and not miss else
                                            ('no master products' if not ps else f'no PRODNAME: {_short(miss)}'))


def _weights(fd):
    with open(os.path.join(fd, 'weights.csv')) as f:
        return {r['date']: r for r in csv.DictReader(f)}


@content_test('installed_cell', name='T2_gate_records')
def ic_t2(path, **ctx):
    """check_rebuild T2 on the installed tree: a night-image record row for every weights.csv night, no FAIL night in the
    master, and (where weights.csv has the column - rebuilt generations) an applied flux_dc for every in-master night."""
    fd = os.path.join(path, 'coadd_final')
    w = _weights(fd)
    with open(os.path.join(fd, 'night_image_gate.csv')) as f:
        g = {r['date']: r for r in csv.DictReader(f)}
    miss = sorted(set(w) - set(g))
    fin = sorted(d for d, r in g.items() if r.get('verdict') == 'FAIL' and w.get(d, {}).get('in_master') == '1')
    has_dc = any('flux_dc' in r for r in w.values())
    nodc = sorted(d for d, r in w.items() if has_dc and r.get('in_master') == '1' and not (r.get('flux_dc') or '').strip())
    ok = not (miss or fin or nodc)
    return ok, len(miss) + len(fin) + len(nodc), (f'{len(g)} gate rows for {len(w)} nights; flux_dc '
                                                  f'{"checked" if has_dc else "column absent (pre-rebuild generation)"}'
                                                  if ok else f'missing {miss[:4]}; FAIL in master {fin[:4]}; '
                                                             f'in-master without flux_dc {nodc[:4]}')


@content_test('installed_cell', name='T3_loo')
def ic_t3(path, **ctx):
    """check_rebuild T3 on the installed tree: every I LOO master has NCOADD = master's - 1 (and PARENTDG = the master's
    CONSTDIG where the LOO carries PARENTDG - builds before PARENTDG lack it: 166 of 456 cells, 2026-10-13); one per
    in-master night when 2 <= N <= 20 (the search's rule)."""
    from exopipe.phase3.core.record import fits_primary_cards
    cell, lst, scope = _cell_of(path)
    fd = os.path.join(path, 'coadd_final')
    mh = fits_primary_cards(os.path.join(fd, f'I_deep_Taper_R075_{scope}.fits'))
    n, cdig = mh.get('NCOADD'), mh.get('CONSTDIG')
    loos = sorted(glob.glob(os.path.join(fd, 'loo', '*_I_loo_*.fits')))
    lbad = []
    for p in loos:
        h = fits_primary_cards(p)
        if (h.get('PARENTDG') is not None and h.get('PARENTDG') != cdig) or n is None or h.get('NCOADD') != n - 1:
            lbad.append(os.path.basename(p))
    inm = [d for d, r in _weights(fd).items() if r.get('in_master') == '1']
    need = n is not None and 2 <= n <= 20
    ok = not lbad and (not need or len(loos) == n == len(inm))
    return ok, len(loos), f'NCOADD {n}, in-master nights {len(inm)}, I LOO masters {len(loos)} (needed: {need})' + (
        f'; not derivatives: {_short(lbad, 3)}' if lbad else '')


@content_test('installed_cell', name='T4_pb_siblings')
def ic_t4(path, **ctx):
    """check_rebuild T4 on the installed tree: every pbcorr sibling x its robust's PB map = its plain (pb_apply.sibling_check,
    in place). The count is reported, not required (6 with cleaned V; 5 after --omit-vclean)."""
    import numpy as np
    from astropy.io import fits
    from exopipe.phase3.lib.masters import pb_apply as PB
    cell, lst, scope = _cell_of(path)
    fd = os.path.join(path, 'coadd_final')
    sibs = sorted(glob.glob(os.path.join(fd, '*_pbcorr_*.fits')))
    bad = []
    pbs = {}
    for p in sibs:
        b = os.path.basename(p)
        plain = p.replace('_pbcorr_', '_', 1)
        pbp = os.path.join(fd, f'PB_deep_R075_{scope}.fits' if '_R075_' in b else f'PB_deep_{scope}.fits')
        if pbp not in pbs:
            pbs[pbp] = np.squeeze(fits.getdata(pbp)).astype(np.float64)
        ok, why = PB.sibling_check(p, plain, pbs[pbp])
        if not ok:
            bad.append(f'{b}: {why}')
    return bool(sibs) and not bad, len(sibs), (f'{len(sibs)} siblings consistent' if sibs and not bad else
                                               ('no pbcorr siblings' if not sibs else _short(bad, 2)))


@content_test('installed_cell', name='T7_date_sets')
def ic_t7(path, **ctx):
    """check_rebuild T7 on the installed tree: residuals_provisional.csv dates = weights.csv dates (the search's rule)."""
    fd = os.path.join(path, 'coadd_final')
    w = set(_weights(fd))
    rp = os.path.join(path, 'residuals_provisional.csv')
    if not os.path.exists(rp):
        return False, 0, f'{rp} missing'
    with open(rp) as f:
        r = {x['date'] for x in csv.DictReader(f)}
    return r == w, len(r ^ w), f'residuals_provisional {len(r)} dates, weights {len(w)}; differ {sorted(r ^ w)[:4]}'


@content_test('installed_cell', name='epochs')
def ic_epochs(path, **ctx):
    """Every publishable epoch (a weights.csv night with a residual, not refused by the night-image or epoch gate) has its
    epoch image and it passes coadd _epoch_content_check (identity; Stokes I ratio 0.8-1.25) - G4 on the installed tree
    (check_rebuild T6 covers the in-master nights only)."""
    from exopipe.phase3.lib.masters import coadd as Cfp, night_image_gate as NG, epoch_gate as EG
    cell, lst, scope = _cell_of(path)
    fd, sd = os.path.join(path, 'coadd_final'), os.path.join(path, 'subtracted')
    w = _weights(fd)
    refused = NG.failed_dates(fd) | EG.failed_dates(fd, scope, check_master=False)
    bad, n = [], 0
    for p in sorted(glob.glob(os.path.join(sd, f'*_sub_{scope}.fits'))):
        e = Cfp._EPOCH_RE.match(os.path.basename(p))
        if not e or e['kind'] != 'sub' or e['date'] not in w or e['date'] in refused:
            continue
        ep = os.path.join(sd, f"{e['date']}_{e['st']}_epoch_{scope}.fits")
        mp = os.path.join(fd, ('I_deep_Taper_R075' if e['st'] == 'I' else 'V_deep_Taper_R0') + f'_{scope}.fits')
        n += 1
        if not os.path.exists(ep):
            bad.append(f"{e['date']} {e['st']}: no epoch image")
            continue
        ok, why = Cfp._epoch_content_check(ep, mp, p, e['st'])
        if not ok:
            bad.append(f"{e['date']} {e['st']}: {why}")
    return not bad, n, (f'{n} publishable epochs pass' if not bad else f'{len(bad)} of {n}: {_short(bad, 3)}')


# ---------------------------------------------------------------------------------------------------------------------
# archive_cell
# ---------------------------------------------------------------------------------------------------------------------

@content_test('archive_cell')
def audit(path, **ctx):
    """The archive cell <archive>/Coadd/<run>/<lst>/<scope> and its per-night deep files (search frames excluded) are
    exactly the working cell's publishable products, each with the md5 of its working source (by PRODNAME), and every
    searched, non-refused night of weights.csv has its archived epoch for each Stokes with a residual
    (phase3_install._archive_audit, phase3_install.py:507-547; a missing epoch_gate.csv is tolerated as there, B-30 c)."""
    from astropy.io import fits
    from exopipe.phase3.lib.masters import coadd as Cfp, night_image_gate as NG, epoch_gate as EG
    SC = _SC()
    mdir = os.path.abspath(path).rstrip('/')
    scope, lst, run = os.path.basename(mdir), os.path.basename(os.path.dirname(mdir)), \
        os.path.basename(os.path.dirname(os.path.dirname(mdir)))
    arch = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(mdir))))
    working, _ = SC.roots_for_archive(arch)
    cdir = os.path.join(working, f'{lst}_{scope}')
    fd, sd = os.path.join(cdir, 'coadd_final'), os.path.join(cdir, 'subtracted')
    C = SC.cfg().install
    now = [p for p in sorted(glob.glob(f'{mdir}/*')) if os.path.isfile(p)]
    now += [p for p in sorted(glob.glob(C.ARCHIVE_NIGHT_GLOB.format(archive=arch, run=run, lst=lst, scope=scope)))
            if not any(t in os.path.basename(p) for t in C.ARCHIVE_SKIP_TOKENS)]
    if not now:
        return False, 0, f'no archive files under {mdir}'
    by_name = {}
    for p in glob.glob(f'{fd}/*.fits') + glob.glob(f'{fd}/loo/*.fits') + glob.glob(f'{sd}/*_epoch_*.fits'):
        try:
            nm = fits.getheader(p).get('PRODNAME')      # a long string (CONTINUE cards): astropy, as phase3_install :521
        except Exception:
            nm = None
        if nm:
            by_name[nm if nm.endswith('.fits') else nm + '.fits'] = p
    for f in (f'PB_deep_{scope}.fits', f'PB_deep_R075_{scope}.fits') + tuple(C.AUDIT_CELL_FILES):
        by_name[f] = f'{fd}/{f}'
    unknown = [p for p in now if os.path.basename(p) not in by_name]
    sums = SC.md5_many([p for p in now if p not in unknown] + [by_name[os.path.basename(p)] for p in now if p not in unknown])
    differ = [p for p in now if p not in unknown and sums.get(p) != sums.get(by_name[os.path.basename(p)])]
    wdates = set(_weights(fd))
    gfail = NG.failed_dates(fd)
    try:
        gfail = gfail | EG.failed_dates(fd, scope)
    except FileNotFoundError:
        pass
    miss = []
    for p in sorted(glob.glob(f'{sd}/*_sub_{scope}.fits')):
        e = Cfp._EPOCH_RE.match(os.path.basename(p))
        if not e or e['date'] not in wdates or e['date'] in gfail:
            continue
        ep = [q for q in glob.glob(f"{arch}/{lst}/{e['date']}/{run}/{scope}/{e['st']}/deep/*.fits")
              if not any(t in os.path.basename(q) for t in ('_msub', '_pbcorr', '_loo'))]
        if not ep:
            miss.append(f"{e['date']} {e['st']}")
    bad = ([f'not a product of the working cell: {_short(unknown, 2)}'] if unknown else []) + \
          ([f'md5 differs from the working source: {_short(differ, 2)}'] if differ else []) + \
          ([f'searched nights without an archived epoch: {_short(miss, 4)}'] if miss else [])
    return not bad, len(now), (f'{len(now)} archive files = working sources ({cdir}); epochs complete' if not bad
                               else '; '.join(bad))


# ---------------------------------------------------------------------------------------------------------------------
# screen_table (the archived screen record mdewarp_screens.csv; one class owned by install, crosscheck 4.2; dewarp writes
# its applied=0 rows as this class too, lib/dewarp/steps_cell.py)
# ---------------------------------------------------------------------------------------------------------------------

@content_test('screen_table', name='rows')
def screen_table_rows(path, **ctx):
    """mdewarp_screens.csv: the 15 phase3_install columns (SCREEN_COLS); one row per (lst, scope); applied in {0, 1};
    applied=1 rows: status OK, the screen file beside the table with md5 = screen_md5; applied=0 rows: no screen file or
    md5, a status of 1 to install.SCREEN_STATUS_MAXLEN characters (phase3_install.py:135-166, :185; dewarp_lst.sh:82-89)."""
    SC = _SC()
    PI = SC.pi_module()
    with open(path, newline='') as f:
        rd = csv.DictReader(f)
        cols = list(rd.fieldnames or [])
        rows = list(rd)
    bad = []
    if cols != PI.SCREEN_COLS:
        bad.append(f'columns {cols[:4]}... != the {len(PI.SCREEN_COLS)} phase3_install.SCREEN_COLS')
    keys = [(r['lst'], r['scope']) for r in rows]
    dup = sorted({k for k in keys if keys.count(k) > 1})
    if dup:
        bad.append(f'duplicate rows {dup[:3]}')
    sdir = os.path.dirname(os.path.abspath(path))
    app = [r for r in rows if r.get('applied') == '1']
    sums = SC.md5_many([os.path.join(sdir, r['screen_file']) for r in app if r.get('screen_file')])
    maxlen = SC.cfg().install.SCREEN_STATUS_MAXLEN
    for r in rows:
        k = f"{r.get('lst')}_{r.get('scope')}"
        if r.get('applied') not in ('0', '1'):
            bad.append(f'{k}: applied {r.get("applied")!r}')
        elif r['applied'] == '1':
            p = os.path.join(sdir, r.get('screen_file') or '')
            if r.get('status') != 'OK':
                bad.append(f'{k}: applied=1 with status {str(r.get("status"))[:30]!r}')
            if not r.get('screen_file') or sums.get(p) != r.get('screen_md5'):
                bad.append(f'{k}: screen {r.get("screen_file")!r} md5 {sums.get(p)} != row {r.get("screen_md5")}')
        else:
            if r.get('screen_file') or r.get('screen_md5'):
                bad.append(f'{k}: applied=0 row names a screen')
            if not (r.get('status') or '').strip() or len(r['status']) > maxlen:
                bad.append(f'{k}: applied=0 status empty or longer than {maxlen} characters')
    return not bad, len(rows), (f'{len(rows)} rows ({len(app)} applied, screens md5-verified)' if not bad
                                else f'{len(bad)} problem(s): {_short(bad)}')


@content_test('screen_table', name='install_cell_row')
def install_cell_row(path, install_cell=None, applied_json=None, **ctx):
    """Step e of the install for one cell (context ``install_cell``, ``applied_json``, given by the install at write time and
    kept on the output entry): exactly one row for the cell, applied=1, naming <lst>_<scope>_warp_screen.fits beside the
    table, md5(screen) = the row's screen_md5 = APPLIED_SCREEN.json's, and the row's numbers and status = the JSON's
    (phase3_install.py:148-166). Without an install context (a dewarp applied=0 write, `verify`, `--resume`) the cell part
    does not apply and the test says so; the table's own rules are screen_table.rows (above), which always run."""
    SC = _SC()
    if not install_cell:
        return True, None, 'no install cell in context: the per-cell applied=1 check applies to install writes only'
    lst, scope = install_cell.split('_', 1)
    with open(path, newline='') as f:
        rows = [r for r in csv.DictReader(f) if (r['lst'], r['scope']) == (lst, scope)]
    if len(rows) != 1:
        return False, len(rows), f'{len(rows)} rows for {install_cell} in {os.path.basename(path)} (one required)'
    r = rows[0]
    want_file = SC.cfg().install.SCREEN_FILE_FMT.format(lst=lst, scope=scope)
    sp = os.path.join(os.path.dirname(os.path.abspath(path)), want_file)
    bad = []
    if r['applied'] != '1':
        bad.append(f'applied {r["applied"]}')
    if r['screen_file'] != want_file:
        bad.append(f'screen_file {r["screen_file"]!r} != {want_file}')
    got = SC.md5(sp) if os.path.isfile(sp) else None
    if got != r['screen_md5']:
        bad.append(f'md5({want_file}) {got} != row {r["screen_md5"]}')
    if applied_json:
        if not os.path.exists(applied_json):
            bad.append(f'{applied_json} missing')
        else:
            j = json.load(open(applied_json))
            pairs = (('screen_md5', 'screen_md5'), ('n_anchors', 'n_anchors'), ('vlssr_med_before_arcmin', 'med_before'),
                     ('vlssr_med_after_arcmin', 'med_after'), ('heldout_resid_med_arcsec', 'cv_resid_med_as'),
                     ('anchor_off_med_arcsec', 'anchor_off_med_as'), ('screen_p50_arcsec', 'warp_p50_as'),
                     ('screen_p95_arcsec', 'warp_p95_as'), ('measured_at', 'measured_at'), ('status', 'status'))
            bad += [f'{x} {r.get(x)!r} != JSON {y} {j.get(y)!r}' for x, y in pairs if str(r.get(x)) != str(j.get(y))]
    return not bad, got, (f'{install_cell}: applied=1 row, screen md5 verified' + (' = APPLIED_SCREEN.json' if applied_json
                                                                                  else '') if not bad else '; '.join(bad))


# ---------------------------------------------------------------------------------------------------------------------
# research_request / rerun_epochs / cleanup_record
# ---------------------------------------------------------------------------------------------------------------------

def _epoch_s(utc):
    import calendar
    import time
    return calendar.timegm(time.strptime(utc, '%Y-%m-%dT%H:%M:%SZ'))


@content_test('research_request')
def queued_once(path, **ctx):
    """The enqueue's edits are in force: the cell once in ready.txt and in assign_<S>.csv for each Stokes; no marker,
    policy-skip or park OLDER than the request remains (newer ones are the re-search's own); every retired copy exists;
    the flow-holder map has its row; the request's I master digest equals the installed master's now (a re-install
    makes the request stale)."""
    from exopipe.phase3.lib.masters import epoch_gate as EG
    C = _SC().cfg().install_enqueue
    req = json.load(open(path))
    S, cell = req['state_root'], req['cell']
    t_req = _epoch_s(req['utc'])
    bad = []

    def lines(p):
        try:
            return [l.rstrip('\n') for l in open(p)]
        except FileNotFoundError:
            return []
    nr = lines(os.path.join(S, C.READY_LIST)).count(cell)
    if nr != 1:
        bad.append(f'{C.READY_LIST}: {nr} lines {cell}')
    for st in req['stokes']:
        na = sum(1 for l in lines(os.path.join(S, C.ASSIGN_FMT.format(stokes=st))) if l.startswith(f'{cell},'))
        if na != 1:
            bad.append(f'{C.ASSIGN_FMT.format(stokes=st)}: {na} lines for {cell}')
        for sub, nm in ((C.MARKERS_FMT, cell), (C.MARKERS_FMT, f'{cell}.policy-skip'), (C.PARKED_FMT, cell)):
            p = os.path.join(S, sub.format(stokes=st), nm)
            if os.path.lexists(p) and os.lstat(p).st_mtime < t_req - 1:
                bad.append(f'{p} older than the request (not retired)')
    for a, b in req.get('retired') or []:
        if not os.path.lexists(b):
            bad.append(f'retired copy {b} missing')
    if not any(l.startswith(f'{cell},') for l in lines(os.path.join(S, C.FLOW_HOLDER_MAP))):
        bad.append(f'{C.FLOW_HOLDER_MAP}: no row for {cell}')
    try:
        cur = EG.master_md5(req['i_master'])
    except (OSError, ValueError) as e:
        cur = f'unreadable ({e})'
    if cur != req['i_master_md5']:
        bad.append(f'installed I master changed since the request ({str(cur)[:12]} != {req["i_master_md5"][:12]})')
    return not bad, len(req['stokes']), (f'{cell} queued for {",".join(req["stokes"])} in {S}' if not bad else _short(bad))


@content_test('rerun_epochs')
def items_match_cells(path, **ctx):
    """Every item is <LST>_<date> of its cells' LST; for each cell named in the header, the items naming it are exactly
    the cell's nights to re-run NOW (install_enqueue.RERUN_NIGHTS rule on the installed cell and data_decisions) - an
    exclusion change or a re-install makes the list stale."""
    from exopipe.phase3.lib.install import steps_enqueue as SE
    head, items = SE.read_rerun_file(path)
    cells_ = [c for c in head.get('cells', '').split(',') if c]
    if not cells_:
        return False, 0, 'header names no cell'
    bad = []
    from exopipe.phase3.core import cells
    for it, cs in items.items():
        try:
            ilst, _ = cells.parse_epoch(it)
        except ValueError as e:
            bad.append(str(e))
            continue
        for c in cs:
            if c not in cells_ or not c.startswith(ilst + '_'):
                bad.append(f'{it}: cell {c} not in the header or of another LST')
    for c in cells_:
        lst, scope = c.split('_', 1)
        want = {f'{lst}_{d}' for d in SE.rerun_nights(os.path.join(head['working_root'], c), lst, scope, head['nights'])}
        got = {it for it, cs in items.items() if c in cs}
        if want != got:
            bad.append(f'{c}: {len(got - want)} items not due now, {len(want - got)} missing (e.g. '
                       f'{sorted(got ^ want)[:3]})')
    return not bad, len(items), (f'{len(items)} items for {len(cells_)} cell(s), current' if not bad else _short(bad))


@content_test('cleanup_record')
def removed(path, **ctx):
    """The rebuild scratch the cleanup removed is gone."""
    rec = json.load(open(path))
    left = [p for p in rec.get('removed') or [] if os.path.exists(p)]
    return bool(rec.get('removed')) and not left, len(rec.get('removed') or []), (
        f'{len(rec["removed"])} paths removed' if not left else f'still present: {left}')
