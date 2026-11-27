#!/usr/bin/env python3
"""TEST - the Release 1 header-repair cards written by the producing steps, the install's sibling card-key gate, and the
cleaned Stokes V source identity and refresh (lib/masters/release1_cards.py, lib/masters/vclean_masters.py, the install
wiring in lib/install/). Data-free: synthetic FITS with the full standard header under <TMPDIR>/exopipe3_tests_release1_cards/
<pid>/ (removed at the end); nothing outside it is read or written.

  history    the repair's HISTORY text and its 72-character wrap (equal to the lines production wrote on 2026-11-13)
  master     a master-referenced cell: FLUXREF = ASTREF = the master ALIGNREF on masters and PB maps, MDWSTAT from the table
             on masters only, P2RUN SCALE_A P1RUN CALBP refused with the reason; a CHECKSUM header and a PB map whose
             CONSTDIG is not a master's refused whole (bytes unchanged); the cards then the HISTORY lines at the end; no
             existing card changed; a second pass adds nothing; provenance_nights.csv; a present key refuses a write
  night      a night-referenced cell: the six provenance cards of the reference night from weights.csv, its Phase 2 log and
             the Phase 1 provenance.json files, in the repair's order and comments
  mdwstat    the table / MDWARP agreement, LSWBSB, a cell missing from the table, no MDWSTAT at the masters build
  gate       the sibling card-key gate: PASS; a missing card key, a missing repair HISTORY line, no repaired sibling, an
             MDWSTAT other than the table's, an uncorrected MDWMED1 each REFUSE; an exempt whole refusal; siblings read
             from another root (the install under --out); the table read by the writer's loader under the writer's key (an
             unusable table refuses); excluded names left out
  notdewarp  a not-dewarped cell (R-GREGG-20261127-PR157-AND-18MHZ-SIBLING-CHECK-RECOMMENDATIONS-APPROVED): the dewarp-only
             cards equal the keys the dewarp apply writes (parsed from master_dewarp._apply), less MDWSTAT on a master; PASS
             with the siblings' '_mdewarp' names and those cards waived (11 per master, 12 on the PB map); a missing FLUXREF
             or master MDWSTAT, an MDWARP T file and an '_mdewarp' file name each REFUSE; a card held by every same-named
             not-dewarped sibling stays required; one 'applied' master, a table saying applied or no MDWSTAT at all
             compares as before (no waiver, refused); a dewarped cell compares with '_mdewarp' siblings only, a missing
             dewarp card refusing it
  vclean     vclean_source_check (current, no VCSRC, another VCSRC, another time block); build_vclean_cell refuses a stale
             pair; refresh_vclean_cell (the sv2 clean and the data_decisions gate stubbed): a check-only call on the stale pair
             runs and writes nothing; the refresh re-cleans it - old pair quarantined with md5, new pair stamped, carded,
             archived and audited - with every file map refused and astropy's switch turned on around the call (no FITS read
             of the refresh or its helpers maps a file; FALSIFIER: the same stand-in refuses PB.sibling_check's map with the
             switch on); a check-only call then reports CURRENT; a current pair without its cards is recarded through the
             working copy, quarantine and publish path of a new round, with no sv2 run and the archived uncarded pair moved
             to the archive quarantine unchanged
  source     the MDWSTAT source (lib/install/mdwstat, gate MDW-REC): no dewarp record - the table; a current record - its
             value, noted when it differs from the table (cell and both values) or the table has no row; an accepted cell's
             apply record (no value) - the table, noted; a record not done, unreadable or malformed - refused; currency on
             the masters' md5 (a differing md5, an unstamped master, a stamped master not installed refuse; --omit-vclean's
             omissions allowed); the rows every reader sees (Source.rows: the record's value, MDWMED1 columns empty); step
             b's card source text, value_problems and the sibling gate compare with the record ('!= dewarp record'); the
             record root; pi_context hands the source to phase3_install; PI's check on the staged masters refuses with the
             cell untouched before b's trial and the first move (install and resume); the plan gate (plan.mdw, the install
             record's mdwstat_source) and the installed-cell check of stages vclean and cards against the install record
  install    install.RELEASE1_CARDS / MDWSTAT_TABLE reach phase3_install b through pi_context only for the calls that run b
             (cards=True); the switch decides the default stages and the gates the enqueue requires; b's writes are tried on
             a trial copy of the staging before the first rename (a failure refuses with the staging unchanged and the trial
             removed); the published and audited cell files are one list (coadd.ARCHIVED_CELL_FILES) including
             provenance_nights.csv; the window check accepts the config; the masters build writes the cards (no MDWSTAT)
             before its publish

Run: python tests/test_install_release1_cards.py
"""
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')

import numpy as np                                                  # noqa: E402
from astropy.io import fits                                         # noqa: E402

ROOT = os.path.join(tempfile.gettempdir(), 'exopipe3_tests_release1_cards', str(os.getpid()))
LST, SCOPE, CELL = '12h', 'Red', '12h_Red'
MREF = 'MASTER:ivw-mean-v3:79ec8a01688d36a88bd398bcaab5338e'
DIG = 'e4448294d4b37717e590826cb04534a0'
NIGHTS = ('2026-01-01', '2026-01-02', '2026-01-03')
N = 64
RESULTS = []


def ok(name, cond, detail=''):
    RESULTS.append((name, bool(cond), detail))
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f'  ({str(detail)[:400]})' if detail else ''), flush=True)
    return cond


def md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


def hdr(p):
    return fits.getheader(p, memmap=False)


# ---------------------------------------------------------------------------------------------------------------------
# synthetic products
# ---------------------------------------------------------------------------------------------------------------------

def product(path, stokes, alignref, constdig=DIG, extra=None, data=None, pbcorr=False, cards=None):
    """A FITS file with the full standard header (products.write_standard_header, validate=True) and the master cards."""
    from exopipe.phase3.core.products import write_standard_header
    os.makedirs(os.path.dirname(path), exist_ok=True)
    h = fits.Header()
    for k, v in dict(CTYPE1='RA---SIN', CTYPE2='DEC--SIN', CRVAL1=180.0, CRVAL2=37.0, CRPIX1=N / 2, CRPIX2=N / 2,
                     CDELT1=-0.1, CDELT2=0.1, CUNIT1='deg', CUNIT2='deg', BUNIT='JY/BEAM', EQUINOX=2000.0, RADESYS='FK5',
                     TELESCOP='OVRO_LWA', NCOADD=len(NIGHTS), COADDOP='ivw-mean-v3').items():
        h[k] = v
    for d in NIGHTS:
        h.add_history(f'COADD constituent {d}')
    for k, v in (cards or {}).items():
        h[k] = v
    if data is None:
        rng = np.random.default_rng(abs(hash(os.path.basename(path))) % 2 ** 32)
        data = rng.normal(0, 0.01, (N, N)).astype(np.float32)
    fits.writeto(path, np.asarray(data, dtype=np.float32), h, overwrite=True)
    ex = {'PRODNAME': (os.path.basename(path)[:-5], 'Phase 3 product name')}
    if constdig is not None:
        ex['CONSTDIG'] = (constdig, 'md5 of sorted constituent dates')
    ex.update(extra or {})
    write_standard_header(path, scope=SCOPE, stokes=stokes, start=f'{NIGHTS[0]}T03:00:00', end=f'{NIGHTS[-1]}T04:00:00',
                          inttime_s=10.0, time_basis='coadd-span', beam=(0.2, 0.2, 0.0), thermnoi=0.01, imgrms=0.02,
                          pbcorr=pbcorr, dewarped=True, aligned=True, alignref=alignref, msub=False, ssub=False, extra=ex,
                          validate=True)
    return path


def weights(fd, ref_date=None, sci_root=None):
    rows = []
    for i, d in enumerate(NIGHTS + ('2026-01-04',)):
        sci = f'{sci_root or "/nowhere"}/{LST}/{d}/Science_{d.replace("-", "")}_030000'
        rows.append(dict(date=d, in_master='1' if i < 3 else '0', sci_dir=sci, flux_dc=f'{1.1 + 0.1 * i:.4f}',
                         qa3_scale='1.2', qa3_guard='reference-identity' if d == ref_date else '', qa3_window=''))
    with open(f'{fd}/weights.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def mdw_table(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['cell', 'mdwstat', 'gate_verdict', 'mdwmed1_header_arcmin',
                                          'mdwmed1_installed_cubic_arcmin', 'source'])
        w.writeheader()
        for cell, st, old, new in rows:
            w.writerow(dict(cell=cell, mdwstat=st, gate_verdict='synthetic', mdwmed1_header_arcmin=old,
                            mdwmed1_installed_cubic_arcmin=new, source='test'))
    return path


MASTERS = ('I_deep_Taper_R075', 'I_deep_Taper_R0', 'I_deep_Taper_R075_pbcorr', 'V_deep_Taper_R0', 'V_deep_Taper_R0_clean')


def master_cell(fd, alignref, mdwarp=True):
    for m in MASTERS:
        product(f'{fd}/{m}_{SCOPE}.fits', m[0], alignref, cards={'MDWARP': True} if mdwarp else None,
                pbcorr='_pbcorr' in m)
    product(f'{fd}/PB_deep_{SCOPE}.fits', 'I', alignref)
    product(f'{fd}/PB_deep_R075_{SCOPE}.fits', 'I', alignref, constdig='ffffffffffffffffffffffffffffffff')
    product(f'{fd}/I_deep_NoTaper_R0_{SCOPE}.fits', 'I', alignref, cards={'MDWARP': True})
    with fits.open(f'{fd}/I_deep_NoTaper_R0_{SCOPE}.fits', mode='update', memmap=False) as h:
        h[0].header['CHECKSUM'] = ('0000000000000000', 'HDU checksum')


# ---------------------------------------------------------------------------------------------------------------------

def test_history():
    from exopipe.phase3.lib.masters import release1_cards as R1
    lines = R1.history_lines(R1.history_text(['FLUXREF', 'ASTREF', 'MDWSTAT'], '2026-11-13'))
    ok('history: the repair line of a master-referenced master equals production\'s 2026-11-13 lines',
       lines == ['2026-11-13 Release 1 header repair (R-GREGG-20261107-HEADER-REPAIR-GO):', 'added FLUXREF ASTREF MDWSTAT'],
       lines)
    lines = R1.history_lines(R1.history_text(list(R1.PROV6) + ['MDWSTAT'], '2026-11-13'))
    ok('history: wrapped at spaces, each line at most 72 characters', all(len(x) <= 72 for x in lines) and len(lines) == 2
       and ' '.join(lines) == R1.history_text(list(R1.PROV6) + ['MDWSTAT'], '2026-11-13'), lines)
    ok('group_values: one value, or value:subbands groups in subband order',
       R1.group_values([('41MHz', 'a'), ('46MHz', 'a')]) == 'a'
       and R1.group_values([('41MHz', 'a'), ('46MHz', 'b'), ('50MHz', 'a')]) == 'a:41MHz,50MHz;b:46MHz')
    ok('scale_a: flux_dc, the reference night 1.0, qa3_scale window, not in master refused',
       R1.scale_a({'in_master': '1', 'flux_dc': '1.25'})[0] == 1.25
       and R1.scale_a({'in_master': '1', 'qa3_guard': 'reference-identity', 'flux_dc': '2'})[0] == 1.0
       and R1.scale_a({'in_master': '1', 'flux_dc': 'affine', 'qa3_scale': '3.5'})[0] == 1.0
       and R1.scale_a({'in_master': '1', 'flux_dc': '', 'qa3_scale': '1.4'})[0] == 1.4
       and R1.scale_a({'in_master': '0', 'flux_dc': '1.2'})[0] is None)


def test_master_cell():
    from exopipe.phase3.lib.masters import release1_cards as R1
    fd = f'{ROOT}/master/{CELL}/coadd_final'
    os.makedirs(fd)
    master_cell(fd, MREF)
    weights(fd)
    t = mdw_table(f'{ROOT}/mdw_master.csv', [('12h Red', 'applied', '0.908', ''), ('13h Red', 'applied', '1.0', '')])
    before = {f: hdr(f'{fd}/{f}') for f in os.listdir(fd) if f.endswith('.fits')}
    sums = {f: md5(f'{fd}/{f}') for f in before}
    rep = R1.write_cell(fd, LST, SCOPE, mdw_table=t, provenance=True, today='2026-11-13')
    F = rep['files']
    ok('master: reference = the I Taper R-0.75 master ALIGNREF (no reference row in weights.csv)',
       rep['ref'] == ['master', MREF], rep['ref'])
    ok('master: every plain, pbcorr and cleaned master gets FLUXREF ASTREF MDWSTAT, in that order',
       all(F[f'{m}_{SCOPE}.fits']['added'] == ['FLUXREF', 'ASTREF', 'MDWSTAT'] for m in MASTERS), rep['lines'])
    ok('master: P2RUN SCALE_A P1RUN CALBP refused (no reference night), named with the reason',
       all(sorted(F[f'{m}_{SCOPE}.fits']['refused']) == sorted(['P2RUN', 'SCALE_A', 'P1RUN', 'CALBP'])
           and 'no reference night' in F[f'{m}_{SCOPE}.fits']['refused']['P2RUN'] for m in MASTERS))
    h = hdr(f'{fd}/I_deep_Taper_R075_{SCOPE}.fits')
    ok('master: values and comments - FLUXREF = ASTREF = the ALIGNREF as written (comment truncated by the card length, as '
       'in production), MDWSTAT the table\'s',
       h['FLUXREF'] == MREF and h['ASTREF'] == MREF and h['MDWSTAT'] == 'applied'
       and h.comments['FLUXREF'] and R1.REF_COMMENT['master']['FLUXREF'].startswith(h.comments['FLUXREF'])
       and h.comments['MDWSTAT'] == 'Master dewarp status', (h['FLUXREF'], h.comments['FLUXREF'], h['MDWSTAT']))
    ok('master: the PB map gets FLUXREF ASTREF only (no MDWSTAT)', F[f'PB_deep_{SCOPE}.fits']['added'] == ['FLUXREF', 'ASTREF']
       and 'MDWSTAT' not in hdr(f'{fd}/PB_deep_{SCOPE}.fits'))
    for f, why in ((f'PB_deep_R075_{SCOPE}.fits', 'CONSTDIG'), (f'I_deep_NoTaper_R0_{SCOPE}.fits', 'CHECKSUM')):
        ok(f'master: {f} refused whole ({why}), its bytes unchanged', F[f]['file_refusals'] and not F[f]['added']
           and why in F[f]['file_refusals'][0] and md5(f'{fd}/{f}') == sums[f], F[f]['file_refusals'])
    cards = list(h.cards)
    tail = [(c.keyword, str(c.value)) for c in cards[-5:]]
    ok('master: the added cards then the repair HISTORY lines end the header',
       tail == [('FLUXREF', MREF), ('ASTREF', MREF), ('MDWSTAT', 'applied'),
                ('HISTORY', '2026-11-13 Release 1 header repair (R-GREGG-20261107-HEADER-REPAIR-GO):'),
                ('HISTORY', 'added FLUXREF ASTREF MDWSTAT')], tail)
    b = before[f'I_deep_Taper_R075_{SCOPE}.fits']
    same = [c.image for c in b.cards] == [c.image for c in cards[:len(b.cards)]]
    ok('master: no existing card changed (the old header is the new one\'s prefix)', same)
    sums2 = {f: md5(f'{fd}/{f}') for f in before}
    rep2 = R1.write_cell(fd, LST, SCOPE, mdw_table=t, provenance=True, today='2026-11-14')
    ok('master: a second pass adds nothing and changes no byte', all(not v['added'] for v in rep2['files'].values())
       and sums2 == {f: md5(f'{fd}/{f}') for f in before})
    adds, _ = R1.planned_adds(fd, LST, SCOPE, mdw_table=t)
    ok('master: the dry run finds nothing to add (refused-whole files named as such)',
       all(not a or a[0].startswith('(refused whole') for a in adds.values()), adds)
    rows = list(csv.DictReader(open(f'{fd}/provenance_nights.csv')))
    ok('master: provenance_nights.csv - one row per in-master night, FLUXREF = ASTREF = the master, P2RUN and SCALE_A of '
       'each night, the unsourced cards named', [r['date'] for r in rows] == list(NIGHTS)
       and all(r['FLUXREF'] == r['ASTREF'] == MREF for r in rows) and rows[0]['P2RUN'] == 'Science_20260101_030000'
       and rows[1]['SCALE_A'] == '1.2' and 'P1RUN: no Phase 2 log' in rows[0]['unsourced']
       and list(rows[0]) == ['date'] + list(R1.PROV6) + ['unsourced'], rows[:1])
    try:
        R1.write_cards(f'{fd}/I_deep_Taper_R0_{SCOPE}.fits', [('FLUXREF', 'x', 'y')])
        refused = False
    except RuntimeError as e:
        refused = 'never changed' in str(e)
    ok('master: writing a key the header already holds refuses (an existing card is never changed)', refused)
    return fd, t


def test_night_cell():
    from exopipe.phase3.core import cells
    from exopipe.phase3.lib.masters import release1_cards as R1
    ref = NIGHTS[1]
    fd = f'{ROOT}/night/{CELL}/coadd_final'
    os.makedirs(fd)
    master_cell(fd, ref)
    sci_root = f'{ROOT}/phase2'
    weights(fd, ref_date=ref, sci_root=sci_root)
    sci = f'{sci_root}/{LST}/{ref}/Science_{ref.replace("-", "")}_030000'
    p1 = f'{ROOT}/phase1/{LST}/{ref}'
    mem = cells.band_members(SCOPE)
    os.makedirs(sci)
    run = f'Run_{ref.replace("-", "")}_020000'
    with open(f'{sci}/phase2_{LST}_{ref}_{os.path.basename(sci)}.log', 'w') as f:
        f.write(f'2026-01-02 05:00:00 - [PHASE2] - INFO - Found {len(mem)} run(s) in {p1}:\n')
        for sb in mem:
            f.write(f'2026-01-02 05:00:00 - [PHASE2] - INFO -   {sb}  {run}  (12 FITS files)\n')
        f.write('2026-01-02 05:00:01 - [PHASE2] - INFO - Copying to NVMe\n')
    for sb in mem:
        os.makedirs(f'{p1}/{run}/{sb}')
        json.dump(dict(subband=sb, obs_date=ref, lst_label=LST, bp_table='/cal/bp_20260102.bcal', run_label=run),
                  open(f'{p1}/{run}/{sb}/provenance.json', 'w'))
    t = mdw_table(f'{ROOT}/mdw_night.csv', [('12h Red', 'applied', '', '')])
    rep = R1.write_cell(fd, LST, SCOPE, mdw_table=t, provenance=True, today='2026-11-13')
    f0 = f'I_deep_Taper_R075_{SCOPE}.fits'
    h = hdr(f'{fd}/{f0}')
    ok('night: reference = the reference-identity row, equal to the master ALIGNREF', rep['ref'] == ['night', ref], rep['ref'])
    ok('night: the six provenance cards and MDWSTAT, in the repair\'s order',
       rep['files'][f0]['added'] == list(R1.PROV6) + ['MDWSTAT'], rep['lines'][:2])
    ok('night: values - P2RUN from sci_dir, SCALE_A 1.0 (reference night), the date, P1RUN and CALBP from the logs',
       (h['P2RUN'], h['SCALE_A'], h['FLUXREF'], h['ASTREF'], h['P1RUN'], h['CALBP'])
       == (os.path.basename(sci), 1.0, ref, ref, run, '/cal/bp_20260102.bcal'),
       {k: h.get(k) for k in R1.PROV6})
    ok('night: comments of a co-add (the reference night\'s)', h.comments['P2RUN'] == R1.COADD_COMMENT['P2RUN']
       and h.comments['FLUXREF'] == R1.COMMENTS['FLUXREF'])
    hist = [str(x) for x in h['HISTORY']][-2:]
    ok('night: the HISTORY line names the seven added cards, wrapped', ' '.join(hist) == R1.history_text(
        list(R1.PROV6) + ['MDWSTAT'], '2026-11-13'), hist)
    rows = {r['date']: r for r in csv.DictReader(open(f'{fd}/provenance_nights.csv'))}
    ok('night: provenance_nights.csv - the reference night fully sourced, the others without a log named',
       rows[ref]['CALBP'] == '/cal/bp_20260102.bcal' and rows[ref]['unsourced'] == ''
       and 'no Phase 2 log' in rows[NIGHTS[0]]['unsourced'] and rows[NIGHTS[0]]['FLUXREF'] == ref, rows[ref])


def test_mdwstat_rules():
    from exopipe.phase3.lib.masters import release1_cards as R1
    fd = f'{ROOT}/mdw/{CELL}/coadd_final'
    os.makedirs(fd)
    master_cell(fd, MREF, mdwarp=False)
    product(f'{fd}/I_deep_Taper_R0_{SCOPE}.fits', 'I', MREF, cards={'LSWBSB': 'Red=41MHz+46MHz'})
    weights(fd)
    c = R1.CellCards(fd, LST, SCOPE)

    def mdw(st, cell='12h Red'):
        return R1.load_mdw_table(mdw_table(f'{ROOT}/mdw_rule.csv', [(cell, st, '', '')]))

    def card(f, table, path='t.csv'):
        return {x['card']: x for x in c.plan_file(f, table, path)['cards']}.get('MDWSTAT')
    m = f'I_deep_Taper_R075_{SCOPE}.fits'
    ok('mdwstat: table "applied" but the file has no MDWARP T - refused (they disagree)',
       card(m, mdw('applied'))['status'] == 'REFUSE' and 'disagree' in card(m, mdw('applied'))['source'])
    ok('mdwstat: table "refused: ..." and no MDWARP - written', card(m, mdw('refused: gain 3.1 arcsec < 5'))['status'] == 'ADD')
    ok('mdwstat: a combination of subband products (LSWBSB, no MDWARP) - refused',
       'LSWBSB' in card(f'I_deep_Taper_R0_{SCOPE}.fits', mdw('not attempted'))['source'])
    miss = c.plan_file(m, mdw('applied', cell='13h Red'), 't.csv')['cards']
    ok('mdwstat: a cell missing from the table refuses MDWSTAT only (FLUXREF ASTREF still added)',
       {x['card']: x['status'] for x in miss}.get('MDWSTAT') == 'REFUSE'
       and {x['card']: x['status'] for x in miss}.get('FLUXREF') == 'ADD')
    ok('mdwstat: an unreadable table refuses MDWSTAT', card(m, R1.load_mdw_table(f'{ROOT}/none.csv'))['status'] == 'REFUSE')
    ok('mdwstat: the masters build (no table) plans no MDWSTAT', card(m, None) is None)
    ok('mdwstat: never on a PB map', card(f'PB_deep_{SCOPE}.fits', mdw('not attempted')) is None)


def _archive_copy(fd, dst_dir, lst):
    """The cell's coadd-class files under archive names (<scope>_<name>_LST<lst>_<span>_N3.fits; the gate keys on the name
    without the LST/span token)."""
    os.makedirs(dst_dir, exist_ok=True)
    out = {}
    for f in sorted(os.listdir(fd)):
        if f.endswith('.fits'):
            d = f'{dst_dir}/{SCOPE}_{f[:-5]}_LST{lst}_20260101-20260103_N3.fits'
            shutil.copy2(f'{fd}/{f}', d)
            out[f] = d
    return out


def test_gate(fd, table):
    from exopipe.phase3.lib.masters import release1_cards as R1
    run = f'{ROOT}/gate/Coadd/Run_X'
    sib_run = f'{ROOT}/gate_sib/Coadd/Run_X'
    for lst in ('13h', '14h'):
        _archive_copy(fd, f'{sib_run}/{lst}/{SCOPE}', lst)
    mine = _archive_copy(fd, f'{run}/{LST}/{SCOPE}', LST)
    for f, p in list(mine.items()):          # the files the repair refused whole are not this cell's archive here
        if f.startswith('PB_deep_R075') or f.startswith('I_deep_NoTaper_R0_'):
            os.remove(p)
            del mine[f]
    for lst in ('13h', '14h'):
        for p in os.listdir(f'{sib_run}/{lst}/{SCOPE}'):
            if 'PB_deep_R075' in p or 'I_deep_NoTaper_R0_' in p:
                os.remove(f'{sib_run}/{lst}/{SCOPE}/{p}')
    t = mdw_table(f'{ROOT}/mdw_gate.csv', [('12h Red', 'applied', '0.908', '')])
    passed, lines = R1.sibling_card_gate(CELL, run, t, sibling_run=sib_run)
    ok('gate: PASS - every archived file carries every card key of its repaired siblings (read from another root, as '
       'under --out)', passed and lines[-1].startswith(f'SIBLING-CARD-GATE PASS {CELL}: {len(mine)} of {len(mine)}'), lines[-1])
    passed, lines = R1.sibling_card_gate(CELL, run, t)
    ok('gate: no repaired sibling in the same run (only this LST archived) - REFUSED, fail closed', not passed
       and 'no repaired sibling' in ' '.join(lines), lines[-1])
    victim = mine[f'I_deep_Taper_R0_{SCOPE}.fits']
    keep = victim + '.keep'
    shutil.copy2(victim, keep)

    def edit(fn):
        shutil.copy2(keep, victim)
        with fits.open(victim, mode='update', memmap=False) as h:
            fn(h[0].header)
        return R1.sibling_card_gate(CELL, run, t, sibling_run=sib_run)
    passed, lines = edit(lambda h: h.remove('FLUXREF'))
    ok('gate: one card removed from one archived file - REFUSED, naming the file and the key', not passed
       and any(os.path.basename(victim) in x and 'FLUXREF' in x for x in lines), lines[0])

    def nohist(h):
        hist = [str(x) for x in h['HISTORY'] if R1.REPAIR not in str(x) and 'added FLUXREF' not in str(x)]
        h.remove('HISTORY', remove_all=True)
        for x in hist:
            h.add_history(x)
    passed, lines = edit(nohist)
    ok('gate: the repair HISTORY line removed - REFUSED', not passed and 'no repair HISTORY line' in lines[0], lines[0])
    passed, lines = edit(lambda h: h.set('MDWSTAT', 'refused: stale'))
    ok('gate: MDWSTAT other than the table\'s - REFUSED', not passed and 'MDWSTAT' in lines[0] and '!= table' in lines[0], lines[0])
    t2 = mdw_table(f'{ROOT}/mdw_gate2.csv', [('12h Red', 'applied', '1.747', '1.7600')])
    passed, lines = edit(lambda h: h.set('MDWMED1', 1.747))
    passed2, lines2 = R1.sibling_card_gate(CELL, run, t2, sibling_run=sib_run)
    ok('gate: an MDWMED1 still holding the uncorrected value - REFUSED (R-GREGG-20261111-MDWMED1-EXCEPTION)', not passed2
       and 'uncorrected' in ' '.join(lines2), lines2[0])
    edit(lambda h: h.set('MDWMED1', 1.76))
    passed2, _ = R1.sibling_card_gate(CELL, run, t2, sibling_run=sib_run)
    ok('gate: the corrected MDWMED1 passes', passed2)
    shutil.copy2(keep, victim)
    os.remove(keep)
    odd = f'{run}/{LST}/{SCOPE}/{SCOPE}_I_odd_product_LST12h_20260101-20260103_N3.fits'
    shutil.copy2(mine[f'PB_deep_{SCOPE}.fits'], odd)
    passed, lines = R1.sibling_card_gate(CELL, run, t, sibling_run=sib_run)
    ok('gate: a file no repaired sibling shares a name with - REFUSED', not passed
       and any('no repaired sibling product of this name' in x for x in lines), lines[-1])
    rf = f'{ROOT}/refusals.csv'
    with open(rf, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['path', 'card', 'reason'])
        w.writeheader()
        for c in ('FLUXREF', 'ASTREF'):
            w.writerow(dict(path=odd, card=c, reason='CONSTDIG is not one of the installed central masters'))
    passed, lines = R1.sibling_card_gate(CELL, run, t, sibling_run=sib_run, refusals=[rf])
    ok('gate: the same file refused whole by the repair (--refusals) - EXEMPT, the gate passes', passed
       and any(x.startswith('EXEMPT') for x in lines), lines[-1])
    os.remove(odd)
    passed, lines = R1.sibling_card_gate('12h_Blue', run, t, sibling_run=sib_run)
    ok('gate: no archived FITS - REFUSED', not passed and 'no archived FITS' in lines[-1])
    bad_t = f'{ROOT}/mdw_twice.csv'
    mdw_table(bad_t, [('12h Red', 'applied', '', ''), ('12h Red', 'applied', '', '')])
    passed, lines = R1.sibling_card_gate(CELL, run, bad_t, sibling_run=sib_run)
    ok('gate: the MDWSTAT table read by the writer\'s loader (load_mdw_table) - a table listing a cell twice is unusable and '
       'REFUSES', not passed and 'cannot be used' in lines[-1] and R1.load_mdw_table(bad_t) in lines[-1], lines[-1])
    ok('gate: the writer\'s key - the table row \'12h Red\' is the cell 12h_Red for both (load_mdw_table)',
       '12h_Red' in R1.load_mdw_table(t))
    keep = victim + '.keep'
    shutil.copy2(victim, keep)
    with fits.open(victim, mode='update', memmap=False) as h:
        del h[0].header['FLUXREF']
    passed, lines = R1.sibling_card_gate(CELL, run, t, sibling_run=sib_run)
    passed_x, lines_x = R1.sibling_card_gate(CELL, run, t, sibling_run=sib_run,
                                             exclude=(f'{SCOPE}_I_deep_Taper_R0_{SCOPE}_LST*.fits',))
    ok('gate: exclude leaves the named files out (a card missing there no longer refuses)', not passed and passed_x
       and f'{len(mine) - 1} of {len(mine) - 1} archived files' in lines_x[-1], lines_x[-1])
    shutil.copy2(keep, victim)
    os.remove(keep)


ND_STAT = 'refused: held-out gain 1.0 arcsec < 5'
TOKEN = '_mdewarp'
DEWARP_VALUES = dict(MDWARP=True, MDWMETH='rbf-multiquadric', MDWINTP='cubic-spline', MDWREF='synthetic', MDWNANC=50, MDWCV=3.0,
                     MDWMED0=1.0, MDWMED1=0.9, MDWP50=10.0, MDWP95=20.0, MDWDATE='2026-10-05T00:00:00', MDWSTAT='applied')


def _set_cards(path, cards=None, remove=()):
    with fits.open(path, mode='update', memmap=False) as h:
        for k in remove:
            if k in h[0].header:
                del h[0].header[k]
        for k, v in (cards or {}).items():
            h[0].header[k] = v


def _nd_layout(fd):
    """A band of three LSTs from the repaired master cell: 13h and 14h dewarped as the apply leaves them (every master named
    with '_mdewarp', every file with the apply's cards - MDWSTAT 'applied' on the PB maps too, as the 2026-11-11 writer
    stamps it; PB maps keep their names, as in the archive), 12h not dewarped (MDWSTAT 'refused: ...', no MDWARP, no
    token). Returns (run, {working name: 12h archive path}, {lst: {working name: path}} of the dewarped LSTs, table)."""
    run = f'{ROOT}/gate_nd/Coadd/Run_X'
    cells = {}
    for lst in ('12h', '13h', '14h'):
        cp = _archive_copy(fd, f'{run}/{lst}/{SCOPE}', lst)
        for f in list(cp):
            if f.startswith('PB_deep_R075') or f.startswith('I_deep_NoTaper_R0_'):
                os.remove(cp.pop(f))
            elif f.startswith('PB'):                 # the archive names a PB map as its working name (PB_deep_<scope>.fits)
                os.rename(cp[f], f'{run}/{lst}/{SCOPE}/{f}')
                cp[f] = f'{run}/{lst}/{SCOPE}/{f}'
        for f, p in list(cp.items()):
            if lst == '12h':
                if not f.startswith('PB'):
                    _set_cards(p, {'MDWSTAT': ND_STAT}, remove=('MDWARP',))
                continue
            _set_cards(p, DEWARP_VALUES)
            if not f.startswith('PB'):
                q = os.path.join(os.path.dirname(p), os.path.basename(p).replace('_LST', TOKEN + '_LST', 1))
                os.rename(p, q)
                cp[f] = q
        cells[lst] = cp
    t = mdw_table(f'{ROOT}/mdw_gate_nd.csv', [('12h Red', ND_STAT, '', ''), ('13h Red', 'applied', '', ''),
                                              ('14h Red', 'applied', '', '')])
    return run, cells.pop('12h'), cells, t



def test_gate_not_dewarped(fd):
    """R-GREGG-20261127-PR157-AND-18MHZ-SIBLING-CHECK-RECOMMENDATIONS-APPROVED: a not-dewarped cell is compared with its
    siblings' names without '_mdewarp' and need not carry the dewarp-only cards; every other card must still match."""
    import ast
    from exopipe.phase3.lib.masters import release1_cards as R1
    src_path = os.path.join(REPO, 'exopipe/phase3/lib/dewarp/master_dewarp.py')
    src = open(src_path).read()
    tree = ast.parse(src)
    apply_ = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_apply']
    written = [kw.arg for n in ast.walk(apply_[0]) if isinstance(n, ast.Assign) and any(getattr(t, 'id', '') == 'cards'
               for t in n.targets) and isinstance(n.value, ast.Call) and getattr(n.value.func, 'id', '') == 'dict'
               for kw in n.value.keywords] if apply_ else []
    ok('not-dewarped: DEWARP_APPLY_CARDS are exactly the card keys the dewarp apply writes (master_dewarp._apply cards)',
       tuple(written) == R1.DEWARP_APPLY_CARDS, written)
    out_name = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_out_name']
    ok('not-dewarped: MDEWARP_TOKEN is the apply\'s output-name token (master_dewarp._out_name)', R1.MDEWARP_TOKEN == TOKEN
       and out_name and R1.MDEWARP_TOKEN in ast.get_source_segment(src, out_name[0]))
    m_only, pb_only = R1.dewarp_only_cards('/x/Red_I_deep_Taper_R0_Red.fits'), R1.dewarp_only_cards('/x/PB_deep_Red.fits')
    ok('not-dewarped: dewarp-only cards - the apply\'s 12 on a PB map, 11 on a master (MDWSTAT: the repair writes it on every '
       'master)', pb_only == set(R1.DEWARP_APPLY_CARDS) and m_only == set(R1.DEWARP_APPLY_CARDS) - {'MDWSTAT'}, sorted(m_only))
    run, mine, dew, t = _nd_layout(fd)
    passed, lines = R1.sibling_card_gate(CELL, run, t)
    waived = {x.split(':')[0][len('WAIVED '):]: int(x.split(': ')[1].split()[0]) for x in lines if x.startswith('WAIVED')}
    pb = os.path.basename(mine[f'PB_deep_{SCOPE}.fits'])
    ok('not-dewarped: PASS - names compared without \'_mdewarp\', 11 dewarp-only keys waived per master, 12 on the PB map',
       passed and lines[-1].startswith(f'SIBLING-CARD-GATE PASS {CELL}: {len(mine)} of {len(mine)}') and 'not dewarped' in lines[-1]
       and len(waived) == len(mine) and waived.pop(pb) == 12 and set(waived.values()) == {11},
       [x for x in lines if x.startswith('REFUSED')] or lines[-1])
    ok('not-dewarped: dewarp_state names the cell not dewarped from MDWSTAT, the names and MDWARP',
       R1.dewarp_state([(p, hdr(p)) for p in mine.values()], R1.load_mdw_table(t)['12h_Red'])[0] == 'not dewarped')
    victim = mine[f'I_deep_Taper_R0_{SCOPE}.fits']
    keep = victim + '.keep'
    shutil.copy2(victim, keep)

    def edit(cards=None, remove=(), cell=CELL, table=t):
        shutil.copy2(keep, victim)
        _set_cards(victim, cards, remove)
        return R1.sibling_card_gate(cell, run, table)
    passed, lines = edit(remove=('FLUXREF',))
    ok('not-dewarped: a master missing a card that is not dewarp-only (FLUXREF) - REFUSED, naming the file and the key',
       not passed and any(os.path.basename(victim) in x and 'missing 1 card keys' in x and 'FLUXREF' in x for x in lines), lines[0])
    passed, lines = edit(remove=('MDWSTAT',))
    ok('not-dewarped: a master missing MDWSTAT (written by the repair on every master, so not dewarp-only there) - REFUSED',
       not passed and any(os.path.basename(victim) in x and 'MDWSTAT' in x for x in lines if x.startswith('REFUSED ')), lines[0])
    passed, lines = edit(cards={'MDWARP': True})
    ok('not-dewarped: a file carrying MDWARP T contradicts the status - REFUSED (fail closed)', not passed
       and 'disagree' in lines[-1] and 'MDWARP' in lines[-1], lines[-1])
    passed, lines = edit(cards={'MDWSTAT': 'applied'})
    ok('not-dewarped: one master saying \'applied\' among \'refused\' ones - compared as before, no waiver: REFUSED (its MDWSTAT '
       'is not the table\'s)', not passed and 'not dewarped' not in lines[-1]
       and any(os.path.basename(victim) in x and '!= table' in x for x in lines), lines[-1])
    t_app = mdw_table(f'{ROOT}/mdw_gate_nd_app.csv', [('12h Red', 'applied', '', ''), ('13h Red', 'applied', '', ''),
                                                      ('14h Red', 'applied', '', '')])
    passed, lines = edit(table=t_app)
    ok('not-dewarped: the MDWSTAT table saying applied for the cell - compared as before, no waiver: REFUSED (\'!= table\')',
       not passed and 'not dewarped' not in lines[-1] and any('!= table' in x for x in lines), lines[-1])
    shutil.copy2(keep, victim)
    tok = os.path.join(os.path.dirname(victim), os.path.basename(victim).replace('_LST', TOKEN + '_LST', 1))
    os.rename(victim, tok)
    passed, lines = R1.sibling_card_gate(CELL, run, t)
    ok('not-dewarped: a file named with \'_mdewarp\' contradicts the status - REFUSED (fail closed)', not passed
       and 'disagree' in lines[-1] and TOKEN in lines[-1], lines[-1])
    os.rename(tok, victim)
    for p in mine.values():
        _set_cards(p, remove=('MDWSTAT',))
    passed, lines = R1.sibling_card_gate(CELL, run, t)
    ok('not-dewarped: no MDWSTAT on any file (no status card) - compared as before: no repaired sibling of the plain name, '
       'REFUSED', not passed and 'not dewarped' not in lines[-1]
       and any('no repaired sibling product of this name' in x for x in lines), lines[-1])
    for f, p in mine.items():
        if not f.startswith('PB'):
            _set_cards(p, {'MDWSTAT': ND_STAT})
    shutil.copy2(victim, keep)
    passed, lines = R1.sibling_card_gate('13h_Red', run, t)
    ok('not-dewarped: a dewarped cell (13h) passes, compared with the same-named \'_mdewarp\' siblings, nothing waived',
       passed and 'not dewarped' not in lines[-1] and not any(x.startswith('WAIVED') for x in lines), lines[-1])
    d13 = dew['13h'][f'I_deep_Taper_R0_{SCOPE}.fits']
    shutil.copy2(d13, d13 + '.keep')
    _set_cards(d13, remove=('MDWMETH',))
    passed, lines = R1.sibling_card_gate('13h_Red', run, t)
    ok('not-dewarped: a dewarped master missing a dewarp card (MDWMETH) - REFUSED (no waiver for a dewarped cell)',
       not passed and any(os.path.basename(d13) in x and 'MDWMETH' in x for x in lines), lines[0])
    shutil.copy2(d13 + '.keep', d13)
    os.remove(d13 + '.keep')
    moved = f'{ROOT}/gate_nd_14h'
    shutil.move(f'{run}/14h', moved)
    passed, lines = R1.sibling_card_gate('13h_Red', run, t)
    ok('not-dewarped: a dewarped cell whose only sibling is not dewarped (14h moved away) - REFUSED, no repaired sibling of '
       'its \'_mdewarp\' names (the token is never stripped for a dewarped cell)', not passed
       and sum('no repaired sibling product of this name' in x for x in lines) == len(dew['13h']) - 1, lines[-1])
    shutil.move(moved, f'{run}/14h')
    passed, lines = R1.sibling_card_gate(CELL, run, t)
    ok('not-dewarped: the layout restored passes again', passed, lines[-1])
    d15 = f'{run}/15h/{SCOPE}'                 # a second not-dewarped LST: plain names, one card the dewarped LSTs lack
    os.makedirs(d15)
    for f, p in mine.items():
        shutil.copy2(p, f'{d15}/{os.path.basename(p).replace("_LST12h_", "_LST15h_")}')
    _set_cards(f'{d15}/{os.path.basename(victim).replace("_LST12h_", "_LST15h_")}', {'P1RUN': 'Run_20260101_030000'})
    passed, lines = R1.sibling_card_gate(CELL, run, t)
    ok('not-dewarped: a card every same-named not-dewarped sibling holds (P1RUN, on 15h) but the dewarped ones lack stays '
       'required - same-named siblings stay the reference, only the dewarp-only cards are waived - REFUSED', not passed
       and any(os.path.basename(victim) in x and 'missing 1 card keys held by all 1 repaired siblings: P1RUN' in x
               for x in lines), [x for x in lines if x.startswith('REFUSED')] or lines[-1])
    shutil.rmtree(f'{run}/15h')
    os.remove(keep)


def test_vclean():
    from exopipe.phase3.lib.masters import vclean_masters as VM
    root = f'{ROOT}/vclean/coadd_sb'
    cf = f'{root}/{CELL}/coadd_final'
    os.makedirs(cf)
    vp = product(f'{cf}/V_deep_Taper_R0_{SCOPE}.fits', 'V', MREF, cards={'MDWARP': True})
    ap = f'{cf}/V_deep_Taper_R0_clean_{SCOPE}.fits'
    shutil.copy2(vp, ap)
    ok('vclean: no VCSRC card - not current', VM.vclean_source_check(ap, vp) == (
        False, 'no VCSRC card (cleaned before 2026-11-10: source unknown)'))
    with fits.open(ap, mode='update', memmap=False) as h:
        h[0].header['VCSRC'] = 'f' * 32
    ok('vclean: VCSRC of another V master - not current (STALE)', 'is not the current V master' in VM.vclean_source_check(ap, vp)[1])
    with fits.open(ap, mode='update', memmap=False) as h:
        h[0].header['VCSRC'] = md5(vp)
    good = VM.vclean_source_check(ap, vp)
    ok('vclean: cleaned from the current V master, time block and constituents equal - current', good[0], good[1])
    with fits.open(ap, mode='update', memmap=False) as h:
        h[0].header['NCOADD'] = 2
    ok('vclean: another time block - not current', VM.vclean_source_check(ap, vp)[1].startswith('time block differs'))
    # a stale pair in a complete cell: build_vclean_cell's skip path refuses it; refresh_vclean_cell re-cleans it
    for m in ('I_deep_Taper_R0', 'I_deep_Taper_R075'):
        product(f'{cf}/{m}_{SCOPE}.fits', 'I', MREF, cards={'MDWARP': True})
    yy, xx = np.mgrid[:N, :N]
    pb = np.clip(1 - 0.8 * (np.hypot(xx - N / 2, yy - N / 2) / (N / 2)) ** 2, 0.02, 1).astype(np.float32)
    product(f'{cf}/PB_deep_{SCOPE}.fits', 'I', MREF, data=pb)
    with fits.open(ap, mode='update', memmap=False) as h:
        h[0].header['NCOADD'] = len(NIGHTS)
        h[0].header['VCSRC'] = 'e' * 32
    VM.write_vclean_sibling(cf, SCOPE)
    st = VM.build_vclean_cell(root, CELL, skip_existing=True)[1]
    ok('vclean: build_vclean_cell refuses a stale pair in its skip path', st.startswith('REFUSED stale cleaned-V pair'), st)
    weights(cf)
    t = mdw_table(f'{ROOT}/mdw_vclean.csv', [('12h Red', 'applied', '', '')])
    _refresh(root, cf, vp, ap, t)


class _NoMap:
    """A stand-in for mmap.mmap in this process: a map of a file under ROOT is recorded and refused (PermissionError, as the
    production gate refuses a Lustre file); every other map (anonymous, astropy's own temporary probe) is made."""
    seen = []

    def __enter__(self):
        import mmap as _mm
        real, root, seen = _mm.mmap, os.path.realpath(ROOT), self.seen

        class NoMap(real):
            def __new__(cls, fileno, *a, **kw):
                path = os.path.realpath(f'/proc/self/fd/{fileno}') if fileno != -1 else ''
                if path.startswith(root + os.sep):
                    seen.append(path)
                    raise PermissionError(13, f'test stand-in: no map of {path}')
                return real.__new__(cls, fileno, *a, **kw)
        self.real, _mm.mmap = real, NoMap
        del seen[:]
        return self

    def __exit__(self, *exc):
        import mmap as _mm
        _mm.mmap = self.real
        return False


def _strip_cards(paths):
    """Remove every Release 1 card and the repair HISTORY line (synthetic test files only)."""
    from exopipe.phase3.lib.masters import release1_cards as R1
    for p in paths:
        with fits.open(p, mode='update', memmap=False) as h:
            hd = h[0].header
            for k in ('FLUXREF', 'ASTREF', 'MDWSTAT', 'P2RUN', 'SCALE_A', 'P1RUN', 'CALBP'):
                if k in hd:
                    del hd[k]
            hist = [str(x) for x in hd.get('HISTORY', []) if R1.REPAIR not in str(x) and 'added FLUXREF' not in str(x)]
            hd.remove('HISTORY', ignore_missing=True, remove_all=True)
            for x in hist:
                hd.add_history(x)


def _refresh(root, cf, vp, ap, table):
    """refresh_vclean_cell on the stale pair: the sv2 clean (a subprocess) and the data_decisions gate stubbed."""
    from exopipe.phase3.lib.install import install as PI, steps_common as SC
    from exopipe.phase3.lib.masters import coadd as C, vclean_masters as VM, pb_apply as PB
    out = f'{ROOT}/vclean'
    roots = SC.Roots(out, True)
    mdir = roots.masters_dir(LST, SCOPE)
    os.makedirs(mdir, exist_ok=True)
    old_arch = []
    for nm in ('clean', 'clean_pbcorr'):
        p = f'{mdir}/{SCOPE}_V_deep_Taper_Robust+0.0_vclean_OLD_{nm}.fits'
        shutil.copy2(f'{cf}/V_deep_Taper_R0_{nm}_{SCOPE}.fits', p)
        old_arch.append((p, md5(p)))
    pair = (os.path.basename(ap), f'V_deep_Taper_R0_clean_pbcorr_{SCOPE}.fits')
    old_pair = {f: md5(f'{cf}/{f}') for f in pair}
    calls = {'sv2': 0, 'excl': 0}

    def fake_run(argv, **kw):
        calls['sv2'] += 1
        d = argv[argv.index('--out_dir') + 1]
        with fits.open(argv[argv.index('--v_image') + 1], memmap=False) as h:
            fits.writeto(f'{d}/V_deep_residual_final.fits', (0.5 * h[0].data).astype(np.float32), h[0].header)
        return types.SimpleNamespace(returncode=0, stderr='')

    def fake_excl(final_dir, lst, fl):
        calls['excl'] += 1
    from exopipe.phase3.config import CFG
    real_run, real_excl, real_t = VM.subprocess.run, C._refuse_excluded_constituents, CFG.install.MDWSTAT_TABLE
    VM.subprocess.run, C._refuse_excluded_constituents, CFG.install.MDWSTAT_TABLE = fake_run, fake_excl, table
    tmp = f'{ROOT}/vtmp'
    nomap = _NoMap()
    try:
        with SC.pi_context(PI, roots, 'r1vclean'):
            st0 = VM.refresh_vclean_cell(CELL, root=f'{out}/coadd_sb', tmp_root=tmp, mdw_table=table, check_only=True)[1]
            ok('vclean refresh: a check-only call on the stale pair reports NOT CURRENT, runs no sv2 and writes nothing',
               st0.startswith('NOT CURRENT (') and calls['sv2'] == 0 and not os.path.exists(tmp)
               and not os.path.exists(roots.wq('r1vclean')) and all(md5(f'{cf}/{f}') == m for f, m in old_pair.items())
               and all(md5(p) == m for p, m in old_arch), st0)
            with fits.conf.set_temp('use_memmap', True), nomap:        # the caller's switch on: the refresh holds it off
                st1 = VM.refresh_vclean_cell(CELL, root=f'{out}/coadd_sb', tmp_root=tmp, mdw_table=table)[1]
            refresh_maps = list(nomap.seen)
            st2 = VM.refresh_vclean_cell(CELL, root=f'{out}/coadd_sb', tmp_root=tmp, mdw_table=table, check_only=True)[1]
            with fits.conf.set_temp('use_memmap', True), nomap:
                try:
                    PB.sibling_check(f'{cf}/{pair[1]}', ap, np.ones((N, N)))
                    fals = 'no refusal'
                except PermissionError as e:
                    fals = str(e)
            fals_maps = list(nomap.seen)
    finally:
        VM.subprocess.run, C._refuse_excluded_constituents, CFG.install.MDWSTAT_TABLE = real_run, real_excl, real_t
    ok('vclean refresh: a stale pair is re-cleaned, archived and audited (OK), then a check-only call reports CURRENT',
       st1.startswith('OK (') and st2.startswith('CURRENT (') and calls == {'sv2': 1, 'excl': 1}, f'{st1[:300]} / {st2[:200]}')
    ok('vclean refresh: no FITS file was memory-mapped (every map of a test file refused by a stand-in, with the caller\'s '
       'astropy switch on)', refresh_maps == [], refresh_maps)
    ok('vclean refresh: FALSIFIER - the same stand-in refuses PB.sibling_check\'s map with the switch on outside the refresh',
       fals_maps and 'no map of' in fals, (fals, fals_maps))
    ok('vclean refresh: the new master is cleaned from the installed V master', VM.vclean_source_check(ap, vp)[0])
    wq = f'{out}/working_quarantine_r1vclean/{CELL}/coadd_final'
    ok('vclean refresh: the old pair in the working quarantine of round <round>vclean with its md5',
       all(os.path.exists(f'{wq}/{f}') and md5(f'{wq}/{f}') == m for f, m in old_pair.items()))
    aq = f'{out}/archive/Run_INSTALL_QUARANTINE_r1vclean'
    ok('vclean refresh: the old archived pair in the archive quarantine with its md5',
       all(md5(f'{aq}/{os.path.relpath(p, out + "/archive")}') == m for p, m in old_arch))
    now = sorted(f for f in os.listdir(mdir) if '_vclean_' in f)
    pub = {f: hdr(f'{mdir}/{f}') for f in now}
    ok('vclean refresh: the archive holds exactly the new pair, each equal to its working file and carrying the Release 1 '
       'cards and the repair line', len(now) == 2 and all(h.get('FLUXREF') == MREF and h.get('MDWSTAT') == 'applied'
                                                          and any('Release 1 header repair' in str(x) for x in h['HISTORY'])
                                                          for h in pub.values())
       and sorted(md5(f'{mdir}/{f}') for f in now) == sorted(md5(f'{cf}/{f}') for f in pair), now)
    steps = [r['step'] for r in csv.DictReader(open(roots.record_csv))]
    ok('vclean refresh: the action log holds the quarantine, publish and refresh rows',
       steps.count('vclean-archive-quarantine') == 2 and steps.count('vclean-published') == 2
       and steps.count('vclean-refreshed') == 1 and steps.count('working-quarantine') == 2, steps)

    # a current pair whose cards are missing (working and archive copies equal): recarded through a new round's working copy,
    # quarantine and publish - no sv2 run, never an in-place edit of the archived file
    _strip_cards([f'{cf}/{f}' for f in pair])
    for f in pair:                                    # the archived name is the working file's PRODNAME (vclean_masters dest)
        nm = hdr(f'{cf}/{f}')['PRODNAME']
        shutil.copy2(f'{cf}/{f}', f"{mdir}/{nm if nm.endswith('.fits') else nm + '.fits'}")
    unc_w = {f: md5(f'{cf}/{f}') for f in pair}
    unc_a = {f: md5(f'{mdir}/{f}') for f in now}
    VM.subprocess.run, C._refuse_excluded_constituents, CFG.install.MDWSTAT_TABLE = fake_run, fake_excl, table
    try:
        with SC.pi_context(PI, roots, 'r1vclean2'):
            sc = VM.refresh_vclean_cell(CELL, root=f'{out}/coadd_sb', tmp_root=tmp, mdw_table=table, check_only=True)[1]
            st3 = VM.refresh_vclean_cell(CELL, root=f'{out}/coadd_sb', tmp_root=tmp, mdw_table=table)[1]
            st4 = VM.refresh_vclean_cell(CELL, root=f'{out}/coadd_sb', tmp_root=tmp, mdw_table=table, check_only=True)[1]
    finally:
        VM.subprocess.run, C._refuse_excluded_constituents, CFG.install.MDWSTAT_TABLE = real_run, real_excl, real_t
    ok('vclean recard: a current pair without its cards is NOT CURRENT under the MDWSTAT table',
       sc.startswith('NOT CURRENT (') and 'Release 1 cards still to add' in sc, sc[:300])
    ok('vclean recard: recarded with no sv2 run (OK, recarded), then CURRENT', st3.startswith('OK (recarded, no sv2 run')
       and st4.startswith('CURRENT (') and calls == {'sv2': 1, 'excl': 2}, f'{st3[:300]} / {st4[:200]}')
    wq2 = f'{out}/working_quarantine_r1vclean2/{CELL}/coadd_final'
    aq2 = f'{out}/archive/Run_INSTALL_QUARANTINE_r1vclean2'
    ok('vclean recard: the uncarded working pair in the new round\'s working quarantine and the uncarded archived pair in its '
       'archive quarantine, bytes unchanged (md5)', all(md5(f'{wq2}/{f}') == m for f, m in unc_w.items())
       and all(md5(f'{aq2}/{os.path.relpath(mdir, out + "/archive")}/{f}') == m for f, m in unc_a.items()))
    pub2 = {f: hdr(f'{mdir}/{f}') for f in now}
    ok('vclean recard: the archive holds the carded pair, equal to the working files, data unchanged',
       sorted(f for f in os.listdir(mdir) if '_vclean_' in f) == now
       and all(h.get('FLUXREF') == MREF and h.get('MDWSTAT') == 'applied' for h in pub2.values())
       and sorted(md5(f'{mdir}/{f}') for f in now) == sorted(md5(f'{cf}/{f}') for f in pair)
       and all(np.array_equal(fits.getdata(f'{cf}/{f}', memmap=False), fits.getdata(f'{wq2}/{f}', memmap=False),
                              equal_nan=True) for f in pair))


class _Gates:
    """A stand-in for steps_common's gate collector in collect mode (a dry run): records (gate, passed, detail) and returns
    the verdict."""
    def __init__(self):
        self.seen = []

    def check(self, gate, ok, detail='', value=None):
        self.seen.append((gate, bool(ok), str(detail)))
        return bool(ok)


def _mdwstat(files, stats=True):
    """meta.mdwstat's files as the dewarp writes them: {path, before, md5, size, mtime_ns} (size and mtime of the file now;
    ``stats`` False: a record made before 2026-11-25, without them)."""
    out = []
    for p, m in files.items():
        x = dict(path=p, before=None, md5=m)
        if stats and os.path.exists(p):
            st = os.stat(p)
            x.update(size=st.st_size, mtime_ns=st.st_mtime_ns)
        out.append(x)
    return out


def _dewarp_record(root, cell, value, files, status='done', install=None, stats=True, last_done=None, reason='synthetic'):
    """A dewarp unit record as the apply or the record stage writes it (meta.mdwstat), or one without (value None);
    ``last_done``: the meta.last_done an attempt carries (lib/dewarp/steps_cell.keep_last_done)."""
    from exopipe.phase3.core import record
    meta = dict(install=install or dict(kind='rebuild', gen=f'{root}/gen/{cell}'))
    if value is not None:
        meta['mdwstat'] = dict(value=value, files=_mdwstat(files, stats))
    if last_done is not None:
        meta['last_done'] = last_done
    p = record.record_path(root, 'dewarp', cell)
    record.atomic_write_json(p, dict(unit=cell, step='dewarp', status=status, meta=meta, reason=reason,
                                     started='2026-11-25T03:00:00Z'))
    return p


def test_mdwstat_source(fd):
    """The install's MDWSTAT source: the dewarp record first, the table only without one (lib/install/mdwstat)."""
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core.record import Refusal
    from exopipe.phase3.lib.install import install as PI, mdwstat as MW, steps_common as SC, steps_deep as SD
    from exopipe.phase3.lib.masters import release1_cards as R1
    root = f'{ROOT}/src_records'
    t = mdw_table(f'{ROOT}/mdw_src.csv', [('12h Red', 'applied', '1.747', '1.7600'), ('13h Red', 'not attempted', '', '')])
    rows = R1.load_mdw_table(t)
    gen = f'{ROOT}/src_gen/{CELL}'
    os.makedirs(f'{gen}/coadd_final')
    names = [f'{m}_{SCOPE}.fits' for m in MASTERS]
    for n in names:
        shutil.copy2(f'{fd}/{n}', f'{gen}/coadd_final/{n}')
    shutil.copy2(f'{fd}/PB_deep_{SCOPE}.fits', f'{gen}/coadd_final/PB_deep_{SCOPE}.fits')
    sums = {n: md5(f'{gen}/coadd_final/{n}') for n in names}
    files = {f'{gen}/coadd_final/{n}': m for n, m in sums.items()}

    def refusal(fn):
        try:
            fn()
            return 'no refusal'
        except Refusal as e:
            return str(e)
    ch = MW.choose(CELL, rows, t, root)
    ok('source: no dewarp record - the table\'s value, noted', ch.source == MW.TABLE and ch.value == 'applied'
       and 'no dewarp record' in ch.notes[0] and MW.problems(ch, {}) == [], ch.notes)
    ok('source: all cells of the table without a record take the table\'s value (13h_Red too)',
       all(MW.choose(c, rows, t, root).value == rows[c]['mdwstat'] for c in rows))
    rv = 'refused: held-out gain 2.0 arcsec < 5'
    _dewarp_record(root, CELL, rv, files)
    ch = MW.choose(CELL, rows, t, root)
    ok('source: a dewarp record - its value wins over the table\'s, noted with the cell and both values',
       ch.source == MW.RECORD and ch.value == rv and ch.table_value == 'applied' and CELL in ch.notes[0]
       and repr(rv) in ch.notes[0] and "('applied')" in ch.notes[0] and 'the record wins' in ch.notes[0], ch.notes)
    _dewarp_record(root, '13h_Red', 'not attempted', {f'{gen}/coadd_final/{names[0]}': sums[names[0]]})
    _dewarp_record(root, '14h_Red', rv, {f'{gen}/coadd_final/{names[0]}': sums[names[0]]})
    ok('source: a record equal to the table - noted as equal; a cell without a table row - the record, noted',
       'equal to the table' in MW.choose('13h_Red', rows, t, root).notes[0]
       and MW.choose('14h_Red', rows, t, root).value == rv and 'has no row' in MW.choose('14h_Red', rows, t, root).notes[0])
    ok('source: current - every master taken has the record\'s md5 -> no problem', MW.problems(ch, dict(sums)) == [])
    bad = dict(sums, **{names[1]: 'f' * 32})
    pr = MW.problems(ch, bad)
    ok('source: a master whose md5 differs from the record\'s - not current, naming the master and both md5',
       len(pr) == 1 and names[1] in pr[0] and 'f' * 32 in pr[0] and sums[names[1]] in pr[0], pr)
    pr = MW.problems(ch, dict(sums, **{'I_deep_NoTaper_R0_Red.fits': 'a' * 32}))
    ok('source: a master the record did not stamp - not current', len(pr) == 1 and 'not a master the record stamped' in pr[0], pr)
    less = {n: m for n, m in sums.items() if n != names[0]}
    pr = MW.problems(ch, less)
    ok('source: a stamped master not installed - not current; left out by --omit-vclean - allowed', len(pr) == 1
       and 'not installed' in pr[0] and MW.problems(ch, {n: m for n, m in sums.items() if '_clean' not in n},
                                                    MW.omitted_masters(ch.files)) == [], pr)
    ok('source: no master taken - not current', MW.problems(ch, {}) != [])
    _dewarp_record(root, '15h_Red', None, {}, install=dict(kind='mdewarp', gen=f'{root}/x'))
    w15 = refusal(lambda: MW.choose('15h_Red', rows, t, root))
    ok('source: a done dewarp record that records no MDWSTAT (an apply record made before 2026-11-25) - REFUSED, never '
       'the table (one rule for accepted and not-accepted cells)', w15.startswith('MDW-REC') and 'records no MDWSTAT' in w15
       and "'mdewarp'" in w15, w15)
    _dewarp_record(root, '15h_Blue', 'applied', files, install=dict(kind='mdewarp', gen=gen))
    c15 = MW.choose('15h_Blue', rows, t, root)
    ok('source: an accepted cell\'s apply record (meta.mdwstat \'applied\', md5 of the new masters) - the record, as for a '
       'not-accepted cell', c15.source == MW.RECORD and c15.value == 'applied' and c15.files == sums
       and set(c15.stats) == set(sums), c15.notes)
    # a rerun that did not finish keeps the last done attempt (meta.last_done, lib/dewarp/steps_cell.keep_last_done)
    last = dict(run_id='dewarp_A', finished='2026-11-25T01:05:00Z', status='done',
                mdwstat=dict(value=rv, files=_mdwstat(files)), install=dict(kind='rebuild', gen=gen))
    _dewarp_record(root, '21h_Red', None, {}, status='refused', last_done=last,
                   reason='G-A3: the generation already holds files')
    c21 = MW.choose('21h_Red', rows, t, root)
    ok('source: a rerun refused (G-A3) after a done attempt - the last done record is used (meta.last_done), the refused '
       'attempt noted; current while the masters keep its md5', c21.source == MW.RECORD and c21.value == rv
       and c21.entry == 'meta.last_done.mdwstat' and "'refused'" in c21.notes[0] and 'G-A3' in c21.notes[0]
       and 'dewarp_A' in c21.notes[0] and MW.problems(c21, dict(sums)) == [], c21.notes)
    _dewarp_record(root, '22h_Red', 'refused: another value', files, status='error', last_done=last, reason='interrupted')
    c22 = MW.choose('22h_Red', rows, t, root)
    pr22 = MW.problems(c22, dict(sums, **{names[0]: 'e' * 32}))
    ok('source: an interrupted stamp (status error, its own half-written meta.mdwstat ignored) - the last done record; a '
       'master the attempt rewrote (md5 changed) makes it STALE, naming the master', c22.source == MW.RECORD
       and c22.value == rv and "'error'" in c22.notes[0] and len(pr22) == 1 and names[0] in pr22[0], (c22.notes, pr22))
    _dewarp_record(root, '23h_Red', None, {}, status='done', last_done=last)
    c23 = MW.choose('23h_Red', rows, t, root)
    ok('source: a done rerun without an MDWSTAT of its own (verify only) - the last done record, noted', c23.value == rv
       and c23.entry == 'meta.last_done.mdwstat' and 'no MDWSTAT of its own' in c23.notes[0], c23.notes)
    _dewarp_record(root, '16h_Red', None, {}, status='refused')
    _dewarp_record(root, '17h_Red', rv, files, status='failed')
    os.makedirs(os.path.dirname(MW.record_file('18h_Red', root)), exist_ok=True)
    open(MW.record_file('18h_Red', root), 'w').write('{not json')
    _dewarp_record(root, '19h_Red', rv, {f'{gen}/coadd_final/{names[0]}': 'not-an-md5'})
    _dewarp_record(root, '20h_Red', '', files)
    why = {c: refusal(lambda c=c: MW.choose(c, rows, t, root)) for c in ('16h_Red', '17h_Red', '18h_Red', '19h_Red', '20h_Red')}
    ok('source: a record whose dewarp did not finish, a not-done record with a value (no earlier done attempt), an '
       'unreadable record, a bad md5, an empty value - each REFUSED (MDW-REC), never the table in its place',
       all(w.startswith('MDW-REC') for w in why.values()) and 'did not finish' in why['16h_Red']
       and "'failed'" in why['17h_Red'] and 'did not finish' in why['17h_Red'] and 'cannot be read' in why['18h_Red']
       and 'malformed' in why['19h_Red'] and 'malformed' in why['20h_Red'], why)
    # the table is read once per Source, and only where needed
    calls = []
    real_load = R1.load_mdw_table

    def counted(path):
        calls.append(path)
        return real_load(path)
    t9 = mdw_table(f'{ROOT}/mdw_once.csv', [('12h Blue', 'applied', '', '')])
    R1.load_mdw_table = counted
    try:
        ch9, s9 = MW.source_for('12h_Blue', t9, SC.Roots(root, True))
        r9a = s9.rows()
        mdw_table(t9, [('12h Blue', 'changed', '', '')])
        r9b = s9.rows()
    finally:
        R1.load_mdw_table = real_load
    ok('source: one Source reads the table once - the choice, then every rows() of the install see the same rows (the '
       'table rewritten in between)', len(calls) == 1 and ch9.value == 'applied' and r9a['12h_Blue']['mdwstat'] == 'applied'
       and r9b['12h_Blue']['mdwstat'] == 'applied' and R1.load_mdw_table(t9)['12h_Blue']['mdwstat'] == 'changed', calls)
    badt = f'{ROOT}/mdw_bad.csv'
    with open(badt, 'w') as f:
        f.write('cell,mdwstat\n12h Red,applied\n12h Red,applied\n12h Blue,applied\n')
    chb = MW.choose(CELL, R1.load_mdw_table(badt), badt, root)
    srcb = MW.Source(badt, [chb])
    wb = refusal(lambda: MW.choose('12h_Blue', R1.load_mdw_table(badt), badt, root))
    ok('source: a malformed table (a cell listed twice) - the record-sourced cell keeps the record\'s value (noted, not '
       'compared) and its row; a table-sourced cell REFUSES (MDW-REC), naming the table',
       chb.source == MW.RECORD and chb.value == rv and 'not compared with the table' in chb.notes[-1]
       and isinstance(R1.load_mdw_table(badt), str) and srcb.rows() == {CELL: MW.record_row(chb, None)}
       and srcb.rows()[CELL]['mdwstat'] == rv and wb.startswith('MDW-REC') and 'cannot be used' in wb
       and isinstance(MW.Source(badt).rows(), str), (chb.notes, wb))
    src = MW.Source(t, [ch])
    srows = src.rows()
    ok('source: the rows every reader sees - the record cell\'s row holds the record\'s value, names the record, MDWMED1 '
       'columns empty; the other rows are the table\'s', srows[CELL]['mdwstat'] == rv
       and srows[CELL]['mdwstat_source'] == MW.RECORD and srows[CELL]['mdwstat_from'] == ch.record
       and srows[CELL]['mdwmed1_header_arcmin'] == '' and srows['13h_Red'] == rows['13h_Red']
       and R1.mdw_rows(t) == rows and R1.mdw_rows(None) is None and R1.mdw_rows(src) == srows)
    h = fits.Header()
    h['MDWSTAT'] = 'applied'
    h['MDWMED1'] = 1.747
    vp_t, vp_r = R1.value_problems(h, rows[CELL]), R1.value_problems(h, srows[CELL])
    h['MDWSTAT'] = rv
    ok('source: value_problems - the table\'s value refused against the record row (\'!= dewarp record\'), the record\'s '
       'value passes; MDWMED1 not checked from a record row', vp_t and 'uncorrected' in vp_t[0]
       and len(vp_r) == 1 and '!= dewarp record' in vp_r[0] and R1.value_problems(h, srows[CELL]) == [], (vp_t, vp_r))
    cf = f'{ROOT}/src_cards/{CELL}/coadd_final'
    os.makedirs(cf)
    master_cell(cf, MREF, mdwarp=False)
    weights(cf)
    p = R1.CellCards(cf, LST, SCOPE).plan_file(f'I_deep_Taper_R075_{SCOPE}.fits', srows, src)
    card = {x['card']: x for x in p['cards']}['MDWSTAT']
    ok('source: step b\'s MDWSTAT card from the source - the record\'s value, the source text naming the dewarp record',
       card['status'] == 'ADD' and card['value'] == rv and ch.record in card['source'], card)
    ok('source: the source prints as the table and the record cell', str(src).startswith(t) and ch.record in str(src))
    # the sibling gate compares with the source the install chose
    run, sib = f'{ROOT}/src_gate/Coadd/Run_X', f'{ROOT}/src_gate_sib/Coadd/Run_X'
    rep_ = R1.write_cell(cf, LST, SCOPE, mdw_table=src, provenance=True)
    for lst in ('13h', '14h'):
        _archive_copy(cf, f'{sib}/{lst}/{SCOPE}', lst)
    _archive_copy(cf, f'{run}/{LST}/{SCOPE}', LST)
    for d in (f'{run}/{LST}/{SCOPE}', f'{sib}/13h/{SCOPE}', f'{sib}/14h/{SCOPE}'):
        for f in os.listdir(d):
            if 'PB_deep_R075' in f or 'I_deep_NoTaper_R0_' in f:
                os.remove(f'{d}/{f}')
    g_r = R1.sibling_card_gate(CELL, run, src, sibling_run=sib)
    g_t = R1.sibling_card_gate(CELL, run, t, sibling_run=sib)
    ok('source: write_cell with the source writes the record\'s value; the sibling gate with the same source PASSES '
       '(\'agree with the dewarp record\'), with the table alone it REFUSES (\'!= table\') - never a mix',
       all(hdr(f'{cf}/{n}').get('MDWSTAT') == rv for n in names) and g_r[0]
       and 'agree with the dewarp record' in g_r[1][-1] and not g_t[0] and any('!= table' in x for x in g_t[1]),
       (rep_['lines'][:2], g_r[1][-1:], g_t[1][:1]))
    # the record root
    old = CFG.install.DEWARP_RECORD_ROOT, CFG.install.MDWSTAT_TABLE, CFG.install.RELEASE1_CARDS
    try:
        CFG.install.DEWARP_RECORD_ROOT = None
        ok('source: the record root - the install\'s --out under --out, else dewarp.GEN_ROOT; install.DEWARP_RECORD_ROOT '
           'overrides', MW.record_root(SC.Roots(f'{ROOT}/o', True)) == os.path.abspath(f'{ROOT}/o')
           and MW.record_root(SC.Roots(f'{ROOT}/o', False)) == CFG.dewarp.GEN_ROOT and MW.record_root() == CFG.dewarp.GEN_ROOT)
        CFG.install.DEWARP_RECORD_ROOT = root
        CFG.install.MDWSTAT_TABLE = t
        ok('source: install.DEWARP_RECORD_ROOT set - read there', MW.record_root(SC.Roots(f'{ROOT}/o', True)) == root)
        roots = SC.Roots(f'{ROOT}/src_wiring', True)
        with SC.pi_context(PI, roots, 'r1', cards=True, mdw=src) as settings:
            inside = PI.RELEASE1_MDW_TABLE
        ok('source: pi_context hands the source to phase3_install b (and records it as text), and restores',
           inside is src and settings['globals']['RELEASE1_MDW_TABLE'] == str(src) and PI.RELEASE1_MDW_TABLE is None)
        CFG.install.RELEASE1_CARDS = False
        with SC.pi_context(PI, roots, 'r1', cards=True, mdw=src):
            inside = PI.RELEASE1_MDW_TABLE
        ok('source: RELEASE1_CARDS False - no cards at b, the source ignored', inside is None)
        CFG.install.RELEASE1_CARDS = True
        # PI's check on the staged masters
        staged = {f'coadd_final/{n}': (f'{ROOT}/stg/coadd_final/{n}', sums[n], f'{gen}/coadd_final/{n}') for n in names}
        staged['coadd_final/PB_deep_Red.fits'] = ('x', 'b' * 32, f'{gen}/coadd_final/PB_deep_{SCOPE}.fits')

        def pi_check(st, mdw, omit=False):
            prev = PI.RELEASE1_MDW_TABLE
            PI.RELEASE1_MDW_TABLE = mdw
            try:
                PI._mdw_record_check(CELL, gen, st, omit)
                return 'no refusal'
            except RuntimeError as e:
                return str(e)
            finally:
                PI.RELEASE1_MDW_TABLE = prev
        stale = dict(staged)
        stale[f'coadd_final/{names[2]}'] = staged[f'coadd_final/{names[2]}'][:1] + ('c' * 32,) + staged[f'coadd_final/{names[2]}'][2:]
        w = pi_check(stale, src)
        ok('source: PI on the staged masters - current passes; a staged md5 other than the record\'s REFUSES with the cell '
           'untouched (gate MDW-REC); a table path checks nothing', pi_check(staged, src) == 'no refusal'
           and 'REFUSING (cell untouched)' in w and SC.pi_gate_of(w) == 'MDW-REC' and names[2] in w
           and pi_check(stale, t) == 'no refusal', w[:300])
        import inspect
        si, sr = inspect.getsource(PI.install), inspect.getsource(PI.resume)
        ok('source: PI checks the staged masters inside the trial\'s try (staging removed on a refusal), before b\'s trial and '
           'the first move; resume (R1) before its trial', si.index('_mdw_record_check(') < si.index('_release1_trial(')
           < si.index('_move_to_wq(') and sr.index('_mdw_record_check(') < sr.index('_release1_trial(')
           and si.rindex('try:', 0, si.index('_mdw_record_check(')) > si.index('if RELEASE1_MDW_TABLE:'))
        # the plan gate
        u = types.SimpleNamespace(meta={}, notes=[], inputs=[], note=lambda x: u.notes.append(x),
                                  add_input=lambda x, extra=None: u.inputs.append(x))
        plan = types.SimpleNamespace(cell=CELL, roots=roots, gen=gen, new=[f'coadd_final/{n}' for n in names]
                                     + [f'coadd_final/PB_deep_{SCOPE}.fits', 'subtracted/x.fits'], omit_vclean=False, mdw=None)
        gates = _Gates()
        real_md5, real_many = SC.md5, SC.md5_many

        def no_read(*a, **k):
            raise AssertionError('the plan read a master\'s bytes')
        SC.md5 = SC.md5_many = no_read
        try:
            SD.mdw_gate(u, plan, gates)
        finally:
            SC.md5, SC.md5_many = real_md5, real_many
        ok('source: plan gate MDW-REC - a current record passes with the cell and both values, by size and mtime alone (no '
           'master read; \'md5 currency checked at staging\'), plan.mdw the source, the install record\'s mdwstat_source '
           'names it', gates.seen and gates.seen[0][:2] == ('MDW-REC', True)
           and f"{CELL}: record {rv!r} != table 'applied'" in gates.seen[0][2]
           and 'md5 currency checked at staging' in gates.seen[0][2] and plan.mdw is not None
           and plan.mdw.choices[CELL].value == rv and u.meta['mdwstat_source']['source'] == MW.RECORD
           and u.meta['mdwstat_source']['files'] == sums and ch.record in u.inputs, gates.seen)
        open(f'{gen}/coadd_final/{names[3]}', 'ab').write(b' ' * 2880)
        plan.mdw, gates = None, _Gates()
        SD.mdw_gate(u, plan, gates)
        ok('source: plan gate MDW-REC - a master changed after the record stamped it (size): STALE, refused, plan.mdw unset',
           gates.seen[0][:2] == ('MDW-REC', False) and 'STALE' in gates.seen[0][2] and names[3] in gates.seen[0][2]
           and plan.mdw is None, gates.seen)
        shutil.copy2(f'{fd}/{names[3]}', f'{gen}/coadd_final/{names[3]}')
        st4 = os.stat(f'{gen}/coadd_final/{names[4]}')
        os.utime(f'{gen}/coadd_final/{names[4]}', ns=(st4.st_atime_ns, st4.st_mtime_ns + 10**9))
        plan.mdw, gates = None, _Gates()
        SD.mdw_gate(u, plan, gates)
        ok('source: plan gate MDW-REC - a master rewritten since the stamp with the same size (mtime only, as an '
           'interrupted rerun of the same value leaves it): passes the plan, left to the md5 at staging (noted)',
           gates.seen[0][:2] == ('MDW-REC', True) and f'1 left to the md5: {names[4]} rewritten since the stamp'
           in gates.seen[0][2] and plan.mdw is not None, gates.seen)
        os.utime(f'{gen}/coadd_final/{names[4]}', ns=(st4.st_atime_ns, st4.st_mtime_ns))
        _dewarp_record(root, CELL, rv, files, stats=False)
        plan.mdw, gates = None, _Gates()
        SD.mdw_gate(u, plan, gates)
        ok('source: plan gate MDW-REC - a record without size and mtime (made before 2026-11-25): passes, noted, the md5 at '
           'staging deciding', gates.seen[0][:2] == ('MDW-REC', True) and f'{len(names)} left to the md5'
           in gates.seen[0][2] and 'no size and mtime in the record' in gates.seen[0][2] and plan.mdw is not None,
           gates.seen)
        _dewarp_record(root, CELL, rv, files)
        CFG.install.MDWSTAT_TABLE = f'{ROOT}/absent_table.csv'
        plan.cell, plan.mdw, gates = '12h_Blue', None, _Gates()
        SD.mdw_gate(u, plan, gates)
        g_abs_t = gates.seen
        plan.cell, plan.mdw, gates = CELL, None, _Gates()
        SD.mdw_gate(u, plan, gates)
        CFG.install.MDWSTAT_TABLE = t
        ok('source: plan gate MDW-REC - a missing table: an MDW-REC failure of a table-sourced cell inside the gate (the '
           'dry run lists it, no abort); a record-sourced cell passes on its record', g_abs_t[0][:2] == ('MDW-REC', False)
           and 'cannot be used' in g_abs_t[0][2] and gates.seen[0][:2] == ('MDW-REC', True) and plan.mdw is not None
           and plan.mdw.choices[CELL].value == rv, (g_abs_t, gates.seen))
        plan.cell, plan.mdw, gates = '12h_Blue', None, _Gates()
        SD.mdw_gate(u, plan, gates)
        ok('source: plan gate MDW-REC - no record: the table (no row: the cards stage refuses MDWSTAT itself)',
           gates.seen[0][:2] == ('MDW-REC', True) and plan.mdw is not None and not plan.mdw.choices, gates.seen)
        plan.cell, plan.mdw, gates = '16h_Red', None, _Gates()
        SD.mdw_gate(u, plan, gates)
        ok('source: plan gate MDW-REC - an unfinished dewarp: refused', gates.seen[0][:2] == ('MDW-REC', False)
           and plan.mdw is None, gates.seen)
        # the plan's MDW-REC failed: stages vclean and cards refuse with its result, never choosing again
        u6 = types.SimpleNamespace(tests=[dict(name='install.MDW-REC', passed=False, detail='synthetic: STALE for gen')])
        g6 = _Gates()
        real_ims = SD.installed_mdw_source
        SD.installed_mdw_source = no_read
        try:
            r6 = SD.post_install_source(u6, CELL, roots, g6, types.SimpleNamespace(mdw=None), ['verify', 'vclean', 'cards'])
        finally:
            SD.installed_mdw_source = real_ims
        ok('source: the plan\'s MDW-REC failed - stages vclean and cards refuse with that result (MDW-REC), never chosen '
           'again against an earlier install record', r6 is False and len(g6.seen) == 1 and g6.seen[0][:2] == ('MDW-REC', False)
           and 'synthetic: STALE for gen' in g6.seen[0][2], g6.seen)
        # the installed cell (stages vclean and cards without install)
        inst = dict(status='done', meta=dict(gen=gen, mdwstat_source=dict(ch.summary(), omitted=[])),
                    inputs={f'{gen}/coadd_final/{n}': dict(md5=sums[n]) for n in names})
        legacy = dict(inst, meta=dict(gen=gen))
        cdir_ = f'{ROOT}/src_inst/{CELL}'                 # an installed cell: the masters carry the record's value
        os.makedirs(f'{cdir_}/coadd_final')
        for n in names + [f'PB_deep_{SCOPE}.fits']:
            shutil.copy2(f'{gen}/coadd_final/{n}', f'{cdir_}/coadd_final/{n}')
            if not n.startswith('PB_'):
                with fits.open(f'{cdir_}/coadd_final/{n}', mode='update', memmap=False) as h_:
                    h_[0].header['MDWSTAT'] = rv
        ct = MW.Choice(CELL, MW.TABLE, 'applied', table_value='applied')
        ct_rv = MW.Choice(CELL, MW.TABLE, rv, table_value=rv)
        lp = MW.installed_problems(ch, legacy)
        hp = MW.installed_problems(ct, legacy, cdir_)
        ok('source: a legacy install record (no mdwstat_source: made before 2026-11-25 or with RELEASE1_CARDS off) means the '
           'table - a record source now refuses (reinstall the cell); a table source passes when the installed masters\' '
           'headers carry its value, and a header that differs refuses, naming both values (PB maps not compared)',
           any('holds no MDWSTAT source' in x and 'reinstall the cell' in x for x in lp)
           and MW.installed_problems(ct_rv, legacy, cdir_) == [] and len(hp) == len(names)
           and all('reinstall the cell' in x and repr(rv) in x and "'applied'" in x for x in hp)
           and MW.installed_problems(ch, inst, cdir_) == [], (lp, hp[:1]))
        ok('source: installed cell - current for the masters the install record installed; the source changed since the '
           'install refuses (reinstall); no install record refuses a record source; a table source with no record passes',
           MW.installed_problems(ch, inst) == []
           and any('reinstall the cell' in x for x in MW.installed_problems(
               ch, dict(inst, meta=dict(inst['meta'], mdwstat_source=dict(source=MW.TABLE, value='applied')))))
           and MW.installed_problems(ch, None) != [] and MW.installed_problems(MW.choose('12h_Blue', rows, t, root), None) == [])
        from exopipe.phase3.core import record
        rr = SC.Roots(root, True)
        record.atomic_write_json(record.record_path(rr.out, 'install', CELL), dict(unit=CELL, finished='2026-11-25T02:00:00Z',
                                                                                   **inst))
        gates = _Gates()
        got = SD.installed_mdw_source(u, CELL, rr, gates)
        inst['inputs'][f'{gen}/coadd_final/{names[0]}'] = dict(md5='d' * 32)
        record.atomic_write_json(record.record_path(rr.out, 'install', CELL), dict(unit=CELL, finished='2026-11-25T02:00:00Z',
                                                                                   **inst))
        g2 = _Gates()
        got2 = SD.installed_mdw_source(u, CELL, rr, g2)
        ok('source: stages vclean / cards - the installed cell\'s source checked against its install record (MDW-REC): '
           'current passes and returns the source; a master installed with another md5 refuses', got is not None
           and got.choices[CELL].value == rv and gates.seen[0][:2] == ('MDW-REC', True) and got2 is None
           and g2.seen[0][:2] == ('MDW-REC', False) and names[0] in g2.seen[0][2], (gates.seen, g2.seen))
    finally:
        CFG.install.DEWARP_RECORD_ROOT, CFG.install.MDWSTAT_TABLE, CFG.install.RELEASE1_CARDS = old
    ok('source: MDW-REC is a gate of the install, never waivable', 'MDW-REC' in SC.GATES and 'MDW-REC' in SC.NEVER_WAIVABLE)


def _tree(d):
    return {os.path.relpath(os.path.join(a, f), d): md5(os.path.join(a, f)) for a, _s, fs in os.walk(d) for f in fs}


def _trial(PI):
    """install b's Release 1 writes tried on a copy of the staging before the first rename (install._release1_trial)."""
    from exopipe.phase3.lib.install import steps_common as SC
    cdir = f'{ROOT}/trial/coadd_sb/{CELL}'
    stg = f'{cdir}/.install_staging'
    master_cell(f'{stg}/coadd_final', MREF)
    wdir = f'{cdir}/coadd_final'
    os.makedirs(wdir)
    weights(wdir)
    t = mdw_table(f'{ROOT}/mdw_trial.csv', [('12h Red', 'applied', '', '')])
    before = _tree(cdir)
    old = PI.RELEASE1_MDW_TABLE, os.environ.get('P3_ARCHIVE_ROOT')
    PI.RELEASE1_MDW_TABLE, os.environ['P3_ARCHIVE_ROOT'] = t, f'{ROOT}/trial/archive'
    try:
        lines = PI._release1_trial(CELL, cdir, stg, wdir, cdir, LST, SCOPE, 'Run_X')
        ok('trial: b\'s cards and provenance_nights.csv pass on a trial copy of the staging; the trial is removed and the '
           'staging and the working cell are byte-identical to before', any('added FLUXREF ASTREF MDWSTAT' in x for x in lines)
           and any(x.startswith('provenance_nights.csv:') and 'constituent nights' in x for x in lines) and not os.path.exists(f'{cdir}/{PI.RELEASE1_TRIAL_DIRNAME}') and _tree(cdir) == before,
           lines)
        with open(f'{stg}/coadd_final/I_deep_Taper_R0_{SCOPE}.fits', 'wb') as f:     # a staged master b cannot read
            f.write(b'NOT A FITS HEADER ' * 160)
        before = _tree(cdir)
        try:
            PI._release1_trial(CELL, cdir, stg, wdir, cdir, LST, SCOPE, 'Run_X')
            why = 'no refusal'
        except RuntimeError as e:
            why = str(e)
        ok('trial: a staged master b cannot read - REFUSED before any move (gate b), the trial removed, staging and cell '
           'unchanged', 'REFUSING (cell untouched)' in why and SC.pi_gate_of(why) == 'b'
           and not os.path.exists(f'{cdir}/{PI.RELEASE1_TRIAL_DIRNAME}') and _tree(cdir) == before, why[:300])
    finally:
        PI.RELEASE1_MDW_TABLE = old[0]
        if old[1] is None:
            os.environ.pop('P3_ARCHIVE_ROOT', None)
        else:
            os.environ['P3_ARCHIVE_ROOT'] = old[1]
    import inspect
    src_i, src_r = inspect.getsource(PI.install), inspect.getsource(PI.resume)
    ok('trial: install() tries b before the gates-passed row and the first move; resume (R1) before _finish',
       src_i.index('_release1_trial(') < src_i.index("rec(cell, 'gates-passed'") < src_i.index('_move_to_wq(')
       and src_r.index('_release1_trial(') < src_r.index('return _finish('))


def _rounds(SC, SD, Refusal):
    """The install round from the newest done install or recover record; the cleaned-V round unique per attempt."""
    from exopipe.phase3.core import record
    roots = SC.Roots(f'{ROOT}/rounds', True)
    args = types.SimpleNamespace(round=None)
    try:
        SD.install_round(CELL, args, roots)
        r0 = 'no refusal'
    except Refusal as e:
        r0 = str(e)
    for unit, rnd, t in ((CELL, '20261120b', '2026-11-20T01:00:00Z'), (f'{CELL}.recover', '20261121a', '2026-11-21T01:00:00Z')):
        record.atomic_write_json(record.record_path(roots.out, 'install', unit), dict(
            unit=unit, status='done', started=t, finished=t, meta=dict(round=rnd)))
    ok('rounds: no done record refuses; the newest done install OR recover record names the round (here the recover\'s)',
       'no done install or recover record' in r0 and SD.install_round(CELL, args, roots) == '20261121a'
       and SD.install_round(CELL, types.SimpleNamespace(round='20261122z'), roots) == '20261122z', r0)
    rel = os.path.relpath(roots.masters_dir(LST, SCOPE), roots.archive)
    r1 = SD.vclean_round(CELL, roots, '20261121a')
    os.makedirs(os.path.join(roots.wq(r1), CELL))
    r2 = SD.vclean_round(CELL, roots, '20261121a')
    os.makedirs(os.path.join(roots.aq(r2), rel))
    r3 = SD.vclean_round(CELL, roots, '20261121a')
    os.makedirs(os.path.join(roots.wq(r3), '13h_Red'))         # another cell's quarantine does not take the round
    ok('rounds: the cleaned-V round is the first free of <round>vclean, <round>vclean2, ... (a working or an archive '
       'quarantine of this cell takes it)', (r1, r2, r3, SD.vclean_round(CELL, roots, '20261121a'))
       == ('20261121avclean', '20261121avclean2', '20261121avclean3', '20261121avclean3'), (r1, r2, r3))


def test_install_wiring():
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core.record import Refusal
    from exopipe.phase3.lib.install import install as PI, steps_common as SC, steps_deep as SD, steps_enqueue as SE
    from exopipe.phase3.lib.masters import release1_cards as R1
    from exopipe.phase3.steps import install as S
    import inspect
    roots = SC.Roots(f'{ROOT}/wiring', True)
    t = mdw_table(f'{ROOT}/mdw_wiring.csv', [('12h Red', 'applied', '', '')])
    old = CFG.install.MDWSTAT_TABLE, CFG.install.RELEASE1_CARDS
    try:
        CFG.install.MDWSTAT_TABLE = t
        with SC.pi_context(PI, roots, 'r1', cards=True):
            inside = PI.RELEASE1_MDW_TABLE
        ok('install: install.RELEASE1_CARDS (ruled True) sets phase3_install b\'s table for a call that runs b (cards=True), '
           'and restores it', CFG.install.RELEASE1_CARDS is True and inside == t and PI.RELEASE1_MDW_TABLE is None)
        CFG.install.MDWSTAT_TABLE = f'{ROOT}/absent.csv'
        with SC.pi_context(PI, roots, 'r1'):
            inside = PI.RELEASE1_MDW_TABLE
        ok('install: a call that does not run b loads no MDWSTAT table (an unreadable one does not refuse it)', inside is None)
        CFG.install.MDWSTAT_TABLE = t
        src_sd = inspect.getsource(SD)
        ok('install: only install and recover --execute (the calls that run b) ask pi_context for the table',
           src_sd.count('cards=True') == 1 and src_sd.index('cards=True') > src_sd.index('def install_cell')
           and src_sd.count('cards=execute') == 1 and src_sd.index('cards=execute') > src_sd.index('def recover_cell')
           and src_sd.count('cards=') == 2)
        CFG.install.RELEASE1_CARDS = False
        with SC.pi_context(PI, roots, 'r1', cards=True):
            inside = PI.RELEASE1_MDW_TABLE
        ok('install: RELEASE1_CARDS False (the parity file\'s production) - no cards at b', inside is None)
        ok('install: RELEASE1_CARDS False - default stages plan,install,verify; the enqueue requires install.V only',
           S.default_stages() == ('plan', 'install', 'verify') and SE.verify_gates() == ('install.V',))
        gates = _Gates()
        SD.cards_cell(None, CELL, roots, types.SimpleNamespace(dry_run=False), gates, lambda *_: None)
        ok('install: RELEASE1_CARDS False - stage cards refuses (CARDS), reading nothing', gates.seen
           and gates.seen[0][:2] == ('CARDS', False) and 'RELEASE1_CARDS is False' in gates.seen[0][2], gates.seen)
        CFG.install.RELEASE1_CARDS = True
        ok('install: RELEASE1_CARDS True - default stages add vclean,cards; the enqueue requires V VCLEAN CARDS CARD-GATE',
           S.default_stages() == S.DEFAULT_STAGES == ('plan', 'install', 'verify', 'vclean', 'cards')
           and SE.verify_gates() == SE.VERIFY_GATES == ('install.V', 'install.VCLEAN', 'install.CARDS', 'install.CARD-GATE'))
        CFG.install.MDWSTAT_TABLE = f'{ROOT}/absent.csv'
        try:
            SC.release1_mdw_table()
            refused = False
        except Refusal:
            refused = True
        ok('install: RELEASE1_CARDS True with no readable MDWSTAT table refuses', refused)
    finally:
        CFG.install.MDWSTAT_TABLE, CFG.install.RELEASE1_CARDS = old
    _trial(PI)
    _rounds(SC, SD, Refusal)
    ok('install: ruled - RELEASE1_CARDS names R-GREGG-20261107-HEADER-REPAIR-GO, the parity file sets it False',
       'R-GREGG-20261107-HEADER-REPAIR-GO' in CFG.ruled('install.RELEASE1_CARDS')
       and '\ninstall.RELEASE1_CARDS = False\n' in open(os.path.join(REPO, 'exopipe', 'phase3', 'parity',
                                                                      'production_20261013.cfg')).read())
    src_b = inspect.getsource(PI._finish)
    ok('install: b writes the cards and provenance_nights.csv after the stamp and the exclusion gate, before the archive '
       'quarantine', src_b.index('_refuse_excluded_constituents') < src_b.index('R1.write_cell')
       < src_b.index('# c: archive quarantine'))
    from exopipe.phase3.lib.masters import coadd as C
    from exopipe.phase3.lib.install import tests as IT
    ok('install: the publish and both audits read one list of archived cell files (coadd.ARCHIVED_CELL_FILES), which holds '
       'provenance_nights.csv', R1.PROVENANCE_TABLE in C.ARCHIVED_CELL_FILES
       and all('ARCHIVED_CELL_FILES' in inspect.getsource(f) for f in (C._archive_publish_cell, PI._archive_audit, IT))
       and not hasattr(CFG.install, 'AUDIT_CELL_FILES'))
    src_m = inspect.getsource(C.run_final_lst_freq)
    i_w = src_m.index('_R1.write_cell(final_dir')
    ok('masters: the build writes the cards (no MDWSTAT) and provenance_nights.csv under HEADER_STANDARD full, after the '
       'exclusion gate and before the publish, and refuses the cell when the table is not written',
       src_m.rindex('_refuse_excluded_constituents', 0, i_w) < i_w < src_m.index('_archive_publish_cell(', i_w)
       and "HEADER_STANDARD == 'full'" in src_m[i_w - 900:i_w] and 'mdw_table=None' in src_m[i_w:i_w + 120]
       and 'REFUSING to archive' in src_m[i_w:i_w + 600])
    try:
        SC.check_window_consistency(PI)
        okw = True
    except Refusal as e:
        okw = str(e)
    ok('install: the window check accepts the config', okw is True, okw)
    ok('install: stages - default plan,install,verify,vclean,cards; cards implies vclean implies verify',
       S.DEFAULT_STAGES == ('plan', 'install', 'verify', 'vclean', 'cards')
       and S.parse_stages('cards') == ['verify', 'vclean', 'cards'] and S.parse_stages('vclean') == ['verify', 'vclean']
       and S.parse_stages('install,cards') == ['plan', 'install', 'verify', 'vclean', 'cards'])
    ok('install: the new gates are never waivable', all(g in SC.NEVER_WAIVABLE for g in ('VCLEAN', 'CARDS', 'CARD-GATE')))


def main():
    if os.path.exists(ROOT):
        shutil.rmtree(ROOT)
    os.makedirs(ROOT)
    t0 = time.time()
    try:
        test_history()
        fd, table = test_master_cell()
        test_night_cell()
        test_mdwstat_rules()
        test_gate(fd, table)
        test_gate_not_dewarped(fd)
        test_mdwstat_source(fd)
        test_vclean()
        test_install_wiring()
    finally:
        shutil.rmtree(ROOT, ignore_errors=True)
        try:
            os.rmdir(os.path.dirname(ROOT))
        except OSError:
            pass
    bad = [r for r in RESULTS if not r[1]]
    print(f'\n{len(RESULTS) - len(bad)} of {len(RESULTS)} checks passed in {time.time() - t0:.0f} s')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
