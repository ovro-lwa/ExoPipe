"""Release 1 header-repair cards of a deep cell's co-adds, written by the producing step, and the install's sibling card-key gate.

WHY. Release 1's header repair (exopipe_ops/header_repair_release1.py, R-GREGG-20261107-HEADER-REPAIR-GO "Yes on the repair.") added
cards to the archived products. A rebuilt master lacks them (FINDING-20261113-DEEP-LOSTSKY-REBUILT-MASTERS-LACK-REPAIR-CARDS), so
production's rebuild chain re-runs the repair after every install and then a refusing gate (membership_rebuild_20261007/
install_cell.sh:35-54, sibling_card_gate.py; exopipe_ops 2421f36, efc3608; DECISION-20261113-DEEP-REBUILD-CHAIN-WRITES-REPAIR-CARDS,
IMPL-20261113-DEEP-INSTALL-CELL-HEADER-REPAIR-STEP-AND-SIBLING-CARD-GATE). The port (CUTOVER_READINESS.md "INSTALL-CELL HEADER
REPAIR + SIBLING CARD GATE"): the producing steps write the cards themselves - the masters build (header standard 'full'), the
install (step b, before the archive publish) and the cleaned Stokes V refresh - with the repair's rules, and the install keeps the
sibling card-key gate as a refusing gate. Cards are written into the working file BEFORE it is published, so the archive copy is
written once and equals its working source (the install's archive audit compares md5).

SCOPE = the repair's class 'coadd' of the central run (header_repair_release1.py:1094-1113): the cell's masters (every coadd_final
file coadd._MASTER_RE names: plain, pbcorr, cleaned Stokes V) and its two PB maps (PB_deep_<s>, PB_deep_R075_<s>), plus the cell's
provenance_nights.csv. Not here (as in install_cell.sh, which runs --classes coadd only): the deep class (epoch images, search
frames, leave-one-out masters - the re-search writes them), the hybrid run, the mosaics, and the MDWMED1 update of
R-GREGG-20261111-MDWMED1-EXCEPTION (a one-time correction of 94 archived cells; the gate below still refuses an uncorrected value).

RULES (header_repair_release1.py file:line; every card is ADDED - a keyword already in the header is never written again, :740-745)
  reference     CellCards.__init__ (:578-591): the weights.csv row with qa3_guard 'reference-identity' (or qa3_window 'reference')
                must equal the ALIGNREF of the cell's Stokes I Taper Robust-0.75 master -> ('night', date); no such row and
                ALIGNREF 'MASTER:...' -> ('master', ALIGNREF as written; R-GREGG-20261110-HEADER-REPAIR-SIX-DECISIONS decision 1,
                --ref-policy master-alignref); anything else -> no reference (all six provenance cards refused).
  link          :1104-1110, :916-934: a file whose CONSTDIG is not one of the cell's masters' is refused whole; a file without
                ALIGNREF or CONSTDIG, or whose ALIGNREF is not the cell reference, gets none of the six provenance cards.
  PROV6         prov_coadd (:916-962): master-referenced cell -> FLUXREF = ASTREF = the master's ALIGNREF (comments
                REF_COMMENT['master']); P2RUN SCALE_A P1RUN CALBP refused (no reference night). Night-referenced cell -> P2RUN
                SCALE_A P1RUN CALBP of the reference night (comments COADD_COMMENT), FLUXREF = ASTREF = that date.
  P2RUN         night (:636-641): basename of the weights.csv sci_dir, a Science_YYYYMMDD_HHMMSS run of /<LST>/<date>/.
  SCALE_A       scale_a (:524-546): in_master != 1 refused; reference-identity 1.0; numeric flux_dc; else qa3_scale if
                0.3 < qa3_scale < 3.0 else 1.0.
  P1RUN, CALBP  phase2_log (:449-482), phase1_bp (:485-511), group_values (:514-521): the Phase 2 log's subband run lines and
                Phase 1 directory; bp_table of {p1dir}/{run}/{subband}/provenance.json (subband, obs_date, lst_label must match).
  MDWSTAT       mdw_cards (:830-863; R-GREGG-20261111-MDWSTAT-CARD "Approved, add the MDWSTAT card"): masters only (not PB maps),
                from the deep chat's table (install.MDWSTAT_TABLE) when the row agrees with the file's MDWARP ('applied' <=>
                MDWARP T); a file with LSWBSB and no MDWARP refused; a cell missing from the table refuses MDWSTAT only.
  HISTORY       policy_files (:1282-1290): '<UTC date> Release 1 header repair (R-GREGG-20261107-HEADER-REPAIR-GO): added <keys>',
                wrapped at 72 characters (history_images :361-374), after the added cards at the end of the header.
  CHECKSUM      file_head: a header carrying CHECKSUM or DATASUM is refused whole (an added card would invalidate it).
  POLICY        partial (decision 3): every card with a source is written; the others are returned with their reasons.
  provenance    :1117-1150 (decision 4): provenance_nights.csv beside weights.csv - date, the six cards, unsourced; one row per
                in_master 1 night.

GATE  sibling_card_gate (sibling_card_gate.py, exopipe_ops 2421f36 + efc3608 value checks + 9a5343a --refusals): every archived FITS
      of Coadd/<run>/<LST>/<band>/ carries every card key that ALL repaired same-named products of the band's other LSTs carry, plus
      the repair HISTORY line; no repaired sibling of that name refuses (fail closed); MDWSTAT must equal the table row; an MDWMED1
      still holding the table's uncorrected value refuses.

No import-time side effects; headers are read with memmap=False (R-GREGG-20261117-NO-MMAP-OF-LUSTRE-FILES-READ-OR-WRITE).
"""
import csv
import glob
import io
import json
import os
import re
import time
import warnings
from collections import OrderedDict, defaultdict

from astropy.io import fits

RULING = 'R-GREGG-20261107-HEADER-REPAIR-GO'                  # header_repair_release1.py:194
RULING_MDWSTAT = 'R-GREGG-20261111-MDWSTAT-CARD'              # :236
RULING_MDWMED1 = 'R-GREGG-20261111-MDWMED1-EXCEPTION'         # :237; sibling_card_gate.py:45
REPAIR = 'Release 1 header repair'                            # sibling_card_gate.py:30 (the HISTORY text, :1284)
PROV6 = ('P2RUN', 'SCALE_A', 'FLUXREF', 'ASTREF', 'P1RUN', 'CALBP')        # :213
COMMENTS = {'P2RUN': 'Phase 2 run label', 'SCALE_A': 'Mean flux scale applied (target/reference)',          # :214-219
            'FLUXREF': 'Flux scale reference date', 'ASTREF': 'Astrometric reference date',
            'P1RUN': 'Phase 1 run label', 'CALBP': 'Bandpass calibration table', 'MDWSTAT': 'Master dewarp status'}
COADD_COMMENT = {'P2RUN': 'Phase 2 run of the reference night', 'SCALE_A': 'Flux scale applied to the reference night',   # :220-221
                 'P1RUN': 'Phase 1 run(s) of the reference night', 'CALBP': 'Bandpass table(s) of the reference night'}
REF_COMMENT = {'night': {'FLUXREF': 'Flux scale reference date', 'ASTREF': 'Astrometric reference date'},          # :238-241
               'master': {'FLUXREF': 'Flux scale reference (previous-generation master)',
                          'ASTREF': 'Astrometric reference (previous-generation master)'}}
MDWMED1_TOL = 0.0005                                          # arcmin; :235, sibling_card_gate.py:28
PROVENANCE_TABLE = 'provenance_nights.csv'                    # :1124 (decision 4)
SPAN = re.compile(r'_LST\d\dh_\d{8}-\d{8}_N\d+')              # sibling_card_gate.py:29
SKIP = {'HISTORY', 'COMMENT', ''}                             # sibling_card_gate.py:31
END = 'END' + ' ' * 77
_P2_RUN_LINE = re.compile(r' - \[PHASE2\] - INFO -\s+(\d+MHz)\s+(Run_\d{8}_\d{6})\s+\(\d+ FITS files\)\s*$')   # :460
_P2_DIR_LINE = re.compile(r' - \[PHASE2\] - INFO - Found \d+ run\(s\) in (\S+):\s*$')                          # :461


def _header(path):
    return fits.getheader(path, 0, memmap=False)


# ---------------------------------------------------------------------------------------------------------------------
# card images and the HISTORY line
# ---------------------------------------------------------------------------------------------------------------------

def card_image(key, value, comment):
    """The card's image (a long string value takes CONTINUE cards; a long comment is shortened, never the value); refused
    when it does not parse back to the value (header_repair_release1.py:345-355)."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', fits.verify.VerifyWarning)
        img = fits.Card(key, value, comment).image
    if len(img) % 80:
        raise RuntimeError(f'{key}: card image of {len(img)} bytes')
    back = fits.Header.fromstring(img + END, sep='')
    if key not in back or back[key] != value:
        raise RuntimeError(f'{key}: card image does not parse back to {value!r}')
    return img


def history_lines(text):
    """HISTORY values of at most 72 characters each, wrapped at spaces (header_repair_release1.py:358-371)."""
    lines, cur = [], ''
    for w in text.split():
        if len(w) > 72:
            raise RuntimeError('HISTORY word longer than 72 characters')
        if cur and len(cur) + 1 + len(w) > 72:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + ' ' + w) if cur else w
    if cur:
        lines.append(cur)
    return lines


def history_text(keys, today=None):
    """The repair's HISTORY text for the added KEYS (policy_files, header_repair_release1.py:1284)."""
    today = today or time.strftime('%Y-%m-%d', time.gmtime())
    return f'{today} {REPAIR} ({RULING}): added ' + ' '.join(keys)


# ---------------------------------------------------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------------------------------------------------

def log_head_lines(b, nmax, stop):
    """The lines of b as b.decode(errors='replace').splitlines() gives them, up to the first containing STOP or the first
    NMAX (header_repair_release1.py:432-445)."""
    n = 1 << 16
    while True:
        whole = n >= len(b)
        lines = b[:n].decode(errors='replace').splitlines()
        done = lines if whole else lines[:-1]
        cut = next((i for i, ln in enumerate(done) if stop in ln), None)
        if whole or cut is not None or len(done) >= nmax:
            return done[:(cut + 1) if cut is not None else nmax]
        n *= 2


def phase2_log(sci_dir, lst, date):
    """({subband: [runs]}, Phase 1 night directory, log path) from the Phase 2 run's log, or an error string
    (header_repair_release1.py:449-482)."""
    name = os.path.basename(sci_dir.rstrip('/'))
    lg = f'{sci_dir}/phase2_{lst}_{date}_{name}.log'
    if not os.path.exists(lg):
        return f'no Phase 2 log {lg}'
    runs, p1dir = defaultdict(list), None
    with open(lg, 'rb') as f:
        lb = f.read()
    for i, line in enumerate(log_head_lines(lb, 402, 'Copying to NVMe')):
        m = _P2_RUN_LINE.search(line)
        if m:
            runs[m.group(1)].append(m.group(2))
        m = _P2_DIR_LINE.search(line)
        if m and p1dir is None:
            p1dir = m.group(1)
        if 'Copying to NVMe' in line or i > 400:
            break
    if not runs:
        return f'no "Discovered subbands" lines in {lg}'
    if p1dir is None:
        return f'no "Found N run(s) in <phase1 dir>" line in {lg}'
    return dict(runs), p1dir, lg


def phase1_bp(p1dir, run, sb, lst, date):
    """(bp_table, source) or (None, reason) from {p1dir}/{run}/{sb}/provenance.json (header_repair_release1.py:485-511)."""
    pj = f'{p1dir}/{run}/{sb}/provenance.json'
    try:
        with open(pj, 'rb') as f:
            d = json.loads(f.read().decode())
    except FileNotFoundError:
        return None, f'the Phase 1 run directory {p1dir}/{run}/{sb} has no provenance.json (no bandpass record)'
    except Exception as ex:
        return None, f'{pj} unreadable ({str(ex)[:60]})'
    bad = [k for k, v in (('subband', sb), ('obs_date', date), ('lst_label', lst)) if str(d.get(k, '')) != v]
    bp = d.get('bp_table')
    if bad:
        return None, f'{pj}: {",".join(bad)} do not match this subband / night'
    if not isinstance(bp, str) or not bp.strip():
        return None, f'{pj}: no bp_table'
    note = '' if str(d.get('run_label', '')) == run else f' (its run_label {d.get("run_label")})'
    return bp.strip(), pj + note


def group_values(pairs):
    """[(subband, value)] -> 'value' when one distinct value, else 'v1:sb,sb;v2:sb' in subband order (:514-521)."""
    order = OrderedDict()
    for sb, v in pairs:
        order.setdefault(v, []).append(sb)
    if len(order) == 1:
        return next(iter(order))
    return ';'.join(f'{v}:{",".join(sbs)}' for v, sbs in order.items())


def scale_a(row):
    """(value, source) or (None, reason): the factor the generation applied to the night, read as
    coadd.rebuild_variants_from_record reads it back (header_repair_release1.py:524-546)."""
    if str(row.get('in_master', '')).strip() != '1':
        return None, (f"weights.csv in_master {row.get('in_master')!r}: the night is not a constituent - no factor was applied "
                      'to any master')
    if (row.get('qa3_guard') or '') == 'reference-identity':
        return 1.0, 'weights.csv qa3_guard reference-identity (the reference night: no flow, no scale)'
    fdc = (row.get('flux_dc') or '').strip()
    q = (row.get('qa3_scale') or '').strip()
    try:
        if fdc not in ('', 'affine'):
            return float(fdc), 'weights.csv flux_dc (the factor applied, recorded since 2026-10-02)'
        qv = float(q)
        why = 'flux_dc affine' if fdc == 'affine' else 'no flux_dc (record before 2026-10-02)'
        return (qv if 0.3 < qv < 3.0 else 1.0), (f'weights.csv qa3_scale, window rule 0.3 < s < 3.0 else 1.0 ({why}; '
                                                 'coadd_full_pipeline.rebuild_variants_from_record)')
    except ValueError:
        return None, f'weights.csv flux_dc {fdc!r} / qa3_scale {q!r} not numeric'


def load_mdw_table(path):
    """{'LSTh_band': row} of the MDWSTAT table, or an error string when it cannot be used (header_repair_release1.py:803-817)."""
    try:
        with open(path, 'rb') as f:
            b = f.read()
        rows = list(csv.DictReader(io.StringIO(b.decode('utf-8'), newline='')))
        by = {r['cell'].strip().replace(' ', '_'): r for r in rows}
        if len(by) != len(rows):
            raise RuntimeError('a cell is listed twice')
        return by
    except Exception as ex:
        return f'the MDWSTAT table {path} cannot be used ({str(ex)[:60]})'


# ---------------------------------------------------------------------------------------------------------------------
# one cell
# ---------------------------------------------------------------------------------------------------------------------

class CellCards:
    """One deep cell's coadd_final directory: weights.csv, the reference, the masters' link digests and the per-night sources
    (header_repair_release1.py class Cell :548-703, on the working names). ``name_dir`` is the directory the messages name
    for weights.csv and the masters (the archive cell when the files are published there), as the repair names its cell."""

    def __init__(self, final_dir, lst, scope, name_dir=None):
        from exopipe.phase3.lib.masters.coadd import _MASTER_RE
        self.fd, self.lst, self.scope = final_dir, lst, scope
        self.cdir = name_dir or final_dir
        self.err = None
        self.weights = OrderedDict()
        self.ref = None
        self.mI = self.mV = None
        self.dig = {}
        self.hdrs = {}
        self.masters, self.pbmaps = [], []
        self.nprov = {}
        self._p2 = {}
        found = []
        for f in sorted(os.listdir(final_dir)):
            m = _MASTER_RE.match(f) if f.endswith('.fits') else None
            if m:
                self.masters.append(f)
            elif f in (f'PB_deep_{scope}.fits', f'PB_deep_R075_{scope}.fits'):
                self.pbmaps.append(f)
            else:
                continue
            found.append((f, m))
        wp = f'{final_dir}/weights.csv'
        if not os.path.exists(wp):          # as the repair's Cell: no headers read, no digests (every CONSTDIG then refuses)
            self.err = f'no {self.cdir}/weights.csv'
            self.ref = ('error', self.err)
            return
        with open(wp, newline='') as f:
            for r in csv.DictReader(f):
                self.weights[r['date']] = r
        for f, m in found:
            h = _header(f'{final_dir}/{f}')
            self.hdrs[f] = h
            if not m:
                continue
            if h.get('CONSTDIG'):
                self.dig[f] = str(h['CONSTDIG'])
            g = m.groupdict()
            plain = g['taper'] == 'Taper' and not g['clean'] and not g['pb']
            if plain and g['st'] == 'I' and g['rob'] == 'R075':
                self.mI = (f, h)
            if plain and g['st'] == 'V' and g['rob'] == 'R0':
                self.mV = (f, h)
        refs = [d for d, r in self.weights.items() if r.get('qa3_guard') == 'reference-identity'
                or r.get('qa3_window') == 'reference']
        aref = str(self.mI[1].get('ALIGNREF', '')) if self.mI else ''
        if self.mI is None:
            self.ref = ('error', f'no Stokes I Taper Robust-0.75 master in {self.cdir}')
        elif len(refs) > 1:
            self.ref = ('error', f'weights.csv names {len(refs)} reference nights {refs}')
        elif refs:
            self.ref = ('night', refs[0]) if aref == refs[0] else \
                ('error', f'weights.csv reference night {refs[0]} != the I Taper R-0.75 master ALIGNREF {aref!r}')
        elif aref.startswith('MASTER:'):
            self.ref = ('master', aref)
        else:
            self.ref = ('error', f'no reference row in weights.csv and master ALIGNREF {aref!r}')

    def ref_card(self):
        """(value, source) or (None, reason) of FLUXREF / ASTREF (:600-610; --ref-policy master-alignref, decision 1)."""
        k, v = self.ref
        if k == 'night':
            return v, f'{self.cdir}/weights.csv reference-identity row = I Taper R-0.75 master ALIGNREF'
        if k == 'master':
            return v, (f'ALIGNREF card of {self.cdir}/{self.mI[0]} as written (flow/.reference_resolved via '
                       '_stamp_master_standard; --ref-policy master-alignref)')
        return None, v

    def night(self, date):
        """{card: (value, source) or (None, reason)} of P2RUN SCALE_A P1RUN CALBP for a night of this cell (:612-695)."""
        if date in self.nprov:
            return self.nprov[date]
        out = {}
        r = self.weights.get(date)
        wp = f'{self.cdir}/weights.csv'
        if r is None:
            why = f'night {date} not in {wp}'
            out = {k: (None, why) for k in ('P2RUN', 'SCALE_A', 'P1RUN', 'CALBP')}
            self.nprov[date] = out
            return out
        sci = (r.get('sci_dir') or '').rstrip('/')
        p2 = os.path.basename(sci)
        if re.match(r'^Science_\d{8}_\d{6}$', p2) and f'/{self.lst}/{date}/' in sci + '/':
            out['P2RUN'] = (p2, f'{wp} sci_dir')
        else:
            out['P2RUN'] = (None, f'{wp} sci_dir {sci!r} is not a Science run of {self.lst} {date}')
        v, src = scale_a(r)
        out['SCALE_A'] = (v, src) if v is not None else (None, src)
        if out['P2RUN'][0] is None:
            out['P1RUN'] = out['CALBP'] = (None, 'no Phase 2 run (P2RUN refused)')
        else:
            if sci not in self._p2:
                self._p2[sci] = phase2_log(sci, self.lst, date)
            lg = self._p2[sci]
            if isinstance(lg, str):
                out['P1RUN'] = out['CALBP'] = (None, lg)
            else:
                from exopipe.phase3.core import cells
                runs, p1dir, lgp = lg
                mem = cells.band_members(self.scope)
                present = [(sb, runs[sb]) for sb in mem if sb in runs]
                amb = [sb for sb, rr in present if len(set(rr)) > 1]
                if not present:
                    out['P1RUN'] = out['CALBP'] = (None, f'{lgp} lists no Phase 1 run for {self.scope}')
                elif amb:
                    out['P1RUN'] = out['CALBP'] = (None, f'{lgp} lists more than one Phase 1 run for {amb}')
                else:
                    pr = [(sb, rr[0]) for sb, rr in present]
                    out['P1RUN'] = (group_values(pr), f'{lgp} "Discovered subbands" ({",".join(sb for sb, _ in pr)})')
                    got = [(sb, phase1_bp(p1dir, run, sb, self.lst, date)) for sb, run in pr]
                    bps = [(sb, bp) for sb, (bp, _) in got if bp is not None]
                    bad = [s2 for sb, (bp, s2) in got if bp is None]
                    notes = sorted({s2.split(' (its run_label')[1][:-1].strip() for sb, (bp, s2) in got
                                    if '(its run_label' in s2})
                    out['CALBP'] = (None, '; '.join(bad[:3])) if bad else \
                        (group_values(bps), f'bp_table of {p1dir}/<run>/<subband>/provenance.json ({len(bps)} subbands)'
                         + (f'; provenance.json run_label {",".join(notes)}' if notes else ''))
        self.nprov[date] = out
        return out

    # ---- the planned cards of one file ------------------------------------------------------------------------------
    def plan_file(self, f, mdw=None, mdw_path=None):
        """{'file_refusals': [...], 'cards': [dict(card, status ADD|PRESENT|REFUSE, value, comment, source)]} of one coadd_final
        file (a master or a PB map). ``mdw``: the loaded MDWSTAT table (dict or error string); None = MDWSTAT is not this
        writer's card (the masters build: the dewarp status is known only after the dewarp)."""
        h = self.hdrs.get(f)
        if h is None:
            h = self.hdrs[f] = _header(f'{self.fd}/{f}')
        kws = set(h.keys())
        rec = dict(file_refusals=[], cards=[])

        def add(key, value, source, comment=None):
            if key != 'HISTORY' and key in kws:
                rec['cards'].append(dict(card=key, status='PRESENT', value='', comment='', source='already in the header'))
                return
            if value is None:
                rec['cards'].append(dict(card=key, status='REFUSE', value='', comment='', source=source))
                return
            com = comment if comment is not None else COMMENTS.get(key, '')
            try:
                card_image(key, value, com)
            except Exception as ex:
                rec['cards'].append(dict(card=key, status='REFUSE', value='', comment='', source=f'card image: {ex}'))
                return
            rec['cards'].append(dict(card=key, status='ADD', value=value, comment=com, source=source))

        for k in ('CHECKSUM', 'DATASUM'):
            if k in kws:
                rec['file_refusals'].append(f'the primary header carries {k} (an added card would invalidate it)')
        cd = h.get('CONSTDIG')
        if cd is not None and str(cd) not in set(self.dig.values()):
            rec['file_refusals'].append(f'CONSTDIG {cd!r} is not one of the installed central masters of the cell')
        # ---- the six provenance cards (prov_coadd, header_repair_release1.py:916-962)
        ref_val, ref_src = self.ref_card()
        ref_kind = 'master' if self.ref[0] == 'master' else 'night'
        aref, link = h.get('ALIGNREF'), None
        if aref is None or cd is None:
            miss = ' and '.join(k for k, v in (('ALIGNREF', aref), ('CONSTDIG', cd)) if v is None)
            link = (f'no recorded link to a master of the cell: the file carries no {miss} card' +
                    (f" (a combination of subband products, LSWBSB {h.get('LSWBSB')})" if h.get('LSWBSB') else ''))
        rv, rs = ref_val, ref_src
        if link:
            rv, rs = None, link
        elif rv is not None and str(aref) != rv:
            rv, rs = None, f'the file ALIGNREF {aref!r} differs from the cell reference {ref_val}'
        if rv is None:
            for k in PROV6:
                add(k, None, rs)
        elif ref_kind == 'master':
            why = ('no reference night: the cell was aligned and flux-scaled to the previous-generation master '
                   f'(ALIGNREF {rv}) - P2RUN SCALE_A P1RUN CALBP of a reference night have no source')
            for k in ('P2RUN', 'SCALE_A'):
                add(k, None, why)
            add('FLUXREF', rv, rs, comment=REF_COMMENT['master']['FLUXREF'])
            add('ASTREF', rv, rs, comment=REF_COMMENT['master']['ASTREF'])
            for k in ('P1RUN', 'CALBP'):
                add(k, None, why)
        else:
            nv = self.night(rv)
            for k in ('P2RUN', 'SCALE_A'):
                add(k, nv[k][0], (nv[k][1] + f' (reference night {rv})') if nv[k][0] is not None else nv[k][1],
                    comment=COADD_COMMENT[k])
            add('FLUXREF', rv, rs)
            add('ASTREF', rv, rs)
            for k in ('P1RUN', 'CALBP'):
                add(k, nv[k][0], (nv[k][1] + f' (reference night {rv})') if nv[k][0] is not None else nv[k][1],
                    comment=COADD_COMMENT[k])
        # ---- MDWSTAT on masters (mdw_cards, :830-863; the PB maps take only the MDWMED1 update, not ported)
        if mdw is not None and f in self.masters:
            cell = f'{self.lst}_{self.scope}'
            if isinstance(mdw, str):
                add('MDWSTAT', None, mdw)
            elif cell not in mdw:
                add('MDWSTAT', None, f'cell {cell} is not in the MDWSTAT table {mdw_path}')
            else:
                v = mdw[cell].get('mdwstat') or ''
                why = None
                if not (0 < len(v) <= 68 and v.isascii() and v.isprintable()):
                    why = f'the table\'s mdwstat {v[:40]!r} is not a one-card value'
                else:
                    has_w, mdwarp, has_sb = 'MDWARP' in kws, h.get('MDWARP'), 'LSWBSB' in kws
                    if has_sb and not has_w:
                        why = ('a combination of subband products (LSWBSB, no MDWARP): never master-dewarped itself; each member '
                               'master carries its own MDWSTAT')
                    elif v.startswith('applied') != (mdwarp is True):
                        why = f'the table says {v[:40]!r} but the file\'s MDWARP is {mdwarp!r}: the table and the file disagree'
                add('MDWSTAT', None if why else v, why or f'{mdw_path} column mdwstat ({RULING_MDWSTAT})')
        return rec

    def provenance_table(self):
        """The cell's provenance_nights.csv text (header_repair_release1.py:1117-1132): date, the six cards, unsourced; one
        row per in_master 1 night, a card blank with its reason where it has no source. None when the cell has no
        weights.csv."""
        if self.err:
            return None
        ref_val, ref_src = self.ref_card()
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator='\n')
        w.writerow(['date'] + list(PROV6) + ['unsourced'])
        for date in sorted(self.weights):
            if str(self.weights[date].get('in_master', '')).strip() != '1':
                continue
            nv = dict(self.night(date))
            nv['FLUXREF'] = nv['ASTREF'] = (ref_val, ref_src)
            w.writerow([date] + ['' if nv[k][0] is None else nv[k][0] for k in PROV6] +
                       ['; '.join(f'{k}: {nv[k][1]}' for k in PROV6 if nv[k][0] is None)])
        return buf.getvalue()


# ---------------------------------------------------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------------------------------------------------

def write_cards(path, adds, today=None):
    """Append the planned ADD cards [(key, value, comment)] and the repair HISTORY line at the end of the primary header (the
    repair's layout: the cards, then the HISTORY lines, policy_files :1289), then the header standard is validated
    (validate=True) and every added card read back. Existing cards are never changed: a planned key now present refuses."""
    if not adds:
        return []
    keys = []
    for k, _, _ in adds:
        if k not in keys:
            keys.append(k)
    lines = history_lines(history_text(keys, today))
    with fits.open(path, mode='update', memmap=False) as hl:
        h = hl[0].header
        now = [k for k in keys if k in h]
        if now:
            raise RuntimeError(f'{path}: {now} already in the header - an existing card is never changed ({RULING})')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', fits.verify.VerifyWarning)
            for k, v, c in adds:
                h.append(fits.Card(k, v, c), end=True)
            for ln in lines:
                h.append(fits.Card('HISTORY', ln), end=True)
    from exopipe.common.validate_fits_headers import validate_header
    report = validate_header(path, level='standard')
    if not report.ok:
        raise RuntimeError(f'Header validation FAILED for {path} after the Release 1 cards:\n'
                           + '\n'.join(f'  ERROR: {e}' for e in report.errors))
    back = _header(path)
    bad = [k for k, v, _ in adds if back.get(k) != v]
    if bad or not any(REPAIR in str(x) and lines[0] in str(x) for x in back.get('HISTORY', [])):
        raise RuntimeError(f'{path}: the Release 1 cards {bad or keys} did not read back as written')
    return keys


def _write_text(path, text):
    tmp = path + '.part'
    with open(tmp, 'w', newline='') as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_cell(final_dir, lst, scope, mdw_table=None, files=None, pbmaps=True, provenance=False, name_dir=None, today=None):
    """Write the Release 1 cards on the cell's coadd-class files in final_dir (the masters, and the PB maps when ``pbmaps``;
    or exactly ``files``, names in final_dir) and, with ``provenance``, its provenance_nights.csv. ``mdw_table``: the MDWSTAT
    table path (None = no MDWSTAT: the masters build). Partial policy (decision 3): a card without a source is not written and
    is returned with its reason. Returns a report dict: ref, files {name: {added, refused {card: reason}, file_refusals}},
    provenance (path or None), lines (one per file, for the log / unit notes)."""
    cell = CellCards(final_dir, lst, scope, name_dir=name_dir)
    mdw = load_mdw_table(mdw_table) if mdw_table else None
    names = list(files) if files is not None else cell.masters + (cell.pbmaps if pbmaps else [])
    rep = dict(cell=f'{lst}_{scope}', ref=list(cell.ref), files={}, provenance=None, lines=[])
    if cell.err or cell.ref[0] == 'error':
        rep['lines'].append(f'cell reference: {cell.ref[1]}')
    for f in names:
        p = cell.plan_file(f, mdw, mdw_table)
        refused = {c['card']: c['source'] for c in p['cards'] if c['status'] == 'REFUSE'}
        added = []
        if not p['file_refusals']:
            added = write_cards(f'{final_dir}/{f}', [(c['card'], c['value'], c['comment']) for c in p['cards']
                                                      if c['status'] == 'ADD'], today)
        rep['files'][f] = dict(added=added, refused=refused, file_refusals=p['file_refusals'],
                               present=[c['card'] for c in p['cards'] if c['status'] == 'PRESENT'])
        rep['lines'].append(f'{f}: ' + (f'REFUSED whole ({"; ".join(p["file_refusals"])[:200]})' if p['file_refusals'] else
                                         f'added {" ".join(added) or "nothing"}' + (f'; unsourced {sorted(refused)}'
                                                                                     if refused else '')))
    if provenance:
        text = cell.provenance_table()
        if text is not None:
            dest = f'{final_dir}/{PROVENANCE_TABLE}'
            _write_text(dest, text)
            rep['provenance'] = dest
            rep['lines'].append(f'{PROVENANCE_TABLE}: {text.count(chr(10)) - 1} constituent nights')
        else:
            rep['lines'].append(f'{PROVENANCE_TABLE}: not written ({cell.err})')
    return rep


def planned_adds(final_dir, lst, scope, mdw_table=None, files=None, pbmaps=True, name_dir=None):
    """{name: [cards a write_cell would add now]} - empty lists when every sourced card is present (the repair's dry run
    finding nothing to add: the install's cards gate on the archived cell)."""
    cell = CellCards(final_dir, lst, scope, name_dir=name_dir)
    mdw = load_mdw_table(mdw_table) if mdw_table else None
    names = list(files) if files is not None else cell.masters + (cell.pbmaps if pbmaps else [])
    out = {}
    for f in names:
        p = cell.plan_file(f, mdw, mdw_table)
        out[f] = [c['card'] for c in p['cards'] if c['status'] == 'ADD'] if not p['file_refusals'] else \
            ['(refused whole: ' + '; '.join(p['file_refusals'])[:120] + ')']
    return out, cell


# ---------------------------------------------------------------------------------------------------------------------
# the sibling card-key gate (sibling_card_gate.py)
# ---------------------------------------------------------------------------------------------------------------------

def value_problems(h, row):
    """The MDWSTAT / MDWMED1 value checks of one header against the cell's table row (None = no row) (sibling_card_gate.py:34-45)."""
    out = []
    if 'MDWSTAT' in h:
        if row is None:
            out.append('MDWSTAT present but the cell has no MDWSTAT table row')
        elif str(h['MDWSTAT']).strip() != row['mdwstat'].strip():
            out.append(f"MDWSTAT {str(h['MDWSTAT'])[:50]!r} != table {row['mdwstat'][:50]!r}")
    if 'MDWMED1' in h and row is not None and row['mdwmed1_header_arcmin'].strip() and row['mdwmed1_installed_cubic_arcmin'].strip():
        old, new = float(row['mdwmed1_header_arcmin']), float(row['mdwmed1_installed_cubic_arcmin'])
        if abs(new - old) > MDWMED1_TOL and abs(float(h['MDWMED1']) - old) <= MDWMED1_TOL:
            out.append(f"MDWMED1 {h['MDWMED1']} is the uncorrected value (table: header {old}, installed {new}; {RULING_MDWMED1})")
    return out


def whole_refusals(tables):
    """{path: reason} of the files whose FLUXREF and ASTREF the repair refused, from refusals.csv tables (:48-55)."""
    cards, why = {}, {}
    for t in tables:
        with open(t, newline='') as f:
            for r in csv.DictReader(f):
                cards.setdefault(r['path'], set()).add(r['card'])
                why.setdefault(r['path'], r['reason'])
    return {p: why[p] for p, c in cards.items() if {'FLUXREF', 'ASTREF'} <= c}


def product_key(path):
    return SPAN.sub('', os.path.basename(path))


def header_keys(path):
    """(card keys without HISTORY/COMMENT/blank, holds the repair HISTORY line, header) (sibling_card_gate.py:62-66)."""
    h = _header(path)
    keys = {k for k in h.keys() if k not in SKIP}
    repaired = any(REPAIR in str(x) for x in h.get('HISTORY', []))
    return keys, repaired, h


def sibling_card_gate(cell, archive_run, mdw_table, sibling_run=None, refusals=()):
    """(passed, lines) of the gate for cell '<LST>_<band>': every FITS of <archive_run>/<LST>/<band>/ against the repaired
    same-named products of the band's other LSTs in <sibling_run>/[0-2][0-9]h/<band>/ (default the same run) (sibling_card_gate.py
    main, :69-117). The last line is 'SIBLING-CARD-GATE PASS|REFUSED <cell> ...'."""
    m = re.fullmatch(r'(\d\dh)_(\w+)', cell)
    if not m:
        return False, [f'SIBLING-CARD-GATE usage: bad cell {cell!r}']
    lst, band = m.groups()
    sibling_run = sibling_run or archive_run
    lines = []
    mine = sorted(glob.glob(f'{archive_run}/{lst}/{band}/*.fits'))
    if not mine:
        return False, [f'SIBLING-CARD-GATE REFUSED {cell}: no archived FITS in {archive_run}/{lst}/{band}']
    sib = {}
    for f in sorted(glob.glob(f'{sibling_run}/[0-2][0-9]h/{band}/*.fits')):
        if f.split('/')[-3] == lst:
            continue
        keys, repaired, _ = header_keys(f)
        if repaired:
            sib.setdefault(product_key(f), []).append(keys)
    with open(mdw_table, newline='') as fh:
        trow = {r['cell']: r for r in csv.DictReader(fh)}.get(f'{lst} {band}')
    exempt_of = whole_refusals(refusals)
    refused = exempt = 0
    for f in mine:
        keys, repaired, h = header_keys(f)
        ref = sib.get(product_key(f))
        vals = value_problems(h, trow)
        if not ref:
            if f in exempt_of and not vals:
                lines.append(f'EXEMPT {os.path.basename(f)}: no repaired sibling; the repair refused it whole ({exempt_of[f][:120]})')
                exempt += 1
                continue
            lines.append(f'REFUSED {os.path.basename(f)}: no repaired sibling product of this name in band {band}'
                         + ''.join(f'; {v}' for v in vals))
            refused += 1
            continue
        need = set.intersection(*ref)
        miss = sorted(need - keys)
        if miss or not repaired or vals:
            lines.append(f'REFUSED {os.path.basename(f)}: missing {len(miss)} card keys held by all {len(ref)} repaired siblings: '
                         f'{",".join(miss[:20])}{" ..." if len(miss) > 20 else ""}' + ('' if repaired else '; no repair HISTORY line')
                         + ''.join(f'; {v}' for v in vals))
            refused += 1
    n_sib = sum(len(v) for v in sib.values())
    if refused:
        lines.append(f'SIBLING-CARD-GATE REFUSED {cell}: {refused} of {len(mine)} archived files fail ({n_sib} repaired sibling '
                     f'files read)')
        return False, lines
    lines.append(f'SIBLING-CARD-GATE PASS {cell}: {len(mine) - exempt} of {len(mine)} archived files carry every card key of their '
                 f'repaired siblings ({n_sib} sibling files read), {exempt} exempt (refused whole by the repair), and MDWSTAT / '
                 f'MDWMED1 agree with the table ({"row " + repr(trow["mdwstat"][:40]) if trow else "no table row"})')
    return True, lines
