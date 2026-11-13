"""Content tests of the deep `exopipe3 search` product classes (core/validate discovers this module).

Classes (spec search.md section 6): detections (master_catalog.csv), gate_ledger (v_deep_ledger.csv), image_ledger
(v_deep_images.csv), refusal_record (refusals/<cell>_<kind>_excluded.csv), candidate_stamps (the candidates/ directory),
fp_gate_verdicts (V_top100_fp_gate_p3.csv), detection_positions (fit_positions.csv); under the B-02 rules (step 3b) frame_noise
(frame_noise.csv: each frame's noise and the medians it was judged by) and table_pulsar_decisions (table_pulsar_decisions.csv);
gate_population (gate_population.csv: the beam gate's population a whole-cell search judged by, read by an append);
search_table_record (search_table_record.csv: every verdict of the search's bright-source table, its inputs and whether it was
applied; production exopipe 55b0263). search_frame and noise_map belong to frames,
master_catalog to masters; the search step runs their tests through the registry when they exist.

Every test reads the file's CONTENT and works with no context: siblings are found from the path in the attempt layout
    <attempt>/transients/Run_<label>/{master_catalog.csv, v_deep_ledger.csv, v_deep_images.csv, refusals/, candidates/}
    <attempt>/Run_<label>/{iresid,vresid}/<lst>_<scope>/<date>_<S>_redo_resid_<scope>.fits
    <attempt>/fit_positions.csv
Optional context (the search step passes it): tier ('deep' default), stokes, cell, catalog (path of master_catalog.csv for
fp_gate_verdicts).

Schemas are those of the in-place writers, carried as literals (deep_search.py:116-128 GLED_COLS / GIMG_COLS; candidates.py
:214-225 fields; core/fitting DET_COLS); tests/test_search-deep.py checks they equal the engine's source.

Ported from: exopipe_ops/verify_p3s3_stage.py conditions 2-5 (frames = image rows, one-to-one catalogue/ledger by nearest
position with a 0.5 deg cap, refusal ledgers), deep_fit_positions.py:138-159 (the fit stage's four refusals),
sublst_fit_positions.py F1 (stamp pixels = frame pixels).
"""
import csv
import glob
import hashlib
import math
import os
import re

from exopipe.phase3.core.validate import content_test
from exopipe.phase3.lib.search import attempts as _ATTS

# ---- schemas of the in-place writers ---------------------------------------------------------------------------------------
GLED_COLS_V1 = ['lst', 'scope', 'stokes', 'date', 'ra', 'dec', 'pix_x', 'pix_y', 'snr', 'peak_jy', 'morph_pass', 'veto_reason',
                'mstruct', 'mstruct_raw', 'pass_mstruct', 'sstack', 'jupiter_deg', 'jupiter_veto', 'final', 'n_v_epoch', 'moon_deg',
                'global_snr', 'size_beams', 'size_flag', 'nearest_ateam', 'fp_prediction', 'axis_norm', 'size_err_beams']
# 2026-10-17: the bright-parent table's inputs appended (deep_search.GLED_COLS_TABLE); every ledger written before has GLED_COLS_V1
GLED_COLS_TABLE = ['parent_jy', 'parent_off_beams', 'table_note', 'table_key', 'table_margin', 'table_parent_S',
                   'table_parent_sep_bw', 'table_R', 'table_bin', 'table_frame_rms_jy']
GLED_COLS = GLED_COLS_V1 + GLED_COLS_TABLE
# the size / axis rule at every S/N (R-GREGG-20261022-ALL-APPROVED-SIZE-GATE-READMISSION-SUBAGENTS; deep_search GLED_COLS_SIZE =
# lib/search/size_rule.GLED_COLS_SIZE): appended after GLED_COLS by a search under SIZE_RULE 'envelope' (its run directory holds
# size_rule.json); production's ledger is GLED_COLS. size_role since size-rule-2 (2026-11-03): every candidate row carries the
# rule's columns ('gate': the rule decided it; 'record': measured after an earlier gate removed it), since size-rule-3
# (2026-11-04) with ':full_size_off (spectrum-aware reference pending)' on every deep Full-band row while that band's size test is
# off (search_detect.FULL_BAND_SIZE, the record's full_band_size); a size-rule-1 search (8957c42) wrote the first seven only
# (GLED_COLS_SIZE_V1)
GLED_COLS_SIZE = ['size_rule', 'size_snr', 'size_thr', 'axis_thr', 'size_excess', 'axis_excess', 'size_verdict', 'size_role']
GLED_COLS_SIZE_V1 = GLED_COLS_SIZE[:7]
GIMG_COLS = ['lst', 'scope', 'stokes', 'date', 'n_ident5', 'n_after_morph', 'n_after_mstruct', 'n_after_jupiter', 'n_final',
             'n_raw_floor', 'elev_min_deg']
CAT_FIELDS = ['cand_id', 'lst', 'date', 'band', 'stokes', 'cadence', 'interval', 'ra', 'dec', 'peak_jy', 'int_jy', 'snr',
              'axis_ratio', 'fwhm_maj_arcmin', 'c50_mjy', 'mstruct', 'mstruct_raw', 'sstack', 'jupiter_deg', 'n_v_epoch', 'body',
              'repeat_group', 'epoch_qa', 'fp_prediction', 'global_snr', 'size_beams', 'size_flag', 'nearest_ateam', 'moon_deg',
              'axis_norm', 'axis_flag']
# every veto_reason the engine writes (deep_search.py _gled_row call sites; identification statuses lower-cased)
VETO_REASONS = frozenset({'', 'reject_local_snr', 'reject_merged', 'reject_sigma_stack', 'reject_bright_prox', 'reject_i_leakage',
                          'reject_local_rms', 'reject_bilobe', 'reject_bright_proximity', 'reject_exclusion',
                          'reject_sidelobe_tier', 'mstruct', 'mstruct_I', 'gaussfit-size', 'axis-ratio', 'jupiter', 'moon', 'ateam',
                          'v110:FP', 'v110:marginal', 'table', 'row-line',
                          'elevation'})   # B-02 step 3b: the deep floor at the frame's mid-time (removal_rules.frame_elevation)
# ... and those with a value after the prefix: the Jupiter track's co-added night (B-02 step 3a, deep_search: 'jupiter-track:<night>')
VETO_REASON_PREFIXES = {'jupiter-track:': re.compile(r'^\d{4}-\d\d-\d\d$'),
                        # the size / axis rule (size-rule-2): a Stokes I row of the multiples fork's member pool (deep, below the
                        # local-SNR floor, 5 <= |snr| < 7) the rule removes - out of the pool (size_rule.engine_early, MEMBER_REASON)
                        'reject_local_snr:': re.compile(r'^(gaussfit-size|axis-ratio)$')}
# B-02 step 3b (removal_rules.FRAME_NOISE_COLS / TABLE_PULSAR_COLS)
FNOISE_COLS = ['lst', 'scope', 'stokes', 'date', 'noise_jy', 'admitted', 'population', 'tblnoise_jy', 'vsanmed_jy', 'vsanity_x',
               'attempt']
TPD_COLS = ['lst', 'scope', 'stokes', 'date', 'ra', 'dec', 'pix_x', 'pix_y', 'cut', 'note']
# the search-time bright-source table's record (deep_search.STR_COLS; production exopipe 55b0263, 2026-10-18)
STR_COLS = ['lst', 'scope', 'stokes', 'date', 'ra', 'dec', 'pix_x', 'pix_y', 'snr', 'verdict', 'note', 'frame_rms_jy', 'beam_deg',
            'applied']


def known_veto_reason(v):
    """A veto_reason the engine writes: one of VETO_REASONS, or a VETO_REASON_PREFIXES prefix with its value."""
    return v in VETO_REASONS or any(v.startswith(p) and rx.match(v[len(p):]) for p, rx in VETO_REASON_PREFIXES.items())


# refusal kinds: (column count, which searches the refusal removes: 'both' | 'I' | 'V' | 'row' = the Stokes in column 4)
REFUSAL_KINDS = {'nightimage': (4, 'both'), 'epochgate': (4, 'both'), 'sanity': (6, 'both'), 'sanity_I': (6, 'I'),
                 'sanity_V': (6, 'V'),
                 'searchtier': (8, 'row'),  # lst, scope, date, Stokes, stage-1 rms, threshold, rule, attempt (the rule since
                                            # 2026-10-30, production exopipe 702dd68: 'ratio2xmedian'; REFUSAL_EARLIER_NCOL)
                 'stale_input': (5, 'both'), 'beam': (6, 'both'),
                 'extraction_I': (7, 'I'), 'extraction_V': (7, 'V'),
                 'daytime': (6, 'row'),     # B-02 step 3a: the daytime gate per frame (lst, scope, date, Stokes, reason, attempt)
                 'wbmember': (7, 'row'),    # R-GREGG-20261023-WIDEBAND-MEMBER-SEARCH-ADMISSION: a wideband epoch whose member
                                            # subband fails the admission in that Stokes (lst, scope, date, Stokes, member, reason, attempt;
                                            # one row per failing member)
                 'datadecisions': (4, 'both')}   # B-02 step 3c: the nights data_decisions excluded (lst, scope, date, attempt)
# kinds whose rows written before a format change have another column count (valid still: the attempt tag stays the last
# column): searchtier rows before 2026-10-30 have no rule column (7). A file may hold both forms - a later search appends to
# its attempt's file, and production's drop-patched attempts of 2026-10-30 ('...stdrop') carry the old rows with the new ones
REFUSAL_EARLIER_NCOL = {'searchtier': 7}
_SEARCHTIER_RULE = re.compile(r'^ratio[0-9.eE+-]+xmedian$')   # deep_search: f'ratio{SEARCH_TIER_RATIO:g}xmedian'
# kinds that record, not refuse (the nights data_decisions excluded when the search ran): refused_dates leaves them out, so
# the gate_ledger and image_ledger content tests do not fail an append that searched a night its base recorded, and the
# search step's unaccounted_epochs judges its own preflight's admitted nights only by real refusals (it runs at search
# time: a later lift is the conformance check's direction (b), conform_check)
RECORD_ONLY_KINDS = ('datadecisions',)
FIT_KINDS = ('gaussian', 'template', 'peak')
FIT_EXTRA = ['frame', 'frame_sha1', 'fit_frac_px']
FP_VERDICTS = frozenset({'FP', 'not-FP', 'marginal', 'inconclusive'})
_CID = re.compile(r'^(?P<s>[IV])(?P<c>[A-Z0-9])_(?P<lst>\d\dh)_(?P<date>\d{4}-\d\d-\d\d(?:T\d\d)?)_(?P<band>\w+?)_(?P<n>\d{4,})$')
_REF = re.compile(r'^(?P<lst>\d\dh)_(?P<scope>\d+MHz|Red|Green|Blue|Full)_(?P<kind>.+)_excluded\.csv$')
_DATE = re.compile(r'^\d{4}-\d\d-\d\d(T\d\d)?$')


def det_cols():
    from exopipe.phase3.core.fitting import DET_COLS
    return list(DET_COLS)


# ---- helpers ---------------------------------------------------------------------------------------------------------------
def _rows(path):
    with open(path, newline='') as f:
        rd = csv.DictReader(f)
        return list(rd.fieldnames or []), list(rd)


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float('nan')


def run_dir_of(path):
    """transients/Run_<label> directory of a product inside it (or the directory itself)."""
    d = path if os.path.isdir(path) else os.path.dirname(path)
    while d and d != '/':
        if os.path.basename(d).startswith('Run_') and os.path.basename(os.path.dirname(d)) == 'transients':
            return d
        d = os.path.dirname(d)
    return None


def attempt_of(path):
    rd = run_dir_of(path)
    return os.path.dirname(os.path.dirname(rd)) if rd else None


def label_of(path):
    rd = run_dir_of(path)
    return os.path.basename(rd)[4:] if rd else None


def frame_path(attempt, label, lst, band, date, stokes):
    sub = 'iresid' if stokes == 'I' else 'vresid'
    return os.path.join(attempt, f'Run_{label}', sub, f'{lst}_{band}', f'{date}_{stokes}_redo_resid_{band}.fits')


def refusal_files(run_dir):
    """{kind: path} of the refusal ledgers in transients/Run_<label>/refusals/."""
    out = {}
    for p in sorted(glob.glob(os.path.join(run_dir, 'refusals', '*_excluded.csv'))):
        m = _REF.match(os.path.basename(p))
        if m:
            out.setdefault(m['kind'], []).append(p)
    return out


def refused_dates(run_dir, stokes):
    """(dates refused from the search of `stokes`, dates whose IMAGE was refused (extraction: frame kept, no image row),
    dates refused from both searches) from the refusal ledgers (not the RECORD_ONLY_KINDS records)."""
    both, mine, extraction = set(), set(), set()
    for kind, paths in refusal_files(run_dir).items():
        spec = REFUSAL_KINDS.get(kind)
        if spec is None or kind in RECORD_ONLY_KINDS:
            continue
        for p in paths:
            with open(p, newline='') as f:
                for r in csv.reader(f):
                    if len(r) < 3:
                        continue
                    d = r[2]
                    if spec[1] == 'both':
                        both.add(d)
                        mine.add(d)
                    elif spec[1] == 'row':
                        if len(r) > 3 and r[3] == stokes:
                            mine.add(d)
                    elif spec[1] == stokes:
                        mine.add(d)
                        if kind.startswith('extraction_'):
                            extraction.add(d)
    return mine, extraction, both


def match_one_to_one(cat_pos, fin_pos, cap_deg=0.5):
    """verify_p3s3_stage.py condition 3: pair catalogue and final positions of one date one-to-one by nearest separation
    (flat-sky with RA wrap, cap 0.5 deg). Returns (unbacked catalogue rows, lost finals)."""
    pairs = []
    for ci, (ra, dec) in enumerate(cat_pos):
        for fi, (fr, fd) in enumerate(fin_pos):
            dra = (fr - ra + 180.0) % 360.0 - 180.0
            dd = math.sqrt(dra ** 2 * math.cos(math.radians(dec)) ** 2 + (fd - dec) ** 2)
            if dd < cap_deg:
                pairs.append((dd, ci, fi))
    pairs.sort()
    uc, uf = set(), set()
    for _dd, ci, fi in pairs:
        if ci in uc or fi in uf:
            continue
        uc.add(ci)
        uf.add(fi)
    return len(cat_pos) - len(uc), len(fin_pos) - len(uf)


# Code generations of the engine that change what an attempt must show (the verifier's "expectation by generation",
# verify_p3s3_stage.py:110-120). An attempt is dated by its directory name <cell>_<S>_<YYYYmmddTHHMMSS>.
ELEV_RECORD_FIX_UTC = '20261007T083010'   # exopipe d8beb0d: v_deep_images elev_min_deg records the floor used (it was a literal
ELEV_RECORD_LITERAL = 20.0                # 20.0 whatever the floor was - deep_search._elev_floor docstring)
FRAME_EVERY_EPOCH_UTC = '20260913T054452' # code floor: a frame + an image row for every epoch not beam/exclusion-refused; before
                                          # it a sanity-refused epoch had an image row but no frame (verify_p3s3_stage.py:110-120)
_ATT_TAG = re.compile(r'_(' + _ATTS.TAG_RX + r')$')   # the tag ending an attempt directory's name (lib/search/attempts.py)


def attempt_tag(path):
    """YYYYmmddTHHMMSS of the search whose rows the attempt holding `path` holds: its directory name's UTC; for production's drop
    patch '<UTC>stdrop' its source's (attempts.rows_utc - the patch carries its source's rows; 58 of the 142 patch sources of
    2026-10-30 were searched before ELEV_RECORD_FIX_UTC), None when the logs name no source. None outside an attempt."""
    att = attempt_of(path)
    m = _ATT_TAG.search(os.path.basename(att.rstrip('/'))) if att else None
    if not m:
        return None
    if m.group(1) == _ATTS.utc(m.group(1)):          # a search attempt: its own UTC
        return m.group(1)
    return _ATTS.rows_utc(att)                         # a drop patch: its source's


def _elev_floor(tier, stokes, tag=None):
    """The elevation floor the detection mask of (tier, Stokes) used, as the image ledger must record it."""
    from exopipe.phase3.config import CFG
    if tag and tag < ELEV_RECORD_FIX_UTC:
        return ELEV_RECORD_LITERAL
    if tier == 'deep':
        sd = CFG.search_detect
        return float(sd.I_MASK_FLOOR_DEG if stokes == 'I' else sd.ELEVATION_FLOOR_DEG)
    sec = getattr(CFG, f'search_{tier}', None)      # the tier draft's section (search_sublst)
    if sec is None:
        raise KeyError(tier)
    return float(sec.I_MASK_FLOOR_DEG if stokes == 'I' else sec.MIN_ELEVATION_DEG)


def _excluded_now(lst, scope, dates, tier='deep'):
    from exopipe.phase3.core import exclusions
    return exclusions.for_tier(tier).excluded_constituents(lst, scope, sorted(dates))


def sha1_file(path):
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def _fail(problems, value):
    return False, value, '; '.join(problems[:6]) + (f' (+{len(problems) - 6} more)' if len(problems) > 6 else '')


# ---- detections -------------------------------------------------------------------------------------------------------------
@content_test('detections')
def schema_and_backing(path, stokes=None, cell=None, tier='deep', **ctx):
    """master_catalog.csv: the 31 fields; cand_id unique, of the writer's form and numbered in file order (candidates.py:
    n0 + i); one unit (lst, band, Stokes); |snr| at the Stokes floor; every row backed ONE-TO-ONE by a final gate-ledger row
    of its date (cap 0.5 deg, verify_p3s3 condition 3); no row on a night data_decisions excludes now. Production's drop patch
    (<UTC>stdrop, lib/search/attempts.py) keeps its source's numbers: they increase in file order, and its rows are exactly its
    source's less those of the nights it dropped (the drop tool's own test D1; when the source's catalogue can be read)."""
    from exopipe.phase3.config import CFG
    cols, rows = _rows(path)
    att = attempt_of(path)
    am = _ATTS.parse(att) if att else None
    patch = bool(am and am['suffix'])
    prev_n = -1
    probs = []
    base = cols[:len(CAT_FIELDS)]
    if base != CAT_FIELDS:
        return False, len(rows), f'header != the 31 writer fields (candidates.py:214-225): {cols[:6]}...'
    extra = cols[len(CAT_FIELDS):]
    if extra and extra != [c for c in det_cols() if c not in CAT_FIELDS]:
        probs.append(f'unexpected extra columns {extra[:4]}')
    seen = set()
    units = {(r['lst'], r['band'], r['stokes']) for r in rows}
    if len(units) > 1:
        probs.append(f'{len(units)} (lst, band, Stokes) units in one attempt catalogue')
    floor = CFG.search_detect.DETECTION_FLOOR_SIGMA
    for i, r in enumerate(rows):
        cid = r['cand_id']
        m = _CID.match(cid)
        if cid in seen:
            probs.append(f'duplicate cand_id {cid}')
        seen.add(cid)
        if not m:
            probs.append(f'cand_id {cid!r} not of the form <S><cadence letter>_<lst>_<date>_<band>_<NNNN>')
            continue
        if (m['s'], m['lst'], m['date'], m['band']) != (r['stokes'], r['lst'], r['date'], r['band']):
            probs.append(f'{cid}: id disagrees with its row (stokes/lst/date/band)')
        if m['c'] != (r['cadence'][:1].upper() if r['cadence'] else '?'):
            probs.append(f'{cid}: cadence letter != cadence {r["cadence"]!r}')
        if patch:                                  # a drop patch: its source's numbers, gaps where it dropped nights
            if int(m['n']) <= prev_n:
                probs.append(f'{cid}: number {int(m["n"])} not above the row before ({prev_n}): a drop patch keeps its source\'s order')
            prev_n = int(m['n'])
        elif int(m['n']) != i:
            probs.append(f'{cid}: number {int(m["n"])} != its row index {i} (the writer numbers rows n0 + i)')
        s = abs(_f(r['snr']))
        fl = floor.get(r['stokes'])
        if fl is not None and not (s >= fl - 0.005):
            probs.append(f'{cid}: |snr| {r["snr"]} below the Stokes {r["stokes"]} floor {fl}')
        if not (math.isfinite(_f(r['ra'])) and math.isfinite(_f(r['dec']))):
            probs.append(f'{cid}: ra/dec not finite')
    # duplicate rows: same date and position to 4 decimals (verify_p3s3_stage.py:144-149, 166)
    pos = [(r['date'], round(_f(r['ra']), 4), round(_f(r['dec']), 4)) for r in rows]
    ndup = len(pos) - len(set(pos))
    if ndup:
        probs.append(f'{ndup} duplicate catalogue rows (same date, ra, dec to 4 decimals)')
    if stokes and any(u[2] != stokes for u in units):
        probs.append(f'Stokes {sorted(u[2] for u in units)} != the unit Stokes {stokes}')
    if cell and any(f'{u[0]}_{u[1]}' != cell for u in units):
        probs.append(f'cell {sorted(f"{u[0]}_{u[1]}" for u in units)} != the unit cell {cell}')
    # one-to-one with the final gate-ledger rows of the same date
    led = os.path.join(os.path.dirname(path), 'v_deep_ledger.csv')
    if not os.path.exists(led):
        probs.append('no sibling v_deep_ledger.csv to back the rows')
    else:
        _c, lrows = _rows(led)
        fin = {}
        for r in lrows:
            if r.get('final') == '1':
                fin.setdefault((r['stokes'], r['date']), []).append((_f(r['ra']), _f(r['dec'])))
        cat = {}
        for r in rows:
            cat.setdefault((r['stokes'], r['date']), []).append((_f(r['ra']), _f(r['dec'])))
        unb = lost = 0
        for k in set(cat) | set(fin):
            a, b = match_one_to_one(cat.get(k, []), fin.get(k, []))
            unb += a
            lost += b
        if unb:
            probs.append(f'{unb} catalogue rows without a final ledger row')
        if lost:
            probs.append(f'{lost} final ledger rows without a catalogue row')
    # exclusions now
    for (lst, band, st) in units:
        bad = _excluded_now(lst, band, {r['date'][:10] for r in rows}, tier)
        if bad:
            probs.append(f'rows on nights data_decisions excludes now: {bad[:5]}')
    if patch:
        probs += _patch_rows(path, att, am, rows)
    if probs:
        return _fail(probs, len(rows))
    return True, len(rows), (f'{len(rows)} rows, ids ' + ('in its source\'s order (a drop patch)' if patch else 'sequential')
                             + ', every row backed by one final ledger row')


def _patch_dropped(att, am):
    """The nights a drop patch dropped: the nights of its refusal rows tagged with its own tag (the drop tool's test D4)."""
    out = set()
    for p in glob.glob(os.path.join(att, 'transients', 'Run_*', 'refusals', '*_excluded.csv')):
        with open(p, newline='') as fh:
            out |= {r[2] for r in csv.reader(fh) if len(r) > 3 and r[-1] == am['att']}
    return out


def _patch_rows(path, att, am, rows):
    """[problems] of a drop patch's table against its source's (production's drop test D1): exactly the source's rows less those
    of the dropped nights in its Stokes. [] when the source's table cannot be read (the patch is then judged by its own rows)."""
    src = _ATTS.stdrop_source(att)
    sp = os.path.join(src, os.path.relpath(path, att)) if src else None
    if not sp or not os.path.exists(sp):
        return []
    dropped = _patch_dropped(att, am)
    want = [r for r in _rows(sp)[1] if not (r['date'] in dropped and r['stokes'] == am['s'])]
    if rows != want:
        return [f'the rows are not its source\'s less those of its {len(dropped)} dropped night(s) in Stokes {am["s"]} '
                f'({len(rows)} rows, {len(want)} expected; production\'s drop test D1)']
    return []


# ---- gate ledger ------------------------------------------------------------------------------------------------------------
def size_rule_rows(path, cols, rows, tier='deep'):
    """The size / axis rule in a gate ledger: [problem]. The header and the run's record agree (GLED_COLS_SIZE exactly when
    size_rule.json says 'envelope', of its version); under the rule EVERY row carries a verdict and its role, under the product
    the record names (spec fp_gates_v2.md section 4: every candidate row); a 'gate' row's removal is the rule's own verdict and
    vice versa ('gaussfit-size' / 'axis-ratio'; a row of the multiples fork's member pool 'reject_local_snr:<verdict>'); a
    'record' row was removed by an earlier gate (its veto_reason, never the rule's); the member pool is exactly the deep tier's
    Stokes I 'reject_local_snr' rows whose ledger S/N lies in catalog_fork.MEMBER_SNR (5 <= |snr| < 7: size_rule.member_window)
    - each of them 'gate', every other local-SNR row 'record', none in the 10-minute tier; every final was judged (point-like,
    fit-failed or no-snr) with excesses <= 1. The deep Full band under the record's full_band_size 'off' (size-rule-3): EVERY
    row of it carries the mark ('<gate|record>:' + FULL_OFF_NOTE), no other row does, and none of its rows is removed by size
    (its size excess recorded, never applied; the axis test applies). A size-rule-1 ledger (the first seven columns, judged rows
    only) is read by that version's checks."""
    from exopipe.phase3.lib.search import size_rule as SZ
    probs = []
    v2, v1 = cols == GLED_COLS + GLED_COLS_SIZE, cols == GLED_COLS + GLED_COLS_SIZE_V1
    try:
        rec = SZ.read_run_record(os.path.dirname(os.path.abspath(path)))
    except RuntimeError as e:
        return [str(e)]
    if (rec['rule'] == 'envelope') != (v2 or v1):
        return [f"the ledger {'carries' if (v2 or v1) else 'lacks'} the size rule's columns while its run's record "
                f"({SZ.RUN_RECORD}) says {rec['rule']!r}"]
    if not (v2 or v1):
        return []
    want_v = (SZ.RECORD_VERSION, 'size-rule-2') if v2 else ('size-rule-1',)   # size-rule-2 and -3 share the columns
    if rec['version'] not in want_v:
        return [f"the ledger carries the columns of {' / '.join(want_v)}, its run's record ({SZ.RUN_RECORD}) says "
                f"{rec['version']!r}"]
    label = f"envelope:{rec['envelope']}:{rec['md5'][:12]}"
    window = SZ.member_window('deep' if tier == 'deep' else '10min', from_env=False)
    full_off = tier == 'deep' and rec['full_band_size'] == 'off'      # the deep Full band's size test off (size-rule-3)
    for r in rows:
        v, sv, where = r['veto_reason'], r.get('size_verdict') or '', f"{r['stokes']} {r['date']} {r['ra']},{r['dec']}"
        if (sv or v2) and sv not in SZ.VERDICTS:
            probs.append(f'{where}: size_verdict {sv!r} is not one of {SZ.VERDICTS}' + (' (every candidate row is judged or '
                                                                                         'recorded)' if v2 and not sv else ''))
            continue
        if sv and r.get('size_rule') != label:
            probs.append(f"{where}: judged under {r.get('size_rule')!r}, the run's record names {label!r}")
        if v1:
            if v in SZ.REMOVING and sv != v:
                probs.append(f'{where}: removed as {v!r} with size_verdict {sv!r}')
            if sv in SZ.REMOVING and v != sv:
                probs.append(f'{where}: the rule removes it ({sv}) but its veto_reason is {v!r}')
            continue
        full_role = r.get('size_role') or ''
        role, off = SZ.role_base(full_role), SZ.role_full_off(full_role)
        if full_role not in (role, f'{role}{SZ.ROLE_SEP}{SZ.FULL_OFF_NOTE}'):
            probs.append(f'{where}: size_role {full_role!r} is neither one of {SZ.ROLES} nor one of them marked '
                         f'{SZ.ROLE_SEP}{SZ.FULL_OFF_NOTE}')
            continue
        if off != (full_off and r['scope'] == SZ.FULL_BAND):           # the deep Full band's size test off: every row marked
            probs.append(f"{where}: a {r['scope']} row with size_role {full_role!r} where the run's record says the deep "
                         f"Full band's size test is {rec['full_band_size']!r} (tier {tier}) - every deep Full-band row carries "
                         f"the mark under 'off', no other row")
            continue
        if off and (sv == 'gaussfit-size' or v in ('gaussfit-size', SZ.MEMBER_REASON + 'gaussfit-size')):
            probs.append(f'{where}: removed by size ({v!r}, size_verdict {sv!r}) in the deep Full band, whose size test is '
                         f'off ({SZ.FULL_OFF_NOTE})')
            continue
        member = v.startswith(SZ.MEMBER_REASON)
        if (v == 'reject_local_snr' or member) and r['stokes'] == 'I':     # the fork's member pool: judged exactly inside it
            try:
                snr_ = round(float(r['snr']), 2)
            except (TypeError, ValueError):
                snr_ = None
            pool = SZ.in_member_pool(snr_, window)
            if pool != (role == 'gate'):
                probs.append(f"{where}: {v!r} at |S/N| {r['snr']} with size_role {full_role!r} - the member pool ("
                             + (f'{window[0]:g} <= |S/N| < {window[1]:g}' if window else 'none in the 10-minute tier')
                             + ") is judged ('gate'), every other local-SNR row recorded ('record')")
                continue
        if role not in SZ.ROLES:
            probs.append(f'{where}: size_role {full_role!r} is not one of {SZ.ROLES}')
        elif role == 'gate':
            if (v in SZ.REMOVING or member) and sv != (v[len(SZ.MEMBER_REASON):] if member else v):
                probs.append(f'{where}: removed as {v!r} with size_verdict {sv!r}')
            if sv in SZ.REMOVING and v not in (sv, SZ.MEMBER_REASON + sv):
                probs.append(f'{where}: the rule removes it ({sv}) but its veto_reason is {v!r}')
            if member and r['stokes'] != 'I':
                probs.append(f"{where}: {v!r} on a Stokes {r['stokes']} row (only the Stokes I local-SNR rows are judged so)")
        elif not v or v in SZ.REMOVING or member:
            probs.append(f"{where}: a 'record' row (measured after an earlier gate removed it) with veto_reason {v!r}")
    for r, _rule, words in SZ.final_findings(rows, full_band=rec['full_band_size'] if tier == 'deep' else 'rule'):
        probs.append(f"{r['stokes']} {r['date']} {r['ra']},{r['dec']}: {words}")
    return probs


@content_test('gate_ledger')
def complete(path, stokes=None, cell=None, tier='deep', **ctx):
    """v_deep_ledger.csv: GLED_COLS; veto_reason known; final=1 exactly when veto_reason is ''; per (Stokes, date) the final
    rows equal the catalogue rows; finals and every I row past identification carry mstruct (GAP-I-CHAIN-GATE-VALUES ledger
    595); no row on a date refused from that Stokes' search; sub-LST: no 'table' row (sublst_search_pass.sh:49-50). A run
    searched under the size / axis rule (size_rule.json beside) carries GLED_COLS_SIZE on every row: its removals are the
    rule's own verdicts ('gate' rows), every other row measured ('record'), every final judged and passing by its recorded
    excesses, every row naming the product (size_rule_rows)."""
    cols, rows = _rows(path)
    if cols not in (GLED_COLS, GLED_COLS_V1, GLED_COLS + GLED_COLS_SIZE, GLED_COLS + GLED_COLS_SIZE_V1):
        return False, len(rows), (f'header is neither GLED_COLS ({len(GLED_COLS)}), GLED_COLS + the size rule\'s columns '
                                  f'({len(GLED_COLS) + len(GLED_COLS_SIZE)}; size-rule-1: {len(GLED_COLS) + len(GLED_COLS_SIZE_V1)}) '
                                  f'nor the pre-2026-10-17 GLED_COLS_V1 ({len(GLED_COLS_V1)}): {cols[:6]}...')
    probs = []
    reasons = {}
    probs += size_rule_rows(path, cols, rows, tier=tier)
    for r in rows:
        v = r['veto_reason']
        reasons[v] = reasons.get(v, 0) + 1
        # the table's recorded inputs agree with its verdict: a 'table' row is one whose margin is < 1 (removals cuts exactly
        # there, before its exemptions; an exempted candidate passes with a margin < 1 and its note says so)
        if v == 'table' and r.get('table_margin') not in (None, '') and not float(r['table_margin']) < 1.0:
            probs.append(f"{r['date']} {r['ra']},{r['dec']}: 'table' row with table_margin {r['table_margin']} >= 1")
        if not known_veto_reason(v):
            probs.append(f'unknown veto_reason {v!r}')
        if (r['final'] == '1') != (v == ''):
            probs.append(f"{r['date']} {r['ra']},{r['dec']}: final={r['final']} with veto_reason {v!r}")
        if r['final'] == '1' and r['mstruct'] == '':
            probs.append(f"{r['stokes']} final {r['date']} {r['ra']},{r['dec']} without mstruct")
        if r['stokes'] == 'I' and not v.startswith('reject_') and r['mstruct'] == '':
            probs.append(f"I row past identification ({v or 'final'}) without mstruct {r['date']} {r['ra']},{r['dec']}")
    units = {(r['lst'], r['scope'], r['stokes']) for r in rows}
    if stokes and any(u[2] != stokes for u in units):
        probs.append(f'Stokes {sorted({u[2] for u in units})} != the unit Stokes {stokes}')
    if cell and any(f'{u[0]}_{u[1]}' != cell for u in units):
        probs.append('rows of another cell')
    if tier == 'sublst' and reasons.get('table'):
        probs.append(f"{reasons['table']} 'table' rows in a sub-LST search (record-only tables; P9)")
    cat = os.path.join(os.path.dirname(path), 'master_catalog.csv')
    fin = {}
    for r in rows:
        if r['final'] == '1':
            fin[(r['stokes'], r['date'])] = fin.get((r['stokes'], r['date']), 0) + 1
    ncat = {}
    if os.path.exists(cat):
        for r in _rows(cat)[1]:
            ncat[(r['stokes'], r['date'])] = ncat.get((r['stokes'], r['date']), 0) + 1
    for k in set(fin) | set(ncat):
        if fin.get(k, 0) != ncat.get(k, 0):
            probs.append(f'{k[0]} {k[1]}: {fin.get(k, 0)} final rows != {ncat.get(k, 0)} catalogue rows')
    rd = run_dir_of(path)
    if rd:
        for st in {u[2] for u in units}:
            mine, _x, _b = refused_dates(rd, st)
            hit = sorted({r['date'] for r in rows if r['stokes'] == st} & mine)
            if hit:
                probs.append(f'Stokes {st} ledger rows on dates refused from its search: {hit[:5]}')
    if probs:
        return _fail(probs, len(rows))
    return True, len(rows), f'{len(rows)} rows, {sum(fin.values())} final; reasons ' + ', '.join(
        f'{k or "final"} {n}' for k, n in sorted(reasons.items(), key=lambda x: -x[1])[:6])


# ---- image ledger -----------------------------------------------------------------------------------------------------------
@content_test('image_ledger')
def one_row_per_searched_epoch(path, stokes=None, cell=None, tier='deep', **ctx):
    """v_deep_images.csv: GIMG_COLS; one row per (date, Stokes); elev_min_deg = the configured mask floor of that Stokes and
    tier (deep 25/25, sub-LST V 25 / I 20; attempts before exopipe d8beb0d recorded a literal 20.0); no row on a date refused
    from both searches or whose image was refused; one search frame per row and no other frame of that Stokes (verify_p3s3
    condition 2: frames = image rows + extraction-refused images; before code floor 20260913T054452 a sanity-refused epoch may
    lack its frame); no image row on a night data_decisions excludes now (verify_p3s3 STALE, p3s3_stale_exclusions.py)."""
    cols, rows = _rows(path)
    if cols != GIMG_COLS:
        return False, len(rows), f'header != GIMG_COLS (11, deep_search.py:126-128): {cols}'
    probs = []
    seen = set()
    for r in rows:
        k = (r['stokes'], r['date'])
        if k in seen:
            probs.append(f'duplicate image row {k}')
        seen.add(k)
        if not _DATE.match(r['date']):
            probs.append(f'bad date {r["date"]!r}')
        try:
            fl = _elev_floor(tier, r['stokes'], attempt_tag(path))
            if abs(_f(r['elev_min_deg']) - fl) > 1e-6:
                probs.append(f"{k}: elev_min_deg {r['elev_min_deg']} != the {tier} {r['stokes']} mask floor {fl}"
                             + (' (pre-d8beb0d literal)' if fl == ELEV_RECORD_LITERAL else ''))
        except (KeyError, AttributeError):
            pass
    if not rows:
        probs.append('no image row (a search that entered epochs writes one per epoch; gate E33)')
    units = {(r['lst'], r['scope'], r['stokes']) for r in rows}
    if stokes and any(u[2] != stokes for u in units):
        probs.append(f'Stokes {sorted({u[2] for u in units})} != the unit Stokes {stokes}')
    if cell and any(f'{u[0]}_{u[1]}' != cell for u in units):
        probs.append('rows of another cell')
    rd = run_dir_of(path)
    att, lab = attempt_of(path), label_of(path)
    nframes = 0
    if rd:
        for (lst, scope, st) in units:
            _mine, extraction, both = refused_dates(rd, st)
            dates = {r['date'] for r in rows if r['stokes'] == st}
            hit = sorted(dates & (both | extraction))
            if hit:
                probs.append(f'Stokes {st}: image rows on refused dates {hit[:5]}')
            old_gen = (attempt_tag(path) or '99') < FRAME_EVERY_EPOCH_UTC
            sanity_only = set()
            if old_gen:
                for kind in (f'sanity_{st}',):
                    for p in refusal_files(rd).get(kind, []):
                        with open(p, newline='') as fh:
                            sanity_only |= {r[2] for r in csv.reader(fh) if len(r) > 2}
            for d in sorted(dates | extraction):
                if os.path.exists(frame_path(att, lab, lst, scope, d, st)):
                    nframes += 1
                elif d in sanity_only:
                    continue
                else:
                    probs.append(f'no search frame for {st} {d} ({os.path.basename(frame_path(att, lab, lst, scope, d, st))})')
            # frames = image rows exactly (verify_p3s3_stage.py:109-131): a frame of this Stokes with neither an image row nor
            # an extraction refusal is an extra frame (the count comparison fails it there)
            fdir = os.path.dirname(frame_path(att, lab, lst, scope, '0000-00-00', st))
            fre = re.compile(r'^(\d{4}-\d\d-\d\d(?:T\d\d)?)_' + st + r'_redo_resid_' + re.escape(scope) + r'\.fits$')
            on_disk = {m.group(1) for m in (fre.match(n) for n in (os.listdir(fdir) if os.path.isdir(fdir) else ())) if m}
            extra = sorted(on_disk - dates - extraction)
            if extra:
                probs.append(f'{len(extra)} {st} search frame(s) without an image row or an extraction refusal: {extra[:5]}')
    # STALE (verify_p3s3_stage.py:106-107; p3s3_stale_exclusions.py:25-27): an epoch searched (image row) on a night
    # data_decisions.csv excludes NOW - the merge gate refuses such an attempt
    for (lst, scope, st) in units:
        bad = _excluded_now(lst, scope, {r['date'][:10] for r in rows if r['stokes'] == st}, tier)
        if bad:
            probs.append(f'STALE: Stokes {st} searched night(s) data_decisions excludes now: {bad[:5]}')
    if probs:
        return _fail(probs, len(rows))
    return True, len(rows), f'{len(rows)} image rows, {nframes} search frames'


# ---- refusal record ---------------------------------------------------------------------------------------------------------
@content_test('refusal_record')
def parses(path, **ctx):
    """refusals/<lst>_<scope>_<kind>_excluded.csv (no header): every row has its kind's column count (or, for a kind in
    REFUSAL_EARLIER_NCOL, the count of its rows written before the format change), the file's lst/scope, a date of the cell's
    epoch grammar; a searchtier row's stage-1 rms is above its threshold (as written, 0.1 mJy) and its rule, when it has one,
    is 'ratio<x>xmedian' (production exopipe 702dd68); the refused dates have no gate-ledger row of the Stokes the refusal
    removes (not the RECORD_ONLY_KINDS records: an append may search a night its base recorded, once data_decisions admits
    it)."""
    m = _REF.match(os.path.basename(path))
    if not m:
        return False, None, f'name is not <lst>_<scope>_<kind>_excluded.csv: {os.path.basename(path)}'
    kind = m['kind']
    if kind not in REFUSAL_KINDS:
        return False, None, f'unknown refusal kind {kind!r} (known: {", ".join(sorted(REFUSAL_KINDS))})'
    ncol, which = REFUSAL_KINDS[kind]
    ncols = {ncol} | ({REFUSAL_EARLIER_NCOL[kind]} if kind in REFUSAL_EARLIER_NCOL else set())
    with open(path, newline='') as f:
        rows = [r for r in csv.reader(f) if r]
    probs = []
    for i, r in enumerate(rows):
        if len(r) not in ncols:
            probs.append(f'row {i + 1}: {len(r)} columns, kind {kind} has {ncol}'
                         + (f' ({REFUSAL_EARLIER_NCOL[kind]} in its earlier rows)' if kind in REFUSAL_EARLIER_NCOL else ''))
            continue
        if (r[0], r[1]) != (m['lst'], m['scope']):
            probs.append(f'row {i + 1}: lst/scope {r[0]}/{r[1]} != file {m["lst"]}/{m["scope"]}')
        if not _DATE.match(r[2]):
            probs.append(f'row {i + 1}: bad date {r[2]!r}')
        if which == 'row' and r[3] not in ('I', 'V'):
            probs.append(f'row {i + 1}: Stokes {r[3]!r}')
        if kind == 'searchtier':
            if not _f(r[4]) >= _f(r[5]):
                probs.append(f'row {i + 1}: stage-1 rms {r[4]!r} is not above its threshold {r[5]!r}')
            if len(r) == ncol and not _SEARCHTIER_RULE.match(r[6]):
                probs.append(f'row {i + 1}: rule {r[6]!r} is not ratio<x>xmedian')
    if not rows:
        probs.append('empty refusal ledger (the engine opens one only to write a refusal)')
    rd = run_dir_of(path)
    led = os.path.join(rd, 'v_deep_ledger.csv') if rd and kind not in RECORD_ONLY_KINDS else None
    if led and os.path.exists(led):
        lrows = _rows(led)[1]
        for r in rows:
            if len(r) not in ncols:
                continue
            sts = ('I', 'V') if which == 'both' else ((r[3],) if which == 'row' else (which,))
            n = sum(1 for x in lrows if x['date'] == r[2] and x['stokes'] in sts)
            if n:
                probs.append(f'{r[2]} refused ({kind}) but {n} gate-ledger rows of Stokes {"/".join(sts)}')
    if probs:
        return _fail(probs, len(rows))
    return True, len(rows), f'{len(rows)} {kind} refusals'


# ---- B-02 step 3b: the frame noise and the table's pulsar decisions ---------------------------------------------------------
def _median9(v):
    from exopipe.phase3.lib.search import removal_rules as RR      # the engine's own median (as a FITS card gives it back)
    return RR.median_noise(v)


@content_test('frame_noise')
def medians_and_cards(path, **ctx):
    """frame_noise.csv (B-02 step 3b): FNOISE_COLS, one cell; one row per (Stokes, night); admitted and population 0/1, a
    population night admitted with a positive noise; per Stokes a population night carries the population's median, an
    admitted night outside it (an admitted or appended night) the median of the population and itself, any other the
    population's median; no admitted V frame above the recorded factor (vsanity_x) times the V noise rule's median it was
    judged with, one median in the file (an append is judged with its base's); every row's archived search frame exists and
    carries IMGRMS = noise_jy (when measured); every image-ledger row of the run has its frame's row."""
    cols, rows = _rows(path)
    if cols != FNOISE_COLS:
        return False, len(rows), f'header != FNOISE_COLS: {cols[:6]}...'
    probs, seen = [], set()
    for r in rows:
        k = (r['stokes'], r['date'])
        if k in seen:
            probs.append(f'{k[0]} {k[1]}: two rows')
        seen.add(k)
        if r['stokes'] not in ('I', 'V') or not _DATE.match(r['date']):
            probs.append(f'bad Stokes/date {r["stokes"]!r} {r["date"]!r}')
        if r['admitted'] not in ('0', '1') or r['population'] not in ('0', '1'):
            probs.append(f'{k[0]} {k[1]}: admitted/population {r["admitted"]!r}/{r["population"]!r}')
        if r['population'] == '1' and not (r['admitted'] == '1' and _f(r['noise_jy']) > 0):
            probs.append(f'{k[0]} {k[1]}: in the population, not admitted with a noise')
        if r['vsanmed_jy'] and r['stokes'] != 'V':
            probs.append(f'{k[0]} {k[1]}: a V noise-rule median on an I row')
        if bool(r['vsanmed_jy']) != bool(r['vsanity_x']):
            probs.append(f'{k[0]} {k[1]}: a V noise-rule median without its factor, or the reverse')
        elif r['stokes'] == 'V' and r['admitted'] == '1' and r['vsanmed_jy'] and _f(r['noise_jy']) > _f(r['vsanity_x']) * _f(r['vsanmed_jy']):
            probs.append(f'V {r["date"]}: admitted with noise {r["noise_jy"]} above {r["vsanity_x"]} x its median '
                         f'{r["vsanmed_jy"]} (the V noise rule refuses it)')
    if len({(r['lst'], r['scope']) for r in rows}) > 1:
        probs.append('rows of more than one cell')
    vs = {r['vsanmed_jy'] for r in rows if r['stokes'] == 'V' and r['vsanmed_jy']}
    if len(vs) > 1:                              # the base's run measured it; every append is judged with the base's
        probs.append(f'{len(vs)} different V noise-rule medians in one attempt')
    elif vs:                                     # ... so no admitted V frame of the attempt escapes it
        miss = sorted(r['date'] for r in rows if r['stokes'] == 'V' and r['admitted'] == '1' and not r['vsanmed_jy'])
        if miss:
            probs.append(f'{len(miss)} admitted V frame(s) without the V noise-rule median the attempt holds, e.g. {miss[:3]}')
    for st in ('I', 'V'):
        R = [r for r in rows if r['stokes'] == st]
        pop = [_f(r['noise_jy']) for r in R if r['population'] == '1']
        m = _median9(pop) if pop else None
        for r in R:
            own = r['population'] == '0' and r['admitted'] == '1' and _f(r['noise_jy']) > 0
            want = _median9(pop + [_f(r['noise_jy'])]) if own else m
            got = _f(r['tblnoise_jy']) if r['tblnoise_jy'] != '' else None
            if want != got:
                probs.append(f'{st} {r["date"]}: tblnoise {got} != {want} (the population of {len(pop)}'
                             + (' and the night)' if own else ')'))
    att, lab = attempt_of(path), label_of(path)
    nfr = 0
    if att and rows:
        from astropy.io import fits
        for r in rows:
            fp = frame_path(att, lab, r['lst'], r['scope'], r['date'], r['stokes'])
            if not os.path.exists(fp):
                probs.append(f'{r["stokes"]} {r["date"]}: no archived search frame {os.path.basename(fp)}')
                continue
            nfr += 1
            if r['noise_jy'] != '':
                h = fits.getheader(fp)
                if h.get('IMGRMS') is None or float(h['IMGRMS']) != _f(r['noise_jy']):
                    probs.append(f'{r["stokes"]} {r["date"]}: frame IMGRMS {h.get("IMGRMS")} != noise_jy {r["noise_jy"]}')
    img = os.path.join(os.path.dirname(path), 'v_deep_images.csv')
    if os.path.exists(img):
        miss = sorted({(x['stokes'], x['date']) for x in _rows(img)[1]} - seen)
        if miss:
            probs.append(f'{len(miss)} image-ledger rows without their frame\'s row, e.g. {miss[:3]}')
    if probs:
        return _fail(probs, len(rows))
    return True, len(rows), f'{len(rows)} frames ({nfr} found); population ' + ', '.join(
        f'{st} {sum(1 for r in rows if r["stokes"] == st and r["population"] == "1")}' for st in ('I', 'V'))


@content_test('gate_population')
def frozen_gate_population(path, cell=None, **ctx):
    """gate_population.csv (lib/search/gate_population: the populations a whole-cell search judged by, read by an append -
    the beam gate at both cadences, and at 10 minutes the search-tier gate per Stokes and the V sanity gate, whose "nights"
    are window keys): COLS; one cell (``cell`` when given); per gate one content (the rows of a base and of a re-written backfill agree);
    n = the nights listed (dates, no repeat); 'applied' exactly when n reaches min_nights, with a positive finite median,
    'below-minimum' / 'off' without one ('off': no nights); the attempt's beam refusals (refusals/<cell>_beam_excluded.csv)
    were made with the row's factor, and none recorded under 'below-minimum' or 'off'."""
    from exopipe.phase3.lib.search import gate_population as GP
    cols, rows = _rows(path)
    if cols != GP.COLS:
        return False, len(rows), f'header != gate_population.COLS: {cols[:6]}...'
    probs = []
    cells = {(r['lst'], r['scope']) for r in rows}
    if len(cells) > 1 or (cell and cells and cells != {tuple(cell.split('_', 1))}):
        probs.append(f'rows of {sorted(cells)[:3]} (one cell{", " + cell if cell else ""})')
    for r in rows:
        nights = [x for x in r['nights'].split(';') if x]
        k = f'{r["lst"]} {r["scope"]} {r["gate"]}'
        if r['gate'] not in GP.GATES or r['outcome'] not in GP.OUTCOMES or r['basis'] not in GP.BASES:
            probs.append(f'{k}: gate / outcome / basis {r["gate"]!r} / {r["outcome"]!r} / {r["basis"]!r}')
            continue
        if str(len(nights)) != r['n'] or len(set(nights)) != len(nights) or not all(_DATE.match(d) for d in nights):
            probs.append(f'{k}: n {r["n"]} != {len(nights)} distinct dated nights')
        med, mn = _f(r['median']) if r['median'] else None, int(_f(r['min_nights']) or 0)
        if r['outcome'] == 'applied' and not (len(nights) >= mn and med is not None and med > 0 and med == med):
            probs.append(f'{k}: applied with {len(nights)} nights (minimum {mn}) and median {r["median"]!r}')
        if r['outcome'] != 'applied' and (r['median'] or (r['outcome'] == 'below-minimum' and len(nights) >= mn)
                                          or (r['outcome'] == 'off' and nights)):
            probs.append(f'{k}: {r["outcome"]} with {len(nights)} nights (minimum {mn}) and median {r["median"]!r}')
    for g in {r['gate'] for r in rows}:
        if len({tuple(r[c] for c in GP._KEY) for r in rows if r['gate'] == g}) > 1:
            probs.append(f'{g}: rows of different content in one attempt')
    beam = [r for r in rows if r['gate'] == 'beam']
    if beam and cells:
        lst, scope = sorted(cells)[0]
        bp = os.path.join(os.path.dirname(path), 'refusals', f'{lst}_{scope}_beam_excluded.csv')
        if os.path.exists(bp):
            with open(bp, newline='') as fh:
                br = [x for x in csv.reader(fh) if x and x[0] == lst and x[1] == scope]
            if br and beam[0]['outcome'] != 'applied':
                probs.append(f'{len(br)} beam refusal(s) recorded under outcome {beam[0]["outcome"]}')
            badf = [x[2] for x in br if len(x) >= 5 and _f(x[4]) != _f(beam[0]['factor'])]
            if badf:
                probs.append(f'beam refusal(s) of {badf[:3]} made with another factor than {beam[0]["factor"]}')
    if probs:
        return _fail(probs, len(rows))
    return True, len(rows), f'{len(rows)} frozen gate population row(s)'


@content_test('table_pulsar_decisions')
def named_and_ledgered(path, **ctx):
    """table_pulsar_decisions.csv (B-02 step 3b): TPD_COLS; Stokes I/V, a night, cut 0/1 and a note naming an ATNF pulsar - 'is the
    parent' on a kept detection, 'NOT exempting' on a cut one; each row's detection (Stokes, night, pixel) has a gate-ledger row,
    veto 'table' exactly when cut."""
    cols, rows = _rows(path)
    if cols != TPD_COLS:
        return False, len(rows), f'header != TPD_COLS: {cols[:6]}...'
    probs = []
    for r in rows:
        k = f'{r["stokes"]} {r["date"]} ({r["pix_x"]},{r["pix_y"]})'
        if r['stokes'] not in ('I', 'V') or not _DATE.match(r['date']) or r['cut'] not in ('0', '1'):
            probs.append(f'{k}: bad Stokes/date/cut {r["cut"]!r}')
            continue
        if 'ATNF pulsar' not in r['note']:
            probs.append(f'{k}: the note names no pulsar')
        elif r['cut'] == '0' and 'is the parent' not in r['note']:
            probs.append(f'{k}: kept without the pulsar-is-the-parent note')
        elif r['cut'] == '1' and 'NOT exempting' not in r['note']:
            probs.append(f'{k}: cut without the not-exempting note')
    if not rows:
        probs.append('empty (the engine writes the file only with a decision)')
    rd = run_dir_of(path)
    led = os.path.join(rd, 'v_deep_ledger.csv') if rd else None
    if led and os.path.exists(led):
        ix = {}
        for x in _rows(led)[1]:
            ix.setdefault((x['stokes'], x['date'], x['pix_x'], x['pix_y']), []).append(x['veto_reason'])
        for r in rows:
            vs = ix.get((r['stokes'], r['date'], r['pix_x'], r['pix_y']))
            if not vs:
                probs.append(f'{r["stokes"]} {r["date"]} ({r["pix_x"]},{r["pix_y"]}): no gate-ledger row')
            elif r['cut'] == '1' and 'table' not in vs:
                probs.append(f'{r["stokes"]} {r["date"]} ({r["pix_x"]},{r["pix_y"]}): cut, ledger {vs}')
            elif r['cut'] == '0' and all(v == 'table' for v in vs):
                probs.append(f'{r["stokes"]} {r["date"]} ({r["pix_x"]},{r["pix_y"]}): kept, ledger {vs}')
    if probs:
        return _fail(probs, len(rows))
    return True, len(rows), f'{len(rows)} decisions, {sum(r["cut"] == "1" for r in rows)} cut'


# ---- the search-time bright-source table's record (production exopipe 55b0263) ----------------------------------------------
@content_test('search_table_record')
def verdicts_applied_as_ruled(path, tier='deep', **ctx):
    """search_table_record.csv (deep_search.STR_COLS; production exopipe 55b0263, the refactor chat's 2026-10-18 decision):
    every verdict of the search's bright-source table with the inputs it was given. STR_COLS; one cell; Stokes I/V, a night of
    the epoch grammar, verdict and applied 0/1; a cut verdict carries the table's cut note ('table: parent S=...', not an
    exemption) and positive inputs, a kept one no note or an exemption's; applied as the attempt was searched - deep under
    production's rules 0 (record-only: the publish conformance applies the table once, with its derivation inputs), deep under
    the B-02 rules ('once-in-search', the attempt's search_rules.json) 1 (the search is the table's one application), the
    10-minute tier (tier 'sublst') 1; every row has its detection's gate-ledger row, and on the nights the record holds the
    ledger's 'table' rows are exactly its applied cut verdicts (Stokes, night, pixel; an append's base searched before the
    record existed has ledger rows the record cannot hold)."""
    import collections
    cols, rows = _rows(path)
    if cols != STR_COLS:
        return False, len(rows), f'header != STR_COLS: {cols[:6]}...'
    probs = []
    cells = sorted({(r['lst'], r['scope']) for r in rows})
    if len(cells) > 1:
        probs.append(f'{len(cells)} cells in one record: {cells[:3]}')
    want = '1' if tier == 'sublst' else None
    att = attempt_of(path)
    if want is None and att:
        from exopipe.phase3.lib.search import removal_rules as RR
        try:
            want = '1' if RR.attempt_rules(att)['rules'] == 'once-in-search' else '0'
        except RuntimeError as e:
            probs.append(str(e))
    applied_cuts, nights = collections.Counter(), set()
    for r in rows:
        k = f'{r["stokes"]} {r["date"]} ({r["pix_x"]},{r["pix_y"]})'
        if (r['stokes'] not in ('I', 'V') or not _DATE.match(r['date']) or r['verdict'] not in ('0', '1')
                or r['applied'] not in ('0', '1')):
            probs.append(f'{k}: bad Stokes / night / verdict / applied ({r["verdict"]!r}, {r["applied"]!r})')
            continue
        nights.add((r['stokes'], r['date']))
        cut_note = 'table: parent S=' in r['note'] and 'table-exempt' not in r['note']
        if r['verdict'] == '1' and not (cut_note and _f(r['frame_rms_jy']) > 0 and _f(r['beam_deg']) > 0):
            probs.append(f'{k}: a cut verdict without the table\'s cut note or with non-positive inputs ({r["note"][:60]!r}, '
                         f'rms {r["frame_rms_jy"]!r}, beam {r["beam_deg"]!r})')
        if r['verdict'] == '0' and r['note'] and 'table-exempt' not in r['note']:
            probs.append(f'{k}: kept with a note that is not an exemption ({r["note"][:60]!r})')
        if want is not None and r['applied'] != want:
            probs.append(f'{k}: applied {r["applied"]}, the attempt\'s rules apply {want}')
        if r['verdict'] == '1' and r['applied'] == '1':
            applied_cuts[(r['stokes'], r['date'], r['pix_x'], r['pix_y'])] += 1
    if not rows:
        probs.append('empty (the engine writes the file only with a verdict)')
    rd = run_dir_of(path)
    led = os.path.join(rd, 'v_deep_ledger.csv') if rd else None
    if led and os.path.exists(led):
        keys, table = set(), collections.Counter()
        for x in _rows(led)[1]:
            kx = (x['stokes'], x['date'], x['pix_x'], x['pix_y'])
            keys.add(kx)
            if x['veto_reason'] == 'table' and (x['stokes'], x['date']) in nights:
                table[kx] += 1
        miss = [r for r in rows if (r['stokes'], r['date'], r['pix_x'], r['pix_y']) not in keys]
        if miss:
            probs.append(f'{len(miss)} verdicts without their detection\'s gate-ledger row, e.g. '
                         f'{[(r["stokes"], r["date"], r["pix_x"], r["pix_y"]) for r in miss[:3]]}')
        if table != applied_cuts:
            probs.append(f'the ledger\'s \'table\' rows ({sum(table.values())}) are not the record\'s applied cuts '
                         f'({sum(applied_cuts.values())}): e.g. {sorted((table - applied_cuts) + (applied_cuts - table))[:3]}')
    if probs:
        return _fail(probs, len(rows))
    return True, len(rows), (f'{len(rows)} verdicts, {sum(r["verdict"] == "1" for r in rows)} cut, '
                             f'{sum(r["applied"] == "1" for r in rows)} applied')


# ---- candidate stamps -------------------------------------------------------------------------------------------------------
@content_test('candidate_stamps')
def match_frame(path, **ctx):
    """candidates/ (one directory per catalogue row): <cid>_resid.fits, <cid>_master.fits and <cid>.png present; the resid
    header passes validate_header(standard), CANDID = cand_id; the stamp's pixels equal the searched frame's pixels at their
    position (NaN as 0; the stamp keeps the residual's NaN, the frame zeroes them) and the catalogue pixel lies at the stamp
    centre +/- 1 px (sublst_fit_positions.py F1)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')          # astropy FITSFixedWarning (4-axis WCS on a 2-D image) on every stamp
        from astropy.utils.exceptions import AstropyWarning
        warnings.simplefilter('ignore', AstropyWarning)
        return _match_frame(path)


def _match_frame(path):
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS
    from exopipe.common.validate_fits_headers import validate_header
    from exopipe.phase3.lib.search.candidates import cutout_half_pix
    rd = run_dir_of(path)
    if not rd:
        return False, None, 'not inside transients/Run_<label>/'
    cat = os.path.join(rd, 'master_catalog.csv')
    rows = _rows(cat)[1] if os.path.exists(cat) else []
    dirs = sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))) if os.path.isdir(path) else []
    probs = []
    want = [r['cand_id'] for r in rows]
    if sorted(want) != dirs:
        probs.append(f'{len(dirs)} stamp directories != {len(want)} catalogue rows '
                     f'(missing {sorted(set(want) - set(dirs))[:3]}, extra {sorted(set(dirs) - set(want))[:3]})')
    att, lab = attempt_of(path), label_of(path)
    frames = {}
    ok = 0
    for r in rows:
        cid = r['cand_id']
        d = os.path.join(path, cid)
        fr, fm, fp = (os.path.join(d, f'{cid}_resid.fits'), os.path.join(d, f'{cid}_master.fits'), os.path.join(d, f'{cid}.png'))
        miss = [os.path.basename(x) for x in (fr, fm, fp) if not os.path.exists(x)]
        if miss:
            probs.append(f'{cid}: missing {miss}')
            continue
        try:
            rep = validate_header(fr, level='standard')
            if not rep.ok:
                probs.append(f'{cid}: resid header fails validate_header(standard): {"; ".join(rep.errors)[:120]}')
            with fits.open(fr, memmap=False) as h:
                sh = h[0].header
                sd = np.asarray(h[0].data, dtype=np.float64)
            if str(sh.get('CANDID', '')) != cid:
                probs.append(f'{cid}: CANDID {sh.get("CANDID")!r}')
            # deep stamps are 5 x 5 deg (verify_p3s3_stage.py:168-177): a legacy 81-px box where the frame's pixel scale
            # needs more pixels fails
            if cid[1:2] == 'D' and sh.get('NAXIS1') == 81 and 2 * cutout_half_pix(sh) + 1 > 81:
                probs.append(f'{cid}: legacy 81-px stamp, not 5 deg ({2 * cutout_half_pix(sh) + 1} px needed)')
            fpath = frame_path(att, lab, r['lst'], r['band'], r['date'], r['stokes'])
            if fpath not in frames:
                if not os.path.exists(fpath):
                    frames[fpath] = None
                else:
                    with fits.open(fpath, memmap=True) as h:
                        frames[fpath] = (h[0].header.copy(), np.squeeze(h[0].data))
            if frames[fpath] is None:
                probs.append(f'{cid}: searched frame missing ({os.path.basename(fpath)})')
                continue
            fh, fd = frames[fpath]
            x0 = float(fh['CRPIX1']) - float(sh['CRPIX1'])
            y0 = float(fh['CRPIX2']) - float(sh['CRPIX2'])
            if abs(x0 - round(x0)) > 1e-6 or abs(y0 - round(y0)) > 1e-6:
                probs.append(f'{cid}: stamp offset ({x0}, {y0}) not whole pixels')
                continue
            x0, y0 = int(round(x0)), int(round(y0))
            ny, nx = sd.shape
            sub = np.asarray(fd[y0:y0 + ny, x0:x0 + nx], dtype=np.float64)
            if sub.shape != sd.shape or not np.array_equal(np.nan_to_num(sd, nan=0.0), np.nan_to_num(sub, nan=0.0)):
                probs.append(f'{cid}: stamp pixels differ from the searched frame at ({x0}, {y0})')
                continue
            px, py = [float(v) for v in WCS(fh).celestial.world_to_pixel_values(_f(r['ra']), _f(r['dec']))]
            cx, cy = round(px) - x0, round(py) - y0
            hx, hy = (nx - 1) / 2.0, (ny - 1) / 2.0
            edge = x0 == 0 or y0 == 0 or x0 + nx >= fd.shape[1] or y0 + ny >= fd.shape[0]
            if not edge and (abs(cx - hx) > 1 or abs(cy - hy) > 1):
                probs.append(f'{cid}: catalogue pixel ({cx}, {cy}) not at the stamp centre ({hx}, {hy}) +/- 1 px')
                continue
            ok += 1
        except Exception as e:      # an unreadable stamp is a failed stamp
            probs.append(f'{cid}: {type(e).__name__}: {e}')
    if probs:
        return _fail(probs, len(rows))
    return True, len(rows), f'{ok} stamps = their searched frames, centred'


# ---- v110 verdicts ----------------------------------------------------------------------------------------------------------
@content_test('fp_gate_verdicts')
def every_survivor(path, catalog=None, **ctx):
    """V_top100_fp_gate_p3.csv: a known fp_prediction on every row (no survivor unverdicted, gate E30); with the V catalogue
    (context `catalog`, or master_catalog.csv beside a v110_out/ sitting in an attempt): every catalogue row's fp_prediction
    equals the verdict of its (date, ra, dec) and no catalogue row carries a removed verdict ('FP')."""
    cols, rows = _rows(path)
    probs = []
    for c in ('cand_id', 'ra', 'dec', 'date', 'fp_prediction'):
        if c not in cols:
            return False, len(rows), f'column {c} missing'
    ver = {}
    for r in rows:
        if r['fp_prediction'] not in FP_VERDICTS:
            probs.append(f"{r['cand_id']}: fp_prediction {r['fp_prediction']!r}")
        ver[(r['date'], round(_f(r['ra']), 5), round(_f(r['dec']), 5))] = r['fp_prediction']
    if not rows:
        probs.append('no verdict row (v110 ran only when V survivors existed)')
    if catalog and os.path.exists(catalog):
        from exopipe.phase3.config import CFG
        removes = set(CFG.search_fpgate.REMOVES)
        for r in _rows(catalog)[1]:
            if r['stokes'] != 'V':
                continue
            k = (r['date'], round(_f(r['ra']), 5), round(_f(r['dec']), 5))
            if r['fp_prediction'] in removes:
                probs.append(f"{r['cand_id']} carries the removed verdict {r['fp_prediction']}")
            if k in ver and ver[k] != r['fp_prediction']:
                probs.append(f"{r['cand_id']}: fp_prediction {r['fp_prediction']!r} != verdict {ver[k]!r}")
    if probs:
        return _fail(probs, len(rows))
    return True, len(rows), f'{len(rows)} verdicts'


# ---- detection positions (fit stage sidecar) --------------------------------------------------------------------------------
@content_test('detection_positions')
def fitted(path, tier='deep', attempt=None, **ctx):
    """fit_positions.csv: cand_id + DET_COLS + frame, frame_sha1, fit_frac_px; the same cand_id set and order as the attempt's
    master_catalog.csv; pos_kind in {gaussian, template, peak}; peak_ra/peak_dec = the catalogue ra/dec; at most 1 % of fitted
    rows within 0.01 px of a pixel centre (deep_fit_positions.py:146-150; the tier's own constants: search_fit for deep,
    search_<tier>.FIT_ON_CENTRE_* otherwise); frame_sha1 = the searched frame on disk; a tier with FIT_BLANKED (sub-LST, error
    model A, sublst_fit_positions.py:56): those columns blank and pos_stat > 0. Reports the peak-fallback fraction (pos_kind
    'peak') per frame."""
    from exopipe.phase3.config import CFG
    cols, rows = _rows(path)
    want = ['cand_id'] + det_cols() + FIT_EXTRA
    if cols != want:
        return False, len(rows), f'header != cand_id + DET_COLS + {FIT_EXTRA}: {cols[:5]}...'
    # the tier's own fit constants: deep = search_fit; sub-LST = search_sublst.FIT_* (error model A blanks FIT_BLANKED)
    if tier == 'deep':
        on_px, on_max, blanked = CFG.search_fit.ON_CENTRE_PX, CFG.search_fit.ON_CENTRE_MAX_FRAC, ()
    else:
        sec = getattr(CFG, f'search_{tier}', None)
        if sec is None or not hasattr(sec, 'FIT_ON_CENTRE_PX'):
            return False, len(rows), f'no fit constants for tier {tier!r} (config section search_{tier})'
        on_px, on_max, blanked = sec.FIT_ON_CENTRE_PX, sec.FIT_ON_CENTRE_MAX_FRAC, tuple(getattr(sec, 'FIT_BLANKED', ()))
    att = attempt or os.path.dirname(path)
    summ = os.path.join(os.path.dirname(path), 'fit_positions_summary.json')
    if not attempt and os.path.exists(summ):     # a replay sidecar names the (read-only) attempt it fitted
        import json
        att = json.load(open(summ)).get('attempt') or att
    cats = sorted(glob.glob(os.path.join(att, 'transients', 'Run_*', 'master_catalog.csv')))
    probs = []
    crow = _rows(cats[0])[1] if cats else []
    if [r['cand_id'] for r in crow] != [r['cand_id'] for r in rows]:
        probs.append(f'cand_id set/order differs from the catalogue ({len(rows)} vs {len(crow)} rows)')
    cbyid = {r['cand_id']: r for r in crow}
    on_ctr = nfit = 0
    shas = {}
    per_frame = {}
    for r in rows:
        k = r['pos_kind']
        if k not in FIT_KINDS:
            probs.append(f"{r['cand_id']}: pos_kind {k!r}")
        c = cbyid.get(r['cand_id'])
        if c and (abs(_f(r['peak_ra']) - _f(c['ra']) % 360.0) > 6e-6 or abs(_f(r['peak_dec']) - _f(c['dec'])) > 6e-6):
            probs.append(f"{r['cand_id']}: peak_ra/peak_dec {r['peak_ra']},{r['peak_dec']} != catalogue {c['ra']},{c['dec']}")
        pf = per_frame.setdefault(r['frame'], [0, 0])
        pf[0] += 1
        if k == 'peak':
            pf[1] += 1
        if k in ('gaussian', 'template'):
            nfit += 1
            if r['fit_frac_px'] not in ('', None) and _f(r['fit_frac_px']) < on_px:
                on_ctr += 1
        if k in FIT_KINDS and not (math.isfinite(_f(r['pos_ra'])) and math.isfinite(_f(r['pos_dec']))):
            probs.append(f"{r['cand_id']}: {k} position not finite")
        if blanked:
            full = [c for c in blanked if r.get(c)]
            if full:
                probs.append(f"{r['cand_id']}: {tier} {full} not blank (error model A)")
            if not _f(r['pos_stat_maj_arcsec']) > 0:
                probs.append(f"{r['cand_id']}: pos_stat not > 0")
        fp = os.path.join(att, r['frame'])
        if r['frame'] not in shas:
            shas[r['frame']] = sha1_file(fp) if os.path.exists(fp) else None
        if shas[r['frame']] is None:
            probs.append(f"{r['cand_id']}: frame {r['frame']} missing")
        elif shas[r['frame']] != r['frame_sha1']:
            probs.append(f"{r['cand_id']}: frame sha1 {r['frame_sha1'][:8]} != the frame on disk {shas[r['frame']][:8]}")
    if nfit and on_ctr / nfit > on_max:
        probs.append(f'{on_ctr} of {nfit} fitted rows ({100.0 * on_ctr / nfit:.2f} %) within {on_px} px of a pixel centre')
    npk = sum(v[1] for v in per_frame.values())
    worst = max(((v[1] / v[0], f) for f, v in per_frame.items() if v[0]), default=(0.0, ''))
    value = round(npk / len(rows), 4) if rows else 0.0
    if probs:
        return _fail(probs, value)
    return True, value, (f'{len(rows)} rows on {len(per_frame)} frames; peak fallback {npk} ({100.0 * value:.1f} %), worst frame '
                         f'{100.0 * worst[0]:.0f} % ({os.path.basename(worst[1])}); {on_ctr}/{nfit} fitted rows on a pixel centre')
