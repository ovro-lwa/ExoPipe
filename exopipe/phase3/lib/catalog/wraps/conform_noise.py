"""The production conformance module (exopipe_ops p3s2_conform.py) IN PLACE, with the ONE population statistic of its
bright-parent table step RECORDED where it is computed, or READ FROZEN (per-hour 10-minute catalog; rulings Gregg 2026-11-12,
R-GREGG-20261112-PER-HOUR-STEPS-SIX-RULINGS (1) "Agreed." and (2) "Approved.": a new night is judged by statistics frozen
from earlier nights, and earlier verdicts stay; DECISION-20261113-FROZEN-POPULATIONS-RECORDED-WHERE-COMPUTED-NO-STATE-STEP:
a frozen statistic is recorded by the step that computes it), and with the ROWS its pulsar step reads from other nights
recorded and read back the same way (THE PULSAR EVIDENCE, below).

    python conform_noise.py <p3s2_conform.py> <run dir>      (env EXOPIPE3_CONFORM_NOISE_*)

THE STATISTIC: the table's noise - per (LST, band, Stokes), the median over the catalog rows the module reads of
|peak flux / global S/N| (Jy; p3s2_conform.py `_RMS`). The table step hands it to the bright-parent table as the noise its
thresholds are scaled by. It is a statistic of the POPULATION of rows: a run over one night's rows alone would take the
median of that night's handful of rows, so a night judged alone and the same night judged in an LST-wide run would be cut
differently. (At the 10-minute cadence the module's two frame-noise gates never act - design section 2.2a, verified
2026-11-13 - so this is its only population statistic there besides the pulsar groups, which read rows, not a statistic.)

THE PULSAR EVIDENCE: the module's pulsar step (Gregg 2026-09-28: "the multi-integration sources that are inconsistent with
the pulsar position are likely to be background and can be removed. Keep the real ones!") groups the detections within one
frame beam of an ATNF pulsar ACROSS NIGHTS and keeps a detection the bright-parent table would remove when its group's
combined position agrees with the pulsar. A run over one night's rows has no other night to group with, so it would
remove a pulsar's detection that the LST-wide run keeps (measured 2026-11-13, LST 17h, night 2025-05-08: 1 of 740
verdicts, PSR J2008+2513). So the LST-wide run RECORDS, after its pulsar step, every row of its catalog that lies within
its own frame beam of an ATNF pulsar - identity, position, S/N, the frame beam, the daytime verdict of its frame, and the
position error the pulsar step uses (from the window-tie table as it stood, which later stages rewrite) - and the name of
the pulsar its table note names. The frozen run appends the OTHER nights' recorded rows to the module's rows for the
pulsar step only: they join a pulsar's group when the module's own tests admit them (within their recorded frame beam of
that pulsar, their frame not daytime, their cell not excluded NOW - exclusion is tested live, never read from the
record), they receive no verdict (ruling (1): earlier verdicts stay), and they are taken out again before the module
applies its verdicts, so no later line and no output file holds them.

SELF-CONTAINED (stdlib only): the child's `exopipe` package can be production's, so this file imports nothing from the
branch. It reads the module's source, applies the PATCHES below - each an exact text that must occur EXACTLY ONCE, with the
census of the names they touch (a new use of `_RMS`, a second assignment, or a second reader of the pulsar step's rows
would bypass the records) - and runs the patched source as the module's own __main__ (its file name, its directory first on sys.path, its argv), so every other line of
production's module runs unchanged.

EXOPIPE3_CONFORM_NOISE_MODE:
  passthrough  the hook returns production's own values: the patched module must write byte-identical files (the check
               that the patching and the __main__ harness change nothing)
  record       production's own values, and the record EXOPIPE3_CONFORM_NOISE_RECORD is written when the module has ended
               (one row per LST, band, Stokes: the number of finite values, the median as repr(float), the module's
               path-normalized sha1; a module that exits early leaves the record of the last run that ended), with the
               pulsar evidence EXOPIPE3_CONFORM_NOISE_EVIDENCE (one row per detection within its frame beam of an ATNF
               pulsar, or named by a table note with one). An LST-wide run.
  frozen       the values of the record, whole (every recorded key, so a look-up of a key this run has no row of answers
               as it did LST-wide). REFUSES: no record; a record of another module version; a module version outside
               EXOPIPE3_CONFORM_NOISE_SHA1 (the path-normalized sha1s the frozen mode was checked for - another version
               may hold a second population statistic, which this run would compute over one night's rows); a (LST, band,
               Stokes) of this run's rows that the record does not hold (no earlier night gives its noise: run the
               LST-wide conformance, which records it). The medians of this run's own rows are printed beside the frozen
               ones and not used. The run is ONE NIGHT's (EXOPIPE3_CONFORM_NOISE_NIGHT; a row of another night refuses)
               and reads the pulsar evidence EXOPIPE3_CONFORM_NOISE_EVIDENCE of the same LST-wide run (no evidence
               record, one of another module version, or an evidence row a group needs that holds no position error:
               refusals).
The records are written and read at the 10-minute cadence only (P3_CADENCE=10min; the pulsar step's frame beam and position
error are that cadence's); another cadence refuses in the record and frozen modes.
Machine lines (stdout): 'CONFORM-NOISE start ...', 'CONFORM-NOISE done ...', 'CONFORM-NOISE REFUSING: ...' (exit 2).
"""
import csv
import hashlib
import io
import math
import os
import sys
import types

MODES = ('passthrough', 'record', 'frozen')
COLS = ['lst', 'band', 'stokes', 'n', 'median_jy', 'module_sha1']
FILE = 'conform_table_noise.csv'            # the record's name in the catalog directory of an LST-wide run
# the pulsar evidence: the catalog's own strings (cand_id ... snr), the frame beam (deg, repr(float); '' = the module's
# beam look-up refused the frame), the frame's daytime verdict ('' = admitted, as the module's `_DAY`), the position error
# the pulsar step uses (arcsec, repr(float); '' = no window tie of the row), the pulsar its table note names ('' = none)
EVIDENCE_COLS = ['cand_id', 'lst', 'band', 'stokes', 'date', 'ra', 'dec', 'snr', 'beam_deg', 'day', 'sig_arcsec',
                 'note_pulsar', 'module_sha1']
EVIDENCE_FILE = 'conform_pulsar_evidence.csv'
MARK = '__conform_noise_evidence__'         # the key an evidence row carries among the module's rows: (beam deg, error arcsec)
CADENCE = '10min'

# (name, old text, new text): the old text must occur EXACTLY ONCE in the module
PATCHES = [
    ('P1-table-noise',
     "_RMS = {k: float(np.nanmedian(v)) for k, v in _RMS.items()}\n",
     "_RMS = __CONFORM_NOISE__.medians(_RMS, np)   # EXOPIPE3 CONFORM-NOISE P1: the table's noise, recorded or frozen\n"),
    # the pulsar step. P2: before its loop over the pulsars (the rows' unit vectors are built, so the module's own search
    # never returns an appended row) the other nights' recorded rows join the rows, and the loop also visits the pulsars
    # only their table notes name. P3: a pulsar's detections of this night are joined by its recorded ones. P4: an
    # evidence row's position error is the recorded one. P5: the appended rows leave before the verdicts are applied;
    # the LST-wide run takes its record here.
    ('P2-pulsar-loop',
     "for _pn, _idx0 in _psr_rows.items():\n",
     "__CONFORM_NOISE__.evidence(raw)   # EXOPIPE3 CONFORM-NOISE P2: other nights' recorded detections near pulsars\n"
     "for _pn, _idx0 in __CONFORM_NOISE__.pulsars(_psr_rows).items():\n"),
    ('P3-pulsar-near',
     "    if len(near) < 2: continue\n",
     "    near = __CONFORM_NOISE__.near(_pn, near, raw, globals())   # EXOPIPE3 CONFORM-NOISE P3\n"
     "    if len(near) < 2: continue\n"),
    ('P4-pulsar-error',
     "sig=_sig_row(raw[i])",
     "sig=__CONFORM_NOISE__.sig(raw[i], _sig_row)"),
    ('P5-pulsar-release',
     "for i, _vs in _PSR_VERDICTS.items():\n",
     "__CONFORM_NOISE__.release(raw, globals())   # EXOPIPE3 CONFORM-NOISE P5: evidence rows out; the LST-wide record\n"
     "for i, _vs in _PSR_VERDICTS.items():\n"),
]
# how often the name occurs in the module the patch was written for (exopipe_ops p3s2_conform.py, sha1 48f69a39...,
# 2026-11-13): its initialisation, the per-row append, the median (twice on that line), three look-ups by the table step;
# two assignments (the initialisation and the median)
# the pulsar step (sha1 6b3fe52d..., 2026-11-13): `_psr_rows` - its initialisation, the append, the reset at another cadence,
# the loop, the summary line; `_PSR_VERDICTS` - its initialisation, the append, the loop that applies them; `_sig_row(` - its
# definition and its one call
CENSUS = {'_RMS': {7}, '_RMS = ': {2}, '_RMS.get(': {3}, '_psr_rows': {5}, '_PSR_VERDICTS': {3}, '_sig_row(': {2}}

LIVE_ROOT = '/lustre/gh/main'               # the live code root a frozen tree's rewrites point away from


class Refused(RuntimeError):
    pass


def normalized_sha1(module, src):
    """sha1 of the module's source with its own code root (a frozen tree's <root>/) replaced by the live root - the same
    module version gives the same value in every frozen tree (as wraps/builder_grouping.normalized_sha1)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(module)))
    if root != LIVE_ROOT:
        src = src.replace(root + '/', LIVE_ROOT + '/')
    return hashlib.sha1(src.encode()).hexdigest()


def patch(src):
    """(patched source, [names]) - REFUSES unless the patch's text occurs exactly once and the census holds."""
    bad = []
    for name, old, _new in PATCHES:
        n = src.count(old)
        if n != 1:
            bad.append(f'{name}: its text occurs {n} times (1 expected)')
    for tok, want in CENSUS.items():
        n = src.count(tok)
        if n not in want:
            bad.append(f'census {tok!r}: {n} occurrences ({sorted(want)} expected)')
    if bad:
        raise Refused('the conformance module is not the one the table-noise patches were written for - ' + '; '.join(bad)
                      + ' - re-port lib/catalog/wraps/conform_noise.py - REFUSING')
    out = src
    for name, old, new in PATCHES:
        out = out.replace(old, new)
    return out, [p[0] for p in PATCHES]


def read_record(path):
    """{(lst, band, stokes): (n, median_jy)}, the set of module sha1s of the record's rows. REFUSES a missing, empty,
    malformed or duplicated record."""
    if not path or not os.path.isfile(path):
        raise Refused(f'no table-noise record {path!r}: the LST-wide conformance writes it where it computes the noise '
                      f'(`exopipe3 catalog --tier sublst --stages conform`) - REFUSING')
    with open(path, 'rb') as f:
        data = f.read()
    rows = list(csv.DictReader(io.StringIO(data.decode('utf-8'), newline='')))
    if not rows or list(rows[0].keys()) != COLS:
        raise Refused(f'{path}: not a table-noise record (columns {COLS}, at least one row) - REFUSING')
    out, shas = {}, set()
    for r in rows:
        k = (r['lst'], r['band'], r['stokes'])
        try:
            n, med = int(r['n']), float(r['median_jy'])
        except (TypeError, ValueError):
            raise Refused(f'{path}: row {k} is not readable (n {r["n"]!r}, median {r["median_jy"]!r}) - REFUSING')
        if k in out:
            raise Refused(f'{path}: {k} twice - REFUSING')
        if n < 0 or (n > 0 and not (math.isfinite(med) and med > 0)) or (n == 0 and not math.isnan(med)):
            raise Refused(f'{path}: row {k} is inconsistent (n {n}, median {med!r}) - REFUSING')
        out[k] = (n, med)
        shas.add(r['module_sha1'])
    return out, shas, hashlib.sha1(data).hexdigest()


def write_record(path, values, counts, module_sha1):
    """The record, atomically: one row per key in sorted order, the median as repr(float) (it reads back to the same
    float)."""
    tmp = f'{path}.tmp.{os.getpid()}'
    with open(tmp, 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(COLS)
        for k in sorted(values):
            w.writerow([k[0], k[1], k[2], counts[k], repr(values[k]), module_sha1])
    os.replace(tmp, path)


def _num(x):
    """float(x) if it is a finite positive number, else None."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v > 0 else None


def read_evidence(path):
    """[row dict], the set of module sha1s of the rows, the file's sha1. A record with no row is a valid one (an LST with no
    detection near a pulsar). REFUSES a missing or malformed record, a cand_id twice, an unreadable beam or error."""
    if not path or not os.path.isfile(path):
        raise Refused(f'no pulsar-evidence record {path!r}: the LST-wide conformance writes it after its pulsar step '
                      f'(`exopipe3 catalog --tier sublst --stages conform`) - REFUSING')
    with open(path, 'rb') as f:
        data = f.read()
    rd = csv.DictReader(io.StringIO(data.decode('utf-8'), newline=''))
    rows = list(rd)
    if rd.fieldnames != EVIDENCE_COLS:
        raise Refused(f'{path}: not a pulsar-evidence record (columns {EVIDENCE_COLS}) - REFUSING')
    seen, shas = set(), set()
    for r in rows:
        if not r['cand_id'] or r['cand_id'] in seen:
            raise Refused(f'{path}: cand_id {r["cand_id"]!r} is empty or occurs twice - REFUSING')
        seen.add(r['cand_id'])
        for c in ('beam_deg', 'sig_arcsec'):
            if r[c] != '' and _num(r[c]) is None:
                raise Refused(f'{path}: {r["cand_id"]}: {c} {r[c]!r} is not a finite positive number - REFUSING')
        if not r['date'] or len(r['date']) < 10:
            raise Refused(f'{path}: {r["cand_id"]}: date {r["date"]!r} names no night - REFUSING')
        shas.add(r['module_sha1'])
    return rows, shas, hashlib.sha1(data).hexdigest()


def write_evidence(path, rows, module_sha1):
    """The pulsar-evidence record, atomically, in the order of the module's rows."""
    tmp = f'{path}.tmp.{os.getpid()}'
    with open(tmp, 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(EVIDENCE_COLS)
        for r in rows:
            w.writerow(list(r) + [module_sha1])
    os.replace(tmp, path)


class Hook:
    """__CONFORM_NOISE__ in the module's namespace."""

    def __init__(self, mode, record=None, module_sha1='', norm_sha1='', accepted=(), evidence=None, night=None):
        if mode not in MODES:
            raise Refused(f'EXOPIPE3_CONFORM_NOISE_MODE={mode!r} is not one of {MODES} - REFUSING')
        self.mode, self.record, self.module_sha1, self.norm_sha1 = mode, record, module_sha1, norm_sha1
        self.called = 0
        self.frozen, self.record_sha1, self.summary, self.pending = None, '', '', None
        # the pulsar evidence: steps counts the calls of P2 and P5; n0 is the number of the module's own rows; ev_rows the
        # other nights' recorded rows (frozen); ev_pending the rows to record (record); ev_used the recorded rows that joined
        self.evidence_path, self.night = evidence, night
        self.steps, self.n0 = {'evidence': 0, 'release': 0}, None
        self.ev_rows, self.ev_sha1, self.ev_pending, self.ev_used, self.ev_lookups, self.ev_named = [], '', None, set(), 0, 0
        if mode == 'record' and not record:
            raise Refused('EXOPIPE3_CONFORM_NOISE_RECORD is not set: the record mode needs the path it writes - REFUSING')
        if mode in ('record', 'frozen'):
            if not evidence:
                raise Refused(f'EXOPIPE3_CONFORM_NOISE_EVIDENCE is not set: the {mode} mode needs the pulsar-evidence record '
                              f'it {"writes" if mode == "record" else "reads"} - REFUSING')
            cad = os.environ.get('P3_CADENCE', 'deep')
            if cad != CADENCE:
                raise Refused(f'P3_CADENCE={cad!r}: the table-noise and pulsar-evidence records are written and read at the '
                              f'{CADENCE} cadence only - REFUSING')
        if mode == 'frozen':
            accepted = [a for a in accepted if a]
            if not accepted:
                raise Refused('EXOPIPE3_CONFORM_NOISE_SHA1 is empty: the module versions the frozen mode was checked for '
                              'must be named (config catalog_conform.NOISE_MODULE_SHA1) - REFUSING')
            if norm_sha1 not in accepted:
                raise Refused(f'the conformance module (path-normalized sha1 {norm_sha1}) is not one the frozen mode was '
                              f'checked for ({len(accepted)} accepted, config catalog_conform.NOISE_MODULE_SHA1): another '
                              f'version may hold a second statistic of the population of rows - check it and re-port '
                              f'lib/catalog/wraps/conform_noise.py - REFUSING')
            self.frozen, shas, self.record_sha1 = read_record(record)
            if shas != {norm_sha1}:
                raise Refused(f'{record}: written by module version(s) {sorted(shas)}, this run is {norm_sha1}: the noise '
                              f'was computed by other code - run the LST-wide conformance again - REFUSING')
            if not night or len(night) != 10 or night[4] != '-' or night[7] != '-':
                raise Refused(f'EXOPIPE3_CONFORM_NOISE_NIGHT={night!r} is not a night YYYY-MM-DD: the frozen mode judges one '
                              f'night\'s rows - REFUSING')
            rows, shas, self.ev_sha1 = read_evidence(evidence)
            if shas - {norm_sha1}:
                raise Refused(f'{evidence}: written by module version(s) {sorted(shas)}, this run is {norm_sha1}: the pulsar '
                              f'evidence was recorded by other code - run the LST-wide conformance again - REFUSING')
            for r in rows:
                if r['date'][:10] != night:                 # the night's own recorded rows are this run's rows again
                    e = dict(r)
                    e[MARK] = (_num(r['beam_deg']), _num(r['sig_arcsec']))
                    self.ev_rows.append(e)

    def medians(self, lists, np):
        """The module's `_RMS` after its own line: {key: float(np.nanmedian(values))} of ``lists`` (production's own
        expression), or the record's values (frozen)."""
        self.called += 1
        if self.called != 1:
            raise Refused('the table noise was computed twice in one run - REFUSING')
        own = {k: float(np.nanmedian(v)) for k, v in lists.items()}        # production's own expression, same order
        counts = {k: int(sum(1 for x in v if isinstance(x, float) and math.isfinite(x))) for k, v in lists.items()}
        if self.mode == 'passthrough':
            self.summary = f'{len(own)} (LST, band, Stokes) computed; nothing recorded'
            return own
        if self.mode == 'record':
            self.pending = (own, counts)        # written by run() when the module has ended: a refused conformance leaves
            return own                          # the record of the last one that ended
        missing = sorted(k for k in own if k not in self.frozen or self.frozen[k][0] == 0)
        if missing:
            raise Refused(f'{self.record}: no recorded noise for {len(missing)} (LST, band, Stokes) of this run\'s rows, '
                          f'e.g. {missing[:4]}: no earlier night gives it - run the LST-wide conformance, which records '
                          f'it (`exopipe3 catalog --tier sublst --stages conform`) - REFUSING')
        for k in sorted(own):
            n, med = self.frozen[k]
            print(f'CONFORM-NOISE frozen {k[0]} {k[1]} {k[2]}: recorded median {med!r} Jy over {n} rows (used); this '
                  f'run\'s own {counts[k]} rows give {own[k]!r} Jy (not used)', flush=True)
        self.summary = (f'{len(own)} (LST, band, Stokes) of this run read frozen from {self.record} (sha1 '
                        f'{self.record_sha1}; {len(self.frozen)} recorded)')
        return {k: v[1] for k, v in self.frozen.items()}    # every recorded key, as the LST-wide run held them

    # ---- the pulsar evidence (patches P2 - P5) ----------------------------------------------------------------------
    def _step(self, name):
        self.steps[name] += 1
        if self.steps[name] != 1 or (name == 'release' and self.steps['evidence'] != 1):
            raise Refused(f'the pulsar step ran out of order ({self.steps}): one loop over the pulsars, then the verdicts, '
                          f'is what the patches were written for - REFUSING')

    @staticmethod
    def _names(g, *names):
        miss = [n for n in names if n not in g]
        if miss:
            raise Refused(f'the conformance module holds no {", ".join(miss)} at its pulsar step - re-port lib/catalog/wraps/'
                          f'conform_noise.py - REFUSING')
        return [g[n] for n in names]

    def evidence(self, raw):
        """P2, before the module's loop over the pulsars. Frozen: every row of this run is the night's (else REFUSES), and
        the other nights' recorded rows are appended to the module's rows - after its unit vectors of the rows were built,
        so its own search for rows near a pulsar never returns one; `near` adds them, `release` removes them."""
        self._step('evidence')
        self.n0 = len(raw)
        if self.mode != 'frozen':
            return
        other = sorted({str(r.get('date', ''))[:10] for r in raw} - {self.night})
        if other:
            raise Refused(f'this run holds rows of {len(other)} other night(s), e.g. {other[:3]}: the frozen mode judges the '
                          f'rows of {self.night} alone (the night view of the search jobs) - REFUSING')
        ids = {r.get('cand_id') for r in raw}
        both = sorted(e['cand_id'] for e in self.ev_rows if e['cand_id'] in ids)
        if both:
            raise Refused(f'{self.evidence_path}: {len(both)} recorded row(s) of other nights carry a cand_id of this run, '
                          f'e.g. {both[:3]} - REFUSING')
        raw.extend(self.ev_rows)

    def pulsars(self, psr_rows):
        """P2, the loop's pulsars: the module's own (the pulsars this run's table notes name), and in the frozen mode also
        the pulsars only the other nights' notes name - the LST-wide run visits those too, and a detection of this night
        in the beam of one takes a verdict from it. The module's own dictionary is not changed."""
        if self.mode != 'frozen':
            return psr_rows
        out = dict(psr_rows)
        for name in sorted({e['note_pulsar'] for e in self.ev_rows if e['note_pulsar']} - set(out)):
            out[name] = []
        self.ev_named = len(out) - len(psr_rows)
        return out

    def near(self, pn, near, raw, g):
        """P3, a pulsar's detections as the module found them among this run's rows; frozen: joined by the other nights'
        recorded rows that pass the module's own tests (within their frame beam of this pulsar - separation by the module's
        own expression - their frame not daytime, their cell not excluded now). A pulsar with no detection of this night
        is left alone: its group would hold no row that takes a verdict."""
        if self.mode != 'frozen' or not len(near) or not self.ev_rows:
            return near
        psr, unit, np, f, is_excl = self._names(g, '_PSR_POS', '_unit', 'np', 'f', '_is_excl')
        pra, pdec = psr[pn]
        pu = unit(pra, pdec)
        cos15 = math.cos(math.radians(1.5))                 # the module's own first cut (every frame beam is < 1.5 deg)
        add = []
        for j in range(self.n0, len(raw)):
            e = raw[j]
            beam = e[MARK][0]
            if beam is None:
                continue                                    # the module's beam look-up refused the frame: never a neighbor
            dot = float(np.dot(pu, unit(f(e['ra']), f(e['dec']))))
            if not dot > cos15:
                continue
            if math.degrees(math.acos(max(-1.0, min(1.0, dot)))) > beam:
                continue
            if is_excl(e['lst'], e['band'], e['date']) or e['day']:
                continue
            add.append(j)
        if add:
            self.ev_lookups += 1
            self.ev_used.update(raw[j]['cand_id'] for j in add)
            print(f'CONFORM-NOISE evidence {pn}: {len(near)} detection(s) of {self.night} joined by {len(add)} recorded '
                  f'detection(s) of other nights', flush=True)
        return list(near) + add

    def sig(self, r, fn):
        """P4, the position error of a row of a pulsar's group: the module's own function, or an evidence row's recorded
        value. REFUSES an evidence row without one (the module refuses a row missing from its window-tie table)."""
        ev = r.get(MARK) if isinstance(r, dict) else None
        if ev is None:
            return fn(r)
        if ev[1] is None:
            raise Refused(f'{self.evidence_path}: {r["cand_id"]} joins a pulsar\'s group and holds no recorded position '
                          f'error (it had no window tie in the LST-wide run) - REFUSING')
        return ev[1]

    def release(self, raw, g):
        """P5, after the loop over the pulsars and before the verdicts are applied. Frozen: the appended rows leave (they
        hold no verdict: the module gives one only to a row its table step decided). Record: the rows to record are taken
        here, where the module's quantities stand (written by run() when the module has ended)."""
        self._step('release')
        if self.mode == 'frozen':
            del raw[self.n0:]
            return
        if len(raw) != self.n0:
            raise Refused(f'the module\'s rows changed inside the pulsar step ({self.n0} -> {len(raw)}) - REFUSING')
        if self.mode != 'record':
            return
        psr, rawu, unit, np, f, gi, day, sig_row, psr_rows = self._names(
            g, '_PSR_POS', '_RAWU', '_unit', 'np', 'f', '_G', '_DAY', '_sig_row', '_psr_rows')
        if len(rawu) != len(raw):
            raise Refused(f'the module\'s unit vectors ({len(rawu)}) are not its rows\' ({len(raw)}) - REFUSING')
        cos15 = math.cos(math.radians(1.5))
        beams, inbeam = {}, set()

        def beam_of(i):
            if i not in beams:
                r = raw[i]
                try:
                    beams[i] = _num(gi.target_beam(r['lst'], r['band'], r['date'], r['stokes']))
                except Exception:                           # noqa: BLE001  (the module skips such a row the same way)
                    beams[i] = None
            return beams[i]
        for pn in psr:                                      # the module's own search, for EVERY pulsar: a later night's
            pra, pdec = psr[pn]                             # notes may name a pulsar this run's notes do not
            pu = unit(pra, pdec)
            for i in np.where(rawu @ pu > cos15)[0]:
                i = int(i)
                if i in inbeam:
                    continue
                bm = beam_of(i)
                if bm is None:
                    continue
                sep = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(pu, unit(f(raw[i]['ra']), f(raw[i]['dec']))))))))
                if sep <= bm:
                    inbeam.add(i)
        named = {}
        for name, idx in psr_rows.items():
            for i in idx:
                named.setdefault(int(i), name)
        rows = []
        for i in sorted(inbeam | set(named)):
            r = raw[i]
            bm = beam_of(i)
            sg = ''
            if i in inbeam:
                try:
                    v = _num(sig_row(r))
                    sg = repr(v) if v is not None else ''
                except Exception:                           # noqa: BLE001  (no window tie of the row: the module would refuse
                    sg = ''                                 # only if a group needed it; the frozen run refuses the same)
            rows.append([r['cand_id'], r['lst'], r['band'], r['stokes'], r['date'], r['ra'], r['dec'], r['snr'],
                         repr(bm) if bm is not None else '', day.get((r['stokes'], r['lst'], r['band'], r['date']), 'x'),
                         sg, named.get(i, '')])
        self.ev_pending = rows


def run(module, args, mode, record=None, accepted=(), evidence=None, night=None):
    """Run the patched module as __main__ (its file name, its directory first on sys.path, sys.argv = [module] + args)."""
    module = os.path.abspath(module)
    with open(module) as f:
        src = f.read()
    sha1 = hashlib.sha1(src.encode()).hexdigest()
    norm = normalized_sha1(module, src)
    patched, names = patch(src)
    hook = Hook(mode, record, sha1, norm, accepted, evidence, night)
    mod = types.ModuleType('__main__')
    mod.__file__ = module
    mod.__dict__['__CONFORM_NOISE__'] = hook
    mod.__dict__['__builtins__'] = __builtins__
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [os.path.dirname(module)] + [p for p in sys.path if os.path.abspath(p or '.') != here]
    sys.argv = [module] + list(args)
    sys.modules['__main__'] = mod
    print(f'CONFORM-NOISE start: module {module} sha1 {sha1} (path-normalized {norm}); patch {",".join(names)}; mode '
          f'{mode}; record {record or "-"}; pulsar evidence {evidence or "-"}' + (f'; night {night}' if night else ''), flush=True)
    exec(compile(patched, module, 'exec'), mod.__dict__)
    if hook.called != 1:
        raise Refused(f'the module ended without computing the table noise ({hook.called} calls of the patched line): '
                      f'nothing was recorded or frozen - REFUSING')
    if hook.steps != {'evidence': 1, 'release': 1}:
        raise Refused(f'the module ended without its pulsar step ({hook.steps} calls of the patched lines): no evidence '
                      f'was recorded or read - REFUSING')
    if hook.pending is not None:
        if hook.ev_pending is None:
            raise Refused('the module ended without the rows of the pulsar-evidence record - REFUSING')
        write_evidence(evidence, hook.ev_pending, norm)
        write_record(record, hook.pending[0], hook.pending[1], norm)
        hook.summary = (f'{len(hook.pending[0])} (LST, band, Stokes) recorded in {record}; {len(hook.ev_pending)} detection(s) '
                        f'near ATNF pulsars recorded in {evidence}')
    if mode == 'frozen':
        hook.summary += (f'; pulsar evidence: {len(hook.ev_rows)} recorded detection(s) of other nights in {evidence} (sha1 '
                         f'{hook.ev_sha1}), {len(hook.ev_used)} joined a group of {night} in {hook.ev_lookups} pulsar '
                         f'look-up(s); {hook.ev_named} pulsar(s) named only by other nights\' table notes')
    print(f'CONFORM-NOISE done: module sha1 {sha1}; mode {mode}; {hook.summary}', flush=True)


def main(argv):
    if len(argv) < 3:
        sys.exit('usage: conform_noise.py <p3s2_conform.py> <run dir>  (env EXOPIPE3_CONFORM_NOISE_*)')
    try:
        run(argv[1], argv[2:], os.environ.get('EXOPIPE3_CONFORM_NOISE_MODE', ''),
            os.environ.get('EXOPIPE3_CONFORM_NOISE_RECORD') or None,
            accepted=os.environ.get('EXOPIPE3_CONFORM_NOISE_SHA1', '').split(','),
            evidence=os.environ.get('EXOPIPE3_CONFORM_NOISE_EVIDENCE') or None,
            night=os.environ.get('EXOPIPE3_CONFORM_NOISE_NIGHT') or None)
    except Refused as e:
        print(f'CONFORM-NOISE REFUSING: {e}', flush=True)
        sys.exit(2)


if __name__ == '__main__':
    main(sys.argv)
