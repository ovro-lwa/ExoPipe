"""The production conformance module (exopipe_ops p3s2_conform.py) IN PLACE, with the ONE population statistic of its
bright-parent table step RECORDED where it is computed, or READ FROZEN (per-hour 10-minute catalog; rulings Gregg 2026-11-12,
R-GREGG-20261112-PER-HOUR-STEPS-SIX-RULINGS (1) "Agreed." and (2) "Approved.": a new night is judged by statistics frozen
from earlier nights, and earlier verdicts stay; DECISION-20261113-FROZEN-POPULATIONS-RECORDED-WHERE-COMPUTED-NO-STATE-STEP:
a frozen statistic is recorded by the step that computes it).

    python conform_noise.py <p3s2_conform.py> <run dir>      (env EXOPIPE3_CONFORM_NOISE_*)

THE STATISTIC: the table's noise - per (LST, band, Stokes), the median over the catalog rows the module reads of
|peak flux / global S/N| (Jy; p3s2_conform.py `_RMS`). The table step hands it to the bright-parent table as the noise its
thresholds are scaled by. It is a statistic of the POPULATION of rows: a run over one night's rows alone would take the
median of that night's handful of rows, so a night judged alone and the same night judged in an LST-wide run would be cut
differently. (At the 10-minute cadence the module's two frame-noise gates never act - design section 2.2a, verified
2026-11-13 - so this is its only population statistic there besides the pulsar groups, which read rows, not a statistic.)

SELF-CONTAINED (stdlib only): the child's `exopipe` package can be production's, so this file imports nothing from the
branch. It reads the module's source, applies the ONE PATCH below - exact text that must occur EXACTLY ONCE, and the census of
the name it replaces (a new use of `_RMS`, or a second assignment, would bypass the record) - and runs the patched source
as the module's own __main__ (its file name, its directory first on sys.path, its argv), so every other line of
production's module runs unchanged.

EXOPIPE3_CONFORM_NOISE_MODE:
  passthrough  the hook returns production's own values: the patched module must write byte-identical files (the check
               that the patching and the __main__ harness change nothing)
  record       production's own values, and the record EXOPIPE3_CONFORM_NOISE_RECORD is written when the module has ended
               (one row per LST, band, Stokes: the number of finite values, the median as repr(float), the module's
               path-normalized sha1; a module that exits early leaves the record of the last run that ended). An LST-wide
               run.
  frozen       the values of the record, whole (every recorded key, so a look-up of a key this run has no row of answers
               as it did LST-wide). REFUSES: no record; a record of another module version; a module version outside
               EXOPIPE3_CONFORM_NOISE_SHA1 (the path-normalized sha1s the frozen mode was checked for - another version
               may hold a second population statistic, which this run would compute over one night's rows); a (LST, band,
               Stokes) of this run's rows that the record does not hold (no earlier night gives its noise: run the
               LST-wide conformance, which records it). The medians of this run's own rows are printed beside the frozen
               ones and not used.
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

# (name, old text, new text): the old text must occur EXACTLY ONCE in the module
PATCHES = [
    ('P1-table-noise',
     "_RMS = {k: float(np.nanmedian(v)) for k, v in _RMS.items()}\n",
     "_RMS = __CONFORM_NOISE__.medians(_RMS, np)   # EXOPIPE3 CONFORM-NOISE P1: the table's noise, recorded or frozen\n"),
]
# how often the name occurs in the module the patch was written for (exopipe_ops p3s2_conform.py, sha1 48f69a39...,
# 2026-11-13): its initialisation, the per-row append, the median (twice on that line), three look-ups by the table step;
# two assignments (the initialisation and the median)
CENSUS = {'_RMS': {7}, '_RMS = ': {2}, '_RMS.get(': {3}}

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
        raise Refused('the conformance module is not the one the table-noise patch was written for - ' + '; '.join(bad)
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


class Hook:
    """__CONFORM_NOISE__ in the module's namespace."""

    def __init__(self, mode, record=None, module_sha1='', norm_sha1='', accepted=()):
        if mode not in MODES:
            raise Refused(f'EXOPIPE3_CONFORM_NOISE_MODE={mode!r} is not one of {MODES} - REFUSING')
        self.mode, self.record, self.module_sha1, self.norm_sha1 = mode, record, module_sha1, norm_sha1
        self.called = 0
        self.frozen, self.record_sha1, self.summary, self.pending = None, '', '', None
        if mode == 'record' and not record:
            raise Refused('EXOPIPE3_CONFORM_NOISE_RECORD is not set: the record mode needs the path it writes - REFUSING')
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


def run(module, args, mode, record=None, accepted=()):
    """Run the patched module as __main__ (its file name, its directory first on sys.path, sys.argv = [module] + args)."""
    module = os.path.abspath(module)
    with open(module) as f:
        src = f.read()
    sha1 = hashlib.sha1(src.encode()).hexdigest()
    norm = normalized_sha1(module, src)
    patched, names = patch(src)
    hook = Hook(mode, record, sha1, norm, accepted)
    mod = types.ModuleType('__main__')
    mod.__file__ = module
    mod.__dict__['__CONFORM_NOISE__'] = hook
    mod.__dict__['__builtins__'] = __builtins__
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [os.path.dirname(module)] + [p for p in sys.path if os.path.abspath(p or '.') != here]
    sys.argv = [module] + list(args)
    sys.modules['__main__'] = mod
    print(f'CONFORM-NOISE start: module {module} sha1 {sha1} (path-normalized {norm}); patch {",".join(names)}; mode '
          f'{mode}; record {record or "-"}', flush=True)
    exec(compile(patched, module, 'exec'), mod.__dict__)
    if hook.called != 1:
        raise Refused(f'the module ended without computing the table noise ({hook.called} calls of the patched line): '
                      f'nothing was recorded or frozen - REFUSING')
    if hook.pending is not None:
        write_record(record, hook.pending[0], hook.pending[1], norm)
        hook.summary = f'{len(hook.pending[0])} (LST, band, Stokes) recorded in {record}'
    print(f'CONFORM-NOISE done: module sha1 {sha1}; mode {mode}; {hook.summary}', flush=True)


def main(argv):
    if len(argv) < 3:
        sys.exit('usage: conform_noise.py <p3s2_conform.py> <run dir>  (env EXOPIPE3_CONFORM_NOISE_*)')
    try:
        run(argv[1], argv[2:], os.environ.get('EXOPIPE3_CONFORM_NOISE_MODE', ''),
            os.environ.get('EXOPIPE3_CONFORM_NOISE_RECORD') or None,
            accepted=os.environ.get('EXOPIPE3_CONFORM_NOISE_SHA1', '').split(','))
    except Refused as e:
        print(f'CONFORM-NOISE REFUSING: {e}', flush=True)
        sys.exit(2)


if __name__ == '__main__':
    main(sys.argv)
