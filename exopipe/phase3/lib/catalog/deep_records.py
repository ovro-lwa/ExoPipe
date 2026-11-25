"""The record chain of the deep catalogue stages (DESIGN 11.5; the lesson of deep defects D9/D10).

Every deep stage is a unit ``<label>.<stage>`` (``<label>.associate.<K>``, ``<label>.fork.<S>``) whose record lists the
fingerprint of every input (core record ``inputs``) and of every output (``meta.out_fp``: core fingerprint plus, for a
CSV up to 16 MB, its row count and a digest of its id column). A stage reads only the files its predecessors' records
list, never a glob, and refuses unless every predecessor record is CURRENT:

- the record is done (an associate key or a fork with no rows may be no-input);
- every recorded input and output still has its recorded fingerprint, OR a later done stage of this run (in time) that
  rewrote it in place recorded the fingerprint it has now (master_catalog.csv: merge -> conform -> fitpos; the tier
  survivors: identity), OR - survivors CSVs only - the forced step (outside the catalog step: it adds forced_* columns
  in place, ops/p3s3_forced_phot.py:70-89) has a done record newer than the writer and the rows and ids are unchanged;
- and, recursively, every predecessor of that record is current.

Cache files the in-place modules keep across passes on purpose (frame_times.csv, frame_beam_ratio.csv,
fit_positions_cache.csv, stale_attempts.csv) are recorded but do not decide currency.
"""
import csv
import glob
import hashlib
import json
import os
import re

from exopipe.phase3.core import record


def csv_identity(path, limit=16 * 2 ** 20):
    """{'rows': n, 'ids': sha1 of the sorted id column} of a CSV up to ``limit`` bytes (id = source_id, else cand_id,
    else identity_id); {} for anything else."""
    try:
        if not path.endswith('.csv') or os.path.getsize(path) > limit:
            return {}
        with open(path, newline='') as f:
            rd = csv.DictReader(f)
            col = next((c for c in ('source_id', 'cand_id', 'identity_id') if c in (rd.fieldnames or [])), None)
            ids = [r.get(col, '') for r in rd] if col else None
        if ids is None:
            return {}
        return {'rows': len(ids), 'ids': hashlib.sha1('\n'.join(sorted(ids)).encode()).hexdigest()[:16], 'id_col': col}
    except (OSError, ValueError, csv.Error):
        return {}


def out_fingerprint(path):
    fp = record.fingerprint(path)
    ci = csv_identity(path)
    if ci:
        fp['csv'] = ci
    return fp


def files_under(d):
    """Every regular file under directory ``d`` (sorted absolute paths): the outputs of a stage that wrote a fresh dir."""
    out = []
    for dp, dn, fn in os.walk(d):
        dn.sort()
        for x in sorted(fn):
            p = os.path.join(dp, x)
            if os.path.isfile(p) and not os.path.islink(p):
                out.append(os.path.abspath(p))
    return out


class Chain:
    """The records of one deep run label under one product root, and the currency check."""

    def __init__(self, out_root, label, units, preds, may_be_noinput=(), forced_step='forced'):
        self.out = out_root
        self.label = label
        self.units = list(units)               # every unit name the chain may hold
        self.preds = dict(preds)               # unit -> [predecessor units]
        self.may_be_noinput = set(may_be_noinput)
        self.forced_step = forced_step
        self.reload()

    def reload(self):
        self.recs = {u: record.load_record(self.out, 'catalog', u) for u in self.units}
        self._memo = {}
        self._fp = {}

    def get(self, unit):
        return self.recs.get(unit)

    def _cur(self, p):
        if p not in self._fp:
            self._fp[p] = record.fingerprint(p)
        return self._fp[p]

    def forced_records(self):
        """Done records of the forced step under the same product root: [(finished, unit)]."""
        out = []
        for p in sorted(glob.glob(os.path.join(self.out, '_records', self.forced_step, '*.json'))):
            try:
                rec = json.load(open(p))
            except (OSError, ValueError):
                continue
            if rec.get('status') == 'done' and not rec.get('dry_run'):
                out.append((rec.get('finished') or '', rec.get('unit')))
        return sorted(out)

    def _rewritten_later(self, p, rec, cur):
        """True when a done record finished after ``rec`` (or ``rec`` itself, for a file it read and rewrote) recorded the
        fingerprint ``p`` has now."""
        own = ((rec.get('meta') or {}).get('out_fp') or {}).get(p)
        if own is not None and record.same_fingerprint(own, cur):
            return 'its own rewrite'
        t0 = rec.get('finished') or ''
        for u, other in self.recs.items():
            if other is None or other is rec or other.get('status') != 'done':
                continue
            if (other.get('finished') or '') < t0:
                continue
            ofp = ((other.get('meta') or {}).get('out_fp') or {}).get(p)
            if ofp is not None and record.same_fingerprint(ofp, cur):
                return f'rewritten by {u}'
        return None

    def _external(self, p, fp, rec):
        """A survivors CSV rewritten in place by the forced step after ``rec``: accepted when the rows and ids are those
        recorded (forced adds columns only)."""
        if not (p.endswith('_sources_survivors.csv') and fp.get('csv')):
            return None
        t0 = rec.get('finished') or ''
        newer = [u for t, u in self.forced_records() if t >= t0]
        if not newer:
            return None
        ci = csv_identity(p)
        if ci and ci.get('rows') == fp['csv'].get('rows') and ci.get('ids') == fp['csv'].get('ids'):
            return f'forced columns added by {self.forced_step} record {newer[-1]} (rows and ids unchanged)'
        return None

    def _file_current(self, p, fp, rec):
        cur = self._cur(p)
        if record.same_fingerprint(fp, cur):
            return True, ''
        why = self._rewritten_later(p, rec, cur) or self._external(p, fp, rec)
        if why:
            return True, why
        return False, ('missing' if cur.get('missing') else 'changed since its record')

    def check(self, unit, _stack=()):
        """(current?, reason) of ``unit``'s record, recursively through its predecessors."""
        if unit in self._memo:
            return self._memo[unit]
        if unit in _stack:
            return True, 'cycle'
        rec = self.recs.get(unit)
        if rec is None:
            res = (False, f'{unit}: no record (run that stage first)')
        elif rec.get('status') != 'done' and not (rec.get('status') == 'no-input' and unit in self.may_be_noinput):
            res = (False, f"{unit}: record status {rec.get('status')} ({rec.get('reason') or ''})")
        else:
            res = (True, f"{unit}: done {rec.get('finished')}")
            caches = set((rec.get('meta') or {}).get('caches') or ())
            items = list((rec.get('inputs') or {}).items()) + \
                [(p, fp) for p, fp in ((rec.get('meta') or {}).get('out_fp') or {}).items() if p not in caches]
            for p, fp in items:
                if p in caches:
                    continue
                ok, why = self._file_current(p, fp, rec)
                if not ok:
                    res = (False, f'{unit}: {p} {why}')
                    break
            if res[0]:
                for pu in self.preds.get(unit, ()):
                    ok, why = self.check(pu, _stack + (unit,))
                    if not ok:
                        res = (False, f'{unit} <- {why}')
                        break
        self._memo[unit] = res
        return res

    def outputs(self, unit, suffix=None, under=None):
        """The output paths ``unit``'s record lists (optionally those ending with ``suffix`` / under directory ``under``)."""
        rec = self.recs.get(unit) or {}
        ps = sorted(((rec.get('meta') or {}).get('out_fp') or {}))
        if suffix:
            ps = [p for p in ps if p.endswith(suffix)]
        if under:
            u = os.path.abspath(under).rstrip('/') + '/'
            ps = [p for p in ps if p.startswith(u)]
        return ps

    def meta(self, unit, key, default=None):
        return ((self.recs.get(unit) or {}).get('meta') or {}).get(key, default)


# ---- the parent snapshot of a deep run's latest table pass (the per-hour steps; B-59) ------------------------------------
_PASS_START = re.compile(r'^\[(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)\] pass start \(parents (\S+)\)\s*$')
_PASS_END = re.compile(r'^\[(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)\] (FORK SERVE DONE|PASS FAILED|REFUSED)')


def _log_pass(path):
    """(started, parent dir, end) of the last pass in production's pass log (catalog.DEEP_PASS_LOG): end is
    'FORK SERVE DONE' (finished), 'PASS FAILED ...' / 'REFUSED ...' (the first line that ended it) or None (no end line:
    running or killed); None when the log names no pass."""
    if not os.path.exists(path):
        return None
    last = None
    with open(path, errors='replace') as f:
        for line in f:
            m = _PASS_START.match(line)
            if m:
                last = [m.group(1), m.group(2), None]
                continue
            m = _PASS_END.match(line)
            if m and last is not None and last[2] is None:
                last[2] = line.strip()[len(m.group(1)) + 3:][:300]
    return tuple(last) if last else None


def _v2_pass(run_dir, label, last_stage):
    """(started, parent dir, end) of the v2 deep chain's pass in ``run_dir`` (lib/catalog/deep_chain.py): its
    ``<label>.preflight`` record (meta.parent_dir); end is 'done' when the record of the last stage ``last_stage`` is done
    or no-input and finished after the preflight, else the state that stops it; None when there is no done preflight."""
    pre = record.load_record(run_dir, 'catalog', f'{label}.preflight')
    if not pre or pre.get('status') != 'done' or pre.get('dry_run') or not (pre.get('meta') or {}).get('parent_dir'):
        return None
    t0 = pre.get('finished') or pre.get('started') or ''
    last = record.load_record(run_dir, 'catalog', f'{label}.{last_stage}')
    if last and last.get('status') in ('done', 'no-input') and (last.get('finished') or '') >= t0:
        end = 'done'
    else:
        end = (f'{label}.{last_stage} record ' + (f"{last.get('status')} at {last.get('finished')}" if last else 'missing')
               + f' (preflight finished {t0})')
    return t0, pre['meta']['parent_dir'], end


def pass_parent_dir(run_dir, label, cfg=None):
    """(parent dir, how) of the deep run ``run_dir``'s LATEST table pass - the snapshot its tables were built against
    (B-59: a per-hour step reads the parent catalogues of the pass whose tables it extends, never a fixed union).
    Two records name a pass: the v2 deep chain's preflight record and production's pass log (catalog.DEEP_PASS_LOG,
    ops/fork_serve_p3s5_20261008.sh: it exports P3_PARENT_DIR = that pass's snapshot to every stage); the pass that
    started last decides. Gate P-PASS refuses (ValueError naming why) when no record names a pass, when the latest pass
    did not finish (failed, refused, running or killed: the run's tables may be mid-pass) or when its snapshot is not a
    directory on this host. ``how`` names the record."""
    if cfg is None:
        from exopipe.phase3.config import CFG as cfg
    c = cfg.catalog
    run_dir = os.path.abspath(run_dir)
    log = os.path.join(run_dir, c.DEEP_PASS_LOG)
    cands = []
    v2 = _v2_pass(run_dir, label, c.DEEP_STAGES[-1])
    if v2:
        cands.append(v2 + (f'the v2 preflight record {record.record_path(run_dir, "catalog", label + ".preflight")}',
                           'done'))
    prod = _log_pass(log)
    if prod:
        cands.append(prod + (f'the production pass log {log}', 'FORK SERVE DONE'))
    if not cands:
        raise ValueError(f'P-PASS: no table pass of deep run {run_dir} is recorded (no done {label}.preflight record, no '
                         f'"pass start (parents DIR)" line in {log}) - run the deep table pass first')
    t0, par, end, how, ok = max(cands, key=lambda x: x[0])
    if end != ok:
        raise ValueError(f'P-PASS: the latest table pass of deep run {run_dir} (started {t0}, parents {par}; {how}) did not '
                         f'finish: {end or "no end line (running or killed)"} - its tables may be mid-pass; finish or '
                         f're-run the pass first')
    if not os.path.isdir(par):
        raise ValueError(f'P-PASS: the parent snapshot {par} of the latest table pass of deep run {run_dir} (started {t0}; '
                         f'{how}) is not a directory on this host')
    return par, f'{how}: pass started {t0}'


def resolve_parent_arg(value, run_dir, label, cfg=None):
    """(parent dir, how, refusal) of a parent-directory argument: the keyword catalog.PARENT_FROM_PASS resolves to the deep
    run's latest finished table pass (pass_parent_dir; ``refusal`` = the P-PASS reason when it does not resolve, dir None);
    any other value is returned as given (how None; the caller's own gate judges it)."""
    if cfg is None:
        from exopipe.phase3.config import CFG as cfg
    if value != cfg.catalog.PARENT_FROM_PASS:
        return value, None, None
    if not run_dir:
        return None, None, f'P-PASS: --parent-dir {value} needs the deep run (--run LABEL or --src-run DIR)'
    if not label:
        b = os.path.basename(os.path.abspath(run_dir))
        label = b[4:] if b.startswith('Run_') else b
    try:
        par, how = pass_parent_dir(run_dir, label, cfg)
    except ValueError as e:
        return None, None, str(e)
    return par, how, None
