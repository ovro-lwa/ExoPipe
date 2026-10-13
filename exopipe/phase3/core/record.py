"""Run records, fingerprints and resume decisions - the replacement for every `.done` marker.

Every step invocation is a ``Run``; every unit of work inside it (a cell, an LST, a frame batch) is a
``UnitRecord`` written atomically to ``<out>/_records/<step>/<unit>.json`` when the unit ends. A record holds:
the command line, code fingerprint (sha1 of every exopipe source file the step imported), config hash and overrides,
host and cores, inputs with fingerprints, the admitted night set (exclusion state), outputs with their product class,
every content test with its value, refusals and waivers.

Rules (CLAUDE.md 3, 4b; architecture 11.1; DESIGN 11.5):
- A unit is ``done`` only when it wrote at least one output and EVERY output passed its content test.
- A unit that declares its PLANNED outputs (``u.plan(path, cls)``) is done only when every planned output was written
  and passed, or a recorded item refusal names it (``u.refuse_item(path, reason)``: one image, band or window refused
  by a named gate while the unit completes). A planned output neither written nor refused is an error (exit 1); a unit
  whose every planned item was refused is refused (exit 2).
- "wrote 0" is never done: a unit with nothing to do ends ``no-input`` with a reason (exit 3).
- A gate refuses (``Refusal``, exit 2); it never warns and continues.
- Exit code of a run: 1 if any unit errored, else 2 if any unit was refused, else 3 only when EVERY unit had no input,
  else 0 (every unit done, skipped as valid, or recorded no-input).
- ``--resume`` skips a unit only when its record says done AND its inputs, admitted nights, code and config are
  unchanged AND every output still passes its content test now (``resume_check``). There are no marker files.
- Input identity: a FITS input is identified by its data digest (DATASUM) and processing cards (COADDOP, FLUXBAKE,
  FLUXVER, MDWARP, CONSTDIG, PBVER, NCOADD) read from the primary header; mtime is never identity when a digest
  exists. A record written from a dirty tree (commit '<sha>-dirty') is never resumed.
"""
import getpass
import hashlib
import json
import os
import socket
import sys
import time
import traceback

RECORD_VERSION = 2          # 2: FITS identity cards in input fingerprints, planned outputs, item refusals

FITS_SUFFIXES = ('.fits', '.fit', '.fts')
# Primary-header cards that identify a FITS input beside its data digest (DESIGN 11.5; critique_rulings 6.1-6.2):
# the co-add operator, flux bake and its version, master dewarp, constituent digest, PB version, constituents.
FITS_PROCESSING_CARDS = ('COADDOP', 'FLUXBAKE', 'FLUXVER', 'MDWARP', 'CONSTDIG', 'PBVER', 'NCOADD')


class Refusal(Exception):
    """A gate refused. Exit code 2. The message is the reason recorded."""


class NoInput(Exception):
    """Nothing to do for this unit (explicit, recorded). Exit code 3. Never a done state."""


EXIT_DONE, EXIT_ERROR, EXIT_REFUSED, EXIT_NO_INPUT = 0, 1, 2, 3


def utcnow():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def _fits_value(text):
    """Value of a FITS header card (the text after '= '): str, bool, int or float (raw text when unparseable)."""
    t = text.strip()
    if t.startswith("'"):
        out, i = [], 1
        while i < len(t):
            if t[i] == "'":
                if t[i + 1:i + 2] == "'":
                    out.append("'")
                    i += 2
                    continue
                break
            out.append(t[i])
            i += 1
        return ''.join(out).rstrip()
    v = t.split('/', 1)[0].strip()
    if v in ('T', 'F'):
        return v == 'T'
    for conv in (int, lambda x: float(x.replace('D', 'E'))):
        try:
            return conv(v)
        except ValueError:
            pass
    return v


def fits_primary_cards(path, max_blocks=400):
    """Cards of the PRIMARY header of a FITS file, read header-only (2880-byte blocks up to END; no astropy, no
    data read). {} for a file that does not start with SIMPLE. The first value of a repeated keyword is kept."""
    cards = {}
    with open(path, 'rb') as f:
        for n in range(max_blocks):
            blk = f.read(2880)
            if len(blk) < 2880 or (n == 0 and not blk.startswith(b'SIMPLE')):
                return cards if n else {}
            for i in range(0, 2880, 80):
                card = blk[i:i + 80].decode('ascii', 'replace')
                key = card[:8].strip()
                if key == 'END':
                    return cards
                if card[8:10] == '= ' and key not in cards:
                    cards[key] = _fits_value(card[10:])
    return cards


def fits_identity(path):
    """{'datasum': ..., 'cards': {...}} of a FITS file from its primary header: DATASUM (the data digest written at
    write time; omitted when absent, or '0' on a primary HDU without data) and the processing cards present."""
    try:
        cards = fits_primary_cards(path)
    except OSError:
        return {}
    out = {'cards': {k: cards[k] for k in FITS_PROCESSING_CARDS if k in cards}}
    ds = cards.get('DATASUM')
    if ds not in (None, '') and not (str(ds).strip() == '0' and cards.get('NAXIS', 0) == 0):
        out['datasum'] = str(ds).strip()
    return out


def fingerprint(path, hash_limit_bytes=16 * 2 ** 20):
    """Identity of an input file: size + mtime (ns) for all files, plus sha1 of the bytes for files up to
    ``hash_limit_bytes`` (CSV tables, small FITS). A FITS file (.fits/.fit/.fts) also carries its DATASUM and the
    processing cards of its primary header (``fits_identity``: a cheap header-only read), so a large FITS input is
    identified by its data digest, never by its modification time."""
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return {'missing': True}
    fp = {'size': st.st_size, 'mtime_ns': st.st_mtime_ns}
    if not os.path.isfile(path):
        return fp
    if st.st_size <= hash_limit_bytes:
        h = hashlib.sha1()
        with open(path, 'rb') as f:
            for blk in iter(lambda: f.read(1 << 20), b''):
                h.update(blk)
        fp['sha1'] = h.hexdigest()
    if path.lower().endswith(FITS_SUFFIXES):
        fp.update(fits_identity(path))
    return fp


def same_fingerprint(a, b):
    """Two fingerprints describe the same content.

    - sha1 of the bytes decides when both have one;
    - else a FITS data digest (DATASUM) decides when either has one: equal only when both carry the same DATASUM, the
      same processing cards and the same size - modification time is never identity when a digest exists;
    - else size + mtime."""
    if a.get('missing') or b.get('missing'):
        return False
    if 'sha1' in a and 'sha1' in b:
        return a['sha1'] == b['sha1']
    if 'datasum' in a or 'datasum' in b:
        return (a.get('datasum') == b.get('datasum') and (a.get('cards') or {}) == (b.get('cards') or {})
                and a.get('size') == b.get('size'))
    return a.get('size') == b.get('size') and a.get('mtime_ns') == b.get('mtime_ns')


def _legacy_roots():
    try:
        from exopipe.phase3.config import CFG
        return tuple(r.rstrip('/') + '/' for r in CFG.site.LEGACY_CODE_ROOTS)
    except Exception:
        return ()


def code_fingerprint():
    """sha1 of every source file the unit imported from exopipe AND from the legacy code roots (clv15, exopipe_ops)
    called in place during the refactor window - the code that produced this unit (DESIGN 11.5)."""
    out = {}
    roots = _legacy_roots()
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, '__file__', None)
        if not (name.startswith('exopipe') or (f and roots and os.path.abspath(f).startswith(roots))):
            continue
        if f and f.endswith('.py') and os.path.exists(f):
            out[name] = hashlib.sha1(open(f, 'rb').read()).hexdigest()[:12]
    return out


def git_commit(path):
    import subprocess
    try:
        c = subprocess.run(['git', '-C', path, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True,
                           timeout=10).stdout.strip()
        d = subprocess.run(['git', '-C', path, 'status', '--porcelain', '--untracked-files=no'],
                           capture_output=True, text=True, timeout=20).stdout.strip()
        return c + ('-dirty' if d else '')
    except Exception:
        return 'unknown'


def unruled_overrides(rec):
    """The overrides of RULED keys in a run record that carry no ledger ruling (allowed only into a non-production
    --out). install and serve refuse a product whose record lists any (DESIGN 11.4)."""
    return [o for o in (rec.get('ruled_overrides') or []) if not str(o.get('authority') or '').startswith('ruling:')]


def atomic_write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f'{path}.tmp.{os.getpid()}'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=str)
    os.replace(tmp, path)


def record_path(out_root, step, unit):
    return os.path.join(out_root, '_records', step, f'{unit}.json')


def load_record(out_root, step, unit):
    p = record_path(out_root, step, unit)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


class Run:
    """One invocation of one step."""

    def __init__(self, step, argv, out_root, cfg, dry_run=False, waivers=None, resume=False, fresh=False,
                 config_sections=None):
        self.step = step
        self.argv = list(argv)
        self.out_root = out_root
        self.cfg = cfg
        self.dry_run = dry_run
        self.resume = resume
        self.fresh = fresh
        self.waivers = list(waivers or [])        # [(test name, ruling text)]
        self.host = socket.gethostname()
        self.pid = os.getpid()
        self.started = utcnow()
        self.run_id = f"{step}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{self.host}_{self.pid}"
        pkg = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.commit = git_commit(pkg)
        self.units = []
        # the sections the step declares (CONFIG) plus the shared ones; None = every section (no CONFIG declared)
        self.config_sections = tuple(config_sections) if config_sections is not None else None

    @property
    def config_hash(self):
        """Hash of the config sections this step reads - the config identity --resume compares."""
        return self.cfg.hash(self.config_sections)

    def resume_check(self, unit, inputs, admitted, retest=True):
        """``resume_check`` for a unit of this run, with this step's config hash."""
        return resume_check(self.out_root, self.step, unit, inputs=inputs, admitted=admitted, cfg_hash=self.config_hash,
                            retest=retest)

    def header(self):
        return {'record_version': RECORD_VERSION, 'step': self.step, 'run_id': self.run_id, 'argv': self.argv,
                'host': self.host, 'pid': self.pid, 'user': getpass.getuser(), 'commit': self.commit,
                'config_hash': self.config_hash, 'config_hash_all': self.cfg.hash(),
                'config_sections': sorted(self.config_sections) if self.config_sections is not None else None,
                'site_sha1': getattr(self.cfg, 'site_sha1', None),
                # [key, repr(old), repr(new), source, authority]; authority: None (unruled key), 'ruling:<ledger ID>',
                # 'non-production out:<dir>' - install and serve refuse products whose ruled override has no ruling
                'config_overrides': [[o[0], repr(o[1]), repr(o[2]), o[3], o[4] if len(o) > 4 else None]
                                     for o in self.cfg.overrides],
                'ruled_overrides': self.cfg.ruled_overrides() if hasattr(self.cfg, 'ruled_overrides') else [],
                'waivers': [{'test': t, 'ruling': r} for t, r in self.waivers], 'dry_run': self.dry_run}

    def unit(self, name, **meta):
        u = UnitRecord(self, name, meta)
        self.units.append(u)
        return u

    def waived(self, test_name):
        """The ruling text if --waive covers this test name, else None."""
        for t, r in self.waivers:
            if t == test_name:
                return r
        return None

    def summary(self):
        by = {}
        for u in self.units:
            by[u.status] = by.get(u.status, 0) + 1
        return by

    def exit_code(self):
        """DESIGN 11.5: 1 if any unit errored; else 2 if any unit was refused (a unit with item refusals that
        completed is done, not refused); else 3 only when EVERY unit had no input; else 0."""
        st = [u.status for u in self.units]
        if not st:
            return EXIT_NO_INPUT
        if 'error' in st:
            return EXIT_ERROR
        if 'refused' in st:
            return EXIT_REFUSED
        if all(s == 'no-input' for s in st):
            return EXIT_NO_INPUT
        if all(s in ('done', 'skipped-valid', 'no-input') for s in st):
            return EXIT_DONE
        return EXIT_ERROR

    def write_summary(self):
        if self.dry_run or getattr(self, 'no_records', False):
            return None
        p = os.path.join(self.out_root, '_records', self.step, '_runs', f'{self.run_id}.json')
        atomic_write_json(p, dict(self.header(), finished=utcnow(), started=self.started, summary=self.summary(),
                                  units=[{'unit': u.name, 'status': u.status, 'reason': u.reason,
                                          'n_planned': len(u.planned), 'n_refused_items': len(u.refused_items)}
                                         for u in self.units]))
        return p


class UnitRecord:
    """One unit of work. Use as a context manager:

        with run.unit('04h_Full') as u:
            u.add_input(path)
            for p in planned_paths:
                u.plan(p, 'search_frame')          # the outputs this unit owes (from the plan / --dry-run logic)
            ...write...
            u.add_output(out_path, 'master')      # runs the class content tests now
            u.refuse_item(p, 'elevation gate: 18 deg < 25 deg', gate='elevation')   # one planned item refused
    """

    def __init__(self, run, name, meta):
        self.run = run
        self.name = name
        self.meta = dict(meta)
        self.status = 'running'
        self.reason = ''
        self.inputs = {}
        self.admitted = None
        self.outputs = []
        self.tests = []
        self.notes = []
        self.planned = {}            # abs path -> product class: the outputs this unit owes
        self.refused_items = []      # [{'path', 'reason', 'gate'}]: planned items refused by a named gate
        self.t0 = time.time()

    # ---- inputs -------------------------------------------------------------------------------------------------
    def add_input(self, path, extra=None):
        fp = fingerprint(path)
        if extra:
            fp.update(extra)
        self.inputs[os.path.abspath(path)] = fp

    def set_admitted(self, nights):
        """The admitted night set this unit used (sorted list of 'YYYY-MM-DD'); part of the resume identity, so an
        exclusion change invalidates the unit (deep requirement (d), 2026-10-13)."""
        self.admitted = sorted(nights)

    # ---- outputs and content tests ------------------------------------------------------------------------------
    def add_output(self, path, cls, test=True, **ctx):
        from exopipe.phase3.core import validate
        entry = {'path': os.path.abspath(path), 'class': cls}
        self.outputs.append(entry)
        if test:
            for r in validate.run(cls, path, **ctx):
                self._record_test(cls, path, r)
        return entry

    def check(self, name, passed, value=None, detail='', cls=None, path=None):
        """Record an inline content test / gate result. A failed check refuses unless --waive covers it."""
        self._record_test(cls, path, {'name': name, 'passed': bool(passed), 'value': value, 'detail': detail})

    def _record_test(self, cls, path, r):
        r = dict(r, cls=cls, path=os.path.abspath(path) if path else None)
        if not r['passed']:
            ruling = self.run.waived(r['name'])
            if ruling:
                r['waived'] = ruling
        self.tests.append(r)

    def failed_tests(self):
        return [t for t in self.tests if not t['passed'] and not t.get('waived')]

    def note(self, text):
        self.notes.append(text)

    def refuse(self, reason):
        """Refuse the WHOLE unit (exit 2)."""
        raise Refusal(reason)

    # ---- planned outputs and item refusals (DESIGN 11.5) --------------------------------------------------------
    def plan(self, path, cls):
        """Declare an output this unit owes. At exit the unit is done only when every planned output was written
        (``add_output``) and passed its content test, or a recorded item refusal names it (``refuse_item``)."""
        p = os.path.abspath(path)
        self.planned[p] = cls
        return p

    def refuse_item(self, path, reason, gate=None):
        """Record that one planned item (an image, a band, a window) is refused by a named gate while the unit goes
        on. A refused item is a complete outcome for that item; the reason is kept in the record."""
        if not reason:
            raise ValueError('refuse_item needs the reason (the gate and its value)')
        self.refused_items.append({'path': os.path.abspath(path), 'reason': str(reason), 'gate': gate})

    # ---- context ------------------------------------------------------------------------------------------------
    def __enter__(self):
        return self

    def _final_status(self):
        refused = {r['path'] for r in self.refused_items}
        failed = [t for t in self.failed_tests() if t.get('path') not in refused]
        if failed:
            return 'refused', 'content test failed: ' + '; '.join(
                f"{t['name']} ({t.get('detail') or t.get('value')})" for t in failed[:5])
        written = {o['path'] for o in self.outputs}
        if self.planned:
            missing = sorted(p for p in self.planned if p not in written and p not in refused)
            if missing:
                return 'error', (f'{len(missing)} of {len(self.planned)} planned outputs neither written nor refused: '
                                 + ', '.join(missing[:5]) + (' ...' if len(missing) > 5 else ''))
            if not (written & set(self.planned)) and all(p in refused for p in self.planned):
                return 'refused', f'every planned item refused ({len(self.planned)}): ' + '; '.join(
                    f"{os.path.basename(r['path'])}: {r['reason']}" for r in self.refused_items[:5])
            return 'done', (f'{len(refused & set(self.planned))} planned item(s) refused by a gate'
                            if refused & set(self.planned) else '')
        if not self.outputs:
            if self.refused_items:
                return 'refused', f'every item refused ({len(self.refused_items)}): ' + '; '.join(
                    f"{os.path.basename(r['path'])}: {r['reason']}" for r in self.refused_items[:5])
            return 'no-input', self.reason or 'no output written (never a done state)'
        return 'done', ''

    def __exit__(self, etype, exc, tb):
        if etype is None:
            self.status, self.reason = self._final_status()
        elif issubclass(etype, Refusal):
            self.status, self.reason = 'refused', str(exc)
        elif issubclass(etype, NoInput):
            self.status, self.reason = 'no-input', str(exc)
        elif issubclass(etype, KeyboardInterrupt):
            self.status, self.reason = 'error', 'interrupted'
        else:
            self.status, self.reason = 'error', ''.join(traceback.format_exception_only(etype, exc)).strip()
            self.notes.append(''.join(traceback.format_exception(etype, exc, tb))[-4000:])
        self.write()
        return etype is not None and issubclass(etype, (Refusal, NoInput))   # swallow gate outcomes, not errors

    def to_dict(self):
        return dict(self.run.header(), unit=self.name, meta=self.meta, status=self.status, reason=self.reason,
                    started=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(self.t0)), finished=utcnow(),
                    wall_s=round(time.time() - self.t0, 1), inputs=self.inputs, admitted=self.admitted,
                    outputs=self.outputs, tests=self.tests, notes=self.notes, code=code_fingerprint(),
                    planned=[{'path': p, 'class': c} for p, c in sorted(self.planned.items())],
                    refused_items=self.refused_items)

    def write(self):
        if self.run.dry_run:
            return
        atomic_write_json(record_path(self.run.out_root, self.run.step, self.name), self.to_dict())


def resume_check(out_root, step, unit, inputs=None, admitted=None, cfg_hash=None, retest=True):
    """(True, reason) when the unit may be skipped under --resume; (False, reason) otherwise.

    Skip only when: a record exists with status done and was written from a clean tree (a commit ending '-dirty' is
    never resumed: its code is not identified by the commit); every recorded input still has the same fingerprint
    (and ``inputs``, when given, equals the recorded input set); the admitted night set is unchanged; the code that
    produced it is unchanged (sha1 of each recorded exopipe module); the config hash is unchanged; and every output
    passes its content test now."""
    rec = load_record(out_root, step, unit)
    if rec is None:
        return False, 'no record'
    if rec.get('status') != 'done':
        return False, f"record status {rec.get('status')}"
    if str(rec.get('commit') or '').endswith('-dirty'):
        return False, f"record written from a dirty tree (commit {rec.get('commit')}): never resumed"
    for p, fp in rec.get('inputs', {}).items():
        if not same_fingerprint(fp, fingerprint(p)):
            return False, f'input changed: {p}'
    if inputs is not None:
        want = {os.path.abspath(p) for p in inputs}
        if want != set(rec.get('inputs', {})):
            return False, 'input set changed'
    if admitted is not None and sorted(admitted) != (rec.get('admitted') or []):
        return False, 'admitted night set changed (exclusions)'
    if cfg_hash is not None and cfg_hash != rec.get('config_hash'):
        return False, 'config changed'
    for mod, h in (rec.get('code') or {}).items():
        m = sys.modules.get(mod)
        f = getattr(m, '__file__', None) if m else None
        if f and f.endswith('.py') and hashlib.sha1(open(f, 'rb').read()).hexdigest()[:12] != h:
            return False, f'code changed: {mod}'
    if retest:
        from exopipe.phase3.core import validate
        for o in rec.get('outputs', []):
            if not os.path.exists(o['path']):
                return False, f"output missing: {o['path']}"
            for r in validate.run(o['class'], o['path']):
                if not r['passed']:
                    return False, f"output fails {r['name']}: {o['path']}"
    return True, 'record done; inputs, admitted nights, code, config unchanged; outputs pass their content tests'
