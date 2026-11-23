"""The flux-scale gate of the 10-s cutout regeneration from the rescaled flow cubes (config photometry_fluxscale).

Ported from the generation-6 drivers (exopipe_ops/gen6_20261117/gen6_cube_gate_DRAFT.py :43-232, gen6_wb_regen_DRAFT.sh :42-45,
gen6_gen5_sample_DRAFT.py; FINDING-20261112-FORCED-SNAPSHOT-PHOTOMETRY-FRAME-UNITS, RESULT-20261117-GEN6-PROBE-CUTOUTS-FROM-
RESCALED-CUBES requirement (2)) so a cutout generation runs in v2 alone:

  records()       {cube path: record}: the cube's newest row by utc in the rescale RESULTS_<host>.csv files ('rescale-results',
                  the cubes snapshot_cube_rescale.py rewrote), else its 'done' row with a factor in the newest rescale plan that
                  lists it ('plan-native', a cube written already scaled); no RESULTS row at all -> RecordsError
  check()         one (cell, subband): every Stokes cube carries HDU-0 FSAPPLD, FSAPPLD = its record within FSAPPLD_TOL, and the
                  V and I factors agree within it -> (ok, reasons, info)
  check_wideband() the member subbands of a wideband group the group search reads (not excluded, with a cube): each existing
                  Stokes cube against its record, V = I where both exist
  changed()       the cube identity (path, FSAPPLD, size, mtime_ns) of two checks differs: the cube changed between them
  write_sidecar() after placement: check again, refuse unless every cube equals the pre-cut check, then write the flux-scale
                  record <root>/_fluxscale/<sb>/<cell>_<sb>_fluxscale.json (atomic), the file the stamp engine reads
  gen5_sample()   the deterministic (cell, subband) sample of content check (b) (comparison with generation 5)

Headers only, by positioned reads with the read-ahead advice off: no Lustre file is memory-mapped (R-GREGG-20261117-NO-MMAP-OF-
LUSTRE-FILES-READ-OR-WRITE) and no sequential read lets the Lustre client read ahead (INCIDENT-20261117-GEN6-GATE-DRYRUN-LUSTRE-
READAHEAD-460GB: astropy's header-only fits.open cost ~67 MB per cube).
"""
import csv
import hashlib
import json
import os
import re
import time


class RecordsError(Exception):
    """The rescale records cannot be read (no RESULTS row): the gate cannot run - every unit it guards refuses."""


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def _fx():
    return _cfg().photometry_fluxscale


def cube_path(cell, sb, stokes, cube_root=None):
    root = cube_root or _cfg().paths.SNAPSHOT_CUBES_ROOT
    return os.path.join(root, cell, '%s_%s_%s_flowcube.fits' % (cell, sb, stokes))


def sidecar_path(root, cell, sb):
    return os.path.join(root, _fx().SIDECAR_SUBDIR, sb, '%s_%s_fluxscale.json' % (cell, sb))


# ---------------------------------------------------------------------------------------------------------------------
# the rescale records
# ---------------------------------------------------------------------------------------------------------------------

def record_files():
    """(RESULTS files, plan files), each sorted by name: the one RESCALE_RESULTS_DIR listed, and paths.OPS_ROOT listed for the
    plan names (one directory each; never a walk)."""
    fx, ops = _fx(), _cfg().paths.OPS_ROOT
    rdir = os.path.join(ops, fx.RESCALE_RESULTS_DIR)
    res = sorted(os.path.join(rdir, n) for n in (os.listdir(rdir) if os.path.isdir(rdir) else ())
                 if re.match(fx.RESCALE_RESULTS_RE, n))
    plans = sorted(os.path.join(ops, n) for n in (os.listdir(ops) if os.path.isdir(ops) else ())
                   if re.match(fx.RESCALE_PLAN_RE, n))
    return res, plans


_REC = {}            # {signature of the record files: records}


def records():
    """{normalized cube path: dict(source, factor, md5_new, quarantine, utc, results | plan)}; RESULTS rows win over plan rows.
    Plans in name order (a newer plan overrides an older one; a newer plan's row that is not 'done' removes an older plan's
    record of the cube). RESULTS: the newest row by utc of each cube (a tie keeps the first, files in name order). Read again
    when a record file changes (size or mtime)."""
    res_files, plan_files = record_files()
    sig = tuple((p, os.stat(p).st_size, os.stat(p).st_mtime_ns) for p in res_files + plan_files)
    if sig in _REC:
        return _REC[sig]
    rec = {}
    for p in plan_files:
        with open(p, newline='') as f:
            for r in csv.DictReader(f):
                k = os.path.normpath(r.get('cube') or '')
                if r.get('status') == 'done' and r.get('factor'):
                    rec[k] = dict(source='plan-native', factor=float(r['factor']), plan=os.path.basename(p))
                elif k in rec and rec[k]['source'] == 'plan-native':
                    del rec[k]
    res = {}
    for p in res_files:
        with open(p, newline='') as f:
            for r in csv.DictReader(f):
                if not r.get('cube') or not r.get('fsappld'):
                    continue
                k = os.path.normpath(r['cube'])
                if k not in res or r['utc'] > res[k]['utc']:
                    res[k] = dict(source='rescale-results', factor=float(r['fsappld']), md5_new=r.get('md5_new', ''),
                                  quarantine=r.get('quarantine', ''), utc=r['utc'], results=os.path.basename(p))
    if not res:
        raise RecordsError('no rescale RESULTS rows in %s (%d files matching %s): the flux-scale gate cannot run without its '
                           'records' % (os.path.join(_cfg().paths.OPS_ROOT, _fx().RESCALE_RESULTS_DIR), len(res_files),
                                        _fx().RESCALE_RESULTS_RE))
    rec.update(res)
    _REC.clear()
    _REC[sig] = rec
    return rec


def record_of(path, rec=None):
    return (records() if rec is None else rec).get(os.path.normpath(path))


# ---------------------------------------------------------------------------------------------------------------------
# headers only, by positioned reads
# ---------------------------------------------------------------------------------------------------------------------

def _read_hdr(fd, off, size):
    """(astropy Header, its length in bytes rounded up to 2880) of the FITS header starting at byte ``off``: positioned reads
    of HEADER_CHUNK bytes until the END card."""
    from astropy.io import fits
    fx = _fx()
    buf = b''
    while off + len(buf) < size:
        b = os.pread(fd, fx.HEADER_CHUNK, off + len(buf))
        if not b:
            break
        start = max(0, len(buf) - len(buf) % 80)
        buf += b
        for k in range(start, len(buf) - 79, 80):
            if buf[k:k + 8] == b'END     ':
                return fits.Header.fromstring(buf[:k + 80].decode('ascii')), (k // 2880 + 1) * 2880
        if len(buf) > fx.HEADER_MAX_CHUNKS * fx.HEADER_CHUNK:
            break
    raise ValueError('no END card within %d bytes at offset %d' % (len(buf), off))


def _data_bytes(h):
    """The HDU's data size in bytes padded to 2880 (|BITPIX|/8 x GCOUNT x (PCOUNT + NAXIS1 x ... x NAXISn))."""
    nax = int(h.get('NAXIS', 0))
    if nax == 0:
        return 0
    n = 1
    for k in range(1, nax + 1):
        n *= int(h['NAXIS%d' % k])
    n = abs(int(h['BITPIX'])) // 8 * int(h.get('GCOUNT', 1)) * (int(h.get('PCOUNT', 0)) + n)
    return -(-n // 2880) * 2880


def read_cube_header(path):
    """dict(fsappld = HDU-0 FSAPPLD or None, nframes = HDU-0 NFRAMES, size, mtime_ns, frames_shape = the FRAMES extension's
    (NAXIS3, NAXIS2, NAXIS1), ZNAXISn when tile-compressed): HDU 0, then each extension header in turn until EXTNAME FRAMES."""
    st = os.stat(path)
    fd = os.open(path, os.O_RDONLY)
    try:
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
        except (AttributeError, OSError):
            pass
        h0, n0 = _read_hdr(fd, 0, st.st_size)
        fs = h0.get('FSAPPLD')
        shape, off = (None, None, None), n0 + _data_bytes(h0)
        for _ in range(8):
            if off >= st.st_size:
                break
            h, n = _read_hdr(fd, off, st.st_size)
            if str(h.get('EXTNAME', '')).strip() == 'FRAMES':
                z = 'Z' if h.get('ZIMAGE') else ''
                shape = (h.get(z + 'NAXIS3'), h.get(z + 'NAXIS2'), h.get(z + 'NAXIS1'))
                break
            off += n + _data_bytes(h)
        return dict(fsappld=None if fs is None else float(fs), nframes=h0.get('NFRAMES'), size=st.st_size,
                    mtime_ns=st.st_mtime_ns, frames_shape=list(shape))
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------------------------------------------------

def check_cube(path, stokes, rec):
    """(info, reasons) of one cube against its record."""
    tol = _fx().FSAPPLD_TOL
    d, reasons = dict(path=path), []
    if not os.path.exists(path):
        return d, ['%s: no cube %s' % (stokes, path)]
    try:
        d.update(read_cube_header(path))
    except Exception as e:                                   # noqa: BLE001 - an unreadable header refuses
        return d, ['%s: header unreadable (%s)' % (stokes, str(e)[:100])]
    r = rec.get(os.path.normpath(path))
    d['record'] = r
    fs = d['fsappld']
    if fs is None:
        reasons.append('%s: no FSAPPLD (frame-unit cube)' % stokes)
    if r is None:
        reasons.append("%s: no rescale record (neither RESULTS row nor plan 'done' row)" % stokes)
    if fs is not None and r is not None and abs(fs - r['factor']) > tol:
        reasons.append('%s: FSAPPLD %.6f != recorded %.6f (%s)' % (stokes, fs, r['factor'], r['source']))
    return d, reasons


def _v_equals_i(info, prefix=''):
    fv, fi = (info.get(S, {}).get('fsappld') for S in ('V', 'I'))
    if fv is not None and fi is not None and abs(fv - fi) > _fx().FSAPPLD_TOL:
        return ['%sV FSAPPLD %.6f != I FSAPPLD %.6f (one factor per hour-band)' % (prefix, fv, fi)]
    return []


def check(cell, sb, cube_root=None, stokes=None):
    """(ok, reasons [str], info {Stokes: dict}) of one (cell, subband): every Stokes cube of ``stokes`` (default
    photometry_fluxscale.STOKES) exists, carries FSAPPLD equal to its record, and V = I. Raises RecordsError."""
    rec = records()
    reasons, info = [], {}
    for S in (stokes or _fx().STOKES):
        info[S], r = check_cube(cube_path(cell, sb, S, cube_root), S, rec)
        reasons += r
    reasons += _v_equals_i(info)
    return not reasons, reasons, info


def members_read(cell, group, cube_root=None, ex=None):
    """The member subbands of wideband ``group`` the group search reads for ``cell``: the group's members
    (instrument.SUBBAND_BANDS) not excluded in the snapshot tiers (core.exclusions) with at least one Stokes cube."""
    from exopipe.phase3.core import exclusions
    c = _cfg()
    ex = ex or exclusions.for_tier('snapshot')
    lst, date = cell.split('_', 1)
    return [sb for sb in c.instrument.SUBBAND_BANDS.get(group, ())
            if not ex.is_excluded(lst, sb, date)
            and any(os.path.exists(cube_path(cell, sb, S, cube_root)) for S in _fx().STOKES)]


def check_wideband(cell, group, cube_root=None, ex=None):
    """(ok, reasons, info {member: {Stokes: dict}}) of the member cubes a wideband group reads: each existing Stokes cube of
    every member (members_read) against its record, V = I where both exist. gen6_wb_regen_DRAFT.sh gated only the members of
    its subband job list (both cubes, a record, not excluded): a member with one Stokes cube, or none recorded, went
    unchecked into the group search; here every cube the search reads is checked."""
    rec = records()
    reasons, info = [], {}
    members = members_read(cell, group, cube_root, ex)
    if not members:
        return False, ['%s: no member subband with a cube that is not excluded' % group], info
    for sb in members:
        info[sb] = {}
        for S in _fx().STOKES:
            p = cube_path(cell, sb, S, cube_root)
            if not os.path.exists(p):
                continue
            info[sb][S], r = check_cube(p, S, rec)
            reasons += ['%s %s' % (sb, x) for x in r]
        reasons += _v_equals_i(info[sb], '%s ' % sb)
    return not reasons, reasons, info


IDENTITY_KEYS = ('path', 'fsappld', 'size', 'mtime_ns')


def changed(before, after):
    """[reason] for each Stokes whose cube identity (path, FSAPPLD, size, mtime_ns) differs between two checks' info."""
    out = []
    for S in sorted(set(before) | set(after)):
        b, a = before.get(S) or {}, after.get(S) or {}
        for k in IDENTITY_KEYS:
            if b.get(k) != a.get(k):
                out.append('%s cube %s changed (%s -> %s)' % (S, k, b.get(k), a.get(k)))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# the flux-scale record (sidecar)
# ---------------------------------------------------------------------------------------------------------------------

def regen_text(line):
    """The regeneration's result line without its log time stamp: 'INFO CUTOUTS-REGEN 00h 2025-09-06 55MHz: V 165 I 165'
    (generation 6 stored the log line from column 25 on, `cut -c25-240`: the 24-character asctime cut off)."""
    s = str(line or '').strip()
    m = re.match(r'^\d{4}-\d\d-\d\d[ T]\d\d:\d\d:\d\d(?:[,.]\d+)?\s+', s)
    return (s[m.end():] if m else s)[:216]


def code_versions(code_root):
    """{'code_dir', '<module>': '<path> <md5>' for the two modules the stamp engine compares (stamp_engine.CODE_MODULES),
    'gate': this module's md5}."""
    from exopipe.phase3.lib.photometry import stamp_engine
    out = dict(code_dir=code_root)
    for m in stamp_engine.CODE_MODULES:
        p = os.path.join(code_root, m)
        with open(p, 'rb') as f:
            out[m] = '%s %s' % (p, hashlib.md5(f.read()).hexdigest())
    with open(os.path.abspath(__file__), 'rb') as f:
        out['gate'] = hashlib.md5(f.read()).hexdigest()
    return out


def sidecar_doc(cell, sb, info, regen_line, code):
    from exopipe.phase3.core import env
    return dict(product='forced-photometry cutouts: flux-scale record of one (cell, subband)',
                cell=cell, subband=sb,
                unit="Jy/beam, Master units: every cutout pixel = frame-unit pixel / FSAPPLD (the cube's own HDU-0 card)",
                cubes={S: dict(path=info[S]['path'], fsappld=info[S]['fsappld'], size=info[S]['size'],
                               mtime_ns=info[S]['mtime_ns'], nframes=info[S]['nframes'],
                               record_source=info[S]['record']['source'], record_factor=info[S]['record']['factor'],
                               md5_recorded=info[S]['record'].get('md5_new', ''),
                               quarantine_original=info[S]['record'].get('quarantine', ''))
                       for S in info},
                regen=regen_text(regen_line), code=code, host=env.host(),
                utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))


class SidecarRefused(Exception):
    pass


def write_sidecar(root, cell, sb, before, regen_line, code_root, cube_root=None):
    """Check the cubes again; refuse (SidecarRefused, nothing written) unless the gate still passes and every cube's path,
    FSAPPLD, size and mtime_ns equal ``before`` (the info of the check made before the cut); then write the flux-scale
    record atomically (fsync, rename). Returns its path."""
    ok, reasons, info = check(cell, sb, cube_root, stokes=tuple(before))
    if not ok:
        raise SidecarRefused('the flux-scale gate now fails: ' + '; '.join(reasons))
    ch = changed(before, info)
    if ch:
        raise SidecarRefused('a cube changed since the pre-cut check: ' + '; '.join(ch))
    doc = sidecar_doc(cell, sb, info, regen_line, code_versions(code_root))
    out = sidecar_path(root, cell, sb)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = '%s.tmp.%d' % (out, os.getpid())
    with open(tmp, 'w') as f:
        json.dump(doc, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# check (b): the generation-5 comparison sample
# ---------------------------------------------------------------------------------------------------------------------

def _jobs(path):
    with open(path) as f:
        return [tuple(ln.split()[:2]) for ln in f if ln.strip() and not ln.startswith('#')]


def _manifest_complete(cell, cube_root=None):
    fx = _fx()
    p = os.path.join(cube_root or _cfg().paths.SNAPSHOT_CUBES_ROOT, cell, fx.CUBE_MANIFEST)
    if not os.path.isfile(p):
        return False
    with open(p, errors='replace') as f:
        return any(ln.startswith(fx.COMPLETE_PREFIX) for ln in f)


def generation_job(cell, sb, rec=None, ex=None, cube_root=None):
    """A (cell, subband) a cutout generation cuts (gen6_build_jobs_DRAFT.py): both Stokes cubes, the cell's cube MANIFEST
    COMPLETE, a rescale record for both cubes, not excluded in the snapshot tiers."""
    from exopipe.phase3.core import exclusions
    rec = records() if rec is None else rec
    ex = ex or exclusions.for_tier('snapshot')
    lst, date = cell.split('_', 1)
    paths = [cube_path(cell, sb, S, cube_root) for S in _fx().STOKES]
    return (all(os.path.exists(p) for p in paths) and not ex.is_excluded(lst, sb, date)
            and all(os.path.normpath(p) in rec for p in paths) and _manifest_complete(cell, cube_root))


def gen5_sample(jobs=None, per_lst=None, cube_root=None, ex=None, lsts=None):
    """[(cell, subband)] of check (b) (gen6_gen5_sample_DRAFT.py): per LST hour, the first ``per_lst`` (GEN5_SAMPLE_PER_LST)
    jobs in md5('<cell>_<subband>') order that are generation-5 jobs (GEN5_JOBS) and generation jobs - ``jobs`` when given,
    else generation_job() - and whose generation-5 input was the frame-unit cube now in the quarantine Run: generation 5's
    done marker of the job exists, and both Stokes have a RESULTS record naming a quarantine original that exists and is
    older (mtime) than the marker. stat calls and the cube MANIFEST only; no cube is read. ``lsts``: only these LST hours (a
    unit asks for its own: the sample of an hour does not depend on the other hours)."""
    c, fx = _cfg(), _fx()
    per_lst = fx.GEN5_SAMPLE_PER_LST if per_lst is None else per_lst
    rec = records()
    g5 = _jobs(os.path.join(c.paths.OPS_ROOT, fx.GEN5_JOBS))
    done = os.path.join(c.paths.OPS_ROOT, fx.GEN5_DONE_DIR)
    if jobs is not None:
        jobs = set(jobs)
        universe = [j for j in g5 if j in jobs]
        member = lambda j: True                       # noqa: E731
    else:
        universe = g5
        member = lambda j: generation_job(j[0], j[1], rec, ex, cube_root)   # noqa: E731
    by = {}
    for cell, sb in dict.fromkeys(universe):
        by.setdefault(cell[:3], []).append((hashlib.md5(('%s_%s' % (cell, sb)).encode()).hexdigest(), cell, sb))
    out = []
    for lst in sorted(by):
        if lsts is not None and lst not in lsts:
            continue
        n = 0
        for _, cell, sb in sorted(by[lst]):
            if n >= per_lst:
                break
            if not member((cell, sb)):
                continue
            mk = os.path.join(done, '%s_%s' % (cell, sb))
            ok = os.path.exists(mk)
            for S in fx.STOKES:
                r = rec.get(os.path.normpath(cube_path(cell, sb, S, cube_root))) or {}
                q = r.get('quarantine', '')
                ok = (ok and r.get('source') == 'rescale-results' and bool(q) and os.path.exists(q)
                      and os.stat(q).st_mtime < os.stat(mk).st_mtime)
            if ok:
                out.append((cell, sb))
                n += 1
    return out
