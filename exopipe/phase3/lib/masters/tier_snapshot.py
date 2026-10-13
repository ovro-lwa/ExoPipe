"""exopipe3 masters --tier snapshot - the snapshot tier's masters (tier D): VERIFIED, never built (contract S-1.3a, "No snapshot
code BUILDS Masters"; Gregg 2026-10-07 Q17 and 2026-10-08 C: the tier stays on the frozen generation-4 masters).

Stages (default --stages = masters.DEFAULT_STAGES['snapshot'] = verify-frozen):
  verify-frozen  unit verify-frozen_<slot> per selected slot (deep cell grammar, 14h_41MHz; default all 456):
                   every frozen file of the slot (I/V Taper R0, I Taper R-0.75, PB_deep) has the md5 its MANIFEST.csv row
                   records (md5 recomputed now), and both registration links of the slot resolve to the slot's frozen copy -
                   never outside masters_snapshot.FROZEN_ROOT and never into the deep archive Coadd/ tree;
                 unit verify-frozen_pblibrary: the model-beam PB library the tier reads (masters_pbmaps.PB_LIBRARY; 15 subbands
                   x (6 bins + deep) = 105 maps): present, readable, finite, inside PB_LIBRARY_VALUE_RANGE, PBBEAM = the model
                   beam, PBSCOPE / PBLSTBIN = the file name (RULING-SNAPSHOT-TIER-PB-CORRECTION-MODEL-BEAM-ONLY);
                 unit verify-frozen_registry (only when every slot is selected): the slot registries joined, with the counts
                   (456 slots, 1,824 frozen files, 912 links, nothing extra in either tree).
  freeze         plan only in the refactor window: under --dry-run, exopipe_ops/snapshot_masters_freeze.py --dry-run runs in
                 place (read only) and its plan is printed; without --dry-run the stage refuses (G-D04: it writes production
                 and re-points the 912 links; masters_snapshot.FREEZE_ALLOWED False, paths.PRODUCTION_WRITES False).
  sources        (review 2026-10-13; selectable once masters.STAGES['snapshot'] lists it) unit sources_snapshot: the snapshot
                 bright-source PARENT catalogues on ONE recipe (M21: exopipe_ops/snapshot_parentcat_contract.py in place,
                 thresh_pix 7.0 / thresh_isl 4.0, no atrous, on the frozen gen-4 I Taper R-0.75 masters) into
                 <out>/snapshot_master_catalogs/master_R075_<slot>.gaul.csv + .gen.json [snapshot_parent_catalog]; the
                 production set (masters_snapshot.PARENT_CATALOG_DIR) is checked with `exopipe3 verify snapshot_parent_catalog`.

Inputs (read only): <LINK_ROOT>/<slot>/coadd_final/{I,V}_deep_Taper_R0_<band>.fits (912 symlinks), <FROZEN_ROOT>/<slot>/coadd_final/
(4 files per slot) and <FROZEN_ROOT>/MANIFEST.csv (sha1 pinned in config), the PB library.
Outputs (a registry in the run root, not a production product; default --out paths.PHASE3_ARCHIVE_ROOT is a production root, so
a real run needs --out while paths.PRODUCTION_WRITES is False):
  <out>/snapshot/registry/<slot>.csv                      [snapshot_master_registry] slot, file, path, size, md5_now,
                                                          md5_manifest, link, link_target, md5_ok, link_ok
  <out>/snapshot/snapshot_master_registry_gen4.csv        [snapshot_master_registry] every slot (full selection only)
  <out>/snapshot/pb_library_registry.csv                  [snapshot_pb_library] file, md5, size, shape, min, max, finite,
                                                          pbbeam, pbscope, pblstbin
  run records <out>/_records/masters/verify-frozen_<slot>.json ...; content tests in lib/masters/tests_snapshot.py.

Failure modes (exit 2 = refused, the gate named): G-D01 a link missing, broken, resolving outside FROZEN_ROOT or into the deep
archive Coadd/ tree, or not to its slot's frozen copy; G-D02 MANIFEST.csv sha1 != masters_snapshot.FROZEN_MANIFEST_SHA1, a
frozen file missing or with an md5 other than its manifest row, a slot without its 4 manifest rows, counts != 456 / 1,824
(full selection); G-D03 a library map missing, unreadable, not finite, outside the value range, or with another beam model /
scope / bin; G-D04 --stages freeze without --dry-run; config differing from an in-place literal it mirrors (freeze script,
snapshot search). Exit 3: no slot selected.

Per-hour vs intermittent (DESIGN 12): intermittent (a check of the frozen tree the per-hour snapshot chain reads); it builds
nothing, so the per-hour chain never runs it.
"""
import csv
import hashlib
import io
import os
import re
import threading

from exopipe.phase3.core import record
from exopipe.phase3.core.record import NoInput, Refusal
from exopipe.phase3.lib.masters import tier_sublst as TS

TIER = 'snapshot'
SECTION = 'masters_snapshot'
REGISTRY_COLUMNS = ['slot', 'file', 'path', 'size', 'md5_now', 'md5_manifest', 'link', 'link_target', 'md5_ok', 'link_ok']
PBLIB_COLUMNS = ['file', 'md5', 'size', 'shape', 'min', 'max', 'finite', 'pbbeam', 'pbscope', 'pblstbin']
COMBINED = 'snapshot_master_registry_gen4.csv'
SCRIPTS = {'freeze': ('exopipe_ops', 'snapshot_masters_freeze.py'), 'snapsearch': ('clv15', 'phase3_snapshot_search.py'),
           'parentcat': ('exopipe_ops', 'snapshot_parentcat_contract.py'),
           # the builder imports phase3_deep._load from the production package; its branch copy (pure git mv) is checked
           'deepload': ('package', 'lib/search/deep_search.py')}
_MAN_LOCK = threading.Lock()
_MAN = {}


def cfg():
    from exopipe.phase3.config import CFG
    return CFG


def _sec():
    return getattr(cfg(), SECTION)


def pb_library():
    return cfg().masters_pbmaps.PB_LIBRARY


def coadd_tree():
    return os.path.join(cfg().paths.PHASE3_ARCHIVE_ROOT, _sec().FORBIDDEN_LINK_TARGET)


def script(short):
    from exopipe.phase3.core import env
    rootname, name = SCRIPTS[short]
    base = (os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) if rootname == 'package'
            else env.legacy_root(rootname))
    p = os.path.join(base, name)
    if not os.path.isfile(p):
        raise Refusal(f'in-place code {p} not found')
    return p


# (label, config getter, in-place file, regex, converter)
MIRRORS = [
    ('masters_snapshot.LINK_ROOT', lambda s: s.LINK_ROOT, 'freeze', r'^LINKS = "([^"]+)"', str),
    ('masters_snapshot.FROZEN_ROOT', lambda s: s.FROZEN_ROOT, 'freeze', r'^FROZEN = "([^"]+)"', str),
    ('masters_pbmaps.PB_LIBRARY', lambda s: pb_library(), 'snapsearch', r'^PB_LIBRARY = "([^"]+)"', str),
    ('masters_snapshot.PB_MODEL_BINS', lambda s: s.PB_MODEL_BINS, 'snapsearch', r'^N_TIME_BINS = (\d+)', int),
    ('masters_snapshot.FROZEN_ROOT', lambda s: s.FROZEN_ROOT, 'parentcat', r'^GEN4 = "([^"]+)"', str),
    ('masters_snapshot.PARENT_THRESH_PIX / PARENT_THRESH_ISL', lambda s: (s.PARENT_THRESH_PIX, s.PARENT_THRESH_ISL),
     'parentcat', r'^THRESH_PIX, THRESH_ISL = ([0-9.]+), ([0-9.]+)', None),
    ('masters_snapshot.PARENT_MASTER_FILE', lambda s: s.PARENT_MASTER_FILE.format(b='%s'), 'parentcat',
     r'coadd_final/(I_deep_Taper_R075_%s\.fits)" % \(GEN4', str),
    ('masters_snapshot.PARENT_ATROUS', lambda s: s.PARENT_ATROUS, 'parentcat', r'"atrous_do": (True|False)',
     lambda t: t == 'True'),
    ('masters_snapshot.PARENT_ZAP_JYB', lambda s: s.PARENT_ZAP_JYB, 'deepload', r'a\[np\.abs\(a\) > ([0-9.e+]+)\] = 0\.0',
     float),
]


def mirror_violations():
    s = _sec()
    bad, texts = [], {}
    for label, get, short, rx, conv in MIRRORS:
        path = script(short)
        if path not in texts:
            texts[path] = open(path).read()
        m = re.search(rx, texts[path], re.M)
        if not m:
            bad.append(f'{label}: literal not found in {path} (pattern {rx!r})')
            continue
        have = tuple(float(x) for x in m.groups()) if conv is None else conv(m.group(1))
        if have != get(s):
            bad.append(f'{label} = {get(s)!r} but {os.path.basename(path)} uses {have!r}')
    return bad


# =====================================================================================================================
# the frozen tree
# =====================================================================================================================

def manifest_path():
    return os.path.join(_sec().FROZEN_ROOT, 'MANIFEST.csv')


def manifest():
    """(sha1 of MANIFEST.csv, {slot: {file: row}}, n rows) - read once per process per (path, size, mtime)."""
    p = manifest_path()
    st = os.stat(p)
    key = (os.path.realpath(p), st.st_size, st.st_mtime_ns)
    with _MAN_LOCK:
        if _MAN.get('key') != key:
            raw = open(p, 'rb').read()
            rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8'), newline='')))
            by = {}
            for r in rows:
                by.setdefault(r['slot'], {})[r['file']] = r
            _MAN.update(key=key, sha1=hashlib.sha1(raw).hexdigest(), by=by, n=len(rows))
        return _MAN['sha1'], _MAN['by'], _MAN['n']


def manifest_problem():
    """'' or the G-D02 reason the manifest cannot be trusted (missing, or sha1 != the pinned one)."""
    p = manifest_path()
    if not os.path.isfile(p):
        return f'G-D02: {p} missing'
    sha1, _, _ = manifest()
    if sha1 != _sec().FROZEN_MANIFEST_SHA1:
        return f'G-D02: {p} sha1 {sha1} != pinned masters_snapshot.FROZEN_MANIFEST_SHA1 {_sec().FROZEN_MANIFEST_SHA1}'
    return ''


def slot_files(slot):
    """[(file name, frozen path, link path or '')] of one slot, in FROZEN_FILES order."""
    from exopipe.phase3.core import cells as CL
    s = _sec()
    _, band = CL.parse_cell(slot)
    linked = {f.format(b=band) for f in s.LINKED_FILES}
    out = []
    for f in s.FROZEN_FILES:
        name = f.format(b=band)
        out.append((name, os.path.join(s.FROZEN_ROOT, slot, 'coadd_final', name),
                    os.path.join(s.LINK_ROOT, slot, 'coadd_final', name) if name in linked else ''))
    return out


def link_state(link, frozen):
    """(target, ok, reason) of a registration link: it must be a symlink that resolves to ``frozen`` (the slot's frozen
    copy), inside FROZEN_ROOT and never into the deep archive Coadd/ tree (install G-1b, phase3_install.py:203-207)."""
    s = _sec()
    if not os.path.islink(link):
        return '', False, 'not a symlink' if os.path.exists(link) else 'missing'
    target = os.path.realpath(link)
    if not os.path.exists(link):
        return target, False, f'broken (-> {os.readlink(link)})'
    if target.startswith(os.path.realpath(coadd_tree()).rstrip('/') + '/'):
        return target, False, 'resolves into the deep archive Coadd/ tree'
    if not target.startswith(os.path.realpath(s.FROZEN_ROOT).rstrip('/') + '/'):
        return target, False, f'resolves outside {s.FROZEN_ROOT}'
    if target != os.path.realpath(frozen):
        return target, False, f'resolves to {target}, not the slot\'s frozen copy'
    return target, True, ''


def selected_slots(args):
    from exopipe.phase3.core import cells as CL
    spec = getattr(args, 'cells', None)
    slots = CL.expand_cells(spec) if spec else list(CL.all_cells())
    if getattr(args, 'lsts', None):
        lsts = set(CL.expand(args.lsts, universe=CL.LSTS, kind='LST'))
        slots = [x for x in slots if x.split('_')[0] in lsts]
    if getattr(args, 'scopes', None):
        sc = set(CL.expand(args.scopes))
        slots = [x for x in slots if x.split('_', 1)[1] in sc]
    if getattr(args, 'exclude_cells', None):
        ex = set(CL.expand_cells(args.exclude_cells))
        slots = [x for x in slots if x not in ex]
    return sorted(slots)


def registry_path(run, slot=None):
    d = os.path.join(run.out_root, 'snapshot')
    return os.path.join(d, 'registry', f'{slot}.csv') if slot else os.path.join(d, COMBINED)


def _write_csv(path, cols, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f'{path}.tmp.{os.getpid()}.{threading.get_ident()}'
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def _slot_unit(run, opts, slot):
    s = _sec()
    name = f'verify-frozen_{slot}'
    files = slot_files(slot)
    inputs = [manifest_path()] + [p for _, p, _ in files if os.path.exists(p)] + [l for _, _, l in files if l]
    if TS.resume_skip(run, opts, name, [p for p in inputs if os.path.lexists(p)], None):
        return
    with run.unit(name, slot=slot, stage='verify-frozen') as u:
        why = manifest_problem()
        if why:
            u.refuse(why)
        for p in inputs:
            if os.path.exists(p):
                u.add_input(p)
        _, by, _ = manifest()
        mrows = by.get(slot, {})
        rows, md5_bad, link_bad = [], [], []
        for fname, frozen, link in files:
            mr = mrows.get(fname)
            want = (mr or {}).get('md5_copy', '')
            if not os.path.isfile(frozen):
                now, size = '', ''
                md5_bad.append(f'{fname}: frozen copy missing')
            else:
                now, size = TS.md5_file(frozen), os.path.getsize(frozen)
                if not want:
                    md5_bad.append(f'{fname}: no manifest md5 (row {"missing" if mr is None else mr.get("note")})')
                elif now != want:
                    md5_bad.append(f'{fname}: md5 {now} != manifest {want}')
            target, lok, lwhy = link_state(link, frozen) if link else ('', True, '')
            if link and not lok:
                link_bad.append(f'{fname} link {link}: {lwhy}')
            rows.append(dict(slot=slot, file=fname, path=frozen, size=size, md5_now=now, md5_manifest=want, link=link,
                             link_target=target, md5_ok=int(bool(now) and now == want), link_ok=int(lok)))
        u.check('snapshot.G-D02_md5_equals_manifest', not md5_bad, value=sum(r['md5_ok'] for r in rows),
                detail='; '.join(md5_bad) or f'{len(rows)} frozen files = manifest md5')
        u.check('snapshot.G-D01_links_into_frozen_tree', not link_bad, value=sum(1 for r in rows if r['link'] and r['link_ok']),
                detail='; '.join(link_bad) or f'{sum(1 for r in rows if r["link"])} links -> the frozen copies')
        reg = registry_path(run, slot)
        _write_csv(reg, REGISTRY_COLUMNS, rows)
        u.add_output(reg, 'snapshot_master_registry', rehash=False)
    TS.say(f'  {name}: {u.status}' + (f' ({u.reason[:200]})' if u.reason else ''))


def _combined_unit(run, slots):
    """The slot registries of this run joined, with the full-tree counts (only when every slot is selected)."""
    s = _sec()
    with run.unit('verify-frozen_registry', stage='verify-frozen') as u:
        why = manifest_problem()
        if why:
            u.refuse(why)
        u.add_input(manifest_path())
        rows = []
        for slot in slots:
            p = registry_path(run, slot)
            rec = record.load_record(run.out_root, run.step, f'verify-frozen_{slot}')
            if not os.path.isfile(p) or not rec or rec.get('status') != 'done':
                u.refuse(f'G-D02: slot {slot} has no done verify-frozen record and registry (status '
                         f'{rec.get("status") if rec else "missing"})')
            rows += list(csv.DictReader(open(p)))
        n_link_dirs = len([d for d in os.listdir(s.LINK_ROOT) if os.path.isdir(os.path.join(s.LINK_ROOT, d))])
        n_frozen_dirs = len([d for d in os.listdir(s.FROZEN_ROOT) if os.path.isdir(os.path.join(s.FROZEN_ROOT, d))])
        _, _, n_man = manifest()
        u.check('snapshot.G-D02_counts', n_man == s.N_FROZEN_FILES and len(rows) == s.N_FROZEN_FILES
                and n_link_dirs == s.N_SLOTS and n_frozen_dirs == s.N_SLOTS, value=len(rows),
                detail=f'manifest rows {n_man}, registry rows {len(rows)} (want {s.N_FROZEN_FILES}); slot dirs: links '
                       f'{n_link_dirs}, frozen {n_frozen_dirs} (want {s.N_SLOTS})')
        out = registry_path(run)
        _write_csv(out, REGISTRY_COLUMNS, rows)
        u.add_output(out, 'snapshot_master_registry', rehash=False)
    TS.say(f'  verify-frozen_registry: {u.status} ({len(rows)} rows)' + (f' ({u.reason[:200]})' if u.reason else ''))


def pblib_expected():
    """[(file, subband, bin or -1)] of the model-beam library the snapshot tier reads."""
    s = _sec()
    out = []
    for sb in cfg().instrument.ALL_SUBBANDS:
        for k in range(s.PB_MODEL_BINS):
            out.append((s.PB_MODEL_FILES[0].format(sb=sb, k=k), sb, k))
        out.append((s.PB_MODEL_FILES[1].format(sb=sb), sb, -1))
    return out


def pblib_row(path):
    """One library map read once: md5 of the bytes, header cards, data range (the content of PBLIB_COLUMNS)."""
    import numpy as np
    from astropy.io import fits
    raw = open(path, 'rb').read()
    with fits.open(io.BytesIO(raw), memmap=False) as h:
        hdr, d = h[0].header, np.asarray(h[0].data, dtype=np.float64)
    fin = bool(np.isfinite(d).all())
    return dict(file=os.path.basename(path), md5=hashlib.md5(raw).hexdigest(), size=len(raw), shape='x'.join(map(str, d.shape)),
                min=repr(float(np.nanmin(d))) if d.size else '', max=repr(float(np.nanmax(d))) if d.size else '',
                finite=int(fin), pbbeam=str(hdr.get('PBBEAM', '')), pbscope=str(hdr.get('PBSCOPE', '')),
                pblstbin=hdr.get('PBLSTBIN', ''))


def _pblib_unit(run, opts):
    s = _sec()
    name = 'verify-frozen_pblibrary'
    exp = pblib_expected()
    paths = [os.path.join(pb_library(), f) for f, _, _ in exp]
    if TS.resume_skip(run, opts, name, [p for p in paths if os.path.exists(p)], None):
        return
    with run.unit(name, stage='verify-frozen') as u:
        miss = [p for p in paths if not os.path.isfile(p)]
        for p in paths:
            if os.path.exists(p):
                u.add_input(p)
        rows, bad = [], []
        for (f, sb, k), p in zip(exp, paths):
            if p in miss:
                bad.append(f'{f} missing')
                continue
            try:
                r = pblib_row(p)
            except Exception as e:
                bad.append(f'{f} unreadable ({type(e).__name__}: {e})')
                continue
            rows.append(r)
        u.check('snapshot.G-D03_library_complete', not bad, value=len(rows),
                detail='; '.join(bad[:6]) or f'{len(rows)} of {len(exp)} library maps read')
        out = os.path.join(run.out_root, 'snapshot', 'pb_library_registry.csv')
        _write_csv(out, PBLIB_COLUMNS, rows)
        u.add_output(out, 'snapshot_pb_library', rehash=False)
    TS.say(f'  {name}: {u.status}' + (f' ({u.reason[:200]})' if u.reason else ''))


def stage_verify(run, opts, slots):
    from exopipe.phase3.core import cells as CL
    s = _sec()
    full = slots == sorted(CL.all_cells())
    if run.dry_run:
        why = manifest_problem()
        TS.say(f'[snapshot verify-frozen] {len(slots)} slot units (md5 of {len(slots) * len(s.FROZEN_FILES)} frozen files '
               f'against {manifest_path()}), 1 PB-library unit ({len(pblib_expected())} maps in {pb_library()})'
               + (', 1 registry unit (full selection)' if full else '') + f' -> {os.path.join(run.out_root, "snapshot")}')
        if why:
            TS.say(f'    WOULD REFUSE every slot: {why}')
        _, by, n_man = manifest() if os.path.isfile(manifest_path()) else (None, {}, 0)
        nl = nbad = nmiss = 0
        for slot in slots:
            for fname, frozen, link in slot_files(slot):
                if fname not in by.get(slot, {}) or not os.path.isfile(frozen):
                    nmiss += 1
                if link:
                    nl += 1
                    nbad += not link_state(link, frozen)[1]
        TS.say(f'    manifest rows {n_man}; frozen files missing or without a manifest row: {nmiss}; links {nl}, failing '
               f'G-D01 now: {nbad} (readlink only; md5 not computed in a dry run)')
        pmiss = [f for f, _, _ in pblib_expected() if not os.path.isfile(os.path.join(pb_library(), f))]
        TS.say(f'    PB library maps missing: {len(pmiss)}' + (f' {pmiss[:4]}' if pmiss else ''))
        return int(bool(why) or bool(nmiss) or bool(nbad) or bool(pmiss))
    if not slots:
        with run.unit('verify-frozen', stage='verify-frozen') as u:
            raise NoInput('no slot selected')
    TS.pool_map(lambda slot: _slot_unit(run, opts, slot), slots, opts.workers or s.MD5_WORKERS, run,
                lambda slot: f'verify-frozen_{slot}')
    _pblib_unit(run, opts)
    if full:
        _combined_unit(run, slots)
    return 0


def stage_freeze(run, opts):
    """Plan only (G-D04): exopipe_ops/snapshot_masters_freeze.py --dry-run in place (reads only) under exopipe3 --dry-run."""
    s = _sec()
    from exopipe.phase3.core import legacy
    if not run.dry_run:
        with run.unit('freeze', stage='freeze') as u:
            u.refuse(f'G-D04: the freeze stage copies masters into {s.FROZEN_ROOT} and re-points the 912 links in '
                     f'{s.LINK_ROOT} (production writes); masters_snapshot.FREEZE_ALLOWED={s.FREEZE_ALLOWED}, '
                     f'paths.PRODUCTION_WRITES={cfg().paths.PRODUCTION_WRITES} - plan only: run with --dry-run')
        return 0
    TS.say(f'[snapshot freeze] plan only: {legacy.interpreter("default")} {script("freeze")} --dry-run (reads the archive '
           f'and the links; writes nothing)')
    res = legacy.run(script('freeze'), ['--dry-run'], python='default', refusal_codes=(), refusal_text=())
    for line in res.tail.splitlines():
        if line.startswith(('DRY-RUN', 'PROBLEM')):
            TS.say(f'    {line}')
    TS.say(f'    freeze --dry-run exit {res.rc}' + (' (problems listed: a freeze would refuse)' if res.rc else ''))
    return int(res.rc != 0)


# =====================================================================================================================
# stage: sources - the snapshot bright-source PARENT catalogues (M21; exopipe_ops a670970 builder, 3ff6565 installed set)
# =====================================================================================================================

_BUILT_RE = re.compile(r'^(BUILT|FAILED) (\d\dh_\w+) (\d+) gaussians [0-9.]+ s ?(.*)$')


def parent_dir(run):
    """Where the stage writes: <out>/snapshot_master_catalogs (the production set is masters_snapshot.PARENT_CATALOG_DIR,
    an ops path: an install writes there only after cutover)."""
    return os.path.join(run.out_root, 'snapshot_master_catalogs')


def parent_paths(run, slot):
    cat = os.path.join(parent_dir(run), f'master_R075_{slot}.gaul.csv')
    return cat, cat + '.gen.json'


def stage_sources(run, opts, slots):
    """One unit sources_snapshot: exopipe_ops/snapshot_parentcat_contract.py in place (core.legacy.run, site 'default'
    interpreter) with --out-dir <out>/snapshot_master_catalogs --work <node-local scratch> --only <slots>; every BUILT
    catalogue is tested (class snapshot_parent_catalog), every FAILED pair is an item refusal; earlier catalogues of the
    selected slots under --out refuse without --fresh --supersede (the builder itself silently skips a pair whose sidecar
    exists, pc:91-92)."""
    from exopipe.phase3.core import fitsio, legacy
    from exopipe.phase3.lib.masters import tests_snapshot as TSN
    s = _sec()
    masters = {slot: TSN.parent_master_path(slot) for slot in slots}
    miss = [sl for sl, p in masters.items() if not os.path.isfile(p)]
    old = TS.earlier_products([p for sl in slots for p in parent_paths(run, sl)])
    if run.dry_run:
        TS.say(f'[snapshot sources] {len(slots)} parent catalogues -> {parent_dir(run)} (production set: '
               f'{s.PARENT_CATALOG_DIR}); recipe thresh_pix {s.PARENT_THRESH_PIX} thresh_isl {s.PARENT_THRESH_ISL} '
               f'atrous {s.PARENT_ATROUS} on {s.PARENT_MASTER_FILE}')
        TS.say(f'    {legacy.interpreter("default")} {script("parentcat")} --out-dir {parent_dir(run)} --work <node-local '
               f'scratch> --nproc {opts.workers or s.PARENT_NPROC} --only <{len(slots)} slots>')
        if miss:
            TS.say(f'    {len(miss)} slot(s) without a parent master (the builder fails them): {miss[:4]}')
        if old and not (opts.fresh and opts.supersede):
            TS.say(f'    WOULD REFUSE (earlier catalogues under --out, e.g. {old[0]}): pass --fresh --supersede TAG')
        return int(bool(old) and not (opts.fresh and opts.supersede))
    if not slots:
        with run.unit('sources_snapshot', stage='sources') as u:
            raise NoInput('no slot selected')
    with run.unit('sources_snapshot', stage='sources', slots=len(slots)) as u:
        for p in masters.values():
            if os.path.exists(p):
                u.add_input(p)
        u.add_input(script('parentcat'))
        TS.clear_earlier(run, opts, u, old, 'snapshot parent catalogues', gate='G-S01')
        for sl in slots:
            u.plan(parent_paths(run, sl)[0], 'snapshot_parent_catalog')
        work = fitsio.scratch_dir('snapshot_parentcat', opts.work)
        log = os.path.join(run.out_root, 'snapshot', 'logs', 'parentcat_contract.log')
        os.makedirs(os.path.dirname(log), exist_ok=True)
        open(log, 'w').close()
        res = legacy.run(script('parentcat'), ['--out-dir', parent_dir(run), '--work', work, '--nproc',
                                               opts.workers or s.PARENT_NPROC, '--only', ','.join(slots)],
                         python='default', log=log, unit=u, refusal_codes=(), refusal_text=())
        lines = open(log, errors='replace').read().splitlines()
        seen = {}
        for ln in lines:
            m = _BUILT_RE.match(ln.strip())
            if m:
                seen[m.group(2)] = (m.group(1), m.group(4))
        if res.rc != 0 and not any(v[0] == 'FAILED' for v in seen.values()):
            raise RuntimeError(f'snapshot_parentcat_contract.py exit {res.rc} with no FAILED pair (log {log})')
        fitsio.cleanup(work)
        for sl in slots:
            cat = parent_paths(run, sl)[0]
            st, why = seen.get(sl, ('', ''))
            if st == 'BUILT':
                u.add_output(cat, 'snapshot_parent_catalog', slot=sl)
            elif st == 'FAILED':
                u.refuse_item(cat, f'builder FAILED {sl}: {why[:300]}', gate='snapshot parent builder')
        u.note(next((ln for ln in reversed(lines) if ln.startswith('PARENTCAT-CONTRACT-END')), f'builder exit {res.rc}'))
    TS.say(f'  sources_snapshot: {u.status}' + (f' ({u.reason[:200]})' if u.reason else ''))


# =====================================================================================================================
# the tier interface (steps/masters.py dispatches here)
# =====================================================================================================================

def add_arguments(p):
    """No snapshot-only flags: slots come from --cells (deep cell grammar, patterns allowed), --lsts, --scopes,
    --exclude-cells; --workers sets the slots hashed at once (default masters_snapshot.MD5_WORKERS)."""
    return None


def default_out(args):
    return cfg().paths.PHASE3_ARCHIVE_ROOT


def run(args, run):
    stages = TS.resolve_stages(args, TIER)
    opts = TS.Opts.from_args(args)
    slots = selected_slots(args)
    TS.say(f'masters --tier snapshot: stages {",".join(stages)}; {len(slots)} slots; out {run.out_root}'
           + ('; DRY RUN (nothing written)' if run.dry_run else ''))
    viol = mirror_violations()
    msg = ('config differs from the in-place code it mirrors: ' + '; '.join(viol)) if viol else TS.waiver_problem(run)
    if msg:
        if run.dry_run:
            TS.say(f'WOULD REFUSE: {msg}')
            return record.EXIT_REFUSED
        raise Refusal(msg)
    bad = 0
    for st in stages:
        if st == 'verify-frozen':
            bad += stage_verify(run, opts, slots)
        elif st == 'freeze':
            bad += stage_freeze(run, opts)
        elif st == 'sources':          # reachable once masters.STAGES['snapshot'] lists it (step owner's draft)
            bad += stage_sources(run, opts, slots) or 0
    if run.dry_run:
        return record.EXIT_REFUSED if bad else record.EXIT_DONE
    return None
