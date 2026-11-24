"""Content tests of the run versions and their retention (lib/serve/versions.py). Read only.

  run_version_placed  placed (BUILDING.json of a version placed from a build directory: its run directory holds the build)
  run_version   sealed_complete (the run's CONTENT against MANIFEST.tsv, versions.verify: every file and link present,
                nothing added, mutable review state left out, every link's target, and the sha256 re-read of every small or
                text file, of the stable sample of the large files and of every file whose size or modification time
                differs - never names, sizes, times or the record checked against itself),
                switched_to (the pointer's log records the switch to this version)
  run_version_adopted  adopted_sealed (an adopted legacy run given its manifest, recorded as adopted at a switch)
  retire_plan   plan_consistent (a dry run never plans the served or a kept version; an applied plan's versions are gone and
                logged)
"""
import json
import os

from exopipe.phase3.core.validate import content_test


def _version(path):
    from exopipe.phase3.lib.serve import versions
    d = os.path.dirname(os.path.abspath(path))
    with open(path) as f:
        rec = json.load(f)
    return versions.Version(os.path.dirname(d), os.path.basename(d), rec.get('name') or ''), rec


@content_test('run_version_placed')
def placed(path, placed_from=None, **ctx):
    """BUILDING.json names this version and the build directory it was placed from, and the version's run directory is a
    directory holding at least one entry (the build, moved in by one rename; sealed later or not)."""
    from exopipe.phase3.lib.serve import versions
    d = os.path.dirname(os.path.abspath(path))
    with open(path) as f:
        b = json.load(f)
    v = versions.Version(os.path.dirname(d), os.path.basename(d), b.get('name') or '')
    if b.get('version') != v.vid or not b.get('name'):
        return False, None, f'BUILDING.json names version {b.get("version")!r}, name {b.get("name")!r} - not {v.vid}'
    if placed_from and os.path.normpath(b.get('placed_from') or '') != os.path.normpath(os.path.abspath(placed_from)):
        return False, None, f'placed from {b.get("placed_from")}, not {placed_from}'
    if not os.path.isdir(v.run) or os.path.islink(v.run):
        return False, None, f'no run directory {v.run}'
    n = len(os.listdir(v.run))
    return n > 0, n, f'{v.run}: {n} entries' + ('' if n else ' - an empty build')


@content_test('run_version')
def sealed_complete(path, **ctx):
    """VERSION.json is a sealed completion record, its manifest's sha256 is the recorded one, and the run on disk is
    MANIFEST.tsv in content (versions.verify without a record: the run walked, every small or text file, the stable sample of
    the large ones and every file changed in size or time re-hashed against the manifest's sha256). The value: the bytes
    re-read. The record is never only compared with itself."""
    from exopipe.phase3.lib.serve import versions
    v, rec = _version(path)
    if rec.get('state') != 'sealed':
        return False, rec.get('state'), f'state {rec.get("state")!r}, not sealed'
    st = {}
    ok, detail = versions.verify(v, stats=st)
    return ok, st.get('bytes_read'), detail


@content_test('run_version_adopted')
def adopted_sealed(path, **ctx):
    """An adopted legacy run given its manifest (versions.seal_adopted): state sealed, adopted, its run walked again and equal
    to MANIFEST.tsv in content (versions.verify: CONTENT), and R/POINTER_LOG.jsonl records its adoption."""
    from exopipe.phase3.lib.serve import versions
    v, rec = _version(path)
    if rec.get('state') != 'sealed' or not rec.get('adopted'):
        return False, rec.get('state'), f'state {rec.get("state")!r}, adopted {rec.get("adopted")!r} - not an adopted sealed run'
    ok, detail = versions.verify(v)
    if not ok:
        return False, rec.get('n_files'), detail
    log = os.path.join(v.root, 'POINTER_LOG.jsonl')
    try:
        with open(log) as f:
            hit = [r for r in map(json.loads, (x for x in f if x.strip())) if r.get('adopted') == v.vid]
    except (OSError, ValueError) as e:
        return False, None, f'{log}: {e}'
    if not hit:
        return False, rec.get('n_files'), f'no adoption of {v.vid} in {log}'
    return True, rec.get('n_files'), f'{detail}; adopted at the switch of {hit[-1].get("utc")}'


@content_test('run_version')
def switched_to(path, pointer=None, **ctx):
    """R/POINTER_LOG.jsonl holds a switch of ``pointer`` to this version."""
    v, rec = _version(path)
    log = os.path.join(v.root, 'POINTER_LOG.jsonl')
    try:
        with open(log) as f:
            rows = [json.loads(x) for x in f if x.strip()]
    except (OSError, ValueError) as e:
        return False, None, f'{log}: {e}'
    hit = [r for r in rows if r.get('to') == v.vid and (pointer is None or os.path.normpath(r.get('pointer') or '')
                                                        == os.path.normpath(os.path.abspath(pointer)))]
    if not hit:
        return False, 0, f'no switch of {pointer} to {v.vid} in {log}'
    return True, len(hit), f'switched {hit[-1].get("utc")} from {hit[-1].get("from")}'


@content_test('retire_plan')
def plan_consistent(path, applied=False, **ctx):
    """The plan never names the version served at its time or a kept one; applied: every planned version is gone from the
    versions root and RETIRE_LOG.jsonl records it under this plan."""
    with open(path) as f:
        doc = json.load(f)
    plan = [p['version'] for p in doc.get('retire', [])]
    kept = {k['version'] for k in doc.get('keep_versions', [])}
    bad = [p for p in plan if p == doc.get('serves') or p in kept]
    if bad or int(doc.get('keep', 0)) < 1:
        return False, len(plan), f'plans served or kept versions {bad} (keep {doc.get("keep")})'
    if not applied:
        return True, len(plan), f'dry run: {len(plan)} version(s) planned, {len(kept)} kept'
    R = doc.get('root') or ''
    left = [p for p in plan if os.path.exists(os.path.join(R, p))]
    try:
        with open(os.path.join(R, 'RETIRE_LOG.jsonl')) as f:
            logged = {r.get('version') for r in map(json.loads, (x for x in f if x.strip()))
                      if os.path.normpath(r.get('plan') or '') == os.path.normpath(os.path.abspath(path))}
    except (OSError, ValueError):
        logged = set()
    missing = [p for p in plan if p not in logged]
    if left or missing:
        return False, len(plan), f'still present {left[:3]}; not logged {missing[:3]}'
    return True, len(plan), f'{len(plan)} version(s) retired and logged'
