"""Digest gate on a directory of pinned input files (the deep parent-catalog unions; FINDING-20261118-PARENT-UNION-OTHER-STEPS-
AND-NO-MANIFEST).

A pinned directory holds a MANIFEST.md5 in md5sum format ('<md5>  <name>' per line, names sorted). The config pins the md5 of
that manifest file itself (paths.PARENT_UNION_MANIFEST_MD5 for the unions), so a regenerated directory with a regenerated
manifest is refused too. ``verify_union`` REFUSES (ManifestRefusal) when:
  M1  the directory has no MANIFEST.md5;
  M2  the md5 of MANIFEST.md5 is not one of the pins;
  M3  the directory's regular files (besides MANIFEST.md5 and the sidecar names in SIDECARS) differ from the manifest's names
      (a file added, removed or renamed);
  M4  a file the caller reads is not in the manifest, or its md5 (plain reads, never a memory map) differs from its line.
A caller names the files it reads (``names``); without names every listed file is checked (about 90 s for a 3.3 GB union on
Lustre). Nothing is cached: each call reads the named files again.
"""
import hashlib
import os

MANIFEST = 'MANIFEST.md5'
SIDECARS = ('MASTER_MD5_DATA.tsv',)     # listed beside the 10-14 union's manifest, not in it (no step reads it)


class ManifestRefusal(RuntimeError):
    """The digest gate refused: the message names the check (M1-M4) and the directory."""


def md5_file(path, block=1 << 22):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(block), b''):
            h.update(b)
    return h.hexdigest()


def read_manifest(path):
    """{name: md5} of an md5sum-format manifest; a malformed or repeated line refuses."""
    out = {}
    with open(path) as f:
        for n, ln in enumerate(f, 1):
            ln = ln.rstrip('\n')
            if not ln:
                continue
            m, sep, name = ln.partition('  ')
            if not sep or len(m) != 32 or not name or name in out:
                raise ManifestRefusal(f'M1: {path}:{n} is not an md5sum line of a new name: {ln[:120]!r}')
            out[name] = m.lower()
    return out


def write_manifest(dirpath, out_path):
    """Write the md5sum-format manifest of every regular file of ``dirpath`` (besides MANIFEST.md5 and SIDECARS) to
    ``out_path``; returns (number of files, md5 of the manifest). For a union about to be pinned; never writes into ``dirpath``
    unless ``out_path`` says so."""
    names = sorted(n for n in os.listdir(dirpath) if n != MANIFEST and n not in SIDECARS
                   and os.path.isfile(os.path.join(dirpath, n)))
    data = ''.join(f'{md5_file(os.path.join(dirpath, n))}  {n}\n' for n in names).encode()
    tmp = f'{out_path}.part'
    with open(tmp, 'wb') as f:
        f.write(data)
    os.replace(tmp, out_path)
    return len(names), hashlib.md5(data).hexdigest()


def verify_union(dirpath, pins, names=None):
    """The digest gate (module docstring). ``pins`` = {label: md5 of MANIFEST.md5}. Returns {'dir', 'union' (the pin's label),
    'manifest_md5', 'n_files', 'checked'} or raises ManifestRefusal."""
    if not dirpath or not os.path.isdir(dirpath):
        raise ManifestRefusal(f'M1: parent-catalog union {dirpath!r} is not a directory')
    mpath = os.path.join(dirpath, MANIFEST)
    if not os.path.isfile(mpath):
        raise ManifestRefusal(f'M1: {dirpath} has no {MANIFEST}: its files cannot be checked against a pinned digest')
    with open(mpath, 'rb') as f:
        mmd5 = hashlib.md5(f.read()).hexdigest()
    label = next((k for k, v in sorted((pins or {}).items()) if v == mmd5), None)
    if label is None:
        raise ManifestRefusal(f'M2: {mpath} has md5 {mmd5}, which is none of the pinned unions '
                              f'({", ".join(f"{k} {v}" for k, v in sorted((pins or {}).items())) or "no pins"}): '
                              f'the union was regenerated or is not a pinned one')
    man = read_manifest(mpath)
    have = {n for n in os.listdir(dirpath) if n != MANIFEST and n not in SIDECARS and os.path.isfile(os.path.join(dirpath, n))}
    extra, missing = sorted(have - set(man)), sorted(set(man) - have)
    if extra or missing:
        raise ManifestRefusal(f'M3: {dirpath} differs from its {MANIFEST} ({label}): {len(extra)} files not listed '
                              f'{extra[:3]}, {len(missing)} listed files absent {missing[:3]}')
    want = sorted(man) if names is None else list(names)
    bad = []
    for n in want:
        if n not in man:
            raise ManifestRefusal(f'M4: {n} is not in {mpath} ({label})')
        got = md5_file(os.path.join(dirpath, n))
        if got != man[n]:
            bad.append(f'{n} md5 {got} != {man[n]}')
    if bad:
        raise ManifestRefusal(f'M4: {len(bad)} of {len(want)} files of {dirpath} differ from its {MANIFEST} ({label}): '
                              + '; '.join(bad[:3]))
    return {'dir': dirpath, 'union': label, 'manifest_md5': mmd5, 'n_files': len(man), 'checked': len(want)}


def parent_union(step_value, cfg=None):
    """The union a step reads: its own key's value (a pinned path) or, when that is None, paths.PARENT_UNION."""
    if step_value:
        return step_value
    if cfg is None:
        from exopipe.phase3.config import CFG as cfg
    return cfg.paths.PARENT_UNION


def union_pins(cfg=None):
    if cfg is None:
        from exopipe.phase3.config import CFG as cfg
    return dict(cfg.paths.PARENT_UNION_MANIFEST_MD5)
