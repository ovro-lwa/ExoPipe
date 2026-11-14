"""The window-tie tables the 10-minute conformance read, recorded and checked (refactor-chat ruling 2026-10-18, option (a)
"record and check"; production exopipe_ops 074f96c + 5d98279, IMPLEMENTED-10MIN-CONFORM-RECORD-CHECK-EVERY-READ-20261019).

The window tie runs twice in the 10-minute catalogue chain: on the rows near ATNF pulsars before the conformance (1 of 2) and
on every selected row after it (2 of 2), which rewrites pertie/. The conformance's pulsar step must stay reproducible from the
tables it read, so:
  record   before the conformance, pertie/conform_inputs/ = copies of pertie_detections.csv and pertie_frames.csv and their
           SHA256SUMS (sha256sum format); the conformance and the tablefilter stage read the copy (P3_TENMIN_PERTIE);
           p3s2_conform.py logs 'conform: window-tie table read: <path> sha256 <hex>' on every run (production code, in place)
  check    the pages, publish and cumulative steps refuse a catalogue whose copy is missing, does not verify against its
           SHA256SUMS, or differs from ANY current read line of its log (every read of the current run of each stage)
Ported from sublst_catalogs.sh:183-190 (record), sublst_catalogs.sh:323-327 (tablefilter's verify), sublst_catalog_publish.sh
ci_reads / ci_check (:94-110; the same functions in sublst_cumulative.sh:76-96) and sublst_pages_prep.py:43-63 (CONFORM_LOG.txt).
"""
import hashlib
import os
import shutil

DIR = 'conform_inputs'
FILES = ('pertie_detections.csv', 'pertie_frames.csv')
SUMS = 'SHA256SUMS'
READ = 'conform: window-tie table read: '
LOG_COPY = 'CONFORM_LOG.txt'          # the read lines that travel with the pages (sublst_pages_prep.py:55-63)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def record(pertie_dir):
    """sublst_catalogs.sh:187-190: replace pertie_dir/conform_inputs/ by copies of the two tie tables (cp -p) and their
    SHA256SUMS ('<hex>  <name>' per line, as sha256sum writes it). Returns (the copy dir, the SHA256SUMS text). Raises OSError
    when a table is missing or a copy fails."""
    ci = os.path.join(pertie_dir, DIR)
    if os.path.lexists(ci):
        shutil.rmtree(ci)
    os.makedirs(ci)
    text = ''
    for n in FILES:
        shutil.copy2(os.path.join(pertie_dir, n), os.path.join(ci, n))
        text += f'{sha256_file(os.path.join(ci, n))}  {n}\n'
    with open(os.path.join(ci, SUMS), 'w') as f:
        f.write(text)
    return ci, text


def sums(ci):
    """{name: sha256} of ci/SHA256SUMS ({} when it is missing)."""
    out = {}
    try:
        with open(os.path.join(ci, SUMS)) as f:
            for ln in f:
                w = ln.split()
                if len(w) == 2:
                    out[w[1].lstrip('*')] = w[0]
    except OSError:
        pass
    return out


def verify(ci, complete=True):
    """'' when ci verifies, else the reason. complete: the two tables and SHA256SUMS each present and non-empty first
    (ci_check's `[ -s ]` loop); then every file SHA256SUMS lists has its listed sha256 (`sha256sum -c --quiet --status`)."""
    if complete:
        for n in FILES + (SUMS,):
            p = os.path.join(ci, n)
            if not (os.path.isfile(p) and os.path.getsize(p) > 0):
                return f'no {p}'
    listed = sums(ci)
    if not listed:
        return f'{os.path.join(ci, SUMS)} does not verify'
    for n, h in listed.items():
        try:
            ok = sha256_file(os.path.join(ci, n)) == h
        except OSError:
            ok = False
        if not ok:
            return f'{os.path.join(ci, SUMS)} does not verify'
    return ''


def reads(log_path):
    """ci_reads LOGFILE: the read lines of the CURRENT run of each catalogue stage. A header '<time> CATALOGS <arm> <stage> ...'
    starts that stage's run and drops its earlier lines; lines before any header (a CONFORM_LOG.txt) all count. In order of
    each stage's first appearance. [] when the log is missing."""
    order, buf, st = [], {}, ''
    try:
        with open(log_path, errors='replace') as f:
            lines = f.read().splitlines()
    except OSError:
        return []
    for ln in lines:
        w = ln.split()
        if len(w) > 1 and w[1] == 'CATALOGS':
            st = w[3] if len(w) > 3 else ''
            if st not in buf:
                order.append(st)
            buf[st] = []
        elif ln.startswith(READ):
            if st not in buf:
                order.append(st)
                buf[st] = []
            buf[st].append(ln)
    return [ln for k in order for ln in buf[k]]


def check(pertie_dir, log_path):
    """ci_check DIR LOGFILE: (ok, message). Passes only when pertie_dir/conform_inputs is complete and verifies, the log holds
    at least one current read line, and EVERY one names the recorded copy (.../conform_inputs/pertie_detections.csv) with the
    copy's sha256. The message is the reason of a refusal or 'conformance inputs verified (N reads of ... sha256 H)'."""
    ci = os.path.join(pertie_dir, DIR)
    why = verify(ci)
    if why:
        return False, why
    want = sums(ci).get('pertie_detections.csv', '')
    lines = reads(log_path)
    for lg in lines:
        if f'{DIR}/pertie_detections.csv sha256 ' not in lg:
            return False, f'conformance did not read the recorded copy: {lg[:200]}'
        got = lg.rsplit(' ', 1)[-1]
        if got != want:
            return False, f"logged sha256 {got} != the recorded copy's {want}"
    if not lines:
        return False, f"no 'conform: window-tie table read' line in {log_path}"
    return True, f'conformance inputs verified ({len(lines)} reads of pertie_detections.csv sha256 {want})'
