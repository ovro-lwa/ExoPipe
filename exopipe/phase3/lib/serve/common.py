"""Small helpers shared by the serve modules (no production logic lives here)."""
import csv
import glob
import hashlib
import os
import time


def cfg():
    from exopipe.phase3.config import CFG
    return CFG


def S():
    """The serve config section."""
    return cfg().serve


def utc_text():
    """'2026-10-13 04:12 UT' - the served time the launchers print (date -u '+%Y-%m-%d %H:%M UT')."""
    return time.strftime(S().UTC_FMT, time.gmtime())


def hhmm():
    return time.strftime('%H:%M', time.gmtime())


def stamp():
    """%Y%m%dT%H%M%S (sublst_cumulative.sh:102)."""
    return time.strftime('%Y%m%dT%H%M%S', time.gmtime())


def inc_log(run_dir, text):
    """Append one line to <run>/incremental.log as the launchers do ('HH:MM text')."""
    try:
        with open(os.path.join(run_dir, 'incremental.log'), 'a') as f:
            f.write(f'{hhmm()} {text}\n')
    except OSError:
        pass


def rows(path):
    """Every row of a CSV, read exactly as the launchers' heredocs read it (csv.DictReader(open(p)))."""
    with open(path) as f:
        return list(csv.DictReader(f))


def sha1_file(path):
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def dir_digest(path, pattern='*.csv'):
    """sha1 over (name, sha1) of the files of a directory (the view digest of a served_page record)."""
    h = hashlib.sha1()
    for p in sorted(glob.glob(os.path.join(path, pattern))):
        h.update(os.path.basename(p).encode() + b'\0' + sha1_file(p).encode() + b'\n')
    return h.hexdigest()


def queue_ids(path):
    """Id lines of a queue file (lines starting 'J', as the launchers' grep '^J')."""
    with open(path) as f:
        return [l.rstrip('\n') for l in f if l.startswith('J')]


def has_j_line(path):
    """[ -s Q ] && grep -q '^J' Q (p3s3_serve_stokes.sh:57)."""
    try:
        if os.path.getsize(path) == 0:
            return False
        with open(path) as f:
            return any(l.startswith('J') for l in f)
    except OSError:
        return False
