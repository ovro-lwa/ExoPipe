"""A view of one LST's 10-minute search jobs that holds one night's rows, or every row except one night's.

A 10-minute search job directory, <search dir>/<LST>_<band>_<Stokes>, holds the cell's CUMULATIVE records under
transients/Run_<label>/: master_catalog.csv, v_deep_ledger.csv, v_deep_images.csv and search_table_record.csv carry
every window searched so far (a per-hour search appends its night's rows and publishes the cumulative files,
lib/search/tier_sublst.py), and candidates/ holds one directory per candidate of every window. The catalog chain's merge
(production's p3s3_merge.py, in place) reads the whole of them.

A per-hour catalog job works on ONE night (Gregg 2026-11-12, R-GREGG-20261112-PER-HOUR-STEPS-SIX-RULINGS (1), (2): a
new night does not change an earlier verdict, and it is judged by frozen statistics). It is handed a VIEW: a directory
with the layout of the search directory in which

- the four record files hold their header and the rows of the chosen windows only, each kept row byte for byte;
- candidates/ holds links to the candidate directories of the chosen windows only;
- lock files and writers' temporary files (<name>.tmp, <name>.tmp.<pid>) are left out; every other entry is a link to
  the job's own.

mode 'night' chooses the windows of the night; mode 'without' chooses every other window (the earlier nights' pass of
the held-out-night test, and nothing else). The chain reads a view through catalog_sublst.SEARCH_DIR.

Nothing here judges a row. build() refuses rather than guess: a record file without a `date` column, a row whose date is
not a night or a window (YYYY-MM-DD or YYYY-MM-DDTnn), a record that spans lines, a candidate directory whose name
carries no window. A refused or failed build leaves no partial view. verify() is the view's content test: it re-reads
the jobs and the view and compares them; it reports a defect, it does not raise on one.

A SEARCH PUBLISHING MEANWHILE: a job's four record files are read one after the other, without the search's lease of
the cell. build_job() takes each file's size, modification time and inode before the first read and after the last and
refuses when one moved; verify(), run after every job is built, then compares the view with the jobs as they are. An
append of ANOTHER night does not change the chosen night's rows, so a view of the night is right before and after it; a
search that rewrites the chosen night's rows while the view is built is caught by one of the two.

Standard library only.
"""
import csv
import glob
import json
import os
import re
import shutil
import sys
import time

RECORDS = ('master_catalog.csv', 'v_deep_ledger.csv', 'v_deep_images.csv', 'search_table_record.csv')
MODES = ('night', 'without')
MANIFEST = 'night_view.json'
_NIGHT = re.compile(r'^\d{4}-\d\d-\d\d$')
_WINDOW = re.compile(r'^(\d{4}-\d\d-\d\d)(T\d\d)?$')
_CAND = re.compile(r'^[A-Za-z0-9]+_\d\dh_(\d{4}-\d\d-\d\d)(T\d\d)?_[A-Za-z0-9]+_\d+$')
_OPEN = dict(newline='', encoding='utf-8', errors='surrogateescape')     # a kept row is written back byte for byte
_LEFT_OUT = re.compile(r'(\.lock|\.tmp(\.\d+)?)$')                         # a lock, a writer's temporary file


class ViewRefusal(RuntimeError):
    """The view cannot be built, or it is not what its jobs say."""


def _check(night, mode):
    if not _NIGHT.match(night or ''):
        raise ViewRefusal(f'night {night!r} is not YYYY-MM-DD')
    if mode not in MODES:
        raise ViewRefusal(f'mode {mode!r} is not one of {MODES}')


def chosen(window_night, night, mode):
    """Whether a row (or a candidate) of the night ``window_night`` belongs to the view."""
    return (window_night == night) if mode == 'night' else (window_night != night)


def jobs_of(src, lst):
    """The LST's search job directories, as the merge stage lists them."""
    return sorted(d for d in glob.glob(os.path.join(src, f'{lst}_*_[IV]')) if os.path.isdir(d))


def run_dir(job, label):
    return os.path.join(job, 'transients', f'Run_{label}')


def read_record(path):
    """(header line, [(night of the row's date, raw line)]) of a record file; refuses what it cannot place."""
    with open(path, **_OPEN) as f:
        parts = f.read().split('\n')                        # lines end at a line feed only (str.splitlines knows ten more ends)
    lines = [p + '\n' for p in parts[:-1]] + ([parts[-1]] if parts[-1] else [])
    if not lines:
        return '', []
    rd = csv.reader(lines)
    try:
        hdr = next(rd)
    except (StopIteration, csv.Error) as e:
        raise ViewRefusal(f'{path}: unreadable header ({e})')
    if rd.line_num != 1:
        raise ViewRefusal(f'{path}: the header spans {rd.line_num} lines')
    if 'date' not in hdr:
        raise ViewRefusal(f'{path}: no `date` column (columns {hdr[:6]}...)')
    i, rows, at = hdr.index('date'), [], 1
    try:
        for r in rd:
            if rd.line_num != at + 1:
                raise ViewRefusal(f'{path}: the record at line {at + 1} spans {rd.line_num - at} lines')
            at = rd.line_num
            if not r:
                raise ViewRefusal(f'{path}: empty line {at}')
            m = _WINDOW.match(r[i]) if len(r) > i else None
            if not m:
                raise ViewRefusal(f'{path} line {at}: date {r[i] if len(r) > i else None!r} is not a night or a window')
            rows.append((m.group(1), lines[at - 1]))
    except csv.Error as e:
        raise ViewRefusal(f'{path}: unreadable at line {at + 1} ({e})')
    return lines[0], rows


def _cut(src, dst, night, mode):
    hdr, rows = read_record(src)
    keep = [ln for d, ln in rows if chosen(d, night, mode)]     # only the file's last line can lack its line feed
    tmp = dst + '.tmp'
    with open(tmp, 'w', **_OPEN) as f:
        f.write(hdr)
        if keep and hdr and not hdr.endswith('\n'):
            raise ViewRefusal(f'{src}: rows after a header without a newline')
        f.writelines(keep)
    os.replace(tmp, dst)
    return len(rows), len(keep)


def _cand_night(name, where):
    m = _CAND.match(name)
    if not m:
        raise ViewRefusal(f'{where}: candidate directory {name!r} carries no window in its name')
    return m.group(1)


def _stat(path):
    """[size, modification time in ns, inode] of a file; None when it is not there."""
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return None
    return [st.st_size, st.st_mtime_ns, st.st_ino]


def _is_link_to(path, target):
    try:
        return os.path.islink(path) and os.readlink(path) == target
    except OSError:
        return False


def build_job(job, dst, label, night, mode):
    """The view of one job directory; returns {'records': {name: [rows, rows kept]}, 'candidates': [n, n kept], 'stat':
    {name: [size, modification time ns, inode] of the job's record file as it was read}}. REFUSES when a record file
    of the job changed between the first read and the last (a search publishing to the cell)."""
    _check(night, mode)
    job = os.path.abspath(job)
    if os.path.lexists(dst):
        raise ViewRefusal(f'{dst} exists (a view is built once, in a new directory)')
    rs = run_dir(job, label)
    out = {'records': {}, 'candidates': [0, 0]}
    if not os.path.isfile(os.path.join(rs, 'v_deep_ledger.csv')):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.symlink(job, dst)                                # not a completed search: the merge skips it by name, as before
        out['linked_whole'] = True
        return out
    rd = run_dir(dst, label)
    os.makedirs(rd)
    before = {e: _stat(os.path.join(rs, e)) for e in RECORDS}
    for e in sorted(os.listdir(job)):
        if e != 'transients':
            os.symlink(os.path.join(job, e), os.path.join(dst, e))
    for e in sorted(os.listdir(os.path.join(job, 'transients'))):
        if e != f'Run_{label}':
            os.symlink(os.path.join(job, 'transients', e), os.path.join(dst, 'transients', e))
    for e in sorted(os.listdir(rs)):
        s, d = os.path.join(rs, e), os.path.join(rd, e)
        if e in RECORDS:
            out['records'][e] = list(_cut(s, d, night, mode))
        elif e == 'candidates' and os.path.isdir(s):
            os.makedirs(d)
            names = sorted(os.listdir(s))
            keep = [n for n in names if chosen(_cand_night(n, s), night, mode)]
            for n in keep:
                os.symlink(os.path.join(s, n), os.path.join(d, n))
            out['candidates'] = [len(names), len(keep)]
        elif _LEFT_OUT.search(e):
            continue
        else:
            os.symlink(s, d)
    moved = sorted(e for e in RECORDS if _stat(os.path.join(rs, e)) != before[e])
    if moved:
        raise ViewRefusal(f'{rs}: {", ".join(moved)} changed while the view was built (a search publishing to the cell?) - '
                          f'build the view again when the search has ended')
    out['stat'] = {e: v for e, v in before.items() if v is not None}
    return out


def build(src, dst, lst, label, night, mode):
    """The view of every job of the LST under ``dst`` (a new directory); writes <dst>/night_view.json; returns it."""
    _check(night, mode)
    src, dst = os.path.abspath(src), os.path.abspath(dst)
    jobs = jobs_of(src, lst)
    if not jobs:
        raise ViewRefusal(f'no search job of {lst} in {src}')
    if os.path.lexists(dst):
        raise ViewRefusal(f'{dst} exists (a view is built once, in a new directory)')
    os.makedirs(dst)
    rec = {'night': night, 'mode': mode, 'lst': lst, 'label': label, 'search_dir': src,
           'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'jobs': {}}
    try:
        for job in jobs:
            rec['jobs'][os.path.basename(job)] = build_job(job, os.path.join(dst, os.path.basename(job)), label, night, mode)
    except BaseException:                                   # a refused or failed build leaves no partial view: the next
        shutil.rmtree(dst, ignore_errors=True)              # build would refuse on it ("exists"). dst was made by this
        raise                                               # call and holds links and cut copies only (links are unlinked)
    tot = {n: [sum(j['records'].get(n, [0, 0])[k] for j in rec['jobs'].values()) for k in (0, 1)] for n in RECORDS}
    rec['rows'] = tot
    rec['candidates'] = [sum(j['candidates'][k] for j in rec['jobs'].values()) for k in (0, 1)]
    tmp = os.path.join(dst, MANIFEST + '.tmp')
    with open(tmp, 'w') as f:
        json.dump(rec, f, indent=1)
    os.replace(tmp, os.path.join(dst, MANIFEST))
    return rec


def _parsed(path):
    with open(path, **_OPEN) as f:
        rows = list(csv.reader(f))
    return (rows[0] if rows else []), rows[1:]


def verify(dst):
    """The view's content test: (ok, number of record rows in the view, detail). Each record file of the view holds the
    job's header and exactly the job's rows of the chosen windows, in the job's order; candidates/ holds exactly the job's
    candidate directories of the chosen windows; every other entry of the job is linked, lock files aside."""
    dst = os.path.abspath(dst)
    try:
        with open(os.path.join(dst, MANIFEST)) as f:
            rec = json.load(f)
        night, mode, src, lst, label = rec['night'], rec['mode'], rec['search_dir'], rec['lst'], rec['label']
        _check(night, mode)
    except (OSError, ValueError, KeyError, ViewRefusal) as e:
        return False, 0, f'{dst}: no readable {MANIFEST} ({e})'
    bad, nrows, nfiles, ncand = [], 0, 0, 0
    jobs = jobs_of(src, lst)
    if sorted(os.path.basename(j) for j in jobs) != sorted(e for e in os.listdir(dst) if not e.startswith(MANIFEST)):
        bad.append('the view does not hold exactly the search directory\'s jobs')
    for job in jobs:
        name = os.path.basename(job)
        try:
            nr, nf, nc = _verify_job(job, os.path.join(dst, name), name, label, night, mode, bad)
        except OSError as x:                                # a path of the job or of the view that cannot be listed or read
            bad.append(f'{name}: unreadable ({x})')
            continue
        nrows, nfiles, ncand = nrows + nr, nfiles + nf, ncand + nc
    words = 'the night' if mode == 'night' else 'every night except'
    detail = (f'{len(jobs)} jobs of {lst}, {words} {night}: {nfiles} record files with {nrows} rows, {ncand} candidate '
              f'directories' + (f'; {len(bad)} defects, e.g. {bad[:3]}' if bad else ''))
    return not bad, nrows, detail


def _verify_job(job, v, name, label, night, mode, bad):
    """The content test of one job's view ``v``: appends its defects to ``bad``; returns (record rows in the view, record
    files, candidate directories)."""
    nrows = nfiles = ncand = 0
    rs, rv = run_dir(job, label), run_dir(v, label)
    if not os.path.isfile(os.path.join(rs, 'v_deep_ledger.csv')):
        if not _is_link_to(v, job):
            bad.append(f'{name}: not a completed search and not linked whole')
        return 0, 0, 0
    if os.path.islink(v) or not os.path.isdir(rv):
        bad.append(f'{name}: a completed search whose view is a link or holds no Run_{label}')
        return 0, 0, 0
    extra = sorted(set(os.listdir(rv)) - set(os.listdir(rs)))
    if extra:
        bad.append(f'{name}: {len(extra)} entries of the view that the job does not hold, e.g. {extra[:3]}')
    for e in sorted(os.listdir(os.path.join(job, 'transients'))):
        if e != f'Run_{label}' and not _is_link_to(os.path.join(v, 'transients', e), os.path.join(job, 'transients', e)):
            bad.append(f'{name}/transients/{e}: not a link to the job\'s')
    for e in sorted(os.listdir(rs)):
        s, d = os.path.join(rs, e), os.path.join(rv, e)
        if e in RECORDS:
            try:
                hs, rows_s = _parsed(s)
                hv, rows_v = _parsed(d)
            except (OSError, csv.Error) as x:
                bad.append(f'{name}/{e}: unreadable ({x})')
                continue
            i = hs.index('date') if 'date' in hs else None
            want = [r for r in rows_s if i is not None and len(r) > i and _WINDOW.match(r[i])
                    and chosen(r[i][:10], night, mode)]
            other = [r for r in rows_v if i is None or len(r) <= i or not chosen(r[i][:10], night, mode)]
            if hs != hv or rows_v != want or other:
                bad.append(f'{name}/{e}: {len(rows_v)} rows in the view, {len(want)} chosen rows in the job, '
                           f'{len(other)} of another night')
            nrows += len(rows_v)
            nfiles += 1
        elif e == 'candidates' and os.path.isdir(s):
            want = sorted(n for n in os.listdir(s) if _CAND.match(n) and chosen(_CAND.match(n).group(1), night, mode))
            got = sorted(os.listdir(d)) if os.path.isdir(d) else None
            if got != want or any(not _is_link_to(os.path.join(d, n), os.path.join(s, n)) for n in want):
                bad.append(f'{name}/candidates: {None if got is None else len(got)} in the view, {len(want)} chosen in the job')
            ncand += len(want)
        elif _LEFT_OUT.search(e):
            if os.path.lexists(d):
                bad.append(f'{name}/{e}: a lock or temporary file in the view')
        elif not _is_link_to(d, s):
            bad.append(f'{name}/{e}: not a link to the job\'s')
    for e in sorted(os.listdir(job)):
        if e != 'transients' and not _is_link_to(os.path.join(v, e), os.path.join(job, e)):
            bad.append(f'{name}/{e}: not a link to the job\'s')
    return nrows, nfiles, ncand


def main(argv):
    if len(argv) == 7 and argv[0] == 'build':
        try:
            rec = build(*argv[1:])
        except ViewRefusal as e:
            print(f'NIGHT-VIEW REFUSING: {e}')
            return 2
        print(f"NIGHT-VIEW built {argv[2]}: {len(rec['jobs'])} jobs, rows kept of rows "
              + ', '.join(f'{n} {v[1]} of {v[0]}' for n, v in rec['rows'].items())
              + f"; candidates {rec['candidates'][1]} of {rec['candidates'][0]}")
        argv = ['verify', argv[2]]
    if len(argv) == 2 and argv[0] == 'verify':
        ok, n, detail = verify(argv[1])
        print(f"NIGHT-VIEW content test {'PASS' if ok else 'FAIL'}: {detail}")
        return 0 if ok else 2
    print('usage: night_view.py build <search dir> <view dir> <LST> <label> <night YYYY-MM-DD> <night|without>\n'
          '       night_view.py verify <view dir>')
    return 2


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
