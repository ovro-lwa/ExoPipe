"""The retention table and the holds (config section retention; plan/ORCHESTRATION_DESIGN_20261117.md 5.4: "Each hour's MS gets
an expiry date of production + 14 days, recorded in a retention table. The team marks holds (hour, subband, time range)
through a review-page action or a command.").

PRODUCTION RECORDS, <out>/retention.TABLE_NAME, one row per recorded range: the raw data of the subbands ``subbands`` ('*' =
every subband) whose 10-s sets START in [start_utc, end_utc) went through the pipeline at produced_utc. The per-hour driver
records each hour it finishes (record_production), a person with `exopipe3 retention record`. A raw file's expiry is the
LATEST produced_utc of the records covering it plus retention.RETENTION_DAYS (a re-processing restarts its clock); a file no
record covers never expires.
HOLDS, <out>/retention.HOLDS_NAME: a hold row (hold_id, start_utc, end_utc, subbands, reason) keeps every raw hour whose UT
hour overlaps [start_utc, end_utc) from deletion until a release row of the same hold_id. Holds are for extraction: a held
range is never deleted, whatever its expiry.
Both files are append only: each row is ONE write on an O_APPEND descriptor (concurrent writers cannot interleave a row), and a
new file appears with its header already in it (written to a temporary file, then hard-linked into place). A row that cannot
be parsed refuses (RT-TABLE): a damaged table is never read as fewer records.
Times are UTC, written YYYY-MM-DDTHH:MM:SSZ.
"""
import calendar
import csv
import getpass
import io
import os
import re
import socket
import time
import uuid

from exopipe.phase3.core.record import Refusal

TABLE_COLUMNS = ('record_id', 'start_utc', 'end_utc', 'subbands', 'produced_utc', 'source', 'recorded_utc', 'recorded_by')
HOLD_COLUMNS = ('hold_id', 'action', 'start_utc', 'end_utc', 'subbands', 'reason', 'recorded_utc', 'recorded_by')
_FORMS = ('%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%MZ', '%Y-%m-%dT%H:%M', '%Y-%m-%dT%HZ', '%Y-%m-%d')


def cfg():
    from exopipe.phase3.config import CFG
    return CFG.retention


def parse_utc(text):
    """Seconds since 1970 (UTC) of 'YYYY-MM-DDTHH:MM:SSZ' (or a shorter form of it, or a date). ValueError otherwise."""
    t = (text or '').strip()
    for f in _FORMS:
        try:
            return calendar.timegm(time.strptime(t, f))
        except ValueError:
            pass
    raise ValueError('not a UTC time YYYY-MM-DDTHH:MM:SSZ: %r' % text)


def iso(t):
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(t))


def parse_subbands(text):
    """None (every subband) for '*' or empty, else the set of names; a name that is not a subband (retention.SUBBAND_RE)
    raises ValueError."""
    t = (text or '').strip()
    if t in ('', '*'):
        return None
    names = {s.strip() for s in re.split(r'[,\s]+', t) if s.strip()}
    bad = sorted(s for s in names if not re.match(cfg().SUBBAND_RE, s))
    if bad:
        raise ValueError('not subband names (%s): %s' % (cfg().SUBBAND_RE, ', '.join(bad)))
    return names


def _who():
    return '%s@%s' % (getpass.getuser(), socket.gethostname().split('.')[0])


def _line(columns, row):
    buf = io.StringIO()
    csv.writer(buf, lineterminator='\n').writerow([row.get(c, '') for c in columns])
    return buf.getvalue()


def append_row(path, columns, row):
    """Append ``row`` (a dict) to the CSV ``path`` in one O_APPEND write, fsynced; a new file is linked into place with its
    header (so no reader or appender ever sees a file without one)."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    if not os.path.exists(path):
        tmp = os.path.join(d, '.%s.%s.tmp' % (os.path.basename(path), uuid.uuid4().hex[:12]))
        with open(tmp, 'w') as f:
            f.write(_line(columns, {c: c for c in columns}))
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            pass                          # another writer created it first: its header is the header
        finally:
            os.unlink(tmp)
    with open(path, newline='') as f:
        head = next(csv.reader(f), [])
    if tuple(head) != tuple(columns):
        raise Refusal('RT-TABLE: %s has the header %s, not %s - not appended' % (path, head, list(columns)))
    fd = os.open(path, os.O_WRONLY | os.O_APPEND)
    try:
        os.write(fd, _line(columns, row).encode())
        os.fsync(fd)
    finally:
        os.close(fd)


def _read(path, columns):
    if not os.path.exists(path):
        return []
    with open(path, newline='') as f:
        rd = csv.reader(f)
        head = next(rd, [])
        if tuple(head) != tuple(columns):
            raise Refusal('RT-TABLE: %s has the header %s, not %s' % (path, head, list(columns)))
        return [(n, dict(zip(columns, r))) for n, r in enumerate(rd, start=2) if r]


def table_path(out):
    return os.path.join(out, cfg().TABLE_NAME)


def holds_path(out):
    return os.path.join(out, cfg().HOLDS_NAME)


def read_records(out):
    """The production records of <out>'s table: [{record_id, start, end, subbands (None = all), produced, source}], times in
    seconds (UTC). No table: []. A row that cannot be parsed refuses (RT-TABLE)."""
    p, out_rows = table_path(out), []
    for n, r in _read(table_path(out), TABLE_COLUMNS):
        try:
            rec = dict(record_id=r['record_id'], start=parse_utc(r['start_utc']), end=parse_utc(r['end_utc']),
                       subbands=parse_subbands(r['subbands']), produced=parse_utc(r['produced_utc']), source=r['source'])
        except (ValueError, KeyError, TypeError) as e:
            raise Refusal('RT-TABLE: %s line %d cannot be read (%s)' % (p, n, e))
        if not rec['record_id'] or rec['end'] <= rec['start']:
            raise Refusal('RT-TABLE: %s line %d: no record_id, or end_utc not after start_utc' % (p, n))
        out_rows.append(rec)
    return out_rows


def read_holds(out):
    """The ACTIVE holds of <out>'s holds file (a hold row without a later release row of its hold_id): [{hold_id, start, end,
    subbands, reason}]. No file: []. A row that cannot be parsed, or a release of an unknown hold, refuses (RT-TABLE)."""
    p, holds = holds_path(out), {}
    for n, r in _read(p, HOLD_COLUMNS):
        try:
            if r['action'] == 'hold':
                h = dict(hold_id=r['hold_id'], start=parse_utc(r['start_utc']), end=parse_utc(r['end_utc']),
                         subbands=parse_subbands(r['subbands']), reason=r['reason'])
                if not h['hold_id'] or h['end'] <= h['start']:
                    raise ValueError('no hold_id, or end_utc not after start_utc')
                holds[h['hold_id']] = h
            elif r['action'] == 'release':
                if r['hold_id'] not in holds:
                    raise ValueError('release of hold %r, which is not held' % r['hold_id'])
                del holds[r['hold_id']]
            else:
                raise ValueError('action %r is neither hold nor release' % r['action'])
        except (ValueError, KeyError, TypeError) as e:
            raise Refusal('RT-TABLE: %s line %d cannot be read (%s)' % (p, n, e))
    return list(holds.values())


def _subbands(subbands, code):
    """parse_subbands of a text or a collection; a name that is not a subband refuses with ``code``."""
    try:
        return parse_subbands(subbands if isinstance(subbands, str) or subbands is None else ','.join(sorted(subbands)))
    except ValueError as e:
        raise Refusal('%s: %s' % (code, e))


def _new_id(prefix):
    return '%s-%s-%s' % (prefix, time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()), uuid.uuid4().hex[:6])


def record_production(out, start, end, subbands, produced, source, now=None):
    """Append a production record: the raw sets of ``subbands`` (None or '*' = every subband) starting in [start, end) went
    through the pipeline at ``produced`` (seconds, UTC). Refuses (RT-RECORD) a range that is empty, a production before the
    range's start or in the future, or no source. Returns the record_id."""
    now = time.time() if now is None else now
    sb = _subbands(subbands, 'RT-RECORD')
    if end <= start:
        raise Refusal('RT-RECORD: end %s is not after start %s' % (iso(end), iso(start)))
    if produced < start or produced > now + 60:
        raise Refusal('RT-RECORD: produced %s is before the range starts (%s) or in the future' % (iso(produced), iso(start)))
    if not (source or '').strip():
        raise Refusal('RT-RECORD: a production record names its source (the chain, job or ledger ID that produced the range)')
    rid = _new_id('P')
    append_row(table_path(out), TABLE_COLUMNS, dict(
        record_id=rid, start_utc=iso(start), end_utc=iso(end), subbands='*' if sb is None else ','.join(sorted(sb)),
        produced_utc=iso(produced), source=source.strip(), recorded_utc=iso(now), recorded_by=_who()))
    return rid


def hold(out, start, end, subbands, reason, now=None):
    """Append a hold of [start, end) on ``subbands`` (None or '*' = every subband). Returns the hold_id."""
    sb = _subbands(subbands, 'RT-HOLD')
    if end <= start:
        raise Refusal('RT-HOLD: end %s is not after start %s' % (iso(end), iso(start)))
    if not (reason or '').strip():
        raise Refusal('RT-HOLD: a hold names its reason (the candidate, the extraction, a ledger ID)')
    hid = _new_id('H')
    append_row(holds_path(out), HOLD_COLUMNS, dict(
        hold_id=hid, action='hold', start_utc=iso(start), end_utc=iso(end),
        subbands='*' if sb is None else ','.join(sorted(sb)), reason=reason.strip(),
        recorded_utc=iso(time.time() if now is None else now), recorded_by=_who()))
    return hid


def release(out, hold_id, reason, now=None):
    """Append the release of an active hold (RT-HOLD when ``hold_id`` is not active)."""
    if hold_id not in {h['hold_id'] for h in read_holds(out)}:
        raise Refusal('RT-HOLD: %s is not an active hold of %s' % (hold_id, holds_path(out)))
    if not (reason or '').strip():
        raise Refusal('RT-HOLD: a release names its reason')
    append_row(holds_path(out), HOLD_COLUMNS, dict(
        hold_id=hold_id, action='release', reason=reason.strip(), recorded_utc=iso(time.time() if now is None else now),
        recorded_by=_who()))
