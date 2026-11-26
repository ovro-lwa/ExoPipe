"""The source of an installed cell's MDWSTAT card value (R-GREGG-20261111-MDWSTAT-CARD): the cell's DEWARP RECORD first, and
install.MDWSTAT_TABLE only when no dewarp record exists for the cell (2026-11-25, the follow-up of
MERGE-20261125-V2-PR173-DEWARP-RECORD-MDWSTAT).

The dewarp record is the unit record <root>/_records/dewarp/<cell>.json of `exopipe3 dewarp`. Both ends of a cell's dewarp
record MDWSTAT there (2026-11-25): the apply of an accepted cell (master_dewarp._apply writes MDWSTAT 'applied' on the new
generation) and the record stage of a cell whose dewarp is not accepted (it stamps its value on the promoted masters),
as meta.mdwstat = {value, files: [{path, before, md5, size, mtime_ns}], loo_files: [the same]}: the value and each
master's md5, size and mtime after the write, and the same of each leave-one-out (LOO) master the value was given to
(coadd_final/loo, gate G-R5 of the dewarp step). The md5 is the record's fingerprint. The record counts only if it is
CURRENT: every master the install takes (the top-level coadd_final files of the generation matching coadd._MASTER_RE,
before step b's stamp) is one the record stamped, with the md5 the record holds, and every master the record stamped is
taken (unless --omit-vclean leaves it out); and the same for the LOO masters (coadd_final/loo, coadd._LOO_RE; a record
without loo_files stamped none, so a generation with LOO masters makes it stale). The plan pre-checks size and mtime (no read of the masters' bytes: a size or name difference refuses there, a
rewrite of the same size is left to the md5); the md5 of the staged copies decides (phase3_install, before b's trial
copy). A stale record REFUSES the cell (gate MDW-REC); it never falls back to the
table. Where record and table differ, the record wins and the difference is noted with the cell and both values.

A rerun of the dewarp that refuses or fails does not erase the cell's last finished record: the cell unit keeps it as
meta.last_done (lib/dewarp/steps_cell.keep_last_done). When the newest attempt is not done, or done without an MDWSTAT of
its own (a verify-only rerun), its meta.last_done is used, with a note naming the attempt; the same md5 rule decides, so
an attempt that wrote on the masters makes it stale. A record with no usable MDWSTAT in either place refuses; the table is
never used in its place.

The table is read only where it is needed: for a cell without a dewarp record (it gives the value; a table that cannot
be used refuses that cell at MDW-REC), and for the record-vs-table note of a record-sourced cell (a table that cannot be
used is noted there and never refuses or replaces the record's value). One ``Source`` reads it once, so every step of one
install sees the same rows.

``Source`` is what phase3_install's RELEASE1_MDW_TABLE holds for an install with install.RELEASE1_CARDS. release1_cards
reads its rows(): the table's rows, with the row of a record-sourced cell replaced by the record's value. So step b, the
cleaned-V refresh and the sibling card gate all compare each header with the value from the source the install chose,
never a mix. PI calls staged_problems() on the staged masters' md5 before b's trial copy, before any move."""
import json
import os

from exopipe.phase3.core.record import Refusal

RECORD = 'dewarp record'
TABLE = 'table'
GATE = 'MDW-REC'
_HEX = set('0123456789abcdef')


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def record_root(roots=None):
    """Where the dewarp records are read: install.DEWARP_RECORD_ROOT; when None, the install's own --out root under --out
    (an --out dewarp run writes its records there; a re-rooted install never reads production's), else dewarp.GEN_ROOT
    (the dewarp step's production out root)."""
    c = _cfg()
    if c.install.DEWARP_RECORD_ROOT:
        return c.install.DEWARP_RECORD_ROOT
    if roots is not None and getattr(roots, 'rerooted', False):
        return roots.out
    return c.dewarp.GEN_ROOT


def record_file(cell, root):
    from exopipe.phase3.core import record
    return record.record_path(root, 'dewarp', cell)


class Choice:
    """The source chosen for one cell: source RECORD or TABLE, the value, the record path (RECORD), the stamped masters
    {name: md5} and {name: (size, mtime_ns)} (RECORD; a master without a recorded size and mtime is absent from
    ``stats``), the same of the stamped leave-one-out masters (``loo_files``, ``loo_stats``), the record entry used
    ('meta.mdwstat', or 'meta.last_done.mdwstat' when the newest attempt did not finish), the table's value (None = no
    row, or no usable table: ``table_show`` says which), and notes."""

    def __init__(self, cell, source, value, record=None, files=None, table_value=None, notes=(), stats=None, entry=None,
                 table_show=None, loo_files=None, loo_stats=None):
        self.cell, self.source, self.value, self.record = cell, source, value, record
        self.files = dict(files or {})
        self.stats = dict(stats or {})
        self.loo_files = dict(loo_files or {})
        self.loo_stats = dict(loo_stats or {})
        self.entry = entry
        self.table_value = table_value
        self.table_show = table_show or ('no row' if table_value is None else repr(table_value))
        self.notes = list(notes)

    def summary(self):
        return dict(source=self.source, value=self.value, record=self.record, record_entry=self.entry,
                    table_value=self.table_value, files=dict(sorted(self.files.items())),
                    loo_files=dict(sorted(self.loo_files.items())))


def _table_value(rows, cell):
    if isinstance(rows, str):              # load_mdw_table's error text: the table is unusable (write_cell refuses MDWSTAT)
        return None
    r = (rows or {}).get(cell)
    return None if r is None else (r.get('mdwstat') or '').strip()


def _bad_mdwstat(mdw):
    """None when meta.mdwstat {value, files, loo_files} is well formed, else what is wrong (loo_files may be absent or
    empty: a record that stamped no leave-one-out master)."""
    value = mdw.get('value') if isinstance(mdw, dict) else None
    files = mdw.get('files') if isinstance(mdw, dict) else None
    if not isinstance(value, str) or not value.strip():
        return f'value {value!r}'
    if not isinstance(files, list) or not files:
        return f'files {str(files)[:80]}'
    loo = mdw.get('loo_files', [])
    if not isinstance(loo, list):
        return f'loo_files {str(loo)[:80]}'
    return _bad_entries(files, 'a master') or _bad_entries(loo, 'a leave-one-out master')


def _bad_entries(files, what):
    for x in files:
        m = x.get('md5') if isinstance(x, dict) else None
        if not (isinstance(x, dict) and x.get('path') and isinstance(m, str) and len(m) == 32 and set(m) <= _HEX):
            return f'file entry {str(x)[:120]}'
        if any(k in x and not (isinstance(x[k], int) and not isinstance(x[k], bool) and x[k] >= 0)
               for k in ('size', 'mtime_ns')):
            return f'file entry size / mtime_ns {str(x)[:120]}'
    names = [os.path.basename(x['path']) for x in files]
    if len(set(names)) != len(names):
        return f'{what} name listed twice'
    return None


def choose(cell, rows, table_path, root):
    """The MDWSTAT source of ``cell`` ('<LST>_<band>'). ``rows``: the table's rows as release1_cards.load_mdw_table returns
    them, or a callable returning them, called only where the table is needed: the value of a cell without a dewarp record
    (a table that cannot be used then refuses), and the record-vs-table note of a record-sourced cell (a table that cannot
    be used is noted there, never refusing). ``root`` = the dewarp record root (record_root). Returns a Choice; a record
    that cannot be used refuses (Refusal, gate MDW-REC), never falling back to the table."""
    def table():
        return rows() if callable(rows) else rows
    p = record_file(cell, root)
    if not os.path.exists(p):
        t = table()
        if isinstance(t, str):
            raise Refusal(f'{GATE}: {cell}: no dewarp record {p}, so MDWSTAT comes from the table, and {t}')
        tv = _table_value(t, cell)
        return Choice(cell, TABLE, tv, table_value=tv, notes=[
            f'{GATE}: {cell}: no dewarp record {p}: MDWSTAT from the table {table_path} '
            f'({"no row" if tv is None else repr(tv)})'])
    try:
        with open(p) as f:
            rec = json.load(f)
        meta = rec.get('meta') or {}
        st = rec.get('status')
    except Exception as ex:
        raise Refusal(f'{GATE}: {cell}: the dewarp record {p} cannot be read ({type(ex).__name__}: {str(ex)[:120]})')
    why = str(rec.get('reason') or '')[:200]
    notes = []
    mdw, entry = (meta.get('mdwstat'), 'meta.mdwstat') if st == 'done' else (None, None)
    if mdw is None:
        last = meta.get('last_done')
        if isinstance(last, dict) and last.get('mdwstat') is not None:
            mdw, entry = last['mdwstat'], 'meta.last_done.mdwstat'
            notes.append(f'{GATE}: {cell}: the newest attempt in the dewarp record {p} is {st!r} '
                         + (f'({why}) ' if st != 'done' else '(no MDWSTAT of its own) ')
                         + f'(started {rec.get("started")}); its last done attempt (run {last.get("run_id")}, finished '
                         f'{last.get("finished")}) recorded MDWSTAT: that is used, current only while every master keeps '
                         f'the md5 it stamped')
        elif st == 'done':
            kind = (meta.get('install') or {}).get('kind')
            raise Refusal(f'{GATE}: {cell}: the dewarp record {p} is done (install kind {kind!r}) but records no MDWSTAT '
                          f'(meta.mdwstat, or meta.last_done): every finished dewarp records it since 2026-11-25 (the apply '
                          f'of an accepted cell \'applied\', the record stage of a not-accepted one its value) - rerun the '
                          f'dewarp of the cell (the table is not used in its place)')
        else:
            raise Refusal(f'{GATE}: {cell}: the dewarp record {p} is {st!r} ({why}) and no earlier attempt of it recorded '
                          f'MDWSTAT (meta.last_done): the cell\'s dewarp did not finish - rerun it before the install (the '
                          f'table is not used in its place)')
    bad = _bad_mdwstat(mdw)
    if bad:
        raise Refusal(f'{GATE}: {cell}: the dewarp record {p} has a malformed {entry} ({bad})')
    value = mdw['value'].strip()
    files = mdw['files']
    fm = {os.path.basename(x['path']): x['md5'] for x in files}
    stats = {os.path.basename(x['path']): (x['size'], x['mtime_ns']) for x in files if 'size' in x and 'mtime_ns' in x}
    loo = mdw.get('loo_files') or []
    lm = {os.path.basename(x['path']): x['md5'] for x in loo}
    lstats = {os.path.basename(x['path']): (x['size'], x['mtime_ns']) for x in loo if 'size' in x and 'mtime_ns' in x}
    try:
        t = table()
    except Exception as ex:                # the record's value never depends on the table
        t = f'the MDWSTAT table {table_path} cannot be read ({type(ex).__name__}: {str(ex)[:60]})'
    if isinstance(t, str):
        tv, tshow = None, f'unusable: {t}'
        cmp = f'not compared with the table ({t}): the record does not depend on it'
    else:
        tv = _table_value(t, cell)
        tshow = 'no row' if tv is None else repr(tv)
        cmp = (f'the table {table_path} has no row for the cell' if tv is None else
               f'it differs from the table {table_path} ({tv!r}): the record wins' if tv != value else 'equal to the table')
    notes.append(f'{GATE}: {cell}: MDWSTAT from the dewarp record {p} ({entry}): {value!r} ({len(fm)} stamped masters, '
                 f'{len(lm)} stamped leave-one-out masters); {cmp}')
    return Choice(cell, RECORD, value, record=p, files=fm, table_value=tv, notes=notes, stats=stats, entry=entry,
                  table_show=tshow, loo_files=lm, loo_stats=lstats)


def problems(choice, taken, omitted=(), loo_taken=None):
    """[] when the dewarp record of ``choice`` is current for the masters being installed: ``taken`` {name: md5} of the
    generation's masters as the install takes them (before step b's stamp); ``omitted``: master names the install leaves out
    (--omit-vclean); ``loo_taken`` {name: md5} of the generation's leave-one-out masters as taken (None = none), compared
    with the record's loo_files by the same rule. Else one line per difference. A table choice is always [] (no
    fingerprint)."""
    if choice.source != RECORD:
        return []
    out = []
    if not taken:
        out.append('no master of the generation is known to the install (none taken, or none recorded)')
    out += _md5_problems(choice.files, taken, omitted, 'a master')
    out += _md5_problems(choice.loo_files, loo_taken or {}, (), 'a leave-one-out master')
    return out


def _md5_problems(stamped, taken, omitted, what):
    out = []
    for n, m in sorted(taken.items()):
        want = stamped.get(n)
        if want is None:
            out.append(f'{n} (md5 {m}) is not {what} the record stamped')
        elif want != m:
            out.append(f'{n} md5 {m} != the record\'s {want}')
    for n in sorted(set(stamped) - set(taken) - set(omitted)):
        out.append(f'{n} (record md5 {stamped[n]}) is stamped by the record but not installed')
    return out


def stat_problems(choice, stats, omitted=(), loo_stats=None):
    """The plan's pre-check, which reads no master's bytes: ``stats`` {name: (size, mtime_ns)} of the generation's masters
    now (os.stat). Returns (problems, deferred): a problem - certain staleness - for a master the record did not stamp, a
    size other than the record's, and a stamped master missing from the generation (unless ``omitted``); ``deferred``
    {name: why} for the masters the plan cannot clear: no size and mtime in the record (made before 2026-11-25), or the
    same size with another mtime (the file was rewritten since the stamp - an interrupted rerun rewrites a master with the
    value it already held - so only its bytes can tell). The md5 of the staged copies (staged_problems) decides, for every
    master. ``loo_stats``: the same of the generation's leave-one-out masters (None = none), against the record's
    loo_files and loo_stats by the same rule."""
    if choice.source != RECORD:
        return [], {}
    out, deferred = [], {}
    if not stats:
        out.append('no master of the generation is known to the install (none taken)')
    _stat_problems(choice.files, choice.stats, stats, omitted, 'a master', out, deferred)
    _stat_problems(choice.loo_files, choice.loo_stats, loo_stats or {}, (), 'a leave-one-out master', out, deferred)
    return out, deferred


def _stat_problems(stamped, rstats, stats, omitted, what, out, deferred):
    for n, got in sorted(stats.items()):
        if n not in stamped:
            out.append(f'{n} is not {what} the record stamped')
            continue
        want = rstats.get(n)
        if want is None:
            deferred[n] = 'no size and mtime in the record'
        elif got[0] != want[0]:
            out.append(f'{n} is {got[0]} bytes, the record stamped {want[0]} bytes (changed since the stamp)')
        elif got[1] != want[1]:
            deferred[n] = f'rewritten since the stamp (mtime {got[1]} ns, the record\'s {want[1]} ns; same size)'
    for n in sorted(set(stamped) - set(stats) - set(omitted)):
        out.append(f'{n} is stamped by the record but not in the generation')


def master_names(rels):
    """The masters among generation-relative paths ('coadd_final/<name>', ...): top level of coadd_final, coadd._MASTER_RE
    (the set the record stage stamps, lib/dewarp/steps_common.mdwstat_files)."""
    from exopipe.phase3.lib.masters.coadd import _MASTER_RE
    out = []
    for r in rels:
        d, n = os.path.split(r)
        if d == 'coadd_final' and n.endswith('.fits') and _MASTER_RE.match(n):
            out.append(r)
    return out


def loo_names(rels):
    """The leave-one-out masters among generation-relative paths: coadd_final/loo/<name> matching coadd._LOO_RE (the set
    the dewarp step records as meta.mdwstat.loo_files, lib/dewarp/steps_common.loo_files)."""
    from exopipe.phase3.lib.masters.coadd import _LOO_RE
    return [r for r in rels if os.path.dirname(r) == os.path.join('coadd_final', 'loo') and _LOO_RE.match(os.path.basename(r))]


def omitted_masters(names):
    """The record's masters --omit-vclean leaves out (install.OMIT_VCLEAN_PATTERN, as phase3_install filters them)."""
    pat = _cfg().install.OMIT_VCLEAN_PATTERN
    return [n for n in names if pat in n]


def record_row(choice, trow):
    """The MDWSTAT table row a record-sourced cell is read with: the table's row (or the table's columns, empty) with mdwstat
    = the record's value and the MDWMED1 columns empty. The MDWMED1 columns describe the table's dewarp install, and the
    record speaks for a master that is not dewarped, so the two are not mixed. mdwstat_source / mdwstat_from name the
    record for the card's source text and the gate's messages."""
    r = dict(trow or {})
    r.update(cell=r.get('cell') or choice.cell.replace('_', ' '), mdwstat=choice.value, gate_verdict='',
             mdwmed1_header_arcmin='', mdwmed1_installed_cubic_arcmin='', source=f'{RECORD} {choice.record}',
             mdwstat_source=RECORD, mdwstat_from=choice.record)
    return r


class Source:
    """PI.RELEASE1_MDW_TABLE for an install whose cells may take MDWSTAT from a dewarp record: the table path and the
    record-sourced choices. Truthy, as the table path it replaces. The table is read once per Source (table_rows), so
    every step of one install - the plan's choice, step b, the cleaned-V refresh, the sibling card gate - sees the same
    rows."""

    def __init__(self, table, choices=(), rows=None):
        self.table = table
        self.choices = {}
        self._rows = rows
        for c in choices:
            self.add(c)

    def add(self, choice):
        if choice.source == RECORD:
            self.choices[choice.cell] = choice

    def table_rows(self):
        """The table's rows as release1_cards.load_mdw_table returns them (or its error text), read on the first call."""
        if self._rows is None:
            from exopipe.phase3.lib.masters.release1_cards import load_mdw_table
            self._rows = load_mdw_table(self.table)
        return self._rows

    def rows(self):
        """{'<LST>_<band>': row}: the table's rows, a record-sourced cell's row replaced (record_row). A table that cannot be
        used gives its error text when no cell is record-sourced, else the record-sourced rows alone: it refuses only the
        cells whose value comes from it and never removes a record's value."""
        t = self.table_rows()
        if isinstance(t, str):
            if not self.choices:
                return t
            t = {}
        t = dict(t)
        for cell, c in self.choices.items():
            t[cell] = record_row(c, t.get(cell))
        return t

    def staged_problems(self, cell, staged, omit_vclean=False):
        """problems() for the masters PI staged: ``staged`` {generation-relative path: md5 of the generation file}."""
        c = self.choices.get(cell)
        if c is None:
            return []
        taken = {os.path.basename(r): staged[r] for r in master_names(staged)}
        loo = {os.path.basename(r): staged[r] for r in loo_names(staged)}
        return problems(c, taken, omitted_masters(c.files) if omit_vclean else (), loo)

    def __str__(self):
        return f'{self.table}' + ''.join(f'; {cell}: MDWSTAT from the dewarp record {c.record} ({c.value!r})'
                                         for cell, c in sorted(self.choices.items()))


def source_for(cell, table, roots):
    """(Choice, Source) for one cell: choose() on the record root of ``roots``, the table read through the Source (once,
    and only where needed)."""
    src = Source(table)
    ch = choose(cell, src.table_rows, table, record_root(roots))
    src.add(ch)
    return ch, src


def installed_values(cell_dir, loo=False):
    """{name: MDWSTAT} of the masters in <cell_dir>/coadd_final (top level, coadd._MASTER_RE), or with ``loo`` of the
    leave-one-out masters in <cell_dir>/coadd_final/loo (coadd._LOO_RE); header only (no memory map)."""
    from exopipe.phase3.core.record import fits_primary_cards
    sub = os.path.join('coadd_final', 'loo') if loo else 'coadd_final'
    try:
        names = os.listdir(os.path.join(cell_dir, sub))
    except (FileNotFoundError, NotADirectoryError):
        return {}
    pick = loo_names if loo else master_names
    return {os.path.basename(r): fits_primary_cards(os.path.join(cell_dir, r)).get('MDWSTAT')
            for r in pick(sorted(os.path.join(sub, n) for n in names))}


def installed_problems(choice, install_rec, cell_dir=None):
    """[] when ``choice`` is the source the cell's install used and holds for what it installed. ``install_rec``: the cell's
    newest done install or recover record (None: none). The source the install recorded (meta.mdwstat_source) must be
    ``choice``'s; a record without it (made before 2026-11-25, or with install.RELEASE1_CARDS off) means the TABLE, as its
    step b wrote the table's value. A record-sourced choice must be current for the masters installed: the inputs the
    install recorded under <meta.gen>/coadd_final with their md5 (phase3_install's 'installed' rows: the generation file's
    md5 before step b's stamp), less the omissions its MDW-REC gate recorded, and so must the leave-one-out masters it
    recorded under <meta.gen>/coadd_final/loo (the record's loo_files). In every case the MDWSTAT in the headers of the
    installed masters (``cell_dir``: the working cell) must be ``choice``'s value, and for a record-sourced choice also in
    the headers of the installed leave-one-out masters (the record stamped them; the install's step b stamps masters
    only, so a table-sourced cell's LOO masters are not compared); a difference names both values. Each problem of the
    source ends 'reinstall the cell'."""
    out = []
    meta = (install_rec or {}).get('meta') or {}
    was = meta.get('mdwstat_source') or None
    if install_rec is not None:
        if not was:
            if choice.source != TABLE:
                out.append(f'the install record holds no MDWSTAT source (made before 2026-11-25, or with '
                           f'install.RELEASE1_CARDS off), so its step b took the table\'s value; the source now is the '
                           f'{choice.source} ({choice.value!r}): reinstall the cell')
        elif (was.get('source'), was.get('value')) != (choice.source, choice.value):
            out.append(f'the install took MDWSTAT from the {was.get("source")} ({was.get("value")!r}), the source now is '
                       f'the {choice.source} ({choice.value!r}): reinstall the cell')
    if cell_dir:
        want = (choice.value or '').strip()
        for n, got in sorted(installed_values(cell_dir).items()):
            if str(got if got is not None else '').strip() != want:
                out.append(f'the installed master {n} carries MDWSTAT {got!r}, the {choice.source} gives '
                           f'{choice.value!r}: reinstall the cell')
        if choice.source == RECORD:
            for n, got in sorted(installed_values(cell_dir, loo=True).items()):
                if str(got if got is not None else '').strip() != want:
                    out.append(f'the installed leave-one-out master {n} carries MDWSTAT {got!r}, the {choice.source} '
                               f'gives {choice.value!r}: reinstall the cell')
    if choice.source != RECORD:
        return out
    if install_rec is None:
        return out + ['no done install record of the cell: the installed masters cannot be shown to be the ones the '
                      'record stamped']
    gen = os.path.abspath(meta['gen']).rstrip('/') if meta.get('gen') else ''         # record inputs are absolute paths
    fd = os.path.join(gen, 'coadd_final') if gen else None
    sums = {}
    for p, fp in (install_rec.get('inputs') or {}).items():
        if fd and os.path.dirname(p) in (fd, os.path.join(fd, 'loo')) and isinstance(fp, dict) and fp.get('md5'):
            sums[os.path.relpath(p, gen)] = fp['md5']
    taken = {os.path.basename(r): sums[r] for r in master_names(sums)}
    loo = {os.path.basename(r): sums[r] for r in loo_names(sums)}
    omitted = (was or {}).get('omitted') or []
    return out + problems(choice, taken, omitted, loo)
