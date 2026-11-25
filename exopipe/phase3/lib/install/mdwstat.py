"""The source of an installed cell's MDWSTAT card value (R-GREGG-20261111-MDWSTAT-CARD): the cell's DEWARP RECORD first, and
install.MDWSTAT_TABLE only when no dewarp record exists for the cell (2026-11-25, the follow-up of
MERGE-20261125-V2-PR173-DEWARP-RECORD-MDWSTAT).

The dewarp record is the unit record <root>/_records/dewarp/<cell>.json of `exopipe3 dewarp`. Its record stage
(lib/dewarp/steps_cell.record) stamps MDWSTAT on the promoted masters of a cell whose dewarp is not accepted, and it
records meta.mdwstat = {value, files: [{path, before, md5}]}: the value and the md5 each stamped master has after the stamp.
That md5 is the record's fingerprint. The record counts only if it is CURRENT: every master the install takes (the
top-level coadd_final files of the generation matching coadd._MASTER_RE, before step b's stamp) is one the record stamped,
with the md5 the record holds, and every master the record stamped is taken (unless --omit-vclean leaves it out). A stale
record REFUSES the cell (gate MDW-REC); it never falls back to the table. Where record and table differ, the record wins
and the difference is noted with the cell and both values.

A record without meta.mdwstat:
  - status done: the apply of an accepted cell (the dewarp's own MDWSTAT 'applied' card, master_dewarp._apply). It carries
    no MDWSTAT value and no md5 of the kind above, so the value comes from the table, with a note (open question: a
    currency rule for applied records);
  - any other status: the cell's dewarp did not finish (refused or failed before the stamp), so the install REFUSES.
    It does not use the table in its place.

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
    {name: md5} (RECORD), the table's value (None = no row), and notes."""

    def __init__(self, cell, source, value, record=None, files=None, table_value=None, notes=()):
        self.cell, self.source, self.value, self.record = cell, source, value, record
        self.files = dict(files or {})
        self.table_value = table_value
        self.notes = list(notes)

    def summary(self):
        return dict(source=self.source, value=self.value, record=self.record, table_value=self.table_value,
                    files=dict(sorted(self.files.items())))


def _table_value(rows, cell):
    if isinstance(rows, str):              # load_mdw_table's error text: the table is unusable (write_cell refuses MDWSTAT)
        return None
    r = (rows or {}).get(cell)
    return None if r is None else (r.get('mdwstat') or '').strip()


def choose(cell, rows, table_path, root):
    """The MDWSTAT source of ``cell`` ('<LST>_<band>'): ``rows`` = release1_cards.load_mdw_table(table_path), ``root`` = the
    dewarp record root (record_root). Returns a Choice; a record that cannot be used refuses (Refusal, gate MDW-REC)."""
    p = record_file(cell, root)
    tv = _table_value(rows, cell)
    tshow = 'no row' if tv is None else repr(tv)
    if not os.path.exists(p):
        return Choice(cell, TABLE, tv, table_value=tv, notes=[
            f'{GATE}: {cell}: no dewarp record {p}: MDWSTAT from the table {table_path} ({tshow})'])
    try:
        with open(p) as f:
            rec = json.load(f)
        meta = rec.get('meta') or {}
        st = rec.get('status')
    except Exception as ex:
        raise Refusal(f'{GATE}: {cell}: the dewarp record {p} cannot be read ({type(ex).__name__}: {str(ex)[:120]})')
    mdw = meta.get('mdwstat')
    if mdw is None:
        if st == 'done':
            kind = (meta.get('install') or {}).get('kind')
            return Choice(cell, TABLE, tv, table_value=tv, notes=[
                f'{GATE}: {cell}: the dewarp record {p} (done; install kind {kind!r}) carries no MDWSTAT value: only the '
                f'record stage of a not-accepted cell writes one (meta.mdwstat); MDWSTAT from the table {table_path} ({tshow})'])
        raise Refusal(f'{GATE}: {cell}: the dewarp record {p} is {st!r} ({str(rec.get("reason") or "")[:200]}) and records no '
                      f'MDWSTAT value: the cell\'s dewarp did not finish - rerun it before the install (the table is not '
                      f'used in its place)')
    if st != 'done':
        raise Refusal(f'{GATE}: {cell}: the dewarp record {p} holds an MDWSTAT value but is {st!r} '
                      f'({str(rec.get("reason") or "")[:200]}): not a finished record - rerun the dewarp record stage')
    value = mdw.get('value') if isinstance(mdw, dict) else None
    files = mdw.get('files') if isinstance(mdw, dict) else None
    bad = None
    if not isinstance(value, str) or not value.strip():
        bad = f'value {value!r}'
    elif not isinstance(files, list) or not files:
        bad = f'files {str(files)[:80]}'
    else:
        for x in files:
            m = x.get('md5') if isinstance(x, dict) else None
            if not (isinstance(x, dict) and x.get('path') and isinstance(m, str) and len(m) == 32 and set(m) <= _HEX):
                bad = f'file entry {str(x)[:120]}'
                break
    if bad is None:
        names = [os.path.basename(x['path']) for x in files]
        if len(set(names)) != len(names):
            bad = 'a master name listed twice'
    if bad:
        raise Refusal(f'{GATE}: {cell}: the dewarp record {p} has a malformed meta.mdwstat ({bad})')
    value = value.strip()
    fm = {os.path.basename(x['path']): x['md5'] for x in files}
    if tv is None:
        why = f'the table {table_path} has no row for the cell'
    elif tv != value:
        why = f'it differs from the table {table_path} ({tv!r}): the record wins'
    else:
        why = 'equal to the table'
    return Choice(cell, RECORD, value, record=p, files=fm, table_value=tv, notes=[
        f'{GATE}: {cell}: MDWSTAT from the dewarp record {p}: {value!r} ({len(fm)} stamped masters); {why}'])


def problems(choice, taken, omitted=()):
    """[] when the dewarp record of ``choice`` is current for the masters being installed: ``taken`` {name: md5} of the
    generation's masters as the install takes them (before step b's stamp); ``omitted``: master names the install leaves out
    (--omit-vclean). Else one line per difference. A table choice is always [] (no fingerprint)."""
    if choice.source != RECORD:
        return []
    out = []
    if not taken:
        out.append('no master of the generation is known to the install (none taken, or none recorded)')
    for n, m in sorted(taken.items()):
        want = choice.files.get(n)
        if want is None:
            out.append(f'{n} (md5 {m}) is not a master the record stamped')
        elif want != m:
            out.append(f'{n} md5 {m} != the record\'s {want}')
    for n in sorted(set(choice.files) - set(taken) - set(omitted)):
        out.append(f'{n} (record md5 {choice.files[n]}) is stamped by the record but not installed')
    return out


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
    record-sourced choices. Truthy, as the table path it replaces."""

    def __init__(self, table, choices=()):
        self.table = table
        self.choices = {c.cell: c for c in choices if c.source == RECORD}

    def rows(self):
        """{'<LST>_<band>': row} as release1_cards.load_mdw_table returns it, a record-sourced cell's row replaced
        (record_row); the loader's error text when the table cannot be used."""
        from exopipe.phase3.lib.masters.release1_cards import load_mdw_table
        t = load_mdw_table(self.table)
        if isinstance(t, str):
            return t
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
        return problems(c, taken, omitted_masters(c.files) if omit_vclean else ())

    def __str__(self):
        return self.table + ''.join(f'; {cell}: MDWSTAT from the dewarp record {c.record} ({c.value!r})'
                                    for cell, c in sorted(self.choices.items()))


def source_for(cell, table, roots):
    """(Choice, Source) for one cell: choose() on the table and the record root of ``roots``."""
    from exopipe.phase3.lib.masters.release1_cards import load_mdw_table
    ch = choose(cell, load_mdw_table(table), table, record_root(roots))
    return ch, Source(table, [ch])


def installed_problems(choice, install_rec):
    """problems() of a record-sourced choice against the masters the cell's install installed, from its done install
    record: the inputs it recorded under <meta.gen>/coadd_final with their md5 (phase3_install's 'installed' rows: the
    generation file's md5 before step b's stamp), and the omissions its MDW-REC gate recorded. Also a difference between the
    source the install recorded (meta.mdwstat_source) and ``choice``."""
    if install_rec is None:
        return ['no done install record of the cell: the installed masters cannot be shown to be the ones the record '
                'stamped'] if choice.source == RECORD else []
    meta = install_rec.get('meta') or {}
    out = []
    was = meta.get('mdwstat_source')
    if was and (was.get('source'), was.get('value')) != (choice.source, choice.value):
        out.append(f'the install took MDWSTAT from the {was.get("source")} ({was.get("value")!r}), the source now is the '
                   f'{choice.source} ({choice.value!r}): reinstall the cell')
    if choice.source != RECORD:
        return out
    gen = os.path.abspath(meta['gen']).rstrip('/') if meta.get('gen') else ''         # record inputs are absolute paths
    fd = os.path.join(gen, 'coadd_final') if gen else None
    sums = {}
    for p, fp in (install_rec.get('inputs') or {}).items():
        if fd and os.path.dirname(p) == fd and isinstance(fp, dict) and fp.get('md5'):
            sums[os.path.relpath(p, gen)] = fp['md5']
    taken = {os.path.basename(r): sums[r] for r in master_names(sums)}
    omitted = (was or {}).get('omitted') or []
    return out + problems(choice, taken, omitted)
