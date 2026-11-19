"""README notes of an archive run (spec hybrid.md 3.7, PR-8c; 2.16 item 5: no inline code writes into the archive).

The last unit of a push of a class with archive_push.README_TEMPLATES[class], push_<class>_readme, appends one section to its archive
run's archive_push.README_NAME and pushes it like any product: the installed README copied byte for byte, then the section (the
template filled from the push's evidence); the staged file content-tested (class archive_readme); the D2 swap of
lib/archive/hybrid.apply_unit with the installed README quarantined into the push's round as README/<stamp>/README.txt (its own
name; several pushes may share a round) and listed in the round's quarantine manifest, not in the push manifest (every later push
replaces the README, so a push manifest listing it would fail its re-audit); the written file content-tested (class
archive_readme_copy: its checksum, and it begins with every byte of the README it replaced - no history is ever lost).

What a section describes. An apply (the transfer stage) describes what the archive shows and the README does not name yet: every
v2 push manifest of the run (<run>/MANIFEST_<stamp>.csv with the archive_push.HYBRID_PUSH_COLUMNS header; production's manifests
have other columns and are described by hand) whose name the README does not hold, and every retire row (no successor) of the
round's quarantine manifest whose quarantined path the README does not hold (a section lists the paths it retired). So a push
whose README unit stopped (an error, a crash) is described by the next push of the class into the run in the same round, with or
without --resume; a rerun after a complete push adds no section (no-input). A plan or dry run describes the units of this job whose
plans change a file, plus what the archive already holds undescribed. The details (ruled notes, the commits of the step records
that built the products, the units not pushed) come from this job's units.

The gate (precheck, called by the push before its first unit): an apply needs the run's README and an --update-note (why the update
was made) citing a row of paths.FIX_LEDGER; otherwise the whole push refuses and nothing is swapped in. A plan or dry run without a
note renders a placeholder and notes that the apply would refuse.
"""
import csv
import glob
import hashlib
import json
import os
import re
import textwrap

from exopipe.phase3.core.record import NoInput, Refusal
from exopipe.phase3.lib.archive import common as K

DONE = ('done', 'skipped-valid', 'planned')
CHANGES = ('publish', 'replace', 'retire', 'keep-or-replace')     # keep-or-replace: a dry run (no checksums) cannot tell yet
COUNTED = ('replace', 'publish', 'retire', 'keep')
_HYB = re.compile(r'^(\d\dh)_(.+)_(R\w+)$')
NO_NOTE = '<the update note: --update-note "<why, ledger ID>" - an apply refuses without it>'


def _p():
    return K.cfg().archive_push


def _md5(b):
    return hashlib.md5(b).hexdigest()


def writes(ctx, args):
    """True when the push writes the archive (an apply with the transfer stage, not a dry run)."""
    return bool(ctx.apply) and 'transfer' in (args._stages or ()) and not ctx.dry


def _note(args):
    return ' '.join(getattr(args, 'update_note', None) or []).strip()


def precheck(ctx, args, cls, adir):
    """Called by a push before its first unit: an apply of a class with a template needs the run's README and an --update-note
    citing a ledger row. Refusal otherwise: nothing of the push is swapped in."""
    if cls not in (_p().README_TEMPLATES or {}) or not writes(ctx, args):
        return
    inst = os.path.join(adir, _p().README_NAME)
    if not os.path.isfile(inst):
        raise Refusal(f'{inst}: the archive run has no {_p().README_NAME} to append the section of this push to (a new run\'s '
                      'README is written by a ruling) - nothing pushed')
    note = _note(args)
    if not (note and K.ruling_ids([note])):
        raise Refusal(f'push --class {cls} --apply needs --update-note "<why, ledger ID>" citing a row of {K.cfg().paths.FIX_LEDGER} '
                      f'(the README section of this push; {"none given" if not note else "no ledger ID in " + repr(note[:120])}) '
                      '- nothing pushed')


def named(token, text):
    """token occurs in text as a whole name, not inside a longer one (push_x_R0 is not named by push_x_R075)."""
    return re.search(r'(?<![\w.-])' + re.escape(token) + r'(?![\w-])', text) is not None


def facts(run, u):
    """What README needs of one push unit u (the unit object unit_fn returned): this job's own view, or for a resumed unit
    (skipped-valid) its earlier record."""
    if u.status == 'skipped-valid':
        from exopipe.phase3.core.record import load_record
        rec = load_record(run.out_root, run.step, u.name) or {}
        meta, notes, inputs = rec.get('meta') or {}, rec.get('notes') or [], rec.get('inputs') or {}
    else:
        meta, notes, inputs = u.meta, u.notes, u.inputs
    return dict(counts=dict(meta.get('counts') or {}), ruled=[n for n in notes if 'by ruling' in str(n)], inputs=list(inputs))


def _changes(counts):
    return sum(int((counts or {}).get(k) or 0) for k in CHANGES)


def _lst_ranges(lsts):
    """'00h-05h, 07h' of a set of LST labels."""
    hs = sorted(int(x[:2]) for x in lsts)
    out, i = [], 0
    while i < len(hs):
        j = i
        while j + 1 < len(hs) and hs[j + 1] == hs[j] + 1:
            j += 1
        out.append(f'{hs[i]:02d}h' + (f'-{hs[j]:02d}h' if j > i else ''))
        i = j + 1
    return ', '.join(out)


def label(cls, name):
    return name[len(f'push_{cls}_'):] if name.startswith(f'push_{cls}_') else name


def summary(cls, names):
    """The changed units in words: hybrid tiles grouped by scope and weighting with their LSTs; other units by name."""
    groups, other = {}, []
    for n in names:
        s = label(cls, n)
        m = _HYB.match(s)
        if m:
            groups.setdefault((m.group(2), m.group(3)), set()).add(m.group(1))
        else:
            other.append(s.replace('_', ' '))
    order = {s: i for i, s in enumerate(K.cfg().hybrid.SCOPES)}                # the config's scope order (subbands, then widebands)
    parts = [f'{sc} {w} at {_lst_ranges(ls)}' for (sc, w), ls in sorted(groups.items(), key=lambda kv: (order.get(kv[0][0], 99),) + kv[0])]
    return '; '.join(parts + sorted(other)) or 'none'


def _commits(paths, cache):
    """Distinct 'commit' of the step records among a unit's inputs (JSON records under a _records/ directory)."""
    out = set()
    for p in paths:
        if '/_records/' not in p or not p.endswith('.json'):
            continue
        if p not in cache:
            try:
                with open(p) as fh:
                    cache[p] = (json.load(fh) or {}).get('commit') or ''
            except (OSError, ValueError):
                cache[p] = ''
        if cache[p]:
            out.add(cache[p])
    return out


def _rows(path):
    with open(path, newline='') as fh:
        rd = csv.DictReader(fh)
        return list(rd.fieldnames or ()), list(rd)


def pending(ctx, adir, qman, text):
    """What the archive holds and the README text does not name (module docstring): ({unit: {action: files}}, [push manifest],
    [quarantined path of a retired file], relative to ctx.archive_root)."""
    cols = list(_p().HYBRID_PUSH_COLUMNS)
    units, mans, retired = {}, [], []
    for m in sorted(glob.glob(os.path.join(adir, 'MANIFEST_*.csv'))):
        if named(os.path.basename(m), text):
            continue
        have, rows = _rows(m)
        if have != cols:
            continue                                             # production's manifests (unit,src,dst,md5,bytes): described by hand
        mans.append(os.path.relpath(m, ctx.archive_root))
        for r in rows:
            c = units.setdefault(r['unit'], {})
            c[r['action']] = c.get(r['action'], 0) + 1
    if os.path.isfile(qman):
        for r in _rows(qman)[1]:
            q = os.path.relpath(r['dest'], ctx.archive_root)
            if not (r.get('replaced_by') or '').strip() and not named(q, text):
                c = units.setdefault(r['unit'], {})
                c['retire'] = c.get('retire', 0) + 1
                retired.append(q)
    return units, mans, retired


def render(cls, fields):
    """The section text: each template paragraph filled and wrapped at README_WIDTH (never inside a word or a path); empty
    paragraphs dropped."""
    paras = []
    for t in _p().README_TEMPLATES[cls]:
        s = t.format(**fields).strip()
        if s:
            paras.append(textwrap.fill(s, _p().README_WIDTH, break_long_words=False, break_on_hyphens=False))
    return '\n'.join(paras) + '\n'


def push_readme(ctx, args, run, unit_fn, cls, adir, qdir, qman_name, units, rnd, staging):
    """The README unit of a push (module docstring). units: the unit objects this push's unit_fn returned, in order."""
    units = [u for u in units if u is not None]
    if cls not in (_p().README_TEMPLATES or {}) or not units:      # no template, or a push of no unit: no README unit
        return None
    from exopipe.phase3.lib.archive.hybrid import apply_unit
    name = f'push_{cls}_readme'

    def body(u):
        inst = os.path.join(adir, _p().README_NAME)
        u.meta.update(push_class=cls, round=rnd, archive_dir=adir, staging_root=staging)
        if not os.path.isfile(inst):
            raise Refusal(f'{inst}: the archive run has no {_p().README_NAME} to append to (a new run\'s README is written by a ruling)')
        u.add_input(inst)
        with open(inst, 'rb') as fh:
            old = fh.read()
        w = writes(ctx, args)
        note = _note(args)
        ids = K.ruling_ids([note]) if note else []
        if w and not ids:                                          # precheck refused the push already; never write without a note
            raise Refusal(f'{name}: --update-note "<why, ledger ID>" with a ledger ID of {K.cfg().paths.FIX_LEDGER} is needed')
        counts, mans, retired = pending(ctx, adir, os.path.join(qdir, qman_name), old.decode('utf-8', 'replace'))
        by = {x.name: x for x in units}
        job = {}
        if not w:                                                  # a plan: this job's planned changes over the archive's pending ones
            for x in units:
                if x.status in DONE:
                    f = job[x.name] = facts(run, x)
                    if _changes(f['counts']):
                        counts[x.name] = f['counts']
        counts = {n: c for n, c in counts.items() if _changes(c)}
        if not counts:
            raise NoInput(f'no unit of this push changed a file that {inst} does not describe yet: the README stays as installed')
        if not note:
            u.note('no --update-note: the apply refuses without one')
        names = sorted(counts)
        for n in names:
            if n in by and n not in job and by[n].status in DONE:
                job[n] = facts(run, by[n])
        cache = {}
        built = sorted(set().union(*[_commits(job[n]['inputs'], cache) for n in names if n in job]))
        earlier = [n for n in names if n not in job]
        ruled = [f'{label(cls, n)}: {r}' for n in names if n in job for r in job[n]['ruled']]
        bad = [x.name for x in units if x.status in ('refused', 'error')]
        tot = {k: sum(int(c.get(k) or 0) for c in counts.values()) for k in COUNTED}
        fields = dict(
            round=rnd, note=note or NO_NOTE, cls=cls, push_commit=run.commit or 'unknown',
            built_commits=(', '.join(built) or 'commits not recorded in the step records') + (
                f'; {len(earlier)} unit(s) applied by an earlier job, see its manifest' if earlier else ''),
            work_run=staging or 'not named', n_units=len(names), changed=summary(cls, names),
            quarantine=os.path.relpath(qdir, ctx.archive_root), quarantine_manifest=qman_name,
            push_manifest=', '.join(mans) or ('none (retired files only)' if w else 'the push manifest(s) of the apply'),
            by_ruling=' '.join(x for x in (('By ruling: ' + '; '.join(ruled) + '.') if ruled else '',
                                           ('Retired without a successor (now in the quarantine): ' + ', '.join(retired) + '.')
                                           if retired else '') if x),
            refused=(f'Not pushed in this job ({len(bad)} unit(s), each record holds the reason): '
                     + ', '.join(label(cls, n) for n in bad[:24]) + (' ...' if len(bad) > 24 else '')) if bad else '',
            **tot)
        sec = render(cls, fields).encode('utf-8')
        sep = b'' if old.endswith(b'\n\n') or not old else (b'\n' if old.endswith(b'\n') else b'\n\n')
        new = old + sep + sec
        u.meta.update(units=names, earlier=earlier, manifests=mans, retired=retired, ledger_ids=ids, prefix_md5=_md5(old),
                      prefix_bytes=len(old))
        u.note(f'section of {len(names)} unit(s): {fields["changed"][:300]}')
        row = dict(unit=name, kind='readme', src='', dst=inst, bytes=len(new), md5=_md5(new), action='replace', replaces=inst,
                   qdest=os.path.join(qdir, 'README', ctx.stamp, _p().README_NAME))
        if ctx.dry:
            return [row]
        staged = os.path.join(ctx.manifest_dir(f'push_{cls}'), f'README_{rnd}_{ctx.stamp}.txt')
        os.makedirs(os.path.dirname(staged), exist_ok=True)
        with open(staged + '.tmp', 'wb') as fh:
            fh.write(new)
        os.replace(staged + '.tmp', staged)
        row['src'] = staged
        u.add_output(staged, 'archive_readme', prefix_md5=_md5(old), prefix_bytes=len(old), round=rnd)
        if not w:
            man = os.path.join(ctx.manifest_dir(f'push_{cls}'), f'readme_{ctx.stamp}.csv')
            K.write_csv(man, _p().HYBRID_PUSH_COLUMNS, [row])
            u.add_output(man, 'csv_table', required_columns=('unit', 'kind', 'src', 'dst', 'action', 'replaces'))
            return [row]
        apply_unit(ctx, u, [row], rnd, cls=cls, adir=adir, qdir=qdir, qman_name=qman_name, copy_class='archive_readme_copy',
                   push_manifest=False)
        return [row]
    return unit_fn(name, body, describe=lambda rows: f'README section appended ({rows[0]["bytes"]} bytes)' if rows else '0 files')
