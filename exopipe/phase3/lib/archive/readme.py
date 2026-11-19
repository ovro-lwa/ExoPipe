"""README notes of an archive run (spec hybrid.md 3.7, PR-8c; 2.16 item 5: no inline code writes into the archive).

The last unit of a push of a class with archive_push.README_TEMPLATES[class], push_<class>_readme, appends one section to its archive
run's archive_push.README_NAME and pushes it like any product: the installed README copied byte for byte, then the section (the
template filled from the records of this push's units); the staged file content-tested (class archive_readme); the D2 swap of
lib/archive/hybrid.apply_unit with the installed README quarantined into the push's round as README/<stamp>/README.txt (its own name; several
pushes may share a round); the written file content-tested (class archive_readme_copy: its checksum, and it begins with every byte of the
README it replaced - no history is ever lost).

The units a section describes: the units this job planned or applied that change a file (publish, replace or retire), and the
units an earlier job applied (resumed, skipped-valid) that the installed README does not describe yet (none of their push manifests
named in it; a retire-only unit, which writes no push manifest: its unit name not in it). A rerun after a complete push therefore
adds no second section (no-input), and a push stopped before its README unit is described by the rerun.

--update-note (why the update was made) must cite a row of paths.FIX_LEDGER when the README is written (an apply with the transfer
stage); a plan or dry run without it renders a placeholder and notes that the apply would refuse.
"""
import hashlib
import json
import os
import re
import textwrap

from exopipe.phase3.core.record import NoInput, Refusal
from exopipe.phase3.lib.archive import common as K

DONE = ('done', 'skipped-valid', 'planned')
CHANGES = ('publish', 'replace', 'retire', 'keep-or-replace')     # keep-or-replace: a dry run (no checksums) cannot tell yet
_HYB = re.compile(r'^(\d\dh)_(.+)_(R\w+)$')
NO_NOTE = '<the update note: --update-note "<why, ledger ID>" - an apply refuses without it>'


def _p():
    return K.cfg().archive_push


def _md5(b):
    return hashlib.md5(b).hexdigest()


def facts(run, u):
    """What README needs of one push unit u (the unit object unit_fn returned): this job's own view, or for a resumed unit
    (skipped-valid) its earlier record."""
    if u.status == 'skipped-valid':
        from exopipe.phase3.core.record import load_record
        rec = load_record(run.out_root, run.step, u.name) or {}
        meta, notes, inputs, outputs = rec.get('meta') or {}, rec.get('notes') or [], rec.get('inputs') or {}, rec.get('outputs') or []
    else:
        meta, notes, inputs, outputs = u.meta, u.notes, u.inputs, u.outputs
    return dict(name=u.name, status=u.status, reason=u.reason, counts=dict(meta.get('counts') or {}),
                ruled=[n for n in notes if 'by ruling' in str(n)], inputs=list(inputs),
                manifests=sorted({o['path'] for o in outputs if o.get('class') == 'archive_manifest'}))


def _changes(f):
    return sum(int((f['counts'] or {}).get(k) or 0) for k in CHANGES)


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


def summary(cls, names):
    """The changed units in words: hybrid tiles grouped by scope and weighting with their LSTs; other units by name."""
    pre = f'push_{cls}_'
    groups, other = {}, []
    for n in names:
        s = n[len(pre):] if n.startswith(pre) else n
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


def selected(units, installed_text):
    """(described, refused) facts: the units of this push the section describes (module docstring) and the units that did not
    finish (refused, error, no-input stays out: it changed nothing)."""
    desc, bad = [], []
    for f in units:
        if f['status'] in ('refused', 'error'):
            bad.append(f)
            continue
        if f['status'] not in DONE or not _changes(f):
            continue
        if f['status'] == 'skipped-valid':
            named = (any(os.path.basename(m) in installed_text for m in f['manifests']) if f['manifests']
                     else f['name'] in installed_text)
            if named:
                continue
        desc.append(f)
    return desc, bad


def render(cls, fields):
    """The section text: each template paragraph filled and wrapped at README_WIDTH; empty paragraphs dropped."""
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
    all_facts = [facts(run, u) for u in units]

    def body(u):
        inst = os.path.join(adir, _p().README_NAME)
        u.meta.update(push_class=cls, round=rnd, archive_dir=adir, staging_root=staging)
        if not os.path.isfile(inst):
            raise Refusal(f'{inst}: the archive run has no {_p().README_NAME} to append to (a new run\'s README is written by a ruling)')
        u.add_input(inst)
        with open(inst, 'rb') as fh:
            old = fh.read()
        desc, bad = selected(all_facts, old.decode('utf-8', 'replace'))
        if not desc:
            raise NoInput(f'no unit of this push changed a file that {inst} does not describe yet: the README stays as installed')
        writes = ctx.apply and 'transfer' in args._stages and not ctx.dry
        note = ' '.join(getattr(args, 'update_note', None) or []).strip()
        ids = K.ruling_ids([note]) if note else []
        if writes and not ids:
            raise Refusal(f'{name}: --update-note "<why, ledger ID>" with a ledger ID of {K.cfg().paths.FIX_LEDGER} is needed to '
                          f'write the README ({"none given" if not note else "no ledger ID found in " + repr(note[:120])})')
        if not note:
            u.note('no --update-note: the apply refuses without one')
        cache = {}
        ruled = [f"{f['name']}: {r}" for f in desc for r in f['ruled']]
        built = sorted(set().union(*[_commits(f['inputs'], cache) for f in desc]))
        tot = {k: sum(int((f['counts'] or {}).get(k) or 0) for f in desc) for k in ('replace', 'publish', 'retire', 'keep')}
        mans = sorted({os.path.relpath(m, ctx.archive_root) for f in desc for m in f['manifests']})
        fields = dict(
            round=rnd, note=note or NO_NOTE, cls=cls, push_commit=run.commit or 'unknown',
            built_commits=', '.join(built) or 'commits not recorded in the step records', work_run=staging or 'not named',
            n_units=len(desc), changed=summary(cls, [f['name'] for f in desc]), quarantine=os.path.relpath(qdir, ctx.archive_root),
            quarantine_manifest=qman_name, push_manifest=', '.join(mans) or (f'MANIFEST_{ctx.stamp}.csv' if writes else
                                                                           'the push manifests of the apply'),
            by_ruling=('By ruling: ' + '; '.join(ruled)) if ruled else '',
            refused=(f'Not pushed in this job ({len(bad)} unit(s), each record holds the reason): '
                     + ', '.join(f['name'] for f in bad[:24]) + (' ...' if len(bad) > 24 else '')) if bad else '',
            **tot)
        sec = render(cls, fields).encode('utf-8')
        sep = b'' if old.endswith(b'\n\n') or not old else (b'\n' if old.endswith(b'\n') else b'\n\n')
        new = old + sep + sec
        u.meta.update(units=[f['name'] for f in desc], ledger_ids=ids, prefix_md5=_md5(old), prefix_bytes=len(old))
        u.note(f'section of {len(desc)} unit(s): {fields["changed"][:300]}')
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
        if not writes:
            man = os.path.join(ctx.manifest_dir(f'push_{cls}'), f'readme_{ctx.stamp}.csv')
            K.write_csv(man, _p().HYBRID_PUSH_COLUMNS, [row])
            u.add_output(man, 'csv_table', required_columns=('unit', 'kind', 'src', 'dst', 'action', 'replaces'))
            return [row]
        apply_unit(ctx, u, [row], rnd, cls=cls, adir=adir, qdir=qdir, qman_name=qman_name, copy_class='archive_readme_copy')
        return [row]
    return unit_fn(name, body, describe=lambda rows: f'README section appended ({rows[0]["bytes"]} bytes)' if rows else '0 files')
