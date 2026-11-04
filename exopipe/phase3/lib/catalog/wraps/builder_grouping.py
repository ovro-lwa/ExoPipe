"""The production catalogue builder (exopipe_ops build_source_catalogs_v3.py) IN PLACE, with its GROUPING and its MULTIPLICITY
decisions taken from the catalog step's grouping (refactor item F.3; lib/catalog/grouping_stage.py). Design B, ruled 2026-10-26
(R-GREGG-20261026-GROUPING-DESIGN-B-APPROVED-IDENTITY-097-MULTIPLES-085-DEEPV-IDENTITY-ONLY-PREVIEW-THEN-REFACTOR), and ONE
multiples rule on every tier, 2026-10-27 (R-GREGG-20261027-HUB-BOTH-LISTENERS-MULTIPLES-CHAIN-FORCED-EVERY-TIER).

    python builder_grouping.py <builder .py> <selected_K.csv> <out dir> <images csv>      (env EXOPIPE3_GROUPING_*)

SELF-CONTAINED (stdlib only): the child's `exopipe` package can be production's (the builder puts /lustre/gh/main/exopipe first on
sys.path), so this file imports nothing from the branch. It reads the builder's source, applies the PATCHES below - each one
exact text that must occur EXACTLY ONCE (else REFUSING: production changed the builder; re-port) and STRUCTURE (the census of
the names the patches replace: a new use of one would bypass the grouping) - and runs the patched source as the builder's own
__main__ (its file name, its directory first on sys.path, its argv), so every other line of production's builder runs
unchanged: positions, errors, cross-matches, E statistics, promotion rungs, repeat significance, PB fluxes, ranks, review
directories, unique names, the files and their columns.

WHAT THE PATCHES CHANGE (EXOPIPE3_GROUPING_MODE=design-b; the hook object __GROUPING__ = Hook below):
  P1 association     the builder's per-band union-find (assoc_groups) -> the (identity, band) groups of the grouping, in the
                     builder's order (groups by first member, members in input order)
  P2 sigma-stack     the singleton gate's exemption "two or more (LST, date) instants" -> the member of a MULTIPLE (its
                     multiples-level group holds MULTIPLE_MIN windows / nights): the one rule
  P3, P4 names       source_id (and the members file's) -> the identity's display name (unique per Stokes: one page per
                     identity; a 10-minute identity's rows of several LSTs share it, and the cumulative joins them by name)
  P5 multi-integration -> the one rule's flag (multi_integration = the multiple flag; the instant / cell / span attributes of
                     the whole multiples group, production's _multi_integration on its detections) + SOURCE_COLS: the stable
                     identity ID, the multiples-level group's ID, name, multiplicity, identities (the multiples link)
  P6-P8 repeat rungs RPT, REP, XREP promote only a member of a multiple (EXOPIPE3_GROUPING_RUNGS 'multiples-level'; 'production':
                     unchanged) - the multiples level decides repeats
  P9 identity merge  the cross-band union-find -> the grouping's identity (identity_id = its display name, identity_bands = its
                     bands in this build)
  P10 7-8 sigma gate for unpromoted singles: a member of a multiple skips it
  P11 the 7-sigma floor: a member of a multiple survives it
  P12 columns        COLS += SOURCE_COLS (every catalogue file the builder writes)
EXOPIPE3_GROUPING_MODE=passthrough: every hook returns production's own value (the original call / condition), so the patched
builder must write byte-identical files - the check that the patching and the __main__ harness change nothing else.
A detection of the builder's input that the grouping table does not hold, or holds with another fingerprint (band, LST,
date, catalogue ra / dec / S/N), REFUSES (a stale grouping: group again). Machine line: 'GROUPING-BUILDER ...' (stdout).
PINS (review 2026-11-04): the builder's PATH-NORMALIZED sha1 (a frozen tree's own root replaced by /lustre/gh/main, the
frozen-code rewrite undone) must be one of EXOPIPE3_GROUPING_BUILDER_SHA1 (config catalog_grouping.BUILDER_SHA1: the builders
the patches were written and checked for) - any other builder REFUSES until the patches are re-ported (F7: a new multiplicity-
dependent decision elsewhere in the file would otherwise run unpatched); with EXOPIPE3_GROUPING_TABLE_SHA1 set, the grouping
table read must have that sha1 (the stage checked the table's record; a regroup in between REFUSES - F2). The table is read
once: its sha1 and its rows come from the same bytes. With EXOPIPE3_GROUPING_PROVENANCE=1 the runner writes
grouping_table.json into the builder's output directory (the table's path and sha1, the builder's sha1) - what the
cumulative build checks (one grouping table for every LST build).
"""
import collections
import csv
import hashlib
import io
import json
import os
import sys
import types

csv.field_size_limit(1 << 30)

SOURCE_COLS = ['group_identity_id', 'multiples_id', 'multiples_name', 'multiple', 'multiples_multiplicity',
               'multiples_n_identities', 'multiples_identities']
MODES = ('design-b', 'passthrough')
PROVENANCE = 'grouping_table.json'          # written into the builder's output directory (design B, EXOPIPE3_GROUPING_PROVENANCE=1)
RUNGS = ('multiples-level', 'production')
TIERS = ('10min', 'deep')

# (name, old text, new text): each old text must occur EXACTLY ONCE in the builder
PATCHES = [
    ('P1-association',
     "    groups = assoc_groups(c_all, [_stat_as(r) for r in br],\n"
     "                          [_sys_det(r) for r in br], K=_KP(band))\n",
     "    groups = __GROUPING__.groups(band, br)   # EXOPIPE3 GROUPING P1: the catalog step's identities\n"),
    ('P2-sigma-stack',
     "        if SSTACK_GATE and len({(m['lst'], m['date']) for m in mem}) == 1:\n",
     "        if SSTACK_GATE and not __GROUPING__.multiple(mem):   # EXOPIPE3 GROUPING P2: the one multiples rule\n"),
    ('P3-members-name',
     "        members.append((jname(ra, dec), band,\n",
     "        members.append((__GROUPING__.name(mem, jname(ra, dec)), band,   # EXOPIPE3 GROUPING P3\n"),
    ('P4-source-name',
     "            source_id=jname(ra, dec), identity_id='', identity_bands='',\n",
     "            source_id=__GROUPING__.name(mem, jname(ra, dec)), identity_id='', identity_bands='',   # EXOPIPE3 GROUPING P4\n"),
    ('P5-multi-integration',
     "            **_multi_integration(mem, cells, dec, sig_stat),\n",
     "            **__GROUPING__.multi_integration(mem, cells, dec, sig_stat, _multi_integration),   # EXOPIPE3 GROUPING P5\n"),
    ('P6-RPT',
     "                    and float(s['_rpt_pct']) < RPT_PCT:\n",
     "                    and float(s['_rpt_pct']) < RPT_PCT and __GROUPING__.repeat_rungs(s):   # EXOPIPE3 GROUPING P6\n"),
    ('P7-REP',
     "            if s['n_det'] >= 2 and s.get('e_rep') not in ('', None) and float(s['e_rep']) < 0.5: lab.append('REP')\n",
     "            if s['n_det'] >= 2 and s.get('e_rep') not in ('', None) and float(s['e_rep']) < 0.5 and __GROUPING__.repeat_rungs(s): lab.append('REP')   # EXOPIPE3 GROUPING P7\n"),
    ('P8-XREP',
     "                    and (s['n_det'] >= 3 or a >= THR): lab.append('XREP')\n",
     "                    and (s['n_det'] >= 3 or a >= THR) and __GROUPING__.repeat_rungs(s): lab.append('XREP')   # EXOPIPE3 GROUPING P8\n"),
    ('P9-identity-merge',
     "if pool:\n"
     "    _xf = _PERTIE is not None\n",
     "if pool and __GROUPING__.active:   # EXOPIPE3 GROUPING P9: the identities are the catalog step's\n"
     "    print(f'identity merge: {__GROUPING__.identities(pool)} cross-band identities (the catalog step\\'s grouping)', flush=True)\n"
     "elif pool:\n"
     "    _xf = _PERTIE is not None\n"),
    ('P10-gate-7-8',
     "        if s['n_det'] == 1 and not s['promotion'] and not s.get('xm_best') and 7.0 <= a_ < 8.0:\n",
     "        if s['n_det'] == 1 and not s['promotion'] and not s.get('xm_best') and 7.0 <= a_ < 8.0 and not __GROUPING__.multiple_src(s):   # EXOPIPE3 GROUPING P10\n"),
    ('P11-floor',
     "                 if abs(s['best_snr']) >= THR or s['promotion']),\n",
     "                 if abs(s['best_snr']) >= THR or s['promotion'] or __GROUPING__.multiple_src(s)),   # EXOPIPE3 GROUPING P11\n"),
    ('P12-columns',
     "\nos.makedirs(outdir, exist_ok=True)\nrows = list(csv.DictReader(open(gated_csv)))\n",
     "\nCOLS += __GROUPING__.extra_cols()   # EXOPIPE3 GROUPING P12\nos.makedirs(outdir, exist_ok=True)\nrows = list(csv.DictReader(open(gated_csv)))\n"),
]
# STRUCTURE: how often each name the patches stand in for occurs in the builders the patches were written for (exopipe_ops
# 0d7861e, sha1 44d2763..., and the 95 frozen 10-minute trees under phase3/sublst/code_frozen, 2026-10-11..11-02: every patch
# text once in each) - a new use would bypass the grouping, so a count outside the accepted set REFUSES. SSTACK_GATE and
# identity_id: 4 before the 10-minute per-detection record (exopipe_ops 6c58469, 2026-10-18: record only), 7 since
CENSUS = {'assoc_groups(': {2}, 'SSTACK_GATE': {4, 7}, '_multi_integration(': {2}, "lab.append('RPT')": {1},
          "lab.append('REP')": {1}, "lab.append('XREP')": {1}, 'SIZE_GATE_7_8': {6}, '>= THR or s[': {1}, '\nCOLS = ': {1},
          'source_id=jname(': {1}, 'members.append((jname(': {1}, 'identity_id': {4, 7}}


LIVE_ROOT = '/lustre/gh/main'               # the live code root a frozen tree's rewrites point away from


class Refused(RuntimeError):
    pass


def normalized_sha1(builder, src):
    """sha1 of the builder's source with its own code root (a frozen tree's <root>/, as its rewrites put it) replaced by the
    live root - the same builder version gives the same value in every frozen tree"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(builder)))
    if root != LIVE_ROOT:
        src = src.replace(root + '/', LIVE_ROOT + '/')
    return hashlib.sha1(src.encode()).hexdigest()


def patch(src):
    """(patched source, [names]) - REFUSES unless every patch's text occurs exactly once and the census holds."""
    bad = []
    for name, old, _new in PATCHES:
        n = src.count(old)
        if n != 1:
            bad.append(f'{name}: its text occurs {n} times (1 expected)')
    for tok, want in CENSUS.items():
        n = src.count(tok)
        if n not in want:
            bad.append(f'census {tok!r}: {n} occurrences ({sorted(want)} expected)')
    if bad:
        raise Refused('the builder is not the one the grouping patches were written for - ' + '; '.join(bad)
                      + ' - re-port lib/catalog/wraps/builder_grouping.py - REFUSING')
    out = src
    for name, old, new in PATCHES:
        out = out.replace(old, new)
    return out, [p[0] for p in PATCHES]


def _f(x):
    return float(x)


class Hook:
    """__GROUPING__ in the builder's namespace. mode 'passthrough': production's own value at every hook; 'design-b': the
    grouping's (table: {cand_id: detections.csv row of grouping_stage}). ns: the builder's module namespace (set before it runs)."""

    def __init__(self, mode, table=None, tier=None, rungs='multiples-level', table_sha1=None):
        if mode not in MODES:
            raise Refused(f'EXOPIPE3_GROUPING_MODE={mode!r} is not one of {MODES} - REFUSING')
        if rungs not in RUNGS:
            raise Refused(f'EXOPIPE3_GROUPING_RUNGS={rungs!r} is not one of {RUNGS} - REFUSING')
        self.mode, self.rungs, self.tier = mode, rungs, tier
        self.active = mode == 'design-b'
        self.ns = None
        self.T, self.mgroup, self.checked = {}, {}, set()
        if self.active:
            if tier not in TIERS:
                raise Refused(f'EXOPIPE3_GROUPING_TIER={tier!r} is not one of {TIERS} - REFUSING')
            if not table or not os.path.exists(table):
                raise Refused(f'EXOPIPE3_GROUPING_TABLE={table!r}: no grouping table - run the grouping stage first - REFUSING')
            with open(table, 'rb') as f:
                data = f.read()                  # ONE read: the sha1 and the rows are the same bytes
            self.table_sha1 = hashlib.sha1(data).hexdigest()
            if table_sha1 and self.table_sha1 != table_sha1:
                raise Refused(f'{table}: sha1 {self.table_sha1[:12]}, the stage checked {table_sha1[:12]} - the grouping changed '
                              f'after the check (a regroup in between): build again - REFUSING')
            mg = collections.defaultdict(list)
            for r in csv.DictReader(io.StringIO(data.decode('utf-8'), newline='')):
                if r['cand_id'] in self.T:
                    raise Refused(f'{table}: cand_id {r["cand_id"]} twice - REFUSING')
                self.T[r['cand_id']] = r
                mg[r['multiples_id']].append(r)
            self.mgroup = dict(mg)
        else:
            self.table_sha1 = ''

    # ---- the table's entry of a builder row (refuses a stale grouping) ---------------------------------------------------
    def entry(self, r):
        c = r['cand_id']
        e = self.T.get(c)
        if e is None:
            raise Refused(f'{c} ({r.get("band")} {r.get("lst")} {r.get("date")}) is not in the grouping table - the grouping is '
                          f'stale: group again - REFUSING')
        if (e['band'], e['lst'], e['date']) != (r['band'], r['lst'], r['date']) or any(
                _f(e[k]) != _f(r[k]) for k in ('ra', 'dec', 'snr')):
            raise Refused(f'{c}: the grouping holds it as ({e["band"]} {e["lst"]} {e["date"]} {e["ra"]} {e["dec"]} {e["snr"]}), '
                          f'the builder input as ({r["band"]} {r["lst"]} {r["date"]} {r["ra"]} {r["dec"]} {r["snr"]}) - a '
                          f'stale grouping: group again - REFUSING')
        self.checked.add(c)
        return e

    def _one(self, mem, field):
        vals = {self.entry(m)[field] for m in mem}
        if len(vals) != 1:
            raise Refused(f'one builder source spans {len(vals)} values of {field} ({sorted(vals)[:3]}) - REFUSING')
        return vals.pop()

    def _src_entry(self, s):
        return self.T[s['best_cand_id']]

    # ---- the hooks ----------------------------------------------------------------------------------------------------
    def groups(self, band, br):
        if not self.active:
            ns = self.ns
            return ns['assoc_groups'](ns['c_all'], [ns['_stat_as'](r) for r in br], [ns['_sys_det'](r) for r in br],
                                      K=ns['_KP'](band))
        order, out = {}, []
        for i, r in enumerate(br):
            k = self.entry(r)['identity_id']
            if k not in order:
                order[k] = len(out)
                out.append([])
            out[order[k]].append(i)
        return out

    def multiple(self, mem):
        if not self.active:
            return len({(m['lst'], m['date']) for m in mem}) != 1
        return self._one(mem, 'multiple') == '1'

    def multiple_src(self, s):
        if not self.active:
            return False
        return self._src_entry(s)['multiple'] == '1'

    def name(self, mem, default):
        if not self.active:
            return default
        return self._one(mem, 'identity_name')

    def multi_integration(self, mem, cells, dec, sig_stat, fn):
        if not self.active:
            return fn(mem, cells, dec, sig_stat)
        m = self._one(mem, 'multiples_id')
        grp = self.mgroup[m]
        e = self.entry(mem[0])
        flag = e['multiple'] == '1'
        dets = [{'lst': g['lst'], 'date': g['date']} for g in grp]
        r = dict(fn(dets, sorted({g['lst'] for g in grp}), dec, sig_stat))
        r['multi_integration'] = int(flag)
        r['mi_tiers'] = self.tier if flag else ''
        ids = sorted({g['identity_name'] for g in grp})
        r.update(group_identity_id=e['identity_id'], multiples_id=m, multiples_name=e['multiples_name'], multiple=int(flag),
                 multiples_multiplicity=int(e['multiplicity']), multiples_n_identities=len(ids),
                 multiples_identities=';'.join(ids))
        return r

    def repeat_rungs(self, s):
        if not self.active or self.rungs == 'production':
            return True
        return self.multiple_src(s)

    def identities(self, pool):
        """identity_id / identity_bands of every source of this build from the grouping; returns the number of identities
        holding two or more bands here (the builder's printed count)."""
        bands = collections.defaultdict(set)
        for s in pool:
            bands[s['source_id']].add(s['band'])
        for s in pool:
            if s['source_id'] != self._src_entry(s)['identity_name']:
                raise Refused(f'{s["source_id"]} {s["band"]}: the source is not named by its identity - REFUSING')
            s['identity_id'] = s['source_id']
            s['identity_bands'] = '+'.join(sorted(bands[s['source_id']]))
        return sum(1 for v in bands.values() if len(v) > 1)

    def extra_cols(self):
        return list(SOURCE_COLS) if self.active else []


def run(builder, args, mode, table=None, tier=None, rungs='multiples-level', accepted=(), table_sha1=None, provenance=False):
    """Run the patched builder as __main__ (its file name, its directory first on sys.path, sys.argv = [builder] + args).
    accepted: the path-normalized sha1s the patches were written for (any other builder REFUSES)."""
    builder = os.path.abspath(builder)
    with open(builder) as f:
        src = f.read()
    sha1 = hashlib.sha1(src.encode()).hexdigest()
    norm = normalized_sha1(builder, src)
    accepted = [a for a in accepted if a]
    if not accepted:
        raise Refused('EXOPIPE3_GROUPING_BUILDER_SHA1 is empty: the accepted builders must be named (config '
                      'catalog_grouping.BUILDER_SHA1) - REFUSING')
    if norm not in accepted:
        raise Refused(f'the builder {builder} (path-normalized sha1 {norm}) is not one the grouping patches were written and '
                      f'checked for ({len(accepted)} accepted, config catalog_grouping.BUILDER_SHA1) - re-port '
                      f'lib/catalog/wraps/builder_grouping.py - REFUSING')
    patched, names = patch(src)
    hook = Hook(mode, table, tier, rungs, table_sha1)
    mod = types.ModuleType('__main__')
    mod.__file__ = builder
    mod.__dict__['__GROUPING__'] = hook
    mod.__dict__['__builtins__'] = __builtins__
    hook.ns = mod.__dict__
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [os.path.dirname(builder)] + [p for p in sys.path if os.path.abspath(p or '.') != here]
    sys.argv = [builder] + list(args)
    sys.modules['__main__'] = mod
    print(f'GROUPING-BUILDER start: builder {builder} sha1 {sha1}; patches {len(names)} ({",".join(names)}); mode {mode}; '
          f'tier {tier}; repeat rungs {rungs}; table {table or "-"} sha1 {hook.table_sha1 or "-"} ({len(hook.T)} detections)',
          flush=True)
    code = compile(patched, builder, 'exec')
    exec(code, mod.__dict__)
    if provenance and hook.active and len(args) >= 2:
        out_dir = args[1] if os.path.isabs(args[1]) else os.path.join(os.getcwd(), args[1])
        p = os.path.join(out_dir, PROVENANCE)
        with open(p + '.tmp', 'w') as f:
            json.dump(dict(table=os.path.abspath(table), table_sha1=hook.table_sha1, tier=tier, repeat_rungs=rungs,
                           builder=builder, builder_sha1=sha1, builder_normalized_sha1=norm, rows_checked=len(hook.checked)),
                      f, indent=1, sort_keys=True)
        os.replace(p + '.tmp', p)
    print(f'GROUPING-BUILDER done: builder sha1 {sha1}; mode {mode}; {len(hook.checked)} input rows checked against the grouping',
          flush=True)


def main(argv):
    if len(argv) < 5:
        sys.exit('usage: builder_grouping.py <builder .py> <selected_K.csv> <out dir> <images csv>  (env EXOPIPE3_GROUPING_*)')
    try:
        run(argv[1], argv[2:], os.environ.get('EXOPIPE3_GROUPING_MODE', ''), os.environ.get('EXOPIPE3_GROUPING_TABLE'),
            os.environ.get('EXOPIPE3_GROUPING_TIER'), os.environ.get('EXOPIPE3_GROUPING_RUNGS', 'multiples-level'),
            accepted=os.environ.get('EXOPIPE3_GROUPING_BUILDER_SHA1', '').split(','),
            table_sha1=os.environ.get('EXOPIPE3_GROUPING_TABLE_SHA1') or None,
            provenance=os.environ.get('EXOPIPE3_GROUPING_PROVENANCE') == '1')
    except Refused as e:
        print(f'GROUPING-BUILDER REFUSING: {e}', flush=True)
        sys.exit(2)


if __name__ == '__main__':
    main(sys.argv)
