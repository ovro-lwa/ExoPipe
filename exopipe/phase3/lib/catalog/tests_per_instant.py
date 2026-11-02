"""The per-instant veto rule as a content test, every tier (FOLD_IN 2026-10-18 05:0x; plan/CUTOVER_READINESS.md "Gregg's
per-instant veto rule ... every tier's removal steps are audited against it").

Gregg 2026-10-18, verbatim (R-GREGG-20261018-VETOES-PER-FRAME-INDEPENDENTLY): "All frames should pass through vetoes
independently. Otherwise, for example, a single VHF reflection could take out a 200x detected transient."; refined
(R-GREGG-20261018-VETOES-PER-FRAME-REFINED-FREQUENCY-YES-TIME-NO): "Vetoing across frequency for a single frame is probably
okay, but not across time". A veto measured at one INSTANT removes that instant's detections in any band, subband, wideband
and Stokes - never another instant's, never another tier's. The instant of a detection, per tier:
    10-s        (cell, '10s', integration label)      the snapshot record's instant_key (clv15 phase3_snapshot_chain.instant_of)
    1-minute    (cell, '1min', integration minute)     the same record
    10-minute   (LST, night, window)                  the row's date '<night>T<kk>' (cadence 10min)
    deep        (LST, night)                          the row's lst and date (cadence deep)
The content test (FOLD_IN: "no removal reason names another instant") is registered as 'per_instant' on the class that holds
each tier's removal record, and refuses the product on any finding:

  conformant_catalog (deep and 10-minute: master_catalog.csv with master_catalog_removed.csv and master_catalog_asrun.csv
  beside, as ops/p3s2_conform.py and the B-02 check write them), fork_catalog (the deep multiples fork: EVERY conformance
  record of fork_<S>/ - master_catalog_removed.csv and each run's master_catalog_removed_<tag>.csv, also in its
  subdirectories - and its row-line record rowline_removed.csv) and rejected_record (rejected.csv: the record the
  10-minute pages read, emit_rejected_record.py - its reason is the removal_reason cut to REJECTED_REASON_CHARS):
    T1 every removed row is of the record's tier (cadence deep / 10min, the date's form) and carries its own instant;
    T2 every removal reason matches, WHOLE, one declared template (REASON_KINDS: the f-strings of the writers themselves,
       lib/search/removal_rules.py and lib/search/parent_curve_table.py, read with ast - nothing hand-copied but the
       patterns of their fields), each with the instant it names and the tiers it is declared for; an undeclared reason
       refuses (a new removal step declares its instant first), and so does the conformance's night-keyed noise-sanity
       removal in the 10-minute tier (inert there today; keyed by night it would remove every window of the night -
       FOLD_IN 2026-10-18: per window only). A reason cut at REJECTED_REASON_CHARS matches a template's prefix;
    T3 a Jupiter-track removal names a co-added night (jupiter_track_night, or the reason's words): another instant's
       Jupiter. It stands only for the MASTER's imprint - the master co-adds that night's Jupiter, so this night's own
       residual carries minus Jupiter there (ops/p3s2_conform.py, the 2026-10-18 deep audit: "measured in the master, not
       on another night's detection"). The verdict reads the product's OWN recorded mask state, never the configuration:
       the 10-minute row's searched frame (<window tree>/<LST>_<band>/frames.csv -> its header JUPMASK, as the conformance
       reads it; the arm from the record's own run name), the deep row's master (catalog.DEEP_ROOTS_CSV -> coadd_final/
       DEEP_MASTERS: a JUPMASK card or a '-jupmask' COADDOP). A file newer than the record is not the product's own and
       is not read. Masked -> the removal crosses time and refuses. With no readable record the TIER RULE (MASK_TIER_RULE)
       decides and the detail says so: deep masters unmasked (R-DEEP-MASTER-JUPITER-MASK-DEFERRED; Run_P3S5's masters
       carry COADDOP ivw-mean-v3 and no JUPMASK), 10-minute window masters masked (RULING-MOVING-JUPITER-MASK-IN-MASTERS);
    T4 a pulsar-group removal ('multi-detection source inconsistent with <PSR> ... (n=N)') is decided by ONE instant's group
       (a2d37b9: across-instant groups may only rescue). With the pulsar stage's group record beside (B-02 check:
       pulsar_groups.csv) exactly: the row is a member of an inconsistent 'per-epoch' group of that pulsar whose members are
       all of its instant. Without it (production's p3s2_conform), the necessary condition: N <= the detections of the row's
       own instant within catalog_conform.PULSAR_PRECUT_DEG of the pulsar in the as-run record (less the rows the daytime gate
       and the exclusions take, which never join a group) - a group of more members than its instant holds spans instants.
       Where the record cut the reason before '(n=N)' (rejected.csv), N >= 2 (a group has two members). The fork's records
       overwrite one as-run copy run by run: there the population is the union of the fork's as-run and every record (a
       superset: the bound only loosens).
  snapshot_run (10-s and 1-minute; the chain run directory):
    S1 vetoes applied per SOURCE refuse: a run whose vetoed_sources.csv holds negative-lobe / reflection-asymmetry removals
       without the per-instant record (chain_summary per_instant_vetoes, vetoed_detections.csv) - the rule before clv15
       99fe3ea, where one member's measurement removed every instant of its source;
    S2 every vetoed_detections.csv row's instant is its own cell and tier, and the member the veto measured (neg_member /
       asym_member) is of the same cell and tier and - by the per-detection record (detection_record.csv.gz) - of the same
       instant;
    S3 every detection the record says a veto removed (first_removed_by neg-lobe / asymmetry) carries that veto's 'remove'
       verdict for ITS OWN instant (gate_neg_lobe / gate_asymmetry), and its instant's measured member is at its instant;
       the record and vetoed_detections.csv agree on the number each veto removed. A run whose vetoes removed detections
       without the record refuses (nothing places them at their instants).
    The 3-D bright-source gate, the seed and floor rules and the routes are SOURCE-level selection rules (the source stands or
    falls on its remaining detections, the ruling's own words), not vetoes measured at an instant: not judged here (their
    across-instant reading is DECISIONS_PENDING "the snapshot bright-source gate's offset rule").
Context keys (all optional): tier ('deep' / 'sublst'); deep_roots_csv (default catalog.DEEP_ROOTS_CSV) and tenmin_tree (the
10-minute window tree; default from the record's run name) for T3.
Every test reads the files' content and returns (passed, value, detail); imports are inside the tests.
"""
import ast
import csv
import glob
import gzip
import json
import math
import os
import re

from exopipe.phase3.core.validate import content_test

csv.field_size_limit(10 ** 9)

TIER_CADENCE = {'deep': 'deep', 'sublst': '10min'}
ALL_TIERS = ('deep', 'sublst')
REJECTED_REASON_CHARS = 160      # ops/emit_rejected_record.py main(): veto=(removal_reason or 'removed')[:160] (drift-tested)
# field readers, applied only to a reason whose kind its whole template decided
PULSAR_GROUP_RE = re.compile(r'multi-detection source inconsistent with (?P<psr>.+?) - \d+ arcsec')
PULSAR_GROUP_N_RE = re.compile(r'outside the 99% radius \d+ arcsec \(n=(?P<n>\d+)\)')
JUPITER_NIGHT_RE = re.compile(r'co-added night (\d{4}-\d\d-\d\d)')
NEVER_GROUPED = ('daytime', 'twilight', 'excluded epoch')   # rows the pulsar step leaves out of every group (p3s2_conform)
# T3: the masters the deep conformance's track stands for (ops/p3s2_frames.master_path: I the Robust -0.75 master, V the
# Robust 0 master, under <co-add root>/<lst>_<band>/coadd_final; drift-tested) and the tier rule where no record is readable
DEEP_MASTERS = {'I': 'I_deep_Taper_R075_{band}.fits', 'V': 'V_deep_Taper_R0_{band}.fits'}
MASK_TIER_RULE = {'deep': ('unmasked', 'R-DEEP-MASTER-JUPITER-MASK-DEFERRED: the deep masters co-add every night unmasked '
                                       '(masters spec amendment 2026-10-18: the deep tier keeps the track until its masters '
                                       'are rebuilt with the moving mask)'),
                  'sublst': ('masked', 'RULING-MOVING-JUPITER-MASK-IN-MASTERS: the 10-minute window masters carry the moving '
                                       'Jupiter mask since 2026-10-05')}
# snapshot chain (clv15 phase3_snapshot_chain.py 99fe3ea; phase3_snapshot_detrecord.py): the two frame-reading vetoes, their
# names in vetoed_detections.csv / vetoed_sources.csv, in the record's first_removed_by, its verdict column and the member column
SNAP_VETOES = {'negative-lobe': ('neg-lobe', 'gate_neg_lobe', 'neg_member'),
               'reflection-asymmetry': ('asymmetry', 'gate_asymmetry', 'asym_member')}
SNAP_REEVAL = ('refl-floor', 'seed')       # survivors of a vetoed instant removed when the source is re-built (not vetoes)
SNAP_TIERS = ('10s', '1min')
RECORD_COLS = ('cell', 'subband', 'stokes', 'tier', 'ts', 'instant_key', 'gate_neg_lobe', 'gate_asymmetry', 'neg_member',
               'asym_member', 'first_removed_by')


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def _rows(path):
    with open(path, newline='') as f:
        rd = csv.DictReader(f)
        return list(rd.fieldnames or []), list(rd)


def _f(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _sep_deg(ra1, de1, ra2, de2):
    a, b, c, d = (math.radians(x) for x in (ra1, de1, ra2, de2))
    x = math.sin(b) * math.sin(d) + math.cos(b) * math.cos(d) * math.cos(a - c)
    return math.degrees(math.acos(max(-1.0, min(1.0, x))))


def _fail(msgs, value, what):
    if msgs:
        return False, len(msgs), '; '.join(msgs[:6]) + (f' (+{len(msgs) - 6} more)' if len(msgs) > 6 else '') + ' | ' + what
    return True, value, what


# ---------------------------------------------------------------------------------------------------------------------
# T2: the removal reasons' grammar, built from the writers' own f-strings (2026-11-02 review: the whole template, never its
# opening words). A template is a tree of ('lit', text) | ('pat', regex, prefix regex) | ('seq', [..]) | ('opt', x) |
# ('alt', [..]); _full compiles it, _prefix compiles 'any prefix of a full match' (a reason cut by the rejected record).
# ---------------------------------------------------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
WRITERS = {'removal_rules': os.path.normpath(os.path.join(_HERE, '..', 'search', 'removal_rules.py')),
           'parent_curve_table': os.path.normpath(os.path.join(_HERE, '..', 'search', 'parent_curve_table.py'))}


class TemplateError(Exception):
    """A writer's template cannot be read as declared (the writer changed): the content test refuses."""


def _pat(full, prefix, name=None):
    return ('pat', full, prefix, name)


def _named(pat, name):
    """A field pattern with the name the test renderer fills (tests/test_per_instant.py); compiling ignores it."""
    return ('pat', pat[1], pat[2], name)


def _lit(text):
    return ('lit', text)


def _seq(*xs):
    return ('seq', list(xs))


def _opt(x):
    return ('opt', x)


def _alt(*xs):
    return ('alt', list(xs))


NUMBER = _pat(r'-?\d+(?:\.\d+)?(?:e[+-]?\d+)?', r'[-\d.e+]*')
INT = _pat(r'\d+', r'\d*')
DATE = _pat(r'\d{4}-\d\d-\d\d', r'[\d-]{0,10}')
STOKES = _pat(r'[IV]', r'[IV]?')
PULSAR = _pat(r'PSR [A-Za-z0-9+._-]+', r'(?:P(?:S(?:R(?: [A-Za-z0-9+._-]*)?)?)?)?')
LABEL = _pat(r'[A-Za-z0-9.-]+', r'[A-Za-z0-9.-]*')


def _full(n):
    k = n[0]
    if k == 'lit':
        return re.escape(n[1])
    if k == 'pat':
        return '(?:%s)' % n[1]
    if k == 'seq':
        return ''.join(_full(x) for x in n[1])
    if k == 'opt':
        return '(?:%s)?' % _full(n[1])
    return '(?:%s)' % '|'.join(_full(x) for x in n[1])


def _prefix(n):
    k = n[0]
    if k == 'lit':
        s = n[1]
        return '(?:%s)' % '|'.join(re.escape(s[:i]) for i in range(len(s), -1, -1))
    if k == 'pat':
        return '(?:%s)' % n[2]
    if k == 'opt':
        return _prefix(n[1])
    if k == 'alt':
        return '(?:%s)' % '|'.join(_prefix(x) for x in n[1])
    xs = n[1]
    return '(?:%s)' % '|'.join(''.join(_full(x) for x in xs[:i]) + _prefix(xs[i]) for i in range(len(xs)))


def _expr(e):
    """Canonical text of a field's expression (what REASON_FIELDS is keyed on); never source positions (f-string node
    positions are unreliable before Python 3.9)."""
    if isinstance(e, ast.Name):
        return e.id
    if isinstance(e, ast.Attribute):
        return f'{_expr(e.value)}.{e.attr}'
    if isinstance(e, ast.Call):
        return f'{_expr(e.func)}(' + ', '.join(_expr(a) for a in e.args) + ')'
    if isinstance(e, ast.Subscript):
        s = e.slice.value if isinstance(e.slice, ast.Index) else e.slice
        return f'{_expr(e.value)}[{_expr(s)}]'
    if isinstance(e, ast.Constant):
        return repr(e.value)
    if isinstance(e, ast.IfExp):
        return f'{_expr(e.body)} if {_expr(e.test)} else {_expr(e.orelse)}'
    return ast.dump(e)


_TREES = {}


def _tree(name):
    p = WRITERS[name]
    st = os.stat(p)
    key = (st.st_mtime_ns, st.st_size)
    if _TREES.get(name, (None,))[0] != key:
        with open(p) as fh:
            _TREES[name] = (key, ast.parse(fh.read(), filename=p))
    return _TREES[name][1]


def _scope(name, func):
    tree = _tree(name)
    if func is None:
        return tree
    fs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == func]
    if len(fs) != 1:
        raise TemplateError(f'{len(fs)} functions {func} in {WRITERS[name]}')
    return fs[0]


def _starts(text):
    return lambda vals: bool(vals) and isinstance(vals[0], ast.Constant) and str(vals[0].value).startswith(text)


def _after(field, text):
    """The suffix templates of parent_curve_table: '{field}<text>...'."""
    return lambda vals: (len(vals) > 1 and isinstance(vals[0], ast.FormattedValue) and _expr(vals[0].value) == field
                         and isinstance(vals[1], ast.Constant) and str(vals[1].value).startswith(text))


def _joined(name, func, pred, what):
    hits = [n for n in ast.walk(_scope(name, func)) if isinstance(n, ast.JoinedStr) and pred(n.values)]
    if len(hits) != 1:
        raise TemplateError(f'{len(hits)} f-strings for {what} in {WRITERS[name]}' + (f' {func}()' if func else '') +
                            ' - the writer changed: declare its template')
    return hits[0]


def _node(js, fields, what, skip_leading=None):
    vals = list(js.values)
    if skip_leading is not None:
        vals = vals[1:]
    parts = []
    for v in vals:
        if isinstance(v, ast.Constant):
            parts.append(_lit(str(v.value)))
            continue
        ex = _expr(v.value)
        spec = ''.join(str(c.value) for c in v.format_spec.values) if v.format_spec else ''
        if ex in fields:
            parts.append(fields[ex])
        elif re.fullmatch(r'\.\d+f', spec):
            nd = int(spec[1:-1])
            parts.append(_pat(r'-?\d+' + (r'\.\d{%d}' % nd if nd else ''), r'[-\d.]*'))
        elif spec == 'g':
            parts.append(NUMBER)
        else:
            raise TemplateError(f'the field {{{ex}{":" + spec if spec else ""}}} of {what} has no declared pattern')
    return _seq(*parts)


def _const(name, var):
    for n in _tree(name).body:
        if (isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id == var and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)):
            return n.value.value
    raise TemplateError(f'no string constant {var} in {WRITERS[name]}')


def _returned_words(name, func):
    """The words a function returns as the second element of its tuples (removal_rules.deep_elevation: 'at mid-hour', ...)."""
    out = [r.value.elts[1].value for r in ast.walk(_scope(name, func)) if isinstance(r, ast.Return)
           and isinstance(r.value, ast.Tuple) and len(r.value.elts) == 2 and isinstance(r.value.elts[1], ast.Constant)]
    if not out:
        raise TemplateError(f'no words returned by {WRITERS[name]} {func}()')
    return out


def _assigned_words(name, var):
    """The string constants a variable is assigned (parent_curve_table _unit = 'bw' if ... else 'bw (image-pixel)')."""
    out = []
    for n in ast.walk(_tree(name)):
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == var for t in n.targets):
            for c in ast.walk(n.value):
                if isinstance(c, ast.Constant) and isinstance(c.value, str):
                    out.append(c.value)
    if not out:
        raise TemplateError(f'no string assigned to {var} in {WRITERS[name]}')
    return sorted(set(out))


def _tuple_words(name, var):
    for n in _tree(name).body:
        if (isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id == var and isinstance(n.value, ast.Tuple)):
            return [c.value for c in n.value.elts if isinstance(c, ast.Constant)]
    raise TemplateError(f'no tuple {var} in {WRITERS[name]}')


def reason_templates():
    """[(kind, instant, tiers, template tree)] of every removal reason the conformance writes (production's p3s2_conform
    writes the same words: tests/test_removal_rules.py, on Run_P3S5). Fields: REASON_FIELDS by expression, else by format
    spec ('.Nf', 'g'); a field with neither refuses (TemplateError)."""
    RR, PCT = 'removal_rules', 'parent_curve_table'
    samples = _tuple_words(PCT, '_XM_SAMPLES')
    fields = {'words': _alt(*[_lit(w) for w in _returned_words(RR, 'deep_elevation')]), 'hit': _named(DATE, 'night'),
              'stokes': _named(STOKES, 'stokes'), 'jupiter_track_radius(band)': NUMBER, '_cc().ROWLINE_MIN': INT,
              'c.SIZE_REMOVE_BEAMS': NUMBER, 'c.SIZE_SNR': NUMBER, "cols['axis_norm']": NUMBER, 'c.AXIS_REMOVE': NUMBER,
              'pn_': _named(PULSAR, 'pulsar'), 'n_': _named(INT, 'n'), 'sep_': _named(_pat(r'-?\d+', r'[-\d]*'), 'sep'),
              'r99_': _named(_pat(r'-?\d+', r'[-\d]*'), 'r99'),
              'pn[k[0]]': PULSAR, 'K_PSR': NUMBER, '_unit': _alt(*[_lit(u) for u in _assigned_words(PCT, '_unit')]),
              "L if rec else 'all-elevation'": LABEL,
              'xn[j]': _pat(r'(?:%s):[^;]+?' % '|'.join(re.escape(s) for s in samples), r'.*')}
    base = _node(_joined(PCT, None, _starts('table: parent S='), 'the table note'), fields, 'the table note')
    elev = _node(_joined(PCT, None, lambda v: (len(v) == 3 and isinstance(v[0], ast.Constant) and v[0].value == '['
                                               and isinstance(v[1], ast.FormattedValue) and 'all-elevation' in _expr(v[1].value)),
                         'the elevation-band prefix'), fields, 'the elevation-band prefix')
    xm = _node(_joined(PCT, None, _after('note[j]', '; xm '), 'the cross-match suffix'), fields, 'the cross-match suffix',
               skip_leading='note[j]')
    psr = _node(_joined(PCT, None, _after('note[j]', '; ATNF pulsar '), 'the pulsar suffix'), fields, 'the pulsar suffix',
                skip_leading='note[j]')
    note = _seq(_opt(elev), base, _opt(xm), _opt(psr))          # a table CUT (the removal's own note)
    # the notes of rows the table KEPT by an exemption (a pulsar group removes pulsar-exempted rows: its 'was: {nt}')
    ex_psr = _node(_joined(PCT, None, _starts('table-exempt: ATNF pulsar '), 'the pulsar exemption'),
                   dict(fields, **{'note[j]': _seq(base, _opt(xm))}), 'the pulsar exemption')
    ex_xm = _node(_joined(PCT, None, _starts('table-exempt: xm '), 'the cross-match exemption'),
                  dict(fields, **{'note[j]': base}), 'the cross-match exemption')
    kept = _seq(_opt(elev), _alt(ex_psr, ex_xm))
    group = _node(_joined(RR, 'pulsar_consistency', _starts('table: multi-detection source inconsistent'), 'the pulsar group'),
                  dict(fields, nt=_alt(note, kept)), 'the pulsar group')
    table_js = _joined(RR, 'table_reason', _starts('per-band per-Stokes table: '), 'the table reason')

    def one(func, start, what):
        return _node(_joined(RR, func, _starts(start), what), fields, what)
    return [
        # the B-02 pulsar stage writes the group's words bare (conform_check.pulsars); production's conformance wraps them
        ('pulsar-group', 'pulsar-group', ALL_TIERS, _alt(_node(table_js, dict(fields, note=group), 'the table reason'), group)),
        ('table', 'own', ALL_TIERS, _node(table_js, dict(fields, note=note), 'the table reason')),
        ('duplicate', 'own', ALL_TIERS, _lit(_const(RR, 'DUPLICATE'))),
        ('excluded epoch', 'own', ALL_TIERS, _lit(_const(RR, 'EXCLUDED'))),           # data_decisions: its own night
        ('elevation floor', 'own', ALL_TIERS, one('elevation_reason', 'elevation ', 'the elevation floor')),
        ('daytime', 'own', ALL_TIERS, one('daytime_verdicts', 'daytime: ', 'the daytime gate')),   # its own frame's times
        ('twilight', 'own', ALL_TIERS, one('daytime_verdicts', 'twilight in ', 'the twilight gate')),
        ('daytime unreadable', 'own', ALL_TIERS, _lit(_const(RR, 'DAYTIME_UNREADABLE'))),
        ('jupiter-track', 'master-night', ALL_TIERS, one('jupiter_track_reason', 'jupiter-track: ', 'the Jupiter track')),
        # keyed by NIGHT (weights.csv dates): the deep instant; at 10 minutes it would remove every window of the night
        ('noise sanity', 'own', ('deep',), one('noise_reason', 'epoch refused: ', 'the noise sanity gate')),
        ('row-line', 'own', ALL_TIERS, one('rowline_reason', 'row-line: ', 'the row line')),   # finals of its own epoch
        ('no ledger verdict', 'own', ALL_TIERS, _lit(_const(RR, 'UNBACKED'))),
        ('size', 'own', ALL_TIERS, one('size_axis', 'gaussfit size ', 'the size gate')),         # its own fit
        ('axis ratio', 'own', ALL_TIERS, one('size_axis', 'axis ratio ', 'the axis-ratio gate')),
    ]


_GRAMMAR = {}


def _grammar():
    key = tuple((os.stat(p).st_mtime_ns, os.stat(p).st_size) for p in WRITERS.values())
    if _GRAMMAR.get('key') != key:
        _GRAMMAR.clear()
        _GRAMMAR['kinds'] = [(k, i, t, re.compile(_full(n)), re.compile(_prefix(n))) for k, i, t, n in reason_templates()]
        _GRAMMAR['key'] = key
    return _GRAMMAR['kinds']


def reason_kind(reason, tier=None, truncated=False):
    """(kind, instant) of a removal reason matched WHOLE by a declared template (``truncated``: a reason cut at
    REJECTED_REASON_CHARS may be any prefix of one); (None, None) when undeclared; (kind, None) when the kind is not declared
    for ``tier``. Raises TemplateError when a writer's template cannot be read."""
    kinds = _grammar()
    r = (reason or '').strip() if not truncated else (reason or '')
    memo = _GRAMMAR.setdefault('memo', {})
    key = (r, bool(truncated))
    if key not in memo:
        memo[key] = next(((kind, inst, tiers) for kind, inst, tiers, rx, rxp in kinds
                          if rx.fullmatch(r) or (truncated and rxp.fullmatch(r))), (None, None, ()))
        if len(memo) > 500000:
            memo.clear()
    kind, inst, tiers = memo[key]
    if kind is None:
        return None, None
    return (kind, inst) if tier is None or tier in tiers else (kind, None)


# ---------------------------------------------------------------------------------------------------------------------
# T1-T4
# ---------------------------------------------------------------------------------------------------------------------

def own_instant(tier, lst, date):
    """The instant key of a deep / 10-minute row: (lst, night) / (lst, '<night>T<kk>'); None when the date is not of the
    tier's form (a row of another tier)."""
    d = (date or '').strip()
    if tier == 'deep' and re.match(r'^\d{4}-\d\d-\d\d$', d):
        return (lst, d)
    if tier == 'sublst' and re.match(r'^\d{4}-\d\d-\d\dT\d\d$', d):
        return (lst, d)
    return None


def pulsar_positions():
    """{ATNF common name: (RA, Dec)} from catalog.PULSARS_CSV, as the conformance reads it (ops/p3s2_conform.py)."""
    pos = {}
    with open(_cfg().catalog.PULSARS_CSV, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            try:
                pos[(r.get('common_name') or '').strip()] = (float(r['ra_current']), float(r['dec_current']))
            except (KeyError, TypeError, ValueError):
                pass
    return pos


def _instant_of_id(cand_id):
    """(lst, date) from a cand_id <prefix>_<lst>_<date>_<band>_<n> (catalog rows, restored <S>R_ rows); None otherwise."""
    p = (cand_id or '').split('_')
    return (p[1], p[2]) if len(p) >= 5 else None


def _tenmin_tree(record):
    """The 10-minute window tree of a record from its own run name (Run_SLST<hh><arm>, Run_SLST<arm>, compare new_<arm>,
    catalog_<hh>h_<arm>): paths.SUBLST_ROOT / catalog_sublst.ARM_TREES[arm]; None when the name carries no arm."""
    C = _cfg()
    for part in reversed(os.path.abspath(record or '').split(os.sep)):
        m = (re.fullmatch(r'Run_SLST(?:\d\d)?(?P<arm>[A-Z]+)', part) or re.fullmatch(r'new_(?P<arm>[A-Z]+)', part)
             or re.fullmatch(r'catalog_\d\dh_(?P<arm>[A-Z]+)', part))
        if m and m.group('arm') in C.catalog_sublst.ARM_TREES:
            return os.path.join(C.paths.SUBLST_ROOT, C.catalog_sublst.ARM_TREES[m.group('arm')])
    return None


def mask_evidence(tier, lst, band, stokes, date, record, ctx, cache):
    """(state 'masked' | 'unmasked', the file read) of the product's own recorded Jupiter-mask state for one row (T3), or
    (None, why none could be read): the 10-minute row's searched frame (JUPMASK), the deep row's master (JUPMASK or a
    '-jupmask' COADDOP). A file newer than the record is not the product's own."""
    from astropy.io import fits
    if tier == 'deep':
        roots_csv = ctx.get('deep_roots_csv') or _cfg().catalog.DEEP_ROOTS_CSV
        if ('roots', roots_csv) not in cache:
            roots = {}
            try:
                for line in open(roots_csv):
                    p = line.strip().split(',')
                    if len(p) >= 3 and p[0]:
                        roots[p[0]] = p[2]
            except OSError:
                pass
            cache[('roots', roots_csv)] = roots
        root = cache[('roots', roots_csv)].get(f'{lst}_{band}')
        if not root or stokes not in DEEP_MASTERS:
            return None, f'no co-add root of {lst}_{band} in {roots_csv}'
        path = os.path.join(root, f'{lst}_{band}', 'coadd_final', DEEP_MASTERS[stokes].format(band=band))
    else:
        tree = ctx.get('tenmin_tree') or _tenmin_tree(record)
        if not tree:
            return None, 'the record names no 10-minute arm (its window tree is unknown)'
        fcsv = os.path.join(tree, f'{lst}_{band}', 'frames.csv')
        if ('frames', fcsv) not in cache:
            fr = {}
            try:
                for r in csv.DictReader(open(fcsv)):
                    fr[(r.get('key', ''), r.get('stokes', ''))] = r.get('sub_path') or r.get('src')   # last row wins
            except OSError:
                pass
            cache[('frames', fcsv)] = fr
        path = cache[('frames', fcsv)].get((date, stokes))
        if not path:
            return None, f'no searched frame of {lst}_{band} {date} {stokes} in {fcsv}'
    if ('hdr', path) not in cache:
        try:
            h = fits.getheader(path)
            cache[('hdr', path)] = (os.path.getmtime(path), h.get('JUPMASK'), str(h.get('COADDOP', '') or ''))
        except (OSError, ValueError) as e:
            cache[('hdr', path)] = (None, f'{type(e).__name__}', '')
    mt, jm, op = cache[('hdr', path)]
    if mt is None:
        return None, f'{path} unreadable ({jm})'
    if record and mt > os.path.getmtime(record) + 1.0:
        return None, f'{os.path.basename(path)} is newer than the record (not the product\'s own state)'
    try:
        masked = (jm is not None and float(jm) > 0) or 'jupmask' in op.lower()
    except (TypeError, ValueError):
        masked = 'jupmask' in op.lower()
    return ('masked' if masked else 'unmasked'), f'{os.path.basename(path)} (JUPMASK {jm}, COADDOP {op or "-"})'


def check_removals(tier, removed, population=None, groups=None, record=None, ctx=None):
    """T1-T4 on removal records. ``removed``: [{'id', 'lst', 'date', 'cadence', 'band', 'stokes', 'reason',
    'jupiter_night'[, 'kind'][, 'truncated']}]; ``population``: [(instant, ra, dec, id)] of the as-run detections (T4 bound),
    None = not available; ``groups``: pulsar_groups.csv rows (exact T4), or None; ``record``: the file the rows came from
    (T3: no evidence newer than it); ``ctx``: deep_roots_csv / tenmin_tree. Returns (passed, value, detail)."""
    C = _cfg()
    ctx = ctx or {}
    msgs, counts, imprint, pgroups, cache = [], {}, {'file': 0, 'tier rule': 0}, [], {}
    try:
        _grammar()
    except (TemplateError, SyntaxError, OSError) as e:
        return False, None, f'T2: the removal templates cannot be read from the writers ({e}) - refusing'
    for x in removed:
        inst = own_instant(tier, x['lst'], x['date'])
        if x.get('cadence') and x['cadence'] != TIER_CADENCE[tier]:
            msgs.append(f"T1 {x['id']}: a {x['cadence']} row in the {tier} record (vetoes never cross tiers)")
            continue
        if inst is None:
            msgs.append(f"T1 {x['id']}: date {x['date']!r} is not a {tier} instant (another tier's row, or no instant)")
            continue
        if x.get('kind'):
            kind, what = x['kind'], 'own'
        else:
            kind, what = reason_kind(x['reason'], tier, truncated=bool(x.get('truncated')))
        if kind is None:
            msgs.append(f"T2 {x['id']}: undeclared removal reason {x['reason'][:90]!r} (no template matches it whole; its "
                        f"instant must be declared)")
            continue
        if what is None:
            msgs.append(f"T2 {x['id']}: a {kind} removal is not declared for the {tier} tier ({x['reason'][:60]!r}): it is "
                        f"not measured at this tier's instant")
            continue
        counts[kind] = counts.get(kind, 0) + 1
        if what == 'master-night':
            m = JUPITER_NIGHT_RE.search(x['reason'] or '')
            night = (x.get('jupiter_night') or '').strip() or (m.group(1) if m else '')
            if not night:
                msgs.append(f"T3 {x['id']}: a Jupiter-track removal that names no night")
            elif night != inst[1][:10]:
                state, where = mask_evidence(tier, x['lst'], x.get('band', ''), x.get('stokes', ''), x['date'], record, ctx,
                                             cache)
                src = 'file'
                if state is None:
                    state, src = MASK_TIER_RULE[tier][0], 'tier rule'
                    where = f'no readable record ({where}) - the tier rule: {MASK_TIER_RULE[tier][1]}'
                if state == 'masked':
                    msgs.append(f"T3 {x['id']} ({inst[0]} {inst[1]}): removed for Jupiter on night {night} - another "
                                f"instant; its master carries the moving Jupiter mask, so no master imprint stands for it "
                                f"({where})")
                else:
                    imprint[src] += 1
        elif what == 'pulsar-group':
            m = PULSAR_GROUP_RE.search(x['reason'] or '')
            if m is None:
                msgs.append(f"T4 {x['id']}: a pulsar-group removal whose pulsar cannot be read: {x['reason'][:80]!r}")
                continue
            nm = PULSAR_GROUP_N_RE.search(x['reason'] or '')
            pgroups.append((x, inst, m.group('psr').strip(), int(nm.group('n')) if nm else None))
    n_pg = 0
    if pgroups:
        if groups is not None:
            by_member = {}
            for g in groups:
                for mid in (g.get('members') or '').split('|'):
                    by_member.setdefault(mid, []).append(g)
            insts = {}
            for _i, _r, _d, mid in (population or ()):
                insts[mid] = _i
            for x, inst, name, _n in pgroups:
                n_pg += 1
                gs = [g for g in by_member.get(x['id'], ()) if g.get('pulsar') == name and str(g.get('consistent')) == '0']
                ep = [g for g in gs if g.get('grouping') == 'per-epoch']
                if not ep:
                    msgs.append(f"T4 {x['id']}: removed by the {name} group but no inconsistent per-epoch group of it holds the "
                                f"row ({len(gs)} across-epoch) - only one instant's group may remove")
                    continue
                other = sorted({mid for g in ep for mid in g['members'].split('|')
                                if (insts.get(mid) or _instant_of_id(mid)) != inst})
                if other:
                    msgs.append(f"T4 {x['id']}: its per-epoch {name} group holds {len(other)} member(s) of another instant, "
                                f"e.g. {other[0]}")
        elif population is None:
            msgs.append(f'T4: {len(pgroups)} pulsar-group removals and no as-run record to bound their groups')
        else:
            psr = pulsar_positions()
            rad = float(C.catalog_conform.PULSAR_PRECUT_DEG)
            byinst = {}
            for i_, ra, dec, _id in population:
                byinst.setdefault(i_, []).append((ra, dec))
            for x, inst, name, n in pgroups:
                n_pg += 1
                if name not in psr:
                    msgs.append(f"T4 {x['id']}: pulsar {name} is not in {C.catalog.PULSARS_CSV}")
                    continue
                pra, pdec = psr[name]
                n_here = sum(1 for ra, dec in byinst.get(inst, ()) if _sep_deg(ra, dec, pra, pdec) <= rad)
                if (n or 2) > n_here:
                    msgs.append(f"T4 {x['id']} ({inst[0]} {inst[1]}): its {name} group has {n or 2} members"
                                f"{'' if n else ' at least (n cut from the record)'} but its instant holds {n_here} "
                                f"detection(s) within {rad:g} deg of the pulsar - the group spans instants")
    n_imp = imprint['file'] + imprint['tier rule']
    what = (f'{len(removed)} removals of the {tier} record: '
            + ', '.join(f'{k} {v}' for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
            + f'; jupiter-track naming another night: {n_imp} as the master\'s imprint'
            + (f' (unmasked by the product\'s own files {imprint["file"]}, by the tier rule {imprint["tier rule"]})'
               if n_imp else '')
            + f'; pulsar-group removals checked against one instant: {n_pg}'
            + (' (group record)' if groups is not None and n_pg else ' (as-run bound)' if n_pg else ''))
    return _fail(msgs, len(removed), what)


def _tier_of(rows, ctx, key='cadence'):
    t = ctx.get('tier')
    if t in TIER_CADENCE:
        return t
    cads = {(r.get(key) or '').strip() for r in rows} - {''}
    return 'sublst' if cads == {'10min'} else 'deep'


def _removed(rows):
    return [dict(id=r.get('cand_id', ''), lst=r.get('lst', ''), date=r.get('date', ''), cadence=r.get('cadence', ''),
                 band=r.get('band', ''), stokes=r.get('stokes', ''), reason=r.get('removal_reason', ''),
                 jupiter_night=r.get('jupiter_track_night', '')) for r in rows]


def _population(tier, rows, skip=()):
    out = []
    for r in rows:
        i_ = own_instant(tier, r.get('lst', ''), r.get('date', ''))
        ra, dec = _f(r.get('ra')), _f(r.get('dec'))
        if i_ and ra is not None and dec is not None and r.get('cand_id') not in skip:
            out.append((i_, ra, dec, r.get('cand_id', '')))
    return out


def _needs_population(removed):
    try:
        return any(reason_kind(x['reason'])[0] == 'pulsar-group' for x in removed if not x.get('kind'))
    except TemplateError:
        return False


def _skip(removed):
    try:
        return {x['id'] for x in removed if not x.get('kind') and reason_kind(x['reason'])[0] in NEVER_GROUPED}
    except TemplateError:
        return set()


@content_test('conformant_catalog', name='per_instant')
def conform_per_instant(path, **ctx):
    """T1-T4 on master_catalog_removed.csv beside the conformant catalogue (module docstring)."""
    from exopipe.phase3.lib.catalog import tests as CT
    from exopipe.phase3.lib.catalog import conform_check as CC
    rmp = CT._sibling(path, 'master_catalog_removed.csv', ctx)
    asp = CT._sibling(path, 'master_catalog_asrun.csv', ctx)
    if not rmp:
        return False, None, 'no master_catalog_removed.csv beside: the removals cannot be placed at their instants'
    _, rem = _rows(rmp)
    if not rem:
        return True, 0, 'no removal in master_catalog_removed.csv: nothing to judge'
    tier = _tier_of(rem, ctx)
    removed = _removed(rem)
    population = None
    if asp and _needs_population(removed):
        population = _population(tier, _rows(asp)[1], _skip(removed))
    gp = CT._sibling(path, CC.PULSAR_GROUPS, ctx)
    groups = _rows(gp)[1] if gp else None
    if groups is not None and population is None:
        population = [(own_instant(tier, r.get('lst', ''), r.get('date', '')), None, None, r.get('cand_id', ''))
                      for r in (_rows(asp)[1] if asp else [])]
    return check_removals(tier, removed, population, groups, record=rmp, ctx=ctx)


def fork_records(fork_dir):
    """Every removal record of a multiples fork's work tree (ops/multiples_fork.py): master_catalog_removed.csv and each
    conformance run's master_catalog_removed_<tag>.csv (validity 'A', table 'B', 'B2', the members' and calibration runs in
    their subdirectories) and the row-line record rowline_removed.csv."""
    return sorted(glob.glob(os.path.join(fork_dir, '**', 'master_catalog_removed*.csv'), recursive=True)
                  + glob.glob(os.path.join(fork_dir, '**', 'rowline_removed*.csv'), recursive=True))


@content_test('fork_catalog', name='per_instant')
def fork_per_instant(path, **ctx):
    """T1-T4 on EVERY removal record of the deep multiples fork (fork_records of <run>/fork_<S>/; module docstring) for a
    fork catalogue directory source_catalogs_<S>_f<wb|sb>: each conformance run copies its removals to its own tag (the
    next run overwrites master_catalog_removed.csv), so the last file alone hides the validity run's removals."""
    from exopipe.phase3.lib.catalog import tests as CT
    key = CT._catalogue_key(path)
    if not key or not key.split('_')[1].startswith('f'):
        return False, None, f'{os.path.basename(os.path.normpath(path))} is not a fork catalogue (source_catalogs_<S>_f<wb|sb>)'
    fd = os.path.join(os.path.dirname(os.path.abspath(os.path.normpath(path))), f'fork_{key[0]}')
    files = fork_records(fd)
    if not files:
        return False, None, f'no removal record under {fd}: the fork\'s removals cannot be placed at their instants'
    recs = {}
    for p in files:
        rows = _rows(p)[1]
        rm = _removed(rows)
        if os.path.basename(p).startswith('rowline_removed'):     # the fork's own row line: no reason column
            for x in rm:
                x['kind'] = 'row-line'
        recs[p] = rm
    population = None
    if any(_needs_population(rm) for rm in recs.values()):
        asp = os.path.join(fd, 'master_catalog_asrun.csv')
        seen, population = set(), []
        for rows in ([_rows(asp)[1]] if os.path.exists(asp) else []) + [_rows(p)[1] for p in files]:
            for e in _population('deep', rows):
                k = (e[0], round(e[1], 5), round(e[2], 5), e[3])
                if k not in seen:
                    seen.add(k)
                    population.append(e)
    msgs, n, parts = [], 0, []
    for p, rm in recs.items():
        ok, val, det = check_removals('deep', rm, population, None, record=p, ctx=ctx)
        n += len(rm)
        m = re.search(r"jupiter-track naming another night: (\d+) as the master's imprint \(unmasked by the product's own "
                      r"files (\d+), by the tier rule (\d+)\)", det)
        parts.append(f'{os.path.relpath(p, fd)} {len(rm)}' + (f' (jupiter-track imprint {m.group(1)}: files {m.group(2)}, '
                                                              f'tier rule {m.group(3)})' if m else ''))
        if not ok:
            msgs.append(f'{os.path.relpath(p, fd)}: {det}')
    return _fail(msgs, n, f'{len(files)} fork records judged ({", ".join(parts)})')


@content_test('rejected_record', name='per_instant')
def rejected_per_instant(path, **ctx):
    """T1-T4 on rejected.csv (the 10-minute / deep pages' record; module docstring): the row's instant from its detail
    '<cand_id> <date> peak ...' and cell '<lst>_<band>'; a reason of exactly REJECTED_REASON_CHARS characters may be cut
    (emit_rejected_record.py) and matches a template's prefix; the as-run population = master_catalog.csv beside + the
    record."""
    _, rows = _rows(path)
    removed = []
    for r in rows:
        det = (r.get('detail') or '').split()
        v = r.get('veto', '')
        removed.append(dict(id=r.get('cand_id', ''), lst=(r.get('cell') or '').split('_')[0],
                            date=det[1] if len(det) > 1 else '', cadence=r.get('tier', ''), band=r.get('subband', ''),
                            stokes=r.get('stokes', ''), reason=v, jupiter_night='',
                            truncated=len(v) == REJECTED_REASON_CHARS))
    tier = ctx.get('tier') if ctx.get('tier') in TIER_CADENCE else (
        'sublst' if {x['cadence'] for x in removed} == {'10min'} else 'deep')
    population = None
    if _needs_population_trunc(removed):
        mc = os.path.join(os.path.dirname(os.path.abspath(path)), 'master_catalog.csv')
        if os.path.exists(mc):
            population = _population(tier, _rows(mc)[1])
            for x, r in zip(removed, rows):
                i_ = own_instant(tier, x['lst'], x['date'])
                ra, dec = _f(r.get('ra_deg')), _f(r.get('dec_deg'))
                if i_ and ra is not None and dec is not None and not _never_grouped(x):
                    population.append((i_, ra, dec, x['id']))
    return check_removals(tier, removed, population, None, record=path, ctx=ctx)


def _never_grouped(x):
    try:
        return reason_kind(x['reason'], truncated=x.get('truncated'))[0] in NEVER_GROUPED
    except TemplateError:
        return False


def _needs_population_trunc(removed):
    try:
        return any(reason_kind(x['reason'], truncated=x.get('truncated'))[0] == 'pulsar-group' for x in removed)
    except TemplateError:
        return False


# ---------------------------------------------------------------------------------------------------------------------
# snapshot runs (10-s and 1-minute)
# ---------------------------------------------------------------------------------------------------------------------

def _parse_member(s):
    """(cell, subband, tier, ts) of a measured member: '<cell> <sb> <ts>' (a 10-s subband member, negative_lobe_veto),
    '<cell> <sb> <tier> <ts>' (a block or group image, negative_lobe_block_veto), '<cell> <sb> <S> <tier> <ts>'
    (reflection_asymmetry_veto); None when unparsable."""
    t = (s or '').split()
    if len(t) == 3:
        return t[0], t[1], '10s', t[2]
    if len(t) in (4, 5) and t[-2] in SNAP_TIERS:
        return t[0], t[1], t[-2], t[-1]
    return None


def read_record(path):
    """The per-detection record's columns RECORD_COLS, row by row (a generator of dicts)."""
    with gzip.open(path, 'rt', newline='') as f:
        rd = csv.reader(f)
        h = next(rd)
        ix = {k: h.index(k) for k in RECORD_COLS if k in h}
        miss = [k for k in RECORD_COLS if k not in ix]
        if miss:
            raise ValueError(f'{os.path.basename(path)} lacks the columns {miss}')
        for r in rd:
            yield {k: r[i] if i < len(r) else '' for k, i in ix.items()}


def check_snapshot_run(run):
    """S1-S3 on a snapshot chain run directory (module docstring): (passed, value, detail)."""
    msgs = []
    sp = os.path.join(run, 'chain_summary.json')
    if not os.path.exists(sp):
        return False, None, 'no chain_summary.json'
    S = json.load(open(sp))
    piv = S.get('per_instant_vetoes')
    vsp, vdp = os.path.join(run, 'vetoed_sources.csv'), os.path.join(run, 'vetoed_detections.csv')
    if not os.path.exists(vsp):
        return False, None, 'no vetoed_sources.csv: the run\'s veto removals cannot be judged'
    src_veto = [r for r in _rows(vsp)[1] if r.get('veto') in SNAP_VETOES]
    if piv is None or not os.path.exists(vdp):
        if src_veto:
            return False, len(src_veto), (f'S1: {len(src_veto)} sources removed by the negative-lobe / reflection-asymmetry veto '
                                          f'with no per-instant record (chain_summary per_instant_vetoes '
                                          f'{"absent" if piv is None else "present"}, vetoed_detections.csv '
                                          f'{"present" if os.path.exists(vdp) else "absent"}): the vetoes were applied per '
                                          f'SOURCE - one member\'s measurement removed every instant (before clv15 99fe3ea)')
        if piv is None and not os.path.exists(vdp):
            return True, 0, 'no veto removed a source and the run predates the per-instant record: nothing to judge'
        return False, None, 'S1: the per-instant summary and vetoed_detections.csv must exist together'
    vd = _rows(vdp)[1]
    n_vd = {k: 0 for k in SNAP_VETOES}
    want_member = []                     # (row id, member frame key, the row's instant label, row's frame key)
    for r in vd:
        rid = f"{r.get('cell')} {r.get('subband')} {r.get('stokes')} {r.get('tier')} {r.get('ts')}"
        by = r.get('removed_by', '')
        if by not in SNAP_VETOES and by not in SNAP_REEVAL:
            msgs.append(f'S2 {rid}: removed_by {by!r} is neither a veto nor a re-evaluation')
            continue
        it = (r.get('instant') or '').split()
        if len(it) != 3 or it[0] != r.get('cell') or it[1] != r.get('tier') or it[1] not in SNAP_TIERS:
            msgs.append(f"S2 {rid}: its instant {r.get('instant')!r} is not of its own cell and tier")
            continue
        if by in SNAP_VETOES:
            n_vd[by] += 1
            mem = _parse_member(r.get(SNAP_VETOES[by][2]))
            if mem is None:
                msgs.append(f'S2 {rid}: {by} without its measured member')
            elif mem[0] != r.get('cell') or mem[2] != r.get('tier'):
                msgs.append(f"S2 {rid}: {by} measured on {r.get(SNAP_VETOES[by][2])!r}, another "
                            f"{'cell' if mem[0] != r.get('cell') else 'tier'} (vetoes never cross tiers or instants)")
            else:
                want_member.append((rid, mem, it[2], (r.get('cell'), r.get('subband'), r.get('tier'), r.get('ts'))))
    rp = os.path.join(run, 'detection_record.csv.gz')
    n_removed_by_veto = sum(n_vd.values())
    if not os.path.exists(rp):
        if n_removed_by_veto:
            msgs.append(f'S3: {n_removed_by_veto} detections removed by a veto and no per-detection record '
                        f'(detection_record.csv.gz) to place them at their instants')
        return _fail(msgs, n_removed_by_veto, f'{len(vd)} vetoed_detections rows; no per-detection record')
    frame_inst = {}                      # (cell, subband, tier, ts) -> {instant_key}
    pending = []                         # (detection id, member frame key, its instant key)
    n_rec = {k: 0 for k in SNAP_VETOES}
    n_rows = 0
    rec_of = {v[0]: k for k, v in SNAP_VETOES.items()}
    try:
        for d in read_record(rp):
            n_rows += 1
            key = (d['cell'], d['subband'], d['tier'], d['ts'])
            frame_inst.setdefault(key, set()).add(d['instant_key'])
            veto = rec_of.get(d['first_removed_by'])
            if not veto:
                continue
            n_rec[veto] += 1
            _rec, gate, mcol = SNAP_VETOES[veto]
            did = f"{d['cell']} {d['subband']} {d['stokes']} {d['tier']} {d['ts']}"
            if d[gate] != 'remove':
                msgs.append(f"S3 {did}: removed by the {veto} veto while its own instant's verdict is {d[gate]!r} - "
                            f"another instant's measurement")
                continue
            mem = _parse_member(d[mcol])
            if mem is None or mem[0] != d['cell'] or mem[2] != d['tier']:
                msgs.append(f"S3 {did}: its {veto} member {d[mcol]!r} is not of its own cell and tier")
                continue
            pending.append((did, mem, d['instant_key']))
    except (OSError, ValueError, EOFError) as e:
        return False, None, f'per-detection record unreadable: {type(e).__name__}: {e}'
    for did, mem, ik in pending:
        got = frame_inst.get(mem, set())
        if ik not in got:
            msgs.append(f"S3 {did}: measured at {' '.join(mem)} (instant {sorted(got) or 'not in the record'}), not at its own "
                        f"instant {ik}")
    for rid, mem, lab, own in want_member:
        if lab not in frame_inst.get(own, set()):
            msgs.append(f'S2 {rid}: the record places its frame at {sorted(frame_inst.get(own, set()))}, not at {lab}')
        elif lab not in frame_inst.get(mem, set()):
            msgs.append(f"S2 {rid}: its measured member {' '.join(mem)} is at instant {sorted(frame_inst.get(mem, set()))}, "
                        f"not {lab}")
    for k in SNAP_VETOES:
        if n_rec[k] != n_vd[k]:
            msgs.append(f'S3: the record says {n_rec[k]} detections were removed by {k}, vetoed_detections.csv lists {n_vd[k]}')
    tiers = {}
    for r in vd:
        if r.get('removed_by') in SNAP_VETOES:
            tiers[r.get('tier')] = tiers.get(r.get('tier'), 0) + 1
    what = (f"per instant: {n_removed_by_veto} detections removed by a veto (negative lobe {n_vd['negative-lobe']}, reflection "
            f"asymmetry {n_vd['reflection-asymmetry']}; " + ', '.join(f'{t} {n}' for t, n in sorted(tiers.items()))
            + f"), each on its own instant's measurement of its own cell and tier ({n_rows} record rows); "
            f"{len(vd) - n_removed_by_veto} survivors removed on re-evaluation (floor / seed); 3-D gate, seed and routes are "
            f"source-level rules, not judged here")
    return _fail(msgs, n_removed_by_veto, what)


@content_test('snapshot_run', name='per_instant')
def snapshot_per_instant(path, **ctx):
    """S1-S3 on a snapshot chain run directory (module docstring)."""
    return check_snapshot_run(path)
