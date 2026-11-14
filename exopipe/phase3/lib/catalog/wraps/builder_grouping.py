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

FROZEN STATISTICS (EXOPIPE3_FROZEN_MODE; the per-hour 10-minute catalog build: Gregg 2026-11-13, fix ledger
R-GREGG-20261113-SPEEDUP-CHECKIN-TEN-ANSWERS answer 3, "the builder runs only on the sources the new night touches, with its
statistics frozen from the last all-nights build, and writes a delta rather than a whole new catalog"; arrival order:
R-GREGG-20261112-PER-HOUR-STEPS-SIX-RULINGS (2)). Design B only.
  unset        nothing below happens: the runner is the one it was.
  record       the all-nights build. No patch is added and no output changes; after the builder ran, the runner writes
               builder_frozen.json into the output directory from the builder's own namespace: the band systematics (SYS_AS), the
               footprint areas and sample densities with what they were computed from (the LSTs, each Master image's path, size
               and modification time, the beams, the cross-match list, the selection module), every source before the 7-8 sigma
               gate (band, best S/N, best detection, first
               matched sample, name), every input detection that passed the leak veto with the grouping it was built under (its
               identity and a digest of its multiples-level group), the detections the leak veto removed, and the sha1 of each input.
  delta        one LST hour's sources that differ from the all-nights build. The input is the all-nights build's own selected
               rows plus the selected rows of the nights appended since; the grouping table is the present one. A source is built
               only for a TOUCHED identity: one that holds a detection the all-nights build did not have, or whose identity or
               multiples-level group is not the one recorded. Every quantity the builder takes from the whole population is, for
               each touched source, the all-nights build's value plus the source's own detections, never another appended
               night's (FROZEN_PATCHES F1 to F9 below), so the same member detections give the same row whichever night arrived
               first. The footprint areas and the sample densities, which no detection enters, are the record's and are not
               computed again (F10, F11; they are 85% of a build's time): a delta build REFUSES when what they were computed from
               is no longer what the record names. The runner writes builder_delta.json (the touched identities, what each replaces, the checks) and removes
               E_params.json and the README, which describe a population this directory does not hold. No rank, no review
               directory.
  passthrough  the eleven frozen patches applied with every hook returning production's own value: the builder must write
               byte-identical files (the check that the patching changes nothing else).
The frozen patches are applied to ONE builder version (EXOPIPE3_FROZEN_BUILDER_SHA1, config catalog_grouping.
FROZEN_BUILDER_SHA1): the list of what the builder computes over the whole population was made by reading that version, so any
other REFUSES until the list is made again. EXOPIPE3_FROZEN_TOUCH=all (tests): every identity is touched; with no appended
night the delta must then equal the all-nights build except for the rank and the review directory.
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

# ---- FROZEN STATISTICS (see the module text). Line numbers are those of the builder with sha1 44d2763087cd...; the builder
# since 2026-11-14 (exopipe_ops 9e25883, sha1 430f83b31708...) has four more lines before its line 322 (`RHO = {}`): add 4 to
# every number from 318 on. Its one change is how the loop of F11 reads the cross-match positions; no quantity was added. -----
FROZEN_MODES = ('', 'record', 'delta', 'passthrough')
FROZEN_RECORD = 'builder_frozen.json'       # written by the all-nights build (record), read by a delta build
DELTA_RECORD = 'builder_delta.json'         # written by a delta build


def group_digest_of(rows):
    """A digest of everything a source row takes from its multiples-level group: the group's detections (``rows``: the
    grouping table's rows of one multiples_id) with their identity, identity name, window, multiplicity and group name. Equal
    digests = the same group for every row built from it. One definition: the builder's delta mode, the frozen and delta
    records, and the certificate of an untouched build (lib/catalog/delta_build.certificate) all use this function."""
    lines = sorted('|'.join((g['cand_id'], g['identity_id'], g['identity_name'], g['lst'], g['date'], g['multiple'],
                             g['multiplicity'], g['multiples_name'])) for g in rows)
    return hashlib.sha1('\n'.join(lines).encode()).hexdigest()[:16]


FROZEN_VERSION = 2                          # 2: the record names what the footprint and the densities were computed from
FROZEN_PATCHES = [
    # other bands' detections a source is matched against (allband_positions, 728-745: column other_bands, the MB rung)
    ('F1-other-band-rows',
     "            if b2 == band or b2 not in allband_positions:\n                continue\n"
     "            c2, s2, f2, k2 = allband_positions[b2]\n",
     "            if b2 == band or b2 not in allband_positions:\n                continue\n"
     "            c2, s2, f2, k2 = __FROZEN__.other_band(b2, allband_positions[b2], mem)   # EXOPIPE3 FROZEN F1\n"
     "            if not len(s2):\n                continue\n"),
    # the other band's median systematic in that match (SYS_AS, 580, used at 737)
    ('F2-other-band-systematic',
     "+ SYS_AS.get(b2, 0.0) ** 2 +",
     "+ __FROZEN__.sys_as(b2, SYS_AS.get(b2, 0.0), mem) ** 2 +"),
    # epochs searched for the source's cells (n_nights_searched, 698)
    ('F3-nights-searched',
     "        nsr = len(set().union(*[nsearched.get((c_, band), set())\n"
     "                                for c_ in cells]))\n",
     "        nsr = len(__FROZEN__.nights(set().union(*[nsearched.get((c_, band), set())\n"
     "                                for c_ in cells]), mem, cells, band))   # EXOPIPE3 FROZEN F3\n"),
    # the band's S/N list behind the null density and the anchor trials of the repeat statistic (_snrs, 860-877, 904)
    ('F4-density-snrs',
     "        dets = sorted((abs(float(d[3])) for d in s_['_dets']), reverse=True) if s_.get('_dets') else [abs(float(s_['best_snr']))]\n",
     "        _snrs = __FROZEN__.snrs(_snrs, band, br, s_)   # EXOPIPE3 FROZEN F4\n"
     "        dets = sorted((abs(float(d[3])) for d in s_['_dets']), reverse=True) if s_.get('_dets') else [abs(float(s_['best_snr']))]\n"),
    # covering epochs, the trials of the repeat statistic (M, 881)
    ('F5-covering-epochs',
     "        M = sum(len(v) for (l_, sc_), v in nsearched.items() if sc_ == band and _elev(l_, ra_, dec_) >= EFLOOR)\n",
     "        M = sum(len(v) for (l_, sc_), v in nsearched.items() if sc_ == band and _elev(l_, ra_, dec_) >= EFLOOR)\n"
     "        M = __FROZEN__.cover(M, s_, band, ra_, dec_)   # EXOPIPE3 FROZEN F5\n"),
    # sources at or above a source's S/N: all bands (xm_P_corr, 981), its band and every band (E, 983), matched (984)
    ('F6-sources-above',
     "            N = n_ge(a)\n",
     "            N = __FROZEN__.n_ge(n_ge(a), a, s)   # EXOPIPE3 FROZEN F6\n"),
    ('F7-chance-load',
     "            e_here = E_band(smp, band, a, _r_obs_deg); e_all = E_allbands(smp, a, _r_obs_deg)\n",
     "            e_here = __FROZEN__.E_band(E_band(smp, band, a, _r_obs_deg), smp, band, a, _r_obs_deg, s); "
     "e_all = __FROZEN__.E_allbands(E_allbands(smp, a, _r_obs_deg), smp, a, _r_obs_deg, s)   # EXOPIPE3 FROZEN F7\n"),
    ('F8-matched-above',
     "            nobs = n_obs_ge(smp, a)\n",
     "            nobs = __FROZEN__.n_obs_ge(n_obs_ge(smp, a), smp, a, s)   # EXOPIPE3 FROZEN F8\n"),
    # rank and review directory: places in the whole catalogue's order (1102-1108)
    ('F9-rank',
     "        s['review_dir'] = f\"candidates/{band}/{s['rank']}_{band}_{s['source_id']}\"\n",
     "        s['review_dir'] = f\"candidates/{band}/{s['rank']}_{band}_{s['source_id']}\"\n"
     "        __FROZEN__.unranked(s)   # EXOPIPE3 FROZEN F9\n"),
    # the searchable footprint per band (AREA, 309-315: 4,000 random sky positions through every Master's valid mask and the
    # parent exclusion) and each sample's density on it (RHO, 318-323: every cross-match source through the same test, 85% of a
    # build's time, measured 2026-11-14): neither depends on a detection, so a delta build takes the all-nights build's
    ('F10-footprint-area',
     "AREA = {}\nfor band in BANDS:\n    frac = np.mean([in_footprint(band, r_, d_)\n",
     "AREA = {}\nfor band in __FROZEN__.measure_area(BANDS, AREA):   # EXOPIPE3 FROZEN F10\n"
     "    frac = np.mean([in_footprint(band, r_, d_)\n"),
    # the loop's head only: the line that follows it was rewritten on 2026-11-14 (exopipe_ops 9e25883: the cross-match positions
    # taken once as arrays, the same values) and the patch must stand on both versions
    ('F11-sample-density',
     "RHO = {}\nfor band in BANDS:\n    inb = np.array([in_footprint(band, ",
     "RHO = {}\nfor band in __FROZEN__.measure_rho(BANDS, RHO):   # EXOPIPE3 FROZEN F11\n"
     "    inb = np.array([in_footprint(band, "),
]


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


def patch(src, frozen=False):
    """(patched source, [names]) - REFUSES unless every patch's text occurs exactly once and the census holds. frozen: the
    eleven FROZEN_PATCHES too (the caller has checked the builder's version)."""
    bad = []
    patches = PATCHES + (FROZEN_PATCHES if frozen else [])
    for name, old, _new in patches:
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
    for name, old, new in patches:
        out = out.replace(old, new)
    return out, [p[0] for p in patches]


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
        self.frozen = None                       # the Frozen object of a delta build (run sets it)
        self.T, self.mgroup, self.checked = {}, {}, set()
        self._digest = {}
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
        delta = self.frozen is not None and self.frozen.active
        for i, r in enumerate(br):
            k = self.entry(r)['identity_id']     # every input row is checked against the table, built or not
            if delta and not self.frozen.touched(k):
                continue                         # a delta build: a source only for a touched identity
            if k not in order:
                order[k] = len(out)
                out.append([])
            out[order[k]].append(i)
        return out

    def group_digest(self, multiples_id):
        """A digest of everything a source row takes from its multiples-level group: the group's detections with their
        identity, window and multiplicity. Equal digests = the same group for every row built from it."""
        d = self._digest.get(multiples_id)
        if d is None:
            d = self._digest[multiples_id] = group_digest_of(self.mgroup[multiples_id])
        return d

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


def _sha1_file(path):
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def _write_json(path, obj):
    with open(path + '.tmp', 'w') as f:
        json.dump(obj, f, indent=1, sort_keys=True)
    os.replace(path + '.tmp', path)


def _ge(sorted_list, thr):
    """how many values of an ascending list are >= thr"""
    import bisect
    return len(sorted_list) - bisect.bisect_left(sorted_list, thr)


class Frozen:
    """__FROZEN__ in the builder's namespace (the module text, FROZEN STATISTICS). mode 'passthrough' (and '', 'record'):
    every hook returns the builder's own value. mode 'delta': the all-nights build's value (record: its builder_frozen.json)
    plus the source's own detections. g: the grouping Hook (design B). ns: the builder's namespace (set before it runs)."""

    def __init__(self, mode, g, record=None, touch=''):
        if mode not in FROZEN_MODES:
            raise Refused(f'EXOPIPE3_FROZEN_MODE={mode!r} is not one of {FROZEN_MODES} - REFUSING')
        if touch not in ('', 'all'):
            raise Refused(f"EXOPIPE3_FROZEN_TOUCH={touch!r} is not '' or 'all' - REFUSING")
        self.mode, self.g, self.ns = mode, g, None
        self.active = mode == 'delta'
        self.touch_all = touch == 'all'
        self.rec, self.rec_path, self.rec_sha1 = None, None, ''
        self._ready = False
        self._fp_checked = False
        if mode and not g.active:
            raise Refused(f'EXOPIPE3_FROZEN_MODE={mode!r} needs the design-B grouping (EXOPIPE3_GROUPING_MODE=design-b) - REFUSING')
        if not self.active:
            return
        for v in ('P3_REP_DENSITY_CSV', 'P3_DETREC_BUILDER_OUT'):
            if os.environ.get(v):
                raise Refused(f'a delta build with {v} set: that input / record is of an all-nights build - REFUSING')
        if not record or not os.path.exists(record):
            raise Refused(f'EXOPIPE3_FROZEN_RECORD={record!r}: no record of the all-nights build ({FROZEN_RECORD}; the all-nights '
                          f'build writes it under EXOPIPE3_FROZEN_MODE=record) - REFUSING')
        with open(record, 'rb') as f:
            data = f.read()
        self.rec_path, self.rec_sha1 = os.path.abspath(record), hashlib.sha1(data).hexdigest()
        self.rec = json.loads(data.decode())
        if self.rec.get('version') != FROZEN_VERSION:
            raise Refused(f'{record}: record version {self.rec.get("version")!r}, this runner reads {FROZEN_VERSION} - REFUSING')

    # ---- what a delta build knows once the builder has read its rows ---------------------------------------------------------
    def _prepare(self):
        if self._ready:
            return
        ns, T, rec = self.ns, self.g.T, self.rec
        np = ns['np']
        rows = ns['rows']                                   # the input rows after the leak veto
        ids = [r['cand_id'] for r in rows]
        have = set(ids)
        frozen = set(rec['rows'])
        gone = sorted(frozen - have)
        if gone:
            raise Refused(f'{len(gone)} detection(s) of the all-nights build are not in this build\'s input after the leak veto '
                          f'(e.g. {gone[:3]}): the input is not that build\'s rows plus appended nights - REFUSING')
        back = sorted(have & set(rec.get('vetoed') or ()))
        if back:
            raise Refused(f'{len(back)} detection(s) the all-nights build\'s leak veto removed are in this build (e.g. {back[:3]}):'
                          f' the leak-veto tables do not hold that build\'s - REFUSING')
        self.new = have - frozen
        miss = [c for c in ids if c not in T]
        if miss:
            raise Refused(f'{len(miss)} input detection(s) are not in the grouping table (e.g. {miss[:3]}) - a stale grouping: '
                          f'group again - REFUSING')
        self.ident = {c: T[c]['identity_id'] for c in ids}
        touched = {}
        for c in ids:
            e = T[c]
            if self.touch_all:
                touched.setdefault(e['identity_id'], 'all (EXOPIPE3_FROZEN_TOUCH)')
            elif c in self.new:
                touched[e['identity_id']] = 'new detection'
            else:
                was = rec['rows'][c]
                if e['identity_id'] != was[0] or self.g.group_digest(e['multiples_id']) != was[1]:
                    touched.setdefault(e['identity_id'], 'regrouped')
        self.touched_why = touched
        # per band: which input rows are the all-nights build's, and each identity's rows
        self.fmask, self.rows_of, self.fsnr = {}, {}, {}
        for band, br in ns['byband'].items():
            self.fmask[band] = np.array([r['cand_id'] in frozen for r in br], bool)
            by = collections.defaultdict(list)
            for i, r in enumerate(br):
                by[self.ident[r['cand_id']]].append(i)
            self.rows_of[band] = dict(by)
            self.fsnr[band] = np.sort(np.array([abs(r['_snr']) for r in br if r['cand_id'] in frozen], float))
        # the all-nights build's sources before its 7-8 sigma gate, with the identity each belongs to NOW
        self.fb, self.fs, self.fx = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
        for band, snr, best, smp, name in rec['sources']:
            if best not in T:
                raise Refused(f'the all-nights build\'s source {name} {band} (best detection {best}) is not in the grouping table '
                              f'- REFUSING')
            v = abs(float(snr))
            self.fb[band].append(v)
            if smp:
                self.fs[smp].append(v)
            self.fx[T[best]['identity_id']].append((band, v, smp, name))
        for d in (self.fb, self.fs):
            for v in d.values():
                v.sort()
        # the bands in the builder's own order of summation (E_allbands sums over its output bands), then any band only the
        # record holds
        self.rec_only_bands = [b for b in rec['bands'] if b not in ns['byband']]
        self._own = {}
        self._ready = True

    def touched(self, identity_id):
        self._prepare()
        return identity_id in self.touched_why

    def _x(self, mem):
        return self.ident[mem[0]['cand_id']]

    def _own_rows(self, band, x, new_only=True):
        br = self.ns['byband'].get(band, [])
        return [br[i] for i in self.rows_of.get(band, {}).get(x, ()) if not new_only or br[i]['cand_id'] in self.new]

    # ---- F1, F2: the other bands a source is matched against ------------------------------------------------------------------
    def other_band(self, b2, pos, mem):
        if not self.active:
            return pos
        self._prepare()
        mask = self.fmask[b2].copy()
        for i in self.rows_of[b2].get(self._x(mem), ()):
            mask[i] = True
        if mask.all():
            return pos
        c2, s2, f2, k2 = pos
        return c2[mask], s2[mask], f2[mask], [k for k, m_ in zip(k2, mask) if m_]

    def sys_as(self, b2, value, mem):
        if not self.active:
            return value
        if b2 in self.rec['sys_as']:
            return self.rec['sys_as'][b2]
        own = self._own_rows(b2, self._x(mem), new_only=False)     # a band the all-nights build had no detection in
        ns = self.ns
        return float(ns['np'].median([ns['_sys_det'](r) for r in own])) if own else value

    # ---- F3, F5: epochs --------------------------------------------------------------------------------------------------------
    def nights(self, base, mem, cells, band):
        if not self.active:
            return base
        return set(base) | {r['date'] for r in self._own_rows(band, self._x(mem)) if r['lst'] in cells}

    def cover(self, M, s_, band, ra_, dec_):
        if not self.active:
            return M
        ns = self.ns
        x = self.g.T[s_['best_cand_id']]['identity_id']
        seen = ns['nsearched']
        wins = {(r['lst'], r['date']) for r in self._own_rows(band, x) if r['date'] not in seen.get((r['lst'], band), ())}
        return M + sum(1 for l_, _d in wins if ns['_elev'](l_, ra_, dec_) >= ns['EFLOOR'])

    # ---- F4: the band's S/N list of the repeat statistic ------------------------------------------------------------------------
    def snrs(self, cur, band, br, s_):
        if not self.active:
            return cur
        if self.ns.get('_wts') is not None:
            raise Refused('a delta build with a weighted null density (P3_REP_DENSITY_CSV) - REFUSING')
        np = self.ns['np']
        x = self.g.T[s_['best_cand_id']]['identity_id']
        own = [abs(r['_snr']) for r in self._own_rows(band, x)]
        return np.sort(np.concatenate([self.fsnr[band], np.array(own, float)])) if own else self.fsnr[band]

    # ---- F6 to F8: how many sources stand at or above a source's S/N -------------------------------------------------------------
    def _own_sources(self, x):
        o = self._own.get(x)
        if o is None:
            T = self.g.T
            o = self._own[x] = [s2 for s2 in self.ns['allsrc'] if T[s2['best_cand_id']]['identity_id'] == x]
        return o

    def _nb(self, band, thr, x):
        """the all-nights build's sources of the band at or above thr, without those that now belong to identity x, plus x's
        sources of this build"""
        return (_ge(self.fb.get(band, []), thr) - sum(1 for b_, v, _s, _n in self.fx.get(x, ()) if b_ == band and v >= thr)
                + sum(1 for s2 in self._own_sources(x) if s2['band'] == band and abs(s2['best_snr']) >= thr))

    def _xs(self, s):
        return self.g.T[s['best_cand_id']]['identity_id']

    def _bands(self):
        return list(self.ns['allout']) + [b for b in self.rec_only_bands if b not in self.ns['allout']]

    def n_ge(self, value, a, s):
        if not self.active:
            return value
        x = self._xs(s)
        return sum(self._nb(b, a, x) for b in self._bands())

    def E_band(self, value, smp, band, a, r_deg, s):
        if not self.active:
            return value
        ns = self.ns
        return self._nb(band, a, self._xs(s)) * ns['RHO'].get((smp, band), 0.0) * ns['np'].pi * r_deg ** 2

    def E_allbands(self, value, smp, a, r_deg, s):
        if not self.active:
            return value
        ns, x = self.ns, self._xs(s)
        return sum(self._nb(b, a, x) * ns['RHO'].get((smp, b), 0.0) for b in self._bands()) * ns['np'].pi * r_deg ** 2

    def n_obs_ge(self, value, smp, a, s):
        if not self.active:
            return value
        x = self._xs(s)
        return (_ge(self.fs.get(smp, []), a) - sum(1 for _b, v, s_, _n in self.fx.get(x, ()) if s_ == smp and v >= a)
                + sum(1 for s2 in self._own_sources(x) if s2['_matches'] and s2['_matches'][0][2] == smp
                      and abs(s2['best_snr']) >= a))

    # ---- F10, F11: the searchable footprint and the samples' densities on it -----------------------------------------------------
    def footprint_inputs(self):
        """What AREA and RHO are computed from, as far as a build can name it without computing them: the LSTs, and per band
        each Master image (path, size, modification time) and the beam; the cross-match list (its length and a digest of its
        positions and samples); the selection module the parent-exclusion test comes from (sha1 of its file)."""
        ns = self.ns
        np = ns['np']
        masters = {}
        for band in ns['BANDS']:
            ms = []
            for lst in ns['LSTS']:
                q = ns['master_path'](lst, band)
                if q and os.path.exists(q):
                    st = os.stat(q)
                    ms.append([lst, os.path.abspath(q), int(st.st_size), int(st.st_mtime_ns)])
            masters[band] = ms
        xc = ns['xc']
        try:
            ra, dec = np.asarray(xc.ra.deg, float), np.asarray(xc.dec.deg, float)
        except AttributeError:
            ra, dec = np.array([c.ra.deg for c in xc], float), np.array([c.dec.deg for c in xc], float)
        h = hashlib.sha1()
        h.update(np.ascontiguousarray(ra).tobytes())
        h.update(np.ascontiguousarray(dec).tobytes())
        h.update('\n'.join(str(x) for x in ns['xsamples']).encode())
        sel = sys.modules.get('select_v3')
        return dict(lsts=[str(x) for x in ns['LSTS']], masters=masters,
                    beams={b: float(ns['BEAM_AS'][b]) for b in ns['BANDS'] if b in ns['BEAM_AS']},
                    xcat=dict(n=int(len(ra)), sha1=h.hexdigest()),
                    select_v3=_sha1_file(sel.__file__) if sel is not None and getattr(sel, '__file__', None) else '')

    def _footprint_check(self):
        """a delta build: the recorded footprint and densities stand only on the same LSTs, Masters, beams, cross-match list
        and selection module (bands the all-nights build did not hold are measured by this build)"""
        if self._fp_checked:
            return
        now, was = json.loads(json.dumps(self.footprint_inputs())), self.rec.get('footprint_inputs')
        if not isinstance(was, dict):
            raise Refused(f'{self.rec_path}: the record does not name what its footprint was computed from - REFUSING')
        bad = [k for k in ('lsts', 'xcat', 'select_v3') if now[k] != was.get(k)]
        for b in self.rec['area']:
            for k in ('masters', 'beams'):
                if now[k].get(b) != was.get(k, {}).get(b):
                    bad.append(f'{k} of {b}')
        if bad:
            raise Refused(f'the searchable footprint and the sample densities of the all-nights build ({self.rec_path}) do not '
                          f'stand for this build: {", ".join(bad[:8])} differ(s) - build all nights again - REFUSING')
        self._fp_checked = True

    def measure_area(self, bands, AREA):
        """the bands whose footprint the builder measures itself: every band (not a delta build), or in a delta build only a
        band the all-nights build did not hold - the others' AREA is the record's"""
        if not self.active:
            return bands
        self._footprint_check()
        for b in bands:
            if b in self.rec['area']:
                AREA[b] = self.rec['area'][b]
        todo = [b for b in bands if b not in self.rec['area']]
        print(f'searchable footprint: the all-nights build\'s for {len(bands) - len(todo)} band(s) ({self.rec_path})'
              + (f'; measured here for {todo} (no detection of the all-nights build)' if todo else ''), flush=True)
        return todo

    def measure_rho(self, bands, RHO):
        if not self.active:
            return bands
        self._footprint_check()
        for k, v in self.rec['rho'].items():
            smp, band = k.rsplit('|', 1)
            if band in bands:
                RHO[(smp, band)] = v
        return [b for b in bands if b not in self.rec['area']]

    # ---- F9 ----------------------------------------------------------------------------------------------------------------------
    def unranked(self, s):
        if self.active:
            s['rank'] = ''
            s['review_dir'] = ''

    # ---- the two records ---------------------------------------------------------------------------------------------------------
    def write_record(self, out_dir, builder_sha1, norm):
        """builder_frozen.json of an all-nights build, from the builder's namespace after it ran."""
        ns, g = self.ns, self.g
        gated, imgcsv = os.path.abspath(ns['gated_csv']), os.path.abspath(ns['imgcsv'])
        with open(gated, newline='') as f:
            sel = [r['cand_id'] for r in csv.DictReader(f)]
        kept = [r['cand_id'] for r in ns['rows']]
        if len(set(sel)) != len(sel) or len(set(kept)) != len(kept):
            raise Refused(f'{gated}: a cand_id twice - no frozen record - REFUSING')
        srcs = [[s['band'], float(s['best_snr']), s['best_cand_id'], (s['_matches'][0][2] if s['_matches'] else ''),
                 s['source_id']] for s in ns['allsrc']]
        ep = os.path.join(out_dir, 'E_params.json')
        try:
            with open(ep) as f:
                e_par = json.load(f)['best_snr']
        except (OSError, ValueError, KeyError) as e:
            raise Refused(f'{ep}: unreadable or without best_snr ({type(e).__name__}: {e}; the builder only prints when it '
                          f'cannot write it) - no frozen record - REFUSING')
        mine = collections.defaultdict(list)
        for band, snr, _b, _s, _n in srcs:
            mine[band].append(abs(snr))
        if {b: sorted(v) for b, v in mine.items() if v} != {b: v for b, v in e_par.items() if v}:
            raise Refused(f'{ep}: its S/N lists are not those of the sources this runner would record - no frozen record - REFUSING')
        lv = ns.get('_LV') or ''
        pt = os.environ.get('P3_TENMIN_PERTIE') or ''
        rec = dict(version=FROZEN_VERSION, builder=os.path.abspath(ns['__file__']), builder_sha1=builder_sha1,
                   builder_normalized_sha1=norm, tier=g.tier, repeat_rungs=g.rungs,
                   grouping_table=dict(path=os.path.abspath(g.table_path), sha1=g.table_sha1),
                   inputs=dict(selected=dict(path=gated, sha1=_sha1_file(gated), rows=len(sel)),
                               images=dict(path=imgcsv, sha1=_sha1_file(imgcsv) if os.path.exists(imgcsv) else ''),
                               pertie=dict(path=os.path.abspath(pt), sha1=_sha1_file(pt)) if pt else None,
                               leakveto=(dict(dir=os.path.abspath(lv),
                                              tables={os.path.basename(q): _sha1_file(q)
                                                      for q in sorted(ns['_glob'].glob(f'{lv}/leakveto_*.csv'))}) if lv else None)),
                   bands=list(ns['BANDS']), sys_as={b: float(v) for b, v in ns['SYS_AS'].items()},
                   area={b: float(v) for b, v in ns['AREA'].items()}, footprint_inputs=self.footprint_inputs(),
                   rho={f'{smp}|{band}': float(v) for (smp, band), v in ns['RHO'].items()},
                   sources=srcs,
                   rows={c: [g.T[c]['identity_id'], g.group_digest(g.T[c]['multiples_id'])] for c in kept},
                   vetoed=sorted(set(sel) - set(kept)))
        p = os.path.join(out_dir, FROZEN_RECORD)
        _write_json(p, rec)
        return p, len(kept), len(srcs)

    def close_delta(self, out_dir):
        """After a delta build: the checks against the record, builder_delta.json, and the two population files removed."""
        self._prepare()
        ns, rec, g = self.ns, self.rec, self.g
        np = ns['np']
        bad = []
        frozen = set(rec['rows'])
        for band, v in rec['sys_as'].items():           # the all-nights rows give the recorded systematic again
            fr = [r for r in ns['byband'].get(band, []) if r['cand_id'] in frozen]
            now = float(np.median([ns['_sys_det'](r) for r in fr])) if fr else None
            if now != v:
                bad.append(f'{band}: the all-nights rows give a median systematic {now!r}, the record holds {v!r}')
        self._footprint_check()                         # AREA and RHO are the record's (F10, F11): what they stand on is equal
        img = os.path.abspath(ns['imgcsv'])
        if rec['inputs']['images']['sha1'] != (_sha1_file(img) if os.path.exists(img) else ''):
            bad.append(f'{img} is not the all-nights build\'s image table (sha1)')
        if bad:
            raise Refused('the delta build does not stand on the all-nights build it names: ' + '; '.join(bad[:6])
                          + (f' (+{len(bad) - 6} more)' if len(bad) > 6 else '') + ' - REFUSING')
        built = collections.defaultdict(list)
        for band, out in ns['allout'].items():
            for s in out:
                built[g.T[s['best_cand_id']]['identity_id']].append(s)
        rows_x = collections.defaultdict(list)
        for r in ns['rows']:
            rows_x[self.ident[r['cand_id']]].append(r['cand_id'])
        touched = []
        for x in sorted(self.touched_why):
            cands = rows_x[x]
            n_new = sum(1 for c in cands if c in self.new)
            touched.append(dict(
                identity_id=x, identity_name=g.T[cands[0]]['identity_name'], why=self.touched_why[x],
                kind='new' if n_new == len(cands) else ('extended' if n_new else 'regrouped'),
                detections=len(cands), new_detections=n_new,
                sources=sorted([s['band'], s['source_id'], int('rank' in s and not s.get('gate_7_8'))] for s in built.get(x, [])),
                replaces=sorted([b_, n_] for b_, _v, _s, n_ in self.fx.get(x, ()))))
        for name in ('E_params.json', 'README_source_catalogs.txt'):
            q = os.path.join(out_dir, name)
            if os.path.exists(q):
                os.remove(q)                 # they describe a population this directory does not hold
        with open(os.path.abspath(ns['gated_csv']), newline='') as f:
            sel = {r['cand_id'] for r in csv.DictReader(f)}
        p = os.path.join(out_dir, DELTA_RECORD)
        _write_json(p, dict(version=FROZEN_VERSION, frozen_record=self.rec_path, frozen_record_sha1=self.rec_sha1,
                            # the selected detections the leak veto removed (as builder_frozen.json): with `rows`, every
                            # detection this build read - the cumulative's gate compares that set with the present table's
                            vetoed=sorted(sel - {r['cand_id'] for r in ns['rows']}),
                            grouping_table=dict(path=os.path.abspath(g.table_path), sha1=g.table_sha1),
                            touch_all=self.touch_all, input_detections=len(ns['rows']), new_detections=len(self.new),
                            rows={r['cand_id']: [g.T[r['cand_id']]['identity_id'],
                                                 g.group_digest(g.T[r['cand_id']]['multiples_id'])] for r in ns['rows']},
                            touched=touched, checks=dict(sys_as=len(rec['sys_as']),
                                                         footprint='the record\'s; LSTs, Masters, beams, cross-match list '
                                                                   'and selection module equal to the all-nights build\'s',
                                                         images='same sha1 as the all-nights build')))
        return p, len(touched), len(self.new)


def run(builder, args, mode, table=None, tier=None, rungs='multiples-level', accepted=(), table_sha1=None, provenance=False,
        frozen_mode='', frozen_record=None, frozen_touch='', frozen_accepted=()):
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
    if frozen_mode not in FROZEN_MODES:
        raise Refused(f'EXOPIPE3_FROZEN_MODE={frozen_mode!r} is not one of {FROZEN_MODES} - REFUSING')
    patch_frozen = frozen_mode in ('delta', 'passthrough')
    if frozen_mode == 'record' and norm not in [a for a in frozen_accepted if a]:
        # the record mode patches nothing: a builder accepted for the grouping but not pinned for the frozen statistics (an older
        # frozen tree's) builds all nights as before and writes NO record - a delta build on it then refuses for want of one
        print(f'FROZEN-BUILDER record skipped: the builder {builder} (path-normalized sha1 {norm}) is not one whose population-wide '
              f'quantities were listed (EXOPIPE3_FROZEN_BUILDER_SHA1): no {FROZEN_RECORD} is written', flush=True)
        frozen_mode = ''
    if frozen_mode and norm not in [a for a in frozen_accepted if a]:
        raise Refused(f'EXOPIPE3_FROZEN_MODE={frozen_mode}: the builder {builder} (path-normalized sha1 {norm}) is not the one '
                      f'whose population-wide quantities were listed for the frozen statistics (EXOPIPE3_FROZEN_BUILDER_SHA1, '
                      f'config catalog_grouping.FROZEN_BUILDER_SHA1) - REFUSING')
    patched, names = patch(src, frozen=patch_frozen)
    hook = Hook(mode, table, tier, rungs, table_sha1)
    hook.table_path = table or ''
    frozen = Frozen(frozen_mode, hook, frozen_record, frozen_touch)
    hook.frozen = frozen
    mod = types.ModuleType('__main__')
    mod.__file__ = builder
    mod.__dict__['__GROUPING__'] = hook
    mod.__dict__['__FROZEN__'] = frozen
    mod.__dict__['__builtins__'] = __builtins__
    hook.ns = frozen.ns = mod.__dict__
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [os.path.dirname(builder)] + [p for p in sys.path if os.path.abspath(p or '.') != here]
    sys.argv = [builder] + list(args)
    sys.modules['__main__'] = mod
    print(f'GROUPING-BUILDER start: builder {builder} sha1 {sha1}; patches {len(names)} ({",".join(names)}); mode {mode}; '
          f'tier {tier}; repeat rungs {rungs}; table {table or "-"} sha1 {hook.table_sha1 or "-"} ({len(hook.T)} detections)'
          + (f'; frozen statistics: {frozen_mode}' + (f', record {frozen.rec_path} sha1 {frozen.rec_sha1}' if frozen.active else '')
             + (', every identity touched' if frozen.active and frozen.touch_all else '') if frozen_mode else ''),
          flush=True)
    code = compile(patched, builder, 'exec')
    exec(code, mod.__dict__)
    out_dir = (args[1] if os.path.isabs(args[1]) else os.path.join(os.getcwd(), args[1])) if len(args) >= 2 else None
    rec_sha256 = ''
    if frozen_mode == 'record':
        p, n_rows, n_src = frozen.write_record(out_dir, sha1, norm)
        with open(p, 'rb') as f:
            rec_sha256 = hashlib.sha256(f.read()).hexdigest()
        print(f'FROZEN-BUILDER record: {p}: {n_rows} input detections after the leak veto, {n_src} sources', flush=True)
    elif frozen.active:
        p, n_t, n_new = frozen.close_delta(out_dir)
        print(f'FROZEN-BUILDER delta: {p}: {n_new} detection(s) the all-nights build did not have, {n_t} touched identities; '
              f'checks against {frozen.rec_path} passed', flush=True)
    if provenance and hook.active and len(args) >= 2:
        p = os.path.join(out_dir, PROVENANCE)
        with open(p + '.tmp', 'w') as f:
            json.dump(dict(table=os.path.abspath(table), table_sha1=hook.table_sha1, tier=tier, repeat_rungs=rungs,
                           builder=builder, builder_sha1=sha1, builder_normalized_sha1=norm, rows_checked=len(hook.checked),
                           # binds builder_frozen.json to this directory: the cumulative's gate (lib/serve/sublst_build.
                           # grouping_mix) certifies a directory only from the record whose sha256 is named here
                           **({'frozen_record_sha256': rec_sha256} if rec_sha256 else {})),
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
            provenance=os.environ.get('EXOPIPE3_GROUPING_PROVENANCE') == '1',
            frozen_mode=os.environ.get('EXOPIPE3_FROZEN_MODE', ''), frozen_record=os.environ.get('EXOPIPE3_FROZEN_RECORD'),
            frozen_touch=os.environ.get('EXOPIPE3_FROZEN_TOUCH', ''),
            frozen_accepted=os.environ.get('EXOPIPE3_FROZEN_BUILDER_SHA1', '').split(','))
    except Refused as e:
        print(f'GROUPING-BUILDER REFUSING: {e}', flush=True)
        sys.exit(2)


if __name__ == '__main__':
    main(sys.argv)
