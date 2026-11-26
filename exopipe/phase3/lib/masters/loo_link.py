"""The link between a leave-one-out (LOO) master and its parent master: the one rule the masters' T3 (deep_verify.t3), the
search's LOO gate as frames previews it (deep_cells.preview, D-G16) and the dewarp step's MDWSTAT gate (G-R5) call.

A LOO master <date>_<S>_loo_<taper>_<robust><tokens>_<band>.fits (coadd._LOO_RE) is the parent master
<S>_deep_<taper>_<robust><tokens>_<band>.fits with the night <date> left out. The masters build writes LOO masters of two
parents only, I Taper R-0.75 and V Taper R0 (coadd.py write_loo_masters calls), and only for a stack of N nights with
N in masters_build.LOO_N_RANGE (R-LOO-THRESHOLD-CORRECTED, R-LOO-PER-STOKES-RATIFIED): ``needed``; the masters' count test
(lib/masters/tests.py loo_master.count_per_stokes) wants exactly N of them per Stokes then and none otherwise: ``expected``
(G-R5 applies the same count). Each carries LOOEXCL = the night left out, NCOADD = the parent's - 1 and PARENTDG = the
parent's CONSTDIG (coadd.py, since 2026-09-09).

The parts of the rule, each a function of the two primary headers (any mapping with .get):
  digest_stale    the search engine's rule (lib/search/deep_search.py, the LOO gate of _loo_masters; D-G16): a LOO
                  master whose PARENTDG and its parent's CONSTDIG are both present and differ is a stale derivative; one
                  without PARENTDG (a build before PARENTDG) is one when its COADDOP differs from the parent's
  digest_problem  digest_stale as a problem line (legacy=True: the search's rule); legacy=False is T3's own form, the raw
                  equality PARENTDG == the parent's CONSTDIG (check_rebuild.py:77), which a fresh build always meets
  ncoadd_problem  T3's NCOADD rule: the LOO master's NCOADD is its parent's - 1
  looexcl_problem LOOEXCL is the night the name leaves out (the masters' loo_master test, the Release 1 header repair)
  link_problems   the rule G-R5 applies: digest_problem (the search's), ncoadd_problem and looexcl_problem
Every function is pure; each caller keeps its own verdict and wording (tests/test_dewarp_loo_mdwstat.py checks that T3's
and D-G16's verdicts are those of their earlier inline code on every combination of the cards)."""

PARENTS = {'I': 'I_deep_Taper_R075', 'V': 'V_deep_Taper_R0'}     # coadd.py: write_loo_masters of the I and V finals


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def n_range():
    """(lo, hi): the stacks that get LOO masters (masters_build.LOO_N_RANGE, nights per Stokes)."""
    lo, hi = _cfg().masters_build.LOO_N_RANGE
    return lo, hi


def needed(n):
    """True when a parent of NCOADD ``n`` must have LOO masters (lo <= n <= hi, masters_build.LOO_N_RANGE; as T3 and the
    masters' count test compare it - a non-number raises, as there)."""
    lo, hi = n_range()
    return lo <= n <= hi


def expected(n):
    """The number of LOO masters a parent of NCOADD ``n`` (an int) must have: ``n`` when ``needed(n)``, else 0 - the masters'
    count test (lib/masters/tests.py loo_master.count_per_stokes, R-LOO-PER-STOKES-RATIFIED), which calls this."""
    return n if needed(n) else 0


def parse(name):
    """coadd._LOO_RE's groups of a LOO master's file name, or None."""
    from exopipe.phase3.lib.masters.coadd import _LOO_RE
    m = _LOO_RE.match(name)
    return m.groupdict() if m else None


def parent_name(name):
    """The file name of the parent master of the LOO master ``name`` (same Stokes, taper, robust weighting, tokens and band),
    or None when ``name`` is not a LOO master's."""
    g = parse(name)
    if g is None:
        return None
    return f"{g['st']}_deep_{g['taper']}_{g['rob']}{g['sh'] or ''}{g['md'] or ''}_{g['fl']}.fits"


def digest_stale(lh, ph):
    """The search engine's LOO rule (D-G16): None when the LOO master (header ``lh``) is a current derivative of its parent
    (header ``ph``), else ('PARENTDG', parentdg, constdig) or ('COADDOP', the LOO's COADDOP, the parent's)."""
    pd = str(lh.get('PARENTDG', '') or '')
    cd = str(ph.get('CONSTDIG', '') or '')
    if pd and cd and pd != cd:
        return ('PARENTDG', pd, cd)
    if not pd and lh.get('COADDOP') != ph.get('COADDOP'):
        return ('COADDOP', lh.get('COADDOP'), ph.get('COADDOP'))
    return None


def digest_problem(lh, ph, legacy=True):
    """None, or why the LOO master is not its parent's by the digest. ``legacy`` True: digest_stale (the search's rule; a
    LOO master without PARENTDG is judged by COADDOP); False: T3's raw equality PARENTDG == the parent's CONSTDIG."""
    if not legacy:
        if lh.get('PARENTDG') != ph.get('CONSTDIG'):
            return f"PARENTDG {lh.get('PARENTDG')!r} is not the parent's CONSTDIG {ph.get('CONSTDIG')!r}"
        return None
    s = digest_stale(lh, ph)
    if s is None:
        return None
    if s[0] == 'PARENTDG':
        return f'PARENTDG {s[1]!r} is not the parent\'s CONSTDIG {s[2]!r}: a stale derivative (the search refuses it, D-G16)'
    return (f'no PARENTDG, and COADDOP {s[1]!r} is not the parent\'s {s[2]!r}: a stale derivative (the search refuses it, '
            f'D-G16)')


def ncoadd_problem(lh, ph):
    """None, or why the LOO master's NCOADD is not its parent's - 1 (T3; a parent without NCOADD raises TypeError, as T3)."""
    if lh.get('NCOADD') != ph.get('NCOADD') - 1:
        return f"NCOADD {lh.get('NCOADD')!r} is not the parent's NCOADD {ph.get('NCOADD')!r} - 1"
    return None


def looexcl_problem(lh, date):
    """None, or why LOOEXCL is not the night ``date`` the LOO master's name leaves out."""
    if str(lh.get('LOOEXCL', '') or '') != date:
        return f"LOOEXCL {lh.get('LOOEXCL')!r} is not the night {date} its name leaves out"
    return None


def link_problems(lh, ph, date):
    """The rule G-R5 applies to one LOO master (header ``lh``, the night ``date`` of its name) and its parent (``ph``): the
    search's digest rule, T3's NCOADD rule and LOOEXCL. A parent without an integer NCOADD is a problem here (the rule
    cannot be evaluated). Returns the problem lines ([] = linked)."""
    out = []
    d = digest_problem(lh, ph, legacy=True)
    if d:
        out.append(d)
    n = ph.get('NCOADD')
    if not isinstance(n, int) or isinstance(n, bool):
        out.append(f'the parent carries no integer NCOADD ({n!r}): the NCOADD rule cannot be evaluated')
    else:
        d = ncoadd_problem(lh, ph)
        if d:
            out.append(d)
    d = looexcl_problem(lh, date)
    if d:
        out.append(d)
    return out
