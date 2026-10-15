"""D-ADMIT (a) slice 6: the deep search with a cell's admitted nights (plan/wave2/DADMIT_DESIGN.md "Slices 5 and 6 as they will
be written"; ruling R-GREGG-20261013-REFACTOR-SHEET-DADMIT-B01-B02-B03-B05-B30-B32 "E15 accepts admitted nights"; Gregg
2026-10-14: an admitted night's own candidates are judged against the cell's searched nights plus that night).

One rule: every population the search computes - a median, a MAD, the sigma stack - is the cell's own, computed exactly as
today over the nights of its record; an admitted night is tested against those populations and never joins one, and its own
candidates are judged against a sigma stack of the cell's searched nights plus itself. A member's result therefore never
changes because a night was admitted, and nights admitted together or one after another are each judged as if alone.

The engine (lib/search/deep_search.run_deep_cell, a moved production module) calls these helpers from marked WAVE2 D-ADMIT
lines; under production's reading ('absent', the parity values) nights() is empty and every helper returns today's values.
Deep cadence only: the 10-minute tier admits through the deep record once agreed with the snapshot / sub-LST chat.
"""
import os

POLICY_ENV = 'EXOPIPE3_ADMITTED_NIGHTS'


def policy():
    """masters_admit.ADMITTED_NIGHTS as the search step passed it (EXOPIPE3_ADMITTED_NIGHTS: deep_step.engine_env); a process
    no step started reads the config. An unknown value refuses (admit.admitted_policy)."""
    from exopipe.phase3.lib.masters import admit as A
    v = os.environ.get(POLICY_ENV)
    return A.admitted_policy(v)


def nights(base, scope, cadence='deep'):
    """({date: admissions.csv row} the search accepts, {date: why} refused) for the cell ``base``: under 'admitted' (deep
    cadence) the nights admitted into the cell's master (admit.admitted_rows) whose every product in subtracted/ carries
    ADMITTED and ADMITREF = the master's CONSTDIG (cut to the card's 32 characters) - a product without them is not the
    admission's, and the night is refused from the search (E15). ({}, {}) otherwise."""
    from astropy.io import fits
    from exopipe.phase3.lib.masters import admit as A
    if cadence != 'deep' or policy() != 'admitted':
        return {}, {}
    rows = A.admitted_rows(base)
    if not rows:
        return {}, {}
    cdg = A.master_identity(base)[0][:32]
    ok, bad = {}, {}
    for d, r in rows.items():
        why = []
        for st in ('I', 'V'):
            p = os.path.join(base, 'subtracted', f'{d}_{st}_sub_{scope}.fits')
            if not os.path.exists(p):
                if st == 'I':
                    why.append('no I product')
                continue
            h = fits.getheader(p)
            if h.get('ADMITTED') is not True or str(h.get('ADMITREF') or '') != cdg:
                why.append(f'{st} product without the admission\'s cards (ADMITTED {h.get("ADMITTED")!r}, ADMITREF '
                           f'{h.get("ADMITREF")!r}, master {cdg!r})')
        if why:
            bad[d] = 'admitted night: ' + '; '.join(why)
        else:
            ok[d] = r
    return ok, bad


def gate_failures(adm):
    """(night-image FAIL dates, epoch FAIL dates) of the admitted nights from their admissions.csv verdicts - E10 / E11's
    rule for the build's records (a FAIL refuses) applied to the record of a night those do not hold. A night with products
    passed both; the gate reads them, so it is applied, not implied."""
    ng = {d for d, r in adm.items() if r.get('night_image') == 'FAIL'}
    eg = {d for d, r in adm.items() if r.get('epoch_gate') == 'FAIL'}
    return ng, eg


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v else None


def values(adm, col):
    """{date: float} of an admissions.csv column over the admitted nights (blank or unreadable: absent - the gate's own
    unmeasured rule applies)."""
    out = {}
    for d, r in adm.items():
        v = _num(r.get(col))
        if v is not None:
            out[d] = v
    return out


def without(dates, adm):
    """``dates`` without the admitted nights (the population a median is taken over)."""
    return [d for d in dates if d not in adm]


def only(resids, keep):
    """``resids`` (date -> residual, or a streaming loader mapping) restricted to the dates ``keep``: the same mapping when
    it holds no other date (production's call is unchanged), else a new mapping of the same kind holding the same entries
    (a streaming mapping keeps its loaders - nothing is loaded here)."""
    keep = set(keep)
    if set(dict.keys(resids)) <= keep:
        return resids
    out = type(resids)()
    for d in dict.keys(resids):
        if d in keep:
            dict.__setitem__(out, d, dict.__getitem__(resids, d))
    return out


def residual_dates(dates, pop, adm):
    """The nights whose residuals the engine builds: ``dates`` (the nights it searches, the same list when no admitted night
    is among them) - and, when an admitted night is, also the cell's searched nights ``pop`` its own sigma stack needs (in
    bulk mode they are among ``dates`` already; an incremental run searches the admitted nights only)."""
    if not any(d in adm for d in dates):
        return dates
    have = set(dates)
    return list(dates) + [d for d in pop if d not in have]


def stacks(dates, pop, adm, resids, build, write, tag=''):
    """{date: sigma stack} for each admitted night searched (in ``dates``, with a residual): ``build`` (the engine's
    _sigma_stack) over the residuals of the cell's searched nights ``pop`` plus the night itself, never another admitted
    night (Gregg 2026-10-14; the result does not depend on the order of admission); each written by ``write(stack, date)``
    and its nights printed (the engine log). A stack the builder cannot make (too few nights) is None, as the cell's own
    would be. {} when no admitted night is searched."""
    out = {}
    base = [p for p in pop if p in resids and p not in adm]
    for a in dates:
        if a in adm and a in resids:
            s = build(only(resids, base + [a]))
            write(s, a)
            out[a] = s
            print(f'  {tag}: admitted night {a}: own sigma stack over {len(base)} of the cell\'s searched nights + itself'
                  f'{"" if s is not None else " - too few nights, none made (as for the cell)"}: {",".join(base)}',
                  flush=True)
    return out
