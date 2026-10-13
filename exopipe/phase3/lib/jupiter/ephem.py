"""`exopipe3 jupiter ephem` - the Phase-0 Horizons ephemeris cache of a night (phase0/jup_ephem_fetch.py, in place).

The module's __main__ block is ported here (10 lines): the night's fetch window = the earliest utc_start and latest utc_end of
its Jupiter-window rows (jupiter.WINDOW_CSV), a night whose cache exists is not fetched again ('have'), and the module's
night(date, window) is called unchanged through the in-place driver (module global D set to the --out root only when it
differs from the module literal, so the cache is written to <out>/ephem/). Only a host of the site's NETWORK_HOSTS reaches
JPL. Unit ephem.<date>; output <out>/ephem/jupiter_ephem_<date>.csv [jupiter_ephem].
"""
import os
import sys

from exopipe.phase3.core.record import NoInput, Refusal
from exopipe.phase3.lib.jupiter import common as K
from exopipe.phase3.lib.jupiter import mirrors


def _C():
    from exopipe.phase3.config import CFG
    return CFG


def windows():
    """{date: [utc_start min, utc_end max]} of the window rows (jup_ephem_fetch.py __main__)."""
    win = {}
    for _, r in K.window_rows():
        w = win.setdefault(r['date'], [None, None])
        w[0] = r['utc_start'] if w[0] is None or r['utc_start'] < w[0] else w[0]
        w[1] = r['utc_end'] if w[1] is None or r['utc_end'] > w[1] else w[1]
    return win


def run(args, run_, roots, stages):
    from exopipe.phase3.core import cells
    if not args.dates:
        raise Refusal('give --dates YYYY-MM-DD[,...] (jup_ephem_fetch.py takes the nights to fetch)')
    dates = [cells.parse_date(d.strip()) for d in args.dates.split(',') if d.strip()]
    win = windows()
    outside = [d for d in dates if d not in win]
    if outside:
        raise Refusal(f'no Jupiter-window rows on {", ".join(outside)} in {_C().jupiter.WINDOW_CSV}: the fetch window is '
                      f'defined by them')
    module = mirrors.module_path('phase0:jup_ephem_fetch.py')
    lit_d = mirrors.module_literals(module).get('D', (None,))[0]
    gl = {} if os.path.abspath(roots.out) == os.path.abspath(lit_d or '') else {'D': roots.out}
    hosts = getattr(_C().site, 'NETWORK_HOSTS', None)
    if run_.dry_run:
        print(f'jupiter ephem: {len(dates)} night(s); {roots.describe()}')
        print(f'  engine {module} night(date, window) in place; module globals {gl or "none"}; interpreter '
              f'{K.interpreter_key("ephem")}; network hosts (site NETWORK_HOSTS) {hosts or "UNSET - a real run refuses"}; '
              f'this host {K.host()}')
        for d in dates:
            p = os.path.join(roots.ephem_out, _C().jupiter.EPHEM_NAME.format(date=d))
            print(f'  {d}: window {win[d][0]} .. {win[d][1]} UTC (+-10 min); {p} '
                  f'{"exists - not fetched again (tested)" if os.path.exists(p) else "would be fetched"}')
        return 0
    if not hosts:
        raise Refusal('the site file names no NETWORK_HOSTS (hosts that reach JPL Horizons; today lwacalim05) - add it there')
    if K.host() not in hosts:
        raise Refusal(f'{K.host()} does not reach JPL: run on one of {", ".join(hosts)} (site NETWORK_HOSTS)')
    for d in dates:
        p = os.path.join(roots.ephem_out, _C().jupiter.EPHEM_NAME.format(date=d))
        name = f'ephem.{d}'
        if os.path.exists(p):
            from exopipe.phase3.core import validate
            res = validate.run('jupiter_ephem', p)
            u = run_.unit(name, action='ephem', date=d)
            if all(r['passed'] for r in res):
                u.status, u.reason = 'skipped-valid', f'{p} exists and passes its content test (not fetched again, as the module)'
            else:
                u.status, u.reason = 'refused', f'{p} exists but fails {[r["name"] for r in res if not r["passed"]]}: move it aside'
            u.write()
            print(f'jupiter ephem {d}: {u.status} ({u.reason})', file=sys.stderr)
            continue
        with run_.unit(name, action='ephem', date=d) as u:
            os.makedirs(roots.ephem_out, exist_ok=True)
            res = K.in_place(u, module, 'night', args=[d, win[d]], globals_=gl, python=K.interpreter_key('ephem'),
                             log=os.path.join(roots.logs, f'ephem_{d}.log'), cwd=os.path.join(roots.work or '/tmp', 'ephem'),
                             guard=roots.guard)
            if res.kind == 'refused':
                u.refuse(res.reason)
            if res.kind != 'ok':
                raise RuntimeError(f'EPHEM-FAILED {d}: {res.reason}')
            if not os.path.exists(p):
                raise NoInput(f'night({d}) wrote no {p}')
            u.add_output(p, 'jupiter_ephem')
        print(f'jupiter ephem {d}: {u.status}' + (f' ({u.reason})' if u.reason else ''), file=sys.stderr)
    return None
