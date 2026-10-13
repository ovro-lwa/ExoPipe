"""Review hub (registry + page, port 8765) and inspection hub (page from inspection_greenlit.csv, port 8795).

Production (paths.PRODUCTION_WRITES and a production --out): build_review_hub.py --set ... in place for every served page,
then the removal of retired hub tiers (port of sublst_cumulative.sh:145-150, the hand edit of registry.json) and one
no-argument rebuild - all under ONE lock on <HUB_DIR>/.registry.lock (edge case E-33: two tiers serving at once lost hub
updates; the in-place script itself does not lock, so only exopipe3 writers are serialised).
Parity: build_review_hub.py hard-codes /data07/gh/phase3/transients/review_hub (build_review_hub.py:12), so the parity hub
is PLANNED: <out>/review_hub/registry.json = the production registry with this run's --set entries applied by the same
rule (port of build_review_hub.py:16-21) and the retired tiers removed; index.html is not rendered (spec serve.md 12.5).
Inspection hub: build_inspection_hub.py <dir> in place (it takes its output directory), production or parity.
Hub HTTP servers: started only when the port does not answer (production only; run_inspection_hub.sh:7; the 8765 hub has
no launcher, interface divergence D21).
"""
import contextlib
import fcntl
import json
import os

from exopipe.phase3.core.record import Refusal
from exopipe.phase3.lib.serve import procs, scripts


def _S():
    from exopipe.phase3.config import CFG
    return CFG.serve


def apply_set(reg, S_, tier, kvs):
    """build_review_hub.py --set S tier k=v ... (build_review_hub.py:16-21)."""
    e = reg[S_].setdefault(tier, {})
    for kv in kvs:
        k, v = kv.split('=', 1)
        e[k] = (v == '1') if k == 'live' else (int(v) if k == 'port' else v)
    return reg


def remove_tiers(reg, tiers):
    for S_ in 'VI':
        for t in tiers:
            reg[S_].pop(t, None)
    return reg


def _dump(reg, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + '.part', 'w') as f:
        json.dump(reg, f, indent=1)
    os.replace(path + '.part', path)


@contextlib.contextmanager
def registry_lock(hub_dir):
    os.makedirs(hub_dir, exist_ok=True)
    fd = os.open(os.path.join(hub_dir, '.registry.lock'), os.O_RDWR | os.O_CREAT, 0o664)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def tunnel_extra_with(reg, ports):
    """The registry's tunnel_extra list with ``ports`` added (build_review_hub.py --tunnel-extra REPLACES the list,
    build_review_hub.py:24-26, so the union is passed), or None when every port is already forwarded."""
    have = set(reg.get('tunnel_extra', []))
    buttons = {e['port'] for S_ in 'VI' for e in reg.get(S_, {}).values()} | {reg.get('round2', {}).get('port')}
    need = {int(p) for p in ports} - have - buttons
    return sorted(have | need) if need else None


def update(plan, entries, remove=(), tunnel=(), log_path=None, unit=None):
    """Apply hub entries [(Stokes, hub tier, [k=v ...])], remove ``remove`` tiers and make sure the button-less ports
    ``tunnel`` are forwarded (registry tunnel_extra). Returns (registry path, index.html path or None, planned argv lists)."""
    S = _S()
    planned = [['--set', s, t] + list(kvs) for s, t, kvs in entries]
    if plan.parity:
        src = os.path.join(S.HUB_DIR, 'registry.json')
        if not os.path.exists(src):
            raise Refusal(f'hub: no production registry at {src} to plan the parity hub from')
        reg = json.load(open(src))
        for s, t, kvs in entries:
            apply_set(reg, s, t, kvs)
        remove_tiers(reg, remove)
        te = tunnel_extra_with(reg, tunnel)
        if te is not None:
            reg['tunnel_extra'] = te
            planned.append(['--tunnel-extra', ','.join(map(str, te))])
        out = os.path.join(plan.hub_dir, 'registry.json')
        _dump(reg, out)
        return out, None, planned
    with registry_lock(plan.hub_dir):
        R = os.path.join(plan.hub_dir, 'registry.json')
        te = tunnel_extra_with(json.load(open(R)), tunnel)
        if te is not None:
            planned.append(['--tunnel-extra', ','.join(map(str, te))])
        for argv in planned:
            res = scripts.run('build_review_hub', argv, log=log_path, unit=unit)
            if not res.ok:
                raise Refusal(f'hub: build_review_hub.py {" ".join(argv[:3])} failed (rc {res.rc}): '
                              f'{scripts.last_line(res.tail)}')
        if remove:
            reg = json.load(open(R))
            _dump(remove_tiers(reg, remove), R)
        res = scripts.run('build_review_hub', [], log=log_path, unit=unit)
        if not res.ok:
            raise Refusal(f'hub: build_review_hub.py rebuild failed (rc {res.rc}): {scripts.last_line(res.tail)}')
    return os.path.join(plan.hub_dir, 'registry.json'), os.path.join(plan.hub_dir, 'index.html'), planned


def inspection_hub(plan, log_path=None, unit=None):
    """build_inspection_hub.py <dir> (run_inspection_hub.sh:5). Returns the index.html path."""
    res = scripts.run('build_inspection_hub', [plan.inspection_hub_dir], log=log_path, unit=unit)
    if not res.ok:
        raise Refusal(f'inspection hub: build_inspection_hub.py failed (rc {res.rc}): {scripts.last_line(res.tail)}')
    return os.path.join(plan.inspection_hub_dir, 'index.html')


def ensure_http_servers(plan):
    """Production only: start `python -m http.server PORT --bind 127.0.0.1` in the hub directories whose port does not
    answer (run_inspection_hub.sh:7). Returns notes."""
    S = _S()
    notes = []
    if plan.parity:
        return ['hub HTTP servers: not started in a parity run (production 8765 / 8795 serve the production hubs)']
    for port, d in ((S.HUB_PORT, plan.hub_dir), (S.INSPECTION_HUB_PORT, plan.inspection_hub_dir)):
        st, _ = procs.http_get(port, timeout=5.0)
        if st == 200:
            notes.append(f'hub HTTP {port}: answering (left as is)')
            continue
        if procs.port_in_use(port):
            raise Refusal(f'G13 hub port {port} is held but does not answer 200 - not replaced')
        p = procs.start([scripts.interpreter('default'), '-m', 'http.server', str(port), '--bind', '127.0.0.1'], d,
                        scripts.server_env({}), os.path.join(d, 'server.log'), append=True)
        ok, detail = procs.wait_ready(port, p, timeout=60, poll=2)
        if not ok:
            raise Refusal(f'G16 hub HTTP {port}: {detail}')
        notes.append(f'hub HTTP {port}: started pid {p.pid} in {d}')
    return notes
