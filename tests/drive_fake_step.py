#!/usr/bin/env python3
"""A stand-in pipeline step for tests/test_drive.py and the driver's cluster content test: no pipeline data, only unit records.

    python drive_fake_step.py NAME --epochs EPOCH --out DIR [--units N] [--sleep S] [--rc K] [--fail-once FLAG]
                              [--fail-always] [--resume] [--record PATH] [anything else: ignored]

Each unit u of the hour writes DIR/_records/NAME/EPOCH_u<u>.json ('running', then 'done') and appends 'start' and 'done' lines
to DIR/ran.log ("NAME EPOCH u<u> start|done|skip pid"). With --resume a unit whose record is 'done' is skipped ('skip' line), as
the steps' --resume skips finished units (core/record.py). SIGTERM: the unit's record becomes 'interrupted' and the step exits
143 (cli: Terminated). --rc K: exit K after the units (2 refused, 3 no-input); --fail-always: exit 1 (or K) after the units;
--fail-once FLAG: exit 1 (or K) after the first unit unless FLAG exists (it is created), and otherwise 0. NAME 'fin' is a
closing step: it copies the chain record's status (--record, the runner's {record}) to DIR/fin_EPOCH.json.
"""
import argparse
import json
import os
import signal
import sys
import time


def _w(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f'{path}.tmp.{os.getpid()}'
    with open(tmp, 'w') as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def _log(out, text):
    with open(os.path.join(out, 'ran.log'), 'a') as f:
        f.write(f'{text} {os.getpid()}\n')


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('name')
    ap.add_argument('--epochs', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--units', type=int, default=2)
    ap.add_argument('--sleep', type=float, default=0.0)
    ap.add_argument('--rc', type=int, default=0)
    ap.add_argument('--fail-once')
    ap.add_argument('--fail-always', action='store_true')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--record')
    a, _ = ap.parse_known_args(argv)
    os.makedirs(a.out, exist_ok=True)
    if a.name == 'fin':
        rec = json.load(open(a.record))
        _w(os.path.join(a.out, f'fin_{a.epochs}.json'), {'chain_status': rec.get('chain_status'), 'status': rec.get('status'),
                                                          'steps': [s.get('status') for s in rec.get('steps') or []]})
        _log(a.out, f'fin {a.epochs} done')
        return 0
    cur = {}

    def term(signum, frame):
        if cur:
            _w(cur['path'], {'status': 'interrupted'})
            _log(a.out, f'{cur["tag"]} interrupted')
        sys.exit(143)
    signal.signal(signal.SIGTERM, term)
    once = bool(a.fail_once) and not os.path.exists(a.fail_once)
    for u in range(a.units):
        tag = f'{a.name} {a.epochs} u{u}'
        path = os.path.join(a.out, '_records', a.name, f'{a.epochs}_u{u}.json')
        if a.resume and os.path.exists(path) and json.load(open(path)).get('status') == 'done':
            _log(a.out, f'{tag} skip')
            continue
        cur.update(path=path, tag=tag)
        _w(path, {'status': 'running'})
        _log(a.out, f'{tag} start')
        time.sleep(a.sleep)
        _w(path, {'status': 'done'})
        _log(a.out, f'{tag} done')
        cur.clear()
        if u == 0 and once:
            open(a.fail_once, 'w').close()
            _log(a.out, f'{tag} fail')
            return a.rc or 1
    if a.fail_always:
        return a.rc or 1
    return 0 if a.fail_once else a.rc


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
