"""`status eligibility` / `status batch-list` (tier snapshot) - ops/snapshot_batch_audit.py and ops/snapshot_batch_lists.py
run IN PLACE in a child (core.legacy.run, site PYTHON['py36'] = the development Python 3.6 production uses for them:
snapshot_batch_audit.py:31, snapshot_batch_lists.py:21) with their own output arguments.

eligibility  argv ['--out', <out>/snapshot/eligibility.csv, '--cells-file', ...][--exclude-file F][--no-store]. The
             function `catalog --tier snapshot` (P-G6) gates on is the same script; status only reports.
batch-list   argv ['--out', LIST, '--base', BASE, '--order', ORDER, '--max-new', ..., ...] (values from config
             status_snapshot.MAX_NEW ... NEW_LOADED = the script's defaults :70-76). In the window --base is
             required (S-G8: without it the script sshes to the review host to read the served run's cells).
"""
import os

from exopipe.phase3.lib.status import steps_common as K


def run_eligibility(out, cells_files, exclude_file, no_store, unit, log):
    from exopipe.phase3.core import legacy
    from exopipe.phase3.lib.status import steps_inplace as IP
    C = K.cfg()
    dest = K.check_write(K.place(out, 'snapshot', C.status_snapshot.ELIGIBILITY_TABLE))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    argv = ['--out', dest]
    if cells_files:
        argv += ['--cells-file'] + list(cells_files)
    if exclude_file:
        argv += ['--exclude-file', exclude_file]
    if no_store:
        argv += ['--no-store']
    res = legacy.run(IP.legacy_script('snapshot_batch_audit.py'), argv, python='py36', log=log, unit=unit)
    return res, dest


def run_batch_list(list_out, args, unit, log):
    from exopipe.phase3.core import legacy
    from exopipe.phase3.lib.status import steps_inplace as IP
    C = K.cfg()
    K.check_write(list_out)
    S = C.status_snapshot
    argv = ['--out', list_out, '--base', args.base, '--order', args.order or K.expand(S.ORDER_LIST),
            '--max-new', S.MAX_NEW, '--max-gate-gb', S.MAX_GATE_GB, '--lc-cache', S.LC_CACHE,
            '--base-detections', S.BASE_DETECTIONS, '--base-loaded', S.BASE_LOADED,
            '--new-detections', S.NEW_DETECTIONS, '--new-loaded', S.NEW_LOADED]
    if args.all:
        argv += ['--all']
    unit.meta['batch_list_argv'] = [str(a) for a in argv]
    res = legacy.run(IP.legacy_script('snapshot_batch_lists.py'), argv, python='py36', log=log, unit=unit)
    return res
