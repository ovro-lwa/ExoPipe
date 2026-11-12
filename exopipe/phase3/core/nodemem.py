"""The memory a job on this node can still count on.

Standard library only and import-safe: the snapshot cell driver (lib/snapshot_engine/cell.py), which runs as a script
outside the package, loads this file by its path; lib/masters/tier_sublst.py imports it. One reading for every memory
budget (2026-11-12): /proc/meminfo's MemAvailable, bounded by the memory limit of the cgroups the process runs in.
"""
import os


def cgroup_available_gb(proc_cgroup='/proc/self/cgroup', root='/sys/fs/cgroup'):
    """What this process may still take (GB = 2**30 bytes) under the memory limits of the cgroups it runs in (cgroup v2),
    or None when no enclosing cgroup sets a limit or nothing can be read (cgroup v1, no /sys). For every cgroup from the
    process's own up to the root that holds a number in memory.max: memory.max - memory.current + the file cache charged
    to it that can be dropped (inactive_file + active_file of memory.stat; shared memory is in neither); the smallest of
    those levels. A level's limit is memory.max or, when it is set and lower, memory.high (above it the kernel throttles
    the cgroup into reclaim instead of killing). A limited level whose usage cannot be read gives no number (it is skipped).
    Why (2026-11-12): the cluster's nodes cap all user sessions together - lwacalim00: /user.slice memory.max = 400 GiB on
    a node of 503 GiB. MemAvailable does not know that cap, and a job that crosses it is killed while the node still shows
    memory available: six high-band units (6 x 51 GB) beside 100 GB of other user jobs would cross it."""
    try:
        with open(proc_cgroup) as f:
            rel = next((line.strip().split('::', 1)[1] for line in f if line.startswith('0::')), None)
    except (OSError, IndexError):
        return None
    if rel is None:
        return None
    root = os.path.normpath(root)
    d = os.path.normpath(os.path.join(root, rel.lstrip('/')))
    best = None
    while d != root and d.startswith(root + os.sep):
        try:
            limits = []
            for name in ('memory.max', 'memory.high'):
                try:
                    with open(os.path.join(d, name)) as f:
                        v = f.read().strip()
                except OSError:
                    if name == 'memory.max':
                        raise
                    continue
                if v != 'max':
                    limits.append(int(v))
            if limits:
                limit = min(limits)
                with open(os.path.join(d, 'memory.current')) as f:
                    current = int(f.read().strip())
                cache = 0
                with open(os.path.join(d, 'memory.stat')) as f:
                    for line in f:
                        k, _, v = line.partition(' ')
                        if k in ('inactive_file', 'active_file'):
                            cache += int(v)
                left = max(0, limit - current + cache) / 2 ** 30
                best = left if best is None else min(best, left)
        except (OSError, ValueError):
            pass                                                               # no memory controller at this level
        d = os.path.dirname(d)
    return best


def mem_available_gb(meminfo='/proc/meminfo'):
    """The memory this process can still count on, in whole GB (0 for less than 1 GB): this node's MemAvailable, or what
    is left under the memory limit of its cgroups (cgroup_available_gb) when that is less. None when MemAvailable cannot
    be read."""
    m = None
    try:
        with open(meminfo) as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    m = int(line.split()[1]) // 2 ** 20
                    break
    except (OSError, ValueError, IndexError):
        m = None
    if m is None:
        return None
    c = cgroup_available_gb()
    return m if c is None else min(m, int(c))
