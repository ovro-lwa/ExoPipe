#!/usr/bin/env python3
"""serve --tier snapshot, stage copy: the completion test after the rsync (lib/serve/snapshot_copy.copy_stage; exopipe_ops
snapshot_batch/next_wave_20261113/rechain_copy_after_enospc_20261115.sh, OPS-20261115-RECHAIN-COPY-AFTER-DATA07-ENOSPC).

  local      a run copied to an empty review directory: done, G22b (size + mtime) 0 entries, G22c md5 equal for every non-panel
             file and the sampled panels (the sample = SNAPSHOT_COPY_MD5_PNG_SAMPLE, the same files on every call); the excluded
             directories are not copied and not compared
  resume     a copy cut short (half the files, one cut to half its size under its final name) completes on a second call
  tampers    a copied table changed with its size and mtime kept -> G22c refuses (rsync's quick check cannot see it); a file only
             in the copy -> G22b refuses; a sampled panel changed the same way -> G22c refuses
  remote     the same on a source named <this host>:<path> (rsync and md5sum over ssh) - skipped when ssh to this host fails
  P-G25      catalog --tier snapshot's review-disk space before the build: read on the review host and over ssh; unreadable = None
Scratch under $TMPDIR (default /tmp; give a /fast path). Read-only outside it.
"""
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exopipe.phase3.config import CFG                                      # noqa: E402
from exopipe.phase3.core import record                                     # noqa: E402
from exopipe.phase3.lib.serve import snapshot_copy as SC                   # noqa: E402

T = tempfile.mkdtemp(prefix='snapcopy_complete_', dir=os.environ.get('TMPDIR') or '/fast/gh')   # never the root disk's /tmp
N_PNG = 7


def make_source(src):
    os.makedirs(os.path.join(src, 'candidates'))
    os.makedirs(os.path.join(src, 'panels', 'B'))
    os.makedirs(os.path.join(src, 'tmp_panels'))
    os.makedirs(os.path.join(src, 'panels', 'tmp_panel_queue'))
    for i in range(5):
        open(os.path.join(src, 'candidates', f'candidates_46MHz_V_median_{i}.csv'), 'w').write('a,b\n%d,%d\n' % (i, i * i))
    open(os.path.join(src, 'chain_summary.json'), 'w').write('{"x": 1}\n')
    for i in range(20):
        open(os.path.join(src, 'panels', 'B', f'p{i:02d}.png'), 'wb').write(os.urandom(2000 + i))
    open(os.path.join(src, 'tmp_panels', 'junk.png'), 'wb').write(b'x' * 10)
    open(os.path.join(src, 'panels', 'tmp_panel_queue', 'q.json'), 'w').write('{}')
    os.symlink('chain_summary.json', os.path.join(src, 'summary_link.json'))


def stage(source, dst, tag):
    run = record.Run('serve', ['test', tag], os.path.join(T, 'out_' + tag), CFG)
    plan = types.SimpleNamespace(source=source, U=dst, run_name='Run_SNAPZ1_20261116', out=os.path.join(T, 'out_' + tag))
    u = SC.copy_stage(run, plan)
    checks = {c['name']: c for c in getattr(u, 'checks', [])}
    return u, checks


def keep_stat(p, f):
    st = os.stat(p)
    data = open(p, 'rb').read()
    open(p, 'wb').write(f(data))
    assert os.path.getsize(p) == st.st_size
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))


def one_case(source, local_src, label):
    dst = os.path.join(T, label, 'review', 'Run_SNAPZ1_20261116')
    u, ch = stage(source, dst, label + '_clean')
    assert u.status == 'done', (label, u.status, u.reason)
    assert not os.path.exists(os.path.join(dst, 'tmp_panels')) and not os.path.exists(os.path.join(dst, 'panels', 'tmp_panel_queue'))
    files = SC.copy_listing(dst)
    sel, k = SC.md5_selection(files, N_PNG)
    assert k == N_PNG and len(sel) == len(files) - 20 + N_PNG, (len(sel), len(files), k)
    assert len(set(SC.md5_selection(files, N_PNG)[0]) - set(sel)) <= N_PNG, 'only the panel sample differs between calls'
    print(f'    {label}: clean copy done; {len(files)} files listed, {len(sel)} md5-compared ({k} sampled panels)')

    # resume: a copy cut short under the final names (the worst case) completes on a second call
    shutil.rmtree(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    subprocess.run(['rsync', '-a', '--exclude', 'tmp_panels', '--exclude', 'tmp_panel_queue', '--exclude', 'panels',
                    local_src + '/', dst + '/'], check=True)
    c0 = os.path.join(dst, 'candidates', 'candidates_46MHz_V_median_0.csv')
    open(c0, 'wb').write(open(c0, 'rb').read()[:3])
    u, ch = stage(source, dst, label + '_resume')
    assert u.status == 'done', (label, 'resume', u.status, u.reason)
    print(f'    {label}: a copy cut short completes on a second call')

    # tampers
    sampled_png = 'panels/B/p00.png'
    for what, edit, gate in (
            ('a table changed, size and mtime kept', lambda: keep_stat(c0, lambda d: d[:-2] + b'9\n'), 'G22c'),
            ('every panel changed, size and mtime kept (a random sample catches it)',
             lambda: [keep_stat(os.path.join(dst, 'panels', 'B', f), lambda d: bytes([d[0] ^ 1]) + d[1:])
                      for f in os.listdir(os.path.join(dst, 'panels', 'B'))], 'G22c'),
            ('a file only in the copy', lambda: open(os.path.join(dst, 'candidates', 'stray.csv'), 'w').write('x\n'), 'G22b')):
        edit()
        u, ch = stage(source, dst, label + '_t')
        assert u.status == 'refused' and gate in (u.reason or ''), (label, what, u.status, u.reason)
        print(f'    {label}: {what} -> refused ({u.reason[:110]})')
        shutil.rmtree(dst)
        assert stage(source, dst, label + '_re')[0].status == 'done'


def review_space():
    """catalog --tier snapshot P-G25: the review disk's free bytes read on the review host (statvfs) or over ssh (df); an
    unreadable root reads None (the gate then refuses a production chain)"""
    from exopipe.phase3.lib.catalog import tier_snapshot as TSN
    saved = (CFG.paths.SNAPSHOT_REVIEW_ROOT, getattr(CFG.site, 'REVIEW_HOST', None))
    try:
        CFG.paths.SNAPSHOT_REVIEW_ROOT, CFG.site.REVIEW_HOST = T, socket.gethostname()
        st = os.statvfs(T)
        local = TSN.review_free_bytes()
        assert local is not None and abs(local - st.f_bavail * st.f_frsize) < 2e9, local
        other = getattr(CFG.site, 'CHAIN_HOST', None)       # another node of the site: read over ssh (/fast exists on each)
        r = subprocess.run(['ssh', '-n', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', str(other), 'df -B1 --output=avail /fast'],
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) if other else None
        if r is not None and r.returncode == 0 and other.split('.')[0] != socket.gethostname().split('.')[0]:
            CFG.site.REVIEW_HOST, CFG.paths.SNAPSHOT_REVIEW_ROOT = other, '/fast'
            remote = TSN.review_free_bytes()
            assert remote is not None and abs(remote - int(r.stdout.split()[-1])) < 2e9, (remote, r.stdout)
            CFG.paths.SNAPSHOT_REVIEW_ROOT = os.path.join(T, 'no_such_dir')
            assert TSN.review_free_bytes() is None
            print(f'    P-G25 review space: {local / 1e9:.0f} GB read locally, {remote / 1e9:.0f} GB over ssh on {other}; an '
                  f'unreadable root reads None')
        else:
            print(f'    P-G25 review space: {local / 1e9:.0f} GB read locally; over ssh SKIP (no other CHAIN_HOST reachable)')
        CFG.paths.SNAPSHOT_REVIEW_ROOT, CFG.site.REVIEW_HOST = os.path.join(T, 'no_such_dir'), socket.gethostname()
        assert TSN.review_free_bytes() is None
    finally:
        CFG.paths.SNAPSHOT_REVIEW_ROOT, CFG.site.REVIEW_HOST = saved


def resume_space(src):
    """G20 on a resume counts only what is still missing: with free space below the full size but above the rest, it passes"""
    dst = os.path.join(T, 'g20', 'Run_SNAPZ1_20261116')
    os.makedirs(os.path.dirname(dst))
    shutil.copytree(src, dst, symlinks=True, ignore=shutil.ignore_patterns('tmp_panels', 'tmp_panel_queue'))
    full = SC.source_bytes(None, src)
    st = os.statvfs(os.path.dirname(dst))
    free = st.f_bavail * st.f_frsize
    saved = CFG.serve.SNAPSHOT_COPY_MIN_FREE_FACTOR
    try:
        CFG.serve.SNAPSHOT_COPY_MIN_FREE_FACTOR = 1.5 * free / full        # the whole run would not fit, the rest (0 B) does
        u, ch = stage(src, dst, 'g20_resume')
        assert u.status == 'done', (u.status, u.reason)
        shutil.rmtree(dst)
        u, ch = stage(src, dst, 'g20_fresh')
        assert u.status == 'refused' and 'G20' in u.reason, (u.status, u.reason)
    finally:
        CFG.serve.SNAPSHOT_COPY_MIN_FREE_FACTOR = saved
    print('    G20 on a resume: the copy already in place counts (passes); the same space for a fresh copy refuses')


def main():
    saved = CFG.serve.SNAPSHOT_COPY_MD5_PNG_SAMPLE
    try:
        review_space()
        CFG.serve.SNAPSHOT_COPY_MD5_PNG_SAMPLE = N_PNG
        src = os.path.join(T, 'src', 'Run_SNAPZ1_20261116')
        make_source(src)
        one_case(src, src, 'local')
        resume_space(src)
        me = socket.gethostname().split('.')[0]
        p = subprocess.run(['ssh', '-n', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', me, 'true'], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
        if p.returncode == 0:
            one_case(f'{me}:{src}', src, 'remote')
        else:
            print(f'    remote: SKIP (ssh to {me} failed)')
    finally:
        CFG.serve.SNAPSHOT_COPY_MD5_PNG_SAMPLE = saved
        shutil.rmtree(T, ignore_errors=True)
    print('PASS test_snapshot_copy_complete')
    return 0


if __name__ == '__main__':
    sys.exit(main())
