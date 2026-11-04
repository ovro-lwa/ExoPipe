"""THE PRODUCTION PIN of the port tests (V_combview review F3, 2026-11-04).

The tests of a PORT compare the refactor's code with production's code it ports. Production keeps moving (exopipe_ops dde1760,
dbc40a6, edf3263, 35aad1a on 2026-11-04 alone), and a test that runs production's LIVE file fails on every production commit:
test_snapshot_view's production side refused the synthetic world after dbc40a6, so 8 of its 9 tests failed, the stage's own gates
and content tests among them, for production's reason. So the PORTED production files are read at ONE pinned exopipe_ops commit,
each by its git blob sha (a commit and its blobs never change), and materialized into the test's scratch:

  materialize(dest)  writes every pinned file under dest (same relative paths), each checked against its pinned blob sha, and
                     returns dest; a test puts dest first on sys.path / reads the files there
  drift()            [(path, pinned blob, live blob, commits since the pin touching it)] of the pinned files production has
                     changed since (its working tree, committed or not) - test_production_pin in each test file turns this into
                     ONE explicit failure that names what to port, instead of cascading failures

What is pinned: production code the refactor PORTS - the combined view's builder, the deep-omission decisions and drop, the
10-minute merge the refactor's test runs, the dated driver whose heredoc is ported. What is NOT pinned: production code the refactor
RUNS IN PLACE (the clv15 snapshot chain and its merge, snapshot_filter_catalog.py, snapshot_localize_tiers.sh, the localizer and the
astrometry figure): the stage and the tests' production side run the same live file, so a parity compares like with like.

Moving the pin is a port: the round that ports production's later commits moves PIN_COMMIT and PIN_BLOBS with it.
"""
import hashlib
import os
import subprocess

OPS_REPO = '/lustre/gh/main/exopipe_ops'
# exopipe_ops c7ca600 (2026-11-04 09:10, dde1760^): production as the V_combview port and the deep-omission port reproduce it -
# the combined view of 5b8b8ff (2026-10-31), the snapshot decisions of 37546f6, the 10-minute decisions and drop of 0d4f1a9 / c6d2c9e
PIN_COMMIT = 'c7ca600a9263a41d874f680433d7b50e83215584'
PIN_BLOBS = {
    'snapshot_combined_view.py': 'f218d72ac4c947c869ffaa44db0fbe98d9605474',        # = 5b8b8ff's
    'snapshot_deep_omission.py': '18b5d06f019ff0e4b3b9c99ce52c1cab51b7f9d0',
    'sublst_deep_omission.py': '64ee51c196a064950fa1c509d74a0ab484de3491',
    'sublst_deep_omission_drop.py': 'bdfddc9affbcb3ae4025267919bd13718591894f',
    'sublst_cumulative.py': '948e9cf7dd95ad9a8421a74bdf8e8af22e008695',
    'snapshot_batch/comb_build_20261102.sh': 'c68c54847628076892dbeac4775e739f68e4e807',
}


def blob_sha(data):
    """git's blob id of ``data`` (bytes): sha1 of 'blob <len>\\0' + data."""
    h = hashlib.sha1(b'blob %d\x00' % len(data))
    h.update(data)
    return h.hexdigest()


def _git(*args):
    p = subprocess.run(['git', '-C', OPS_REPO] + list(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode:
        raise RuntimeError(f'git {" ".join(args)} in {OPS_REPO}: {p.stderr.decode("utf-8", "replace").strip()[:300]}')
    return p.stdout


def materialize(dest, files=None):
    """Write the pinned files (all, or ``files``) under ``dest``, each verified against its pinned blob sha; returns ``dest``.
    The pinned commit must name the same blob (the pin is one commit, the blob sha its double check)."""
    for path in (files or sorted(PIN_BLOBS)):
        want = PIN_BLOBS[path]
        at_pin = _git('rev-parse', f'{PIN_COMMIT}:{path}').decode().strip()
        if at_pin != want:
            raise RuntimeError(f'production pin: {path} at {PIN_COMMIT[:12]} is blob {at_pin[:12]}, not the pinned {want[:12]}')
        data = _git('cat-file', 'blob', want)
        if blob_sha(data) != want:
            raise RuntimeError(f'production pin: {path} read back as {blob_sha(data)[:12]}, not {want[:12]}')
        p = os.path.join(dest, path)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, 'wb') as f:
            f.write(data)
    return dest


def drift(files=None):
    """[(path, pinned blob, live blob, [commits since the pin touching it])] of the pinned files that production's working tree
    holds in another version now (committed or not)."""
    out = []
    for path in (files or sorted(PIN_BLOBS)):
        live_p = os.path.join(OPS_REPO, path)
        live = blob_sha(open(live_p, 'rb').read()) if os.path.exists(live_p) else 'absent'
        if live != PIN_BLOBS[path]:
            log = _git('log', '--format=%h %cs %s', f'{PIN_COMMIT}..HEAD', '--', path).decode().splitlines()
            out.append((path, PIN_BLOBS[path], live, [ln[:110] for ln in log] or ['(uncommitted in the working tree)']))
    return out


def drift_message(rows):
    return ('production moved since the pin %s: ' % PIN_COMMIT[:7] + '; '.join(
        f'{p} (pinned {pin[:7]}, live {live[:7]}: {", ".join(c.split()[0] for c in commits)})' for p, pin, live, commits in rows)
        + ' - port these commits, then move tests/production_pin.py PIN_COMMIT / PIN_BLOBS with the port')
