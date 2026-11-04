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
re-judging module (snapshot_rejudge.py), the 10-minute merge the refactor's test runs, the dated drivers whose heredoc is ported. What is NOT pinned: production code the refactor
RUNS IN PLACE (the clv15 snapshot chain and its merge, snapshot_filter_catalog.py, snapshot_localize_tiers.sh, the localizer and the
astrometry figure): the stage and the tests' production side run the same live file, so a parity compares like with like.

Moving the pin is a port: the round that ports production's later commits moves PIN_COMMIT and PIN_BLOBS with it.
"""
import hashlib
import os
import subprocess

OPS_REPO = '/lustre/gh/main/exopipe_ops'
# exopipe_ops fa90670 (2026-11-04 15:38, production HEAD when the V2 follow-up round started): production as the follow-up ports it -
# the combined view of dbc40a6 (Card 2 per frame, re-judging, the re-point fix, the rate population; dde1760's per-image wideband
# omission), the snapshot decisions of dbc40a6 / dde1760, the 10-minute decisions of dde1760 (gate pairing), the 10-minute drop of
# 35aad1a (edf3263's re-judging + the approval marker), snapshot_rejudge.py (dbc40a6), the dated drivers 20261102 / 20261104 (their
# step-2 heredocs identical). Moved from c7ca600 (dde1760^) by the V2 follow-up round (2026-11-04).
PIN_COMMIT = 'fa9067019528da47e1174c16a3a647054ec41bc2'
PIN_BLOBS = {
    'snapshot_combined_view.py': 'afc2bb3ce64e1210a0d87fae99d35e46094d6245',        # dbc40a6
    'snapshot_deep_omission.py': 'b55e6d76d64ed2d289834edce963119a2e48fdb1',        # dbc40a6
    'snapshot_rejudge.py': '61c44e4496d1a4691b4b1173aa414b026dd17229',              # dbc40a6 (new)
    'sublst_deep_omission.py': 'a031a978422129f40fc83285169ccde89ec3f2f9',          # dde1760
    'sublst_deep_omission_drop.py': '6d12fa03ebe1684ae35790b7879407d81c4323fe',     # 35aad1a
    'sublst_cumulative.py': '948e9cf7dd95ad9a8421a74bdf8e8af22e008695',             # 0d4f1a9 (unchanged)
    'snapshot_batch/comb_build_20261102.sh': 'c68c54847628076892dbeac4775e739f68e4e807',
    'snapshot_batch/comb_build_20261104.sh': '1be26d8d82c916601edbd9c6743ac350ce93d145',   # dbc40a6
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
