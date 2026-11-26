"""Content test of the sample-catalog record of a snapshot chain run (cutover row 70; lib/catalog/run_samples.py).

  sample_catalogs   <run>/sample_catalogs/sample_catalogs.json, or the run directory itself (a directory holding
                    chain_summary.json): the record of the catalogs the run's chain read - every recorded copy beside it present
                    with its recorded md5 and size, no unlisted file beside them, the files exactly catalog_snapshot.SAMPLES (the
                    chain's load_samples list). Only that name is a record (a staging never is). Given run directories, it is the
                    backfill's worklist: `exopipe3 verify sample_catalogs --glob '<root>/Run_SNAP*/' --emit-failing FILE`.
"""
import os

from exopipe.phase3.core.validate import content_test


@content_test('sample_catalogs')
def sample_catalog_record(path, **ctx):
    """(passed, number of catalogs, detail) of a run's record (run_samples.verified)."""
    from exopipe.phase3.lib.catalog import run_samples as RS
    p = os.path.abspath(path)
    if os.path.isdir(p):
        run = p
        if not os.path.isfile(os.path.join(run, 'chain_summary.json')):
            return False, None, '%s has no chain_summary.json: not a snapshot chain run' % run
    elif os.path.basename(p) == RS.RECORD and os.path.basename(os.path.dirname(p)) == RS.COPY_DIR:
        run = os.path.dirname(os.path.dirname(p))
    else:
        return False, None, '%s is neither a run directory nor a run\'s %s/%s (a staging is never a record)' % (
            path, RS.COPY_DIR, RS.RECORD)
    try:
        _d, rec = RS.verified(run, RS.chain_names())
    except RS.SamplesRefusal as e:
        return False, None, str(e)
    return True, len(rec['files']), '%d sample catalogs verified against the record (written by %s %s from %s): %s' % (
        len(rec['files']), rec['how'], rec.get('written_utc'), rec.get('source_dir'),
        ', '.join('%s %s' % kv for kv in RS.md5_map(rec).items()))
