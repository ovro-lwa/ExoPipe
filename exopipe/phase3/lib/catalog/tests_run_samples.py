"""Content test of the sample-catalog record of a snapshot chain run (cutover row 70; lib/catalog/run_samples.py).

  sample_catalogs   <run>/sample_catalogs.json: the record of the catalogs the run's chain read - every recorded copy in
                    <run>/sample_catalogs/ present with its recorded md5 and size, no unlisted file beside them, the files exactly
                    catalog_snapshot.SAMPLES (the chain's load_samples list)
"""
import os

from exopipe.phase3.core.validate import content_test


@content_test('sample_catalogs')
def sample_catalog_record(path, **ctx):
    """(passed, number of catalogs, detail) of a run's sample_catalogs.json (run_samples.verified)."""
    from exopipe.phase3.lib.catalog import run_samples as RS
    if os.path.basename(path) != RS.RECORD:
        return False, None, '%s is not a %s' % (path, RS.RECORD)
    try:
        _d, rec = RS.verified(os.path.dirname(os.path.abspath(path)), RS.chain_names())
    except RS.SamplesRefusal as e:
        return False, None, str(e)
    return True, len(rec['files']), '%d sample catalogs verified against the record (written by %s %s from %s): %s' % (
        len(rec['files']), rec['how'], rec.get('written_utc'), rec.get('source_dir'),
        ', '.join('%s %s' % kv for kv in RS.md5_map(rec).items()))
