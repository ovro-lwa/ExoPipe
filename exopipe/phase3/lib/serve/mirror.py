"""Parity mirror of a served deep run (spec serve.md 4 and 10): the files serve rewrites are COPIED into the parity root,
everything else is SYMLINKED to the production run, so the parity serve reads the same inputs and writes only under --out.

Copied (serve or its in-place scripts rewrite or append them): judgements/ (queues), source_catalogs/ (the union's review_dir
column), source_catalogs_V/, source_catalogs_I/, views/ (the per-Stokes views, rewritten in place as production does, so
stale files behave identically), judgements_V/, judgements_I/ (journals, carry), incremental.log.
Not mirrored: server logs (rewritten by each start). Symlinked: everything else (candidates/, the tier catalogues, master
catalogues, localizations, ...). Known limit: the review server refuses images whose realpath leaves its run directory
(classify_transients.py /img handler), so parity pages carry the same HTML but their panel images answer 403; the parity
comparison is of HTML, queues, views and argv (spec 10).
"""
import os
import shutil

COPY = ('judgements', 'source_catalogs', 'source_catalogs_V', 'source_catalogs_I', 'views', 'judgements_V', 'judgements_I',
        'incremental.log')
SKIP_PREFIXES = ('server_',)


def build(src, dst):
    """(Re)build the mirror ``dst`` of the production run ``src``. Returns {'copied': [...], 'linked': n}."""
    if not os.path.isdir(src):
        raise FileNotFoundError(f'no served run at {src}')
    if os.path.islink(dst):
        os.unlink(dst)
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    copied, linked = [], 0
    for name in sorted(os.listdir(src)):
        s, d = os.path.join(src, name), os.path.join(dst, name)
        if name.startswith(SKIP_PREFIXES) and name.endswith('.log'):
            continue
        if name in COPY:
            if os.path.isdir(s):
                shutil.copytree(s, d, symlinks=True)
            else:
                shutil.copy2(s, d)
            copied.append(name)
        else:
            os.symlink(os.path.realpath(s), d)
            linked += 1
    return {'copied': copied, 'linked': linked}
