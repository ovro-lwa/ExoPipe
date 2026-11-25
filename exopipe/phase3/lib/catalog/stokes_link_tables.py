"""The cross-Stokes link's tables: their names, columns and the ONE reader of each (lib/catalog/stokes_link.py writes them).

Every reader of a link directory - the catalog stages (stokes_link.py re-exports these names), the transient list, the
two-port server's link banner and label carrying (lib/serve/stokes_banner.py, lib/serve/label_identity.py) and the
localization wrap (lib/catalog/wraps/source_localization.py, which loads this file by its path: a localization child's
`exopipe` can be production's package) - reads through the functions here, so the tables have one set of names and one
reading. Standard library only, and nothing imported from the branch, for that reason.

LAYOUT (in the grouping directory, or in a served run that holds its own copy)
  stokes_link                 a symbolic link to the current version, switched atomically (stokes_link.py switch_to):
                              a reader that opens stokes_link/<file> gets one complete version, never none
  stokes_link.versions/<vid>/ the versions (vid: UTC to the microsecond and the writer's pid); the current one and the
                              KEEP_SUPERSEDED newest before it are kept, older ones deleted (intermediates)
  in each version             stokes_links.csv (LINK_COLS), stokes_link_pairs.csv (PAIR_COLS), identity_merges.csv
                              (MERGE_COLS), summary.json (FORMAT, the two grouping tables and their sha1, the counts with the
                              test's p_threshold and u_cap)
"""
import collections
import csv
import json
import os

DIRNAME = 'stokes_link'
VERSIONS = 'stokes_link.versions'
KEEP_SUPERSEDED = 2
LINKS = 'stokes_links.csv'
PAIRS = 'stokes_link_pairs.csv'
MERGES = 'identity_merges.csv'
SUMMARY = 'summary.json'
FORMAT = 'exopipe3-stokes-link-2'
LINK_COLS = ('link_id', 'link_name', 'link_size', 'link_ambiguous', 'stokes', 'identity_id', 'identity_name', 'linked_ids',
             'linked_names', 'best_p', 'best_u', 'best_offset_arcsec', 'best_offset_pct', 'best_p_chance', 'mean_ra', 'mean_dec',
             'pos_err_arcsec', 'n_det')
PAIR_COLS = ('link_id', 'link_ambiguous', 'I_identity_id', 'I_identity_name', 'V_identity_id', 'V_identity_name', 'offset_arcsec',
             'u', 'p', 'offset_pct', 'p_chance', 'rho_I_per_deg2', 'rho_V_per_deg2', 'I_pos_err_arcsec', 'V_pos_err_arcsec')
# into_status: the status in the grouping state of the ID the 'into' chain ends at - 'live' (a label made on the merged
# identity follows the merge to it) or 'retired' (valid history: the survivor itself left the grouping later; a label made on
# the merged identity has no page to land on and is refused with HTTP 409, lib/serve/label_identity.py)
MERGE_COLS = ('stokes', 'identity_id', 'into_id', 'into_status', 'run')
INTO_STATUS = ('live', 'retired')
LIST_COLS = ('stokes_link_id', 'stokes_link_name', 'stokes_link_ambiguous', 'stokes_linked_names')
LOC_ENV = 'P3_LOC_LINKS'          # the localization's link table (lib/catalog/wraps/source_localization.py link_detections)


class TableRefused(RuntimeError):
    """a link table that is missing or not in this module's format (stokes_link.LinkRefused derives from it)"""


def link_dir(out):
    return os.path.join(out, DIRNAME)


def dir_of(out_or_dir):
    """the stokes_link directory of a grouping or served-run directory; a stokes_link directory or one of its versions
    (stokes_link.versions/<vid>) itself"""
    n = os.path.normpath(out_or_dir)
    if os.path.basename(n) == DIRNAME or os.path.basename(os.path.dirname(n)) == VERSIONS:
        return out_or_dir
    return link_dir(out_or_dir)


def read_rows(path, cols=None):
    """[row dicts] of a link table; cols: the columns it must hold (a table without one REFUSES)"""
    if not os.path.exists(path):
        raise TableRefused(f'stokes link table {path} is missing - REFUSING')
    with open(path, newline='') as f:
        rd = csv.DictReader(f)
        rows = list(rd)
        head = rd.fieldnames or []
    miss = [c for c in (cols or ()) if c not in head]
    if miss:
        raise TableRefused(f'stokes link table {path} lacks the column(s) {miss} (another format; re-run the link) - REFUSING')
    return rows


def read_summary(out_or_dir):
    p = os.path.join(dir_of(out_or_dir), SUMMARY)
    if not os.path.exists(p):
        raise TableRefused(f'stokes link summary {p} is missing - REFUSING')
    with open(p) as f:
        return json.load(f)


def link_test_values(summ):
    """(P threshold, u cap) the link was computed with (its summary's counts: config catalog_grouping P_IDENTITY, U_CAP)"""
    c = summ.get('counts') or {}
    return float(c['p_threshold']), float(c['u_cap'])


def read_links(out_or_dir):
    """{(stokes, identity_id): link row} of a stokes_link directory (or the directory holding it); a missing table REFUSES
    (a reader that merges on the link must not silently fall back to per-Stokes sources)"""
    return {(r['stokes'], r['identity_id']): r for r in read_rows(os.path.join(dir_of(out_or_dir), LINKS), LINK_COLS)}


def read_pairs(out_or_dir):
    return read_rows(os.path.join(dir_of(out_or_dir), PAIRS), PAIR_COLS)


def members(rows):
    """{link_id: [its rows]} of link rows (the components, indexed once)"""
    comp = collections.defaultdict(list)
    for r in rows:
        comp[r['link_id']].append(r)
    return comp


def partner_names(links_path):
    """{display name: [the names it is localized with]} of a stokes_links.csv: the other members of its component, for an
    UNAMBIGUOUS component only (a one-to-many component pools nothing)"""
    rows = read_rows(links_path, LINK_COLS)
    comp = members(rows)
    return {r['identity_name']: [m['identity_name'] for m in comp[r['link_id']] if m['identity_name'] != r['identity_name']]
            for r in rows if r['link_ambiguous'] == '0'}


def read_merge_rows(out_or_dir):
    """[MERGE_COLS rows] of a stokes_link directory; a missing table or one without into_status REFUSES"""
    return read_rows(os.path.join(dir_of(out_or_dir), MERGES), MERGE_COLS)


def read_merges(out_or_dir):
    """{merged identity ID: the LIVE ID it stands for now} of a stokes_link directory; a merged ID whose chain ends at a
    retired identity is left out (no page holds it: a label made on it is refused, never moved). {} when the directory holds
    no merge table (production's grouping: no link)."""
    if not os.path.exists(os.path.join(dir_of(out_or_dir), MERGES)):
        return {}
    return {r['identity_id']: r['into_id'] for r in read_merge_rows(out_or_dir) if r['into_status'] == 'live'}


def link_key(links, stokes, identity_id):
    """the link id of a (Stokes, stable identity ID): its component's link_id, or the identity's own ID (a source of one)"""
    r = links.get((stokes, identity_id))
    return r['link_id'] if r else identity_id
