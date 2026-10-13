"""The deep page plan of one page set: what build_i14 (ops/build_i14_pages_std7.py, run in place) must write.

Recomputed from the same inputs and with the same rules as the builder, without importing it:
- rows of source_catalogs/pages_input_<K>.csv sorted by -|snr| (P3_SORT_SNR=1; a stable sort, build_i14:55-56), rank =
  position + 1; page id = <J-name>_<LST>_<date>_<band> for a J-name source id, else i14_<rank:04d>_<LST>_<date>_<band>
  (build_i14:57-67);
- one panel per row plus one per MEMBER detection of the source in this set (source_catalogs_<K>/<band>_source_members.csv,
  keyed by (bare J-name, band); a member at the row's own (LST, night) is the row itself; a member's LST is token 1 of its
  cand_id; build_i14:657-697); every panel of a page sits in the page's directory and carries the page's rank:
  candidates/I/<page id>/rank<rank:04d>_<LST>_<night>_<band>.png (build_i14:865-867);
- the page index rows (non-member rows, by band) and the raw queue (every row's page id in rank order; build_i14:992-1018).
"""
import csv
import glob
import math
import os
import re


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float('nan')


def ordered_rows(pages_input_path, sort_key='snr'):
    rows = list(csv.DictReader(open(pages_input_path)))
    key = 'snr' if sort_key == 'snr' else 'peak_jy'
    rows.sort(key=lambda r: -abs(fnum(r[key])))
    for i, r in enumerate(rows):
        r['_rank'] = i + 1
        j = (r.get('_sid') or '').strip()
        r['_pid'] = (f"{j}_{r['lst']}_{r['date']}_{r['band']}" if j.startswith('J')
                     else f"i14_{i + 1:04d}_{r['lst']}_{r['date']}_{r['band']}")
    return rows


def members(U, K):
    out = {}
    for mp in sorted(glob.glob(f'{U}/source_catalogs_{K}/*_source_members.csv')):
        for m in csv.DictReader(open(mp)):
            out.setdefault((m['source_id'], m['band']), []).append(m)
    return out


def index_tag(K):
    """The tag of the set's index file names (build_i14:996-1001): the Stokes for wb/sb sets, the set itself otherwise."""
    m = re.match(r'^([A-Za-z]+_[a-z]+)$', K)
    st = K.split('_')[0] if K[:1] in 'VI' else ''
    if m and K.split('_')[-1] not in ('wb', 'sb'):
        return K
    return st


class Plan:
    """rows (ordered, with _rank/_pid), panels {abs png path: (page id, lst, date, band, cand_id)}, pages {page id: [png]},
    index {band: abs path}, queue ids, and the page directory root."""

    def __init__(self, U, K, sort_key='snr', layout='shared', only=None):
        self.U, self.K = U, K
        self.input = f'{U}/source_catalogs/pages_input_{K}.csv'
        self.rows = ordered_rows(self.input, sort_key) if os.path.exists(self.input) else []
        self.subdir = 'I' if layout == 'shared' else K
        self.cand_root = f'{U}/candidates/{self.subdir}'
        mem = members(U, K)
        self.panels, self.pages = {}, {}
        for r in self.rows:
            pid = r['_pid']
            if only is not None and pid not in only:
                continue
            ddir = f'{self.cand_root}/{pid}'
            keys = [(r['lst'], r['date'], r['band'], '')]
            sid0 = (r.get('_sid') or '').rsplit('_', 3)[0] if (r.get('_sid') or '').startswith('J') else ''
            for m in mem.get((sid0, r['band']), []):
                cid = m.get('cand_id', '')
                lst = cid.split('_')[1] if cid.count('_') >= 3 else r['lst']
                if (lst, m.get('det_date')) == (r['lst'], r['date']):
                    if not keys[0][3]:
                        keys[0] = (r['lst'], r['date'], r['band'], cid)  # the primary (best) detection is the row itself
                    continue
                keys.append((lst, m.get('det_date'), r['band'], cid))
            for lst, date, band, cid in keys:
                png = os.path.abspath(f'{ddir}/rank{r["_rank"]:04d}_{lst}_{date}_{band}.png')
                self.panels.setdefault(png, (pid, lst, date, band, cid))
                self.pages.setdefault(pid, [])
                if png not in self.pages[pid]:
                    self.pages[pid].append(png)
        tag = index_tag(K)
        self.index = {}
        if only is None:
            for b in sorted({r['band'] for r in self.rows}):
                self.index[b] = os.path.abspath(f'{U}/source_catalogs/{b}_pages_index' + (f'_{tag}' if tag else '') + '.csv')
        self.queue_ids = [r['_pid'] for r in self.rows]

    def page_ids(self):
        return [r['_pid'] for r in self.rows]

    def cells(self):
        """(lst, band) of every planned panel."""
        return sorted({(v[1], v[3]) for v in self.panels.values()})

    def panel_keys(self):
        """(lst, band, date) of every planned panel (exclusion bookkeeping)."""
        return [(v[1], v[3], v[2]) for v in self.panels.values()]

    def identities(self):
        """identity id of every page (the astrometry figure key, build_i14:955)."""
        out = set()
        for r in self.rows:
            ident = (r.get('identity_id') or '').strip() or (r.get('_sid') or '').split('_')[0]
            if ident:
                out.add(ident)
        return out

    def epoch_panels(self, epochs):
        """Panels whose (LST, night) is one of the epoch items <LST>_<date>."""
        want = set()
        for e in epochs or ():
            lst, _, d = e.partition('_')
            want.add((lst, d[:10]))
        return {p: v for p, v in self.panels.items() if (v[1], (v[2] or '')[:10]) in want}

    def pages_in_cells(self, cells):
        """Page ids with a panel in one of the cells <LST>_<band> (the --cells re-render)."""
        want = set(cells or ())
        return sorted({v[0] for v in self.panels.values() if f'{v[1]}_{v[3]}' in want})


def master_paths(roots_csv, cells):
    """Master files the builder reads for (lst, band) cells through the roots map (p3s2_frames.master_path:31-41)."""
    roots = {}
    try:
        for line in open(roots_csv):
            p = line.strip().split(',')
            if len(p) >= 3 and p[0]:
                roots[p[0]] = p[2]
    except OSError:
        return []
    out = []
    for lst, band in cells:
        r = roots.get(f'{lst}_{band}')
        if not r:
            continue
        for fn in (f'I_deep_Taper_R075_{band}.fits', f'V_deep_Taper_R0_{band}.fits'):
            p = f'{r}/{lst}_{band}/coadd_final/{fn}'
            if os.path.exists(p):
                out.append(p)
    return out


def cell_inputs(U, cells):
    """Per-cell run inputs of the builder: residual links ([iv]resid/<cell>), I parents, the v110 review CSV."""
    out = []
    for lst, band in cells:
        for p in (f'{U}/iresid/{lst}_{band}', f'{U}/vresid/{lst}_{band}', f'{U}/parent_catalogs/master_R075_{lst}_{band}.gaul.csv',
                  f'{U}/v110_products/{lst}_{band}/V_top100_p3_REVIEW.csv'):
            if os.path.lexists(p):
                out.append(p)
    return out


def finite(x):
    v = fnum(x)
    return v if math.isfinite(v) else None
