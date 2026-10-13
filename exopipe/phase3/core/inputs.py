"""Inputs - the ONE resolver of Phase 1 snapshot frames (and, as steps move in, Phase 2 images) for every tier.

Phase 1 frames: ``phase1_frames(lst, date, subband)`` -> {'I': [(key, path)], 'V': [(key, path)]}, one frame per 10-s
INTEGRATION. A cell can hold several Phase 1 runs (re-images, patch runs). The pre-refactor discovery
(clv15/phase3_snapshot_search.discover_frames) keyed frames by their FILE-NAME timestamp with 'latest run wins'; two
runs of the same hour carry file-name timestamps 1-9 s apart, so it kept BOTH: 83 cell-bands in 41 cells mixed 2-3 runs
(12h 2025-01-10 55 MHz: 681 Stokes I frames for one hour; ledger IMPLEMENTED-DATA-DECISIONS-VIS50-20261012, FINDING;
handed to the refactor by Gregg 2026-10-13 ~00:35 UTC, "Hand it to refactor").

The rule here: frames are keyed by their INTEGRATION (the integration midpoint from the header DATE-OBS, truncated to the
second - the label Phase 1 names frames by since exopipe 192c81c), and the LATEST run wins per integration; an older run
contributes only integrations the later runs lack (a patch run fills its gap and nothing else). Measured on 12h 2025-01-10
55 MHz: the three runs share 357-358 integration labels exactly (DATE-OBS agrees across runs); the union is 358, not 681.
Headers are read only where more than one run holds frames of the subband (elsewhere a file-name timestamp is already one
integration). ``phase1_frames`` refuses a result with more frames than integrations in the hour (content test).

Frame file names (Phase 1): old era '<sb>-Pilot-Snapshot-<ts>-<I|V>-image.fits[.fs|.fz]' in snapshots/; new era
'<sb>-Pilot-Snapshot-<ts>-image.fits' = Stokes V in snapshots/, '<sb>-Clean-Snapshot-<ts>-image.fits' = Stokes I in
snapshots_clean/. CLEAN files named by interval index (t0000...) come from CLEAN runs that stopped partway and are not
frames (refactor finding D12): they are counted and reported, never used.
"""
import datetime
import glob
import os
import re

from exopipe.phase3.core.record import Refusal

_PAT_OLD = re.compile(r'^(?P<sb>\d+MHz)-Pilot-Snapshot-(?P<ts>\d{8}_\d{6})-(?P<st>[IV])-image\.fits(?P<fz>\.f[sz])?$')
_PAT_NEW_V = re.compile(r'^(?P<sb>\d+MHz)-Pilot-Snapshot-(?P<ts>\d{8}_\d{6})-image\.fits(?P<fz>\.f[sz])?$')
_PAT_NEW_I = re.compile(r'^(?P<sb>\d+MHz)-Clean-Snapshot-(?P<ts>\d{8}_\d{6})-image\.fits(?P<fz>\.f[sz])?$')
_PAT_INDEX = re.compile(r'^(?P<sb>\d+MHz)-.*-t\d{4}-image\.fits(\.f[sz])?$')
_EPOCH = datetime.datetime(1970, 1, 1)
_DATEOBS_KEY = b'DATE-OBS='


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def read_dateobs(path, nblk=8):
    """DATE-OBS of a FITS file (plain or tile-compressed), read from the first nblk 2880-byte blocks; '' if absent."""
    fd = os.open(path, os.O_RDONLY)
    try:
        buf = os.pread(fd, 2880 * nblk, 0)
    finally:
        os.close(fd)
    for i in range(0, len(buf) - 79, 80):
        card = buf[i:i + 80]
        if card.startswith(_DATEOBS_KEY):
            return card[10:].split(b'/')[0].strip().strip(b"'").strip().decode()
    if len(buf) == 2880 * nblk:
        from astropy.io import fits
        with fits.open(path, memmap=False) as hl:
            for hdu in hl:
                if 'DATE-OBS' in hdu.header:
                    return str(hdu.header['DATE-OBS']).strip()
    return ''


def integration_label(date_obs):
    """'YYYYMMDD_HHMMSS' of the integration MIDPOINT (DATE-OBS = integration start) truncated to the second."""
    s = date_obs.strip().replace('T', ' ')
    fmt = '%Y-%m-%d %H:%M:%S.%f' if '.' in s else '%Y-%m-%d %H:%M:%S'
    t0 = datetime.datetime.strptime(s, fmt)
    mid = (t0 - _EPOCH).total_seconds() + _cfg().instrument.INTEGRATION_S / 2.0
    return (_EPOCH + datetime.timedelta(seconds=int(mid // 1))).strftime('%Y%m%d_%H%M%S')


def _dateobs_many(paths, nthreads=16):
    from concurrent.futures import ThreadPoolExecutor
    if not paths:
        return {}
    with ThreadPoolExecutor(max_workers=max(1, min(nthreads, len(paths)))) as ex:
        return dict(zip(paths, ex.map(read_dateobs, paths)))


def _scan_run(rd, subband):
    """{'I': {ts: path}, 'V': {ts: path}} of one run directory, plus the count of index-named CLEAN files."""
    out = {'I': {}, 'V': {}}
    n_index = 0
    for sub, hint in (('snapshots', None), ('snapshots_clean', 'I')):
        d = os.path.join(rd, subband, sub)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            m = _PAT_OLD.match(fn)
            if m:
                out[m['st']][m['ts']] = os.path.join(d, fn)
                continue
            m = _PAT_NEW_I.match(fn)
            if m:
                out['I'][m['ts']] = os.path.join(d, fn)
                continue
            m = _PAT_NEW_V.match(fn)
            if m and hint != 'I':
                out['V'][m['ts']] = os.path.join(d, fn)
                continue
            if _PAT_INDEX.match(fn):
                n_index += 1
    return out, n_index


def phase1_frames(lst, date, subband, phase1_root=None, report=None):
    """One frame per integration per Stokes for (LST, date, subband): {'I': [(key, path)], 'V': [(key, path)]}, sorted.

    key = the file-name timestamp when only one run holds frames of this subband, else the integration label from the
    header. ``report`` (a dict, optional) receives: runs (run -> frame counts), mixed (bool), dropped (frames of older
    runs superseded per integration), index_named (CLEAN files named by interval index, not used)."""
    root = phase1_root or _cfg().paths.PHASE1_ROOT
    rundirs = sorted(glob.glob(os.path.join(root, lst, date, 'Run_*')))
    if not rundirs:
        raise Refusal(f'no Phase 1 runs for {lst} {date} under {root}')
    per_run = []
    n_index = 0
    for rd in rundirs:
        fr, ni = _scan_run(rd, subband)
        n_index += ni
        if fr['I'] or fr['V']:
            per_run.append((os.path.basename(rd), fr))
    rep = {'runs': {r: {'I': len(f['I']), 'V': len(f['V'])} for r, f in per_run}, 'index_named': n_index,
           'mixed': False, 'dropped': {'I': 0, 'V': 0}}
    out = {}
    for st in ('I', 'V'):
        contributing = [(r, f[st]) for r, f in per_run if f[st]]
        if len(contributing) <= 1:
            items = sorted(contributing[0][1].items()) if contributing else []
        else:
            rep['mixed'] = True
            chosen = {}
            # later runs first: the latest run wins per integration; older runs only fill integrations still missing
            for r, frames in reversed(contributing):
                dob = _dateobs_many(list(frames.values()))
                for ts, p in frames.items():
                    d = dob.get(p, '')
                    if not d:
                        raise Refusal(f'{p}: no DATE-OBS - cannot assign the frame to an integration')
                    key = integration_label(d)
                    if key in chosen:
                        rep['dropped'][st] += 1
                    else:
                        chosen[key] = p
            items = sorted(chosen.items())
        out[st] = items
    # content test: never more frames than integrations in the hour (+1 for a frame straddling the boundary)
    max_frames = int(3600.0 / _cfg().instrument.INTEGRATION_S) + 2
    for st in ('I', 'V'):
        if len(out[st]) > max_frames:
            raise Refusal(f'{lst} {date} {subband} Stokes {st}: {len(out[st])} frames for one hour > {max_frames} '
                          f'integrations - frames from several runs are still mixed')
    if report is not None:
        report.update(rep)
    return out
