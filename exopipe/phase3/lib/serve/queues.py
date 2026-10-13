"""Served queues: sectioning of the tier queues and the multiples / multi-integration queues (ports of the launchers'
heredocs, byte-identical output; tests/test_serve.py runs the production heredocs on the same synthetic input and compares).

  section_tier_queue(U, K, 'deep')    p3s3_serve_tier.sh:13-32 (PYQ): multi-epoch (n_det >= MULTI_MIN_NDET) first, single,
                                      v110 marginal last, keyed by the composed page id Jname_LST_date_band (ledgers 629, 730)
  section_tier_queue(U, K, 'sublst')  sublst_serve.sh:57-72 (PYQ): multi-integration first, then single (no marginal)
  deep_multi_queue(U, S)              p3s3_serve_multiples.sh:14-85 (PYQ, fork branch only): the fork raw queues ordered by
                                      SNR_rep, multi vs single sections per fork tier, de-duplicated; refuses without the
                                      fork (G3; the chain-kept / recovered fallback sections are dead for deep, not ported)
  sublst_multi_queue(Q, QM)           sublst_cumulative.sh:79 (awk): the MULTI-INTEGRATION section of the Stokes queue
The Stokes queue itself (wideband + subband merged) is p3s3_stokes_queue.py, called in place.
"""
import csv
import glob
import os

from exopipe.phase3.core.record import Refusal


def _S():
    from exopipe.phase3.config import CFG
    return CFG.serve


def page_id(r):
    """Jname_LST_date_band from a pages_input row (p3s3_serve_tier.sh:19)."""
    return f"{r['_sid']}_{r['lst']}_{r['date']}_{r['band']}"


def tier_queue_path(U, K):
    return f'{U}/judgements/queue_p3s3_{K}.txt'


def section_tier_queue(U, K, tier, lsts_text=''):
    """Re-section judgements/queue_p3s3_<K>.txt in place. Returns {'multi', 'single', 'marginal'} counts, or None when the
    tier has no queue (the launchers skip it: p3s3_serve_tier.sh:17 exits for that tier only; sublst_serve.sh:57)."""
    S = _S()
    qp = tier_queue_path(U, K)
    if not os.path.exists(qp):
        return None
    ids = [l.rstrip('\n') for l in open(qp) if l.strip() and not l.startswith('#')]
    pin = f'{U}/source_catalogs/pages_input_{K}.csv'
    marg = []
    if tier == 'deep':
        lab = {page_id(r): (r.get('label') or '') for r in csv.DictReader(open(pin))}
        marg = [i for i in ids if lab.get(i) == S.MARGINAL_LABEL]
        ids = [i for i in ids if lab.get(i) != S.MARGINAL_LABEL]
    nd = {page_id(r): int(float(r.get('n_det') or 1)) for r in csv.DictReader(open(pin))}
    multi = [i for i in ids if nd.get(i, 1) >= S.MULTI_MIN_NDET]
    single = [i for i in ids if nd.get(i, 1) < S.MULTI_MIN_NDET]
    if tier == 'deep':
        H = S.QUEUE_HEADERS['deep_tier']
        text = (f'{H[0].format(K=K)}\n{H[1]}\n' + '\n'.join(multi) + ('\n' if multi else '') + f'{H[2]}\n'
                + '\n'.join(single) + ('\n' if single else ''))
        if marg:
            text += f'{H[3]}\n' + '\n'.join(marg) + '\n'
    else:
        H = S.QUEUE_HEADERS['sublst_tier']
        text = (f'{H[0].format(lsts=lsts_text, K=K)}\n{H[1]}\n' + '\n'.join(multi) + ('\n' if multi else '')
                + f'{H[2]}\n' + '\n'.join(single) + ('\n' if single else ''))
    with open(qp + '.new', 'w') as f:
        f.write(text)
    os.replace(qp + '.new', qp)
    return {'multi': len(multi), 'single': len(single), 'marginal': len(marg)}


def fork_raw_queues(U, ST):
    """The fork raw queues of a Stokes that exist: [(t, path)] for t in wb, sb (p3s3_serve_multiples.sh:46-48)."""
    out = []
    for ft in _S().DEEP_FORK_TIERS:
        t = ft[1:]
        qp = f'{U}/judgements/queue_p3s3_{ST}_f{t}_raw.txt'
        if os.path.exists(qp):
            out.append((t, qp))
    return out


def deep_multi_queue(U, ST):
    """Write judgements/queue_p3s3_<ST>_multi.txt from the fork raw queues. Returns (queue path, keep set of
    (J-name, band), sections [(header, ids)])."""
    S = _S()
    H = S.QUEUE_HEADERS['deep_multi']
    fork = []
    for t, qp in fork_raw_queues(U, ST):
        ids = [l.rstrip('\n') for l in open(qp) if l.strip() and not l.startswith('#')]
        nd = {}
        pi = f'{U}/source_catalogs/pages_input_{ST}_f{t}.csv'
        if os.path.exists(pi):
            for r in csv.DictReader(open(pi)):
                nd[page_id(r)] = int(float(r.get('n_det') or 1))
        srep = {}
        for sp in glob.glob(f'{U}/source_catalogs_{ST}_f{t}/*_sources_survivors.csv'):
            for r in csv.DictReader(open(sp)):
                try:
                    srep[(r['source_id'], r['band'])] = float(r.get('snr_rep') or 0)
                except ValueError:
                    pass

        def _key(i, srep=srep):
            base, lst, date, band = i.rsplit('_', 3)
            return -srep.get((base, band), 0.0)
        if srep:
            ids = sorted(ids, key=_key)
        multi = [i for i in ids if nd.get(i, 1) >= S.MULTI_MIN_NDET]
        single = [i for i in ids if nd.get(i, 1) < S.MULTI_MIN_NDET]
        fork.append((H[1].format(T=t.upper()), multi))
        fork.append((H[2].format(T=t.upper()), single))
    if not fork:
        raise Refusal(f'G3 deep multiples need the fork: no judgements/queue_p3s3_{ST}_f{{wb,sb}}_raw.txt under {U} - '
                      f'REFUSING the fallback sections (R-MULTIPLES-FORK-INTO-PRODUCTION-PREAMBLE-APPROVED; '
                      f'p3s3_serve_multiples.sh:66-71)')
    secs = fork
    seen, keep = set(), set()
    lines = [H[0].format(S=ST), H[3].format(counts=', '.join(f'{len(ids)} ids' for _, ids in secs))]
    out_secs = []
    for hdr, ids in secs:
        ids = [i for i in ids if not (i in seen or seen.add(i))]
        lines.append(hdr)
        lines.extend(ids)
        out_secs.append((hdr, ids))
        for i in ids:
            base, lst, date, band = i.rsplit('_', 3)
            keep.add((base, band))
    dst = f'{U}/judgements/queue_p3s3_{ST}_multi.txt'
    with open(dst + '.new', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    os.replace(dst + '.new', dst)
    return dst, keep, out_secs


def sublst_multi_queue(Q, QM):
    """awk '/^# selection: MULTI-INTEGRATION/{m=1; print; next} /^# selection:/{m=0} m' Q > QM (sublst_cumulative.sh:79).
    Returns True when QM holds a line starting 'J'."""
    marker = _S().QUEUE_HEADERS['sublst_multi_marker']
    out, m = [], False
    for line in open(Q):
        rec = line.rstrip('\n')
        if rec.startswith(marker):
            m = True
            out.append(rec + '\n')
            continue
        if rec.startswith('# selection:'):
            m = False
        if m:
            out.append(rec + '\n')
    with open(QM, 'w') as f:
        f.write(''.join(out))
    return any(l.startswith('J') for l in out)
