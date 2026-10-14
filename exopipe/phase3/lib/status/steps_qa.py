"""`status qa` stages - the QA tables and plots, each the production script run IN PLACE (spec 7.2), outputs under --out.

qa-table    ops/build_qa_table.py in a child (core.legacy.run), argv [<out>/QA/qa_all_phases.csv]; env P3_ARCHIVE_RUN =
            paths.ARCHIVE_RUN, P3_QA_SEARCH_LABEL = status.SEARCH_LABEL['deep'] (the script reads them, :134, :155). It reads
            the production bandpass_phase_qa.csv (literal path, :38).
solar       ops/solar_contamination_audit.py in-process (module-level script) with its literal output redirected to
            <out>/QA/solar_contamination_audit.candidate_<UTC>.csv - NEVER the installed table (S-G3/S-G4: a pinned
            masters gate input; installing a regenerated audit is `calibrate`'s, with --ruling). A diff against the
            installed table (per verdict) is written beside it.
beam        ops/beam_audit.py in a child, argv [<out>/QA/beam_audit, status_qa.BEAM_NPROC].
noise       ops/qa_noise_vs_freq_plot.py in a child, argv [<out>/QA/median_noise_vs_freq_all_lsts_v2.png].
outliers    NOT in place (the tool rewrites data_decisions.csv, :52-59, and reads the deleted /fast coadd tree, :19): the
            selection of pkg/../tools/propagate_date_outliers.py:17-51 over <COADD_SB_ROOT>/*/residuals_provisional.csv,
            written as PROPOSALS for Gregg (<out>/QA/date_outlier_proposals.csv); data_decisions.csv sha1 checked unchanged.
bandpass    ops/bandpass_phase_warp.py by path with site PYTHON['casa'], status_qa.BANDPASS_SHARDS shards in parallel
            (inheriting the step's --cores pinning, not `taskset -c 0-15`), merged sorted by (cal_date, cal_lst) - the
            heredoc of ops/run_bandpass_phase_qa.sh:14-24 - into <out>/QA/bandpass_phase_qa.csv.
provenance  ops/candidate_provenance_audit.py and ops/perframe_solar_check.py in-process with argv [<run dir>] (--run
            required, S-G12) and their literal outputs redirected to <out>/QA/.
"""
import collections
import hashlib
import os

from exopipe.phase3.lib.status import steps_common as K

STAGES = ('bandpass', 'qa-table', 'solar', 'provenance', 'beam', 'noise', 'outliers')
OUTLIER_COLUMNS = ('lst', 'subband', 'date', 'verdict_proposed', 'reason', 'n_rejected', 'n_cells', 'leaked_lsts')
SOLAR_DIFF_COLUMNS = ('lst', 'date', 'subband', 'installed_verdict', 'candidate_verdict', 'change')
BEAM_COLUMNS = ('lst', 'date', 'scope', 'stokes', 'bmaj_as', 'bmin_as', 'cell_median_area', 'area_ratio', 'err', 'path')  # beam_audit.py:73-79


def sha1_of(path):
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def stage_inputs(stage, run_dir=None):
    """Input roots a stage needs (S-G5 / S-G11 checks)."""
    C = K.cfg()
    if stage == 'qa-table':
        return [C.status_qa.QA_ASSEMBLY, C.paths.COADD_SB_ROOT, C.paths.PHASE2_ROOT]
    if stage in ('solar', 'beam', 'noise'):
        return [C.paths.PHASE2_ROOT]
    if stage == 'outliers':
        return [C.paths.COADD_SB_ROOT]
    if stage == 'bandpass':
        return [os.path.dirname(C.status_qa.BANDPASS_TABLES_GLOB.split('*')[0].rstrip('/')) or '/']
    if stage == 'provenance':
        return [run_dir, K.expand(C.status_qa.SOLAR_INSTALLED)]
    return []


def stage_outputs(stage, out, utc):
    C = K.cfg()
    Q = C.status_qa
    if stage == 'qa-table':
        return [(K.place(out, 'QA', Q.QA_TABLE), 'qa_table')]
    if stage == 'solar':
        cand = K.place(out, 'QA', Q.SOLAR_CANDIDATE.format(utc=utc))
        return [(cand, 'solar_audit'), (cand[:-4] + '.diff.csv', 'csv_table')]
    if stage == 'beam':
        d = K.place(out, 'QA', Q.BEAM_OUT)
        return [(os.path.join(d, 'beam_audit_all.csv'), 'beam_audit'), (os.path.join(d, 'beam_audit_flagged.csv'),
                                                                        'csv_table')]
    if stage == 'noise':
        return [(K.place(out, 'QA', Q.NOISE_PLOT), 'noise_plot')]
    if stage == 'outliers':
        return [(K.place(out, 'QA', Q.OUTLIER_PROPOSALS), 'date_outlier_proposals')]
    if stage == 'bandpass':
        return [(K.place(out, 'QA', Q.BANDPASS_QA), 'bandpass_phase_qa')]
    if stage == 'provenance':
        return [(K.place(out, 'QA', Q.PROVENANCE_TABLE), 'provenance_audit'),
                (K.place(out, 'QA', Q.PERFRAME_TABLE), 'provenance_audit')]
    return []


# ---------------------------------------------------------------------------------------------------------------------

def run_qa_table(out, unit, log):
    from exopipe.phase3.core import legacy
    from exopipe.phase3.lib.status import steps_inplace as IP
    C = K.cfg()
    dest = K.check_write(K.place(out, 'QA', C.status_qa.QA_TABLE))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    env = {C.status_qa.ARCHIVE_RUN_ENV: C.paths.ARCHIVE_RUN, C.status_qa.SEARCH_LABEL_ENV: C.status.SEARCH_LABEL['deep']}
    unit.meta['env_set'] = env
    res = legacy.run(IP.legacy_script('build_qa_table.py'), [dest], python='default', env=env, log=log, unit=unit)
    return res, dest


def run_solar(out, utc, unit, log, allowed):
    from exopipe.phase3.lib.status import steps_inplace as IP
    C = K.cfg()
    installed = K.expand(C.status_qa.SOLAR_INSTALLED)
    cand = K.check_write(K.place(out, 'QA', C.status_qa.SOLAR_CANDIDATE.format(utc=utc)))
    os.makedirs(os.path.dirname(cand), exist_ok=True)
    before = sha1_of(installed) if os.path.exists(installed) else None
    rc, text = IP.run_path(IP.legacy_script('solar_contamination_audit.py'), [], mapping={installed: cand},
                           allowed_roots=allowed, log=log, unit=unit)
    after = sha1_of(installed) if os.path.exists(installed) else None
    unit.check('S-G3.installed_solar_audit_untouched', before == after, value=after, detail=installed)
    diff = cand[:-4] + '.diff.csv'
    if rc == 0 and os.path.exists(cand):
        old = {(r['lst'], r['date'], r['subband']): r.get('verdict', '') for r in K.read_csv(installed)} \
            if os.path.exists(installed) else {}
        new = {(r['lst'], r['date'], r['subband']): r.get('verdict', '') for r in K.read_csv(cand)}
        rows = []
        for k in sorted(set(old) | set(new)):
            a, b = old.get(k), new.get(k)
            if a != b:
                rows.append(dict(lst=k[0], date=k[1], subband=k[2], installed_verdict=a or '', candidate_verdict=b or '',
                                 change='added' if a is None else ('removed' if b is None else 'changed')))
        K.write_csv(diff, SOLAR_DIFF_COLUMNS, rows)
        unit.meta['solar_diff'] = dict(collections.Counter(r['change'] for r in rows))
    return rc, text, cand, diff


def run_beam(out, unit, log):
    from exopipe.phase3.core import legacy
    from exopipe.phase3.lib.status import steps_inplace as IP
    C = K.cfg()
    d = K.place(out, 'QA', C.status_qa.BEAM_OUT)
    K.check_write(os.path.join(d, 'beam_audit_all.csv'))
    return legacy.run(IP.legacy_script('beam_audit.py'), [d, C.status_qa.BEAM_NPROC], python='default', log=log,
                      unit=unit), d


def run_noise(out, unit, log):
    from exopipe.phase3.core import legacy
    from exopipe.phase3.lib.status import steps_inplace as IP
    C = K.cfg()
    dest = K.check_write(K.place(out, 'QA', C.status_qa.NOISE_PLOT))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    return legacy.run(IP.legacy_script('qa_noise_vs_freq_plot.py'), [dest], python='default', log=log, unit=unit), dest


# The rows that keep a date-level proposal out: production skips (lst, subband, date) keys with ANY row in data_decisions
# (propagate_date_outliers.py:40-42: exclude, keep, ... - a 'keep' is Gregg's readmission). core/exclusions (the only
# reader of the file) exposes exclude rows only, so proposals on a night Gregg kept are possible here; the shared fix
# (core/exclusions keeping every row's key) is reported in plan/impl/status.md. The rule used is recorded per run.
OUTLIER_EXISTING_RULE = 'exclude rows (core.exclusions); production: every row of any verdict'


def date_outliers(ex, unreadable=None):
    """(proposals, files read) - propagate_date_outliers.py:17-51 selection over the CURRENT working tree. Each file is
    read row by row as the tool reads it (csv.DictReader(open(f)), :23-24: rows before an error are kept); a file the
    tool's bare except would cut short (:25-26) is cut short here too and appended to ``unreadable`` (S-G16: counted)."""
    import csv
    import glob
    C = K.cfg()
    Q = C.status_qa
    verdict = collections.defaultdict(lambda: collections.defaultdict(dict))   # date -> lst -> {cell: outlier}
    files = sorted(glob.glob(os.path.join(C.paths.COADD_SB_ROOT, '*_*', 'residuals_provisional.csv')))
    for f in files:
        cell = os.path.basename(os.path.dirname(f))
        lst = cell.split('_')[0]
        try:
            with open(f) as fh:
                for r in csv.DictReader(fh):
                    verdict[r['date']][lst][cell] = (str(r['outlier']) == 'True')
        except Exception as e:                        # noqa: BLE001 - as the tool (:25-26)
            if unreadable is not None:
                unreadable.append(f'{f}: {type(e).__name__}: {e}'[:300])
    rows = []
    for date, by_lst in sorted(verdict.items()):
        cells = [(lst, c, o) for lst, cc in by_lst.items() for c, o in cc.items()]
        n_tot = len(cells)
        n_rej = sum(1 for *_, o in cells if o)
        if n_tot >= Q.OUTLIER_MIN_CELLS and n_rej / n_tot >= Q.OUTLIER_MIN_FRAC and n_rej < n_tot:
            leak = sorted({lst for lst, cc in by_lst.items() if any(not o for o in cc.values())})
            for lst in leak:
                for sb in Q.OUTLIER_SUBBANDS:
                    if (lst, sb.replace('MHz', ''), date) in ex.rows:
                        continue
                    rows.append(dict(lst=lst, subband=sb, date=date, verdict_proposed='exclude',
                                     reason=f'date-level outlier (rejected {n_rej}/{n_tot} cells) - PROPOSAL for Gregg',
                                     n_rejected=n_rej, n_cells=n_tot, leaked_lsts=';'.join(leak)))
    return rows, files


def bandpass_tables():
    import glob
    return sorted(glob.glob(K.cfg().status_qa.BANDPASS_TABLES_GLOB))


def run_bandpass(out, work, unit, log):
    from exopipe.phase3.core import legacy
    from exopipe.phase3.lib.status import steps_inplace as IP
    C = K.cfg()
    Q = C.status_qa
    dest = K.check_write(K.place(out, 'QA', Q.BANDPASS_QA))
    wd = os.path.join(work or K.place(out, 'status', ''), 'bpqa')
    os.makedirs(wd, exist_ok=True)
    import glob
    for old in glob.glob(os.path.join(wd, 'bpqa_*.csv')):   # a fresh shard set: never merge an earlier run's shards
        os.remove(old)
    tables = bandpass_tables()
    lst = os.path.join(wd, 'tables.txt')
    with open(lst, 'w') as f:
        f.write(''.join(t + '\n' for t in tables))
    script = IP.legacy_script(Q.BANDPASS_SCRIPT)
    n = Q.BANDPASS_SHARDS

    def shard(i):
        return legacy.run(script, ['--csv', os.path.join(wd, f'bpqa_{i}.csv'), '--shard', i, n, lst], python='casa',
                          log=os.path.join(wd, f'bpqa_{i}.log'), unit=unit)
    res = K.thread_map(shard, range(n), n)
    rows = []
    for p in sorted(glob.glob(os.path.join(wd, 'bpqa_*.csv'))):   # run_bandpass_phase_qa.sh:17 (name order)
        rows += K.read_csv(p)
    rows.sort(key=lambda r: (r['cal_date'], r['cal_lst']))     # run_bandpass_phase_qa.sh:19
    if rows:
        K.write_csv(dest, list(rows[0].keys()), rows)
    unit.meta['bandpass'] = {'tables': len(tables), 'rows': len(rows), 'failed': sum(1 for r in rows if r.get('error')),
                             'shard_rc': [r.rc for r in res], 'list': lst}
    return res, dest, tables


def run_provenance(out, run_dir, unit, log, allowed):
    from exopipe.phase3.lib.status import steps_inplace as IP
    C = K.cfg()
    Q = C.status_qa
    qa = C.status.QA_DIR
    res = []
    for script, name in (('candidate_provenance_audit.py', Q.PROVENANCE_TABLE),
                         ('perframe_solar_check.py', Q.PERFRAME_TABLE)):
        dest = K.check_write(K.place(out, 'QA', name))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        rc, text = IP.run_path(IP.legacy_script(script), [run_dir], mapping={os.path.join(qa, name): dest},
                               allowed_roots=allowed, log=log, unit=unit)
        res.append((script, rc, dest, text[-400:]))
    return res
