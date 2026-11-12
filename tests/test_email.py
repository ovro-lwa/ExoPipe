#!/usr/bin/env python3
"""STANDING TEST - the hour's ONE email (Gregg 2026-11-12: "one email at the end of Phase 3 that is close to identical to
the current one in what is included, just with the transient and forced photometry highlights added"), on synthetic data.

  mailer   the Phase 2 report rebuilt by a later process (restore_email_inputs) is the report the Phase 2 process itself
           would have sent: same body, same attachments in the same order; the highlights block is the only addition;
           --defer_email records the command and the duration; the review copy (write_to) sends nothing
  block    the Phase 3 highlights block: targets first, then the top transients; every limit and missing input is said in
           the block (tests added with lib/email/highlights.py)

Nothing is sent: every report is written to a scratch file. Run standalone:  python tests/test_email.py [test_name ...]
"""
import email
import email.header
import json
import os
import shutil
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def _run_dir(root, detections=True):
    """A Phase 2 run directory with the files the report reads (names as in a real run of 04h 2024-12-19)."""
    rd = os.path.join(root, '04h', '2024-12-19', 'Science_20260527_182537')
    qa = os.path.join(rd, 'QA')
    diag = os.path.join(rd, 'Dewarp_Diagnostics')
    os.makedirs(qa)
    os.makedirs(diag)
    os.makedirs(os.path.join(rd, 'detections'))
    pre = '2024-12-19_04h_'
    for name in (pre + 'thermal_noise_vs_freq.png', pre + 'dewarp_vs_frequency.png',
                 'I_deep_Taper_Robust-0.75_20241219_LST04h_3color.png', pre + 'Flux_Check_Hybrid_3C147.png'):
        open(os.path.join(qa, name), 'wb').write(b'PNG ' + name.encode())
    open(os.path.join(qa, pre + 'thermal_noise.csv'), 'w').write('freq,rms\n41,0.050\n73,0.020\n')
    open(os.path.join(qa, pre + 'flux_check_hybrid.csv'), 'w').write('source,ratio\n3C147,1.01\n')
    json.dump({'disturbance_rms_arcmin': 1.25, 'disturbance_quality': 'Good', 'median_shift_arcmin': 2.0, 'n_anchors': 800},
              open(os.path.join(diag, pre + 'warp_stats_73MHz.json'), 'w'))
    open(os.path.join(diag, pre + 'anchor_vectors_73MHz.png'), 'wb').write(b'PNG anchors')
    open(os.path.join(rd, 'phase2_04h_2024-12-19_Science_20260527_182537.log'), 'w').write('log\n')
    open(os.path.join(rd, 'detections', 'search_73MHz.csv'), 'w').write('status,snr,ra_hms,dec_dms\nPASS,9.5,04:10:00,+30:00:00\n')
    if detections:
        for name in ('subband_detections.csv', 'wideband_detections.csv', 'combined_detections.csv'):
            open(os.path.join(rd, name), 'w').write('image,snr\nx.fits,9.5\n')
    return rd


def _parts(eml):
    """(subject, html body, [attachment file names in order]) of a written report."""
    msg = email.message_from_string(open(eml).read())
    body, names = None, []
    for part in msg.walk():
        if part.get_content_type() == 'text/html':
            body = part.get_payload(decode=True).decode()
        elif part.get_filename():
            names.append(part.get_filename())
    return str(email.header.make_header(email.header.decode_header(msg['Subject']))), body, names


def _as_phase2_fills(agg, rd):
    """The two module lists as the Phase 2 process leaves them before the send (run_wideband_stacking: thermal-noise
    plot, dewarp summary plot, 3-color PNG; then gather_detections' tables)."""
    del agg.ATTACHMENT_FILES[:]
    del agg.EMAIL_REPORT_LINES[:]
    qa = os.path.join(rd, 'QA')
    for name in ('2024-12-19_04h_thermal_noise_vs_freq.png', '2024-12-19_04h_dewarp_vs_frequency.png',
                 'I_deep_Taper_Robust-0.75_20241219_LST04h_3color.png'):
        agg.ATTACHMENT_FILES.append(os.path.join(qa, name))
    if os.path.exists(os.path.join(rd, 'combined_detections.csv')):
        agg.ATTACHMENT_FILES.extend(os.path.join(rd, n) for n in ('subband_detections.csv', 'wideband_detections.csv'))
    else:
        agg.EMAIL_REPORT_LINES.append('No significant detections found.')


def test_mailer_rebuilt_report_is_the_phase2_report():
    from exopipe.phase2 import phase2_science_agg as agg
    for detections in (True, False):
        d = tempfile.mkdtemp(prefix='test_email_')
        try:
            rd = _run_dir(d, detections=detections)
            kw = dict(run_command='phase2_controller.py --lst 04h --date 2024-12-19', run_duration=1448.0, lst='04h',
                      date='2024-12-19')
            _as_phase2_fills(agg, rd)
            agg.send_email_report(rd, write_to=os.path.join(d, 'a'), **kw)
            # a later process: the deferred record + the files of the run
            agg.write_deferred_email(rd, **kw)
            saved = agg.restore_email_inputs(rd)
            assert saved == kw, saved
            agg.send_email_report(rd, write_to=os.path.join(d, 'b'), **saved)
            sa, ba, na = _parts(os.path.join(d, 'a.eml'))
            sb, bb, nb = _parts(os.path.join(d, 'b.eml'))
            assert (sa, ba, na) == (sb, bb, nb), (na, nb)
            assert sa == 'OVRO-LWA Phase 2: 04h 2024-12-19 — Science_20260527_182537', sa
            assert na[:3] == ['2024-12-19_04h_thermal_noise_vs_freq.png', '2024-12-19_04h_dewarp_vs_frequency.png',
                              'I_deep_Taper_Robust-0.75_20241219_LST04h_3color.png'], na
            assert ('subband_detections.csv' in na) == detections and len(na) == len(set(na)), na
            assert open(os.path.join(d, 'a.html')).read() == ba
            # the highlights block is the only addition, placed above the Phase 2 sections
            block = '<h4>Phase 3 highlights</h4><p>TEST BLOCK</p>'
            agg.restore_email_inputs(rd)
            agg.send_email_report(rd, write_to=os.path.join(d, 'c'), highlights_html=block, phases='Phases 2-3', **saved)
            sc, bc, nc = _parts(os.path.join(d, 'c.eml'))
            assert sc == sa.replace('Phase 2:', 'Phases 2-3:') and nc == na, (sc, nc)
            assert bc.replace(block, '').replace('Phases 2-3 Pipeline', 'Phase 2 Pipeline') == ba
            assert bc.index(block) < bc.index('<h4>Thermal Noise</h4>') and bc.count(block) == 1
        finally:
            shutil.rmtree(d, ignore_errors=True)


def test_mailer_run_that_was_not_deferred_has_no_record():
    from exopipe.phase2 import phase2_science_agg as agg
    d = tempfile.mkdtemp(prefix='test_email_')
    try:
        rd = _run_dir(d)
        assert agg.restore_email_inputs(rd) == {}
        agg.ATTACHMENT_FILES.append('/nonexistent/left_over.png')       # a second call starts from empty lists
        agg.restore_email_inputs(rd)
        assert '/nonexistent/left_over.png' not in agg.ATTACHMENT_FILES and len(agg.ATTACHMENT_FILES) == 5
        before = sorted(os.listdir(rd)) + sorted(os.listdir(os.path.join(rd, 'QA')))
        agg.send_email_report(rd, write_to=os.path.join(d, 'x'), lst='04h', date='2024-12-19')
        assert sorted(os.listdir(rd)) + sorted(os.listdir(os.path.join(rd, 'QA'))) == before   # nothing written in the run
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_block_states_chain_outcome_limits_targets_then_transients():
    from exopipe.phase3.lib.notify import block
    d = tempfile.mkdtemp(prefix='test_email_')
    try:
        step = lambda name, status, rc: {'name': name, 'argv': ['py', '-m', 'exopipe.phase3', name, '--tier', 'deep'],
                                         'status': status, 'rc': rc}
        recs = {'done': {'chain_status': 'done', 'steps': [step('search', 'done', 0), step('catalog', 'no-input', 3)]},
                'refused': {'chain_status': 'refused', 'log': '/x/logs/04h.log',
                            'steps': [step('search', 'done', 0), step('catalog', 'refused', 2)]},
                'failed': {'chain_status': 'failed', 'steps': [step('search', 'error', 1)]}}
        for k, r in recs.items():
            json.dump(r, open(os.path.join(d, k + '.json'), 'w'))
        t, ok = block.chain_summary(os.path.join(d, 'done.json'))
        assert ok is True and 'done (2 steps: 1 done, 1 no-input)' in t, t
        t, ok = block.chain_summary(os.path.join(d, 'refused.json'))
        assert ok is False and 'REFUSED at step 2 (<code>catalog --tier</code>, exit code 2)' in t and '/x/logs/04h.log' in t, t
        t, ok = block.chain_summary(os.path.join(d, 'failed.json'))
        assert ok is False and 'FAILED at step 1' in t, t
        t, ok = block.chain_summary(None)
        assert ok is None and 'none given' in t
        t, ok = block.chain_summary(os.path.join(d, 'missing.json'))
        assert ok is None and 'unreadable' in t
        targets = [{'target': 'HD 189733 <b>', 'sample': 'Hot / Warm Jupiter', 'found_by': 'forced photometry',
                    'tier': '10-minute', 'band': '73 MHz', 'stokes': 'V', 'time_utc': '2024-12-19 08:10:00',
                    'flux_jy': 1.2345, 'snr_formal': 7.26, 'snr_local': None, 'link': 'http://h/p1'}]
        trans = [{'rank': 1, 'tier': 'deep', 'band': '41 MHz', 'stokes': 'I', 'time_utc': '2024-12-19',
                  'ra_deg': 62.55, 'dec_deg': -0.5, 'snr_formal': 12.04, 'snr_local': float('nan'), 'cls': 'single',
                  'link': 'http://h/p2'}]
        h = block.build('04h_2024-12-19', 'CHAIN', ['the 10-s table is missing'], targets, '1 detection of 40 targets',
                        [('V', [], 0), ('I', trans, 37)], 37, {'deep': (120, 37)}, 20)
        order = [h.index(x) for x in ('CHAIN', 'Limits and missing inputs', 'the 10-s table is missing',
                                      'Detections of Hot / Warm Jupiter and 10 pc sample targets', 'HD 189733 &lt;b&gt;',
                                      '1 detection of 40 targets', '(37 pass every default filter', 'deep: 37 of 120',
                                      'Stokes V: top 0 of 0', '<p>None.</p>', 'Stokes I: top 1 of 37', '12.0')]
        assert order == sorted(order), order
        assert '<td>7.3</td><td>&ndash;</td>' in h and '<td>12.0</td><td>&ndash;</td>' in h   # a missing value is a dash
        assert '04:10:12.0 -00:30:00' in h, h
        assert 'HD 189733 <b>' not in h
        none = block.build('04h_2024-12-19', 'CHAIN', [], [], 'No target detected (40 measured).', [('V', [], 0), ('I', [], 0)],
                           0, {}, 20)
        assert 'Limits and missing inputs' not in none and '<p>None.</p>' in none and 'No target detected' in none
        assert block.sexagesimal(359.99999, 89.99999) == '00:00:00.0 +90:00:00', block.sexagesimal(359.99999, 89.99999)
        assert block.sexagesimal(None, 1) == '&ndash;'
    finally:
        shutil.rmtree(d, ignore_errors=True)


STUB_FILTERS = """
FP_FILE, FP_META = 'fp_filters.csv', 'fp_filters.json'
SCINT_FILE, SCINT_META = 'scintillation_index.csv', 'scintillation_index.json'
SCINT_STATES = ['worst 10% - hidden by default', 'below', 'not measured']
SCINT_FROM = {'worst 10% of its band': SCINT_STATES[0], 'other': SCINT_STATES[1], 'not measured': SCINT_STATES[2]}
SCINT_CAT = dict(key='fp_c_scint', states=SCINT_STATES, default_off=[SCINT_STATES[0]])
NO_DEFAULT_CATS = frozenset(('fp_c_track', 'fp_c_flood'))
def passes(dets, flt):
    if not flt:
        return True
    return any(all(d.get(k) in rng for k, rng in flt.items()) for d in dets)
"""


def _w(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(text)


def test_sources_read_served_tables_apply_page_defaults_and_group_targets():
    """lib/notify/sources.py on synthetic tables of both layouts (members + survivors: deep, 10-minute; page index:
    10-s, 1-minute) and a forced event list: only served identities, only detections that pass the page's default
    filters, exempt categories ignored, one row per identity, targets grouped with their association verdict."""
    import gzip
    from exopipe.phase3.config import CFG
    from exopipe.phase3.lib.notify import sources
    d = tempfile.mkdtemp(prefix='test_email_')
    old = {k: getattr(CFG.notify, k) for k in ('REGISTRY', 'PAGES')}
    old_root = CFG.paths.SNAPSHOT_REVIEW_ROOT
    try:
        code = os.path.join(d, 'code', 'classify_transients.py')
        _w(code, '')
        _w(os.path.join(d, 'code', 'review_filters.py'), STUB_FILTERS)
        r1, r2, rev = os.path.join(d, 'run_deep'), os.path.join(d, 'run_snap'), os.path.join(d, 'review')
        c1, c2 = os.path.join(r1, 'source_catalogs_V'), os.path.join(r2, 'source_catalogs_V_10s')
        _w(os.path.join(c1, '41MHz_source_members.csv'), 'source_id,band,cand_id,det_ra,det_dec,det_snr,det_date\n'
           'J0100+1000,41MHz,VD_04h_2024-12-19_41MHz_0001,15.0,10.0,9.5,2024-12-19\n'       # served, passes
           'J0100+1000,41MHz,VD_04h_2024-12-20_41MHz_0001,15.0,10.0,30.0,2024-12-20\n'      # another night
           'J0200+2000,41MHz,VD_04h_2024-12-19_41MHz_0002,30.0,20.0,12.0,2024-12-19\n'      # served, hidden by a default
           'J0300+3000,41MHz,VD_04h_2024-12-19_41MHz_0003,45.0,30.0,50.0,2024-12-19\n'      # not served
           'J0400+4000,41MHz,VD_04h_2024-12-19_41MHz_0004,60.0,40.0,7.0,2024-12-19\n')      # served, exempt category only
        _w(os.path.join(c1, '41MHz_sources_survivors.csv'),
           'source_id,identity_id,ra,dec,promotion,multi_integration,xm_best,xm_offset_pct,xm_p_chance,xm_ambiguous\n'
           'J0100+1000,J0100+1000,15.001,10.001,REP,1,10pc:GJ 1000,40,0.002,0\n'
           'J0200+2000,J0200+2000,30.0,20.0,,0,,,,\nJ0400+4000,,60.0,40.0,,0,ATNF_psr:PSR J0400+4000,5,0.0001,0\n')
        _w(os.path.join(r1, 'master_catalog.csv'), 'cand_id,stokes,global_snr\nVD_04h_2024-12-19_41MHz_0001,V,11.25\n')
        json.dump({'cats': [{'key': 'fp_c_zone', 'states': ['in zone', 'clear', 'not evaluated'], 'default_off': ['in zone']},
                            {'key': 'fp_c_track', 'states': ['tracked', 'clear'], 'default_off': ['tracked']},
                            {'key': 'fp_c_iony', 'stokes': 'I', 'states': ['bad', 'ok'], 'default_off': ['bad']}]},
                  open(os.path.join(r1, 'fp_filters.json'), 'w'))
        _w(os.path.join(r1, 'fp_filters.csv'), '# comment\ncand_id,ra,dec,fp_c_zone,fp_c_track,fp_c_iony\n'
           'VD_04h_2024-12-19_41MHz_0002,30.0,20.0,in zone,clear,ok\n'
           'VD_04h_2024-12-19_41MHz_0004,60.0,40.0,clear,tracked,bad\n')
        # the page-index layout: one served candidate with two detections of the hour (one in a hidden scintillation hour-band)
        det_head = 'lst,date,band,stokes,ts,snr,ra,dec,tier,cand_id\n'
        det = (det_head + '15h,2025-05-01,27MHz,V,20250501_083820,-8.5,170.0,31.0,10s,15h_2025-05-01_27MHz_V_10s_20250501_083820\n'
               '15h,2025-05-01,27MHz,V,20250501_083900,-5.0,170.0,31.0,1min,15h_2025-05-01_27MHz_V_1min_20250501_083900\n'
               '16h,2025-05-01,27MHz,V,20250501_093820,-99.0,170.0,31.0,10s,16h_2025-05-01_27MHz_V_10s_20250501_093820\n')
        det23 = det_head + '15h,2025-05-01,23MHz,V,20250501_083830,-20.0,170.0,31.0,10s,15h_2025-05-01_23MHz_V_10s_20250501_083830\n'
        os.makedirs(os.path.join(r2, 'page_index'))
        _w(os.path.join(r2, 'page_index', 'manifest.json'), '{}')
        with gzip.open(os.path.join(r2, 'page_index', 'shard_000.tsv.gz'), 'wt') as f:
            f.write('candidates/V/J1120p3100_27MHz\t[[1,2]]\t' + json.dumps({'d': True, 't': det}) + '\n')
            f.write('candidates/V/J1120p3100_23MHz\t[[1,2]]\t' + json.dumps({'d': True, 't': det23}) + '\n')
            f.write('candidates/V/J0000p0000_27MHz\t[[1,2]]\t' + json.dumps({'d': True, 't': det_head}) + '\n')
        _w(os.path.join(c2, '27MHz_sources_survivors.csv'), 'rank,source_id,identity_id,review_dir,xm_best,multi_integration\n'
           'rank0001,J1120+3100_27MHz,J1120+3100,candidates/V/J1120p3100_27MHz,'
           '"Hot_Warm_Jupiters:HIP 57274(5.7 arcmin, 0.19 beam, pct 25, E=0.360, p=1.3e-04)",1\n')
        _w(os.path.join(c2, '23MHz_sources_survivors.csv'), 'rank,source_id,identity_id,review_dir,xm_best,multi_integration\n'
           'rank0001,J1120+3100_23MHz,J1120+3100,candidates/V/J1120p3100_23MHz,'
           '"Hot_Warm_Jupiters:HIP 57274(9.9 arcmin, 0.30 beam, pct 60, E=0.9, p=2.0e-03)",1\n')
        json.dump({'default_hidden': True}, open(os.path.join(r2, 'scintillation_index.json'), 'w'))
        _w(os.path.join(r2, 'scintillation_index.csv'), 'lst,date,band,scint_index,state\n15h,2025-05-01,23MHz,0.9,worst 10% of its band\n'
           '15h,2025-05-01,27MHz,0.1,other\n')
        fr = os.path.join(rev, 'Run_SNAPFORCED_V10s_gen5')
        _w(os.path.join(fr, 'forced_assoc_events.csv'),
           'cand_id,identity_id,sample,target,band,tier,ts,snr,identity_label,sep_target_arcsec,pct_target,bg_name,pct_bg\n'
           'F_15h_2025-05-01_27MHz_20250501_083820_HIP_57274,HIP_57274,HWJ,HIP_57274,27MHz,10s,20250501_083820,-6.5,target,30,17,,\n'
           'F_15h_2025-05-01_27MHz_20250501_083830_HIP_57274,HIP_57274,HWJ,HIP_57274,27MHz,10s,20250501_083830,-9.0,target,30,12,,\n'
           'F_15h_2025-05-01_27MHz_20250501_084000_GJ_1,GJ_1,10pc,GJ_1,27MHz,10s,20250501_084000,7.0,background source,300,99,VLSSr J1,3\n'
           'F_15h_2025-05-01_27MHz_20250501_084000_X_1,X_1,flare,X_1,27MHz,10s,20250501_084000,70.0,target,3,9,,\n'
           'F_15h_2025-05-02_27MHz_20250502_084000_HIP_57274,HIP_57274,HWJ,HIP_57274,27MHz,10s,20250502_084000,-60.0,target,3,9,,\n')
        _w(os.path.join(fr, 'forced_assoc_identity.csv'), 'identity_id,label,target_pcc\nHIP_57274,target,0.00013\n')
        _w(os.path.join(fr, 'forced_fits.csv'), 'cand_id,fit_peak_jyb\nF_15h_2025-05-01_27MHz_20250501_083830_HIP_57274,-7.5\n')
        reg = os.path.join(d, 'registry.json')
        page = lambda i, run, cat: {'id': i, 'code': code, 'argv': ['--run', run] + (['--catalog-dir', cat] if cat else [])}
        json.dump({'pages': [page('pD', r1, c1), page('pS', r2, c2), page('pF', fr, None)]}, open(reg, 'w'))
        CFG.notify.REGISTRY, CFG.notify.PAGES = reg, {('deep', 'V'): 'pD', ('10-s', 'V'): 'pS', ('10-minute', 'I'): 'pGone'}
        CFG.paths.SNAPSHOT_REVIEW_ROOT = rev
        sources._CACHE.clear(); sources._INDEX_CACHE.clear()
        parts, n_pass, per_tier, limits = sources.transients('04h_2024-12-19', CFG, 10)
        (sv, rows_v, nv), (si, rows_i, ni) = parts
        assert (sv, si, nv, ni, n_pass) == ('V', 'I', 2, 0, 2) and rows_i == [], parts
        assert [(r['link_text'], r['snr_local'], r['snr_formal'], r['rank']) for r in rows_v] == \
            [('J0100+1000', 9.5, 11.25, 1), ('J0400+4000', 7.0, None, 2)], rows_v     # 0002 hidden, 0003 not served
        assert rows_v[0]['cls'] == 'REP, multi-integration' and rows_v[0]['link'].endswith('/p/pD/source/J0100+1000?b=41MHz')
        assert (rows_v[0]['ra_deg'], rows_v[0]['time_utc']) == (15.001, '2024-12-19, LST 04h hour'), rows_v[0]
        assert per_tier == {'deep Stokes V': (3, 2)}, per_tier
        assert any('pGone' in x for x in limits) and any('10-s Stokes V (page pS)' in x for x in limits), limits
        tg, note, lim = sources.targets('04h_2024-12-19', CFG)
        assert [(t['target'], t['sample'], t['found_by'], t['offset_percentile'], t['p_chance']) for t in tg] == \
            [('GJ 1000', '10 pc', 'untargeted search', 40.0, 0.002)], tg                # the pulsar match is not a target
        assert any('no served event list' in x for x in lim) and '0 forced-photometry events' in note, (lim, note)
        # the page-index layout, and the forced list of the same hour
        parts, n_pass, per_tier, limits = sources.transients('15h_2025-05-01', CFG, 10)
        rows_v = parts[0][1]
        assert n_pass == 1 and [(r['link_text'], r['band'], r['snr_local'], r['time_utc']) for r in rows_v] == \
            [('J1120+3100', '27MHz', -8.5, '2025-05-01 08:38:20')], rows_v             # 23 MHz: hidden scintillation hour
        assert per_tier == {'10-s Stokes V': (1, 1)}, per_tier
        tg, note, lim = sources.targets('15h_2025-05-01', CFG)
        assert [(t['target'], t['found_by'], t['snr_formal'], t['flux_jy'], t['offset_percentile'], t['p_chance']) for t in tg] == \
            [('HIP 57274', 'forced photometry', -9.0, -7.5, 12.0, 0.00013),
             ('HIP 57274', 'untargeted search', None, None, 25.0, 0.00013)], tg
        assert '2 events of this target' in tg[0]['note'] and tg[0]['link'].endswith('/p/pF/source/HIP_57274'), tg[0]
        assert '3 forced-photometry events' in note and '1 given to a catalogued background source' in note, note
        assert tg[1]['snr_local'] == -8.5 and 'hidden' not in tg[1]['note'] and '2 detections' in tg[1]['note'], tg[1]
    finally:
        for k, v in old.items():
            setattr(CFG.notify, k, v)
        CFG.paths.SNAPSHOT_REVIEW_ROOT = old_root
        sources._CACHE.clear(); sources._INDEX_CACHE.clear()
        shutil.rmtree(d, ignore_errors=True)


def _all_tests():
    return sorted((n, f) for n, f in globals().items() if n.startswith('test_') and callable(f))


def main(argv):
    names = argv or [n for n, _ in _all_tests()]
    fails = 0
    for n, f in _all_tests():
        if n not in names:
            continue
        try:
            f()
            print(f'PASS  {n}')
        except Exception as e:                     # noqa: BLE001 - report every failure
            fails += 1
            import traceback
            traceback.print_exc()
            print(f'FAIL  {n}: {e}')
    print(f'{len([n for n, _ in _all_tests() if n in names]) - fails}/{len([n for n, _ in _all_tests() if n in names])} passed')
    return 1 if fails else 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
