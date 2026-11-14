#!/usr/bin/env python3
"""STANDING TEST - the hour's ONE email (Gregg 2026-11-12: "one email at the end of Phase 3 that is close to identical to
the current one in what is included, just with the transient and forced photometry highlights added"), on synthetic data.

  mailer   the Phase 2 report rebuilt by a later process (restore_email_inputs) is the report the Phase 2 process itself
           would have sent: same body, same attachments in the same order; --defer_email records the command and the
           duration; the review copy (write_to) sends nothing. The one email (Gregg 2026-11-14): Phase 2's QA sections,
           then the Phase 3 QA percentiles, then the highlights; no "Transient Detections" section; the ionosphere
           numbers worded by their percentiles
  block    the Phase 3 highlights block: targets first, then the top transients; every limit and missing input is said in
           the block; Galactic latitude in bold near the plane
  sources  the listed rows on a stub of the page code: the five value filters, the page's defaults, its bright-source
           staircase and its pulsar match; the same value filters on the forced-photometry events
  qa       the Phase 3 QA percentile table (lib/qa/percentiles.py: computed, written as a product, its content test, a
           stale table not used) and the ionosphere percentiles
  served   the page rule on the real served deep Stokes V page, read only (skipped where the registry is absent)

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
            # the one email: the QA sections, then the Phase 3 QA percentiles, then the highlights; nothing else
            block = '<h4>Phase 3 highlights</h4><p>TEST BLOCK</p>'
            qa = '<h4>Phase 3 QA percentiles</h4><p>QA BLOCK</p>'
            agg.restore_email_inputs(rd)
            agg.send_email_report(rd, write_to=os.path.join(d, 'c'), highlights_html=block, phases='Phases 2-3',
                                  qa_extra_html=qa, ionosphere_rank={'disturbance': 'RMS 1.25 at percentile 40'}, **saved)
            sc, bc, nc = _parts(os.path.join(d, 'c.eml'))
            assert sc == sa.replace('Phase 2:', 'Phases 2-3:') and nc == na, (sc, nc)
            head = ba[:ba.index('<h4>Transient Detections</h4>')]
            assert "Disturbance: Good (RMS 1.25')" in head
            assert bc == (head.replace('Phase 2 Pipeline', 'Phases 2-3 Pipeline')
                          .replace("Good (RMS 1.25')", 'RMS 1.25 at percentile 40') + qa + block), bc
            assert 'Transient Detections' not in bc and 'No significant detections found' not in bc
            assert bc.index('<h4>Thermal Noise</h4>') < bc.index('<h4>Ionosphere Quality</h4>') < bc.index(qa) < bc.index(block)
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
        from exopipe.phase3.config import CFG
        trans.append(dict(trans[0], rank=2, gal_lat_deg=45.04, elevation_deg=61.4))
        trans[0].update(gal_lat_deg=-5.04, elevation_deg=40.2)
        counts = {'served': 120, 'candidates': 90, 'snr': 100, 'elev': 80, 'nve': 70, 'size': 60, 'axis': 50, 'defaults': 119,
                  'bright_pass': 110, 'bright_no_data': 2, 'pulsar': 3, 'pass': 40, 'listed': 37, 'n_max': 9.0,
                  'bright': {'mean_sky_loss': 0.0499, 'bands': 19, 'candidates_removed': 863, 'listed': 5280}}
        h = block.build('04h_2024-12-19', 'CHAIN', ['the 10-s table is missing'], targets, '1 detection of 40 targets',
                        [('V', [], 0), ('I', trans, 37)], 37, {'deep Stokes I': counts}, 20, cfg=CFG.notify)
        order = [h.index(x) for x in ('CHAIN', 'Limits and missing inputs', 'the 10-s table is missing',
                                      'Detections of Hot / Warm Jupiter and 10 pc sample targets', 'HD 189733 &lt;b&gt;',
                                      '1 detection of 40 targets', '(37 pass every filter)', '|S/N| above 7, elevation above 35 deg',
                                      'allowed mean sky loss of 5%',
                                      'deep Stokes I: 120 detections of 90 listed candidates', 'Passing S/N: 100; elevation: 80; '
                                      'count in the epoch image (at most 9): 70; size: 60; axis ratio: 50',
                                      'the bright-source filter: 110 (mean sky loss 5.0% over 19 searched bands; it removes 863 of '
                                      'the page\'s 5280 listed candidates; 2 of this hour\'s detections have no precomputed data '
                                      'for it and are never removed by it, as on the page)', 'known pulsar: 3. Passing all: 40 detections of 37 '
                                      'candidates', 'Stokes V: top 0 of 0', '<p>None.</p>', 'Stokes I: top 2 of 37', '12.0')]
        assert order == sorted(order), order
        assert '<td>7.3</td><td>&ndash;</td>' in h and '<td>12.0</td><td>&ndash;</td>' in h   # a missing value is a dash
        assert '<td><strong>-5.0</strong></td><td>40</td>' in h and '<td>+45.0</td><td>61</td>' in h, h   # bold below 10 deg
        assert '04:10:12.0 -00:30:00' in h, h
        assert 'HD 189733 <b>' not in h
        none = block.build('04h_2024-12-19', 'CHAIN', [], [], 'No target detected (40 measured).', [('V', [], 0), ('I', [], 0)],
                           0, {}, 20, cfg=CFG.notify)
        assert 'Limits and missing inputs' not in none and '<p>None.</p>' in none and 'No target detected' in none
        assert block.gal_lat({}, 10.0) == '&ndash;' and block.gal_lat({'gal_lat_deg': 10.0}, 10.0) == '+10.0'
        assert block.sexagesimal(359.99999, 89.99999) == '00:00:00.0 +90:00:00', block.sexagesimal(359.99999, 89.99999)
        assert block.sexagesimal(None, 1) == '&ndash;'
    finally:
        shutil.rmtree(d, ignore_errors=True)


STUB_FILTERS = """
def passes(dets, flt):
    if not flt:
        return True
    return any(all(d.get(k) in rng for k, rng in flt.items()) for d in dets)
"""

# A stub of the page server's script with the names lib/notify/page_rule.py uses. A page's data come from two files:
# <catalog-dir>/sources.json (its sources) and <run>/page_<P3_PAGE_ID>.json (its percentile breakpoints, defaults and
# staircase), so the page's environment must be set when its object is built.
STUB_PAGE = """
import json, os, types
BSF_DEFAULT_B, BSF_DEFAULT_X = (0.6, 0.0), (1.0, 2.5)
BSF_OPT_LEVELS, BSF_OPT_RSTEP, BSF_OPT_NR = (5, 10), 0.05, 200
PAGE = {}
class L(list):
    def tolist(self):
        return list(self)
def read_queue(path):
    return [(l.split()[0], 'p1') for l in open(path) if l.strip()], {}, {}
def load_sources(run, blind=False, catalog_dir=None, extra_dir=None):
    print('stub page: loading', catalog_dir)
    J = json.load(open(os.path.join(catalog_dir, 'sources.json')))
    return {sid: types.SimpleNamespace(detections=s['detections'], bands=s['bands']) for sid, s in J.items()}
def bsf_path(app):
    return os.path.join(app.run_dir, 'bright_source_filter.npz')
class App:
    MCHAIN_RAW, FILTER_DEFAULTS = None, True
    def __init__(self, run_dir, sources, queue_entries, queue_prov, labels_dir, panel_format, hide_gate=False, entry_sel=None):
        import review_filters as RF
        PAGE.update(json.load(open(os.path.join(run_dir, 'page_' + os.environ['P3_PAGE_ID'] + '.json'))))
        self.run_dir, self.sources, self.RF, self.mstokes = run_dir, sources, RF, PAGE['stokes']
        if PAGE.get('stale'):
            print('bright-source filter: x.npz: 1 parent catalogs changed or missing since it was made (first c) - '
                  'their 1 cells never remove a detection and leave the sky loss; the other 3 cells exact')
        self.queue_ids = [s for s, p in queue_entries if s in sources]
        self.mdets = {sid: [d['m'] for d in s.detections] for sid, s in sources.items()}
        self.metrics_json = json.dumps({'ref': PAGE['ref']}).encode()
        self.fp_default_allowed = PAGE['defaults']
        sids = list(sources)
        self.bsf = None if PAGE.get('no_bsf') else dict(
            BP=types.SimpleNamespace(psr_match=lambda r: r.get('credible') == '1'), sids=sids, n_miss=0,
            ok=L((s, p) not in {tuple(x) for x in PAGE.get('no_data', [])}
                 for s in sids for p in range(len(sources[s].detections))),
            d_src=L(i for i, s in enumerate(sids) for _ in sources[s].detections),
            d_pos=L(p for s in sids for p in range(len(sources[s].detections))))
def _bsf_opt_prep(app, key, levels, rstep, nr):
    assert key == (0.0, 0.6, 0.0, False, 1.0, 2.5, False), key
    return {'n_listed': len(app.queue_ids)}
def bsf_search(prep, loss, n_starts=24, budget_s=10.0):
    assert budget_s == float('inf') and loss == 0.05, (budget_s, loss)
    return 'K', PAGE['n_removed'], loss, [], 'greedy', 1
def bsf_steps_of(prep, K):
    return PAGE['steps']
def bsf_sky_browser(app, steps, a, bh, bl):
    return {'41MHz': PAGE['mean_loss'], '46MHz': PAGE['mean_loss'], 'Full': None}
def bsf_det_state(P, key):
    gone = {tuple(x) for x in PAGE['removed']}
    rem = L((P['sids'][s], p) in gone for s, p in zip(P['d_src'], P['d_pos']))
    return rem, rem, rem
"""


def _w(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(text)


def _stub_pages(d):
    """A stub page code, two pages (deep Stokes V 'pD', 10-minute Stokes I 'pT') and their registry under d; the hour is
    04h 2024-12-19. Returns (registry path, run of pD, forced review root)."""
    code = os.path.join(d, 'code', 'classify_transients.py')
    _w(code, STUB_PAGE)
    _w(os.path.join(d, 'code', 'review_filters.py'), STUB_FILTERS)
    r1, r2, rev = os.path.join(d, 'run_deep'), os.path.join(d, 'run_tenmin'), os.path.join(d, 'review')
    good = dict(snr=9.5, elev=60.0, nve=12.0, size=1.0, axis=1.2, glat=5.04, fp_c_zone='clear')

    def det(n, band='41MHz', date='2024-12-19', lst='04h', snr=9.5, **m):
        return dict(cand_id=f'VD_{lst}_{date}_{band}_{n:04d}', lst=lst, date=date, band=band, snr=snr, ra=15.0, dec=10.0,
                    m=dict(good, snr=abs(snr), **m))

    def src(dets, **row):
        return dict(detections=dets, bands={b: dict(dict(ra='15.001', dec='10.001'), **row) for b in {x['band'] for x in dets}})
    S = {
        'J0100+1000': src([det(1), det(1, date='2024-12-20', snr=30.0)], promotion='REP', multi_integration='1',
                          xm_best='10pc:GJ 1000', xm_offset_pct='40', xm_p_chance='0.002', xm_ambiguous='0'),
        'J0200+2000': src([det(2, snr=12.0, fp_c_zone='in zone')], xm_best='10pc:GJ 2000'),      # hidden by a page default
        'J0300+3000': src([det(3, snr=50.0)]),                                                   # not in the page's queue
        'J0400+4000': src([det(4, snr=40.0)], credible='1', xm_best='ATNF_psr:PSR J0400+4000'),  # a known pulsar
        'J0500+5000': src([det(5, snr=41.0, elev=35.0)]),                                        # elevation not above 35
        'J0600+6000': src([det(6, snr=42.0, nve=41.0)]),                                         # busy epoch image
        'J0700+7000': src([det(7, snr=43.0, size=None)]),                                        # no fitted size
        'J0800+8000': src([det(8, snr=44.0)]),                                                   # removed by the staircase
        'J0900+9000': src([det(9, snr=-7.0)]),                                                   # |S/N| not above 7
        'J1000+1000': src([det(10, snr=-8.0)], credible='1', xm_best='LPT:GLEAM-X J1'),          # a long-period transient
        'J1100+1100': src([det(11, snr=20.0, glat=-45.0), det(11, band='46MHz', snr=15.0), det(12, snr=18.0, axis=1.75)],
                          xm_best='Hot_Warm_Jupiters:HIP 1', xm_offset_pct='70', xm_p_chance='0.01', xm_ambiguous='1'),
    }
    c1, c2 = os.path.join(r1, 'cat_V'), os.path.join(r2, 'cat_I')
    _w(os.path.join(c1, 'sources.json'), json.dumps(S))
    _w(os.path.join(c2, 'sources.json'), json.dumps({'J2000+2000': src([det(1, band='Red', date='2024-12-19T03', snr=8.0)])}))
    _w(os.path.join(r1, 'queue.txt'), ''.join(f'{s}\n' for s in S if s != 'J0300+3000') + 'J9999+9999\n')
    _w(os.path.join(r2, 'queue.txt'), 'J2000+2000\n')
    ref = {'nve': [float(i) * 40 / 90 for i in range(101)]}                 # the 90th percentile breakpoint: 40
    json.dump(dict(stokes='V', ref=ref, defaults={'fp_c_zone': ['clear', 'not evaluated']}, removed=[['J0800+8000', 0]],
                   steps=[[1.0, 5.0]], mean_loss=0.0499, n_removed=1), open(os.path.join(r1, 'page_pD.json'), 'w'))
    json.dump(dict(stokes='I', ref=ref, defaults={}, removed=[], steps=[], mean_loss=0.0, n_removed=0),
              open(os.path.join(r2, 'page_pT.json'), 'w'))
    _w(os.path.join(r1, 'master_catalog.csv'), 'cand_id,lst,date,band,stokes,n_v_epoch,global_snr\n'
       'VD_04h_2024-12-19_41MHz_0001,04h,2024-12-19,41MHz,V,12,11.25\n'
       'VD_04h_2024-12-19_41MHz_0002,04h,2024-12-19,41MHz,V,12,\n'
       'VD_04h_2024-12-19_46MHz_0011,04h,2024-12-19,46MHz,V,50,\n'
       'VD_04h_2024-12-20_41MHz_0001,04h,2024-12-20,41MHz,V,20,\n'
       'VD_05h_2024-12-19_41MHz_0001,05h,2024-12-19,41MHz,V,4,\n'
       'ID_04h_2024-12-19_41MHz_0001,04h,2024-12-19,41MHz,I,,\n')
    _w(os.path.join(r2, 'master_catalog.csv'), 'cand_id,lst,date,band,stokes,n_v_epoch,global_snr\n'
       'I1_04h_2024-12-19T03_Red_0001,04h,2024-12-19T03,Red,I,,\nI1_04h_2024-12-19T03_Red_0002,04h,2024-12-19T03,Red,I,,\n'
       'I1_04h_2024-12-19T04_Red_0001,04h,2024-12-19T04,Red,I,,\nI1_07h_2024-12-19T01_Red_0001,07h,2024-12-19T01,Red,I,,\n')
    img = 'lst,scope,stokes,date,n_ident5,n_after_morph,n_after_mstruct,n_after_jupiter,n_final,n_raw_floor,elev_min_deg\n'
    _w(os.path.join(r1, 'v_deep_images.csv'), img + '04h,41MHz,V,2024-12-19,90,80,12,12,12,90,25.0\n'
       '04h,41MHz,V,2024-12-20,90,80,20,20,20,95,25.0\n05h,41MHz,V,2024-12-19,90,80,4,4,4,60,25.0\n'
       '04h,46MHz,V,2024-12-19,90,80,0,0,0,70,25.0\n04h,46MHz,V,2024-12-20,90,80,,0,0,70,25.0\n'
       '04h,41MHz,I,2024-12-19,300,7,,7,7,900,25.0\n')
    _w(os.path.join(r2, 'v_deep_images.csv'), img + '04h,Red,I,2024-12-19T03,50,2,,2,2,400,25.0\n'
       '04h,Red,I,2024-12-19T04,50,1,,1,1,300,25.0\n04h,Red,I,2024-12-19T05,50,0,,0,0,200,25.0\n'
       '07h,Red,I,2024-12-19T01,50,1,,1,1,300,25.0\n04h,Red,V,2024-12-19T03,9,9,3,3,3,9,25.0\n')
    _w(os.path.join(r1, 'scintillation_index.csv'), 'lst,date,band,scint_index,n_sources,n_frames,pct_in_band,state\n'
       '04h,2024-12-19,41MHz,0.1234,50,300,61.5,other\n04h,2024-12-20,41MHz,0.9,50,300,99,other\n')
    fr = os.path.join(rev, 'Run_SNAPFORCED_Vdeep_gen5')
    ev = lambda who, sample, band, snr, label, ra='67.5', dec='37.0', date='2024-12-19': (     # noqa: E731
        f'F_04h_{date}_{band}_deep_{sample}_{who},{who},{sample},{who},{band},deep,20241219_120000,{snr},{label},{ra},{dec},30,17,,\n')
    _w(os.path.join(fr, 'forced_assoc_events.csv'),
       'cand_id,identity_id,sample,target,band,tier,ts,snr,identity_label,ra,dec,sep_target_arcsec,pct_target,bg_name,pct_bg\n'
       + ev('HIP_57274', 'HWJ', '41MHz', -9.0, 'target') + ev('HIP_57275', 'HWJ', '41MHz', -6.5, 'target')
       + ev('GJ_1', '10pc', '46MHz', 8.0, 'target')                      # its epoch image holds 50 detections
       + ev('GJ_2', '10pc', '41MHz', 9.0, 'target')                      # no fit
       + ev('GJ_3', '10pc', '41MHz', 10.0, 'target', dec='-30.0')        # elevation 23 deg
       + ev('GJ_4', '10pc', '41MHz', 12.0, 'background source')
       + ev('GJ_5', '10pc', '59MHz', 8.5, 'ambiguous', ra='', dec='')    # position from the fit; no count for its image
       + ev('X_1', 'flare', '41MHz', 70.0, 'target') + ev('HIP_57274', 'HWJ', '41MHz', -60.0, 'target', date='2024-12-20'))
    _w(os.path.join(fr, 'forced_assoc_identity.csv'), 'identity_id,label,target_pcc\nHIP_57274,target,0.00013\n')
    fit = lambda who, sample, band, extra='': f'F_04h_2024-12-19_{band}_deep_{sample}_{who},-7.5,1.1,1.3{extra}\n'   # noqa: E731
    _w(os.path.join(fr, 'forced_fits.csv'), 'cand_id,fit_peak_jyb,size_beams,axis_ratio,fit_ra_deg,fit_dec_deg\n'
       + ''.join(fit(*x) for x in (('HIP_57274', 'HWJ', '41MHz'), ('HIP_57275', 'HWJ', '41MHz'), ('GJ_1', '10pc', '46MHz'),
                                   ('GJ_3', '10pc', '41MHz'), ('GJ_4', '10pc', '41MHz')))
       + fit('GJ_5', '10pc', '59MHz', ',67.5,37.0'))
    reg = os.path.join(d, 'registry.json')
    page = lambda i, run, cat: {'id': i, 'code': code, 'env': {'P3_PAGE_ID': i},                     # noqa: E731
                                'argv': ['--run', run, '--queue', os.path.join(run, 'queue.txt'), '--catalog-dir', cat]}
    json.dump({'pages': [page('pD', r1, c1), page('pT', r2, c2), {'id': 'pF', 'code': code, 'argv': ['--run', fr]}]},
              open(reg, 'w'))
    return reg, r1, rev


class _Config:
    """config notify and paths pointed at the stub for the length of a test."""

    def __init__(self, reg, rev, pages, phase2_root=None, **notify):
        from exopipe.phase3.config import CFG
        from exopipe.phase3.lib.notify import qa, sources
        self.CFG, self.mods = CFG, (sources, qa)
        self.new = {'notify': {'REGISTRY': reg, 'PAGES': pages}, 'paths': {'SNAPSHOT_REVIEW_ROOT': rev}}
        self.new['notify'].update(notify)
        if phase2_root:
            self.new['paths']['PHASE2_ROOT'] = phase2_root

    def __enter__(self):
        self.old = {s: {k: getattr(getattr(self.CFG, s), k) for k in kv} for s, kv in self.new.items()}
        for s, kv in self.new.items():
            for k, v in kv.items():
                setattr(getattr(self.CFG, s), k, v)
        self._clear()
        return self.CFG

    def _clear(self):
        sources, qa = self.mods
        sources._CACHE.clear(); sources._RULES.clear(); qa._COUNTS.clear()

    def __exit__(self, *exc):
        for s, kv in self.old.items():
            for k, v in kv.items():
                setattr(getattr(self.CFG, s), k, v)
        self._clear()


def test_sources_apply_the_filters_of_the_list_through_the_page_code():
    from exopipe.phase3.lib.notify import sources
    d = tempfile.mkdtemp(prefix='test_email_')
    env0, path0 = os.environ.get('P3_PAGE_ID'), list(sys.path)
    try:
        reg, _r1, rev = _stub_pages(d)
        pages = {('deep', 'V'): 'pD', ('10-minute', 'I'): 'pT', ('10-minute', 'V'): 'pGone'}
        with _Config(reg, rev, pages, DEEP_MIN_FRACTION=0.0) as CFG:
            parts, n_pass, per_tier, limits = sources.transients('04h_2024-12-19', CFG, 10)
            (sv, rows_v, nv), (si, rows_i, ni) = parts
            assert (sv, si, nv, ni, n_pass) == ('V', 'I', 3, 1, 4), parts
            # most significant passing detection per candidate; the long-period transient stays, the pulsar does not
            assert [(r['link_text'], r['band'], r['snr_local'], r['snr_formal'], r['rank']) for r in rows_v] == \
                [('J1100+1100', '41MHz', 20.0, None, 1), ('J0100+1000', '41MHz', 9.5, 11.25, 2),
                 ('J1000+1000', '41MHz', -8.0, None, 3)], rows_v
            assert rows_v[1]['cls'] == 'REP, multi-integration' and rows_v[1]['link'].endswith('/p/pD/source/J0100+1000?b=41MHz')
            assert (rows_v[1]['ra_deg'], rows_v[1]['time_utc'], rows_v[1]['gal_lat_deg']) == \
                (15.001, '2024-12-19, LST 04h hour', 5.04), rows_v[1]
            assert rows_v[0]['cls'] == '2 bands this hour' and rows_v[0]['gal_lat_deg'] == -45.0, rows_v[0]
            assert [(r['link_text'], r['time_utc']) for r in rows_i] == \
                [('J2000+2000', '2024-12-19, 10-minute window 4 of 6 of LST 04h')], rows_i
            c = per_tier['deep Stokes V']
            assert {k: c[k] for k in ('served', 'candidates', 'snr', 'elev', 'nve', 'size', 'axis', 'defaults', 'bright_pass',
                                      'pulsar', 'pass', 'listed', 'n_max')} == \
                dict(served=12, candidates=10, snr=11, elev=11, nve=11, size=11, axis=11, defaults=11, bright_pass=11, pulsar=1,
                     **{'pass': 4}, listed=3, n_max=40.0), c
            assert c['bright']['mean_sky_loss'] == 0.0499 and c['bright']['bands'] == 2 and c['bright']['listed'] == 10, c
            assert per_tier['10-minute Stokes I']['pass'] == 1 and len(per_tier) == 2
            # at least half of the listed candidates of a Stokes are deep: no deep Stokes I candidate, so none is listed
            CFG.notify.DEEP_MIN_FRACTION = 0.5
            (_sv, rows_v2, _nv), (_si, rows_i2, ni2) = sources.transients('04h_2024-12-19', CFG, 10)[0]
            assert rows_v2 == rows_v and rows_i2 == [] and ni2 == 1, (rows_v2, rows_i2)
            R = lambda tiers: [{'tier': 'deep' if t == 'd' else '10-minute', 'rank': i + 1} for i, t in enumerate(tiers)]   # noqa: E731
            pick = lambda tiers, n, f=0.5: [r['rank'] for r in sources.at_least_deep(R(tiers), n, f)]                     # noqa: E731
            assert pick('ttttdddd', 4) == [1, 2, 5, 6]            # two 10-minute rows leave, the two best deep rows enter
            assert pick('ttdtttd', 4) == [1, 2, 3, 7]
            assert pick('tttd', 4) == [1, 4] and pick('tttt', 4) == [] and pick('dtdt', 4) == [1, 2, 3, 4]
            assert pick('tdt', 3) == [1, 2] and pick('tttt', 3, 0.0) == [1, 2, 3] and pick('', 4) == []
            CFG.notify.DEEP_MIN_FRACTION = 0.0
            assert len(limits) == 1 and 'pGone' in limits[0], limits
            # the page's environment and the module path are as before; each page had its own module
            assert os.environ.get('P3_PAGE_ID') == env0 and sys.path == path0
            assert sources._RULES['pD'].CT is not sources._RULES['pT'].CT and 'stub page: loading' in sources._RULES['pD'].log
            # targets: the forced events under the same value filters, then the passing cross-matched detections
            tg, note, lim = sources.targets('04h_2024-12-19', CFG)
            assert [(t['target'], t['found_by'], t['snr_formal'], t['flux_jy']) for t in tg] == \
                [('HIP 57274', 'forced photometry', -9.0, -7.5), ('GJ 5', 'forced photometry', 8.5, -7.5),
                 ('HIP 1', 'untargeted search', None, None), ('GJ 1000', 'untargeted search', 11.25, None)], tg
            assert tg[2]['note'].startswith('AMBIGUOUS association') and '2 detections' in tg[2]['note'], tg[2]
            assert tg[0]['link'].endswith('/p/pF/source/HIP_57274') and tg[0]['p_chance'] == 0.00013, tg[0]
            for text in ('7 forced-photometry events', '3 pass every filter', 'S/N above 7: 6', 'elevation above 35 deg: 6',
                         'fitted size below 1.75 beams: 6', 'fitted axis ratio below 1.75: 6',
                         '90th percentile: 6, of which 0 are 10-second or 1-minute events (no listed page, no count) and 1 are '
                         'of an epoch image without a catalog detection, and pass',
                         '1 events have no fitted size or axis ratio and fail, of which 0 have |S/N| at or below 7',
                         '1 given to a catalogued background source',
                         '2 served untargeted detections', '(2 more are cross-matched and do not pass)'):
                assert text in note, (text, note)
            assert sum('no served event list' in x for x in lim) == 7, lim
            assert 'because their page' not in note, note
            # the hour's detections without precomputed bright-source data are counted; the page's stale-catalog warning
            # is a limit
            J = json.load(open(os.path.join(d, 'run_deep', 'page_pD.json')))
            json.dump(dict(J, no_data=[['J0100+1000', 0], ['J0100+1000', 1]], stale=True),
                      open(os.path.join(d, 'run_deep', 'page_pD.json'), 'w'))
            sources._CACHE.clear(); sources._RULES.clear()
            parts, n_pass, per_tier, limits = sources.transients('04h_2024-12-19', CFG, 10)
            assert per_tier['deep Stokes V']['bright_no_data'] == 1 and n_pass == 4, per_tier
            assert any(x.startswith('deep Stokes V, page pD: bright-source filter:') and 'changed or missing' in x
                       for x in limits), limits
            # a page whose own Stokes is not the one config notify.PAGES names is refused
            CFG.notify.PAGES = {**pages, ('10-minute', 'V'): 'pT'}
            sources._CACHE.clear(); sources._RULES.clear()
            parts, n_pass, per_tier, limits = sources.transients('04h_2024-12-19', CFG, 10)
            assert any('10-minute Stokes V: page pT is a Stokes I page by its own code' in x for x in limits), limits
            assert list(per_tier) == ['deep Stokes V', '10-minute Stokes I'], per_tier
            CFG.notify.PAGES = pages
            # a page without a bright-source filter file is not listed, and says why
            _w(os.path.join(d, 'run_tenmin', 'page_pT.json'), json.dumps(dict(stokes='I', ref={}, defaults={}, no_bsf=True)))
            sources._CACHE.clear(); sources._RULES.clear()
            parts, n_pass, per_tier, limits = sources.transients('04h_2024-12-19', CFG, 10)
            assert n_pass == 3 and list(per_tier) == ['deep Stokes V'], per_tier
            assert any('10-minute Stokes I' in x and 'no readable bright-source filter file' in x for x in limits), limits
            # a staircase that removes more sky than allowed refuses the page
            J = json.load(open(os.path.join(d, 'run_deep', 'page_pD.json')))
            json.dump(dict(J, mean_loss=0.0502), open(os.path.join(d, 'run_deep', 'page_pD.json'), 'w'))
            sources._CACHE.clear(); sources._RULES.clear()
            parts, n_pass, per_tier, limits = sources.transients('04h_2024-12-19', CFG, 10)
            assert n_pass == 0 and any('removes 0.0502 of the searched sky' in x for x in limits), limits
            # ... and then the deep forced events have no count: they FAIL the count filter instead of passing it
            tg, note, lim = sources.targets('04h_2024-12-19', CFG)
            assert [t['target'] for t in tg if t['found_by'] == 'forced photometry'] == [], tg
            assert 'S/N above 7: 6' in note and '90th percentile: 0, of which 0 are 10-second' in note, note
            assert '7 deep or 10-minute events fail it because their page' in note, note
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_qa_percentile_table_and_ionosphere_percentiles():
    from exopipe.phase3.lib.notify import qa
    d = tempfile.mkdtemp(prefix='test_email_')
    try:
        assert qa.midrank(2, [1, 2, 2, 3]) == 50.0 and qa.midrank(1, [1, 2, 3]) == 100 / 6 and qa.midrank(None, [1]) is None
        assert qa.midrank(1, []) is None
        reg, _r1, rev = _stub_pages(d)
        root = os.path.join(d, 'phase2')
        rd = _run_dir(root)                                             # 04h 2024-12-19: disturbance RMS 1.25'
        for date, rms, var in (('2024-12-20', 0.30, 0.2), ('2024-12-21', 0.50, 0.4)):
            dd = os.path.join(root, '04h', date, 'Science_20260527_000000', 'Dewarp_Diagnostics')
            os.makedirs(dd)
            json.dump({'disturbance_rms_arcmin': rms}, open(os.path.join(dd, 'x_warp_stats_73MHz.json'), 'w'))
            json.dump({'mean_variability': var}, open(os.path.join(dd, 'x_ionosphere_variability_73MHz.json'), 'w'))
        os.makedirs(os.path.join(root, '05h', '2024-12-19', 'Science_20260527_000000'))      # no diagnostics yet
        pages = {('deep', 'V'): 'pD', ('10-minute', 'I'): 'pT', ('10-minute', 'V'): 'pGone'}
        with _Config(reg, rev, pages, phase2_root=root) as CFG:
            html, limits = qa.phase3_percentiles('04h_2024-12-19', CFG)
            assert html.startswith('<h4>Phase 3 QA percentiles</h4>') and len(limits) == 1 and 'pGone' in limits[0], limits
            # deep V, 41 MHz: counts after the master-structure gate 12 (this hour), 20, 4 -> mid-rank 50; 46 MHz: a
            # searched image without a candidate is in the table (count 0), one without a recorded count is not
            for text in ('<td>41MHz</td><td>12 (50)</td><td>&ndash;</td><td>0.123 (62)</td>',
                         '<td>46MHz</td><td>0 (50)</td><td>&ndash;</td><td>&ndash;</td>',
                         # 10-minute I, Red: searched windows of 2, 1 and 0 candidates this hour, 1 in another hour:
                         # the largest, 2 -> (3 + 0.5) / 4 = 88
                         '<td>Red</td><td>&ndash;</td><td>2 (88); 3 windows</td><td>&ndash;</td>',
                         'deep, Stokes V: 4 images; 10-minute, Stokes I: 4 images'):
                assert text in html, (text, html)
            assert html.index('<td>41MHz</td>') < html.index('<td>46MHz</td>') < html.index('<td>Red</td>')
            # the table as a product: written beside the per-image table, its content test passes, a reader takes the
            # file while it is current and computes again when the per-image table has moved
            from exopipe.phase3.core import validate
            from exopipe.phase3.lib.qa import percentiles
            r1 = os.path.join(d, 'run_deep')
            path, n = percentiles.write(r1, 'deep')
            assert n == 5 and os.path.basename(path) == 'qa_percentiles.csv', (path, n)
            first = open(path).read().split('\n')[1]
            assert first == f'04h,2024-12-19,41MHz,V,deep,12,after_mstruct,50.0,90,50.0,3,{percentiles.version(r1)}', first
            assert 'I,deep,7,after_morph,50.0,900,50.0,1,' in open(path).read()
            res = [f(path) for _n, f in validate.tests_for('qa_percentiles')]
            assert res and all(r[0] for r in res), res
            assert percentiles.read(r1, 'deep')[1] == 'table' and percentiles.read(r1, 'deep')[0] == percentiles.table(r1, 'deep')
            with open(os.path.join(r1, 'v_deep_images.csv'), 'a') as f:
                f.write('06h,41MHz,V,2024-12-19,90,80,30,30,30,99,25.0\n')
            assert percentiles.read(r1, 'deep')[1] == 'computed' and len(percentiles.read(r1, 'deep')[0]) == 6
            res = [f(path) for _n, f in validate.tests_for('qa_percentiles')]
            assert not res[0][0] and 'stale table' in res[0][2], res
            out = os.path.join(d, 'out')
            words, lim = qa.ionosphere_rank(rd, '04h_2024-12-19', CFG, out)
            assert words == {'disturbance': "RMS 1.25', percentile 83 of the 3 processed hours with this measurement (0 = the "
                                            "calmest hour on record, 100 = the most disturbed)"}, words
            assert len(lim) == 1 and 'variability' in lim[0], lim           # this hour's run has no variability file
            pop = json.load(open(os.path.join(out, 'notify', CFG.notify.IONOSPHERE_POPULATION)))
            assert sorted(pop) == ['04h_2024-12-19', '04h_2024-12-20', '04h_2024-12-21'], pop
            # an hour whose diagnostics are gone leaves the population; a rewritten file is read again
            shutil.rmtree(os.path.join(root, '04h', '2024-12-20'))
            json.dump({'disturbance_rms_arcmin': 0.2, 'median_shift_arcmin': 2.0, 'n_anchors': 800},
                      open(os.path.join(rd, 'Dewarp_Diagnostics', '2024-12-19_04h_warp_stats_73MHz.json'), 'w'))
            words, _lim = qa.ionosphere_rank(rd, '04h_2024-12-19', CFG, out)
            assert words['disturbance'].startswith("RMS 0.20', percentile 25 of the 2 processed hours"), words
            # an hour that gains its diagnostics, and an hour whose file is rewritten, are read again
            dd = os.path.join(root, '05h', '2024-12-19', 'Science_20260527_000000', 'Dewarp_Diagnostics')
            os.makedirs(dd)
            json.dump({'disturbance_rms_arcmin': 0.1}, open(os.path.join(dd, 'x_warp_stats_73MHz.json'), 'w'))
            json.dump({'disturbance_rms_arcmin': 0.15}, open(os.path.join(
                root, '04h', '2024-12-21', 'Science_20260527_000000', 'Dewarp_Diagnostics', 'x_warp_stats_73MHz.json'), 'w'))
            words, _lim = qa.ionosphere_rank(rd, '04h_2024-12-19', CFG, out)
            assert words['disturbance'].startswith("RMS 0.20', percentile 83 of the 3 processed hours"), words
            pop = json.load(open(os.path.join(out, 'notify', CFG.notify.IONOSPHERE_POPULATION)))
            assert sorted(pop) == ['04h_2024-12-19', '04h_2024-12-21', '05h_2024-12-19'] and \
                pop['04h_2024-12-21']['var'] == 0.4 and pop['04h_2024-12-21']['rms'] == 0.15, pop
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_qa_percentile_blank_count_after_an_empty_morphology_gate_is_zero():
    # the Stokes V search records no master-structure count for an image with nothing left after the morphology gate
    # (Run_SLSTB: 319 of 31,078 V images): its count is 0 and it is in the population; any other blank count is left out
    from exopipe.phase3.lib.qa import percentiles
    d = tempfile.mkdtemp(prefix='test_email_')
    try:
        _w(os.path.join(d, 'v_deep_images.csv'),
           'lst,scope,stokes,date,n_after_morph,n_after_mstruct,n_raw_floor\n'
           '15h,41MHz,V,2025-05-01T00,0,,0\n'           # nothing after the morphology gate: count 0
           '15h,41MHz,V,2025-05-01T01,6,2,9\n'
           '15h,41MHz,V,2025-05-01T02,6,,9\n'           # a count that is not recorded: left out
           '15h,41MHz,V,2025-05-01T03,,,9\n'            # no count at all: left out
           '15h,41MHz,I,2025-05-01T00,,,40\n')          # Stokes I counts after the morphology gate: blank = left out
        rows = percentiles.table(d, percentiles.TENMIN)
        got = [(r['date'], r['stokes'], r['count'], r['count_kind'], r['percentile'], r['population']) for r in rows]
        assert got == [('2025-05-01T00', 'V', 0, 'after_mstruct', 25.0, 2),
                       ('2025-05-01T01', 'V', 2, 'after_mstruct', 75.0, 2)], got
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_qa_percentile_review59_optional_raw_count_refusal_and_no_stray_tmp():
    # review of PR 59: n_raw_floor is reference only (an older per-image table without it gives count_raw ''); the chain's
    # call raises core.record.Refusal itself; a write that fails leaves no tmp file beside the run's products
    from exopipe.phase3.core.record import Refusal
    from exopipe.phase3.lib.qa import percentiles
    d = tempfile.mkdtemp(prefix='test_email_')
    try:
        _w(os.path.join(d, 'v_deep_images.csv'),
           'lst,scope,stokes,date,n_after_morph,n_after_mstruct\n'
           '15h,41MHz,V,2025-05-01T00,3,1\n15h,41MHz,V,2025-05-01T01,6,2\n')
        path = percentiles.write_for_chain(d, percentiles.TENMIN)
        rows = percentiles.read(d, percentiles.TENMIN)
        assert rows[1] == 'table' and [(r['count'], r['count_raw'], r['percentile_raw']) for r in rows[0]] == [(1, '', ''), (2, '', '')], rows
        _w(os.path.join(d, 'v_deep_images.csv'), 'lst,scope,stokes,date,n_after_morph,n_after_mstruct\n15h,41MHz,V,x,7,\n')
        try:
            percentiles.write_for_chain(d, percentiles.TENMIN)
            why = ''
        except Refusal as e:
            why = str(e)
        assert 'no searched image with a recorded count' in why and not os.path.exists(path), why
        _w(os.path.join(d, 'v_deep_images.csv'), 'lst,scope,stokes,n_after_morph\n')
        try:
            percentiles.write_for_chain(d, percentiles.TENMIN)
            why = ''
        except Refusal as e:
            why = str(e)
        assert 'lacks' in why and 'n_raw_floor' not in why, why
        _w(os.path.join(d, 'v_deep_images.csv'), 'lst,scope,stokes,date,n_after_morph,n_after_mstruct\n15h,41MHz,V,x,3,1\n')
        os.makedirs(os.path.join(d, 'blocked'))
        try:
            percentiles.write(d, percentiles.TENMIN, path=os.path.join(d, 'blocked'))   # os.replace onto a directory fails
            failed = False
        except OSError:
            failed = True
        assert failed and not [f for f in os.listdir(d) if '.tmp' in f], os.listdir(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_page_rule_on_the_served_deep_page_readonly():
    """The rule of the real served deep Stokes V page, built from its own code (read only): the staircase holds the
    allowed mean sky loss, and the process is left as it was. Skipped where the registry or the page is absent."""
    from exopipe.phase3.config import CFG
    from exopipe.phase3.lib.notify import page_rule, sources
    try:
        p = sources._registry(CFG).get(CFG.notify.PAGES[('deep', 'V')])
    except (OSError, ValueError, KeyError):
        p = None
    if not p or not p.get('code') or not os.path.exists(p['code']) or not os.path.isdir(p['run'] or ''):
        print('      (skipped: the served deep Stokes V page is not on this node)')
        return
    env0, path0, mods0 = dict(os.environ), list(sys.path), set(sys.modules)
    rule = page_rule.PageRule(p['entry'], CFG.notify.BRIGHT_SKY_LOSS)
    assert dict(os.environ) == env0 and sys.path == path0
    here = os.path.dirname(os.path.abspath(p['code']))
    assert not [m for m in set(sys.modules) - mods0
                if os.path.dirname(str(getattr(sys.modules[m], '__file__', None) or '')) == here]
    b = rule.bright
    assert 0.8 * b['allowed'] < b['mean_sky_loss'] <= b['allowed'] + 1e-6 and b['bands'] >= 15, b
    assert 0 < b['candidates_removed'] < b['listed'] and 0 < b['detections_removed'] == len(rule.removed), b
    assert rule.stokes == 'V' and all(len(rule.ref[k]) == 101 for k in ('snr', 'elev', 'nve', 'size', 'axis')), rule.ref.keys()
    sid = next(iter(rule.app.queue_ids))
    d0 = rule.app.sources[sid].detections[0]
    hour = rule.hour(d0['lst'], str(d0['date'])[:10])
    assert hour and all({'snr', 'elev', 'nve', 'size', 'axis', 'glat'} <= set(m) for _s, _p, _d, m, _r in hour)
    assert isinstance(rule.passes_defaults(hour[0][3]), bool) and isinstance(rule.pulsar(sid), bool)


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
