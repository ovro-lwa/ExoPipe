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
