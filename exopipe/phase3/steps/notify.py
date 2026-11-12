"""exopipe3 notify - the ONE email of an LST hour of a night, sent at the end of Phase 3.

Gregg 2026-11-12: "one email at the end of Phase 3 that is close to identical to the current one in what is included, just
with the transient and forced photometry highlights added." The email is Phase 2's report (thermal noise, ionosphere
quality, Phase 2's transient summary, the same attachments in the same order: phase2_science_agg.send_email_report) with
one block placed above its sections, the Phase 3 highlights of the hour:
  1. the state of the hour's Phase 3 chain, and every limit or missing input;
  2. every DETECTION of a target of the Hot / Warm Jupiter and 10 pc samples, in forced photometry or as an untargeted
     candidate associated with the target (association through core/astrometry: offset percentile and chance-coincidence
     probability, no radius);
  3. the most significant transient candidates of the hour that pass every default filter of their served page: Stokes V
     first, then Stokes I (notify.TOP_N of each, default 10).
Nothing is computed: the block is read from the tables the hour's searches, catalogs, forced photometry and pages wrote.

    exopipe3 notify --epochs 04h_2025-10-14                               # the closing step of the per-hour chain: sends when
                                                                          # notify.SEND is True, else writes <out>/notify/<item>.eml
    exopipe3 notify --epochs 04h_2025-10-14 --write-to /fast/gh/tmp/mail  # review copy: nothing is sent or posted
    exopipe3 notify --epochs 04h_2025-10-14 --chain-record <jobs>/<job>/items/04h_2025-10-14.json

Unit of work: one hour (item <LST>_<date>); record <out>/_records/notify/<item>.json.
Inputs:  the hour's Phase 2 run directory (core/inputs.latest_science_dir: QA/, Dewarp_Diagnostics/, detections/, the
         detection tables, QA/email_deferred.json when Phase 2 ran with --defer_email); the tables named in config section
         `notify` (one reader per tier in lib/notify/sources.py); the chain's item record (--chain-record; runner.CHAIN_FINALLY).
Outputs: the email (and the Slack post of the Phase 2 mailer, when a webhook is configured), or with --write-to the files
         <prefix>_<item>.eml and <prefix>_<item>.html; <out>/notify/<item>_highlights.html (the block alone, always).
Failure modes (exit 2 = refused; 3 = no input; 1 = error):
  N1  the hour has no Phase 2 run directory: no report can be built (exit 3; nothing is sent).
  N2  a table a tier's reader needs is missing or lacks a column: NOT a refusal - the block says which tier is missing
      and why, and the email is sent (a failed or refused hour is reported, never silent).
  N3  the mailer raises: exit 1 (the block is kept under <out>/notify/).
Per hour: the closing step of the 'epoch' chain (runner.CHAIN_FINALLY): it runs whatever the chain's outcome. Phase 2 runs
with --defer_email in that mode, so the hour sends one email, not two.
"""
import os
import sys

SUMMARY = "the hour's one email: the Phase 2 report plus the Phase 3 highlights (target detections, top transients)"
TIERS = ()
SELECTORS = ('epochs',)
CONFIG = ('notify', 'paths', 'runner')
WRITES = True


def add_arguments(p):
    g = p.add_argument_group('notify')
    g.add_argument('--chain-record', metavar='FILE', help="the per-hour chain's item record (runner.CHAIN_FINALLY passes it)")
    g.add_argument('--write-to', metavar='PREFIX', help='review copy: write <PREFIX>_<item>.eml and .html; send nothing')
    g.add_argument('--top', type=int, help='number of transient candidates listed (default notify.TOP_N)')
    g.add_argument('--phase2-run', metavar='DIR', help="the hour's Phase 2 run directory (default: the latest Science_* "
                                                       'directory of the hour under paths.PHASE2_ROOT); one --epochs item only')


def default_out(args):
    from exopipe.phase3.config import CFG
    return CFG.notify.OUT_ROOT


def run(args, run):
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import cells, inputs, record
    from exopipe.phase3.lib.notify import block, sources
    if not args.epochs:
        raise record.Refusal('notify needs --epochs <LST>_<date>')
    items = cells.expand_epochs(args.epochs)
    if args.phase2_run and len(items) != 1:
        raise record.Refusal('--phase2-run names one run directory: give one --epochs item')
    top_n = CFG.notify.TOP_N if args.top is None else args.top
    for item in items:
        lst, date = cells.parse_epoch(item)
        rd = args.phase2_run or inputs.latest_science_dir(lst, date)
        if run.dry_run:
            print(f'{item}: would build the highlights block and '
                  + (f'write {args.write_to}_{item}.eml/.html' if args.write_to else 'SEND the email' if CFG.notify.SEND
                     else 'write it under <out>/notify/ (notify.SEND is False: nothing is sent)')
                  + f' (Phase 2 run {rd})')
            continue
        with run.unit(item) as u:
            if not rd or not os.path.isdir(rd):
                raise record.NoInput(f'N1: no Phase 2 run directory for {item} under {CFG.paths.PHASE2_ROOT}')
            u.add_input(rd)
            chain_text, _ok = block.chain_summary(args.chain_record)
            limits = []
            targets, target_note, lim = sources.targets(item, CFG)
            limits += lim
            transients, n_pass, per_tier, lim = sources.transients(item, CFG, top_n)
            limits += lim
            html = block.build(item, chain_text, limits, targets, target_note, transients, n_pass, per_tier, top_n)
            keep = os.path.join(run.out_root, 'notify', f'{item}_highlights.html')
            os.makedirs(os.path.dirname(keep), exist_ok=True)
            with open(keep, 'w') as f:
                f.write(html)
            u.add_output(keep, 'notify_block', test=False)
            # Phase 2's own mailer, in this process: its two module lists are refilled from the run's files
            from exopipe.phase2 import phase2_science_agg as agg
            saved = agg.restore_email_inputs(rd)
            # notify.SEND False (the default until Gregg's go): the email is written beside the block, never sent
            copy = (f'{args.write_to}_{item}' if args.write_to
                    else None if CFG.notify.SEND else os.path.join(run.out_root, 'notify', item))
            sent = agg.send_email_report(rd, run_command=saved.get('run_command'), run_duration=saved.get('run_duration'),
                                         lst=lst, date=date, highlights_html=html, phases='Phases 2-3', write_to=copy)
            n_list = sum(len(r) for _s, r, _n in transients)
            u.note(f'{len(targets)} target rows, {n_list} of {n_pass} filter-passing transient candidates listed; '
                   f'limits: {len(limits)}; '
                   + (f'NOT sent, written to {copy}.eml' if copy else 'sent' if sent else 'SEND FAILED'))
            if not sent:
                raise RuntimeError(f'N3: the mailer did not send the email of {item} (its log line "Email failed" names '
                                   f'the cause); the block is kept at {keep}')
            print(f'{item}: {len(targets)} target rows, {n_list} of {n_pass} filter-passing candidates listed; '
                  + (f'NOT sent - written to {copy}.eml and {copy}.html' if copy else 'email sent'), file=sys.stderr)
    return record.EXIT_DONE if run.dry_run else run.exit_code()
