"""The Phase 3 highlights block of the hour's email: HTML only, from rows the readers of sources.py return.

Order (Gregg 2026-11-12): the state of the hour's Phase 3 chain and every limit or missing input first; then every
DETECTION of a target of the named samples ("Just detections!!"); then the most significant transient candidates of the
deep and 10-minute tiers that pass the filters of config notify ("Ideally I want real transients not junk"; the filters:
Gregg 2026-11-14). Every significance is given formal AND local-empirical; a column that a tier's table does not hold is
shown as a dash, never filled in. Phase 2's QA sections and the Phase 3 QA percentiles (qa.py) stand above this block in
the email.
"""
import html as _html
import json

TABLE = ('<table border="1" cellspacing="0" cellpadding="4" style="border-collapse:collapse;font-size:12px">'
         '<tr>{head}</tr>{rows}</table>')


def esc(x):
    return _html.escape('' if x is None else str(x))


def num(x, fmt):
    """A number in the given format, or a dash when the table does not hold it (None, empty, NaN)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return '&ndash;'
    if v != v:
        return '&ndash;'
    return format(v, fmt)


def sexagesimal(ra_deg, dec_deg):
    """'04:10:12.3 +30:05:11' from degrees (J2000 as the tables give them); a dash pair when either is missing."""
    try:
        ra, dec = float(ra_deg) % 360.0, float(dec_deg)
    except (TypeError, ValueError):
        return '&ndash;'
    if ra != ra or dec != dec:
        return '&ndash;'
    h = ra / 15.0
    hh = int(h)
    mm = int((h - hh) * 60)
    ss = (h - hh - mm / 60.0) * 3600
    if round(ss, 1) >= 60.0:
        ss, mm = 0.0, mm + 1
    if mm >= 60:
        mm, hh = 0, (hh + 1) % 24
    sign = '-' if dec < 0 else '+'
    a = abs(dec)
    dd = int(a)
    dm = int((a - dd) * 60)
    ds = (a - dd - dm / 60.0) * 3600
    if round(ds) >= 60:
        ds, dm = 0.0, dm + 1
    if dm >= 60:
        dm, dd = 0, dd + 1
    return f'{hh:02d}:{mm:02d}:{ss:04.1f} {sign}{dd:02d}:{dm:02d}:{ds:02.0f}'


def link(url, text):
    return f'<a href="{esc(url)}">{esc(text)}</a>' if url else esc(text)


def table(columns, rows):
    """columns: [(header, function of a row -> HTML cell text)]."""
    head = ''.join(f'<th>{esc(h)}</th>' for h, _ in columns)
    body = ''.join('<tr>' + ''.join(f'<td>{f(r)}</td>' for _, f in columns) + '</tr>' for r in rows)
    return TABLE.format(head=head, rows=body)


def chain_summary(record_path):
    """One sentence on the hour's Phase 3 chain from the runner's item record (runner.CHAIN_FINALLY passes its path):
    the outcome and, for a chain that did not finish, the step that stopped it. (text, ok). No record: said so."""
    if not record_path:
        return 'Phase 3 chain record: none given (this email was built outside the per-hour chain).', None
    try:
        with open(record_path) as f:
            rec = json.load(f)
    except (OSError, ValueError) as e:
        return f'Phase 3 chain record unreadable ({type(e).__name__}): {esc(record_path)}.', None
    status = rec.get('chain_status') or rec.get('status')
    steps = rec.get('steps') or []
    names = [' '.join(s.get('argv', [])[3:5]) or s.get('name', '?') for s in steps]
    by = {}
    for s in steps:
        by[s.get('status')] = by.get(s.get('status'), 0) + 1
    counts = ', '.join(f'{n} {k}' for k, n in sorted(by.items()))
    if status == 'done':
        return f'Phase 3 chain: done ({len(steps)} steps: {esc(counts)}).', True
    if status == 'no-input':
        return f'Phase 3 chain: every step had nothing to do for this hour ({len(steps)} steps).', False
    stop = next((i for i, s in enumerate(steps) if s.get('status') not in ('done', 'no-input')), None)
    where = (f' at step {stop + 1} (<code>{esc(names[stop])}</code>, exit code {esc(steps[stop].get("rc"))})'
             if stop is not None else '')
    return (f'<strong>Phase 3 chain: {esc(str(status).upper())}{where}</strong>; steps before it: {esc(counts)}. '
            f'Log: <code>{esc(rec.get("log"))}</code>. The tables below hold what the finished steps wrote.'), False


def gal_lat(r, bold_below):
    """Galactic latitude in degrees, in bold when its absolute value is below the limit (Gregg 2026-11-14)."""
    text = num(r.get('gal_lat_deg'), '+.1f')
    try:
        low = abs(float(r.get('gal_lat_deg'))) < bold_below
    except (TypeError, ValueError):
        low = False
    return f'<strong>{text}</strong>' if low else text


def filter_sentence(name, c):
    """One page's counts: the hour's detections of listed candidates that pass each filter, and all of them."""
    b = c['bright']
    return (f'<li>{esc(name)}: {c["served"]} detections of {c["candidates"]} listed candidates in this hour. Passing S/N: '
            f'{c["snr"]}; elevation: {c["elev"]}; count in the epoch image (at most {num(c["n_max"], "g")}): {c["nve"]}; '
            f'size: {c["size"]}; axis ratio: {c["axis"]}; the page\'s default filters: {c["defaults"]}; the bright-source '
            f'filter: {c["bright_pass"]} (mean sky loss {100 * b["mean_sky_loss"]:.1f}% over {b["bands"]} searched bands; it '
            f'removes {b["candidates_removed"]} of the page\'s {b["listed"]} listed candidates); detections of candidates '
            f'matched to a known pulsar: {c["pulsar"]}. Passing all: {c["pass"]} detections of {c["listed"]} candidates.</li>')


def build(item, chain_text, limits, targets, target_note, transients, n_pass, per_tier, top_n, cfg=None):
    """The block. limits: sentences on missing inputs (shown first). targets: rows (dicts) from sources.py. transients:
    [(Stokes, rows, number passing in that Stokes)], Stokes V first. n_pass: candidates of the hour that pass every
    filter (both tiers, both Stokes); per_tier: {page: counts (sources.collect)}; cfg: config notify (the filter values
    named in the text)."""
    lst, date = item.split('_', 1)
    out = [f'<h4>Phase 3 highlights: LST {esc(lst)}, {esc(date)}</h4>', f'<p>{chain_text}</p>']
    if limits:
        out.append('<p><strong>Limits and missing inputs:</strong></p><ul>'
                   + ''.join(f'<li>{x}</li>' for x in limits) + '</ul>')
    out.append('<h4>Detections of Hot / Warm Jupiter and 10 pc sample targets</h4>')
    if targets:
        out.append(table([
            ('Target', lambda r: link(r.get('link'), r.get('target'))),
            ('Sample', lambda r: esc(r.get('sample'))),
            ('Found by', lambda r: esc(r.get('found_by'))),
            ('Tier', lambda r: esc(r.get('tier'))),
            ('Band', lambda r: esc(r.get('band'))),
            ('Stokes', lambda r: esc(r.get('stokes'))),
            ('Time (UTC)', lambda r: esc(r.get('time_utc'))),
            ('Flux density (Jy)', lambda r: num(r.get('flux_jy'), '.3f')),
            ('S/N, formal', lambda r: num(r.get('snr_formal'), '.1f')),
            ('S/N, local-empirical', lambda r: num(r.get('snr_local'), '.1f')),
            ('Offset percentile', lambda r: num(r.get('offset_percentile'), '.0f')),
            ('Chance-coincidence probability', lambda r: num(r.get('p_chance'), '.2g')),
            ('Note', lambda r: esc(r.get('note'))),
        ], targets))
    out.append(f'<p>{target_note}</p>')
    out.append(f'<h4>Most significant transient candidates of the hour, deep and 10-minute tiers ({n_pass} pass every '
               f'filter)</h4>')
    if cfg is not None:
        out.append(f'<p>Listed: a detection of this hour with |S/N| above {cfg.SNR_MIN:g}, elevation above '
                   f'{cfg.ELEVATION_MIN_DEG:g} deg at the middle of the LST hour, a count of detections of its Stokes in its '
                   f'epoch image at or below the {cfg.N_EPOCH_MAX_PERCENTILE}th percentile of its page\'s slider, a fitted '
                   f'size below {cfg.SIZE_MAX_BEAMS:g} beams and a fitted axis ratio below {cfg.AXIS_RATIO_MAX:g}; passing '
                   f'its page\'s default filters; not removed by its page\'s bright-source filter at an allowed mean sky '
                   f'loss of {100 * cfg.BRIGHT_SKY_LOSS:g}%; and not of a candidate its page matches to a known pulsar. A '
                   f'missing value fails its filter. One row per candidate, tier and Stokes: its most significant passing '
                   f'detection of the hour. S/N, local-empirical is the search\'s own local statistic; S/N, formal is given '
                   f'where the tier\'s table records one. Galactic latitude is in bold below {cfg.GAL_LAT_BOLD_DEG:g} deg '
                   f'from the plane.</p>')
    if per_tier:
        out.append('<ul>' + ''.join(filter_sentence(n, c) for n, c in per_tier.items()) + '</ul>')
    bold = cfg.GAL_LAT_BOLD_DEG if cfg is not None else 10.0
    cols = [
        ('#', lambda r: esc(r.get('rank'))),
        ('Tier', lambda r: esc(r.get('tier'))),
        ('Band', lambda r: esc(r.get('band'))),
        ('Time (UTC)', lambda r: esc(r.get('time_utc'))),
        ('RA Dec (J2000)', lambda r: sexagesimal(r.get('ra_deg'), r.get('dec_deg'))),
        ('Galactic latitude (deg)', lambda r: gal_lat(r, bold)),
        ('Elevation (deg)', lambda r: num(r.get('elevation_deg'), '.0f')),
        ('S/N, formal', lambda r: num(r.get('snr_formal'), '.1f')),
        ('S/N, local-empirical', lambda r: num(r.get('snr_local'), '.1f')),
        ('Class', lambda r: esc(r.get('cls'))),
        ('Page', lambda r: link(r.get('link'), r.get('link_text') or 'page')),
    ]
    for stokes, rows, n in transients:
        out.append(f'<p><strong>Stokes {esc(stokes)}: top {len(rows)} of {n}</strong></p>')
        out.append(table(cols, rows) if rows else '<p>None.</p>')
    return '\n'.join(out)
