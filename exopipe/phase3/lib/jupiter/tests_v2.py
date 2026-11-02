"""Content tests of the Jupiter V2 and occurrence classes (spec plan/specs/jupiter.md section 6), registered with
@content_test. During the refactor window these check each product against the record written with it - manifests
re-hashed, verdict files consistent, freshness (no product file newer than its verdict), the occurrence failures equal the
accepted waiver exactly - and run no V2 code. The component's own full test (jupv2_masters test, jblue_test, test_cell,
jint_test_metrics, jsite_test, content_test.py, jint_content_summary) runs inside the v2 / occurrence stage that writes
the product (it writes its verdict into the product directory, so it cannot be a side-effect-free class test).

  jupiter_geom_raw          nightgeom/horizons_<date>/manifest.json            raw Horizons tables re-hashed = the fetch manifest
  jupiter_geom              nightgeom/<cell>/CONTENT_TEST.txt                  verdict ALL-PASS, TEST lines, freshness of geom_*
  jupiter_v2_masters        masters/CONTENT_TEST_<cell>.txt                    verdict ALL-PASS, no FAIL line, freshness
  jupiter_v2_root           masters/derived/<V>/<cell>/CONTENT_TEST.txt        verdict, links resolve, assign rows = meas rows
  jupiter_crop_archive      provenance/<cell>/crops_<label>.tar                members re-hashed = crops_<label>.csv; verdict
  jupiter_blue_anchor       anchor/<cell>/<label>/MANIFEST.sha256              members re-hashed; CONTENT_TEST_blue_<label>
  jupiter_refcheck          integrate/inputs/ref_offsets_<cell>.csv            window_summary tie columns recomputed from ref_offsets
  jupiter_metrics           integrate/<cell>/CONTENT_TEST_metrics.txt          verdict, metrics_<label>.json per root, freshness
  jupiter_site_render       site/<cell>/renders/<label>.r<UTC>/MANIFEST.sha256 members re-hashed; RENDER_TEST.txt ALL-PASS
  jupiter_site_summary      site/<cell>/site.json                              the live site (production's convention since
                                                                               2026-10-25/28): roots SITE_ROOTS of SITE_GENERATION,
                                                                               SITE_VERSION; each live render's RENDER_TEST.txt
                                                                               ALL-PASS (T8 PASS) recorded in site.json; products
                                                                               and metric file unchanged since the render; stated
                                                                               occurrence failures = the accepted set. Or the
                                                                               2026-10-06 flow's site/<cell>/CONTENT_TEST.txt:
                                                                               ALL-PASS (except exactly the accepted failures),
                                                                               not older than a site.json beside it, no v3 site
  jupiter_render_quarantine site/<cell>/<qdir>/MANIFEST.sha256                 moved files re-hashed (recorded deletions excepted)
  jupiter_occurrence        occurrence/validation/content_test_results.json    failing gating set = the accepted waiver, exactly
  jupiter_disclosure        integrate/occurrence_disclosed_failures.json       the waiver record = the live results
"""
import csv
import glob
import hashlib
import json
import os
import re
import tarfile

from exopipe.phase3.core.validate import content_test


def _C():
    from exopipe.phase3.config import CFG
    return CFG


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def _lines(path):
    return [l.rstrip('\n') for l in open(path, errors='replace') if l.strip()]


def _verdict(path, accept_prefix=False):
    lines = _lines(path)
    if not lines:
        return False, 'empty verdict file', lines
    last = lines[-1]
    ok = last == 'ALL-PASS' or (accept_prefix and last.startswith('ALL-PASS'))
    fails = [l for l in lines if re.search(r'->\s*FAIL', l) and 'informational, not a gate' not in l]
    if ok and fails and not accept_prefix:
        return False, f'last line ALL-PASS but {len(fails)} line(s) end -> FAIL', lines
    return ok, last[:200], lines


def _fresh(verdict_path, files):
    """Files newer than their verdict (by more than 2 s): the verdict does not cover them."""
    t = os.path.getmtime(verdict_path)
    return [f for f in files if os.path.isfile(f) and os.path.getmtime(f) > t + 2]


def _manifest_check(mpath, base=None, accounted=(), accounted_dirs=()):
    """Re-hash every member of a 'sha256  [size  ]name' manifest. Returns (n, missing, differ). A missing member is
    accounted when its name, or its directory under renders/ (accounted_dirs), is recorded as deleted."""
    base = base or os.path.dirname(mpath)
    n, missing, differ = 0, [], []
    adirs = set(accounted_dirs)
    for line in _lines(mpath):
        parts = line.split(None, 2)
        if len(parts) == 3 and parts[1].isdigit():
            h, size, name = parts
        else:
            h, name = line.split(None, 1)
            size = None
        name = name.strip()
        p = os.path.join(base, name)
        n += 1
        if not os.path.exists(p):
            parts = name.split('/')
            gone = name in accounted or os.path.basename(name) in accounted or (len(parts) > 1 and parts[1] in adirs)
            if not gone:
                missing.append(name)
            continue
        if (size is not None and os.path.getsize(p) != int(size)) or _sha256_file(p) != h:
            differ.append(name)
    return n, missing, differ


def _short(xs, n=3):
    xs = list(xs)
    return ', '.join(xs[:n]) + (f' (+{len(xs) - n})' if len(xs) > n else '')


@content_test('jupiter_geom_raw')
def manifest(path, **ctx):
    """Every raw Horizons table named in the fetch manifest (<query>.txt beside it) re-hashed = its recorded sha256."""
    d = os.path.dirname(os.path.abspath(path))
    m = json.load(open(path))
    q = m.get('queries') or {}
    missing, differ = [], []
    for key, info in q.items():
        p = os.path.join(d, key + '.txt')
        if not os.path.exists(p):
            missing.append(key)
        elif _sha256_file(p) != info.get('sha256'):
            differ.append(key)
    ok = bool(q) and not missing and not differ
    return ok, len(q), f'{len(q)} raw tables; missing {_short(missing) or 0}; sha256 differs {_short(differ) or 0}'


@content_test('jupiter_geom')
def nightgeom_verdict(path, **ctx):
    """nightgeom/<cell>/CONTENT_TEST.txt (test_cell.py TEST0-TEST9): last line ALL-PASS, no TEST line ends -> FAIL, the
    geometry tables (geom_*.csv, cutout_wcs_10min.json, geom_events.csv) exist and none is newer than the verdict."""
    ok, last, lines = _verdict(path)
    d = os.path.dirname(os.path.abspath(path))
    files = glob.glob(os.path.join(d, 'geom_*.csv')) + [os.path.join(d, 'cutout_wcs_10min.json')]
    absent = [os.path.basename(f) for f in files if not os.path.exists(f)]
    newer = _fresh(path, files)
    tests = [l for l in lines if re.match(r'^TEST\d', l)]
    return ok and not absent and not newer, len(tests), (f'{last}; {len(tests)} TEST lines; absent {_short(absent) or 0}; '
                                                         f'newer than the verdict {_short(map(os.path.basename, newer)) or 0}')


@content_test('jupiter_v2_masters')
def masters_verdict(path, **ctx):
    """masters/CONTENT_TEST_<cell>.txt (jupv2_masters stage_test T1-T11): ALL-PASS, no FAIL line, and no file of the
    variants' cell directories (masters/<V>/<cell>/) is newer than the verdict."""
    ok, last, lines = _verdict(path)
    d = os.path.dirname(os.path.abspath(path))
    cell = re.sub(r'^CONTENT_TEST_|\.txt$', '', os.path.basename(path))
    files = []
    for v in _C().jupiter_v2.VARIANTS:
        files += glob.glob(os.path.join(d, v, cell, '*'))
    newer = _fresh(path, files)
    return ok and bool(files) and not newer, len(files), (f'{last}; {len(files)} variant files; newer than the verdict '
                                                          f'{_short(map(os.path.basename, newer)) or 0}')


@content_test('jupiter_v2_root')
def prepared_root(path, **ctx):
    """masters/derived/<V>/<cell>/CONTENT_TEST.txt (jsite_prepare_variant.py): ALL-PASS; every symlink in the root resolves;
    each assign_<tier>.csv has exactly the rows (cell, band, Stokes, key, in order) of its meas_<tier>.csv."""
    ok, last, _ = _verdict(path)
    d = os.path.dirname(os.path.abspath(path))
    broken = [n for n in os.listdir(d) if os.path.islink(os.path.join(d, n)) and not os.path.exists(os.path.join(d, n))]
    bad = []
    for t in ('10s', '1min', '10min'):
        a, m = os.path.join(d, f'assign_{t}.csv'), os.path.join(d, f'meas_{t}.csv')
        if os.path.exists(a) and os.path.exists(m):
            ka = [(r['band'], r['stokes'], r['key']) for r in csv.DictReader(open(a, newline=''))]
            km = [(r['band'], r['stokes'], r['key']) for r in csv.DictReader(open(m, newline=''))]
            if ka != km:
                bad.append(t)
    return ok and not broken and not bad, len(os.listdir(d)), (f'{last}; broken links {_short(broken) or 0}; assign rows != '
                                                               f'meas rows in tiers {bad or "none"}')


@content_test('jupiter_crop_archive')
def archive_audit(path, **ctx):
    """provenance/<cell>/crops_<label>.tar: every member re-hashed = its row of crops_<label>.csv (member, bytes, sha256),
    every row has its member, and CONTENT_TEST_crops_<label>.txt ends ALL-PASS."""
    d = os.path.dirname(os.path.abspath(path))
    label = re.sub(r'^crops_|\.tar$', '', os.path.basename(path))
    rows = {r['member']: r for r in csv.DictReader(open(os.path.join(d, f'crops_{label}.csv'), newline=''))}
    seen, differ = set(), []
    with tarfile.open(path) as tf:
        for m in tf:
            if not m.isfile():
                continue
            seen.add(m.name)
            r = rows.get(m.name)
            data = tf.extractfile(m).read()
            if r is None or int(r['bytes']) != len(data) or hashlib.sha256(data).hexdigest() != r['sha256']:
                differ.append(m.name)
    missing = set(rows) - seen
    vok, last, _ = _verdict(os.path.join(d, f'CONTENT_TEST_crops_{label}.txt'))
    ok = vok and not differ and not missing and bool(rows)
    return ok, len(seen), f'{len(seen)} members; differ {len(differ)}; rows without member {len(missing)}; verdict {last}'


@content_test('jupiter_blue_anchor')
def blue_manifest(path, **ctx):
    """anchor/<cell>/<label>/MANIFEST.sha256: every member re-hashed (sha256 and size); anchor/<cell>/CONTENT_TEST_blue_<label>.txt
    (jblue_test B1-B7) ends ALL-PASS and is not older than the members."""
    d = os.path.dirname(os.path.abspath(path))
    label = os.path.basename(d)
    n, missing, differ = _manifest_check(path)
    vp = os.path.join(os.path.dirname(d), f'CONTENT_TEST_blue_{label}.txt')
    vok, last, _ = _verdict(vp) if os.path.exists(vp) else (False, 'no verdict file', [])
    newer = _fresh(vp, glob.glob(os.path.join(d, '*'))) if os.path.exists(vp) else []
    ok = n > 0 and not missing and not differ and vok and not newer
    return ok, n, f'{n} members; missing {_short(missing) or 0}; differ {_short(differ) or 0}; verdict {last}; newer {len(newer)}'


@content_test('jupiter_refcheck')
def ref_inputs(path, **ctx):
    """integrate/inputs/ref_offsets_<cell>.csv and window_summary_<cell>.csv (the reference-tie inputs made by
    v2/integrate/inputs/jup_refcheck_20241229.py): both parse with rows; the summary's windows = the (subband, window) pairs
    with reference fits, in the script's order; per window the reference-tie columns recomputed from ref_offsets by the
    script's rules (:61-70): n_ref, n_near6 (sep_deg < 6), ref_dx/ref_dy = median offsets, ref_scatter = median distance
    from that median, near_dx/near_dy = medians of the near ones (2 decimals); jup<S>_dx_tied = jup<S>_dx - ref_dx (to the
    2-decimal rounding of both). Not checked here: the reference fits themselves (re-made only by `v2 --stages refcheck`,
    which keeps an existing table only when it is byte-identical to the fresh one)."""
    import statistics as st
    import math
    d = os.path.dirname(os.path.abspath(path))
    cell = re.sub(r'^ref_offsets_|\.csv$', '', os.path.basename(path))
    ws = os.path.join(d, f'window_summary_{cell}.csv')
    if not os.path.exists(ws):
        return False, None, f'{os.path.basename(ws)} missing'
    refs = list(csv.DictReader(open(path, newline='')))
    summ = list(csv.DictReader(open(ws, newline='')))
    if not refs or not summ:
        return False, len(refs), f'ref_offsets {len(refs)} rows, window_summary {len(summ)} rows'
    groups = {}
    for r in refs:
        groups.setdefault((r['sb'], r['w']), []).append(r)
    order = sorted(groups, key=lambda k: (_C().jupiter.SUBBANDS.index(k[0]) if k[0] in _C().jupiter.SUBBANDS else 99, k[1]))
    if [(e['sb'], e['w']) for e in summ] != order:
        return False, len(summ), f'summary windows {len(summ)} != the {len(order)} (subband, window) pairs with reference fits'

    def f(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return float('nan')

    def same(a, b, tol=0.0051):
        return (math.isnan(a) and math.isnan(b)) or abs(a - b) <= tol
    bad = []
    for e in summ:
        rr = groups[(e['sb'], e['w'])]
        dx, dy = [f(r['dx']) for r in rr], [f(r['dy']) for r in rr]
        near = [r for r in rr if f(r['sep_deg']) < 6]
        mdx, mdy = st.median(dx), st.median(dy)
        want = {'n_ref': len(rr), 'n_near6': len(near), 'ref_dx': round(mdx, 2), 'ref_dy': round(mdy, 2),
                'ref_scatter': round(st.median(math.hypot(a - mdx, b - mdy) for a, b in zip(dx, dy)), 2),
                'near_dx': round(st.median(f(r['dx']) for r in near), 2) if near else float('nan'),
                'near_dy': round(st.median(f(r['dy']) for r in near), 2) if near else float('nan')}
        for k, v in want.items():
            if not same(f(e.get(k)), float(v)):
                bad.append(f"{e['sb']} {e['w']} {k} {e.get(k)} != {v}")
        for S in ('I', 'V'):
            for ax, m in (('dx', mdx), ('dy', mdy)):
                if e.get(f'jup{S}_{ax}') not in (None, ''):
                    if not same(f(e.get(f'jup{S}_{ax}_tied')), f(e[f'jup{S}_{ax}']) - m, tol=0.0101):
                        bad.append(f"{e['sb']} {e['w']} jup{S}_{ax}_tied")
    return not bad, len(summ), (f'ref_offsets {len(refs)} rows over {len(order)} windows; window_summary {len(summ)} rows; '
                                f'{len(bad)} recomputed values differ' + (f' ({"; ".join(bad[:3])})' if bad else ''))


@content_test('jupiter_metrics')
def metrics_verdict(path, **ctx):
    """integrate/<cell>/CONTENT_TEST_metrics.txt (jint_test_metrics M1-M16): ALL-PASS; metrics_<label>.json parses for every
    root label; no metrics file newer than the verdict."""
    ok, last, _ = _verdict(path)
    d = os.path.dirname(os.path.abspath(path))
    bad = []
    for lab in _C().jupiter_v2.ROOT_LABELS:
        p = os.path.join(d, f'metrics_{lab}.json')
        try:
            json.load(open(p))
        except (OSError, ValueError):
            bad.append(lab)
    newer = _fresh(path, glob.glob(os.path.join(d, 'metrics_*')) + glob.glob(os.path.join(d, '*.csv')))
    return ok and not bad and not newer, len(_C().jupiter_v2.ROOT_LABELS) - len(bad), (
        f'{last}; unreadable metrics_<label>.json {bad or "none"}; newer than the verdict {_short(map(os.path.basename, newer)) or 0}')


@content_test('jupiter_site_render')
def render_manifest(path, **ctx):
    """site/<cell>/renders/<label>.r<UTC>/MANIFEST.sha256: every member re-hashed; RENDER_TEST.txt (jsite_test T1-T10 and the
    self-test) ends ALL-PASS."""
    d = os.path.dirname(os.path.abspath(path))
    n, missing, differ = _manifest_check(path)
    vok, last, _ = _verdict(os.path.join(d, 'RENDER_TEST.txt'))
    ok = n > 0 and not missing and not differ and vok
    return ok, n, f'{n} members; missing {_short(missing) or 0}; differ {_short(differ) or 0}; verdict {last}'


def _site_version(S):
    """The highest site_version a site record carries (site.json and its roots; 0 when none: the 2026-10-06 flow writes 2 per
    root and none for the site), or None when one does not parse."""
    try:
        return max([int(S.get('site_version') or 0)] + [int(v.get('site_version') or 0) for v in S.get('variants') or []])
    except (TypeError, ValueError, AttributeError):
        return None


def _site_record(path):
    """site/<cell>/site.json: the live site of the cell, its roots and their live renders (summary_valid, first convention)."""
    J, V, O = _C().jupiter, _C().jupiter_v2, _C().jupiter_occurrence
    sc = os.path.dirname(os.path.abspath(path))
    try:
        S = json.load(open(path))
    except (OSError, ValueError) as e:
        return False, None, f'site.json unreadable ({type(e).__name__}: {e})'
    cell = S.get('cell') or os.path.basename(sc)
    roots = S.get('variants') or []
    labels = [v.get('label') for v in roots]
    acc = sorted(O.ACCEPTED_FAILURES)
    probs, live, occ_states, n_prod, n_hash = [], [], set(), 0, 0
    if labels != list(V.SITE_ROOTS):
        probs.append(f'roots {labels}, the live site shows {list(V.SITE_ROOTS)} (jupiter_v2.SITE_ROOTS: V2 only, Gregg 2026-10-25)')
    if S.get('default') != V.RULED_ROOT or labels[:1] != [V.RULED_ROOT]:
        probs.append(f'default root {S.get("default")}, order {labels}: the ruled root {V.RULED_ROOT} must be the default and first')
    if S.get('site_version') != V.SITE_VERSION:
        probs.append(f'site_version {S.get("site_version")}, the live site\'s is {V.SITE_VERSION} (jupiter_v2.SITE_VERSION)')
    if S.get('preview_failed'):
        probs.append(f'published around its content tests: {S["preview_failed"]}')
    for v in roots:
        lab = v.get('label') or '?'
        if v.get('site_version') != V.SITE_VERSION:
            probs.append(f'{lab}: rendered with site_version {v.get("site_version")}, the live site\'s is {V.SITE_VERSION}')
        want = os.path.join(J.ROOT, V.SITE_GENERATION, 'derived', lab)
        if not v.get('root') or os.path.realpath(str(v['root'])) != os.path.realpath(want):
            probs.append(f'{lab}: renders the product root {v.get("root")}, the live generation\'s is {want} (jupiter_v2.SITE_GENERATION)')
        rd = os.path.join(sc, 'renders', f'{lab}.r{v.get("render_id")}')
        link = os.path.join(sc, lab)
        if not os.path.islink(link) or not os.path.isdir(rd) or os.path.realpath(link) != os.path.realpath(rd):
            probs.append(f'{lab}: link {"-> " + os.readlink(link) if os.path.islink(link) else "missing"}, but site.json names '
                         f'renders/{os.path.basename(rd)}{"" if os.path.isdir(rd) else " (absent)"}')
            continue
        rt = os.path.join(rd, 'RENDER_TEST.txt')
        if not os.path.exists(rt):
            probs.append(f'{lab}: renders/{os.path.basename(rd)} has no RENDER_TEST.txt')
            continue
        live.append(os.path.basename(rd))
        vok, last, lines = _verdict(rt)
        if not vok:
            probs.append(f'{lab}: RENDER_TEST.txt {last}')
        t8 = [l for l in lines if l.startswith('T8 ')]
        if len(t8) != 1 or not re.search(r'->\s*PASS\s*$', t8[0]):
            probs.append(f'{lab}: T8 (products unchanged since the render) {"absent" if not t8 else "not PASS"}')
        if v.get('test') != (lines[-1] if lines else None):
            probs.append(f'{lab}: site.json records the test {v.get("test")!r}, RENDER_TEST.txt ends {last!r}')
        if v.get('preview_failed'):
            probs.append(f'{lab}: published around its content tests: {v["preview_failed"]}')
        st = v.get('occurrence_status')
        occ = next((x for x in v.get('limits') or [] if x.get('id') == 'occ_failures'), None) or {}
        named = sorted(str(f.get('test')) for f in ((occ.get('data') or {}).get('failures') or []))
        occ_states.add(f'{st} {named}' if named else str(st))
        if st == 'used-accepted':
            if named != acc:
                probs.append(f'{lab}: occurrence failures stated {named}, accepted {acc}')
        elif st != 'used' or named:
            probs.append(f'{lab}: occurrence status {st!r} with failures {named} (the maps must pass their own test, or fail exactly '
                         f'the accepted set {acc})')
        t = os.path.getmtime(rt)
        newer = [e for e in sorted(os.listdir(rd)) if e not in ('RENDER_TEST.txt', 'MANIFEST.sha256')
                 and os.path.getmtime(os.path.join(rd, e)) > t + 2]
        if newer:
            probs.append(f'{lab}: newer than its RENDER_TEST.txt: {_short(newer)}')
        # freshness re-checked here, not only read from T8: the product files the render recorded (render_info.json 'products':
        # file -> mtime, jsite_lib.product_files) against the root now, and the metric file it shows against its sha256
        try:
            ri = json.load(open(os.path.join(rd, 'render_info.json')))
        except (OSError, ValueError) as e:
            probs.append(f'{lab}: render_info.json unreadable ({type(e).__name__})')
            continue
        if (os.path.realpath(str(ri.get('root'))) != os.path.realpath(str(v.get('root'))) or ri.get('render_id') != v.get('render_id')
                or ri.get('site_version') != v.get('site_version')):
            probs.append(f'{lab}: render_info.json (root {ri.get("root")}, render {ri.get("render_id")}, site_version '
                         f'{ri.get("site_version")}) differs from site.json')
        rec = ri.get('products') if isinstance(ri.get('products'), dict) else {}
        cdir = os.path.join(str(ri.get('root')), cell)
        now = {os.path.basename(p): os.path.getmtime(p) for p in glob.glob(os.path.join(cdir, '*')) if os.path.isfile(p)}
        changed = sorted(f for f, m in rec.items() if now.get(f) != m)
        added = sorted(set(now) - set(rec))
        n_prod += len(rec)
        if not rec or changed or added:
            probs.append(f'{lab}: product files of {cdir} since the render: {len(rec)} recorded, changed or gone '
                         f'{_short(changed) or 0}, new {_short(added) or 0}')
        for k in ('reflect', 'metrics'):
            m = ri.get(k)
            if not isinstance(m, dict) or not m.get('path'):
                if k == 'reflect' and (v.get('site_version') or 0) >= 3:
                    probs.append(f'{lab}: a site v3 render without its reflection-metric record')
                continue
            n_hash += 1
            if not os.path.isfile(m['path']) or _sha256_file(m['path']) != m.get('sha256'):
                probs.append(f'{lab}: the {k} file {m["path"]} is not the one rendered (sha256 {str(m.get("sha256"))[:12]} recorded)')
    if probs:
        return False, len(roots), '; '.join(probs[:5]) + (f' (+{len(probs) - 5})' if len(probs) > 5 else '')
    return True, len(roots), (f'{len(roots)} root(s) {labels} of {V.SITE_GENERATION}, site_version {V.SITE_VERSION}: live '
                              f'render(s) {", ".join(live)} - RENDER_TEST ALL-PASS, T8 PASS, recorded in site.json; {n_prod} product '
                              f'files unchanged since the render, {n_hash} metric file(s) = their recorded sha256; occurrence '
                              f'{"; ".join(sorted(occ_states))} (accepted set {acc})')


@content_test('jupiter_site_summary')
def summary_valid(path, **ctx):
    """The content-test record of a site cell, in either convention production has used:

    site/<cell>/site.json - production since 2026-10-25 ('V2 only': the cell summary file was quarantined with the comparison
    roots) and the 2026-10-28 site rebuild: no cell summary is written; each published render carries its own RENDER_TEST.txt
    (jsite_test.py, T1-T10; the renderer publishes only on ALL-PASS) and site.json names the live render of every root.
    Checked: the roots are jupiter_v2.SITE_ROOTS (V2 only) with the ruled root the default and first, site_version
    jupiter_v2.SITE_VERSION for the site and every root, each root rendering <jupiter.ROOT>/<SITE_GENERATION>/derived/<label>;
    per root, the <label> link resolves to renders/<label>.r<render_id>; its RENDER_TEST.txt ends ALL-PASS with no line ending
    -> FAIL, and its T8 line (the products unchanged since the render) ends -> PASS; site.json's 'test' = that last line; nothing
    published around the tests (preview_failed empty: jsite_publish_preview.py states failures instead of refusing); the
    occurrence failures the root states (limit occ_failures) = exactly jupiter_occurrence.ACCEPTED_FAILURES; no entry of the
    render's top level (tier directories included, MANIFEST.sha256 excepted: written after the test) newer than its
    RENDER_TEST.txt; render_info.json agrees with site.json, every product file it recorded has its recorded mtime and none
    was added (T8 re-checked, not trusted), and the metric file it shows (reflect, or metrics in the 2026-10-06 flow) has its
    recorded sha256. The render's own files are re-hashed by jupiter_site_render.

    site/<cell>/CONTENT_TEST.txt - jint_content_summary.py, the 2026-10-06 flow's summary that `v2 --stages summary` still runs
    in place: FRESHNESS -> PASS and the last line ALL-PASS, or 'ALL-PASS EXCEPT THE ACCEPTED OCCURRENCE FAILURES' with exactly
    the accepted set (jupiter_occurrence.ACCEPTED_FAILURES) named; and it must describe the site beside it: a site.json there
    that is a site v3 record (site_version above 2) or newer than the summary fails it."""
    if os.path.basename(path) == 'site.json':
        return _site_record(path)
    ok, last, lines = _verdict(path, accept_prefix=True)
    fresh = any(l.strip() == 'FRESHNESS -> PASS' for l in lines)
    acc = _C().jupiter_occurrence.ACCEPTED_FAILURES
    if 'EXCEPT THE ACCEPTED' in last:
        summ = next((l for l in lines if l.startswith('SUMMARY:')), '')
        m = re.search(r'gating checks fail \(([^)]*)\)', summ)
        named = tuple(x.strip() for x in m.group(1).split(',')) if m else ()
        if sorted(named) != sorted(acc):
            return False, None, f'summary names {named}, accepted {acc}'
    sj = os.path.join(os.path.dirname(os.path.abspath(path)), 'site.json')
    if os.path.exists(sj):
        try:
            sv = _site_version(json.load(open(sj)))
        except (OSError, ValueError):
            sv = None
        if sv is None or sv > 2:
            return False, None, (f'the site.json beside it is {"unreadable" if sv is None else f"a site_version {sv} record"} - '
                                 f'the live convention since 2026-10-25/28 (no cell summary): this summary does not describe it')
        if os.path.getmtime(sj) > os.path.getmtime(path) + 2:
            return False, None, 'the site.json beside it is newer: the site changed after this summary'
    return ok and fresh, None, f'{last[:120]}; freshness {"PASS" if fresh else "MISSING/FAIL"}'


@content_test('jupiter_render_quarantine')
def quarantine_rehash(path, **ctx):
    """site/<cell>/<qdir>/MANIFEST.sha256: every moved file re-hashed = the manifest; the render directories a recorded
    deletion names (DELETED_*_list.txt: one directory under renders/ per line; deletion only on Gregg's word, Q19) are
    accounted, not missing."""
    d = os.path.dirname(os.path.abspath(path))
    gone = set()
    for lst in glob.glob(os.path.join(d, 'DELETED_*_list.txt')):
        gone |= {l.split()[-1].rstrip('/') for l in _lines(lst)}
    n, missing, differ = _manifest_check(path, accounted_dirs=gone)
    return n > 0 and not missing and not differ, n, (f'{n} manifest lines; missing {_short(missing) or 0}; differ '
                                                     f'{_short(differ) or 0}; {len(gone)} directories recorded as deleted')


def _failed_gating(res):
    return {r['test']: r for r in res.get('results', []) if r.get('gating') and not r.get('passed')}


def _disclosure_check(results_path, disclosure_path):
    O = _C().jupiter_occurrence
    res = json.load(open(results_path))
    dis = json.load(open(disclosure_path))
    live = _failed_gating(res)
    rec = {f['test']: f for f in dis.get('failures', [])}
    probs = []
    if res.get('n_gating') != O.GATING_TESTS or dis.get('n_gating') != O.GATING_TESTS:
        probs.append(f"gating tests live {res.get('n_gating')}, record {dis.get('n_gating')}, ruled {O.GATING_TESTS}")
    if sorted(live) != sorted(O.ACCEPTED_FAILURES):
        probs.append(f'live failing set {sorted(live)} != accepted {sorted(O.ACCEPTED_FAILURES)}')
    if sorted(rec) != sorted(O.ACCEPTED_FAILURES):
        probs.append(f'record failing set {sorted(rec)} != accepted {sorted(O.ACCEPTED_FAILURES)}')
    for t in set(live) & set(rec):
        if live[t].get('value') != rec[t].get('value') or live[t].get('expected') != rec[t].get('expected'):
            probs.append(f'{t}: live value {live[t].get("value")} vs recorded {rec[t].get("value")}')
    return not probs, len(live), '; '.join(probs) or (f'{len(live)} of {res.get("n_gating")} gating checks fail = exactly the '
                                                      f'accepted set {sorted(O.ACCEPTED_FAILURES)} with the recorded values')


@content_test('jupiter_occurrence')
def occurrence_results(path, **ctx):
    """occurrence/validation/content_test_results.json (content_test.py M/L/G/S/T/C/F): every gating test passes, or the
    failing gating set and its values equal the disclosure record and the accepted set exactly (decision 2, Q18), with 92
    gating tests."""
    dis = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(path)))),
                       _C().jupiter_occurrence.DISCLOSURE)
    res = json.load(open(path))
    if not _failed_gating(res):
        return True, 0, f'every one of {res.get("n_gating")} gating tests passes'
    if not os.path.exists(dis):
        return False, None, f'gating failures without the disclosure record {dis}'
    return _disclosure_check(path, dis)


@content_test('jupiter_disclosure')
def record_matches_live(path, **ctx):
    """integrate/occurrence_disclosed_failures.json: its failing set, values and gating count = the live occurrence results
    (the file its 'source' names) and the accepted set."""
    dis = json.load(open(path))
    src = dis.get('source') or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(path))),
                                             _C().jupiter_occurrence.RESULTS)
    if not os.path.exists(src):
        return False, None, f'live results {src} missing'
    return _disclosure_check(src, path)
