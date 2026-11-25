"""The serve Plan: which run is served from where, which pages on which ports, with which argv and environment.

Production vs parity (DESIGN 11.1; spec serve.md 10): when ``--out`` lies under a production root (paths.PRODUCTION_ROOTS)
the plan serves the production run directory itself (only once paths.PRODUCTION_WRITES is True and the cutover is
approved); otherwise it is a PARITY plan: every file serve writes goes under ``<out>/<tier>/<run name>`` (deep: a mirror of
the served run, lib/serve/mirror.py; sub-LST: the cumulative build and swap; snapshot and forced pages: a byte copy), the
hub registry and page under ``<out>/review_hub`` (written by lib/serve/hub: build_review_hub.py hard-codes its directory),
the shared lists under ``<out>/review_lists``, and every port is shifted by ``--port-offset`` (>= serve.PARITY_PORT_OFFSET_MIN;
the cross-tier link strings keep the production ports so pages compare): a review page is registered as p<port + offset> in
``<out>/two_port`` (lib/serve/register.py), served by a parity two-port server on serve.HUB_PORT / INSPECTION_HUB_PORT +
offset; the Jupiter pages bind port + offset and read their production directories directly (read-only servers).

Argv per page family reproduces the launchers exactly (orders included); a review page's registry entry carries it without
'--port' (register.page_entry), as the production entries do:
  deep review single  p3s3_serve_stokes.sh:79      --run --queue --labels-dir --catalog-dir --port --title
                      --mchain-queue --mchain-catalog-dir (the multiples chain, Gregg 2026-10-27: exopipe_ops 4c42f91;
                      the two-port registry entries p8766/p8768 - no launcher passes them)
  deep review multi   p3s3_serve_multiples.sh:152  --run --queue --labels-dir --catalog-dir --port --title (the title of the
                      served entries p8772/p8773 since 2026-10-31; the launcher's per-port server has none, divergence D2)
  sub-LST review      sublst_serve.sh:140 / sublst_cumulative.sh:139-140  ... --port [--loc-offsets] --title
                      [--mchain-queue] (singles: the two-port registry entries p8787/p8788, which sublst_register_pages.py
                      re-registers as they are)
  snapshot review     snapshot_serve.sh:72-76      --run --queue --catalog-dir --port --default-fp-reason --loc-dir
                      --loc-offsets --title --sibling x3 (the combined view's pages, family comb, the same: snapshot_serve.sh
                      SNAP_PORTS="8986 8987 8988 8989"; their siblings are the other comb pages)
  snapshot forced     snapshot_forced_serve_20261007.sh:49-51  --run --queue --catalog-dir --port --loc-offsets
                      --default-fp-reason --sibling (what each other forced page serves AFTER this serve: its registry
                      entry's run or the run this serve registers for it, snapshot_forced_gen5_serve_20261026.sh:47-56) --title
  jupiter             jupiter_server.py ROOT PORT (cwd ops); python -m http.server PORT --bind 127.0.0.1 --directory DIR
                      (8801: py310_orca, cwd the site's parent - its standing command since 2026-11-14)
plus, in a parity plan only, --lists-dir <out>/review_lists on classify pages (spec 7.2).
"""
import glob
import os
import re

from exopipe.phase3.core.record import Refusal
from exopipe.phase3.lib.serve import scripts

SERVER_DEFAULT_FP_REASON = 'bilobe'      # classify_transients.py:2438 (the in-place server's own default)
# the order of the key=value arguments each launcher passes to build_review_hub.py --set (the registry entry's key order)
HUB_KV_ORDER = {('deep', 'review'): ('run', 'generation', 'served', 'live', 'port'),          # p3s3_serve_stokes.sh:84
                ('sublst', 'review'): ('run', 'generation', 'served', 'live', 'port'),        # sublst_serve.sh:142
                ('snapshot', 'review'): ('port', 'run', 'generation', 'served', 'live'),      # snapshot_serve.sh:101
                ('snapshot', 'comb'): ('port', 'run', 'generation', 'served', 'live'),        # the same (SNAP_HUB_SUFFIX=comb)
                ('snapshot', 'forced'): ('port', 'run', 'generation', 'served', 'live', 'note'),   # snapshot_forced_serve:96-98
                ('snapshot', 'forced-tier'): ('run', 'generation', 'served', 'live', 'port')}   # the live hub entries
                                                                                                 # <S> 10-min-forced / 1-hour-forced
FORCED_FAMILIES = ('forced', 'forced-tier')   # one run per page; --sibling links to the family's other pages
FAMILY_DEFAULT = {'deep': 'review', 'sublst': 'review', 'snapshot': 'review', 'jupiter': 'jupiter'}


def _CFG():
    from exopipe.phase3.config import CFG
    return CFG


def run_root(tier):
    """The production root of a tier's served runs (config.paths via serve.RUN_ROOT_KEY), or None (jupiter)."""
    CFG = _CFG()
    key = CFG.serve.RUN_ROOT_KEY.get(tier)
    return getattr(CFG.paths, key) if key else None


class Page:
    """One page of the plan. Paths of a review page are relative to a run directory (``at(field, root)``) because the
    sub-LST pages are prepared on the build directory and served from the swapped-in run; a forced page has its own run."""

    def __init__(self, plan, key, c):
        self.plan, self.key, self.c = plan, key, dict(c)
        self.tier = plan.tier
        self.family, self.program, self.start = c['family'], c['program'], c['start']
        self.stokes, self.kind = c.get('stokes'), c.get('kind')
        self.port_cfg = int(c['port'])
        self.port = self.port_cfg + plan.offset
        self.twin_cfg = int(c['twin']) if c.get('twin') else None
        self.twin = self.twin_cfg + plan.offset if self.twin_cfg else None
        self.hub = tuple(c['hub']) if c.get('hub') else None
        self.loc_offsets = bool(c.get('loc_offsets'))
        self.run_name = plan.run_name
        self.root = plan.U
        self.root_src = plan.U_src
        if self.family == 'forced':
            S = _CFG().serve
            self.run_name = S.FORCED_RUN_NAME.format(S=self.stokes, T=self.kind, tag=plan.forced_tag or '<TAG>')
            self.root_src = os.path.join(run_root('snapshot'), self.run_name)
            self.root = os.path.join(plan.out, 'snapshot', self.run_name) if plan.parity else self.root_src
        elif self.family == 'forced-tier':      # family T's run Run_SNAPFORCED_<S><10min|deep>_gen<N> (installed from <run>.new)
            C = _CFG()
            self.run_name = C.forced_tier_pages.RUN_NAME.format(S=self.stokes, tier=self.kind)
            self.root_src = os.path.join(run_root('snapshot'), self.run_name)
            # a parity serve installs where family T writes under the same --out: <out>/forced_snapshot.REVIEW_SUBDIR_UNDER_OUT
            self.root = (os.path.join(plan.out, C.forced_snapshot.REVIEW_SUBDIR_UNDER_OUT, self.run_name) if plan.parity
                         else self.root_src)
        elif self.family == 'pinned':           # a page registered once by hand: its run under the tier's run root
            self.run_name = c['run']
            self.root_src = os.path.join(run_root(self.tier), self.run_name)
            self.root = os.path.join(plan.out, self.tier, self.run_name) if plan.parity else self.root_src
        elif self.program in ('jupiter', 'http'):
            self.run_name = key
            self.root = self.root_src = _CFG().serve.JUPITER_ROOTS[c['root']]
        self.unit = f'{self.tier}_{self.run_name}_{key}'

    def at(self, field, root=None):
        rel = self.c.get(field)
        return os.path.join(root or self.root, rel) if rel else None

    @property
    def multi(self):
        return self.kind == 'multi'

    def title_kind(self):
        return 'multi' if self.kind == 'multi' else 'single'

    def log_path(self, root=None):
        if self.program in ('jupiter', 'http'):
            return os.path.join(_CFG().serve.JUPITER_LOG_DIR.format(out=self.plan.out), self.c['log'])
        return self.at('log', root)

    def cwd(self):
        # http: the site's parent (8801's standing command, 2026-11-14: --directory names the site, so a site swap by rename
        # leaves the process no stale working directory)
        return os.path.dirname(os.path.normpath(self.root)) if self.program == 'http' else self.plan.ops

    def record_path(self, port=None):
        return os.path.join(self.plan.out, '_records', 'serve', 'pages', f'{self.tier}_{port or self.port}.json')


class Plan:
    def __init__(self):
        self.gates = []          # [(gate, message)] refusals found while planning (raised by the caller unless dry-run)
        self.notes = []
        self.forced_after = {}   # forced page key -> the run this serve registers for it (forced_siblings)
        self.forced_view = None  # the transient-search view the forced pages are paired with (forced.paired_view; the switch)

    def refuse_later(self, gate, msg):
        self.gates.append((gate, msg))

    def first_page(self):
        """The first page of this plan's family in config order, whatever --pages selects: the page whose served run is the
        family's 'served' run (the carry source 'served', journals.resolve_carry_from)."""
        return next(pg for pg in self.all_pages if pg.family == self.family)

    # ---- environment and argv of the servers -----------------------------------------------------------------------
    def env_key(self):
        """serve.SERVER_ENV key of this plan's servers: '<tier>-<family>' when config has one (the snapshot forced pages,
        whose launcher is not snapshot_serve.sh), else the tier."""
        k = f'{self.tier}-{self.family}'
        return k if k in _CFG().serve.SERVER_ENV else self.tier

    def server_env_extra(self, page=None):
        """The page servers' extra environment: serve.SERVER_ENV of env_key(); a pinned page's own 'env' when it has one."""
        S = _CFG().serve
        fmt = dict(deep_xtier=S.DEEP_XTIER.format(label=self.label or ''), other_deep_xtier=S.OTHER_TIERS_DEEP_XTIER,
                   tenmin_xtier=S.TENMIN_XTIER)
        env = page.c['env'] if page is not None and page.c.get('env') is not None else S.SERVER_ENV[self.env_key()]
        return {k: str(v).format(**fmt) for k, v in env.items()}

    def server_env(self):
        return scripts.server_env(self.server_env_extra())

    def title(self, page, utc, root=None):
        S = _CFG().serve
        if page.family == 'forced':
            run_name = os.path.basename(os.path.normpath(root)) if root else page.run_name
            samples, gen = forced_generation(page, run_name)
            return S.FORCED_TITLE.format(S=page.stokes, what=S.SNAPSHOT_WHAT.get(page.kind, ''), run_name=run_name, utc=utc,
                                         samples=samples, generation=gen)
        if page.family == 'forced-tier':
            D = root or page.root
            samples, _gen = forced_tier_words(page, D)
            return S.FORCED_TIER_TITLE.format(S=page.stokes, what=S.FORCED_TIER_WHAT[page.kind], samples=samples, utc=utc,
                                              run_name=os.path.basename(os.path.normpath(D)))
        if page.family == 'pinned':
            return page.c['title']
        t = S.TITLE.get((self.tier, page.title_kind()))
        if t is None:
            return None
        return t.format(tier_text=S.DEEP_TIER_TEXT, S=page.stokes, label=self.label or '', utc=utc, lsts=self.lsts_text,
                        what=S.SNAPSHOT_WHAT.get(page.kind, ''))

    def registered_run(self, page):
        """The --run of the page's entry p<port> in this plan's two-port registry (register.registry_dir), or None."""
        from exopipe.phase3.lib.serve import register
        e = register.registered_entry(register.registry_dir(self), f'p{page.port}')
        a = (e or {}).get('argv') or []
        i = a.index('--run') + 1 if '--run' in a else len(a)
        return a[i] if i < len(a) else None

    def forced_siblings(self, page):
        """--sibling args of a forced page: each OTHER page of its family (forced: the 10-s / 1-min pages; forced-tier: the
        10-minute / deep pages) - its catalogue view as served AFTER this serve: the run this serve registers for it
        (self.forced_after) or else the run its registry entry serves now - in the family's config order
        (snapshot_forced_gen5_serve_20261026.sh:47-56, '<port>:<run>/source_catalogs_<S>:<label>'; the live entries p9120-p9123
        the same); a page with neither gives no link."""
        S = _CFG().serve
        out = []
        for o in self.all_pages:
            if o.family != page.family or o.key == page.key:
                continue
            run = self.forced_after.get(o.key) or self.registered_run(o)
            if run:
                T = (S.FORCED_TIER_SIBLING_T[o.kind] if o.family == 'forced-tier' else '10-s' if o.kind == '10s' else '1-min')
                out += ['--sibling', f'{o.port}:{o.at("catalog", run)}:{S.FORCED_SIBLING_LABEL.format(S=o.stokes, T=T)}']
        return out

    def argv(self, page, utc, root=None):
        """The server command line of ``page`` (a review page served from ``root``, default the page's run directory)."""
        S = _CFG().serve
        py = scripts.interpreter(page.c['python']) if page.c.get('python') else self.python
        if page.program == 'jupiter':
            return [py, 'jupiter_server.py', page.root, str(page.port)]
        if page.program == 'http':
            return [py, '-m', 'http.server', str(page.port), '--bind', '127.0.0.1', '--directory', page.root]
        U = root or page.root
        a0 = page.c.get('code') or S.SERVER_ARGV0[self.tier].format(ops=self.ops)    # a pinned page: its own code
        if page.family in FORCED_FAMILIES:
            argv = [py, a0, '--run', U, '--queue', page.at('queue', U), '--catalog-dir', page.at('catalog', U),
                    '--port', str(page.port), '--loc-offsets', '--default-fp-reason', S.DEFAULT_FP_REASON['snapshot']]
            argv += self.forced_siblings(page) + ['--title', self.title(page, utc, U)]
            if self.parity:
                argv += ['--lists-dir', self.lists_dir]
            return argv
        argv = [py, a0, '--run', U, '--queue', page.at('queue', U)]
        if self.tier in ('deep', 'sublst'):
            argv += ['--labels-dir', page.at('labels', U)]
        argv += ['--catalog-dir', page.at('catalog', U), '--port', str(page.port)]
        fp = S.DEFAULT_FP_REASON.get(self.tier, SERVER_DEFAULT_FP_REASON)
        if fp != SERVER_DEFAULT_FP_REASON:
            argv += ['--default-fp-reason', fp]
        if page.c.get('loc_dir'):
            argv += ['--loc-dir', page.at('loc_dir', U)]
        if page.loc_offsets:
            argv += ['--loc-offsets']
        t = self.title(page, utc)
        if t is not None:
            argv += ['--title', t]
        if self.tier == 'snapshot':
            for o in self.all_pages:            # the other pages of the page's own family (review: 8779-8782; comb: 8986-8989)
                if o.key == page.key or o.family != page.family:
                    continue
                lab = S.SNAPSHOT_SIBLING_LABEL.format(S=o.stokes, T='10-s' if o.kind == '10s' else '1-min')
                argv += ['--sibling', f'{o.port}:{o.at("catalog", U)}:{lab}']
        # the page's MULTIPLES CHAIN (Gregg 2026-10-27; exopipe_ops 4c42f91): after the title, as the two-port registry
        # entries of the deep and 10-minute singles pages carry them (p8766/p8768 both, p8787/p8788 the queue)
        if page.c.get('mchain_queue'):
            argv += ['--mchain-queue', page.at('mchain_queue', U)]
        if page.c.get('mchain_catalog'):
            argv += ['--mchain-catalog-dir', page.at('mchain_catalog', U)]
        argv += list(page.c.get('extra') or ())      # a pinned page's trailing arguments (p9411 / p9412: --filter-defaults off)
        if self.parity:
            argv += ['--lists-dir', self.lists_dir]
        return argv

    def hub_entry(self, page, utc, root=None):
        """(Stokes, hub tier, [key=value ...]) of build_review_hub.py --set for ``page``, in the launcher's order; None for a
        page without a button (tunnel only). A forced page's run name, as title(): from ``root``, else from the run its entry
        in this plan's registry serves (registered_run: what the button points at after a switch or a sibling refresh), else
        the plan's name; G26 refuses a name whose tag has no serve.FORCED_GENERATIONS text."""
        if page.hub is None:
            return None
        S = _CFG().serve
        run_name = page.run_name
        if page.family == 'forced':
            root = root or self.registered_run(page)
            run_name = os.path.basename(os.path.normpath(root)) if root else page.run_name
            samples, g = forced_generation(page, run_name)
            gen = S.FORCED_HUB_GENERATION.format(S=page.stokes, kind=page.kind, samples=samples, generation=g)
        elif page.family == 'forced-tier':
            D = root or self.registered_run(page) or page.root
            run_name = os.path.basename(os.path.normpath(D))
            samples, g = forced_tier_words(page, D)
            gen = S.FORCED_TIER_HUB_GENERATION.format(S=page.stokes, kind=page.kind, samples=samples, gen=g)
        else:
            gen = S.HUB_GENERATION[self.tier].format(lsts=self.lsts_text, label=self.label or '')
        vals = {'run': run_name, 'generation': gen, 'served': utc, 'live': '1', 'port': str(page.port), 'note': ''}
        return page.hub[0], page.hub[1], [f'{k}={vals[k]}' for k in HUB_KV_ORDER[(self.tier, page.family)]]


def forced_generation(page, run_name):
    """(samples text, cutout generation text) of a forced page serving ``run_name`` = Run_SNAPFORCED_<S><T>_<tag>
    (serve.FORCED_RUN_NAME): the first two fields of serve.FORCED_GENERATIONS[tag]. G26 refuses a run of another name or a tag without an entry -
    the title and the hub button name the samples and the generation the page shows (generation 6 adds LPT)."""
    S = _CFG().serve
    pre = S.FORCED_RUN_NAME.format(S=page.stokes, T=page.kind, tag='')
    tag = run_name[len(pre):] if run_name.startswith(pre) else None
    if tag not in S.FORCED_GENERATIONS:
        raise Refusal(f'G26 forced page {page.key}: run {run_name!r} - no serve.FORCED_GENERATIONS entry for its tag {tag!r} '
                      f'(the samples and cutout generation its title and hub button name; known: '
                      f'{", ".join(sorted(S.FORCED_GENERATIONS))})')
    return tuple(S.FORCED_GENERATIONS[tag][:2])


def forced_tier_words(page, D, name=None):
    """(samples text, generation number) of a 10-minute / deep forced page serving run directory ``D`` (named ``name``, default
    its own name: a new build <run>.new is read before it is installed): the samples from the run's own queue header
    (serve.FORCED_TIER_QUEUE_HEADER, tier_forced_page_20261027.py samples_short()), the generation from its name
    forced_tier_pages.RUN_NAME (..._gen<N>). G26 refuses a run of another name, a queue without that header, or a header naming
    another Stokes or tier than the page's - the title and the hub button name what the run holds."""
    import re
    C = _CFG()
    name = name or os.path.basename(os.path.normpath(D))
    if name != C.forced_tier_pages.RUN_NAME.format(S=page.stokes, tier=page.kind):
        raise Refusal(f'G26 forced page {page.key}: run {name!r} is not forced_tier_pages.RUN_NAME for Stokes {page.stokes}, '
                      f'{page.kind}')
    m = re.search(r'_gen(\d+)$', name)
    q = page.at('queue', D)
    try:
        with open(q) as f:
            first = f.readline().rstrip('\n')
    except OSError as e:
        raise Refusal(f'G26 forced page {page.key}: no queue {q} ({e})')
    h = re.match(C.serve.FORCED_TIER_QUEUE_HEADER, first)
    if not (m and h and h['S'] == page.stokes and h['tier'] == page.kind):
        raise Refusal(f'G26 forced page {page.key}: {q} begins {first[:120]!r}, not the family T queue header of Stokes '
                      f'{page.stokes}, {page.kind} tier (serve.FORCED_TIER_QUEUE_HEADER)')
    return h['samples'], m.group(1)


def _strip_run(label):
    return label[4:] if label and label.startswith('Run_') else label


def sublst_lsts(spec=None):
    """The LSTs of the cumulative product: --lsts, else every Run_SLST<hh>B under the runs root (sublst_publish_lst.sh:42,
    'ls -d .../Run_SLST[0-9][0-9]B | sed ... | sort')."""
    from exopipe.phase3.core import cells
    S = _CFG().serve
    if spec:
        out = cells.expand(spec, universe=cells.LSTS, kind='LST')
        for l in out:
            try:
                cells.parse_lst(l)
            except ValueError as e:
                raise Refusal(f'--lsts: {e}')
        return out
    found = []
    for d in sorted(glob.glob(os.path.join(run_root('sublst'), S.SUBLST_LST_RUN_GLOB))):
        m = re.search(r'Run_SLST(\d\d)B$', d)
        if m and os.path.isdir(d):
            found.append(m.group(1) + 'h')
    return sorted(found)


def _chain_host():
    ch = scripts.site_value('CHAIN_HOST')
    return ch.get('snapshot') if isinstance(ch, dict) else ch


def make(args, out):
    """Build the Plan from the parsed arguments and the run's product root ``out``. Raises Refusal for what makes a plan
    impossible (G26 label, unknown page or family); collects the other plan-time refusals in plan.gates."""
    CFG = _CFG()
    S = CFG.serve
    p = Plan()
    p.tier = args.tier
    p.family = getattr(args, 'family', None) or FAMILY_DEFAULT[p.tier]
    p.out = os.path.abspath(out)
    p.parity = CFG.production_root_of(p.out) is None
    p.offset = int(getattr(args, 'port_offset', 0) or 0)
    p.ops = scripts.ops_root()
    p.python = scripts.interpreter('default')
    p.label = None
    p.lsts, p.lsts_text = [], ''
    p.snap_label = getattr(args, 'label', None)
    p.forced_tag = getattr(args, 'forced_tag', None)
    p.source = None
    families = sorted({c['family'] for (t, k), c in S.PAGES.items() if t == p.tier})
    if p.family not in families:
        raise Refusal(f'--family {p.family}: tier {p.tier} has the page families {", ".join(families)}')
    label = getattr(args, 'run_label', None)
    restart_like = bool(getattr(args, 'restart', None) or getattr(args, 'stop', None))
    if (p.family != 'review' and (p.tier, p.family) != ('snapshot', 'comb')) or restart_like:
        p.label = _strip_run(label) if label else None
        p.run_name = 'jupiter' if p.tier == 'jupiter' else (f'Run_{p.label}' if p.label else p.family)
    elif p.tier == 'deep':
        if not label:
            raise Refusal('G26 label required: --run P3S5 (or Run_P3S5) - the deep serve has no default run (edge case E-6; '
                          'p3s3_serve_stokes.sh:9 defaulted to P3S3)')
        p.label = _strip_run(label)
        p.run_name = S.RUN_NAME['deep'].format(label=p.label)
    elif p.tier == 'sublst':
        p.run_name = f'Run_{_strip_run(label)}' if label else S.RUN_NAME['sublst']
        p.label = _strip_run(p.run_name)
        p.lsts = sublst_lsts(getattr(args, 'lsts', None))
        p.lsts_text = ' '.join(p.lsts)
    else:   # snapshot review, and the combined view's pages (family comb: the view itself is the review copy, no source)
        if not label:
            raise Refusal(f'G26 label required: --run {S.SNAPSHOT_RUN_PREFIX}... (the chain run name) - the snapshot serve has '
                          f'no default run')
        if not label.startswith(S.SNAPSHOT_RUN_PREFIX):
            raise Refusal(f'--run {label!r}: a snapshot run name starts with {S.SNAPSHOT_RUN_PREFIX!r} '
                          f'(snapshot_rechain_batch.sh:49)')
        p.run_name = S.RUN_NAME['snapshot'].format(label=label)
        p.label = p.snap_label
        src = getattr(args, 'source', None)
        if p.family == 'comb':
            if src:
                raise Refusal('--source: the combined view is built in place under the snapshot review root and served from there '
                              '(snapshot_serve.sh: "copy skipped: $RUN is the review host\'s copy") - no source to copy')
        elif src:
            p.source = src
        elif _chain_host():
            p.source = f'{_chain_host()}:{S.SNAPSHOT_SOURCE.format(label=p.run_name)}'
    if p.family == 'forced' and not p.forced_tag and not restart_like and not getattr(args, 'forced_refresh_siblings', False):
        raise Refusal('G26 label required: --forced-tag TAG (the forced runs are Run_SNAPFORCED_<S><T>_<TAG>, '
                      'snapshot_forced_serve_20261007.sh:30)')
    rr = run_root(p.tier) if p.family in ('review', 'comb') else None    # forced / pinned pages: one run per page (Page.root)
    # the combined view's member batch runs (G27's restore-only go reads them): production's, beside the view under the snapshot
    # review root - also for a parity view under --out, whose members are the production batch runs it was built from
    p.member_root = run_root(p.tier) if p.family == 'comb' else None
    p.U_src = os.path.join(rr, p.run_name) if rr else None
    p.U = (os.path.join(p.out, p.tier, p.run_name) if p.parity else p.U_src) if rr else None
    p.hub_dir = os.path.join(p.out, 'review_hub') if p.parity else S.HUB_DIR
    p.lists_dir = os.path.join(p.out, 'review_lists') if p.parity else None
    p.log_path = os.path.join(p.U, 'incremental.log') if p.U and p.family in ('review', 'comb') else None

    # ---- pages ----------------------------------------------------------------------------------------------------
    p.all_pages = [Page(p, k, S.PAGES[(t, k)]) for (t, k) in S.PAGES if t == p.tier]
    fam = [pg for pg in p.all_pages if pg.family == p.family]
    keys = [pg.key for pg in fam]
    want = [x.strip() for x in (getattr(args, 'pages', None) or '').split(',') if x.strip()]
    bad = [w for w in want if w not in keys]
    if bad:
        raise Refusal(f'--pages {",".join(bad)}: not a page of tier {p.tier} family {p.family} (pages: {", ".join(keys)})')
    stokes = [x.strip() for x in (getattr(args, 'stokes', None) or '').split(',') if x.strip()]
    p.pages = [pg for pg in fam if (not want or pg.key in want) and (not stokes or pg.stokes in stokes)]
    if not restart_like:
        ro = [pg.key for pg in p.pages if pg.start == 'restart-only']
        if ro:
            raise Refusal(f'pages {", ".join(ro)} ({p.family}) have no production launcher defining their argv: serve '
                          f'restarts or stops them with their exact argv only (--restart PORTS / --stop PORTS)')

    # ---- plan-time gates ------------------------------------------------------------------------------------------
    if p.parity and p.offset < S.PARITY_PORT_OFFSET_MIN:
        p.refuse_later('G23', f'parity run (--out {p.out} is not a production root) needs --port-offset >= '
                              f'{S.PARITY_PORT_OFFSET_MIN}: a parity run never replaces a production server')
    if not p.parity and p.offset:
        p.refuse_later('G23', f'a production serve binds the ruled ports (--port-offset {p.offset} refused)')
    seen = {}
    for (t, k), c in S.PAGES.items():
        for port in [c['port']] + ([c['twin']] if c.get('twin') else []):
            if port in S.OTHER_SERVED_PORTS:
                p.refuse_later('G13', f'config: page {t}/{k} names port {port}, served by something else '
                                      f'({S.OTHER_SERVED_PORTS[port]})')
            if port in seen and seen[port] != (t, k):
                p.refuse_later('G13', f'config: port {port} is given to two pages ({seen[port]} and {(t, k)})')
            seen[port] = (t, k)
    if not p.parity:
        from exopipe.phase3.core.env import host
        rh = scripts.site_value('REVIEW_HOST')
        me = host()
        if not rh:
            p.refuse_later('G25', 'site REVIEW_HOST is unset: a production serve runs only on the review host '
                                  '(sublst_serve.sh:12; sublst_cumulative.sh:13)')
        elif rh != me:
            p.refuse_later('G25', f'production serve on {me}, but the review host is {rh} (sublst_serve.sh:12)')
    return p
