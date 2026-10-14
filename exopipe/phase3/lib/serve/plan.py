"""The serve Plan: which run is served from where, which pages on which ports, with which argv and environment.

Production vs parity (DESIGN 11.1; spec serve.md 10): when ``--out`` lies under a production root (paths.PRODUCTION_ROOTS)
the plan serves the production run directory itself (only once paths.PRODUCTION_WRITES is True and the cutover is
approved); otherwise it is a PARITY plan: every file serve writes goes under ``<out>/<tier>/<run name>`` (deep: a mirror of
the served run, lib/serve/mirror.py; sub-LST: the cumulative build and swap; snapshot and forced pages: a byte copy), the
hub registry under ``<out>/review_hub`` (planned, not rendered: build_review_hub.py hard-codes its directory, spec 12.5),
the inspection hub under ``<out>/inspection_hub``, the shared lists under ``<out>/review_lists``, and every port is bound at
port + ``--port-offset`` (>= serve.PARITY_PORT_OFFSET_MIN; the cross-tier link strings keep the production ports so pages
compare). The Jupiter pages read their production directories directly (read-only servers) in both modes.

Argv per page family reproduces the launchers exactly (orders included):
  deep review single  p3s3_serve_stokes.sh:79      --run --queue --labels-dir --catalog-dir --port --title
  deep review multi   p3s3_serve_multiples.sh:152  --run --queue --labels-dir --catalog-dir --port (no title, divergence D2)
  sub-LST review      sublst_serve.sh:140 / sublst_cumulative.sh:139-140  ... --port [--loc-offsets] --title
  snapshot review     snapshot_serve.sh:72-76      --run --queue --catalog-dir --port --default-fp-reason --loc-dir
                      --loc-offsets --title --sibling x3
  snapshot forced     snapshot_forced_serve_20261007.sh:49-51  --run --queue --catalog-dir --port --loc-offsets
                      --default-fp-reason --sibling (what each other forced port serves NOW) --title
  jupiter             jupiter_server.py ROOT PORT (cwd ops); python -m http.server PORT --bind 127.0.0.1 --directory DIR
plus, in a parity plan only, --lists-dir <out>/review_lists on classify pages (spec 7.2).
"""
import glob
import os
import re

from exopipe.phase3.core.record import Refusal
from exopipe.phase3.lib.serve import procs, scripts

SERVER_DEFAULT_FP_REASON = 'bilobe'      # classify_transients.py:2438 (the in-place server's own default)
# the order of the key=value arguments each launcher passes to build_review_hub.py --set (the registry entry's key order)
HUB_KV_ORDER = {('deep', 'review'): ('run', 'generation', 'served', 'live', 'port'),          # p3s3_serve_stokes.sh:84
                ('sublst', 'review'): ('run', 'generation', 'served', 'live', 'port'),        # sublst_serve.sh:142
                ('snapshot', 'review'): ('port', 'run', 'generation', 'served', 'live'),      # snapshot_serve.sh:101
                ('snapshot', 'forced'): ('port', 'run', 'generation', 'served', 'live', 'note')}   # snapshot_forced_serve:96-98
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
        return self.root if self.program == 'http' else self.plan.ops

    def record_path(self, port=None):
        return os.path.join(self.plan.out, '_records', 'serve', 'pages', f'{self.tier}_{port or self.port}.json')


class Plan:
    def __init__(self):
        self.gates = []          # [(gate, message)] refusals found while planning (raised by the caller unless dry-run)
        self.notes = []

    def refuse_later(self, gate, msg):
        self.gates.append((gate, msg))

    # ---- environment and argv of the servers -----------------------------------------------------------------------
    def env_key(self):
        """serve.SERVER_ENV key of this plan's servers: '<tier>-<family>' when config has one (the snapshot forced pages,
        whose launcher is not snapshot_serve.sh), else the tier."""
        k = f'{self.tier}-{self.family}'
        return k if k in _CFG().serve.SERVER_ENV else self.tier

    def server_env_extra(self):
        S = _CFG().serve
        fmt = dict(deep_xtier=S.DEEP_XTIER.format(label=self.label or ''), other_deep_xtier=S.OTHER_TIERS_DEEP_XTIER,
                   tenmin_xtier=S.TENMIN_XTIER)
        return {k: str(v).format(**fmt) for k, v in S.SERVER_ENV[self.env_key()].items()}

    def server_env(self):
        return scripts.server_env(self.server_env_extra())

    def title(self, page, utc):
        S = _CFG().serve
        if page.family == 'forced':
            return S.FORCED_TITLE.format(S=page.stokes, what=S.SNAPSHOT_WHAT.get(page.kind, ''), utc=utc)
        t = S.TITLE.get((self.tier, page.title_kind()))
        if t is None:
            return None
        return t.format(tier_text=S.DEEP_TIER_TEXT, S=page.stokes, label=self.label or '', utc=utc, lsts=self.lsts_text,
                        what=S.SNAPSHOT_WHAT.get(page.kind, ''))

    def forced_siblings(self, page):
        """--sibling args of a forced page: each OTHER forced port's catalogue directory as served NOW (its server's
        --catalog-dir from /proc); a port with no review server gives no link (snapshot_forced_serve_20261007.sh:35-41)."""
        S = _CFG().serve
        out = []
        for o in self.all_pages:
            if o.family != 'forced' or o.key == page.key:
                continue
            for pid in procs.listeners(o.port):
                info = procs.proc_info(pid)
                if procs.is_review_server(info) and procs.argv_value(info['argv'], '--port') == str(o.port):
                    cd = procs.argv_value(info['argv'], '--catalog-dir')
                    if cd:
                        T = '10-s' if o.kind == '10s' else '1-min'
                        out += ['--sibling', f'{o.port}:{cd}:{S.FORCED_SIBLING_LABEL.format(S=o.stokes, T=T)}']
                    break
        return out

    def argv(self, page, utc, root=None):
        """The server command line of ``page`` (a review page served from ``root``, default the page's run directory)."""
        S = _CFG().serve
        py = self.python
        if page.program == 'jupiter':
            return [py, 'jupiter_server.py', page.root, str(page.port)]
        if page.program == 'http':
            return [py, '-m', 'http.server', str(page.port), '--bind', '127.0.0.1', '--directory', page.root]
        U = root or page.root
        a0 = S.SERVER_ARGV0[self.tier].format(ops=self.ops)
        if page.family == 'forced':
            argv = [py, a0, '--run', U, '--queue', page.at('queue', U), '--catalog-dir', page.at('catalog', U),
                    '--port', str(page.port), '--loc-offsets', '--default-fp-reason', S.DEFAULT_FP_REASON['snapshot']]
            argv += self.forced_siblings(page) + ['--title', self.title(page, utc)]
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
            for o in self.all_pages:
                if o.key == page.key or o.family != 'review':
                    continue
                lab = S.SNAPSHOT_SIBLING_LABEL.format(S=o.stokes, T='10-s' if o.kind == '10s' else '1-min')
                argv += ['--sibling', f'{o.port}:{o.at("catalog", U)}:{lab}']
        if self.parity:
            argv += ['--lists-dir', self.lists_dir]
        return argv

    def hub_entry(self, page, utc):
        """(Stokes, hub tier, [key=value ...]) of build_review_hub.py --set for ``page``, in the launcher's order; None for a
        page without a button (tunnel only)."""
        if page.hub is None:
            return None
        S = _CFG().serve
        if page.family == 'forced':
            gen = S.FORCED_HUB_GENERATION.format(S=page.stokes, kind=page.kind)
        else:
            gen = S.HUB_GENERATION[self.tier].format(lsts=self.lsts_text, label=self.label or '')
        vals = {'run': page.run_name, 'generation': gen, 'served': utc, 'live': '1', 'port': str(page.port), 'note': ''}
        return page.hub[0], page.hub[1], [f'{k}={vals[k]}' for k in HUB_KV_ORDER[(self.tier, page.family)]]


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
    if p.family != 'review' or restart_like:
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
    else:   # snapshot review
        if not label:
            raise Refusal(f'G26 label required: --run {S.SNAPSHOT_RUN_PREFIX}... (the chain run name) - the snapshot serve has '
                          f'no default run')
        if not label.startswith(S.SNAPSHOT_RUN_PREFIX):
            raise Refusal(f'--run {label!r}: a snapshot run name starts with {S.SNAPSHOT_RUN_PREFIX!r} '
                          f'(snapshot_rechain_batch.sh:49)')
        p.run_name = S.RUN_NAME['snapshot'].format(label=label)
        p.label = p.snap_label
        src = getattr(args, 'source', None)
        if src:
            p.source = src
        elif _chain_host():
            p.source = f'{_chain_host()}:{S.SNAPSHOT_SOURCE.format(label=p.run_name)}'
    if p.family == 'forced' and not p.forced_tag and not restart_like and not getattr(args, 'forced_refresh_siblings', False):
        raise Refusal('G26 label required: --forced-tag TAG (the forced runs are Run_SNAPFORCED_<S><T>_<TAG>, '
                      'snapshot_forced_serve_20261007.sh:30)')
    rr = run_root(p.tier) if p.family == 'review' else None    # forced pages: one run per page (Page.root)
    p.U_src = os.path.join(rr, p.run_name) if rr else None
    p.U = (os.path.join(p.out, p.tier, p.run_name) if p.parity else p.U_src) if rr else None
    p.hub_dir = os.path.join(p.out, 'review_hub') if p.parity else S.HUB_DIR
    p.inspection_hub_dir = os.path.join(p.out, 'inspection_hub') if p.parity else S.INSPECTION_HUB_DIR
    p.lists_dir = os.path.join(p.out, 'review_lists') if p.parity else None
    p.log_path = os.path.join(p.U, 'incremental.log') if p.U and p.family == 'review' else None

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
