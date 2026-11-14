"""One served review page's own rule, evaluated outside its server.

The page's code (the registry's `code`: the page server's script, with its filter module and its bright-source filter
module beside it) is loaded by its path, and the page's own object is built from the page's own arguments and environment
exactly as its server builds it at startup (read_queue, load_sources, App). These are readers: nothing is written and no
server is started. From that object:
  - every detection's filter values and the percentile breakpoints, as the page's sliders have them (review_filters.build);
  - the page's default filters;
  - the page's match to a known pulsar;
  - the page's bright-source filter at an allowed mean sky loss: the staircase the page's "go" button finds
    (bsf_search), here without the live server's 10-second budget, so the answer does not depend on the load of a node.
Nothing of the rule is re-derived here; a page whose code no longer offers these names raises, and the caller says so.

The process is left as it was: the page's environment variables and the module search path are restored (the page code
puts its own directory and production's code directory on the path), and the modules the page imported from its own
directory are dropped from the module table, so each page has its own copies, as each has its own server process.
One thing is the running process's and not the page's: the page reads the band frequency table from
`exopipe.common.phase2_config`, and in this process that name is this package (the table is the same one by rule).
"""
import contextlib
import importlib.util
import io
import itertools
import json
import os
import sys

_SERIAL = itertools.count()


def _arg(argv, key):
    return argv[argv.index(key) + 1] if key in argv and argv.index(key) + 1 < len(argv) else None


class PageRule:
    def __init__(self, entry, sky_loss):
        """entry: the page's registry entry (code, argv, env). sky_loss: the allowed mean sky loss of the bright-source
        filter, a fraction."""
        code, argv, env = entry['code'], list(entry.get('argv') or []), dict(entry.get('env') or {})
        here = os.path.dirname(os.path.abspath(code))
        spec = importlib.util.spec_from_file_location(f'exopipe3_notify_page_{next(_SERIAL)}', code)
        CT = importlib.util.module_from_spec(spec)
        old = {k: os.environ.get(k) for k in env}
        path_obj, path0, mods0 = sys.path, list(sys.path), set(sys.modules)
        if here not in sys.path:
            sys.path.append(here)                # the page imports the modules beside it by name; last, so it shadows nothing
        os.environ.update({k: str(v) for k, v in env.items()})
        log = io.StringIO()
        try:
            with contextlib.redirect_stdout(log):
                spec.loader.exec_module(CT)      # a fresh module per page: the page class keeps per-page state on itself
                run = os.path.realpath(_arg(argv, '--run'))
                entries, prov, sel = CT.read_queue(_arg(argv, '--queue'))
                sources = CT.load_sources(run, blind=False, catalog_dir=_arg(argv, '--catalog-dir'),
                                          extra_dir=_arg(argv, '--mchain-catalog-dir'))
                if _arg(argv, '--mchain-queue'):
                    m_entries, _prov, m_sel = CT.read_queue(_arg(argv, '--mchain-queue'))
                    CT.App.MCHAIN_RAW = [(s, p, m_sel.get((s, p), '')) for s, p in m_entries]
                CT.App.FILTER_DEFAULTS = _arg(argv, '--filter-defaults') != 'off'
                app = CT.App(run, sources, entries, prov, _arg(argv, '--labels-dir') or os.path.join(run, 'judgements'),
                             _arg(argv, '--panel-format') or 'regen_panels7', entry_sel=sel)
                self.CT, self.app, self.run = CT, app, run
                self.ref = json.loads(app.metrics_json)['ref']           # {filter key: 101 percentile breakpoints}
                self.defaults = dict(getattr(app, 'fp_default_allowed', None) or {})
                self.stokes = getattr(app, 'mstokes', None)
                self._bright(sky_loss)
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            path_obj[:] = path0
            sys.path = path_obj
            for name in set(sys.modules) - mods0:
                f = getattr(sys.modules.get(name), '__file__', None)
                if f and os.path.dirname(os.path.abspath(f)) == here:
                    del sys.modules[name]
            self.log = log.getvalue()
        # the page's own warnings on its bright-source filter (parent catalogs changed or missing since its file was
        # made: those cells never remove a detection and are left out of the sky loss), for the email's limits
        self.warnings = [ln.strip() for ln in self.log.splitlines()
                         if ln.startswith('bright-source filter:') and 'changed or missing' in ln]

    def _bright(self, loss):
        """The staircase of the page's bright-source filter for the allowed mean sky loss, with the page's default settings
        (elevation exponents as the panel starts; the readmission rule and the pulsar exemption off, as the panel starts),
        and the detections it removes: self.removed = {(source id, position among the source's detections)}."""
        CT, app = self.CT, self.app
        P = getattr(app, 'bsf', None)
        if P is None:
            raise RuntimeError(f'the page has no readable bright-source filter file ({CT.bsf_path(app)})')
        key = (0.0,) + tuple(CT.BSF_DEFAULT_B) + (False,) + tuple(CT.BSF_DEFAULT_X) + (False,)
        prep = CT._bsf_opt_prep(app, key, CT.BSF_OPT_LEVELS, CT.BSF_OPT_RSTEP, CT.BSF_OPT_NR)
        K, n_removed, _used, _moves, how, _runs = CT.bsf_search(prep, loss, budget_s=float('inf'))
        steps = CT.bsf_steps_of(prep, K)
        sky = [v for v in CT.bsf_sky_browser(app, steps, key[0], key[1], key[2]).values() if v is not None]
        mean = sum(sky) / len(sky) if sky else 0.0
        if mean > loss + 1e-6:
            raise RuntimeError(f'the staircase found for an allowed mean sky loss of {loss:.3f} removes {mean:.4f} of the '
                               f'searched sky')
        self.removed = set()
        if steps:
            rem = CT.bsf_det_state(P, (tuple(sorted(tuple(s) for s in steps)),) + key)[0]
            self.removed = {(P['sids'][s], int(p)) for s, p, r in zip(P['d_src'].tolist(), P['d_pos'].tolist(), rem.tolist())
                            if r}
        # detections the filter has no precomputed data for (no parent catalog, or a stale one): the page never removes
        # them; the email counts them per hour (sources.collect)
        self.no_data = {(P['sids'][s], int(p)) for s, p, k in zip(P['d_src'].tolist(), P['d_pos'].tolist(),
                                                                   P['ok'].tolist()) if not k}
        self.bright = dict(allowed=loss, mean_sky_loss=mean, bands=len(sky), steps=[list(s) for s in steps], how=how,
                           candidates_removed=int(n_removed), listed=int(prep['n_listed']),
                           detections_removed=len(self.removed))
        self._psr = P['BP'].psr_match

    def hour(self, lst, date):
        """The detections of the hour of the candidates the page lists: [(source id, position among the source's
        detections, the detection, its filter values, the source's catalog row of the detection's band)]."""
        app, out = self.app, []
        for sid in dict.fromkeys(app.queue_ids):
            s = app.sources.get(sid)
            if s is None or not s.detections:
                continue
            for pos, (d, m) in enumerate(zip(s.detections, app.mdets[sid])):
                if d.get('lst') == lst and str(d.get('date') or '')[:10] == date:
                    row = s.bands.get(d.get('band')) or next(iter(s.bands.values()), {})
                    out.append((sid, pos, d, m, row))
        return out

    def passes_defaults(self, m):
        """Whether a detection with these values passes every default filter of the page (review_filters.passes)."""
        return bool(self.app.RF.passes([m], self.defaults))

    def pulsar(self, sid):
        """Whether the page's own match gives the candidate to a known pulsar: a catalog row of the candidate that the
        bright-source filter's match finds credible (offset within the 99% position error, fewer than 0.5 chance matches
        expected) and whose cross-match names a pulsar. That match also covers long-period transients; those are not
        pulsars and are not removed here."""
        s = self.app.sources.get(sid)
        return any(self._psr(r) and 'psr' in str(r.get('xm_best') or '').lower() for r in (s.bands.values() if s else ()))
