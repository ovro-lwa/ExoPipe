"""Old module names of the Phase 3 refactor: the table of moves, and the helpers the compatibility shims use.

On the refactor branch NO file sits at an old path. Every package move is a PURE rename commit (byte-identical, so git
follows the rename when main is merged in), followed by a separate commit that rewrites the package's importers to
the new names. The compatibility layer for code outside the package - ops shell launchers that run
``python -m exopipe.phase3.<old>``, the Phase 2 controller's by-path check of search_v_transients_v2.py, the clv15 and
exopipe_ops modules that import old names - is written ONCE, as the last commit at the final merge into main, from
the tables below (MOVE_PLAN.md sections 3 and 8; DESIGN.md 11.1):

``LEGACY``    old dotted name -> new dotted name, one entry per move made on the branch (filled by each move commit).
``RUNNABLE``  the old names that are LAUNCHED, either ``python -m <old>`` (kind '-m') or ``python <old path>.py``
              (kind 'path'). Their shim must be a FORWARDER: a sys.modules swap alone does not run the new module's
              __main__ block, so the launch would do nothing and exit 0.

Shim file bodies at the old path (cutover only; never on the branch):

    from exopipe.phase3.core._shim import alias      # imported names
    alias(__name__, 'exopipe.phase3.lib.masters.epoch_gate')

    from exopipe.phase3.core._shim import forward    # launched names (also aliases when imported)
    forward(__name__, 'exopipe.phase3.lib.masters.coadd')

A 'path' forwarder runs with its own directory as sys.path[0]: the repository root must be importable (both of today's
callers put it on PYTHONPATH).

``install_legacy_finder()`` resolves the old names in-process without any file at the old path (an import hook over
LEGACY, returning the new module object itself). Nothing installs it by default; tests/test_refactor_shims.py uses it
for the old-name check (EXOPIPE_SHIM_CHECK=finder).
"""
import importlib
import importlib.abc
import importlib.util
import runpy
import sys

# old dotted name -> new dotted name (one entry per move made on the branch)
LEGACY = {
    # stage 3 (core)
    'exopipe.common.phase3_products': 'exopipe.phase3.core.products',     # moved 742bff3; old-path alias removed
    'exopipe.phase3.residuals': 'exopipe.phase3.core.residuals',
    'exopipe.common.roots': 'exopipe.phase3.config',                     # absorbed verbatim (module-level names)
    'exopipe.common.phase3_paths': 'exopipe.phase3.config',              # absorbed verbatim (module-level names)
    'exopipe.phase3.identify': 'exopipe.phase3.core.fitting',            # fit_empirical_beam absorbed verbatim
    # stage 4: masters
    'exopipe.phase3.coadd_full_pipeline': 'exopipe.phase3.lib.masters.coadd',
    'exopipe.phase3.epoch_gate': 'exopipe.phase3.lib.masters.epoch_gate',
    'exopipe.phase3.flux_correction': 'exopipe.phase3.lib.masters.flux_correction',
    'exopipe.phase3.membership': 'exopipe.phase3.lib.masters.membership',
    'exopipe.phase3.night_image_gate': 'exopipe.phase3.lib.masters.night_image_gate',
    'exopipe.phase3.pb_apply': 'exopipe.phase3.lib.masters.pb_apply',
    'exopipe.phase3.phase3_master_shflux': 'exopipe.phase3.lib.masters.shflux',
    'exopipe.phase3.phase3_pbcorr_masters': 'exopipe.phase3.lib.masters.pbcorr_masters',
    'exopipe.phase3.vclean_masters': 'exopipe.phase3.lib.masters.vclean_masters',
    # stage 4: dewarp
    'exopipe.phase3.phase3_master_dewarp': 'exopipe.phase3.lib.dewarp.master_dewarp',
    # stage 4: install
    'exopipe.phase3.phase3_install': 'exopipe.phase3.lib.install.install',
}

# (kind, old dotted name, --help probe argv or None = never executed by the test, launched by)
RUNNABLE = (
    ('-m', 'exopipe.phase3.coadd_full_pipeline', ['--help'], 'rebuild_cell.sh:126'),
    ('-m', 'exopipe.phase3.membership', ['--help'], 'install_cell.sh:23'),
    ('-m', 'exopipe.phase3.phase3_master_shflux', ['--help'], 'rebuild_cell.sh:143'),
    ('-m', 'exopipe.phase3.phase3_master_dewarp', ['--help'], 'dewarp_lst.sh:58, measure_on_09.sh:14'),
    ('-m', 'exopipe.phase3.phase3_install', ['--help'], 'install_cell.sh:19'),
    ('-m', 'exopipe.phase3.mosaic_coadd_jysr', ['--help'], 'run_mosaic_v2.sh:22, run_mosaic_v2_sub.sh:29, '
                                                          'run_mosaic_jobs.sh:19'),
    ('-m', 'exopipe.phase3.make_band_products', ['--help'], 'run_mosaic_v2_renders.sh:11'),
    # launched by non-production ops fleets (pbcorr_chain_t2*, run_droppass_fleet*, t2*_v_fleet*, ...); phase3_deep with
    # no argument stops at its first line (IndexError on sys.argv) before any I/O; pbcorr_masters is never executed
    # by the test (with or without arguments it lists the co-add roots and processes them).
    ('-m', 'exopipe.phase3.phase3_deep', [], 'non-production ops fleets'),
    ('-m', 'exopipe.phase3.phase3_pbcorr_masters', None, 'non-production ops fleets'),
    ('path', 'exopipe.phase3.search_v_transients_v2', ['--help'], 'phase2_controller.py:750/1049 (R-GC9), '
                                                                  'vclean_masters.py:59'),
    ('path', 'exopipe.common.batch_flow_align', ['--help'], 'coadd_full_pipeline.py:76, phase3_pipeline.py:149 '
                                                            '(py310_orca)'),
)


def new_name(old):
    """The current dotted name of a module that may have moved (unchanged when it has not)."""
    return LEGACY.get(old, old)


def module_path(dotted):
    """Repository-relative file path of a dotted module name (not checked for existence)."""
    return dotted.replace('.', '/') + '.py'


def alias(name, new):
    """Shim body at an old import path: the module being imported as ``name`` IS the module ``new`` (one object,
    shared state)."""
    mod = importlib.import_module(new)
    sys.modules[name] = mod
    return mod


def forward(name, new):
    """Shim body at an old RUNNABLE path: launched (``name == '__main__'``) it runs ``new`` as __main__ with sys.argv
    kept (exit status and SystemExit propagate); imported it aliases ``new`` like ``alias``."""
    if name == '__main__':
        runpy.run_module(new, run_name='__main__', alter_sys=True)
        return None
    return alias(name, new)


class _LegacyFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Import hook over LEGACY: ``import <old>`` returns the new module object itself (its own __spec__ kept)."""

    def __init__(self, table):
        self.table = table
        self._spec = {}

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self.table:
            return None
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        mod = importlib.import_module(self.table[spec.name])
        self._spec[spec.name] = getattr(mod, '__spec__', None)
        return mod

    def exec_module(self, module):
        # the import system set module.__spec__ to the alias spec; give the module back its own
        spec = self._spec.pop(getattr(module.__spec__, 'name', None), None)
        if spec is not None:
            module.__spec__ = spec


def install_legacy_finder(table=None):
    """Put the LEGACY import hook first on sys.meta_path (idempotent). Returns the finder."""
    for f in sys.meta_path:
        if isinstance(f, _LegacyFinder):
            return f
    f = _LegacyFinder(LEGACY if table is None else table)
    sys.meta_path.insert(0, f)
    return f
