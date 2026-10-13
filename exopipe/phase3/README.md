# Phase 3 - how to find things and run them

Phase 3 turns the Phase 2 dewarped images (and the Phase 1 10-s snapshots) into co-add masters, residual search frames,
transient detections, catalogues, review pages and archived products, in three tiers: **deep** (1-hour epochs),
**sub-LST** (10-minute windows) and **snapshot** (10-s and 1-min frames). Everything runs through ONE command:

```
exopipe3 --help                     # the steps, in pipeline order
exopipe3 <step> --help              # what it reads, what it writes, how it fails, every flag
exopipe3 config --section <step>    # that step's tunables, with units and the rulings that set them
exopipe3 config --step <step>       # every section the step declares (CONFIG), one place
```

## Where things are

| You want to... | Run | Code | Parameters |
|---|---|---|---|
| build co-add masters (and LOO masters, V clean, flux scale, PB maps, master catalogue) | `exopipe3 masters` | `steps/masters.py` -> `lib/masters/` | `config.py` class `masters` |
| compute alignment flows | `exopipe3 flow` | `steps/flow.py` -> `lib/flow/` | `flow` |
| dewarp masters (per LST) | `exopipe3 dewarp` | `steps/dewarp.py` -> `lib/dewarp/` | `dewarp` |
| install rebuilt products into the working tree | `exopipe3 install` | `steps/install.py` -> `lib/install/` | `install` |
| make residual / search frames | `exopipe3 frames` | `steps/frames.py` -> `lib/frames/` | `frames` |
| search for transients | `exopipe3 search` | `steps/search.py` -> `lib/search/` | `search` |
| build candidate catalogues (merge, associate, select, names; snapshot chain) | `exopipe3 catalog` | `steps/catalog.py` -> `lib/catalog/` | `catalog` |
| forced photometry | `exopipe3 forced` | `steps/forced.py` -> `lib/forced/` | `forced`, `fitting` |
| light curves | `exopipe3 curves` | `steps/curves.py` -> `lib/curves/` | `curves` |
| review pages / servers | `exopipe3 pages`, `exopipe3 serve` | `steps/pages.py`, `steps/serve.py` -> `lib/pages/`, `lib/serve/` | `pages`, `serve` |
| push to / clean the archive | `exopipe3 archive` | `steps/archive.py` -> `lib/archive/` | `archive` |
| mosaics | `exopipe3 mosaic` | `steps/mosaic.py` -> `lib/mosaic/` | `mosaic` |
| photometry of input samples | `exopipe3 photometry` | `steps/photometry.py` -> `lib/photometry/` | `photometry` |
| follow up a candidate | `exopipe3 followup` | `steps/followup.py` -> `followup/` | `followup` |
| Jupiter reflection program | `exopipe3 jupiter` | `steps/jupiter.py` -> `lib/jupiter/` | `jupiter` |
| check any product's content | `exopipe3 verify <class> <files>` | `steps/verify.py`; tests live with their step | - |
| see what is done / stale / missing | `exopipe3 status` | `steps/status.py` | - |
| run a step over many cells on many nodes | `exopipe3 run submit --nodes ... -- <step> ...` | `steps/run.py` -> `core/runner.py` | `runner`, site file |

`core/` holds what every step shares and nothing else: `exclusions` (the only reader of data_decisions.csv),
`products` (names and standard headers), `astrometry` (position errors), `fitting` (Gaussian / template fits, fitted
positions), `fitsio` (atomic writes, node-local scratch), `validate` (content-test registry), `record` (run records and
resume), `cells` (cell / item-list grammar), `runner` (multi-node jobs), `env` (process environment).

## Rules every step follows

1. **Inputs, outputs and failure modes in `--help`.** The module docstring is the help text.
2. **Every tunable in `config.py`**, in the step's section, with its unit and ruling. No constants that change results
   inside `steps/` or `lib/`; no `os.environ` switches; no host names (the site file holds them). Overrides:
   `--set section.KEY=V` or one dict entry `--set section.KEY[entry]=V` (a whole dict that omits entries is refused);
   `config.PROTECTED` keys (exclusions, production-root guard, ledger, pinned inputs) cannot be overridden; a key in a
   section's `RULED` needs `--ruling-for section.KEY=<ledger ID>` (checked in clv15/fix_ledger.csv) or a non-production
   `--out`, and the run record shows the override with its authority.
3. **Gates refuse.** A failed gate raises `record.Refusal` (exit 2) with the reason; it never warns and continues.
4. **Every product class has a content test** registered with `@content_test('<class>')`. A unit is done only when
   it wrote at least one output and every output passed. "Wrote 0" is never done (exit 3, recorded). A unit that
   declares its planned outputs (`u.plan(path, cls)`) is done only when each was written and passed or was refused
   by a named gate (`u.refuse_item(path, reason)`); exit 2 only when a unit itself is refused, exit 3 only when every
   unit had no input.
5. **No marker files.** `--resume` trusts the run record only when inputs, admitted nights, code and config are
   unchanged and the outputs pass their tests now.
6. **Exclusions are absolute.** Ask `core.exclusions`; record the admitted night set on every unit that used one.
7. **Scratch is node-local** (`core.fitsio.scratch_dir`); Lustre holds only bulk products.
8. **`--dry-run`** prints the plan (inputs found, outputs, gates) and writes nothing.
9. **Rulings that waive a test are arguments**: `--waive TEST --ruling "<Gregg's words, UTC, ledger ID>"`.
10. **While `paths.PRODUCTION_WRITES` is False** (the refactor window) no step writes under a production root; parity
    runs pass `--out` to their own root.

## Writing a step (for maintainers)

```python
"""exopipe3 <name> - one line.

Inputs:  ...
Outputs: ... (product classes: ...)
Failure modes: ... (exit 2 = refused, with the gate named)
"""
SUMMARY = '...'
TIERS = ('deep', 'sublst')          # or None
SELECTORS = ('tier', 'cells')       # flags cli.py adds
CONFIG = ('search', 'fitting')      # config sections it reads: its resume hash + `exopipe3 config --step`
WRITES = True

def add_arguments(p): ...           # step-specific flags only
def default_out(args): ...          # production root from config
def run(args, run):                 # one `with run.unit(name) as u:` per cell/LST/batch
    ...
```
