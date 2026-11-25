"""Placement coverage guard (PR 177, placement P1; placement P2; placement P3; core/placement.py): in the modules P1, P2 and P3
converted to placement.atomic, no bare os.replace / os.rename / os.renames / shutil.move remains except the sites listed
below, each with its reason or the placement PR planned for it (/fast/gh/claude_tmp/placement-classify_20261125/PLAN.md: P3
copy2 sources, P4 mosaic and hybrid sites and P5 others behind a handler on the path, P6 archive writers and per-file volume,
P7 directory renames). A new bare rename in these modules fails this test; so does a listed site that is gone (the list must
shrink with each later placement PR). In the same modules every record.atomic_write_json call passes ``place=`` explicitly
(True for a product, False for a unit or run record), so a product never takes the unplaced default by omission. Every
listed entry carries its reason ('stays bare: ...') or planned PR ('P<n>: ...'); an entry without one fails. Data-free: the
source is parsed, nothing is imported or run (self_test checks that the guards fail on injected input)."""
import ast
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P3 = 'exopipe/phase3/'
MODULES = ['core/manifest.py', 'lib/calibrate/common.py', 'lib/calibrate/fluxscale.py', 'lib/catalog/conform_check.py',
           'lib/catalog/deep_chain.py', 'lib/catalog/deep_omission.py', 'lib/catalog/deep_ported.py', 'lib/catalog/delta_build.py',
           'lib/catalog/grouping_stage.py', 'lib/catalog/incremental.py', 'lib/catalog/restore_wave.py',
           'lib/catalog/tier_sublst.py',
           # P2
           'lib/forced/common.py', 'lib/forced/night_activity.py', 'lib/forced/wide_jobs.py', 'lib/forced/wide_stages.py',
           'lib/forced/wide_v3.py', 'lib/frames/tier_snapshot.py', 'lib/hybrid/stack.py', 'lib/hybrid/warp.py',
           'lib/jupiter/serve.py', 'lib/masters/admit.py', 'lib/masters/admit_stages.py', 'lib/masters/build_stats.py',
           'lib/masters/deep_stages.py', 'lib/masters/membership.py', 'lib/masters/night_image_gate.py',
           'lib/masters/sublst_nonmember.py', 'lib/masters/tier_snapshot.py', 'lib/mosaic/conform_v5.py',
           'lib/mosaic/steps_worker.py', 'lib/notify/qa.py', 'lib/photometry/steps_common.py', 'lib/photometry/steps_deep.py',
           'lib/photometry/tests_cutouts.py', 'lib/retention/plan.py', 'lib/search/c50.py', 'lib/search/deep_step.py',
           'phase3_i_deep_final.py', 'phase3_i_deep_package_deep.py', 'steps/followup.py', 'steps/hybrid.py',
           # P3
           'lib/jupiter/v2.py', 'lib/mosaic/gen_common.py', 'lib/pages/night.py', 'lib/pages/pages_input.py',
           'lib/serve/fp_filters.py', 'lib/serve/hub.py', 'lib/serve/register.py', 'lib/serve/snapshot_view.py',
           'lib/serve/versions.py', 'lib/serve/views.py', 'lib/status/steps_common.py', 'lib/status/steps_hub.py']
# every entry's reason: 'stays bare[ until ...]: <why>' or the planned PR 'P<n>: <what>' - nothing else (an entry without
# a reason or a planned PR fails the guard)
REASON = re.compile(r'^(stays bare( until [^:]+)?|P[1-9][0-9]*): \S.{9,}$', re.S)
MOVED_ASIDE = 'stays bare: moves an existing product aside under a new name (a rename keeps its inode, so its group, mode and ACL)'
MOVED_DIR = 'stays bare: moves an existing version or run directory (a rename keeps its inode, so its group, mode and ACL)'
# (module, function, call): (count, reason or planned PR)
EXCEPTIONS = {
    ('lib/calibrate/common.py', 'fresh_stage_dir', 'shutil.move'): (1, MOVED_ASIDE + ' - the earlier stage directory'),
    ('lib/calibrate/common.py', 'verify', 'shutil.move'): (1, MOVED_ASIDE + ' - the earlier verification directory'),
    ('lib/catalog/deep_chain.py', 'DeepRun.move_aside', 'os.rename'): (1, MOVED_ASIDE + ' - an earlier stage output'),
    ('lib/catalog/deep_chain.py', 'DeepRun.st_list', 'os.replace'): (1, 'stays bare: a symlink (placement follows symlinks)'),
    ('lib/catalog/deep_omission.py', 'TenminRecords.write_cache', 'os.replace'): (
        1, 'stays bare: the window-rms cache, re-derivable, not a product'),
    ('lib/catalog/deep_omission.py', 'run_snapshot', 'os.rename'): (
        1, 'stays bare: the re-judging work directory renamed to <out>/_rejudge (the children\'s scratch)'),
    ('lib/catalog/grouping_stage.py', 'run_group', 'os.rename'): (
        3, 'P7: grouping_<S>.tmp renamed to the kept and to the final directory (directory renames); ' + MOVED_ASIDE),
    ('lib/catalog/incremental.py', 'union_tiers', 'os.replace'): (1, MOVED_ASIDE + ' - earlier union files'),
    ('lib/catalog/incremental.py', '_epoch_unit', 'os.rename'): (2, 'P7: the epoch block directory; ' + MOVED_ASIDE),
    ('lib/catalog/incremental.py', 'prune_frame_caches', 'os.replace'): (
        1, 'stays bare: an unreadable cache set aside as .unreadable (a rename of the file itself)'),
    ('lib/catalog/incremental.py', '_build_stage', 'os.rename'): (2, 'P7: the catalog directory; ' + MOVED_ASIDE),
    ('lib/catalog/tier_sublst.py', 'night_view', 'shutil.move'): (1, MOVED_ASIDE + ' - the earlier night view'),
    ('lib/catalog/tier_sublst.py', 'st_merge', 'shutil.move'): (1, MOVED_ASIDE + ' - earlier collect and state directories'),
    ('lib/catalog/tier_sublst.py', 'st_reftie', 'os.replace'): (1, 'stays bare: a failed tie log kept under a new name'),
    # P2
    ('lib/forced/wide_jobs.py', 'write_jobs', 'os.replace'): (1, 'stays bare: the run\'s job state document, not a product'),
    ('lib/forced/wide_stages.py', 'search_job', 'os.replace'): (
        1, 'P5: the search child\'s result, behind a handler on the path (the caller catches Exception)'),
    ('lib/frames/tier_snapshot.py', 'stage_stage.unpack', 'os.replace'): (
        1, 'stays bare: unpacks the staged inputs into the stage\'s scratch directory'),
    ('lib/frames/tier_snapshot.py', 'stage_cube', 'os.replace'): (1, 'P6: an archive writer (the snapshot cube, a copyfile)'),
    ('lib/frames/tier_snapshot.py', 'strip_legacy_header', 'os.replace'): (
        1, 'stays bare: log - rewrites an engine log without its legacy header line, not a product (a cosmetic rewrite never '
           'fails a movie unit)'),
    ('lib/frames/tier_snapshot.py', 'flow_movies', 'os.replace'): (1, 'P6: per-file volume (the flow movies)'),
    ('lib/hybrid/warp.py', 'derive_warp_screen', 'os.replace'): (
        2, 'stays bare: the warp-screen cache and its record, node-local (fitsio.scratch_root) and re-derivable, not a product'),
    ('lib/masters/admit.py', 'stale_admissions', 'os.replace'): (1, MOVED_ASIDE + ' - a stale admission'),
    ('lib/masters/admit_stages.py', 'move_aside', 'os.replace'): (1, MOVED_ASIDE + ' - earlier admit outputs'),
    ('lib/masters/admit_stages.py', 'move_aside_files', 'os.replace'): (1, MOVED_ASIDE + ' - earlier admit outputs'),
    ('lib/masters/admit_stages.py', 'admit_stage', 'os.replace'): (1, 'P6: per-file volume (the admitted night images)'),
    ('lib/masters/deep_stages.py', '_place_build_stats', 'os.replace'): (
        2, MOVED_ASIDE + ' - the earlier build_stats.json, and that same file moved back under its name when nothing is placed'),
    ('lib/masters/deep_stages.py', 'own_note', 'os.replace'): (
        1, 'P5: behind a handler on the path (the caller catches Exception)'),
    ('lib/masters/deep_stages.py', 'striation', 'os.replace'): (
        1, 'stays bare: renames existing products to *.FAILED (a rename keeps its inode, so its group, mode and ACL)'),
    ('lib/masters/membership.py', 'write_verdict', 'os.replace'): (
        1, 'stays bare until production changes with it: the status S-G6 gate requires this module to equal production\'s '
           'exopipe/phase3/membership.py except imports'),
    ('lib/mosaic/steps_worker.py', 'main', 'os.replace'): (1, 'stays bare: the worker\'s result record in the stage\'s scratch'),
    ('lib/notify/qa.py', 'ionosphere_rank', 'os.replace'): (
        1, 'stays bare: cache - the ionosphere-rank population cache, derived and re-derivable, not a product'),
    ('lib/mosaic/steps_worker.py', 'task_qa', 'os.replace'): (
        1, 'stays bare: writes into the stage\'s temporary directory; P4 places the table (steps_qa.run_qa)'),
    ('lib/photometry/steps_common.py', 'atomic_copy', 'os.replace'): (
        1, 'P6: per-file volume; the temporary file is a shutil.copy2 of the work file'),
    ('lib/search/deep_step.py', 'stage_quarantine', 'os.replace'): (1, 'stays bare: a run record (quarantine plan)'),
    ('lib/search/deep_step.py', 'run_unit', 'os.replace'): (1, 'stays bare: a run record (the unit\'s test report)'),
    ('lib/search/deep_step.py', '_copy_into', 'os.replace'): (
        1, 'P6: per-file volume; the temporary file is a shutil.copy2 of the attempt\'s file'),
    ('lib/search/deep_step.py', 'stage_publish', 'os.replace'): (
        1, 'P6: the published attempt (a directory of copies renamed into place)'),
    ('lib/search/deep_step.py', '_child', 'os.replace'): (1, 'stays bare: the child\'s result record in the run\'s scratch'),
    ('phase3_i_deep_package_deep.py', '_sweep_stale', 'shutil.move'): (1, MOVED_ASIDE + ' - stale review directories'),
    ('steps/followup.py', '_atomic_text', 'os.replace'): (
        1, 'P5: behind a handler on the path (the caller catches Exception)'),
    ('steps/hybrid.py', 'tilefill_field', 'os.replace'): (
        1, 'P4: the field FITS, behind except (TilefillRefusal, OSError, KeyError, ValueError), which a refusal escapes'),
    ('steps/hybrid.py', 'tilefill_unit', 'os.replace'): (1, 'P4: the unit\'s record, behind a handler on the path'),
    # P3
    ('lib/jupiter/v2.py', 'st_prepare', 'os.replace'): (
        1, MOVED_ASIDE + ' - earlier derived roots (directories) moved to _superseded/<tag>'),
    ('lib/mosaic/gen_common.py', 'pkcorr', 'os.replace'): (
        1, 'P4: the pkcorr products, moved from the unit\'s temporary subdirectory (not a sibling temporary file)'),
    ('lib/mosaic/gen_common.py', 'test2_write', 'os.replace'): (1, 'P4: the test-2 record of a mosaic unit'),
    ('lib/pages/night.py', 'rename', 'os.rename'): (
        1, 'stays bare: renames the images this run rendered to their full-selection names in the same directory (a rename '
           'keeps the group, mode and ACL they took when rendered there)'),
    ('lib/serve/fp_filters.py', 'write_files', 'os.replace'): (
        2, 'P5: the filter CSV and JSON installed as a pair behind an except BaseException that removes both and re-raises'),
    ('lib/serve/register.py', 'set_hold', 'os.replace'): (
        1, 'stays bare: the publish hold, a state marker (not a product, mode 0664 as production); it replaces a dangling link '
           'at its name, which placement cannot (it stats through links)'),
    ('lib/serve/register.py', 'lift_hold', 'os.rename'): (1, MOVED_ASIDE + ' - the lifted publish hold'),
    ('lib/serve/snapshot_view.py', 'build', 'os.rename'): (
        1, 'P7: the combined view directory renamed from its .part name (a directory rename)'),
    ('lib/serve/snapshot_view.py', 'child_main', 'os.replace'): (
        1, 'stays bare: the build child\'s result record (scratch), written by the Python 3.6 child without exopipe'),
    ('lib/serve/snapshot_view.py', 'move_aside', 'os.rename'): (1, MOVED_ASIDE + ' - an earlier view'),
    ('lib/serve/versions.py', '_renew_one', 'os.replace'): (
        1, 'stays bare: the lease renewal of a state record (BUILDING.json, a hold), behind an (OSError, ValueError) '
           'handler'),
    ('lib/serve/versions.py', 'place', 'os.rename'): (1, MOVED_DIR + ' (the built run into its version slot)'),
    ('lib/serve/versions.py', 'unplace', 'os.rename'): (1, MOVED_DIR + ' (a version back to its build directory)'),
    ('lib/serve/versions.py', '_write_manifest', 'os.replace'): (
        1, 'P5: the version manifest, behind a handler on the path (a caller catches Exception)'),
    ('lib/serve/versions.py', '_unadopt', 'os.rename'): (1, MOVED_DIR + ' (an adopted run moved back)'),
    ('lib/serve/versions.py', '_adopt', 'os.rename'): (1, MOVED_DIR + ' (a served run adopted as a version)'),
    ('lib/serve/versions.py', 'switch', 'os.replace'): (
        1, 'stays bare: the served pointer, a symlink (placement follows symlinks)'),
    ('lib/serve/versions.py', 'discard', 'os.rename'): (1, MOVED_DIR + ' (a version renamed out before its deletion)'),
    ('lib/serve/versions.py', '_to_work', 'os.rename'): (1, MOVED_DIR + ' (a legacy run moved to its work directory)'),
    ('lib/serve/versions.py', '_unwork', 'os.rename'): (1, MOVED_DIR + ' (the work directory moved back)'),
    ('lib/serve/versions.py', 'retire', 'os.rename'): (1, MOVED_DIR + ' (a retired version renamed out)'),
}
CALLS = {'os': ('replace', 'rename', 'renames'), 'shutil': ('move',)}


def _functions(tree):
    """[(qualified function name or '<module>', node)] of every node, by a walk that tracks the enclosing def / class."""
    out = []

    def visit(node, fn):
        for ch in ast.iter_child_nodes(node):
            f = fn
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                f = f'{fn}.{ch.name}' if fn else ch.name
            out.append((f or '<module>', ch))
            visit(ch, f)
    visit(tree, '')
    return out


def json_calls(path, text=None):
    """[(function, line, place= passed by name)] of the record.atomic_write_json calls of one module (a ``**kwargs`` call
    counts as not explicit): through any module alias (record.atomic_write_json, _R.atomic_write_json) or a from-import
    under any name."""
    tree = ast.parse(text if text is not None else open(path).read(), path)
    names = {a.asname or a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names
             if a.name == 'atomic_write_json'}
    out = []
    for fn, n in _functions(tree):
        if not isinstance(n, ast.Call):
            continue
        c = n.func
        if (isinstance(c, ast.Attribute) and c.attr == 'atomic_write_json') or (isinstance(c, ast.Name) and c.id in names):
            out.append((fn, n.lineno, 'place' in [k.arg for k in n.keywords]))
    return out


def bare_renames(path, text=None):
    """{(function, 'os.replace' | ...): [line, ...]} of one module, through any alias of os / shutil or a from-import."""
    tree = ast.parse(text if text is not None else open(path).read(), path)
    mods, names = {}, {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name in CALLS:
                    mods[a.asname or a.name] = a.name
        elif isinstance(n, ast.ImportFrom) and n.module in CALLS:
            for a in n.names:
                if a.name in CALLS[n.module]:
                    names[a.asname or a.name] = f'{n.module}.{a.name}'
    out = {}

    def visit(node, fn):
        for ch in ast.iter_child_nodes(node):
            f = fn
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                f = f'{fn}.{ch.name}' if fn else ch.name
            if isinstance(ch, ast.Call):
                c = ch.func
                call = None
                if isinstance(c, ast.Attribute) and isinstance(c.value, ast.Name) and c.value.id in mods \
                        and c.attr in CALLS[mods[c.value.id]]:
                    call = f'{mods[c.value.id]}.{c.attr}'
                elif isinstance(c, ast.Name) and c.id in names:
                    call = names[c.id]
                if call:
                    out.setdefault((f or '<module>', call), []).append(ch.lineno)
            visit(ch, f)
    visit(tree, '')
    return out


INJECTED = """import os as o
from exopipe.phase3.core import record as R
from exopipe.phase3.core.record import atomic_write_json as w


def f(p):
    R.atomic_write_json(p, {})
    w(p, {}, durable=True)
    R.atomic_write_json(p, {}, place=False)
    w(p, {}, place=True)
    o.replace(p + '.tmp', p)
"""


def bad_entries(exceptions):
    """The EXCEPTIONS entries without a count of at least 1 or without a reason or planned PR (REASON)."""
    return sorted(k for k, v in exceptions.items()
                  if not (isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], int) and v[0] >= 1
                          and isinstance(v[1], str) and REASON.match(v[1])))


def self_test():
    """The guards must fail on injected input: two atomic_write_json calls without place= (a module alias and a renamed
    from-import), two with it, and one bare os.replace through an alias; list entries without a reason or planned PR.
    Returns the problems (empty: the guards work)."""
    probs = []
    injected = {('m', 'a', 'os.replace'): (1, 'unclassified: a copy renamed over itself'), ('m', 'b', 'os.replace'): (1, ''),
                ('m', 'c', 'os.replace'): (0, 'P4: the products of c'), ('m', 'd', 'os.replace'): (1, 'P4: the products of d'),
                ('m', 'e', 'os.rename'): (1, MOVED_DIR), ('m', 'f', 'os.replace'): (1, 'stays bare until production changes: '
                                                                                     'the gate compares the module')}
    got = bad_entries(injected)
    if got != [('m', 'a', 'os.replace'), ('m', 'b', 'os.replace'), ('m', 'c', 'os.replace')]:
        probs.append(f'the reason guard found {got} in the injected entries, not a, b and c')
    got = [(fn, line) for fn, line, explicit in json_calls('<injected>', INJECTED) if not explicit]
    if got != [('f', 7), ('f', 8)]:
        probs.append(f'the place= guard found {got} in the injected source, not [(\'f\', 7), (\'f\', 8)]')
    got = bare_renames('<injected>', INJECTED)
    if got != {('f', 'os.replace'): [11]}:
        probs.append(f'the rename guard found {got} in the injected source, not one os.replace at line 11')
    return probs


def main():
    fails, n_sites = [], 0
    seen = set()
    fails += [f'self-test: {p}' for p in self_test()]
    n_json = 0
    for m in MODULES:
        for fn, line, explicit in json_calls(os.path.join(REPO, P3 + m)):
            n_json += 1
            if not explicit:
                fails.append(f'{P3}{m}:{line} {fn}: record.atomic_write_json without place= - pass place=True for a '
                             f'product, place=False for a unit or run record')
        for (fn, call), lines in sorted(bare_renames(os.path.join(REPO, P3 + m)).items()):
            n_sites += len(lines)
            key = (m, fn, call)
            seen.add(key)
            if key not in EXCEPTIONS:
                fails.append(f'{P3}{m}:{lines[0]} {fn}: bare {call} ({len(lines)}) - write through placement.atomic, or list '
                             f'it here with its reason or planned PR')
            elif EXCEPTIONS[key][0] != len(lines):
                fails.append(f'{P3}{m} {fn}: {len(lines)} bare {call} (lines {lines}), the list says {EXCEPTIONS[key][0]} - '
                             f'update the list')
    for key in bad_entries(EXCEPTIONS):
        fails.append(f'{P3}{key[0]} {key[1]}: listed bare {key[2]} without a reason or planned PR ({EXCEPTIONS[key]!r}) - '
                     f'give "stays bare: <why>" or "P<n>: <what>"')
    for key in sorted(set(EXCEPTIONS) - seen):
        fails.append(f'{P3}{key[0]} {key[1]}: listed bare {key[2]} is gone - remove it from EXCEPTIONS')
    for f in fails:
        print('FAIL', f)
    if fails:
        sys.exit(1)
    print(f'PASS: {len(MODULES)} modules, {n_sites} bare rename/move sites, all listed with a reason or planned PR '
          f'({len(EXCEPTIONS)} entries); {n_json} record.atomic_write_json calls, each with place= explicit; every entry with a reason '
          f'or planned PR; the three guards fail on injected input')


if __name__ == '__main__':
    main()
