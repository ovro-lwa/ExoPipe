"""Placement coverage guard (PR 177, placement P1; core/placement.py): in the modules P1 converted to placement.atomic, no
bare os.replace / os.rename / os.renames / shutil.move remains except the sites listed below, each with its reason or the
placement PR planned for it (/fast/gh/claude_tmp/placement-classify_20261125/PLAN.md: P3 copy2 sources, P7 directory
renames). A new bare rename in these modules fails this test; so does a listed site that is gone (the list must shrink with
each later placement PR). Data-free: the source is parsed, nothing is imported or run."""
import ast
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P3 = 'exopipe/phase3/'
MODULES = ['core/manifest.py', 'lib/calibrate/common.py', 'lib/calibrate/fluxscale.py', 'lib/catalog/conform_check.py',
           'lib/catalog/deep_chain.py', 'lib/catalog/deep_omission.py', 'lib/catalog/deep_ported.py', 'lib/catalog/delta_build.py',
           'lib/catalog/grouping_stage.py', 'lib/catalog/incremental.py', 'lib/catalog/restore_wave.py',
           'lib/catalog/tier_sublst.py']
MOVED_ASIDE = 'stays bare: moves an existing product aside under a new name (a rename keeps its inode, so its group, mode and ACL)'
# (module, function, call): (count, reason or planned PR)
EXCEPTIONS = {
    ('lib/calibrate/common.py', 'fresh_stage_dir', 'shutil.move'): (1, MOVED_ASIDE + ' - the earlier stage directory'),
    ('lib/calibrate/common.py', 'verify', 'shutil.move'): (1, MOVED_ASIDE + ' - the earlier verification directory'),
    ('lib/catalog/deep_chain.py', 'DeepRun.move_aside', 'os.rename'): (1, MOVED_ASIDE + ' - an earlier stage output'),
    ('lib/catalog/deep_chain.py', 'DeepRun.copy', 'os.replace'): (1, 'P3: the temporary file is a shutil.copy2 of the source'),
    ('lib/catalog/deep_chain.py', 'DeepRun.ensure_master', 'os.replace'): (1, 'P3: a shutil.copy2 of the stage copy'),
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
}
CALLS = {'os': ('replace', 'rename', 'renames'), 'shutil': ('move',)}


def bare_renames(path):
    """{(function, 'os.replace' | ...): [line, ...]} of one module, through any alias of os / shutil or a from-import."""
    tree = ast.parse(open(path).read(), path)
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


def main():
    fails, n_sites = [], 0
    seen = set()
    for m in MODULES:
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
    for key in sorted(set(EXCEPTIONS) - seen):
        fails.append(f'{P3}{key[0]} {key[1]}: listed bare {key[2]} is gone - remove it from EXCEPTIONS')
    for f in fails:
        print('FAIL', f)
    if fails:
        sys.exit(1)
    print(f'PASS: {len(MODULES)} modules, {n_sites} bare rename/move sites, all listed with a reason or planned PR '
          f'({len(EXCEPTIONS)} entries)')


if __name__ == '__main__':
    main()
