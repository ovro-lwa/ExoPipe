"""Review pages of every tier (`exopipe3 pages`): the step's orchestration around the production page modules, which run
IN PLACE during the refactor window (DESIGN 11.1).

Modules (nothing here has a command line of its own):
  common.py        legacy code roots, the scrubbed step environment and lrun (a thin wrapper of core.legacy.run, the one
                   child runner), the server's panel-name pattern, CSV helpers, the run-directory lease (gate P9),
                   exclusion bookkeeping, record chain and resume
  mirror.py        gate P3: the config values of constants that live inside the in-place modules, read from their source
  pages_input.py   the deep singles pages input (port of the p3s3_pass.sh:137-169 heredoc) and the expected fork rows
  panels.py        the deep page plan: page ids, ranks and every panel file a set owes (rows + member detections)
  m14.py           deep item M14: per-set panel directories, the (Stokes, source_id)-keyed review_dir merge, fork-index names
  merge.py         THE review_dir merge (pages owns it and pages.REVIEW_DIR_MERGE; serve calls merge.run at serve time)
  tier_deep.py, tier_sublst.py, tier_snapshot.py   the stages of each tier
  tests.py         content tests of every product class the step writes (registered with @content_test)

The production modules keep their homes until absorption (MOVE_PLAN 2.10 names: deep_pages.py, merge_review_dir.py,
pages_index.py, sublst_pages*.py, snapshot_pages.py, std7_panel.py, tess_coverage.py, tier_pages_input.py, ...); no file
here uses those names.
"""
