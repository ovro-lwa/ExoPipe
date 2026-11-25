"""Two-week retention of the raw night data (`exopipe3 retention`; plan/ORCHESTRATION_DESIGN_20261117.md 5.4 and section 6 step 5).

table.py   the retention table (production records: which raw time ranges went through the pipeline, and when) and the holds
plan.py    the tree's items and their status, the dry-run manifest, and the delete that re-checks it
protect.py the do-not-delete gate: data_decisions.csv's do-not-delete cells are never planned or deleted (every subband)
tests.py   content tests of the files the step writes
"""
