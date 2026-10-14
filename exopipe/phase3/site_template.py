"""Template of the exopipe3 SITE FILE - where Phase 3 runs (hosts, interpreters, scratch, legacy code roots).

Copy to an ops location (not git), edit, and point EXOPIPE3_SITE at it. exopipe/phase3/config loads every UPPER_CASE name into
CFG.site; a step that needs an unset name refuses with a clear message. Products and science tunables are NOT here:
they live in exopipe/phase3/config/. The site file's sha1 is written into every run record.
"""
NODE_POOLS = {}            # pool name -> [host, ...] or [(host, lanes), ...] ('exopipe3 run submit --nodes <pool>')
GPU_HOSTS = []             # hosts whose GPU runs the flow batch ('run submit --gpu' refuses any other host)
PYTHON = {}                # keys: 'default' (most steps), 'flow' (JAX flows), 'py36' (legacy scripts on Python 3.6), 'casa', 'tess'
SCRATCH_ROOT = {}          # host -> node-local scratch root (never Lustre)
NODE_LOCAL_ROOTS = {}      # host -> [roots of node-local product trees], for data locality
LEGACY_CODE_ROOTS = []     # refactor window only: roots of clv15 / exopipe_ops code the steps call in place
