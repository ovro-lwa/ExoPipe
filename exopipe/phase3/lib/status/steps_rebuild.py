"""`status --rebuild-due` - which deep masters are due for a rebuild (NEW; PHASE3_ARCHITECTURE.md section 0, C-10:
"default trigger = +50% accumulated observing time since the last build"; DESIGN 12.2).

The measure (config status_rebuild, whose docstring is the definition of record; repeated in every table row and the
unit record):
  t(night)  = INTTIME (s) of the night's Phase 2 deep image of the cell's scope (I, Taper, Robust -0.75; dewarped, else its
              plain parent: the same integrations) - the quantity the master's own INTTIME card sums over its constituents;
              a build night with no such image in its sci_dir had no data of the scope at the build (t = 0; listed);
  build set = the rows of the installed master's coadd_final/weights.csv (the nights its build admitted, in_master 0/1),
              measured in the Science run the build used (sci_dir);
  new set   = nights admitted NOW (a Phase 2 image of the scope exists; core.exclusions does not exclude the night) that
              are not in the build set, measured in the newest Science run holding the image;
  DUE       when T_new >= status_rebuild.DUE_FRACTION x T_build. No build record + admitted nights = 'never-built' (due).
  A night without an image or without INTTIME is 'unmeasured': the cell is listed 'unmeasured' (due undecided), never
  given a default duration.
"""
import fnmatch
import os

from exopipe.phase3.lib.status import steps_common as K

COLUMNS = ('cell', 'lst', 'scope', 'status', 'due', 'frac_new', 'build_s', 'new_s', 'n_build', 'n_build_in_master',
           'build_in_master_s', 'master_inttime_s', 'master_ncoadd', 'n_new', 'new_nights', 'n_unmeasured', 'unmeasured',
           'build_without_image', 'excluded_since_build', 'due_fraction', 'measure')
STATUSES = ('due', 'not-due', 'never-built', 'no-data', 'unmeasured')
MEASURE = ('t=INTTIME of the Phase 2 deep I Taper Robust-0.75 image of the night (dewarped, else its plain parent; '
           'scope of the cell); build=coadd_final/weights.csv rows in their sci_dir (no image there: t=0); '
           'new=admitted now (image present in the newest Science run, not excluded) and not measured in the build; '
           'due iff new_s >= due_fraction x build_s')


def _patterns(lst, scope):
    G = K.cfg().status_rebuild.IMAGE_GLOB
    pats = G['band'] if scope in ('Red', 'Green', 'Blue', 'Full') else G['subband']
    pats = (pats,) if isinstance(pats, str) else tuple(pats)
    return [p.format(lst=lst, scope=scope) for p in pats]


def image_in(run_dir, lst, scope):
    """The survey-variant Phase 2 deep image of the scope in one Science run (first IMAGE_GLOB pattern present), or None."""
    d = os.path.join(run_dir, scope, 'I', 'deep')
    names = K.listdir(d)
    for pat in _patterns(lst, scope):
        hits = sorted(fnmatch.filter(names, pat))
        if hits:
            return os.path.join(d, hits[-1])
    return None


def night_time(path):
    """(seconds or None, reason)."""
    if not path:
        return None, 'no Phase 2 image of the survey variant'
    try:
        v = K.fits_cards(path).get(K.cfg().status_rebuild.TIME_CARD)
    except OSError as e:
        return None, f'unreadable ({e})'
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None, f'no {K.cfg().status_rebuild.TIME_CARD} card'
    return (v, '') if v > 0 else (None, f'{K.cfg().status_rebuild.TIME_CARD} = {v}')


def _p2_runs(lst, date):
    C = K.cfg()
    dd = os.path.join(C.paths.PHASE2_ROOT, lst, date)
    return [os.path.join(dd, r) for r in K.listdir(dd) if r.startswith(C.status_coverage.PHASE2_RUN_PREFIX)]


def cell_row(cell, ex, p2_dates):
    """The rebuild-due row of one deep cell. p2_dates: {lst: [dates with a Phase 2 night dir]}."""
    C = K.cfg()
    R = C.status_rebuild
    lst, scope = cell.split('_', 1)
    fin = os.path.join(C.paths.COADD_SB_ROOT, cell, 'coadd_final')
    wpath = os.path.join(fin, R.BUILD_RECORD)
    mpath = os.path.join(fin, R.MASTER.format(scope=scope))
    build = K.read_csv(wpath) if os.path.exists(wpath) else None
    mc = K.fits_cards(mpath) if os.path.exists(mpath) else {}
    unmeasured, no_image = [], []
    build_s = in_s = 0.0
    n_in = 0
    bdates = set()
    for r in build or ():
        d = r['date']
        bdates.add(d)
        img = image_in(r['sci_dir'], lst, scope) if r.get('sci_dir') else None
        if img is None:
            no_image.append(d)            # no data of the scope at the build: t = 0 for the build
            continue
        t, why = night_time(img)
        if t is None:
            unmeasured.append(f'{d}(build: {why})')
            continue
        build_s += t
        if str(r.get('in_master')) == '1':
            in_s += t
            n_in += 1
    new, new_s = [], 0.0
    for d in p2_dates.get(lst, ()):
        if (d in bdates and d not in no_image) or ex.is_excluded(lst, scope, d):
            continue
        img = None
        for rd in sorted(_p2_runs(lst, d), reverse=(R.NEW_RUN_RULE == 'newest')):
            img = image_in(rd, lst, scope)
            if img:
                break
        if not img:
            continue                                  # no Phase 2 product of this scope: not admitted
        t, why = night_time(img)
        if t is None:
            unmeasured.append(f'{d}(new: {why})')
            continue
        new.append(d)
        new_s += t
    excl_since = sorted(d for d in bdates if ex.is_excluded(lst, scope, d))
    row = dict(cell=cell, lst=lst, scope=scope, build_s=round(build_s, 3), new_s=round(new_s, 3),
               n_build=len(bdates), n_build_in_master=n_in, build_in_master_s=round(in_s, 3),
               master_inttime_s=mc.get('INTTIME', ''), master_ncoadd=mc.get('NCOADD', ''), n_new=len(new),
               new_nights=';'.join(new), n_unmeasured=len(unmeasured), unmeasured=';'.join(unmeasured),
               build_without_image=';'.join(no_image),
               excluded_since_build=';'.join(excl_since), due_fraction=R.DUE_FRACTION, measure=MEASURE)
    if build is None:
        row.update(status='never-built' if new else 'no-data', due=int(bool(new)), frac_new='')
    elif unmeasured:
        row.update(status='unmeasured', due='', frac_new='')
    elif build_s <= 0:
        row.update(status='never-built' if new else 'no-data', due=int(bool(new)), frac_new='')
    else:
        frac = new_s / build_s
        due = new_s >= R.DUE_FRACTION * build_s
        row.update(status='due' if due else 'not-due', due=int(due), frac_new=round(frac, 6))
    return row


def rebuild_due(cells, ex, workers=None):
    C = K.cfg()
    import re
    lsts = sorted({c.split('_', 1)[0] for c in cells})
    p2_dates = {l: [d for d in K.listdir(os.path.join(C.paths.PHASE2_ROOT, l)) if re.match(r'^\d{4}-\d{2}-\d{2}$', d)]
                for l in lsts}
    return K.thread_map(lambda c: cell_row(c, ex, p2_dates), cells, workers)


def is_due(row):
    return row.get('status') in ('due', 'never-built')
