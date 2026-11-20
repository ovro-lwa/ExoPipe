"""The per-person review lists ("Add to Prioritized transient", "Add to Visibility Data Required"; one pair of lists for every tier,
Gregg 2026-09-20; one pair per person, 2026-10-27) as files, written for the two-port server's static sites that opt in (Gregg
2026-11-19, the Jovian event pages: "I want the same \"Add to Visibility Data Required\" Button that we have for the transients
pages").

The review pages (exopipe_ops classify_transients.py, App.list_users / list_add_user / list_members / list_toggle) write the same
files under the same lock, and this module writes them as those do:
- <lists_dir>/users.json: {"users": [{"name", "slug", "added_utc"}]}, indent 1; without it the people are DEFAULT_USERS;
- <lists_dir>/users/<slug>/<LIST_FILES[which]>: CSV with COLUMNS, one row per (run, source id);
- each file replaced whole (<file>.part, then a rename) under an exclusive flock of <lists_dir>/.lock, so a static site's write
  and a review page's write (another process) never interleave.
A row is keyed by (run, source id): a toggle replaces or removes only its own run's row, and keeps every other run's rows.
Two writers until the review pages are ported (they run frozen exopipe_ops code today): a change to the file layout, the lock, the
name rule or the columns is made here and in classify_transients.py together. The rows differ only in their values: a static site
writes labels '' and ra/dec from its ids file (5 decimals), a review page its labels directory and its catalog's ra/dec.
"""
import csv
import datetime
import fcntl
import json
import os
import re

LIST_FILES = {'prioritized': 'prioritized.csv', 'visibility-data-required': 'visibility_required.csv'}
COLUMNS = ['run', 'labels', 'source_id', 'ra', 'dec', 'added_utc', 'title']
USER_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9 .'-]{0,38}[A-Za-z0-9.]$")
DEFAULT_USERS = [{'name': 'Gregg', 'slug': 'gregg'}]
BAD_NAME = 'a name is 2-40 characters: letters first, then letters, digits, spaces, dots, hyphens or apostrophes'


def utcnow():
    return datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')


def _locked(lists_dir):
    os.makedirs(lists_dir, exist_ok=True)
    lk = open(os.path.join(lists_dir, '.lock'), 'a')
    fcntl.flock(lk, fcntl.LOCK_EX)
    return lk


def users(lists_dir):
    """[{'name', 'slug', ...}] of the people with lists (users.json; DEFAULT_USERS without it)."""
    try:
        with open(os.path.join(lists_dir, 'users.json')) as f:
            return [u for u in json.load(f).get('users', []) if u.get('slug')]
    except (OSError, ValueError):
        return [dict(u) for u in DEFAULT_USERS]


def slug_of(name):
    return re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')


def add_user(lists_dir, name):
    """The slug of the person named (added when new), or None for a name that is not a plain name."""
    name = ' '.join((name or '').split())
    if not USER_NAME_RE.match(name):
        return None
    slug = slug_of(name)
    lk = _locked(lists_dir)
    try:
        us = users(lists_dir)
        if not any(u['slug'] == slug for u in us):
            us.append(dict(name=name, slug=slug, added_utc=utcnow()))
            fp = os.path.join(lists_dir, 'users.json')
            with open(fp + '.part', 'w') as f:
                json.dump({'users': us}, f, indent=1)
            os.replace(fp + '.part', fp)
        return slug
    finally:
        lk.close()


def members(lists_dir, which, user):
    """{(run, source id): row} of one person's list."""
    fp = os.path.join(lists_dir, 'users', user, LIST_FILES[which])
    out = {}
    if os.path.exists(fp):
        with open(fp, newline='') as f:
            for r in csv.DictReader(f):
                out[(r.get('run', ''), r.get('source_id', ''))] = r
    return out


def toggle(lists_dir, which, run, source_id, on, user, row):
    """Put (on) or take (not on) the row of (run, source_id) on one person's list; row gives labels, ra, dec and title."""
    lk = _locked(lists_dir)
    try:
        d = os.path.join(lists_dir, 'users', user)
        os.makedirs(d, exist_ok=True)
        fp = os.path.join(d, LIST_FILES[which])
        rows = [r for r in members(lists_dir, which, user).values() if not (r.get('run') == run and r.get('source_id') == source_id)]
        if on:
            rows.append(dict(row, run=run, source_id=source_id, added_utc=utcnow()))
        with open(fp + '.part', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, '') for k in COLUMNS})
        os.replace(fp + '.part', fp)
    finally:
        lk.close()
