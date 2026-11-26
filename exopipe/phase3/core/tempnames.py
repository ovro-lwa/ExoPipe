"""The temporary-file names of core/placement (atomic, atomic_many, copy): ``<dst>[.<tag>].<pid>.<8 hex digits>.tmp``.

placement makes, globs and sweeps them through this module (placement._temp_name, temp_glob, sweep). It imports the standard
library alone, so a child without the package - the re-judging child (Python 3.6, clv15 first on sys.path, whose exopipe/ shadows
this one; lib/catalog/rejudge.py) - loads it from its file (lib/catalog/run_samples._tempnames) and matches the same names.
"""
import os
import re

HEX8 = '[0-9a-f]' * 8


def name(dst, tag=None):
    """A new temporary name for ``dst`` (not created): ``<dst>[.<tag>].<pid>.<8 random hex digits>.tmp``."""
    if tag is not None and not (tag[:1].isalpha() and tag.replace('_', '').isalnum()):
        raise ValueError('placement.atomic: tag %r is not letters, digits and \'_\' starting with a letter' % (tag,))
    return '%s.%s%d.%s.tmp' % (dst, tag + '.' if tag else '', os.getpid(), os.urandom(4).hex())


def glob(dst, tag=None):
    """The glob of the temporary names of ``dst`` (``dst`` is a glob itself: glob.escape a literal path)."""
    return '%s.%s[0-9]*.%s.tmp' % (dst, tag + '.' if tag else '', HEX8)


def tail(tag=None):
    """The compiled pattern of a temporary name's tail (``[.<tag>].<pid>.<8 hex digits>.tmp`` at the end of a name)."""
    return re.compile((re.escape('.' + tag) if tag else '') + r'\.\d+\.[0-9a-f]{8}\.tmp$')


def product_of(fname, tag=None):
    """The product name a temporary file name belongs to (``<product>[.<tag>].<pid>.<hex>.tmp`` -> ``<product>``), or None when
    ``fname`` is not a temporary name."""
    m = tail(tag).search(fname)
    return fname[:m.start()] if m else None
