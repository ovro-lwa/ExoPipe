"""Moved to exopipe.phase3.core.products (refactor 2026-10-13). This alias keeps pre-refactor importers working
during the transition; it is removed at cutover."""
import sys

from exopipe.phase3.core import products as _products

sys.modules[__name__] = _products
