"""Numba kernels, split by the domain module that uses them.

Nothing here imports numba at package-import time -- each submodule
(`apa`, `background`, `local_decay`, `stats`) does that itself, and callers
only ever `from touche.numba.<domain> import <kernel>` inside a function
body, never at their own module top level. Importing `touche.numba` itself
must stay free of that cost.
"""

from __future__ import annotations
