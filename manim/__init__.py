#!/usr/bin/env python


from __future__ import annotations

import pkg_resources

__version__: str = pkg_resources.get_distribution(__name__).version

# isort: off

# Importing the config module should be the first thing we do, since other
# modules depend on the global config dict for initialization.

from . import _config

# many scripts depend on this -> has to be loaded first

# isort: on


try:
    from IPython import get_ipython

    from .utils.ipython_magic import ManimMagic
except ImportError:
    pass
else:
    ipy = get_ipython()
    if ipy is not None:
        ipy.register_magics(ManimMagic)

from .plugins import *
