"""Create an abstraction over the progress bar used."""

from __future__ import annotations

import contextlib
from typing import Protocol, cast

from tqdm.asyncio import tqdm as asyncio_tqdm
from tqdm.auto import tqdm as auto_tqdm
from tqdm.rich import tqdm as rich_tqdm
from tqdm.std import TqdmExperimentalWarning

__all__ = [
    "ProgressBar",
    "ProgressBarProtocol",
    "NullProgressBar",
    "ExperimentalProgressBarWarning",
]


# let tqdm figure out whether we're in a notebook
# but replace the basic tqdm with tqdm.rich.tqdm
if auto_tqdm is asyncio_tqdm:
    tqdm = rich_tqdm
else:
    # we're in a notebook
    # tell typecheckers to pretend like it's tqdm.rich.tqdm
    tqdm = cast(type[rich_tqdm], auto_tqdm)


class ProgressBarProtocol(Protocol):
    def update(self, n: int) -> object: ...


class ProgressBar(tqdm, contextlib.AbstractContextManager, ProgressBarProtocol):
    """A manim progress bar.

    This abstracts away whether a progress bar is used in a notebook, or via the terminal,
    or something else.

    You may need to ignore warnings from ``tqdm``, due to the experimental nature of
    ``tqdm.notebook.tqdm`` and ``tqdm.rich.tqdm``. This can be done with something like:

    .. code-block:: python

        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ExperimentalProgressBarWarning)
            return ProgressBar(...)


    .. note::

        This warning filtering could have been done in the constructor, but would
        have caused the loss of autocomplete with the ``__init__`` of ``tqdm``, as
        well as possibly hide issues with the progressbar. Therefore, the warning
        filtering is left to the user.
    """

    pass


class NullProgressBar(ProgressBarProtocol):
    """Fake progressbar."""

    def update(self, n: int) -> None:
        """Do nothing"""


ExperimentalProgressBarWarning = TqdmExperimentalWarning
