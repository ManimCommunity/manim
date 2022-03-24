from __future__ import annotations

import pluggy

hookspec = pluggy.HookspecMarker("manim")


@hookspec
def default_font(requestor: str) -> str:
    """Look who's requesting and return the default font to be used by
    them.

    Parameters
    ----------
    requestor:
        The ``__name__`` of the requesting class. This can be one of
        ['Text', 'MarkupText', 'Paragraph', 'Code']

    Returns
    -------
    str:
        Can be empty string to use the default font as defined by the system.
    """
