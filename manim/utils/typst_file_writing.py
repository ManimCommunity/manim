"""Interface for writing, compiling, and converting ``.typ`` files via the ``typst`` Python package.

.. SEEALSO::

    :mod:`.mobject.text.typst_mobject`

"""

from __future__ import annotations

import hashlib
from pathlib import Path

from manim import config, logger

__all__ = ["typst_to_svg_file"]

TYPST_COMPILATION_FONT_SIZE = 11  # pt — Typst's default

TYPST_TEMPLATE = """\
#set page(width: auto, height: auto, margin: 0pt, fill: none)
#set text(size: {text_size}pt)
{preamble}
{body}
"""


def _typst_hash(content: str) -> str:
    """Return a truncated SHA-256 hex digest of *content*."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def typst_to_svg_file(
    typst_code: str,
    preamble: str = "",
    text_size: float = TYPST_COMPILATION_FONT_SIZE,
    font_paths: list[str | Path] | None = None,
) -> Path:
    """Compile a Typst string to SVG via the ``typst`` Python package.

    The compiled SVG and the intermediate ``.typ`` source are cached
    under :func:`config.get_dir("tex_dir") <manim.ManimConfig.get_dir>`
    using a content-hash filename scheme (identical to the LaTeX pipeline).

    Parameters
    ----------
    typst_code
        The body of the Typst document (user-supplied markup).
    preamble
        Extra Typst code inserted between the ``#set`` rules and the body.
        Useful for ``#import``, ``#set``, or ``#show`` rules.
    text_size
        Font size in Typst points used during compilation.
    font_paths
        Optional list of additional font directories passed to the Typst
        compiler.

    Returns
    -------
    :class:`Path`
        Path to the generated SVG file.

    Raises
    ------
    ImportError
        If the ``typst`` Python package is not installed.
    """
    try:
        import typst as typst_compiler
    except ImportError:
        raise ImportError(
            "TypstMobject requires the 'typst' Python package. "
            "Install it with:  pip install typst>=0.14"
        )

    full_source = TYPST_TEMPLATE.format(
        text_size=text_size,
        preamble=preamble,
        body=typst_code,
    )
    content_hash = _typst_hash(full_source)
    typst_dir = config.get_dir("tex_dir")
    typst_dir.mkdir(parents=True, exist_ok=True)

    typ_file = typst_dir / f"{content_hash}.typ"
    svg_file = typst_dir / f"{content_hash}.svg"

    if svg_file.exists():
        return svg_file

    typ_file.write_text(full_source, encoding="utf-8")

    logger.info(
        "Compiling Typst source %(path)s ...",
        {"path": f"{typ_file}"},
    )

    svg_bytes = typst_compiler.compile(
        str(typ_file),
        format="svg",
        font_paths=font_paths or [],
    )
    svg_file.write_bytes(svg_bytes)
    return svg_file
