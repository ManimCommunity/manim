"""Utilities for processing LaTeX templates."""

from __future__ import annotations

__all__ = [
    "TexTemplate",
]

import copy
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing_extensions import Self

    from manim.typing import StrPath

_DEFAULT_PREAMBLE = r"""\usepackage[english]{babel}
\usepackage{amsmath}
\usepackage{amssymb}"""

_BEGIN_DOCUMENT = r"\begin{document}"
_END_DOCUMENT = r"\end{document}"


@dataclass(eq=True)
class TexTemplate:
    """TeX templates are used to create ``Tex`` and ``MathTex`` objects."""

    _body: str = field(default="", init=False)
    """A custom body, can be set from a file."""

    tex_compiler: str = "latex"
    """The TeX compiler to be used, e.g. ``latex``, ``pdflatex`` or ``lualatex``."""

    output_format: str = ".dvi"
    """The output format resulting from compilation, e.g. ``.dvi`` or ``.pdf``."""

    documentclass: str = r"\documentclass[preview]{standalone}"
    r"""The command defining the documentclass, e.g. ``\documentclass[preview]{standalone}``."""

    preamble: str = _DEFAULT_PREAMBLE
    r"""The document's preamble, i.e. the part between ``\documentclass`` and ``\begin{document}``."""

    placeholder_text: str = "YourTextHere"
    """Text in the document that will be replaced by the expression to be rendered."""

    post_doc_commands: str = ""
    r"""Text (definitions, commands) to be inserted at right after ``\begin{document}``, e.g. ``\boldmath``."""

    @property
    def body(self) -> str:
        """The entire TeX template."""
        return self._body or "\n".join(
            filter(
                None,
                [
                    self.documentclass,
                    self.preamble,
                    _BEGIN_DOCUMENT,
                    self.post_doc_commands,
                    self.placeholder_text,
                    _END_DOCUMENT,
                ],
            )
        )

    @body.setter
    def body(self, value: str) -> None:
        self._body = value

    @classmethod
    def from_file(cls, file: StrPath = "tex_template.tex", **kwargs: Any) -> Self:
        """Create an instance by reading the content of a file.

        Using the ``add_to_preamble`` and ``add_to_document`` methods on this instance
        will have no effect, as the body is read from the file.
        """
        instance = cls(**kwargs)
        instance.body = Path(file).read_text(encoding="utf-8")
        return instance

    def add_to_preamble(self, txt: str, prepend: bool = False) -> Self:
        r"""Adds text to the TeX template's preamble (e.g. definitions, packages). Text can be inserted at the beginning or at the end of the preamble.

        Parameters
        ----------
        txt
            String containing the text to be added, e.g. ``\usepackage{hyperref}``.
        prepend
            Whether the text should be added at the beginning of the preamble, i.e. right after ``\documentclass``.
            Default is to add it at the end of the preamble, i.e. right before ``\begin{document}``.
        """
        if self._body:
            warnings.warn(
                "This TeX template was created with a fixed body, trying to add text the preamble will have no effect.",
                UserWarning,
                stacklevel=2,
            )
        if prepend:
            self.preamble = txt + "\n" + self.preamble
        else:
            self.preamble += "\n" + txt
        return self

    def add_to_document(self, txt: str) -> Self:
        r"""Adds text to the TeX template just after \begin{document}, e.g. ``\boldmath``.

        Parameters
        ----------
        txt
            String containing the text to be added.
        """
        if self._body:
            warnings.warn(
                "This TeX template was created with a fixed body, trying to add text the document will have no effect.",
                UserWarning,
                stacklevel=2,
            )
        self.post_doc_commands += txt
        return self

    def get_texcode_for_expression(self, expression: str) -> str:
        r"""Inserts expression verbatim into TeX template.

        Parameters
        ----------
        expression
            The string containing the expression to be typeset, e.g. ``$\sqrt{2}$``

        Returns
        -------
        :class:`str`
            LaTeX code based on current template, containing the given ``expression`` and ready for typesetting
        """
        return self.body.replace(self.placeholder_text, expression)

    def get_texcode_for_expression_in_env(
        self, expression: str, environment: str
    ) -> str:
        r"""Inserts expression into TeX template wrapped in ``\begin{environment}`` and ``\end{environment}``.

        Parameters
        ----------
        expression
            The string containing the expression to be typeset, e.g. ``$\sqrt{2}$``.
        environment
            The string containing the environment in which the expression should be typeset, e.g. ``align*``.

        Returns
        -------
        :class:`str`
            LaTeX code based on template, containing the given expression inside its environment, ready for typesetting
        """
        begin, end = _texcode_for_environment(environment)
        return self.body.replace(
            self.placeholder_text, "\n".join([begin, expression, end])
        )

    def copy(self) -> Self:
        """Create a deep copy of the TeX template instance."""
        return copy.deepcopy(self)


def _texcode_for_environment(environment: str) -> tuple[str, str]:
    r"""Processes the tex_environment string to return the correct ``\begin{environment}[extra]{extra}`` and
    ``\end{environment}`` strings.

    Parameters
    ----------
    environment
        The tex_environment as a string. Acceptable formats include:
        ``{align*}``, ``align*``, ``{tabular}[t]{cccl}``, ``tabular}{cccl``, ``\begin{tabular}[t]{cccl}``.

    Returns
    -------
    Tuple[:class:`str`, :class:`str`]
        A pair of strings representing the opening and closing of the tex environment, e.g.
        ``\begin{tabular}{cccl}`` and ``\end{tabular}``
    """

    environment.removeprefix(r"\begin").removeprefix("{")

    # The \begin command takes everything and closes with a brace
    begin = r"\begin{" + environment
    # If it doesn't end on } or ], assume missing }
    if not begin.endswith(("}", "]")):
        begin += "}"

    # While the \end command terminates at the first closing brace
    split_at_brace = re.split("}", environment, 1)
    end = r"\end{" + split_at_brace[0] + "}"

    return begin, end
