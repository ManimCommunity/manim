"""Utilities for processing LaTeX templates."""

from __future__ import annotations

__all__ = [
    "TexTemplate",
    "TexTemplateFromFile",
]

import copy
import os
import re
from pathlib import Path


class TexTemplate:
    """TeX templates are used for creating Tex() and MathTex() objects.

    Parameters
    ----------
    tex_compiler
        The TeX compiler to be used, e.g. ``latex``, ``pdflatex`` or ``lualatex``
    output_format
        The output format resulting from compilation, e.g. ``.dvi`` or ``.pdf``
    documentclass
        The command defining the documentclass, e.g. ``\\documentclass[preview]{standalone}``
    preamble
        The document's preamble, i.e. the part between ``\\documentclass`` and ``\\begin{document}``
    placeholder_text
        Text in the document that will be replaced by the expression to be rendered
    post_doc_commands
        Text (definitions, commands) to be inserted at right after ``\\begin{document}``, e.g. ``\\boldmath``

    Attributes
    ----------
    tex_compiler : :class:`str`
        The TeX compiler to be used, e.g. ``latex``, ``pdflatex`` or ``lualatex``
    output_format : :class:`str`
        The output format resulting from compilation, e.g. ``.dvi`` or ``.pdf``
    documentclass : :class:`str`
        The command defining the documentclass, e.g. ``\\documentclass[preview]{standalone}``
    preamble : :class:`str`
        The document's preamble, i.e. the part between ``\\documentclass`` and ``\\begin{document}``
    placeholder_text : :class:`str`
        Text in the document that will be replaced by the expression to be rendered
    post_doc_commands : :class:`str`
        Text (definitions, commands) to be inserted at right after ``\\begin{document}``, e.g. ``\\boldmath``
    """

    default_documentclass = r"\documentclass[preview]{standalone}"
    default_preamble = r"""
\usepackage[english]{babel}
\usepackage{amsmath}
\usepackage{amssymb}
"""
    default_placeholder_text = "YourTextHere"
    default_tex_compiler = "latex"
    default_output_format = ".dvi"
    default_post_doc_commands = ""

    def __init__(
        self,
        tex_compiler: str | None = None,
        output_format: str | None = None,
        documentclass: str | None = None,
        preamble: str | None = None,
        placeholder_text: str | None = None,
        post_doc_commands: str | None = None,
        **kwargs,
    ):
        self.tex_compiler = (
            tex_compiler
            if tex_compiler is not None
            else TexTemplate.default_tex_compiler
        )
        self.output_format = (
            output_format
            if output_format is not None
            else TexTemplate.default_output_format
        )
        self.documentclass = (
            documentclass
            if documentclass is not None
            else TexTemplate.default_documentclass
        )
        self.preamble = (
            preamble if preamble is not None else TexTemplate.default_preamble
        )
        self.placeholder_text = (
            placeholder_text
            if placeholder_text is not None
            else TexTemplate.default_placeholder_text
        )
        self.post_doc_commands = (
            post_doc_commands
            if post_doc_commands is not None
            else TexTemplate.default_post_doc_commands
        )
        self._rebuild()

    def __eq__(self, other: TexTemplate) -> bool:
        return (
            self.body == other.body
            and self.tex_compiler == other.tex_compiler
            and self.output_format == other.output_format
            and self.post_doc_commands == other.post_doc_commands
        )

    def _rebuild(self):
        """Rebuilds the entire TeX template text from ``\\documentclass`` to ``\\end{document}`` according to all settings and choices."""
        self.body = (
            self.documentclass
            + "\n"
            + self.preamble
            + "\n"
            + r"\begin{document}"
            + "\n"
            + self.post_doc_commands
            + "\n"
            + self.placeholder_text
            + "\n"
            + "\n"
            + r"\end{document}"
            + "\n"
        )

    def add_to_preamble(self, txt: str, prepend: bool = False):
        """Adds stuff to the TeX template's preamble (e.g. definitions, packages). Text can be inserted at the beginning or at the end of the preamble.

        Parameters
        ----------
        txt
            String containing the text to be added, e.g. ``\\usepackage{hyperref}``
        prepend
            Whether the text should be added at the beginning of the preamble, i.e. right after ``\\documentclass``. Default is to add it at the end of the preamble, i.e. right before ``\\begin{document}``
        """
        if prepend:
            self.preamble = txt + "\n" + self.preamble
        else:
            self.preamble += "\n" + txt
        self._rebuild()

    def add_to_document(self, txt: str):
        """Adds txt to the TeX template just after \\begin{document}, e.g. ``\\boldmath``

        Parameters
        ----------
        txt
            String containing the text to be added.
        """
        self.post_doc_commands += "\n" + txt + "\n"
        self._rebuild()

    def get_texcode_for_expression(self, expression: str):
        """Inserts expression verbatim into TeX template.

        Parameters
        ----------
        expression
            The string containing the expression to be typeset, e.g. ``$\\sqrt{2}$``

        Returns
        -------
        :class:`str`
            LaTeX code based on current template, containing the given ``expression`` and ready for typesetting
        """
        return self.body.replace(self.placeholder_text, expression)

    def _texcode_for_environment(self, environment: str):
        """Processes the tex_environment string to return the correct ``\\begin{environment}[extra]{extra}`` and
        ``\\end{environment}`` strings

        Parameters
        ----------
        environment
            The tex_environment as a string. Acceptable formats include:
            ``{align*}``, ``align*``, ``{tabular}[t]{cccl}``, ``tabular}{cccl``, ``\\begin{tabular}[t]{cccl}``.

        Returns
        -------
        Tuple[:class:`str`, :class:`str`]
            A pair of strings representing the opening and closing of the tex environment, e.g.
            ``\\begin{tabular}{cccl}`` and ``\\end{tabular}``
        """

        # If the environment starts with \begin, remove it
        if environment[0:6] == r"\begin":
            environment = environment[6:]

        # If environment begins with { strip it
        if environment[0] == r"{":
            environment = environment[1:]

        # The \begin command takes everything and closes with a brace
        begin = r"\begin{" + environment
        if (
            begin[-1] != r"}" and begin[-1] != r"]"
        ):  # If it doesn't end on } or ], assume missing }
            begin += r"}"

        # While the \end command terminates at the first closing brace
        split_at_brace = re.split(r"}", environment, 1)
        end = r"\end{" + split_at_brace[0] + r"}"

        return begin, end

    def get_texcode_for_expression_in_env(self, expression: str, environment: str):
        r"""Inserts expression into TeX template wrapped in \begin{environment} and \end{environment}

        Parameters
        ----------
        expression
            The string containing the expression to be typeset, e.g. ``$\\sqrt{2}$``
        environment
            The string containing the environment in which the expression should be typeset, e.g. ``align*``

        Returns
        -------
        :class:`str`
            LaTeX code based on template, containing the given expression inside its environment, ready for typesetting
        """
        begin, end = self._texcode_for_environment(environment)
        return self.body.replace(self.placeholder_text, f"{begin}\n{expression}\n{end}")

    def copy(self) -> TexTemplate:
        return copy.deepcopy(self)


class TexTemplateFromFile(TexTemplate):
    """A TexTemplate object created from a template file (default: tex_template.tex)

    Parameters
    ----------
    tex_filename
        Path to a valid TeX template file
    kwargs
        Arguments for :class:`~.TexTemplate`.

    Attributes
    ----------
    template_file : :class:`str`
        Path to a valid TeX template file
    body : :class:`str`
        Content of the TeX template file
    tex_compiler : :class:`str`
        The TeX compiler to be used, e.g. ``latex``, ``pdflatex`` or ``lualatex``
    output_format : :class:`str`
        The output format resulting from compilation, e.g. ``.dvi`` or ``.pdf``
    """

    def __init__(
        self, *, tex_filename: str | os.PathLike = "tex_template.tex", **kwargs
    ):
        self.template_file = Path(tex_filename)
        super().__init__(**kwargs)

    def _rebuild(self):
        self.body = self.template_file.read_text()

    def file_not_mutable(self):
        raise Exception("Cannot modify TexTemplate when using a template file.")

    def add_to_preamble(self, txt, prepend=False):
        self.file_not_mutable()

    def add_to_document(self, txt):
        self.file_not_mutable()
