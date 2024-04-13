"""Mobjects used to display Text using Pango or LaTeX.

Modules
=======

.. autosummary::
    :toctree: ../reference

    ~code_mobject
    ~numbers
    ~tex_mobject
    ~text_mobject
"""
def register_font(font_file: str | Path):
    """Temporarily add a font file to Pango's search path.

    This searches for the font_file at various places. The order it searches it described below.

    1. Absolute path.
    2. In ``assets/fonts`` folder.
    3. In ``font/`` folder.
    4. In the same directory.

    Parameters
    ----------
    font_file
        The font file to add.

    Examples
    --------
    Use ``with register_font(...)`` to add a font file to search
    path.

    .. code-block:: python

        with register_font("path/to/font_file.ttf"):
            a = Text("Hello", font="Custom Font Name")

    Raises
    ------
    FileNotFoundError:
        If the font doesn't exists.

    AttributeError:
        If this method is used on macOS.

    .. important ::

        This method is available for macOS for ``ManimPango>=v0.2.3``. Using this
        method with previous releases will raise an :class:`AttributeError` on macOS.
    """

    input_folder = Path(config.input_file).parent.resolve()
    possible_paths = [
        Path(font_file),
        input_folder / "assets/fonts" / font_file,
        input_folder / "fonts" / font_file,
        input_folder / font_file,
    ]
    for path in possible_paths:
        path = path.resolve()
        if path.exists():
            file_path = path
            logger.debug("Found file at %s", file_path.absolute())
            break
    else:
        error = f"Can't find {font_file}." f"Tried these : {possible_paths}"
        raise FileNotFoundError(error)

    try:
        assert manimpango.register_font(str(file_path))
        yield
    finally:
        manimpango.unregister_font(str(file_path))
