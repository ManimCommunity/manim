==============
Adding Typings
==============

.. warning::
   This section is still a work in progress.

Adding type hints to functions and parameters
---------------------------------------------

Manim is currently in the process of adding type hints into the library. In this
section, you will find information about the standards used and some general
guidelines.

If you've never used type hints before, this is a good place to get started:
https://realpython.com/python-type-checking/#hello-types.

Typing standards
~~~~~~~~~~~~~~~~

Manim uses `mypy`_ to type check its codebase. You will find a list of
configuration values in the ``mypy.ini`` configuration file.

To be able to use the newest typing features not available in the lowest
supported Python version, make use of `typing_extensions`_.

To be able to use the new Union syntax (``|``) and builtins subscripting, use
the ``from __future__ import annotations`` import.

.. _mypy: https://mypy-lang.org/
.. _typing_extensions: https://pypi.org/project/typing-extensions/

Typing guidelines
~~~~~~~~~~~~~~~~~

* Manim has a dedicated :mod:`~.typing` module where type aliases are provided.
  Most of them may seem redundant, in particular the ones related to ``numpy``.
  This is in anticipation of the support for shape type hinting
  (`related issue <https://github.com/numpy/numpy/issues/16544>`_). Besides the
  pending shape support, using the correct type aliases will help users understand
  which shape should be used.

* Always use a type hint of ``None`` for functions that does not return
  a value (this also applies to ``__init__``), e.g.:

.. code:: py

    def height(self, value) -> None:
        self.scale_to_fit_height(value)

* For variables representing paths, use the ``StrPath`` or ``StrOrBytesPath``
  type alias defined in the :mod:`~.typing` module.

* ``*args`` and ``**kwargs`` shouldn't be left untyped (in most cases you can
  use ``Any``).

* Following `PEP 484 <https://peps.python.org/pep-0484/#the-numeric-tower>`_,
  use ``float`` instead of ``int | float``.

* Mobjects have the typehint ``Mobject``, e.g.:

.. code:: py

    def match_color(self, mobject: "Mobject"):
        """Match the color with the color of another :class:`~.Mobject`."""
        return self.set_color(mobject.get_color())

* Always parametrize generics (``list[int]`` instead of ``list``,
  ``type[Any]`` instead of ``type``, etc.). This also applies to callables:

.. code:: py

    rate_func: Callable[[float], float] = lambda t: smooth(1 - t)

Missing Sections for typehints are:
-----------------------------------

* Mypy and numpy import errors: https://realpython.com/python-type-checking/#running-mypy
* When to use ``object`` vs ``Any``
* The use of a TypeVar on the type hints for ``copy()``.
* The definition and use of Protocols (like ``Sized``, ``Sequence``, ``Iterable``...)
