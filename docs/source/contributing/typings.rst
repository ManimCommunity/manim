==============
Adding Typings
==============

Adding type hints to functions and parameters
---------------------------------------------

.. warning::
   This section is still a work in progress.

If you've never used type hints before, this is a good place to get started:
https://realpython.com/python-type-checking/#hello-types.

When adding type hints to manim, there are some guidelines that should be followed:

* Coordinates have the typehint ``Sequence[float]``, e.g.

.. code:: py

    def set_points_as_corners(self, points: Sequence[float]) -> "VMobject":
        """Given an array of points, set them as corner of the Vmobject."""

* ``**kwargs`` has no typehint

* Mobjects have the typehint "Mobject", e.g.

.. code:: py

    def match_color(self, mobject: "Mobject"):
        """Match the color with the color of another :class:`~.Mobject`."""
        return self.set_color(mobject.get_color())

* Colors have the typehint ``Color``, e.g.

.. code:: py

    def set_color(self, color: Color = YELLOW_C, family: bool = True):
        """Condition is function which takes in one arguments, (x, y, z)."""

* As ``float`` and ``Union[int, float]`` are the same, use only ``float``

* For numpy arrays use the typehint ``np.ndarray``

* Functions that does not return a value should get the type hint ``None``. (This annotations help catch the kinds of subtle bugs where you are trying to use a meaningless return value. )

.. code:: py

    def height(self, value) -> None:
        self.scale_to_fit_height(value)

* Parameters that are None by default should get the type hint ``Optional``

.. code:: py

    def rotate(
        self,
        angle,
        axis=OUT,
        about_point: Optional[Sequence[float]] = None,
        **kwargs,
    ):
        pass


* The ``__init__()`` method always should have None as its return type.

* Functions and lambda functions should get the typehint ``Callable``

.. code:: py

    rate_func: Callable[[float], float] = lambda t: smooth(1 - t)


* Assuming that typical path objects are either Paths or strs, one can use the typehint ``typing.Union[str, pathlib.Path]``

.. note::
   As a helper for tool for typesets, you can use `typestring-parser
   <https://github.com/Dominik1123/typestring-parser>`_ 
   which can be accessed by first installing it via ``pip`` - ``pip install typestring-parser`` and
   then using ``from typestring_parser import parse``.

.. doctest::
    :options: +SKIP
    
    >>> from typestring_parser import parse
    >>> parse("int")
    <class 'int'>
    >>> parse("int or str")
    typing.Union[int, str]
    >>> parse("list of str or str")
    typing.Union[typing.List[str], str]
    >>> parse("list of (int, str)")
    typing.List[typing.Tuple[int, str]]

Missing Sections for typehints are:
-----------------------------------
* Tools for typehinting
* Link to MyPy
* Mypy and numpy import errors: https://realpython.com/python-type-checking/#running-mypy
* Where to find the alias
* When to use Object and when to use "Object".
* The use of a TypeVar on the type hints for copy().
* The definition and use of Protocols (like Sized, or Sequence, or Iterable...)