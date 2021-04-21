====================
Adding Documentation
====================

When submitting a new class through a PR, or any changes in general,
there should be documentation where possible. Here are the guidelines
for writing it.

Formatting and Running Tests
----------------------------

Please begin the description of the class/function in the same line as
the 3 quotes:

.. code:: py

    def do_this():
        """This is correct.
        (...)
        """


    def dont_do_this():
        """
        This is incorrect.
        (...)
        """

NumPy Format
------------

Use the numpy format for sections and formatting - see
https://numpydoc.readthedocs.io/en/latest/format.html.

This includes:

1. The usage of ``Attributes`` to specify ALL ATTRIBUTES that a
   class can have, their respective types, and a brief (or long, if
   needed) description. (See more on :ref:`_types`)

Also, ``__init__`` parameters should be specified as ``Parameters`` **on
the class docstring**, *rather than under* ``__init__``. Note that this
can be omitted if the parameters and the attributes are the same
(i.e., dataclass) - priority should be given to the ``Attributes``
section, in this case, which must **always be present**, unless the
class specifies no attributes at all. (See more on Parameters in number
2 of this list.)

Example:

.. code:: py

    class MyClass:
        """My cool class. Long or short (whatever is more appropriate) description here.

        Parameters
        ----------
        name : :class:`str`
            The class's name.
        id : :class:`int`
            The class's id.
        mobj : Optional[:class:`~.Mobject`], optional
            The mobject linked to this instance. Defaults to `Mobject()` \
    (is set to that if `None` is specified).

        Attributes
        ----------
        name : :class:`str`
            The user's name.
        id : :class:`int`
            The user's id.
        singleton : :class:`MyClass`
            Something.
        mobj : :class:`~.Mobject`
            The mobject linked to this instance.
        """

        def __init__(
            name, id, singleton, mobj=None
        ):  # in-code typehints are optional for now
            ...

2. The usage of ``Parameters`` on functions to specify how
   every parameter works and what it does. This should be excluded if
   the function has no parameters. Note that you **should not** specify
   the default value of the parameter on the type. On the documentation
   render, this is already specified on the function's signature. If you
   need to indicate a further consequence of value omission or simply
   want to specify it on the docs, make sure to **specify it in the
   parameter's DESCRIPTION**.

See an example on list item 4.

.. NOTE::

   When documenting varargs (args and kwargs), make sure to
   document them by listing the possible types of each value specified,
   like this:

::

    Parameters
    ----------
    args : Union[:class:`int`, :class:`float`]
      The args specified can be either an int or a float.
    kwargs : :class:`float`
      The kwargs specified can only be a float.

Note that, if the kwargs expect specific values, those can be specified
in a section such as ``Other Parameters``:

::

    Other Parameters
    ----------------
    kwarg_param_1 : :class:`int`
      Parameter documentation here
    (etc)

3. The usage of ``Returns`` to indicate what is the type of this
   function's return value and what exactly it returns (i.e., a brief -
   or long, if needed - description of what this function returns). Can
   be omitted if the function does not explicitly return (i.e., always
   returns ``None`` because ``return`` is never specified, and it is
   very clear why this function does not return at all). In all other
   cases, it should be specified.

See an example on list item 4.

4. The usage of ``Examples`` in order to specify an example of usage of
   a function **is highly encouraged** and, in general, should be
   specified for *every function* unless its usage is **extremely
   obvious**, which can be debatable. Even if it is, it's always a good
   idea to add an example in order to give a better orientation to the
   documentation user. Use the following format for Python code:

   .. code:: rst

       ::

       # python code here

**NOTE: Also, if this is a video- or animation-related change, please
try to add an example GIF or video if possible for demonstration
purposes.**

Make sure to be as explicit as possible in your documentation. We all
want the users to have an easier time using this library.

Example:

.. code:: py

    def my_function(thing, other, name, *, d, test=45):  # typings are optional for now
        """My cool function. Builds and modifies an :class:`EpicClassInThisFile` instance with the given parameters.

      Parameters
      ----------
      thing : :class:`int`
          Specifies the index of life.
      other : :class:`numpy.ndarray`
          Specifies something cool.
      name : :class:`str`
          Specifies my name.
      d : :class:`~.SomeClassFromFarAway`
          Sets thing D to this value.
      test : :class:`int`, optional
          Defines the amount of times things should be tested. \
    Defaults to 45, because that is almost the meaning of life.

      Returns
      -------
      :class:`EpicClassInThisFile`
          The generated EpicClass with the specified attributes and modifications.

      Examples
      --------
      Normal usage::

          my_function(5, np.array([1, 2, 3]), "Chelovek", d=SomeClassFromFarAway(cool=True), test=5)
      """
        # code...
        pass

.. _types:

Reference to types in documentaion
----------------------------------

Always specify types with the correct **role** (see
https://www.sphinx-doc.org/en/1.7/domains.html#python-roles) for the
sake of proper rendering. E.g.: Use ``:class:`int``` to refer to an int
type, and in general ``:class:`<path>`​`` to refer to a certain class
(see ``Path specification`` below). See after for more specific
instructions.

Path specifications
~~~~~~~~~~~~~~~~~~~

1. If it's on stdlib: Use ``<name>`` directly. If it's a class, just the
   name is enough. If it's a method (``:meth:``) or attribute
   (``:attr:``), dotted names may be used (e.g.
   ``:meth:`str.to_lower`​``).

Example: ``:class:`int`​``, ``:class:`str`​``, ``:class:`float`​``,
``:class:`bool`​``

2. If it's on the same file as the docstring or, for methods and
   attributes, under the same class, then the name may also be specified
   directly.

Example: ``:class:`MyClass`​`` referring to a class in the same file;
``:meth:`push`​`` referring to a method in the same class;
``:meth:`MyClass.push`​`` referring to a method in a different class in
the same file; ``:attr:`color`​`` referring to an attribute in the same
class; ``:attr:`MyClass.color`​`` referring to an attribute in a
different class in the same file.

3. If it's on a different file, then you may either use the full dotted
   name (e.g. ``~manim.animations.Animation``) or simply use the
   shortened way (``~.Animation``). Note that, if there is ambiguity,
   then the full dotted name must be used where the actual class can't
   be deduced. Also, note the ``~`` before the path - this is so that it
   displays just ``Animation`` instead of the full location in the
   rendering. It can be removed for disambiguation purposes only.

Example: ``:class:`~.Animation`​``, ``:meth:`~.VMobject.set_color`​``,
``:attr:`~.VMobject.color`​``

4. If it's a class from a different module, specify the full dotted
   syntax.

Example: ``:class:`numpy.ndarray`​`` for a numpy ndarray.

Reference type specifications
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**The following instructions refer to types of attributes, parameters,
and return values.** When specifying a type mid-text, it does not
necessarily have to be typeset. However, if it's a class name, a method,
or an enum's attribute/variant, then it is recommended to be typeset at
least on the first occurrence of the name so that the users can quickly
jump to the related documentation.

1. Class names should be wrapped in ``:class:`path_goes_here`​``. See
   examples in the subsection above.
2. Method names should be wrapped in ``:meth:`path_goes_here`​``. See
   examples in the subsection above.
3. Attribute names should be wrapped in ``:attr:`path_goes_here`​``. See
   examples in the subsection above.
4. If ``None`` can also be specified, use ``Optional[type]``, where
   ``type`` must follow the guidelines in the current section.

Example: ``Optional[:class:`str`]`` means you can either specify a
``str`` or ``None``.

5. If more than one type is possible, use
   ``Union[type_1, type_2, (...), type_n]``, where all the ``type_n``
   must follow the guidelines in the current section. Note that, if one
   of these types is ``None``, then the Union should be wrapped with
   ``Optional`` instead.

Example: ``Union[:class:`str`, :class:`int`]`` for either ``str`` or
``int``. ``Optional[Union[:class:`int`, :class:`bool`]]`` for either
``int``, ``bool`` or ``None``.

6. **Dictionaries:** Use ``Dict[key_type, value_type]``, where
   ``key_type`` and ``value_type`` must follow the guidelines in the
   current section.

Example: ``Dict[:class:`str`, :class:`~.Mobject`]`` is a dictionary that
maps strings to Mobjects.
``Dict[:class:`str`, Union[:class:`int`, :class:`MyClass`]]`` is a
dictionary that maps a string to either an int or an instance of
``MyClass``.

7. **If the parameter is a list:** Note that it is very rare to require
   the parameter to be exactly a ``list`` type. One could usually
   specify a ``tuple`` instead, for example. So, in order to cover all
   cases, consider:

   1. If the parameter only needs to be an ``Iterable``, i.e., if the
      function only requires being able to iterate over this parameter's
      value (e.g. can be a ``list``, ``tuple``, ``str``, but also
      ``zip()``, ``iter()`` and so on), then specify
      ``Iterable[type_here]``, where ``type_here`` is the type of the
      iterable's yielded elements and should follow the format in the
      present section (``Type specifications``).

   Example: ``Iterable[:class:`str`]`` for any iterable of strings;
   ``Iterable[:class:`~.Mobject`]`` for an iterable of Mobjects; etc.

   2. If you require being able to index the parameter (i.e. ``x[n]``)
      or retrieve its length (i.e. ``len(x)``), or even just pass it to
      a function that requires any of those, then specify ``Sequence``,
      which allows any list-like object to be specified (e.g. ``list``,
      ``tuple``...)

   Example: ``Sequence[:class:`str`]`` for a sequence of strings;
   ``Sequence[Union[:class:`str`, :class:`int`]]`` for a sequence of
   integers or strings.

   3. If you EXPLICITLY REQUIRE it to be a ``list`` for some reason,
      then use ``List[type]``, where ``type`` is the type that any
      element in the list will have. It must follow the guidelines in
      the current section.

8. **If the return type is a list or tuple:** Specify ``List[type]`` for
   a list, ``Tuple[type_a, type_b, (...), type_n]`` for a tuple (if the
   elements are all different) or ``Tuple[type, ...]`` (if all elements
   have the same type). Each ``type_n`` on those representations
   corresponds to elements in the returned list/tuple and must follow
   the guidelines in the current section.

Example: ``List[Optional[:class:`str`]]`` for a list that returns
elements that are either a ``str`` or ``None``;
``Tuple[:class:`str`, :class:`int`]`` for a tuple of type
``(str, int)``; ``Tuple[:class:`int`, ...]`` for a tuple of variable
length with only integers.


Adding type hints to functions and parameters
---------------------------------------------

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


* the .__init__() method always should have None as its return type.

* functions and lambda functions should get the typehint ``Callable``

.. code:: py

    rate_func: Callable[[float], float] = lambda t: smooth(1 - t)

*  numpy arrays can get type hints with ``np.ndarray``

Missing Sections
----------------
* Tools for typehinting
* Link to MyPy
* Mypy and numpy import errors: https://realpython.com/python-type-checking/#running-mypy
* Where to find the alias
* When to use Object and when to use "Object".
* The use of a TypeVar on the type hints for copy().
* The definition and use of Protocols (like Sized, or Sequence, or Iterable...)
