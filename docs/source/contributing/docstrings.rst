=================
Adding Docstrings
=================

A docstring is a string literal that is used right after the definition
of a module, function, class, or method. They are used to document our code.
This page will give you a set of guidelines to write efficient and correct docstrings.


Formatting the Docstrings
-------------------------

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
The Manim Community uses numpy format for the documentation.

Use the numpy format for sections and formatting - see
https://numpydoc.readthedocs.io/en/latest/format.html.

This includes:

1. The usage of ``Attributes`` to specify ALL ATTRIBUTES that a
   class can have and a brief (or long, if
   needed) description.

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
        name
            The class's name.
        id
            The class's id.
        mobj
            The mobject linked to this instance. Defaults to `Mobject()` \
    (is set to that if `None` is specified).

        Attributes
        ----------
        name
            The user's name.
        id
            The user's id.
        singleton
            Something.
        mobj
            The mobject linked to this instance.
        """

        def __init__(name: str, id: int, singleton: MyClass, mobj: Mobject = None):
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

.. note::

   When documenting varargs (args and kwargs), make sure to
   document them by listing the possible types of each value specified,
   like this:

::

    Parameters
    ----------
    args
      The args specified can be either an int or a float.
    kwargs
      The kwargs specified can only be a float.

Note that, if the kwargs expect specific values, those can be specified
in a section such as ``Other Parameters``:

::

    Other Parameters
    ----------------
    kwarg_param_1
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

.. note::
   Also, if this is a video- or animation-related change, please
   try to add an example GIF or video if possible for demonstration
   purposes.

Make sure to be as explicit as possible in your documentation. We all
want the users to have an easier time using this library.

Example:

.. code:: py

    def my_function(
        thing: int,
        other: np.ndarray,
        name: str,
        *,
        d: "SomeClassFromFarAway",
        test: Optional[int] = 45
    ) -> "EpicClassInThisFile":  # typings are optional for now
        """My cool function. Builds and modifies an :class:`EpicClassInThisFile` instance with the given 
            parameters.

        Parameters
        ----------
        thing
            Specifies the index of life.
        other
            Specifies something cool.
        name
            Specifies my name.
        d
            Sets thing D to this value.
        test
            Defines the number of times things should be tested. \
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
