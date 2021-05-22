=================
Adding References
=================

Reference to types in documentation
-----------------------------------

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