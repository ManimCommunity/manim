==================
Adding Admonitions
==================

Adding Blocks for Tip, Note, Important etc. (Admonitions)
---------------------------------------------------------

The following directives are called Admonitions. You
can use them to point out additional or important
information. Here are some examples:

See also
~~~~~~~~

.. code-block:: rest

   .. seealso::
        Some ideas at :mod:`~.tex_mobject`, :class:`~.Mobject`, :meth:`~.Mobject.add_updater`, :attr:`~.Mobject.depth`, :func:`~.make_config_parser`

.. seealso::
    Some ideas at :mod:`~.tex_mobject`, :class:`~.Mobject`, :meth:`~.Mobject.add_updater`, :attr:`~.Mobject.depth`, :func:`~.make_config_parser`

.. index:: reST directives; note



Note
~~~~

.. code-block:: rest

   .. note::
      A note

.. note::
   A note

Tip
~~~

.. code-block:: rest

   .. tip::
      A tip

.. tip::
   A tip

You may also use the admonition **hint**, but this is very similar
and **tip** is more commonly used in the documentation.

Important
~~~~~~~~~

.. code-block:: rest

   .. important::
      Some important information which should be considered.

.. important::
   Some important information which should be considered.

Warning
~~~~~~~

.. code-block:: rest

   .. warning::
      Some text pointing out something that people should be warned about.

.. warning::
   Some text pointing out something that people should be warned about.

You may also use the admonitions **caution** or even **danger** if the
severity of the warning must be stressed.

Attention
~~~~~~~~~

.. code-block:: rest

   .. attention::
      A attention

.. attention::
   A attention

You can find further information about Admonitions here: https://pradyunsg.me/furo/reference/admonitions/
