.. _plugins:

=======
Plugins
=======

Plugins are features that extend Manim's core functionality. Since Manim is
extensible and not everything belongs in its core, we'll go over how to
install, use, and create your own plugins.

*Note: The standard naming convention for plugins is to prefix the plugin*
*with* ``manim-``

Installing Plugins
******************
Plugins can be easily installed to your python site-packages via the ``pip``
command:

.. code-block:: bash

    pip install manim-*

The standard naming conventions makes plugins easy to find on organizations
such as `PyPi.org <https://pypi.org/>`_. After installing plugins, you may
use the ``manim plugins`` command to list and update Manim to import your
plugins, see the following help output:

.. code-block:: bash

    manim plugins -h
    usage: manim plugins -h -u -l

    Utility command for managing plugins

    optional arguments:
    -h, --help    show this help message and exit
    -l, --list    Lists all available plugins
    -u, --update  Updates plugins/__init__.py

    Made with <3 by the manim community devs

or

.. code-block:: bash

    manim plugins -lu

Using Plugins in Projects
*************************

Plugins are imported automatically by updating manim's
``plugins/__init__.py`` via the ``manim plugins`` command. The modules in
each respective plugin are imported in this file and made accessible to Manim
and your projects via:

.. code-block:: python

    from manim import *



Creating Plugins
****************
Plugins are intended to extend Manim's core functionality. If you aren't sure
whether a feature should be included in Manim's core, feel free to ask over
on the `Discord server <https://discord.gg/mMRrZQW>`_. Visit
`manim-plugintemplate <https://pypi.org/project/manim-plugintemplate/>`_
on PyPI.org which servers as an in-depth tutorial for creating plugins.

.. code-block:: bash

    pip install manim-plugintemplate
