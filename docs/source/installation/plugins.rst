.. _plugins:

=======
Plugins
=======

Plugins are features that extend Manim's core functionality. Since Manim is
extensible and not everything belongs in its core, we'll go over how to
install, use, and create your own plugins.

.. note:: The standard naming convention for plugins is to prefix the plugin with ``manim-``. This makes them easy to find on packages indexes such as PyPI.

.. WARNING::

    The plugin feature is new and under active development. Expect updates
    for the best practices on installing, using, and creating plugins; as
    well as new subcommands/flags for ``manim plugins``

Installing Plugins
******************
Plugins can be easily installed via the ``pip``
command:

.. code-block:: bash

    pip install manim-*

After installing a plugin, you may use the ``manim plugins`` command to list
your available plugins, see the following help output:

.. code-block:: bash

    manim plugins -h
    usage: manim plugins -h -l

    Utility command for managing plugins

    optional arguments:
    -h, --help    show this help message and exit
    -l, --list    Lists all available plugins

    Made with <3 by the manim community devs

Using Plugins in Projects
*************************
Plugins specified in ``plugins/__init__.py`` are imported automatically by
manim's ``__init__.py``. As such, writing:

.. code-block:: python

    from manim import *

in your projects will import any of the plugins and packages written in the
``plugins/__init__.py``.

.. code-block:: python

    import manim_cool_plugin
    # or
    from manim_cool_plugin import feature_x, feature_y, ...

This is especially useful to modify if your projects will involve the same
plugins. Alternatively, you can manually specify the same imports into your
project scripts as well. 

Creating Plugins
****************
Plugins are intended to extend Manim's core functionality. If you aren't sure
whether a feature should be included in Manim's core, feel free to ask over
on the `Discord server <https://discord.gg/mMRrZQW>`_. Visit
`manim-plugintemplate <https://pypi.org/project/manim-plugintemplate/>`_
on PyPI.org which servers as an in-depth tutorial for creating plugins.

.. code-block:: bash

    pip install manim-plugintemplate
