.. _plugins:

=======
Plugins
=======

Plugins are features that extend Manim's core functionality. Since Manim is
extensible and not everything belongs in its core, we'll go over how to
install, use, and create your own plugins.

.. note::

    The standard naming convention for plugins is to prefix the plugin with
    ``manim-``. This makes them easy for users to find on package
    repositories such as PyPI.

.. WARNING::

    The plugin feature is new and under active development. Expect updates
    for the best practices on installing, using, and creating plugins; as
    well as new subcommands/flags for ``manim plugins``

.. tip::

    See https://plugins.manim.community/ for the list of plugins available.

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
    Usage: manim plugins [OPTIONS]

      Manages Manim plugins.

    Options:
    -l, --list  List available plugins
    -h, --help  Show this message and exit.

    Made with <3 by Manim Community developers.

You can list plugins as such:

.. code-block:: bash

    manim plugins -l
    Plugins:
    • manim_plugintemplate

Using Plugins in Projects
*************************
For enabling a plugin ``manim.cfg`` or command line parameters should be used.

.. important::

    The plugins should be module name of the plugin and not PyPi name.

Enabling plugins through ``manim.cfg``

.. code-block:: ini

    [CLI]
    plugins = manim_rubikscube

For specifying multiple plugins, comma-separated values must be used.

.. code-block:: ini

    [CLI]
    plugins = manim_rubikscube, manim_plugintemplate

Creating Plugins
****************

Plugins are intended to extend Manim's core functionality. If you aren't sure
whether a feature should be included in Manim's core, feel free to ask over
on the `Discord server <https://www.manim.community/discord/>`_. Visit
`manim-plugintemplate <https://pypi.org/project/manim-plugintemplate/>`_
on PyPI.org which serves as an in-depth tutorial for creating plugins.

.. code-block:: bash

    pip install manim-plugintemplate

The only requirement of manim plugins is that they specify an entry point
with the group, ``"manim.plugins"``. This allows Manim to discover plugins
available in the user's environment. Everything regarding the plugin's
directory structure, build system, and naming are completely up to your
discretion as an author.

The standard way to specify an entry point (see
`the Python packaging guide <https://packaging.python.org/specifications/entry-points/>`__
for details) is to include the following in your ``pyproject.toml``:

.. code-block:: toml

    [project.entry-points."manim.plugins"]
    "name" = "object_reference"

.. versionremoved:: 0.18.1

    Plugins should be imported explicitly to be usable in user code. The plugin
    system will probably be refactored in the future to provide a more structured
    interface.

A note on Renderer Compatibility
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Depending on which renderer is currently active, custom mobjects
created in your plugin might want to behave differently as the
corresponding mobject base classes are (unfortunately) not fully
compatible.

The currently active renderer can be queried by checking the value
of ``manim.config.renderer``. All possible renderer types are given
by :class:`.constants.RendererType`. The module :mod:`.manim.mobject.utils`
contains utility functions that return the base class for the currently
active renderer.

A simple form of renderer compatibility (by hot-swapping the class
inheritance chain) for Mobjects directly inheriting from
:class:`.Mobject` or :class:`.VMobject` can be achieved by using the
:class:`.mobject.opengl.opengl_compatibility.ConvertToOpenGL` metaclass.
