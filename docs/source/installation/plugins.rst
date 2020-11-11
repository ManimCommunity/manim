Plugins
=======

Plugins, or addons, are features meant to extend the core of Manim. Since
Manim is extensible, we'll go over how to install plugins as well as create
your own.

Installing Plugins
******************
Plugins can be installed directly in two ways:
1. Placing the respective plugin directly underneath the ``manim/plugins``
   folder. This is only recommended if the plugin is not directly supported
   by pip. 

2. Via ``pip install manimce-YourPluginName``. The naming convention for
   plugins is to prefix the plugin with ``manimce-`` (e.g. ``pip install
   manimce-{}``) which makes them easier to dynamically locate as well as
   easier for user to find on organizations such as `PyPi.org`<https://pypi.org/>_.

Both pip supported plugins and plugins placed in the ``manim/plugins`` folder
will be loaded dynamically at runtime via the ``importlib`` standard library
package.

Creating Plugins
****************
Depending on how you would like to distribute your plugin determines how
plugins should be created. There's more boilerplate code involved with
distributing packages with pip, but the process involves shared steps.

Create your feature
-------------------
The recommended plugin structure is as follows:
.. code-block:: bash
    PluginName/
    └─ src/
        └─pluginname.py
    └─ setup.py

Using Plugins in Projects
*************************