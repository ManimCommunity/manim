Installation
============

Depending on your use case, different installation options are recommended:
if you just want to play around with Manim for a bit, interactive in-browser
notebooks are a really simple way of exploring the library as they
require no local installation. Head over to
https://try.manim.community to give our interactive tutorial a try.

Otherwise, if you intend to use Manim to work on an animation project,
we recommend installing the library locally (either to a conda environment,
your system's Python, or via Docker).

.. warning::

   Note that there are several different versions of Manim. The
   instructions on this website are **only** for the *community edition*.
   Find out more about the :ref:`differences between Manim
   versions <different-versions>` if you are unsure which
   version you should install.

#. :ref:`Installing Manim to a conda environment <conda-installation>`
#. :ref:`Installing Manim to your system's Python <local-installation>`
#. :ref:`Using Manim via Docker <docker-installation>`
#. :ref:`Interactive Jupyter notebooks via Binder / Google Colab
   <interactive-online>`


.. _conda-installation:

Installing Manim via Conda and related environment managers
***********************************************************

Conda is a package manager for Python that allows creating environments
where all your dependencies are stored. Like this, you don't clutter up your PC with
unwanted libraries and you can just delete the environment when you don't need it anymore.
It is a good way to install manim since all dependencies like ``pycairo``, etc. come with it.
Also, the installation steps are the same, no matter if you are
on Windows, Linux, Intel Macs or on Apple Silicon.

.. NOTE::

   There are various popular alternatives to Conda like
   `mamba <https://mamba.readthedocs.io/en/latest/>`__ /
   `micromamba <https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html>`__,
   or `pixi <https://pixi.sh>`__.
   They all can be used to setup a suitable, isolated environment
   for your Manim projects.

The following pages show how to install Manim in a conda environment:

.. toctree::
   :maxdepth: 2

   installation/conda



.. _local-installation:

Installing Manim locally
************************

Manim is a Python library, and it can be
installed via `pip <https://pypi.org/project/manim/>`__
or `conda <https://anaconda.org/conda-forge/manim/>`__. However,
in order for Manim to work properly, some additional system
dependencies need to be installed first.

Manim requires Python version ``3.9`` or above to run.

.. hint::

   Depending on your particular setup, the installation process
   might be slightly different. Make sure that you have tried to
   follow the steps on the following pages carefully, but in case
   you hit a wall we are happy to help: either `join our Discord
   <https://www.manim.community/discord/>`__, or start a new
   Discussion `directly on GitHub
   <https://github.com/ManimCommunity/manim/discussions>`__.


To install Manim locally, check out the following pages. Note
that the process for Linux is slightly different - if you're
on Linux please follow the instructions in the `Linux section <installation/linux>`_.

.. toctree::
   :maxdepth: 1

   installation/locally


Once you've installed the core dependencies, you can proceed to
install the optional dependencies, depending on your system.

- :doc:`installation/windows`
- :doc:`installation/macos`
- :doc:`installation/linux`

Once Manim is installed locally, you can proceed to our
:doc:`quickstart guide <tutorials/quickstart>` which walks you
through rendering a first simple scene.

As mentioned above, do not worry if there are errors or other
problems: consult our :doc:`FAQ section </faq/index>` for help
(including instructions for how to ask Manim's community for help).



.. _docker-installation:

Using Manim via Docker
**********************

`Docker <https://www.docker.com>`__ is a virtualization tool that
allows the distribution of encapsulated software environments (containers).

The following pages contain more information about the docker image
maintained by the community, ``manimcommunity/manim``:

.. toctree::

   installation/docker


.. _interactive-online:

Interactive Jupyter notebooks for your browser
**********************************************

Manim ships with a built-in ``%%manim`` IPython magic command
designed for the use within `Jupyter notebooks <https://jupyter.org>`__.
Our interactive tutorial over at https://try.manim.community illustrates
how Manim can be used from within a Jupyter notebook.

The following pages explain how you can setup interactive environments
like that yourself:

.. toctree::

   installation/jupyter

.. _editor-addons:

Editors
********

If you're using Visual Studio Code you can install an extension called
*Manim Sideview* which provides automated rendering and an integrated preview
of the animation inside the editor. The extension can be installed through the
`marketplace of VS Code <https://marketplace.visualstudio.com/items?itemName=Rickaym.manim-sideview>`__.


Installation for developers
***************************

In order to change code in the library, it is recommended to
install Manim in a different way. Please follow the instructions
in our :doc:`contribution guide <contributing>` if you are
interested in that.
