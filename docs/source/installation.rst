Installation
============

Depending on your use case, different installation options are recommended:
if you just want to play around with Manim for a bit, interactive in-browser
notebooks are a really simple way of exploring the library as they
require no local installation. Head over to
https://try.manim.community to give our interactive tutorial a try.

Otherwise, if you intend to use Manim to work on an animation project,
we recommend installing the library locally (preferably to some isolated
virtual Python environment, or a conda-like environment, or via Docker).

.. warning::

   Note that there are several different versions of Manim. The
   instructions on this website are **only** for the *community edition*.
   Find out more about the :ref:`differences between Manim
   versions <different-versions>` if you are unsure which
   version you should install.

#. :ref:`(Recommended) Installing Manim via Python's package manager pip
   <local-installation>`
#. :ref:`Installing Manim to a conda environment <conda-installation>`
#. :ref:`Using Manim via Docker <docker-installation>`
#. :ref:`Interactive Jupyter notebooks via Binder / Google Colab
   <interactive-online>`


.. _local-installation:

Installing Manim locally via pip
********************************

The recommended way of installing Manim is by using Python's package manager
pip. If you already have a Python environment set up, you can simply run
``pip install manim`` to install the library.

Our :doc:`local installation guide <installation/uv>` provides more detailed
instructions, including best practices for setting up a suitable local environment.

.. toctree::
   :hidden:

   installation/uv

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

.. caution::

   This extension is not officially maintained by the Manim Community.
   If you run into issues, please report them to the extension's author.


Installation for developers
***************************

In order to change code in the library, it is recommended to
install Manim in a different way. Please follow the instructions
in our :doc:`contribution guide <contributing>` if you are
interested in that.
