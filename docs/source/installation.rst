Installation
============

Depending on your use case, different installation options are recommended:
if you just want to play around with Manim for a bit, interactive in-browser
notebooks are a really simple way of exploring the library as they
require no local installation. Head over to
https://try.manim.community to give our interactive tutorial a try.

Otherwise, if you intend to use Manim to work on an animation project,
we recommend installing the library locally (either to your system's
Python, or via Docker).

.. warning::

   Note that there are several different versions of Manim. The
   instructions on this website are **only** for the *community edition*.
   Find out more about the :doc:`differences between Manim
   versions <installation/versions>` if you are unsure which
   version you should install.

#. :ref:`Installing Manim to your system's Python <local-installation>`
#. :ref:`Using Manim via Docker <docker-installation>`
#. :ref:`Interactive Jupyter notebooks via Binder / Google Colab
   <interactive-online>`


.. _local-installation:

Installing Manim locally
************************

Manim is a Python library, and it can be
`installed via pip <https://pypi.org/project/manim/>`__. However,
in order for Manim to work properly, some additional system
dependencies need to be installed first. The following pages have
operating system specific instructions for you to follow.

Manim is **only** compatible with Python versions `3.7–3.9`, but not `3.10` for now.

.. hint::

   Depending on your particular setup, the installation process
   might be slightly different. Make sure that you have tried to
   follow the steps on the following pages carefully, but in case
   you hit a wall we are happy to help: either `join our Discord
   <https://www.manim.community/discord/>`__, or start a new
   Discussion `directly on GitHub
   <https://github.com/ManimCommunity/manim/discussions>`__.

.. toctree::
   :maxdepth: 2

   installation/windows
   installation/macos
   installation/linux
   installation/troubleshooting

Once Manim is installed locally, you can proceed to our
:doc:`quickstart guide <tutorials/quickstart>` which walks you
through rendering a first simple scene.

As mentioned above, do not worry if there are errors or other
problems: consult our :doc:`troubleshooting
guide <installation/troubleshooting>` for help, or get in touch
with the community via `GitHub discussions
<https://github.com/ManimCommunity/manim/discussions>`__ or
`Discord <https://www.manim.community/discord/>`__.



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
