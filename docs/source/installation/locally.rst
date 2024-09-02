Installing Manim Locally
************************
For the most part, installing Manim is the same across operating systems. However,
you can take some shortcuts depending on your operating system.

- MacOS users can use `Homebrew <https://brew.sh>`_ to install Manim - check out the section for :ref:`MacOS<macos_homebrew>`.


However, if you don't want to use a package manager, check out the section for :ref:`all operating systems<all_os>`.

.. _all_os:

All Operating Systems
=====================
Manim requires a Python version of at least ``3.9`` to run.
If you're not sure if you have python installed, or want to check
what version of Python you have, try running::

  python --version

If it errors out, you most likely don't have Python installed. Otherwise, if your
python version is ``3.9`` or higher, you can proceed to :ref:`installing Manim with Pip<manim_pip>`.

.. hint::

   On MacOS and some Linux distributions, you may have to use ``python3`` instead of ``python``.
   In this document, we will use ``python``, but depending on your operating system you may have to
   use ``python3``.

Installing Python
-----------------
If you don't have Python installed, head over to https://www.python.org, download an installer
for a recent (preferably the latest) version of Python, and follow its instructions to get Python
installed on your system.

.. note::

   We have received reports of problems caused by using the version of
   Python that can be installed from the Windows Store. At this point,
   we recommend staying away from the Windows Store version. Instead,
   install Python directly from the `official website <https://www.python.org>`__.


After installing Python, running the command::

  python --version

Should be successful. If it is not, try checking out :ref:`this FAQ entry<not-on-path>`.

.. _manim_pip:

Installing Manim
----------------
At this point, installing manim should be as easy as running::

  python -m pip install manim

To confirm Manim is working, you can run::

  manim --version


.. _macos_homebrew:

MacOS
=====
The easiest way to install Manim on macOS is via the popular `package manager Homebrew <https://brew.sh>`__.
If you want to use Homebrew but do not have it installed yet, please
follow `Homebrew's installation instructions <https://docs.brew.sh/Installation>`__.

.. note::

   For a while after Apple released its new ARM-based processors (the Apple Silicon chips like the *"M1 chip"*),
   the recommended way of installing Manim relied on *Rosetta*, Apple's compatibility
   layer between Intel and ARM architectures. This is no longer necessary, Manim can
   (and is recommended to) be installed natively.

Manim has a Homebrew formula, so you can just run::

  brew install manim

And you should have Manim all installed! Head on over to install the :ref:`optional dependencies<optional_dependencies>`.


.. _optional_dependencies:

Optional Dependencies
=====================
At this point, Manim should be fully working! However,
many Manim objects depend on ``LaTeX``, a math typesetting system.
How to install a texlive distribution varies across operating systems,
so check out the document that matches your operating system below:

.. toctree::
   :maxdepth: 1

   windows
   macos
   linux

To confirm all dependencies are installed and working, run::

  manim checkhealth


Working with Manim
==================
Head over to our :doc:`Quickstart Tutorial <../tutorials/quickstart>` to learn
how to make your own *Manimations*!
