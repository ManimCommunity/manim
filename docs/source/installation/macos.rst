MacOS
=====
For installing Manim, please refer to the :doc:`installation instructions <../installation>`.

.. _macos-optional-dependencies:

Optional Dependencies
---------------------

In order to make use of Manim's interface to LaTeX for, e.g., rendering
equations, LaTeX has to be installed as well. Note that this is an optional
dependency: if you don't intend to use LaTeX, you don't have to install it.

For macOS, the recommended LaTeX distribution is
`MacTeX <http://www.tug.org/mactex/>`__. You can install it by following
the instructions from the link, or alternatively also via Homebrew by
running:

.. code-block:: bash

   brew install --cask mactex-no-gui

.. warning::

   MacTeX is a *full* LaTeX distribution and will require more than 4GB of
   disk space. If this is an issue for you, consider installing a smaller
   distribution like
   `BasicTeX <http://www.tug.org/mactex/morepackages.html>`__.

Should you choose to work with some partial TeX distribution, the full list
of LaTeX packages which Manim interacts with in some way (a subset might
be sufficient for your particular application) is::

   amsmath babel-english cbfonts-fd cm-super count1to ctex doublestroke dvisvgm everysel
   fontspec frcursive fundus-calligra gnu-freefont jknapltx latex-bin
   mathastext microtype multitoc physics preview prelim2e ragged2e relsize rsfs
   setspace standalone tipa wasy wasysym xcolor xetex xkeyval


Working with Manim
------------------

At this point, you should have a working installation of Manim. Head
over to our :doc:`Quickstart Tutorial <../tutorials/quickstart>` to learn
how to make your own *Manimations*!
