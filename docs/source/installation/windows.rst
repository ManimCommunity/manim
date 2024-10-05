Windows
=======
For installing Manim, please refer to the :doc:`installation instructions <../installation>`.

.. _win-optional-dependencies:

Optional Dependencies
---------------------

In order to make use of Manim's interface to LaTeX to, for example, render
equations, LaTeX has to be installed as well. Note that this is an optional
dependency: if you don't intend to use LaTeX, you don't have to install it.

For Windows, the recommended LaTeX distribution is
`MiKTeX <https://miktex.org/download>`__. You can install it by using the
installer from the linked MiKTeX site, or by using the package manager
of your choice (Chocolatey: ``choco install miktex.install``,
Scoop: ``scoop install latex``, Winget: ``winget install MiKTeX.MiKTeX``).

If you are concerned about disk space, there are some alternative,
smaller distributions of LaTeX.

**Using Chocolatey:** If you used Chocolatey to install manim or are already
a chocolatey user, then you can simply run ``choco install manim-latex``. It
is a dedicated package for Manim based on TinyTeX which contains all the
required packages that Manim interacts with.

**Manual Installation:**
You can also use `TinyTeX <https://yihui.org/tinytex/>`__ (Chocolatey: ``choco install tinytex``,
Scoop: first ``scoop bucket add r-bucket https://github.com/cderv/r-bucket.git``,
then ``scoop install tinytex``) alternative installation instructions can be found at their website.
Keep in mind that you will have to manage the LaTeX packages installed on your system yourself via ``tlmgr``.
Therefore we only recommend this option if you know what you are doing.

The full list of LaTeX packages which Manim interacts with in some way
(a subset might be sufficient for your particular application) are::

   amsmath babel-english cbfonts-fd cm-super count1to ctex doublestroke dvisvgm everysel
   fontspec frcursive fundus-calligra gnu-freefont jknapltx latex-bin
   mathastext microtype multitoc physics preview prelim2e ragged2e relsize rsfs
   setspace standalone tipa wasy wasysym xcolor xetex xkeyval



Working with Manim
------------------

At this point, you should have a working installation of Manim, head
over to our :doc:`Quickstart Tutorial <../tutorials/quickstart>` to learn
how to make your own *Manimations*!
