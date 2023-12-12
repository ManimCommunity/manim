Windows
=======

The easiest way of installing Manim and its dependencies is by using a
package manager like `Chocolatey <https://chocolatey.org/>`__
or `Scoop <https://scoop.sh>`__, especially if you need optional dependecies
like LaTeX support.

If you choose to use one of the package managers, please follow
their installation instructions
(`for Chocolatey <https://chocolatey.org/install#install-step2>`__,
`for Scoop <https://scoop-docs.now.sh/docs/getting-started/Quick-Start.html>`__)
to make one of them available on your system.


Required Dependencies
---------------------

Manim requires a recent version of Python (3.9 or above)
in order to work.

Chocolatey
**********

Manim can be installed via Chocolatey simply by running:

.. code-block:: powershell

   choco install manimce

That's it, no further steps required. You can continue with installing
the :ref:`optional dependencies <win-optional-dependencies>` below.

Pip
***

As mentioned above, Manim needs a reasonably recent version of
Python 3 (3.9 or above).

**Python:** Head over to https://www.python.org, download an installer
for a recent version of Python, and follow its instructions to get Python
installed on your system.

.. note::

   We have received reports of problems caused by using the version of
   Python that can be installed from the Windows Store. At this point,
   we recommend staying away from the Windows Store version. Instead,
   install Python directly from the
   `official website <https://www.python.org>`__.

Then, Manim can be installed via Pip simply by running:

.. code-block:: powershell

   python -m pip install manim

Manim should now be installed on your system. Continue reading
the :ref:`optional dependencies <win-optional-dependencies>` section
below.


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

   amsmath babel-english cbfonts-fd cm-super ctex doublestroke dvisvgm everysel
   fontspec frcursive fundus-calligra gnu-freefont jknapltx latex-bin
   mathastext microtype ms physics preview ragged2e relsize rsfs
   setspace standalone tipa wasy wasysym xcolor xetex xkeyval



Working with Manim
------------------

At this point, you should have a working installation of Manim, head
over to our :doc:`Quickstart Tutorial <../tutorials/quickstart>` to learn
how to make your own *Manimations*!
