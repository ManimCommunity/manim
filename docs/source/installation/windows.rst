Windows
=======

The easiest way of installing Manim and its dependencies is by using a
package manager like `Chocolatey <https://chocolatey.org/>`__
or `Scoop <https://scoop.sh>`__. If you are not afraid of editing
your System's ``PATH``, a manual installation is also possible.
In fact, if you already have an existing Python
installation (3.7-3.10), it might be the easiest way to get
everything up and running.

If you choose to use one of the package managers, please follow
their installation instructions
(`for Chocolatey <https://chocolatey.org/install#install-step2>`__,
`for Scoop <https://scoop-docs.now.sh/docs/getting-started/Quick-Start.html>`__)
to make one of them available on your system.


Required Dependencies
---------------------

Manim requires a recent version of Python (3.7–3.10) and ``ffmpeg``
in order to work.

Chocolatey
**********

Manim can be installed via Chocolatey simply by running:

.. code-block:: powershell

   choco install manimce

That's it, no further steps required. You can continue with installing
the :ref:`optional dependencies <win-optional-dependencies>` below.

Scoop
*****

While there is no recipe for installing Manim with Scoop directly,
you can install all requirements by running:

.. code-block:: powershell

   scoop install python ffmpeg

and then Manim can be installed by running:

.. code-block:: powershell

   python -m pip install manim

Manim should now be installed on your system. Continue reading
the :ref:`optional dependencies <win-optional-dependencies>` section
below.

Manual Installation
*******************

As mentioned above, Manim needs a reasonably recent version of
Python 3 (3.7–3.10) and FFmpeg.

**Python:** Head over to https://www.python.org, download an installer
for Python (3.7–3.10), and follow its instructions to get Python
installed on your system.

.. note::

   We have received reports of problems caused by using the version of
   Python that can be installed from the Windows Store. At this point,
   we recommend staying away from the Windows Store version. Instead,
   install Python directly from the
   `official website <https://www.python.org>`__.

**FFmpeg:** In order to install FFmpeg, you can get a
pre-compiled and ready-to-use version from one of the resources
linked at https://ffmpeg.org/download.html#build-windows, such as
`the version available here
<https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.7z>`__
(recommended), or if you know exactly what you are doing
you can alternatively get the source code
from https://ffmpeg.org/download.html and compile it yourself.


After downloading the pre-compiled archive,
`unzip it <https://www.7-zip.org>`__ and, if you like, move the
extracted directory to some more permanent place (e.g.,
``C:\Program Files\``). Next, edit the ``PATH`` environment variable:
first, visit ``Control Panel`` > ``System`` > ``System settings`` >
``Environment Variables``, then add the full path to the ``bin``
directory inside of the (moved) ffmpeg directory to the
``PATH`` variable. Finally, save your changes and exit.

If you now open a new command line prompt (or PowerShell) and
run ``ffmpeg``, the command should be recognized.

At this point, you have all the required dependencies and can now
install Manim via

.. code-block:: powershell

   python -m pip install manim


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
Scoop: ``scoop install latex``).

If you are concerned about disk space, there are some alternative,
smaller distributions of LaTeX like
`TinyTeX <https://yihui.org/tinytex/>`__ (Chocolatey: ``choco install tinytex``,
Scoop: first ``scoop bucket add r-bucket https://github.com/cderv/r-bucket.git``,
then ``scoop install tinytex``). In this case, you will have to manage the
LaTeX packages installed on your system yourself via ``tlmgr``. Therefore we only
recommend this option if you know what you are doing. The full list
of LaTeX packages which Manim interacts with in some way (a subset might
be sufficient for your particular application) is::

   amsmath babel-english cbfonts-fd cm-super ctex doublestroke dvisvgm everysel
   fontspec frcursive fundus-calligra gnu-freefont jknapltx latex-bin
   mathastext microtype ms physics preview ragged2e relsize rsfs
   setspace standalone tipa wasy wasysym xcolor xetex xkeyval

.. note::

   For Chocolatey there is a dedicated ``manim-latex`` package providing a
   small LaTeX distribution based on TinyTeX which contains these packages;
   if you use Chocolatey you can get it with ``choco install manim-latex``.


Working with Manim
------------------

At this point, you should have a working installation of Manim, head
over to our :doc:`Quickstart Tutorial <../tutorials/quickstart>` to learn
how to make your own *Manimations*!
