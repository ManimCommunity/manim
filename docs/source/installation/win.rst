Windows
=======

There are two simple ways to download manim's dependencies, using the popular package
managers `Scoop <https://scoop.sh>`_ and `Chocolatey <https://chocolatey.org/install>`_

.. _scoop:

Installing using Scoop
**********************

First you need to install Scoop, which is a command-line installer for Windows
systems. Please refer to `this link
<https://scoop-docs.now.sh/docs/getting-started/Quick-Start.html>`_ for
instructions.

While a manifest for manim doesn't currently exist, it is sufficient to install the dependencies
via scoop and manim itself via ``pip``.

After installing Scoop, add these "buckets" as we'll need to install things from them later:

.. code-block:: powershell

      scoop bucket add extras
      scoop bucket add r-bucket https://github.com/cderv/r-bucket.git

FFmpeg installation
-------------------
1. Run ``scoop install ffmpeg``
2. Check whether ffmpeg has been properly installed by running ``ffmpeg``.

LaTeX Installation
------------------
There are two ways of installing a LaTeX distribution that will be covered here:

1. using MikTeX (approx. 2GB, takes a while to install, but provides "on-the-fly"
package installation)

2. using TinyTeX (approx. 500MB, installs quickly, but you have to install the required
packages manually)

Using Miktex
++++++++++++
Run ``scoop install latex`` and wait for the installer to complete its work. Most
packages required by manim should be installed, but if there are some that aren't
you'll get a popup to install them while rendering.

Using TinyTeX
+++++++++++++
Run ``scoop install tinytex`` and wait for the install to finish.
Now run the following command to install all necessary packages for using manim:

.. code-block:: powershell

      tlmgr install standalone everysel preview doublestroke ms setspace rsfs relsize ragged2e
      fundus-calligra microtype wasysym physics dvisvgm jknapltx wasy cm-super babel-english
      gnu-freefont mathastext cbfonts-fd

You can check whether they were installed properly by rendering an example scene which uses
:class:`~.Tex` or :class:`~.MathTex`.

.. _choco:

Installing using Chocolatey
***************************

First, you need to install Chocolatey, which is a package manager for Windows
systems.  Please refer to `this link <https://chocolatey.org/install>`_ for
instructions.

You can install manim very easily using chocolatey, by typing the following command.

.. code-block:: powershell

      choco install manimce


And then you can skip all the other steps and move to installing :ref:`latex-installation`.
Please see :doc:`troubleshooting` section for details about OSError.

FFmpeg installation
-------------------

1. To install ``ffmpeg`` and add it to your PATH, install `Chocolatey
   <https://chocolatey.org/>`_ and run ``choco install ffmpeg``.

2. You can check if you did it right by running ``refreshenv`` to update your
   environment variable and running ``ffmpeg``.


.. _latex-installation:

LaTeX Installation
------------------
You can install latex by either of the two methods below. MiKTex is very large (2 GB) while ManimLaTeX is small  (500Mb).

Using a custom distribution for Manim based on Texlive
++++++++++++++++++++++++++++++++++++++++++++++++++++++

This is the smallest latex distribution just enough to run Manim. Extra latex packages for fonts can be
installed using ``tlmgr``. See https://www.tug.org/texlive/tlmgr.html for more information.

1. Install chocolatey if you haven't already.

2. Run the following command

   .. code-block:: powershell

      choco install manim-latex

3. Finally, check whether it installed properly by running an example scene.

Using MiKTex
++++++++++++
1. Download the MiKTex installer from `this page
   <https://miktex.org/download>`_ and execute it.

   .. image:: ../_static/windows_miktex.png
       :align: center
       :width: 500px
       :alt: windows latex download page

2. You can check if you did it right by running ``refreshenv`` to update your
   environment variable and running ``latex``.

Certifying a clean install
**************************

After making sure you have a clean install following the instructions for each
dependency above, you can go back to :ref:`installing-manim`.
