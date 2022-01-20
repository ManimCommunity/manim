Linux
=====

The installation instructions depend on your particular operating
system and package manager. If you happen to know exactly what you are doing,
you can also simply ensure that your system has:

- a reasonably recent version of Python 3 (3.7–3.10),
- with working Cairo bindings in the form of
  `pycairo <https://cairographics.org/pycairo/>`__,
- FFmpeg accessible from the command line as ``ffmpeg``,
- and `Pango <https://pango.gnome.org>`__ headers.

Then, installing Manim is just a matter of running:

.. code-block:: bash

   pip3 install manim

.. note::

   In light of the current efforts of migrating to rendering via OpenGL,
   this list might be incomplete. Please `let us know
   <https://github.com/ManimCommunity/manim/issues/new/choose>` if you
   ran into missing dependencies while installing.

In any case, we have also compiled instructions for several common
combinations of operating systems and package managers below.

Required Dependencies
---------------------

apt – Ubuntu / Mint / Debian
****************************

To first update your sources, and then install Cairo, Pango, and FFmpeg
simply run:

.. code-block:: bash

   sudo apt update
   sudo apt install libcairo2-dev libpango1.0-dev ffmpeg

If you don't have python3-pip installed, install it via:

.. code-block:: bash

   sudo apt install python3-pip

Then, to install Manim, run:

.. code-block:: bash

   pip3 install manim

Continue by reading the :ref:`optional dependencies <linux-optional-dependencies>`
section.

dnf – Fedora / CentOS / RHEL
****************************

To install Cairo and Pango:

.. code-block:: bash

  sudo dnf install cairo-devel pango-devel

In order to successfully build the ``pycairo`` wheel, you will also
need the Python development headers:

.. code-block:: bash

   sudo dnf install python3-devel

FFmpeg is only available via the RPMfusion repository which you have to
configure first – please consult https://rpmfusion.org/Configuration/ for
instructions. Then, install FFmpeg:

.. code-block:: bash

   sudo dnf install ffmpeg

At this point you have all required dependencies and can install
Manim by running:

.. code-block:: bash

   pip3 install manim

Continue by reading the :ref:`optional dependencies <linux-optional-dependencies>`
section.

pacman – Arch / Manjaro
***********************

.. tip::

   Thanks to *groctel*, there is a `dedicated Manim package
   on the AUR! <https://aur.archlinux.org/packages/manim/>`

If you don't want to use the packaged version from AUR, here is what
you need to do manually: Update your package sources, then install
Cairo, Pango, and FFmpeg:

.. code-block:: bash

   sudo pacman -Syu
   sudo pacman -S cairo pango ffmpeg

If you don't have ``python-pip`` installed, get it by running:

.. code-block:: bash

   sudo pacman -S python-pip

then simply install Manim via:

.. code-block:: bash

   pip3 install manim


Continue by reading the :ref:`optional dependencies <linux-optional-dependencies>`
section.


.. _linux-optional-dependencies:

Optional Dependencies
---------------------

In order to make use of Manim's interface to LaTeX for, e.g., rendering
equations, LaTeX has to be installed as well. Note that this is an optional
dependency: if you don't intend to use LaTeX, you don't have to install it.

You can use whichever LaTeX distribution you like or whichever is easiest
to install with your package manager. Usually,
`TeX Live <https://www.tug.org/texlive/>`__ is a good candidate if you don't
care too much about disk space.

Should you choose to work with some smaller TeX distribution like
`TinyTeX <https://yihui.org/tinytex/>`__ , the full list
of LaTeX packages which Manim interacts with in some way (a subset might
be sufficient for your particular application) is::

   collection-basic amsmath babel-english cbfonts-fd cm-super ctex doublestroke
   dvisvgm everysel fontspec frcursive fundus-calligra gnu-freefont jknapltx
   latex-bin mathastext microtype ms physics preview ragged2e relsize rsfs
   setspace standalone tipa wasy wasysym xcolor xetex xkeyval
