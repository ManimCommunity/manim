Troubleshooting
===============

Version incompatibility
***********************

Confusion and conflict between versions is by far the most common reason
for installation failures. Some signs and errors resulting from this are
as follows: 

- ``There are no scenes in that module`` 
- ``ModuleNotFoundError: No module named 'manim'`` 
- ``ModuleNotFoundError: No module named 'manimlib'`` 
- You followed any tutorial created before October 2020 (because the community edition did not exist before then) 
- You cloned a repository on GitHub (installation of the community version for normal use does not require the cloning of any repository) 
- Different import statements (explained below) 
- You used documentation for multiple versions (such as the readme for 3b1b/manim and this documentation)

.. note::
   As this is the documentation for the community version, we can
   only help with the installation of this library. If you would like to
   install other versions of manim, please refer to their documentation.

Identifying files written for a different version of manim
----------------------------------------------------------

There are some distinctive features of different versions of manim that
can help in identifying what version of manim files are written for:

+--------------+-------------------------+----------------------------+-----------------------------------------+
| Feature      | ManimCE (this version)  | ManimGL                    | ManimCairo                              |
+==============+=========================+============================+=========================================+
| Import       | ``from manim import *`` | ``from manimlib import *`` | ``from manimlib.imports import *``      |
| statement    |                         |                            |                                         |
+--------------+-------------------------+----------------------------+-----------------------------------------+

If you are a beginner, you should only attempt to run files written for
your version. Files written for a different version of manim will
generally not work without some modification.

Identifying the version you are running
---------------------------------------

The community edition of manim should always state `Manim Community <version_number>` 
as its first line of any command you run.

Identifying and removing conflicting versions of manim
------------------------------------------------------

Within the system or environment you are using to run manim, run the
following command in the terminal:

.. code-block:: bash

   pip list

The correct package for the community edition is simply ``manim``. If
you do not see this package listed, please refer back to our
installation guide to install it. If you see ``manimlib`` or ``manimce``
(actually an old version of the community edition), you should remove
them with:

.. code-block:: bash

   pip uninstall <package>


If you have cloned a repository from GitHub, you should either remove it
or run manim outside that folder.

Other errors
************

``pip install manim`` fails when installing manimpango?
-------------------------------------------------------
Most likely this means that pip was not able to use our pre-built wheels
of ``manimpango``. Let us know (via our `Discord <https://www.manim.community/discord/>`_
or by opening a
`new issue on GitHub <https://github.com/ManimCommunity/ManimPango/issues/new>`_)
which architecture you would like to see supported, and we'll see what we
can do about it.

To fix errors when installing ``manimpango``, you need to make sure you
have all the necessary build requirements. Check out the detailed
instructions given in
`the BUILDING section <https://github.com/ManimCommunity/ManimPango#BUILDING>`_
of the corresponding `GitHub repository <https://github.com/ManimCommunity/ManimPango>`_.


(Windows) OSError: dlopen() failed to load a library: pango?
------------------------------------------------------------

This should be fixed in Manim's latest version, update
using ``pip install --upgrade manim``.



Some letters are missing from Text/Tex output?
------------------------------------------------------------

If you have recently installed TeX you may need to build the fonts it
uses. Which can be done by running:

.. code-block:: bash

  fmtutil -sys --all


.. _dvisvgm-troubleshoot:

Installation does not support converting PDF to SVG?
----------------------------------------------------

First, make sure your ``dvisvgm`` version is at least 2.4:

.. code-block:: bash

  dvisvgm --version


If you do not know how to update ``dvisvgm``, please refer to your operating system's documentation.

Second, check whether your ``dvisvgm`` supports PostScript specials. This is 
needed to convert from PDF to SVG.

.. code-block:: bash

  dvisvgm -l


If the output to this command does **not** contain ``ps  dvips PostScript specials``, 
this is a bad sign. In this case, run

.. code-block:: bash

  dvisvgm -h


If the output does **not** contain ``--libgs=filename``, this means your 
``dvisvgm`` does not currently support PostScript. You must get another binary.

If, however, ``--libgs=filename`` appears in the help, that means that your 
``dvisvgm`` needs the Ghostscript library to support PostScript. Search for 
``libgs.so`` (on Linux, probably in ``/usr/local/lib`` or ``/usr/lib``) or 
``gsdll32.dll`` (on 32-bit Windows, probably in ``C:\windows\system32``) or 
``gsdll64.dll`` (on 64-bit Windows, probably in ``c:\windows\system32`` -- yes 
32) or ``libgsl.dylib`` (on Mac OS, probably in ``/usr/local/lib`` or 
``/opt/local/lib``). Please look carefully, as the file might be located 
elsewhere, e.g. in the directory where Ghostscript is installed.

As soon as you have found the library, try (on Mac OS or Linux)

.. code-block:: bash

  export LIBGS=<path to your library including the file name>
  dvisvgm -l

or (on Windows)

.. code-block:: bat

  set LIBGS=<path to your library including the file name>
  dvisvgm -l


You should now see ``ps    dvips PostScript specials`` in the output. Refer to 
your operating system's documentation to find out how you can set or export the 
environment variable ``LIBGS`` automatically whenever you open a shell.

As a last check, you can run

.. code-block:: bash

  dvisvgm -V1

while still having ``LIBGS`` set to the correct path, of course. If ``dvisvgm`` 
can find your Ghostscript installation, it will be shown in the output together 
with the version number.

If you do not have the necessary library on your system, please refer to your 
operating system's documentation to find out where you can get it and how you 
have to install it.

If you are unable to solve your problem, check out the `dvisvgm FAQ <https://dvisvgm.de/FAQ/>`_.
