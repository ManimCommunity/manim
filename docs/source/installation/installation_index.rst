Installation
============

Dependencies
************

Before installing manim, there are some additional dependencies that you must
have installed: Cairo, FFmpeg, Sox (optional, for sound), and LaTeX (optional,
but strongly recommended).  The following documents contain instructions to
install all of these dependencies, depending on your operating system.

.. toctree::
   :maxdepth: 1

   installation_win
   installation_linux
   installation_mac

After following the instructions therein, come back to this document to install
manim, following the next section.  For any problems encountered when
installing the dependencies or manim itself, visit the :doc:`troubleshooting`
page.

.. This is here so that sphinx doesn't complain about troubleshooting.rst not
   being included in any toctree
.. toctree::
   :hidden:

   troubleshooting


Installing manim
****************

After certifying that you have you have a clean install, you can now install
manim itself.

1. Manim-Community runs on Python 3.6+.  It is common to have more than one
   python version installed.  To check your version, execute the following
   command:

   .. code-block:: bash

      python --version

   If it shows your version is at least 3.6, then go to the next step.  If your
   python version is less than 3.6, try the following command instead.

   .. code-block:: bash

      python3 --version

   If it shows your version is at least 3.6, then go to the next step.  If not,
   you need to update your python installation.


   .. note:: The remaining steps in this guide assume that the command
             ``python`` points to the python installation with the appropriate
             version (3.6+).  If you are using the ``python3`` command instead,
             replace ``python`` by ``python3`` in the remaining instructions.
             In this case, you also have to replace the ``pip`` command by
             ``pip3``.

2. If you'd like to just use the library, you can install it from PyPI via pip:

   .. code-block:: bash

      pip install manimce

3. However, if you'd like to contribute to and/or help develop manim-community,
   you have to clone this repository to your local device.  To do this, first
   make sure you have ``git`` installed.  Then, clone this repo by executing
   either of the following commands, depending on whether you want to use HTTPS
   or SSH (do not execute both commands!)

   .. code-block:: bash

      git clone git@github.com:ManimCommunity/manim.git
      git clone https://github.com/ManimCommunity/manim.git

4. Finally, after having cloned this repo, go to the new ``manim/`` folder and
   run the following:

   .. code-block:: bash

      pip install -r requirements.txt
