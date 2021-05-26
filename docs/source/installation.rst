Installation
============

Manim has a few dependencies that need to be installed before it can be used.
The following pages have instructions that are specific to your system. Once
you are done installing the dependencies, come back to this page to install
manim itself.

.. note::

   Before installing manim, you should understand that there are a few main versions of manim 
   today that are generally incompatible with each other.
   This documentation **only** covers the installation of the *community edition*; 
   trying to use instructions intended for other versions of manim or vice versa will likely result in failure.
   In particular, most video tutorials are outdated. For more information, please read :doc:`Differences between Manim Versions <installation/versions>`.


.. tip::

   In case that you want to try manim online without installation, open it in 
   `Binder <https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb>`_.


Installing dependencies
***********************

.. toctree::

   installation/versions
   installation/win
   installation/mac
   installation/linux
   installation/colab
   installation/troubleshooting
   installation/for_dev
   installation/plugins


.. _installing-manim:

Installing Manim
****************

Manim-Community runs on Python 3.6+. If you'd like to just use the library, you
can install it from PyPI via pip:

.. code-block:: bash

   pip install manim

You can replace ``pip`` with ``pip3`` if you need to in your system.

Alternatively, you can work with Manim using our Docker image that can be
found at `Docker Hub <https://hub.docker.com/r/manimcommunity/manim>`_.

Installation For Developers
***************************
If you want to contribute to manim, follow the :doc:`contributing` instructions.

Verifying installation
**********************
Please proceed to our :doc:`quickstart guide <tutorials/quickstart>` to run a simple file to test your installation.
If it did not work, please refer to our :doc:`troubleshooting guide <installation/troubleshooting>` for help.
