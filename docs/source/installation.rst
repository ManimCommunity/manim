Installation
============

Manim has a few dependencies that need to be installed before it.  The
following pages have instructions that are specific to your system.  Once you
are done installing the dependencies, come back to this page to install manim
itself.

Installing dependencies
***********************

.. toctree::

   installation/win
   installation/mac
   installation/linux
   installation/troubleshooting


.. _installing-manim:

Installing Manim
****************

Manim-Community runs on Python 3.6+. If you'd like to just use the library, you
can install it from PyPI via pip:

.. code-block:: bash

   pip install manimce

You can replace ``pip`` with ``pip3`` is you need to in your system.

If you'd like to contribute to and/or help develop ``manim-community``, you can
clone this repository to your local device.  To do this, first make sure you
have ``git`` installed. Then, clone this repo by executing either

.. code-block:: bash

   git clone git@github.com:ManimCommunity/manim.git

or

.. code-block:: bash

   git clone https://github.com/ManimCommunity/manim.git

depending on whether you want to use HTTPS or SSH.  Finally, after having
cloned this repo, run the following:

.. code-block:: bash

   python3 -m pip install -r requirements.txt
