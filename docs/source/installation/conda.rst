Conda
=====

Required Dependencies
---------------------

To create a conda environment, you must first install
`conda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html>`__
or `mamba <https://mamba.readthedocs.io/en/latest/installation.html>`__,
the two most popular conda clients.

After installing conda, you can create a new environment and install ``manim`` inside by running

.. code-block:: bash

   conda create -n my-manim-environment
   conda activate my-manim-environment
   conda install -c conda-forge manim

Since all dependencies (except LaTeX) are handled by conda, you don't need to worry
about needing to install additional dependencies.



Optional Dependencies
---------------------

In order to make use of Manim's interface to LaTeX to, for example, render
equations, LaTeX has to be installed as well. Note that this is an optional
dependency: if you don't intend to use LaTeX, you don't have to install it.

You can install LaTeX by following the optional dependencies steps
for :ref:`Windows <win-optional-dependencies>`,
:ref:`Linux <linux-optional-dependencies>` or
:ref:`macOS <macos-optional-dependencies>`.



Working with Manim
------------------

At this point, you should have a working installation of Manim, head
over to our :doc:`Quickstart Tutorial <../tutorials/quickstart>` to learn
how to make your own *Manimations*!
