Conda
=====

Required Dependencies
---------------------

There are several package managers that work with conda packages,
namely `conda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html>`__,
`mamba <https://mamba.readthedocs.io>`__ and `pixi <https://pixi.sh>`__.

After installing your package manager, you can create a new environment and install ``manim`` inside by running

.. code-block:: bash

   # using conda
   conda create -n my-manim-environment
   conda activate my-manim-environment
   conda install -c conda-forge manim
   # using pixi
   pixi init
   pixi add manim

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
