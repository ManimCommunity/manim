Google Colaboratory
===================

You can install Colab by running the following in a cell:

.. code-block::

   !sudo apt install libcairo2-dev ffmpeg texlive texlive-latex-extra texlive-fonts-extra texlive-latex-recommended texlive-science tipa

Then, run this in a separate cell:

.. code-block::
   !sudo apt install libpango1.0-dev
   !pip install manim
   !pip install IPython --upgrade

After the execution has completed, you will be prompted to restart the runtime. Click the "restart runtime" button at the bottom of the cell output.

.. note:: Due to a bug in Colab, the installation has to be completed in two separate cells.