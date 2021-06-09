Google Colaboratory
===================

You can install Colab by running the following in a cell:

.. code-block::

   !sudo apt update
   !sudo apt install libcairo2-dev ffmpeg texlive texlive-latex-extra texlive-fonts-extra texlive-latex-recommended texlive-science tipa libpango1.0-dev
   !pip install manim
   !pip install IPython --upgrade

After the execution has completed, you will be prompted to restart the runtime. Click the "restart runtime" button at the bottom of the cell output.

In the next cell, import manim like so:

.. code-block::

   from manim import *

Try and run the following code in the next cell to confirm installation.

.. code-block::

   %%manim SquareToCircle
   
   class SquareToCircle(Scene):
       def construct(self):
           circle = Circle()  
           circle.set_fill(PINK, opacity=0.5)  
           self.play(Create(circle))
