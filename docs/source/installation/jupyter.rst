Jupyter Notebooks
=================


Binder
------

`Binder <https://mybinder.readthedocs.io/en/latest/>`__ is an online
platform that hosts shareable and customizable computing environments
in the form of Jupyter notebooks. Manim ships with a built-in ``%%manim``
Jupyter magic command which makes it easy to use in these notebooks.

To see an example for such an environment, visit our interactive
tutorial over at https://try.manim.community/.

It is relatively straightforward to prepare your own notebooks in
a way that allows them to be shared interactively via Binder as well:

#. First, prepare a directory containing one or multiple notebooks
   which you would like to share in an interactive environment. You
   can create these notebooks by using Jupyter notebooks with a
   local installation of Manim, or also by working in our pre-existing
   `interactive tutorial environment <https://try.manim.community/>`__.
#. In the same directory containing your notebooks, add a
   file named ``Dockerfile`` with the following content:

   .. code-block:: dockerfile

      FROM docker.io/manimcommunity/manim:v0.9.0

      COPY --chown=manimuser:manimuser . /manim

   Don't forget to change the version tag ``v0.9.0`` to the version you
   were working with locally when creating your notebooks.
#. Make the directory with your worksheets and the ``Dockerfile``
   available to the public (and in particular: to Binder!). There are
   `several different options to do so
   <https://mybinder.readthedocs.io/en/latest/introduction.html#how-can-i-prepare-a-repository-for-binder>`__,
   within the community we usually work with GitHub
   repositories or gists.
#. Once your material is publicly available, visit
   https://mybinder.org and follow the instructions there to
   generate an interactive environment for your worksheets.

.. hint::

   The repository containing our `interactive tutorial
   <https://try.manim.community>`__ can be found at
   https://github.com/ManimCommunity/jupyter_examples.


Google Colaboratory
-------------------

It is also possible to install Manim in a
`Google Colaboratory <https://colab.research.google.com/>`__ environment.
In contrast to Binder, where you can customize and prepare the environment
beforehand (such that Manim is already installed and ready to be used), you
will have to take care of that every time you start
a new notebook in Google Colab. Fortunately, this
is not particularly difficult.

After creating a new notebook, paste the following code block in a cell,
then execute it.

.. code-block::

   !sudo apt update
   !sudo apt install libcairo2-dev ffmpeg \
       texlive texlive-latex-extra texlive-fonts-extra \
       texlive-latex-recommended texlive-science \
       tipa libpango1.0-dev
   !pip install manim
   !pip install IPython --upgrade

You should start to see Colab installing all the dependencies specified
in these commands. After the execution has completed, you will be prompted
to restart the runtime. Click the "restart runtime" button at the bottom of
the cell output. You are now ready to use Manim in Colab!

To check that everything works as expected, first import Manim by running

.. code-block::

   from manim import *

in a new code cell. Then create another cell containing the
following code::

   %%manim -qm -v WARNING SquareToCircle

   class SquareToCircle(Scene):
      def construct(self):
         square = Square()
         circle = Circle()
         circle.set_fill(PINK, opacity=0.5)
         self.play(Create(square))
         self.play(Transform(square, circle))
         self.wait()

Upon running this cell, a short animation transforming a square
into a circle should be rendered and displayed.
