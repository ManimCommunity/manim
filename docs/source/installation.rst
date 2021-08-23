Installation
============

There are a few different options for making Manim available to you:

#. :ref:`Installing Manim to your system's Python <local-installation>`
#. :ref:`Using Manim via Docker <docker-installation>`
#. :ref:`Interactive Jupyter notebooks via Binder / Google Colab 
   <interactive-online>`

Depending on your use case, different options are recommended: if you
just want to play around with Manim for a bit, interactive in-browser
worksheets are a really simple way of exploring the library as they
require virtually no additional installation. Head over to
https://try.manim.community to give our interactive tutorial a try.

Otherwise, if you intend to use Manim to work on an animation project,
we recommend installing the library locally (either to your system's
Python, or via Docker).

.. warning:: 

   Note that there are several different versions of Manim. The
   instructions on this website are **only** for the *community edition*.
   Find out more about the :doc:`differences between Manim
   versions <installation/versions>` in case you are unsure which
   version you should install.


.. _local-installation:

Installing Manim locally
************************

Manim is a Python library, and it can be
`installed via pip <https://pypi.org/project/manim/>`__. However,
in order to enable all features of Manim, some additional system
dependencies need to be installed first. The following pages have
operating system specific instructions for you to follow.

.. hint::

   Depending on your particular setup, the installation process
   might be slightly different. Make sure that you have tried to
   follow the steps on the following pages carefully, but in case
   you hit a wall we are happy to help: either `join our Discord
   <https://www.manim.community/discord/>`__, or start a new
   Discussion `directly on GitHub 
   <https://github.com/ManimCommunity/manim/discussions>`__.

.. toctree::
   :maxdepth: 2

   installation/win
   installation/mac
   installation/linux

Once Manim is installed locally, you can proceed to our
:doc:`quickstart guide <tutorials/quickstart>` which walks you
through rendering a first simple scene.

As mentioned above, do not worry if there are errors or other
problems: consult our :doc:`troubleshooting 
guide <installation/troubleshooting>` for help, or get in touch
with the community via `GitHub discussions 
<https://github.com/ManimCommunity/manim/discussions>`__ or 
`Discord <https://www.manim.community/discord/>`__.
   


.. _docker-installation:

Using Manim via Docker
**********************

`Docker <https://www.docker.com>`__ is a virtualization tool that
allows to distribute encapsulated software environments (containers).

The community maintains a docker image, which can be found
`on DockerHub <https://hub.docker.com/r/manimcommunity/manim>`__. 
For our image ``manimcommunity/manim``, there are the following tags:

- ``latest``: the most recent version corresponding 
  to `the main branch <https://github.com/ManimCommunity/manim>`__,
- ``stable``: the latest released version (according to 
  `the releases page <https://github.com/ManimCommunity/manim/releases>`__),
- ``vX.Y.Z``: any particular released version (according to 
  `the releases page <https://github.com/ManimCommunity/manim/releases>`__).

.. note::

   When using Manim's CLI within a Docker container, some flags like 
   ``-p`` (preview file) and ``-f`` (show output file in the file browser)
   are not supported.


Basic usage of the Docker container
-----------------------------------

Assuming that you can access the docker installation on your system
from a terminal (bash / PowerShell) via ``docker``, you can 
render a scene ``CircleToSquare`` in a file `test_scenes.py`
with the following command.

.. code-block:: bash

   docker run --rm -it -v "/full/path/to/your/directory:/manim" manimcommunity/manim manim -qm test_scenes.py CircleToSquare

.. tip::

   For Linux users there might be permission problems when letting the
   user in the container write to the mounted volume.
   Add ``--user="$(id -u):$(id -g)"`` to the ``docker`` CLI arguments
   to prevent the creation of output files not belonging to your user.


Instead of using the "throwaway container" approach sketched
above, you can also create a named container that you can
modify to your liking. First, run

.. code-block:: sh

   docker run -it --name my-manim-container -v "/full/path/to/your/directory:/manim" manimcommunity/manim /bin/bash   


to obtain an interactive shell inside your container allowing you
to, e.g., install further dependencies (like texlive packages using
``tlmgr``). Exit the container as soon as you are satisfied. Then,
before using it, start the container by running

.. code-block:: sh

   docker start my-manim-container

which starts the container in the background. Then, to render
a scene ``CircleToSquare`` in a file ``test_scenes.py``, run

.. code-block:: sh

   docker exec -it my-manim-container manim -qm test_scenes.py CircleToSquare


Running JupyterLab via Docker
-----------------------------

Another alternative is to use the Docker image to spin up a
local JupyterLab instance. To do that, simply run

.. code-block:: sh

   docker run -it -p 8888:8888 manimcommunity/manim jupyter lab --ip=0.0.0.0

and then follow the instructions in the terminal.


.. _interactive-online:

Interactive Jupyter notebooks for your browser
**********************************************

Binder
------

`Binder <https://mybinder.readthedocs.io/en/latest/>`__ is a online
platform that hosts shareable and customizable computing environments
via Jupyter notebooks. Manim ships with a built-in ``%%manim`` Jupyter
magic command which makes it easy to use in these notebooks.

As an example for such an environment, visit our interactive
tutorial over at https://try.manim.community/.

It is relatively straightforward to prepare your own notebooks in
a way that allows them to be shared interactively via Binder as well:

#. First, prepare a directory containing one or multiple notebooks
   which you would like to share in an interactive environment. You
   can create these notebooks by using Jupyter notebooks with a
   local installation of Manim, or also by working in our pre-existing
   `interactive tutorial environment <https://try.manim.community/>`__.
#. In the same directory containing your notebooks, you need to add a
   file named ``Dockerfile`` with the following content:

   .. code-block:: dockerfile

      FROM manimcommunity/manim:v0.9.0

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
In contrast to Binder, where we can customize and prepare the environment
for you (such that Manim is already installed and ready to be used), you
will have to take care of that yourself in Google Colab. Fortunately, this
is not particularly difficult.

After creating a new notebook, paste the following code block in a cell,
and then execute the cell.

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

in a new code cell, and then create another cell containing the
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


Installation for developers
***************************

In order to change code in the library, it is recommended to
install Manim in a different way. Please follow the instructions
in our :doc:`contribution guide <contributing>` if you are
interested in that.



