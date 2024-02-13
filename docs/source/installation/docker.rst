Docker
======

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


Instead of using the "throwaway container" approach outlined
above, you can also create a named container that you can
modify to your liking. First, run

.. code-block:: sh

   docker run -it --name my-manim-container -v "/full/path/to/your/directory:/manim" manimcommunity/manim bash


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

Another alternative to using the Docker image is to spin up a
local JupyterLab instance. To do that, simply run

.. code-block:: sh

   docker run -it -p 8888:8888 manimcommunity/manim jupyter lab --ip=0.0.0.0

and then follow the instructions in the terminal.
