For Developers
==============

This documentation is for developers who want to contribute to 
`ManimCommunity/manim <https://github.com/ManimCommunity/manim>`_.

Thank you for your interest in contributing! Please see our documentation on
:doc:`../contributing` to take the necessary steps before installing Manim as a
developer. This documentation assumes you have already taken the necessary
steps to clone your fork.

.. warning::

   If you have installed a non-developer version of manim, please uninstall
   it. This is to avoid any accidental usage of the non-developer version
   when developing and testing your local copy of the repository. This
   warning doesn't apply to users who use `poetry
   <https://python-poetry.org>`_ (chapter below.)

For Developers with Poetry (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Poetry is a tool for dependency management and packaging in Python. It allows to 
declare the libraries your project depends on and it will manage (install/update) them 
for you. In addition, Poetry provides a simple interface for managing virtual environments.

Please follow `Poetry's installation guidelines
<https://python-poetry.org/docs/master/#installation>`_
in order to get Poetry up and running on your system.

Installing System Dependencies
******************************

Please follow the instructions under :ref:`installing-manim` to install all
dependencies (e.g. ``LaTeX``, ``ffmpeg``, etc.). Afterwards, proceed with the
installation with Poetry.

.. important:: Windows users can skip the steps to install Pycairo.


.. _install-manim-poetry:

Installing Manim using Poetry
*****************************

#.  First start by cloning the `manim repository <https://github.com/manimcommunity/manim>`_ 
    (see the first few steps in :doc:`../contributing/development` for more details),
    then navigate into the freshly cloned directory in the terminal.

#.  Use the following command to install python dependencies. This will use the system python:

    .. code-block:: bash
	
         poetry install

    .. note:: The first time running this command, poetry will create and
              enter a virtual environment rooted in the current directory.
    
    For more information, you can visit the `poetry documentation
    <https://python-poetry.org/docs/managing-environments/>`_.

#. If you exit the virtual environment, you can reactivate the
   ``Poetry`` virtual environment with:

   .. code-block:: bash

      poetry shell
   
   If you only need to run a single command, use:

   .. code-block:: bash

      poetry run <your-command>

Now you are free to start developing!

Running the Tests Using Poetry
******************************

After completing :ref:`install-manim-poetry`, you can run manim's test suite
by activating a shell using ``poetry shell`` command and then running the
command ``pytest`` to run the tests.

.. code-block:: bash

   poetry shell
   pytest

.. important:: 

   You should always run the test suite before making a PR. See
   :doc:`../contributing` for details.


Code Formatting and Linting Using Poetry
****************************************

Once you are done with :ref:`install-manim-poetry`, you can run the code 
formatter ``black`` by activating entering the virtual environment:

.. code-block:: bash

   poetry shell
   black manim

Or alternatively, without entering the virtual environment: 

.. code-block:: bash

   poetry run black manim

For example, if you have written some new example and want to format it and see 
lint information use the commands below.

.. code-block:: bash

    poetry run black example_scenes

Similarly, you can see linting information for a given file, or directory, 
by the ``black`` command with ``pylint``.


For Developers with pip
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python3 -m pip install .

Please see :doc:`../contributing` for more details about contributing to Manim.
Since `pip` doesn't implement editable installations from our ``pyproject.toml``
