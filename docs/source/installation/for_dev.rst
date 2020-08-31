Manim Installation For Developers
=================================

	This documentation is for developers who wats to contribute to ``ManimCommunity/manim``.

Installing Poetry
*****************

Poetry can be easily installed in any OS by just running the below command.

.. code-block:: bash
	
	curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python


If you are a Windows developer and want to use PowerShell, you can use the below command.

.. code-block:: powershell
	
	(Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py -UseBasicParsing).Content | python

It will add it to path automatically.

.. code-block:: bash

	poetry --version


.. note:: You may need to restart your shell in order for the changes to take effect.

See the `docs on installation python poetry for more information
<https://python-poetry.org/docs/>`_.

Installing System Dependencies
******************************

This section had to followed as in :ref:`installing-manim`.

.. important:: Windows Users can Ignore Pycairo Installation.


Additional `git` has to be installed. For instructions see `Documentation
<https://git-scm.com/>`_.

.. _Install Manim Poetry:
Installing Manim using Poetry
*****************************

1.  First, clone the Manim repo locally using git.

    .. code-block:: bash
		
		git clone https://github.com/ManimCommunity/manim.git

    or

    .. code-block:: bash
		
		git clone git@github.com:ManimCommunity/manim.git

2.  Open a Terminal/Powershell/Command Prompt and cd into the cloned directory.

    .. code-block:: bash
		
		cd path/to/manim
    

    .. note:: This path should contain a file called `pyproject.toml` if it doesn't contain it, you would need to go a level up.

3.  Use the below command to install dependencies. This will use default python version installed.

    .. code-block:: bash
	
         poetry install


    .. note:: Poetry creates a virtual environment by default and no need to worry about it.
    You can select the required python version using 

    .. code-block:: bash
	
         poetry env use <python version you need>

    For example you can use for python 3.7.

    .. code-block:: bash
	
         poetry env use 3.7
    For more information about this you can visit the `docs
    <https://python-poetry.org/docs/managing-environments/>`_.

4. Now you can activate the virtual environment, ``Poetry`` had created in the previous step, use the below command.

  .. code-block:: bash

       poetry shell
    
  Or if you want to run a single command use

  .. code-block:: bash

       poetry run manim -h


Running the Tests Using Poetry
******************************

After you have :ref:`Install Manim Poetry`, you can run the tests using by activating a shell using ``poetry shell`` command and then running the command ``pytest`` to run the tests. 

.. important:: You should run this test before making a PR.


Code Formatting Using Poetry
****************************

After you have :ref:`Install Manim Poetry`, you can run the code formatter ``black`` by activating a shell using ``poetry shell`` command and then running the command ``black manim`` to run the tests. Or alternatively just use the command ``poetry run black manim``. 

.. note:: Here ``manim`` used in command ``black manim`` or ``poetry run black manim`` is folder which ``black`` formats the code.

For example, if you have written some new example and want to format it use the below command

.. code-block:: bash

    poetry run black example_scenes


