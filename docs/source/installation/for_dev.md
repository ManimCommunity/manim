# Manim Installation For Developers



## Installing Poetry

Poetry can be easily installed in any OS by just running the below command.

```sh
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
```

If you are a windows developer and want to use Powershell, you can use the below command.

```powershell
(Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py -UseBasicParsing).Content | python
```

It will add it to path automatically.

```sh
poetry --version
```

>   You may need to restart your shell in order for the changes to take effect.

See the [docs on installation python poetry for more info](https://python-poetry.org/docs/)

## Installing System Dependencies

This section had to followed as in [installing-manim](win.rst). 
``` important:: Windows Users can Ignore Pycairo Installation.
```


Additional `git` has to be installed. For instructions see https://git-scm.com/

## Installing Manim using Poetry

1.  First, clone the Manim repo locally using git.

    ```sh
    git clone https://github.com/ManimCommunity/manim.git
    ```

    or

    ```sh
    git clone git@github.com:ManimCommunity/manim.git
    ```

2.  Open a Terminal/Powershell/Command Prompt and cd into the cloned directory.

    ```sh
    cd path/to/manim
    ```

    ``` note:: This path should contain a file called `pyproject.toml` if it doesn't contain it, you would need to go a level up.
    ```
3.  Use the below command to install dependencies.

    ```sh
    poetry env use <3.6/3.7/3.8>
    poetry install
    ```

    ``` note:: Poetry creates a virtual environment by default and no need to worry about it.
	```