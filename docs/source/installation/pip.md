# Installing Manim from PyPI

The standard way of installing Manim is by using
Python's package manager `pip` to install the latest
release from [PyPI](https://pypi.org/project/manim/).

To make it easier for you to follow best practices when it
comes to setting up a Python project for your Manim animations,
we strongly recommend using a tool for managing Python environments
and dependencies. In particular,
[we recommend using `uv`](https://docs.astral.sh/uv/#getting-started).

For the two main ways of installing Manim described below, we assume
that `uv` is available; we think it is particularly helpful if you are
new to Python or programming in general. It is not a hard requirement
whatsoever; if you know what you are doing you can just use `pip` to
install Manim directly.

:::::{hint}

One way to install `uv` is via the dedicated console installer supporting
all large operating systems. Simply paste the following snippet into 
your terminal / PowerShell -- or
[consult `uv`'s documentation](https://docs.astral.sh/uv/#getting-started)
for alternative ways to install the tool.

::::{tab-set}
:::{tab-item} MacOS and Linux
```console
$ curl -LsSf https://astral.sh/uv/install.sh | sh
```
:::
:::{tab-item} Windows
```console
$ powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
:::
::::

:::::

## Different Installation Methods

**Preliminaries:** check that Python version is available, otherwise install
one via uv.

### Recommended: Installation in project environment (using `uv`)

For this (strongly recommended) method we create a project directory holding
our Manim scene files. Manim is installed in a local virtual environment within
the project directory.

- `uv init my-project-name`

### Global installation (using `uv`)

This will make Manim (and in particular the console command `manim`) available
as a global tool such that you can run it anywhere on your PC. This method requires
an additional step to enable code completion and syntax highlighting in your code
editor.



(installation-optional-latex)=
## Optional Dependency: LaTeX

