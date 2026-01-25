# Installing Manim locally

The standard way of installing Manim is by using
Python's package manager `pip` to install the latest
release from [PyPI](https://pypi.org/project/manim/).

To make it easier for you to follow best practices when it
comes to setting up a Python project for your Manim animations,
we strongly recommend using a tool for managing Python environments
and dependencies. In particular,
[we strongly recommend using `uv`](https://docs.astral.sh/uv/#getting-started).

For the two main ways of installing Manim described below, we assume
that `uv` is available; we think it is particularly helpful if you are
new to Python or programming in general. It is not a hard requirement
whatsoever; if you know what you are doing you can just use `pip` to
install Manim directly.

:::::{admonition} Installing the Python management tool `uv`
:class: seealso

One way to install `uv` is via the dedicated console installer supporting
all large operating systems. Simply paste the following snippet into
your terminal / PowerShell -- or
[consult `uv`'s documentation](https://docs.astral.sh/uv/#getting-started)
for alternative ways to install the tool.

::::{tab-set}
:::{tab-item} MacOS and Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
:::
:::{tab-item} Windows
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
:::
::::

:::::

Of course, if you know what you are doing and prefer to setup a virtual
environment yourself, feel free to do so!

:::{important}
If you run into issues when following our instructions below, do
not worry: check our [installation FAQs](<project:/faq/installation.md>) to
see whether the problem is already addressed there -- and otherwise go and
check [how to contact our community](<project:/faq/help.md>) to get help.
:::



## Installation

### Step 1: Installing Python

We first need to check that an appropriate version of Python is available
on your machine. Open a terminal to run
```bash
uv python install
```
to install the latest version of Python. If this is successful, continue
to the next step.

(installation-optional-latex)=
### Step 2 (optional): Installing LaTeX

[LaTeX](https://en.wikibooks.org/wiki/LaTeX/Mathematics) is a very well-known
and widely used typesetting system allowing you to write formulas like

\begin{equation*}
\frac{1}{2\pi i} \oint_{\gamma} \frac{f(z)}{(z - z_0)^{n+1}}~dz
= \frac{f^{(n)}(z_0)}{n!}.
\end{equation*}

If rendering plain text is sufficient for your needs and you don't want
to render any typeset formulas, you can technically skip this step. Otherwise
select your operating system from the tab list below and follow the instructions.

:::::{tab-set}

::::{tab-item} Windows
For Windows we recommend installing LaTeX via the
[MiKTeX distribution](https://miktex.org). Simply grab
the Windows installer available from their download page,
<https://miktex.org/download> and run it.
::::

::::{tab-item} MacOS
If you are running MacOS, we recommend installing the
[MacTeX distribution](https://www.tug.org/mactex/). The latest
available PKG file can be downloaded from
<https://www.tug.org/mactex/mactex-download.html>.
Get it and follow the standard installation procedure.
::::

::::{tab-item} Linux
Given the large number of Linux distributions with different ways
of installing packages, we cannot give detailed instructions for
all package managers.

In general we recommend to install a *TeX Live* distribution
(<https://www.tug.org/texlive/>). For most Linux distributions,
TeX Live has already been packaged such that it can be installed
easily with your system package manager. Search the internet and
your usual OS resources for detailed instructions.

For example, on Debian-based systems with the package manager `apt`,
a full TeX Live distribution can be installed by running
```bash
sudo apt install texlive-full
```
For Fedora (managed via `dnf`), the corresponding command is
```bash
sudo dnf install texlive-scheme-full
```
As soon as LaTeX is installed, continue with actually installing Manim
itself.

::::

:::::

:::{dropdown} I know what I am doing and I would like to setup a minimal LaTeX installation
You are welcome to use a smaller, more customizable LaTeX distribution like
[TinyTeX](https://yihui.org/tinytex/). Manim overall requires the following
LaTeX packages to be installed in your distribution:
```text
amsmath babel-english cbfonts-fd cm-super count1to ctex doublestroke dvisvgm everysel
fontspec frcursive fundus-calligra gnu-freefont jknapltx latex-bin
mathastext microtype multitoc physics preview prelim2e ragged2e relsize rsfs
setspace standalone tipa wasy wasysym xcolor xetex xkeyval
```
:::

### Step 3: Installing Manim

These steps again differ slightly between different operating systems. Make
sure you select the correct one from the tab list below, then follow
the instructions below.

::::::{tab-set}

:::::{tab-item} Windows
The following commands will

- create a new directory for a Python project,
- and add Manim as a dependency, which installs it into the corresponding
  local Python environment.

The name for the Python project is *manimations*, which you can change
to anything you like.

```bash
uv init manimations
cd manimations
uv add manim
```

Manim is now installed in your local project environment!

:::::

:::::{tab-item} MacOS
Before we can install Manim, we need to make sure that the system utilities
`cairo` and `pkg-config` are present. They are needed for the [`pycairo` Python
package](https://pycairo.readthedocs.io/en/latest/), a dependency of Manim.

The easiest way of installing these utilities is by using [Homebrew](https://brew.sh/),
a fairly popular 3rd party package manager for MacOS. Check whether Homebrew is
already installed by running

```bash
brew --version
```

which will report something along the lines of `Homebrew 4.4.15-54-...`
if it is installed, and a message `command not found: brew` otherwise. In this
case, use the shell installer [as instructed on Homebrew's website](https://brew.sh/),
or get a `.pkg`-installer from
[their GitHub release page](https://github.com/Homebrew/brew/releases). Make sure to
follow the instructions of the installer carefully, especially when prompted to
modify your `.zprofile` to add Homebrew to your system's PATH.

With Homebrew available, the required utilities can be installed by running

```bash
brew install cairo pkg-config
```

With all of this preparation out of the way, now it is time to actually install
Manim itself! The following commands will

- create a new directory for a Python project,
- and add Manim as a dependency, which installs it into the corresponding
  local Python environment.

The name for the Python project is *manimations*, which you can change
to anything you like.

```bash
uv init manimations
cd manimations
uv add manim
```

Manim is now installed in your local project environment!

:::::

:::::{tab-item} Linux
Practically, the instructions given in the *Windows* tab
also apply for Linux -- however, some additional dependencies are
required as Linux users need to build
[ManimPango](https://github.com/ManimCommunity/ManimPango)
(and potentially [pycairo](https://pycairo.readthedocs.io/en/latest/))
from source. More specifically, this includes:

- A C compiler,
- Python's development headers,
- the `pkg-config` tool,
- Pango and its development headers,
- and Cairo and its development headers.

Instructions for popular systems / package managers are given below.

::::{tab-set}

:::{tab-item} Debian-based / apt
```bash
sudo apt update
sudo apt install build-essential python3-dev libcairo2-dev libpango1.0-dev
```
:::

:::{tab-item} Fedora / dnf
```bash
sudo dnf install python3-devel pkg-config cairo-devel pango-devel
```
:::

:::{tab-item} Arch Linux / pacman
```bash
sudo pacman -Syu base-devel cairo pango
```
:::

::::

As soon as the required dependencies are installed, you can create
a Python project (feel free to change the name *manimations* used below
to some other name) with a local environment containing Manim by running
```bash
uv init manimations
cd manimations
uv add manim
```

:::::

::::::

To verify that your local Python project is setup correctly
and that Manim is available, simply run
```bash
uv run manim checkhealth
```

At this point, you can also open your project folder with the
IDE of your choice. All modern Python IDEs (for example VS Code
with the Python extension, or PyCharm) should automatically detect
the local environment created by `uv` such that if you put
```py
import manim
```
into a new file `my-first-animation.py`, the import is resolved
correctly and autocompletion is available.

*Happy Manimating!*


:::{dropdown} Alternative: Installing Manim as a global `uv`-managed tool
If you have Manim projects in many different directories and you do not
want to setup a local project environment for each of them, you could
also install Manim as a `uv`-managed tool.

See [`uv`'s documentation for more information](https://docs.astral.sh/uv/concepts/tools/)
on their tool mechanism.

To install Manim as a global `uv` tool, simply run
```bash
uv tool install manim
```
after which the `manim` executable will be available on your
global system path, without the need to activate any virtual
environment or prefixing your commands with `uv run`.

Note that when using this approach, setting up your code editor
to properly resolve `import manim` requires additional work, as
the global tool environment is not automatically detected: the
base path of all tool environments can be determined by running
```
uv tool dir
```
which should now contain a directory `manim` in which the appropriate
virtual environment is located. Set the Python interpreter of your IDE
to this environment to make imports properly resolve themselves.
:::

:::{dropdown} Installing Manim for a different version of Python
In case you would like to use a different version of Python
(for example, due to compatibility issues with other packages),
then `uv` allows you to do so in a fairly straightforward way.

When initializing the local Python project, simply pass the Python
version you want to use as an argument to the `init` command:
```
uv init --python 3.12 manimations
cd manimations
uv add manim
```
To change the version for an existing package, you will need to
edit the `pyproject.toml` file. If you are downgrading the python version, the
`requires-python` entry needs to be updated such that your chosen
version satisfies the requirement. Change the line to, for example
`requires-python = ">=3.12"`. After that, run `uv python pin 3.12`
to pin the python version to `3.12`. Finally, run `uv sync`, and your
environment is updated!
:::
