# FAQ: Installation

(different-versions)=
## Why are there different versions of Manim?

Manim was originally created by Grant Sanderson as a personal project and for use
in his YouTube channel,
[3Blue1Brown](https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw).
As his channel gained popularity, many grew to like the style of his animations and
wanted to use manim for their own projects. However, as manim was only intended for
personal use, it was very difficult for other users to install and use it.

In late 2019, Grant started working on faster OpenGL rendering in a new branch,
known as the `shaders` branch. In mid-2020, a group of developers forked it into what is
now the community edition; this is the version documented on this website.
In early 2021, Grant merged the shaders branch back into master, making it the default branch in his repository -- and this is what `manimgl` is.
The old version, before merging the `shaders` branch is sometimes referred to as
`ManimCairo` and is, at this point, only useful for one singular purpose: rendering
Grant's old videos locally on your machine. It is still available in his GitHub
repository in form of the `cairo-backend` branch.

To summarize:
- [**Manim**, or **ManimCE**](https://manim.community) refers to the community
  maintained version of the library. This is the version documented on this website;
  the package name on PyPI is [`manim`](https://pypi.org/project/manim/).
- [ManimGL](https://github.com/3b1b/manim) is the latest released version of the
  version of the library developed by Grant "3b1b" Sanderson. It has more experimental
  features and breaking changes between versions are not documented. Check out
  its documentation [here](https://3b1b.github.io/manim/index.html); on PyPI the
  package name is [`manimgl`](https://pypi.org/project/manimgl/).
- [ManimCairo](https://github.com/3b1b/manim/tree/cairo-backend) is the name that
  is sometimes used for the old, pre-OpenGL version of `manimgl`. The latest version
  of it is available [on PyPI as `manimlib`](https://pypi.org/project/manimgl/),
  but note that if you intend to use it to compile some old project of Grant,
  you will likely have to install the exact version from the time the project
  was created from source.

---

## Which version should I use?

We recommend the community maintained version especially for beginners. It has been
developed to be more stable, better tested and documented (!), and quicker to respond
to community contributions. It is also perfectly reasonable to start learning with the
community maintained version and then switch to a different version later on.

If you do not care so much about documentation or stability, and would like to use
the exact same version that Grant is using, then use ManimGL.

And as mentioned above, ManimCairo should only be used for (re)rendering old
3Blue1Brown projects (basically 2019 and before).

---

## What are the differences between Manim, ManimGL, ManimCairo? Can I tell for which version a scene was written for?

You can! The thing that usually gives it away is the `import` statement
at the top of the file; depending on how the code imports Manim you can tell
for which version of the code it was written for:

- If the code imports from `manim` (i.e., `from manim import *`, `import manim as mn`, etc.),
  then the code you are reading is supposed to be run with the community maintained version.
- If the import reads `import manimlib` (or `from manimlib import *`), you are likely
  reading a file to be rendered with ManimGL.
- And if the import reads `from manimlib.imports import *`, or perhaps even
  `from big_ol_pile_of_manim_imports import *` you are reading a snippet that is
  supposed to be rendered with an early, or very early version of ManimCairo, respectively.

---

## How do I know which version of Manim I have installed?

Assuming you can run `manim` in your terminal and there is some output, check the
first line of the text being produced. If you are using the community maintained
version, the first line of any output will be `Manim Community <version number>`.
If it does not say that, you are likely using ManimGL.

You can also check the list of packages you have installed: if typing `python`
in your terminal spawns the interpreter that corresponds to the Python
installation you use (might also be `py`, or `python3`, depending on your
operating system), running `python -m pip list` will print a list of all
installed packages. Check whether `manim` or `manimgl` appear in that list.

Similarly, you can use `python -m pip install <package name>` and
`python -m pip uninstall <package name>` to install and uninstall
packages from that list, respectively.

---

## I am following the video guide X to install Manim, but some step fails. What do I do?

It is only natural that there are many video guides on installing Manim
out there, given that Manim is a library used for creating videos. Unfortunately
however, (YouTube) videos can't be updated easily (without uploading a new one, that is)
when some step in the installation process changes, and so there are many
**severely outdated** resources out there.

This is why we strongly recommend following our
{doc}`written installation guide </installation>` to guide you through the process.
In case you prefer using a video guide regardless, please check whether the
creator whose guide you have been watching has made a more recent version available,
and otherwise please contact them directly. Asking for help in the community will
likely lead to being suggested to follow our written guide.

---

## Why does ManimPango fail to install when running `pip install manim`?

This most likely means that pip was not able to use our pre-built wheels
of the `manimpango` dependency. Let us know (via
[Discord](https://www.manim.community/discord/) or by opening a
[new issue on GitHub](https://github.com/ManimCommunity/ManimPango/issues/new))
which architecture you would like to see supported, and we'll see what we
can do about it.

To fix errors when installing `manimpango`, you need to make sure you
have all the necessary build requirements. Check out the detailed
instructions given in [the BUILDING section](https://github.com/ManimCommunity/ManimPango#BUILDING)
of [ManimPango's README](https://github.com/ManimCommunity/ManimPango).

---

(not-on-path)=
## I am using Windows and get the error `X is not recognized as an internal or external command, operable program or batch file`

If you have followed {doc}`our local installation instructions </installation/uv>` and
have not activated the corresponding virtual environment, make sure to use `uv run manim ...`
instead of just `manim` (or activate the virtual environment by following the instructions
printed when running `uv venv`).

Otherwise there is a problem with the directories where your system is looking for
executables (the `PATH` variable).
If `python` is recognized, you can try running
commands by prepending `python -m`. That is, `manim` becomes `python -m manim`,
and `pip` becomes `python -m pip`.

Otherwise see
[this StackExchange answer](https://superuser.com/questions/143119/how-do-i-add-python-to-the-windows-path/143121#143121)
to get help with editing the `PATH` variable manually.

---

## I have tried using Chocolatey (`choco install manimce`) to install Manim, but it failed!

Make sure that you were running the command with administrator permissions,
otherwise there can be problems. If this is not the issue, read Chocolatey's
output carefully, it should mention a `.log` file containing information why
the process failed.

You are welcome to take this file (and any other input you feel might be
relevant) and submit it to Manim's community to ask for help with
your problem. See the {doc}`FAQ on getting help </faq/help>` for instructions.

---

## On Windows, when typing `python` or `python3` the Windows store is opened, can I fix this?

Yes: you can remove these aliases with these steps:

1. Go to the Windows Setting.
2. Under *Apps and Features* you will find application execution aliases.
3. Within this menu, disable the alias(es) that are causing the issue
   (`python` and/or `python3`).

---

## I am using Anaconda and get an `ImportError` mentioning that some Symbol is not found.

This is because Anaconda environments come with their own preinstalled
version of `cairo` which is not compatible with the version of `pycairo`
required by Manim. Usually it can be fixed by running

```bash
conda install -c conda-forge pycairo
```

---

## How can I fix the error that `manimpango/cmanimpango.c` could not be found when trying to install Manim?

This occasionally happens when your system has to build a wheel for
[ManimPango](https://github.com/ManimCommunity/ManimPango) locally because there
is no compatible version for your architecture available on PyPI.

Very often, the problem is resolved by installing Cython (e.g., via
`pip3 install Cython`) and then trying to reinstall Manim. If this
does not fix it:

- Make sure that you have installed all build dependencies mentioned
  in [ManimPango's README](https://github.com/ManimCommunity/ManimPango),
- and if you still run into troubles after that, please reach out to
  us as described in the {doc}`Getting Help FAQs </faq/help>`.
