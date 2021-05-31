<p align="center">
    <a href="https://www.manim.community/"><img src="https://raw.githubusercontent.com/ManimCommunity/manim/master/logo/cropped.png"></a>
    <br />
    <br />
    <a href="https://pypi.org/project/manim/"><img src="https://img.shields.io/pypi/v/manim.svg?style=flat&logo=pypi" alt="PyPI Latest Release"></a>
    <a href="https://hub.docker.com/r/manimcommunity/manim"><img src="https://img.shields.io/docker/v/manimcommunity/manim?color=%23099cec&label=docker%20image&logo=docker" alt="Docker image"> </a>
    <a href="https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb"><img src="https://mybinder.org/badge_logo.svg"></a>
    <a href="http://choosealicense.com/licenses/mit/"><img src="https://img.shields.io/badge/license-MIT-red.svg?style=flat" alt="MIT License"></a>
    <a href="https://www.reddit.com/r/manim/"><img src="https://img.shields.io/reddit/subreddit-subscribers/manim.svg?color=orange&label=reddit&logo=reddit" alt="Reddit" href=></a>
    <a href="https://twitter.com/manim_community/"><img src="https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40manim_community" alt="Twitter">
    <a href="https://www.manim.community/discord/"><img src="https://img.shields.io/discord/581738731934056449.svg?label=discord&color=yellow&logo=discord" alt="Discord"></a>
    <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black">
    <a href="https://docs.manim.community/"><img src="https://readthedocs.org/projects/manimce/badge/?version=latest" alt="Documentation Status"></a>
    <a href="https://pepy.tech/project/manim"><img src="https://pepy.tech/badge/manim/month?" alt="Downloads"> </a>
    <img src="https://github.com/ManimCommunity/manim/workflows/CI/badge.svg" alt="CI">
    <br />
    <br />
    <i>An animation engine for explanatory math videos</i>
</p>
<hr />

Manim is an animation engine for explanatory math videos. It's used to create precise animations programmatically, as demonstrated in the videos of [3Blue1Brown](https://www.3blue1brown.com/).

> NOTE: This repository is maintained by the Manim Community, and is not associated with Grant Sanderson or 3Blue1Brown in any way (although we are definitely indebted to him for providing his work to the world). If you would like to study how Grant makes his videos, head over to his repository ([3b1b/manim](https://github.com/3b1b/manim)). This fork is updated more frequently than his, and it's recommended to use this fork if you'd like to use Manim for your own projects.

## Table of Contents:

-  [Installation](#installation)
-  [Usage](#usage)
-  [Documentation](#documentation)
-  [Docker](#docker)
-  [Help with Manim](#help-with-manim)
-  [Contributing](#contributing)
-  [License](#license)

## Installation

Manim requires a few dependencies that must be installed prior to using it. If you
want to try it out first before installing it locally, you can do so
[in our online Jupyter environment](https://mybinder.org/v2/gist/behackl/725d956ec80969226b7bf9b4aef40b78/HEAD?filepath=basic%20example%20scenes.ipynb).

For the local installation, please visit the [Documentation](https://docs.manim.community/en/stable/installation.html)
and follow the appropriate instructions for your operating system.

Once the dependencies have been installed, run the following in a terminal window:

```bash
pip install manim
```

## Usage

Manim is an extremely versatile package. The following is an example `Scene` you can construct:

```python
from manim import *


class SquareToCircle(Scene):
    def construct(self):
        circle = Circle()
        square = Square()
        square.flip(RIGHT)
        square.rotate(-3 * TAU / 8)
        circle.set_fill(PINK, opacity=0.5)

        self.play(Create(square))
        self.play(Transform(square, circle))
        self.play(FadeOut(square))
```

In order to view the output of this scene, save the code in a file called `example.py`. Then, run the following in a terminal window:

```sh
manim -p -ql example.py SquareToCircle
```

You should see your native video player program pop up and play a simple scene in which a square is transformed into a circle. You may find some more simple examples within this
[GitHub repository](example_scenes). You can also visit the [official gallery](https://docs.manim.community/en/stable/examples.html) for more advanced examples.

Manim also ships with a `%%manim` IPython magic which allows to use it conveniently in JupyterLab (as well as classic Jupyter) notebooks. See the

[corresponding documentation](https://docs.manim.community/en/stable/reference/manim.utils.ipython_magic.ManimMagic.html) for some guidance and
[try it out online](https://mybinder.org/v2/gh/ManimCommunity/jupyter_examples/HEAD?filepath=basic_example_scenes.ipynb).

## Command line arguments

The general usage of Manim is as follows:

![manim-illustration](https://raw.githubusercontent.com/ManimCommunity/manim/master/docs/source/_static/command.png)

The `-p` flag in the command above is for previewing, meaning the video file will automatically open when it is done rendering. The `-ql` flag is for a faster rendering at a lower quality.

Some other useful flags include:

-  `-s` to skip to the end and just show the final frame.
-  `-n <number>` to skip ahead to the `n`'th animation of a scene.
-  `-f` show the file in the file browser.

For a thorough list of command line arguments, visit the [documentation](https://docs.manim.community/en/stable/tutorials/configuration.html).

## Documentation

Documentation is in progress at [ReadTheDocs](https://docs.manim.community/).

## Docker

The community also maintains a docker image (`manimcommunity/manim`), which can be found [on DockerHub](https://hub.docker.com/r/manimcommunity/manim). The following tags are supported:

- `latest` -- the most recent version corresponding to [the master branch](https://github.com/ManimCommunity/manim)
- `stable` -- the latest released version (according to [the releases page](https://github.com/ManimCommunity/manim/releases))
- `vX.Y.Z` -- any particular released version (according to [the releases page](https://github.com/ManimCommunity/manim/releases))

### Instructions for running the docker image

#### Quick Example
To render a scene `CircleToSquare` in a file `test_scenes.py` contained in your current working directory while preserving your user and group ID, use
```
docker run --rm -it  --user="$(id -u):$(id -g)" -v "$(pwd)":/manim manimcommunity/manim manim test_scenes.py CircleToSquare -qm
```

#### Running the image in the background
Instead of using the "throwaway container" approach sketched above, you can also create a named container that you can also modify to your liking. First, run
```
docker run -it --name my-manim-container -v "$(pwd):/manim" manimcommunity/manim /bin/bash
```
to obtain an interactive shell inside your container allowing you to, e.g., install further dependencies (like texlive packages using `tlmgr`). Exit the container as soon as you are satisfied. Then, before using it, start the container by running
```
docker start my-manim-container
```
Then, to render a scene `CircleToSquare` in a file `test_scenes.py`, call
```
docker exec -it --user="$(id -u):$(id -g)" my-manim-container manim test.py CircleToSquare -qm
```

#### Jupyterlab
Another alternative is to use the docker image to spin up a local webserver running
JupyterLab in whose Python kernel manim is installed and can be accessed via the `%%manim` cell magic.
To use JupyterLab, run
```
docker run -it -p 8888:8888 manimcommunity/manim jupyter lab --ip=0.0.0.0
```
and then follow the instructions in the terminal.

#### Important notes

When executing `manim` within a Docker container, several command line flags (in particular `-p` (preview file) and `-f` (show output file in the file browser)) are not supported.

## Help with Manim

If you need help installing or using Manim, feel free to reach out to our [Discord
Server](https://www.manim.community/discord/) or [Reddit Community](https://www.reddit.com/r/manim). If you would like to submit a bug report or feature request, please open an issue.

## Contributing

Contributions to Manim are always welcome. In particular, there is a dire need for tests and documentation. For contribution guidelines, please see the [documentation](https://docs.manim.community/en/stable/contributing.html).

Most developers on the project use [Poetry](https://python-poetry.org/docs/) for management. You'll want to have poetry installed and available in your environment. You can learn more about `poetry` and how to use it at its [documentation](https://docs.manim.community/en/stable/installation/for_dev.html).

## How to Cite Manim

We acknowledge the importance of good software to support research, and we note
that research becomes more valuable when it is communicated effectively. To
demonstrate the value of Manim, we ask that you cite Manim in your work.
Currently, the best way to cite Manim is to reference the
[Manim home page](https://www.manim.community) using this BibTeX entry (the
entry is for release `v0.6.0`, but can be adapted easily):

```
@Manual{Manim:v0.6.0,
  key =          {Manim},
  author =       {{The Manim Community Developers}},
  title =        {{Manim} -- {M}athematical {A}nimation {F}ramework ({V}ersion v0.6.0)},
  note =         {\url{https://www.manim.community}},
  year =         2021,
}
```

This should render a reference that looks more or less like this:

42. The Manim Community Developers,
    [Manim – Mathematical Animation Framework (Version v0.6.0)](https://www.manim.community).
    2021.

## Code of Conduct

Our full code of conduct, and how we enforce it, can be read on [our website](https://docs.manim.community/en/stable/conduct.html).

## License

The software is double-licensed under the MIT license, with copyright by 3blue1brown LLC (see LICENSE), and copyright by Manim Community Developers (see LICENSE.community).
