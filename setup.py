# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['manim',
 'manim._config',
 'manim.animation',
 'manim.camera',
 'manim.grpc',
 'manim.grpc.gen',
 'manim.grpc.impl',
 'manim.mobject',
 'manim.mobject.svg',
 'manim.mobject.types',
 'manim.renderer',
 'manim.scene',
 'manim.utils']

package_data = \
{'': ['*'], 'manim.grpc': ['proto/*']}

install_requires = \
['Pillow',
 'cairocffi>=1.1.0,<2.0.0',
 'colour',
 'numpy',
 'pangocairocffi>=0.4.0,<0.5.0',
 'pangocffi>=0.8.0,<0.9.0',
 'progressbar',
 'pycairo>=1.19,<2.0',
 'pydub',
 'pygments',
 'rich>=6.0,<7.0',
 'scipy',
 'tqdm']

extras_require = \
{'js_renderer': ['grpcio', 'grpcio-tools', 'watchdog']}

entry_points = \
{'console_scripts': ['manim = manim.__main__:main',
                     'manimce = manim.__main__:main']}

setup_kwargs = {
    'name': 'manimce',
    'version': '0.1.0',
    'description': 'Animation engine for explanatory math videos.',
    'long_description': '<p align="center">\n    <a href="https://www.manim.community/"><img src="https://raw.githubusercontent.com/ManimCommunity/manim/master/logo/cropped.png"></a>\n    <br />\n    <br />\n    <a href="http://choosealicense.com/licenses/mit/"><img src="https://img.shields.io/badge/license-MIT-red.svg?style=flat" alt="MIT License"></a>\n    <a href="https://www.reddit.com/r/manim/"><img src="https://img.shields.io/reddit/subreddit-subscribers/manim.svg?color=orange&label=reddit" alt="Reddit" href=></a>\n    <a href="https://discord.gg/mMRrZQW"><img src="https://img.shields.io/discord/581738731934056449.svg?label=discord&color=yellow" alt="Discord"></a>\n    <a href="https://manimce.readthedocs.io/en/latest/"><img src="https://readthedocs.org/projects/manimce/badge/?version=latest" alt="Documentation Status"></a>\n    <a href="https://hub.docker.com/r/manimcommunity/manim"><img src="https://img.shields.io/docker/v/manimcommunity/manim?color=%23099cec&label=docker%20image" alt="Docker image"> </a>\n    <img src="https://github.com/ManimCommunity/manim/workflows/CI/badge.svg" alt="CI">\n    <br />\n    <br />\n    <i>An animation engine for explanatory math videos</i>\n</p>\n\n<hr />\n\n`manim` is an animation engine for explanatory math videos. It\'s used to create precise animations programmatically, as demonstrated in the videos of [3Blue1Brown](https://www.3blue1brown.com/).\n\n> NOTE: This repository is maintained by the Manim Community, and is not associated with Grant Sanderson or 3Blue1Brown in any way (although we are definitely indebted to him for providing his work to the world). If you would like to study how Grant makes his videos, head over to his repository ([3b1b/manim](https://github.com/3b1b/manim)). This fork is updated more frequently than that his, and it\'s recommended to use this fork if you\'d like to use Manim for your own projects.\n\n## Table of Contents:\n\n-  [Installation](#installation)\n-  [Usage](#usage)\n-  [Documentation](#documentation)\n-  [Help with Manim](#help-with-manim)\n-  [Contributing](#contributing)\n-  [License](#license)\n\n## Installation\n\nManim requires a few dependencies that must be installed prior to using it. Please visit the [documentation](https://manimce.readthedocs.io/en/latest/installation.html) and follow the appropriate instructions for your operating system.\n\nOnce the dependencies have been installed, run the following in a terminal window:\n\n```bash\npip install manimce\n```\n\n## Usage\n\nManim is an extremely versatile package. The following is an example `Scene` you can construct:\n\n```python\nfrom manim import *\n\nclass SquareToCircle(Scene):\n    def construct(self):\n        circle = Circle()\n        square = Square()\n        square.flip(RIGHT)\n        square.rotate(-3 * TAU / 8)\n        circle.set_fill(PINK, opacity=0.5)\n\n        self.play(ShowCreation(square))\n        self.play(Transform(square, circle))\n        self.play(FadeOut(square))\n```\n\nIn order to view the output of this scene, save the code in a file called `example.py`. Then, run the following in a terminal window:\n\n```sh\nmanim example.py SquareToCircle -p -ql\n```\n\nYou should see your native video player program pop up and play a simple scene in which a square is transformed into a circle. You may find some more simple examples within this\n[GitHub repository](master/example_scenes). You can also visit the [official gallery](https://manimce.readthedocs.io/en/latest/examples.html) for more advanced examples.\n\n## Command line arguments\n\nThe general usage of manim is as follows:\n\n![manim-illustration](https://raw.githubusercontent.com/ManimCommunity/manim/master/docs/source/_static/command.png)\n\nThe `-p` flag in the command above is for previewing, meaning the video file will automatically open when it is done rendering. The `-ql` flag is for a faster rendering at a lower quality.\n\nSome other useful flags include:\n\n-  `-s` to skip to the end and just show the final frame.\n-  `-n <number>` to skip ahead to the `n`\'th animation of a scene.\n-  `-f` show the file in the file browser.\n\nFor a thorough list of command line arguments, visit the [documentation](https://manimce.readthedocs.io/en/latest/tutorials/configuration.html).\n\n## Documentation\n\nDocumentation is in progress at [ReadTheDocs](https://manimce.readthedocs.io/en/latest/).\n\n## Help with Manim\n\nIf you need help installing or using Manim, feel free to reach out to our [Discord\nServer](https://discord.gg/mMRrZQW) or [Reddit Community](https://www.reddit.com/r/manim). If you would like to submit bug report or feature request, please open an issue.\n\n## Contributing\n\nContributions to `manim` are always welcome. In particular, there is a dire need for tests and documentation. For contribution guidelines, please see the [documentation](https://manimce.readthedocs.io/en/latest/contributing.html).\n\nThis project uses [Poetry](https://python-poetry.org/docs/) for management. You must have poetry installed and available in your environment. You can learn more `poetry` and how to use it at its [Documentation](https://manimce.readthedocs.io/en/latest/installation/for_dev.html).\n\n## License\n\nThe software is double-licensed under the MIT license, with copyright by 3blue1brown LLC (see LICENSE), and copyright by Manim Community Developers (see LICENSE.community).\n',
    'author': 'The Manim Community Developers',
    'author_email': None,
    'maintainer': None,
    'maintainer_email': None,
    'url': 'https://github.com/manimcommunity/manim',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'extras_require': extras_require,
    'entry_points': entry_points,
    'python_requires': '>=3.6,<4.0',
}


setup(**setup_kwargs)
