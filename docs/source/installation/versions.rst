Differences between Manim Versions
==================================

While originally a single library, there are now three main versions of Manim, 
each with their own advantages, disadvantages and ideal use cases. 
It is important to understand these differences in order to select the best version 
for your use case and avoid confusion arising from version mismatches.

A brief history of Manim
************************

Manim was originally created by Grant Sanderson as a personal project and 
for use in his YouTube channel, `3blue1brown <https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw>`_. As his channel gained popularity, 
many grew to like the style of his animations and wanted to use Manim for their own projects. 
However, as Manim was only intended for personal use, 
it was very difficult for other users to install and use it.

In late 2019, Grant started working on faster OpenGL rendering in a new branch, 
known as  the shaders branch. In mid 2020, a group of developers forked it into what is now the community edition; 
this is the version which is documented by this website. 
In early 2021, Grant merged the shaders branch back into master, making it the default branch in his repository. 
The old version is still available as the branch ``cairo-backend``.

The three versions of Manim
****************************

There are currently three main versions of Manim. They are as follows:

- **ManimCE**: The community edition of Manim. This is the version documented by this website, and is named `manim <https://pypi.org/project/manim/https://pypi.org/project/manim/>`_ on pip.
- **`ManimGL <https://github.com/3b1b/manim>`_**: The current version of Manim that is used by 3blue1brown. It supports OpenGL rendering and interactivity, and is named ``manimgl`` on pip.
- **`ManimCairo <https://github.com/3b1b/manim/tree/cairo-backend>`_**: The old version of Manim originally used by 3blue1brown. It is not available on pip.

Which version to use
********************
We recommend using the community edition for most purposes, as it has been developed to be more stable, 
better tested, quicker to respond to community contributions, and easier for beginners to use. 
It also has partial experimental OpenGL support, and should have full support shortly.

If you would like to use a version with full OpenGL support or render recent 3blue1brown videos (2020 onwards), you should use ManimGL.

If you would like to render old 3blue1brown projects (2019 and before), you should use ManimCairo.

Notes on installation, documentation and use
********************************************
If you are a beginner, it is very important that you only use the documentation for your desired version. 
Trying to install or learn Manim using documentation or guides made for different versions will likely fail and only lead to more confusion. 
As many tutorials and guides on the internet are outdated, we do not recommend you follow them. 
You should only read tutorials and documentation for other versions once you are aware of the differences between them 
and know how to adapt code for your desired version.
