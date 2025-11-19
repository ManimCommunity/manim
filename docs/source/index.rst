.. manim documentation master file, created by
   sphinx-quickstart on Tue Aug  4 13:58:07 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Manim Community Edition
=======================

Animating technical concepts is traditionally pretty tedious since it can be
difficult to make the animations precise enough to convey them accurately.
Manim relies on Python's simplicity to generate animations programmatically,
making it convenient to specify exactly how each one should run. Take a look
at the :doc:`Example Gallery <../examples>` for some inspiration on how to
create beautiful images and videos with Manim.

First Steps
-----------

Are you new to Manim and are looking for where to get started? Then you are
in the right place!

.. note::

   Please be aware that there are different, incompatible versions of Manim available.
   This version, the Community Edition of Manim (`ManimCE <https://github.com/ManimCommunity/manim>`_),
   is a separate project maintained by the community, but it was forked from `3b1b/manim <https://github.com/3b1b/manim>`_,
   the original Manim created and open-sourced by Grant Sanderson, creator of `3Blue1Brown <https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw>`_ educational math videos.
   Check our :ref:`installation FAQ <different-versions>`
   to learn more!

- The :doc:`Installation <installation>` section has the latest and
  up-to-date installation instructions for Windows, macOS, and Linux.
  You can also find information on Manim's docker images and (online)
  notebook environments there.
- Want to try the library before installing it? Take a look at our
  interactive online playground at https://try.manim.community in the form
  of a Jupyter notebook.
- In our :doc:`Tutorials <tutorials/index>` section you will find a
  collection of resources that will teach you how to use Manim. In particular,
  the :doc:`tutorials/quickstart` tutorial teaches you Manim's basics,
  and in :doc:`tutorials/building_blocks` the classes used to compose
  your animations are described in more detail.


Finding Help
------------

Are you struggling with installing or using Manim? Don't worry, we've all been
there. Here are some good resources to help you out:

- Perhaps your problem is one that occurs frequently, then chances are it is
  addressed in our :doc:`collection of FAQs <faq/index>`.
- If you are looking for information on some specific class, look for it
  in the :doc:`reference manual <reference>` and/or use the search feature
  of the documentation.
- Still no luck? Then you are welcome to ask the community for help, together
  we usually manage to find a solution for your problem! Consult the
  :doc:`FAQ page on getting help <faq/help>` for instructions.


Navigating the Documentation
----------------------------

Here are some short summaries for all of the sections in this documentation:

- The :doc:`Example Gallery </examples>` is a collection of examples (rendered videos
  and images together with the code they were generated from) that show a few different,
  simple things that you can do with Manim.
- The :doc:`Installation </installation>` section has information on installing Manim.
- In :doc:`Tutorials & Guides </tutorials_guides>` you can find learning resources: proper
  tutorials that guide you through the process of creating a video are in
  the :doc:`Tutorial </tutorials/index>` section; guides on specific topics are in the
  :doc:`Guides </guides/index>` section, and the answers to frequently asked questions
  can be found in the :doc:`FAQ </faq/index>` section.
- The :doc:`Reference Manual </reference>` contains a comprehensive list of all of Manim's
  (documented) modules, classes, and functions. If you are somewhat familiar with Manim's
  module structure, feel free to browse the manual directly. If you are searching for
  something specific, feel free to use the documentation's search feature in the sidebar.
  Many classes and methods come with their own illustrated examples too!
- The :doc:`Plugins </plugins>` page documents how to install, write, and distribute
  plugins (that is, separate Python packages that extend the feature set of the core library).
- Changes between versions are documented in our :doc:`Changelog </changelog>`.
- If you are looking into contributing to the development of Manim, you can find information
  on how to get involved in our :doc:`Contributing </contributing>` section.
- And finally, the :doc:`Code of Conduct </conduct>` page has a formal description of
  the rules you should abide by when interacting within our community.

Sharing Your Work
-----------------

We'd love to hear from you and see your manimations
`on Twitter <https://twitter.com/manim_community>`_, `Reddit <https://www.reddit.com/r/manim/>`_,
or `Discord <https://www.manim.community/discord/>`_. If you're using Manim in a scientific
context, instructions on how to cite a particular release can be found
`in our README <https://github.com/ManimCommunity/manim/blob/main/README.md>`_.

License Information
-------------------

Manim is an open-source library licensed under the **MIT License**, which applies to both the
original and the community editions of the software. This means you are free to use, modify,
and distribute the code in accordance with the MIT License terms. However, there are some
additional points to be aware of:

- **Copyrighted Assets:** Specific assets, such as the "Pi creatures" in Grant Sanderson's
  (3Blue1Brown) videos, are copyrighted and protected. Please avoid using these characters in
  any derivative works.
- **Content Creation and Sharing:** Videos and animations created with Manim can be freely
  shared, and no attribution to Manim is required—although it is much appreciated! You are
  encouraged to showcase your work online and share it with the Manim community.

Index
-----

.. toctree::
   :maxdepth: 2

   examples
   installation
   tutorials_guides
   reference
   plugins
   changelog
   contributing
   conduct

.. image:: _static/crowdin-badge.svg
  :align: center
  :alt: Localized with Crowdin
  :target: https://translate.manim.community
