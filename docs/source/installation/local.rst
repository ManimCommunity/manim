Installing Manim on all Operating Systems
*****************************************


Manim is a Python library, and it can be
installed via `pip <https://pypi.org/project/manim/>`__
or `conda <https://anaconda.org/conda-forge/manim/>`__. However,
in order for Manim to work properly, some additional system
dependencies need to be installed first.

If you are new to programming or to Python, we recommend installing
a tool called `uv <https://docs.astral.sh/uv/#getting-started>`__ for using Manim.

.. tip::

   If you're on Windows and you get an error about running an untrusted script, you
   can run the following command to allow the script to run::

     Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

The instructions given below assume that you have ``uv`` installed.

.. tip::

   On some Linux distributions, you can install ``uv`` via the package manager.

   .. tab-set::
       :sync-group: linux-package-manager

       .. tab-item:: dnf
           :sync: dnf

           .. code-block::

              sudo dnf install uv

       .. tab-item:: pacman
           :sync: pacman

           .. code-block::

              sudo pacman -S uv



Depending on your particular setup, the installation process
might be slightly different. Make sure that you have tried to
follow the steps on the following page carefully, but in case
you hit a wall, we are happy to help: either `join our Discord
<https://www.manim.community/discord/>`__ and ask in ``#help-forum``, or start a new
Discussion `directly on GitHub
<https://github.com/ManimCommunity/manim/discussions>`__.



.. tab-set::
    :sync-group: operating-system

    .. tab-item:: Windows
        :sync: windows

        Manim requires at least python ``3.9``. To check if a python satisfies this requirement, run::

            uv python find ">=3.9"

        If ``uv`` does not find a python installation that satisfies the requirement, you can run::

            uv python install

    .. tab-item:: macOS
        :sync: macos

        Manim requires at least python ``3.9``. To check if a python satisfies this requirement, run::

            uv python find ">=3.9"

        If ``uv`` does not find a python installation that satisfies the requirement, you can run::

            uv python install


    .. tab-item:: Linux
        :sync: linux

        The installation instructions depend on your particular operating
        system and package manager. If you happen to know exactly what you are doing,
        you can also simply ensure that your system has:

        - a reasonably recent version of Python 3 (3.9 or above),
        - with working Cairo bindings in the form of
          `pycairo <https://cairographics.org/pycairo/>`__,
        - and `Pango <https://pango.gnome.org>`__ headers.

        .. note::

          In light of the current efforts of migrating to rendering via OpenGL,
          this list might be incomplete. Please `let us know
          <https://github.com/ManimCommunity/manim/issues/new/choose>`__ if you
          ran into missing dependencies while installing.

        In any case, we have also compiled instructions for several common
        combinations of operating systems and package managers below.

        .. tip::

          If you have multiple Python versions installed, you might need to install the python
          development headers for the correct version. For example, if you have Python 3.9 installed,
          you would need to install python3.9-dev.


        .. tab-set::
            :sync-group: linux-package-manager

            .. tab-item:: apt
                :sync: apt

                You will have to update your sources, and then install Cairo and Pango::

                  sudo apt update
                  sudo apt install build-essential libcairo2-dev libpango1.0-dev

            .. tab-item:: dnf
                :sync: dnf

                To install Cairo and Pango::

                  sudo dnf install cairo-devel pango-devel

                In order to successfully build the ``pycairo`` wheel, you will also
                need the Python development headers (and a C++ compiler)::

                  sudo dnf install python3-devel

            .. tab-item:: pacman
                :sync: pacman

                .. tip::

                  Thanks to *groctel*, there is a `dedicated Manim package
                  on the AUR! <https://aur.archlinux.org/packages/manim/>`_.
                  If you use this, you can skip to the Optional Dependencies section.

                If you don't want to use the packaged version from AUR, here is what
                you need to do manually: Update your package sources, then install
                Cairo and Pango::

                  sudo pacman -Syu cairo pango uv

After that, you can install Manim with uv::

  uv tool install manim

You can check if Manim is installed correctly by running::

  uvx manim checkhealth

After installing Manim, you may be interested in installing the optional dependencies.

.. important::

    In the rest of the documentation, we will use ``manim`` as a shorthand for ``uvx manim``.


.. tip::

   If you don't want to have to type ``uvx``, you can add the directory given by
   ``uv tool dir --bin`` to your ``PATH`` environment variable::

      uv tool update-shell



Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

In order to make use of Manim's interface to LaTeX for, e.g., rendering
equations, LaTeX has to be installed as well. Note that this is an optional
dependency: if you don't intend to use LaTeX, you don't have to install it.


.. tab-set::
    :sync-group: operating-system

    .. tab-item:: Windows
        :sync: windows

        For Windows, the recommended LaTeX distribution is
        `MiKTeX <https://miktex.org/download>`__. You can install it by using the
        installer from the linked MiKTeX site, or by using the package manager
        of your choice (Chocolatey: ``choco install miktex.install``,
        Scoop: ``scoop install latex``, Winget: ``winget install MiKTeX.MiKTeX``).

        If you are concerned about disk space, there are some alternative,
        smaller distributions of LaTeX.

        You can also use `TinyTeX <https://yihui.org/tinytex/>`__ (Chocolatey: ``choco install tinytex``,
        Scoop: first ``scoop bucket add r-bucket https://github.com/cderv/r-bucket.git``,
        then ``scoop install tinytex``) alternative installation instructions can be found at their website.
        Keep in mind that you will have to manage the LaTeX packages installed on your system yourself via ``tlmgr``.
        Therefore we only recommend this option if you know what you are doing.

    .. tab-item:: macOS
        :sync: macos

        For macOS, the recommended LaTeX distribution is
        `MacTeX <http://www.tug.org/mactex/>`__. You can install it by following
        the instructions from the link, or alternatively also via Homebrew by
        running:

        .. code-block:: bash

          brew install --cask mactex-no-gui

        .. warning::

          MacTeX is a *full* LaTeX distribution and will require more than 4GB of
          disk space. If this is an issue for you, consider installing a smaller
          distribution like
          `BasicTeX <http://www.tug.org/mactex/morepackages.html>`__.


    .. tab-item:: Linux
        :sync: linux

        You can use whichever LaTeX distribution you like or whichever is easiest
        to install with your package manager. Usually,
        `TeX Live <https://www.tug.org/texlive/>`__ is a good candidate if you don't
        care too much about disk space.

        .. tab-set::
            :sync-group: linux-package-manager


            .. tab-item:: apt
                :sync: apt

                To install TeX Live, run::

                    sudo apt install texlive texlive-latex-extra

            .. tab-item:: dnf
                :sync: dnf

                For Fedora (see `docs <https://docs.fedoraproject.org/en-US/neurofedora/latex/>`__),
                run::

                    sudo dnf install texlive-scheme-full

            .. tab-item:: pacman
                :sync: pacman

                See the `Arch Wiki <https://wiki.archlinux.org/title/TeX_Live>`__ for more information.

                .. code-block::

                    sudo pacman -Syu texlive-latexextra texlive-fontsrecommended

        Should you choose to work with some smaller TeX distribution like
        `TinyTeX <https://yihui.org/tinytex/>`__ , the full list
        of LaTeX packages which Manim interacts with in some way (a subset might
        be sufficient for your particular application) is::

          amsmath babel-english cbfonts-fd cm-super count1to ctex doublestroke dvisvgm everysel
          fontspec frcursive fundus-calligra gnu-freefont jknapltx latex-bin
          mathastext microtype multitoc physics preview prelim2e ragged2e relsize rsfs
          setspace standalone tipa wasy wasysym xcolor xetex xkeyval


Basic Editor Support
********************
In order to get code editors like VS Code or Pycharm to find the Manim library, you will
need to point them to the virtual environment where Manim is installed. You can find
this directory by running::

  echo $(uv tool dir)/manim

For example, in VS Code, you can set the Python interpreter by hitting Ctrl+P (Windows/Linux)
or Cmd+P (macOS) and typing ``Python: Select Interpreter``. Then you can paste the path given
by the above command.

Alternatively, if you're going to work on a project in a single directory, you can run::

  uv venv
  uv pip install manim

After that, if you open your IDE in that directory, it should automatically find and use the virtual
environment created by ``uv venv``.


Next Steps
**********

Once Manim is installed locally, you can proceed to our
:doc:`quickstart guide </tutorials/quickstart>` which walks you
through rendering a first simple scene.

As mentioned above, do not worry if there are errors or other
problems: consult our :doc:`FAQ section </faq/index>` for help
(including instructions for how to ask Manim's community for help).
