Manim is a Python library, and it can be
installed via `pip <https://pypi.org/project/manim/>`__
or `conda <https://anaconda.org/conda-forge/manim/>`__. However,
in order for Manim to work properly, some additional system
dependencies need to be installed first.

Manim requires Python version ``3.9`` or above to run.

.. hint::

   Depending on your particular setup, the installation process
   might be slightly different. Make sure that you have tried to
   follow the steps on the following pages carefully, but in case
   you hit a wall we are happy to help: either `join our Discord
   <https://www.manim.community/discord/>`__, or start a new
   Discussion `directly on GitHub
   <https://github.com/ManimCommunity/manim/discussions>`__.


The installation of Manim is OS dependent, so please follow
the instructions for your operating system.

.. tab-set::
    :sync-group: operating-system

    .. tab-item:: Windows
        :sync: windows

        Manim requires a Python version of at least ``3.9`` to run.
        If you're not sure if you have python installed, or want to check
        what version of Python you have, try running::

          python --version

        If it errors out, you most likely don't have Python installed. Otherwise, if your
        python version is ``3.9`` or higher, you can skip the next step and proceed to pip installing Manim.

        If you don't have Python installed, head over to https://www.python.org, download an installer
        for a recent (preferably the latest) version of Python, and follow its instructions to get Python
        installed on your system.

        .. note::

          We have received reports of problems caused by using the version of
          Python that can be installed from the Windows Store. At this point,
          we recommend staying away from the Windows Store version. Instead,
          install Python directly from the `official website <https://www.python.org>`__.

        After installing Python, running the command::

          python --version

        Should be successful. If it is not, try checking out :ref:`this FAQ entry<not-on-path>`.

        At this point, installing manim should be as easy as running::

          python -m pip install manim

        To confirm Manim is working, you can run::

          manim --version



    .. tab-item:: macOS
        :sync: macos

        The easiest way to install Manim on macOS is via the popular `package manager Homebrew <https://brew.sh>`__.
        If you want to use Homebrew but do not have it installed yet, please
        follow `Homebrew's installation instructions <https://docs.brew.sh/Installation>`__.

        After that, you can run ``brew install manim`` and you should be all set! To confirm that your
        Manim is working, run::

            manim checkhealth


    .. tab-item:: Linux
        :sync: linux

        The installation instructions depend on your particular operating
        system and package manager. If you happen to know exactly what you are doing,
        you can also simply ensure that your system has:

        - a reasonably recent version of Python 3 (3.9 or above),
        - with working Cairo bindings in the form of
          `pycairo <https://cairographics.org/pycairo/>`__,
        - and `Pango <https://pango.gnome.org>`__ headers.

        Then, installing Manim is just a matter of running:

        .. code-block:: bash

          pip3 install manim

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

                First update your sources, and then install Cairo and Pango.

                .. code-block:: bash

                  sudo apt update
                  sudo apt install build-essential python3-dev libcairo2-dev libpango1.0-dev

                If you don't have python3-pip installed, install it via:

                .. code-block:: bash

                  sudo apt install python3-pip

            .. tab-item:: dnf
                :sync: dnf

                To install Cairo and Pango:

                .. code-block:: bash

                  sudo dnf install cairo-devel pango-devel

                In order to successfully build the ``pycairo`` wheel, you will also
                need the Python development headers:

                .. code-block:: bash

                  sudo dnf install python3-devel

            .. tab-item:: pacman
                :sync: pacman

                .. tip::

                  Thanks to *groctel*, there is a `dedicated Manim package
                  on the AUR! <https://aur.archlinux.org/packages/manim/>`_

                If you don't want to use the packaged version from AUR, here is what
                you need to do manually: Update your package sources, then install
                Cairo and Pango:

                .. code-block:: bash

                  sudo pacman -Syu cairo pango

                If you don't have python3-pip installed, install it via:

                .. code-block:: bash

                  sudo pacman -Syu python-pip

        Installing python packages globally is disallowed on Linux systems. As
        a result, you will have to create a virtual environment to install Manim::

            python3 -m venv .venv
            source .venv/bin/activate
            manim checkhealth

        You will have to activate the venv every time you want to use Manim.

        .. tip::

            If you use an IDE (such as VS Code or Pycharm), they will autoactivate
            the virtual environment if you open that folder in the IDE.

After installing Manim, you may be interested in installing the optional dependencies.

.. note::

   Although these dependencies are strictly optional, we highly
   recommend installing them as they greatly increase the capabilities of Manim.


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


Once Manim is installed locally, you can proceed to our
:doc:`quickstart guide </tutorials/quickstart>` which walks you
through rendering a first simple scene.

As mentioned above, do not worry if there are errors or other
problems: consult our :doc:`FAQ section </faq/index>` for help
(including instructions for how to ask Manim's community for help).
