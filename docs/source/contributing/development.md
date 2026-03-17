# Manim Development Process

## For first-time contributors

1. Install git:

   For instructions see <https://git-scm.com/>.

2. Fork the project:

   Go to <https://github.com/ManimCommunity/manim> and click the "fork" button
   to create a copy of the project for you to work on. You will need a
   GitHub account. This will allow you to make a "Pull Request" (PR)
   to the ManimCommunity repo later on.

3. Clone your fork to your local computer:

   ```shell
   git clone https://github.com/<your-username>/manim.git
   ```

   GitHub will provide both a SSH (`git@github.com:<your-username>/manim.git`) and
   HTTPS (`https://github.com/<your-username>/manim.git`) URL for cloning.
   You can use SSH if you have SSH keys setup.

   :::{WARNING}
   Do not clone the ManimCommunity repository. You must clone your own
   fork.
   :::

4. Change the directory to enter the project folder:

   ```shell
   cd manim
   ```

5. Add the upstream repository, ManimCommunity:

   ```shell
   git remote add upstream https://github.com/ManimCommunity/manim.git
   ```

6. Now, `git remote -v` should show two remote repositories named:

   - `origin`, your forked repository
   - `upstream` the ManimCommunity repository

7. Install the Python project management tool `uv`, as recommended
   in our {doc}`installation guide for users </installation/uv>`.

8. Let `uv` create a virtual environment for your development
   installation by running

   ```shell
   uv sync
   ```

   In case you need (or want) to install some of the optional dependency
   groups defined in our [`pyproject.toml`](https://github.com/ManimCommunity/manim/blob/main/pyproject.toml),
   run `uv sync --all-extras`, or pass the `--extra` flag with the
   name of a group, for example `uv sync --extra jupyterhub`.

9. Install Pre-Commit:

   ```shell
   uv run pre-commit install
   ```

   This will ensure during development that each of your commits is properly
   formatted against our linter and formatters.

You are now ready to work on Manim!

## Develop your contribution

1. Checkout your local repository's main branch and pull the latest
   changes from ManimCommunity, `upstream`, into your local repository:

   ```shell
   git switch main
   git pull --rebase upstream main
   ```

2. Create a branch for the changes you want to work on rather than working
   off of your local main branch:

   ```shell
   git switch -c <new branch name> upstream/main
   ```

   This ensures you can easily update your local repository's main with the
   first step and switch branches to work on multiple features.

3. Write some awesome code!

   You're ready to make changes in your local repository's branch.
   You can add local files you've changed within the current directory with
   `git add .`, or add specific files with

   ```shell
   git add <file/directory>
   ```

   and commit these changes to your local history with `git commit`. If you
   have installed pre-commit, your commit will succeed only if none of the
   hooks fail.

   :::{tip}
   When crafting commit messages, it is highly recommended that
   you adhere to [these guidelines](https://www.conventionalcommits.org/en/v1.0.0/).
   :::

4. Add new or update existing tests.

   Depending on your changes, you may need to update or add new tests. For new
   features, it is required that you include tests with your PR. Details of
   our testing system are explained in the {doc}`testing guide <testing>`.

5. Update docstrings and documentation:

   Update the docstrings (the text in triple quotation marks) of any functions
   or classes you change and include them with any new functions you add.
   See the {doc}`documentation guide <docs/docstrings>` for more information about how we
   prefer our code to be documented. The content of the docstrings will be
   rendered in the {doc}`reference manual <../reference>`.

   :::{tip}
   Use the {mod}`manim directive for Sphinx <manim.utils.docbuild.manim_directive>` to add examples
   to the documentation!
   :::

As far as development on your local machine goes, these are the main steps you
should follow.

(polishing-changes-and-submitting-a-pull-request)=

## Polishing Changes and Submitting a Pull Request

As soon as you are ready to share your local changes with the community
so that they can be discussed, go through the following steps to open a
pull request. A pull request signifies to the ManimCommunity organization,
"Here are some changes I wrote; I think it's worthwhile for you to maintain
them."

:::{note}
You do not need to have everything (code/documentation/tests) complete
to open a pull request (PR). If the PR is still under development, please
mark it as a draft. Community developers will still be able to review the
changes, discuss yet-to-be-implemented changes, and offer advice; however,
the more complete your PR, the quicker it will be merged.
:::

1. Update your fork on GitHub to reflect your local changes:

   ```shell
   git push -u origin <branch name>
   ```

   Doing so creates a new branch on your remote fork, `origin`, with the
   contents of your local repository on GitHub. In subsequent pushes, this
   local branch will track the branch `origin` and `git push` is enough.

2. Make a pull request (PR) on GitHub.

   In order to make the ManimCommunity development team aware of your changes,
   you can make a PR to the ManimCommunity repository from your fork.

   :::{WARNING}
   Make sure to select `ManimCommunity/manim` instead of `3b1b/manim`
   as the base repository!
   :::

   Choose the branch from your fork as the head repository - see the
   screenshot below.

   ```{image} /_static/pull-requests.png
   :align: center
   ```

   Please make sure you follow the template (this is the default
   text you are shown when first opening the 'New Pull Request' page).

Your changes are eligible to be merged if:

1. there are no merge conflicts
2. the tests in our pipeline pass
3. at least one (two for more complex changes) Community Developer approves the changes

You can check for merge conflicts between the current upstream/main and
your branch by executing `git pull upstream main` locally. If this
generates any merge conflicts, you need to resolve them and push an
updated version of the branch to your fork of the repository.

Our pipeline consists of a series of different tests that ensure
that Manim still works as intended and that the code you added
sticks to our coding conventions.

- **Code style**: We use the code style imposed
  by [Black](https://black.readthedocs.io/en/stable/), [isort](https://pycqa.github.io/isort/)
  and [flake8](https://flake8.pycqa.org/en/latest/). The GitHub pipeline
  makes sure that the (Python) files changed in your pull request
  also adhere to this code style. If this step of the pipeline fails,
  fix your code formatting automatically by running `black <file or directory>` and `isort <file or directory>`.
  To fix code style problems, run `flake8 <file or directory>` for a style report, and then fix the problems
  manually that were detected by `flake8`.
- **Tests**: The pipeline runs Manim's test suite on different operating systems
  (the latest versions of Ubuntu, macOS, and Windows) for different versions of Python.
  The test suite consists of two different kinds of tests: integration tests
  and doctests. You can run them locally by executing `uv run pytest`
  and `uv run pytest --doctest-modules manim`, respectively, from the
  root directory of your cloned fork.
- **Documentation**: We also build a version of the documentation corresponding
  to your pull request. Make sure not to introduce any Sphinx errors, and have
  a look at the built HTML files to see whether the formatting of the documentation
  you added looks as you intended. You can build the documentation locally
  by running `make html` from the `docs` directory. Make sure you have [Graphviz](https://graphviz.org/)
  installed locally in order to build the inheritance diagrams. See {doc}`docs` for
  more information.

Finally, if the pipeline passes and you are satisfied with your changes: wait for
feedback and iterate over any requested changes. You will likely be asked to
edit or modify your PR in one way or another during this process. This is not
an indictment of your work, but rather a strong signal that the community
wants to merge your changes! Once approved, your changes may be merged!

### Further useful guidelines

1. When submitting a PR, please mention explicitly if it includes breaking changes.
2. When submitting a PR, make sure that your proposed changes are as general as
   possible, and ready to be taken advantage of by all of Manim's users. In
   particular, leave out any machine-specific configurations, or any personal
   information it may contain.
3. If you are a maintainer, please label issues and PRs appropriately and
   frequently.
4. When opening a new issue, if there are old issues that are related, add a link
   to them in your new issue (even if the old ones are closed).
5. When submitting a code review, it is highly recommended that you adhere to
   [these general guidelines](https://conventionalcomments.org/).
6. If you find stale or inactive issues that seem to be irrelevant, please post
   a comment saying 'This issue should be closed', and a community developer
   will take a look.
7. Please do as much as possible to keep issues, PRs, and development in
   general as tidy as possible.

You can find examples for the `docs` in several places:
the {doc}`Example Gallery <../examples>`, {doc}`Tutorials <../tutorials/index>`,
and {doc}`Reference Classes <../reference>`.

**Thank you for contributing!**
