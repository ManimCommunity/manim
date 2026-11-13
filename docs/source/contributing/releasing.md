# Releasing Manim

This page is for maintainers. It describes how releases and preview
prereleases are made; the tools for both are in `scripts/release.py`
(`uv run python scripts/release.py --help`).

## Overview

| | Releases | Preview prereleases |
|---|---|---|
| Versions | `0.22.0`, `0.22.1` | `0.30.0a1`, `0.30.0a2`, … |
| Made from | `main` | a `preview/*` branch, e.g. `preview/v0.30.x` |
| GitHub release | a release | a release marked as a *pre-release* |
| PyPI | `pip install manim` | `pip install --pre manim` or `pip install manim==0.30.0a1` |
| Docker | `:stable`, `:latest`, `:vX.Y.Z` | `:preview`, `:vX.Y.ZaN` |
| Documentation | [stable](https://docs.manim.community/en/stable/) | [preview](https://docs.manim.community/en/preview/) |
| Changelog | `docs/source/changelog/X.Y.Z-changelog.md` | `docs/source/changelog/preview.md` |

[latest](https://docs.manim.community/en/latest/) always follows `main`.

Publishing a GitHub release runs the *Publish Release* workflow
(`.github/workflows/python-publish.yml`). It first checks that

- the tag is `v` followed by the version in `pyproject.toml`;
- the release is marked as a pre-release exactly when that version is a
  prerelease (in the sense of [PEP 440](https://peps.python.org/pep-0440/):
  `a`, `b`, `rc`, `.dev`);
- a prerelease is tagged on a `preview/*` branch, a release on `main`;

and then builds the package and uploads it to PyPI. Run the same check
locally with `scripts/release.py check-release vX.Y.Z --kind release` (or
`--kind prerelease`).

## Making a release

1. Open a pull request `Prepare new release: vX.Y.Z` (label `release`) from a
   branch of `main`:

   ```shell
   uv version X.Y.Z
   uv run python scripts/release.py changelog --base vX.Y.W --version X.Y.Z --update-citation
   ```

   `uv version` updates `pyproject.toml` and `uv.lock`. The changelog is
   generated from the pull requests merged since `vX.Y.W` and their labels
   (see `.github/release.yml`); edit it as needed (running the command again
   keeps the edits of the entries) and list it first in
   `docs/source/changelog.rst`. GitHub only lists this pull request in the
   changelog once it is merged; add it with `--include-extra <number>`.
2. Paste the release notes into the pull request for review:

   ```shell
   uv run python scripts/release.py export-changelog X.Y.Z
   ```

3. After merging, create the release `vX.Y.Z` on `main` on GitHub, with the
   exported notes, and publish it.

## Preview prereleases

A preview branch makes the larger changes of the next minor version
available for testing before they are ready for a release from `main`. Its
prereleases are alphas of that version: `preview/v0.30.x` publishes
`0.30.0a1`, `0.30.0a2`, and so on. They are never installed by
`pip install manim` without `--pre`.

### Starting a preview branch

1. Create the branch from `main` and set the version of the first alpha:

   ```shell
   git switch -c preview/v0.30.x upstream/main
   uv version 0.30.0a1
   ```

2. Write its changelog (see below) and push the branch.
3. On Read the Docs, activate the version `preview/v0.30.x` and change its
   URL identifier (slug) to `preview`, so that its documentation is at
   <https://docs.manim.community/en/preview/>. The documentation is rebuilt
   on every push to the branch. Read the Docs never points `stable` at a
   prerelease; the tags of prereleases do not need to be activated.

The branch is protected like `main`: changes arrive through pull requests
into it, which run the same checks as those into `main`.

### Keeping it up to date

Merge `main` into the preview branch regularly, as a merge commit (not a
squashed pull request, so that later merges do not conflict again):

```shell
git switch preview/v0.30.x
git pull
git merge upstream/main
git push upstream preview/v0.30.x
```

Keep the preview branch's version when `pyproject.toml` or `uv.lock`
conflict. Releases from `main` meanwhile (`0.22.2`, …) are made as usual.

### Publishing a prerelease

1. Open a pull request into the preview branch (label `release`) that sets
   the version and adds the changelog section of the prerelease:

   ```shell
   uv version 0.30.0a2
   uv run python scripts/release.py changelog --base v0.30.0a1 --version 0.30.0a2 --head preview/v0.30.x
   ```

   The section goes on top of `docs/source/changelog/preview.md`, which
   lists all prereleases of the branch, newest first. For the first alpha,
   use the last release (e.g. `--base v0.22.1`) as the base; the command
   creates the page, which is then listed first in
   `docs/source/changelog.rst`. Prereleases do not update `CITATION.cff`.
2. After merging, create the release `v0.30.0a2` on the preview branch on
   GitHub (target: `preview/v0.30.x`), tick *Set as a pre-release*, and use
   the exported notes:

   ```shell
   uv run python scripts/release.py export-changelog 0.30.0a2
   ```

### Ending a preview cycle

When `main` is ready for the version of the preview branch, merge the
preview branch into `main` (or the remaining changes, if the branch was kept
separate), turn the sections of `preview.md` into the changelog of the
release `X.Y.0`, and remove `preview.md`. The branch can then be deleted,
and the Read the Docs version `preview` deactivated or pointed at the next
preview branch.
