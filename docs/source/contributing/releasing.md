# Releasing Manim

For maintainers. The tooling lives in `scripts/release.py`
(`uv run python scripts/release.py --help`).

## Overview

| | Release | Preview prerelease |
|---|---|---|
| Version | `0.22.1` | `0.30.0a1`, `0.30.0a2`, … |
| Branch | `main` | `preview/*`, e.g. `preview/v0.30.x` |
| GitHub | release | release marked as *pre-release* |
| Install | `pip install manim` | `pip install --pre manim` |
| Docker | `:stable`, `:vX.Y.Z` | `:preview`, `:vX.Y.ZaN` |
| Docs | [stable](https://docs.manim.community/en/stable/) | [preview](https://docs.manim.community/en/preview/) |
| Changelog | `changelog/X.Y.Z-changelog.md` | `changelog/preview.md` |

[latest](https://docs.manim.community/en/latest/) (and Docker `:latest`)
follow `main`.

Publishing a GitHub release uploads the package to PyPI
(`.github/workflows/python-publish.yml`). The workflow refuses unless the tag
matches the version in `pyproject.toml`, the pre-release checkbox matches
whether that version is a prerelease, and the tag is on `main` (releases) or a
`preview/*` branch (prereleases). To run this check locally:
`scripts/release.py check-release vX.Y.Z --kind release`.

## Making a release

1. Open a PR `Prepare new release: vX.Y.Z` (label `release`) against `main`:

   ```shell
   uv version X.Y.Z
   uv run python scripts/release.py changelog --base vX.Y.W --version X.Y.Z --update-citation
   ```

   Edit the generated changelog as needed (rerunning keeps your edits) and
   list it first in `docs/source/changelog.rst`. To list the release PR
   itself, rerun with `--include-extra <PR number>`.
2. Paste the output of `scripts/release.py export-changelog X.Y.Z` into the
   PR description for review.
3. After merging, publish the GitHub release `vX.Y.Z` on `main` with those
   notes.

## Preview prereleases

A preview branch ships alphas of the next minor version for testing, while
releases continue from `main` as usual.

### Starting a preview branch

```shell
git switch -c preview/v0.30.x upstream/main
git push upstream preview/v0.30.x
```

On Read the Docs, activate the version `preview/v0.30.x` and set its slug
to `preview`. It rebuilds on every push to the branch; `stable` is never
moved to a prerelease.

### Merging `main` into it

Do this regularly, as a merge commit pushed directly (a squashed PR would
make every later merge conflict again):

```shell
git switch preview/v0.30.x && git pull
git merge upstream/main   # on conflicts, keep the preview version in pyproject.toml / uv.lock
git push upstream preview/v0.30.x
```

### Publishing a prerelease

1. Open a PR against the preview branch (label `release`):

   ```shell
   uv version 0.30.0a2
   uv run python scripts/release.py changelog --base v0.30.0a1 --version 0.30.0a2 --head preview/v0.30.x
   ```

   This adds a section on top of `changelog/preview.md`. For the first
   alpha, use the last release as `--base`, and list `changelog/preview`
   first in `docs/source/changelog.rst`.
2. After merging, publish the GitHub release `v0.30.0a2` with target
   `preview/v0.30.x`, tick *Set as a pre-release*, and use the output of
   `scripts/release.py export-changelog 0.30.0a2` as notes.

### Ending a preview cycle

Once the preview branch's changes are on `main`, fold `preview.md` into the
changelog of `X.Y.0` and delete it. Delete the branch, then deactivate the
`preview` docs version or point it at the next preview branch.
