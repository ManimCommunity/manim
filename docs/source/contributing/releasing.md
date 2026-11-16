# Releasing Manim

This page is intended for maintainers and documents Manim's release process
and infrastructure. The central tool is the release script which is located
at `scripts/release.py`. The script documentation can be accessed via
`uv run python scripts/release.py --help`.

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
(`.github/workflows/python-publish.yml`) automatically. The corresponding
workflow requires that the tag published via the release matches the version
set in `pyproject.toml`. It will fail and not publish a version if this is not
the case.

When creating the GitHub release, tick the _pre-release_ checkbox exactly
when the version is a prerelease (e.g., `v0.30.0a1`). You also need to
select the correct branch to tag: for stable releases, it needs to be placed
on the `main` branch, for prereleases on a `preview/*` branch.

## Checklist: making a release

1. Open a PR `Prepare new release: vX.Y.Z` (label `release`) against `main`:

   ```shell
   uv version X.Y.Z
   uv run python scripts/release.py changelog --base vX.Y.W --version X.Y.Z --update-citation
   ```

   Edit the generated changelog as needed (rerunning keeps your edits) and
   list it first in `docs/source/changelog.rst`. To include the release preparation
   PR in the rendered changelog, rerun the changelog script with `--include-extra <PR number>`.
2. Paste the output of `scripts/release.py export-changelog X.Y.Z` into the
   PR description for review.
3. After merging, publish the GitHub release `vX.Y.Z` on `main` with those
   notes.

## Preview prereleases

A preview branch ships alphas of the next version for testing, while
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

Once the preview branch's changes are on `main`, include the content from `preview.md`
in the changelog of the new version, then delete `preview.md`.
At that point, the preview branch can also be deleted. If you choose to do so, make sure
to also deactivate the `preview` version of the documentation.
