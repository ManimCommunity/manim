## Commit message format
We have precise rules over how our Git messages must be formatted.
This format leads to **easier to read commit history** and enables automated changelog generation and semantic versioning.

Each commit message consists of a **header**, a **body**, and a **footer**
```
<header>
<BLANK LINE>
<body>
<BLANK LINE>
<footer>
```
The `header` is mandatory and must conform to the Commit Message Header format.
The `body` is mandatory for all commits except for those of type `docs`.
The `footer` is optional. The Commit Message Footer format describes what the footer is used for and the structure it must have.

---

## Commit Message Header
```
<type>(<scope>): <short summary>
  │       │             │
  │       │             └─⫸ Summary in present tense. Not capitalized. No period at the end.
  │       │
  │       └─⫸ Commit Scope: mobject|scene|animation|renderer|camera|cli|config|utils|docs|tests
  │
  └─⫸ Commit Type: build|ci|docs|feat|fix|perf|refactor|test
```
The `<type>` and `<summary>` fields are mandatory, the `(<scope>)` field is optional.

### Summary
Use the summary field to provide a succinct description of the change:
- Use the imperative, present tense: "change" not "changed" nor "changes"
- Don't capitalize the first letter
- No period (.) at the end

---

## Type
Must be one of the following:
| Type     | Description                                                              | Manim Examples                                                              |
|----------|--------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| build    | Changes that affect the build system or external dependencies            | `pyproject.toml`, dependency upgrades, `hatchling build config`             |
| ci       | Changes to our CI configuration files and scripts                        | GitHub Actions workflows, `pre-commit` configuration                        |
| docs     | Documentation-only changes                                               | Sphinx `rst` guides, Markdown `md` files, inline docstrings                 |
| feat     | A new feature                                                            | A new geometric mobject, a new animation class, or a new CLI flag           |
| fix      | A bug fix                                                                | Fixing rendering crashes, coordinate misalignments, or type errors          |
| perf     | A code change that improves performance                                  | Optimizing shader compilation, caching, or numpy operations                 |
| refactor | A code change that neither fixes a bug nor adds a feature                | Decomposing large files, extracting helper classes                          |
| test     | Adding missing tests or correcting existing tests                        | pytest modules, graphical output assertions, video regression tests         |

---

## Scope
The scope should be the name of the module or architectural area affected by the change.
| Type       | Description                                                              | Manim Examples                                                           |
|------------|--------------------------------------------------------------------------|--------------------------------------------------------------------------|
| `build`    | Changes that affect the build system or external dependencies            | `pyproject.toml`, dependency upgrades, `hatchling` build config          |
| `ci`       | Changes to our CI configuration files and scripts                        | GitHub Actions workflows, `pre-commit` configuration                     |
| `docs`     | Documentation-only changes                                               | Sphinx `.rst` guides, Markdown `.md` files, inline docstrings            |
| `feat`     | A new feature                                                            | A new geometric mobject, a new animation class, or a new CLI flag        |
| `fix`      | A bug fix                                                                | Fixing rendering crashes, coordinate misalignments, or type errors       |
| `perf`     | A code change that improves performance                                  | Optimizing shader compilation, caching, or numpy operations              |
| `refactor` | A code change that neither fixes a bug nor adds a feature                | Decomposing large files, extracting helper classes                       |
| `test`     | Adding missing tests or correcting existing tests                        | `pytest` modules, graphical output assertions, video regression tests    |
Exceptions:
- `packaging`: changes to Python packaging layout (e.g. `pyproject.toml` metadata, `__init__.py` exports)
- `changelog`: updating release notes
- none/empty: cross-cutting changes across multiple scopes

---

## Commit Message Body
Use the imperative, present tense: "fix" not "fixed" nor "fixes"
Explain the **motivation** for the change. Include a comparison of previous vs. new behavior when it helps illustrate impact.

---

## Commit Message Footer
The footer can contain information about breaking changes and deprecations, and references to GitHub issues or PRs.
```
BREAKING CHANGE: <breaking change summary>
<BLANK LINE>
<breaking change description + migration instructions>
<BLANK LINE>
Fixes #<issue number>
```

---

## Revert Commits
If the commit reverts a previous commit, begin with `revert: `, followed by the header of the reverted commit.
The body should contain:
- `This reverts commit <SHA>.`
- A clear description of the reason for reverting.

---

## Examples
### Simple fix:
```
fix(mobject): resolve vectorized fill opacity bug
```

### Feature with body:
```
feat(animation): add staggered fade-in animation for submobjects

The existing FadeIn animation applies the effect to the entire mobject at once.
This new StaggeredFadeIn class applies a cascading delay to each submobject,
which is useful for animating grouped elements like matrix entries or graph nodes.

Closes #2145
```

### Documentation change (body not required):
```
docs(config): add examples for ManimConfig programmatic usage
```

### Breaking change:
```
feat(renderer): remove deprecated Cairo-only rendering path

BREAKING CHANGE: The `use_cairo_renderer` config flag has been removed.
All scenes now render through the unified pipeline. Users who relied on
Cairo-specific behavior should migrate to the OpenGL renderer or use
the compatibility shim documented in the migration guide.

Closes #1892
```

### Revert:
```
revert: fix(mobject): resolve vectorized fill opacity bug

This reverts commit 21cf999abc123.

The original fix introduced a regression in 3D mobject rendering where
transparency values were incorrectly inherited by child submobjects.
```
