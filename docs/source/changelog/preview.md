---
short-title: Preview
description: Changelog of the preview prereleases
---

# Preview

The prereleases of the next version, newest first. They are published from a
`preview/*` branch and installed with `pip install --pre manim`; see
{doc}`/contributing/releasing`.

## v0.30.0a1

Date
: November 21, 2026

### What's Changed
#### Breaking Changes 🚨
* Refactor: separate camera state from renderer-owned rasterization (cairo) + consolidate renderer related modules by {user}`behackl` in {pr}`4989`
* Refactor: move more render lifecycle responsibilities to Manager by {user}`behackl` in {pr}`5002`
* Refactor: run play calls through one loop with listeners by {user}`behackl` in {pr}`5004`
* Refactor: evaluate scenes without rendering, and draw requested frames by {user}`behackl` in {pr}`5005`

#### New Features ✨
* Add timeline capture and JSON export by {user}`behackl` in {pr}`5006`

#### Bug Fixes 🐛
* Raise a clear ValueError for unsupported axes in Percent by {user}`manan21bhalla` in {pr}`5098`
* Read and write manim.cfg as UTF-8 by {user}`b84330808` in {pr}`5103`

#### Performance ⚡
* Improve CLI startup time by using lazy imports by {user}`GniLudio` in {pr}`5124`
* Use direct formula instead of SciPy in `space_ops.rotation_matrix()` by {user}`chopan050` in {pr}`5129`

#### Infrastructure & Build 🔨
* Support preview prereleases from `preview/*` branches by {user}`behackl` in {pr}`5130`

#### Type Hints 📝
* Fix typing in `logo.py` by {user}`Kayyz1603` in {pr}`5112`

#### Other Changes
* Replace pydub with a PyAV sound mixer and place audio in movie time by {user}`behackl` in {pr}`5059`
* Fix return type of invert_image by {user}`GniLudio` in {pr}`5132`
* Embed saved scene sections as MP4 chapters by {user}`Mist-wu` in {pr}`5125`
* Speed up the documentation build and cache rendered examples by {user}`GniLudio` in {pr}`5133`
* Fix Line.set_points_by_ends ignoring the path_arc argument by {user}`Timotty` in {pr}`5058`
* Use attribute access for global config in scene and animation by {user}`Hangeerh` in {pr}`5138`
* Use attribute access for the global config in manim/scene by {user}`Q9Kawaii` in {pr}`5142`

### New Contributors
* {user}`manan21bhalla` made their first contribution in {pr}`5098`
* {user}`Mist-wu` made their first contribution in {pr}`5125`
* {user}`Timotty` made their first contribution in {pr}`5058`
* {user}`Hangeerh` made their first contribution in {pr}`5138`
* {user}`Q9Kawaii` made their first contribution in {pr}`5142`
* {user}`b84330808` made their first contribution in {pr}`5103`

**Full Changelog**: [Compare view](https://github.com/ManimCommunity/manim/compare/v0.22.1...v0.30.0a1)
