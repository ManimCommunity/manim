---
short-title: v0.21.0
description: Changelog for v0.21.0
---

# v0.21.0

Date
: August 10, 2026


## What's Changed
### Breaking Changes 🚨
* Delegate color handling in {class}`.Code` to Pygments and document custom styles by {user}`Helbronner` in {pr}`4816`

  {class}`.Code` no longer honors foreground color overrides passed through
  `paragraph_config={"color": ...}` or `Code.default_paragraph_config["color"]`;
  syntax and line-number colors now come from the selected Pygments
  `formatter_style`. To customize them, pass a custom Pygments `Style` subclass
  through `formatter_style` and configure its token styles and
  `line_number_color`. Background colors can still be overridden with
  `background_config={"fill_color": ...}`; otherwise, the selected Pygments
  style provides the background. See the note in the documentation of {class}`.Code`
  for an explicit example.
* Speed up Cairo renderer by 2.2x on animation-heavy scenes by {user}`HamdiBarkous` in {pr}`4695`

  Cairo path construction now uses vectorized subpath splitting, and pixel-array
  handling avoids redundant copies while producing pixel-identical output. This
  changes `Camera.convert_pixel_array()`: the `convert_from_floats` argument has
  been removed, and the method now always converts floats to the camera's pixel
  format. Omit the argument if it was previously `True`; use
  `np.array(pixel_array)` instead if relying on the old default copying behavior.

### Highlights 🌟
* Introduce first-class support for rendering text and markup via Typst (optional dependency) by {user}`behackl` in {pr}`4681`

  The new {class}`.Typst` and {class}`.MathTypst` mobjects compile Typst markup
  and mathematical expressions directly to SVG without requiring a TeX
  installation. They support label-based subgroup selection, `{{ ... }}`
  expression grouping, and integration with existing label APIs. Install the
  optional dependency with `pip install "manim[typst]"`.
* Hash NumPy arrays without JSON expansion by {user}`dexhunter` in {pr}`4883`

  NumPy arrays in scene cache keys are now represented by deterministic hashes
  of their complete logical contents instead of sampled JSON data. This
  prevents stale cache hits when visible image data changes and reduces hashing
  CPU time by roughly 94–98% in representative benchmarks.
* Add support for encoding partial movie files in parallel by {user}`pjfo` in {pr}`4899`

  The new `max_inflight_encoders` configuration option allows partial movie
  encoding to overlap with rendering while bounding the number of encoder
  workers. Try it with
  `manim render --max-inflight-encoders 4 --encoder-queue-size 8 scene.py Scene`;
  `--encoder-queue-size` limits the pending frame buffers held by each encoder.
  The default of one in-flight encoder preserves serial behavior; higher values
  reduced an encoding-heavy benchmark by 56% and larger real-world renders by
  roughly 8–11%.

### New Features ✨
* Allow inner lines to be omitted from {class}`.Table` by {user}`leqo-c` in {pr}`4731`
* Support multiple compiler passes in {class}`.TexTemplate` by {user}`Sazzzzzz` in {pr}`4810`
* Add a `dot_buff` parameter to {class}`.BulletedList` by {user}`GniLudio` in {pr}`4892`

### Enhancements 🚀
* Support tuple-based edge lookup in `GenericGraph.__getitem__()` by {user}`HairlessVillager` in {pr}`3799`
* Raise a clear error when rendered glyphs do not match input characters by {user}`SupernovaIa` in {pr}`4868`
* Avoid copying `Mobject.submobjects` in sequence methods by {user}`nikolajmunk` in {pr}`4915`
* Make additional mobject methods chainable by {user}`GniLudio` in {pr}`4913`

### Bug Fixes 🐛
* Fix OpenGL crashes in `arrange_in_grid()` and `get_location()` by {user}`Croustit` in {pr}`4622`
* Prevent `get_nth_subpath()` from raising `IndexError` and ensure it returns a NumPy array by {user}`MqCreaple` in {pr}`4630`
* Preserve closed curves in `put_start_and_end_on()` by {user}`GoThrones` in {pr}`4658`
* Fix inconsistent stroke-width scaling for text in compound objects by {user}`vihdutta` in {pr}`4694`
* Accept a single `Group` or `VGroup` in `CyclicReplace` and `Swap` by {user}`irvanalhaq9` in {pr}`4211`
* Prevent empty submobjects from distorting parent dimensions by {user}`gvauvert` in {pr}`4088`
* Honor `language` when constructing `Code` from `code_file` by {user}`GniLudio` in {pr}`4706`
* Fix `ThreeDScene.set_to_default_angled_camera_orientation()` raising `AttributeError` by {user}`siddhu-pikachu` in {pr}`4704`
* Fix reversed default resolution in `manim init project --default` by {user}`aoxiangtianyu-go` in {pr}`4839`
* Prevent memoizer collisions between object hashes and IDs by {user}`pjfo` in {pr}`4901`
* Prevent `round_corners()` failures for collinear vertices by {user}`MeiSiristhebest` in {pr}`4906`
* Fix `checkhealth` false positives for uv launcher shims by {user}`srinu2003` in {pr}`4889`
* Fix OpenGL triangulation errors at ring boundaries by {user}`Theelx` in {pr}`4819`
* Fix {class}`.Angle` crashes for parallel or collinear lines by {user}`MeiSiristhebest` in {pr}`4907`
* Stabilize cache hashes for transient objects by {user}`pjfo` in {pr}`4896`
* Ignore SVG shape elements without points by {user}`behackl` in {pr}`4917`
* Normalize {class}`.Code` baseline alignment in {class}`.Paragraph` by {user}`Croustit` in {pr}`4692`
* Preserve z-index ordering in animation groups by {user}`behackl` in {pr}`4918`
* Let {class}`.MathTex` handle multiline strings by {user}`henrikmidtiby` in {pr}`4634`
* Ensure that glyphs at end of {class}`.Code` lines are no longer deleted when `disable_ligatures` is `False` by {user}`nikolajmunk` in {pr}`4926`

### Documentation 📚
* Correct `SquareToCircle` terminology in the quickstart guide by {user}`xuruidong` in {pr}`4631`
* Correct angle ranges in {class}`.Sphere` documentation and add examples by {user}`Jackyuanx` in {pr}`3973`
* Document edge lookup in `GenericGraph.__getitem__()` by {user}`mayank-dev-15` in {pr}`4761`
* Correct the {class}`.FadeOut` example name by {user}`GniLudio` in {pr}`4832`
* Document whitespace-stripped indexing in {class}`.Text` by {user}`SupernovaIa` in {pr}`4866`
* Preserve line color in the `MovingDots` example by {user}`shenao010415` in {pr}`4895`
* Correct the font-directory lookup documented for `register_font()` by {user}`ExploreMaths` in {pr}`4880`
* Correct angle ordering in the `cartesian_to_spherical()` and `spherical_to_cartesian()` docstrings by {user}`changsheng0804-blip` in {pr}`4764`
* Clarify frame terminology in the {class}`.MovingCameraScene` example by {user}`nikolajmunk` in {pr}`4910`

### Testing 🧪
* Add unit tests for color helper functions by {user}`THE-RAF` in {pr}`4749`
* Add regression tests for four recent `VMobject` fixes by {user}`THE-RAF` in {pr}`4750`
* Add tests for edge lookup in `GenericGraph.__getitem__()` by {user}`sridhar-3009` in {pr}`4753`
* Add a regression test for text stroke-width scaling in compound objects by {user}`vihdutta` in {pr}`4703`

### Infrastructure & Build 🔨
* Update TinyTeX installation on Windows and macOS to fix CI failures by {user}`chopan050` in {pr}`4679`
* Update CI workflow dependencies by {user}`behackl` in {pr}`4677`
* Switch package builds from Hatchling to uv-build by {user}`JasonGrace2282` in {pr}`4619`

### Code Quality & Refactoring 🧹
* Update the project metadata link to X (formerly Twitter) by {user}`henrikmidtiby` in {pr}`4642`
* Replace a bare `except` in `rate_functions.py` by {user}`koteshyelamati` in {pr}`4891`
* Simplify the {class}`.Code` syntax-coloring implementation by {user}`GniLudio` in {pr}`4894`
* Update pre-commit hooks by {user}`behackl` in {pr}`4920`
* Rename `TypstMath` to {class}`.MathTypst` to match existing {class}`.Tex` / {class}`.MathTex` coupling by {user}`behackl` in {pr}`4925`

### Type Hints 📝
* Add type annotations to `mobject.py` by {user}`henrikmidtiby` in {pr}`4388`
* Use `npt.NDArray[np.float64]` for `VMobject._bezier_t_values` by {user}`GoThrones` in {pr}`4675`
* Improve `Callable` annotations for mobject updater utilities by {user}`treszkai` in {pr}`4728`
* Add type annotations to `transform_matching_parts.py` by {user}`henrikmidtiby` in {pr}`4400`
* Fix the `_on_finish` annotation in `Animation.__init__` by {user}`GniLudio` in {pr}`4812`
* Add type annotations to `table.py` by {user}`henrikmidtiby` in {pr}`4641`

### Other Changes
* Prepare new release: `v0.21.0` by {user}`behackl` in {pr}`4919`

## New Contributors
* {user}`Croustit` made their first contribution in {pr}`4622`
* {user}`xuruidong` made their first contribution in {pr}`4631`
* {user}`MqCreaple` made their first contribution in {pr}`4630`
* {user}`vihdutta` made their first contribution in {pr}`4694`
* {user}`THE-RAF` made their first contribution in {pr}`4749`
* {user}`leqo-c` made their first contribution in {pr}`4731`
* {user}`treszkai` made their first contribution in {pr}`4728`
* {user}`gvauvert` made their first contribution in {pr}`4088`
* {user}`sridhar-3009` made their first contribution in {pr}`4753`
* {user}`GniLudio` made their first contribution in {pr}`4706`
* {user}`mayank-dev-15` made their first contribution in {pr}`4761`
* {user}`siddhu-pikachu` made their first contribution in {pr}`4704`
* {user}`Sazzzzzz` made their first contribution in {pr}`4810`
* {user}`aoxiangtianyu-go` made their first contribution in {pr}`4839`
* {user}`SupernovaIa` made their first contribution in {pr}`4866`
* {user}`koteshyelamati` made their first contribution in {pr}`4891`
* {user}`pjfo` made their first contribution in {pr}`4901`
* {user}`MeiSiristhebest` made their first contribution in {pr}`4906`
* {user}`shenao010415` made their first contribution in {pr}`4895`
* {user}`ExploreMaths` made their first contribution in {pr}`4880`
* {user}`srinu2003` made their first contribution in {pr}`4889`
* {user}`changsheng0804-blip` made their first contribution in {pr}`4764`
* {user}`Theelx` made their first contribution in {pr}`4819`
* {user}`dexhunter` made their first contribution in {pr}`4883`
* {user}`nikolajmunk` made their first contribution in {pr}`4910`
* {user}`Helbronner` made their first contribution in {pr}`4816`
* {user}`HamdiBarkous` made their first contribution in {pr}`4695`

**Full Changelog**: [Compare view](https://github.com/ManimCommunity/manim/compare/v0.20.1...v0.21.0)
