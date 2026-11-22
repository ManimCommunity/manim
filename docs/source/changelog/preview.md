---
short-title: Preview
description: Changelog of the preview prereleases
---

# Preview

This page holds a detailed overview of the changes included
in Manim's latest prereleases, which can be installed via
`pip install --pre manim`. Each prerelease has its own section,
newest first. The corresponding source version
can be found [in our repository](https://github.com/ManimCommunity/manim)
on a `preview/...` branch.

The current previews lead up to
[the v0.30.0 release](https://github.com/ManimCommunity/manim/tree/preview/v0.30.0),
which fundamentally reworks how scenes are rendered. Please try it with your
scenes and [report problems on GitHub](https://github.com/ManimCommunity/manim/issues)
/ share your thoughts [in our Discord](https://manim.community/discord).

## v0.30.0a1

Date
: November 22, 2026

The first preview of v0.30.0 features a restructured rendering pipeline
and changes to bring the cairo and the OpenGL renderers closer together.

### Highlights

The {stack}`5104` of six PRs ({pr}`4989`, {pr}`5002`, {pr}`5004`, {pr}`5005`,
{pr}`5006`, {pr}`5059`) restructures the rendering pipeline:

- **Cameras describe views, renderers draw.** Cairo cameras no longer hold
  pixel buffers or draw; the renderer does. Every camera has an animatable
  `camera.frame` and `auto_zoom()`, so moving the camera no longer needs
  `MovingCameraScene`.
- **The `Manager` runs scenes.** It creates the file writer, plays every
  animation through one run loop shared by Cairo and OpenGL, and closes all
  resources on success, failure or interrupt. Skipped and cached plays step
  through their frames without drawing, so later plays start from the same
  state as in a full render.
- **Evaluating without rendering.** `Manager.evaluate()` runs a scene without
  producing media, and draws only the frames you ask for
  ({doc}`/guides/evaluation`).
- **Timelines.** `--timeline-output PATH` exports the plays, waits, sections,
  captions and sounds of a scene with their times as JSON
  ({doc}`/guides/timeline`).
- **Audio without pydub.** Sounds are mixed with PyAV and placed at scene
  time; partial renders cut sound like video.

### Breaking changes

**Module paths.** Top-level imports (`from manim import ...`) are unchanged.
Implementation modules and the OpenGL shaders moved, without compatibility
aliases:

| Removed | Use instead |
|---|---|
| `manim.camera.camera`, `.moving_camera`, `.multi_camera`, `.three_d_camera` | `manim.renderer.cairo.camera` |
| `manim.renderer.cairo_renderer` | `manim.renderer.cairo` |
| `manim.renderer.opengl_renderer` | `manim.renderer.opengl` |
| `manim.renderer.opengl_renderer_window` | `manim.renderer.opengl.window` |
| `manim.renderer.shader`, `.shader_wrapper`, `.vectorized_mobject_rendering` | the same modules in `manim.renderer.opengl` |
| `manim/renderer/shaders` | `manim/renderer/opengl/shaders` |

**Cameras** ({pr}`4989`)

- Cairo cameras take view settings only. `pixel_width`, `pixel_height`,
  `frame_rate`, `image_mode`, `n_channels`, `pixel_array_dtype`, `background`
  (pixel arrays), `cairo_line_width_multiple` and `fixed_dimension` are
  removed: use `config` for render settings, and `background_color`,
  `background_opacity` or `background_image`. Unknown keywords raise
  `TypeError`.
- Cameras no longer draw: `camera.pixel_array`, `capture_mobjects()`,
  `get_image()` and the `display_*` / `apply_*` helpers are removed. Use
  `scene.get_image()`, `mobject.get_image(camera=...)`, or render through
  `CairoRenderer`.
- `MappingCamera`, `OldMultiCamera`, `SplitScreenCamera` and
  `BackgroundColoredVMobjectDisplayer` are removed. Use `apply_function()`,
  `MultiCamera` with `ImageMobjectFromCamera`, and
  `VMobject.color_using_background_image()`.
- `ImageMobjectFromCamera` is no longer an `AbstractImageMobject` and has no
  pixel array.
- `ThreeDScene` requires a `ThreeDCamera` (or an OpenGL camera).
  `ThreeDCamera.rotation_matrix` is read-only.
- Deprecated: `MovingCamera` and `MovingCameraScene` (use `Camera` and
  `Scene`). `ZoomedScene` is now a plain `Scene` subclass.

**Rendering lifecycle** ({pr}`5002`)

- A scene renders once: calling `render()` again raises `RuntimeError`.
- A renderer belongs to one scene. `Scene.renderer` and `renderer.file_writer`
  are read-only; pick a custom writer with the renderer's `file_writer_class`
  argument. `init_scene()` no longer takes `file_writer_settings`.
- The renderer is closed after rendering. To read its last frame, render
  inside `with Manager(scene) as manager:`.
- The log file is created when rendering starts, not when the scene is
  constructed.

**Playback** ({pr}`5004`, {pr}`5006`)

- `scene.time` advances in whole frames, and waits round their frame count up
  like plays: after `wait(0.5)` and a 0.5 s play at 15 fps, it is 16/15 s
  instead of 1 s.
- `-s`, `-n`, skipped sections and cached plays run the updaters of every
  frame. The resulting state is correct, but heavy updaters make these modes
  slower than before.
- `play()` or `wait()` called during another play (for example from an
  updater) raises `RuntimeError`. Before, it silently replaced the running
  play's animations.
- Removed: `Scene.play_internal()` and
  `manim.utils.caching.handle_caching_play`. Deprecated:
  `Scene.get_time_progression()`.

**Sound** ({pr}`5059`)

- `pydub` and `audioop-lts` are no longer dependencies; install them yourself
  if you use them.
- `add_sound()` checks the file when called (`OSError` if it is missing,
  `ValueError` if it is not audio), and takes `gain_to_background` instead of
  `**kwargs`.
- The pydub-based audio API of `SceneFileWriter` is removed (`init_audio`,
  `create_audio_segment`, `add_audio_segment`, `audio_segment`,
  `includes_sound`, `convert_audio`); `SceneFileWriter.add_sound()` requires
  `time`.

### What's Changed
#### Breaking Changes 🚨
* Refactor: separate camera state from renderer-owned rasterization (cairo) + consolidate renderer related modules by {user}`behackl` in {pr}`4989`
* Refactor: move more render lifecycle responsibilities to Manager by {user}`behackl` in {pr}`5002`
* Refactor: run play calls through one loop with listeners by {user}`behackl` in {pr}`5004`
* Refactor: evaluate scenes without rendering, and draw requested frames by {user}`behackl` in {pr}`5005`
* Replace pydub with a PyAV sound mixer and place audio in movie time by {user}`behackl` in {pr}`5059`

#### New Features ✨
* Add timeline capture and JSON export by {user}`behackl` in {pr}`5006`
* Embed saved scene sections as MP4 chapters by {user}`Mist-wu` in {pr}`5125`

#### Bug Fixes 🐛
* Raise a clear ValueError for unsupported axes in Percent by {user}`manan21bhalla` in {pr}`5098`
* Fix Line.set_points_by_ends ignoring the path_arc argument by {user}`Timotty` in {pr}`5058`
* Read and write manim.cfg as UTF-8 by {user}`b84330808` in {pr}`5103`

#### Performance ⚡
* Improve CLI startup time by using lazy imports by {user}`GniLudio` in {pr}`5124`
* Use direct formula instead of SciPy in `space_ops.rotation_matrix()` by {user}`chopan050` in {pr}`5129`

#### Documentation 📚
* Speed up the documentation build and cache rendered examples by {user}`GniLudio` in {pr}`5133`
* Fix docbuild warnings by {user}`GniLudio` in {pr}`5155`
* Make side drawer and toc always collapsible by {user}`GniLudio` in {pr}`5131`

#### Infrastructure & Build 🔨
* Support preview prereleases from `preview/*` branches by {user}`behackl` in {pr}`5130`
* Keep changelog prose on regeneration, and link prerelease notes to the preview docs by {user}`behackl` in {pr}`5152`
* Use uv for the readthedocs build by {user}`GniLudio` in {pr}`5153`

#### Code Quality & Refactoring 🧹
* Use attribute access for global config in scene and animation by {user}`Hangeerh` in {pr}`5138`
* Use attribute access for the global config in manim/scene by {user}`Q9Kawaii` in {pr}`5142`

#### Type Hints 📝
* Fix typing in `logo.py` by {user}`Kayyz1603` in {pr}`5112`
* Fix return type of invert_image by {user}`GniLudio` in {pr}`5132`

#### Other Changes
* Prepare preview prerelease: `v0.30.0a1` by {user}`behackl` in {pr}`5154`

### New Contributors
* {user}`manan21bhalla` made their first contribution in {pr}`5098`
* {user}`Mist-wu` made their first contribution in {pr}`5125`
* {user}`Timotty` made their first contribution in {pr}`5058`
* {user}`Hangeerh` made their first contribution in {pr}`5138`
* {user}`Q9Kawaii` made their first contribution in {pr}`5142`
* {user}`b84330808` made their first contribution in {pr}`5103`

**Full Changelog**: [Compare view](https://github.com/ManimCommunity/manim/compare/v0.22.1...v0.30.0a1)
