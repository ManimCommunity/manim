########################################
Playing a frame sequence (video footage)
########################################

Manim has no video mobject. To show real footage inside a scene, such as a screen
recording, you export it to numbered image files and play those frames back.

There is a natural-looking way to do this that scales badly, and a small amount of
code that does not. This guide covers both, because the difference is large enough
to change how long a render takes.

Why the obvious approach is slow
================================

Building one :class:`~.ImageMobject` per frame and playing the group with
:class:`~.ShowSubmobjectsOneByOne` reads like the right tool, and its behaviour is
correct:

.. code-block:: python

    # Correct output, but every frame stays in the scene.
    frames = Group(*[ImageMobject(p).set_width(6) for p in paths])
    self.play(ShowSubmobjectsOneByOne(frames), run_time=6)

The cost model is what makes this expensive. **The renderer does per-mobject work on
every frame of output.** Each image mobject in the scene is resampled once per frame,
whether it is visible or not, so a group of *N* frames costs *N* resamples per output
frame rather than one. Setting the opacity of the frames you are not showing does not
avoid that work.

.. note::

    On a 159-frame screen recording at 1100x294, played over six seconds, the group
    approach took **200 s** to render where the single-mobject approach below took
    **3.2 s**. Both produced the same six-second video. In one real clip this was the
    difference between an eight-and-a-half-minute draft render and 36 seconds.

Playing the footage with one mobject
====================================

Keep the frames as raw arrays and swap them into a single :class:`~.ImageMobject`.
Only one mobject is ever in the scene, so the per-frame cost stops growing with the
length of the footage.

.. code-block:: python

    import glob

    from manim import *


    class Filmstrip:
        """A frame sequence played through a single ImageMobject."""

        def __init__(self, paths, width):
            frames = [ImageMobject(p) for p in paths]
            shapes = {f.pixel_array.shape for f in frames}
            if len(shapes) != 1:
                # One odd frame would otherwise stretch silently.
                raise ValueError(f"frames differ in size: {sorted(shapes)}")
            self.arrays = [f.pixel_array for f in frames]
            self.mobject = frames[0]
            self.mobject.set_width(width)

        def roll(self, **kwargs):
            return Roll(self, **kwargs)


    class Roll(Animation):
        """Advance a Filmstrip by swapping the displayed pixel array."""

        def __init__(self, strip, rate_func=linear, **kwargs):
            self._arrays = strip.arrays
            super().__init__(strip.mobject, rate_func=rate_func, **kwargs)

        def interpolate_mobject(self, alpha):
            # interpolate_mobject receives the raw alpha: Animation.interpolate
            # does not apply the rate function for you.
            n = len(self._arrays)
            index = min(n - 1, max(0, int(self.rate_func(alpha) * n)))
            if getattr(self.mobject, "_frame_index", None) != index:
                self.mobject._frame_index = index
                self.mobject.pixel_array = self._arrays[index]


    class Footage(Scene):
        def construct(self):
            strip = Filmstrip(sorted(glob.glob("frames/*.png")), width=6)
            self.add(strip.mobject)
            self.play(strip.roll(run_time=6))

Two details worth knowing
=========================

``interpolate_mobject`` is given the raw alpha. :meth:`.Animation.interpolate` does
not apply the rate function before calling it, so an animation that needs easing
must call ``self.rate_func`` itself. Getting this wrong plays the footage on the
wrong curve rather than raising an error.

:class:`~.ImageMobject` records ``orig_alpha_pixel_array`` from the image it was
built with, and :meth:`~.Mobject.set_opacity` reads that cached array. Assigning a
new ``pixel_array`` leaves the cache describing the previous frame, so fading a
rolling filmstrip may not fade as expected. Refresh the cache alongside the swap if
the footage needs to fade.

Preparing the frames
====================

``ffmpeg`` exports a numbered sequence, and cropping to the region of interest keeps
both the file count and the resample cost down:

.. code-block:: bash

    ffmpeg -i recording.mov -vf "crop=1100:294:0:120,fps=12" frames/f%03d.png

Choose a frame rate for the footage rather than matching the render frame rate. A
screen recording read at 12 frames per second is usually legible, and it needs far
fewer files than 60.
