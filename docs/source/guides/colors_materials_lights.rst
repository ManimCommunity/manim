Colours, materials and lights
=============================

What a mobject looks like comes from three things: the *paints* that colour its
areas and lines, the *material* that says whether and how light falls on it, and
the *lights* of the scene. This page explains each and how they work together.

Paints
------

A :class:`.VMobject` has three paints: ``fill`` (its area), ``stroke`` (its
outline) and ``background_stroke`` (an outline drawn behind the fill). A paint is
an immutable value from :mod:`manim.paint`:

:class:`.SolidColor`
    One colour, with an opacity.
:class:`.LinearGradient`
    Colours evenly spaced along a direction, between the edges of the mobject's
    own bounding box. The direction is ``UL`` unless you give another.
:class:`.ColorField`
    Colours that lie in the scene instead of on the mobject (see below).

:meth:`~.VMobject.set_fill`, :meth:`~.VMobject.set_stroke` and
:meth:`~.VMobject.set_color` make paints from colours: one colour gives a solid
paint, several give a gradient. They also take a paint directly. Because paints
are values, you change a mobject's colours by giving it a new paint; changing a
paint in place is not possible.

.. manim:: PaintsExample
    :save_last_frame:

    class PaintsExample(Scene):
        def construct(self):
            solid = Square().set_fill(BLUE, opacity=0.8)
            default_direction = Square().set_fill([RED, YELLOW], opacity=1)
            sideways = Square()
            sideways.fill = LinearGradient([GREEN, BLUE], direction=RIGHT)
            outlined = Circle().set_stroke([PINK, ORANGE], width=12)
            self.add(
                VGroup(solid, default_direction, sideways, outlined).arrange(
                    RIGHT, buff=0.5
                )
            )

:meth:`~.VMobject.rotate` turns the direction of a gradient with the mobject.
Animations interpolate paints: a gradient fades into another one stop by stop,
and paints with different numbers of stops are aligned first
(:func:`.interpolate_paint`).

Paints of your own subclass :class:`.Paint` and say what renderers draw for
them with :meth:`~.Paint.resolve`, which returns one of the snapshot paints of
:mod:`manim.renderer.snapshot` (:class:`.RSolidColor`, :class:`.RLinearGradient`,
:class:`.RColorField`); :meth:`~.Paint.interpolate` animates them. Every
renderer draws a paint that resolves to these, and raises
:class:`.CapabilityError` for anything else.

Colour fields
~~~~~~~~~~~~~

A :class:`.ColorField` is a grid of colours placed in the scene. A mobject
painted with a field shows the field's colour wherever the mobject is drawn.
Moving, trimming or transforming the mobject shows other colours, and a moving
camera keeps the colours where they are in the scene. :class:`.StreamLines` use
one to colour their lines by the strength of the vector field.

.. manim:: ColorFieldExample

    class ColorFieldExample(Scene):
        def construct(self):
            field = ColorField.from_function(
                lambda point: (abs(point[0]) / 7, 0.4, 1 - abs(point[1]) / 4),
                x_range=(-7, 7),
                y_range=(-4, 4),
                resolution=(140, 80),
            )
            wave = FunctionGraph(np.sin, x_range=(-7, 7)).set_stroke(field, width=12)
            self.play(Create(wave))
            self.play(wave.animate.shift(2 * UP))

The Cairo and OpenGL renderers draw colour fields that lie in a plane parallel to
the frame, under a flat camera. Other cameras and planes, and mobjects fixed in
the frame, raise :class:`~.CapabilityError`.

Materials
---------

A material (:mod:`manim.material`) says how a mobject responds to light:

:class:`.UnlitMaterial`
    The mobject shows its colours as they are.
:class:`.PbrMaterial`
    The mobject is lit as a physical surface by the lights of the scene. Its
    ``roughness`` (from smooth with sharp highlights to rough without any),
    ``metallic`` and ``emissive`` parameters are those of glTF 2.0.

A material holds no colour: the colour of a lit surface comes from its paint,
which plays the role of the base colour (albedo) of a material in glTF and other
3D engines. A mobject has up to three paints but one material, so changing or
animating a colour never touches the material, and mobjects of any colour can
share one.

Materials of your own subclass :class:`.Material`: a frozen dataclass whose fields
are its parameters, with :meth:`~.Material.interpolate` overridden to animate them.
Every renderer raises :class:`.CapabilityError` for a material it does not know
instead of drawing it unlit, so a new material needs a renderer that draws it.

The rule for the default is: **what has a surface in space is lit, what is drawn
is not.** Meshes and surfaces (:class:`.Sphere`, :class:`.Surface`, the faces of
:class:`.Cube` and the polyhedra) and :class:`.ThreeDVMobject` have a
:class:`.PbrMaterial`. Everything else (text, lines, axes, plots, flat shapes, point
clouds and images) has an :class:`.UnlitMaterial`. Mobjects fixed in the frame are
never lit.

Use :meth:`~.VMobject.set_material` to change it, for example to light a flat shape
in a 3D scene or to show a surface's colours unshaded. A lit vectorized mobject
is drawn in one colour per path, from the direction its plane faces, so it must be
flat and have a solid fill; its outline is not lit.

Under a 3D camera, mobjects are drawn in scene order unless they are ordered by
depth: meshes hide each other where they overlap (``depth_test``), and vectorized
mobjects with ``sort_by_depth`` (:meth:`~.VMobject.set_sort_by_depth`) are drawn
far to near. ``sort_by_depth`` orders whole paths by their centers and does not
resolve intersecting shapes: one is drawn over the other.

.. manim:: MaterialsExample
    :save_last_frame:

    class MaterialsExample(ThreeDScene):
        def construct(self):
            self.set_camera_orientation(phi=65 * DEGREES, theta=-90 * DEGREES)
            def ball():
                return Sphere(radius=1, resolution=(32, 16)).set_color(BLUE)

            rough = ball().set_material(PbrMaterial(roughness=0.9))
            shiny = ball().set_material(PbrMaterial(roughness=0.3))
            unlit = ball().set_material(UnlitMaterial())
            card = Square(2, fill_color=ORANGE, fill_opacity=1, stroke_width=0)
            card.rotate(PI / 4, RIGHT).rotate(PI / 6, OUT)
            card.set_material(PbrMaterial()).set_sort_by_depth(True)
            for x, mob in zip([-4.5, -1.5, 1.5, 4.5], [rough, shiny, unlit, card]):
                mob.shift(x * RIGHT)
            self.add(rough, shiny, unlit, card)

Lights
------

Lit materials are shaded by the scene's lights. A light (:class:`.PointLight`,
in :mod:`manim.light`) is a mobject that draws nothing: you add it, move it,
animate it and give it updaters like any other mobject. Its intensity is in
candela, and its light falls off with the square of the distance.

A :class:`.ThreeDScene` has one light, ``self.light``, above and to the left of
the default camera. It is not one of ``self.mobjects``, so
:meth:`~.Scene.clear` and fading out every mobject leave it on. Set
``self.light = None`` to turn it off. Lights you add with :meth:`~.Scene.add`
are ordinary mobjects of the scene: fading one out turns it off.
``self.ambient_light`` (0.25 by default) is the white light that reaches every
surface everywhere.

**Without lights, nothing is shaded**: lit materials show their base colour. A
plain :class:`.Scene` has no light.

.. manim:: LightsExample

    class LightsExample(ThreeDScene):
        def construct(self):
            self.set_camera_orientation(phi=70 * DEGREES, theta=-60 * DEGREES)
            self.add(Torus(), Sphere(radius=0.5).set_color(GREY_B))
            self.play(self.light.animate.move_to(4 * RIGHT + 2 * OUT))
            warm = PointLight(4 * LEFT + 2 * OUT, color=ORANGE, intensity=400)
            self.add(warm)
            self.play(warm.animate.move_to(4 * DOWN + 2 * OUT))
            self.play(FadeOut(warm))

What the renderers approximate
------------------------------

The shading follows glTF 2.0 and its ``KHR_lights_punctual`` lights, in linear
RGB. The Cairo and OpenGL renderers approximate it:

- They use a Lambert diffuse term and a normalized Blinn-Phong highlight instead
  of the GGX term of glTF, and no tone mapping.
- Cairo colours each face of a mesh once, at its middle; OpenGL shades every
  pixel of a mesh. Both colour a lit vectorized path once.
- The OpenGL renderer takes up to eight lights; more raise
  :class:`~.CapabilityError`.

A renderer that cannot draw something faithfully raises
:class:`~.CapabilityError` instead of drawing something else
(:doc:`writing_a_renderer`).
