Colors, materials and lights
============================

What a mobject looks like comes from three things: the *paints* that color its
areas and lines, the *material* that says whether and how light falls on it, and
the *lights* of the scene.

This page explains each and how they work together.

Paints
------

A :class:`.VMobject` has three paints: ``fill`` (its area), ``stroke`` (its
outline) and ``background_stroke`` (an outline drawn behind the fill). Each can be
one of these (:mod:`manim.appearance.paint`):

:class:`.SolidColor`
    One color with an opacity.
:class:`.LinearGradient`
    Colors evenly spaced along a direction.
:class:`.ColorField`
    Colors that lie in the scene instead of on the mobject (see below).

:meth:`~.VMobject.set_fill`, :meth:`~.VMobject.set_stroke` and
:meth:`~.VMobject.set_color` make paints from colors: one color gives a solid
paint, several give a gradient. They also take a paint directly.

Colors change only the colors and keep the opacity the mobject has: a new
:class:`.Circle` has no fill, so ``set_fill(RED)`` leaves it transparent until an
``opacity`` is given. A paint brings its own opacity (1 unless it says otherwise),
so ``set_fill(SolidColor(RED))`` fills the circle.

.. note::

    Paints are immutable: assign a new one to change a mobject's colors.

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

Color fields
~~~~~~~~~~~~~

A :class:`.ColorField` is a grid of colors placed in the scene. A mobject
painted with a field shows the field's color wherever the mobject is drawn.
Moving, trimming or transforming the mobject shows other colors, and a moving
camera keeps the colors where they are in the scene.

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

.. note::

    :class:`.StreamLines` color their lines with a color field, by the strength of
    the vector field.

Color fields can only be drawn under a flat camera, in a plane parallel to the
frame, and not on mobjects fixed in the frame.

Materials
---------

A material (:mod:`manim.appearance.material`) describes how a mobject responds to
light:

:class:`.UnlitMaterial`
    The mobject shows its colors as they are.
:class:`.PbrMaterial`
    The mobject is lit as a physical surface by the lights of the scene. Its
    ``roughness``, ``metallic`` and ``emissive`` parameters are those of glTF 2.0.

A material holds no color: the color of a lit surface comes from its paint,
which plays the role of the base color (albedo) of a material in glTF and other
3D engines. A mobject has up to three paints but one material, so changing or
animating a color never touches the material, and mobjects of any color can
share one.

The rule for the default is: **what has a surface in space is lit, what is drawn
is not.** Meshes and surfaces (:class:`.Sphere`, :class:`.Surface`, the faces of
:class:`.Cube` and the polyhedra) and :class:`.ThreeDVMobject` have a
:class:`.PbrMaterial`. Everything else (text, lines, axes, plots, flat shapes, point
clouds and images) has an :class:`.UnlitMaterial`. Mobjects fixed in the frame are
never lit.

Use :meth:`~.VMobject.set_material` to change it, for example to light a flat shape
in a 3D scene or to show a surface's colors unshaded. A lit vectorized mobject must
be flat and have a solid fill.

Under a 3D camera, mobjects are drawn in the order they are in the scene, with
two exceptions:

- Meshes hide whatever is behind them (``depth_test``).
- Vectorized mobjects with ``sort_by_depth``
  (:meth:`~.VMobject.set_sort_by_depth`) are drawn from far to near, each by its
  center. Two of them that cross each other are drawn one over the other.

.. manim:: MaterialsExample
    :save_last_frame:
    :renderer: opengl

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

Lit materials are shaded by the scene's lights. A light (:class:`.PointLight`)
is a mobject that draws nothing: move it, animate it, give it updaters and add
new ones to the scene, like any other mobject. Its intensity is in candela, and
its light falls off with the square of the distance.

A :class:`.ThreeDScene` has one light, ``self.light``, above and to the left of
the default camera. It is not one of ``self.mobjects``, so
:meth:`~.Scene.clear` and fading out every mobject leave it on. Set
``self.light = None`` to turn it off. Lights you add with :meth:`~.Scene.add`
are ordinary mobjects of the scene: fading one out turns it off.
``self.ambient_light`` (0.25 by default) is the white light that reaches every
surface everywhere.

**Without lights, nothing is shaded**: lit materials show their base color. A
plain :class:`.Scene` has no light.

.. manim:: LightsExample
    :renderer: opengl

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
- Cairo colors each face of a mesh once, at its middle; OpenGL shades every
  pixel of a mesh. Both color a lit vectorized path once.
- The OpenGL renderer takes up to eight lights; more raise
  :class:`~.CapabilityError`.

A renderer that cannot draw something faithfully raises
:class:`~.CapabilityError` (:doc:`writing_a_renderer`).
