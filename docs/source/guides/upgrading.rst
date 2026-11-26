Upgrading from Manim 0.22
=========================

This page lists what code written for Manim 0.22 needs to change.

Mobject arrays are replaced, not changed in place
-------------------------------------------------

The arrays that mobjects are drawn from can be replaced, but not changed in place:
``points``, the color arrays of vectorized mobjects (``fill_rgbas``,
``stroke_rgbas``, ``background_stroke_rgbas``) and their ``sheen_direction``, and
the ``pixel_array`` of an :class:`.ImageMobject`. Renderers keep track of which

The arrays that mobjects are drawn from are read-only: ``points``, the colour
stops of paints (``mob.fill.rgbas``, see below), and the ``pixel_array`` of an
:class:`.ImageMobject`. Renderers keep track of which
mobjects changed by the assignments to these arrays, so changing one in place
raises numpy's error ``ValueError: assignment destination is read-only``:

.. code-block:: python

    mob.points[0] = ORIGIN  # raises
    mob.points += RIGHT  # raises

Assign a new array instead:

.. code-block:: python

    points = mob.points.copy()
    points[0] = ORIGIN
    mob.points = points

    mob.points = mob.points + RIGHT

The assigned array is copied, so changing it afterwards does not change the
mobject. To give the array another shape, assign a reshaped copy
(``mob.points = mob.points.reshape(...)``); changing ``shape`` or ``dtype`` of
the array itself is not supported.

The same holds for the other properties mobjects are drawn from, such as
``stroke_width`` or ``z_index``. A subclass that sets one of them in its class
body raises a ``TypeError`` as soon as the class is defined:

.. code-block:: python

    class ThickLine(Line):
        stroke_width = 8  # raises TypeError

Set the value in ``__init__`` instead:

.. code-block:: python

    class ThickLine(Line):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.stroke_width = 8

Removed 3D shading options
--------------------------

Shading of vectorized mobjects in 3D is drawn from the mobjects' own fields now. The
options that only served the old shading code are gone, without a deprecation
period; passing them raises ``TypeError``:

.. list-table::
   :header-rows: 1

   * - Removed
     - Use instead
   * - ``ThreeDAxes(light_source=...)``, ``ThreeDAxes(gloss=...)``
     - Nothing: the axes are drawn in their own colors, without a sheen
   * - ``set_shade_in_3d(value, z_index_as_group=True)``, ``z_index_group``
     - ``set_shade_in_3d(value)``; each path is ordered by its own points

Fixed mobjects are marked on the mobjects
-----------------------------------------

Whether a mobject is fixed in the frame or keeps facing the camera is stored on
the mobject and its family: read ``mob.is_fixed_in_frame`` and
``mob.is_fixed_orientation``. :class:`.ThreeDCamera` no longer keeps its own
records, so its ``fixed_in_frame_mobjects`` and ``fixed_orientation_mobjects``
attributes are gone.

.. _upgrading-opengl-mobjects:

The OpenGL mobject classes are removed
--------------------------------------

Earlier versions of Manim had two sets of mobject classes: the common ones
(:class:`.Mobject`, :class:`.VMobject`, ...) and OpenGL ones (``OpenGLMobject``,
``OpenGLVMobject``, ...). With ``--renderer=opengl``, a metaclass swapped the base
classes of every mobject class for their OpenGL counterparts, so the same scene
code ran on different classes depending on the renderer.

Now both renderers draw the same classes. Scenes that only use the classes
exported by ``from manim import *`` keep working; under ``--renderer=opengl`` they
now look like they do with Cairo (strokes, fills, gradients, ``z_index``, cameras).
This section lists what was removed and what to use instead.

Removed classes
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Removed
     - Use instead
   * - ``OpenGLMobject``, ``OpenGLGroup``, ``OpenGLPoint``
     - :class:`.Mobject`, :class:`.Group`, :class:`.Point`
   * - ``OpenGLVMobject``, ``OpenGLVGroup``, ``OpenGLVectorizedPoint``,
       ``OpenGLCurvesAsSubmobjects``, ``OpenGLDashedVMobject``
     - :class:`.VMobject`, :class:`.VGroup`, :class:`.VectorizedPoint`,
       :class:`.CurvesAsSubmobjects`, :class:`.DashedVMobject`
   * - ``OpenGLPMobject``, ``OpenGLPGroup``, ``OpenGLPMPoint``
     - :class:`.PMobject`, :class:`.PGroup`, :class:`.Point`
   * - ``TrueDot``, ``DotCloud``
     - :class:`.Dot`, or :class:`.PMobject` / :class:`.PointCloudDot` for clouds
   * - ``OpenGLImageMobject``
     - :class:`.ImageMobject`
   * - ``OpenGLSurface``
     - :class:`.ParametricSurface`, a smooth lit surface, or :class:`.Surface`,
       its checkerboard-colored version
   * - ``OpenGLTexturedSurface``
     - :class:`.ParametricSurface` with ``texture=``. One texture: the separate
       night-side texture is gone.
   * - ``OpenGLSurfaceMesh``
     - :meth:`.MeshMobject.get_wireframe`
   * - ``OpenGLArc``, ``OpenGLCircle``, ``OpenGLLine``, ... (``opengl_geometry``)
     - :class:`.Arc`, :class:`.Circle`, :class:`.Line`, ...
   * - ``OpenGLCamera``
     - :class:`.Camera`, :class:`.MovingCamera`, :class:`.ThreeDCamera`
       (see :doc:`cameras`)

The modules ``manim.mobject.opengl``, ``manim.mobject.utils``, ``manim.opengl``
and ``manim.utils.opengl`` are gone.

Removed functions and options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``ConvertToOpenGL`` and ``uses_opengl_mobjects()``: subclass the common classes
  directly; they work with both renderers.
- ``get_mobject_class()``, ``get_vectorized_mobject_class()``,
  ``get_point_mobject_class()``: use :class:`.Mobject`, :class:`.VMobject` and
  :class:`.PMobject`.
- The ``use_projection_fill_shaders`` and ``use_projection_stroke_shaders`` config
  options and ``--use_projection_fill_shaders`` / ``--use_projection_stroke_shaders``
  CLI flags.
- Raw GPU meshes: ``Object3D``, ``Mesh`` and ``FullScreenQuad`` (from
  ``manim.opengl``), ``Scene.meshes`` and ``Scene.update_meshes``. Use
  :class:`.MeshMobject` or :class:`.ParametricSurface`, which both renderers draw.
  Custom GLSL shaders in scenes are no longer supported.
- The ``long_lines``, ``should_subdivide_sharp_curves`` and
  ``should_remove_null_curves`` arguments of :class:`.VMobjectFromSVGPath`, which
  only had an effect on the OpenGL classes.

Changed behavior of mobject methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Methods of the OpenGL classes that the common classes do not have are gone. The
most common ones:

.. list-table::
   :header-rows: 1

   * - OpenGL classes
     - Common classes
   * - ``mob.fix_in_frame()``, ``mob.unfix_from_frame()``,
       ``mob.fix_orientation()``
     - The same methods exist on :class:`.Mobject`, and now work on both
       renderers and with every camera.
   * - ``mob.set_color(..., recurse=False)``
     - ``mob.set_color(..., family=False)``
   * - ``mob.data[...]``, ``mob.uniforms[...]``
     - The attributes themselves (``mob.points``, ``mob.fill``, ...).
   * - ``mob.set_color_by_code(glsl)``
     - :meth:`.MeshMobject.set_color_by_xyz_func`, which takes a Python function
       of the points.
   * - ``set_gloss``, ``set_shadow``
     - The ``material`` of :class:`.MeshMobject` (:meth:`.MeshMobject.set_material`):
       :class:`.PbrMaterial` with ``roughness`` and ``metallic`` for lit surfaces,
       :class:`.UnlitMaterial` for flat colors.
   * - ``apply_depth_test``
     - The ``depth_test`` argument and attribute of :class:`.MeshMobject`.
   * - Stroke widths per point
     - One stroke width per :class:`.VMobject`.
   * - ``get_shader_wrapper_list()`` and other shader-wrapper methods
     - None.

Things that look different
~~~~~~~~~~~~~~~~~~~~~~~~~~

- :class:`.Sphere` and :class:`.Torus` have the same default resolution on both
  renderers; on OpenGL it used to be much finer. Use :class:`.ParametricSurface`
  for smooth spheres.
- Under :class:`.ThreeDScene`, lines and other paths are drawn in the order they
  are added, as with Cairo; only :class:`.MeshMobject` takes part in the depth
  test.

Surfaces are meshes
-------------------

:class:`.Surface` and the mobjects built on it (:class:`.Sphere`, :class:`.Dot3D`,
:class:`.Torus`, :class:`.Cone`, :class:`.Cylinder`, :class:`.Line3D`,
:class:`.Arrow3D`) used to be a :class:`.VGroup` of
one :class:`.VMobject` per face. They are now :class:`.ParametricSurface` meshes,
drawn as meshes on OpenGL and as sorted faces on Cairo:

- The faces are not submobjects anymore: ``surface[i]``, iterating over a surface
  and ``list_of_faces`` are gone. Color faces with
  :meth:`~.Surface.set_fill_by_checkerboard`, :meth:`~.Surface.set_fill_by_value`
  or :meth:`.MeshMobject.set_vertex_colors`.
- ``set_fill``, ``set_stroke``, ``set_style``, ``set_color`` and ``set_opacity`` set
  the colors of the faces and of their outlines; meshes can still be put into a
  :class:`.VGroup`.
- The arguments ``should_make_jagged``, ``surface_piece_config`` and
  ``pre_function_handle_to_anchor_scale_factor`` are gone.
- They are lit by their ``material`` (:class:`.PbrMaterial` by default) under the
  scene's lights (:mod:`manim.appearance.light`; a :class:`.ThreeDScene` has one):
  faces turned away from the light are darker than before, and the surface has
  highlights.
  ``material=UnlitMaterial()`` shows its colors as they are. Without lights,
  surfaces are not shaded.
- Surfaces are smooth, in one colour and without outlines by default: they used to
  have a ``[BLUE_D, BLUE_E]`` checkerboard and light grey outlines of width 0.5.
  ``checkerboard_colors=[BLUE_D, BLUE_E]`` and ``stroke_width=0.5`` give that look
  back; with a checkerboard, each face is lit flat.
- :meth:`~.Surface.set_fill_by_value` colours the points of a smooth surface, so the
  colours blend across faces on OpenGL (Cairo draws each face in the mean colour of
  its corners); on a surface with a checkerboard it colours each face.
- :class:`.Line3D` is white by default, as :class:`.Line` and :class:`.Arrow3D`
  are; it was the blue of the surface.
- The end caps of :class:`.Cylinder` (``base_top``, ``base_bottom``) and the base of
  :class:`.Cone` (``base_circle``) are discs of the same kind, lit as the side is,
  not :class:`.Circle` paths.
- The sides of the faces are straight lines between the sampled points (they were
  slightly curved).
- A surface can only be transformed into a mesh with the same triangles: a surface
  of the same resolution and the same kind (smooth, or with a checkerboard). A
  surface and a path cannot be transformed into each other.

The faces of solids are meshes
------------------------------

The faces of :class:`.Cube`, :class:`.Prism` and the polyhedra
(:class:`.Polyhedron` and its subclasses, :class:`.ConvexHull3D`) used to be
:class:`.Square` and :class:`.Polygon` paths shaded with ``shade_in_3d``. They are
now :class:`.PolygonFace` meshes, lit by a ``material`` (:class:`.PbrMaterial` by
default; ``Cube(material=...)`` and ``faces_config={"material": ...}`` set another):

- ``cube[i]`` and ``polyhedron.faces[i]`` are still the faces, in a
  :class:`.VGroup`, and ``set_fill``, ``set_stroke``, ``set_color`` and
  ``set_opacity`` work on them as before. Methods of paths
  (``get_vertices``, ``point_from_proportion``, ...) do not; a face's corners are
  its ``points``.
- ``faces_config`` takes the styling arguments it took
  (``fill_color``, ``fill_opacity``, ``stroke_color``, ``stroke_width``,
  ``stroke_opacity``).
- A face is transformed only into a face with as many corners; ``Create`` shows a
  face at once instead of drawing its outline.

.. _upgrading-renderers:

Renderers only draw
-------------------

The Cairo and OpenGL renderers only draw: the :class:`.Manager` runs the scene,
is responsible for the animation clock and the output files, and hands the
renderer what to draw in each frame. This section lists what changed for code
that used the renderers, cameras or scene classes directly.

Removed scene classes and camera options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``SpecialThreeDScene``: use :class:`.ThreeDScene` and create the axes and
  spheres it offered directly.
- ``ThreeDCamera.exponential_projection``: points are projected with the factor
  ``focal_distance / (focal_distance - z)``.

Changed drawing
~~~~~~~~~~~~~~~

- Both renderers fill the background of a frame with the camera's
  ``background_color`` and ``background_opacity``. The OpenGL renderer used the
  ``background_color`` of the configuration instead.
- An inset (an :class:`.ImageMobjectFromCamera`) shows the scene without other
  insets in it. Before, the Cairo renderer drew the insets inside a
  :class:`.MultiCamera` that was itself shown in an inset. Cameras that show each
  other in their insets are no longer an error.
- Mobjects fixed in the frame (:meth:`.Mobject.fix_in_frame`) are drawn after all
  other mobjects, in scene order and by ``z_index`` among themselves. Earlier, a
  fixed mobject added before another mobject could be covered by it. A fixed mesh
  hides its own triangles, but neither hides nor is hidden by the depth of the
  scene.

Drawing frames
~~~~~~~~~~~~~~

- Renderers draw what they are given:
  :meth:`renderer.render(nodes, config) <.Renderer.render>` draws the render
  snapshots of a frame as its :class:`.RenderConfig` sets them up and returns a
  :class:`.Frame`; :meth:`.Frame.pixels` reads its RGBA pixels. The
  manager's :class:`.FrameBuilder` makes both from the scene's mobjects and
  camera. ``update_frame()``, ``render_mobjects()``, ``get_frame()``,
  ``get_image()`` and ``render(scene, time)`` of the renderers are gone; use
  :meth:`.Scene.get_image` for an image of the scene's current state.
- Renderers keep no camera: ``renderer.camera`` and the ``camera`` and
  ``camera_class`` arguments of :class:`.CairoRenderer` and
  :class:`.OpenGLRenderer` are gone. The scene's camera is ``scene.camera``.
- :class:`.OpenGLRenderer` takes its options as arguments (``wireframe``,
  ``anti_alias_width``); the scene passes ``config.enable_wireframe`` when it
  creates one. Both options are part of the play cache's key.
- Frames are drawn at the size the scene was created with
  (``scene.session_spec.pixel_size``); renderers have no pixel size of their own.

Removed renderer methods
~~~~~~~~~~~~~~~~~~~~~~~~

Renderers only draw (see :class:`.Renderer`). These methods are gone:

.. list-table::
   :header-rows: 1

   * - Removed
     - Use instead
   * - ``renderer.file_writer``
     - ``scene.manager.file_writer`` (or ``scene._get_manager().file_writer``)
   * - ``renderer.play(scene, ...)``
     - :meth:`.Scene.play`
   * - ``renderer.scene_finished(scene)``
     - Nothing: :meth:`.Manager.render` finishes the movie and saves the last
       frame.
   * - ``renderer.freeze_current_frame(duration)``, ``renderer.add_frame(...)``
     - :meth:`.Scene.pause` (a frozen :meth:`.Scene.wait`)
   * - ``renderer.update_skipping_status()``
     - Nothing: each play checks whether it is part of the output.
   * - ``renderer.show_frame(scene)``
     - ``scene.get_image().show()``

Removed camera methods
~~~~~~~~~~~~~~~~~~~~~~

Cameras no longer draw: renderers draw the render snapshots of the mobjects
(:mod:`manim.renderer.snapshot`), and a camera only says where it looks. The
methods through which a camera took part in drawing are gone, so a camera
subclass that overrides one of them no longer changes what is drawn:

.. list-table::
   :header-rows: 1

   * - Removed
     - Use instead
   * - ``transform_points_pre_display``
     - Move or deform the mobjects themselves (for example with an updater or
       ``apply_function``); :meth:`.ThreeDCamera.project_points` still projects
       points as the renderers do
   * - ``get_mobjects_to_display``
     - Nothing: what is drawn is the scene's mobjects; leave a mobject out by
       removing it from the scene
   * - ``get_fill_rgbas``, ``get_stroke_rgbas``, ``modified_rgbas``
     - Set the colors on the mobjects; lighting comes from the scene's lights
   * - ``get_mobjects_indicating_movement``, ``get_view_transform_center``
     - Nothing
   * - ``Mobject.get_z_index_reference_point``
     - Nothing: a 3D camera orders shaded paths by the center of their own
       points

A renderer draws the frames of one scene: passing a renderer that another scene
uses to :class:`.Scene` raises ``RuntimeError``.

The animation clock and counters belong to the :class:`.Manager`:
``renderer.time``, ``renderer.num_plays``, ``renderer.skip_animations`` and
``renderer.animations_hashes`` are gone; use ``scene.time`` or the attributes of
the same names on ``scene.manager``. ``Scene(skip_animations=True)`` now leaves
plays out with any renderer, and the renderers' ``skip_animations`` argument is
gone.

Preview window
~~~~~~~~~~~~~~

- The OpenGL renderer draws every frame offscreen at the scene's size; the
  preview window shows it, scaled to the window. Movies written while
  previewing in a window of another size have the scene's size.
- The preview is reached through a handle: ``scene.manager.preview`` (a
  :class:`.Preview`, or ``None`` without a live preview) has ``show()``,
  ``present(frame)``, ``is_closing`` and ``pressed_keys``. The window opens
  before :meth:`.Scene.setup` when a scene renders with a live preview.
- ``renderer.pressed_keys``, ``renderer.pixel_coords_to_space_coords()``,
  ``renderer.animation_start_time`` and ``renderer.animation_elapsed_time``,
  ``renderer.clear_screen()`` and ``renderer.init_scene()`` are gone; the
  window maps its positions to scene points itself. ``Scene.get_image()`` and
  evaluation never open a window.

Selecting a renderer
~~~~~~~~~~~~~~~~~~~~

- A scene takes a renderer *factory* or the name of a built-in renderer, not a
  renderer: ``Scene(renderer="opengl")`` or
  ``Scene(renderer=OpenGL(anti_alias_width=2))`` with the factories of
  :mod:`manim.renderer.backends` (``Cairo()``, ``OpenGL(wireframe=...,
  anti_alias_width=...)``). Without it, ``config.renderer`` decides, as before.
  Passing a renderer instance raises ``TypeError``. The scene keeps its factory as
  ``scene.backend`` and creates its own renderer from it; two scenes never share
  one.
- ``config.renderer`` and ``--renderer`` stay the list of built-in names; other
  renderers are selected in code with their factory (see
  :class:`.RendererFactory`).
- A renderer creates its resources when it first draws: the protocol has no
  ``open()``, and ``renderer.capabilities`` and ``renderer.cache_identity`` moved
  to the factory. ``Scene.get_image()`` and evaluation draw with a second renderer
  of the scene's factory, kept for a ``with manager:`` block, a render or an
  evaluation and closed with it.
- The graphical test harness takes ``renderer="opengl"`` instead of
  ``renderer_class=OpenGLRenderer``.
- The OpenGL renderer raises :class:`~.CapabilityError` for a camera's
  ``background_image``; it used to leave the image out. Use the Cairo renderer
  for background images.
- Removed from :class:`.OpenGLRenderer`: ``get_frame()``,
  ``get_raw_frame_buffer_object_data()``, ``get_pixel_shape()`` and
  ``get_frame_buffer_object()``; read a frame's pixels with ``frame.pixels()``.

Colours of vectorized mobjects are paints
-----------------------------------------

A :class:`.VMobject` stores its colours as three *paints*: ``mob.fill``, ``mob.stroke``
and ``mob.background_stroke`` (see :mod:`manim.appearance.paint`). A paint is an
immutable value: :class:`.SolidColor` (one colour) or :class:`.LinearGradient` (evenly
spaced colours along a direction, ``UL`` by default, between the edges of the mobject's
own bounding box).

.. code-block:: python

    square.set_fill([RED, BLUE], opacity=1)  # as before: a gradient along UL
    square.fill = LinearGradient([RED, BLUE], direction=RIGHT)
    square.set_stroke(SolidColor(GREEN, opacity=0.5))

:meth:`~.VMobject.set_fill`, :meth:`~.VMobject.set_stroke`,
:meth:`~.VMobject.set_color`, :meth:`~.VMobject.set_opacity` and the getters
(``get_fill_color``, ``get_fill_colors``, ``get_fill_opacities``, ...) work as
before; ``set_fill`` and ``set_stroke`` also take a paint. ``fill_opacity``,
``stroke_opacity``, ``background_stroke_color`` and
``background_stroke_opacity`` are properties over the paints.
:meth:`~.VMobject.rotate` turns the direction of gradients.

:meth:`~.VMobject.get_style` returns the paints themselves, with their opacities,
under ``fill_color``, ``stroke_color`` and ``background_stroke_color``; it no longer
has ``fill_opacity``, ``stroke_opacity`` or ``background_stroke_opacity``. A
subclass that overrides :meth:`~.VMobject.set_style` (as :meth:`~.VMobject.match_style`
calls it with such a style) must not require them.

.. list-table::
   :header-rows: 1

   * - Removed
     - Use instead
   * - ``fill_rgbas``, ``stroke_rgbas``, ``background_stroke_rgbas``,
       ``get_fill_rgbas()``, ``get_stroke_rgbas()``
     - ``mob.fill.rgbas`` (read-only), ``mob.stroke``,
       ``mob.background_stroke``; ``get_fill_colors()``,
       ``get_fill_opacities()``
   * - ``generate_rgbas_array``, ``update_rgbas_array``, ``align_rgbas``
     - :meth:`~.VMobject.set_fill` / :meth:`~.VMobject.set_stroke`;
       :func:`.interpolate_paint` aligns paints of different lengths
   * - ``set_sheen(factor, direction)``, ``sheen_factor=``
     - a two-stop gradient: ``set_fill([color, lighter_color])``; there is no
       automatic lightened colour
   * - ``set_sheen_direction(d)``, ``rotate_sheen_direction``,
       ``sheen_direction=``, ``get_sheen_direction``
     - ``mob.fill = mob.fill.with_direction(d)`` (for a gradient), or
       ``LinearGradient(colors, direction=d)``
   * - ``get_gradient_start_and_end_points``
     - none (renderers compute it)

Colours that lie in the scene: colour fields
--------------------------------------------

``color_using_background_image`` coloured a mobject with an image fixed to the
frame. It is replaced by a third kind of paint, :class:`.ColorField`: a grid of
colours that lies in the scene. A mobject painted with a field shows the field's
colour wherever it is drawn, so moving, trimming or transforming it shows other
colours, and a moving camera keeps them where they are in the scene.

.. code-block:: python

    field = ColorField.from_function(
        lambda point: (abs(point[0]) / 7, 0.5, 1 - abs(point[1]) / 4),
        x_range=(-7, 7),
        y_range=(-4, 4),
        resolution=(140, 80),
    )
    line.set_stroke(field)

:class:`.StreamLines` colour their lines with a field over the region the lines
can reach (the seed ranges widened by ``padding``); in 3D they keep a colour list
per line along its bounding box, an approximation. The Cairo and OpenGL renderers
draw fields in a plane parallel to the frame under a flat camera, and raise
``CapabilityError`` (:mod:`manim.renderer.protocol`) for other cameras or planes,
and for mobjects fixed in the frame.

.. list-table::
   :header-rows: 1

   * - Removed
     - Use instead
   * - ``color_using_background_image(image)``, ``match_background_image``,
       ``get_background_image``, ``background_image=`` of vectorized mobjects
     - ``set_stroke(field)`` / ``set_fill(field)`` with a :class:`.ColorField`
       (in scene coordinates, not fixed to the frame)
   * - ``StreamLines.get_colored_background_image``, ``background_img``
     - the lines' ``stroke`` paint

Lights belong to the scene
--------------------------

Lit materials (:class:`.PbrMaterial`, the default of meshes and surfaces) are shaded by
:class:`.PointLight` mobjects (see :mod:`manim.appearance.light`). A
:class:`.ThreeDScene` has one, ``scene.light``, at the position of the camera's former
light; it is not in ``scene.mobjects``, so :meth:`.Scene.clear` and fading out every
mobject leave it. Move or animate it like any mobject, or add more lights with
:meth:`.Scene.add`. ``scene.ambient_light`` (0.25) is the white light that reaches every
surface. A frame without lights is not shaded: lit materials show their base colour (a
plain :class:`.Scene` has no light).

.. code-block:: python

    class Lit(ThreeDScene):
        def construct(self):
            self.add(Sphere())
            self.play(self.light.animate.move_to(3 * IN))
            self.add(PointLight(2 * UP, color=YELLOW))

.. list-table::
   :header-rows: 1

   * - Removed
     - Use instead
   * - ``ThreeDCamera.light_source``,
       ``ThreeDCamera(light_source_start_point=...)``
     - ``scene.light`` (a :class:`.PointLight`), ``scene.light.move_to(...)``
   * - ``ThreeDCamera(should_apply_shading=False)``
     - ``scene.light = None`` (no shading)
   * - ``ThreeDCamera(shading_factor=...)``
     - none: lit materials follow :class:`.PbrMaterial`'s parameters

Vector mobjects have a material and a depth ordering
----------------------------------------------------

``shade_in_3d`` meant two things: "shade towards the light" and "sort by depth under
a 3D camera". They are now two settings of :class:`.VMobject`:

- ``material`` (:mod:`manim.appearance.material`, ``set_material``):
  :class:`.UnlitMaterial` by default; :class:`.PbrMaterial` lights the path by the
  scene's lights in one colour, from the normal of its plane. A lit path must be flat
  and have a solid fill, otherwise drawing it raises :class:`.CapabilityError`. Its
  stroke is not lit, and mobjects fixed in the frame are never lit.
- ``sort_by_depth`` (``set_sort_by_depth``): under a perspective camera, the path is
  ordered far to near by the center of its own points. This is not a depth test:
  of two intersecting shapes, one is drawn over the other.

:class:`.ThreeDVMobject` has both (``PbrMaterial()`` and ``True``). The axes of
:class:`.ThreeDAxes` are ordered by depth and no longer shaded, so they no longer
darken or brighten along their length.

.. list-table::
   :header-rows: 1

   * - Removed
     - Use instead
   * - ``VMobject(shade_in_3d=True)``, ``set_shade_in_3d(True)``
     - ``set_material(PbrMaterial())`` and ``set_sort_by_depth(True)``, or only the
       one you need
   * - ``manim.utils.color.get_shaded_rgb``, ``manim.mobject.three_d.three_d_utils``
     - none

Other removed names
-------------------

.. list-table::
   :header-rows: 1

   * - Removed
     - Use instead
   * - ``z_index_group`` (set by ``set_shade_in_3d(z_index_as_group=True)``)
     - ``set_sort_by_depth(True)`` orders each path by its own center
   * - ``ThreeDAxes(light_source=...)``
     - none: the axes are not shaded

See :doc:`colors_materials_lights` for how paints, materials and lights work
together.
