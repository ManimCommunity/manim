Upgrading from Manim 0.22
=========================

This page lists what code written for Manim 0.22 needs to change.

Mobject arrays are replaced, not changed in place
-------------------------------------------------

The arrays that mobjects are drawn from can be replaced, but not changed in place:
``points``, the color arrays of vectorized mobjects (``fill_rgbas``,
``stroke_rgbas``, ``background_stroke_rgbas``) and their ``sheen_direction``, and
the ``pixel_array`` of an :class:`.ImageMobject`. Renderers keep track of which
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
     - The attributes themselves (``mob.points``, ``mob.fill_rgbas``, ...).
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
- They are lit by their ``material`` (:class:`.PbrMaterial` by default) under a
  :class:`.ThreeDCamera`'s light: faces turned away from the light are darker than
  before, and the surface has highlights. ``material=UnlitMaterial()`` shows its
  colors as they are. Under other cameras, surfaces are not lit.
- A surface without ``checkerboard_colors`` is lit smoothly; with them, each face
  is lit flat.
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
  ``stroke_opacity``); ``shade_in_3d`` has no effect.
- A face is transformed only into a face with as many corners; ``Create`` shows a
  face at once instead of drawing its outline.

.. _upgrading-renderers:

Renderers only draw
-------------------

The Cairo and OpenGL renderers only draw: the :class:`.Manager` runs the scene,
owns the animation clock and the output files, and hands the renderer what to draw
in each frame. This section lists what changed for code that used the renderers,
cameras or scene classes directly.

Removed scene classes and camera options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``SpecialThreeDScene`` is gone. It could not be constructed (it read a
  renderer setting that no longer existed); use :class:`.ThreeDScene` and create
  the axes and spheres it offered directly.
- The ``exponential_projection`` argument and attribute of :class:`.ThreeDCamera`
  are gone. Only ``SpecialThreeDScene`` used them, and the OpenGL renderer
  ignored them. Points are projected with the factor
  ``focal_distance / (focal_distance - z)``.
