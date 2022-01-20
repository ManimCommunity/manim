Reference Manual
================

This reference manual details modules, functions, and variables included in
Manim, describing what they are and what they do.  For learning how to use
Manim, see :doc:`tutorials`.  For a list of changes since the last release, see
the :doc:`changelog`.

.. warning:: The pages linked to here are currently a work in progress.

Inheritance Graphs
------------------

Animations
**********

.. inheritance-diagram::
   manim.animation.animation
   manim.animation.composition
   manim.animation.creation
   manim.animation.fading
   manim.animation.growing
   manim.animation.indication
   manim.animation.movement
   manim.animation.numbers
   manim.animation.rotation
   manim.animation.specialized
   manim.animation.transform
   manim.animation.transform_matching_parts
   manim.animation.update
   :parts: 1
   :top-classes: manim.animation.animation.Animation

Cameras
*******

.. inheritance-diagram::
   manim.camera.camera
   manim.camera.mapping_camera
   manim.camera.moving_camera
   manim.camera.multi_camera
   manim.camera.three_d_camera
   :parts: 1
   :top-classes: manim.camera.camera.Camera, manim.mobject.mobject.Mobject

Mobjects
********

.. inheritance-diagram::
   manim.mobject.boolean_ops
   manim.mobject.changing
   manim.mobject.coordinate_systems
   manim.mobject.frame
   manim.mobject.functions
   manim.mobject.geometry
   manim.mobject.graph
   manim.mobject.logo
   manim.mobject.matrix
   manim.mobject.mobject
   manim.mobject.mobject_update_utils
   manim.mobject.number_line
   manim.mobject.numbers
   manim.mobject.probability
   manim.mobject.shape_matchers
   manim.mobject.table
   manim.mobject.three_d_utils
   manim.mobject.three_dimensions
   manim.mobject.value_tracker
   manim.mobject.vector_field
   manim.mobject.svg.brace
   manim.mobject.svg.code_mobject
   manim.mobject.svg.svg_mobject
   manim.mobject.svg.tex_mobject
   manim.mobject.svg.text_mobject
   manim.mobject.types.image_mobject
   manim.mobject.types.point_cloud_mobject
   manim.mobject.types.vectorized_mobject
   :parts: 1
   :top-classes: manim.mobject.mobject.Mobject

Scenes
******

.. inheritance-diagram::
   manim.scene.moving_camera_scene
   manim.scene.scene
   manim.scene.scene_file_writer
   manim.scene.three_d_scene
   manim.scene.vector_space_scene
   manim.scene.zoomed_scene
   :parts: 1
   :top-classes: manim.scene.scene.Scene, manim.scene.scene.RerunSceneHandler


Module Index
------------

.. toctree::
   :maxdepth: 3

   reference_index/animations
   reference_index/cameras
   reference_index/configuration
   reference_index/mobjects
   reference_index/scenes
   reference_index/utilities_misc
