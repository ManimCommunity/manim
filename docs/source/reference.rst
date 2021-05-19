Reference Manual
================

This reference manual details modules, functions, and variables included in
Manim, describing what they are and what they do.  For learning how to use
Manim, see :doc:`tutorials`.  For a list of changes since the last release, see
the :doc:`changelog`.

.. warning:: The pages linked to here are currently a work in progress.

.. currentmodule:: manim

********
Mobjects
********

.. autosummary::
   :toctree: reference

   ~mobject.changing
   ~mobject.coordinate_systems
   ~mobject.frame
   ~mobject.functions
   ~mobject.geometry
   ~mobject.graph
   ~mobject.logo
   ~mobject.matrix
   ~mobject.mobject
   ~mobject.mobject_update_utils
   ~mobject.number_line
   ~mobject.numbers
   ~mobject.polyhedra
   ~mobject.probability
   ~mobject.shape_matchers
   ~mobject.three_d_utils
   ~mobject.three_dimensions
   ~mobject.value_tracker
   ~mobject.vector_field
   ~mobject.svg.brace
   ~mobject.svg.code_mobject
   ~mobject.svg.style_utils
   ~mobject.svg.svg_path
   ~mobject.svg.svg_mobject
   ~mobject.svg.tex_mobject
   ~mobject.svg.text_mobject
   ~mobject.types.image_mobject
   ~mobject.types.point_cloud_mobject
   ~mobject.types.vectorized_mobject

**Mobject** Inheritance Graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. inheritance-diagram::
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
   manim.mobject.three_d_utils
   manim.mobject.three_dimensions
   manim.mobject.value_tracker
   manim.mobject.vector_field
   manim.mobject.svg.brace
   manim.mobject.svg.code_mobject
   manim.mobject.svg.style_utils
   manim.mobject.svg.svg_path
   manim.mobject.svg.svg_mobject
   manim.mobject.svg.tex_mobject
   manim.mobject.svg.text_mobject
   manim.mobject.types.image_mobject
   manim.mobject.types.point_cloud_mobject
   manim.mobject.types.vectorized_mobject
   :parts: 1
   :top-classes: manim.mobject.mobject.Mobject



******
Scenes
******

.. autosummary::
   :toctree: reference

   ~scene.graph_scene
   ~scene.moving_camera_scene
   ~scene.reconfigurable_scene
   ~scene.sample_space_scene
   ~scene.scene
   ~scene.scene_file_writer
   ~scene.three_d_scene
   ~scene.vector_space_scene
   ~scene.zoomed_scene

**Scene** Inheritance Graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. inheritance-diagram::
   manim.scene.graph_scene
   manim.scene.moving_camera_scene
   manim.scene.reconfigurable_scene
   manim.scene.sample_space_scene
   manim.scene.scene
   manim.scene.scene_file_writer
   manim.scene.three_d_scene
   manim.scene.vector_space_scene
   manim.scene.zoomed_scene
   :parts: 1
   :top-classes: manim.scene.scene.Scene, manim.scene.scene.RerunSceneHandler


**********
Animations
**********

.. autosummary::
   :toctree: reference

   ~animation.animation
   ~animation.composition
   ~animation.creation
   ~animation.fading
   ~animation.growing
   ~animation.indication
   ~animation.movement
   ~animation.numbers
   ~animation.rotation
   ~animation.transform
   ~animation.transform_matching_parts
   ~animation.update

**Animation** Inheritance Graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
   manim.animation.transform
   manim.animation.transform_matching_parts
   manim.animation.update
   :parts: 1
   :top-classes: manim.animation.animation.Animation
 

*******
Cameras
*******

.. autosummary::
   :toctree: reference

   ~camera.camera
   ~camera.mapping_camera
   ~camera.moving_camera
   ~camera.multi_camera
   ~camera.three_d_camera

**Camera** Inheritance Graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. inheritance-diagram::
   manim.camera.camera
   manim.camera.mapping_camera
   manim.camera.moving_camera
   manim.camera.multi_camera
   manim.camera.three_d_camera
   :parts: 1
   :top-classes: manim.camera.camera.Camera, manim.mobject.mobject.Mobject

*************
Configuration
*************

.. autosummary::
   :toctree: reference

   ~_config
   ~_config.utils
   ~_config.logger_utils


*********
Utilities
*********

.. autosummary::
   :toctree: reference

   ~utils.bezier
   ~utils.color
   ~utils.config_ops
   ~utils.deprecation
   ~utils.hashing
   ~utils.ipython_magic
   ~utils.images
   ~utils.iterables
   ~utils.paths
   ~utils.rate_functions
   ~utils.simple_functions
   ~utils.sounds
   ~utils.space_ops
   ~utils.strings
   ~utils.tex
   ~utils.tex_templates
   ~utils.tex_file_writing


*************
Other modules
*************

.. autosummary::
   :toctree: reference

   constants
   container       
