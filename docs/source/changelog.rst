Contributors
============

A total of 30 people contributed to this
release. People with a '+' by their names authored a patch for the first
time.

* Abel Aebker +
* Abhijith Muthyala
* AntonBallmaier +
* Aron
* Benjamin Hackl
* Bogdan Stăncescu +
* Darylgolden
* Devin Neal
* GameDungeon +
* Hugues Devimeux
* Jason Villanueva
* Kapil Sachdeva
* KingWampy
* Lionel Ray +
* Mark Miller
* Mohammad Al-Fetyani +
* Naveen M K
* Niklas Dewally +
* Oliver +
* Roopesh +
* Seb Pearce +
* aebkea +
* friedkeenan
* hydrobeam +
* kolibril13
* tfglynn +


The patches included in this release have been reviewed by
the following contributors.

* Abel Aebker
* Abhijith Muthyala
* Benjamin Hackl
* Devin Neal
* Jason Villanueva
* KingWampy
* Lionel Ray
* Mark Miller
* Naveen M K
* Oliver
* vector67

Pull requests merged
====================

A total of 17 pull requests were merged for this release.

Highlight
---------

* `#1075 <https://github.com/ManimCommunity/manim/pull/1075>`__: Add OpenGL Renderer
   Adds an OpenGLRenderer, OpenGLCamera, OpenGL-enabled Mobjects, and a `--use_opengl_renderer` flag. When this flag is passed, you can pass the `-p` flag to preview animations, the `-w` flag to generate video, and the `-q` flag to specify render quality. If you don't pass either the `-p` or the `-w` flag, nothing will happen. Scenes rendered with the OpenGL renderer must *only* use OpenGL-enabled Mobjects.
New feature
-----------

* `#1107 <https://github.com/ManimCommunity/manim/pull/1107>`__: Added :class:`~.Unwrite` animation class to complement :class:`~.Write`
   Added :class:`Unwrite` which inherits from :class:`~.Write`. It automatically reverses the animation of :class:`~.Write` by passing the reversed rate function, but it also takes an additional boolean parameter `reverse` which, if `False`, renders the animation from left to right (assuming text oriented in the usual way), but if `True`, it renders right to left.
* `#1085 <https://github.com/ManimCommunity/manim/pull/1085>`__: Added Angle/RightAngle classes for two intersecting lines
   ``Angle`` and ``RightAngle`` both take two lines as input. If they intersect, or share a common vertex, an angle is drawn between them. Users can customize the look of the angle and also use a dotted right angle.
Enhancement
-----------

* `#718 <https://github.com/ManimCommunity/manim/pull/718>`__: Rotating the numbers in y axis
   In Axes, the y axis will be rotated 90deg but the numbers are
   also rotated and shouldn't be. Fixes this issue.
* `#1070 <https://github.com/ManimCommunity/manim/pull/1070>`__: Raise FileNotFoundError when unable to locate the .cfg file specified via `--config_file`
   Raising the error will stop script execution and let the user know that there are problems with the `--config_file` location instead of reverting back to the default configuration.
Bug
---

* `#1115 <https://github.com/ManimCommunity/manim/pull/1115>`__: Fixed bugs in :class:`~.OpenGLMobject` and added :class:`ApplyMethod` support 
   Fixed undefined variables and converted :class:`Mobject` to :class:`OpenGLMobject`. Also, fixed assert statement in :class:`ApplyMethod`.
* `#1092 <https://github.com/ManimCommunity/manim/pull/1092>`__: Refactored coordinate_systems.py, fixed bugs, added :class:`~.NumberPlane` test
   The default behavior of :meth:`~.Mobject.rotate` is to rotate about the center of :class:`~.Mobject`. :class:`~.NumberLine` is symmetric about the point at the number 0 only when ``|x_min|`` == ``|x_max|``. Ideally, the rotation should coincide with
   the point at number 0 on the line.

   Added a regression test and additionally fixed some bugs introduced in :pr:`718`.
* `#1078 <https://github.com/ManimCommunity/manim/pull/1078>`__: Removed stray print statements from `__main__.py`
   Uses rich's print traceback instead and fixes an issue in printing the version twice when `manim --version` is called.
* `#1086 <https://github.com/ManimCommunity/manim/pull/1086>`__: Fixed broken line spacing in :class:`~.Text`
   The `line_spacing` kwarg was missing when creating :class:`Text` Mobjects; this adds it.
* `#1083 <https://github.com/ManimCommunity/manim/pull/1083>`__: Corrected the shape of :class:`~.Torus`
   :class:`Torus` draws a surface with an elliptical cross-section when `minor_radius` is different from 1. This PR ensures the cross-section is always a circle.
Deprecation
-----------

* `#1110 <https://github.com/ManimCommunity/manim/pull/1110>`__: Deprecated SmallDot + OpenGLSmallDot
   `SmallDot` isn't necessary and a deprecation warning will be raised. This will be removed in a future release.
Documentation
-------------

* `#1101 <https://github.com/ManimCommunity/manim/pull/1101>`__: Added documentation to :class:`~.Mobject`
   Methods for which documentation was added or improved:
   - :meth:`~.reset_points`
   - :meth:`~.init_colors`
   - :meth:`~.generate_points`
   - :meth:`~.add`
   - :meth:`~.add_to_back`
   - :meth:`~.remove`
   - :meth:`~.copy`
   - :meth:`~.update`
   - :meth:`~.get_time_based_updaters`
   - :meth:`~.has_time_based_updater`
   - :meth:`~.get_updaters`
   - :meth:`~.add_updater`
   - :meth:`~.remove_updater`
   - :meth:`~.clear_updaters`
   - :meth:`~.match_updaters`
   - :meth:`~.suspend_updating`
   - :meth:`~.resume_updating`
   - :meth:`~.apply_to_family`
   - :meth:`~.shift`
   - :meth:`~.scale`
   - :meth:`~.add_background_rectangle`
* `#1088 <https://github.com/ManimCommunity/manim/pull/1088>`__: Added new svg files to documentation and imports
   In particular, SVGPathMobject, VMobjectFromPathstring, and the style_utils functions to manim's namespace.
* `#1076 <https://github.com/ManimCommunity/manim/pull/1076>`__: Improve documentation for GraphScene
   Updated `coords_to_point` and `point_to_coords` under `manim/scene/graph_scene.py` as the dosctring of each function confusingly described the opposite of what it is supposed to do.
Release
-------

* `#1073 <https://github.com/ManimCommunity/manim/pull/1073>`__: Removed "one line summary" from PULL_REQUEST_TEMPLATE.md

Testing
-------

* `#1100 <https://github.com/ManimCommunity/manim/pull/1100>`__: Rewrote test cases to use sys.executable in the command instead of "python"
   Tests would fail due to `capture()` not spawning a subshell in the correct environment, so when python was called, the test would be unable to find necessary packages.
* `#1079 <https://github.com/ManimCommunity/manim/pull/1079>`__: Removed the hardcoded value, `manim`, in `test_version.py`

