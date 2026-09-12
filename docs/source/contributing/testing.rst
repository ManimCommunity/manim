============
Adding Tests
============

If you add a feature or fix a bug in Manim, you should also add tests that
cover the changed behavior. Tests help prevent later changes from breaking
existing functionality.

.. warning::

   The full test suite requires Cairo 1.18 or later. If an older version of
   Cairo is installed, graphical tests that depend on Cairo are skipped
   locally, but they still run in continuous integration (CI).

   If Cairo 1.18 is not available from your package manager, download a recent
   release from the `Cairo releases page
   <https://www.cairographics.org/releases/>`_ and follow the installation
   instructions included with it.

Running the Test Suite
----------------------

Manim uses `pytest <https://docs.pytest.org/>`_ as its test framework. From
the root directory of the repository, run the test suite with:

.. code-block:: bash

   uv run pytest

The following pytest options are particularly useful while developing tests:

``-x``
   Stop after the first failure.

``-s``
   Show output written while the test is running, including Manim's render
   logs.

``--skip_slow``
   Skip tests marked as slow.

``--show_diff``
   Open a visual comparison when a graphical test fails.

You can also run a single file or test by passing its path and node ID. For
example:

.. code-block:: bash

   uv run pytest tests/test_graphical_units/test_geometry.py::test_Circle

To run the doctests embedded in Manim's source code, use:

.. code-block:: bash

   uv run pytest --doctest-modules manim

Types of Tests
--------------

Manim's test suite contains three broad kinds of tests:

Unit tests
   These test individual functions, classes, and other behavior that can be
   checked directly. Most unit tests live in ``tests/module/``.

Graphical tests
   These render a small scene and compare the resulting frame data against a
   committed ``.npz`` control file. A graphical test can compare only the last
   frame or multiple frames from an animation. Graphical tests live in
   ``tests/test_graphical_units/``.

Render and output tests
   These exercise scene rendering, command-line options, caching, sections,
   and generated files. Video comparison tests check metadata and section
   output recorded in committed JSON control files. They do not compare the
   encoded video pixel by pixel because encoding can vary between systems.
   Most of these tests live in ``tests/test_scene_rendering/``.

Test Directory Layout
---------------------

The main testing directories are organized as follows:

::

   tests/
   ├── conftest.py
   ├── control_data/
   │   └── videos_data/
   ├── helpers/
   │   ├── graphical_units.py
   │   └── video_utils.py
   ├── module/
   ├── test_graphical_units/
   │   ├── control_data/
   │   └── test_*.py
   ├── test_scene_rendering/
   └── utils/
       └── video_tester.py

``conftest.py``
   Defines shared fixtures and custom pytest options.

``control_data/``
   Stores expected data used by tests. Video metadata and section layouts are
   stored in ``tests/control_data/videos_data/``. Graphical frame data is
   stored next to the graphical tests in
   ``tests/test_graphical_units/control_data/``.

``helpers/``
   Contains utilities for generating control data.

``module/``
   Contains unit tests grouped to mirror Manim's source modules.

``test_graphical_units/``
   Contains frame-comparison tests and their control data.

``test_scene_rendering/``
   Contains tests that render scenes or exercise output-related behavior.

``utils/``
   Contains internal utilities used by the test suite. Shared pytest fixtures
   belong in a ``conftest.py`` file rather than this directory.

Adding a Unit Test
------------------

Pytest discovers files named ``test_*.py``. Within those files, it discovers
functions named ``test_*`` and classes named ``Test*``. Add unit tests to the
directory that corresponds to the source module being tested; for example,
tests for vectorized mobjects belong in
``tests/module/mobject/types/vectorized_mobject/test_vectorized_mobject.py``.

Keep each test focused on one behavior. When fixing a bug, add a regression
test that fails without the fix and passes with it.

Adding a Graphical Test
-----------------------

Use a graphical test when correctness is best expressed by how a scene looks.
Place it in the appropriate file under ``tests/test_graphical_units/`` and use
the ``frames_comparison`` decorator. Every graphical test module must define
``__module_test__``; this value determines the subdirectory that stores its
control data.

The decorated test must accept a parameter named ``scene``. Use it in the same
way that you would use ``self`` in a scene's ``construct`` method:

.. code-block:: python

   from manim import Circle
   from manim.utils.testing.frames_comparison import frames_comparison

   __module_test__ = "geometry"


   @frames_comparison
   def test_Circle(scene):
       scene.add(Circle())

The decorator can be used with or without parentheses. By default, only the
last rendered frame is compared. Set ``last_frame=False`` when the intermediate
frames of an animation are significant:

.. code-block:: python

   from manim import Circle, Create


   @frames_comparison(last_frame=False)
   def test_CircleCreation(scene):
       scene.play(Create(Circle()))

You can also select a different base scene when necessary:

.. code-block:: python

   from manim import ThreeDScene


   @frames_comparison(base_scene=ThreeDScene)
   def test_ThreeDCircle(scene):
       scene.add(Circle())

See the documentation of ``frames_comparison`` for the complete set of
options.

Generating graphical control data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A new graphical test initially fails because its control data does not exist.
After carefully checking that the rendered result is correct, generate the
control file with ``--set_test``:

.. code-block:: bash

   uv run pytest tests/test_graphical_units/test_geometry.py::test_Circle --set_test -s

For the example above, this writes
``tests/test_graphical_units/control_data/geometry/Circle.npz``. Review the
generated frames before committing them. The frame extraction script converts
the control data into PNG files for inspection:

.. code-block:: bash

   uv run python scripts/extract_frames.py \
       tests/test_graphical_units/control_data/geometry/Circle.npz output

The output directory will contain ``frame0.png``, ``frame1.png``, and so on.
Commit the reviewed ``.npz`` file together with the test.

Adding a Video Comparison Test
------------------------------

Use ``tests.utils.video_tester.video_comparison`` when a test needs to verify
the metadata and section files produced by a rendered video. The decorator
takes the name of a JSON control file and the expected path of the generated
video relative to the media directory:

.. code-block:: python

   import sys

   from manim import capture
   from tests.utils.video_tester import video_comparison


   @video_comparison(
       "SquareToCircleWithlFlag.json",
       "videos/simple_scenes/480p15/SquareToCircle.mp4",
   )
   def test_basic_scene_l_flag(tmp_path, simple_scenes_path):
       command = [
           sys.executable,
           "-m",
           "manim",
           "-ql",
           "--media_dir",
           str(tmp_path),
           str(simple_scenes_path),
           "SquareToCircle",
       ]
       _, err, exit_code = capture(command)
       assert exit_code == 0, err

Here, ``tmp_path`` is pytest's temporary-directory fixture. The test directs
Manim's output there, and the decorator locates the generated video relative
to that directory. Fixtures such as ``simple_scenes_path`` are defined in the
nearest ``conftest.py`` file.

Generating video control data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Render the scene using the same options as the test, inspect the result, and
then pass the generated video path to
``tests.helpers.video_utils.save_control_data_from_video``:

.. code-block:: python

   from pathlib import Path

   from tests.helpers.video_utils import save_control_data_from_video


   save_control_data_from_video(
       Path("<path-to-video>"),
       "SquareToCircleWithlFlag",
   )

This writes
``tests/control_data/videos_data/SquareToCircleWithlFlag.json``. The file
records the movie metadata, section directory layout, and section index used by
the comparison test. Review and commit the JSON control file with the test.

If you have questions, ask on `Discord
<https://www.manim.community/discord/>`_, in your pull request, or in an issue.
