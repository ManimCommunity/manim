============
Adding Tests
============
When adding a new feature, it should always be tested. Tests prevent
manim from breaking at each new feature added by checking if any other
feature has been broken and/or been unintentionally modified.

How Manim Tests
---------------

To conduct our tests, we use ``pytest``. Running ``pytest`` in the root of
the project will start the testing process, and will show if there is
something wrong.

Some useful pytest flags: 
- ``-x``, that will make pytest stop at the first fail,
- ``-s``, that will make pytest display all the print messages (including those during scene generation, like DEBUG messages).
- ``--skip_slow`` will skip the (arbitrarly) slow tests. 
- ``--show_diff`` will show a visual comparison in case an unit test is
failing. 

How it Works
~~~~~~~~~~~~

At the moment there are three type of tests:

#. Unit Tests:

   Basically test for pretty much everything. For example, there a test for
   ``Mobject``, that checks if it can be added to a Scene, etc ..

#. Graphical unit tests:

   Because ``manim`` is a video library, we tests frames. To do so, we take a
   frame of control data for each feature and compare the last frame of the
   feature rendered (in the form of a numpy array). If it matches, the tests
   are successful. If one wants to visually see the what has changed, you can
   use ``--show_diff`` flag along with ``pytest`` to be able to visualize
   what is different.

#. Videos format tests:

   As Manim is a video library, we have to test videos as well. Unfortunalty,
   we can't test directly video content as manim outputs videos that can
   differ slightly from one system to another (for reasons related to
   ffmpeg). As such, we just compare videos configuration values, exported in
   .json.

Architecture
------------

``manim/tests`` directory looks like this:

::

    .
    ├── conftest.py
    ├── control_data
    │   ├── graphical_units_data
    │   │   ├── creation
    │   │   │   ├── DrawBorderThenFillTest.npy
    │   │   │   ├── FadeInFromDownTest.npy
    │   │   │   ├── FadeInFromLargeTest.npy
    │   │   │   ├── FadeInFromTest.npy
    │   │   │   ├── FadeInTest.npy
    │   │   │   ├── ...
    │   │   ├── geometry
    │   │   │   ├── AnnularSectorTest.npy
    │   │   │   ├── AnnulusTest.npy
    │   │   │   ├── ArcBetweenPointsTest.npy
    │   │   │   ├── ArcTest.npy
    │   │   │   ├── CircleTest.npy
    │   │   │   ├── CoordinatesTest.npy
    │   │   │   ├── ...
    │   │   ├── graph
    │   │   │   ├── ...
    |   |   |   | ...
    │   └── videos_data
    │       ├── SquareToCircleWithDefaultValues.json
    │       └── SquareToCircleWithlFlag.json
    ├── helpers
    │   ├── graphical_units.py
    │   ├── __init__.py
    │   └── video_utils.py
    ├── __init__.py
    ├── test_camera.py
    ├── test_config.py
    ├── test_container.py
    ├── test_copy.py
    ├── test_vectorized_mobject.py
    ├── test_graphical_units
    │   ├── conftest.py
    │   ├── __init__.py
    │   ├── test_creation.py
    │   ├── test_geometry.py
    │   ├── test_graph.py
    │   ├── test_indication.py
    │   ├── test_movements.py
    │   ├── test_threed.py
    │   ├── test_transform.py
    │   └── test_updaters.py
    ├── test_logging
    │   ├── basic_scenes.py
    │   ├── expected.txt
    │   ├── testloggingconfig.cfg
    │   └── test_logging.py
    ├── test_scene_rendering
    │   ├── conftest.py
    │   ├── __init__.py
    │   ├── simple_scenes.py
    │   ├── standard_config.cfg
    │   └── test_cli_flags.py
    └── utils
        ├── commands.py
        ├── GraphicalUnitTester.py
        ├── __init__.py
        ├── testing_utils.py
        └── video_tester.py
       ...

The Main Directories
--------------------

- ``control_data/``:

  Here control data is saved. These are generally frames
  that we expect to see. In ``control_data/graphical_units_data/`` are all the
  .npz (represented the last frame) used in graphical unit tests videos, and in
  ``control_data/videos_data/`` some .json used to check videos.

- ``test_graphical_units/``:

  For tests related to visual items that can appear in media
    
- ``test_scene_rendering/``:

  For tests that need to render a scene in a way or another. For example, CLI
  flags (end-to-end tests).

- ``utils/``:

  Useful internal functions used by pytest to test.

  .. Note:: fixtures are not contained here, they are in ``conftest.py``.

- ``helpers/``:

  Helper function for developers to setup graphical/video tests.

Adding a New Test
-----------------

Unit Tests
~~~~~~~~~~

Pytest determines which functions are tests by searching for files whose
names begin with "test\_" and then within those files for functions
beginning with "test" or classes beginning with "Test". These kind of
tests must be in ``tests/`` (e.g. ``tests/test_container.py``).

Graphical Unit Test
~~~~~~~~~~~~~~~~~~~

The test must be written in the correct file and follow the structure
of unit tests.

For example, to test the ``Circle`` VMobject which resides in
``manim/mobject/geometry.py``, add the CircleTest to
``test/test_geometry.py``.

In ``test_geometry.py``:

.. code:: python

    class CircleTest(Scene):
        def construct(self):
            circle = Circle()
            self.play(Animation(circle))

Scene names follow the syntax: ``<thing_to_test>Test``. In the example above,
we are testing whether Circle properly shows up with the generic
``Animation`` and not any specific animation.

.. Note:: 

   If the file already exists, just add to its content. The 
   ``Scene`` will be tested thanks to the ``GraphicalUnitTester`` that lives
   in ``tests/utils/GraphicalUnitTester.py``. Import it with ``from
   ..utils.GraphicalUnitTester import GraphicalUnitTester``.

To test all the scenes in the module, we do the following:

.. code:: python

    @pytest.mark.parametrize("scene_to_test", get_scenes_to_test(__name__), indirect=False)
    def test_scene(scene_to_test, tmpdir, show_diff):
        GraphicalUnitTester(scene_to_test[1], MODULE_NAME, tmpdir).test(show_diff=show_diff)

The first line is a `pytest decorator
<https://docs.pytest.org/en/stable/parametrize.html>`_.
It is used to run a test function several times with different
parameters. Here, we pass in all the scenes as arguments.

.. warning::
  If you run pytest now, you will get a ``FileNotFound`` error. It's because
  you haven't created control data for your test. 

Next, we'll want to create control data for ``CircleTest``. In
``tests/template_generate_graphical_units_data.py``, there exist the
function, ``set_test_scene``, for this purpose.

It will looks like this :

.. code:: python

    class CircleTest(Scene):
        def construct(self):
            circle = Circle()
            self.play(Animation(circle))

    set_test_scene(CircleTest, "geometry") 

``set_test_scene`` takes two parameters : the scene to test, and the
module name. It will automatically generate the control data at the
right place (in this case,
``tests/control_data/graphical_units_data/geometry/CircleTest.npz``).

That's all there is to it. Please make sure to add the control data to git as
soon as it is produced with ``git add <your-control-data.npz>`` but do NOT
include changes to the template script in your pull request so that others
may continue to use the template file
(template\_generate\_graphical\_units\_data.py) will be still available for
others.

Videos tests
~~~~~~~~~~~~

To test videos generated, we use the decorator
``tests.utils.videos_tester.video_comparison``:

.. code:: python

    @video_comparison(
        "SquareToCircleWithlFlag.json", "videos/simple_scenes/480p15/SquareToCircle.mp4"
    )
    def test_basic_scene_l_flag(tmp_path, manim_cfg_file, simple_scenes_path):
        scene_name = "SquareToCircle"
        command = [
            "python",
            "-m",
            "manim",
            simple_scenes_path,
            scene_name,
            "-l",
            "--media_dir",
            str(tmp_path),
        ]
        out, err, exit_code = capture(command)
        assert exit_code == 0, err

.. Note:: ``assert exit*\ code == 0, err`` is used in case of the command fails
to run. The decorator takes two arguments: json name and the path
to where the video should be generated, starting from the ``medias/`` dir.

Note the fixtures here:

- tmp_path is a pytest fixture to get a tmp_path. Manim will output here, according to the flag ``--media_dir``.

- ``manim_cfg_file`` fixture that return a path pointing to ``test_scene_rendering/standard_config.cfg``. It's just to shorten the code, in the case multiple tests need to use this cfg file.

- ``simple_scenes_path`` same as above, except for ``test_scene_rendering/simple_scene.py``

You have to generate a ``.json`` file first to be able to test your video. To
do that, use ``helpers.save_control_data_from_video``.

For instance, a test that will check if the l flag works properly will first
require rendering a video using the -l flag from a scene. Then we will test
(in this case, SquareToCircle), that lives in
``test_scene_rendering/simple_scene.py``. Change directories to ``tests/``,
create a file (e.g. ``create\_data.py``) that you will remove as soon as
you're done. Then run:

.. code:: python

    save_control_data_from_video(<path-to-video>, "SquareToCircleWithlFlag.json"). 

Running this will save
``control_data/videos_data/SquareToCircleWithlFlag.json``, whoch will
looks like this :

.. code:: json

    {
        "name": "SquareToCircleWithlFlag",
        "config": {
            "codec_name": "h264",
            "width": 854,
            "height": 480,
            "avg_frame_rate": "15/1",
            "duration": "1.000000",
            "nb_frames": "15"
        }
    }

If you have any question don't hesitate to ask on `Discord
<https://discord.gg/mMRrZQW>`_, in your pull request, or open an issue.
