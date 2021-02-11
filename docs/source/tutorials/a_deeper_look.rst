A deeper look
=============

This document will focus on understanding manim's output files and some of the
main command line flags available.

.. note:: This tutorial picks up where :doc:`quickstart` left of, so please
          read that document before starting this one.

Manim output folders
********************

At this point, you have just executed the following command.

.. code-block:: bash

   $ manim scene.py SquareToCircle -pql

Let's dissect what just happened step by step.  First, this command executes
manim on the file ``scene.py``, which contains our animation code.  Further,
this command tells manim exactly which ``Scene`` to be rendered, in this case
it is ``SquareToCircle``.  This is necessary because a single scene file may
contain more than one scene.  Next, the flag `-p` tells manim to play the scene
once it's rendered, and the `-ql` flag tells manim to render the scene in low
quality.

After the video is rendered, you will see that manim has generated some new
files and the project folder will look as follows.

.. code-block:: bash

   project/
   ├─scene.py
   └─media
     ├─videos
     |  └─scene
     |     └─480p15
     |        ├─SquareToCircle.mp4
     |        └─partial_movie_files
     ├─text
     └─Tex


There are quite a few new files.  The main output is in
``media/videos/scene/480p15/SquareToCircle.mp4``.  By default, the ``media``
folder will contain all of manim's output files.  The ``media/videos``
subfolder contains the rendered videos.  Inside of it, you will find one folder
for each different video quality.  In our case, since we used the ``-l`` flag,
the video was generated at 480 resolution at 15 frames per second from the
``scene.py`` file.  Therefore, the output can be found inside
``media/videos/scene/480p15``.  The additional folders
``media/videos/scene/480p15/partial_movie_files`` as well as ``media/text`` and
``media/Tex`` contain files that are used by manim internally.

You can see how manim makes use of the generated folder structure by executing
the following command,

.. code-block:: bash

   $ manim scene.py SquareToCircle -pqh

The ``-ql`` flag (for low quality) has been replaced by the ``-qh`` flag, for
high quality.  Manim will take considerably longer to render this file, and it
will play it once it's done since we are using the ``-p`` flag.  The output
should look like this:

.. manim:: SquareToCircle3
   :hide_source:
   :quality: high

   class SquareToCircle3(Scene):
       def construct(self):
           circle = Circle()                    # create a circle
           circle.set_fill(PINK, opacity=0.5)   # set color and transparency

           square = Square()                    # create a square
           square.flip(RIGHT)                   # flip horizontally
           square.rotate(-3 * TAU / 8)          # rotate a certain amount

           self.play(ShowCreation(square))      # animate the creation of the square
           self.play(Transform(square, circle)) # interpolate the square into the circle
           self.play(FadeOut(square))           # fade out animation

And the folder structure should look as follows.

.. code-block:: bash

   project/
   ├─scene.py
   └─media
     ├─videos
     | └─scene
     |   ├─480p15
     |   | ├─SquareToCircle.mp4
     |   | └─partial_movie_files
     |   └─1080p60
     |     ├─SquareToCircle.mp4
     |     └─partial_movie_files
     ├─text
     └─Tex

Manim has created a new folder ``media/videos/1080p60``, which corresponds to
the high resolution and the 60 frames per second.  Inside of it, you can find
the new ``SquareToCircle.mp4``, as well as the corresponding
``partial_movie_files``.

When working on a project with multiple scenes, and trying out multiple
resolutions, the structure of the output directories will keep all your videos
organized.

Further, manim has the option to output the last frame of a scene, when adding
the flag ``-s``. This is the fastest option to quickly get a preview of a scene.
The corresponding folder structure looks like this:

.. code-block:: bash

   project/
   ├─scene.py
   └─media
     ├─images
     | └─scene
     |   ├─SquareToCircle.png
     ├─videos
     | └─scene
     |   ├─480p15
     |   | ├─SquareToCircle.mp4
     |   | └─partial_movie_files
     |   └─1080p60
     |     ├─SquareToCircle.mp4
     |     └─partial_movie_files
     ├─text
     └─Tex

Saving the last frame with ``-s`` can be combined with the flags for different
resolutions, e.g. ``-s -ql``, ``-s -qh``




Some command line flags
***********************

When executing the command

.. code-block:: bash

   $ manim scene.py SquareToCircle -pql

it was necessary to specify which ``Scene`` class to render.  This is because a
single file can contain more than one ``Scene`` class.  If your file contains
multiple ``Scene`` classes, and you want to render them all, you can use the
``-a`` flag.

As discussed previously, the ``-ql`` specifies low render quality.  This does
not look very good, but is very useful for rapid prototyping and testing.  The
other options that specify render quality are ``-qm``, ``-qh``, and ``-qk`` for
medium, high, and 4k quality, respectively.

The ``-p`` flag plays the animation once it is rendered.  If you want to open
the file browser at the location of the animation instead of playing it, you
can use the ``-f`` flag.  You can also omit these two flags.

Finally, by default manim will output .mp4 files.  If you want your animations
in .gif format instead, use the ``-i`` flag.  The output files will be in the
same folder as the .mp4 files, and with the same name, but different file
extension.

This was a quick review of some of the most frequent command line flags.  For a
thorough review of all flags available, see :doc:`configuration`.


*****************
Debugging a scene
*****************

You may need to access some attributes of some mobjects, of the scene or of 
the animation, and you may want to see these values at each frame. 

:class:`.SceneDebugger` can do it for you, by enabling the user to display at 
each frame debug information. You can access it by using either ``--debug`` flag, ``debug = True`` in a config file or 
by putting ``config.debug = True`` in your code file.  

By default, the debugger shows some insights such as the frame number, mobjects on the scene, etc: 

.. manim:: DebugWithDefaultValues

    config.debug = True

    class DebugWithDefaultValues(Scene): 
       def construct(self): 
          square = Square()
          self.play(ShowCreation(square))


You can debug attributes by adding their names in these 
three ``set`` ``debug_animation_attributes``, 
``debug_mobjects_attributes`` or 
``debug_scene_attributes`` of :class:`.SceneDebugger`. 

Example: 

.. manim:: DebugWithCustomAttributes

    from manim import debugger

    config.debug = True
    
    debugger.debug_animation_attributes.add("mobject")
    debugger.debug_mobjects_attributes.update(["stroke_opacity","z_index"])

    class DebugWithCustomAttributes(Scene): 
       def construct(self): 
          square = Square()
          self.play(ShowCreation(square))

Spying functions or methods is also possible. You can achieve this by using :meth:`~.SceneDebugger.spy_function`.

If needed, the debugger can artificially call the function at each frame. 
This can be done by using the ``force_call`` parameter (you can additonnaly 
pass function's parameters in ``args`` and ``kwargs`` arguments).

Example : 

.. manim:: SpyFunctionDebug

    from manim import debugger
    
    config.debug = True

    class SpyFunctionDebug(Scene):
       def construct(self):
          square = Square()
          debugger.spy_function(square.get_color, force_call = True)
          self.play(square.animate.set_color(RED))

.. warning::
   Spying inner functions is not yet supported.


You can as well record values that will be displayed on the debug layout with :meth:`~.SceneDebugger.record_value`. 
The usage is similar to :meth:`~.SceneDebugger.spy_function`. 
