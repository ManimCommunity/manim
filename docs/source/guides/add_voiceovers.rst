###########################
Adding Voiceovers to Videos
###########################

Creating a full-fledged video with voiceovers is a bit more involved than
creating purely visual Manim scenes. One has to use `a video editing
program <https://en.wikipedia.org/wiki/List_of_video_editing_software>`__
to add the voiceovers after the video has been rendered. This process
can be difficult and time-consuming, since it requires a lot of planning
and preparation.

To ease the process of adding voiceovers to videos, we have created
`Manim Voiceover <https://voiceover.manim.community>`__, a plugin
that lets you add voiceovers to scenes directly in Python. To install it, run

.. code-block:: bash

    pip install "manim-voiceover[azure,gtts]"

Visit `the installation page <https://voiceover.manim.community/en/latest/installation.html>`__
for more details on how to install Manim Voiceover.

Basic Usage
###########

Manim Voiceover lets you ...

- Add voiceovers to Manim videos directly in Python, without having to use a video editor.
- Record voiceovers with your microphone during rendering through a simple command line interface.
- Develop animations with auto-generated AI voices from various free and proprietary services.

It provides a very simple API that lets you specify your voiceover script
and then record it during rendering:

.. code-block:: python

    from manim import *
    from manim_voiceover import VoiceoverScene
    from manim_voiceover.services.recorder import RecorderService


    # Simply inherit from VoiceoverScene instead of Scene to get all the
    # voiceover functionality.
    class RecorderExample(VoiceoverScene):
        def construct(self):
            # You can choose from a multitude of TTS services,
            # or in this example, record your own voice:
            self.set_speech_service(RecorderService())

            circle = Circle()

            # Surround animation sections with with-statements:
            with self.voiceover(text="This circle is drawn as I speak.") as tracker:
                self.play(Create(circle), run_time=tracker.duration)
                # The duration of the animation is received from the audio file
                # and passed to the tracker automatically.

            # This part will not start playing until the previous voiceover is finished.
            with self.voiceover(text="Let's shift it to the left 2 units.") as tracker:
                self.play(circle.animate.shift(2 * LEFT), run_time=tracker.duration)

To get started with Manim Voiceover,
visit the `Quick Start Guide <https://voiceover.manim.community/en/latest/quickstart.html>`__.

Visit the `Example Gallery <https://voiceover.manim.community/en/latest/examples.html>`__
to see some examples of Manim Voiceover in action.
