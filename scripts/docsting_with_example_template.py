# see more documentation guidelines online here: https://github.com/ManimCommunity/manim/wiki/Documentation-guidelines-(WIP)


class SomeClass:
    """Some Description of the Class

    Parameters
    ----------
    scale_factor
        The factor used for scaling.


    Returns
    -------
    :class:`VMobject`
        Returns self.

    Raises
    ------
    TypeError
        If one element of the list is not an instance of VMobject


    See Also
    --------
    :class:`ShowCreation`, :class:`~.ShowPassingFlash`


    .. tip::
        This is currently only possible for class:`~.Text` and not for class:`~.MathTex`

    Examples
    --------
    .. manim:: AddTextLetterByLetterScene
        :save_last_frame:
        class AddTextLetterByLetterScene(Scene):
            def construct(self):
                t = Text("Hello")
                self.play(AddTextLetterByLetter(t))
                self.wait()
    """
