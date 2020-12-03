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


    Examples
    --------

    .. manim:: GeometricShapes
        :save_last_frame:

        class GeometricShapes(Scene):
            def construct(self):
                d = Dot()
                c = Circle()
                s = Square()
                t = Triangle()
                d.next_to(c, RIGHT)
                s.next_to(c, LEFT)
                t.next_to(c, DOWN)
                self.add(d, c, s, t)
                """