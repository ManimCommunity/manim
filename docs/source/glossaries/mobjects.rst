========
Mobjects
========

.. tab:: List

    .. tab:: With title

        .. tab:: With Source

            .. tab:: With Reference

                .. rubric:: Circle
                    :class: sd-text-center
                .. manim:: CircleScene
                    :save_last_frame:
                    :ref_classes: Circle

                    class CircleScene(Scene):
                        def construct(self) -> None:
                            self.add(
                                Circle(
                                    radius=3.0,
                                )
                            )

                .. rubric:: Square
                    :class: sd-text-center
                .. manim:: SquareScene
                    :save_last_frame:
                    :ref_classes: Square

                    class SquareScene(Scene):
                        def construct(self) -> None:
                            self.add(
                                Square(
                                    side_length=3.0,
                                )
                            )

                .. rubric:: Star
                    :class: sd-text-center
                .. manim:: StarScene
                    :save_last_frame:
                    :ref_classes: Star

                    class StarScene(Scene):
                        def construct(self) -> None:
                            self.add(
                                Star(
                                    outer_radius=3.0,
                                )
                            )

                .. rubric:: Line
                    :class: sd-text-center
                .. manim:: LineScene
                    :save_last_frame:
                    :ref_classes: Line

                    class LineScene(Scene):
                        def construct(self) -> None:
                            self.add(
                                Line(
                                    (-3, 0, 0),
                                    (3, 0, 0),
                                )
                            )

            .. tab:: Without Reference

                .. rubric:: Circle
                    :class: sd-text-center
                .. manim:: CircleScene
                    :save_last_frame:

                    class CircleScene(Scene):
                        def construct(self) -> None:
                            self.add(
                                Circle(
                                    radius=3.0,
                                )
                            )

                .. rubric:: Square
                    :class: sd-text-center
                .. manim:: SquareScene
                    :save_last_frame:

                    class SquareScene(Scene):
                        def construct(self) -> None:
                            self.add(
                                Square(
                                    side_length=3.0,
                                )
                            )

                .. rubric:: Star
                    :class: sd-text-center
                .. manim:: StarScene
                    :save_last_frame:

                    class StarScene(Scene):
                        def construct(self) -> None:
                            self.add(
                                Star(
                                    outer_radius=3.0,
                                )
                            )

                .. rubric:: Line
                    :class: sd-text-center
                .. manim:: LineScene
                    :save_last_frame:

                    class LineScene(Scene):
                        def construct(self) -> None:
                            self.add(
                                Line(
                                    (-3, 0, 0),
                                    (3, 0, 0),
                                )
                            )

        .. tab:: Without Source

            .. tab:: Without Reference

                .. rubric:: Circle
                    :class: sd-text-center
                .. manim:: CircleScene
                    :save_last_frame:
                    :hide_source:

                    class CircleScene(Scene):
                        def construct(self) -> None:
                            self.add(
                                Circle(
                                    radius=3.0,
                                )
                            )

                .. rubric:: Square
                    :class: sd-text-center
                .. manim:: SquareScene
                    :save_last_frame:
                    :hide_source:

                    class SquareScene(Scene):
                        def construct(self) -> None:
                            self.add(
                                Square(
                                    side_length=3.0,
                                )
                            )

                .. rubric:: Star
                    :class: sd-text-center
                .. manim:: StarScene
                    :save_last_frame:
                    :hide_source:

                    class StarScene(Scene):
                        def construct(self) -> None:
                            self.add(
                                Star(
                                    outer_radius=3.0,
                                )
                            )

                .. rubric:: Line
                    :class: sd-text-center
                .. manim:: LineScene
                    :save_last_frame:
                    :hide_source:

                    class LineScene(Scene):
                        def construct(self) -> None:
                            self.add(
                                Line(
                                    (-3, 0, 0),
                                    (3, 0, 0),
                                )
                            )

    .. tab:: Without title

        .. tab:: With Source

            .. tab:: With Reference

                .. manim:: CircleScene
                    :save_last_frame:
                    :ref_classes: Circle

                    class CircleScene(Scene):
                        def construct(self) -> None:
                            self.add(
                                Circle(
                                    radius=3.0,
                                )
                            )

                .. manim:: SquareScene
                    :save_last_frame:
                    :ref_classes: Square

                    class SquareScene(Scene):
                        def construct(self) -> None:
                            self.add(
                                Square(
                                    side_length=3.0,
                                )
                            )

                .. manim:: StarScene
                    :save_last_frame:
                    :ref_classes: Star

                    class StarScene(Scene):
                        def construct(self) -> None:
                            self.add(
                                Star(
                                    outer_radius=3.0,
                                )
                            )

                .. manim:: LineScene
                    :save_last_frame:
                    :ref_classes: Line

                    class LineScene(Scene):
                        def construct(self) -> None:
                            self.add(
                                Line(
                                    (-3, 0, 0),
                                    (3, 0, 0),
                                )
                            )

            .. tab:: Without Reference

                .. manim:: CircleScene
                    :save_last_frame:

                    class CircleScene(Scene):
                        def construct(self) -> None:
                            self.add(
                                Circle(
                                    radius=3.0,
                                )
                            )

                .. manim:: SquareScene
                    :save_last_frame:

                    class SquareScene(Scene):
                        def construct(self) -> None:
                            self.add(
                                Square(
                                    side_length=3.0,
                                )
                            )

                .. manim:: StarScene
                    :save_last_frame:

                    class StarScene(Scene):
                        def construct(self) -> None:
                            self.add(
                                Star(
                                    outer_radius=3.0,
                                )
                            )

                .. manim:: LineScene
                    :save_last_frame:

                    class LineScene(Scene):
                        def construct(self) -> None:
                            self.add(
                                Line(
                                    (-3, 0, 0),
                                    (3, 0, 0),
                                )
                            )

        .. tab:: Without Source

            .. tab:: Without Reference

                .. manim:: CircleScene
                    :save_last_frame:
                    :hide_source:

                    class CircleScene(Scene):
                        def construct(self) -> None:
                            self.add(
                                Circle(
                                    radius=3.0,
                                )
                            )

                .. manim:: SquareScene
                    :save_last_frame:
                    :hide_source:

                    class SquareScene(Scene):
                        def construct(self) -> None:
                            self.add(
                                Square(
                                    side_length=3.0,
                                )
                            )

                .. manim:: StarScene
                    :save_last_frame:
                    :hide_source:

                    class StarScene(Scene):
                        def construct(self) -> None:
                            self.add(
                                Star(
                                    outer_radius=3.0,
                                )
                            )

                .. manim:: LineScene
                    :save_last_frame:
                    :hide_source:

                    class LineScene(Scene):
                        def construct(self) -> None:
                            self.add(
                                Line(
                                    (-3, 0, 0),
                                    (3, 0, 0),
                                )
                            )

.. tab:: Grid

    .. tab:: Items

        .. tab:: With title

            .. tab:: With Source

                .. tab:: With Reference

                    .. grid:: 2
                        :gutter: 1

                        .. grid-item::

                            .. rubric:: Circle
                                :class: sd-text-center
                            .. manim:: CircleScene
                                :save_last_frame:
                                :ref_classes: Circle

                                class CircleScene(Scene):
                                    def construct(self) -> None:
                                        self.add(
                                            Circle(
                                                radius=3.0,
                                            )
                                        )

                        .. grid-item::

                            .. rubric:: Square
                                :class: sd-text-center
                            .. manim:: SquareScene
                                :save_last_frame:
                                :ref_classes: Square

                                class SquareScene(Scene):
                                    def construct(self) -> None:
                                        self.add(
                                            Square(
                                                side_length=3.0,
                                            )
                                        )

                        .. grid-item::

                            .. rubric:: Star
                                :class: sd-text-center
                            .. manim:: StarScene
                                :save_last_frame:
                                :ref_classes: Star

                                class StarScene(Scene):
                                    def construct(self) -> None:
                                        self.add(
                                            Star(
                                                outer_radius=3.0,
                                            )
                                        )

                        .. grid-item::

                            .. rubric:: Line
                                :class: sd-text-center
                            .. manim:: LineScene
                                :save_last_frame:
                                :ref_classes: Line

                                class LineScene(Scene):
                                    def construct(self) -> None:
                                        self.add(
                                            Line(
                                                (-3, 0, 0),
                                                (3, 0, 0),
                                            )
                                        )

                .. tab:: Without Reference

                    .. grid:: 2
                        :gutter: 1

                        .. grid-item::

                            .. rubric:: Circle
                                :class: sd-text-center
                            .. manim:: CircleScene
                                :save_last_frame:

                                class CircleScene(Scene):
                                    def construct(self) -> None:
                                        self.add(
                                            Circle(
                                                radius=3.0,
                                            )
                                        )

                        .. grid-item::

                            .. rubric:: Square
                                :class: sd-text-center
                            .. manim:: SquareScene
                                :save_last_frame:

                                class SquareScene(Scene):
                                    def construct(self) -> None:
                                        self.add(
                                            Square(
                                                side_length=3.0,
                                            )
                                        )

                        .. grid-item::

                            .. rubric:: Star
                                :class: sd-text-center
                            .. manim:: StarScene
                                :save_last_frame:

                                class StarScene(Scene):
                                    def construct(self) -> None:
                                        self.add(
                                            Star(
                                                outer_radius=3.0,
                                            )
                                        )

                        .. grid-item::

                            .. rubric:: Line
                                :class: sd-text-center
                            .. manim:: LineScene
                                :save_last_frame:

                                class LineScene(Scene):
                                    def construct(self) -> None:
                                        self.add(
                                            Line(
                                                (-3, 0, 0),
                                                (3, 0, 0),
                                            )
                                        )

            .. tab:: Without Source

                .. tab:: Without Reference

                    .. grid:: 2
                        :gutter: 1

                        .. grid-item::

                            .. rubric:: Circle
                                :class: sd-text-center
                            .. manim:: CircleScene
                                :save_last_frame:
                                :hide_source:

                                class CircleScene(Scene):
                                    def construct(self) -> None:
                                        self.add(
                                            Circle(
                                                radius=3.0,
                                            )
                                        )

                        .. grid-item::

                            .. rubric:: Square
                                :class: sd-text-center
                            .. manim:: SquareScene
                                :save_last_frame:
                                :hide_source:

                                class SquareScene(Scene):
                                    def construct(self) -> None:
                                        self.add(
                                            Square(
                                                side_length=3.0,
                                            )
                                        )

                        .. grid-item::

                            .. rubric:: Star
                                :class: sd-text-center
                            .. manim:: StarScene
                                :save_last_frame:
                                :hide_source:

                                class StarScene(Scene):
                                    def construct(self) -> None:
                                        self.add(
                                            Star(
                                                outer_radius=3.0,
                                            )
                                        )

                        .. grid-item::

                            .. rubric:: Line
                                :class: sd-text-center
                            .. manim:: LineScene
                                :save_last_frame:
                                :hide_source:

                                class LineScene(Scene):
                                    def construct(self) -> None:
                                        self.add(
                                            Line(
                                                (-3, 0, 0),
                                                (3, 0, 0),
                                            )
                                        )

        .. tab:: Without title

            .. tab:: With Source

                .. tab:: With Reference

                    .. grid:: 2
                        :gutter: 1

                        .. grid-item::

                            .. manim:: CircleScene
                                :save_last_frame:
                                :ref_classes: Circle

                                class CircleScene(Scene):
                                    def construct(self) -> None:
                                        self.add(
                                            Circle(
                                                radius=3.0,
                                            )
                                        )

                        .. grid-item::

                            .. manim:: SquareScene
                                :save_last_frame:
                                :ref_classes: Square

                                class SquareScene(Scene):
                                    def construct(self) -> None:
                                        self.add(
                                            Square(
                                                side_length=3.0,
                                            )
                                        )

                        .. grid-item::

                            .. manim:: StarScene
                                :save_last_frame:
                                :ref_classes: Star

                                class StarScene(Scene):
                                    def construct(self) -> None:
                                        self.add(
                                            Star(
                                                outer_radius=3.0,
                                            )
                                        )

                        .. grid-item::

                            .. manim:: LineScene
                                :save_last_frame:
                                :ref_classes: Line

                                class LineScene(Scene):
                                    def construct(self) -> None:
                                        self.add(
                                            Line(
                                                (-3, 0, 0),
                                                (3, 0, 0),
                                            )
                                        )

                .. tab:: Without Reference

                    .. grid:: 2
                        :gutter: 1

                        .. grid-item::

                            .. manim:: CircleScene
                                :save_last_frame:

                                class CircleScene(Scene):
                                    def construct(self) -> None:
                                        self.add(
                                            Circle(
                                                radius=3.0,
                                            )
                                        )

                        .. grid-item::

                            .. manim:: SquareScene
                                :save_last_frame:

                                class SquareScene(Scene):
                                    def construct(self) -> None:
                                        self.add(
                                            Square(
                                                side_length=3.0,
                                            )
                                        )

                        .. grid-item::

                            .. manim:: StarScene
                                :save_last_frame:

                                class StarScene(Scene):
                                    def construct(self) -> None:
                                        self.add(
                                            Star(
                                                outer_radius=3.0,
                                            )
                                        )

                        .. grid-item::

                            .. manim:: LineScene
                                :save_last_frame:

                                class LineScene(Scene):
                                    def construct(self) -> None:
                                        self.add(
                                            Line(
                                                (-3, 0, 0),
                                                (3, 0, 0),
                                            )
                                        )

            .. tab:: Without Source

                .. tab:: Without Reference

                    .. grid:: 2
                        :gutter: 1

                        .. grid-item::

                            .. manim:: CircleScene
                                :save_last_frame:
                                :hide_source:

                                class CircleScene(Scene):
                                    def construct(self) -> None:
                                        self.add(
                                            Circle(
                                                radius=3.0,
                                            )
                                        )

                        .. grid-item::

                            .. manim:: SquareScene
                                :save_last_frame:
                                :hide_source:

                                class SquareScene(Scene):
                                    def construct(self) -> None:
                                        self.add(
                                            Square(
                                                side_length=3.0,
                                            )
                                        )

                        .. grid-item::

                            .. manim:: StarScene
                                :save_last_frame:
                                :hide_source:

                                class StarScene(Scene):
                                    def construct(self) -> None:
                                        self.add(
                                            Star(
                                                outer_radius=3.0,
                                            )
                                        )

                        .. grid-item::

                            .. manim:: LineScene
                                :save_last_frame:
                                :hide_source:

                                class LineScene(Scene):
                                    def construct(self) -> None:
                                        self.add(
                                            Line(
                                                (-3, 0, 0),
                                                (3, 0, 0),
                                            )
                                        )

    .. tab:: Cards

        .. tab:: With title

            .. tab:: With Source

                .. tab:: With Reference

                    .. grid:: 2
                        :gutter: 1

                        .. grid-item-card::
                            :class-header: sd-text-center

                            Circle
                            ^^^
                            .. manim:: CircleScene
                                :save_last_frame:
                                :ref_classes: Circle

                                class CircleScene(Scene):
                                    def construct(self) -> None:
                                        self.add(
                                            Circle(
                                                radius=3.0,
                                            )
                                        )

                        .. grid-item-card::
                            :class-header: sd-text-center

                            Square
                            ^^^
                            .. manim:: SquareScene
                                :save_last_frame:
                                :ref_classes: Square

                                class SquareScene(Scene):
                                    def construct(self) -> None:
                                        self.add(
                                            Square(
                                                side_length=3.0,
                                            )
                                        )

                        .. grid-item-card::
                            :class-header: sd-text-center

                            Star
                            ^^^
                            .. manim:: StarScene
                                :save_last_frame:
                                :ref_classes: Star

                                class StarScene(Scene):
                                    def construct(self) -> None:
                                        self.add(
                                            Star(
                                                outer_radius=3.0,
                                            )
                                        )

                        .. grid-item-card::
                            :class-header: sd-text-center

                            Line
                            ^^^
                            .. manim:: LineScene
                                :save_last_frame:
                                :ref_classes: Line

                                class LineScene(Scene):
                                    def construct(self) -> None:
                                        self.add(
                                            Line(
                                                (-3, 0, 0),
                                                (3, 0, 0),
                                            )
                                        )

                .. tab:: Without Reference

                    .. grid:: 2
                        :gutter: 1

                        .. grid-item-card::
                            :class-header: sd-text-center

                            Circle
                            ^^^
                            .. manim:: CircleScene
                                :save_last_frame:

                                class CircleScene(Scene):
                                    def construct(self) -> None:
                                        self.add(
                                            Circle(
                                                radius=3.0,
                                            )
                                        )

                        .. grid-item-card::
                            :class-header: sd-text-center

                            Square
                            ^^^
                            .. manim:: SquareScene
                                :save_last_frame:

                                class SquareScene(Scene):
                                    def construct(self) -> None:
                                        self.add(
                                            Square(
                                                side_length=3.0,
                                            )
                                        )

                        .. grid-item-card::
                            :class-header: sd-text-center

                            Star
                            ^^^
                            .. manim:: StarScene
                                :save_last_frame:

                                class StarScene(Scene):
                                    def construct(self) -> None:
                                        self.add(
                                            Star(
                                                outer_radius=3.0,
                                            )
                                        )

                        .. grid-item-card::
                            :class-header: sd-text-center

                            Line
                            ^^^
                            .. manim:: LineScene
                                :save_last_frame:

                                class LineScene(Scene):
                                    def construct(self) -> None:
                                        self.add(
                                            Line(
                                                (-3, 0, 0),
                                                (3, 0, 0),
                                            )
                                        )

            .. tab:: Without Source

                .. tab:: Without Reference

                    .. grid:: 2
                        :gutter: 1

                        .. grid-item-card::
                            :class-header: sd-text-center

                            Circle
                            ^^^
                            .. manim:: CircleScene
                                :save_last_frame:
                                :hide_source:

                                class CircleScene(Scene):
                                    def construct(self) -> None:
                                        self.add(
                                            Circle(
                                                radius=3.0,
                                            )
                                        )

                        .. grid-item-card::
                            :class-header: sd-text-center

                            Square
                            ^^^
                            .. manim:: SquareScene
                                :save_last_frame:
                                :hide_source:

                                class SquareScene(Scene):
                                    def construct(self) -> None:
                                        self.add(
                                            Square(
                                                side_length=3.0,
                                            )
                                        )

                        .. grid-item-card::
                            :class-header: sd-text-center

                            Star
                            ^^^
                            .. manim:: StarScene
                                :save_last_frame:
                                :hide_source:

                                class StarScene(Scene):
                                    def construct(self) -> None:
                                        self.add(
                                            Star(
                                                outer_radius=3.0,
                                            )
                                        )

                        .. grid-item-card::
                            :class-header: sd-text-center

                            Line
                            ^^^
                            .. manim:: LineScene
                                :save_last_frame:
                                :hide_source:

                                class LineScene(Scene):
                                    def construct(self) -> None:
                                        self.add(
                                            Line(
                                                (-3, 0, 0),
                                                (3, 0, 0),
                                            )
                                        )

        .. tab:: Without title

            .. tab:: With Source

                .. tab:: With Reference

                    .. grid:: 2
                        :gutter: 1

                        .. grid-item-card::

                            .. manim:: CircleScene
                                :save_last_frame:
                                :ref_classes: Circle

                                class CircleScene(Scene):
                                    def construct(self) -> None:
                                        self.add(
                                            Circle(
                                                radius=3.0,
                                            )
                                        )

                        .. grid-item-card::

                            .. manim:: SquareScene
                                :save_last_frame:
                                :ref_classes: Square

                                class SquareScene(Scene):
                                    def construct(self) -> None:
                                        self.add(
                                            Square(
                                                side_length=3.0,
                                            )
                                        )

                        .. grid-item-card::

                            .. manim:: StarScene
                                :save_last_frame:
                                :ref_classes: Star

                                class StarScene(Scene):
                                    def construct(self) -> None:
                                        self.add(
                                            Star(
                                                outer_radius=3.0,
                                            )
                                        )

                        .. grid-item-card::

                            .. manim:: LineScene
                                :save_last_frame:
                                :ref_classes: Line

                                class LineScene(Scene):
                                    def construct(self) -> None:
                                        self.add(
                                            Line(
                                                (-3, 0, 0),
                                                (3, 0, 0),
                                            )
                                        )

                .. tab:: Without Reference

                    .. grid:: 2
                        :gutter: 1

                        .. grid-item-card::

                            .. manim:: CircleScene
                                :save_last_frame:

                                class CircleScene(Scene):
                                    def construct(self) -> None:
                                        self.add(
                                            Circle(
                                                radius=3.0,
                                            )
                                        )

                        .. grid-item-card::

                            .. manim:: SquareScene
                                :save_last_frame:

                                class SquareScene(Scene):
                                    def construct(self) -> None:
                                        self.add(
                                            Square(
                                                side_length=3.0,
                                            )
                                        )

                        .. grid-item-card::

                            .. manim:: StarScene
                                :save_last_frame:

                                class StarScene(Scene):
                                    def construct(self) -> None:
                                        self.add(
                                            Star(
                                                outer_radius=3.0,
                                            )
                                        )

                        .. grid-item-card::

                            .. manim:: LineScene
                                :save_last_frame:

                                class LineScene(Scene):
                                    def construct(self) -> None:
                                        self.add(
                                            Line(
                                                (-3, 0, 0),
                                                (3, 0, 0),
                                            )
                                        )

            .. tab:: Without Source

                .. tab:: Without Reference

                    .. grid:: 2
                        :gutter: 1

                        .. grid-item-card::

                            .. manim:: CircleScene
                                :save_last_frame:
                                :hide_source:

                                class CircleScene(Scene):
                                    def construct(self) -> None:
                                        self.add(
                                            Circle(
                                                radius=3.0,
                                            )
                                        )

                        .. grid-item-card::

                            .. manim:: SquareScene
                                :save_last_frame:
                                :hide_source:

                                class SquareScene(Scene):
                                    def construct(self) -> None:
                                        self.add(
                                            Square(
                                                side_length=3.0,
                                            )
                                        )

                        .. grid-item-card::

                            .. manim:: StarScene
                                :save_last_frame:
                                :hide_source:

                                class StarScene(Scene):
                                    def construct(self) -> None:
                                        self.add(
                                            Star(
                                                outer_radius=3.0,
                                            )
                                        )

                        .. grid-item-card::

                            .. manim:: LineScene
                                :save_last_frame:
                                :hide_source:

                                class LineScene(Scene):
                                    def construct(self) -> None:
                                        self.add(
                                            Line(
                                                (-3, 0, 0),
                                                (3, 0, 0),
                                            )
                                        )
