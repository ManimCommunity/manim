.. manim:: Adventure
    :hide_source:

    class VectorGroup(VGroup):
        def __init__(
            self, start, end, labelname: str,
            vector_color: ParsableManimColor, direction = RIGHT,
            plane: NumberPlane | None = None, **kwargs
        ) -> None:
            if plane is not None:
                # if using a plane convert from plane units
                # to Munits
                start = plane.c2p(*start)
                end = plane.c2p(*end)

            self.vector = Arrow(
                start,
                end,
                color=vector_color,
                buff=0
            )
            self.label = MathTex(labelname, color=vector_color)

            def label_updater(m: MathTex, d=direction):
                m.next_to(self.vector, direction=d, **kwargs)

            self.label.add_updater(label_updater, call_updater=True)
            super().__init__(self.vector, self.label, **kwargs)

        @override_animation(Create)
        def _create_vec_write_label(self) -> AnimationGroup:
            return AnimationGroup(
                Create(self.vector),
                Write(self.label),
                lag_ratio=0
            )

        @override_animation(Uncreate)
        def _uncreate_vec_unwrite_label(self) -> AnimationGroup:
            return AnimationGroup(
                Uncreate(self.vector),
                Unwrite(self.label),
                lag_ratio=0
            )
    class Adventure(Scene):
        """Goal: Make an example showcasing manim's features"""

        def construct(self) -> None:
            intro = Text("Let's try to add two vectors!")
            vec_txts = Tex(r"We'll use $\boldsymbol{\vec{v}_1}=(2, 2)$ and $\boldsymbol{\vec{v}_2}=(0, -3)$")
            self.play(Create(intro))
            self.wait(1)
            self.play(intro.animate.shift(2*UP).set_opacity(0.5), Write(vec_txts))
            self.wait(1)
            self.play(Unwrite(intro), Unwrite(vec_txts), run_time=.5)
            self.wait(0.2)

            self.show_addition_math()
            self.wait(0.2)
            self.show_vector_addition()

            outro = Text("Thanks for watching!")
            self.play(Create(outro))
            self.wait()

        def show_addition_math(self) -> None:
            title = Title("Vector Addition Algebraically")

            v1x, v1y = (2, 2)
            v2x, v2y = (0, -3)
            math = MathTex(r"""
                \begin{bmatrix} %(v1x)d \\ %(v1y)d \end{bmatrix}
                +\begin{bmatrix} %(v2x)d \\ %(v2y)d \end{bmatrix}
            """ % {
                'v1x': v1x,
                'v2x': v2x,
                'v1y': v1y,
                'v2y': v2y
            }).shift(DOWN)

            resultant_vector = r"=\begin{bmatrix} %(x)d \\ %(y)d \end{bmatrix}" % {
                'x': v1x+v2x,
                'y': v1y+v2y
            }
            math_with_answer = MathTex(
                math.get_tex_string()+resultant_vector
            ).move_to(math.get_center())

            self.play(Write(math), FadeIn(title))
            self.wait(2)
            self.play(
                math.animate.shift(2*UP).set_opacity(0.5),
                Write(math_with_answer)
            )
            conclusion = Paragraph("As you can see,\nYou add each component individually").to_edge(DOWN)
            self.play(Write(conclusion))
            self.wait(2)
            self.play(Unwrite(math), Unwrite(math_with_answer), Unwrite(conclusion), Unwrite(title))

        def show_vector_addition(self) -> None:
            title = Text("Now let's take a look at it geometrically")
            self.play(Write(title))
            self.wait(2)
            self.play(Unwrite(title))

            plane = NumberPlane()

            sum_point = (2, -1, 0)

            v1 = VectorGroup(
                ORIGIN,
                (2, 2, 0),
                r"\boldsymbol{\vec{v}_1}",
                RED,
                direction=UP,
                plane=plane
            )

            v2 = VectorGroup(
                ORIGIN,
                (0, -3, 0),
                r"\boldsymbol{\vec{v}_2}",
                YELLOW,
                direction=LEFT,
                plane=plane
            )

            v1moved = VectorGroup(
                (0, -3, 0),
                sum_point,
                r"\boldsymbol{\vec{v}_1}",
                v1.vector.get_color(),
                plane=plane
            )

            v2moved = VectorGroup(
                (2, 2, 0),
                sum_point,
                r"\boldsymbol{\vec{v}_2}",
                v2.vector.get_color(),
                plane=plane
            )

            sum_vec = VectorGroup(
                ORIGIN,
                sum_point,
                r"\boldsymbol{\vec{v}_1}+\boldsymbol{\vec{v}_2}",
                ORANGE,
                direction=DOWN,
                plane=plane
            )

            self.play(Create(plane), Create(v1))
            self.wait(0.5)
            self.play(Create(v2))
            self.wait()

            # animate movement of vectors
            self.play(
                Succession(
                    ReplacementTransform(v1.copy(), v1moved),
                    ReplacementTransform(v2.copy(), v2moved)
                )
            )
            self.wait()
            # draw sum vector
            self.play(Create(sum_vec))
            self.wait()
            self.play(*[
                Uncreate(x)
                for x in (
                    plane,
                    v1,
                    v2,
                    v1moved,
                    v2moved,
                    sum_vec
                )
            ])
