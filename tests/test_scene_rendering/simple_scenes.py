from manim import *


class SquareToCircle(Scene):
    def construct(self):
        square = Square()
        circle = Circle()
        self.play(Transform(square, circle))


class SceneWithMultipleCalls(Scene):
    def construct(self):
        number = Integer(0)
        self.add(number)
        for _i in range(10):
            self.play(Animation(Square()))


class SceneWithMultipleWaitCalls(Scene):
    def construct(self):
        self.play(Create(Square()))
        self.wait(1)
        self.play(Create(Square().shift(DOWN)))
        self.wait(1)
        self.play(Create(Square().shift(2 * DOWN)))
        self.wait(1)
        self.play(Create(Square().shift(3 * DOWN)))
        self.wait(1)


class NoAnimations(Scene):
    def construct(self):
        dot = Dot().set_color(GREEN)
        self.add(dot)
        self.wait(0.1)


class SceneWithStaticWait(Scene):
    def construct(self):
        self.add(Square())
        self.wait()


class SceneWithNonStaticWait(Scene):
    def construct(self):
        s = Square()
        # Non static wait are triggered by mobject with time based updaters.
        s.add_updater(lambda mob, dt: None)
        self.add(s)
        self.wait()


class StaticScene(Scene):
    def construct(self):
        dot = Dot().set_color(GREEN)
        self.add(dot)


class InteractiveStaticScene(Scene):
    def construct(self):
        dot = Dot().set_color(GREEN)
        self.add(dot)
        self.interactive_mode = True


class SceneWithSections(Scene):
    def construct(self):
        # this would be defined in a third party application using the segmented video API
        class PresentationSectionType(str, Enum):
            # start, end, wait for continuation by user
            NORMAL = "presentation.normal"
            # start, end, immediately continue to next section
            SKIP = "presentation.skip"
            # start, end, restart, immediately continue to next section when continued by user
            LOOP = "presentation.loop"
            # start, end, restart, finish animation first when user continues
            COMPLETE_LOOP = "presentation.complete_loop"

        # this animation is part of the first, automatically created section
        self.wait()

        self.next_section()
        self.wait(2)

        self.next_section(name="test")
        self.wait()

        self.next_section(
            "Prepare For Unforeseen Consequences.", DefaultSectionType.NORMAL
        )
        self.wait(2)

        self.next_section(type=PresentationSectionType.SKIP)
        self.wait()

        self.next_section(
            name="this section should be removed as it doesn't contain any animations"
        )


class ElaborateSceneWithSections(Scene):
    def construct(self):
        # the first automatically created section should be deleted <- it's empty
        self.next_section("create square")
        square = Square()
        self.play(FadeIn(square))
        self.wait()

        self.next_section("transform to circle")
        circle = Circle()
        self.play(Transform(square, circle))
        self.wait()

        self.next_section("fade out")
        self.play(FadeOut(square))
        self.wait()
