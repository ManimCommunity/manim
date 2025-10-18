from manim import *

class C2PTest(Scene):
    def construct(self):
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 5, 1],
            x_length=5,
            y_length=5,
        )

        self.add(axes)

        # Test different inputs
        print("Single coordinates:")
        print(axes.c2p(1, 2))  # Works as expected → np.array([X, Y])

        print("\n1D list input:")
        print(axes.c2p([1, 2]))  # Confusing behavior! May not return [X, Y]

        print("\nList of lists input:")
        print(axes.c2p([[1, 2]]))  # Returns a list of np.arrays

        print("\nList of multiple coordinates:")
        print(axes.c2p([[1, 2], [3, 4]]))  # Returns list of np.arrays

        # Add dots at coordinates to visualize
        dot1 = Dot(axes.c2p(1, 2), color=RED)
        dot2 = Dot(axes.c2p([[3, 4]])[0], color=BLUE)
        self.add(dot1, dot2)
