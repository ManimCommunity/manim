#!/usr/bin/env python

from manim.imports import * # Import the manim library

# To watch one of these scenes, run the following:
# python -m manim example_scenes.py SquareToCircle -pl
#
# Use the flat -l for a faster rendering at a lower
# quality.
# Use -s to skip to the end and just save the final frame
# Use the -p to have the animation (or image, if -s was
# used) pop up once done.
# Use -n <number> to skip ahead to the n'th animation of a scene.
# Use -r <number> to specify a resolution (for example, -r 1080
# for a 1920x1080 video)


class OpeningManimExample(Scene): # Define a new scene named OpeningManimScene inheriting from Scene
    def construct(self): # Define the animations to be constructed
        title = TextMobject("This is some \\LaTeX") # Define a TextMobject, doesn't use LaTeX
        basel = TexMobject(
            "\\sum_{n=1}^\\infty "
            "\\frac{1}{n^2} = \\frac{\\pi^2}{6}"
        ) # Define a TexMobject, uses LaTeX
        VGroup(title, basel).arrange(DOWN) # Create a VGroup containing title and basel, and arrange them to show with one under the other
        self.play(
            Write(title),
            FadeInFrom(basel, UP),
        ) # Play an animation that Write()s title and FadeInFrom() on basel with the direction being up
        self.wait() # Wait for the default wait time

        transform_title = TextMobject("That was a transform") # Define a TextMobject, doesn't use LaTeX
        transform_title.to_corner(UP + LEFT) # Move transform_title to the upper left corner
        self.play(
            Transform(title, transform_title),
            LaggedStart(*map(FadeOutAndShiftDown, basel)),
        ) # Transform title into transform_title and FadeOutAndShiftDown() basel with a LaggedStart()
        self.wait() # Wait for the default wait time

        grid = NumberPlane() # Define a number plane
        grid_title = TextMobject("This is a grid") # Define a TextMobject, doesn't use LaTeX
        grid_title.scale(1.5) # Scale the number grid up by 1.5
        grid_title.move_to(transform_title) # Move grid_title to where transform_title was

        self.add(grid, grid_title)  # Make sure title is on top of grid
        self.play(
            FadeOut(title),
            FadeInFromDown(grid_title),
            ShowCreation(grid, run_time=3, lag_ratio=0.1),
        ) # Play an animation the FadeOut()s title, FadeInFromDown()s grid_title, and ShowCreation() on grid with the animation running time being three and inducing a lag ratio of 0.1
        self.wait() # Wait for the default wait time

        grid_transform_title = TextMobject(
            "That was a non-linear function \\\\"
            "applied to the grid"
        ) # Define a TextMobject, doesn't use LaTeX
        grid_transform_title.move_to(grid_title, UL) # Move grid_transform_title to the upper-left corner of grid_title but with a buffer
        grid.prepare_for_nonlinear_transform() # Prepare the grid for a nonlinear tranformation
        self.play(
            grid.apply_function,
            lambda p: p + np.array([
                np.sin(p[1]),
                np.sin(p[0]),
                0,
            ]),
            run_time=3,
        ) # Apply a function to the grid that is nonlinear
        self.wait() # Wait for the default wait time
        self.play(
            Transform(grid_title, grid_transform_title)
        ) # Transform the grid_title into the grid_transform_title
        self.wait() # Wait for the default wait time


class SquareToCircle(Scene): # Define a new scene called SquareToCircle inheriting from Scene
    def construct(self): # Define the animations to be constructed
        circle = Circle() # Define a circle with the default radius
        square = Square() # Define a square with the default side length
        square.flip(RIGHT) # Flip the square to the right
        square.rotate(-3 * TAU / 8) # Rotate the square -3/8 Tau radians
        circle.set_fill(PINK, opacity=0.5) # Set the fill color of the circle to be pink with 50% opacity

        self.play(ShowCreation(square)) # ShowCreation() square
        self.play(Transform(square, circle)) # Transform() the square into the circle
        self.play(FadeOut(square)) # FadeOut the circle (which is called square)


class WarpSquare(Scene): # Define a new scene called WarpSquare inheriting from Scene
    def construct(self): # Define the animations to be constructed
        square = Square() # Define a square with the default side length
        self.play(ApplyPointwiseFunction(
            lambda point: complex_to_R3(np.exp(R3_to_complex(point))),
            square
        )) # Apply the preceding pointwise function to the square
        self.wait() # Wait for the default wait time


class WriteStuff(Scene): # Define a new scene called WriteStuff inheriting from Scene
    def construct(self): # Define the animations to be constructed
        example_text = TextMobject(
            "This is a some text",
            tex_to_color_map={"text": YELLOW} # Define a color map to set any substring with "text" to YELLOW (defined in constants.py)
        ) # Define a TextMobject, doesn't use LaTeX
        example_tex = TexMobject(
            "\\sum_{k=1}^\\infty {1 \\over k^2} = {\\pi^2 \\over 6}",
        ) # Define a TexMobject, uses LaTeX
        group = VGroup(example_text, example_tex).arrange(DOWN) # Define a VGroup containing example_text and example_tex and arrange it to have one under the other
        group.set_width(FRAME_WIDTH - 2 * LARGE_BUFF) # Set the width of the VGroup() group to be the frame width (usually 14) - 2 (12) time the size of a LARGE_BUFF (defined in constants.py)

        self.play(Write(example_text)) # Write() example_text
        self.play(Write(example_tex)) # Write() example_tex
        self.wait() # Wait for the default wait time


class UpdatersExample(Scene): # Define a new scene called Updaters inheriting from Scene
    def construct(self): # Define the animations to be constructed
        decimal = DecimalNumber( # Define a DecimalNumber() called decimal (to be used with updaters)
            0, # Set the initial value to 0
            show_ellipsis=True, # Show ellipses
            num_decimal_places=3, # Set the number of decimal places to 3
            include_sign=True, # Include the position/negative sign
        )
        square = Square().to_edge(UP) # Create a Square() named square with the default side length and arrange to be at the top edge of the frame

        decimal.add_updater(lambda d: d.next_to(square, RIGHT)) # Add an updater to the decimal value that, every frame, moves it to the right pf the square with a buffer
        decimal.add_updater(lambda d: d.set_value(square.get_center()[1]))  # Set the value of the decimal number to be the y-coordinate of the square every frame
        self.add(square, decimal) # Add the square and decimal to the scene
        self.play(
            square.to_edge, DOWN,
            rate_func=there_and_back,
            run_time=5,
        ) # Play an animation the moves the square to the bottom edge of the screen, uses the rate function there_and_back (moves it to the bottom and then back to where it was), and uses a run time of 5
        self.wait() # Wait for the default wait time
