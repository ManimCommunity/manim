import sys
from pathlib import Path

import pytest

from manim import *

from ..helpers.graphical_units import set_test_scene
from ..helpers.path_utils import get_project_root, get_svg_resource
from ..utils.GraphicalUnitTester import GraphicalUnitTester
from ..utils.testing_utils import get_scenes_to_test

# Tests break down into two kinds: one where the SVG is simple enough to step through
# and ones where the SVG is realistically complex, and the output should be visually inspected.

# First are the simple tests.


class LineTest(Scene):
    def construct(self):
        line_demo = SVGMobject(get_svg_resource("line.svg"))
        self.add(line_demo)
        self.wait()


class CubicPathTest(Scene):
    def construct(self):
        cubic_demo = SVGMobject(get_svg_resource("cubic_demo.svg"))
        self.add(cubic_demo)
        self.wait()


class CubicAndLinetoTest(Scene):
    def construct(self):
        cubic_lineto = SVGMobject(get_svg_resource("cubic_and_lineto.svg"))
        self.add(cubic_lineto)
        self.wait()


class RhomboidTest(Scene):
    """Test the default fill and parsed stroke of a rhomboid"""

    def construct(self):
        rhomboid = SVGMobject(get_svg_resource("rhomboid.svg")).scale(0.5)
        rhomboid_fill = rhomboid.copy().set_fill(opacity=1).shift(UP * 2)
        rhomboid_no_fill = rhomboid.copy().set_fill(opacity=0).shift(DOWN * 2)

        self.add(rhomboid, rhomboid_fill, rhomboid_no_fill)
        self.wait()


class InheritanceTest(Scene):
    """Ensure SVG inheritance is followed"""

    def construct(self):
        three_arrows = SVGMobject(get_svg_resource("inheritance_test.svg")).scale(0.5)
        self.add(three_arrows)
        self.wait()


class MultiPartPathTest(Scene):
    def construct(self):
        mpp = SVGMobject(get_svg_resource("multi_part_path.svg"))
        self.add(mpp)
        self.wait()


class QuadraticPathTest(Scene):
    def construct(self):
        quad = SVGMobject(get_svg_resource("qcurve_demo.svg"))
        self.add(quad)
        self.wait()


class SmoothCurvesTest(Scene):
    def construct(self):
        smooths = SVGMobject(get_svg_resource("smooth_curves.svg"))
        self.add(smooths)
        self.wait()


class WatchTheDecimals(Scene):
    def construct(self):
        decimal = SVGMobject(get_svg_resource("watch_the_decimals.svg"))
        self.add(decimal)
        self.wait()


class UseTagInheritanceTest(Scene):
    def construct(self):
        aabbb = SVGMobject(get_svg_resource("aabbb.svg"))
        self.add(aabbb)
        self.wait()


class HalfEllipseTest(Scene):
    def construct(self):
        half_ellipse = SVGMobject(get_svg_resource("half_ellipse.svg"))
        self.add(half_ellipse)
        self.wait()


class HeartTest(Scene):
    def construct(self):
        heart = SVGMobject(get_svg_resource("heart.svg"))
        self.add(heart)
        self.wait()


class Arcs01Test(Scene):
    # See: https://www.w3.org/TR/SVG11/images/paths/arcs01.svg
    def construct(self):
        arcs = SVGMobject(get_svg_resource("arcs01.svg"))
        self.add(arcs)
        self.wait()


class Arcs02Test(Scene):
    # See: https://www.w3.org/TR/SVG11/images/paths/arcs02.svg
    def construct(self):
        arcs = SVGMobject(get_svg_resource("arcs02.svg"))
        self.add(arcs)
        self.wait()


# Second are the visual tests - these are probably too complex to verify step-by-step, so
# these are really more of a spot-check


class WeightSVGTest(Scene):
    def construct(self):
        path = get_svg_resource("weight.svg")
        svg_obj = SVGMobject(path)
        self.add(svg_obj)
        self.wait()


class BrachistochroneCurveTest(Scene):
    def construct(self):
        brach_curve = SVGMobject(get_svg_resource("curve.svg"))
        self.add(brach_curve)
        self.wait()


class DesmosGraph1Test(Scene):
    def construct(self):
        dgraph = SVGMobject(get_svg_resource("desmos-graph_1.svg")).scale(3)
        self.add(dgraph)
        self.wait()


class PenroseTest(Scene):
    def construct(self):
        penrose = SVGMobject(get_svg_resource("penrose.svg"))
        self.add(penrose)
        self.wait()


class ManimLogoTest(Scene):
    def construct(self):
        background_rect = Rectangle(color=WHITE, fill_opacity=1).scale(2)
        manim_logo = SVGMobject(get_svg_resource("manim-logo-sidebar.svg"))
        self.add(background_rect, manim_logo)
        self.wait()


class UKFlagTest(Scene):
    def construct(self):
        uk_flag = SVGMobject(get_svg_resource("united-kingdom.svg"))
        self.add(uk_flag)
        self.wait()


class SingleUSStateTest(Scene):
    def construct(self):
        single_state = SVGMobject(get_svg_resource("single_state.svg"))
        self.add(single_state)
        self.wait()


class ContiguousUSMapTest(Scene):
    def construct(self):
        states = SVGMobject(get_svg_resource("states_map.svg")).scale(3)
        self.add(states)
        self.wait()


class PixelizedTextTest(Scene):
    def construct(self):
        background_rect = Rectangle(color=WHITE, fill_opacity=1).scale(2)
        rgb_svg = SVGMobject(get_svg_resource("pixelated_text.svg"))
        self.add(background_rect, rgb_svg)
        self.wait()


class VideoIconTest(Scene):
    def construct(self):
        video_icon = SVGMobject(get_svg_resource("video_icon.svg"))
        self.add(video_icon)
        self.wait()


class MultipleTransformTest(Scene):
    def construct(self):
        svg_obj = SVGMobject(get_svg_resource("multiple_transforms.svg"))
        self.add(svg_obj)
        self.wait()


class MatrixTransformTest(Scene):
    def construct(self):
        svg_obj = SVGMobject(get_svg_resource("matrix.svg"))
        self.add(svg_obj)
        self.wait()


class ScaleTransformTest(Scene):
    def construct(self):
        svg_obj = SVGMobject(get_svg_resource("scale.svg"))
        self.add(svg_obj)
        self.wait()


class TranslateTransformTest(Scene):
    def construct(self):
        svg_obj = SVGMobject(get_svg_resource("translate.svg"))
        self.add(svg_obj)
        self.wait()


class SkewXTransformTest(Scene):
    def construct(self):
        svg_obj = SVGMobject(get_svg_resource("skewX.svg"))
        self.add(svg_obj)
        self.wait()


class SkewYTransformTest(Scene):
    def construct(self):
        svg_obj = SVGMobject(get_svg_resource("skewY.svg"))
        self.add(svg_obj)
        self.wait()


class RotateTransformTest(Scene):
    def construct(self):
        svg_obj = SVGMobject(get_svg_resource("rotate.svg"))
        self.add(svg_obj)
        self.wait()


class ImageMobjectTest(Scene):
    def construct(self):
        file_path = get_svg_resource("tree_img_640x351.png")

        im1 = ImageMobject(file_path).shift(4 * LEFT + UP)
        im2 = ImageMobject(file_path, scale_to_resolution=1080).shift(
            4 * LEFT + 2 * DOWN
        )
        im3 = ImageMobject(file_path, scale_to_resolution=540).shift(4 * RIGHT)
        self.add(im1, im2, im3)
        self.wait(1)


class ImageInterpolationTest(Scene):
    def construct(self):
        img = ImageMobject(
            np.uint8([[63, 0, 0, 0], [0, 127, 0, 0], [0, 0, 191, 0], [0, 0, 0, 255]])
        )

        img.height = 2
        img1 = img.copy()
        img2 = img.copy()
        img3 = img.copy()
        img4 = img.copy()
        img5 = img.copy()

        img1.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        img2.set_resampling_algorithm(RESAMPLING_ALGORITHMS["lanczos"])
        img3.set_resampling_algorithm(RESAMPLING_ALGORITHMS["linear"])
        img4.set_resampling_algorithm(RESAMPLING_ALGORITHMS["cubic"])
        img5.set_resampling_algorithm(RESAMPLING_ALGORITHMS["box"])

        self.add(img1, img2, img3, img4, img5)
        [s.shift(4 * LEFT + pos * 2 * RIGHT) for pos, s in enumerate(self.mobjects)]
        self.wait()


MODULE_NAME = "img_and_svg"


@pytest.mark.parametrize("scene_to_test", get_scenes_to_test(__name__), indirect=False)
def test_scene(scene_to_test, tmpdir, show_diff):
    GraphicalUnitTester(scene_to_test[1], MODULE_NAME, tmpdir).test(show_diff=show_diff)
