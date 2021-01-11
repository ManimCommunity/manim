import sys
from pathlib import Path

import pytest
from manim import *
from ..helpers.path_utils import get_project_root
from ..utils.testing_utils import get_scenes_to_test
from ..utils.GraphicalUnitTester import GraphicalUnitTester

from ..helpers.graphical_units import set_test_scene

# To update the reference object after a visual inspection, use this line:
# set_test_scene(NameOfTestScene, "img_and_svg")


def get_test_resource(filename):
    return str(
        get_project_root() / "tests/test_graphical_units/img_svg_resources" / filename
    )


# Tests break down into two kinds: one where the SVG is simple enough to step through
# and ones where the SVG is realistically complex, and the output should be visually inspected.

# First are the simple tests.


class CubicPathTest(Scene):
    def construct(self):
        cubic_demo = SVGMobject(get_test_resource("cubic_demo.svg"))
        self.add(cubic_demo)
        self.wait()


class CubicAndLinetoTest(Scene):
    def construct(self):
        cubic_lineto = SVGMobject(get_test_resource("cubic_and_lineto.svg"))
        self.add(cubic_lineto)
        self.wait()


class RhomboidTest(Scene):
    pytest_skip = True

    def construct(self):
        # TODO: Fix behavior such that the polygon results in a closed shape, even without the closing z.
        # TODO: Discuss whether, upon loading an SVG, whether to obey the fill and stroke properties.
        rhomboid = SVGMobject(get_test_resource("rhomboid.svg")).shift(UP * 2)
        rhomboid_no_fill = (
            rhomboid.copy()
            .set_fill(opacity=0)
            .set_stroke(color=WHITE, width=1)
            .shift(DOWN * 4)
        )
        self.add(rhomboid, rhomboid_no_fill)
        self.wait()


class MultiPartPathTest(Scene):
    def construct(self):
        mpp = SVGMobject(get_test_resource("multi_part_path.svg"))
        self.add(mpp)
        self.wait()


class QuadraticPathTest(Scene):
    def construct(self):
        quad = SVGMobject(get_test_resource("qcurve_demo.svg"))
        self.add(quad)
        self.wait()


class SmoothCurvesTest(Scene):
    def construct(self):
        smooths = SVGMobject(get_test_resource("smooth_curves.svg")).set_fill(opacity=0)
        self.add(smooths)
        self.wait()


# TODO: test in a 3D Scene
class ThreeDSVGTest(ThreeDScene):
    pytest_skip = True


# Second are the visual tests - these are probably too complex to verify step-by-step, so
# these are really more of a spot-check


class WeightSVGTest(Scene):
    def construct(self):
        path = get_test_resource("weight.svg")
        svg_obj = SVGMobject(path)
        self.add(svg_obj)
        self.wait()


class BrachistochroneCurveTest(Scene):
    pytest_skip = True

    def construct(self):
        # TODO: The stroke option of the path has an RGB color specified.
        # TODO: The rect's fill="none" attribute should be obeyed.
        # TODO: The path's fill is unstated, and should not be obeyed.
        brach_curve = SVGMobject(get_test_resource("curve.svg")).set_fill(opacity=0)
        self.add(brach_curve)
        self.wait()


class DesmosGraph1Test(Scene):
    pytest_skip = True

    def construct(self):
        # TODO: fill and stroke attributes of the rectangle and paths.
        dgraph = SVGMobject(get_test_resource("desmos-graph_1.svg")).set_fill(opacity=0)
        self.add(dgraph)
        self.wait()


class Drawing4(Scene):
    pytest_skip = True

    def construct(self):
        # TODO: There is a linear gradient specified. That sounds complicated, and that also makes it not a VMobject
        #  without some vector gradient object. That seems complicated.
        # TODO: Should the engine handle url calls in fill/stroke?
        # TODO: white hex background, and stroke / fill colors aren't parsed.
        draw4 = SVGMobject(get_test_resource("drawing4.svg")).scale(3)
        self.add(draw4)
        self.wait()


class FancyGTest(Scene):
    def construct(self):
        fancy_g = SVGMobject(get_test_resource("fancy_g.svg")).scale(3)
        self.add(fancy_g)
        self.wait()


class LogoTest(Scene):
    def construct(self):
        logo = SVGMobject(get_test_resource("logo.svg"))
        self.add(logo)
        self.wait()


class MSModelTest(Scene):
    # TODO: Have to specify no fill, and possibly mitre issues?
    def construct(self):
        ms_model = SVGMobject(get_test_resource("ms_model.svg")).set_fill(opacity=0)
        self.add(ms_model)
        self.wait()


class SingleUSStateTest(Scene):
    def construct(self):
        single_state = SVGMobject(get_test_resource("single_state.svg"))
        self.add(single_state)
        self.wait()


class ContiguousUSMapTest(Scene):
    def construct(self):
        states = SVGMobject(get_test_resource("states_map.svg")).set_fill(opacity=0).scale(3)
        self.add(states)
        self.wait()


class PeriodicTableTest(Scene):
    pytest_skip = True

    # TODO: I'm sure there will be issues of fill and stroke.
    # color is in the style attribute of the parent element (g)
    def construct(self):
        ptable = SVGMobject(get_test_resource("Periodic_table_1.svg")).scale(3)
        self.add(ptable)
        self.wait()


class SomethingSmileTest(Scene):
    pytest_skip = True

    # TODO: Looks like it's a fill / color issue.
    # fill and stroke is in the path element.
    def construct(self):
        smile = SVGMobject(get_test_resource("something.svg"))
        self.add(smile)
        self.wait()


class StarchTest(Scene):
    def construct(self):
        starch = SVGMobject(get_test_resource("starch.svg"))
        self.add(starch)
        self.wait()


class PixelizedTextTest(Scene):
    pytest_skip = True

    # TODO: Obey the fill and fill-opacity attributes
    def construct(self):
        rgb_svg = SVGMobject(get_test_resource("test.svg"))
        self.add(rgb_svg)
        self.wait()


class VideoIconTest(Scene):
    # TODO: Obey the fill attribute, technically.
    def construct(self):
        video_icon = SVGMobject(get_test_resource("video_icon.svg"))
        self.add(video_icon)
        self.wait()


"""

At the moment, we're having syntax issues with this test... 
we're diving into the deep world of things other than paths.

class RCTest(Scene):
    pytest_skip = True

    def construct(self):
        rc_svg = SVGMobject(get_test_resource("RC.svg"))
        self.add(rc_svg)
        self.wait()

# set_test_scene(RCTest, "img_and_svg")
"""

"""
class WhiteLogoTest(Scene):
    # TODO: Arc issues. Also, probably fill/stroke stuff too.
    def construct(self):
        white_logo = SVGMobject(get_test_resource("white.svg"))
        self.add(white_logo)
        self.wait()
"""


class ImageMobjectTest(Scene):
    def construct(self):
        file_path = get_test_resource("tree_img_640x351.png")

        im1 = ImageMobject(file_path).shift(4 * LEFT + UP)
        im2 = ImageMobject(file_path, scale_to_resolution=1080).shift(
            4 * LEFT + 2 * DOWN
        )
        im3 = ImageMobject(file_path, scale_to_resolution=540).shift(4 * RIGHT)
        self.add(im1, im2, im3)
        self.wait(1)


MODULE_NAME = "img_and_svg"


@pytest.mark.parametrize("scene_to_test", get_scenes_to_test(__name__), indirect=False)
def test_scene(scene_to_test, tmpdir, show_diff):
    GraphicalUnitTester(scene_to_test[1], MODULE_NAME, tmpdir).test(show_diff=True)
