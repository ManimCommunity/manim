from manim import *
from tests.test_graphical_units.testing.frames_comparison import frames_comparison

from ..helpers.path_utils import get_svg_resource

__module_test__ = "img_and_svg"

# Tests break down into two kinds: one where the SVG is simple enough to step through
# and ones where the SVG is realistically complex, and the output should be visually inspected.

# First are the simple tests.


@frames_comparison
def test_Line(scene):
    line_demo = SVGMobject(get_svg_resource("line.svg"))
    scene.add(line_demo)
    scene.wait()


@frames_comparison
def test_CubicPath(scene):
    cubic_demo = SVGMobject(get_svg_resource("cubic_demo.svg"))
    scene.add(cubic_demo)
    scene.wait()


@frames_comparison
def test_CubicAndLineto(scene):
    cubic_lineto = SVGMobject(get_svg_resource("cubic_and_lineto.svg"))
    scene.add(cubic_lineto)
    scene.wait()


@frames_comparison
def test_Rhomboid(scene):
    rhomboid = SVGMobject(get_svg_resource("rhomboid.svg")).scale(0.5)
    rhomboid_fill = rhomboid.copy().set_fill(opacity=1).shift(UP * 2)
    rhomboid_no_fill = rhomboid.copy().set_fill(opacity=0).shift(DOWN * 2)
    scene.add(rhomboid, rhomboid_fill, rhomboid_no_fill)
    scene.wait()


@frames_comparison
def test_Inheritance(scene):
    three_arrows = SVGMobject(get_svg_resource("inheritance_test.svg")).scale(0.5)
    scene.add(three_arrows)
    scene.wait()


@frames_comparison
def test_MultiPartPath(scene):
    mpp = SVGMobject(get_svg_resource("multi_part_path.svg"))
    scene.add(mpp)
    scene.wait()


@frames_comparison
def test_QuadraticPath(scene):
    quad = SVGMobject(get_svg_resource("qcurve_demo.svg"))
    scene.add(quad)
    scene.wait()


@frames_comparison
def test_SmoothCurves(scene):
    smooths = SVGMobject(get_svg_resource("smooth_curves.svg"))
    scene.add(smooths)
    scene.wait()


@frames_comparison
def test_WatchTheDecimals(scene):
    def construct(scene):
        decimal = SVGMobject(get_svg_resource("watch_the_decimals.svg"))
        scene.add(decimal)
        scene.wait()


@frames_comparison
def test_UseTagInheritance(scene):
    aabbb = SVGMobject(get_svg_resource("aabbb.svg"))
    scene.add(aabbb)
    scene.wait()


@frames_comparison
def test_HalfEllipse(scene):
    half_ellipse = SVGMobject(get_svg_resource("half_ellipse.svg"))
    scene.add(half_ellipse)
    scene.wait()


@frames_comparison
def test_Heart(scene):
    heart = SVGMobject(get_svg_resource("heart.svg"))
    scene.add(heart)
    scene.wait()


@frames_comparison
def test_Arcs01(scene):
    # See: https://www.w3.org/TR/SVG11/images/paths/arcs01.svg
    arcs = SVGMobject(get_svg_resource("arcs01.svg"))
    scene.add(arcs)
    scene.wait()


@frames_comparison(last_frame=False)
def test_Arcs02(scene):
    # See: https://www.w3.org/TR/SVG11/images/paths/arcs02.svg
    arcs = SVGMobject(get_svg_resource("arcs02.svg"))
    scene.add(arcs)
    scene.wait()


# Second are the visual tests - these are probably too complex to verify step-by-step, so
# these are really more of a spot-check


@frames_comparison(last_frame=False)
def test_WeightSVG(scene):
    path = get_svg_resource("weight.svg")
    svg_obj = SVGMobject(path)
    scene.add(svg_obj)
    scene.wait()


@frames_comparison
def test_BrachistochroneCurve(scene):
    brach_curve = SVGMobject(get_svg_resource("curve.svg"))
    scene.add(brach_curve)
    scene.wait()


@frames_comparison
def test_DesmosGraph1(scene):
    dgraph = SVGMobject(get_svg_resource("desmos-graph_1.svg")).scale(3)
    scene.add(dgraph)
    scene.wait()


@frames_comparison
def test_Penrose(scene):
    penrose = SVGMobject(get_svg_resource("penrose.svg"))
    scene.add(penrose)
    scene.wait()


@frames_comparison
def test_ManimLogo(scene):
    background_rect = Rectangle(color=WHITE, fill_opacity=1).scale(2)
    manim_logo = SVGMobject(get_svg_resource("manim-logo-sidebar.svg"))
    scene.add(background_rect, manim_logo)
    scene.wait()


@frames_comparison
def test_UKFlag(scene):
    uk_flag = SVGMobject(get_svg_resource("united-kingdom.svg"))
    scene.add(uk_flag)
    scene.wait()


@frames_comparison
def test_SingleUSState(scene):
    single_state = SVGMobject(get_svg_resource("single_state.svg"))
    scene.add(single_state)
    scene.wait()


@frames_comparison
def test_ContiguousUSMap(scene):
    states = SVGMobject(get_svg_resource("states_map.svg")).scale(3)
    scene.add(states)
    scene.wait()


@frames_comparison
def test_PixelizedText(scene):
    background_rect = Rectangle(color=WHITE, fill_opacity=1).scale(2)
    rgb_svg = SVGMobject(get_svg_resource("pixelated_text.svg"))
    scene.add(background_rect, rgb_svg)
    scene.wait()


@frames_comparison
def test_VideoIcon(scene):
    video_icon = SVGMobject(get_svg_resource("video_icon.svg"))
    scene.add(video_icon)
    scene.wait()


@frames_comparison
def test_MultipleTransform(scene):
    svg_obj = SVGMobject(get_svg_resource("multiple_transforms.svg"))
    scene.add(svg_obj)
    scene.wait()


@frames_comparison
def test_MatrixTransform(scene):
    svg_obj = SVGMobject(get_svg_resource("matrix.svg"))
    scene.add(svg_obj)
    scene.wait()


@frames_comparison
def test_ScaleTransform(scene):
    svg_obj = SVGMobject(get_svg_resource("scale.svg"))
    scene.add(svg_obj)
    scene.wait()


@frames_comparison
def test_TranslateTransform(scene):
    svg_obj = SVGMobject(get_svg_resource("translate.svg"))
    scene.add(svg_obj)
    scene.wait()


@frames_comparison
def test_SkewXTransform(scene):
    svg_obj = SVGMobject(get_svg_resource("skewX.svg"))
    scene.add(svg_obj)
    scene.wait()


@frames_comparison
def test_SkewYTransform(scene):
    svg_obj = SVGMobject(get_svg_resource("skewY.svg"))
    scene.add(svg_obj)
    scene.wait()


@frames_comparison
def test_RotateTransform(scene):
    svg_obj = SVGMobject(get_svg_resource("rotate.svg"))
    scene.add(svg_obj)
    scene.wait()


@frames_comparison
def test_ImageMobject(scene):
    file_path = get_svg_resource("tree_img_640x351.png")
    im1 = ImageMobject(file_path).shift(4 * LEFT + UP)
    im2 = ImageMobject(file_path, scale_to_resolution=1080).shift(4 * LEFT + 2 * DOWN)
    im3 = ImageMobject(file_path, scale_to_resolution=540).shift(4 * RIGHT)
    scene.add(im1, im2, im3)
    scene.wait(1)


@frames_comparison
def test_ImageInterpolation(scene):
    img = ImageMobject(
        np.uint8([[63, 0, 0, 0], [0, 127, 0, 0], [0, 0, 191, 0], [0, 0, 0, 255]]),
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

    scene.add(img1, img2, img3, img4, img5)
    [s.shift(4 * LEFT + pos * 2 * RIGHT) for pos, s in enumerate(scene.mobjects)]
    scene.wait()
