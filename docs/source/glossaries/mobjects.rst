========
Mobjects
========


.. note::

    An exhaustive list of all mobjects and their properties can be found in the :doc:`Reference Manual </reference_index/mobjects>`.


.. grid:: 1 1 1 2
    :gutter: 3 3 3 3

    .. grid-item-card:: Angle
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: AngleScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Angle

            class AngleScene(Scene):
                def construct(self) -> None:
                    self.add(
                        Angle(
                            Line((-3, -2, 0), (3, -2, 0)),
                            Line((-3, -2, 0), (2, 2, 0)),
                            radius=2,
                        )
                    )

    .. grid-item-card:: AnnotationDot
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: AnnotationDotScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: AnnotationDot

            class AnnotationDotScene(Scene):
                def construct(self) -> None:
                    self.add(
                        AnnotationDot(
                            radius=1,
                            stroke_width=20,
                        )
                    )

    .. grid-item-card:: AnnularSector
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: AnnularSectorScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: AnnularSector

            class AnnularSectorScene(Scene):
                def construct(self) -> None:
                    self.add(
                        AnnularSector(
                            inner_radius=2,
                            outer_radius=3,
                            angle=TAU / 4,
                            start_angle=0,
                        )
                    )

    .. grid-item-card:: Annulus
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: AnnulusScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Annulus

            class AnnulusScene(Scene):
                def construct(self) -> None:
                    self.add(
                        Annulus(
                            inner_radius=2,
                            outer_radius=3,
                        )
                    )

    .. grid-item-card:: Arc
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: ArcScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Arc

            class ArcScene(Scene):
                def construct(self) -> None:
                    self.add(
                        Arc(
                            radius=3,
                            start_angle=0,
                            angle=TAU / 4,
                        )
                    )

    .. grid-item-card:: ArcBetweenPoints
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: ArcBetweenPointsScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: ArcBetweenPoints

            class ArcBetweenPointsScene(Scene):
                def construct(self) -> None:
                    self.add(
                        ArcBetweenPoints(
                            (2, 0, 0),
                            (-2, 0, 0),
                            angle=TAU / 4,
                        )
                    )

    .. grid-item-card:: ArcBrace
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: ArcBraceScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: ArcBrace

            class ArcBraceScene(Scene):
                def construct(self) -> None:
                    self.add(
                        ArcBrace(
                            Arc(
                                radius=3,
                                start_angle=0,
                                angle=TAU / 4,
                            )
                        )
                    )

    .. grid-item-card:: ArcPolygon
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: ArcPolygonScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: ArcPolygon

            class ArcPolygonScene(Scene):
                def construct(self) -> None:
                    self.add(
                        ArcPolygon(
                            (-2, -2, 0),
                            (3, -2, 0),
                            (-2, 3, 0),
                            radius=5,
                        )
                    )

    .. grid-item-card:: ArcPolygonFromArcs
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: ArcPolygonFromArcsScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: ArcPolygonFromArcs

            class ArcPolygonFromArcsScene(Scene):
                def construct(self) -> None:
                    self.add(
                        ArcPolygonFromArcs(
                            ArcBetweenPoints((-3, -2, 0), (3, -2, 0), radius=6),
                            ArcBetweenPoints((3, -2, 0), (0, 3, 0), radius=6),
                            ArcBetweenPoints((0, 3, 0), (-3, -2, 0), radius=6),
                        )
                    )

    .. grid-item-card:: Arrow
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: ArrowScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Arrow

            class ArrowScene(Scene):
                def construct(self) -> None:
                    self.add(
                        Arrow(
                            (-4, -1, 0),
                            (4, 1, 0),
                        )
                    )

    .. grid-item-card:: Arrow3D
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: Arrow3DScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Arrow3D

            class Arrow3DScene(ThreeDScene):
                def construct(self) -> None:
                    self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
                    self.add(
                        Arrow3D(
                            start=(0, -3, -1),
                            end=(0, 3, 1),
                            thickness=0.05,
                            height=0.8,
                            base_radius=0.25,
                        )
                    )

    .. grid-item-card:: ArrowCircleFilledTip
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: ArrowCircleFilledTipScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: ArrowCircleFilledTip

            class ArrowCircleFilledTipScene(Scene):
                def construct(self) -> None:
                    self.add(
                        Line((-4, 0, 0), (4, 0, 0)).add_tip(
                            ArrowCircleFilledTip(
                                length=1.5,
                            )
                        )
                    )

    .. grid-item-card:: ArrowCircleTip
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: ArrowCircleTipScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: ArrowCircleTip

            class ArrowCircleTipScene(Scene):
                def construct(self) -> None:
                    self.add(
                        Line((-4, 0, 0), (4, 0, 0)).add_tip(
                            ArrowCircleTip(
                                length=1.5,
                            )
                        )
                    )

    .. grid-item-card:: ArrowSquareFilledTip
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: ArrowSquareFilledTipScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: ArrowSquareFilledTip

            class ArrowSquareFilledTipScene(Scene):
                def construct(self) -> None:
                    self.add(
                        Line((-4, 0, 0), (4, 0, 0)).add_tip(
                            ArrowSquareFilledTip(
                                length=1.5,
                            )
                        )
                    )

    .. grid-item-card:: ArrowSquareTip
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: ArrowSquareTipScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: ArrowSquareTip

            class ArrowSquareTipScene(Scene):
                def construct(self) -> None:
                    self.add(
                        Line((-4, 0, 0), (4, 0, 0)).add_tip(
                            ArrowSquareTip(
                                length=1.5,
                            )
                        )
                    )

    .. grid-item-card:: ArrowTriangleFilledTip
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: ArrowTriangleFilledTipScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: ArrowTriangleFilledTip

            class ArrowTriangleFilledTipScene(Scene):
                def construct(self) -> None:
                    self.add(
                        Line((-4, 0, 0), (4, 0, 0)).add_tip(
                            ArrowTriangleFilledTip(
                                length=1.5,
                                width=1.5,
                            )
                        )
                    )

    .. grid-item-card:: ArrowTriangleTip
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: ArrowTriangleTipScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: ArrowTriangleTip

            class ArrowTriangleTipScene(Scene):
                def construct(self) -> None:
                    self.add(
                        Line((-4, 0, 0), (4, 0, 0)).add_tip(
                            ArrowTriangleTip(
                                length=1.5,
                                width=1.5,
                            )
                        )
                    )

    .. grid-item-card:: ArrowVectorField
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: ArrowVectorFieldScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: ArrowVectorField

            class ArrowVectorFieldScene(Scene):
                def construct(self) -> None:
                    self.add(
                        ArrowVectorField(
                            lambda pos: (-pos[1] * RIGHT + pos[0] * UP) / 3,
                        )
                    )

    .. grid-item-card:: Axes
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: AxesScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Axes

            class AxesScene(Scene):
                def construct(self) -> None:
                    self.add(
                        Axes(
                            x_range=(-5, 5, 1),
                            y_range=(-3, 3, 1),
                        )
                    )

    .. grid-item-card:: BackgroundRectangle
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: BackgroundRectangleScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: BackgroundRectangle

            class BackgroundRectangleScene(Scene):
                def construct(self) -> None:
                    text = Text("Hello", font_size=144)
                    self.add(
                        BackgroundRectangle(
                            text,
                            color=DARK_GRAY,
                            fill_opacity=1,
                        ),
                        text,
                    )

    .. grid-item-card:: BarChart
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: BarChartScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: BarChart

            class BarChartScene(Scene):
                def construct(self) -> None:
                    self.add(
                        BarChart(
                            values=[3, 5, 2, 6, 4],
                            bar_names=["A", "B", "C", "D", "E"],
                            y_range=(0, 7, 1),
                        )
                    )

    .. grid-item-card:: Brace
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: BraceScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Brace

            class BraceScene(Scene):
                def construct(self) -> None:
                    self.add(
                        Brace(
                            Square(side_length=4),
                            direction=DOWN,
                        )
                    )

    .. grid-item-card:: BraceBetweenPoints
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: BraceBetweenPointsScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: BraceBetweenPoints

            class BraceBetweenPointsScene(Scene):
                def construct(self) -> None:
                    self.add(
                        BraceBetweenPoints(
                            (-4, 0, 0),
                            (4, 0, 0),
                        )
                    )

    .. grid-item-card:: BraceLabel
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: BraceLabelScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: BraceLabel

            class BraceLabelScene(Scene):
                def construct(self) -> None:
                    self.add(
                        BraceLabel(
                            Square(side_length=4),
                            "x",
                            brace_direction=DOWN,
                            font_size=96,
                        ),
                    )

    .. grid-item-card:: BraceText
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: BraceTextScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: BraceText

            class BraceTextScene(Scene):
                def construct(self) -> None:
                    self.add(
                        BraceText(
                            Square(side_length=4),
                            "width",
                        ),
                    )

    .. grid-item-card:: BulletedList
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: BulletedListScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: BulletedList

            class BulletedListScene(Scene):
                def construct(self) -> None:
                    self.add(
                        BulletedList(
                            "First",
                            "Second",
                            "Third",
                            font_size=96,
                        )
                    )

    .. grid-item-card:: Circle
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: CircleScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Circle

            class CircleScene(Scene):
                def construct(self) -> None:
                    self.add(
                        Circle(
                            radius=3,
                        )
                    )

    .. grid-item-card:: Code
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: CodeScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Code

            class CodeScene(Scene):
                def construct(self) -> None:
                    self.add(
                        Code(
                            code_string='def hello():\n    print("Hello")\n',
                            language="python",
                            paragraph_config={"font_size": 48},
                        )
                    )

    .. grid-item-card:: ComplexPlane
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: ComplexPlaneScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: ComplexPlane

            class ComplexPlaneScene(Scene):
                def construct(self) -> None:
                    self.add(
                        ComplexPlane(
                            x_range=(-4, 4, 1),
                            y_range=(-3, 3, 1),
                        )
                    )

    .. grid-item-card:: Cone
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: ConeScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Cone

            class ConeScene(ThreeDScene):
                def construct(self) -> None:
                    self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
                    self.add(
                        Cone(
                            base_radius=2,
                            height=3,
                        )
                    )

    .. grid-item-card:: ConvexHull
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: ConvexHullScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: ConvexHull

            class ConvexHullScene(Scene):
                def construct(self) -> None:
                    self.add(
                        ConvexHull(
                            (-3, -2, 0),
                            (3, -2, 0),
                            (4, 1, 0),
                            (1, 3, 0),
                            (-2, 2, 0),
                            (0, 0, 0),
                            (1, -1, 0),
                            (-1, 1, 0),
                        )
                    )

    .. grid-item-card:: ConvexHull3D
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: ConvexHull3DScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: ConvexHull3D

            class ConvexHull3DScene(ThreeDScene):
                def construct(self) -> None:
                    self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
                    self.add(
                        ConvexHull3D(
                            *np.array(
                                [
                                    (2, 0, 0),
                                    (-2, 0, 0),
                                    (0, 2, 0),
                                    (0, -2, 0),
                                    (0, 0, 2),
                                    (0, 0, -2),
                                    (1.5, 1.5, 1.5),
                                    (-1.5, -1.5, 1.5),
                                ]
                            )
                        )
                    )

    .. grid-item-card:: Cross
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: CrossScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Cross

            class CrossScene(Scene):
                def construct(self) -> None:
                    square = Square(side_length=4)
                    self.add(
                        Cross()
                    )

    .. grid-item-card:: Cube
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: CubeScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Cube

            class CubeScene(ThreeDScene):
                def construct(self) -> None:
                    self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
                    self.add(
                        Cube(
                            side_length=3,
                        )
                    )

    .. grid-item-card:: CubicBezier
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: CubicBezierScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: CubicBezier

            class CubicBezierScene(Scene):
                def construct(self) -> None:
                    self.add(
                        CubicBezier(
                            (-4, 2, 0),
                            (-4, -2, 0),
                            (4, 2, 0),
                            (4, -2, 0),
                        )
                    )

    .. grid-item-card:: CurvedArrow
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: CurvedArrowScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: CurvedArrow

            class CurvedArrowScene(Scene):
                def construct(self) -> None:
                    self.add(
                        CurvedArrow(
                            (-4, 0, 0),
                            (4, 0, 0),
                            angle=TAU / 4,
                            start_angle=0,
                        )
                    )

    .. grid-item-card:: CurvedDoubleArrow
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: CurvedDoubleArrowScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: CurvedDoubleArrow

            class CurvedDoubleArrowScene(Scene):
                def construct(self) -> None:
                    self.add(
                        CurvedDoubleArrow(
                            (-4, 0, 0),
                            (4, 0, 0),
                            angle=TAU / 4,
                            start_angle=0,
                        )
                    )

    .. grid-item-card:: CurvesAsSubmobjects
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: CurvesAsSubmobjectsScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: CurvesAsSubmobjects

            class CurvesAsSubmobjectsScene(Scene):
                def construct(self) -> None:
                    self.add(
                        CurvesAsSubmobjects(
                            Circle(radius=3, stroke_width=12),
                        ).set_color_by_gradient(BLUE, YELLOW)
                    )

    .. grid-item-card:: Cutout
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: CutoutScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Cutout

            class CutoutScene(Scene):
                def construct(self) -> None:
                    self.add(
                        Cutout(
                            Square(side_length=6),
                            Circle(radius=1).shift(UR),
                            Square(side_length=2).shift(DL),
                            fill_opacity=1,
                        )
                    )

    .. grid-item-card:: Cylinder
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: CylinderScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Cylinder

            class CylinderScene(ThreeDScene):
                def construct(self) -> None:
                    self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
                    self.add(
                        Cylinder(
                            radius=1.5,
                            height=3,
                        )
                    )

    .. grid-item-card:: DashedLine
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: DashedLineScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: DashedLine

            class DashedLineScene(Scene):
                def construct(self) -> None:
                    self.add(
                        DashedLine(
                            (-4, -1, 0),
                            (4, 1, 0),
                            dash_length=0.5,
                        )
                    )

    .. grid-item-card:: DashedVMobject
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: DashedVMobjectScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: DashedVMobject

            class DashedVMobjectScene(Scene):
                def construct(self) -> None:
                    self.add(
                        DashedVMobject(
                            Circle(radius=3),
                            num_dashes=20,
                        )
                    )

    .. grid-item-card:: DecimalMatrix
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: DecimalMatrixScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: DecimalMatrix

            class DecimalMatrixScene(Scene):
                def construct(self) -> None:
                    self.add(
                        DecimalMatrix(
                            [
                                [1.5, 2.25],
                                [3.0, -4.75],
                            ],
                        ).scale(2)
                    )

    .. grid-item-card:: DecimalNumber
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: DecimalNumberScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: DecimalNumber

            class DecimalNumberScene(Scene):
                def construct(self) -> None:
                    self.add(
                        DecimalNumber(
                            3.14159,
                            num_decimal_places=3,
                            font_size=144,
                        )
                    )

    .. grid-item-card:: DecimalTable
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: DecimalTableScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: DecimalTable

            class DecimalTableScene(Scene):
                def construct(self) -> None:
                    self.add(
                        DecimalTable(
                            [
                                [1.5, 2.5],
                                [3.0, -4.5],
                            ],
                        ).scale(2)
                    )

    .. grid-item-card:: Difference
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: DifferenceScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Difference

            class DifferenceScene(Scene):
                def construct(self) -> None:
                    self.add(
                        Difference(
                            Square(side_length=4),
                            Circle(radius=1.5).move_to((1.5, 1.5, 0)),
                            fill_opacity=1,
                        )
                    )

    .. grid-item-card:: DiGraph
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: DiGraphScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: DiGraph

            class DiGraphScene(Scene):
                def construct(self) -> None:
                    self.add(
                        DiGraph(
                            [1, 2, 3, 4],
                            [(1, 2), (2, 3), (3, 4), (4, 1), (1, 3)],
                            layout="circular",
                            layout_scale=3,
                            labels=True,
                        )
                    )

    .. grid-item-card:: Dodecahedron
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: DodecahedronScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Dodecahedron

            class DodecahedronScene(ThreeDScene):
                def construct(self) -> None:
                    self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
                    self.add(
                        Dodecahedron(
                            edge_length=2,
                        )
                    )

    .. grid-item-card:: Dot
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: DotScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Dot

            class DotScene(Scene):
                def construct(self) -> None:
                    self.add(
                        Dot(
                            (0, 0, 0),
                            radius=1,
                        )
                    )

    .. grid-item-card:: Dot3D
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: Dot3DScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Dot3D

            class Dot3DScene(ThreeDScene):
                def construct(self) -> None:
                    self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
                    self.add(
                        Dot3D(
                            point=(0, 0, 0),
                            radius=1,
                        )
                    )

    .. grid-item-card:: DoubleArrow
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: DoubleArrowScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: DoubleArrow

            class DoubleArrowScene(Scene):
                def construct(self) -> None:
                    self.add(
                        DoubleArrow(
                            (-4, -1, 0),
                            (4, 1, 0),
                        )
                    )

    .. grid-item-card:: Elbow
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: ElbowScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Elbow

            class ElbowScene(Scene):
                def construct(self) -> None:
                    self.add(
                        Elbow(
                            width=3,
                            angle=0,
                        )
                    )

    .. grid-item-card:: Ellipse
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: EllipseScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Ellipse

            class EllipseScene(Scene):
                def construct(self) -> None:
                    self.add(
                        Ellipse(
                            width=8,
                            height=4,
                        )
                    )

    .. grid-item-card:: Exclusion
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: ExclusionScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Exclusion

            class ExclusionScene(Scene):
                def construct(self) -> None:
                    self.add(
                        Exclusion(
                            Square(side_length=4),
                            Circle(radius=1.5).move_to((1.5, 1.5, 0)),
                            fill_opacity=1,
                        )
                    )

    .. grid-item-card:: FullScreenRectangle
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: FullScreenRectangleScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: FullScreenRectangle

            class FullScreenRectangleScene(Scene):
                def construct(self) -> None:
                    self.add(
                        FullScreenRectangle(
                            fill_color=GRAY,
                            fill_opacity=1,
                        )
                    )

    .. grid-item-card:: FunctionGraph
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: FunctionGraphScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: FunctionGraph

            class FunctionGraphScene(Scene):
                def construct(self) -> None:
                    self.add(
                        FunctionGraph(
                            lambda x: 2 * np.sin(x),
                            x_range=(-2 * PI, 2 * PI),
                        )
                    )

    .. grid-item-card:: Graph
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: GraphScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Graph

            class GraphScene(Scene):
                def construct(self) -> None:
                    self.add(
                        Graph(
                            [1, 2, 3, 4],
                            [(1, 2), (2, 3), (3, 4), (4, 1), (1, 3)],
                            layout="circular",
                            layout_scale=3,
                            labels=True,
                        )
                    )

    .. grid-item-card:: Group
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: GroupScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Group

            class GroupScene(Scene):
                def construct(self) -> None:
                    self.add(
                        Group(
                            Circle(radius=1.5),
                            Square(side_length=3),
                            Triangle(radius=1.5),
                        ).arrange(RIGHT)
                    )

    .. grid-item-card:: Icosahedron
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: IcosahedronScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Icosahedron

            class IcosahedronScene(ThreeDScene):
                def construct(self) -> None:
                    self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
                    self.add(
                        Icosahedron(
                            edge_length=3,
                        )
                    )

    .. grid-item-card:: ImageMobject
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: ImageMobjectScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: ImageMobject

            class ImageMobjectScene(Scene):
                def construct(self) -> None:
                    self.add(
                        ImageMobject(
                            np.uint8(
                                [
                                    [
                                        [252, 98, 85, 255],
                                        [131, 193, 103, 255],
                                    ],
                                    [
                                        [88, 196, 221, 255],
                                        [255, 255, 0, 255],
                                    ],
                                ]
                            ),
                            scale_to_resolution=4,
                        )
                    )

    .. grid-item-card:: ImplicitFunction
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: ImplicitFunctionScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: ImplicitFunction

            class ImplicitFunctionScene(Scene):
                def construct(self) -> None:
                    self.add(
                        ImplicitFunction(
                            lambda x, y: x * y**2 - x**2 * y - 2,
                        )
                    )

    .. grid-item-card:: Integer
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: IntegerScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Integer

            class IntegerScene(Scene):
                def construct(self) -> None:
                    self.add(
                        Integer(
                            42,
                            font_size=144,
                        )
                    )

    .. grid-item-card:: IntegerMatrix
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: IntegerMatrixScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: IntegerMatrix

            class IntegerMatrixScene(Scene):
                def construct(self) -> None:
                    self.add(
                        IntegerMatrix(
                            [
                                [1, 2],
                                [3, 4],
                            ],
                        ).scale(2)
                    )

    .. grid-item-card:: IntegerTable
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: IntegerTableScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: IntegerTable

            class IntegerTableScene(Scene):
                def construct(self) -> None:
                    self.add(
                        IntegerTable(
                            [
                                [1, 2],
                                [3, 4],
                            ],
                        ).scale(2)
                    )

    .. grid-item-card:: Intersection
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: IntersectionScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Intersection

            class IntersectionScene(Scene):
                def construct(self) -> None:
                    self.add(
                        Intersection(
                            Square(side_length=4),
                            Circle(radius=1.5).move_to((1.5, 1.5, 0)),
                            fill_opacity=1,
                        )
                    )

    .. grid-item-card:: Label
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: LabelScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Label

            class LabelScene(Scene):
                def construct(self) -> None:
                    self.add(
                        Label(
                            "Hello",
                            label_config={
                                "font_size": 256,
                            },
                            box_config={
                                "fill_color": DARK_GRAY,
                            },
                        )
                    )

    .. grid-item-card:: LabeledArrow
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: LabeledArrowScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: LabeledArrow

            class LabeledArrowScene(Scene):
                def construct(self) -> None:
                    self.add(
                        LabeledArrow(
                            "Hello",
                            start=(-4, -1, 0),
                            end=(4, 1, 0),
                        )
                    )

    .. grid-item-card:: LabeledDot
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: LabeledDotScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: LabeledDot

            class LabeledDotScene(Scene):
                def construct(self) -> None:
                    self.add(
                        LabeledDot(
                            "42",
                            radius=1,
                        )
                    )

    .. grid-item-card:: LabeledLine
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: LabeledLineScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: LabeledLine

            class LabeledLineScene(Scene):
                def construct(self) -> None:
                    self.add(
                        LabeledLine(
                            "Hello",
                            start=(-4, -1, 0),
                            end=(4, 1, 0),
                        )
                    )

    .. grid-item-card:: LabeledPolygram
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: LabeledPolygramScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: LabeledPolygram

            class LabeledPolygramScene(Scene):
                def construct(self) -> None:
                    self.add(
                        LabeledPolygram(
                            [
                                (-3, -2, 0),
                                (-1, 2, 0),
                                (1, 2, 0),
                                (3, -2, 0),
                            ],
                            label="Hello",
                        )
                    )

    .. grid-item-card:: Line
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: LineScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Line

            class LineScene(Scene):
                def construct(self) -> None:
                    self.add(
                        Line(
                            (-4, -1, 0),
                            (4, 1, 0),
                        )
                    )

    .. grid-item-card:: Line3D
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: Line3DScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Line3D

            class Line3DScene(ThreeDScene):
                def construct(self) -> None:
                    self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
                    self.add(
                        Line3D(
                            start=(0, -3, -1),
                            end=(0, 3, 1),
                            thickness=0.1,
                        )
                    )

    .. grid-item-card:: ManimBanner
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: ManimBannerScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: ManimBanner

            class ManimBannerScene(Scene):
                def construct(self) -> None:
                    self.add(
                        ManimBanner(
                            dark_theme=True,
                        )
                    )

    .. grid-item-card:: MarkupText
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: MarkupTextScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: MarkupText

            class MarkupTextScene(Scene):
                def construct(self) -> None:
                    self.add(
                        MarkupText(
                            '<b>Hello</b> <i>World</i>',
                            font_size=96,
                        )
                    )

    .. grid-item-card:: MathTable
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: MathTableScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: MathTable

            class MathTableScene(Scene):
                def construct(self) -> None:
                    self.add(
                        MathTable(
                            [
                                ["x", "x^2"],
                                ["2", "4"],
                            ],
                        ).scale(2)
                    )

    .. grid-item-card:: MathTex
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: MathTexScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: MathTex

            class MathTexScene(Scene):
                def construct(self) -> None:
                    self.add(
                        MathTex(
                            r"e^{i \pi} + 1 = 0",
                            font_size=144,
                        )
                    )

    .. grid-item-card:: MathTypst
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: MathTypstScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: MathTypst

            class MathTypstScene(Scene):
                def construct(self) -> None:
                    self.add(
                        MathTypst(
                            "e^(i pi) + 1 = 0",
                            font_size=144,
                        )
                    )

    .. grid-item-card:: Matrix
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: MatrixScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Matrix

            class MatrixScene(Scene):
                def construct(self) -> None:
                    self.add(
                        Matrix(
                            [
                                ["a", "b"],
                                ["c", "d"],
                            ],
                        ).scale(2)
                    )

    .. grid-item-card:: Mobject1D
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: Mobject1DScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Mobject1D

            class Mobject1DScene(Scene):
                def construct(self) -> None:
                    self.add(
                        Mobject1D(
                            density=10,
                            stroke_width=8,
                        ).add_line(
                            (-4, -1, 0),
                            (4, 1, 0),
                        )
                    )

    .. grid-item-card:: MobjectMatrix
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: MobjectMatrixScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: MobjectMatrix

            class MobjectMatrixScene(Scene):
                def construct(self) -> None:
                    self.add(
                        MobjectMatrix(
                            [
                                [Circle(radius=0.4), Square(side_length=0.8)],
                                [Triangle(radius=0.4), Star(outer_radius=0.4)],
                            ],
                        ).scale(2)
                    )

    .. grid-item-card:: MobjectTable
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: MobjectTableScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: MobjectTable

            class MobjectTableScene(Scene):
                def construct(self) -> None:
                    self.add(
                        MobjectTable(
                            [
                                [Circle(radius=1), Square(side_length=2)],
                                [Triangle(radius=1), Star(outer_radius=1)],
                            ],
                        )
                    )

    .. grid-item-card:: NumberLine
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: NumberLineScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: NumberLine

            class NumberLineScene(Scene):
                def construct(self) -> None:
                    self.add(
                        NumberLine(
                            x_range=(-5, 5, 1),
                            length=12,
                            include_numbers=True,
                        )
                    )

    .. grid-item-card:: NumberPlane
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: NumberPlaneScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: NumberPlane

            class NumberPlaneScene(Scene):
                def construct(self) -> None:
                    self.add(
                        NumberPlane(
                            x_range=(-7, 7, 1),
                            y_range=(-4, 4, 1),
                        )
                    )

    .. grid-item-card:: Octahedron
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: OctahedronScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Octahedron

            class OctahedronScene(ThreeDScene):
                def construct(self) -> None:
                    self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
                    self.add(
                        Octahedron(
                            edge_length=4,
                        )
                    )

    .. grid-item-card:: Paragraph
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: ParagraphScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Paragraph

            class ParagraphScene(Scene):
                def construct(self) -> None:
                    self.add(
                        Paragraph(
                            "Hello",
                            "World",
                            alignment="center",
                            font_size=96,
                        )
                    )

    .. grid-item-card:: ParametricFunction
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: ParametricFunctionScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: ParametricFunction

            class ParametricFunctionScene(Scene):
                def construct(self) -> None:
                    self.add(
                        ParametricFunction(
                            lambda t: (4 * np.sin(3 * t), 3 * np.sin(2 * t), 0),
                            t_range=(0, TAU),
                        )
                    )

    .. grid-item-card:: PGroup
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: PGroupScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: PGroup

            class PGroupScene(Scene):
                def construct(self) -> None:
                    self.add(
                        PGroup(
                            PointCloudDot(center=(-2, 0, 0), radius=1.5),
                            PointCloudDot(center=(2, 0, 0), radius=1.5, color=BLUE),
                        )
                    )

    .. grid-item-card:: PMobject
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: PMobjectScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: PMobject

            class PMobjectScene(Scene):
                def construct(self) -> None:
                    self.add(
                        PMobject(
                            stroke_width=20,
                        ).add_points(
                            [(x, y, 0) for x in range(-4, 5) for y in range(-2, 3)],
                        )
                    )

    .. grid-item-card:: PointCloudDot
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: PointCloudDotScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: PointCloudDot

            class PointCloudDotScene(Scene):
                def construct(self) -> None:
                    self.add(
                        PointCloudDot(
                            radius=2.5,
                        )
                    )

    .. grid-item-card:: PolarPlane
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: PolarPlaneScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: PolarPlane

            class PolarPlaneScene(Scene):
                def construct(self) -> None:
                    self.add(
                        PolarPlane(
                            radius_max=3,
                        )
                    )

    .. grid-item-card:: Polygon
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: PolygonScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Polygon

            class PolygonScene(Scene):
                def construct(self) -> None:
                    self.add(
                        Polygon(
                            (-3, -2, 0),
                            (3, -2, 0),
                            (2, 2, 0),
                            (-2, 3, 0),
                        )
                    )

    .. grid-item-card:: Polygram
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: PolygramScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Polygram

            class PolygramScene(Scene):
                def construct(self) -> None:
                    self.add(
                        Polygram(
                            [(-4, -2, 0), (-2, 2, 0), (0, -2, 0)],
                            [(0, -2, 0), (2, 2, 0), (4, -2, 0)],
                        )
                    )

    .. grid-item-card:: Polyhedron
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: PolyhedronScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Polyhedron

            class PolyhedronScene(ThreeDScene):
                def construct(self) -> None:
                    self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
                    self.add(
                        Polyhedron(
                            vertex_coords=[
                                (-2, -2, -1),
                                (2, -2, -1),
                                (2, 2, -1),
                                (-2, 2, -1),
                                (0, 0, 2),
                            ],
                            faces_list=[
                                [0, 1, 2, 3],
                                [0, 1, 4],
                                [1, 2, 4],
                                [2, 3, 4],
                                [3, 0, 4],
                            ],
                        )
                    )

    .. grid-item-card:: Prism
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: PrismScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Prism

            class PrismScene(ThreeDScene):
                def construct(self) -> None:
                    self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
                    self.add(
                        Prism(
                            dimensions=(4, 3, 2),
                        )
                    )

    .. grid-item-card:: Rectangle
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: RectangleScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Rectangle

            class RectangleScene(Scene):
                def construct(self) -> None:
                    self.add(
                        Rectangle(
                            width=6,
                            height=4,
                        )
                    )

    .. grid-item-card:: RegularPolygon
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: RegularPolygonScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: RegularPolygon

            class RegularPolygonScene(Scene):
                def construct(self) -> None:
                    self.add(
                        RegularPolygon(
                            n=6,
                            radius=3,
                        )
                    )

    .. grid-item-card:: RegularPolygram
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: RegularPolygramScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: RegularPolygram

            class RegularPolygramScene(Scene):
                def construct(self) -> None:
                    self.add(
                        RegularPolygram(
                            num_vertices=7,
                            density=3,
                            radius=3,
                        )
                    )

    .. grid-item-card:: RightAngle
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: RightAngleScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: RightAngle

            class RightAngleScene(Scene):
                def construct(self) -> None:
                    self.add(
                        RightAngle(
                            Line((-3, -2, 0), (3, -2, 0)),
                            Line((-3, -2, 0), (-3, 3, 0)),
                            length=1,
                        ),
                    )

    .. grid-item-card:: RoundedRectangle
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: RoundedRectangleScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: RoundedRectangle

            class RoundedRectangleScene(Scene):
                def construct(self) -> None:
                    self.add(
                        RoundedRectangle(
                            corner_radius=1,
                            width=6,
                            height=4,
                        )
                    )

    .. grid-item-card:: SampleSpace
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: SampleSpaceScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: SampleSpace

            class SampleSpaceScene(Scene):
                def construct(self) -> None:
                    self.add(
                        SampleSpace(
                            height=5,
                            width=5,
                        ).divide_horizontally(0.3)
                    )

    .. grid-item-card:: ScreenRectangle
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: ScreenRectangleScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: ScreenRectangle

            class ScreenRectangleScene(Scene):
                def construct(self) -> None:
                    self.add(
                        ScreenRectangle(
                            height=6,
                        )
                    )

    .. grid-item-card:: Sector
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: SectorScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Sector

            class SectorScene(Scene):
                def construct(self) -> None:
                    self.add(
                        Sector(
                            radius=3,
                            start_angle=0,
                            angle=TAU / 4,
                        )
                    )

    .. grid-item-card:: SingleStringMathTex
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: SingleStringMathTexScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: SingleStringMathTex

            class SingleStringMathTexScene(Scene):
                def construct(self) -> None:
                    self.add(
                        SingleStringMathTex(
                            r"\int_0^1 x^2 \, dx",
                            font_size=144,
                        )
                    )

    .. grid-item-card:: Sphere
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: SphereScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Sphere

            class SphereScene(ThreeDScene):
                def construct(self) -> None:
                    self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
                    self.add(
                        Sphere(
                            radius=2,
                        )
                    )

    .. grid-item-card:: Square
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: SquareScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Square

            class SquareScene(Scene):
                def construct(self) -> None:
                    self.add(
                        Square(
                            side_length=5,
                        )
                    )

    .. grid-item-card:: Star
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: StarScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Star

            class StarScene(Scene):
                def construct(self) -> None:
                    self.add(
                        Star(
                            n=5,
                            outer_radius=3,
                        )
                    )

    .. grid-item-card:: StealthTip
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: StealthTipScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: StealthTip

            class StealthTipScene(Scene):
                def construct(self) -> None:
                    self.add(
                        Line((-4, 0, 0), (4, 0, 0)).add_tip(
                            StealthTip(
                                length=1.5,
                            )
                        )
                    )

    .. grid-item-card:: StreamLines
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: StreamLinesScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: StreamLines

            class StreamLinesScene(Scene):
                def construct(self) -> None:
                    self.add(
                        StreamLines(
                            lambda pos: (-pos[1] * RIGHT + pos[0] * UP) / 3,
                            stroke_width=2,
                        )
                    )

    .. grid-item-card:: Surface
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: SurfaceScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Surface

            class SurfaceScene(ThreeDScene):
                def construct(self) -> None:
                    self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
                    self.add(
                        Surface(
                            lambda u, v: np.array([u, v, np.sin(u) * np.cos(v)]),
                            u_range=(-3, 3),
                            v_range=(-3, 3),
                        )
                    )

    .. grid-item-card:: SurroundingRectangle
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: SurroundingRectangleScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: SurroundingRectangle

            class SurroundingRectangleScene(Scene):
                def construct(self) -> None:
                    self.add(
                        Text("Hello", font_size=144),
                        SurroundingRectangle(
                            Text("Hello", font_size=144),
                        ),
                    )

    .. grid-item-card:: SVGMobject
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: SVGMobjectScene
            :save_last_frame:
            :hide_levels: 2
            :title: None
            :ref_classes: SVGMobject

            import tempfile
            from pathlib import Path

            class SVGMobjectScene(Scene):
                def construct(self) -> None:
                    svg_file = Path(tempfile.mkdtemp()) / "heart.svg"
                    svg_file.write_text(
                        '<svg xmlns="http://www.w3.org/2000/svg">'
                        '<path d="M50 90 L10 50 A20 20 0 0 1 50 20 A20 20 0 0 1 90 50 Z" fill="#FC6255"/>'
                        "</svg>"
                    )
                    self.add(
                        SVGMobject(
                            svg_file,
                            height=6,
                        )
                    )

    .. grid-item-card:: Table
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: TableScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Table

            class TableScene(Scene):
                def construct(self) -> None:
                    self.add(
                        Table(
                            [
                                ["A", "B"],
                                ["C", "D"],
                            ],
                        ).scale(2)
                    )

    .. grid-item-card:: TangentialArc
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: TangentialArcScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: TangentialArc

            class TangentialArcScene(Scene):
                def construct(self) -> None:
                    self.add(
                        TangentialArc(
                            Line((0, 0, 0), (0, 2, 0)),
                            Line((0, 0, 0), (2, 0, 0)),
                            radius=3,
                            corner=(1, 1),
                        ),
                    )

    .. grid-item-card:: TangentLine
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: TangentLineScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: TangentLine

            class TangentLineScene(Scene):
                def construct(self) -> None:
                    circle = Circle(radius=3)
                    self.add(
                        circle,
                        TangentLine(
                            circle,
                            alpha=1 / 8,
                            length=6,
                        ),
                    )

    .. grid-item-card:: Tetrahedron
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: TetrahedronScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Tetrahedron

            class TetrahedronScene(ThreeDScene):
                def construct(self) -> None:
                    self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
                    self.add(
                        Tetrahedron(
                            edge_length=4,
                        )
                    )

    .. grid-item-card:: Tex
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: TexScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Tex

            class TexScene(Scene):
                def construct(self) -> None:
                    self.add(
                        Tex(
                            r"Hello \LaTeX",
                            font_size=144,
                        )
                    )

    .. grid-item-card:: Text
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: TextScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Text

            class TextScene(Scene):
                def construct(self) -> None:
                    self.add(
                        Text(
                            "Hello",
                            font_size=144,
                        )
                    )

    .. grid-item-card:: ThreeDAxes
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: ThreeDAxesScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: ThreeDAxes

            class ThreeDAxesScene(ThreeDScene):
                def construct(self) -> None:
                    self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
                    self.add(
                        ThreeDAxes(
                            x_range=(-4, 4, 1),
                            y_range=(-4, 4, 1),
                            z_range=(-3, 3, 1),
                        )
                    )

    .. grid-item-card:: TipableVMobject
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: TipableVMobjectScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: TipableVMobject

            class TipableVMobjectScene(Scene):
                def construct(self) -> None:
                    self.add(
                        TipableVMobject()
                        .set_points_smoothly(
                            [
                                (-4, -2, 0),
                                (-1, 2, 0),
                                (1, -2, 0),
                                (4, 2, 0),
                            ]
                        )
                        .add_tip()
                        .add_tip(at_start=True)
                    )

    .. grid-item-card:: Title
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: TitleScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Title

            class TitleScene(Scene):
                def construct(self) -> None:
                    self.add(
                        Title(
                            "Hello",
                            font_size=96,
                        )
                    )

    .. grid-item-card:: Torus
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: TorusScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Torus

            class TorusScene(ThreeDScene):
                def construct(self) -> None:
                    self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
                    self.add(
                        Torus(
                            major_radius=2.5,
                            minor_radius=1,
                        )
                    )

    .. grid-item-card:: Triangle
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: TriangleScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Triangle

            class TriangleScene(Scene):
                def construct(self) -> None:
                    self.add(
                        Triangle(
                            radius=3,
                        )
                    )

    .. grid-item-card:: Typst
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: TypstScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Typst

            class TypstScene(Scene):
                def construct(self) -> None:
                    self.add(
                        Typst(
                            "*Hello* _Typst_",
                            font_size=144,
                        )
                    )

    .. grid-item-card:: Underline
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: UnderlineScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Underline

            class UnderlineScene(Scene):
                def construct(self) -> None:
                    self.add(
                        Underline(
                            Text("Hello", font_size=144),
                        )
                    )

    .. grid-item-card:: Union
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: UnionScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Union

            class UnionScene(Scene):
                def construct(self) -> None:
                    self.add(
                        Union(
                            Square(side_length=4),
                            Circle(radius=1.5).move_to((1.5, 1.5, 0)),
                            fill_opacity=1,
                        )
                    )

    .. grid-item-card:: UnitInterval
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: UnitIntervalScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: UnitInterval

            class UnitIntervalScene(Scene):
                def construct(self) -> None:
                    self.add(
                        UnitInterval(
                            unit_size=10,
                            include_numbers=True,
                        )
                    )

    .. grid-item-card:: Variable
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: VariableScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Variable

            class VariableScene(Scene):
                def construct(self) -> None:
                    self.add(
                        Variable(
                            3.14,
                            label="x",
                            num_decimal_places=2,
                        ).scale(2)
                    )

    .. grid-item-card:: VDict
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: VDictScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: VDict

            class VDictScene(Scene):
                def construct(self) -> None:
                    self.add(
                        VDict(
                            {
                                "circle": Circle(),
                                "square": Square(),
                                "triangle": Triangle(),
                                "star": Star(),
                            },
                            show_keys=True,
                        ).arrange_in_grid(2)
                    )

    .. grid-item-card:: Vector
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: VectorScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: Vector

            class VectorScene(Scene):
                def construct(self) -> None:
                    self.add(
                        Vector(
                            (3, 2),
                        )
                    )

    .. grid-item-card:: VGroup
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: VGroupScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: VGroup

            class VGroupScene(Scene):
                def construct(self) -> None:
                    self.add(
                        VGroup(
                            Circle(radius=1.5),
                            Square(side_length=3),
                            Triangle(radius=1.5),
                        ).arrange(RIGHT)
                    )

    .. grid-item-card:: VMobject
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: VMobjectScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: VMobject

            class VMobjectScene(Scene):
                def construct(self) -> None:
                    self.add(
                        VMobject().set_points_as_corners(
                            [
                                (-4, -2, 0),
                                (-2, 2, 0),
                                (0, -2, 0),
                                (2, 2, 0),
                                (4, -2, 0),
                            ]
                        )
                    )

    .. grid-item-card:: VMobjectFromSVGPath
        :class-title: sd-text-center
        :class-body: sd-p-1

        .. manim:: VMobjectFromSVGPathScene
            :save_last_frame:
            :hide_levels: 3
            :title: None
            :ref_classes: VMobjectFromSVGPath

            class VMobjectFromSVGPathScene(Scene):
                def construct(self) -> None:
                    import svgelements as se

                    self.add(
                        VMobjectFromSVGPath(
                            se.Path("M -4 0 C -2 -4 2 4 4 0"),
                        )
                    )
