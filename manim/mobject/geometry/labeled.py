r"""Mobjects that inherit from lines and contain a label along the length."""

from __future__ import annotations

__all__ = ["Label", "LabeledLine", "LabeledArrow", "LabeledPolygram"]

from typing import TYPE_CHECKING

import numpy as np

from manim.constants import *
from manim.mobject.geometry.line import Arrow, Line
from manim.mobject.geometry.polygram import Polygram
from manim.mobject.geometry.shape_matchers import (
    BackgroundRectangle,
    SurroundingRectangle,
)
from manim.mobject.text.tex_mobject import MathTex, Tex
from manim.mobject.text.text_mobject import Text
from manim.mobject.types.vectorized_mobject import VGroup
from manim.utils.color import WHITE
from manim.utils.polylabel import polylabel

if TYPE_CHECKING:
    from typing import Any

    from manim.typing import Point3DLike_Array


class Label(VGroup):
    """A Label consisting of text surrounded by a frame.

    Parameters
    ----------
    label
        Label that will be displayed.
    label_config
        A dictionary containing the configuration for the label.
        This is only applied if ``label`` is of type ``str``.
    box_config
        A dictionary containing the configuration for the background box.
    frame_config
         A dictionary containing the configuration for the frame.

    Examples
    --------
    .. manim:: LabelExample
        :save_last_frame:
        :quality: high

        class LabelExample(Scene):
            def construct(self):
                label = Label(
                    label=Text('Label Text', font='sans-serif'),
                    box_config = {
                        "color" : BLUE,
                        "fill_opacity" : 0.75
                    }
                )
                label.scale(3)
                self.add(label)
    """

    def __init__(
        self,
        label: str | Tex | MathTex | Text,
        label_config: dict[str, Any] | None = None,
        box_config: dict[str, Any] | None = None,
        frame_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # Setup Defaults
        default_label_config: dict[str, Any] = {
            "color": WHITE,
            "font_size": DEFAULT_FONT_SIZE,
        }

        default_box_config: dict[str, Any] = {
            "color": None,
            "buff": 0.05,
            "fill_opacity": 1,
            "stroke_width": 0.5,
        }

        default_frame_config: dict[str, Any] = {
            "color": WHITE,
            "buff": 0.05,
            "stroke_width": 0.5,
        }

        # Merge Defaults
        label_config = default_label_config | (label_config or {})
        box_config = default_box_config | (box_config or {})
        frame_config = default_frame_config | (frame_config or {})

        # Determine the type of label and instantiate the appropriate object
        self.rendered_label: MathTex | Tex | Text
        if isinstance(label, str):
            self.rendered_label = MathTex(label, **label_config)
        elif isinstance(label, (MathTex, Tex, Text)):
            self.rendered_label = label
        else:
            raise TypeError("Unsupported label type. Must be MathTex, Tex, or Text.")

        # Add a background box
        self.background_rect = BackgroundRectangle(self.rendered_label, **box_config)

        # Add a frame around the label
        self.frame = SurroundingRectangle(self.rendered_label, **frame_config)

        # Add components to the VGroup
        self.add(self.background_rect, self.rendered_label, self.frame)


class LabeledLine(Line):
    """Constructs a line containing a label box somewhere along its length.

    Parameters
    ----------
    label
        Label that will be displayed on the line.
    label_position
        A ratio in the range [0-1] to indicate the position of the label with respect to the length of the line. Default value is 0.5.
    label_config
        A dictionary containing the configuration for the label.
        This is only applied if ``label`` is of type ``str``.
    box_config
        A dictionary containing the configuration for the background box.
    frame_config
         A dictionary containing the configuration for the frame.

        .. seealso::
            :class:`LabeledArrow`

    Examples
    --------
    .. manim:: LabeledLineExample
        :save_last_frame:
        :quality: high

        class LabeledLineExample(Scene):
            def construct(self):
                line = LabeledLine(
                    label          = '0.5',
                    label_position = 0.8,
                    label_config = {
                        "font_size" : 20
                    },
                    start=LEFT+DOWN,
                    end=RIGHT+UP)

                line.set_length(line.get_length() * 2)
                self.add(line)
    """

    def __init__(
        self,
        label: str | Tex | MathTex | Text,
        label_position: float = 0.5,
        label_config: dict[str, Any] | None = None,
        box_config: dict[str, Any] | None = None,
        frame_config: dict[str, Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        # Create Label
        self.label = Label(
            label=label,
            label_config=label_config,
            box_config=box_config,
            frame_config=frame_config,
        )

        # Compute Label Position
        line_start, line_end = self.get_start_and_end()
        new_vec = (line_end - line_start) * label_position
        label_coords = line_start + new_vec

        self.label.move_to(label_coords)
        self.add(self.label)


class LabeledArrow(LabeledLine, Arrow):
    """Constructs an arrow containing a label box somewhere along its length.
    This class inherits its label properties from `LabeledLine`, so the main parameters controlling it are the same.

    Parameters
    ----------
    label
        Label that will be displayed on the Arrow.
    label_position
        A ratio in the range [0-1] to indicate the position of the label with respect to the length of the line. Default value is 0.5.
    label_config
        A dictionary containing the configuration for the label.
        This is only applied if ``label`` is of type ``str``.
    box_config
        A dictionary containing the configuration for the background box.
    frame_config
         A dictionary containing the configuration for the frame.

        .. seealso::
            :class:`LabeledLine`

    Examples
    --------
    .. manim:: LabeledArrowExample
        :save_last_frame:
        :quality: high

        class LabeledArrowExample(Scene):
            def construct(self):
                l_arrow = LabeledArrow("0.5", start=LEFT*3, end=RIGHT*3 + UP*2, label_position=0.5)

                self.add(l_arrow)
    """

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)


class LabeledPolygram(Polygram):
    """Constructs a polygram containing a label box at its pole of inaccessibility.

    Parameters
    ----------
    vertex_groups
        Vertices passed to the :class:`~.Polygram` constructor.
    label
        Label that will be displayed on the Polygram.
    precision
        The precision used by the PolyLabel algorithm.
    label_config
        A dictionary containing the configuration for the label.
        This is only applied if ``label`` is of type ``str``.
    box_config
        A dictionary containing the configuration for the background box.
    frame_config
         A dictionary containing the configuration for the frame.

        .. note::
            The PolyLabel Algorithm expects each vertex group to form a closed ring.
            If the input is open, :class:`LabeledPolygram` will attempt to close it.
            This may cause the polygon to intersect itself leading to unexpected results.

        .. tip::
            Make sure the precision corresponds to the scale of your inputs!
            For instance, if the bounding box of your polygon stretches from 0 to 10,000, a precision of 1.0 or 10.0 should be sufficient.

    Examples
    --------
    .. manim:: LabeledPolygramExample
        :save_last_frame:
        :quality: high

        class LabeledPolygramExample(Scene):
            def construct(self):
                # Define Rings
                ring1 = [
                    [-3.8, -2.4, 0], [-2.4, -2.5, 0], [-1.3, -1.6, 0], [-0.2, -1.7, 0],
                    [1.7, -2.5, 0], [2.9, -2.6, 0], [3.5, -1.5, 0], [4.9, -1.4, 0],
                    [4.5, 0.2, 0], [4.7, 1.6, 0], [3.5, 2.4, 0], [1.1, 2.5, 0],
                    [-0.1, 0.9, 0], [-1.2, 0.5, 0], [-1.6, 0.7, 0], [-1.4, 1.9, 0],
                    [-2.6, 2.6, 0], [-4.4, 1.2, 0], [-4.9, -0.8, 0], [-3.8, -2.4, 0]
                ]
                ring2 = [
                    [0.2, -1.2, 0], [0.9, -1.2, 0], [1.4, -2.0, 0], [2.1, -1.6, 0],
                    [2.2, -0.5, 0], [1.4, 0.0, 0], [0.4, -0.2, 0], [0.2, -1.2, 0]
                ]
                ring3 = [[-2.7, 1.4, 0], [-2.3, 1.7, 0], [-2.8, 1.9, 0], [-2.7, 1.4, 0]]

                # Create Polygons (for reference)
                p1 = Polygon(*ring1, fill_opacity=0.75)
                p2 = Polygon(*ring2, fill_color=BLACK, fill_opacity=1)
                p3 = Polygon(*ring3, fill_color=BLACK, fill_opacity=1)

                # Create Labeled Polygram
                polygram = LabeledPolygram(
                    *[ring1, ring2, ring3],
                    label=Text('Pole', font='sans-serif'),
                    precision=0.01,
                )

                # Display Circle (for reference)
                circle = Circle(radius=polygram.radius, color=WHITE).move_to(polygram.pole)

                self.add(p1, p2, p3)
                self.add(polygram)
                self.add(circle)

    .. manim:: LabeledCountryExample
        :save_last_frame:
        :quality: high

        import requests
        import json

        class LabeledCountryExample(Scene):
            def construct(self):
                # Fetch JSON data and process arcs
                data = requests.get('https://cdn.jsdelivr.net/npm/us-atlas@3/nation-10m.json').json()
                arcs, transform = data['arcs'], data['transform']
                sarcs = [np.cumsum(arc, axis=0) * transform['scale'] + transform['translate'] for arc in arcs]
                ssarcs = sorted(sarcs, key=len, reverse=True)[:1]

                # Compute Bounding Box
                points = np.concatenate(ssarcs)
                mins, maxs = np.min(points, axis=0), np.max(points, axis=0)

                # Build Axes
                ax = Axes(
                    x_range=[mins[0], maxs[0], maxs[0] - mins[0]], x_length=10,
                    y_range=[mins[1], maxs[1], maxs[1] - mins[1]], y_length=7,
                    tips=False
                )

                # Adjust Coordinates
                array = [[ax.c2p(*point) for point in sarc] for sarc in ssarcs]

                # Add Polygram
                polygram = LabeledPolygram(
                    *array,
                    label=Text('USA', font='sans-serif'),
                    precision=0.01,
                    fill_color=BLUE,
                    stroke_width=0,
                    fill_opacity=0.75
                )

                # Display Circle (for reference)
                circle = Circle(radius=polygram.radius, color=WHITE).move_to(polygram.pole)

                self.add(ax)
                self.add(polygram)
                self.add(circle)
    """

    def __init__(
        self,
        *vertex_groups: Point3DLike_Array,
        label: str | Tex | MathTex | Text,
        precision: float = 0.01,
        label_config: dict[str, Any] | None = None,
        box_config: dict[str, Any] | None = None,
        frame_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        # Initialize the Polygram with the vertex groups
        super().__init__(*vertex_groups, **kwargs)

        # Create Label
        self.label = Label(
            label=label,
            label_config=label_config,
            box_config=box_config,
            frame_config=frame_config,
        )

        # Close Vertex Groups
        rings = [
            group if np.array_equal(group[0], group[-1]) else list(group) + [group[0]]
            for group in vertex_groups
        ]

        # Compute the Pole of Inaccessibility
        cell = polylabel(rings, precision=precision)
        self.pole, self.radius = np.pad(cell.c, (0, 1), "constant"), cell.d

        # Position the label at the pole
        self.label.move_to(self.pole)
        self.add(self.label)
