"""Mobjects representing text rendered using Typst.

.. _typst-mobjects:

.. important::

   The ``typst`` Python package must be installed to use these classes.
   Install it via ``pip install typst>=0.14`` or add the ``typst`` optional
   dependency group (``pip install manim[typst]``).

Typst mobjects compile Typst markup directly to SVG using the ``typst``
Python package and then import the result through :class:`~.SVGMobject`.
Use :class:`~.Typst` for general Typst markup and :class:`~.MathTypst`
for display-style math.

Examples
--------

Basic text and math
^^^^^^^^^^^^^^^^^^^

.. manim:: TypstTextReferenceExample
    :save_last_frame:
    :ref_classes: Typst

    class TypstTextReferenceExample(Scene):
        def construct(self):
            text = Typst(
                r"*Hello* from _Typst!_",
                color=YELLOW,
                font_size=72,
            )
            self.add(text)

.. manim:: MathTypstReferenceExample
    :save_last_frame:
    :ref_classes: MathTypst

    class MathTypstReferenceExample(Scene):
        def construct(self):
            equation = MathTypst(
                r"sum_(k=1)^n k = frac(n(n + 1), 2)",
                font_size=72,
            )
            self.add(equation)

Selecting subexpressions
^^^^^^^^^^^^^^^^^^^^^^^^

Typst mobjects expose label-based selection via :meth:`~.Typst.select`.
There are two common ways to create selectable groups:

- use ordinary Typst labels in :class:`~.Typst`
- use Manim's ``{{ ... }}`` shorthand in :class:`~.MathTypst`

.. note::

   The ``{{ ... }}`` shorthand is currently only supported by
   :class:`~.MathTypst`. For :class:`~.Typst`, create labels directly in the
   Typst source, for example with ``#box[body] <label>``.

.. manim:: TypstLabelSelectionExample
    :save_last_frame:
    :ref_classes: Typst
    :ref_methods: Typst.select

    class TypstLabelSelectionExample(Scene):
        def construct(self):
            text = Typst(
                r'''
                #box[
                    *Typst* labels also work in regular markup.
                ] <headline>

                #let pick(body) = [#box(body) <picked>]
                We can highlight #pick[multiple] #pick[fragments] at once.
                ''',
                font_size=42,
            )
            text.select("headline").set_color(BLUE)
            text.select("picked").set_color(YELLOW)
            self.add(text)

.. manim:: MathTypstSelectionExample
    :save_last_frame:
    :ref_classes: MathTypst
    :ref_methods: Typst.select

    class MathTypstSelectionExample(Scene):
        def construct(self):
            equation = MathTypst(
                "{{ a^2 + b^2 : lhs }} = {{ c^2 }}",
                font_size=72,
            )
            equation.select("lhs").set_color(BLUE)
            equation.select(0).set_color(YELLOW)
            self.add(equation)

Inspecting baseline frames
^^^^^^^^^^^^^^^^^^^^^^^^^^

For debugging or alignment tasks, Typst mobjects can optionally track a
per-element baseline frame. Enable this with ``track_baselines=True`` and
query either :attr:`~.Typst.baseline_frames` for all tracked leaf elements or
:meth:`~.Typst.get_baseline_frame` for a specific selected submobject.

.. code-block:: python

    text = Typst("Ggf", track_baselines=True)
    orig, right, up = text.baseline_frames[0]

    eq = MathTypst("{{ a^2 + b^2 : lhs }} = c^2", track_baselines=True)
    for part in eq.select("lhs"):
        orig, right, up = eq.get_baseline_frame(part)
        print(orig, right, up)
"""

from __future__ import annotations

__all__ = [
    "Typst",
    "MathTypst",
]

import re
from pathlib import Path
from typing import Any, Self, cast
from xml.etree import ElementTree as ET

import numpy as np
import svgelements as se

from manim import config
from manim.constants import DEFAULT_FONT_SIZE, SCALE_FACTOR_PER_FONT_POINT, RendererType
from manim.mobject.svg.svg_mobject import SVGMobject
from manim.mobject.types.vectorized_mobject import VGroup, VMobject
from manim.utils.color import BLACK, ParsableManimColor
from manim.utils.typst_file_writing import typst_to_svg_file

_MANIMGRP_TARGET = "__manim_typst_capture_target"

# Pattern for the label part of {{ content : label }}.
# The label must be a valid Typst label identifier.
_LABEL_RE = re.compile(r"^(.*)\s*:\s*([a-zA-Z_][a-zA-Z0-9_-]*)\s*$", re.DOTALL)
_INTERNAL_TYPST_ID_RE = re.compile(r"g[0-9A-Fa-f]+")
_DUPLICATE_LABEL_SUFFIX = "__manim_typst_dup_"
# Empirical correction so Typst-authored SVG strokes (fraction bars,
# underlines, etc.) visually match the weight of TeX-derived geometry more
# closely after import into Manim's pixel-based stroke model.
_TYPST_SVG_STROKE_WIDTH_SCALE = 0.5
_SVG_LEAF_TAGS = {
    "circle",
    "ellipse",
    "image",
    "line",
    "path",
    "polygon",
    "polyline",
    "rect",
    "text",
    "use",
}


def _manimgrp_preamble(target: str | None) -> str:
    """Return a layout-transparent Typst capture helper.

    The final render uses ``none`` as the target and therefore returns every
    group body unchanged. Probe renders target one label at a time and use
    Typst's layout-preserving ``hide`` element to remove just that group's SVG
    leaves.
    """
    target_value = "none" if target is None else f'"{target}"'
    return (
        f"#let {_MANIMGRP_TARGET} = {target_value}\n"
        "#let manimgrp(lbl, body) = if "
        f"lbl == {_MANIMGRP_TARGET} {{ hide(body) }} else {{ body }}"
    )


def _svg_tag_name(element: ET.Element) -> str:
    """Return an XML element's local tag name."""
    if not isinstance(element.tag, str):
        return ""
    return element.tag.rsplit("}", 1)[-1]


def _svg_leaf_signatures(svg_file: Path) -> list[tuple[Any, ...]]:
    """Return stable signatures for rendered SVG leaves in drawing order."""
    root = ET.parse(svg_file).getroot()
    signatures: list[tuple[Any, ...]] = []

    def visit(
        element: ET.Element,
        transform: se.Matrix,
        inside_defs: bool,
    ) -> None:
        tag = _svg_tag_name(element)
        inside_defs = inside_defs or tag == "defs"
        local_transform = (
            se.Matrix(element.get("transform"))
            if element.get("transform") is not None
            else se.Matrix()
        )
        effective_transform = transform * local_transform

        if not inside_defs and tag in _SVG_LEAF_TAGS:
            attributes = tuple(
                sorted(
                    (key, value)
                    for key, value in element.attrib.items()
                    if key != "transform"
                )
            )
            matrix = tuple(
                round(value, 12)
                for value in (
                    effective_transform.a,
                    effective_transform.b,
                    effective_transform.c,
                    effective_transform.d,
                    effective_transform.e,
                    effective_transform.f,
                )
            )
            signatures.append((tag, attributes, matrix))

        for child in element:
            visit(child, effective_transform, inside_defs)

    visit(root, se.Matrix(), False)
    return signatures


def _hidden_leaf_indices(
    visible: list[tuple[Any, ...]],
    probe: list[tuple[Any, ...]],
) -> set[int]:
    """Return visible leaf indices removed from a layout-preserving probe.

    A valid ``hide`` probe is the final SVG leaf sequence with zero or more
    entries deleted. Longest-common-subsequence matching handles repeated
    glyphs while retaining their drawing order and effective positions.
    """
    visible_count = len(visible)
    probe_count = len(probe)
    matches = [[0] * (probe_count + 1) for _ in range(visible_count + 1)]

    for visible_index in range(visible_count - 1, -1, -1):
        for probe_index in range(probe_count - 1, -1, -1):
            if visible[visible_index] == probe[probe_index]:
                matches[visible_index][probe_index] = (
                    1 + matches[visible_index + 1][probe_index + 1]
                )
            else:
                matches[visible_index][probe_index] = max(
                    matches[visible_index + 1][probe_index],
                    matches[visible_index][probe_index + 1],
                )

    if matches[0][0] != probe_count:
        raise ValueError(
            "The MathTypst grouping probe changed visible SVG geometry instead of "
            "only hiding captured leaves. A custom Typst show rule for `hide` may "
            "be interfering with subexpression selection."
        )

    retained: set[int] = set()
    visible_index = 0
    probe_index = 0
    while visible_index < visible_count and probe_index < probe_count:
        if visible[visible_index] == probe[probe_index]:
            retained.add(visible_index)
            visible_index += 1
            probe_index += 1
        elif (
            matches[visible_index + 1][probe_index]
            >= matches[visible_index][probe_index + 1]
        ):
            visible_index += 1
        else:
            probe_index += 1

    return set(range(visible_count)) - retained


class Typst(SVGMobject):
    """A mobject rendered from a Typst markup string.

    The Typst source is compiled to SVG via the ``typst`` Python package
    (a self-contained Rust binary extension — no system-level install
    required) and then imported through :class:`~.SVGMobject`.

    Parameters
    ----------
    typst_code
        Raw Typst markup to be compiled. This string is placed verbatim
        into the body of a minimal Typst document.
    font_size
        Font size in Manim font-size units (default: ``DEFAULT_FONT_SIZE``,
        i.e. 48). The actual scaling is applied *after* SVG import, matching
        the approach used by :class:`~.SingleStringMathTex`.
    typst_preamble
        Extra Typst code inserted before the body. Useful for ``#import``,
        ``#set``, or ``#show`` rules. Default: ``""``.
    color
        The color of the mobject. By default the standard VMobject color
        (white in dark mode). Overrides the Typst text fill color.
    stroke_width
        SVG stroke width override. If ``None`` (default), the stroke widths
        from Typst's SVG output are preserved.
    font_paths
        Optional list of additional font directories passed to the Typst
        compiler (e.g. for custom fonts not installed system-wide).
    track_baselines
        Whether to keep enough per-element reference data to recover the
        current Typst baseline frame for each imported submobject.
        When enabled, :attr:`baseline_frames` and
        :meth:`get_baseline_frame` can be used to retrieve the current
        ``(orig, right, up)`` positions for the imported SVG elements.
    should_center
        Whether to center the mobject after import (default ``True``).
    height
        Target height of the mobject. If ``None`` (default), the height is
        determined by ``font_size``.
    **kwargs
        Forwarded to :class:`~.SVGMobject`.

    Examples
    --------
    .. manim:: TypstExample
        :ref_classes: Typst

        class TypstExample(Scene):
            def construct(self):
                formula = Typst(r"$ integral_a^b f(x) dif x $")
                self.play(Write(formula))

    .. manim:: TypstTextExample
        :save_last_frame:
        :ref_classes: Typst

        class TypstTextExample(Scene):
            def construct(self):
                text = Typst(
                    r"*Hello* from _Typst!_",
                    font_size=72,
                )
                self.add(text)
    """

    def __init__(
        self,
        typst_code: str,
        *,
        font_size: float = DEFAULT_FONT_SIZE,
        typst_preamble: str = "",
        color: ParsableManimColor | None = None,
        stroke_width: float | None = None,
        font_paths: list[str | Path] | None = None,
        track_baselines: bool = False,
        should_center: bool = True,
        height: float | None = None,
        **kwargs: Any,
    ):
        if color is None:
            color = VMobject().color

        self._font_size = font_size
        self.typst_code = typst_code
        self.typst_preamble = typst_preamble
        self.track_baselines = track_baselines
        self._preserve_svg_stroke_widths = stroke_width is None
        self._baseline_tracked_submobjects: list[VMobject] = []
        self._stroke_width_tracked_submobjects: list[VMobject] = []
        self._label_aliases: dict[str, list[str]] = {}
        self._svg_leaf_labels: dict[int, list[str]] = getattr(
            self,
            "_svg_leaf_labels",
            {},
        )
        self._known_svg_labels: list[str] = getattr(
            self,
            "_known_svg_labels",
            [],
        )

        file_name = typst_to_svg_file(
            typst_code,
            preamble=typst_preamble,
            font_paths=font_paths,
        )
        super().__init__(
            file_name=file_name,
            should_center=should_center,
            stroke_width=stroke_width,
            height=height,
            color=color,
            path_string_config={
                "should_subdivide_sharp_curves": True,
                "should_remove_null_curves": True,
            },
            **kwargs,
        )
        self._rebuild_label_aliases()
        self._refresh_svg_stroke_widths()
        self.init_colors()

        # Used for scaling via font_size property (mirrors SingleStringMathTex).
        self.initial_height = self.height

        if height is None:
            self.font_size = self._font_size

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.typst_code!r})"

    @property
    def hash_seed(self) -> tuple:
        """Include baseline tracking in the SVG cache key."""
        return (*super().hash_seed, self.track_baselines)

    # -- font_size property (same approach as SingleStringMathTex) -----------

    @property
    def font_size(self) -> float:
        """The font size of the Typst mobject."""
        return self.height / self.initial_height / SCALE_FACTOR_PER_FONT_POINT

    @font_size.setter
    def font_size(self, val: float) -> None:
        if val <= 0:
            raise ValueError("font_size must be greater than 0.")
        if self.height > 0:
            self.scale(val / self.font_size)

    def scale(
        self,
        scale_factor: float,
        scale_stroke: bool = False,
        *,
        about_point: np.ndarray | None = None,
        about_edge: np.ndarray | None = None,
    ) -> Self:
        result = super().scale(
            scale_factor,
            scale_stroke=scale_stroke,
            about_point=about_point,
            about_edge=about_edge,
        )
        self._refresh_svg_stroke_widths()
        return result

    def _refresh_svg_stroke_widths(self) -> None:
        """Refresh pixel stroke widths for Typst-authored SVG strokes.

        SVG stroke widths are specified in the SVG's local coordinate system,
        while Manim stroke widths are pixel-based. For Typst-authored strokes
        such as fraction bars or underlines, rescale them according to the
        current geometric scale of the imported element so their visual weight
        stays proportional to the rest of the expression.
        """
        if not self._preserve_svg_stroke_widths:
            return

        pixels_per_unit = config.pixel_width / config.frame_width
        for submobject in self._stroke_width_tracked_submobjects:
            submobject_any = cast(Any, submobject)
            reference_size = cast(float, submobject_any._typst_reference_size)
            source_stroke_width = cast(
                float,
                submobject_any._typst_source_stroke_width,
            )
            current_size = max(submobject.width, submobject.height)
            if reference_size <= 0:
                continue
            current_stroke_width = source_stroke_width * current_size / reference_size
            submobject.set_stroke(
                width=current_stroke_width
                * pixels_per_unit
                * _TYPST_SVG_STROKE_WIDTH_SCALE,
                family=False,
            )

    # -- baseline frame tracking ---------------------------------------------

    def get_mob_from_shape_element(self, shape: se.SVGElement) -> VMobject | None:
        """Attach Typst-specific metadata to imported shape mobjects."""
        mob = super().get_mob_from_shape_element(shape)
        if mob is None or not mob.has_points():
            return mob

        if self._preserve_svg_stroke_widths and shape.stroke_width not in (None, 0):
            reference_size = max(mob.width, mob.height)
            if reference_size > 0:
                mob_any = cast(Any, mob)
                mob_any._typst_reference_size = reference_size
                mob_any._typst_source_stroke_width = shape.stroke_width
                self._stroke_width_tracked_submobjects.append(mob)

        if not self.track_baselines:
            return mob

        baseline_marks = self._get_reference_baseline_frame(shape)
        if baseline_marks is None:
            return mob

        reference_points = mob.points.copy()
        reference_xy = np.column_stack(
            [
                reference_points[:, 0],
                reference_points[:, 1],
                np.ones(len(reference_points)),
            ],
        )
        if np.linalg.matrix_rank(reference_xy) < 3:
            return mob

        mob_any = cast(Any, mob)
        mob_any._typst_reference_points = reference_points
        mob_any._typst_reference_baseline_frame = baseline_marks
        self._baseline_tracked_submobjects.append(mob)
        return mob

    @staticmethod
    def _get_reference_baseline_frame(
        shape: se.SVGElement,
    ) -> np.ndarray | None:
        """Return the reference ``(orig, right, up)`` frame for a Typst SVG element.

        The frame is expressed in the same pre-centering coordinate system as the
        imported submobject points after the element's own SVG transform has been
        applied.
        """
        if not isinstance(shape, se.Transformable):
            return None

        matrix = shape.transform if shape.apply else se.Matrix()
        return np.array(
            [
                [matrix.e, matrix.f, 0.0],
                [matrix.a + matrix.e, matrix.b + matrix.f, 0.0],
                [matrix.c + matrix.e, matrix.d + matrix.f, 0.0],
            ],
        )

    def get_baseline_frame(
        self, submobject: VMobject
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the current Typst baseline frame for a tracked submobject.

        The returned tuple contains the current positions of ``(orig, right, up)``.
        These are recovered from the stored reference frame and the submobject's
        current affine position in the scene.
        """
        try:
            submobject_any = cast(Any, submobject)
            reference_points = cast(
                np.ndarray,
                submobject_any._typst_reference_points,
            )
            reference_frame = cast(
                np.ndarray,
                submobject_any._typst_reference_baseline_frame,
            )
        except AttributeError as err:
            raise ValueError(
                "No tracked Typst baseline frame is available for this submobject. "
                "Construct the Typst mobject with track_baselines=True.",
            ) from err

        reference_xy = np.column_stack(
            [
                reference_points[:, 0],
                reference_points[:, 1],
                np.ones(len(reference_points)),
            ],
        )
        if np.linalg.matrix_rank(reference_xy) < 3:
            raise ValueError(
                "The stored Typst reference geometry is degenerate, so its baseline "
                "frame cannot be recovered.",
            )

        transform, _, _, _ = np.linalg.lstsq(
            reference_xy, submobject.points, rcond=None
        )
        frame_xy = np.column_stack(
            [
                reference_frame[:, 0],
                reference_frame[:, 1],
                np.ones(len(reference_frame)),
            ],
        )
        current_frame = frame_xy @ transform
        return tuple(
            cast(tuple[np.ndarray, np.ndarray, np.ndarray], tuple(current_frame))
        )

    @property
    def baseline_frames(self) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Current Typst baseline frames for all tracked leaf submobjects."""
        if not self.track_baselines:
            return []
        return [
            self.get_baseline_frame(submobject)
            for submobject in self._baseline_tracked_submobjects
        ]

    def _rebuild_label_aliases(self) -> None:
        """Rebuild user-facing label aliases from imported SVG ids."""
        aliases: dict[str, list[str]] = {}
        for key in self.id_to_vgroup_dict:
            if key == "root" or key.startswith("numbered_group_"):
                continue
            if _INTERNAL_TYPST_ID_RE.fullmatch(key) is not None:
                continue

            base_label = key
            if _DUPLICATE_LABEL_SUFFIX in key:
                base_label, _, _ = key.partition(_DUPLICATE_LABEL_SUFFIX)
            aliases.setdefault(base_label, []).append(key)
        for label in self._known_svg_labels:
            aliases.setdefault(label, [])
        self._label_aliases = aliases

    def _select_label(self, label: str) -> VGroup:
        if label not in self._label_aliases:
            raise KeyError(
                f"No group with label {label!r} found. "
                f"Available labels: {self._user_label_keys()}"
            )

        result = VGroup()
        seen_ids: set[int] = set()
        for group_id in self._label_aliases[label]:
            for submobject in self.id_to_vgroup_dict[group_id]:
                submobject_id = id(submobject)
                if submobject_id in seen_ids:
                    continue
                seen_ids.add(submobject_id)
                result.add(submobject)
        return result

    # -- SVG post-processing -------------------------------------------------

    def modify_xml_tree(self, element_tree: ET.ElementTree) -> ET.ElementTree:
        """Add Manim group IDs to Typst SVG elements before parsing.

        Ordinary Typst labels are promoted from ``data-typst-label`` to ``id``.
        :class:`MathTypst` capture groups additionally provide a leaf-index to
        label mapping obtained from layout-preserving ``hide`` probes. Each
        captured leaf is wrapped in one or more identified SVG groups, allowing
        nested captures without altering the final Typst layout.
        """
        label_counts: dict[str, int] = {}

        if self._svg_leaf_labels:
            leaf_index = 0

            def wrap_leaves(element: ET.Element, inside_defs: bool = False) -> None:
                nonlocal leaf_index
                children: list[ET.Element] = []
                for child in element:
                    tag = _svg_tag_name(child)
                    child_inside_defs = inside_defs or tag == "defs"
                    if not child_inside_defs and tag in _SVG_LEAF_TAGS:
                        wrapped = child
                        for label in reversed(
                            self._svg_leaf_labels.get(leaf_index, [])
                        ):
                            count = label_counts.get(label, 0)
                            label_counts[label] = count + 1
                            svg_id = label
                            if count > 0:
                                svg_id = f"{label}{_DUPLICATE_LABEL_SUFFIX}{count}"
                            namespace = (
                                wrapped.tag.partition("}")[0] + "}"
                                if "}" in wrapped.tag
                                else ""
                            )
                            group = ET.Element(
                                f"{namespace}g",
                                {"id": svg_id},
                            )
                            group.append(wrapped)
                            wrapped = group
                        children.append(wrapped)
                        leaf_index += 1
                    else:
                        wrap_leaves(child, child_inside_defs)
                        children.append(child)
                element[:] = children

            wrap_leaves(element_tree.getroot())

        # Let the base class inject default style wrappers after capture leaves
        # have been grouped in the original SVG namespace.
        element_tree = super().modify_xml_tree(element_tree)

        # ElementTree qualifies SVG tags with a namespace URI, so walk all
        # elements rather than matching a bare ``g`` tag.
        for element in element_tree.iter():
            label = element.get("data-typst-label")
            if label is None:
                continue
            count = label_counts.get(label, 0)
            label_counts[label] = count + 1
            svg_id = label
            if count > 0:
                svg_id = f"{label}{_DUPLICATE_LABEL_SUFFIX}{count}"
            element.set("id", svg_id)
            del element.attrib["data-typst-label"]

        return element_tree

    # -- sub-expression selection --------------------------------------------

    def select(self, key: str | int) -> VGroup:
        """Select a labeled sub-expression.

        Labels are created either directly in Typst markup or through the
        ``{{ }}`` double-brace notation in :class:`MathTypst`.

        Parameters
        ----------
        key
            A label name (``str``) available in the rendered SVG, or an
            integer index into the auto-numbered ``{{ }}`` groups
            (``_grp-0``, ``_grp-1``, …).

        Returns
        -------
        VGroup
            The submobjects corresponding to the selected group.

        Raises
        ------
        KeyError
            If no group with the given label exists.
        IndexError
            If an integer index is out of range.

        Examples
        --------
        .. manim:: TypstSelectExample
            :save_last_frame:
            :ref_classes: MathTypst
            :ref_methods: Typst.select

            class TypstSelectExample(Scene):
                def construct(self):
                    eq = MathTypst(
                        "{{ a + b : num }} / {{ c : den }} = {{ lambda }} {{ x }}"
                    )
                    eq.select("num").set_color(RED)  # "a + b"
                    eq.select("den").set_color(BLUE) # "c"
                    eq.select(0).set_color(YELLOW)   # "lambda" (auto-numbered: "grp-0")
                    eq.select(1).set_color(GREEN)    # "x" (auto-numbered: "grp-1")

                    self.add(eq)
        """
        if isinstance(key, int):
            label = f"_grp-{key}"
            if label not in self._label_aliases:
                raise IndexError(
                    f"Group index {key} out of range. "
                    f"Available labels: {self._user_label_keys()}"
                )
            return self._select_label(label)

        return self._select_label(key)

    def _user_label_keys(self) -> list[str]:
        """Return user-facing label keys, excluding internal SVG group IDs."""
        return list(self._label_aliases)

    # -- color handling ------------------------------------------------------

    def init_colors(self, propagate_colors: bool = True) -> Self:
        """Recolor black submobjects to ``self.color``.

        Typst renders text in black (``fill="#000000"``) by default.
        This mirrors the approach of :meth:`SingleStringMathTex.init_colors`:
        any submobject whose color is black is recolored to ``self.color``,
        while explicitly colored submobjects (non-black) are preserved.
        """
        for submobject in self.submobjects:
            if submobject.color != BLACK:
                continue
            submobject.color = self.color
            if config.renderer == RendererType.OPENGL:
                submobject.init_colors()
            elif config.renderer == RendererType.CAIRO:
                submobject.init_colors(propagate_colors=propagate_colors)
        return self


class MathTypst(Typst):
    r"""Convenience wrapper: wraps the input in Typst math delimiters.

    The expression is rendered as a display-level equation
    (``$ ... $`` with surrounding spaces).

    Supports the ``{{ ... }}`` double-brace notation for grouping
    sub-expressions. The final expression is rendered without a layout
    wrapper; additional layout-preserving ``hide`` probes map each capture
    to SVG leaves accessible via :meth:`~.Typst.select`.

    Groups can optionally be given explicit labels:
    ``{{ content : label }}``. Without a label, groups are
    auto-numbered (``_grp-0``, ``_grp-1``, …). Groups may be nested; an inner
    group's SVG leaves are then selectable through both labels.

    .. note::

       Custom Typst show rules targeting ``hide`` are not supported together
       with ``{{ ... }}`` capture groups.

    Parameters
    ----------
    math_expression
        Typst math-mode content **without** the ``$ ... $`` delimiters.
        May contain ``{{ ... }}`` groups.
    **kwargs
        Forwarded to :class:`Typst`.

    Examples
    --------
    .. manim:: DisplayMath
        :save_last_frame:
        :ref_classes: MathTypst

        class DisplayMath(Scene):
            def construct(self):
                eq = MathTypst(r"sum_(k=0)^n k = (n(n+1)) / 2")
                self.add(eq)

    .. manim:: GroupedMath
        :save_last_frame:
        :ref_classes: MathTypst
        :ref_methods: Typst.select

        class GroupedMath(Scene):
            def construct(self):
                eq = MathTypst("{{ a^2 + b^2 : lhs }} = {{ c^2 }}")
                eq.select("lhs").set_color(RED) # "a^2 + b^2"
                eq.select(0).set_color(BLUE)    # "c^2" (auto-numbered: "grp-0")
                self.add(eq)
    """

    def __init__(self, math_expression: str, **kwargs: Any):
        processed, labels = self._preprocess_groups(math_expression)
        self._group_labels = labels
        typst_code = f"$ {processed} $"

        if labels:
            user_preamble = kwargs.get("typst_preamble", "")
            font_paths = kwargs.get("font_paths")
            final_preamble = _manimgrp_preamble(None)
            if user_preamble:
                final_preamble = f"{final_preamble}\n{user_preamble}"

            final_svg = typst_to_svg_file(
                typst_code,
                preamble=final_preamble,
                font_paths=font_paths,
            )
            visible_leaves = _svg_leaf_signatures(final_svg)
            distinct_labels = list(dict.fromkeys(labels))
            leaf_labels: dict[int, list[str]] = {}

            for label in distinct_labels:
                probe_preamble = _manimgrp_preamble(label)
                if user_preamble:
                    probe_preamble = f"{probe_preamble}\n{user_preamble}"
                probe_svg = typst_to_svg_file(
                    typst_code,
                    preamble=probe_preamble,
                    font_paths=font_paths,
                )
                try:
                    hidden_indices = _hidden_leaf_indices(
                        visible_leaves,
                        _svg_leaf_signatures(probe_svg),
                    )
                except ValueError as error:
                    raise ValueError(
                        f"Could not map MathTypst group {label!r} to SVG leaves. "
                        f"{error}"
                    ) from error
                for leaf_index in sorted(hidden_indices):
                    leaf_labels.setdefault(leaf_index, []).append(label)

            self._svg_leaf_labels = leaf_labels
            self._known_svg_labels = distinct_labels
            kwargs["typst_preamble"] = final_preamble

        super().__init__(typst_code, **kwargs)

    # -- double-brace preprocessor -------------------------------------------

    @staticmethod
    def _preprocess_groups(math_expr: str) -> tuple[str, list[str]]:
        """Replace ``{{ ... }}`` groups with ``manimgrp(...)`` calls.

        Parameters
        ----------
        math_expr
            The raw math expression (without ``$ ... $`` delimiters).

        Returns
        -------
        tuple[str, list[str]]
            The processed expression and an ordered list of group labels.
        """
        labels: list[str] = []
        auto_index = 0

        def process(expression: str) -> str:
            nonlocal auto_index
            result: list[str] = []
            i = 0
            n = len(expression)
            in_string = False
            bracket_depth = 0

            while i < n:
                ch = expression[i]

                if in_string:
                    result.append(ch)
                    if ch == "\\" and i + 1 < n:
                        result.append(expression[i + 1])
                        i += 2
                        continue
                    if ch == '"':
                        in_string = False
                    i += 1
                    continue
                if ch == '"':
                    in_string = True
                    result.append(ch)
                    i += 1
                    continue

                if ch == "[":
                    bracket_depth += 1
                    result.append(ch)
                    i += 1
                    continue
                if ch == "]" and bracket_depth > 0:
                    bracket_depth -= 1
                    result.append(ch)
                    i += 1
                    continue
                if bracket_depth > 0:
                    result.append(ch)
                    i += 1
                    continue

                if i + 1 >= n or ch != "{" or expression[i + 1] != "{":
                    result.append(ch)
                    i += 1
                    continue

                group_start = i
                i += 2
                content_start = i
                depth = 1
                group_in_string = False
                group_bracket_depth = 0

                while i < n and depth > 0:
                    ch = expression[i]
                    if group_in_string:
                        if ch == "\\" and i + 1 < n:
                            i += 2
                            continue
                        if ch == '"':
                            group_in_string = False
                        i += 1
                        continue
                    if ch == '"':
                        group_in_string = True
                        i += 1
                        continue
                    if ch == "[":
                        group_bracket_depth += 1
                        i += 1
                        continue
                    if ch == "]" and group_bracket_depth > 0:
                        group_bracket_depth -= 1
                        i += 1
                        continue
                    if group_bracket_depth > 0:
                        i += 1
                        continue
                    if ch == "{" and i + 1 < n and expression[i + 1] == "{":
                        depth += 1
                        i += 2
                        continue
                    if ch == "}" and i + 1 < n and expression[i + 1] == "}":
                        depth -= 1
                        if depth == 0:
                            content = expression[content_start:i]
                            i += 2
                            break
                        i += 2
                        continue
                    i += 1
                else:
                    result.append(expression[group_start:])
                    return "".join(result)

                match = _LABEL_RE.match(content)
                if match is not None:
                    body = match.group(1).strip()
                    label = match.group(2)
                else:
                    body = content.strip()
                    label = f"_grp-{auto_index}"
                    auto_index += 1

                # Record the outer capture before recursively processing its
                # body, preserving source opening order for nested groups.
                labels.append(label)
                result.append(f'manimgrp("{label}", {process(body)})')

            return "".join(result)

        return process(math_expr), labels
