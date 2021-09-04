"""Utility functions for parsing SVG styles."""


__all__ = ["cascade_element_style", "parse_style", "parse_color_string"]

from typing import Dict, List
from xml.dom.minidom import Element as MinidomElement

from colour import web2hex

from ...utils.color import rgb_to_hex

CASCADING_STYLING_ATTRIBUTES: List[str] = [
    "fill",
    "stroke",
    "fill-opacity",
    "stroke-opacity",
]


# The default styling specifications for SVG images,
# according to https://www.w3.org/TR/SVG/painting.html
# (ctrl-F for "initial")
SVG_DEFAULT_ATTRIBUTES: Dict[str, str] = {
    "fill": "black",
    "fill-opacity": "1",
    "stroke": "none",
    "stroke-opacity": "1",
}


def cascade_element_style(
    element: MinidomElement,
    inherited: Dict[str, str],
) -> Dict[str, str]:
    """Collect the element's style attributes based upon both its inheritance and its own attributes.

    SVG uses cascading element styles. A closer ancestor's style takes precedence over a more distant ancestor's
    style. In order to correctly calculate the styles, the attributes are passed down through the inheritance tree,
    updating where necessary.

    Note that this method only copies the values and does not parse them. See :meth:`parse_color_string` for converting
    from SVG attributes to manim keyword arguments.

    Parameters
    ----------
    element : :class:`MinidomElement`
        Element of the SVG parse tree

    inherited : :class:`dict`
        Dictionary of SVG attributes inherited from the parent element.

    Returns
    -------
    :class:`dict`
        Dictionary mapping svg attributes to values with `element`'s values overriding inherited values.
    """

    style = inherited.copy()

    # cascade the regular elements.
    for attr in CASCADING_STYLING_ATTRIBUTES:
        entry = element.getAttribute(attr)
        if entry:
            style[attr] = entry

    # the style attribute should be handled separately in order to
    # break it up nicely. furthermore, style takes priority over other
    # attributes in the same element.
    style_specs = element.getAttribute("style")
    if style_specs:
        for style_spec in style_specs.split(";"):
            try:
                key, value = style_spec.split(":")
            except ValueError as e:
                if not style_spec.strip():
                    # there was just a stray semicolon at the end, producing an emptystring
                    pass
                else:
                    raise e
            else:
                style[key.strip()] = value.strip()

    return style


def parse_color_string(color_spec: str) -> str:
    """Handle the SVG-specific color strings and convert them to HTML #rrggbb format.

    Parameters
    ----------
    color_spec : :class:`str`
        String in any web-compatible format

    Returns
    -------
    :class:`str`
        Hexadecimal color string in the format `#rrggbb`
    """

    if color_spec[0:3] == "rgb":
        # these are only in integer form, but the Colour module wants them in floats.
        splits = color_spec[4:-1].split(",")
        if splits[0][-1] == "%":
            # if the last character of the first number is a percentage,
            # then interpret the number as a percentage
            parsed_rgbs = [float(i[:-1]) / 100.0 for i in splits]
        else:
            parsed_rgbs = [int(i) / 255.0 for i in splits]

        hex_color = rgb_to_hex(parsed_rgbs)

    elif color_spec[0] == "#":
        # its OK, parse as hex color standard.
        hex_color = color_spec

    else:
        # attempt to convert color names like "red" to hex color
        hex_color = web2hex(color_spec, force_long=True)

    return hex_color


def fill_default_values(svg_style: Dict) -> None:
    """
    Fill in the default values for properties of SVG elements,
    if they are not currently set in the style dictionary.

    Parameters
    ----------
    svg_style : :class:`dict`
        Style dictionary with SVG property names. Some may be missing.

    Returns
    -------
    :class:`dict`
        Style attributes; none are missing.
    """
    for key in SVG_DEFAULT_ATTRIBUTES:
        if key not in svg_style:
            svg_style[key] = SVG_DEFAULT_ATTRIBUTES[key]


def parse_style(svg_style: Dict[str, str]) -> Dict:
    """Convert a dictionary of SVG attributes to Manim VMobject keyword arguments.

    Parameters
    ----------
    svg_style : :class:`dict`
        Style attributes as a string-to-string dictionary. Keys are valid SVG element attributes (fill, stroke, etc)

    Returns
    -------
    :class:`dict`
        Style attributes, but in manim kwargs form, e.g., keys are fill_color, stroke_color
    """

    manim_style = {}
    fill_default_values(svg_style)

    if "fill-opacity" in svg_style:
        manim_style["fill_opacity"] = float(svg_style["fill-opacity"])

    if "stroke-opacity" in svg_style:
        manim_style["stroke_opacity"] = float(svg_style["stroke-opacity"])

    # nones need to be handled specially
    if "fill" in svg_style:
        if svg_style["fill"] == "none":
            manim_style["fill_opacity"] = 0
        else:
            manim_style["fill_color"] = parse_color_string(svg_style["fill"])

    if "stroke" in svg_style:
        if svg_style["stroke"] == "none":
            # In order to not break animations.creation.Write,
            # we interpret no stroke as stroke-width of zero and
            # color the same as the fill color, if it exists.
            manim_style["stroke_width"] = 0
            if "fill_color" in manim_style:
                manim_style["stroke_color"] = manim_style["fill_color"]
        else:
            manim_style["stroke_color"] = parse_color_string(svg_style["stroke"])

    return manim_style
