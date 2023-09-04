"""Colors included in the global name space.

These colors form Manim's default color space.

.. manim:: ColorsOverview
    :save_last_frame:
    :hide_source:

    import manim.utils.color.manim_colors as Colors

    class ColorsOverview(Scene):
        def construct(self):
            def color_group(color):
                group = VGroup(
                    *[
                        Line(ORIGIN, RIGHT * 1.5, stroke_width=35, color=getattr(Colors, name.upper()))
                        for name in subnames(color)
                    ]
                ).arrange_submobjects(buff=0.4, direction=DOWN)

                name = Text(color).scale(0.6).next_to(group, UP, buff=0.3)
                if any(decender in color for decender in "gjpqy"):
                    name.shift(DOWN * 0.08)
                group.add(name)
                return group

            def subnames(name):
                return [name + "_" + char for char in "abcde"]

            color_groups = VGroup(
                *[
                    color_group(color)
                    for color in [
                        "blue",
                        "teal",
                        "green",
                        "yellow",
                        "gold",
                        "red",
                        "maroon",
                        "purple",
                    ]
                ]
            ).arrange_submobjects(buff=0.2, aligned_edge=DOWN)

            for line, char in zip(color_groups[0], "abcde"):
                color_groups.add(Text(char).scale(0.6).next_to(line, LEFT, buff=0.2))

            def named_lines_group(length, colors, names, text_colors, align_to_block):
                lines = VGroup(
                    *[
                        Line(
                            ORIGIN,
                            RIGHT * length,
                            stroke_width=55,
                            color=getattr(Colors, color.upper()),
                        )
                        for color in colors
                    ]
                ).arrange_submobjects(buff=0.6, direction=DOWN)

                for line, name, color in zip(lines, names, text_colors):
                    line.add(Text(name, color=color).scale(0.6).move_to(line))
                lines.next_to(color_groups, DOWN, buff=0.5).align_to(
                    color_groups[align_to_block], LEFT
                )
                return lines

            other_colors = (
                "pink",
                "light_pink",
                "orange",
                "light_brown",
                "dark_brown",
                "gray_brown",
            )

            other_lines = named_lines_group(
                3.2,
                other_colors,
                other_colors,
                [BLACK] * 4 + [WHITE] * 2,
                0,
            )

            gray_lines = named_lines_group(
                6.6,
                ["white"] + subnames("gray") + ["black"],
                [
                    "white",
                    "lighter_gray / gray_a",
                    "light_gray / gray_b",
                    "gray / gray_c",
                    "dark_gray / gray_d",
                    "darker_gray / gray_e",
                    "black",
                ],
                [BLACK] * 3 + [WHITE] * 4,
                2,
            )

            pure_colors = (
                "pure_red",
                "pure_green",
                "pure_blue",
            )

            pure_lines = named_lines_group(
                3.2,
                pure_colors,
                pure_colors,
                [BLACK, BLACK, WHITE],
                6,
            )

            self.add(color_groups, other_lines, gray_lines, pure_lines)

            VGroup(*self.mobjects).move_to(ORIGIN)

.. automanimcolormodule:: manim.utils.color.manim_colors

"""

from typing import List

from .core import ManimColor

WHITE: ManimColor = ManimColor("#FFFFFF")
GRAY_A: ManimColor = ManimColor("#DDDDDD")
GREY_A: ManimColor = ManimColor("#DDDDDD")
GRAY_B: ManimColor = ManimColor("#BBBBBB")
GREY_B: ManimColor = ManimColor("#BBBBBB")
GRAY_C: ManimColor = ManimColor("#888888")
GREY_C: ManimColor = ManimColor("#888888")
GRAY_D: ManimColor = ManimColor("#444444")
GREY_D: ManimColor = ManimColor("#444444")
GRAY_E: ManimColor = ManimColor("#222222")
GREY_E: ManimColor = ManimColor("#222222")
BLACK: ManimColor = ManimColor("#000000")
LIGHTER_GRAY: ManimColor = ManimColor("#DDDDDD")
LIGHTER_GREY: ManimColor = ManimColor("#DDDDDD")
LIGHT_GRAY: ManimColor = ManimColor("#BBBBBB")
LIGHT_GREY: ManimColor = ManimColor("#BBBBBB")
GRAY: ManimColor = ManimColor("#888888")
GREY: ManimColor = ManimColor("#888888")
DARK_GRAY: ManimColor = ManimColor("#444444")
DARK_GREY: ManimColor = ManimColor("#444444")
DARKER_GRAY: ManimColor = ManimColor("#222222")
DARKER_GREY: ManimColor = ManimColor("#222222")
BLUE_A: ManimColor = ManimColor("#C7E9F1")
BLUE_B: ManimColor = ManimColor("#9CDCEB")
BLUE_C: ManimColor = ManimColor("#58C4DD")
BLUE_D: ManimColor = ManimColor("#29ABCA")
BLUE_E: ManimColor = ManimColor("#236B8E")
PURE_BLUE: ManimColor = ManimColor("#0000FF")
BLUE: ManimColor = ManimColor("#58C4DD")
DARK_BLUE: ManimColor = ManimColor("#236B8E")
TEAL_A: ManimColor = ManimColor("#ACEAD7")
TEAL_B: ManimColor = ManimColor("#76DDC0")
TEAL_C: ManimColor = ManimColor("#5CD0B3")
TEAL_D: ManimColor = ManimColor("#55C1A7")
TEAL_E: ManimColor = ManimColor("#49A88F")
TEAL: ManimColor = ManimColor("#5CD0B3")
GREEN_A: ManimColor = ManimColor("#C9E2AE")
GREEN_B: ManimColor = ManimColor("#A6CF8C")
GREEN_C: ManimColor = ManimColor("#83C167")
GREEN_D: ManimColor = ManimColor("#77B05D")
GREEN_E: ManimColor = ManimColor("#699C52")
PURE_GREEN: ManimColor = ManimColor("#00FF00")
GREEN: ManimColor = ManimColor("#83C167")
YELLOW_A: ManimColor = ManimColor("#FFF1B6")
YELLOW_B: ManimColor = ManimColor("#FFEA94")
YELLOW_C: ManimColor = ManimColor("#FFFF00")
YELLOW_D: ManimColor = ManimColor("#F4D345")
YELLOW_E: ManimColor = ManimColor("#E8C11C")
YELLOW: ManimColor = ManimColor("#FFFF00")
GOLD_A: ManimColor = ManimColor("#F7C797")
GOLD_B: ManimColor = ManimColor("#F9B775")
GOLD_C: ManimColor = ManimColor("#F0AC5F")
GOLD_D: ManimColor = ManimColor("#E1A158")
GOLD_E: ManimColor = ManimColor("#C78D46")
GOLD: ManimColor = ManimColor("#F0AC5F")
RED_A: ManimColor = ManimColor("#F7A1A3")
RED_B: ManimColor = ManimColor("#FF8080")
RED_C: ManimColor = ManimColor("#FC6255")
RED_D: ManimColor = ManimColor("#E65A4C")
RED_E: ManimColor = ManimColor("#CF5044")
PURE_RED: ManimColor = ManimColor("#FF0000")
RED: ManimColor = ManimColor("#FC6255")
MAROON_A: ManimColor = ManimColor("#ECABC1")
MAROON_B: ManimColor = ManimColor("#EC92AB")
MAROON_C: ManimColor = ManimColor("#C55F73")
MAROON_D: ManimColor = ManimColor("#A24D61")
MAROON_E: ManimColor = ManimColor("#94424F")
MAROON: ManimColor = ManimColor("#C55F73")
PURPLE_A: ManimColor = ManimColor("#CAA3E8")
PURPLE_B: ManimColor = ManimColor("#B189C6")
PURPLE_C: ManimColor = ManimColor("#9A72AC")
PURPLE_D: ManimColor = ManimColor("#715582")
PURPLE_E: ManimColor = ManimColor("#644172")
PURPLE: ManimColor = ManimColor("#9A72AC")
PINK: ManimColor = ManimColor("#D147BD")
LIGHT_PINK: ManimColor = ManimColor("#DC75CD")
ORANGE: ManimColor = ManimColor("#FF862F")
LIGHT_BROWN: ManimColor = ManimColor("#CD853F")
DARK_BROWN: ManimColor = ManimColor("#8B4513")
GRAY_BROWN: ManimColor = ManimColor("#736357")
GREY_BROWN: ManimColor = ManimColor("#736357")

# Colors used for Manim Community's logo and banner

LOGO_WHITE = ManimColor("#ECE7E2")
LOGO_GREEN = ManimColor("#87C2A5")
LOGO_BLUE = ManimColor("#525893")
LOGO_RED = ManimColor("#E07A5F")
LOGO_BLACK = ManimColor("#343434")

_all_manim_colors: List[ManimColor] = [
    x for x in globals().values() if isinstance(x, ManimColor)
]
