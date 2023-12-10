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

WHITE = ManimColor("#FFFFFF")
GRAY_A = ManimColor("#DDDDDD")
GREY_A = ManimColor("#DDDDDD")
GRAY_B = ManimColor("#BBBBBB")
GREY_B = ManimColor("#BBBBBB")
GRAY_C = ManimColor("#888888")
GREY_C = ManimColor("#888888")
GRAY_D = ManimColor("#444444")
GREY_D = ManimColor("#444444")
GRAY_E = ManimColor("#222222")
GREY_E = ManimColor("#222222")
BLACK = ManimColor("#000000")
LIGHTER_GRAY = ManimColor("#DDDDDD")
LIGHTER_GREY = ManimColor("#DDDDDD")
LIGHT_GRAY = ManimColor("#BBBBBB")
LIGHT_GREY = ManimColor("#BBBBBB")
GRAY = ManimColor("#888888")
GREY = ManimColor("#888888")
DARK_GRAY = ManimColor("#444444")
DARK_GREY = ManimColor("#444444")
DARKER_GRAY = ManimColor("#222222")
DARKER_GREY = ManimColor("#222222")
BLUE_A = ManimColor("#C7E9F1")
BLUE_B = ManimColor("#9CDCEB")
BLUE_C = ManimColor("#58C4DD")
BLUE_D = ManimColor("#29ABCA")
BLUE_E = ManimColor("#236B8E")
PURE_BLUE = ManimColor("#0000FF")
BLUE = ManimColor("#58C4DD")
DARK_BLUE = ManimColor("#236B8E")
TEAL_A = ManimColor("#ACEAD7")
TEAL_B = ManimColor("#76DDC0")
TEAL_C = ManimColor("#5CD0B3")
TEAL_D = ManimColor("#55C1A7")
TEAL_E = ManimColor("#49A88F")
TEAL = ManimColor("#5CD0B3")
GREEN_A = ManimColor("#C9E2AE")
GREEN_B = ManimColor("#A6CF8C")
GREEN_C = ManimColor("#83C167")
GREEN_D = ManimColor("#77B05D")
GREEN_E = ManimColor("#699C52")
PURE_GREEN = ManimColor("#00FF00")
GREEN = ManimColor("#83C167")
YELLOW_A = ManimColor("#FFF1B6")
YELLOW_B = ManimColor("#FFEA94")
YELLOW_C = ManimColor("#FFFF00")
YELLOW_D = ManimColor("#F4D345")
YELLOW_E = ManimColor("#E8C11C")
YELLOW = ManimColor("#FFFF00")
GOLD_A = ManimColor("#F7C797")
GOLD_B = ManimColor("#F9B775")
GOLD_C = ManimColor("#F0AC5F")
GOLD_D = ManimColor("#E1A158")
GOLD_E = ManimColor("#C78D46")
GOLD = ManimColor("#F0AC5F")
RED_A = ManimColor("#F7A1A3")
RED_B = ManimColor("#FF8080")
RED_C = ManimColor("#FC6255")
RED_D = ManimColor("#E65A4C")
RED_E = ManimColor("#CF5044")
PURE_RED = ManimColor("#FF0000")
RED = ManimColor("#FC6255")
MAROON_A = ManimColor("#ECABC1")
MAROON_B = ManimColor("#EC92AB")
MAROON_C = ManimColor("#C55F73")
MAROON_D = ManimColor("#A24D61")
MAROON_E = ManimColor("#94424F")
MAROON = ManimColor("#C55F73")
PURPLE_A = ManimColor("#CAA3E8")
PURPLE_B = ManimColor("#B189C6")
PURPLE_C = ManimColor("#9A72AC")
PURPLE_D = ManimColor("#715582")
PURPLE_E = ManimColor("#644172")
PURPLE = ManimColor("#9A72AC")
PINK = ManimColor("#D147BD")
LIGHT_PINK = ManimColor("#DC75CD")
ORANGE = ManimColor("#FF862F")
LIGHT_BROWN = ManimColor("#CD853F")
DARK_BROWN = ManimColor("#8B4513")
GRAY_BROWN = ManimColor("#736357")
GREY_BROWN = ManimColor("#736357")

# Colors used for Manim Community's logo and banner

LOGO_WHITE = ManimColor("#ECE7E2")
LOGO_GREEN = ManimColor("#87C2A5")
LOGO_BLUE = ManimColor("#525893")
LOGO_RED = ManimColor("#E07A5F")
LOGO_BLACK = ManimColor("#343434")

_all_manim_colors: List[ManimColor] = [
    x for x in globals().values() if isinstance(x, ManimColor)
]
