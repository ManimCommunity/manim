from __future__ import annotations

from manim.mobject.text.code_mobject import Code


def test_code_indentation():
    co = Code(
        code="""\
    def test()
        print("Hi")
        """,
        language="Python",
        indentation_chars="    ",
    )

    assert co.tab_spaces[0] == 1
    assert co.tab_spaces[1] == 2
