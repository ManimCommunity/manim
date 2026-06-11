from __future__ import annotations

from manim import AnimationGroup, Code, FadeIn, FadeOut, LaggedStart, Transform, linear


def find_line_matches(before: Code, after: Code):
    before_lines = [
        line.lstrip() if line.strip() != "" else None
        for line in before.code_string.splitlines()
    ]
    after_lines = [
        line.lstrip() if line.strip() != "" else None
        for line in after.code_string.splitlines()
    ]

    matches = []

    for i, b_line in enumerate(before_lines):
        if b_line is None:
            continue
        for j, a_line in enumerate(after_lines):
            if a_line is not None and b_line == a_line:
                matches.append((i, j))
                before_lines[i] = None
                after_lines[j] = None
                break

    deletions = []
    for i, line in enumerate(before_lines):
        if before_lines[i] is not None:
            deletions.append((i, len(line)))

    additions = []
    for j, line in enumerate(after_lines):
        if after_lines[j] is not None:
            additions.append((j, len(line)))

    return matches, deletions, additions


class CodeTransform(AnimationGroup):
    """
    An animation that smoothly transitions between two Code objects.

    PARAMETERS
    ----------
    before : Code
        The initial Code object.
    after : Code
        The target Code object after the transition.
    """

    def __init__(self, before: Code, after: Code, **kwargs):
        matches, deletions, additions = find_line_matches(before, after)

        transform_pairs = [(before.code[i], after.code[j]) for i, j in matches]

        delete_lines = [before.code[i] for i, _ in deletions]

        add_lines = [after.code[j] for j, _ in additions]

        animations = []

        if hasattr(before, "background_mobject") and hasattr(
            after, "background_mobject"
        ):
            animations.append(
                Transform(before.background_mobject, after.background_mobject)
            )

        if hasattr(before, "line_numbers") and hasattr(after, "line_numbers"):
            animations.append(Transform(before.line_numbers, after.line_numbers))

        if delete_lines:
            animations.append(FadeOut(*delete_lines))

        if transform_pairs:
            animations.append(
                LaggedStart(
                    *[
                        Transform(before_line, after_line)
                        for before_line, after_line in transform_pairs
                    ]
                )
            )

        if add_lines:
            animations.append(FadeIn(*add_lines))

        super().__init__(
            *animations,
            group=None,
            run_time=None,
            rate_func=linear,
            lag_ratio=0.0,
            **kwargs,
        )
