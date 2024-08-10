# This file is to test file functionalities of Code-Class
# test_code_mobject.py test will call this file
import numpy as np

CODE_1 = """\
def test()
    print("Hi")
    for i in out:
        print(i, "see you")
"""

IDENTATION_CHAR = "    "
i, b, f = 0, False, 3.1415
list = [1, 2, 3, 4, 5]
array = np.array(list)

mapping, html = [], ""


class Foo:
    def __init__(self, i):
        print(Foo.__name__ + "t", i)


class AnnoyingLinter(Exception):
    pass


def fun_gus(
    a,
    b: str,
) -> Foo:
    a += b
    if b > a:  # Inline
        raise SystemError


try:
    for i in list:
        # Intendented
        bar_length = Foo(i)
        k = 0
        while k < i:
            k += 2
except AnnoyingLinter:
    print("upsie")
finally:
    recovery = "Why not"

print("Resolution")
# Last bit
