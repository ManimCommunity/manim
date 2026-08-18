from manim.mobject.abstract.positionable import Positionable
from manim.mobject.mobject import Mobject
from manim.mobject.opengl.opengl_mobject import OpenGLMobject
from manim.mobject.opengl.opengl_vectorized_mobject import OpenGLVMobject
from manim.mobject.types.vectorized_mobject import VMobject


def main() -> None:
    seen: set[str] = set()

    for cls in [Mobject, VMobject, OpenGLMobject, OpenGLVMobject]:
        assert isinstance(cls, type)
        print(cls.__name__)
        for name, attr in sorted(cls.__dict__.items()):
            if (
                name in seen
                or name.startswith("__")
                or attr is getattr(cls.__base__, name, None)
            ):
                continue
            print(
                f"\t{'-+'[getattr(Positionable, name, None) is not getattr(Positionable.__base__, name, None)]} {name}"
            )
        seen |= cls.__dict__.keys()

    print(Positionable.__name__)
    for name, attr in Positionable.__dict__.items():
        if name.startswith("__") or attr is getattr(Positionable.__base__, name, None):
            continue
        print(f"\t* {name}", "(new)" if name not in seen else "")


if __name__ == "__main__":
    main()
