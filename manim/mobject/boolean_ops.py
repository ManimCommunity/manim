from .types.vectorized_mobject import VMobject

class Union(VMobject):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(self,*args, **kwargs)

class Difference(VMobject):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(self, *args, **kwargs)

class Intersection(VMobject):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(self, *args, **kwargs)


