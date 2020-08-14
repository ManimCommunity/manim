# Test


```python
from manim import *

class TestScene(Scene):
    def construct(self):
        dot = Dot()
        self.add(dot)
        self.wait(1)
```


and one more:

```python
from manim import *

class TestSceneXX(Scene):
    def construct(self):
        sq = Square()
        self.add(sq)
        self.wait(1)
```
