import inspect
import manim

for name, obj in inspect.getmembers(manim):
    if inspect.isclass(obj) or inspect.ismethod(obj):
        try:
            obj()
        except Exception as e:
            print(obj.__name__)
