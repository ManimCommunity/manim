import inspect
import importlib
import manim
import os
import sys

sys.path.append('tests')

for file in os.listdir("tests"):
    if file.startswith("test_"):
        try:
            importlib.import_module(file[:-3])
        except Exception as e:
            print("Can't run this test: " + file)

for name, obj in inspect.getmembers(manim):
    if callable(obj):
        try:
            obj()
        except TypeError:
            pass
        except Exception as e:
            print(f"Error: {e} from {obj.__name__}")

