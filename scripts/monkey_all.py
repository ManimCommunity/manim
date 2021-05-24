"""Gathers Info for MonkeyType"""

import inspect
import importlib
import manim
import os
import sys


def all_tests() -> None:
    sys.path.append('tests')
    all_files = []

    for root, dirs, files in os.walk("."):
        if "__pycache__" in root:
            continue
        for file in files:
            all_files.append(file)
        for directory in dirs:
            if directory != "__pycache__":
                sys.path.append(f"{root}\\{directory}")

    for file in all_files:
        if file.startswith("test_"):
            try:
                importlib.import_module(file)
            except Exception as e:
                print(f"Can't run this test: {file}, Beacuse: {e}")

    print("Tests Done")


def all_classes() -> None:
    for name, obj in inspect.getmembers(manim):
        if callable(obj):
            try:
                obj()
            except TypeError:
                pass
            except Exception as e:
                print(f"Error: {e} from {obj.__name__}")
    print("Classes Done")


all_tests()
# all_classes()
