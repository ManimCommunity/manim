from manim import __name__, __version__
import pkg_resources


def test_version():
    assert __version__ == pkg_resources.get_distribution(__name__).version
