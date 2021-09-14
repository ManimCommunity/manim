import pkg_resources

from manim import __name__, __version__


def test_version():
    assert __version__ == pkg_resources.get_distribution(__name__).version
