import pytest

@pytest.fixture(autouse=True)
def enable_info_logging():
    import logging
    logging.getLogger("manim").setLevel(logging.INFO)

