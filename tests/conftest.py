import pytest
import sys


@pytest.fixture(scope="module")
def python_version():
    return "python3" if sys.platform == "darwin" else "python"
