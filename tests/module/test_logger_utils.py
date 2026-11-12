import io
import logging

from manim._config.logger_utils import JSONFormatter


def test_the_json_formatter_writes_logged_exceptions():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JSONFormatter())
    log = logging.getLogger("manim.test_json_formatter")
    log.addHandler(handler)
    log.propagate = False
    try:
        try:
            raise OSError("cannot close")
        except OSError:
            log.exception("cleanup failed for %(name)s", {"name": "scene"})
    finally:
        log.removeHandler(handler)
    assert "cleanup failed for <>" in stream.getvalue()
    assert "OSError: cannot close" in stream.getvalue()
