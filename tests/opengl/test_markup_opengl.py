from manim import MarkupText


def test_good_markup(using_opengl_renderer):
    """Test creation of valid :class:`MarkupText` object"""
    try:
        MarkupText("<b>foo</b>")
        MarkupText("foo")
        success = True
    except ValueError:
        success = False
    assert success, "'<b>foo</b>' and 'foo' should not fail validation"


def test_special_tags_markup(using_opengl_renderer):
    """Test creation of valid :class:`MarkupText` object with unofficial tags"""
    try:
        MarkupText('<color col="RED">foo</color>')
        MarkupText('<gradient from="RED" to="YELLOW">foo</gradient>')
        success = True
    except ValueError:
        success = False
    assert (
        success
    ), '\'<color col="RED">foo</color>\' and \'<gradient from="RED" to="YELLOW">foo</gradient>\' should not fail validation'


def test_unbalanced_tag_markup(using_opengl_renderer):
    """Test creation of invalid :class:`MarkupText` object (unbalanced tag)"""
    try:
        MarkupText("<b>foo")
        success = False
    except ValueError:
        success = True
    assert success, "'<b>foo' should fail validation"


def test_invalid_tag_markup(using_opengl_renderer):
    """Test creation of invalid :class:`MarkupText` object (invalid tag)"""
    try:
        MarkupText("<invalidtag>foo</invalidtag>")
        success = False
    except ValueError:
        success = True

    assert success, "'<invalidtag>foo</invalidtag>' should fail validation"
