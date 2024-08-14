from manim.mobject.mobject import Mobject
from manim.mobject.text.code_mobject import Code, CodeColorFormatter, create_code_string
from manim.mobject.text.text_mobject import Paragraph
from manim.mobject.types.vectorized_mobject import VGroup

PATH = "tests/utility_for_code_test.py"

INFILE = """\
def test()
    print("Hi")
    for i in out:
        print(i, "see you")
"""


class TestInternals:
    def test_formatter(self):
        code_str = create_code_string(PATH, None)
        formatter = CodeColorFormatter("dracula", code_str, "python", PATH)
        mapping = formatter.get_mapping()
        for line in mapping:
            for stylemap in line:
                if "\n" in stylemap[0]:
                    raise SyntaxError("Found uncatched newline in {line}")


class TestCreation:
    def test_from_file(self):
        file = Code(PATH)
        assert isinstance(file, VGroup)
        self.attributes(file)

    def test_from_code_snippet(self):
        code = Code(code=INFILE, language="python")
        assert isinstance(code, VGroup)
        self.attributes(code)

    def attributes(self, inst: Code):
        assert hasattr(inst, "code")
        assert isinstance(inst.code, Paragraph)
        assert hasattr(inst, "line_numbers")
        assert isinstance(inst.line_numbers, Paragraph)
        assert hasattr(inst, "background_mobject")
        assert isinstance(inst.background_mobject, Mobject)
