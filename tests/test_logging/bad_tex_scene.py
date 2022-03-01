from manim import Scene, Tex, TexTemplate


class BadTex(Scene):
    def construct(self):
        tex_template = TexTemplate(preamble=r"\usepackage{notapackage}")
        some_tex = r"\frac{1}{0}"
        my_tex = Tex(some_tex, tex_template=tex_template)
        self.add(my_tex)
