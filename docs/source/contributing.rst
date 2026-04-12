from manim import *

class FractionAddition(Scene):
    def construct(self):
        # 1. إنشاء الكسور الأساسية
        equation1 = MathTex(r"\frac{1}{2}", "+", r"\frac{1}{4}")
        equation1.scale(1.5)
        
        # 2. عرض المسألة في البداية
        self.play(Write(equation1))
        self.wait(1)

        # 3. تحويل الكسر 1/2 ليكون 2/4 (توحيد المقامات)
        # هنكتب الكسر الجديد في نفس مكان القديم
        fraction_step2 = MathTex(r"\frac{1 \times 2}{2 \times 2}", "+", r"\frac{1}{4}")
        fraction_step2.scale(1.5)
        
        fraction_step3 = MathTex(r"\frac{2}{4}", "+", r"\frac{1}{4}")
        fraction_step3.scale(1.5)

        self.play(ReplacementTransform(equation1[0], fraction_step3[0]))
        self.wait(1)

        # 4. تداخل الأرقام ليكون المقام موحد
        common_denominator_step = MathTex(r"\frac{2+1}{4}")
        common_denominator_step.scale(1.5)
        
        # إخفاء علامة الزائد والكسر التاني ودمجهم في كسر واحد
        self.play(
            ReplacementTransform(VGroup(fraction_step3[0], fraction_step3[1], fraction_step3[2]), common_denominator_step)
        )
        self.wait(1)

        # 5. ظهور علامة اليساوي "كأنها وقعت من فوق"
        equals_sign = MathTex("=")
        equals_sign.scale(1.5)
        equals_sign.next_to(common_denominator_step, RIGHT)
        
        # بنحطها فوق الشاشة الأول وبعدين بننزلها
        equals_sign.shift(UP * 5) 
        self.play(equals_sign.animate.shift(DOWN * 5), run_time=1, rate_func=bounce)
        
        # 6. كتابة الناتج النهائي
        final_result = MathTex(r"\frac{3}{4}")
        final_result.scale(1.5)
        final_result.next_to(equals_sign, RIGHT)
        
        self.play(Write(final_result))
        self.wait(2)
do so here. Manim is Free and Open Source Software (FOSS) for mathematical
animations. As such, **we welcome everyone** who is interested in
mathematics, pedagogy, computer animations, open-source,
software development, and beyond. Manim accepts many kinds of contributions,
some are detailed below:

*  Code maintenance and development
*  DevOps
*  Documentation
*  Developing educational content & narrative documentation
*  Plugins to extend Manim functionality
*  Testing (graphical, unit & video)
*  Website design and development
*  Translating documentation and docstrings

To get an overview of what our community is currently working on, check out
`our development project board <https://github.com/orgs/ManimCommunity/projects/7/views/1>`__.

.. note::
   Please ensure that you are reading the latest version of this guide by ensuring that "latest" is selected in the version switcher.



Contributing can be confusing, so here are a few guides:

.. toctree::
   :maxdepth: 3

   contributing/development
   contributing/docs
   contributing/testing
   contributing/performance
   contributing/internationalization
