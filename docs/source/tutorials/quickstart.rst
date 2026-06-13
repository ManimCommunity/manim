from manim import *
import numpy as np

class AbsoluteRefineryPro(Scene):
    def construct(self):
        # ========== 1. معرفی داستان ==========
        story_title = Text("🏭 داستان قدر مطلق", font="B Nazanin", font_size=48, color=GOLD)
        story_subtitle = Text("پالایشگاه اعداد منفی", font="B Nazanin", font_size=32, color=BLUE_C)
        story_subtitle.next_to(story_title, DOWN)

        self.play(Write(story_title), run_time=1.5)
        self.play(FadeIn(story_subtitle, shift=UP), run_time=1)
        self.wait(1.5)
        self.play(FadeOut(story_title), FadeOut(story_subtitle))

        # ========== 2. ساختن پالایشگاه پیشرفته ==========
        # بدنه اصلی
        refinery_body = Rectangle(width=5, height=3.5, color=BLUE_D, fill_opacity=0.3, stroke_width=3)
        refinery_body.set_stroke(color=BLUE_C, width=3)

        # دودکش‌ها
        chimney1 = Rectangle(width=0.4, height=1.2, color=GRAY, fill_opacity=0.8)
        chimney1.move_to(refinery_body.get_top() + UP*0.2 + LEFT*1.2)
        chimney2 = Rectangle(width=0.4, height=1, color=GRAY, fill_opacity=0.8)
        chimney2.move_to(refinery_body.get_top() + UP*0.2 + RIGHT*1.2)

        # چرخ‌دنده‌ها
        gear1 = Circle(radius=0.3, color=YELLOW, fill_opacity=0.5)
        gear1.move_to(refinery_body.get_center() + LEFT*1)
        gear2 = Circle(radius=0.3, color=YELLOW, fill_opacity=0.5)
        gear2.move_to(refinery_body.get_center() + RIGHT*1)

        # دایره‌های داخلی چرخ‌دنده
        inner_gear1 = Circle(radius=0.15, color=RED, fill_opacity=0.8)
        inner_gear1.move_to(gear1.get_center())
        inner_gear2 = Circle(radius=0.15, color=RED, fill_opacity=0.8)
        inner_gear2.move_to(gear2.get_center())

        # تسمه نقاله (فلش ورودی و خروجی)
        conveyor_belt_in = Arrow(LEFT*4.5, LEFT*2.8, color=GRAY, stroke_width=5)
        conveyor_belt_out = Arrow(RIGHT*2.8, RIGHT*4.5, color=GRAY, stroke_width=5)

        # تابلو پالایشگاه
        sign_board = Rectangle(width=3, height=0.8, color=GREEN, fill_opacity=0.9)
        sign_board.next_to(refinery_body, UP, buff=0.3)
        sign_text = Text("پالایشگاه |x|", font_size=28, color=BLACK, font="B Nazanin")
        sign_text.move_to(sign_board.get_center())

        # جمع‌آوری پالایشگاه
        refinery_group = VGroup(refinery_body, chimney1, chimney2, gear1, gear2,
                                inner_gear1, inner_gear2, sign_board, sign_text)
        conveyor_group = VGroup(conveyor_belt_in, conveyor_belt_out)

        self.play(
            Create(refinery_body),
            *[Create(chimney) for chimney in [chimney1, chimney2]],
            run_time=1.5
        )
        self.play(
            *[Create(gear) for gear in [gear1, gear2]],
            *[Create(inner) for inner in [inner_gear1, inner_gear2]],
            run_time=1
        )
        self.play(
            Create(sign_board),
            Write(sign_text),
            *[Create(conv) for conv in [conveyor_belt_in, conveyor_belt_out]],
            run_time=1.5
        )

        # ========== 3. دود و جرقه ==========
        smoke1 = VGroup(*[Dot(point=chimney1.get_top() + UP*i, color=GRAY, radius=0.05) for i in range(1, 4)])
        smoke2 = VGroup(*[Dot(point=chimney2.get_top() + UP*i, color=GRAY, radius=0.05) for i in range(1, 4)])

        self.play(
            *[FadeIn(smoke, scale=0.5) for smoke in [smoke1, smoke2]],
            run_time=0.5
        )

        # ========== 4. تعریف اعداد با شخصیت ==========
        numbers_data = [
            {"input": "-5", "output": "5", "color": RED, "emoji": "😢"},
            {"input": "-3", "output": "3", "color": RED, "emoji": "😞"},
            {"input": "+4", "output": "4", "color": GREEN, "emoji": "😊"},
            {"input": "-2", "output": "2", "color": RED, "emoji": "🥺"},
            {"input": "+7", "output": "7", "color": GREEN, "emoji": "😄"},
            {"input": "-8", "output": "8", "color": RED, "emoji": "😭"}
        ]

        # لیست برای ذخیره انیمیشن‌های چرخ‌دنده
        gear_animations = []

        for i, num in enumerate(numbers_data):
            # ساخت عدد با کاراکتر صورتک
            number_group = VGroup(
                Text(num["emoji"], font_size=36),
                MathTex(num["input"], color=num["color"], font_size=48)
            )
            number_group.arrange(RIGHT, buff=0.2)
            number_group.move_to(conveyor_belt_in.get_end() + LEFT*0.5)

            # ورودی
            self.play(FadeIn(number_group, shift=RIGHT, scale=0.8), run_time=0.5)
            self.wait(0.3)

            # حرکت به سمت پالایشگاه
            self.play(
                number_group.animate.move_to(refinery_body.get_center()),
                run_time=1,
                rate_func=linear
            )

            # چرخش چرخ‌دنده‌ها و جرقه
            self.play(
                Rotate(gear1, angle=2*PI, run_time=0.5, rate_func=linear),
                Rotate(gear2, angle=-2*PI, run_time=0.5, rate_func=linear),
                Flash(refinery_body.get_center(), color=YELLOW, flash_radius=0.5, line_length=0.2),
                run_time=0.8
            )

            # دود اضافی
            puff = Circle(radius=0.2, color=GRAY, fill_opacity=0.6)
            puff.move_to(refinery_body.get_center())
            self.play(FadeOut(puff, scale=1.5), run_time=0.3)

            # تبدیل عدد
            new_number = MathTex(num["output"], color=GREEN, font_size=48)
            new_number.move_to(number_group.get_center())
            smile = Text("😊", font_size=36)
            smile.move_to(number_group.get_center() + LEFT*0.8)

            if num["color"] == RED:  # اگر منفی بود تبدیل شود
                self.play(
                    Transform(number_group, VGroup(smile, new_number)),
                    run_time=0.6
                )
            else:  # اگر مثبت بود فقط خوشحال می‌شود
                self.play(
                    number_group[0].animate.set_color(YELLOW),
                    run_time=0.3
                )

            # خروج از پالایشگاه
            self.play(
                number_group.animate.move_to(conveyor_belt_out.get_start() + RIGHT*0.5),
                run_time=1,
                rate_func=linear
            )

            # نمایش قانون
            if i == 2:  # وسط کار قانون را نشان بده
                rule = MathTex("|x| = \\begin{cases} x & x \\geq 0 \\\\ -x & x < 0 \\end{cases}", color=YELLOW)
                rule.scale(0.7)
                rule.to_edge(DOWN, buff=0.5)
                rule_box = SurroundingRectangle(rule, color=GOLD, buff=0.2)
                self.play(
                    Write(rule),
                    Create(rule_box),
                    run_time=2
                )
                self.wait(1)

            # محو شدن عدد بعد از خروج
            self.play(FadeOut(number_group, shift=RIGHT), run_time=0.4)

        # ========== 5. نتیجه‌گیری نهایی ==========
        conclusion = Text(
            "✨ هر عدد منفی که وارد پالایشگاه شود،\nبا قدر مطلق به عدد مثبت تبدیل می‌شود! ✨",
            font_size=36,
            color=GOLD,
            font="B Nazanin",
            line_spacing=1.5
        )
        conclusion.move_to(ORIGIN)
        conclusion_box = SurroundingRectangle(conclusion, color=PURPLE, buff=0.5, stroke_width=3)

        self.play(
            FadeOut(conveyor_group),
            FadeOut(rule),
            FadeOut(rule_box),
            run_time=1
        )

        self.play(
            Write(conclusion),
            Create(conclusion_box),
            run_time=2
        )

        # چرخش نهایی چرخ‌دنده‌ها
        self.play(
            Rotate(gear1, angle=4*PI, run_time=2, rate_func=linear),
            Rotate(gear2, angle=-4*PI, run_time=2, rate_func=linear),
            *[FadeIn(smoke, scale=0.8) for smoke in [smoke1, smoke2]],
            run_time=2
        )

        self.wait(2)

        # ========== 6. پایان ==========
        end_text = Text("پایان داستان | قدر مطلق یعنی فاصله از صفر |", font_size=30, color=BLUE)
        end_text.next_to(conclusion, DOWN, buff=0.8)

        self.play(Write(end_text), run_time=1.5)
        self.wait(3)
