from manim import *

class SeriIspat(Scene):
    def construct(self):
        # 1. Ana Kare Oluşturma
        ana_kare = Square(side_length=5, color=WHITE)
        self.play(Create(ana_kare))

        # 2. 1/7'lik Sütun (Mavi)
        # Genişlik = 5 birim / 7
        mavi_sutun = Rectangle(
            width=5/7, height=5,
            fill_opacity=0.8, color=BLUE, stroke_width=2
        ).align_to(ana_kare, LEFT)

        etiket1 = Text("1/7", font_size=24).move_to(mavi_sutun.get_center())

        self.play(FadeIn(mavi_sutun), Write(etiket1))
        self.wait(0.5)

        # 3. 1/49'luk Kare (Turuncu)
        # Kenar = 5 / 7
        turuncu_kare = Square(
            side_length=5/7,
            fill_opacity=1, color=ORANGE, stroke_width=2
        ).next_to(mavi_sutun, RIGHT, buff=0).align_to(ana_kare, DOWN)

        etiket2 = Text("1/49", font_size=18).move_to(turuncu_kare.get_center())

        self.play(FadeIn(turuncu_kare), Write(etiket2))
        self.wait(0.5)

        # 4. 1/343'lük Kare (Sarı)
        sari_kare = Rectangle(
            width=5/7, height=5/49,
            fill_opacity=1, color=YELLOW, stroke_width=1
        ).next_to(turuncu_kare, DOWN, buff=0).align_to(turuncu_kare, LEFT)

        # Çok küçük olduğu için etiketini yanına koyalım
        etiket3 = Text("1/343", font_size=14).next_to(sari_kare, RIGHT, buff=0.1)

        self.play(FadeIn(sari_kare), Write(etiket3))
        self.wait(1)

        # 5. Final Yazısı
        toplam_yazi = Text("Toplam Boyalı Alan = 1/6", color=YELLOW).to_edge(UP)
        self.play(Write(toplam_yazi))
        self.wait(2)
