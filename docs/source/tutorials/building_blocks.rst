from manim import *

class AsteroideGIAVA(Scene):
    def construct(self):
        # --- CONFIGURAZIONE ---
        self.camera.background_color = BLACK

        # 1. Creazione della Terra
        # Creiamo un cerchio blu per rappresentare la Terra
        terra = Circle(radius=2, color=BLUE, fill_opacity=1)
        # Aggiungiamo un po' di contorni/ombre per dargli un asse più 3D
        terra_ombra = Circle(radius=2, color=BLACK, fill_opacity=0.3)
        terra_ombra.shift(UP * 0.5 + RIGHT * 0.5)

        # Gruppiamo gli elementi della Terra
        earth_group = VGroup(terra, terra_ombra)
        earth_group.shift(DOWN * 1.5) # Abbassiamo la Terra in basso allo schermo

        # 2. Creazione dell'Asteroide
        # Una sfera rossa/arancione con una scia (usiamo un punto o una piccola sfera)
        asteroide = Dot(point=UP * 3.5, color=ORANGE, radius=0.3)

        # Creiamo una scia di fuoco (una linea che segue l'asteroide)
        scia_fiamma = Line(start=UP * 3.5, end=UP * 4.5, color=RED, stroke_width=10)
        scia_fiamma.add_updater(lambda m: m.put_start_and_end_on(asteroide.get_center(), asteroide.get_center() + UP * 1.5))

        # 3. Animazione dell'impatto
        self.play(FadeIn(earth_group))
        self.play(FadeIn(asteroide), FadeIn(scia_fiamma))

        # L'asteroide cade verso la Terra
        # duration=2 rende l'impatto veloce
        self.play(asteroide.animate.move_to(earth_group.get_center() + UP * 0.5), run_time=2, rate_func=linear)

        # 4. L'Esplosione
        # Creiamo cerchi concentrici che si espandono velocemente (effetto onda d'urto)
        esplosione_1 = Circle(color=YELLOW, radius=0.1).move_to(asteroide.get_center())
        esplosione_2 = Circle(color=RED, radius=0.1).move_to(asteroide.get_center())
        esplosione_3 = Circle(color=ORANGE, radius=0.1).move_to(asteroide.get_center())

        # Facciamo sparire asteroide e scia
        self.remove(asteroide, scia_fiamma)

        # Animiamo l'espansione dell'esplosione
        self.play(
            FadeOut(earth_group, shift=DOWN, scale=0.5), # La Terra viene oscurata
            AnimationGroup(
                esplosione_1.animate.set_fill(opacity=0).set_stroke(width=0).scale(15),
                esplosione_2.animate.set_fill(opacity=0).set_stroke(width=0).scale(12),
                esplosione_3.animate.set_fill(opacity=1).scale(8),
                lag_ratio=0.1
            ),
            run_time=1
        )
        self.play(FadeOut(esplosione_1), FadeOut(esplosione_2), FadeOut(esplosione_3), run_time=0.5)

        # 5. Apparizione della scritta "GIAVA"
        # Creiamo il testo
        testo_giava = Text("GIAVA", font_size=120, color=WHITE).scale(1.5)
        testo_giava.set_color_by_gradient(RED, ORANGE, YELLOW) # Effetto fuoco sul testo

        # Animazione di entrata spettacolare (scala dal centro)
        self.play(Write(testo_giava), run_time=1.5)

        # Teniamo il testo fermo per riempire i 10 secondi totali
        self.wait(4)
