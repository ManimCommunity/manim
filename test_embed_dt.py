"""Test manual para verificar que embed() usa deltaTime real."""
from manim import *

class TestEmbedDt(Scene):
    def construct(self):
        # Usar una flecha para ver la rotación claramente
        arrow = Arrow(ORIGIN, RIGHT * 2, color=BLUE, buff=0)
        self.add(arrow)
        
        # Agregar un updater que rota la flecha
        # Si dt funciona correctamente, rotará a velocidad constante
        arrow.add_updater(lambda m, dt: m.rotate(PI * dt))
        
        # Entrar en modo interactivo
        self.manager._interact()
