"""A CLI utility helping to diagnose problems with
your Manim installation.

"""

from __future__ import annotations

import sys

import click
import cloup

from .checks import HEALTH_CHECKS

__all__ = ["checkhealth"]


@cloup.command(
    context_settings=None,
)
def checkhealth():
    """This subcommand checks whether Manim is installed correctly
    and has access to its required (and optional) system dependencies.
    """
    click.echo(f"Python executable: {sys.executable}\n")
    click.echo("Checking whether your installation of Manim Community is healthy...")
    failed_checks = []

    for check in HEALTH_CHECKS:
        click.echo(f"- {check.description} ... ", nl=False)
        if any(
            failed_check.__name__ in check.skip_on_failed
            for failed_check in failed_checks
        ):
            click.secho("SKIPPED", fg="blue")
            continue
        check_result = check()
        if check_result:
            click.secho("PASSED", fg="green")
        else:
            click.secho("FAILED", fg="red")
            failed_checks.append(check)

    click.echo()

    if failed_checks:
        click.echo(
            "There are problems with your installation, "
            "here are some recommendations to fix them:"
        )
        for ind, failed_check in enumerate(failed_checks):
            click.echo(failed_check.recommendation)
            if ind + 1 < len(failed_checks):
                click.confirm("Continue with next recommendation?")

    else:  # no problems detected!
        click.echo("No problems detected, your installation seems healthy!")
        render_test_scene = click.confirm(
            "Would you like to render and preview a test scene?"
        )
        if render_test_scene:
            import manim as mn

            class CheckHealthDemo(mn.Scene):
                def construct(self):
                    banner = mn.ManimBanner().shift(mn.UP * 0.5)
                    self.play(banner.create())
                    self.wait(0.5)
                    self.play(banner.expand())
                    self.wait(0.5)
                    text_left = mn.Text("All systems operational!")
                    formula_right = mn.MathTex(r"\oint_{\gamma} f(z)~dz = 0")
                    text_tex_group = mn.VGroup(text_left, formula_right)
                    text_tex_group.arrange(mn.RIGHT, buff=1).next_to(banner, mn.DOWN)
                    self.play(mn.Write(text_tex_group))
                    self.wait(0.5)
                    self.play(
                        mn.FadeOut(banner, shift=mn.UP),
                        mn.FadeOut(text_tex_group, shift=mn.DOWN),
                    )

            with mn.tempconfig({"preview": True, "disable_caching": True}):
                CheckHealthDemo().render()
