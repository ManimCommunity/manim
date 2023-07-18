"""A CLI utility helping to diagnose problems with
your Manim installation.

"""

from __future__ import annotations

import click
import cloup

from .checks import HEALTH_CHECKS


@cloup.command(
    context_settings=None,
)
def checkhealth():
    """This subcommand checks whether Manim is installed correctly
    and has access to its required (and optional) system dependencies.
    """
    click.echo(
        "Checking whether your installation of Manim Community "
        "is healthy..."
    )
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
            from manim import tempconfig, Scene, ManimBanner, FadeOut, UP

            class CheckHealthDemo(Scene):
                def construct(self):
                    banner = ManimBanner()
                    self.play(banner.create())
                    self.wait()
                    self.play(banner.expand())
                    self.wait()
                    self.play(FadeOut(banner, shift=UP))
            
            with tempconfig({"preview": True, "disable_caching": True}):
                CheckHealthDemo().render()
