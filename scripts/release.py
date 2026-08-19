#!/usr/bin/env python3
"""
Release management tools for Manim.

This script provides commands for preparing and managing Manim releases:
- Generate changelogs from GitHub's release notes API
- Export documentation changelogs as GitHub-flavored release notes
- Update CITATION.cff with new version information
- Fetch existing release notes for documentation

Usage:
    # Generate changelog for an upcoming release
    uv run python scripts/release.py changelog --base v0.19.0 --version 0.20.0

    # Include a PR omitted by GitHub's generated release notes
    uv run python scripts/release.py changelog --base v0.19.0 --version 0.20.0 --include-extra 1234

    # Also update CITATION.cff at the same time
    uv run python scripts/release.py changelog --base v0.19.0 --version 0.20.0 --update-citation

    # Export an edited changelog for pasting into a GitHub release
    uv run python scripts/release.py export-changelog 0.20.0

    # Update only CITATION.cff
    uv run python scripts/release.py citation --version 0.20.0

    # Fetch existing release changelogs for documentation
    uv run python scripts/release.py fetch-releases

Requirements:
    - gh CLI installed and authenticated
    - Python 3.11+
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from collections.abc import Sequence

# =============================================================================
# Constants
# =============================================================================

REPO = "manimcommunity/manim"

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CHANGELOG_DIR = REPO_ROOT / "docs" / "source" / "changelog"
CITATION_TEMPLATE = SCRIPT_DIR / "TEMPLATE.cff"
CITATION_FILE = REPO_ROOT / "CITATION.cff"

# Minimum version for fetching historical releases
DEFAULT_MIN_VERSION = "0.18.0"


# =============================================================================
# GitHub CLI Helpers
# =============================================================================


def run_gh(
    args: Sequence[str],
    *,
    check: bool = True,
    suppress_errors: bool = False,
) -> subprocess.CompletedProcess[str]:
    """
    Run a gh CLI command.

    Args:
        args: Arguments to pass to gh
        check: If True, raise on non-zero exit
        suppress_errors: If True, don't print errors to stderr

    Returns:
        CompletedProcess with stdout/stderr
    """
    result = subprocess.run(
        ["gh", *args],
        capture_output=True,
        text=True,
    )
    if (
        result.returncode != 0
        and not suppress_errors
        and "not found" not in result.stderr.lower()
    ):
        click.echo(f"gh error: {result.stderr}", err=True)
    if check and result.returncode != 0:
        raise click.ClickException(f"gh command failed: gh {' '.join(args)}")
    return result


def get_release_tags() -> list[str]:
    """Get all published release tags from GitHub, newest first."""
    result = run_gh(
        ["release", "list", "--repo", REPO, "--limit", "100", "--json", "tagName"],
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return []

    try:
        data = json.loads(result.stdout)
        return [item["tagName"] for item in data]
    except (json.JSONDecodeError, KeyError):
        return []


def get_release_body(tag: str) -> str | None:
    """Get the release body for a published release."""
    result = run_gh(
        ["release", "view", tag, "--repo", REPO, "--json", "body", "--jq", ".body"],
        check=False,
        suppress_errors=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def get_release_date(tag: str) -> str | None:
    """Get the release date formatted as 'Month DD, YYYY'."""
    result = run_gh(
        [
            "release",
            "view",
            tag,
            "--repo",
            REPO,
            "--json",
            "createdAt",
            "--jq",
            ".createdAt",
        ],
        check=False,
    )
    if result.returncode != 0:
        return None

    date_str = result.stdout.strip().strip('"')
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return dt.strftime("%B %d, %Y")
    except ValueError:
        return None


def get_pull_request_release_note(number: int) -> str:
    """Return a GitHub-style release-note bullet for a pull request."""
    result = run_gh(
        [
            "pr",
            "view",
            str(number),
            "--repo",
            REPO,
            "--json",
            "title,author,url",
        ]
    )
    try:
        data = json.loads(result.stdout)
        title = data["title"]
        author = data["author"]["login"]
        url = data["url"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise click.ClickException(f"Could not read pull request #{number}") from error
    return f"* {title} by @{author} in {url}"


def generate_release_notes(head_tag: str, base_tag: str) -> str:
    """
    Generate release notes using GitHub's API.

    This respects .github/release.yml for PR categorization.
    """
    result = run_gh(
        [
            "api",
            f"repos/{REPO}/releases/generate-notes",
            "--field",
            f"tag_name={head_tag}",
            "--field",
            f"previous_tag_name={base_tag}",
            "--jq",
            ".body",
        ]
    )
    body = result.stdout.strip()
    if not body:
        raise click.ClickException("GitHub API returned empty release notes")
    return body


# =============================================================================
# Version Utilities
# =============================================================================


def normalize_tag(tag: str) -> str:
    """Ensure tag has 'v' prefix."""
    return tag if tag.startswith("v") else f"v{tag}"


def version_from_tag(tag: str) -> str:
    """Extract version from tag (e.g., 'v0.18.0' -> '0.18.0')."""
    return tag[1:] if tag.startswith("v") else tag


def parse_version(version: str) -> tuple[int, ...]:
    """Parse version string into comparable tuple."""
    # Handle post-releases like '0.18.0.post0'
    version = version.replace(".post", "-post")
    parts = []
    for part in version.replace("-", ".").split("."):
        try:
            parts.append(int(part))
        except ValueError:
            continue
    # Pad to at least 3 components
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def version_gte(version: str, min_version: str) -> bool:
    """Check if version >= min_version."""
    return parse_version(version) >= parse_version(min_version)


# =============================================================================
# Markdown Conversion
# =============================================================================


_PR_REFERENCE = re.compile(
    r"(?:\{pr\}`(?P<myst>\d+)`|"
    r"https://github\.com/ManimCommunity/manim/pull/(?P<url>\d+))"
)
_TOP_LEVEL_BULLET = re.compile(r"^[*-]\s")
_SECOND_LEVEL_HEADING = re.compile(r"^##(?!#)\s+(.+?)\s*$")
_ANY_HEADING = re.compile(r"^#{1,6}\s")
_CHANGELOG_SECTION_SCOPES = {
    "What's Changed": "changes",
    "New Contributors": "contributors",
}


def include_extra_pull_requests(body: str, numbers: Sequence[int]) -> str:
    """Add explicitly requested PRs to the generated "Other Changes" category."""
    existing_numbers = {
        int(match.group("url"))
        for match in _PR_REFERENCE.finditer(body)
        if match.group("url") is not None
    }
    entries = [
        get_pull_request_release_note(number)
        for number in numbers
        if number not in existing_numbers
    ]
    if not entries:
        return body

    lines = body.splitlines()
    category_heading = "### Other Changes"
    try:
        category_start = lines.index(category_heading)
    except ValueError:
        insertion_point = next(
            (
                index
                for index, line in enumerate(lines)
                if line == "## New Contributors"
                or line.startswith("**Full Changelog**:")
            ),
            len(lines),
        )
        lines[insertion_point:insertion_point] = [category_heading, *entries]
    else:
        insertion_point = category_start + 1
        while insertion_point < len(lines) and not lines[insertion_point].startswith(
            "#"
        ):
            insertion_point += 1
        while (
            insertion_point > category_start + 1
            and not lines[insertion_point - 1].strip()
        ):
            insertion_point -= 1
        lines[insertion_point:insertion_point] = entries

    return "\n".join(lines)


def _find_pull_request_blocks(
    lines: list[str],
) -> list[tuple[int, int, tuple[str, str]]]:
    """Find PR list items and their continuation lines in a changelog.

    The returned key includes the enclosing second-level section so that a PR's
    release note and its potential "New Contributors" entry are kept separate.
    Third-level category headings are deliberately not part of the key: the
    newly generated changelog remains authoritative for PR categorization.
    """
    blocks = []
    section_scope: str | None = None
    line_number = 0

    while line_number < len(lines):
        line = lines[line_number]
        heading = _SECOND_LEVEL_HEADING.match(line.rstrip("\r\n"))
        if heading:
            section_scope = _CHANGELOG_SECTION_SCOPES.get(heading.group(1))
            line_number += 1
            continue

        if section_scope is None or not _TOP_LEVEL_BULLET.match(line):
            line_number += 1
            continue

        pr_reference = _PR_REFERENCE.search(line)
        if pr_reference is None:
            line_number += 1
            continue

        pr_number = pr_reference.group("myst") or pr_reference.group("url")
        block_end = line_number + 1
        while block_end < len(lines):
            next_line = lines[block_end]
            if (
                _TOP_LEVEL_BULLET.match(next_line)
                or _ANY_HEADING.match(next_line)
                or next_line.startswith("**Full Changelog**:")
            ):
                break
            block_end += 1
        while block_end > line_number + 1 and not lines[block_end - 1].strip():
            block_end -= 1

        blocks.append(
            (line_number, block_end, (section_scope, pr_number)),
        )
        line_number = block_end

    return blocks


def merge_changelog(generated: str, existing: str) -> str:
    """Merge existing PR note edits into a newly generated changelog.

    The generated changelog controls which PRs are present, their ordering, and
    their category headings. For every PR that is present in both versions, its
    existing list item (including indented descriptions) is retained. Thus,
    deleted entries are restored and label changes move entries to their newly
    generated categories without discarding copy edits.
    """
    generated_lines = generated.splitlines(keepends=True)
    existing_lines = existing.splitlines(keepends=True)
    existing_blocks = {
        key: "".join(existing_lines[start:end])
        for start, end, key in _find_pull_request_blocks(existing_lines)
    }

    merged = []
    previous_end = 0
    for start, end, key in _find_pull_request_blocks(generated_lines):
        merged.extend(generated_lines[previous_end:start])
        existing_block = existing_blocks.get(key)
        if existing_block is None:
            merged.extend(generated_lines[start:end])
        else:
            merged.append(existing_block)
        previous_end = end
    merged.extend(generated_lines[previous_end:])
    return "".join(merged)


def convert_to_myst(body: str) -> str:
    """
    Convert GitHub markdown to MyST format.

    - PR URLs -> {pr}`NUMBER`
    - Issue URLs -> {issue}`NUMBER`
    - @mentions -> {user}`USERNAME`
    - Strips HTML comments
    - Ensures proper list spacing
    """
    lines = body.split("\n")
    result = []
    prev_bullet = False

    for line in lines:
        # Skip HTML comments
        if line.strip().startswith("<!--") and line.strip().endswith("-->"):
            continue

        # Convert URLs to extlinks
        line = re.sub(
            r"https://github\.com/ManimCommunity/manim/pull/(\d+)",
            r"{pr}`\1`",
            line,
        )
        line = re.sub(
            r"https://github\.com/ManimCommunity/manim/issues/(\d+)",
            r"{issue}`\1`",
            line,
        )
        line = re.sub(r"@([a-zA-Z0-9_-]+)", r"{user}`\1`", line)

        if line.startswith("**Full Changelog**:"):
            _, url = line.split(":", 1)
            url = url.strip().replace("vmain", "main")
            line = f"**Full Changelog**: [Compare view]({url})"

        # Handle list spacing
        is_bullet = line.strip().startswith("*") and not line.strip().startswith("**")
        if prev_bullet and not is_bullet and line.strip():
            result.append("")

        result.append(line)
        prev_bullet = is_bullet

    return "\n".join(result)


def format_changelog(
    version: str,
    body: str,
    date: str | None = None,
    title: str | None = None,
) -> str:
    """Create changelog file content with MyST frontmatter."""
    title = title or f"v{version}"
    body = convert_to_myst(body)
    date_section = f"Date\n: {date}\n" if date else ""

    return f"""---
short-title: {title}
description: Changelog for {title}
---

# {title}

{date_section}
{body}
"""


def convert_to_github(body: str) -> str:
    """Convert the MyST roles used by changelogs to GitHub Markdown."""
    body = re.sub(
        r"\{pr\}`(\d+)`",
        rf"https://github.com/{REPO}/pull/\1",
        body,
    )
    body = re.sub(
        r"\{issue\}`(\d+)`",
        rf"https://github.com/{REPO}/issues/\1",
        body,
    )
    body = re.sub(r"\{user\}`([a-zA-Z0-9_-]+)`", r"@\1", body)
    body = re.sub(r"\{class\}`\.?(.*?)`", r"`\1`", body)
    return body


def format_github_release_notes(version: str, content: str) -> str:
    """Create a GitHub release body from a documentation changelog."""
    content = re.sub(r"\A---\r?\n.*?\r?\n---\r?\n", "", content, flags=re.DOTALL)
    lines = content.splitlines()

    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and lines[0].startswith("# "):
        lines.pop(0)
    while lines and not lines[0].strip():
        lines.pop(0)
    if len(lines) >= 2 and lines[0] == "Date" and lines[1].startswith(": "):
        del lines[:2]
    while lines and not lines[0].strip():
        lines.pop(0)

    body = convert_to_github("\n".join(lines)).rstrip()
    changelog_url = (
        f"https://docs.manim.community/en/stable/changelog/{version}-changelog.html"
    )
    return f"See [our rendered changelog]({changelog_url}).\n\n{body}\n"


# =============================================================================
# File Operations
# =============================================================================


def get_existing_versions() -> set[str]:
    """Get versions that already have changelog files."""
    if not CHANGELOG_DIR.exists():
        return set()
    return {
        f.stem.replace("-changelog", "") for f in CHANGELOG_DIR.glob("*-changelog.*")
    }


def save_changelog(version: str, content: str) -> Path:
    """Save changelog and return filepath."""
    filepath = CHANGELOG_DIR / f"{version}-changelog.md"
    filepath.write_text(content)
    return filepath


def update_citation(version: str, date: str | None = None) -> Path:
    """Update CITATION.cff from template."""
    if not CITATION_TEMPLATE.exists():
        raise click.ClickException(f"Citation template not found: {CITATION_TEMPLATE}")

    date = date or datetime.now().strftime("%Y-%m-%d")
    version_tag = normalize_tag(version)

    content = CITATION_TEMPLATE.read_text()
    content = content.replace("<version>", version_tag)
    content = content.replace("<date_released>", date)

    CITATION_FILE.write_text(content)
    return CITATION_FILE


# =============================================================================
# CLI Commands
# =============================================================================


@click.group()
@click.option(
    "--dry-run", is_flag=True, help="Show what would be done without making changes"
)
@click.pass_context
def cli(ctx: click.Context, dry_run: bool) -> None:
    """Release management tools for Manim."""
    ctx.ensure_object(dict)
    ctx.obj["dry_run"] = dry_run


@cli.command()
@click.option("--base", required=True, help="Base tag for comparison (e.g., v0.19.0)")
@click.option(
    "--version", "version", required=True, help="New version number (e.g., 0.20.0)"
)
@click.option("--head", default="main", help="Head ref for comparison (default: main)")
@click.option("--title", help="Custom changelog title (default: vX.Y.Z)")
@click.option(
    "--include-extra",
    "extra_prs",
    type=click.IntRange(min=1),
    multiple=True,
    help="Include a PR omitted by GitHub's generated notes; may be repeated",
)
@click.option(
    "--update-citation",
    "also_update_citation",
    is_flag=True,
    help="Also update CITATION.cff",
)
@click.pass_context
def changelog(
    ctx: click.Context,
    base: str,
    version: str,
    head: str,
    title: str | None,
    extra_prs: tuple[int, ...],
    also_update_citation: bool,
) -> None:
    """Generate changelog for an upcoming release.

    Uses GitHub's release notes API with .github/release.yml categorization.
    """
    dry_run = ctx.obj["dry_run"]
    base_tag = normalize_tag(base)
    head_tag = normalize_tag(head) if head != "main" else normalize_tag(version)

    click.echo(f"Generating changelog for v{version}...")
    click.echo(f"  Comparing: {base_tag} → {head}")

    body = generate_release_notes(head_tag, base_tag)
    if extra_prs:
        click.echo(f"  Including extra PRs: {', '.join(f'#{pr}' for pr in extra_prs)}")
        body = include_extra_pull_requests(body, extra_prs)
    date = datetime.now().strftime("%B %d, %Y")
    content = format_changelog(version, body, date=date, title=title)
    filepath = CHANGELOG_DIR / f"{version}-changelog.md"

    if filepath.exists():
        click.echo("  Preserving existing PR note edits...")
        content = merge_changelog(content, filepath.read_text())

    if dry_run:
        click.echo()
        click.secho("[DRY RUN]", fg="yellow", bold=True)
        click.echo(f"  Would save: {filepath}")
        if also_update_citation:
            click.echo(f"  Would update: {CITATION_FILE}")
        click.echo()
        click.echo("--- Generated changelog ---")
        click.echo(content)
        return

    filepath = save_changelog(version, content)
    click.secho(f"  ✓ Saved: {filepath}", fg="green")

    if also_update_citation:
        citation_path = update_citation(version)
        click.secho(f"  ✓ Updated: {citation_path}", fg="green")

    click.echo()
    click.echo("Next steps:")
    click.echo("  • Review and edit the changelog as needed")
    click.echo("  • Update docs/source/changelog.rst to include the new file")


@cli.command("export-changelog")
@click.argument("version")
def export_changelog(version: str) -> None:
    """Export VERSION's changelog as GitHub-flavored release notes.

    The output is written to stdout for pasting or piping to another command.
    """
    version = version_from_tag(version)
    filepath = CHANGELOG_DIR / f"{version}-changelog.md"
    if not filepath.exists():
        raise click.ClickException(f"Changelog not found: {filepath}")

    click.echo(
        format_github_release_notes(version, filepath.read_text()),
        nl=False,
    )


@cli.command()
@click.option(
    "--version", "version", required=True, help="Version number (e.g., 0.20.0)"
)
@click.option("--date", help="Release date as YYYY-MM-DD (default: today)")
@click.pass_context
def citation(ctx: click.Context, version: str, date: str | None) -> None:
    """Update CITATION.cff for a release."""
    dry_run = ctx.obj["dry_run"]
    display_date = date or datetime.now().strftime("%Y-%m-%d")

    if dry_run:
        click.secho("[DRY RUN]", fg="yellow", bold=True)
        click.echo(f"  Would update: {CITATION_FILE}")
        click.echo(f"  Version: v{version}")
        click.echo(f"  Date: {display_date}")
        return

    filepath = update_citation(version, date)
    click.secho(f"✓ Updated: {filepath}", fg="green")
    click.echo(f"  Version: v{version}")
    click.echo(f"  Date: {display_date}")


@cli.command("fetch-releases")
@click.option("--tag", help="Fetch a specific release tag")
@click.option(
    "--min-version",
    default=DEFAULT_MIN_VERSION,
    help=f"Minimum version to fetch (default: {DEFAULT_MIN_VERSION})",
)
@click.option("--force", is_flag=True, help="Overwrite existing changelog files")
@click.pass_context
def fetch_releases(
    ctx: click.Context,
    tag: str | None,
    min_version: str,
    force: bool,
) -> None:
    """Fetch existing release changelogs from GitHub.

    Converts GitHub release notes to documentation-ready MyST markdown.
    """
    dry_run = ctx.obj["dry_run"]
    existing = get_existing_versions()

    # Single tag mode
    if tag:
        tag = normalize_tag(tag)
        version = version_from_tag(tag)

        if version in existing and not force:
            click.echo(
                f"Changelog for {version} already exists. Use --force to overwrite."
            )
            return

        if dry_run:
            click.secho("[DRY RUN]", fg="yellow", bold=True)
            click.echo(f"  Would fetch: {version}")
            return

        _fetch_single_release(tag, version)
        return

    # Batch mode
    click.echo(f"Existing versions: {', '.join(sorted(existing)) or '(none)'}")
    click.echo("Fetching release list...")

    tags = get_release_tags()
    click.echo(f"Found {len(tags)} releases")

    fetched = 0
    prev_tag = None

    for tag in reversed(tags):
        version = version_from_tag(tag)

        if not version_gte(version, min_version):
            prev_tag = tag
            continue

        if version in existing and not force:
            click.echo(f"  Skipping {version} (exists)")
            prev_tag = tag
            continue

        if dry_run:
            click.echo(f"  [DRY RUN] Would fetch {version}")
            fetched += 1
        else:
            if _fetch_single_release(tag, version, prev_tag):
                existing.add(version)
                fetched += 1

        prev_tag = tag

    click.echo()
    click.echo(f"Processed {fetched} changelog(s)")

    if fetched > 0 and not dry_run:
        click.echo()
        click.echo("Next steps:")
        click.echo("  • Update docs/source/changelog.rst to include new files")


def _fetch_single_release(tag: str, version: str, prev_tag: str | None = None) -> bool:
    """Fetch and save a single release changelog."""
    click.echo(f"  Fetching {version}...")

    body = get_release_body(tag)

    if not body and prev_tag:
        click.echo(f"    No body, generating from {prev_tag}...")
        try:
            body = generate_release_notes(tag, prev_tag)
        except click.ClickException:
            body = None

    if not body:
        click.secho(f"    ✗ Could not get release notes for {tag}", fg="red", err=True)
        return False

    date = get_release_date(tag)
    content = format_changelog(version, body, date=date)

    filepath = save_changelog(version, content)
    click.secho(f"    ✓ Saved: {filepath}", fg="green")
    return True


# =============================================================================
# Entry Point
# =============================================================================


def main() -> None:
    """Entry point."""
    cli()


if __name__ == "__main__":
    sys.exit(main() or 0)
