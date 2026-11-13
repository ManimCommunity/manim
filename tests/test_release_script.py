import json
import subprocess

import click
import pytest
from packaging.version import Version

from scripts import release
from scripts.release import (
    format_github_release_notes,
    include_extra_pull_requests,
    merge_changelog,
)


def test_merge_changelog_preserves_edits_but_uses_generated_categories():
    existing = """## What's Changed
### Bug Fixes 🐛
* Carefully copy-edited note by {user}`alice` in {pr}`1`

  A manually added description.
* Existing note by {user}`bob` in {pr}`2`
* Obsolete note by {user}`dave` in {pr}`4`

## New Contributors
* {user}`alice` made their first contribution in {pr}`1`
"""
    generated = """## What's Changed
### Enhancements 🚀
* Original generated note by {user}`alice` in {pr}`1`
* Newly merged PR by {user}`carol` in {pr}`3`

### Bug Fixes 🐛
* Regenerated note by {user}`bob` in {pr}`2`

## New Contributors
* {user}`alice` made their first contribution in {pr}`1`
* {user}`carol` made their first contribution in {pr}`3`
"""

    merged = merge_changelog(generated, existing)

    assert "Carefully copy-edited note" in merged
    assert "A manually added description." in merged
    assert "Original generated note" not in merged
    assert "Newly merged PR" in merged
    assert "Obsolete note" not in merged
    assert merged.index("### Enhancements") < merged.index("Carefully copy-edited note")
    assert merged.index("Carefully copy-edited note") < merged.index("### Bug Fixes")


def test_merge_changelog_keeps_release_and_contributor_entries_separate():
    existing = """## What's Changed
### Enhancements
* Edited release note in {pr}`10`

## New Contributors
* Edited welcome for the author of {pr}`10`

**Full Changelog**: existing compare link
"""
    generated = """## What's Changed
### Documentation
* Generated release note in {pr}`10`

## New Contributors
* Generated welcome for the author of {pr}`10`
* Welcome for the author of {pr}`11`

**Full Changelog**: generated compare link
"""

    merged = merge_changelog(generated, existing)

    assert "Edited release note" in merged
    assert "Edited welcome" in merged
    assert "Generated release note" not in merged
    assert "Generated welcome" not in merged
    assert merged.count("{pr}`10`") == 2
    assert "Welcome for the author of {pr}`11`" in merged
    assert merged.count("**Full Changelog**") == 1
    assert "generated compare link" in merged


def test_include_extra_pull_requests_adds_other_changes_category(monkeypatch):
    body = """## What's Changed
### Bug Fixes
* Fix a bug by @alice in https://github.com/ManimCommunity/manim/pull/1

## New Contributors
* @alice made their first contribution in https://github.com/ManimCommunity/manim/pull/1

**Full Changelog**: compare link
"""
    monkeypatch.setattr(
        release,
        "get_pull_request_release_note",
        lambda number: (
            f"* Prepare a release by @bob in "
            f"https://github.com/ManimCommunity/manim/pull/{number}"
        ),
    )

    result = include_extra_pull_requests(body, [99])

    assert "### Other Changes\n* Prepare a release by @bob" in result
    assert result.index("### Other Changes") < result.index("## New Contributors")


def test_include_extra_pull_requests_skips_existing_pr(monkeypatch):
    body = """## What's Changed
### Other Changes
* Existing entry in https://github.com/ManimCommunity/manim/pull/99
"""

    def fail_if_called(number):
        raise AssertionError(f"unexpected fetch for #{number}")

    monkeypatch.setattr(release, "get_pull_request_release_note", fail_if_called)

    assert include_extra_pull_requests(body, [99]) == body


def test_merge_changelog_without_matching_entries_is_unchanged():
    generated = """## What's Changed
### New Features
* A new entry in {pr}`42`
"""

    assert merge_changelog(generated, "") == generated


def test_format_github_release_notes_removes_docs_metadata_and_converts_roles():
    changelog = """---
short-title: v1.2.3
description: Changelog for v1.2.3
---

# v1.2.3

Date
: January 02, 2026

## What's Changed
* Improve {class}`.Example` by {user}`alice` in {pr}`123`
* Fix {issue}`456`

See the {doc}`configuration guide </guides/configuration>` and {doc}`/guides/deep_dive`.

**Full Changelog**: [Compare view](https://example.com/compare)
"""

    result = format_github_release_notes("1.2.3", changelog)

    assert result.startswith(
        "See [our rendered changelog](https://docs.manim.community/en/stable/"
        "changelog/1.2.3-changelog.html).\n\n## What's Changed"
    )
    assert "frontmatter" not in result
    assert "# v1.2.3" not in result
    assert "January 02, 2026" not in result
    assert "`Example` by @alice" in result
    assert "https://github.com/manimcommunity/manim/pull/123" in result
    assert "https://github.com/manimcommunity/manim/issues/456" in result
    assert (
        "[configuration guide]"
        "(https://docs.manim.community/en/stable/guides/configuration.html)"
    ) in result
    assert (
        "[guides/deep_dive]"
        "(https://docs.manim.community/en/stable/guides/deep_dive.html)"
    ) in result
    assert result.endswith("\n")


GENERATED_NOTES = """## What's Changed
### New Features ✨
* Draw with wgpu by @alice in https://github.com/ManimCommunity/manim/pull/7

## New Contributors
* @alice made their first contribution in https://github.com/ManimCommunity/manim/pull/7

**Full Changelog**: https://github.com/ManimCommunity/manim/compare/v2.0.0a1...v2.0.0a2
"""


def test_preview_changelog_lists_prereleases_newest_first():
    first = release.format_preview_section("2.0.0a1", GENERATED_NOTES, "May 1, 2030")
    content = release.update_preview_changelog("", "2.0.0a1", first)
    second = release.format_preview_section("2.0.0a2", GENERATED_NOTES, "June 1, 2030")
    content = release.update_preview_changelog(content, "2.0.0a2", second)

    assert content.startswith(release.PREVIEW_INTRO.rstrip())
    assert content.index("## v2.0.0a2") < content.index("## v2.0.0a1")
    # The notes' own headings are nested in the prerelease's section.
    assert "\n### What's Changed\n#### New Features ✨\n" in content
    assert "\n### New Contributors\n" in content
    assert "\n## What's Changed" not in content
    assert "{pr}`7`" in content
    assert "{user}`alice`" in content


def test_regenerating_a_preview_section_keeps_its_place_and_edits():
    content = release.update_preview_changelog(
        "", "2.0.0a1", release.format_preview_section("2.0.0a1", GENERATED_NOTES)
    )
    content = release.update_preview_changelog(
        content,
        "2.0.0a2",
        release.format_preview_section("2.0.0a2", GENERATED_NOTES),
    )
    content = content.replace(
        "* Draw with wgpu by {user}`alice` in {pr}`7`",
        "* Draw frames with wgpu (experimental) by {user}`alice` in {pr}`7`",
        1,
    )

    regenerated = release.update_preview_changelog(
        content,
        "2.0.0a2",
        release.format_preview_section("2.0.0a2", GENERATED_NOTES),
    )

    assert regenerated.count("## v2.0.0a2") == 1
    assert regenerated.index("## v2.0.0a2") < regenerated.index("## v2.0.0a1")
    _, sections = release.split_preview_changelog(regenerated)
    assert "Draw frames with wgpu (experimental)" in sections["2.0.0a2"]
    assert "Draw frames with wgpu" not in sections["2.0.0a1"]


def test_prerelease_notes_are_exported_from_their_preview_section():
    content = release.update_preview_changelog(
        "",
        "2.0.0a1",
        release.format_preview_section("2.0.0a1", GENERATED_NOTES, "May 1, 2030"),
    )
    content = release.update_preview_changelog(
        content,
        "2.0.0a2",
        release.format_preview_section(
            "2.0.0a2",
            GENERATED_NOTES.replace("Draw with wgpu", "Add lights"),
            "June 1, 2030",
        ),
    )

    notes = release.format_github_prerelease_notes("2.0.0a1", content)

    assert notes.startswith("This is a preview prerelease.")
    assert "pip install manim==2.0.0a1" in notes
    assert "https://docs.manim.community/en/preview/changelog/preview.html" in notes
    assert "\n## What's Changed\n### New Features ✨\n" in notes
    assert "Draw with wgpu by @alice" in notes
    assert "Add lights" not in notes
    assert "May 1, 2030" not in notes


def test_exporting_a_prerelease_without_a_section_fails():
    with pytest.raises(click.ClickException, match="v2.0.0a3"):
        release.format_github_prerelease_notes("2.0.0a3", release.PREVIEW_INTRO)


@pytest.mark.parametrize(
    ("tag", "version", "prerelease", "branches", "problem"),
    [
        ("v1.4.2", "1.4.2", False, ["main"], None),
        ("v2.0.0a1", "2.0.0a1", True, ["preview/v2.0.x"], None),
        ("v2.0.0rc1", "2.0.0rc1", True, ["main", "preview/v2.0.x"], None),
        ("v2.0.0a1", "2.0.0a2", True, ["preview/v2.0.x"], "is not v2.0.0a2"),
        ("v2.0.0a1", "2.0.0a1", False, ["preview/v2.0.x"], "not marked"),
        ("v1.4.2", "1.4.2", True, ["main"], "is not a prerelease"),
        ("v2.0.0a1", "2.0.0a1", True, ["main"], "not on a preview/* branch"),
        ("v2.0.0", "2.0.0", False, ["preview/v2.0.x"], "not on main"),
    ],
)
def test_release_checks(tag, version, prerelease, branches, problem):
    problems = release.release_problems(
        tag, Version(version), prerelease=prerelease, branches=branches
    )
    if problem is None:
        assert problems == []
    else:
        assert len(problems) == 1
        assert problem in problems[0]


def test_versions_compare_by_pep_440():
    assert release.version_gte("v2.0.0a1", "1.4.2")
    assert not release.version_gte("2.0.0a1", "2.0.0")
    assert release.version_gte("1.3.0.post0", "1.3.0")
    with pytest.raises(click.ClickException):
        release.parse_version("main")


def test_prereleases_are_not_fetched_as_release_changelogs(monkeypatch):
    listed = [
        {"tagName": "v2.0.0a1", "isPrerelease": True},
        {"tagName": "v1.4.2", "isPrerelease": False},
    ]
    monkeypatch.setattr(
        release,
        "run_gh",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args, 0, stdout=json.dumps(listed)
        ),
    )

    assert release.get_release_tags() == ["v1.4.2"]


def test_changelog_notes_run_up_to_the_release_branch(monkeypatch):
    calls = []

    def run_gh(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="notes")

    monkeypatch.setattr(release, "run_gh", run_gh)

    assert release.generate_release_notes("v2.0.0a2", "v2.0.0a1", "preview/v2.0.x")
    assert "target_commitish=preview/v2.0.x" in calls[0]
    assert "tag_name=v2.0.0a2" in calls[0]
