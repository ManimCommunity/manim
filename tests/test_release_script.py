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
    assert result.endswith("\n")
