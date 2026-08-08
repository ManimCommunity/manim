from scripts.release import merge_changelog


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
"""
    generated = """## What's Changed
### Documentation
* Generated release note in {pr}`10`

## New Contributors
* Generated welcome for the author of {pr}`10`
"""

    merged = merge_changelog(generated, existing)

    assert "Edited release note" in merged
    assert "Edited welcome" in merged
    assert "Generated release note" not in merged
    assert "Generated welcome" not in merged
    assert merged.count("{pr}`10`") == 2


def test_merge_changelog_without_matching_entries_is_unchanged():
    generated = """## What's Changed
### New Features
* A new entry in {pr}`42`
"""

    assert merge_changelog(generated, "") == generated
