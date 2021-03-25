#!/usr/bin/env python
"""
Script to generate contributor and pull request lists
This script generates contributor and pull request lists for release
changelogs using Github v3 protocol. Use requires an authentication token in
order to have sufficient bandwidth, you can get one following the directions at
`<https://help.github.com/articles/creating-an-access-token-for-command-line-use/>_
Don't add any scope, as the default is read access to public information. The
token may be stored in an environment variable as you only get one chance to
see it.
Usage::
    $ ./scripts/dev_changelog.py <token> <revision range>
The output is utf8 rst.
Dependencies
------------
- gitpython
- pygithub

Examples
--------
From the bash command line with $GITHUB token::
    $ ./scripts/dev_changelog.py $GITHUB v0.3.0..v0.4.0 > 1.14.0-changelog.rst
Note
----
This script was taken from Numpy under the terms of BSD-3-Clause license.
"""
import re
import sys
from collections import defaultdict
from textwrap import dedent
from pathlib import Path
import git
from tqdm import tqdm
from git import Repo
from github import Github

if sys.version_info[:2] < (3, 6):
    raise RuntimeError("Python version must be >= 3.6")

this_repo = Repo(str(Path(__file__).resolve().parent.parent))


def get_authors(revision_range):
    pat = "^.*\\t(.*)$"
    lst_release, cur_release = [r.strip() for r in revision_range.split("..")]

    # authors, in current release and previous to current release.
    cur = set(re.findall(pat, this_repo.git.shortlog("-s", revision_range), re.M))
    pre = set(re.findall(pat, this_repo.git.shortlog("-s", lst_release), re.M))

    # Append '+' to new authors.
    authors = [s + " +" for s in cur - pre] + [s for s in cur & pre]
    authors.sort()
    return authors


def get_pr_nums(repo, revision_range):
    print("Getting PR Numbers:")
    prnums = []

    # From regular merges
    merges = this_repo.git.log("--oneline", "--merges", revision_range)
    issues = re.findall("Merge pull request \\#(\\d*)", merges)
    prnums.extend(int(s) for s in issues)

    # From fast forward squash-merges
    commits = this_repo.git.log(
        "--oneline", "--no-merges", "--first-parent", revision_range
    )
    issues = re.findall("^.*\\(\\#(\\d+)\\)$", commits, re.M)
    prnums.extend(int(s) for s in issues)

    print(prnums)
    return prnums


def sort_by_labels(github_repo, pr_nums):
    """Sorts PR into groups based on labels.

    This implementation sorts based on importance into a singular group. If a
    PR uses multiple labels, it is sorted under one label.

    The importance order (for the end-user):
    - breaking changes
    - highlight
    - feature
    - enhancement
    - bug
    - documentation
    - testing
    - infrastructure
    - unlabeled
    """
    pr_by_labels = defaultdict(list)
    for num in tqdm(pr_nums, desc="Sorting by labels"):
        pr = github_repo.get_pull(num)
        labels = [label.name for label in pr.labels]
        if "breaking changes" in labels:
            pr_by_labels["breaking changes"].append(pr)
        elif "highlight" in labels:
            pr_by_labels["highlight"].append(pr)
        elif "new feature" in labels:
            pr_by_labels["new feature"].append(pr)
        elif "enhancement" in labels:
            pr_by_labels["enhancement"].append(pr)
        elif "bug" in labels:
            pr_by_labels["bug"].append(pr)
        elif "documentation" in labels:
            pr_by_labels["documentation"].append(pr)
        elif "testing" in labels:
            pr_by_labels["testing"].append(pr)
        elif "infrastructure" in labels:
            pr_by_labels["infrastructure"].append(pr)
        else:  # PR doesn't have label :( Create one!
            pr_by_labels["unlabeled"].append(pr)

    return pr_by_labels


def main(token, revision_range):
    lst_release, cur_release = [r.strip() for r in revision_range.split("..")]

    github = Github(token)
    github_repo = github.get_repo("ManimCommunity/manim")

    # document authors
    authors = get_authors(revision_range)

    # TODO: Decide where to place changelog file
    changelog_file = (
        Path(__file__).resolve().parent.parent / "docs" / "source" / "changelog.rst"
    )

    with changelog_file.open("w", encoding="utf8") as f:
        heading = "Contributors"
        f.write(f"{heading}\n")
        f.write("=" * len(heading) + "\n\n")
        f.write(
            dedent(
                f"""\
            A total of {len(authors)} people contributed to this release.
            People with a '+' by their names contributed a patch for the first time.\n
            """
            )
        )

        for author in authors:
            f.write("* " + author + "\n")

        # document pull requests
        pr_nums = get_pr_nums(github_repo, revision_range)

        heading = "Pull requests merged"
        f.write("\n")
        f.write(heading + "\n")
        f.write("=" * len(heading) + "\n\n")
        f.write(
            f"A total of {len(pr_nums)} pull requests were merged for this release.\n\n"
        )

        labels = [
            "breaking changes",
            "highlight",
            "new feature",
            "enhancement",
            "bug",
            "documentation",
            "testing",
            "infrastructure",
            "unlabeled",
        ]
        pr_by_labels = sort_by_labels(github_repo, pr_nums)
        for label in labels:
            f.write(f"{label.capitalize()}\n")
            f.write("-" * len(label) + "\n\n")

            for PR in pr_by_labels[label]:
                num = PR.number
                url = PR.html_url
                title = PR.title
                label = PR.labels
                f.write(f"* `#{num} <{url}>`__: {title}\n")

    print(f"Wrote changelog to: {changelog_file}")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Generate author/pr lists for release")
    parser.add_argument("token", help="github access token")
    parser.add_argument("revision_range", help="<revision>..<revision>")
    args = parser.parse_args()
    main(args.token, args.revision_range)
