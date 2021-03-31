#!/usr/bin/env python
"""Script to generate contributor and pull request lists
This script generates contributor and pull request lists for release
changelogs using Github v3 protocol. Use requires an authentication token in
order to have sufficient bandwidth, you can get one following the directions at
`<https://help.github.com/articles/creating-an-access-token-for-command-line-use/>_
Don't add any scope, as the default is read access to public information. The
token may be stored in an environment variable as you only get one chance to
see it.

Usage::
    $ ./scripts/dev_changelog.py <token> <revision range> <output_file>

The output is utf8 rst.

Dependencies
------------
- gitpython
- pygithub

Examples
--------
From the bash command line with $GITHUB token::
    $ ./scripts/dev_changelog.py $GITHUB v0.3.0..v0.4.0 -o 0.4.0-changelog.rst

Note
----
This script was taken from Numpy under the terms of BSD-3-Clause license.
"""
import re
import sys
from collections import defaultdict
from textwrap import dedent, indent
from pathlib import Path
import git
from tqdm import tqdm
from git import Repo
from github import Github


this_repo = Repo(str(Path(__file__).resolve().parent.parent))


def get_authors_and_reviewers(revision_range, github_repo, pr_nums):
    pat = r"^.*\t(.*)$"
    lst_release, cur_release = [r.strip() for r in revision_range.split("..")]

    # authors, in current release and previous to current release.
    cur = set(re.findall(pat, this_repo.git.shortlog("-s", revision_range), re.M))
    pre = set(re.findall(pat, this_repo.git.shortlog("-s", lst_release), re.M))

    # Append '+' to new authors.
    authors = [s + " +" for s in cur - pre] + [s for s in cur & pre]
    authors.sort()

    reviewers = []
    for num in tqdm(pr_nums, desc="Fetching reviewer comments"):
        pr = github_repo.get_pull(num)
        reviewers.extend(rev.user.name for rev in pr.get_reviews())
    reviewers = sorted(set(rev for rev in reviewers if rev is not None))

    return {"authors": authors, "reviewers": reviewers}


def get_pr_nums(revision_range):
    print("Getting PR Numbers:")
    prnums = []

    # From regular merges
    merges = this_repo.git.log("--oneline", "--merges", revision_range)
    issues = re.findall(r".*\(\#(\d+)\)", merges)
    prnums.extend(int(s) for s in issues)

    # From fast forward squash-merges
    commits = this_repo.git.log(
        "--oneline", "--no-merges", "--first-parent", revision_range
    )
    issues = re.findall(r"^.*\(\#(\d+)\)$", commits, re.M)
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
        # TODO: Make use of label names directly from main
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
        elif "deprecation" in labels:
            pr_by_labels["deprecation"].append(pr)
        elif "documentation" in labels:
            pr_by_labels["documentation"].append(pr)
        elif "release" in labels:
            pr_by_labels["release"].append(pr)
        elif "testing" in labels:
            pr_by_labels["testing"].append(pr)
        elif "infrastructure" in labels:
            pr_by_labels["infrastructure"].append(pr)
        elif "maintenance":
            pr_by_labels["maintenance"].append(pr)
        elif "style" in labels:
            pr_by_labels["style"].append(pr)
        else:  # PR doesn't have label :( Create one!
            pr_by_labels["unlabeled"].append(pr)

    return pr_by_labels


def get_summary(body):
    pattern = '<!--changelog-start-->([^"]*)<!--changelog-end-->'
    has_changelog_pattern = re.search(pattern, body)
    if has_changelog_pattern:

        return has_changelog_pattern.group()[22:-21].strip()


def main(token, revision_range, outfile=None):
    lst_release, cur_release = [r.strip() for r in revision_range.split("..")]

    github = Github(token)
    github_repo = github.get_repo("ManimCommunity/manim")

    pr_nums = get_pr_nums(revision_range)

    # document authors
    contributors = get_authors_and_reviewers(revision_range, github_repo, pr_nums)
    authors = contributors["authors"]
    reviewers = contributors["reviewers"]

    if not outfile:
        outfile = (
            Path(__file__).resolve().parent.parent / "docs" / "source" / "changelog"
        )
        outfile = outfile / f"{cur_release[1:]}-changelog.rst"
    else:
        outfile = Path(outfile).resolve()

    with outfile.open("w", encoding="utf8") as f:
        heading = "Contributors"
        f.write(f"{heading}\n")
        f.write("=" * len(heading) + "\n\n")
        f.write(
            dedent(
                f"""\
                A total of {len(set(authors).union(set(reviewers)))} people contributed to this
                release. People with a '+' by their names authored a patch for the first
                time.\n
                """
            )
        )

        for author in authors:
            f.write(f"* {author}\n")

        f.write("\n")
        f.write(
            dedent(
                """
                The patches included in this release have been reviewed by
                the following contributors.\n
                """
            )
        )

        for reviewer in reviewers:
            f.write(f"* {reviewer}\n")

        # document pull requests
        heading = "Pull requests merged"
        f.write("\n")
        f.write(heading + "\n")
        f.write("=" * len(heading) + "\n\n")
        f.write(
            f"A total of {len(pr_nums)} pull requests were merged for this release.\n\n"
        )

        # TODO: Use labels list in sort_by_labels, simplify logic
        labels = [
            "breaking changes",
            "highlight",
            "new feature",
            "enhancement",
            "bug",
            "deprecation",
            "documentation",
            "release",
            "testing",
            "infrastructure",
            "maintenance",
            "style",
            "unlabeled",
        ]
        pr_by_labels = sort_by_labels(github_repo, pr_nums)
        for label in labels:
            pr_of_label = pr_by_labels[label]

            if pr_of_label:
                f.write(f"{label.capitalize()}\n")
                f.write("-" * len(label) + "\n\n")

                for PR in pr_by_labels[label]:
                    num = PR.number
                    url = PR.html_url
                    title = PR.title
                    label = PR.labels
                    f.write(f"* `#{num} <{url}>`__: {title}\n")
                    overview = get_summary(PR.body)
                    if overview:
                        f.write(indent(f"{overview}\n", "   "))
                    else:
                        f.write("\n")

    print(f"Wrote changelog to: {outfile}")


if __name__ == "__main__":
    from argparse import ArgumentParser, FileType

    parser = ArgumentParser(description="Generate author/pr lists for release")
    parser.add_argument("token", help="github access token")
    parser.add_argument("revision_range", help="<revision>..<revision>")
    parser.add_argument(
        "-o", "--outfile", type=str, help="path and file name of the changelog output"
    )
    args = parser.parse_args()
    main(args.token, args.revision_range, args.outfile)
