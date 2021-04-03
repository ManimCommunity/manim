#!/usr/bin/env python
"""Script to generate contributor and pull request lists.

This script generates contributor and pull request lists for release
changelogs using Github v3 protocol. Use requires an authentication token in
order to have sufficient bandwidth, you can get one following the directions at
`<https://help.github.com/articles/creating-an-access-token-for-command-line-use/>_
Don't add any scope, as the default is read access to public information. The
token may be stored in an environment variable as you only get one chance to
see it.

Usage::

    $ ./scripts/dev_changelog.py [OPTIONS] TOKEN PRIOR TAG [ADDITIONAL]...

The output is utf8 rst.

Dependencies
------------

- gitpython
- pygithub

Examples
--------

From a bash command line with $GITHUB environment variable as the GitHub token::

    $ ./scripts/dev_changelog.py $GITHUB v0.3.0 v0.4.0

This would generate 0.4.0-changelog.rst file and place it automatically under
docs/source/changelog/.

As another example, you may also run include PRs that have been excluded by
providing a space separated list of ticket numbers after TAG::

    $ ./scripts/dev_changelog.py $GITHUB v0.3.0 v0.4.0 1911 1234 1492 ...


Note
----

This script was taken from Numpy under the terms of BSD-3-Clause license.
"""

import datetime
import re
from collections import defaultdict
from pathlib import Path
from textwrap import dedent, indent

import click
from git import Repo
from github import Github
from tqdm import tqdm

from manim.constants import CONTEXT_SETTINGS, EPILOG

this_repo = Repo(str(Path(__file__).resolve().parent.parent))

PR_LABELS = {
    "breaking changes": "Breaking changes",
    "highlight": "Highlights",
    "deprecation": "Deprecated classes and functions",
    "new feature": "New features",
    "enhancement": "Enhancements",
    "bug": "Fixed bugs",
    "documentation": "Documentation-related changes",
    "testing": "Changes concerning the testing system",
    "infrastructure": "Changes to our development infrastructure",
    "maintenance": "Code quality improvements and similar refactors",
    "reverts": "Changes that needed to be reverted again",
    "unlabeled": "Unclassified changes",
}


def get_authors_and_reviewers(lst, cur, github_repo, pr_nums):
    pat = r"^.*\t(.*)$"

    # authors, in current release and previous to current release.
    cur = set(re.findall(pat, this_repo.git.shortlog("-s", f"{lst}..{cur}"), re.M))
    pre = set(re.findall(pat, this_repo.git.shortlog("-s", lst), re.M))

    # Append '+' to new authors.
    authors = [s + " +" for s in cur - pre] + [s for s in cur & pre]
    authors.sort()

    reviewers = []
    for num in tqdm(pr_nums, desc="Fetching reviewer comments"):
        pr = github_repo.get_pull(num)
        reviewers.extend(rev.user.name for rev in pr.get_reviews())
    reviewers = sorted(set(rev for rev in reviewers if rev is not None))

    return {"authors": authors, "reviewers": reviewers}


def get_pr_nums(lst, cur):
    print("Getting PR Numbers:")
    prnums = []

    # From regular merges
    merges = this_repo.git.log("--oneline", "--merges", f"{lst}..{cur}")
    issues = re.findall(r".*\(\#(\d+)\)", merges)
    prnums.extend(int(s) for s in issues)

    # From fast forward squash-merges
    commits = this_repo.git.log(
        "--oneline", "--no-merges", "--first-parent", f"{lst}..{cur}"
    )
    issues = re.findall(r"^.*\(\#(\d+)\)$", commits, re.M)
    prnums.extend(int(s) for s in issues)

    print(prnums)
    return prnums


def sort_by_labels(github_repo, pr_nums):
    """Sorts PR into groups based on labels.

    This implementation sorts based on importance into a singular group. If a
    PR uses multiple labels, it is sorted under one label.

    """
    pr_by_labels = defaultdict(list)
    for num in tqdm(pr_nums, desc="Sorting by labels"):
        pr = github_repo.get_pull(num)
        pr_labels = [label.name for label in pr.labels]
        for label in PR_LABELS.keys():
            if label in pr_labels:
                pr_by_labels[label].append(pr)
                break  # ensure that PR is only added in one category
        else:
            pr_by_labels["unlabeled"].append(pr)

    return pr_by_labels


def get_summary(body):
    pattern = '<!--changelog-start-->([^"]*)<!--changelog-end-->'
    has_changelog_pattern = re.search(pattern, body)
    if has_changelog_pattern:

        return has_changelog_pattern.group()[22:-21].strip()


@click.command(
    context_settings=CONTEXT_SETTINGS,
    epilog=EPILOG,
)
@click.argument("token")
@click.argument("prior")
@click.argument("tag")
@click.argument(
    "additional",
    nargs=-1,
    required=False,
)
@click.option(
    "-o", "--outfile", type=str, help="Path and file name of the changelog output."
)
def main(token, prior, tag, additional, outfile):
    """Generate Changelog/List of contributors/PRs for release.

    TOKEN is your GitHub Personal Access Token.

    PRIOR is the tag/commit SHA of the previous release.

    TAG is the tag of the new release.

    ADDITIONAL includes additional PR(s) that have not been recognized automatically.
    """

    lst_release, cur_release = prior, tag

    github = Github(token)
    github_repo = github.get_repo("ManimCommunity/manim")

    pr_nums = get_pr_nums(lst_release, cur_release)
    if additional:
        print(f"Adding {additional} to the mix!")
        pr_nums = pr_nums + list(additional)

    # document authors
    contributors = get_authors_and_reviewers(
        lst_release, cur_release, github_repo, pr_nums
    )
    authors = contributors["authors"]
    reviewers = contributors["reviewers"]

    if not outfile:
        outfile = (
            Path(__file__).resolve().parent.parent / "docs" / "source" / "changelog"
        )
        outfile = outfile / f"{tag[1:] if tag.startswith('v') else tag}-changelog.rst"
    else:
        outfile = Path(outfile).resolve()

    with outfile.open("w", encoding="utf8") as f:
        f.write("*" * len(tag) + "\n")
        f.write(f"{tag}\n")
        f.write("*" * len(tag) + "\n\n")

        today = datetime.date.today()
        f.write(f":Date: {today.strftime('%B %d, %Y')}\n\n")

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

        pr_by_labels = sort_by_labels(github_repo, pr_nums)
        for label in PR_LABELS.keys():
            pr_of_label = pr_by_labels[label]

            if pr_of_label:
                heading = PR_LABELS[label]
                f.write(f"{heading}\n")
                f.write("-" * len(heading) + "\n\n")

                for PR in pr_by_labels[label]:
                    num = PR.number
                    url = PR.html_url
                    title = PR.title
                    label = PR.labels
                    f.write(f"* `#{num} <{url}>`__: {title}\n")
                    overview = get_summary(PR.body)
                    if overview:
                        f.write(indent(f"{overview}\n\n", "   "))
                    else:
                        f.write("\n\n")

    print(f"Wrote changelog to: {outfile}")


if __name__ == "__main__":
    main()
