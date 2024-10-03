# Title: Add weighted edges and self loop edges to graphs
<!-- Thank you for contributing to Manim! Learn more about the process in our contributing guidelines: https://docs.manim.community/en/latest/contributing.html -->

## Overview: What does this pull request change?
<!-- If there is more information than the PR title that should be added to our release changelog, add it in the following changelog section. This is optional, but recommended for larger pull requests. -->
This pull request addresses the issue #3153. It makes possible to create self loop edges that are correctly rendered using
curved lines as edges. It also adds the possibility to set weights (or any other label) to the edges.
<!--changelog-start-->

<!--changelog-end-->

## Motivation and Explanation: Why and how do your changes improve the library?
<!-- Optional for bugfixes, small enhancements, and documentation-related PRs. Otherwise, please give a short reasoning for your changes. -->
As it is explained in the linked issue, these changes make graphs ready for state machines representation.

## Links to added or changed documentation pages
<!-- Please add links to the affected documentation pages (edit the description after opening the PR). The link to the documentation for your PR is https://manimce--####.org.readthedocs.build/en/####/, where #### represents the PR number. -->


## Further Information and Comments
<!-- If applicable, put further comments for the reviewers here. -->
We may need to modify the label properties of edges. For instance, the background color is set to BLACK for now but it should get the scene's background color.
Yet, I have not found any clean way to do it for now.
Moreover, the two added methods may be moved to another location. Though the objective here was to make it as general as possible without creating new classes
that would support these new features. This way, it can be used in every future graphs.
The class used to create edges labels is customizable by giving a `label_type` attribute in the `edge_config` dictionary. The only prerequisite is that
this custom class can be initialized with a `label` and a `color` keyword.
Similarly, a label can be added to a graph edge by either passing a `weights` dictionary when creating the graph, or by adding a `label` keyword in the
`edge_config` dictionary.



<!-- Thank you again for contributing! Do not modify the lines below, they are for reviewers. -->
## Reviewer Checklist
- [ ] The PR title is descriptive enough for the changelog, and the PR is labeled correctly
- [ ] If applicable: newly added non-private functions and classes have a docstring including a short summary and a PARAMETERS section
- [ ] If applicable: newly added functions and classes are tested
