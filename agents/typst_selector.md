# Design: Sub-Expression Selection for `Typst` / `TypstMath`

## Problem Statement

Users need to interact with individual parts of a Typst-rendered expression:
color a variable, animate the numerator of a fraction, morph one sub-expression
into another, etc. The `MathTex` class solves this with:

1. **`{{ ... }}` double-brace notation** — splits the TeX string into named
   submobject groups at compile time.
2. **`substrings_to_isolate` / `get_part_by_tex`** — identifies submobjects
   whose TeX source matches a given string.

Both mechanisms ultimately rely on injecting `\special{dvisvgm:raw <g id='...'>}`
markers into the LaTeX source so that the resulting SVG contains `<g>` elements
with known `id` attributes, which SVGMobject's parser maps to `VGroup`
sub-trees via `id_to_vgroup_dict`.

We need an analogous mechanism for Typst.

## Key Discovery: `data-typst-label` in SVG Output

Typst's SVG renderer (`typst-svg` crate) already emits a `data-typst-label`
attribute on `<g>` elements whenever a `GroupItem` (hard frame) carries a
label. The relevant code path:

```rust
// typst-svg/src/lib.rs — render_group()
if let Some(label) = group.label {
    svg.init().attr("data-typst-label", label.resolve());
}
```

A **hard frame** is created by the `box` element (and `block`, etc.). Crucially,
`box` can be used inline inside math mode, and labels can be attached to it.

### Proof of Concept

The following Typst helper wraps content in a labeled `box`:

```typst
#let grp(lbl, body) = [#box(body) #label(lbl)]
```

When used in math:

```typst
$ #grp("numer", $a + b$) / #grp("denom", $c - d$) = #grp("result", $x$) $
```

The compiled SVG contains:

```xml
<g class="typst-group" ... data-typst-label="numer">
  <!-- glyphs for a + b -->
</g>
<g class="typst-group" ... data-typst-label="denom">
  <!-- glyphs for c - d -->
</g>
<g class="typst-group" ... data-typst-label="result">
  <!-- glyph for x -->
</g>
```

**Nesting works.** A `grp` wrapping a fraction that itself contains `grp`-ed
sub-parts produces nested `data-typst-label` groups:

```typst
$ #grp("whole-frac", $frac(#grp("numer", $a + b$), #grp("denom", $c - d$))$) $
```

SVG output:

```xml
<g ... data-typst-label="whole-frac">
  <g ... data-typst-label="numer"> ... </g>
  <g ... data-typst-label="denom"> ... </g>
  <path class="typst-shape" ... />  <!-- fraction bar -->
</g>
```

### SVG Parser Compatibility

Manim uses `svgelements` to parse SVGs. The library preserves
`data-typst-label` in the `values` dictionary of `Group` objects, and it
propagates to child elements. Manim's `SVGMobject.get_mobjects_from()` already
iterates over groups and builds `id_to_vgroup_dict` keyed by the `id` attribute.
Extending this to also key by `data-typst-label` is straightforward.

## Proposed Interface

### 1. Explicit Groups via `{{ ... }}` Notation (Compile-Time)

Mirror the `MathTex` double-brace convention. Users write:

```python
eq = TypstMath("{{ a + b }} / {{ c - d }} = {{ x }}")
```

The pre-processor splits on `{{ ... }}` (reusing the same whitespace-guard
rules as `MathTex._split_double_braces`) and wraps each group in a labeled
`box` call:

```typst
$ #box[$a + b$] <_grp-0> / #box[$c - d$] <_grp-1> = #box[$x$] <_grp-2> $
```

Each group gets an auto-generated label (`_grp-0`, `_grp-1`, ...).
The `data-typst-label` attributes then appear in the SVG, and
`SVGMobject.get_mobjects_from()` can map them to `VGroup` entries in
`label_to_vgroup_dict` (or reuse `id_to_vgroup_dict`).

These groups become sub-mobjects of the `TypstMath` instance, accessible by
index:

```python
eq[0]  # VGroup for "a + b"
eq[1]  # VGroup for "c - d"
eq[2]  # VGroup for "x"
```

(Non-group content between groups — like `/` and `=` — also becomes
its own submobject, mirroring `MathTex` behavior.)

**For `Typst` (text mode):** the same `{{ ... }}` notation applies, but the
wrapper is `#box[...]` without math delimiters.

### 2. Named Groups via Labels

Users can also assign explicit label names for retrieval by name:

```python
eq = Typst(
    r"$ #box[$a + b$] <numer> / #box[$c - d$] <denom> $"
)
eq.select("numer").set_color(RED)
eq.select("denom").set_color(BLUE)
```

Alternatively, an even more ergonomic approach that hides the `box` boilerplate
and uses the `{{ ... : label }}` notation:

```python
eq = TypstMath("{{ a + b : numer }} / {{ c - d : denom }}")
eq.select("numer").set_color(RED)
```

Here the pre-processor recognizes `{{ content : label }}` and emits
`#box[$content$] <label>` in the Typst source.

### 3. The `.select()` Method

```python
def select(self, key: str | int) -> VGroup:
    """Select a labeled sub-expression.

    Parameters
    ----------
    key
        Either a label name (string) matching a ``data-typst-label``
        in the SVG, or an integer index into the auto-numbered
        ``{{ ... }}`` groups.

    Returns
    -------
    VGroup
        The sub-mobjects corresponding to the selected group.

    Raises
    ------
    KeyError
        If no group with the given label/index exists.
    """
```

This returns a `VGroup` containing exactly the submobjects (paths) that
were rendered inside the corresponding `<g data-typst-label="...">` in the SVG.

## Implementation Plan

### Step 1: Extend `SVGMobject.get_mobjects_from()` to Track Labels

In `manim/mobject/svg/svg_mobject.py`, the group-walking loop already checks
for `id` attributes. Add a parallel check for `data-typst-label`:

```python
try:
    group_name = str(element.values["id"])
except Exception:
    # Fall back to data-typst-label if available
    label = element.values.get("data-typst-label")
    if label:
        group_name = f"typst-label:{label}"
    else:
        group_name = f"numbered_group_{group_id_number}"
        group_id_number += 1
```

This automatically populates `id_to_vgroup_dict` with label-keyed entries.

### Step 2: Pre-Processing `{{ ... }}` in Typst Source

Add a `_split_and_label_groups()` method that:

1. Scans the input for `{{ ... }}` or `{{ ... : label }}` patterns
   (using the same whitespace-guard rules as `MathTex._split_double_braces`).
2. Replaces each group with `#box[$content$] <label>` (math mode) or
   `#box[content] <label>` (text mode).
3. Records the mapping from label → original source string for later lookup.

### Step 3: `Typst.select()` / Index Access

- Store the ordered list of group labels and their source strings.
- `select(label_or_index)` looks up the corresponding `VGroup` from
  `id_to_vgroup_dict` (using the `typst-label:...` key).
- `__getitem__(int)` returns the *n*-th group's `VGroup`.

### Step 4: Compatibility with `TransformMatchingTex` (future)

`TransformMatchingTex` (and its successor `TransformMatchingShapes`) works by
matching submobjects between two `MathTex` instances by their TeX string keys.
The same approach extends to `Typst` if each `{{ ... }}` group carries its
original source string as metadata. A `TransformMatchingTypst` animation could
match groups by label name or by source string equality.

## Open Design Questions

### Q1: Context-Aware Wrapping — Math vs. Text Mode

The `box` + `label` mechanism works identically in math and text mode, but the
**wrapping** of group content must match the surrounding context:

- **In text mode:** `{{ Hello : greeting }}` → `#box[Hello] <greeting>`
- **In math mode:** `{{ y^2 : second }}` → `#box[$y^2$] <second>`

Getting this wrong is not a silent error — it produces visually broken output.
Wrapping math content with `#box[y^2]` (no `$...$`) renders `y^2` as literal
text in the body font instead of as a math superscript.

This is a real problem for `Typst()`, where a single source string can mix text
and math freely:

```python
Typst("hello world, here is a formula: $x^2 + {{ y^2 : second }} = z^2$")
```

Here `{{ y^2 : second }}` is inside a `$ ... $` block, so it needs the
math-mode wrapper, but the pre-processor has no way to know this unless it
tracks `$` delimiters.

### The `#` prefix problem and math calls

A natural idea is to translate `{{ content }}` into a Typst function call like
`grp("lbl", content)`. However, this has a subtle but critical context
sensitivity: Typst has two different call conventions depending on context:

- **Math call** (no `#` prefix): `$ grp("lbl", a^2 + b) $` — arguments are
  parsed **in math mode**. The content `a^2 + b` is math. ✓
- **Code call** (`#` prefix): `$ #grp("lbl", a^2 + b) $` — arguments are
  parsed **in code mode**. `a^2` is a syntax error in code! ✗

So in math mode, the function MUST be called without `#` for args to stay in
math mode. In text/markup mode, the function MUST be called WITH `#` (that's
how you invoke code from markup), and content arguments need `[...]` wrapping:

```typst
// Text context:  #grp("lbl", [Hello world])
// Math context:  grp("lbl", a^2 + b)
```

The function definition is the same either way:
```typst
#let grp(lbl, body) = [#box(body) #label(lbl)]
```

This means the function call approach has **exactly the same context problem**
as the raw `#box` approach: the pre-processor must know whether it's in math or
text to emit the right calling convention.

### Further complication: string literals and content blocks

Even inside `TypstMath` (where everything is math), the scanner must avoid
`{{ }}` matches inside string literals or content blocks:

```python
TypstMath('x^2 + y^2 =_("Hello {{ world }}") z^2')
```

Here `{{ world }}` is inside a `"..."` string literal — it should NOT be
processed. Similarly, content blocks `[...]` inside math switch back to text
mode.

### Options

**A. `TypstMath`: math calls with simple string-aware scanning.**
For `TypstMath`, the entire body is math, so `{{ content }}` always becomes
`grp("_grp-N", content)` (no `#`, no `$...$`). The scanner just needs to
skip `"..."` string literals and `[...]` content blocks — no `$` tracking
needed. This is clean and robust.

**B. `Typst`: context-aware scanning (full parser).**
For the general `Typst` class, the scanner must additionally track `$...$`
math blocks (toggling a mode flag on unescaped `$`) to choose between
`grp(...)` (in math) and `#grp("lbl", [...])` (in text). It must also handle
string literals and content blocks inside math that switch context back. This
is doable but non-trivial — essentially a mini Typst lexer.

**C. `Typst`: no `{{ }}`, manual groups only.**
For the general `Typst` class, don't support `{{ }}` at all. Users write
`grp(...)` / `#grp(...)` themselves (with the helper injected into the
preamble). `{{ }}` is only available on `TypstMath`. This is simpler and
avoids the parsing complexity, at the cost of ergonomics for mixed-mode
documents.

**Recommendation:** Start with A (TypstMath only) and C (manual for Typst).
Upgrade to B later if demand warrants it — the function call infrastructure
is already in place, it's only the scanner that needs upgrading.

### Q2: What about "unlabeled" content between groups?

Like `MathTex`, the pieces of content *between* `{{ ... }}` groups should also
become their own submobjects (auto-labeled with sequential indices). For
example:

```python
TypstMath("{{ a }} + {{ b }} = {{ c }}")
#  group-0: "a"
#  group-1: "+"       (auto-group for inter-group content)
#  group-2: "b"
#  group-3: "="       (auto-group for inter-group content)
#  group-4: "c"
```

Each segment (group or inter-group) gets wrapped in its own labeled `box`.

### Q3: What happens with `box` and baseline alignment?

`box` is an inline element in Typst, and when used inside math mode it
participates in math layout. Testing confirms that fractions, superscripts, and
other constructs render correctly when their children are `box`-wrapped.
However, `box` creates a "hard frame" boundary which may subtly affect spacing
in edge cases (e.g., math operator spacing around a boxed expression). This
needs further testing; if issues arise, we could explore `block(breakable: false)`
or invisible `rect` wrappers as alternatives.

### Q4: Can we avoid the `#grp(...)` / `#box[...] <label>` verbosity?

Yes — the `{{ ... }}` double-brace notation is purely syntactic sugar that gets
pre-processed by Manim before the source reaches the Typst compiler. Users never
need to write raw `#box` or `#label()` calls unless they want finer control.

### Q5: String-based selection without explicit groups?

A future enhancement could support:

```python
eq = TypstMath(r"a + b = c")
eq.select("a")  # finds submobjects corresponding to the glyph "a"
```

This is hard to do reliably because:
- Typst SVGs embed glyphs as `<use xlink:href="#gXXX">` references; there's no
  text content in the SVG itself.
- A single variable in Typst may span multiple glyphs (e.g., `"alpha"` → one
  glyph) or identical glyphs may appear multiple times.

A possible approach: at pre-processing time, wrap every "token" in the Typst
math source in its own labeled `box`. This would require a Typst math tokenizer
and is better suited for a v2 implementation.

## Summary: What Typst Gives Us

| Mechanism | How it works | SVG output |
|---|---|---|
| `#box(body) <label>` | Creates a hard-frame `GroupItem` with a `Label` | `<g data-typst-label="label">...</g>` |
| `#metadata(val) <label>` | Invisible; queryable via `typst query` CLI | No visual output (useful for CLI queries, not SVG) |
| Show rules on labels | `#show <label>: ...` | Transforms visual output but no automatic SVG grouping |
| `context query(<label>)` | Document introspection (positions, counters) | In-document only; not available from Python |

The `box` + `label` mechanism is the **only** one that produces identifiable
groups in the SVG output, making it the correct tool for sub-expression
selection in Manim.
