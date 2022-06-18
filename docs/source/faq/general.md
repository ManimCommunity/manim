# FAQ: General Usage

## Why does Manim say that "there are no scenes inside that module"?

There are two main reasons why this error appears: if you have edited
the file containing your `Scene` class and forgot to save it, or if you
have accidentally passed the name of a wrong file to `manim`, this is
a likely outcome. Check that you have spelled everything correctly.

Otherwise you are likely mixing up Manim versions. See {ref}`this FAQ answer <different-versions>`
for an explanation regarding why there are different versions. Under the assumption
that you are trying to use the `manim` executable from the terminal to run
a scene that has been written for the community version (i.e., there is
`from manim import *`, or more specifically `from manim import Scene`)

---

## No matter what code I put in my file, Manim only renders a black frame! Why?

If you are using the usual pattern to write a `Scene`, i.e.,
```python
class MyAwesomeScene(Scene):
    def construct(self):
        ...
        # your animation code
```
then double check whether you have spelled `construct` correctly.
If the method containing your code is not called `construct` (or
if you are not calling a different, custom method from `construct`),
Manim will not call your method and simply output a black frame.

---

## What are the default measurements for Manim's scene?

The scene measures 8 units in height and has a default ratio of 16:9,
which means that it measures {math}`8 \cdot 16 / 9 = 14 + 2/9` units in width.
The origin is in the center of the scene, which means that, for example, the
upper left corner of the scene has coordinates `[-7-1/9, 4, 0]`.

---

## How do I find out which keyword arguments I can pass when creating a `Mobject`?

Let us consider some specific example, like the {class}`.Circle` class. When looking
at its documentation page, only two specific keyword arguments are listed
(`radius`, and `color`). Besides these concrete arguments, there is also a
catchall `**kwargs` argument which captures all other arguments that are passed
to `Circle`, and passes them on to the base class of {class}`.Circle`, {class}`.Arc`.

The same holds for {class}`.Arc`: some arguments are explicitly documented, and
there is again a catchall `**kwargs` argument that passes unprocessed arguments
to the next base class -- and so on.

The most important keyword arguments relevant to styling your mobjects are the
ones that are documented for the base classes {class}`.VMobject` and
{class}`.Mobject`.

---

## Can Manim render a video with transparent background?

Yes: simply pass the CLI flag `-t` (or its long form `--transparent`).
Note that the default video file format does not support transparency,
which is why Manim will output a `.mov` instead of a `.mp4` when
rendering with a transparent background. Other movie file formats
that support transparency can be obtained by passing
`--format=webm` or `--format=gif`.

---

## I have watched a video where a creator ran command X, but it does not work for me. Why?

The video you have been watching is likely outdated. If you want to follow
along, you either need to use the same version used in the video, or
modify the code (in many cases it is just a method having been renamed etc.)
accordingly. Check the video description, in some cases creators point out
whether changes need to be applied to the code shown in the video.

---

## When using `Tex` or `MathTex`, some letters are missing. How can I fix this?

It is possible that you have to (re)build some fonts used by LaTeX. For
some distributions, you can do this manually by running
```bash
fmtutil -sys --all
```
We recommend consulting the documentation of your LaTeX distribution
for more information.

---

## I want to translate some code from `manimgl` to `manim`, what do I do with `CONFIG` dictionaries?

The community maintained version has dropped the use of `CONFIG` dictionaries very
early, with {doc}`version v0.2.0 </changelog/0.2.0-changelog>` released in
January 2021.

Before that, Manim's classes basically processed `CONFIG` dictionaries
by mimicking inheritance (to properly process `CONFIG` dictionaries set
by parent classes) and then assigning all of the key-value-pairs in the
dictionary as attributes of the corresponding object.

In situations where there is not much inheritance going on,
or for any custom setting, you should set these attributes yourself.
For example, for an old-style `Scene` with custom attributes like

```python
class OldStyle(Scene):
    CONFIG = {"a": 1, "b": 2}
```

should be written as

```python
class NewStyle(Scene):
    a = 1
    b = 2
```

In situations where values should be properly inherited, the arguments
should be added to the initialization function of the class. An old-style
mobject `Thing` could look like

```python
class Thing(VMobject):
    CONFIG = {
        "stroke_color": RED,
        "fill_opacity": 0.7,
        "my_awesome_argument": 42,
    }
```

where `stroke_color` and `fill_opacity` are arguments that concern the
parent class of `Thing`, and `my_awesome_argument` is a custom argument
that only concerns `Thing`. A version without `CONFIG` could look like this:

```python
class Thing(VMobject):
    def __init__(
        self, stroke_color=RED, fill_opacity=0.7, my_awesome_argument=42, **kwargs
    ):
        self.my_awesome_argument = my_awesome_argument
        super().__init__(stroke_color=stroke_color, fill_opacity=fill_opacity, **kwargs)
```

---

## My installation does not support converting PDF to SVG, help?

This is an issue with `dvisvgm`, the tool shipped with LaTeX that
transforms LaTeX output to a `.svg` file that
Manim can parse.

First, make sure your ``dvisvgm`` version is at least 2.4 by
checking the output of

```bash
dvisvgm --version
```

If you do not know how to update `dvisvgm`, please refer to your
LaTeX distributions documentation (or the documentation of your
operating system, if `dvisvgm` was installed as a system package).

Second, check whether your ``dvisvgm`` supports PostScript specials. This is
needed to convert from PDF to SVG. Run:

```bash
dvisvgm -l
```

If the output to this command does **not** contain `ps  dvips PostScript specials`,
this is a bad sign. In this case, run

```bash
dvisvgm -h
```

If the output does **not** contain `--libgs=filename`, this means your
`dvisvgm` does not currently support PostScript. You must get another binary.

If, however, `--libgs=filename` appears in the help, that means that your
`dvisvgm` needs the Ghostscript library to support PostScript. Search for
`libgs.so` (on Linux, probably in `/usr/local/lib` or `/usr/lib`) or
`gsdll32.dll` (on 32-bit Windows, probably in `C:\windows\system32`) or
`gsdll64.dll` (on 64-bit Windows, also probably in `C:\windows\system32`)
or `libgsl.dylib` (on MacOS, probably in `/usr/local/lib` or
`/opt/local/lib`). Please look carefully, as the file might be located
elsewhere, e.g. in the directory where Ghostscript is installed.

When you have found the library, try (on MacOS or Linux)

```bash
export LIBGS=<path to your library including the file name>
dvisvgm -l
```

or (on Windows)

```bat
set LIBGS=<path to your library including the file name>
dvisvgm -l
```

You should now see `ps    dvips PostScript specials` in the output. Refer to
your operating system's documentation to find out how you can set or export the
environment variable ``LIBGS`` automatically whenever you open a shell.

As a last check, you can run

```bash
dvisvgm -V1
```

(while still having `LIBGS` set to the correct path, of course.) If `dvisvgm`
can find your Ghostscript installation, it will be shown in the output together
with the version number.

If you do not have the necessary library on your system, please refer to your
operating system's documentation to find out where you can get it and how you
have to install it.

If you are unable to solve your problem, check out the
[dvisvgm FAQ](https://dvisvgm.de/FAQ/).

---

## Where can I find more resources for learning Manim?

In our [Discord server](https://manim.community/discord), we have the community maintained
`#beginner-resources` channel in which links to helpful learning resources are collected.
You are welcome to join our Discord and take a look yourself! If you have found some
guides or tutorials yourself that are not on our list yet, feel free to add them!
