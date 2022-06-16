# FAQ: OpenGL rendering

## Are there any resources on how the OpenGL renderer in the community maintained version can be used?

Yes. Unfortunately, at this point, the official online documentation does
not contain the relevant base classes like `OpenGLMobject` and `OpenGLVMobject`
or specific OpenGL classes like `OpenGLSurface`, but documentation for some of
them is available in form of docstrings
[in the source code](https://github.com/ManimCommunity/manim/tree/main/manim/mobject/opengl).

Furthermore, [this user guide by *aquabeam*](https://www.aquabeam.me/manim/opengl_guide/)
can be helpful to get started using the OpenGL renderer.

---

## I am trying to run an interactive scene with `--renderer=opengl` and `Scene.interactive_embed`, but an error (`sqlite3.ProgrammingError`) is raised. How can I fix this?

This seems to be an issue with a recent IPython release,
in our experience it helps to downgrade the installed `IPython`
package to `8.0.1`: `pip install IPython==8.0.1`.
