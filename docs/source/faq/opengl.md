# FAQ: OpenGL rendering

## Are there any resources on how the OpenGL renderer in the community maintained version can be used?

Render with `--renderer=opengl`; add `-l` (or `--live-preview`) for an
interactive preview window. The rest of the documentation applies to both
renderers. Smooth, lit surfaces
({class}`.ParametricSurface`, {class}`.MeshMobject`) are currently drawn by the
OpenGL renderer only.

Earlier versions had separate OpenGL mobject classes (`OpenGLMobject`,
`OpenGLVMobject`, `OpenGLSurface`, ...); see
{ref}`the upgrading guide <upgrading-opengl-mobjects>` for how to update code that used them.

---

## I am trying to run an interactive scene with `--renderer=opengl` and `Scene.interactive_embed`, but an error (`sqlite3.ProgrammingError`) is raised. How can I fix this?

This seems to be an issue with a recent IPython release,
in our experience it helps to downgrade the installed `IPython`
package to `8.0.1`: `pip install IPython==8.0.1`.
