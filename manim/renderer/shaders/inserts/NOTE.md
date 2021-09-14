There seems to be no analog to #include in C++ for OpenGL shaders.  While there are other options for sharing code between shaders, a lot of them aren't great, especially if the goal is to have all the logic for which specific bits of code to share handled in the shader file itself.  So the way manim currently works is to replace any line which looks like

#INSERT <file_name>

with the code from one of the files in this folder.

The functions in this file often include reference to uniforms which are assumed to be part of the surrounding context into which they are inserted.
