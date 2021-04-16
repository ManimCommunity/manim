#version 330

#INSERT camera_uniform_declarations.glsl

in vec3 point;

// Analog of import for manim only
#INSERT get_gl_Position.glsl
#INSERT position_point_into_frame.glsl

void main(){
    gl_Position = get_gl_Position(position_point_into_frame(point));
}
