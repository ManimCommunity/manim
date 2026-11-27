#version 330

// Square points of whole pixels, as the Cairo renderer draws point clouds.
in vec4 v_color;

out vec4 frag_color;

void main(){
    frag_color = v_color;
}
