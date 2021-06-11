#version 330

uniform vec3 manim_unit_normal;
uniform mat4 u_model_view_matrix;
uniform mat4 u_projection_matrix;
uniform float stroke_width;
uniform vec4 color;

in vec3[3] previous_curve;
in vec3[3] current_curve;
in vec3[3] next_curve;
in vec3 point;
in vec2 tile_coordinate;

out vec2[3] v_previous_curve;
out vec2[3] v_current_curve;
out vec2[3] v_next_curve;
out vec2 v_point;
out vec4 v_color;
out float v_thickness;

vec2 project_to_tile(
    vec3 tile_origin,
    vec3 tile_x_vec,
    vec3 tile_y_vec,
    float x_uv_max,
    float y_uv_max,
    vec3 v
) {
    vec3 translated_current_curve = v - tile_origin;

    float tile_x_length = length(tile_x_vec);
    float x_projection_length = dot(translated_current_curve, tile_x_vec) / tile_x_length;
    float x_uv_ratio = x_projection_length / tile_x_length;
    float x_uv_coordinate = x_uv_ratio * x_uv_max;

    float tile_y_length = length(tile_y_vec);
    float y_projection_length = dot(translated_current_curve, tile_y_vec) / tile_y_length;
    float y_uv_ratio = y_projection_length / tile_y_length;
    float y_uv_coordinate = y_uv_ratio * y_uv_max;

    return vec2(x_uv_coordinate, y_uv_coordinate);
}

float angle_between_vectors(vec3 v1, vec3 v2){
    float v1_norm = length(v1);
    float v2_norm = length(v2);
    if(v1_norm == 0 || v2_norm == 0) return 0.0;
    float dot_product = dot(v1, v2) / (v1_norm * v2_norm);
    float angle = acos(clamp(dot_product, -1.0, 1.0));
    return angle;
}

void main() {
    int x = 0;
    if (point[0] > 0) { x += 1; }
    if (manim_unit_normal[0] > 0) { x += 1; }
    if (previous_curve[0][0] == 0.0) { x += 1; }
    if (current_curve[0][0] == 0.0) { x += 1; }
    if (next_curve[0][0] == 0.0) { x += 1; }
    if (tile_coordinate[0] == 0.0) { x += 1; }

    vec3 v01_vec = current_curve[1] - current_curve[0];
    vec3 v02_vec = current_curve[2] - current_curve[0];
    vec3 unit_normal = manim_unit_normal;

    // Prefer to compute a normal if possible rather than rely on the one from manim.
    float angle_threshold = 1e-3;
    if (angle_between_vectors(v01_vec, v02_vec) > angle_threshold) {
        unit_normal = normalize(cross(v02_vec, v01_vec));
    }

    // Compute tile data.
    vec3 tile_origin = current_curve[0];
    vec3 tile_x_vec = v02_vec;
    vec3 tile_x_unit = normalize(tile_x_vec);

    // Ensure tile_y_unit is pointing toward p1 from p0.
    vec3 tile_y_unit = cross(unit_normal, tile_x_unit);
    if (dot(tile_y_unit, v01_vec) < 0) {
        tile_y_unit *= -1;
    }
    vec3 tile_y_vec = dot(v01_vec, tile_y_unit) * tile_y_unit;

    // Expand the tile.
    tile_origin -= 1 * tile_x_unit;
    tile_x_vec += 2 * tile_x_unit;
    tile_origin -= 1 * tile_y_unit;
    tile_y_vec += 2 * tile_y_unit;

    // Compute uv coordinates.
    float tile_x_length = length(tile_x_vec);
    float tile_y_length = length(tile_y_vec);

    float tile_max_dimension = max(tile_x_length, tile_y_length);
    float x_uv_max = tile_x_length / tile_max_dimension;
    float y_uv_max = tile_y_length / tile_max_dimension;
    v_thickness = 0.004 * stroke_width / tile_max_dimension;

    for(int i = 0; i < 3; i++) {
        // Project the curve onto the tile;
        v_current_curve[i] = project_to_tile(
            tile_origin,
            tile_x_vec,
            tile_y_vec,
            x_uv_max,
            y_uv_max,
            current_curve[i]
        );
    }

    vec3 tile_point = tile_origin + \
                      tile_coordinate[0] * tile_x_vec + \
                      tile_coordinate[1] * tile_y_vec;
    gl_Position = u_projection_matrix * u_model_view_matrix * vec4(tile_point, 1.0);
    v_point = project_to_tile(
        tile_origin,
        tile_x_vec,
        tile_y_vec,
        x_uv_max,
        y_uv_max,
        tile_point
    );
    v_color = color;
}
