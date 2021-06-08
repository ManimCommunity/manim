#version 330

uniform vec3 unit_normal;
uniform mat4 u_model_view_matrix;
uniform mat4 u_projection_matrix;

in vec3 point;
in vec3[3] previous_curve;
in vec3[3] current_curve;
in vec3[3] next_curve;
in vec2 tile_coordinate;

out vec2 v_point;
out vec2[3] v_previous_curve;
out vec2[3] v_current_curve;
out vec2[3] v_next_curve;
out vec4 v_color;

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

void main() {
    int x = 0;
    if (point[0] > 0) { x += 1; }
    if (unit_normal[0] > 0) { x += 1; }
    if (previous_curve[0][0] == 0.0) { x += 1; }
    if (current_curve[0][0] == 0.0) { x += 1; }
    if (next_curve[0][0] == 0.0) { x += 1; }
    if (tile_coordinate[0] == 0.0) { x += 1; }

    // Compute tile data.
    // TODO: Make the origin the lower left point always.
    vec3 tile_origin = current_curve[0];
    vec3 tile_x_vec = current_curve[2] - current_curve[0];
    vec3 tile_x_unit = normalize(tile_x_vec);

    vec3 tile_y_unit = cross(unit_normal, normalize(tile_x_vec));
    vec3 tile_y_vec = dot((current_curve[1] - current_curve[0]), tile_y_unit) * tile_y_unit;

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

    if (v_current_curve[1][0] == 0.5) {
        v_color = vec4(1.0, 0.0, 0.0, 1.0);
    } else {
        v_color = vec4(0.0, 0.0, 1.0, 1.0);
    }

    vec3 tile_point = tile_origin + tile_coordinate[0] * tile_x_vec + tile_coordinate[1] * tile_y_vec;
    gl_Position = u_projection_matrix * u_model_view_matrix * vec4(tile_point, 1.0);
    v_point = project_to_tile(
        tile_origin,
        tile_x_vec,
        tile_y_vec,
        x_uv_max,
        y_uv_max,
        tile_point
    );
}
