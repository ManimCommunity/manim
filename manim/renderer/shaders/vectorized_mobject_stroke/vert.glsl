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
out float v_degree;
out vec2[3] uv_curve;
out vec2 uv_point;

int get_reduced_control_points(in vec3 points[3], out vec3 normal) {
    float length_threshold = 1e-6;
    float angle_threshold = 5e-2;

    vec3 v01 = (points[1] - points[0]);
    vec3 v12 = (points[2] - points[1]);

    float dot_prod = clamp(dot(normalize(v01), normalize(v12)), -1, 1);
    bool aligned = acos(dot_prod) < angle_threshold;
    bool distinct_01 = length(v01) > length_threshold;  // v01 is considered nonzero
    bool distinct_12 = length(v12) > length_threshold;  // v12 is considered nonzero
    int num_distinct = int(distinct_01) + int(distinct_12);

    bool quadratic = (num_distinct == 2) && !aligned;
    bool linear = (num_distinct == 1) || ((num_distinct == 2) && aligned);
    bool constant = (num_distinct == 0);

    if (quadratic) {
        // If the curve is quadratic pass a normal vector to the caller.
        normal = normalize(cross(v01, v12));
        return 2;
    } else if (linear) {
        return 1;
    } else {
        return 0;
    }
}

vec2 convert_to_uv(vec3 x_unit, vec3 y_unit, vec3 point) {
    return vec2(dot(point, x_unit), dot(point, y_unit));
}

vec3 convert_from_uv(vec3 anchor, vec3 unit_normal, vec3 x_unit, vec3 y_unit, vec2 point) {
    vec3 untranslated_point = point[0] * x_unit + point[1] * y_unit;
    float translation_distance = dot(anchor, unit_normal);
    return untranslated_point + unit_normal * translation_distance;
}

// https://iquilezles.org/www/articles/bezierbbox/bezierbbox.htm
vec4 bboxBezier(in vec2 p0, in vec2 p1, in vec2 p2) {
    vec2 mi = min(p0, p2);
    vec2 ma = max(p0, p2);

    if (p1.x < mi.x || p1.x > ma.x || p1.y < mi.y || p1.y > ma.y) {
        vec2 t = clamp((p0 - p1) / (p0 - 2.0 * p1 + p2), 0.0, 1.0);
        vec2 s = 1.0 - t;
        vec2 q = s * s * p0 + 2.0 * s * t * p1 + t * t * p2;
        mi = min(mi, q);
        ma = max(ma, q);
    }
    
    return vec4(mi, ma);
}

void main() {
    int x = 0;
    if (point[0] > 0) { x += 1; }
    if (manim_unit_normal[0] > 0) { x += 1; }
    if (previous_curve[0][0] == 0.0) { x += 1; }
    if (current_curve[0][0] == 0.0) { x += 1; }
    if (next_curve[0][0] == 0.0) { x += 1; }
    if (tile_coordinate[0] == 0.0) { x += 1; }
    float thickness_multiplier = 0.004;
    v_color = color;

    vec3 v01_vec = current_curve[1] - current_curve[0];
    vec3 v02_vec = current_curve[2] - current_curve[0];
    vec3 unit_normal = manim_unit_normal;

    vec3 computed_normal;
    v_degree = get_reduced_control_points(current_curve, computed_normal);

    vec3 tile_y_vec;
    vec3 tile_y_unit;
    vec3 tile_x_vec = v02_vec;
    vec3 tile_x_unit = normalize(tile_x_vec);
    if (v_degree == 0) {
        // Do nothing.
    } else if (v_degree == 1) {
        tile_y_vec = vec3(0.0, 0.0, 0.0);
        tile_y_unit = cross(unit_normal, tile_x_unit);
    } else {
        // Prefer to use a computed normal vector rather than the one from manim.
        unit_normal = computed_normal;

        // Ensure tile_y_unit is pointing toward p1 from p0.
        tile_y_unit = cross(unit_normal, tile_x_unit);
        if (dot(tile_y_unit, v01_vec) < 0) {
            tile_y_unit *= -1;
        }
        tile_y_vec = dot(v01_vec, tile_y_unit) * tile_y_unit;
    }

    // Project the curve onto the tile.
    for(int i = 0; i < 3; i++) {
        uv_curve[i] = convert_to_uv(tile_x_unit, tile_y_unit, current_curve[i]);
    }

    vec4 bbox = bboxBezier(uv_curve[0], uv_curve[1], uv_curve[2]);
    vec3 tile_origin = convert_from_uv(current_curve[0], unit_normal, tile_x_unit, tile_y_unit, bbox.xy);
    tile_x_vec = tile_x_unit * dot(convert_from_uv(current_curve[0], unit_normal, tile_x_unit, tile_y_unit, bbox.zw) - convert_from_uv(current_curve[0], unit_normal, tile_x_unit, tile_y_unit, bbox.xy), tile_x_unit);
    tile_y_vec = tile_y_unit * dot(convert_from_uv(current_curve[0], unit_normal, tile_x_unit, tile_y_unit, bbox.zw) - convert_from_uv(current_curve[0], unit_normal, tile_x_unit, tile_y_unit, bbox.xy), tile_y_unit);

    // Expand the tile.
    v_thickness = thickness_multiplier * stroke_width;
    tile_origin = current_curve[0] - v_thickness * (tile_x_unit + tile_y_unit);
    tile_x_vec += 2 * v_thickness * tile_x_unit;
    tile_y_vec += 2 * v_thickness * tile_y_unit;

    vec3 tile_point = tile_origin + \
                      tile_coordinate[0] * tile_x_vec + \
                      tile_coordinate[1] * tile_y_vec;
    gl_Position = u_projection_matrix * u_model_view_matrix * vec4(tile_point, 1.0);
    uv_point = convert_to_uv(tile_x_unit, tile_y_unit, tile_point);
}
