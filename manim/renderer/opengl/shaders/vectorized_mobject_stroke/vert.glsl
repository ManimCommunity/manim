#version 330

uniform vec3 manim_unit_normal;
uniform mat4 u_model_view_matrix;
uniform mat4 u_projection_matrix;

in vec3[3] current_curve;
in vec2 tile_coordinate;
in vec4 in_color;
in float in_width;

out float v_degree;
out float v_thickness;
out vec2 uv_point;
out vec2[3] uv_curve;
out vec4 v_color;

int get_degree(in vec3 points[3], out vec3 normal) {
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

vec2 convert_to_uv(vec3 x_unit, vec3 y_unit, vec3 point) {
    return vec2(dot(point, x_unit), dot(point, y_unit));
}

vec3 convert_from_uv(vec3 translation, vec3 x_unit, vec3 y_unit, vec2 point) {
    vec3 untranslated_point = point[0] * x_unit + point[1] * y_unit;
    return untranslated_point + translation;
}

void main() {
    float thickness_multiplier = 0.004;
    v_color = in_color;

    vec3 computed_normal;
    v_degree = get_degree(current_curve, computed_normal);

    vec3 tile_x_unit = normalize(current_curve[2] - current_curve[0]);
    vec3 unit_normal;
    vec3 tile_y_unit;
    if (v_degree == 0) {
        tile_y_unit = vec3(0.0, 0.0, 0.0);
    } else if (v_degree == 1) {
        // Since the curve forms a straight line there's no way to compute a normal.
        unit_normal = manim_unit_normal;

        tile_y_unit = cross(unit_normal, tile_x_unit);
    } else {
        // Prefer to use a computed normal vector rather than the one from manim.
        unit_normal = computed_normal;

        // Ensure tile_y_unit is pointing toward p1 from p0.
        tile_y_unit = cross(unit_normal, tile_x_unit);
        if (dot(tile_y_unit, current_curve[1] - current_curve[0]) < 0) {
            tile_y_unit *= -1;
        }
    }

    // Project the curve onto the tile.
    for(int i = 0; i < 3; i++) {
        uv_curve[i] = convert_to_uv(tile_x_unit, tile_y_unit, current_curve[i]);
    }

    // Compute the curve's bounding box.
    vec4 uv_bounding_box = bboxBezier(uv_curve[0], uv_curve[1], uv_curve[2]);
    vec3 tile_translation = unit_normal * dot(current_curve[0], unit_normal);
    vec3 bounding_box_min = convert_from_uv(tile_translation, tile_x_unit, tile_y_unit, uv_bounding_box.xy);
    vec3 bounding_box_max = convert_from_uv(tile_translation, tile_x_unit, tile_y_unit, uv_bounding_box.zw);
    vec3 bounding_box_vec = bounding_box_max - bounding_box_min;
    vec3 tile_origin = bounding_box_min;
    vec3 tile_x_vec = tile_x_unit * dot(tile_x_unit, bounding_box_vec);
    vec3 tile_y_vec = tile_y_unit * dot(tile_y_unit, bounding_box_vec);

    // Expand the tile according to the line's thickness.
    v_thickness = thickness_multiplier * in_width;
    tile_origin = current_curve[0] - v_thickness * (tile_x_unit + tile_y_unit);
    tile_x_vec += 2 * v_thickness * tile_x_unit;
    tile_y_vec += 2 * v_thickness * tile_y_unit;

    vec3 tile_point = tile_origin + \
                      tile_coordinate[0] * tile_x_vec + \
                      tile_coordinate[1] * tile_y_vec;
    gl_Position = u_projection_matrix * u_model_view_matrix * vec4(tile_point, 1.0);
    uv_point = convert_to_uv(tile_x_unit, tile_y_unit, tile_point);
}
