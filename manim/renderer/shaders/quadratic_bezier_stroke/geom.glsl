#version 330

layout(triangles) in;
layout(triangle_strip, max_vertices = 6) out;

uniform float anti_alias_width;
uniform float flat_stroke;
uniform float pixel_size;
uniform float joint_type;

in vec3 verts[3];

in vec4 v_joint_product[3];
in float v_stroke_width[3];
in vec4 v_color[3];

out vec4 color;
out float uv_stroke_width;
out float uv_anti_alias_width;

out float is_linear;

out vec2 uv_coords;

// Codes for joint types
const int NO_JOINT = 0;
const int AUTO_JOINT = 1;
const int BEVEL_JOINT = 2;
const int MITER_JOINT = 3;

// When the cosine of the angle between
// two vectors is larger than this, we
// consider them aligned
const float COS_THRESHOLD = 0.99;

vec3 unit_normal = vec3(0.0, 0.0, 1.0);

#include "../include/get_gl_Position.glsl"
#include "../include/get_xyz_to_uv.glsl"
#include "../include/finalize_color.glsl"

vec3 get_joint_unit_normal(vec4 joint_product) {
    vec3 result = (joint_product.w < COS_THRESHOLD) ?
        joint_product.xyz : v_joint_product[1].xyz;
    float norm = length(result);
    return (norm > 1e-5) ? result / norm : vec3(0.0, 0.0, 1.0);
}

vec4 normalized_joint_product(vec4 joint_product) {
    float norm = length(joint_product);
    return (norm > 1e-10) ? joint_product / norm : vec4(0.0, 0.0, 0.0, 1.0);
}

void create_joint(
    vec4 joint_product,
    vec3 unit_tan,
    float buff,
    vec3 static_c0,
    out vec3 changing_c0,
    vec3 static_c1,
    out vec3 changing_c1
) {
    float cos_angle = joint_product.w;
    if (abs(cos_angle) > COS_THRESHOLD || int(joint_type) == NO_JOINT) {
        // No joint
        changing_c0 = static_c0;
        changing_c1 = static_c1;
        return;
    }

    float shift;
    float sin_angle = length(joint_product.xyz) * sign(joint_product.z);
    if (int(joint_type) == MITER_JOINT) {
        shift = buff * (-1.0 - cos_angle) / sin_angle;
    } else {
        // For a Bevel joint
        shift = buff * (1.0 - cos_angle) / sin_angle;
    }
    changing_c0 = static_c0 - shift * unit_tan;
    changing_c1 = static_c1 + shift * unit_tan;
}

vec3 get_perp(int index, vec4 joint_product, vec3 point, vec3 tangent, float aaw) {
    /*
                Perpendicular vectors to the left of the curve
                */
    float buff = 0.5 * v_stroke_width[index] + aaw;
    // Add correction for sharp angles to prevent weird bevel effects
    if (joint_product.w < -0.75) buff *= 4 * (joint_product.w + 1.0);
    vec3 normal = get_joint_unit_normal(joint_product);
    // Set global unit normal
    unit_normal = normal;
    // Choose the "outward" normal direction
    if (normal.z < 0) normal *= -1;
    if (bool(flat_stroke)) {
        return buff * normalize(cross(normal, tangent));
    } else {
        return buff * normalize(cross(camera_position - point, tangent));
    }
}

// This function is responsible for finding the corners of
// a bounding region around the bezier curve, which can be
// emitted as a triangle fan, with vertices vaguely close
// to control points so that the passage of vert data to
// frag shaders is most natural.
void get_corners(
    // Control points for a bezier curve
    vec3 p0,
    vec3 p1,
    vec3 p2,
    // Unit tangent vectors at p0 and p2
    vec3 v01,
    vec3 v12,
    // Anti-alias width
    float aaw,
    out vec3 corners[6]
) {
    bool linear = bool(is_linear);
    vec4 jp0 = normalized_joint_product(v_joint_product[0]);
    vec4 jp2 = normalized_joint_product(v_joint_product[2]);
    vec3 p0_perp = get_perp(0, jp0, p0, v01, aaw);
    vec3 p2_perp = get_perp(2, jp2, p2, v12, aaw);
    vec3 p1_perp = 0.5 * (p0_perp + p2_perp);
    if (linear) {
        p1_perp *= (0.5 * v_stroke_width[1] + aaw) / length(p1_perp);
    }

    // The order of corners should be for a triangle_strip.
    vec3 c0 = p0 + p0_perp;
    vec3 c1 = p0 - p0_perp;
    vec3 c2 = p1 + p1_perp;
    vec3 c3 = p1 - p1_perp;
    vec3 c4 = p2 + p2_perp;
    vec3 c5 = p2 - p2_perp;
    // Move the inner middle control point to make
    // room for the curve
    // float orientation = dot(unit_normal, v_joint_product[1].xyz);
    float orientation = v_joint_product[1].z;
    if (!linear && orientation >= 0.0) c2 = 0.5 * (c0 + c4);
    else if (!linear && orientation < 0.0) c3 = 0.5 * (c1 + c5);

    // Account for previous and next control points
    if (bool(flat_stroke)) {
        create_joint(jp0, v01, length(p0_perp), c1, c1, c0, c0);
        create_joint(jp2, -v12, length(p2_perp), c5, c5, c4, c4);
    }

    corners = vec3[6](c0, c1, c2, c3, c4, c5);
}

void main() {
    // Curves are marked as ended when the handle after
    // the first anchor is set equal to that anchor
    if (verts[0] == verts[1]) return;

    vec3 p0 = verts[0];
    vec3 p1 = verts[1];
    vec3 p2 = verts[2];
    vec3 v01 = normalize(p1 - p0);
    vec3 v12 = normalize(p2 - p1);

    vec4 jp1 = normalized_joint_product(v_joint_product[1]);
    is_linear = float(jp1.w > COS_THRESHOLD);

    // We want to change the coordinates to a space where the curve
    // coincides with y = x^2, between some values x0 and x2. Or, in
    // the case of a linear curve just put it on the x-axis
    mat4 xyz_to_uv;
    float uv_scale_factor;
    if (!bool(is_linear)) {
        bool too_steep;
        xyz_to_uv = get_xyz_to_uv(p0, p1, p2, 2.0, too_steep);
        is_linear = float(too_steep);
        uv_scale_factor = length(xyz_to_uv[0].xyz);
    }

    float scaled_aaw = anti_alias_width * pixel_size;
    vec3 corners[6];
    get_corners(p0, p1, p2, v01, v12, scaled_aaw, corners);

    // Emit each corner
    float max_sw = max(v_stroke_width[0], v_stroke_width[2]);
    for (int i = 0; i < 6; i++) {
        float stroke_width = v_stroke_width[i / 2];

        if (bool(is_linear)) {
            float sign = vec2(-1, 1)[i % 2];
            // In this case, we only really care about
            // the v coordinate
            uv_coords = vec2(0, sign * (0.5 * stroke_width + scaled_aaw));
            uv_anti_alias_width = scaled_aaw;
            uv_stroke_width = stroke_width;
        } else {
            uv_coords = (xyz_to_uv * vec4(corners[i], 1.0)).xy;
            uv_stroke_width = uv_scale_factor * stroke_width;
            uv_anti_alias_width = uv_scale_factor * scaled_aaw;
        }

        color = finalize_color(v_color[i / 2], corners[i], unit_normal);
        emit_gl_Position(corners[i]);
        EmitVertex();
    }
    EndPrimitive();
}
