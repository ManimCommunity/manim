#version 330

in float v_degree;
in float v_thickness;
in vec2 uv_point;
in vec2[3] uv_curve;
in vec4 v_color;

out vec4 frag_color;

// https://www.shadertoy.com/view/ltXSDB
// Test if point p crosses line (a, b), returns sign of result
float testCross(vec2 a, vec2 b, vec2 p) {
    return sign((b.y-a.y) * (p.x-a.x) - (b.x-a.x) * (p.y-a.y));
}

// Determine which side we're on (using barycentric parameterization)
float signBezier(vec2 A, vec2 B, vec2 C, vec2 p)
{
    vec2 a = C - A, b = B - A, c = p - A;
    vec2 bary = vec2(c.x*b.y-b.x*c.y,a.x*c.y-c.x*a.y) / (a.x*b.y-b.x*a.y);
    vec2 d = vec2(bary.y * 0.5, 0.0) + 1.0 - bary.x - bary.y;
    return mix(sign(d.x * d.x - d.y), mix(-1.0, 1.0,
        step(testCross(A, B, p) * testCross(B, C, p), 0.0)),
        step((d.x - d.y), 0.0)) * testCross(A, C, B);
}

// Solve cubic equation for roots
vec3 solveCubic(float a, float b, float c)
{
    float p = b - a*a / 3.0, p3 = p*p*p;
    float q = a * (2.0*a*a - 9.0*b) / 27.0 + c;
    float d = q*q + 4.0*p3 / 27.0;
    float offset = -a / 3.0;
    if(d >= 0.0) {
        float z = sqrt(d);
        vec2 x = (vec2(z, -z) - q) / 2.0;
        vec2 uv = sign(x)*pow(abs(x), vec2(1.0/3.0));
        return vec3(offset + uv.x + uv.y);
    }
    float v = acos(-sqrt(-27.0 / p3) * q / 2.0) / 3.0;
    float m = cos(v), n = sin(v)*1.732050808;
    return vec3(m + m, -n - m, n - m) * sqrt(-p / 3.0) + offset;
}

// Find the signed distance from a point to a bezier curve
float sdBezier(vec2 A, vec2 B, vec2 C, vec2 p)
{
    B = mix(B + vec2(1e-4), B, abs(sign(B * 2.0 - A - C)));
    vec2 a = B - A, b = A - B * 2.0 + C, c = a * 2.0, d = A - p;
    vec3 k = vec3(3.*dot(a,b),2.*dot(a,a)+dot(d,b),dot(d,a)) / dot(b,b);
    vec3 t = clamp(solveCubic(k.x, k.y, k.z), 0.0, 1.0);
    vec2 pos = A + (c + b*t.x)*t.x;
    float dis = length(pos - p);
    pos = A + (c + b*t.y)*t.y;
    dis = min(dis, length(pos - p));
    pos = A + (c + b*t.z)*t.z;
    dis = min(dis, length(pos - p));
    return dis * signBezier(A, B, C, p);
}

// https://www.shadertoy.com/view/llcfR7
float dLine(vec2 p1, vec2 p2, vec2 x) {
    vec4 colA = vec4(clamp (5.0 - length (x - p1), 0.0, 1.0));
    vec4 colB = vec4(clamp (5.0 - length (x - p2), 0.0, 1.0));

    vec2 a_p1 = x - p1;
    vec2 p2_p1 = p2 - p1;
    float h = clamp (dot (a_p1, p2_p1) / dot (p2_p1, p2_p1), 0.0, 1.0);
    return length (a_p1 - p2_p1 * h);
}

void main() {
    float distance;
    if (v_degree == 2.0) {
        distance = sdBezier(uv_curve[0], uv_curve[1], uv_curve[2], uv_point);
    } else {
        distance = dLine(uv_curve[0], uv_curve[2], uv_point);
    }
    if (abs(distance) < v_thickness) {
        frag_color = v_color;
    } else {
        discard;
    }
}
