#ifndef GET_UNIT_NORMAL_GLSL
#define GET_UNIT_NORMAL_GLSL

vec3 get_unit_normal(in vec3[3] points)
{
    float tol = 1e-6;
    vec3 v1 = normalize(points[1] - points[0]);
    vec3 v2 = normalize(points[2] - points[1]);
    vec3 cp = cross(v1, v2);
    float cp_norm = length(cp);
    if (cp_norm < tol)
    {
        // Three points form a line, so find a normal vector
        // to that line in the plane shared with the z-axis
        vec3 k_hat = vec3(0.0, 0.0, 1.0);
        vec3 comb = v1 + v2;
        vec3 new_cp = cross(cross(comb, k_hat), comb);
        float new_cp_norm = length(new_cp);
        if (new_cp_norm < tol)
        {
            // We only come here if all three points line up
            // on the z-axis.
            return vec3(0.0, -1.0, 0.0);
        }
        return new_cp / new_cp_norm;
    }
    return cp / cp_norm;
}
#endif // GET_UNIT_NORMAL_GLSL
