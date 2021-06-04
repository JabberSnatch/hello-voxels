R"__lstr__(

#version 430 core

layout(location = 0) out vec3 delta_irradiance;

uniform sampler2D trtex;

layout(std140, binding = 0) uniform ViewportBlock
{
    vec2 resolution;
} viewport;

struct LinearLayer
{
    float linear;
    float constant;
    vec2 padding;
};

layout(std140, binding = 1) uniform AtmosphereBlock
{
    vec3 sun_irradiance;
    float sun_angular_radius;

    vec2 bounds;
    vec2 padding;

    vec4 rscat; // rgb + expo_scale
    vec4 mext; // rgb + expo_scale

    LinearLayer odensity[2];
    vec4 oext; // rgb + obound

} atmos;

vec2 IrradianceUVtoRMu(vec2 radius_bounds, vec2 uv)
{
    float r = radius_bounds.x + uv.y * (radius_bounds.y - radius_bounds.x);
    float mu = min(1.0, max(-1.0, 2.0 * uv.x - 1.0));
    return vec2(r, mu);
}

float BoundaryDistance(float boundary_radius, float r, float mu)
{
    float delta = r*r * (mu*mu - 1.0) + boundary_radius*boundary_radius;
    return max(-r * mu + sqrt(max(delta, 0.0)), 0.0);
}

vec2 TransmittanceRMutoUV(vec2 radius_bounds, float r, float mu)
{
    vec2 boundssqr = radius_bounds * radius_bounds;
    float H = sqrt(boundssqr[1] - boundssqr[0]);
    float rho = sqrt(max(0.0, r*r - boundssqr[0]));
    float d = BoundaryDistance(radius_bounds[1], r, mu);
    float d_min = radius_bounds[1] - r;
    float d_max = rho + H;
    float u = (d - d_min) / (d_max - d_min);
    float v = rho / H;
    return vec2(u, v);
}

vec3 DirectIrradiance(float r, float mu)
{
    float sun_edge = mu + atmos.sun_angular_radius;
    float average_cosine_factor =
        max(0.0, min(mu, (sun_edge * sun_edge) / (4.0 * atmos.sun_angular_radius)));

    vec2 uv = TransmittanceRMutoUV(atmos.bounds, r, mu);

    return atmos.sun_irradiance * average_cosine_factor * texture(trtex, uv).xyz;
}

void main()
{
    vec2 uv = vec2(gl_FragCoord) / viewport.resolution;
    vec2 rmu = IrradianceUVtoRMu(atmos.bounds, uv);
    delta_irradiance = DirectIrradiance(rmu.x, rmu.y);
}

)__lstr__"
