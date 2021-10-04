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
    vec4 mscat; // rgb + padding

    LinearLayer odensity[2];
    vec4 oext; // rgb + obound

} atmos;

vec2 IrradianceUVtoRMu(vec2 radius_bounds, vec2 uv)
{
    float r = radius_bounds.x + uv.y * (radius_bounds.y - radius_bounds.x);
    float mu = min(1.0, max(-1.0, 2.0 * uv.x - 1.0));
    return vec2(r, mu);
}

float ExtBoundaryDistance(float r, float mu)
{
    float delta = r*r * (mu*mu - 1.0) + atmos.bounds[1]*atmos.bounds[1];
    return max(-r * mu + sqrt(max(delta, 0.0)), 0.0);
}

vec2 TransmittanceRMutoUV(float r, float mu)
{
    vec2 boundssqr = atmos.bounds * atmos.bounds;
    float H = sqrt(boundssqr[1] - boundssqr[0]);
    float rho = sqrt(max(0.0, r*r - boundssqr[0]));
    float d = ExtBoundaryDistance(r, mu);
    float d_min = atmos.bounds[1] - r;
    float d_max = rho + H;
    float u = (d - d_min) / (d_max - d_min);
    float v = rho / H;
    return vec2(u, v);
}

void main()
{
    vec2 rmu = IrradianceUVtoRMu(atmos.bounds, vec2(gl_FragCoord) / viewport.resolution);

    float r = rmu.x;
    float mu = rmu.y;

    float sun_edge = mu + atmos.sun_angular_radius;
    float average_cosine_factor =
        max(0.0, min(mu, (sun_edge * sun_edge) / (4.0 * atmos.sun_angular_radius)));

    vec2 uv = TransmittanceRMutoUV(r, mu);

    delta_irradiance = atmos.sun_irradiance * average_cosine_factor * texture(trtex, uv).xyz;
}
