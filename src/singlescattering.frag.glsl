R"__lstr__(

#version 430 core

layout(location = 0) out vec3 rayleigh;
layout(location = 1) out vec3 mie;
layout(location = 2) out vec4 scattering;

uniform sampler2D trtex;
uniform float layer;

layout(std140, binding = 0) uniform ViewportBlock
{
    float nu;
    float mus;
    float muv;
    float r;
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
    float mus_min;
    float padding;

    vec4 rscat; // rgb + expo_scale
    vec4 mext; // rgb + expo_scale

    LinearLayer odensity[2];
    vec4 oext; // rgb + obound

} atmos;


float BoundaryDistance(float boundary_radius, float r, float mu)
{
    float delta = r*r * (mu*mu - 1.0) + boundary_radius*boundary_radius;
    return max(-r * mu + sqrt(max(delta, 0.0)), 0.0);
}

vec4 ScatteringTexCoordstoRMuVMuSNu(vec4 texCoords)
{
    vec2 boundssqr = atmos.bounds * atmos.bounds;
    float Hsqr = boundssqr[1] - boundssqr[0];
    float H = sqrt(Hsqr);
    float rho = H * texCoords.w;
    float rhosqr = rho*rho;
    float r = sqrt(rhosqr + boundssqr[0]);
    float muv = 0.0;

    if (texCoords.z < 0.5)
    {
        float d_min = r - atmos.bounds[0];
        float d_max = rho;
        float d = d_min + (d_max - d_min) * (1.0 - 2.0 * texCoords.z);
        muv = (abs(d) < 0.0001)
            ? -1.0
            : clamp(-(rhosqr + d*d) / (2.0 * r * d), -1.0, 1.0);
    }
    else
    {
        float d_min = atmos.bounds[1] - r;
        float d_max = rho + H;
        float d = d_min + (d_max - d_min) * (2.0 * texCoords.z - 1.0);
        muv = (abs(d) < 0.0001)
            ? 1.0
            : clamp((Hsqr - rhosqr - d*d) / (2.0 * r * d), -1.0, 1.0);
    }

    float d_min = atmos.bounds[1] - atmos.bounds[0];
    float d_max = H;
    float D = BoundaryDistance(atmos.bounds[1], atmos.bounds[0], atmos.mus_min);
    float A = (D - d_min) / (d_max - d_min);
    float a = (A - texCoords.y * A) / (1.0 + texCoords.y * A);
    float d = d_min + min(a, A) * (d_max - d_min);

    float mus = (abs(d) < 0.0001)
        ? 1.0
        : clamp((Hsqr - d*d) / (2.0 * atmos.bounds[0] * d), -1.0, 1.0);

    float nu = texCoords.x * 2.0 - 1.0;

    return vec4(r, muv, mus, nu);
}

void main()
{
    vec4 texCoords = vec4(gl_FragCoord.x / viewport.mus,
                          mod(gl_FragCoord.x, viewport.mus),
                          gl_FragCoord.y,
                          layer + 0.5) /
        vec4(viewport.nu - 1, viewport.mus, viewport.muv, viewport.r);

    vec4 rmuvmusnu = ScatteringTexCoordstoRMuVMuSNu(texCoords);

    scattering = rmuvmusnu;
}

)__lstr__"
