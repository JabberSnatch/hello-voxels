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
    return max(-r * mu + sign(boundary_radius) * sqrt(max(delta, 0.0)), 0.0);
}

float ExtBoundaryDistance(float r, float mu)
{
    float delta = r*r * (mu*mu - 1.0) + atmos.bounds[1]*atmos.bounds[1];
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
    float D = ExtBoundaryDistance(atmos.bounds[0], atmos.mus_min);
    float A = (D - d_min) / (d_max - d_min);
    float a = (A - texCoords.y * A) / (1.0 + texCoords.y * A);
    float d = d_min + min(a, A) * (d_max - d_min);

    float mus = (abs(d) < 0.0001)
        ? 1.0
        : clamp((Hsqr - d*d) / (2.0 * atmos.bounds[0] * d), -1.0, 1.0);

    float nu = texCoords.x * 2.0 - 1.0;
    nu = clamp(nu,
               muv * mus - sqrt((1.0 - muv * muv) * (1.0 - mus * mus)),
               muv * mus + sqrt((1.0 - muv * muv) * (1.0 - mus * mus)));

    return vec4(r, muv, mus, nu);
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
    vec4 texCoords = vec4(floor(float(gl_FragCoord.x) / viewport.mus),
                          mod(float(gl_FragCoord.x), viewport.mus),
                          gl_FragCoord.y,
                          layer + 0.5) /
        vec4(viewport.nu - 1, viewport.mus, viewport.muv, viewport.r);

    vec4 rmuvmusnu = ScatteringTexCoordstoRMuVMuSNu(texCoords);
    float ray_outbound = (texCoords.z < 0.5) ? -atmos.bounds[0] : atmos.bounds[1];

    const float kSampleCount = 64.0;
    float r = rmuvmusnu.x;
    float muv = rmuvmusnu.y;
    float mus = rmuvmusnu.z;
    float nu = rmuvmusnu.w;

    float dt = BoundaryDistance(ray_outbound, r, muv) / kSampleCount;
    vec3 rayleigh_sum = vec3(0.0);
    vec3 mie_sum = vec3(0.0);
    vec3 trs_sum = vec3(0.0);
    for (float i = 0.0; i <= kSampleCount; i += 1.0)
    {
        float weight = (i == 0.0 || i >= kSampleCount) ? 0.5 : 1.0;

        float t = i * dt;

        float rt = clamp(sqrt(t*t + 2.0*r*muv*t + r*r),
                         atmos.bounds[0], atmos.bounds[1]);

        // Transmittance(r, mu, t)
        float muvt = clamp((r * muv + t) / rt, -1.0, 1.0);
        vec3 tr0 = texture(trtex,
                           TransmittanceRMutoUV(rt, sign(ray_outbound)*muvt)).xyz;
        vec3 tr1 = texture(trtex,
                           TransmittanceRMutoUV(r, sign(ray_outbound)*muv)).xyz;

        vec3 tr = min((ray_outbound < 0) ? (tr0 / tr1) : (tr1 / tr0), vec3(1.0));

        // TransmittanceSun(r, mu)
        float must = clamp((r*mus + t*nu) / rt, -1.0, 1.0);
        float sin_theta = atmos.bounds[0] / rt;
        float cos_theta = -sqrt(max(1.0 - sin_theta*sin_theta, 0.0));
        vec3 trs = texture(trtex,
                           TransmittanceRMutoUV(rt, must)).xyz
            * smoothstep(-sin_theta * atmos.sun_angular_radius,
                         sin_theta * atmos.sun_angular_radius,
                         must - cos_theta);

        rayleigh_sum += weight * tr * trs * exp(atmos.rscat.w * (rt - atmos.bounds[0]));
        mie_sum += weight * tr * trs * exp(atmos.mext.w * (rt - atmos.bounds[0]));
    }

    rayleigh = rayleigh_sum * dt * atmos.sun_irradiance * atmos.rscat.xyz;
    mie = mie_sum * dt * atmos.sun_irradiance * atmos.mext.xyz;

    scattering = vec4(rayleigh, mie.r);
}
