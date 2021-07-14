R"__lstr__(

#version 430 core

layout(location = 0) out vec3 delta_irradiance;
layout(location = 1) out vec3 irradiance;

uniform sampler3D drsctex;
uniform sampler3D dmsctex;
uniform sampler3D msctex;

uniform float scatorder;
uniform float nu_tex_size;

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
    float mus_min;
    float padding;

    vec4 rscat; // rgb + expo_scale
    vec4 mext; // rgb + expo_scale

    LinearLayer odensity[2];
    vec4 oext; // rgb + obound
} atmos;

const float kPi = 3.1415926536;
const float kMieG = 0.8;

float RayleighPhaseFunction(float cos_mu)
{
    return (3.0 * (1.0 + cos_mu*cos_mu)) / (16.0 * kPi);
}

float MiePhaseFunction(float g, float cos_mu)
{
    float gsqr = g*g;
    float t0 = 3.0 / (8.0 * kPi);
    float t1 = (1.0 - gsqr) / (2.0 + gsqr);
    float t2 = (1.0 + cos_mu*cos_mu) / pow(1.0 + gsqr - 2.0 * g * cos_mu, 1.5);
    return t0*t1*t2;
}

vec2 IrradianceUVtoRMu(vec2 uv)
{
    float r = atmos.bounds.x + uv.y * (atmos.bounds.y - atmos.bounds.x);
    float mu = min(1.0, max(-1.0, 2.0 * uv.x - 1.0));
    return vec2(r, mu);
}

float ExtBoundaryDistance(float r, float mu)
{
    float delta = r*r * (mu*mu - 1.0) + atmos.bounds[1]*atmos.bounds[1];
    return max(-r * mu + sqrt(max(delta, 0.0)), 0.0);
}

vec4 ScatteringRMuVMuSNutoTexCoords(float r, float muv, float mus, float nu,
                                    bool ray_hits_ground)
{
    vec2 boundssqr = atmos.bounds * atmos.bounds;
    float H = sqrt(boundssqr[1] - boundssqr[0]);
    float rho = sqrt(max(r*r - boundssqr[0], 0.0));
    float u_r = clamp(rho/H, 0.0, 1.0);

    float rmuv = r * muv;
    float delta = rmuv*rmuv - r*r + boundssqr[0];
    float u_muv = 0.0;
    if (ray_hits_ground)
    {
        float d = -rmuv - sqrt(max(delta, 0.0));
        float dmin = r - atmos.bounds[0];
        float dmax = rho;
        float tex = (dmax != dmin) ? (d - dmin) / (dmax - dmin) : 0.0;
        u_muv = 0.5 - 0.5 * tex;
    }
    else
    {
        float d = -rmuv - sqrt(max(delta + H*H, 0.0));
        float dmin = atmos.bounds[1] - r;
        float dmax = rho + H;
        float tex = (d - dmin) / (dmax - dmin);
        u_muv = 0.5 + 0.5 * tex;
    }

    float d = ExtBoundaryDistance(atmos.bounds[0], mus);
    float dmin = atmos.bounds[1] - atmos.bounds[0];
    float dmax = H;
    float a = (d - dmin) / (dmax - dmin);
    float D = ExtBoundaryDistance(atmos.bounds[0], atmos.mus_min);
    float A = (D - dmin) / (dmax - dmin);
    float u_mus = max(1.0 - a / A, 0.0) / (1.0 + a);
    float u_nu = (nu + 1.0) / 2.0;

    return vec4(u_nu, u_mus, u_muv, u_r);
}

vec3 SampleScattering(sampler3D lut, vec4 texCoords)
{
    float pixelSpace_x = texCoords.x * (nu_tex_size - 1.0);
    float fragCoord_x = floor(pixelSpace_x);
    float interp = fract(pixelSpace_x);
    vec3 uvw0 = vec3((fragCoord_x + texCoords.y) / nu_tex_size, texCoords.z, texCoords.w);
    vec3 uvw1 = vec3((fragCoord_x + 1.0 + texCoords.y) / nu_tex_size, texCoords.z, texCoords.w);
    return mix(texture(lut, uvw0).xyz, texture(lut, uvw1).xyz, interp);
}

void main()
{
    vec2 rmu = IrradianceUVtoRMu(vec2(gl_FragCoord) / viewport.resolution);

    float r = rmu.x;
    float mus = rmu.y;

    const float kSampleCount = 32.0;
    const float dphi = kPi / kSampleCount;
    const float dtheta = kPi / kSampleCount;

    vec3 irradiance_sum = vec3(0.0);
    vec3 sundir = vec3(sqrt(1.0 - mus*mus), 0.0, mus);

    for (float j = 0.0; j < kSampleCount / 2.0; j += 1.0)
    {
        float theta = (j + 0.5) * dtheta;
        float dsampledir = dtheta * dphi * sin(theta);

        for (float i = 0.0; i < kSampleCount * 2.0; i += 1.0)
        {
            float phi = (i + 0.5) * dphi;
            vec3 sampledir = vec3(cos(phi) * sin(theta),
                                  sin(phi) * sin(theta),
                                  cos(theta));

            float nu = dot(sampledir, sundir);
            float muv = sampledir.z;

            vec4 texCoords = ScatteringRMuVMuSNutoTexCoords(r, muv, mus, nu, false);

            vec3 Li = vec3(0.0);
            if (scatorder == 2.0)
            {
                vec3 r = SampleScattering(drsctex, texCoords);
                vec3 m = SampleScattering(dmsctex, texCoords);
                Li = r * RayleighPhaseFunction(nu) + m * MiePhaseFunction(kMieG, nu);
            }
            else
            {
                Li = SampleScattering(msctex, texCoords);
            }

            irradiance_sum += Li * muv * dsampledir;
        }
    }

    delta_irradiance = irradiance_sum;
    irradiance = delta_irradiance;
}

)__lstr__"
