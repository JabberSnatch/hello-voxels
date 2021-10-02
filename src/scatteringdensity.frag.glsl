#version 430 core

layout(location = 0) out vec3 scattering;

uniform sampler2D trtex;
uniform sampler3D drsctex;
uniform sampler3D dmsctex;
uniform sampler3D msctex;
uniform sampler2D dirtex;

uniform float scatorder;
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

float ExtBoundaryDistance(float r, float mu)
{
    float delta = r*r * (mu*mu - 1.0) + atmos.bounds[1]*atmos.bounds[1];
    return max(-r * mu + sqrt(max(delta, 0.0)), 0.0);
}

float IntBoundaryDistance(float r, float mu)
{
    float delta = r*r * (mu*mu - 1.0) + atmos.bounds[0]*atmos.bounds[0];
    return max(-r * mu - sqrt(max(delta, 0.0)), 0.0);
}

bool RayHitsGround(float r, float mu)
{
    return (mu < 0.0) && (r*r * (mu*mu - 1.0) + atmos.bounds[0]*atmos.bounds[0] >= 0.0);
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

vec3 SampleTransmittance(float r, float mu, float t, bool ray_hits_ground)
{
    float rt = clamp(sqrt(t*t + 2.0*r*mu*t + r*r),
                     atmos.bounds[0], atmos.bounds[1]);
    float mut = clamp((r*mu + t) / rt, -1.0, 1.0);

    if (ray_hits_ground)
    {
        vec3 tr0 = texture(trtex,
                           TransmittanceRMutoUV(rt, -mut)).xyz;
        vec3 tr1 = texture(trtex,
                           TransmittanceRMutoUV(r, -mu)).xyz;

        return min((tr0 / tr1), vec3(1.0));
    }
    else
    {
        vec3 tr0 = texture(trtex,
                           TransmittanceRMutoUV(rt, mut)).xyz;
        vec3 tr1 = texture(trtex,
                           TransmittanceRMutoUV(r, mu)).xyz;

        return min((tr1 / tr0), vec3(1.0));
    }
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
    float pixelSpace_x = texCoords.x * (viewport.nu - 1.0);
    float fragCoord_x = floor(pixelSpace_x);
    float interp = fract(pixelSpace_x);
    vec3 uvw0 = vec3((fragCoord_x + texCoords.y) / viewport.nu, texCoords.z, texCoords.w);
    vec3 uvw1 = vec3((fragCoord_x + 1.0 + texCoords.y) / viewport.nu, texCoords.z, texCoords.w);
    return mix(texture(lut, uvw0).xyz, texture(lut, uvw1).xyz, interp);
}

vec2 IrradianceRMutoUV(vec2 rmu)
{
    float u = rmu.y * 0.5 + 0.5;
    float v = (rmu.x - atmos.bounds[0]) / (atmos.bounds[1] - atmos.bounds[0]);
    return vec2(u, v);
}

vec3 SampleIrradiance(vec2 uv)
{
    return texture(dirtex, uv).xyz;
}

void main()
{
    vec4 texCoords = vec4(floor(float(gl_FragCoord.x) / viewport.mus),
                          mod(float(gl_FragCoord.x), viewport.mus),
                          gl_FragCoord.y,
                          layer + 0.5) /
        vec4(viewport.nu - 1, viewport.mus, viewport.muv, viewport.r);

    vec4 rmuvmusnu = ScatteringTexCoordstoRMuVMuSNu(texCoords);

    float r = rmuvmusnu.x;
    float muv = rmuvmusnu.y;
    float mus = rmuvmusnu.z;
    float nu = rmuvmusnu.w;

    vec3 viewdir = vec3(sqrt(1.0 - muv*muv), 0.0, muv);
    float sundx = (abs(viewdir.x) > 0.0001) ? (nu - muv*mus) / viewdir.x : 0.0;
    vec3 sundir = vec3(sundx,
                       sqrt(max(1.0 - sundx*sundx - mus*mus, 0.0)),
                       mus);

    const float kSampleCount = 16.0;
    float dphi = kPi / kSampleCount;
    float dtheta = kPi / kSampleCount;

    vec3 scatdensity = vec3(0.0);

    for (float i = 0.0; i < kSampleCount; i += 1.0)
    {
        float theta = (i + 0.5) * dtheta;
        float cos_theta = cos(theta);
        float sin_theta = sin(theta);
        bool r_theta_hits_ground = RayHitsGround(r, cos_theta);

        float groundhit_t = 0.0;
        vec3 tr_to_ground = vec3(0.0);
        vec3 ground_albedo = vec3(0.0);

        if (r_theta_hits_ground) {
            groundhit_t = IntBoundaryDistance(r, cos_theta);
            tr_to_ground = SampleTransmittance(r, cos_theta, groundhit_t, true);
            ground_albedo = vec3(0.1);
        }

        for (float j = 0.0; j < 2.0 * kSampleCount; j += 1.0)
        {
            float phi = j + 0.5 * dphi;
            vec3 sampledir = vec3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);

            float nu0 = dot(sundir, sampledir);
            vec3 Li = vec3(0.0);
            if (scatorder == 2.0)
            {
                vec3 r = SampleScattering(drsctex, texCoords);
                vec3 m = SampleScattering(dmsctex, texCoords);
                Li = r * RayleighPhaseFunction(nu0) + m * MiePhaseFunction(kMieG, nu0);
            }
            else
            {
                Li = SampleScattering(msctex, texCoords);
            }

            if (r_theta_hits_ground)
            {
                vec3 ground_normal = normalize(vec3(0.0, 0.0, r) + sampledir * groundhit_t);
                vec2 uv = IrradianceRMutoUV(vec2(atmos.bounds[0], dot(ground_normal, sundir)));
                vec3 ground_irradiance = SampleIrradiance(uv);
                Li += tr_to_ground * ground_albedo * (1.0/kPi) * ground_irradiance;
            }

            float nu1 = dot(viewdir, sampledir);
            float altitude = r - atmos.bounds[0];
            float rayleigh_density = exp(atmos.rscat.w * altitude);
            float mie_density = exp(atmos.mext.w * altitude);

            float dsampledir = dtheta * dphi * sin_theta;

            scatdensity += Li * (atmos.rscat.xyz * rayleigh_density * RayleighPhaseFunction(nu1)
                                 + atmos.mext.xyz * mie_density * MiePhaseFunction(kMieG, nu1))
                                 * dsampledir;
        }
    }

    scattering = scatdensity;
}
