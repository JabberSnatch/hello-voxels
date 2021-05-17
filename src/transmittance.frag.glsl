R"__lstr__(

#version 430 core

layout(location = 0) out vec3 transmittance;

layout(std140, binding = 0) uniform ViewportBlock
{
    vec2 resolution;
} viewport;


struct LinearLayer
{
    float linear;
    float constant;
};

layout(std140, binding = 1) uniform AtmosphereBlock
{
    vec3 sun_irradiance;
    float sun_angular_radius;

    vec2 bounds;

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

float OpticalLengthExpo(float scale, float r, float mu)
{
    const float kSampleCount = 512.0;
    float dx = BoundaryDistance(atmos.bounds[1], r, mu) / kSampleCount;
    float result = 0.0;
    for (float i = 0.0; i <= kSampleCount; i += 1.0)
    {
        float t = i * dx;
        float ri = sqrt(t*t + 2.0*r*mu*t + r*r) - atmos.bounds[0];
        float di = exp(scale * ri);
        float wi = (i != 0.0 && i != kSampleCount) ? 1.0 : 0.5;
        result += di * wi * dx;
    }
    return result;
}

float OpticalLengthLinear(LinearLayer density[2], float bound, float r, float mu)
{
    const float kSampleCount = 512.0;
    float dx = BoundaryDistance(atmos.bounds[1], r, mu) / kSampleCount;
    float result = 0.0;
    for (float i = 0.0; i <= kSampleCount; i += 1.0)
    {
        float t = i * dx;
        float ri = sqrt(t*t + 2.0*r*mu*t + r*r) - atmos.bounds[0];
        float di = (ri <= bound)
                ? density[0].linear * ri + density[0].constant
                : density[1].linear * ri + density[1].constant;
        float wi = (i != 0.0 && i != kSampleCount) ? 1.0 : 0.5;
        result += di * wi * dx;
    }
    return result;
}

vec3 Transmittance(float r, float mu)
{
    vec3 rayleigh_term = atmos.rscat.xyz * OpticalLengthExpo(atmos.rscat.w, r, mu);
    vec3 mie_term = atmos.mext.xyz * OpticalLengthExpo(atmos.mext.w, r, mu);
    vec3 absorp_term = atmos.oext.xyz * OpticalLengthLinear(atmos.odensity, atmos.oext.w, r, mu);
    return exp(-(rayleigh_term + mie_term + absorp_term));
}

vec2 TransmittanceUVtoRMu(vec2 radius_bounds, vec2 uv)
{
    vec2 uvsqr = radius_bounds * radius_bounds;
    float H = sqrt(uvsqr[1] - uvsqr[0]);
    float rho = H * uv[1];
    float rhosqr = rho * rho;
    float r = sqrt(rhosqr + uvsqr[0]);
    float d_min = radius_bounds[1] - r;
    float d_max = rho + H;
    float d = d_min + uv[0] * (d_max - d_min);
    float mu = (abs(d) < 0.0001)
        ? 1.0
        : clamp((uvsqr[1]-uvsqr[0] - rhosqr - d*d) / (2.0 * r * d), -1.0, 1.0);
    return vec2(r, mu);
}

void main() {
    vec2 uv = vec2(gl_FragCoord) / viewport.resolution;
    vec2 rmu = TransmittanceUVtoRMu(atmos.bounds, uv);
    transmittance = vec3(rmu.xy, 0.0);//Transmittance(rmu.x, rmu.y);
}

)__lstr__"
