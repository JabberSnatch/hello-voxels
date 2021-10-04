#version 430 core

#extension GL_ARB_bindless_texture : require

vec3 raydir_frommat(mat4 perspective_inverse, vec2 clip_coord)
{
    vec4 target = vec4(clip_coord, 1.0, 1.0);
    vec4 ray_direction = perspective_inverse * target;
    ray_direction = ray_direction / ray_direction.w;
    return normalize(ray_direction.xyz);
}

float maxc(vec3 v) {return max(max(v.x, v.y), v.z); }

uniform vec3 iWSCamPos;
uniform mat4 iInvProj;
uniform vec2 iResolution;

uniform float iExtent;

layout(bindless_sampler) uniform sampler3D iChunk;
uniform float iChunkExtent;
uniform vec3 iChunkLocalCamPos; // Normalized on chunk size

#define kPi 3.1415926536

#define SPHERICAL_HARMONICS
#ifdef SPHERICAL_HARMONICS
uniform float iSHBuffer_red[9];
uniform float iSHBuffer_green[9];
uniform float iSHBuffer_blue[9];
#endif

#define CLEARSKY_MODEL
#ifdef CLEARSKY_MODEL
uniform vec3 iSunDir;
layout(bindless_sampler) uniform sampler2D trtex;
layout(bindless_sampler) uniform sampler2D irrtex;
layout(bindless_sampler) uniform sampler3D sctex;

struct LinearLayer
{
    float linear;
    float constant;
    vec2 padding;
};

#define SCTEX_NU_SIZE 8.0

layout(std140, binding = 0) uniform AtmosphereBlock
{
    vec3 sun_irradiance;
    float sun_angular_radius;

    vec2 bounds;
    float mus_min;
    float padding;

    vec4 rscat; // rgb + expo_scale
    vec4 mext; // rgb + expo_scale
    vec4 mscat; // rgb + padding

    LinearLayer odensity[2];
    vec4 oext; // rgb + obound
} atmos;

float ExtBoundaryDistance(float r, float mu)
{
    float delta = r*r * (mu*mu - 1.0) + atmos.bounds[1]*atmos.bounds[1];
    return max(-r * mu + sqrt(max(delta, 0.0)), 0.0);
}

vec4 SampleScattering(sampler3D lut, vec4 texCoords)
{
    float pixelSpace_x = texCoords.x * (SCTEX_NU_SIZE - 1.0);
    float fragCoord_x = floor(pixelSpace_x);
    float interp = fract(pixelSpace_x);
    vec3 uvw0 = vec3((fragCoord_x + texCoords.y) / SCTEX_NU_SIZE, texCoords.z, texCoords.w);
    vec3 uvw1 = vec3((fragCoord_x + 1.0 + texCoords.y) / SCTEX_NU_SIZE, texCoords.z, texCoords.w);
    return mix(texture(lut, uvw0), texture(lut, uvw1), interp);
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

float RayleighPhaseFunction(float cos_mu)
{
    return (3.0 * (1.0 + cos_mu*cos_mu)) / (16.0 * kPi);
}

float MiePhaseFunction(float cos_mu)
{
    #define kMieG 0.8

    const float g = kMieG;
    float gsqr = g*g;
    float t0 = 3.0 / (8.0 * kPi);
    float t1 = (1.0 - gsqr) / (2.0 + gsqr);
    float t2 = (1.0 + cos_mu*cos_mu) / pow(1.0 + gsqr - 2.0 * g * cos_mu, 1.5);
    return t0*t1*t2;

    #undef kMieG
}

vec3 SkyRadiance(vec3 ro, vec3 rd, vec3 sundir, out vec3 tr)
{
    const float r = max(atmos.bounds[0], atmos.bounds[0] + ro.y);

    #define LAT (90.0 - 47.47) * 3.1415926536 / 180.0
    #define LONG -0.55968 * 3.1415926536 / 180.0
    const vec3 sphpos = r * vec3(0.0, 1.0, 0.0);
    //vec3(cos(LONG) * sin(LAT), cos(LAT), sin(LONG) * sin(LAT));
    #undef LAT
    #undef LONG

    const float muv = dot(sphpos, rd) / r;
    const float mus = dot(sphpos, sundir) / r;
    const float nu = dot(rd, sundir);

    // Transmittance to upper atmosphere boundary
    tr = texture(trtex, TransmittanceRMutoUV(r, muv)).xyz;
    vec4 texCoords = ScatteringRMuVMuSNutoTexCoords(r, muv, mus, nu, false);
    vec4 scattering_mie = SampleScattering(sctex, texCoords);
    vec3 rayleigh = scattering_mie.xyz;
    vec3 mie = vec3(0.0);
    if (rayleigh.x > 0.0)
    {
        mie = rayleigh * scattering_mie.w / rayleigh.x *
            (atmos.rscat.x / atmos.mscat.x) *
            (atmos.mscat.xyz / atmos.rscat.xyz);
    }

    //return vec3(fract(texCoords.z));
    return rayleigh * RayleighPhaseFunction(nu) + mie * MiePhaseFunction(nu);
}
#endif

layout(location = 0) out vec4 color;

void sh_second_order(vec3 w, out float v[9])
{
    const float kInvPi = 1.f / kPi;

    v[0] = .5 * sqrt(kInvPi);
    v[1] = w[1] * sqrt(.75f * kInvPi);
    v[2] = w[2] * sqrt(.75f * kInvPi);
    v[3] = w[0] * sqrt(.75f * kInvPi);
    v[4] = .5f * w[0]*w[2] * sqrt(15.f * kInvPi);
    v[5] = .5f * w[2]*w[1] * sqrt(15.f * kInvPi);
    v[6] = .25f * (-w[0]*w[0] - w[2]*w[2] + 2.f*w[1]*w[1]) * sqrt(5.f * kInvPi);
    v[7] = .5f * w[0]*w[1] * sqrt(15.f * kInvPi);
    v[8] = .25f * (w[0]*w[0] - w[2]*w[2]) * sqrt(15.f * kInvPi);
}

float sh_dot(float lhs[9], float rhs[9])
{
    return lhs[0] * rhs[0] +
        lhs[1] * rhs[1] +
        lhs[2] * rhs[2] +
        lhs[3] * rhs[3] +
        lhs[4] * rhs[4] +
        lhs[5] * rhs[5] +
        lhs[6] * rhs[6] +
        lhs[7] * rhs[7] +
        lhs[8] * rhs[8];
}

void main()
{
    vec2 frag_coord = vec2(gl_FragCoord.xy);
    vec2 clip_coord = ((frag_coord / iResolution) - 0.5) * 2.0;

    const vec3 rd = raydir_frommat(iInvProj, clip_coord);
    vec3 ro = iChunkLocalCamPos;// - vec3(0.5);

#if 0
    vec3 tr = vec3(0.0);
    color.xyz = SkyRadiance(iWSCamPos, rd, iSunDir, tr);
    return;
#endif

    vec3 n = vec3(0.0);

    #if 1
    if (ro.x < 0.0 || ro.x > 1.0
        || ro.y < 0.0 || ro.y > 1.0
        || ro.z < 0.0 || ro.z > 1.0)
    {
        ro = iChunkLocalCamPos - vec3(0.5);

        float winding = (maxc(abs(ro) * 2.0) < 1.0) ? -1.0 : 1.0;
        vec3 sgn = -sign(rd);
        vec3 d = (0.5 * winding * sgn - ro) / rd;

        #define TEST(U, V, W)                                           \
            (d.U >= 0.0) && all(lessThan(abs(vec2(ro.V, ro.W) + vec2(rd.V, rd.W)*d.U), vec2(0.5)))

        bvec3 test = bvec3(
            TEST(x, y, z),
            TEST(y, z, x),
            TEST(z, x, y));

        #undef TEST


        sgn = test.x ? vec3(sgn.x,0,0) : (test.y ? vec3(0,sgn.y,0) : vec3(0, 0,test.z ? sgn.z : 0));

        float distance = (sgn.x != 0) ? d.x : ((sgn.y != 0) ? d.y : d.z);
        //vec3 normal = sgn;
        //n = sgn;
        bool hit = (sgn.x != 0) || (sgn.y != 0) || (sgn.z != 0);

        // bounds intersection
        //color = (vec4(rd, 1.0));

        if (hit)
        {
            //color = vec4(normal * 0.5 + vec3(0.5), 1.0);
            //vec3 puvw = (ro + rd * distance * 1.0001) * 0.5 + vec3(0.5);
            //vec3 pvoxel = puvw * iChunkExtent;

            //gl_FragDepth = distance / 1000.f;
            //color.xyz = puvw;////vec3(distance*0.5);//texture(iChunk, puvw).xxx;
            ro = (ro + rd * distance * 1.0001) + vec3(0.5);
            n = sgn;
        }
        else
        {
            discard;
        }
    }
    else
    {
        //discard;
    }
    #endif

    #if 1

    vec3 stepSign = sign(rd);
    vec3 vsro = ro * iChunkExtent;
    vec3 p = floor(vsro);

    vec3 manh = stepSign * (vec3(0.5) - fract(vsro)) + 0.5;
    vec3 compv = (stepSign * 0.5 + vec3(0.5));

    vec3 tDelta = 1.f / abs(rd);
    vec3 tMax = manh * tDelta;
    vec3 uvwstep = stepSign / iChunkExtent;
    vec3 puvw = (p + vec3(0.5))  / iChunkExtent;
    vec3 start = puvw;

    //color.xyz = tMax;

    #if 1
    float accum = 0.f;
    float t = 0.f;

    while( (puvw.x * stepSign.x < compv.x)
           && (puvw.y * stepSign.y < compv.y)
           && (puvw.z * stepSign.z < compv.z))
    {
        accum += texture(iChunk, puvw).x;
        if (accum.x >= 1.0) break;

        if (tMax.x < tMax.y)
        {
            if (tMax.x < tMax.z)
            {
                puvw.x += uvwstep.x;
                tMax.x += tDelta.x;
                t += tDelta.x;
                n = -vec3(stepSign.x, 0.0, 0.0);
            }
            else
            {
                puvw.z += uvwstep.z;
                tMax.z += tDelta.z;
                t += tDelta.z;
                n = -vec3(0.0, 0.0, stepSign.z);
            }
        }

        else
        {
            if (tMax.y < tMax.z)
            {
                puvw.y += uvwstep.y;
                tMax.y += tDelta.y;
                t += tDelta.y;
                n = -vec3(0.0, stepSign.y, 0.0);
            }
            else
            {
                puvw.z += uvwstep.z;
                tMax.z += tDelta.z;
                t += tDelta.z;
                n = -vec3(0.0, 0.0, stepSign.z);
            }
        }

    }

    if (accum.x >= 1.0)
    {
        //color.xyz = mix(vec3(0.5), vec3(puvw * (0.1 / distance(puvw,start))), accum.x);
        //distance(puvw, start));//puvw * 0.5;// + vec3(0.5);
#ifdef SPHERICAL_HARMONICS
        float nsh[9];
        sh_second_order(n, nsh);
        vec3 Li = vec3(sh_dot(nsh, iSHBuffer_red),
                       sh_dot(nsh, iSHBuffer_green),
                       sh_dot(nsh, iSHBuffer_blue));
#endif

        //color.xyz = vec3(0.7, 0.6, 0.2) * vec3(exp(-distance(iChunkLocalCamPos, puvw) / 10.0)) * dot(n, vec3(0.5, 1.0, 0.5));
        vec3 voxel_index = floor(puvw) + vec3(16.0);

        vec3 groundcolor =
            vec3(0.2, 0.56862745098*0.8, 0.043137254902 * 2.0) * max(0.0, n[1]) * 0.5f+
            vec3(0.207843137255, 0.254901960784, 0.117647058824) * min(1.0, (abs(n[0]) + abs(n[2]))) +
            vec3(0.262745098039, 0.239215686275, 0.0627450980392) * abs(min(0.0, n[1])) +
            vec3(0.207843137255, 0.211764705882, 0.196078431373) * abs(min(0.0, n[1])) * 0.f;
        groundcolor *= 2.f;

        color.xyz = groundcolor;

#ifdef SPHERICAL_HARMONICS
        color.xyz = mix(vec3(0.36, 0.4, 0.58), Li, min(1.0, exp(1.f-distance(iChunkLocalCamPos, puvw) / 2.f)))
            * groundcolor;
#endif

        gl_FragDepth = distance(iChunkLocalCamPos, puvw) / 1000.f;
    }
    else
    {
#ifdef CLEARSKY_MODEL
        vec3 tr = vec3(0.0);
        color.xyz = SkyRadiance(iWSCamPos, rd, iSunDir, tr);
        gl_FragDepth = 0.99;
#else
        discard;
#endif
    }
    #endif
    #endif
}
