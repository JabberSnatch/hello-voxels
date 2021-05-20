R"__lstr__(
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

uniform mat4 iInvProj;
uniform vec2 iResolution;

uniform float iExtent;

layout(bindless_sampler) uniform sampler3D iChunk;
uniform float iChunkExtent;
uniform vec3 iChunkLocalCamPos; // Normalized on chunk size

uniform float iSHBuffer_red[9];
uniform float iSHBuffer_green[9];
uniform float iSHBuffer_blue[9];

#if 0
layout(std140, binding = 0) uniform SHBlock
{
    float r[9];
    float g[9];
    float b[9];
} iSHBuffer;
#endif

layout(location = 0) out vec4 color;

void sh_second_order(vec3 w, out float v[9])
{
    const float kPi = 3.1415926535f;
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

    vec3 rd = raydir_frommat(iInvProj, clip_coord);
    vec3 ro = iChunkLocalCamPos;// - vec3(0.5);

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
        float nsh[9];
        sh_second_order(n, nsh);
        vec3 Li = vec3(sh_dot(nsh, iSHBuffer_red),
                       sh_dot(nsh, iSHBuffer_green),
                       sh_dot(nsh, iSHBuffer_blue));

        //color.xyz = vec3(0.7, 0.6, 0.2) * vec3(exp(-distance(iChunkLocalCamPos, puvw) / 10.0)) * dot(n, vec3(0.5, 1.0, 0.5));
        vec3 voxel_index = floor(puvw) + vec3(16.0);

        vec3 groundcolor =
            vec3(0.2, 0.56862745098, 0.043137254902) * max(0.0, n[1]) * 0.5f+
            vec3(0.207843137255, 0.254901960784, 0.117647058824) * min(1.0, (abs(n[0]) + abs(n[2]))) +
            vec3(0.262745098039, 0.239215686275, 0.0627450980392) * abs(min(0.0, n[1])) +
            vec3(0.207843137255, 0.211764705882, 0.196078431373) * abs(min(0.0, n[1])) * 0.f;
        //groundcolor *= 2.f;

        color.xyz = Li * exp(-distance(iChunkLocalCamPos, puvw) / 10.f);
color.xyz = color.xyz * groundcolor * 3.f;
        gl_FragDepth = distance(iChunkLocalCamPos, puvw) / 1000.f;
    }
    else
    {
        discard;
    }
    #endif
    #endif
}

)__lstr__"
