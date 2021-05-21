#include "rng.h"
#include "numtk.h"

#include <cmath>

namespace rng {

std::uint32_t hash32(void const* d, std::size_t size, std::uint32_t salt)
{
    constexpr std::uint32_t c1 = 0xcc9e2d51u;
    constexpr std::uint32_t c2 = 0x1b873593u;

    std::uint8_t const* stream = (std::uint8_t const*)d;

    int const block_count = size / 4u;
    std::uint32_t const* blocks = (std::uint32_t const*)(stream + block_count * 4u);

    std::uint32_t h = salt;
    for (int i = -block_count; i; ++i)
    {
        std::uint32_t k = blocks[i];
        k *= c1;
        k = rotl32(k, 15u);
        k *= c2;

        h ^= k;
        h = rotl32(h, 13u);
        h = h * 5u + 0xe6546b64u;
    }

    std::uint8_t const* tail = (std::uint8_t const*)blocks;
    std::uint32_t k = 0u;

    switch (size & 3u)
    {
    case 3: k ^= (std::uint32_t)tail[2] << 16;
    case 2: k ^= (std::uint32_t)tail[1] << 8;
    case 1: k ^= (std::uint32_t)tail[0];
        k *= c1;
        k = rotl32(k, 15);
        k *= c2;
        h ^= k;
    }

    h ^= size;
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}

float noise3(float const* pp)
{
    constexpr float kPhi = 1.6180339887f;
    constexpr float kInvPhi = 1.f / kPhi;
    static float const f = ((std::sqrt(4.f) - 1.f) / 3.f) * kPhi;
    static float const g = ((1.f - 1.f/std::sqrt(4.f)) / 3.f) * kInvPhi;
    static numtk::Mat3_t const skew{
        kPhi + f, f, f,
        f, kPhi + f, f,
        f, f, kPhi + f
    };
    static numtk::Mat3_t const invskew{
        kInvPhi - g, -g, -g,
        -g, kInvPhi - g, -g,
        -g, -g, kInvPhi - g
    };

    numtk::Vec3_t const& p = *(numtk::Vec3_t const*)pp;
    numtk::Vec3_t const sp = numtk::mat3_vec3_mul(skew, p);
    numtk::Vec3_t const cell = numtk::vec3_unary_op(sp, floorf);
    numtk::Vec3_t const d0 = numtk::vec3_unary_op(sp, [](float v)->float{ return std::fmod(v, 1.f); });
    float const x0 = (d0[0] <= d0[1]) ? 1.f : 0.f;
    float const x1 = (d0[1] <= d0[2]) ? 1.f : 0.f;
    float const x2 = (d0[2] <= d0[0]) ? 1.f : 0.f;
    numtk::Vec3_t const s0{
        x2 * (1.f-x0),
        x0 * (1.f-x1),
        x1 * (1.f-x2)
    };

    numtk::Vec3_t const s1{
        std::min(1.f, 1.f + x2 - x0),
        std::min(1.f, 1.f + x0 - x1),
        std::min(1.f, 1.f + x1 - x2)
    };

    numtk::Vec3_t const sv[4]{
        cell,
        numtk::vec3_add(cell, s0),
        numtk::vec3_add(cell, s1),
        numtk::vec3_add(cell, numtk::vec3_constant(1.f))
    };

    numtk::Vec3_t const wv[4]{
        numtk::mat3_vec3_mul(invskew, sv[0]),
        numtk::mat3_vec3_mul(invskew, sv[1]),
        numtk::mat3_vec3_mul(invskew, sv[2]),
        numtk::mat3_vec3_mul(invskew, sv[3])
    };

    numtk::Vec3_t const d[4]{
        numtk::vec3_sub(p, wv[0]),
        numtk::vec3_sub(p, wv[1]),
        numtk::vec3_sub(p, wv[2]),
        numtk::vec3_sub(p, wv[3])
    };

    numtk::Vec4_t const sqrnorm{
        numtk::vec3_sqrnorm(d[0]),
        numtk::vec3_sqrnorm(d[1]),
        numtk::vec3_sqrnorm(d[2]),
        numtk::vec3_sqrnorm(d[3])
    };

    numtk::Vec4_t const weights = numtk::vec4_unary_op(sqrnorm,
        [](float v){
            return std::pow(std::max(0.f, .5f - v), 4.f);
        }
    );

    auto hash3 = [](numtk::Vec3_t const& v) {
        numtk::Vec3_t res = numtk::vec3_normalise({
            hash2unit(hash32(&v[0], 4u, 0xdeadbeefu)) * 2.f - 1.f,
            hash2unit(hash32(&v[1], 4u, 0xc0ffee87u)) * 2.f - 1.f,
            hash2unit(hash32(&v[2], 4u, 0xed832fabu)) * 2.f - 1.f
        });
        return res;
    };

    numtk::Vec3_t const hashed[4]{
        hash3(sv[0]),
        hash3(sv[1]),
        hash3(sv[2]),
        hash3(sv[3])
    };

    return (numtk::vec3_dot(hashed[0], d[0]) * weights[0] +
            numtk::vec3_dot(hashed[1], d[1]) * weights[1] +
            numtk::vec3_dot(hashed[2], d[2]) * weights[2] +
            numtk::vec3_dot(hashed[3], d[3]) * weights[3])
        * 16.f;
}

} // namespace rng
