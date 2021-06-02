/*
 * ----------------------------------------------------------------------------
 * "THE BEER-WARE LICENSE" (Revision 42):
 * Samuel Bourasseau wrote this file. You can do whatever you want with this
 * stuff. If we meet some day, and you think this stuff is worth it, you can
 * buy me a beer in return.
 * ----------------------------------------------------------------------------
 */

#include "rng.h"
#include "numtk.h"

#include <cmath>
#include <tuple>

namespace rng {

// murmur_x86_32
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
    constexpr float kPhi = 1.f;//1.6180339887f;
    constexpr float kInvPhi = 1.f;//1.f / kPhi;
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

#if 1
    std::uint64_t const svhash[4]{
        (std::uint64_t)hash32(&sv[0], 12u),
        (std::uint64_t)hash32(&sv[1], 12u),
        (std::uint64_t)hash32(&sv[2], 12u),
        (std::uint64_t)hash32(&sv[3], 12u)
    };

    PCG<0u> pcg[4]{
        PCG<0u>{ svhash[0] },
        PCG<0u>{ svhash[1] },
        PCG<0u>{ svhash[2] },
        PCG<0u>{ svhash[3] }
    };

    numtk::Vec3_t const hashed[4]{
        numtk::vec3_normalise({ pcg[0].get_float(), pcg[0].get_float(), pcg[0].get_float() }),
        numtk::vec3_normalise({ pcg[1].get_float(), pcg[1].get_float(), pcg[1].get_float() }),
        numtk::vec3_normalise({ pcg[2].get_float(), pcg[2].get_float(), pcg[2].get_float() }),
        numtk::vec3_normalise({ pcg[3].get_float(), pcg[3].get_float(), pcg[3].get_float() })
    };

#else

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
#endif

    return (numtk::vec3_dot(hashed[0], d[0]) * weights[0] +
            numtk::vec3_dot(hashed[1], d[1]) * weights[1] +
            numtk::vec3_dot(hashed[2], d[2]) * weights[2] +
            numtk::vec3_dot(hashed[3], d[3]) * weights[3])
        * 16.f;
}

std::uint32_t xorsh32_inv_rec(std::uint32_t v, std::size_t bitcount, unsigned d)
{
    if (2 * d >= bitcount)
        return xorsh32(v, d);
    std::uint32_t const lowmask1 = (1u << (bitcount - d * 2)) - 1u;
    std::uint32_t const highmask1 = ~lowmask1;
    std::uint32_t const bottom1 = v & lowmask1;
    std::uint32_t const top = xorsh32(v, d) & highmask1;
    std::uint32_t const combined1 = top | bottom1;
    std::uint32_t const lowmask2 = (1u << (bitcount - d)) - 1u;
    std::uint32_t const bottom2 = combined1 & lowmask2;
    return top | (lowmask1 & xorsh32_inv_rec(bottom2, bitcount - d, d));
}

std::uint32_t xorsh32_inv(std::uint32_t v, unsigned d)
{
    return xorsh32_inv_rec(v, 32u, d);
}

template <Bitcount_t logD>
struct PCG_xsh_rr : private PCG<logD>
{
    static constexpr size_t         kDimensionCount = PCG<logD>::kDimensionCount;
    static constexpr uint64_t   kDimensionMask = PCG<logD>::kDimensionMask;

    static constexpr std::uint64_t  kMultiplier = 6364136223846793005ull;
    static constexpr std::uint64_t  kIncrement = 1442695040888963407ull;

    static constexpr Bitcount_t     kOutputSize_ = 32u;
    static constexpr Bitcount_t     kInputSize_ = 64u;
    static constexpr Bitcount_t     kSpareSize_ = kInputSize_ - kOutputSize_;

    static constexpr Bitcount_t kLog2Advance = 16u;
    static constexpr std::uint64_t  kTickMask_ = (1ull << kLog2Advance) - 1ull;

    struct xsh_rr
    {
        static constexpr Bitcount_t     kOpcodeSize = 5; // depends on input size
        static constexpr Bitcount_t     kOpcodeMask = (1u << kOpcodeSize) - 1u;
        static constexpr Bitcount_t     kBottomSpare = kSpareSize_ - kOpcodeSize;
        static constexpr Bitcount_t     kInputShift = (kOpcodeSize + 32u) / 2;

        static std::uint32_t Apply(std::uint64_t _input)
        {
            const Bitcount_t output_rotation =
                static_cast<Bitcount_t>(_input >> (kInputSize_ - kOpcodeSize)) & kOpcodeMask;
            const uint64_t xorshifted_input = _input ^ (_input >> kInputShift);
            const uint32_t truncated_input = static_cast<uint32_t>(xorshifted_input >> kBottomSpare);
            return rotr32(truncated_input, output_rotation);
        }
    };

    struct rxs_m_xs
    {
        using BoolUint32Pair_t = std::tuple<bool, std::uint32_t>;

        static BoolUint32Pair_t InverseStep(std::uint32_t v, std::size_t _i)
        {
            const uint32_t state = Inverse(v) * kMultiplier + kIncrement
                + static_cast<uint32_t>(_i * 2);
            const uint32_t result = Apply(state);
            const bool is_zero = (state & 3u);
            return { is_zero, result };
        }

        static std::uint32_t Apply(std::uint32_t _input)
        {
            const Bitcount_t random_shift =
                static_cast<Bitcount_t>(_input >> (32u - kOpcodeSize)) & kOpcodeMask;
            const uint32_t xorshifted_input = xorsh32(_input, kOpcodeSize + random_shift);
            const uint32_t multiplied_rxs_input = xorshifted_input * kMcgMultiplier;
            return xorsh32(multiplied_rxs_input, kXsShift);
        }

        static std::uint32_t Inverse(std::uint32_t v)
        {
            const uint32_t ixs_output = xorsh32_inv(v, kXsShift);
            const uint32_t im_ixs_output = ixs_output * kInvMcgMultiplier;
            const Bitcount_t random_shift =
                static_cast<Bitcount_t>((im_ixs_output >> (32u - kOpcodeSize)) & kOpcodeMask);
            return xorsh32_inv(im_ixs_output, kOpcodeSize + random_shift);
        }

        static constexpr std::uint32_t kIncrement = 2891336453u;
        static constexpr std::uint32_t kMultiplier = 747796405u;

        static constexpr std::uint32_t kMcgMultiplier = 277803737u;
        static constexpr std::uint32_t kInvMcgMultiplier = 2897767785u;

        static constexpr Bitcount_t kOpcodeSize = 4; // depends on input size
        static constexpr Bitcount_t kOpcodeMask = (1u << kOpcodeSize) - 1u;

        static constexpr Bitcount_t kXsShift = (2u * 32u + 2u) / 3u;
    };

    std::uint32_t GeneratorValue_()
    {
        std::uint64_t const current_state = this->state_;
        this->state_ = this->state_ * kMultiplier + kIncrement;
        return xsh_rr::Apply(current_state);
    }

    std::uint32_t ExtensionValue_()
    {
        std::size_t const extension_index = static_cast<std::size_t>(this->state_ & kDimensionMask);
        bool const does_tick = ((this->state_ & kTickMask_) == 0u);
        if (does_tick)
            AdvanceExtension_();
        return this->extension_[extension_index];
    }

    void AdvanceExtension_()
    {
        bool carry = false;
        for (unsigned index = 0u; index < kDimensionCount; ++index)
        {
            uint32_t& ext = this->extension_[index];
            if (carry)
            {
                typename rxs_m_xs::BoolUint32Pair_t const step_result =
                    rxs_m_xs::InverseStep(ext, index + 1u);
                carry = std::get<0>(step_result);
                ext = std::get<1>(step_result);
            }

            typename rxs_m_xs::BoolUint32Pair_t const step_result =
                rxs_m_xs::InverseStep(ext, index + 1u);
            ext = std::get<1>(step_result);
            carry = carry || std::get<0>(step_result);
        }
    }

    std::uint32_t next32b()
    {
        std::uint32_t const rhs = ExtensionValue_();
        std::uint32_t const lhs = GeneratorValue_();
        return lhs ^ rhs;
    }
};

template <Bitcount_t logD>
PCG<logD>::PCG(std::uint64_t _seed) :
    state_{
        (_seed + PCG_xsh_rr<logD>::kIncrement) * PCG_xsh_rr<logD>::kMultiplier
        + PCG_xsh_rr<logD>::kIncrement
    },
    extension_{}
{
    std::uint32_t const xdiff =
        ((PCG_xsh_rr<logD>*)this)->GeneratorValue_()
        - ((PCG_xsh_rr<logD>*)this)->GeneratorValue_();
    for (std::size_t index = 0u; index < kDimensionCount; ++index)
        extension_[index] = ((PCG_xsh_rr<logD>*)this)->GeneratorValue_() ^ xdiff;
}

template <Bitcount_t logD>
float PCG<logD>::get_float(float* d)
{
    if (d)
    {
        for (std::size_t index = 0u; index < kDimensionCount; ++index)
            d[index] =  hash2unit(((PCG_xsh_rr<logD>*)this)->next32b());
        return d[0];
    }
    else
    {
        float res = hash2unit(((PCG_xsh_rr<logD>*)this)->next32b());
        for (std::size_t index = 1u; index < kDimensionCount; ++index)
            ((PCG_xsh_rr<logD>*)this)->next32b();
        return res;
    }
}

template <Bitcount_t logD>
std::uint32_t PCG<logD>::get_uint(std::uint32_t* d)
{
    if (d)
    {
        for (std::size_t index = 0u; index < kDimensionCount; ++index)
            d[index] = ((PCG_xsh_rr<logD>*)this)->next32b();
        return d[0];
    }
    else
    {
        std::uint32_t res = ((PCG_xsh_rr<logD>*)this)->next32b();
        for (std::size_t index = 1u; index < kDimensionCount; ++index)
            ((PCG_xsh_rr<logD>*)this)->next32b();
        return res;
    }
}


template struct PCG<0u>;
template struct PCG<1u>;
template struct PCG<2u>;


} // namespace rng
