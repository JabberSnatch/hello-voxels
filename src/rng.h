/*
 * ----------------------------------------------------------------------------
 * "THE BEER-WARE LICENSE" (Revision 42):
 * Samuel Bourasseau wrote this file. You can do whatever you want with this
 * stuff. If we meet some day, and you think this stuff is worth it, you can
 * buy me a beer in return.
 * ----------------------------------------------------------------------------
 */

#pragma once

#include <array>
#include <cstdint>
#include <limits>

namespace rng
{

using Bitcount_t = std::uint8_t;

std::uint32_t hash32(void const* d, std::size_t size, std::uint32_t salt = 0u);
float noise3(float const* pp);
inline std::uint32_t rotl32(std::uint32_t v, unsigned d);
inline std::uint32_t rotr32(std::uint32_t v, unsigned d);
inline std::uint32_t xorsh32(std::uint32_t v, unsigned d);
std::uint32_t xorsh32_inv(std::uint32_t v, unsigned d);
inline float hash2unit(std::uint32_t v);

template <Bitcount_t logD>
struct PCG
{
    static constexpr Bitcount_t     kLog2Dimension = logD;
    static constexpr size_t         kDimensionCount = (1u << kLog2Dimension);
    static constexpr uint64_t   kDimensionMask = kDimensionCount - 1u;

    PCG(std::uint64_t _seed = 0u);

    // kDimensionCount_ sized arrays as optional input
    // returns first value
    float get_float(float* d = nullptr);
    std::uint32_t get_uint(std::uint32_t* d = nullptr);

    using ExtensionArray_t = std::array<uint32_t, kDimensionCount>;
    std::uint64_t state_;
    ExtensionArray_t extension_;
};


inline std::uint32_t rotl32(std::uint32_t v, unsigned d)
{
    return (v << d) | (v >> (32u - d));
}

inline std::uint32_t rotr32(std::uint32_t v, unsigned d)
{
    return (v >> d) | (v << (32u - d));
}

inline uint32_t xorsh32(std::uint32_t v, unsigned d)
{
    return v ^ (v >> d);
}

inline float hash2unit(std::uint32_t v)
{
    constexpr float kFactor =
        (1.f - (.5f * std::numeric_limits<float>::epsilon())) /
        static_cast<float>(std::numeric_limits<uint32_t>::max());
    float res = static_cast<float>(v) * kFactor;
    return res;
}

} // namespace rng
