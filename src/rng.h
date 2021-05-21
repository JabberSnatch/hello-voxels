/*
 * ----------------------------------------------------------------------------
 * "THE BEER-WARE LICENSE" (Revision 42):
 * Samuel Bourasseau wrote this file. You can do whatever you want with this
 * stuff. If we meet some day, and you think this stuff is worth it, you can
 * buy me a beer in return.
 * ----------------------------------------------------------------------------
 */

#pragma once

#include <cstdint>
#include <limits>

namespace rng
{

// murmur32
std::uint32_t hash32(void const* d, std::size_t size, std::uint32_t salt = 0u);
float noise3(float const* pp);

inline std::uint32_t rotl32(std::uint32_t v, unsigned d)
{
    return (v << d) | (v >> (32u - d));
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
