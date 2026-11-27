/**
 * @file odin_common.hpp
 * @brief Common bare-metal C++17 definitions for ODIN modules on STM32G070RB
 *        (Cortex-M0+, no FPU, 128 KB Flash, 36 KB SRAM, zero dynamic allocation).
 */
#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <algorithm>

namespace odin {

constexpr std::size_t MAX_CHANNELS = 4;

inline bool is_finite_f32(float v) noexcept {
    return std::isfinite(v);
}

inline float clamp_f32(float v, float lo, float hi) noexcept {
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}

// Fast insertion-sort median on small fixed buffers (no heap, deterministic WCET)
template <std::size_t N>
inline float median_fixed(const std::array<float, N>& src, std::size_t count) noexcept {
    if (count == 0) return 0.0f;
    if (count > N) count = N;
    std::array<float, N> buf{};
    for (std::size_t i = 0; i < count; ++i) {
        buf[i] = src[i];
    }
    for (std::size_t i = 1; i < count; ++i) {
        float key = buf[i];
        std::size_t j = i;
        while (j > 0 && buf[j - 1] > key) {
            buf[j] = buf[j - 1];
            --j;
        }
        buf[j] = key;
    }
    if ((count & 1u) != 0u) {
        return buf[count / 2u];
    }
    return 0.5f * (buf[(count / 2u) - 1u] + buf[count / 2u]);
}

} // namespace odin
