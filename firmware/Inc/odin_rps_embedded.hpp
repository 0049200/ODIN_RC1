/**
 * @file odin_rps_embedded.hpp
 * @brief ODIN-RPS (Runtime Protection, Policy Supervisor & Failsafe Recovery)
 *        Target: STM32G070RBTx (ARM Cortex-M0+, ISO C++17 freestanding, zero heap).
 */

#ifndef ODIN_RPS_EMBEDDED_HPP
#define ODIN_RPS_EMBEDDED_HPP

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cstring>

namespace odin::mod_rps {

enum class HealthState : uint8_t {
    NOMINAL   = 0u,
    DEGRADED  = 1u,
    SAFE_HOLD = 2u,
    LOCKOUT   = 3u
};

struct RuntimeTelemetry {
    uint32_t tick_id;
    float    timestamp_s;
    float    actuator_cmd;
    float    current_ma;
    float    temp_c;
    uint32_t wcet_cycles;
    uint8_t  crc_ok;
    uint8_t  watchdog_fed;
};

struct SupervisorOutput {
    float       authorized_cmd;
    HealthState state;
    uint8_t     clamped;
    uint8_t     tripped;
    uint8_t     recovered;
    uint32_t    fault_flags;
    uint32_t    crc32;
};

class OdinRuntimeSupervisor {
public:
    OdinRuntimeSupervisor() noexcept { reset(); }

    static uint32_t crc32_bytes(const void* data, size_t len) noexcept {
        const uint8_t* p = static_cast<const uint8_t*>(data);
        uint32_t crc = 0xFFFFFFFFu;
        for (size_t i = 0u; i < len; ++i) {
            crc ^= static_cast<uint32_t>(p[i]);
            for (uint8_t b = 0u; b < 8u; ++b) {
                const uint32_t mask = -(crc & 1u);
                crc = (crc >> 1u) ^ (0xEDB88320u & mask);
            }
        }
        return ~crc;
    }

    void reset() noexcept {
        state_ = HealthState::NOMINAL;
        last_cmd_ = 0.0f;
        last_ts_ = -1.0f;
        max_cmd_ = 1.0f;
        max_slew_ = 0.25f;
        max_current_ma_ = 1500.0f;
        max_temp_c_ = 85.0f;
        max_wcet_cycles_ = 120000u;
        consecutive_healthy_ = 0u;
        total_trips_ = 0u;
        total_recoveries_ = 0u;
        ticks_processed_ = 0u;
    }

    void configure_limits(float max_cmd, float max_slew, float max_ma, float max_temp, uint32_t max_wcet) noexcept {
        if (std::isfinite(max_cmd) && max_cmd > 0.0f) max_cmd_ = max_cmd;
        if (std::isfinite(max_slew) && max_slew > 0.0f) max_slew_ = max_slew;
        if (std::isfinite(max_ma) && max_ma > 0.0f) max_current_ma_ = max_ma;
        if (std::isfinite(max_temp) && max_temp > 0.0f) max_temp_c_ = max_temp;
        if (max_wcet > 0u) max_wcet_cycles_ = max_wcet;
    }

    SupervisorOutput supervise(const RuntimeTelemetry& rt) noexcept {
        SupervisorOutput out{};
        ++ticks_processed_;

        uint32_t flags = 0u;
        if (!std::isfinite(rt.timestamp_s) || rt.timestamp_s <= last_ts_) flags |= (1u << 0u);
        if (!std::isfinite(rt.actuator_cmd) || !std::isfinite(rt.current_ma) || !std::isfinite(rt.temp_c)) flags |= (1u << 1u);
        if (rt.crc_ok == 0u) flags |= (1u << 2u);
        if (rt.watchdog_fed == 0u) flags |= (1u << 3u);
        if (std::isfinite(rt.current_ma) && rt.current_ma > max_current_ma_) flags |= (1u << 4u);
        if (std::isfinite(rt.temp_c) && rt.temp_c > max_temp_c_) flags |= (1u << 5u);
        if (rt.wcet_cycles > max_wcet_cycles_) flags |= (1u << 6u);

        if (std::isfinite(rt.timestamp_s) && rt.timestamp_s > last_ts_) {
            last_ts_ = rt.timestamp_s;
        }

        if (flags != 0u) {
            consecutive_healthy_ = 0u;
            out.tripped = 1u;
            ++total_trips_;
            if ((flags & ((1u << 1u) | (1u << 2u) | (1u << 4u) | (1u << 5u))) != 0u) {
                state_ = HealthState::SAFE_HOLD;
                last_cmd_ = 0.0f;
            } else if (state_ == HealthState::NOMINAL) {
                state_ = HealthState::DEGRADED;
            }
        } else {
            ++consecutive_healthy_;
            if (state_ == HealthState::SAFE_HOLD && consecutive_healthy_ >= 5u) {
                state_ = HealthState::DEGRADED;
                out.recovered = 1u;
                ++total_recoveries_;
            } else if (state_ == HealthState::DEGRADED && consecutive_healthy_ >= 10u) {
                state_ = HealthState::NOMINAL;
                out.recovered = 1u;
                ++total_recoveries_;
            }
        }

        float cmd = std::isfinite(rt.actuator_cmd) ? rt.actuator_cmd : 0.0f;
        if (state_ == HealthState::SAFE_HOLD || state_ == HealthState::LOCKOUT) {
            cmd = 0.0f;
            out.clamped = 1u;
        } else {
            const float eff_limit = (state_ == HealthState::DEGRADED) ? (0.5f * max_cmd_) : max_cmd_;
            if (cmd > eff_limit) { cmd = eff_limit; out.clamped = 1u; }
            if (cmd < -eff_limit) { cmd = -eff_limit; out.clamped = 1u; }

            const float delta = cmd - last_cmd_;
            if (delta > max_slew_) { cmd = last_cmd_ + max_slew_; out.clamped = 1u; }
            if (delta < -max_slew_) { cmd = last_cmd_ - max_slew_; out.clamped = 1u; }
        }

        last_cmd_ = cmd;
        out.authorized_cmd = cmd;
        out.state = state_;
        out.fault_flags = flags;
        out.crc32 = crc32_bytes(&out, offsetof(SupervisorOutput, crc32));
        return out;
    }

    void force_lockout() noexcept {
        state_ = HealthState::LOCKOUT;
        last_cmd_ = 0.0f;
    }

    bool verify_output(const SupervisorOutput& out) const noexcept {
        return out.crc32 == crc32_bytes(&out, offsetof(SupervisorOutput, crc32));
    }

    HealthState state() const noexcept { return state_; }
    uint32_t total_trips() const noexcept { return total_trips_; }
    uint32_t total_recoveries() const noexcept { return total_recoveries_; }
    uint32_t ticks_processed() const noexcept { return ticks_processed_; }
    size_t bytes() const noexcept { return sizeof(*this); }

private:
    HealthState state_ = HealthState::NOMINAL;
    float       last_cmd_ = 0.0f;
    float       last_ts_ = -1.0f;
    float       max_cmd_ = 1.0f;
    float       max_slew_ = 0.25f;
    float       max_current_ma_ = 1500.0f;
    float       max_temp_c_ = 85.0f;
    uint32_t    max_wcet_cycles_ = 120000u;
    uint32_t    consecutive_healthy_ = 0u;
    uint32_t    total_trips_ = 0u;
    uint32_t    total_recoveries_ = 0u;
    uint32_t    ticks_processed_ = 0u;
};

} // namespace odin::mod_rps

#endif
