/**
 * @file odin_s_embedded.hpp
 * @brief ODIN-S (State Estimation, Multi-Channel Sensor Fusion & Innovation Gate)
 *        Target: STM32G070RBTx (ARM Cortex-M0+, ISO C++17 freestanding, zero heap).
 */

#ifndef ODIN_S_EMBEDDED_HPP
#define ODIN_S_EMBEDDED_HPP

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cstring>

namespace odin::mod_s {

constexpr size_t kChannels = 4u;

struct SensorFrame {
    uint32_t frame_id;
    float    timestamp_s;
    float    primary[kChannels];
    float    redundant[kChannels];
    uint8_t  valid_mask;
};

struct StateEstimate {
    float    x_hat[kChannels];
    float    v_hat[kChannels];
    float    innovation_rms;
    uint8_t  outlier_rejected_mask;
    uint8_t  degraded_sensor_mask;
    uint32_t crc32;
};

class OdinStateEstimator {
public:
    OdinStateEstimator() noexcept { reset(); }

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
        std::memset(x_hat_, 0, sizeof(x_hat_));
        std::memset(v_hat_, 0, sizeof(v_hat_));
        initialized_ = false;
        last_ts_ = -1.0f;
        alpha_ = 0.45f;
        beta_ = 0.12f;
        gate_sigma_ = 3.5f;
        frames_processed_ = 0u;
        outliers_caught_ = 0u;
        failovers_ = 0u;
    }

    void set_gains(float alpha, float beta, float gate_threshold) noexcept {
        if (std::isfinite(alpha) && alpha > 0.0f && alpha < 1.0f) alpha_ = alpha;
        if (std::isfinite(beta) && beta > 0.0f && beta < 1.0f) beta_ = beta;
        if (std::isfinite(gate_threshold) && gate_threshold > 0.1f) gate_sigma_ = gate_threshold;
    }

    StateEstimate update(const SensorFrame& f) noexcept {
        StateEstimate out{};
        if (!std::isfinite(f.timestamp_s) || f.timestamp_s <= last_ts_) {
            for (size_t c = 0u; c < kChannels; ++c) {
                out.x_hat[c] = x_hat_[c];
                out.v_hat[c] = v_hat_[c];
            }
            out.crc32 = crc32_bytes(&out, offsetof(StateEstimate, crc32));
            return out;
        }

        const float dt = initialized_ ? (f.timestamp_s - last_ts_) : 0.02f;
        const float safe_dt = (dt > 1e-4f && dt < 5.0f) ? dt : 0.02f;
        last_ts_ = f.timestamp_s;
        ++frames_processed_;

        float innov_sq_sum = 0.0f;
        for (size_t c = 0u; c < kChannels; ++c) {
            const bool p_ok = ((f.valid_mask & (1u << c)) != 0u) && std::isfinite(f.primary[c]);
            const bool r_ok = ((f.valid_mask & (1u << (c + 4u))) != 0u) && std::isfinite(f.redundant[c]);

            const float x_pred = initialized_ ? (x_hat_[c] + v_hat_[c] * safe_dt) : (p_ok ? f.primary[c] : (r_ok ? f.redundant[c] : 0.0f));
            float z_meas = x_pred;
            bool has_meas = false;

            if (p_ok && r_ok) {
                const float diff_pr = std::fabs(f.primary[c] - f.redundant[c]);
                if (diff_pr > gate_sigma_ && initialized_) {
                    const float ep = std::fabs(f.primary[c] - x_pred);
                    const float er = std::fabs(f.redundant[c] - x_pred);
                    z_meas = (ep <= er) ? f.primary[c] : f.redundant[c];
                    out.degraded_sensor_mask |= static_cast<uint8_t>(1u << c);
                    ++failovers_;
                } else {
                    z_meas = 0.6f * f.primary[c] + 0.4f * f.redundant[c];
                }
                has_meas = true;
            } else if (p_ok) {
                z_meas = f.primary[c];
                has_meas = true;
            } else if (r_ok) {
                z_meas = f.redundant[c];
                out.degraded_sensor_mask |= static_cast<uint8_t>(1u << c);
                ++failovers_;
                has_meas = true;
            } else {
                out.degraded_sensor_mask |= static_cast<uint8_t>(1u << c);
            }

            if (!initialized_ && has_meas) {
                x_hat_[c] = z_meas;
                v_hat_[c] = 0.0f;
            } else if (has_meas) {
                const float resid = z_meas - x_pred;
                if (std::fabs(resid) > gate_sigma_) {
                    x_hat_[c] = x_pred;
                    out.outlier_rejected_mask |= static_cast<uint8_t>(1u << c);
                    ++outliers_caught_;
                } else {
                    x_hat_[c] = x_pred + alpha_ * resid;
                    v_hat_[c] = v_hat_[c] + (beta_ / safe_dt) * resid;
                    innov_sq_sum += resid * resid;
                }
            } else {
                x_hat_[c] = x_pred;
            }

            out.x_hat[c] = x_hat_[c];
            out.v_hat[c] = v_hat_[c];
        }

        initialized_ = true;
        out.innovation_rms = std::sqrt(innov_sq_sum / static_cast<float>(kChannels));
        out.crc32 = crc32_bytes(&out, offsetof(StateEstimate, crc32));
        return out;
    }

    bool verify_estimate(const StateEstimate& est) const noexcept {
        return est.crc32 == crc32_bytes(&est, offsetof(StateEstimate, crc32));
    }

    uint32_t frames_processed() const noexcept { return frames_processed_; }
    uint32_t outliers_caught() const noexcept { return outliers_caught_; }
    uint32_t failovers() const noexcept { return failovers_; }
    size_t bytes() const noexcept { return sizeof(*this); }

private:
    float    x_hat_[kChannels]{};
    float    v_hat_[kChannels]{};
    float    last_ts_ = -1.0f;
    float    alpha_ = 0.45f;
    float    beta_ = 0.12f;
    float    gate_sigma_ = 3.5f;
    uint32_t frames_processed_ = 0u;
    uint32_t outliers_caught_ = 0u;
    uint32_t failovers_ = 0u;
    bool     initialized_ = false;
};

} // namespace odin::mod_s

#endif
