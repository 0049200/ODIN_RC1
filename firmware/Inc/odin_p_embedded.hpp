/**
 * @file odin_p_embedded.hpp
 * @brief ODIN-P (Predictive & Planning Engine) — Standalone C++17 Freestanding
 *        Target: STM32G070RBTx (ARM Cortex-M0+, zero dynamic allocation).
 */

#ifndef ODIN_P_EMBEDDED_HPP
#define ODIN_P_EMBEDDED_HPP

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cstring>

namespace odin::mod_p {

constexpr size_t kStateDim       = 4u;
constexpr size_t kActionDim      = 2u;
constexpr size_t kHistoryDepth   = 16u;
constexpr size_t kHorizonSteps   = 4u;
constexpr size_t kCandidateCount = 4u;

struct StateSample {
    uint32_t step_id;
    float    timestamp_s;
    float    x[kStateDim];
    float    u[kActionDim];
};

struct HorizonPrediction {
    float    pred_x[kHorizonSteps][kStateDim];
    float    expected_error;
    float    regime_drift_score;
    bool     anomaly_warning;
};

struct PlanDecision {
    float    best_u[kActionDim];
    float    predicted_cost;
    float    safety_margin;
    uint8_t  selected_candidate;
    bool     safe;
};

struct PlannerSnapshot {
    uint32_t samples_seen;
    uint32_t alarms_raised;
    uint32_t plans_computed;
    uint32_t crc32;
    uint8_t  valid;
};

class OdinPredictorPlanner {
public:
    OdinPredictorPlanner() noexcept { reset(); }

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
        std::memset(history_, 0, sizeof(history_));
        std::memset(vel_, 0, sizeof(vel_));
        std::memset(err_ema_, 0, sizeof(err_ema_));
        head_ = 0u;
        count_ = 0u;
        samples_seen_ = 0u;
        alarms_raised_ = 0u;
        plans_computed_ = 0u;
        last_ts_ = -1.0f;
        safety_limit_ = 10.0f;
        has_prev_pred_ = false;
        std::memset(last_pred_x1_, 0, sizeof(last_pred_x1_));
    }

    void set_safety_limit(float limit) noexcept {
        if (std::isfinite(limit) && limit > 0.1f) {
            safety_limit_ = limit;
        }
    }

    float safety_limit() const noexcept { return safety_limit_; }

    bool ingest(const StateSample& s) noexcept {
        if (!std::isfinite(s.timestamp_s) || s.timestamp_s <= last_ts_) return false;
        for (size_t i = 0u; i < kStateDim; ++i) {
            if (!std::isfinite(s.x[i])) return false;
        }
        for (size_t j = 0u; j < kActionDim; ++j) {
            if (!std::isfinite(s.u[j])) return false;
        }

        const float dt = (last_ts_ >= 0.0f) ? (s.timestamp_s - last_ts_) : 0.05f;
        const float safe_dt = (dt > 1e-4f && dt < 10.0f) ? dt : 0.05f;

        if (has_prev_pred_) {
            for (size_t i = 0u; i < kStateDim; ++i) {
                const float e = std::fabs(s.x[i] - last_pred_x1_[i]);
                err_ema_[i] = 0.80f * err_ema_[i] + 0.20f * e;
            }
        }

        if (count_ > 0u) {
            const size_t prev_idx = (head_ + kHistoryDepth - 1u) % kHistoryDepth;
            const StateSample& prev = history_[prev_idx];
            for (size_t i = 0u; i < kStateDim; ++i) {
                const float inst_v = (s.x[i] - prev.x[i]) / safe_dt;
                vel_[i] = 0.65f * vel_[i] + 0.35f * inst_v;
            }
        }

        history_[head_] = s;
        head_ = (head_ + 1u) % kHistoryDepth;
        if (count_ < kHistoryDepth) ++count_;
        ++samples_seen_;
        last_ts_ = s.timestamp_s;

        // Precompute 1-step ahead reference for next observation error tracking
        const float step_dt = safe_dt;
        for (size_t i = 0u; i < kStateDim; ++i) {
            const float u_eff = (i < kActionDim) ? s.u[i] : s.u[i % kActionDim];
            last_pred_x1_[i] = s.x[i] + vel_[i] * step_dt + 0.15f * u_eff * step_dt;
        }
        has_prev_pred_ = true;
        return true;
    }

    HorizonPrediction predict_horizon(const float u_plan[kActionDim], float dt_step = 0.05f) noexcept {
        HorizonPrediction hp{};
        const float h = (std::isfinite(dt_step) && dt_step > 1e-4f) ? dt_step : 0.05f;

        float cur_x[kStateDim]{};
        float cur_v[kStateDim]{};
        if (count_ > 0u) {
            const size_t last_idx = (head_ + kHistoryDepth - 1u) % kHistoryDepth;
            for (size_t i = 0u; i < kStateDim; ++i) {
                cur_x[i] = history_[last_idx].x[i];
                cur_v[i] = vel_[i];
            }
        }

        float err_sum = 0.0f;
        float max_norm = 0.0f;
        for (size_t t = 0u; t < kHorizonSteps; ++t) {
            for (size_t i = 0u; i < kStateDim; ++i) {
                const float damp = 0.92f;
                cur_v[i] *= damp;
                const float u_term = (i < kActionDim) ? u_plan[i] : u_plan[i % kActionDim];
                cur_x[i] = cur_x[i] + cur_v[i] * h + 0.15f * u_term * h;
                hp.pred_x[t][i] = cur_x[i];
                err_sum += err_ema_[i];
                const float ax = std::fabs(cur_x[i]);
                if (ax > max_norm) max_norm = ax;
            }
        }

        hp.expected_error = err_sum / static_cast<float>(kHorizonSteps * kStateDim);
        hp.regime_drift_score = hp.expected_error + 0.10f * max_norm;
        hp.anomaly_warning = (hp.expected_error > 0.65f) || (max_norm > safety_limit_);
        if (hp.anomaly_warning) {
            ++alarms_raised_;
        }
        return hp;
    }

    PlanDecision plan_action(const float target_x[kStateDim], float dt_step = 0.05f) noexcept {
        PlanDecision best{};
        best.predicted_cost = 1e30f;
        best.safety_margin = -1e30f;
        best.selected_candidate = 0u;
        best.safe = false;

        const float candidates[kCandidateCount][kActionDim] = {
            { 0.0f,  0.0f},
            { 0.5f,  0.5f},
            {-0.5f, -0.5f},
            { 0.25f, -0.25f}
        };

        float cur_x[kStateDim]{};
        if (count_ > 0u) {
            const size_t last_idx = (head_ + kHistoryDepth - 1u) % kHistoryDepth;
            for (size_t i = 0u; i < kStateDim; ++i) cur_x[i] = history_[last_idx].x[i];
        }

        for (size_t c = 0u; c < kCandidateCount; ++c) {
            const HorizonPrediction hp = predict_horizon(candidates[c], dt_step);
            const float* final_x = hp.pred_x[kHorizonSteps - 1u];

            float tracking_err = 0.0f;
            float max_abs = 0.0f;
            for (size_t i = 0u; i < kStateDim; ++i) {
                const float d = final_x[i] - target_x[i];
                tracking_err += d * d;
                const float ax = std::fabs(final_x[i]);
                if (ax > max_abs) max_abs = ax;
            }
            const float effort = candidates[c][0] * candidates[c][0] + candidates[c][1] * candidates[c][1];
            const float margin = safety_limit_ - max_abs;
            const float barrier_penalty = (margin < 0.0f) ? (1000.0f * margin * margin) : (1.0f / (0.25f + margin));
            const float cost = tracking_err + 0.05f * effort + barrier_penalty;

            if (cost < best.predicted_cost) {
                best.predicted_cost = cost;
                best.safety_margin = margin;
                best.selected_candidate = static_cast<uint8_t>(c);
                best.best_u[0] = candidates[c][0];
                best.best_u[1] = candidates[c][1];
                best.safe = (margin >= 0.0f);
            }
        }

        ++plans_computed_;
        return best;
    }

    PlannerSnapshot make_snapshot() const noexcept {
        PlannerSnapshot snap{};
        snap.samples_seen = samples_seen_;
        snap.alarms_raised = alarms_raised_;
        snap.plans_computed = plans_computed_;
        snap.valid = 1u;
        snap.crc32 = crc32_bytes(&snap, offsetof(PlannerSnapshot, crc32));
        return snap;
    }

    bool verify_snapshot(const PlannerSnapshot& snap) const noexcept {
        if (!snap.valid) return false;
        return snap.crc32 == crc32_bytes(&snap, offsetof(PlannerSnapshot, crc32));
    }

    uint32_t samples_seen() const noexcept { return samples_seen_; }
    uint32_t alarms_raised() const noexcept { return alarms_raised_; }
    uint32_t plans_computed() const noexcept { return plans_computed_; }
    size_t bytes() const noexcept { return sizeof(*this); }

private:
    StateSample history_[kHistoryDepth]{};
    float       vel_[kStateDim]{};
    float       err_ema_[kStateDim]{};
    float       last_pred_x1_[kStateDim]{};
    size_t      head_ = 0u;
    size_t      count_ = 0u;
    uint32_t    samples_seen_ = 0u;
    uint32_t    alarms_raised_ = 0u;
    uint32_t    plans_computed_ = 0u;
    float       last_ts_ = -1.0f;
    float       safety_limit_ = 10.0f;
    bool        has_prev_pred_ = false;
};

} // namespace odin::mod_p

#endif
