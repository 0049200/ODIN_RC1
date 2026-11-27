/**
 * @file odin_c_embedded.hpp
 * @brief ODIN-C (Contextual State Projection, Robust MAD Deviation, Continuity, Nonlinear Feature Lift & Prequential SGD)
 *        Standalone C++17 bare-metal port of odin_c_advanced.py + mnesis/core.py for STM32G070RB (no SQLite/heap).
 */
#pragma once

#include "odin_common.hpp"

namespace odin::mod_c {

struct ContextState {
    float    timestamp_s = 0.0f;
    float    continuity = 1.0f;
    float    temporal_change = 0.0f;
    float    deviation = 0.0f;
    float    mnesis_prediction = 0.5f;
    float    uncertainty = 1.0f;
    bool     deviation_ready = false;
    bool     valid = false;
};

template <std::size_t DIMS = 4, std::size_t HISTORY_LEN = 32>
class EngineC {
public:
    static_assert(DIMS == 4, "ODIN-C embedded currently configured for 4D feature vectors");
    // Lift dims: 1 (bias) + 4 (x) + 6 (pairs) + 8 (2 EMAs * 4) + 4 (delta) = 23 features
    static constexpr std::size_t LIFT_DIMS = 23;

    explicit EngineC(float lr = 0.045f, float l2 = 0.0008f) noexcept
        : lr_(lr), l2_(l2) {
        reset();
    }

    void reset() noexcept {
        hist_count_ = 0;
        hist_head_ = 0;
        has_prev_ = false;
        previous_t_ = 0.0f;
        continuity_ = 1.0f;
        step_n_ = 0;
        for (std::size_t d = 0; d < DIMS; ++d) {
            ema_fast_[d] = 0.0f;
            ema_slow_[d] = 0.0f;
            prev_x_[d] = 0.0f;
        }
        for (std::size_t k = 0; k < LIFT_DIMS; ++k) {
            weights_[k] = 0.0f;
            last_feat_[k] = 0.0f;
        }
    }

    ContextState project(float timestamp_s,
                         const std::array<float, DIMS>& features,
                         float memory_confidence = 0.0f) noexcept {
        ContextState out{};
        if (!std::isfinite(timestamp_s)) return out;
        for (std::size_t d = 0; d < DIMS; ++d) {
            if (!std::isfinite(features[d])) return out;
        }
        if (has_prev_ && timestamp_s <= previous_t_) return out;

        // 1. Temporal change & smoothed continuity
        float change = 0.0f;
        if (has_prev_) {
            float d2 = 0.0f;
            for (std::size_t d = 0; d < DIMS; ++d) {
                float diff = features[d] - prev_x_[d];
                d2 += diff * diff;
            }
            change = std::sqrt(d2);
        }
        continuity_ = clamp_f32(0.75f * continuity_ + 0.25f * std::exp(-change), 0.0f, 1.0f);

        // 2. Robust MAD deviation over sliding history (requires >= 8 samples)
        float deviation = 0.0f;
        bool dev_ready = false;
        if (hist_count_ >= 8) {
            float z_norm2 = 0.0f;
            for (std::size_t d = 0; d < DIMS; ++d) {
                std::array<float, HISTORY_LEN> col{};
                for (std::size_t i = 0; i < hist_count_; ++i) {
                    col[i] = history_[i][d];
                }
                float center = median_fixed<HISTORY_LEN>(col, hist_count_);
                std::array<float, HISTORY_LEN> abs_dev{};
                for (std::size_t i = 0; i < hist_count_; ++i) {
                    abs_dev[i] = std::fabs(col[i] - center);
                }
                float mad = median_fixed<HISTORY_LEN>(abs_dev, hist_count_);
                float z = clamp_f32((features[d] - center) / (1.4826f * mad + 0.05f), -20.0f, 20.0f);
                z_norm2 += z * z;
            }
            deviation = std::sqrt(z_norm2) * 0.5f;
            dev_ready = true;
        }

        // 3. MNESIS nonlinear multiscale lift & prequential prediction
        transform_lift(features, last_feat_);
        float pred = predict_logit(last_feat_);

        // 4. Push into ring history
        history_[hist_head_] = features;
        hist_head_ = (hist_head_ + 1u) % HISTORY_LEN;
        if (hist_count_ < HISTORY_LEN) ++hist_count_;

        prev_x_ = features;
        previous_t_ = timestamp_s;
        has_prev_ = true;
        ++step_n_;

        float conf = clamp_f32(memory_confidence, 0.0f, 1.0f);
        out.timestamp_s       = timestamp_s;
        out.continuity        = continuity_;
        out.temporal_change   = change;
        out.deviation         = deviation;
        out.mnesis_prediction = pred;
        out.uncertainty       = 1.0f - conf;
        out.deviation_ready   = dev_ready;
        out.valid             = true;
        return out;
    }

    // Optional causal online SGD update when explicit ground-truth label is available
    float learn(float target_y) noexcept {
        float p = predict_logit(last_feat_);
        float err = clamp_f32(target_y, 0.0f, 1.0f) - p;
        float eta = lr_ / (1.0f + 0.00006f * static_cast<float>(step_n_));
        for (std::size_t k = 0; k < LIFT_DIMS; ++k) {
            float grad = clamp_f32(err * last_feat_[k], -3.0f, 3.0f);
            weights_[k] = clamp_f32(weights_[k] + eta * (grad - l2_ * weights_[k]), -8.0f, 8.0f);
        }
        return p;
    }

private:
    float lr_;
    float l2_;
    std::array<std::array<float, DIMS>, HISTORY_LEN> history_{};
    std::size_t hist_count_ = 0;
    std::size_t hist_head_ = 0;
    bool  has_prev_ = false;
    float previous_t_ = 0.0f;
    float continuity_ = 1.0f;
    uint32_t step_n_ = 0;

    std::array<float, DIMS> ema_fast_{};
    std::array<float, DIMS> ema_slow_{};
    std::array<float, DIMS> prev_x_{};
    std::array<float, LIFT_DIMS> weights_{};
    std::array<float, LIFT_DIMS> last_feat_{};

    static float sigmoid(float z) noexcept {
        z = clamp_f32(z, -30.0f, 30.0f);
        return 1.0f / (1.0f + std::exp(-z));
    }

    void transform_lift(const std::array<float, DIMS>& raw_x, std::array<float, LIFT_DIMS>& feat) noexcept {
        std::array<float, DIMS> x{};
        for (std::size_t d = 0; d < DIMS; ++d) x[d] = clamp_f32(raw_x[d], -4.0f, 4.0f);

        std::size_t idx = 0;
        feat[idx++] = 1.0f; // bias
        for (std::size_t i = 0; i < DIMS; ++i) feat[idx++] = x[i];
        for (std::size_t i = 0; i < DIMS; ++i) {
            for (std::size_t j = i + 1; j < DIMS; ++j) {
                feat[idx++] = x[i] * x[j];
            }
        }
        // Two EMA scales: 0.45 and 0.08
        for (std::size_t i = 0; i < DIMS; ++i) {
            ema_fast_[i] = 0.45f * x[i] + 0.55f * ema_fast_[i];
            feat[idx++] = ema_fast_[i];
        }
        for (std::size_t i = 0; i < DIMS; ++i) {
            ema_slow_[i] = 0.08f * x[i] + 0.92f * ema_slow_[i];
            feat[idx++] = ema_slow_[i];
        }
        for (std::size_t i = 0; i < DIMS; ++i) {
            feat[idx++] = has_prev_ ? (x[i] - clamp_f32(prev_x_[i], -4.0f, 4.0f)) : 0.0f;
        }
    }

    float predict_logit(const std::array<float, LIFT_DIMS>& feat) const noexcept {
        float dot = 0.0f;
        for (std::size_t k = 0; k < LIFT_DIMS; ++k) dot += weights_[k] * feat[k];
        return sigmoid(dot);
    }
};

} // namespace odin::mod_c
