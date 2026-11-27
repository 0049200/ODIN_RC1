/**
 * @file odin_d_embedded.hpp
 * @brief ODIN-D (D1 Temporal Difference, D2 Gaussian State Classifier, D3 Streaming Segmentation, D4 Versioned State)
 *        Standalone C++17 bare-metal port of ODIN_D_INTEGRATED (d1.py, pipeline.py, odin_d3, odin_d4) for STM32G070RB.
 */
#pragma once

#include "odin_common.hpp"

namespace odin::mod_d {

struct GaussianClass {
    float mean = 0.0f;
    float variance = 1.0f;
    bool  configured = false;
};

struct DiscriminationResult {
    float    timestamp_s = 0.0f;
    float    delta = 0.0f;
    float    squared_change = 0.0f;
    int8_t   d2_class = -1;       // -1 = abstained / uncertain
    float    d2_posterior = 0.0f;
    uint32_t d3_segment_id = 0;
    float    d3_zscore = 0.0f;
    bool     d3_transition = false;
    int8_t   d4_confirmed_state = -1;
    uint32_t d4_version = 0;
    bool     valid = false;
};

template <std::size_t NUM_CLASSES = 3>
class EngineD {
public:
    static_assert(NUM_CLASSES >= 2 && NUM_CLASSES <= 8, "NUM_CLASSES in [2, 8]");

    explicit EngineD(float min_margin = 0.20f,
                     float d3_z_threshold = 3.0f,
                     uint32_t d3_min_baseline = 8,
                     float d3_alpha = 0.12f,
                     uint32_t d3_consecutive = 2,
                     uint32_t d4_min_support = 2) noexcept
        : min_margin_(min_margin),
          d3_z_threshold_(d3_z_threshold),
          d3_min_baseline_(d3_min_baseline),
          d3_alpha_(d3_alpha),
          d3_consecutive_(d3_consecutive),
          d4_min_support_(d4_min_support) {
        reset();
    }

    void set_class_model(std::size_t class_idx, float mean, float variance) noexcept {
        if (class_idx >= NUM_CLASSES || !std::isfinite(mean) || !std::isfinite(variance) || variance <= 1e-6f) return;
        classes_[class_idx] = GaussianClass{mean, variance, true};
    }

    void reset() noexcept {
        has_prev_ = false;
        last_t_ = 0.0f;
        prev_val_ = 0.0f;
        d3_mean_ = 0.0f;
        d3_var_ = 0.0f;
        d3_count_ = 0;
        d3_segment_ = 0;
        d3_pending_ = 0;
        d4_state_ = -1;
        d4_candidate_ = -1;
        d4_support_ = 0;
        d4_version_ = 0;
    }

    DiscriminationResult step(float timestamp_s, float value, float quality = 1.0f) noexcept {
        return process(timestamp_s, value, quality);
    }

    DiscriminationResult process(float timestamp_s, float value, float quality = 1.0f) noexcept {
        DiscriminationResult out{};
        if (!std::isfinite(timestamp_s) || !std::isfinite(value) || !std::isfinite(quality) ||
            quality < 0.0f || quality > 1.0f) {
            return out;
        }
        if (has_prev_ && timestamp_s <= last_t_) return out;

        // D1: Lossless temporal difference
        float delta = has_prev_ ? (value - prev_val_) : 0.0f;
        float sq    = has_prev_ ? (delta * delta) : 0.0f;
        prev_val_   = value;
        last_t_     = timestamp_s;
        has_prev_   = true;

        // D2: Independent Gaussian classifier with abstention margin
        int8_t best_cls = -1;
        float  best_p   = 0.0f;
        float  log_scores[NUM_CLASSES]{};
        bool   any_cls = false;
        float  max_log = -1e30f;

        for (std::size_t c = 0; c < NUM_CLASSES; ++c) {
            if (!classes_[c].configured) {
                log_scores[c] = -1e30f;
                continue;
            }
            any_cls = true;
            float diff = value - classes_[c].mean;
            float var  = classes_[c].variance;
            float ls   = -0.5f * (std::log(6.2831853f * var) + (diff * diff) / var);
            log_scores[c] = ls;
            if (ls > max_log) max_log = ls;
        }

        if (any_cls) {
            float den = 0.0f;
            float top_w = 0.0f;
            std::size_t top_i = 0;
            for (std::size_t c = 0; c < NUM_CLASSES; ++c) {
                if (!classes_[c].configured) continue;
                float w = std::exp(log_scores[c] - max_log);
                den += w;
                if (w > top_w) {
                    top_w = w;
                    top_i = c;
                }
            }
            if (den > 0.0f) {
                best_p = top_w / den;
                if (best_p >= (0.5f + min_margin_)) {
                    best_cls = static_cast<int8_t>(top_i);
                }
            }
        }

        // D3: Streaming event segmentation (z-score against adaptive baseline)
        bool transition = false;
        float zscore = 0.0f;
        if (d3_count_ < d3_min_baseline_) {
            ++d3_count_;
            float diff = value - d3_mean_;
            d3_mean_ += diff / static_cast<float>(d3_count_);
            d3_var_  += diff * (value - d3_mean_);
        } else {
            float std_dev = std::sqrt((d3_var_ / static_cast<float>(d3_count_ - 1u)) + 1e-6f);
            zscore = std::fabs(value - d3_mean_) / std_dev;
            if (zscore >= d3_z_threshold_) {
                ++d3_pending_;
                if (d3_pending_ >= d3_consecutive_) {
                    ++d3_segment_;
                    transition = true;
                    d3_pending_ = 0;
                    // Re-seed baseline around new regime
                    d3_mean_ = value;
                    d3_var_  = 1e-4f;
                    d3_count_ = 1;
                }
            } else {
                d3_pending_ = 0;
                float diff = value - d3_mean_;
                d3_mean_ += d3_alpha_ * diff;
                d3_var_   = (1.0f - d3_alpha_) * d3_var_ + d3_alpha_ * diff * diff * static_cast<float>(d3_count_);
            }
        }

        // D4: Evidence revision & versioned state confirmation
        if (quality >= 0.5f && best_cls >= 0) {
            if (best_cls == d4_candidate_) {
                ++d4_support_;
            } else {
                d4_candidate_ = best_cls;
                d4_support_ = 1;
            }
            if (d4_support_ >= d4_min_support_ && d4_state_ != d4_candidate_) {
                d4_state_ = d4_candidate_;
                ++d4_version_;
            }
        }

        out.timestamp_s        = timestamp_s;
        out.delta              = delta;
        out.squared_change     = sq;
        out.d2_class           = best_cls;
        out.d2_posterior       = best_p;
        out.d3_segment_id      = d3_segment_;
        out.d3_zscore          = zscore;
        out.d3_transition      = transition;
        out.d4_confirmed_state = d4_state_;
        out.d4_version         = d4_version_;
        out.valid              = true;
        return out;
    }

private:
    float    min_margin_;
    float    d3_z_threshold_;
    uint32_t d3_min_baseline_;
    float    d3_alpha_;
    uint32_t d3_consecutive_;
    uint32_t d4_min_support_;

    std::array<GaussianClass, NUM_CLASSES> classes_{};
    bool     has_prev_ = false;
    float    last_t_ = 0.0f;
    float    prev_val_ = 0.0f;

    float    d3_mean_ = 0.0f;
    float    d3_var_ = 0.0f;
    uint32_t d3_count_ = 0;
    uint32_t d3_segment_ = 0;
    uint32_t d3_pending_ = 0;

    int8_t   d4_state_ = -1;
    int8_t   d4_candidate_ = -1;
    uint32_t d4_support_ = 0;
    uint32_t d4_version_ = 0;
};

using DiscriminationEngine = EngineD<3>;

} // namespace odin::mod_d
