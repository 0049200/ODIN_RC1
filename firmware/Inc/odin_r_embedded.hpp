/**
 * @file odin_r_embedded.hpp
 * @brief ODIN-R (Reasoning, Evidence Fusion & Rule Inference Engine)
 *        Target: STM32G070RBTx (ARM Cortex-M0+, ISO C++17 freestanding, zero heap).
 */

#ifndef ODIN_R_EMBEDDED_HPP
#define ODIN_R_EMBEDDED_HPP

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cstring>

namespace odin::mod_r {

constexpr size_t kMaxRules        = 8u;
constexpr size_t kMaxHypotheses   = 4u;
constexpr size_t kEvidenceWindow  = 16u;

struct EvidencePacket {
    uint16_t source_id;
    uint16_t hypothesis_id;
    float    timestamp_s;
    float    support;        // [0.0, 1.0]
    float    reliability;    // [0.0, 1.0]
    uint8_t  contradicts;    // 1 if explicit refutation
};

struct InferenceRule {
    uint8_t  active;
    uint8_t  priority;
    uint16_t hypothesis_id;
    float    min_support;
    float    max_conflict;
    uint16_t action_code;
};

struct ReasoningVerdict {
    uint16_t winning_hypothesis;
    uint16_t recommended_action;
    float    posterior_confidence;
    float    conflict_index;
    uint8_t  rule_fired;
    uint8_t  abstained;
    uint32_t crc32;
};

class OdinReasoner {
public:
    OdinReasoner() noexcept { reset(); }

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
        std::memset(pos_mass_, 0, sizeof(pos_mass_));
        std::memset(neg_mass_, 0, sizeof(neg_mass_));
        std::memset(rules_, 0, sizeof(rules_));
        std::memset(recent_, 0, sizeof(recent_));
        recent_head_ = 0u;
        evidence_count_ = 0u;
        inferences_run_ = 0u;
        abstentions_ = 0u;
        last_ts_ = -1.0f;
        conflict_threshold_ = 0.45f;
        min_confidence_ = 0.55f;
    }

    void set_thresholds(float min_conf, float max_conflict) noexcept {
        if (std::isfinite(min_conf) && min_conf > 0.0f && min_conf < 1.0f) min_confidence_ = min_conf;
        if (std::isfinite(max_conflict) && max_conflict > 0.0f && max_conflict <= 1.0f) conflict_threshold_ = max_conflict;
    }

    bool register_rule(size_t slot, uint16_t hyp_id, float min_sup, float max_conf, uint16_t action_code, uint8_t priority) noexcept {
        if (slot >= kMaxRules || hyp_id >= kMaxHypotheses) return false;
        if (!std::isfinite(min_sup) || !std::isfinite(max_conf)) return false;
        rules_[slot].active = 1u;
        rules_[slot].priority = priority;
        rules_[slot].hypothesis_id = hyp_id;
        rules_[slot].min_support = min_sup;
        rules_[slot].max_conflict = max_conf;
        rules_[slot].action_code = action_code;
        return true;
    }

    bool assert_evidence(const EvidencePacket& ev) noexcept {
        if (ev.hypothesis_id >= kMaxHypotheses) return false;
        if (!std::isfinite(ev.timestamp_s) || ev.timestamp_s <= last_ts_) return false;
        if (!std::isfinite(ev.support) || ev.support < 0.0f || ev.support > 1.0f) return false;
        if (!std::isfinite(ev.reliability) || ev.reliability < 0.0f || ev.reliability > 1.0f) return false;

        last_ts_ = ev.timestamp_s;
        ++evidence_count_;
        recent_[recent_head_] = ev;
        recent_head_ = (recent_head_ + 1u) % kEvidenceWindow;

        const float w = ev.support * ev.reliability;
        for (size_t h = 0u; h < kMaxHypotheses; ++h) {
            pos_mass_[h] *= 0.96f;
            neg_mass_[h] *= 0.96f;
        }
        if (ev.contradicts != 0u) {
            neg_mass_[ev.hypothesis_id] += w;
        } else {
            pos_mass_[ev.hypothesis_id] += w;
        }
        return true;
    }

    ReasoningVerdict evaluate() noexcept {
        ReasoningVerdict v{};
        ++inferences_run_;

        float total_system_mass = 0.0f;
        for (size_t h = 0u; h < kMaxHypotheses; ++h) {
            total_system_mass += pos_mass_[h] + neg_mass_[h];
        }

        uint16_t best_h = 0u;
        float best_conf = -1.0f;
        float best_net_mass = -1.0f;
        float best_conflict = 1.0f;

        for (size_t h = 0u; h < kMaxHypotheses; ++h) {
            const float tot = pos_mass_[h] + neg_mass_[h];
            if (tot <= 1e-6f) continue;
            const float net_mass = pos_mass_[h] - neg_mass_[h];
            const float net_ratio = (net_mass > 0.0f) ? (net_mass / tot) : 0.0f;
            const float share = (total_system_mass > 1e-6f) ? (tot / total_system_mass) : 0.0f;
            const float conf = net_ratio * (0.5f + 0.5f * share);
            const float min_m = (pos_mass_[h] < neg_mass_[h]) ? pos_mass_[h] : neg_mass_[h];
            const float conflict = (2.0f * min_m) / tot;

            if (conf > best_conf || (conf == best_conf && net_mass > best_net_mass)) {
                best_conf = conf;
                best_net_mass = net_mass;
                best_conflict = conflict;
                best_h = static_cast<uint16_t>(h);
            }
        }

        if (best_conf < 0.0f) {
            best_conf = 0.0f;
            best_conflict = 0.0f;
        }

        v.winning_hypothesis = best_h;
        v.posterior_confidence = best_conf;
        v.conflict_index = best_conflict;

        if (best_conf < min_confidence_ || best_conflict > conflict_threshold_) {
            v.abstained = 1u;
            v.rule_fired = 0u;
            v.recommended_action = 0u;
            ++abstentions_;
            v.crc32 = crc32_bytes(&v, offsetof(ReasoningVerdict, crc32));
            return v;
        }

        int chosen_rule = -1;
        uint8_t best_prio = 0u;
        for (size_t r = 0u; r < kMaxRules; ++r) {
            if (!rules_[r].active) continue;
            if (rules_[r].hypothesis_id == best_h &&
                best_conf >= rules_[r].min_support &&
                best_conflict <= rules_[r].max_conflict) {
                if (chosen_rule < 0 || rules_[r].priority > best_prio) {
                    chosen_rule = static_cast<int>(r);
                    best_prio = rules_[r].priority;
                }
            }
        }

        if (chosen_rule >= 0) {
            v.rule_fired = 1u;
            v.abstained = 0u;
            v.recommended_action = rules_[chosen_rule].action_code;
        } else {
            v.rule_fired = 0u;
            v.abstained = 1u;
            v.recommended_action = 0u;
            ++abstentions_;
        }

        v.crc32 = crc32_bytes(&v, offsetof(ReasoningVerdict, crc32));
        return v;
    }

    bool verify_verdict(const ReasoningVerdict& v) const noexcept {
        return v.crc32 == crc32_bytes(&v, offsetof(ReasoningVerdict, crc32));
    }

    uint32_t evidence_count() const noexcept { return evidence_count_; }
    uint32_t inferences_run() const noexcept { return inferences_run_; }
    uint32_t abstentions() const noexcept { return abstentions_; }
    size_t bytes() const noexcept { return sizeof(*this); }

private:
    float          pos_mass_[kMaxHypotheses]{};
    float          neg_mass_[kMaxHypotheses]{};
    InferenceRule  rules_[kMaxRules]{};
    EvidencePacket recent_[kEvidenceWindow]{};
    size_t         recent_head_ = 0u;
    uint32_t       evidence_count_ = 0u;
    uint32_t       inferences_run_ = 0u;
    uint32_t       abstentions_ = 0u;
    float          last_ts_ = -1.0f;
    float          conflict_threshold_ = 0.45f;
    float          min_confidence_ = 0.55f;
};

} // namespace odin::mod_r

#endif
