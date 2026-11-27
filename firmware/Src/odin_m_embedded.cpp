#include "odin_m_embedded.hpp"
#include <cmath>
#include <cstring>

namespace odin {
namespace mod_m {

namespace {
constexpr float kAshiKc = 1.441f;
constexpr float kAshiGamma = 0.25f;
constexpr float kMatchRadiusSq = 0.25f;
} // namespace

uint32_t OdinMemory::crc32_bytes(const void* data, size_t len) noexcept {
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

uint32_t OdinMemory::compute_proto_crc(const PrototypeSlot& p) const noexcept {
    return crc32_bytes(&p, offsetof(PrototypeSlot, crc32));
}

uint32_t OdinMemory::compute_rev_crc(const RevisionRecord& r) const noexcept {
    return crc32_bytes(&r, offsetof(RevisionRecord, crc32));
}

uint32_t OdinMemory::compute_integrity_crc32() const noexcept {
    uint32_t acc = crc32_bytes(prototypes_, sizeof(prototypes_));
    acc ^= crc32_bytes(revisions_, sizeof(revisions_));
    acc ^= crc32_bytes(key_head_version_, sizeof(key_head_version_));
    return acc;
}

void OdinMemory::refresh_global_crc() noexcept {
    global_crc32_ = compute_integrity_crc32();
}

bool OdinMemory::verify_integrity() const noexcept {
    if (global_crc32_ != compute_integrity_crc32()) {
        return false;
    }
    for (size_t i = 0u; i < kMaxPrototypes; ++i) {
        if (prototypes_[i].active && prototypes_[i].crc32 != compute_proto_crc(prototypes_[i])) {
            return false;
        }
    }
    for (size_t k = 0u; k < kMaxKeys; ++k) {
        for (size_t r = 0u; r < kMaxRevisions; ++r) {
            if (revisions_[k][r].valid && revisions_[k][r].crc32 != compute_rev_crc(revisions_[k][r])) {
                return false;
            }
        }
    }
    return true;
}

void OdinMemory::inject_bitflip_corruption(size_t byte_offset, uint8_t bit_mask) noexcept {
    uint8_t* raw = reinterpret_cast<uint8_t*>(prototypes_);
    const size_t max_bytes = sizeof(prototypes_);
    raw[byte_offset % max_bytes] ^= bit_mask;
}

uint32_t OdinMemory::sanitize_and_recover() noexcept {
    uint32_t recovered = 0u;
    for (size_t i = 0u; i < kMaxPrototypes; ++i) {
        if (prototypes_[i].active) {
            const bool bad_crc = (prototypes_[i].crc32 != compute_proto_crc(prototypes_[i]));
            const bool bad_vec = !is_finite_vec(prototypes_[i].center);
            const bool bad_key = (prototypes_[i].key_id >= kMaxKeys);
            if (bad_crc || bad_vec || bad_key) {
                // Attempt recovery from latest valid revision record for that key
                bool restored = false;
                if (!bad_key && key_head_version_[prototypes_[i].key_id] > 0u) {
                    const uint16_t hv = key_head_version_[prototypes_[i].key_id];
                    const size_t slot = static_cast<size_t>((hv - 1u) % kMaxRevisions);
                    const RevisionRecord& rr = revisions_[prototypes_[i].key_id][slot];
                    if (rr.valid && rr.crc32 == compute_rev_crc(rr) && is_finite_vec(rr.value)) {
                        for (size_t d = 0u; d < kFeatureDim; ++d) {
                            prototypes_[i].center[d] = rr.value[d];
                        }
                        prototypes_[i].version = rr.version;
                        prototypes_[i].crc32 = compute_proto_crc(prototypes_[i]);
                        restored = true;
                    }
                }
                if (!restored) {
                    std::memset(&prototypes_[i], 0, sizeof(PrototypeSlot));
                }
                ++recovered;
            }
        }
    }
    refresh_global_crc();
    if (recovered > 0u) {
        append_journal(5u, 0u, 0u, 0u, 0u, 0u, 1u, last_global_ts_, static_cast<float>(recovered));
    }
    return recovered;
}

bool OdinMemory::is_finite_vec(const float v[kFeatureDim]) noexcept {
    if (!v) return false;
    for (size_t i = 0u; i < kFeatureDim; ++i) {
        if (!std::isfinite(v[i])) return false;
    }
    return true;
}

float OdinMemory::dist_sq(const float a[kFeatureDim], const float b[kFeatureDim]) noexcept {
    float acc = 0.0f;
    for (size_t i = 0u; i < kFeatureDim; ++i) {
        const float d = a[i] - b[i];
        acc += d * d;
    }
    return acc;
}

void OdinMemory::reset() noexcept {
    std::memset(prototypes_, 0, sizeof(prototypes_));
    std::memset(revisions_, 0, sizeof(revisions_));
    std::memset(key_head_version_, 0, sizeof(key_head_version_));
    std::memset(ashi_, 0, sizeof(ashi_));
    std::memset(episodes_, 0, sizeof(episodes_));
    std::memset(journal_, 0, sizeof(journal_));
    episode_head_ = 0u;
    episode_count_ = 0u;
    journal_seq_ = 0u;
    total_observations_ = 0u;
    total_revisions_ = 0u;
    total_rollbacks_ = 0u;
    ashi_triggers_ = 0u;
    pruned_prototypes_ = 0u;
    last_global_ts_ = -1.0f;
    refresh_global_crc();
}

void OdinMemory::append_journal(uint8_t op_type, uint16_t key_id, uint16_t version,
                                uint32_t event_id, uint8_t source_id, uint8_t ashi_fired,
                                uint8_t committed, float ts, float imp) noexcept {
    const size_t idx = static_cast<size_t>(journal_seq_ % kMaxJournalItems);
    ++journal_seq_;
    JournalEntry& je = journal_[idx];
    je.seq = journal_seq_;
    je.event_id = event_id;
    je.key_id = key_id;
    je.version = version;
    je.op_type = op_type;
    je.source_id = source_id;
    je.ashi_fired = ashi_fired;
    je.committed = committed;
    je.timestamp_s = ts;
    je.importance = imp;
    je.crc32 = crc32_bytes(&je, offsetof(JournalEntry, crc32));
}

void OdinMemory::record_revision(uint16_t key_id, uint16_t version, uint8_t source_id,
                                 float ts, const float val[kFeatureDim]) noexcept {
    if (key_id >= kMaxKeys) return;
    const size_t slot = static_cast<size_t>((version - 1u) % kMaxRevisions);
    RevisionRecord& rr = revisions_[key_id][slot];
    rr.valid = 1u;
    rr.source_id = source_id;
    rr.key_id = key_id;
    rr.version = version;
    rr.pad = 0u;
    rr.timestamp_s = ts;
    for (size_t i = 0u; i < kFeatureDim; ++i) {
        rr.value[i] = val[i];
    }
    rr.crc32 = compute_rev_crc(rr);
}

bool OdinMemory::observe(const Sample& s,
                         RejectReason* out_reason,
                         bool simulate_power_loss_mid_write) noexcept {
    if (s.key_id >= kMaxKeys) {
        if (out_reason) *out_reason = RejectReason::INVALID_KEY;
        return false;
    }
    if (!is_finite_vec(s.features)) {
        if (out_reason) *out_reason = RejectReason::NON_FINITE_FEATURE;
        return false;
    }
    if (!std::isfinite(s.importance) || s.importance < 0.0f) {
        if (out_reason) *out_reason = RejectReason::INVALID_IMPORTANCE;
        return false;
    }
    if (!std::isfinite(s.timestamp_s) || s.timestamp_s <= last_global_ts_) {
        if (out_reason) *out_reason = RejectReason::NON_MONOTONIC_TIME;
        return false;
    }

    // Write-Ahead Intent Journal entry (uncommitted) for atomic power-loss resilience (M10)
    append_journal(1u, s.key_id, key_head_version_[s.key_id], s.event_id,
                   s.source_id, 0u, 0u, s.timestamp_s, s.importance);
    if (simulate_power_loss_mid_write) {
        // Atomic abort: committed state, prototypes, counters, and global CRC remain untouched
        if (out_reason) *out_reason = RejectReason::PARTIAL_WRITE_ABORTED;
        return false;
    }

    last_global_ts_ = s.timestamp_s;
    ++total_observations_;

    // ASHI online Welford statistics + empirical threshold kc = 1.441
    AshiKeyState& ak = ashi_[s.key_id];
    bool fired = false;
    if (ak.count >= 5u) {
        const float var = ak.m2 / static_cast<float>(ak.count - 1u);
        const float sd = (var > 0.0f) ? std::sqrt(var) : 0.0f;
        fired = (s.importance > (ak.mean + kAshiKc * sd));
    }
    ++ak.count;
    const float delta = s.importance - ak.mean;
    ak.mean += delta / static_cast<float>(ak.count);
    const float delta2 = s.importance - ak.mean;
    ak.m2 += delta * delta2;

    if (fired) {
        ak.latched = 1u;
        ++ashi_triggers_;
    }
    ak.accumulator += s.importance;
    const float excess = (s.importance > ak.mean) ? (s.importance - ak.mean) : 0.0f;
    ak.consolidated_importance += s.importance + (ak.latched ? (kAshiGamma * excess * excess) : 0.0f);

    // Bounded episodic ring buffer (M4)
    EpisodeSlot& ep = episodes_[episode_head_];
    ep.event_id = s.event_id;
    ep.key_id = s.key_id;
    ep.source_id = s.source_id;
    ep.ashi_fired = fired ? 1u : 0u;
    ep.timestamp_s = s.timestamp_s;
    ep.importance = s.importance;
    for (size_t i = 0u; i < kFeatureDim; ++i) {
        ep.features[i] = s.features[i];
    }
    episode_head_ = (episode_head_ + 1u) % kMaxEpisodes;
    if (episode_count_ < kMaxEpisodes) {
        ++episode_count_;
    }

    // Prototype acquisition & plasticity-modulated update (M1 + M5 + M8)
    int best_idx = -1;
    float best_d2 = 1e30f;
    int free_idx = -1;
    int lowest_imp_idx = 0;
    float lowest_imp = 1e30f;

    for (size_t i = 0u; i < kMaxPrototypes; ++i) {
        if (!prototypes_[i].active) {
            if (free_idx < 0) free_idx = static_cast<int>(i);
            continue;
        }
        if (prototypes_[i].importance_acc < lowest_imp) {
            lowest_imp = prototypes_[i].importance_acc;
            lowest_imp_idx = static_cast<int>(i);
        }
        if (prototypes_[i].key_id == s.key_id) {
            const float d2 = dist_sq(prototypes_[i].center, s.features);
            if (d2 < best_d2) {
                best_d2 = d2;
                best_idx = static_cast<int>(i);
            }
        }
    }

    const float pw = protection_weight(s.key_id);
    const float alpha = 0.25f / (1.0f + pw);

    if (best_idx >= 0 && best_d2 <= kMatchRadiusSq) {
        PrototypeSlot& ps = prototypes_[best_idx];
        for (size_t i = 0u; i < kFeatureDim; ++i) {
            ps.center[i] = (1.0f - alpha) * ps.center[i] + alpha * s.features[i];
        }
        ps.importance_acc += s.importance;
        ps.last_timestamp_s = s.timestamp_s;
        ps.crc32 = compute_proto_crc(ps);
    } else {
        const int target_idx = (free_idx >= 0) ? free_idx : lowest_imp_idx;
        if (free_idx < 0 && prototypes_[target_idx].active) {
            ++pruned_prototypes_;
        }
        PrototypeSlot& ps = prototypes_[target_idx];
        ps.active = 1u;
        ps.key_id = s.key_id;
        ps.version = (key_head_version_[s.key_id] == 0u) ? 1u : key_head_version_[s.key_id];
        ps.recall_count = 0u;
        for (size_t i = 0u; i < kFeatureDim; ++i) {
            ps.center[i] = s.features[i];
        }
        ps.importance_acc = s.importance;
        ps.last_timestamp_s = s.timestamp_s;
        ps.crc32 = compute_proto_crc(ps);
        if (key_head_version_[s.key_id] == 0u) {
            key_head_version_[s.key_id] = 1u;
            record_revision(s.key_id, 1u, s.source_id, s.timestamp_s, s.features);
        }
    }

    // Commit record in journal & refresh global CRC
    append_journal(1u, s.key_id, key_head_version_[s.key_id], s.event_id,
                   s.source_id, fired ? 1u : 0u, 1u, s.timestamp_s, s.importance);
    refresh_global_crc();
    if (out_reason) *out_reason = RejectReason::NONE;
    return true;
}

RecallResult OdinMemory::recall(uint16_t key_id,
                                const float query[kFeatureDim],
                                float min_confidence) const noexcept {
    RecallResult res{};
    res.key_id = key_id;

    if (key_id >= kMaxKeys) {
        res.reason = RejectReason::INVALID_KEY;
        return res;
    }
    if (!is_finite_vec(query) || !std::isfinite(min_confidence)) {
        res.reason = RejectReason::NON_FINITE_FEATURE;
        return res;
    }

    int best_idx = -1;
    float best_d2 = 1e30f;
    for (size_t i = 0u; i < kMaxPrototypes; ++i) {
        if (prototypes_[i].active && prototypes_[i].key_id == key_id) {
            if (prototypes_[i].crc32 != compute_proto_crc(prototypes_[i])) {
                res.reason = RejectReason::INTEGRITY_CORRUPTION;
                return res;
            }
            const float d2 = dist_sq(prototypes_[i].center, query);
            if (d2 < best_d2) {
                best_d2 = d2;
                best_idx = static_cast<int>(i);
            }
        }
    }

    if (best_idx < 0) {
        res.reason = RejectReason::NO_MATCHING_PROTOTYPE;
        return res;
    }

    const float dist = std::sqrt(best_d2);
    const float conf = 1.0f / (1.0f + dist);
    const PrototypeSlot& ps = prototypes_[best_idx];

    res.prototype_index = static_cast<uint16_t>(best_idx);
    res.version = ps.version;
    res.recall_count = ps.recall_count;
    res.confidence = conf;
    res.distance = dist;
    res.protection_weight = protection_weight(key_id);
    for (size_t i = 0u; i < kFeatureDim; ++i) {
        res.features[i] = ps.center[i];
    }

    if (conf < min_confidence) {
        res.found = false;
        res.reason = RejectReason::LOW_CONFIDENCE;
        return res;
    }

    res.found = true;
    res.reason = RejectReason::NONE;
    return res;
}

bool OdinMemory::revise(uint16_t key_id,
                        const float value[kFeatureDim],
                        uint8_t source_id,
                        bool trusted,
                        float timestamp_s,
                        RejectReason* out_reason) noexcept {
    if (key_id >= kMaxKeys) {
        if (out_reason) *out_reason = RejectReason::INVALID_KEY;
        return false;
    }
    if (!is_finite_vec(value)) {
        if (out_reason) *out_reason = RejectReason::NON_FINITE_FEATURE;
        return false;
    }
    if (!std::isfinite(timestamp_s) || timestamp_s <= last_global_ts_) {
        if (out_reason) *out_reason = RejectReason::NON_MONOTONIC_TIME;
        return false;
    }
    if (!trusted) {
        if (out_reason) *out_reason = RejectReason::UNTRUSTED_REVISION;
        return false;
    }

    last_global_ts_ = timestamp_s;
    const uint16_t next_ver = static_cast<uint16_t>(key_head_version_[key_id] + 1u);
    key_head_version_[key_id] = next_ver;
    ++total_revisions_;

    record_revision(key_id, next_ver, source_id, timestamp_s, value);

    int target_idx = -1;
    int free_idx = -1;
    for (size_t i = 0u; i < kMaxPrototypes; ++i) {
        if (prototypes_[i].active && prototypes_[i].key_id == key_id) {
            target_idx = static_cast<int>(i);
            break;
        }
        if (!prototypes_[i].active && free_idx < 0) {
            free_idx = static_cast<int>(i);
        }
    }
    if (target_idx < 0) {
        target_idx = (free_idx >= 0) ? free_idx : 0;
    }

    PrototypeSlot& ps = prototypes_[target_idx];
    ps.active = 1u;
    ps.key_id = key_id;
    ps.version = next_ver;
    for (size_t i = 0u; i < kFeatureDim; ++i) {
        ps.center[i] = value[i];
    }
    ps.last_timestamp_s = timestamp_s;
    ps.crc32 = compute_proto_crc(ps);

    append_journal(2u, key_id, next_ver, 0u, source_id, 0u, 1u, timestamp_s, 1.0f);
    refresh_global_crc();
    if (out_reason) *out_reason = RejectReason::NONE;
    return true;
}

bool OdinMemory::rollback(uint16_t key_id,
                          uint16_t target_version,
                          float timestamp_s,
                          RejectReason* out_reason) noexcept {
    if (key_id >= kMaxKeys) {
        if (out_reason) *out_reason = RejectReason::INVALID_KEY;
        return false;
    }
    if (!std::isfinite(timestamp_s) || timestamp_s <= last_global_ts_) {
        if (out_reason) *out_reason = RejectReason::NON_MONOTONIC_TIME;
        return false;
    }
    if (target_version == 0u || target_version > key_head_version_[key_id]) {
        if (out_reason) *out_reason = RejectReason::VERSION_NOT_FOUND;
        return false;
    }

    const size_t slot = static_cast<size_t>((target_version - 1u) % kMaxRevisions);
    const RevisionRecord& rr = revisions_[key_id][slot];
    if (!rr.valid || rr.version != target_version || rr.crc32 != compute_rev_crc(rr)) {
        if (out_reason) *out_reason = RejectReason::VERSION_NOT_FOUND;
        return false;
    }

    last_global_ts_ = timestamp_s;
    const uint16_t new_ver = static_cast<uint16_t>(key_head_version_[key_id] + 1u);
    key_head_version_[key_id] = new_ver;
    ++total_rollbacks_;

    record_revision(key_id, new_ver, rr.source_id, timestamp_s, rr.value);

    for (size_t i = 0u; i < kMaxPrototypes; ++i) {
        if (prototypes_[i].active && prototypes_[i].key_id == key_id) {
            prototypes_[i].version = new_ver;
            for (size_t d = 0u; d < kFeatureDim; ++d) {
                prototypes_[i].center[d] = rr.value[d];
            }
            prototypes_[i].last_timestamp_s = timestamp_s;
            prototypes_[i].crc32 = compute_proto_crc(prototypes_[i]);
            break;
        }
    }

    append_journal(3u, key_id, new_ver, 0u, rr.source_id, 0u, 1u, timestamp_s, 1.0f);
    refresh_global_crc();
    if (out_reason) *out_reason = RejectReason::NONE;
    return true;
}

uint32_t OdinMemory::consolidate(float merge_dist_sq, float decay_factor) noexcept {
    uint32_t merged = 0u;
    for (size_t i = 0u; i < kMaxPrototypes; ++i) {
        if (!prototypes_[i].active) continue;
        prototypes_[i].importance_acc *= decay_factor;
        for (size_t j = i + 1u; j < kMaxPrototypes; ++j) {
            if (!prototypes_[j].active) continue;
            if (prototypes_[i].key_id == prototypes_[j].key_id) {
                if (dist_sq(prototypes_[i].center, prototypes_[j].center) <= merge_dist_sq) {
                    const float wi = prototypes_[i].importance_acc + 1e-3f;
                    const float wj = prototypes_[j].importance_acc + 1e-3f;
                    const float wsum = wi + wj;
                    for (size_t d = 0u; d < kFeatureDim; ++d) {
                        prototypes_[i].center[d] =
                            (wi * prototypes_[i].center[d] + wj * prototypes_[j].center[d]) / wsum;
                    }
                    prototypes_[i].importance_acc += prototypes_[j].importance_acc;
                    prototypes_[j].active = 0u;
                    ++merged;
                    ++pruned_prototypes_;
                }
            }
        }
        prototypes_[i].crc32 = compute_proto_crc(prototypes_[i]);
    }
    if (merged > 0u) {
        append_journal(4u, 0u, 0u, 0u, 0u, 0u, 1u, last_global_ts_, static_cast<float>(merged));
    }
    refresh_global_crc();
    return merged;
}

Snapshot OdinMemory::export_snapshot() const noexcept {
    Snapshot s{};
    uint16_t active = 0u;
    for (size_t i = 0u; i < kMaxPrototypes; ++i) {
        if (prototypes_[i].active) ++active;
    }
    s.journal_seq = journal_seq_;
    s.active_prototypes = active;
    s.episode_count = static_cast<uint16_t>(episode_count_);
    s.total_observations = total_observations_;
    s.total_revisions = total_revisions_;
    s.total_rollbacks = total_rollbacks_;
    s.ashi_triggers = ashi_triggers_;
    s.pruned_prototypes = pruned_prototypes_;
    s.state_crc32 = compute_integrity_crc32();
    s.valid = 1u;
    return s;
}

bool OdinMemory::restore_from_snapshot(const Snapshot& snap) noexcept {
    if (!snap.valid || snap.state_crc32 != compute_integrity_crc32()) {
        return false;
    }
    journal_seq_ = snap.journal_seq;
    total_observations_ = snap.total_observations;
    total_revisions_ = snap.total_revisions;
    total_rollbacks_ = snap.total_rollbacks;
    ashi_triggers_ = snap.ashi_triggers;
    pruned_prototypes_ = snap.pruned_prototypes;
    std::memset(ashi_, 0, sizeof(ashi_));
    refresh_global_crc();
    return true;
}

uint16_t OdinMemory::head(uint16_t key_id) const noexcept {
    return (key_id < kMaxKeys) ? key_head_version_[key_id] : 0u;
}

size_t OdinMemory::cells(uint16_t key_id) const noexcept {
    if (key_id >= kMaxKeys) return 0u;
    size_t n = 0u;
    for (size_t i = 0u; i < kMaxPrototypes; ++i) {
        if (prototypes_[i].active && prototypes_[i].key_id == key_id) ++n;
    }
    return n;
}

size_t OdinMemory::active_prototypes_total() const noexcept {
    size_t n = 0u;
    for (size_t i = 0u; i < kMaxPrototypes; ++i) {
        if (prototypes_[i].active) ++n;
    }
    return n;
}

float OdinMemory::protection_weight(uint16_t key_id) const noexcept {
    if (key_id >= kMaxKeys) return 0.0f;
    const AshiKeyState& ak = ashi_[key_id];
    return ak.latched ? (1.0f + 0.1f * ak.consolidated_importance) : 0.0f;
}

bool OdinMemory::ashi_gate_latched(uint16_t key_id) const noexcept {
    return (key_id < kMaxKeys) ? (ashi_[key_id].latched != 0u) : false;
}

} // namespace mod_m
} // namespace odin
