#ifndef ODIN_M_EMBEDDED_HPP
#define ODIN_M_EMBEDDED_HPP

#include <cstdint>
#include <cstddef>

namespace odin {
namespace mod_m {

static constexpr size_t kFeatureDim      = 4u;
static constexpr size_t kMaxKeys         = 4u;
static constexpr size_t kMaxPrototypes   = 8u;
static constexpr size_t kMaxEpisodes     = 16u;
static constexpr size_t kMaxRevisions    = 8u;
static constexpr size_t kMaxJournalItems = 32u;

enum class RejectReason : uint8_t {
    NONE = 0u,
    INVALID_KEY = 1u,
    NON_FINITE_FEATURE = 2u,
    INVALID_IMPORTANCE = 3u,
    NON_MONOTONIC_TIME = 4u,
    UNTRUSTED_REVISION = 5u,
    LOW_CONFIDENCE = 6u,
    NO_MATCHING_PROTOTYPE = 7u,
    VERSION_NOT_FOUND = 8u,
    INTEGRITY_CORRUPTION = 9u,
    PARTIAL_WRITE_ABORTED = 10u
};

struct Sample {
    uint16_t key_id;
    uint32_t event_id;
    float    timestamp_s;
    float    features[kFeatureDim];
    float    importance;
    uint8_t  source_id;
    uint8_t  trusted;
};

struct RecallResult {
    bool         found;
    uint16_t     key_id;
    uint16_t     prototype_index;
    uint16_t     version;
    uint16_t     recall_count;
    float        features[kFeatureDim];
    float        confidence;
    float        distance;
    float        protection_weight;
    RejectReason reason;
};

struct JournalEntry {
    uint32_t seq;
    uint32_t event_id;
    uint16_t key_id;
    uint16_t version;
    uint8_t  op_type; // 1=observe, 2=revise, 3=rollback, 4=consolidate, 5=repair
    uint8_t  source_id;
    uint8_t  ashi_fired;
    uint8_t  committed;
    float    timestamp_s;
    float    importance;
    uint32_t crc32;
};

struct Snapshot {
    uint32_t journal_seq;
    uint16_t active_prototypes;
    uint16_t episode_count;
    uint32_t total_observations;
    uint32_t total_revisions;
    uint32_t total_rollbacks;
    uint32_t ashi_triggers;
    uint32_t pruned_prototypes;
    uint32_t state_crc32;
    uint8_t  valid;
    uint8_t  pad[3];
};

class OdinMemory {
public:
    OdinMemory() noexcept { reset(); }

    void reset() noexcept;

    // M1 + M2 + M4 + M5: Acquire numerical experience, update ASHI & prototypes, log journal
    bool observe(const Sample& s,
                 RejectReason* out_reason = nullptr,
                 bool simulate_power_loss_mid_write = false) noexcept;

    // M3 + M7: Associative retrieval with confidence & abstention policy
    RecallResult recall(uint16_t key_id,
                        const float query[kFeatureDim],
                        float min_confidence = 0.55f) const noexcept;

    // M6: Versioned revision & point-in-time rollback
    bool revise(uint16_t key_id,
                const float value[kFeatureDim],
                uint8_t source_id,
                bool trusted,
                float timestamp_s,
                RejectReason* out_reason = nullptr) noexcept;

    bool rollback(uint16_t key_id,
                  uint16_t target_version,
                  float timestamp_s,
                  RejectReason* out_reason = nullptr) noexcept;

    // M5 + M8: Consolidate redundant prototypes & decay low-importance slots
    uint32_t consolidate(float merge_dist_sq = 0.04f, float decay_factor = 0.95f) noexcept;

    // Persistence boundary & fault-injection/integrity audit primitives (M02, M09, M10)
    Snapshot export_snapshot() const noexcept;
    bool     restore_from_snapshot(const Snapshot& snap) noexcept;
    uint32_t compute_integrity_crc32() const noexcept;
    bool     verify_integrity() const noexcept;
    void     inject_bitflip_corruption(size_t byte_offset, uint8_t bit_mask) noexcept;
    uint32_t sanitize_and_recover() noexcept;

    // Observability getters
    uint16_t head(uint16_t key_id) const noexcept;
    size_t   cells(uint16_t key_id) const noexcept;
    size_t   active_prototypes_total() const noexcept;
    size_t   episode_count() const noexcept { return episode_count_; }
    float    protection_weight(uint16_t key_id) const noexcept;
    bool     ashi_gate_latched(uint16_t key_id) const noexcept;
    uint32_t journal_count() const noexcept { return journal_seq_; }
    uint32_t pruned_count() const noexcept { return pruned_prototypes_; }
    size_t   bytes() const noexcept { return sizeof(*this); }

private:
    struct PrototypeSlot {
        uint8_t  active;
        uint8_t  pad;
        uint16_t key_id;
        uint16_t version;
        uint16_t recall_count;
        float    center[kFeatureDim];
        float    importance_acc;
        float    last_timestamp_s;
        uint32_t crc32;
    };

    struct RevisionRecord {
        uint8_t  valid;
        uint8_t  source_id;
        uint16_t key_id;
        uint16_t version;
        uint16_t pad;
        float    timestamp_s;
        float    value[kFeatureDim];
        uint32_t crc32;
    };

    struct AshiKeyState {
        uint32_t count;
        float    mean;
        float    m2;
        float    accumulator;
        float    consolidated_importance;
        uint8_t  latched;
        uint8_t  pad[3];
    };

    struct EpisodeSlot {
        uint32_t event_id;
        uint16_t key_id;
        uint8_t  source_id;
        uint8_t  ashi_fired;
        float    timestamp_s;
        float    importance;
        float    features[kFeatureDim];
    };

    PrototypeSlot  prototypes_[kMaxPrototypes]{};
    RevisionRecord revisions_[kMaxKeys][kMaxRevisions]{};
    uint16_t       key_head_version_[kMaxKeys]{};
    AshiKeyState   ashi_[kMaxKeys]{};
    EpisodeSlot    episodes_[kMaxEpisodes]{};
    JournalEntry   journal_[kMaxJournalItems]{};

    size_t   episode_head_ = 0u;
    size_t   episode_count_ = 0u;
    uint32_t journal_seq_ = 0u;
    uint32_t total_observations_ = 0u;
    uint32_t total_revisions_ = 0u;
    uint32_t total_rollbacks_ = 0u;
    uint32_t ashi_triggers_ = 0u;
    uint32_t pruned_prototypes_ = 0u;
    float    last_global_ts_ = -1.0f;
    uint32_t global_crc32_ = 0u;

    static bool  is_finite_vec(const float v[kFeatureDim]) noexcept;
    static float dist_sq(const float a[kFeatureDim], const float b[kFeatureDim]) noexcept;
    static uint32_t crc32_bytes(const void* data, size_t len) noexcept;
    uint32_t compute_proto_crc(const PrototypeSlot& p) const noexcept;
    uint32_t compute_rev_crc(const RevisionRecord& r) const noexcept;
    void refresh_global_crc() noexcept;

    void append_journal(uint8_t op_type, uint16_t key_id, uint16_t version,
                        uint32_t event_id, uint8_t source_id, uint8_t ashi_fired,
                        uint8_t committed, float ts, float imp) noexcept;
    void record_revision(uint16_t key_id, uint16_t version, uint8_t source_id,
                         float ts, const float val[kFeatureDim]) noexcept;
};

} // namespace mod_m
} // namespace odin

#endif // ODIN_M_EMBEDDED_HPP
