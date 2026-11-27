/**
 * @file main.cpp
 * @brief Bare-Metal Embedded Validation Suite for ODIN Native RC1 (V01 - V10)
 *        Target: STM32G070RBTx (ARM Cortex-M0+, ISO C++17 freestanding, zero heap).
 */

#include "odin_native.hpp"

#if defined(__has_include)
#  if __has_include("odin_native_validation_suite.hpp")
#    include "odin_native_validation_suite.hpp"
#  endif
#endif

#ifndef ODIN_NATIVE_VALIDATION_SUITE_HPP
#define ODIN_NATIVE_VALIDATION_SUITE_HPP
#include <cstdint>

struct ValidationResultSlot {
    uint16_t test_code;
    uint8_t  passed;
    uint8_t  completed;
    uint64_t simulated_events;
    uint32_t physical_ticks_executed;
    float    unified_metric;
    float    isolated_metric;
    float    synergy_gain;
    uint32_t fault_count_detected;
    uint32_t recovery_count;
    uint32_t min_cycles;
    uint32_t avg_cycles;
    uint32_t max_cycles;
    uint32_t sram_footprint_bytes;
};

struct NativeValidationCampaign {
    ValidationResultSlot results[10];
    uint32_t tests_passed_count;
    uint32_t tests_completed_count;
    uint32_t engine_sram_bytes;
    uint32_t output_sram_bytes;
    uint8_t  campaign_passed;
    uint8_t  done;
    uint8_t  pad[2];
};
#endif

#include <cmath>
#include <cstring>

namespace {

constexpr uintptr_t kSysTickBase = 0xE000E010u;
volatile uint32_t& SYST_CSR = *reinterpret_cast<volatile uint32_t*>(kSysTickBase + 0x00u);
volatile uint32_t& SYST_RVR = *reinterpret_cast<volatile uint32_t*>(kSysTickBase + 0x04u);
volatile uint32_t& SYST_CVR = *reinterpret_cast<volatile uint32_t*>(kSysTickBase + 0x08u);

inline void systick_init() noexcept {
    SYST_CSR = 0u;
    SYST_RVR = 0x00FFFFFFu;
    SYST_CVR = 0u;
    SYST_CSR = 0x5u;
}

inline uint32_t systick_now() noexcept {
    return SYST_CVR & 0x00FFFFFFu;
}

inline uint32_t systick_elapsed(uint32_t start, uint32_t end) noexcept {
    return (start >= end) ? (start - end) : (start + (0x01000000u - end));
}

odin::native::Engine g_engine;
odin::native::Output g_out;

inline void configure_standard_calibration(odin::native::Engine& eng) noexcept {
    eng.init();
    eng.d.set_class_model(0u, 0.20f, 0.15f);
    eng.d.set_class_model(1u, 1.50f, 0.25f);
    eng.d.set_class_model(2u, 3.50f, 0.35f);
    (void)eng.r.register_rule(0u, 0u, 0.55f, 0.45f, 10u, 5u);
    (void)eng.r.register_rule(1u, 1u, 0.55f, 0.45f, 20u, 10u);
    (void)eng.r.register_rule(2u, 2u, 0.55f, 0.45f, 30u, 20u);
}

inline odin::native::Input make_nominal_input(uint32_t id, float ts, float val, float raw_d) noexcept {
    odin::native::Input in{};
    in.sensors.frame_id = id;
    in.sensors.timestamp_s = ts;
    in.sensors.valid_mask = 0xFFu;
    for (size_t c = 0u; c < 4u; ++c) {
        in.sensors.primary[c] = val;
        in.sensors.redundant[c] = val;
        in.relevance[c] = 160u;
        in.target[c] = 0.20f;
    }
    in.quality = 1.0f;
    in.raw_discrimination = raw_d;
    in.applied_u[0] = 0.0f;
    in.applied_u[1] = 0.0f;
    in.memory_key = 0u;
    in.source_id = 1u;
    in.trusted = 1u;
    in.importance = 0.40f;
    in.evidence_valid = false;
    in.label_valid = false;
    in.runtime.tick_id = id;
    in.runtime.timestamp_s = ts;
    in.runtime.actuator_cmd = 0.0f;
    in.runtime.current_ma = 350.0f;
    in.runtime.temp_c = 36.0f;
    in.runtime.wcet_cycles = 25000u;
    in.runtime.crc_ok = 1u;
    in.runtime.watchdog_fed = 1u;
    return in;
}

} // namespace

extern "C" {
volatile uint32_t g_val_progress = 0u;
volatile uint32_t g_val_passed   = 0u;
volatile uint8_t  g_val_done     = 0u;
volatile NativeValidationCampaign g_val_campaign{};
}

namespace {

inline void publish_val(size_t idx, const ValidationResultSlot& src, uint32_t& acc) noexcept {
    volatile ValidationResultSlot* dst = &g_val_campaign.results[idx];
    dst->test_code = src.test_code;
    dst->passed = src.passed;
    dst->simulated_events = src.simulated_events;
    dst->physical_ticks_executed = src.physical_ticks_executed;
    dst->unified_metric = src.unified_metric;
    dst->isolated_metric = src.isolated_metric;
    dst->synergy_gain = src.synergy_gain;
    dst->fault_count_detected = src.fault_count_detected;
    dst->recovery_count = src.recovery_count;
    dst->min_cycles = src.min_cycles;
    dst->avg_cycles = src.avg_cycles;
    dst->max_cycles = src.max_cycles;
    dst->sram_footprint_bytes = src.sram_footprint_bytes;
    dst->completed = 1u;

    g_val_campaign.tests_completed_count = static_cast<uint32_t>(idx + 1u);
    g_val_progress = static_cast<uint32_t>(idx + 1u);
    if (src.passed) {
        ++acc;
        g_val_campaign.tests_passed_count = acc;
        g_val_passed = acc;
    }
}

ValidationResultSlot run_V01() noexcept {
    configure_standard_calibration(g_engine);
    constexpr uint32_t kTicks = 100u;
    uint32_t min_c = 0xFFFFFFFFu, max_c = 0u, faults = 0u;
    uint64_t tot_c = 0ull;
    float ts = 0.0f;

    for (uint32_t k = 1u; k <= kTicks; ++k) {
        ts += 0.05f;
        auto in = make_nominal_input(k, ts, 0.20f, 0.20f);
        const uint32_t c0 = systick_now();
        const bool ok = g_engine.tick(in, g_out);
        const uint32_t dt = systick_elapsed(c0, systick_now());
        if (!ok || !g_out.valid || !g_engine.m.verify_integrity() || !g_engine.s.verify_estimate(g_out.s)) {
            ++faults;
        }
        if (dt < min_c) min_c = dt;
        if (dt > max_c) max_c = dt;
        tot_c += dt;
    }

    const bool sync_ok = (g_out.x.sequence == kTicks) &&
                         (g_engine.p.samples_seen() == kTicks) &&
                         (g_engine.s.frames_processed() == kTicks) &&
                         (g_engine.rps.ticks_processed() == kTicks);
    const uint8_t pass = (faults == 0u && sync_ok) ? 1u : 0u;
    return {1u, pass, 1u, kTicks, kTicks, 100.0f, 0.0f, 100.0f, faults, 0u,
            min_c, static_cast<uint32_t>(tot_c / kTicks), max_c, static_cast<uint32_t>(sizeof(g_engine))};
}

ValidationResultSlot run_V02() noexcept {
    configure_standard_calibration(g_engine);
    float ts = 0.0f;
    for (uint32_t k = 1u; k <= 20u; ++k) {
        ts += 0.05f;
        (void)g_engine.tick(make_nominal_input(k, ts, 0.50f, 0.20f), g_out);
    }

    ts += 0.05f;
    auto spike_in = make_nominal_input(21u, ts, 0.50f, 0.20f);
    spike_in.sensors.primary[0] = 6.50f;
    spike_in.sensors.redundant[0] = 6.50f;

    const uint32_t c0 = systick_now();
    const bool ok = g_engine.tick(spike_in, g_out);
    const uint32_t dt = systick_elapsed(c0, systick_now());

    const float unified_err = std::fabs(g_out.s.x_hat[0] - 0.50f);
    const float isolated_raw_err = std::fabs(spike_in.sensors.primary[0] - 0.50f);
    const float gain = isolated_raw_err - unified_err;
    const uint8_t pass = (ok && ((g_out.s.outlier_rejected_mask & 1u) != 0u) && unified_err < 0.15f && gain > 5.5f) ? 1u : 0u;
    return {2u, pass, 1u, 1ull, 1u, unified_err, isolated_raw_err, gain, 1u, 1u,
            dt, dt, dt, static_cast<uint32_t>(sizeof(g_engine))};
}

ValidationResultSlot run_V03() noexcept {
    configure_standard_calibration(g_engine);
    float ts = 0.0f;
    for (uint32_t k = 1u; k <= 12u; ++k) {
        ts += 0.05f;
        auto in = make_nominal_input(k, ts, 0.20f, 1.50f);
        in.evidence_valid = true;
        in.evidence = {static_cast<uint16_t>(k % 2u), 1u, ts, 0.90f, 0.95f, static_cast<uint8_t>(k % 2u)};
        (void)g_engine.tick(in, g_out);
    }

    const uint32_t c0 = systick_now();
    ts += 0.05f;
    auto check_in = make_nominal_input(13u, ts, 0.20f, 1.50f);
    check_in.evidence_valid = true;
    check_in.evidence = {1u, 1u, ts, 0.90f, 0.95f, 1u};
    (void)g_engine.tick(check_in, g_out);
    const uint32_t dt = systick_elapsed(c0, systick_now());

    const float unified_false_consolidation = (g_out.x.consolidation == ODIN_X_CONSOLIDATED) ? 1.0f : 0.0f;
    const float isolated_false_consolidation = 1.0f;
    const uint8_t pass = (g_out.r.abstained == 1u && g_out.x.consolidation != ODIN_X_CONSOLIDATED) ? 1u : 0u;
    return {3u, pass, 1u, 13ull, 13u, unified_false_consolidation, isolated_false_consolidation, 1.0f,
            1u, 0u, dt, dt, dt, static_cast<uint32_t>(sizeof(g_engine))};
}

ValidationResultSlot run_V04() noexcept {
    configure_standard_calibration(g_engine);
    float ts = 0.0f;
    uint32_t min_c = 0xFFFFFFFFu, max_c = 0u;
    uint64_t tot_c = 0ull;

    for (uint32_t k = 1u; k <= 15u; ++k) {
        ts += 0.05f;
        auto in = make_nominal_input(k, ts, 0.80f, 3.50f);
        const uint32_t c0 = systick_now();
        (void)g_engine.tick(in, g_out);
        const uint32_t dt = systick_elapsed(c0, systick_now());
        if (dt < min_c) min_c = dt;
        if (dt > max_c) max_c = dt;
        tot_c += dt;
    }

    const float unified_action = static_cast<float>(g_out.r.recommended_action);
    const float isolated_d_action = 0.0f;
    const uint8_t pass = (g_out.d.d2_class == 2 && g_out.r.rule_fired == 1u && g_out.r.recommended_action == 30u) ? 1u : 0u;
    return {4u, pass, 1u, 15ull, 15u, unified_action, isolated_d_action, unified_action, 0u, 0u,
            min_c, static_cast<uint32_t>(tot_c / 15u), max_c, static_cast<uint32_t>(sizeof(g_engine))};
}

ValidationResultSlot run_V05() noexcept {
    configure_standard_calibration(g_engine);
    float ts = 0.0f;
    for (uint32_t k = 1u; k <= 10u; ++k) {
        ts += 0.05f;
        auto in = make_nominal_input(k, ts, 0.60f, 0.20f);
        in.importance = (k == 6u) ? 5.0f : 0.3f;
        in.label_valid = true;
        in.label = 1.0f;
        (void)g_engine.tick(in, g_out);
    }

    const uint32_t c0 = systick_now();
    ts += 0.05f;
    auto q_in = make_nominal_input(11u, ts, 0.60f, 0.20f);
    q_in.label_valid = true;
    q_in.label = 1.0f;
    (void)g_engine.tick(q_in, g_out);
    const uint32_t dt = systick_elapsed(c0, systick_now());

    const float unified_uncertainty = g_out.c.uncertainty;
    const float isolated_c_uncertainty = 1.0f;
    const float uncertainty_reduction = isolated_c_uncertainty - unified_uncertainty;
    const uint8_t pass = (g_out.m.found && unified_uncertainty < 0.15f && uncertainty_reduction > 0.85f) ? 1u : 0u;
    return {5u, pass, 1u, 11ull, 11u, unified_uncertainty, isolated_c_uncertainty, uncertainty_reduction,
            0u, 1u, dt, dt, dt, static_cast<uint32_t>(sizeof(g_engine))};
}

ValidationResultSlot run_V06() noexcept {
    configure_standard_calibration(g_engine);
    (void)g_engine.tick(make_nominal_input(1u, 0.05f, 0.20f, 0.20f), g_out);
    const uint32_t j_before = g_engine.m.journal_count();

    auto fault_in = make_nominal_input(2u, 0.10f, 3.00f, 3.50f);
    fault_in.runtime.current_ma = 2400.0f;

    const uint32_t c0 = systick_now();
    const bool tick_allowed = g_engine.tick(fault_in, g_out);
    const uint32_t dt = systick_elapsed(c0, systick_now());

    const uint32_t j_after = g_engine.m.journal_count();
    const uint8_t pass = (!tick_allowed && g_out.rps.state == odin::mod_rps::HealthState::SAFE_HOLD &&
                          j_after == j_before) ? 1u : 0u;
    return {6u, pass, 1u, 2ull, 2u, static_cast<float>(j_after - j_before), 1.0f, 1.0f,
            1u, 0u, dt, dt, dt, static_cast<uint32_t>(sizeof(g_engine))};
}

ValidationResultSlot run_V07() noexcept {
    configure_standard_calibration(g_engine);
    float ts = 0.0f;
    for (uint32_t k = 1u; k <= 12u; ++k) {
        ts += 0.05f;
        (void)g_engine.tick(make_nominal_input(k, ts, 0.40f, 0.20f), g_out);
    }
    const uint32_t c0 = systick_now();
    ts += 0.05f;
    const bool ok = g_engine.tick(make_nominal_input(13u, ts, 0.40f, 0.20f), g_out);
    const uint32_t dt = systick_elapsed(c0, systick_now());

    const bool advisory_only = (!g_out.x.actuator_permitted) && (!g_out.ka.actuation_enabled);
    const uint8_t pass = (ok && advisory_only && g_out.x.status == ODIN_X_OK) ? 1u : 0u;
    return {7u, pass, 1u, 13ull, 13u, static_cast<float>(g_out.x.residual_l1),
            static_cast<float>(g_out.x.innovation_l1), 100.0f, 0u, 0u,
            dt, dt, dt, static_cast<uint32_t>(sizeof(g_engine))};
}

ValidationResultSlot run_V08() noexcept {
    configure_standard_calibration(g_engine);
    float ts = 0.0f;
    for (uint32_t k = 1u; k <= 20u; ++k) {
        ts += 0.05f;
        const float ramp = 0.05f * static_cast<float>(k);
        (void)g_engine.tick(make_nominal_input(k, ts, ramp, 0.20f), g_out);
    }

    const uint32_t c0 = systick_now();
    bool all_ok = true;
    for (uint32_t b = 1u; b <= 3u; ++b) {
        ts += 0.05f;
        auto blind = make_nominal_input(20u + b, ts, 0.0f, 0.20f);
        blind.sensors.valid_mask = 0x00u;
        if (!g_engine.tick(blind, g_out)) all_ok = false;
    }
    const uint32_t dt = systick_elapsed(c0, systick_now());

    const float expected_ramp = 0.05f * 23.0f;
    const float coast_err = std::fabs(g_out.s.x_hat[0] - expected_ramp);
    const uint8_t pass = (all_ok && g_out.valid && coast_err < 0.15f) ? 1u : 0u;
    return {8u, pass, 1u, 23ull, 23u, coast_err, expected_ramp, expected_ramp - coast_err,
            1u, 1u, dt / 3u, dt / 3u, dt / 3u, static_cast<uint32_t>(sizeof(g_engine))};
}

ValidationResultSlot run_V09() noexcept {
    configure_standard_calibration(g_engine);
    (void)g_engine.tick(make_nominal_input(1u, 1.0f, 0.30f, 0.20f), g_out);
    const uint32_t seq_before = g_out.x.sequence;

    const uint32_t c0 = systick_now();
    auto bad_ts = make_nominal_input(2u, 1.0f, 0.30f, 0.20f);
    const bool r1 = g_engine.tick(bad_ts, g_out);

    auto bad_oob = make_nominal_input(3u, 2.0f, 12.0f, 0.20f);
    const bool r2 = g_engine.tick(bad_oob, g_out);

    auto bad_nan = make_nominal_input(4u, 3.0f, std::sqrt(-1.0f), 0.20f);
    const bool r3 = g_engine.tick(bad_nan, g_out);
    const uint32_t dt = systick_elapsed(c0, systick_now());

    const uint32_t rejected = (!r1 ? 1u : 0u) + (!r2 ? 1u : 0u) + (!r3 ? 1u : 0u);
    const uint8_t pass = (rejected == 3u && g_engine.s.frames_processed() == seq_before && g_engine.m.verify_integrity()) ? 1u : 0u;
    return {9u, pass, 1u, 3ull, 3u, static_cast<float>(rejected), 100.0f, 0.0f, rejected, 0u,
            dt, dt, dt, static_cast<uint32_t>(sizeof(g_engine))};
}

ValidationResultSlot run_V10() noexcept {
    configure_standard_calibration(g_engine);
    constexpr uint32_t kTicks = 200u;
    uint32_t min_c = 0xFFFFFFFFu, max_c = 0u, faults = 0u;
    uint64_t tot_c = 0ull;
    float ts = 0.0f;

    for (uint32_t k = 1u; k <= kTicks; ++k) {
        ts += 0.02f;
        const float sig = 0.5f * std::sin(ts);
        auto in = make_nominal_input(k, ts, sig, 0.20f);
        const uint32_t c0 = systick_now();
        const bool ok = g_engine.tick(in, g_out);
        const uint32_t dt = systick_elapsed(c0, systick_now());
        if (!ok || !g_engine.m.verify_integrity()) ++faults;
        if (dt < min_c) min_c = dt;
        if (dt > max_c) max_c = dt;
        tot_c += dt;
    }

    const uint32_t avg_c = static_cast<uint32_t>(tot_c / kTicks);
    const uint32_t total_bytes = static_cast<uint32_t>(sizeof(g_engine) + sizeof(g_out));
    const uint8_t pass = (faults == 0u && total_bytes < 16384u) ? 1u : 0u;
    return {10u, pass, 1u, kTicks, kTicks, static_cast<float>(avg_c) / 64.0f,
            static_cast<float>(max_c) / 64.0f, static_cast<float>(total_bytes), faults, 0u,
            min_c, avg_c, max_c, total_bytes};
}

} // namespace

int main(void) {
    systick_init();
    configure_standard_calibration(g_engine);

    g_val_progress = 0u;
    g_val_passed   = 0u;
    g_val_done     = 0u;
    g_val_campaign.tests_completed_count = 0u;
    g_val_campaign.tests_passed_count    = 0u;
    g_val_campaign.engine_sram_bytes     = static_cast<uint32_t>(sizeof(g_engine));
    g_val_campaign.output_sram_bytes     = static_cast<uint32_t>(sizeof(g_out));
    g_val_campaign.campaign_passed       = 0u;
    g_val_campaign.done                  = 0u;

    uint32_t acc = 0u;
    publish_val(0u, run_V01(), acc);
    publish_val(1u, run_V02(), acc);
    publish_val(2u, run_V03(), acc);
    publish_val(3u, run_V04(), acc);
    publish_val(4u, run_V05(), acc);
    publish_val(5u, run_V06(), acc);
    publish_val(6u, run_V07(), acc);
    publish_val(7u, run_V08(), acc);
    publish_val(8u, run_V09(), acc);
    publish_val(9u, run_V10(), acc);

    g_val_campaign.campaign_passed = (acc == 10u) ? 1u : 0u;
    g_val_campaign.done            = 1u;
    g_val_done                     = 1u;

    while (true) {
    }
}
