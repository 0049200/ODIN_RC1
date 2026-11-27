/**
 * @file odin_native_validation_suite.hpp
 * @brief Embedded Comparative & Integration Validation Contracts (V01 - V10)
 *        Compares Unified ODIN Native Pipeline vs Isolated / Uncoupled Execution
 *        on STM32G070RBTx (ARM Cortex-M0+, C++17 freestanding, zero heap).
 */

#ifndef ODIN_NATIVE_VALIDATION_SUITE_HPP
#define ODIN_NATIVE_VALIDATION_SUITE_HPP

#include <cstdint>

struct ValidationResultSlot {
    uint16_t test_code;
    uint8_t  passed;
    uint8_t  completed;
    uint64_t simulated_events;
    uint32_t physical_ticks_executed;
    float    unified_metric;      // Primary metric under full unified pipeline
    float    isolated_metric;     // Baseline metric when modules run uncoupled / isolated
    float    synergy_gain;        // Quantitative advantage of unified architecture
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
