#ifndef ODIN_KA_ADVANCED_HPP
#define ODIN_KA_ADVANCED_HPP

#include <stdint.h>
#include <stddef.h>

namespace odin_ka {

// ============================================================================
// CONFIGURAZIONE BOUNDED PER STM32G070RB (36 KB SRAM / 128 KB FLASH)
// ============================================================================
constexpr size_t STATE_DIM       = 4;   // Dimensione stato x_t
constexpr size_t OBS_DIM         = 4;   // Dimensione osservazione y_t
constexpr size_t ACT_DIM         = 2;   // Dimensione comando u_t
constexpr size_t LATENT_DIM      = 6;   // Spazio latente distillato (LeRobot/TD-MPC2)
constexpr size_t HORIZON_H       = 5;   // Orizzonte predittivo MPC / Forecast
constexpr size_t CAUSAL_LAGS     = 2;   // Ritardi temporali per grafo PCMCI+ leggero
constexpr size_t SCENARIO_COUNT  = 3;   // Scenari controfattuali (K4)
constexpr size_t PROTOTYPE_COUNT = 8;   // Memoria consolidata di prototipi (K6)
constexpr size_t SLOTS_PER_MOD   = 5;   // 5 algoritmi per ciascuno dei 12 moduli = 60 slot

// Aritmetica Q15.16 per determinismo su Cortex-M0+ (1.0 == 65536)
using q16_t = int32_t;
constexpr q16_t Q16_ONE  = 65536;
constexpr q16_t Q16_HALF = 32768;

inline q16_t q16_mul(q16_t a, q16_t b) {
    return static_cast<q16_t>((static_cast<int64_t>(a) * b) >> 16);
}
inline q16_t q16_abs(q16_t x) { return (x < 0) ? -x : x; }
inline q16_t q16_clamp(q16_t x, q16_t lo, q16_t hi) {
    return (x < lo) ? lo : ((x > hi) ? hi : x);
}

// ============================================================================
// R1: TRACCIABILITÀ E PROVENIENZA DEI MODELLI (60 SLOT CANDIDATI)
// ============================================================================
enum class SubsystemId : uint8_t {
    K1_EVENT_ENCODING = 0,
    K2_DYNAMIC_LEARNING,
    K3_CAUSAL_LEARNING,
    K4_COUNTERFACTUALS,
    K5_ERROR_AWARENESS,
    K6_CONSOLIDATION,
    A1_GOAL_EVALUATION,
    A2_HOMEOSTATIC_REG,
    A3_PREDICTIVE_ADAPT,
    A4_UNCERTAINTY_REG,
    A5_RECOVERY,
    A6_OUTCOME_LEARNING,
    SUBSYSTEM_COUNT = 12
};

struct ModelProvenance {
    uint16_t    model_id;         // ID univoco [0..59]
    SubsystemId subsystem;        // Sottosistema di appartenenza
    uint8_t     slot_index;       // [0..4]
    uint16_t    version_major_minor;
    uint32_t    weights_crc32;    // Hash dei pesi distillati / parametri
    uint16_t    wcet_us;          // Worst-Case Execution Time misurato su M0+
};

// ============================================================================
// R2 & R3: CONTRATTI DEI MESSAGGI TIPIZZATI (NESSUNA FUSIONE SCALARE IMPROPRIA)
// ============================================================================
struct MessageHeader {
    uint16_t schema_version;
    uint32_t sequence;
    uint32_t timestamp_ms;
    uint32_t valid_until_ms;
    uint16_t source_model_id;
    uint8_t  quality_q8;          // [0..255]

    bool is_valid(uint32_t now_ms, uint32_t last_seq) const {
        return (schema_version == 0x0100) &&
               (sequence >= last_seq) &&
               (now_ms <= valid_until_ms);
    }
};

// Input da ODIN-UNIFIED (ODIN-P/C/R/D/M/S/X)
struct ObservationFrame {
    MessageHeader header;
    q16_t         state_est[STATE_DIM];     // x_hat da ODIN-X
    q16_t         observation[OBS_DIM];     // r / sensori
    q16_t         residual_l1[OBS_DIM];     // Residui L1 da ODIN-X
    q16_t         covariance_diag[STATE_DIM];
    uint16_t      context_id;
};

// K1 Output: Event Encoding (ruptures / TCN encoder)
struct EventSet {
    MessageHeader header;
    q16_t         change_scores[OBS_DIM];   // CUSUM / Page-Hinkley scores
    q16_t         latent_z[LATENT_DIM];     // Embedding temporale TCN
    uint16_t      segment_id;
    uint8_t       changepoint_detectedMask; // Bitmask per canale
};

// K2 Output: Dynamic Learning (Adaptive State-Space / DreamerV3 RSSM distillato)
struct StateForecast {
    MessageHeader header;
    q16_t         mean_traj[HORIZON_H][STATE_DIM];
    q16_t         quantile_lo[HORIZON_H][STATE_DIM]; // 5% quantile
    q16_t         quantile_hi[HORIZON_H][STATE_DIM]; // 95% quantile
    q16_t         model_residual[STATE_DIM];
    uint16_t      calibration_id;
};

// K3 Output: Causal Learning (Tigramite PCMCI+ / Lagged Partial Correlation)
enum class CausalStatus : uint8_t {
    OBSERVATIONAL_ONLY = 0,
    INTERVENTION_VERIFIED = 1,
    CONFOUNDED_SUSPECTED = 2
};

struct CausalEdge {
    uint8_t      src_var;
    uint8_t      dst_var;
    uint8_t      lag_steps;
    q16_t        strength_q16;              // Effetto parziale stimato [-1..+1]
    q16_t        confidence_q16;            // 1 - p_value approssimato
    CausalStatus status;
};

struct CausalHypothesis {
    MessageHeader header;
    CausalEdge    edges[STATE_DIM * ACT_DIM];
    uint8_t       active_edges_count;
    uint8_t       assumptions_flags;        // Faithfulness, Causal Sufficiency, ecc.
};

// K4 Output: Counterfactuals (SCM Interventions do(u = u_alt))
struct CounterfactualBranch {
    q16_t intervened_action[ACT_DIM];       // do(U = u_k)
    q16_t predicted_state[HORIZON_H][STATE_DIM];
    q16_t causal_effect_delta[STATE_DIM];   // E[X | do(u_k)] - E[X | u_factual]
};

struct ScenarioSet {
    MessageHeader        header;
    CounterfactualBranch branches[SCENARIO_COUNT];
    uint8_t              best_branch_idx;
};

// K5 Output: Error Awareness (River Drift + Conformal Prediction)
struct CalibrationReport {
    MessageHeader header;
    q16_t         conformal_alpha_quantile; // Raggio di incertezza conforme
    q16_t         empirical_coverage;       // Coverage reale su finestra mobile
    q16_t         mahalanobis_ood_score;
    bool          drift_alert;
    bool          ood_alert;
};

// K6 Output: Consolidation (Experience Replay / EWC / Prototype Memory)
struct MemoryUpdate {
    MessageHeader header;
    uint8_t       updated_prototype_idx;
    q16_t         retention_score;
    q16_t         ewc_fisher_penalty;
    uint32_t      consolidated_hash;
};

// A1 Output: Goal Evaluation (Pareto Multi-Objective)
struct GoalSet {
    MessageHeader header;
    q16_t         target_state[STATE_DIM];
    q16_t         objective_weights[STATE_DIM]; // Pesi Pareto dinamici
    q16_t         pareto_optimality_score;
};

// A2/A3 Output: Regulation Proposal (LQR/BLF + TD-MPC2/MPPI)
struct RegulationProposal {
    MessageHeader header;
    uint32_t      proposal_id;
    q16_t         nominal_action[ACT_DIM];          // Da A2 (Homeostatic LQR)
    q16_t         predictive_traj_u[HORIZON_H][ACT_DIM]; // Da A3 (TD-MPC2 / MPPI)
    q16_t         expected_outcome[STATE_DIM];
    q16_t         lyapunov_margin;                  // > 0 indica stabilità asintotica
    q16_t         action_uncertainty[ACT_DIM];
};

// A4 Output: Uncertainty Regulation (Risk-Sensitive Conformal Gate)
struct RiskAssessment {
    MessageHeader header;
    q16_t         action_scale_limit[ACT_DIM];      // Restringimento [0..1] in Q16
    q16_t         cvar_risk_score;
    bool          confidence_gate_open;
};

// A5 Output: Recovery (Runtime Assurance Simplex FSM)
enum class RecoveryMode : uint8_t {
    NOMINAL_KA_ACTIVE = 0,
    CAUTIOUS_CONSTRAINED = 1,
    SAFE_LQR_FALLBACK = 2,
    HARDWARE_LOCKOUT = 3
};

struct RecoveryState {
    MessageHeader header;
    RecoveryMode  mode;
    uint16_t      fault_code;
    bool          fallback_requested;
};

// A6 Output: Outcome Learning (Contextual Bandit / RLS)
struct OutcomeReport {
    MessageHeader header;
    uint32_t      evaluated_proposal_id;
    q16_t         observed_effect_error[STATE_DIM];
    q16_t         incurred_cost;
    q16_t         policy_trust_update[SLOTS_PER_MOD]; // Aggiornamento pesi LinUCB
    uint8_t       constraint_violations;
};

// Output del Supervisore di Sicurezza Indipendente (R5)
struct SafeActuationCommand {
    uint32_t timestamp_ms;
    q16_t    authorized_action[ACT_DIM];
    bool     actuation_enabled;             // In Fase 1 (Shadow Mode) può essere bloccato
    RecoveryMode active_safety_mode;
};

// ============================================================================
// BRIDGE BIDIREZIONALE K <-> A (D-01, D-04)
// ============================================================================
struct KABridgeBus {
    // K -> A (Conoscenza consolidata verso Azione)
    EventSet          k1_events;
    StateForecast     k2_forecast;
    CausalHypothesis  k3_causal;
    ScenarioSet       k4_scenarios;
    CalibrationReport k5_calibration;
    MemoryUpdate      k6_memory;

    // A -> K (Proposte, Rischio ed Esiti verso Conoscenza)
    GoalSet            a1_goals;
    RegulationProposal a2_a3_proposal;
    RiskAssessment     a4_risk;
    RecoveryState      a5_recovery;
    OutcomeReport      a6_outcome;
};

// Telemetria compatta per invio DMA su USART2 (ODIN-UNIFIED)
#pragma pack(push, 1)
struct OdinKATelemetryPacket {
    uint8_t  magic[4];              // "ODKA"
    uint32_t tick;
    int16_t  forecast_x0_q8;        // Stima K2 orizzonte H=1
    int16_t  conformal_bound_q8;    // Soglia K5
    int16_t  causal_top_strength_q8;// Forza legame causale primario K3
    int16_t  proposed_u0_q8;        // Comando proposto A3
    int16_t  authorized_u0_q8;      // Comando filtrato dal Supervisore
    uint8_t  recovery_mode;         // Stato A5
    uint8_t  flags;                 // Bit 0: Changepoint, Bit 1: Drift, Bit 2: OOD, Bit 3: GateOpen
    uint16_t crc16;
};
#pragma pack(pop)

// ============================================================================
// CLASSE PRINCIPALE DEL MOTORE ODIN-KA (ZERO-HEAP, DETERMINISTICA)
// ============================================================================
class OdinKAEngine {
public:
    OdinKAEngine();

    void init(bool allow_physical_actuation = false);

    // Esecuzione deterministica di un tick (da chiamare nel loop TIM6 a 100 Hz)
    SafeActuationCommand step(const ObservationFrame& obs_in,
                              OdinKATelemetryPacket* telemetry_out = nullptr);

    const KABridgeBus& get_bus() const { return bus_; }
    const ModelProvenance* get_registry(size_t* count_out) const;

private:
    // Sottosistemi ODIN-K
    void run_K1_EventEncoding(const ObservationFrame& obs);
    void run_K2_DynamicLearning(const ObservationFrame& obs);
    void run_K3_CausalLearning(const ObservationFrame& obs);
    void run_K4_Counterfactuals(const ObservationFrame& obs);
    void run_K5_ErrorAwareness(const ObservationFrame& obs);
    void run_K6_Consolidation(const ObservationFrame& obs);

    // Sottosistemi ODIN-A
    void run_A1_GoalEvaluation(const ObservationFrame& obs);
    void run_A2_HomeostaticRegulation(const ObservationFrame& obs);
    void run_A3_PredictiveAdaptation(const ObservationFrame& obs);
    void run_A4_UncertaintyRegulation();
    void run_A5_Recovery(const ObservationFrame& obs);
    void run_A6_OutcomeLearning(const ObservationFrame& obs);

    // R5: Supervisore di sicurezza indipendente
    SafeActuationCommand run_IndependentSafetySupervisor(const ObservationFrame& obs);

    // Stato interno statico (Zero-Heap)
    KABridgeBus     bus_;
    ModelProvenance registry_[static_cast<size_t>(SubsystemId::SUBSYSTEM_COUNT) * SLOTS_PER_MOD];
    uint32_t        tick_counter_;
    bool            physical_actuation_allowed_;

    // Buffer circolari per K1 (TCN/CUSUM), K2 (Kalman/RSSM), K3 (PCMCI+ Lags)
    q16_t history_state_[CAUSAL_LAGS + 1][STATE_DIM];
    q16_t history_act_[CAUSAL_LAGS + 1][ACT_DIM];
    q16_t cusum_pos_[OBS_DIM];
    q16_t cusum_neg_[OBS_DIM];
    q16_t running_mean_[OBS_DIM];

    // Matrice di transizione adattiva K2 (A_hat, B_hat in Q16)
    q16_t A_hat_[STATE_DIM][STATE_DIM];
    q16_t B_hat_[STATE_DIM][ACT_DIM];
    q16_t last_predicted_state_[STATE_DIM];

    // Statistiche sufficienti per K3 (Correlazione parziale online)
    q16_t cov_state_act_[STATE_DIM][ACT_DIM];
    q16_t var_act_[ACT_DIM];

    // Quantili conformi per K5 (River/Conformal)
    q16_t conformal_quantile_;
    q16_t empirical_hit_rate_;

    // Memoria di prototipi per K6 (Reservoir + Centroidi)
    q16_t prototypes_[PROTOTYPE_COUNT][STATE_DIM];
    uint32_t reservoir_count_;

    // Pesi di fiducia contestuali per A6 (LinUCB / RLS)
    q16_t slot_weights_[static_cast<size_t>(SubsystemId::SUBSYSTEM_COUNT)][SLOTS_PER_MOD];
};

} // namespace odin_ka

#endif // ODIN_KA_ADVANCED_HPP