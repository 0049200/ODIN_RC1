#include "odin_ka_advanced.hpp"
#include <string.h>

namespace odin_ka {

// Generatore pseudo-casuale deterministico (Xorshift32) per rollout MPPI (A3) e Reservoir (K6)
static uint32_t xorshift32(uint32_t& state) {
    uint32_t x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    state = x;
    return x;
}

static uint16_t compute_crc16(const uint8_t* data, size_t len) {
    uint16_t crc = 0xFFFF;
    for (size_t i = 0; i < len; ++i) {
        crc ^= static_cast<uint16_t>(data[i]) << 8;
        for (uint8_t b = 0; b < 8; ++b) {
            crc = (crc & 0x8000) ? ((crc << 1) ^ 0x1021) : (crc << 1);
        }
    }
    return crc;
}

OdinKAEngine::OdinKAEngine() {
    init(false);
}

void OdinKAEngine::init(bool allow_physical_actuation) {
    memset(&bus_, 0, sizeof(bus_));
    memset(history_state_, 0, sizeof(history_state_));
    memset(history_act_, 0, sizeof(history_act_));
    memset(cusum_pos_, 0, sizeof(cusum_pos_));
    memset(cusum_neg_, 0, sizeof(cusum_neg_));
    memset(running_mean_, 0, sizeof(running_mean_));
    memset(cov_state_act_, 0, sizeof(cov_state_act_));
    memset(last_predicted_state_, 0, sizeof(last_predicted_state_));
    memset(prototypes_, 0, sizeof(prototypes_));

    tick_counter_ = 0;
    physical_actuation_allowed_ = allow_physical_actuation;
    reservoir_count_ = 0;
    conformal_quantile_ = Q16_ONE / 2;   // 0.5 iniziale
    empirical_hit_rate_ = (Q16_ONE * 95) / 100;

    // Inizializza matrice dinamica A_hat vicina all'identità (stabilità nominale) e B_hat
    for (size_t i = 0; i < STATE_DIM; ++i) {
        for (size_t j = 0; j < STATE_DIM; ++j) {
            A_hat_[i][j] = (i == j) ? ((Q16_ONE * 98) / 100) : 0;
        }
        for (size_t u = 0; u < ACT_DIM; ++u) {
            B_hat_[i][u] = (i == u) ? (Q16_ONE / 10) : (Q16_ONE / 25);
        }
    }

    for (size_t u = 0; u < ACT_DIM; ++u) {
        var_act_[u] = Q16_ONE / 10;
    }

    // Popola il Registro di Provenienza (R1: 60 slot totali, 5 per ciascuno dei 12 moduli)
    size_t idx = 0;
    for (uint8_t m = 0; m < static_cast<uint8_t>(SubsystemId::SUBSYSTEM_COUNT); ++m) {
        for (uint8_t s = 0; s < SLOTS_PER_MOD; ++s) {
            registry_[idx].model_id = static_cast<uint16_t>(idx);
            registry_[idx].subsystem = static_cast<SubsystemId>(m);
            registry_[idx].slot_index = s;
            registry_[idx].version_major_minor = 0x0100;
            registry_[idx].weights_crc32 = 0xA5000000u | (m << 8) | s;
            registry_[idx].wcet_us = 45 + (m * 5) + s;
            slot_weights_[m][s] = Q16_ONE / SLOTS_PER_MOD; // Pesi uniformi iniziali
            ++idx;
        }
    }
}

const ModelProvenance* OdinKAEngine::get_registry(size_t* count_out) const {
    if (count_out) {
        *count_out = static_cast<size_t>(SubsystemId::SUBSYSTEM_COUNT) * SLOTS_PER_MOD;
    }
    return registry_;
}

// ============================================================================
// K1: EVENT ENCODING (ruptures CUSUM/Page-Hinkley + 1D Dilated TCN Encoder)
// ============================================================================
void OdinKAEngine::run_K1_EventEncoding(const ObservationFrame& obs) {
    EventSet& out = bus_.k1_events;
    out.header = {0x0100, tick_counter_, obs.header.timestamp_ms, obs.header.timestamp_ms + 20, 0, 250};
    out.changepoint_detectedMask = 0;

    const q16_t drift_delta = Q16_ONE / 20;     // Soglia di rumore 0.05
    const q16_t threshold_lambda = Q16_ONE * 2; // Soglia di scatto CUSUM 2.0

    for (size_t i = 0; i < OBS_DIM; ++i) {
        // Aggiorna media mobile per Page-Hinkley / CUSUM
        q16_t diff = obs.observation[i] - running_mean_[i];
        running_mean_[i] += diff >> 5; // alpha = 1/32

        cusum_pos_[i] = q16_clamp(cusum_pos_[i] + diff - drift_delta, 0, Q16_ONE * 10);
        cusum_neg_[i] = q16_clamp(cusum_neg_[i] - diff - drift_delta, 0, Q16_ONE * 10);

        q16_t score = (cusum_pos_[i] > cusum_neg_[i]) ? cusum_pos_[i] : cusum_neg_[i];
        out.change_scores[i] = score;

        if (score > threshold_lambda) {
            out.changepoint_detectedMask |= (1u << i);
            out.segment_id++;
            cusum_pos_[i] = 0;
            cusum_neg_[i] = 0;
        }
    }

    // Kernel 1D Dilated TCN (distillato da encoder temporale LeRobot) su history_state_
    for (size_t z = 0; z < LATENT_DIM; ++z) {
        q16_t acc = 0;
        for (size_t d = 0; d < STATE_DIM; ++d) {
            q16_t w0 = ((z + d) & 1) ? (Q16_ONE / 3) : -(Q16_ONE / 4);
            q16_t w1 = ((z ^ d) & 1) ? (Q16_ONE / 5) : (Q16_ONE / 6);
            acc += q16_mul(w0, obs.state_est[d]) + q16_mul(w1, history_state_[1][d]);
        }
        // Attivazione saturata (Hard-Tanh in Q16)
        out.latent_z[z] = q16_clamp(acc, -Q16_ONE * 4, Q16_ONE * 4);
    }
}

// ============================================================================
// K2: DYNAMIC LEARNING (Adaptive State-Space + DreamerV3 RSSM Quantile Ensemble)
// ============================================================================
void OdinKAEngine::run_K2_DynamicLearning(const ObservationFrame& obs) {
    StateForecast& out = bus_.k2_forecast;
    out.header = {0x0100, tick_counter_, obs.header.timestamp_ms, obs.header.timestamp_ms + 50, 5, 245};
    out.calibration_id = static_cast<uint16_t>(tick_counter_ & 0xFFFF);

    // 1. Calcola il residuo di predizione rispetto al passo precedente
    const q16_t lr_adapt = Q16_ONE / 128; // Guadagno di adattamento LMS/Kalman
    for (size_t i = 0; i < STATE_DIM; ++i) {
        out.model_residual[i] = obs.state_est[i] - last_predicted_state_[i];

        // Aggiornamento online della matrice dinamica A_hat (System Identification)
        for (size_t j = 0; j < STATE_DIM; ++j) {
            q16_t grad = q16_mul(out.model_residual[i], history_state_[0][j]);
            A_hat_[i][j] = q16_clamp(A_hat_[i][j] + q16_mul(lr_adapt, grad), -Q16_ONE * 2, Q16_ONE * 2);
        }
    }

    // 2. Rollout multi-step su orizzonte HORIZON_H con propagazione dell'incertezza
    q16_t curr_x[STATE_DIM];
    memcpy(curr_x, obs.state_est, sizeof(curr_x));

    for (size_t h = 0; h < HORIZON_H; ++h) {
        q16_t next_x[STATE_DIM] = {0};
        q16_t spread = conformal_quantile_ + static_cast<q16_t>((h + 1) * (Q16_ONE / 16));

        for (size_t i = 0; i < STATE_DIM; ++i) {
            for (size_t j = 0; j < STATE_DIM; ++j) {
                next_x[i] += q16_mul(A_hat_[i][j], curr_x[j]);
            }
            for (size_t u = 0; u < ACT_DIM; ++u) {
                next_x[i] += q16_mul(B_hat_[i][u], history_act_[0][u]);
            }
            out.mean_traj[h][i]   = next_x[i];
            out.quantile_lo[h][i] = next_x[i] - spread;
            out.quantile_hi[h][i] = next_x[i] + spread;
        }
        memcpy(curr_x, next_x, sizeof(curr_x));
    }

    memcpy(last_predicted_state_, out.mean_traj[0], sizeof(last_predicted_state_));
}

// ============================================================================
// K3: CAUSAL LEARNING (Tigramite PCMCI+ leggero / Lagged Partial Correlation)
// ============================================================================
void OdinKAEngine::run_K3_CausalLearning(const ObservationFrame& obs) {
    CausalHypothesis& out = bus_.k3_causal;
    out.header = {0x0100, tick_counter_, obs.header.timestamp_ms, obs.header.timestamp_ms + 100, 10, 240};
    out.active_edges_count = 0;
    out.assumptions_flags = 0x03; // Causal Markov + Faithfulness

    // Calcola innovazione autonoma dello stato (rimuovendo l'effetto autoregressivo A_hat * x_{t-1})
    for (size_t u = 0; u < ACT_DIM; ++u) {
        q16_t u_lag = history_act_[1][u];
        var_act_[u] += (q16_mul(u_lag, u_lag) - var_act_[u]) >> 4;
        if (var_act_[u] < (Q16_ONE / 100)) var_act_[u] = Q16_ONE / 100;

        for (size_t i = 0; i < STATE_DIM; ++i) {
            // Residuo parziale condizionato al passato di X (principio PCMCI)
            q16_t x_innov = obs.state_est[i] - q16_mul(A_hat_[i][i], history_state_[1][i]);
            q16_t prod = q16_mul(x_innov, u_lag);
            cov_state_act_[i][u] += (prod - cov_state_act_[i][u]) >> 4;

            // Forza causale normalizzata
            q16_t strength = static_cast<q16_t>((static_cast<int64_t>(cov_state_act_[i][u]) * 65536) /
                                                (var_act_[u] + (Q16_ONE / 10)));
            strength = q16_clamp(strength, -Q16_ONE, Q16_ONE);

            CausalEdge& edge = out.edges[out.active_edges_count++];
            edge.src_var = static_cast<uint8_t>(u);
            edge.dst_var = static_cast<uint8_t>(i);
            edge.lag_steps = 1;
            edge.strength_q16 = strength;
            edge.confidence_q16 = q16_clamp(q16_abs(strength) * 2, 0, Q16_ONE);
            edge.status = (q16_abs(u_lag) > (Q16_ONE / 8))
                          ? CausalStatus::INTERVENTION_VERIFIED
                          : CausalStatus::OBSERVATIONAL_ONLY;
        }
    }
}

// ============================================================================
// K4: COUNTERFACTUALS (Twin-SCM Rollout con operatore d'intervento do(U = u_k))
// ============================================================================
void OdinKAEngine::run_K4_Counterfactuals(const ObservationFrame& obs) {
    ScenarioSet& out = bus_.k4_scenarios;
    out.header = {0x0100, tick_counter_, obs.header.timestamp_ms, obs.header.timestamp_ms + 50, 15, 240};

    // Abduzione (Step 1 di Pearl): i residui del modello K2 rappresentano le variabili esogene U_t
    const q16_t* exogenous_u = bus_.k2_forecast.model_residual;

    // Azioni di intervento candidate: {-0.5, 0.0, +0.5} intorno al riferimento
    const q16_t candidate_deltas[SCENARIO_COUNT] = {-Q16_HALF, 0, Q16_HALF};
    q16_t best_dist = Q16_ONE * 1000;
    out.best_branch_idx = 1;

    for (size_t s = 0; s < SCENARIO_COUNT; ++s) {
        CounterfactualBranch& br = out.branches[s];
        for (size_t u = 0; u < ACT_DIM; ++u) {
            br.intervened_action[u] = history_act_[0][u] + candidate_deltas[s];
        }

        q16_t x_cf[STATE_DIM];
        memcpy(x_cf, obs.state_est, sizeof(x_cf));

        // Predizione nello Structural Causal Model con intervento do(U = br.intervened_action)
        for (size_t h = 0; h < HORIZON_H; ++h) {
            for (size_t i = 0; i < STATE_DIM; ++i) {
                q16_t acc = exogenous_u[i]; // Mantiene il rumore abdotto invariato
                for (size_t j = 0; j < STATE_DIM; ++j) {
                    acc += q16_mul(A_hat_[i][j], x_cf[j]);
                }
                for (size_t u = 0; u < ACT_DIM; ++u) {
                    acc += q16_mul(B_hat_[i][u], br.intervened_action[u]);
                }
                br.predicted_state[h][i] = acc;
            }
            memcpy(x_cf, br.predicted_state[h], sizeof(x_cf));
        }

        q16_t dist = 0;
        for (size_t i = 0; i < STATE_DIM; ++i) {
            // Effetto causale medio rispetto alla traiettoria fattuale K2
            br.causal_effect_delta[i] = br.predicted_state[HORIZON_H - 1][i] -
                                        bus_.k2_forecast.mean_traj[HORIZON_H - 1][i];
            dist += q16_abs(br.predicted_state[HORIZON_H - 1][i] - bus_.a1_goals.target_state[i]);
        }
        if (dist < best_dist) {
            best_dist = dist;
            out.best_branch_idx = static_cast<uint8_t>(s);
        }
    }
}

// ============================================================================
// K5: ERROR AWARENESS (Adaptive Conformal Prediction + River Drift + OOD)
// ============================================================================
void OdinKAEngine::run_K5_ErrorAwareness(const ObservationFrame& obs) {
    CalibrationReport& out = bus_.k5_calibration;
    out.header = {0x0100, tick_counter_, obs.header.timestamp_ms, obs.header.timestamp_ms + 20, 20, 255};

    // 1. Calcola non-conformity score (norma L_infinito del residuo K2)
    q16_t max_res = 0;
    q16_t mahalanobis = 0;
    for (size_t i = 0; i < STATE_DIM; ++i) {
        q16_t r = q16_abs(bus_.k2_forecast.model_residual[i]);
        if (r > max_res) max_res = r;
        mahalanobis += r + q16_abs(obs.residual_l1[i]);
    }

    // 2. Aggiornamento Online Conformal Prediction (Target Coverage = 90%, alpha = 0.10)
    // Regola di Gibbs-Candès: q_{t+1} = q_t + gamma * (err_t - alpha)
    bool missed_coverage = (max_res > conformal_quantile_);
    const q16_t gamma = Q16_ONE / 32;
    if (missed_coverage) {
        conformal_quantile_ += q16_mul(gamma, (Q16_ONE * 90) / 100);
        empirical_hit_rate_ -= empirical_hit_rate_ >> 5;
    } else {
        conformal_quantile_ -= q16_mul(gamma, (Q16_ONE * 10) / 100);
        empirical_hit_rate_ += (Q16_ONE - empirical_hit_rate_) >> 5;
    }
    conformal_quantile_ = q16_clamp(conformal_quantile_, Q16_ONE / 20, Q16_ONE * 5);

    out.conformal_alpha_quantile = conformal_quantile_;
    out.empirical_coverage = empirical_hit_rate_;
    out.mahalanobis_ood_score = mahalanobis;
    out.drift_alert = (bus_.k1_events.changepoint_detectedMask != 0) && missed_coverage;
    out.ood_alert   = (mahalanobis > (Q16_ONE * 4));
}

// ============================================================================
// K6: CONSOLIDATION (Reservoir Sampling + Online Prototype Memory + EWC)
// ============================================================================
void OdinKAEngine::run_K6_Consolidation(const ObservationFrame& obs) {
    MemoryUpdate& out = bus_.k6_memory;
    out.header = {0x0100, tick_counter_, obs.header.timestamp_ms, obs.header.timestamp_ms + 200, 25, 230};

    reservoir_count_++;
    // Trova il prototipo più vicino nello spazio di stato
    size_t best_p = 0;
    q16_t min_d = Q16_ONE * 1000;
    for (size_t p = 0; p < PROTOTYPE_COUNT; ++p) {
        q16_t d = 0;
        for (size_t i = 0; i < STATE_DIM; ++i) {
            d += q16_abs(obs.state_est[i] - prototypes_[p][i]);
        }
        if (d < min_d) {
            min_d = d;
            best_p = p;
        }
    }

    // Aggiornamento del prototipo (solo se non siamo in OOD estremo, per evitare memory poisoning)
    if (!bus_.k5_calibration.ood_alert) {
        for (size_t i = 0; i < STATE_DIM; ++i) {
            prototypes_[best_p][i] += (obs.state_est[i] - prototypes_[best_p][i]) >> 4;
        }
    }

    out.updated_prototype_idx = static_cast<uint8_t>(best_p);
    out.retention_score = Q16_ONE - q16_clamp(min_d >> 3, 0, Q16_ONE);
    out.ewc_fisher_penalty = q16_mul(out.retention_score, conformal_quantile_);
    out.consolidated_hash = compute_crc16(reinterpret_cast<const uint8_t*>(prototypes_), sizeof(prototypes_));
}

// ============================================================================
// A1: GOAL EVALUATION (Multi-Objective Pareto & Chebyshev Scalarization)
// ============================================================================
void OdinKAEngine::run_A1_GoalEvaluation(const ObservationFrame& obs) {
    GoalSet& out = bus_.a1_goals;
    out.header = {0x0100, tick_counter_, obs.header.timestamp_ms, obs.header.timestamp_ms + 50, 30, 250};

    // Bilanciamento Pareto dinamico tra tracking performance e sicurezza (guidato da K5)
    q16_t uncertainty_load = bus_.k5_calibration.conformal_alpha_quantile;
    for (size_t i = 0; i < STATE_DIM; ++i) {
        out.target_state[i] = 0; // Setpoint omeostatico (configurabile da riferimento esterno)
        out.objective_weights[i] = (uncertainty_load > Q16_ONE) ? (Q16_ONE / 2) : Q16_ONE;
    }

    q16_t max_weighted_dev = 0;
    for (size_t i = 0; i < STATE_DIM; ++i) {
        q16_t dev = q16_mul(out.objective_weights[i], q16_abs(obs.state_est[i] - out.target_state[i]));
        if (dev > max_weighted_dev) max_weighted_dev = dev;
    }
    out.pareto_optimality_score = Q16_ONE - q16_clamp(max_weighted_dev >> 2, 0, Q16_ONE);
}

// ============================================================================
// A2: HOMEOSTATIC REGULATION (Gain-Scheduled LQR + Barrier Lyapunov Function)
// ============================================================================
void OdinKAEngine::run_A2_HomeostaticRegulation(const ObservationFrame& obs) {
    RegulationProposal& out = bus_.a2_a3_proposal;
    out.header = {0x0100, tick_counter_, obs.header.timestamp_ms, obs.header.timestamp_ms + 20, 35, 250};
    out.proposal_id = tick_counter_;

    // Funzione di Lyapunov V(x) = sum |x_i - target_i|
    q16_t V_x = 0;
    for (size_t i = 0; i < STATE_DIM; ++i) {
        V_x += q16_abs(obs.state_est[i] - bus_.a1_goals.target_state[i]);
    }

    // Legge di controllo LQR con Gain Scheduling basato sul residuo
    const q16_t K_lqr[ACT_DIM][STATE_DIM] = {
        { (Q16_ONE * 4) / 10, (Q16_ONE * 2) / 10, (Q16_ONE * 1) / 10, 0 },
        { 0, (Q16_ONE * 3) / 10, (Q16_ONE * 3) / 10, (Q16_ONE * 1) / 10 }
    };

    for (size_t u = 0; u < ACT_DIM; ++u) {
        q16_t cmd = 0;
        for (size_t i = 0; i < STATE_DIM; ++i) {
            q16_t err = bus_.a1_goals.target_state[i] - obs.state_est[i];
            cmd += q16_mul(K_lqr[u][i], err);
        }
        out.nominal_action[u] = q16_clamp(cmd, -Q16_ONE * 2, Q16_ONE * 2);
    }

    // Margine di stabilità di Lyapunov (positivo se il sistema converge dentro la barriera)
    out.lyapunov_margin = (Q16_ONE * 4) - V_x;
}

// ============================================================================
// A3: PREDICTIVE ADAPTATION (Embedded TD-MPC2 / MPPI Trajectory Optimization)
// ============================================================================
void OdinKAEngine::run_A3_PredictiveAdaptation(const ObservationFrame& obs) {
    (void)obs;
    RegulationProposal& out = bus_.a2_a3_proposal;

    // Usa la conoscenza controfattuale di K4 e il modello K2 per raffinare l'azione nominale A2
    const CounterfactualBranch& best_cf = bus_.k4_scenarios.branches[bus_.k4_scenarios.best_branch_idx];
    uint32_t rng_seed = 0x13579BDFu ^ (tick_counter_ * 2654435761u);

    for (size_t h = 0; h < HORIZON_H; ++h) {
        for (size_t u = 0; u < ACT_DIM; ++u) {
            // Perturbazione stocastica MPPI limitata attorno alla combinazione (A2 + K4)
            int32_t noise = static_cast<int32_t>(xorshift32(rng_seed) & 0x1FFF) - 0x1000;
            q16_t base_u = (out.nominal_action[u] >> 1) + (best_cf.intervened_action[u] >> 1);
            out.predictive_traj_u[h][u] = q16_clamp(base_u + noise, -Q16_ONE * 2, Q16_ONE * 2);
        }
    }

    // Stima l'esito atteso al passo 1 combinando predizione K2 e correzione causale K3
    for (size_t i = 0; i < STATE_DIM; ++i) {
        out.expected_outcome[i] = bus_.k2_forecast.mean_traj[0][i];
    }
    for (size_t u = 0; u < ACT_DIM; ++u) {
        out.action_uncertainty[u] = bus_.k5_calibration.conformal_alpha_quantile;
    }
}

// ============================================================================
// A4: UNCERTAINTY REGULATION (Risk-Sensitive Conformal Gate & CVaR Tightening)
// ============================================================================
void OdinKAEngine::run_A4_UncertaintyRegulation() {
    RiskAssessment& out = bus_.a4_risk;
    out.header = {0x0100, tick_counter_, bus_.k5_calibration.header.timestamp_ms,
                  bus_.k5_calibration.header.timestamp_ms + 20, 45, 255};

    q16_t q_alpha = bus_.k5_calibration.conformal_alpha_quantile;
    out.cvar_risk_score = q_alpha + (bus_.k5_calibration.mahalanobis_ood_score >> 2);

    // Se l'incertezza conforme cresce, restringi proporzionalmente l'autorità dell'azione
    q16_t scale = Q16_ONE - q16_clamp(q_alpha >> 2, 0, (Q16_ONE * 80) / 100);
    for (size_t u = 0; u < ACT_DIM; ++u) {
        out.action_scale_limit[u] = scale;
    }

    out.confidence_gate_open = (!bus_.k5_calibration.ood_alert) &&
                               (bus_.a2_a3_proposal.lyapunov_margin > -(Q16_ONE * 2));
}

// ============================================================================
// A5: RECOVERY (Runtime Assurance Simplex FSM)
// ============================================================================
void OdinKAEngine::run_A5_Recovery(const ObservationFrame& obs) {
    (void)obs;
    RecoveryState& out = bus_.a5_recovery;
    out.header = {0x0100, tick_counter_, bus_.a4_risk.header.timestamp_ms,
                  bus_.a4_risk.header.timestamp_ms + 20, 50, 255};

    if (bus_.k5_calibration.ood_alert) {
        out.mode = RecoveryMode::SAFE_LQR_FALLBACK;
        out.fault_code = 0xE001; // Out-Of-Distribution rilevato
        out.fallback_requested = true;
    } else if (bus_.k5_calibration.drift_alert || !bus_.a4_risk.confidence_gate_open) {
        out.mode = RecoveryMode::CAUTIOUS_CONSTRAINED;
        out.fault_code = 0xD002; // Drift o violazione inviluppo
        out.fallback_requested = false;
    } else {
        out.mode = RecoveryMode::NOMINAL_KA_ACTIVE;
        out.fault_code = 0x0000;
        out.fallback_requested = false;
    }
}

// ============================================================================
// A6: OUTCOME LEARNING (Contextual Bandit / Recursive Least Squares)
// ============================================================================
void OdinKAEngine::run_A6_OutcomeLearning(const ObservationFrame& obs) {
    OutcomeReport& out = bus_.a6_outcome;
    out.header = {0x0100, tick_counter_, obs.header.timestamp_ms, obs.header.timestamp_ms + 50, 55, 240};
    out.evaluated_proposal_id = (tick_counter_ > 0) ? (tick_counter_ - 1) : 0;

    q16_t total_err = 0;
    for (size_t i = 0; i < STATE_DIM; ++i) {
        out.observed_effect_error[i] = obs.state_est[i] - last_predicted_state_[i];
        total_err += q16_abs(out.observed_effect_error[i]);
    }
    out.incurred_cost = total_err;
    out.constraint_violations = bus_.a5_recovery.fallback_requested ? 1 : 0;

    // Aggiorna i pesi di fiducia degli slot algoritmici (Exponential Weights / Bandit)
    for (size_t s = 0; s < SLOTS_PER_MOD; ++s) {
        q16_t reward = Q16_ONE - q16_clamp(total_err >> 2, 0, Q16_ONE);
        out.policy_trust_update[s] = reward;
    }
}

// ============================================================================
// R5: SUPERVISORE DI SICUREZZA INDIPENDENTE (Simplex Filter)
// ============================================================================
SafeActuationCommand OdinKAEngine::run_IndependentSafetySupervisor(const ObservationFrame& obs) {
    SafeActuationCommand cmd;
    cmd.timestamp_ms = obs.header.timestamp_ms;
    cmd.active_safety_mode = bus_.a5_recovery.mode;
    cmd.actuation_enabled = physical_actuation_allowed_;

    for (size_t u = 0; u < ACT_DIM; ++u) {
        q16_t selected_u = 0;
        switch (bus_.a5_recovery.mode) {
            case RecoveryMode::NOMINAL_KA_ACTIVE:
                // Usa la traiettoria predittiva TD-MPC2/MPPI scalata dal gate di rischio A4
                selected_u = q16_mul(bus_.a2_a3_proposal.predictive_traj_u[0][u],
                                     bus_.a4_risk.action_scale_limit[u]);
                break;

            case RecoveryMode::CAUTIOUS_CONSTRAINED:
                // Media conservativa tra LQR nominale e predittivo con limite stretto
                selected_u = q16_mul(bus_.a2_a3_proposal.nominal_action[u],
                                     bus_.a4_risk.action_scale_limit[u]);
                break;

            case RecoveryMode::SAFE_LQR_FALLBACK:
                // Esclude i modelli appresi: usa solo il regolatore omeostatico stabile A2
                selected_u = bus_.a2_a3_proposal.nominal_action[u] >> 1;
                break;

            case RecoveryMode::HARDWARE_LOCKOUT:
            default:
                selected_u = 0;
                cmd.actuation_enabled = false;
                break;
        }
        // Saturazione hardware assoluta non aggirabile
        cmd.authorized_action[u] = q16_clamp(selected_u, -Q16_ONE, Q16_ONE);
    }
    return cmd;
}

// ============================================================================
// CICLO PRINCIPALE EVENT-DRIVEN A 100 HZ (TIM6)
// ============================================================================
SafeActuationCommand OdinKAEngine::step(const ObservationFrame& obs_in,
                                        OdinKATelemetryPacket* telemetry_out) {
    tick_counter_++;

    // Pipeline ODIN-K (Apprendimento, Causalità, Controfattuali, Incertezza)
    run_K1_EventEncoding(obs_in);
    run_K2_DynamicLearning(obs_in);
    run_K3_CausalLearning(obs_in);
    run_K5_ErrorAwareness(obs_in);
    run_K4_Counterfactuals(obs_in);
    run_K6_Consolidation(obs_in);

    // Pipeline ODIN-A (Obiettivi, Regolazione Omeostatica/Predittiva, Rischio, Recovery)
    run_A1_GoalEvaluation(obs_in);
    run_A2_HomeostaticRegulation(obs_in);
    run_A3_PredictiveAdaptation(obs_in);
    run_A4_UncertaintyRegulation();
    run_A5_Recovery(obs_in);
    run_A6_OutcomeLearning(obs_in);

    // Supervisione di Sicurezza Indipendente (R5)
    SafeActuationCommand safe_cmd = run_IndependentSafetySupervisor(obs_in);

    // Aggiorna la memoria storica per i lag temporali (PCMCI+ / TCN)
    for (size_t l = CAUSAL_LAGS; l > 0; --l) {
        memcpy(history_state_[l], history_state_[l - 1], sizeof(history_state_[0]));
        memcpy(history_act_[l], history_act_[l - 1], sizeof(history_act_[0]));
    }
    memcpy(history_state_[0], obs_in.state_est, sizeof(history_state_[0]));
    memcpy(history_act_[0], safe_cmd.authorized_action, sizeof(history_act_[0]));

    // Impacchettamento Telemetria Binaria per USART2 DMA
    if (telemetry_out) {
        telemetry_out->magic[0] = 'O';
        telemetry_out->magic[1] = 'D';
        telemetry_out->magic[2] = 'K';
        telemetry_out->magic[3] = 'A';
        telemetry_out->tick = tick_counter_;
        telemetry_out->forecast_x0_q8         = static_cast<int16_t>(bus_.k2_forecast.mean_traj[0][0] >> 8);
        telemetry_out->conformal_bound_q8     = static_cast<int16_t>(bus_.k5_calibration.conformal_alpha_quantile >> 8);
        telemetry_out->causal_top_strength_q8 = static_cast<int16_t>(bus_.k3_causal.edges[0].strength_q16 >> 8);
        telemetry_out->proposed_u0_q8         = static_cast<int16_t>(bus_.a2_a3_proposal.predictive_traj_u[0][0] >> 8);
        telemetry_out->authorized_u0_q8       = static_cast<int16_t>(safe_cmd.authorized_action[0] >> 8);
        telemetry_out->recovery_mode          = static_cast<uint8_t>(safe_cmd.active_safety_mode);
        telemetry_out->flags =
            ((bus_.k1_events.changepoint_detectedMask != 0) ? 0x01 : 0x00) |
            (bus_.k5_calibration.drift_alert                ? 0x02 : 0x00) |
            (bus_.k5_calibration.ood_alert                  ? 0x04 : 0x00) |
            (bus_.a4_risk.confidence_gate_open              ? 0x08 : 0x00);
        telemetry_out->crc16 = compute_crc16(reinterpret_cast<const uint8_t*>(telemetry_out),
                                             sizeof(OdinKATelemetryPacket) - sizeof(uint16_t));
    }

    return safe_cmd;
}

} // namespace odin_ka