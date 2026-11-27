#pragma once
#include "odin_p_embedded.hpp"
#include "odin_c_embedded.hpp"
#include "odin_d_embedded.hpp"
#include "odin_r_embedded.hpp"
#include "odin_m_embedded.hpp"
#include "odin_s_embedded.hpp"
#include "odin_rps_embedded.hpp"
#include "odin_x_core.hpp"
#include "odin_ka_advanced.hpp"
namespace odin::native {
struct Input {
 mod_s::SensorFrame sensors;
 float quality, raw_discrimination, applied_u[2], target[4];
 uint16_t memory_key; uint8_t source_id, trusted; float importance;
 uint8_t relevance[4];
 bool evidence_valid; mod_r::EvidencePacket evidence;
 bool label_valid; float label;
 mod_rps::RuntimeTelemetry runtime;
};
struct Output {
 mod_s::StateEstimate s; mod_p::HorizonPrediction p; mod_p::PlanDecision plan;
 mod_c::ContextState c; mod_d::DiscriminationResult d; mod_r::ReasoningVerdict r;
 mod_m::RecallResult m; OdinXOutput_t x; odin_ka::SafeActuationCommand ka;
 mod_rps::SupervisorOutput rps; mod_m::RejectReason memory_reason; bool observed, valid;
};
class Engine {
 float last_=-1; bool previous_context_=false;
public:
 mod_s::OdinStateEstimator s; mod_p::OdinPredictorPlanner p; mod_c::EngineC<> c;
 mod_d::EngineD<> d; mod_r::OdinReasoner r; mod_m::OdinMemory m;
 odin_ka::OdinKAEngine ka; mod_rps::OdinRuntimeSupervisor rps;
 void init() noexcept; bool tick(const Input&, Output&) noexcept;
};
}
