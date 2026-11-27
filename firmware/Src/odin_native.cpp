#include "odin_native.hpp"
namespace odin::native {
void Engine::init() noexcept {s.reset();p.reset();c.reset();d.reset();r.reset();m.reset();rps.reset();ka.init(false);ODIN_X_Init();last_=-1;previous_context_=false;}
bool Engine::tick(const Input& in, Output& o) noexcept {
 o={}; float t=in.sensors.timestamp_s;
 if(!std::isfinite(t)||t<0||t<=last_||t>4000000.f||!std::isfinite(in.quality)||in.quality<0||in.quality>1||!std::isfinite(in.raw_discrimination)||in.memory_key>=mod_m::kMaxKeys||!std::isfinite(in.importance)||in.importance<0)return false;
 for(unsigned i=0;i<4;++i){if(!std::isfinite(in.target[i])||std::fabs(in.target[i])>8)return false;if((in.sensors.valid_mask&(1u<<i))&&(!std::isfinite(in.sensors.primary[i])||std::fabs(in.sensors.primary[i])>8))return false;if((in.sensors.valid_mask&(16u<<i))&&(!std::isfinite(in.sensors.redundant[i])||std::fabs(in.sensors.redundant[i])>8))return false;}
 for(float u:in.applied_u)if(!std::isfinite(u)||std::fabs(u)>8)return false;
 if(in.label_valid&&(!std::isfinite(in.label)||in.label<0||in.label>1))return false;
 last_=t;o.s=s.update(in.sensors);auto rt=in.runtime;rt.timestamp_s=t;rt.actuator_cmd=0;rt.crc_ok=rt.crc_ok&&s.verify_estimate(o.s);o.rps=rps.supervise(rt);
 if(!rps.verify_output(o.rps)||o.rps.state==mod_rps::HealthState::SAFE_HOLD||o.rps.state==mod_rps::HealthState::LOCKOUT||in.quality==0)return false;
 mod_p::StateSample sample{};sample.step_id=in.sensors.frame_id;sample.timestamp_s=t;std::array<float,4> f{};
 for(unsigned i=0;i<4;++i){f[i]=o.s.x_hat[i];if(!std::isfinite(f[i])||std::fabs(f[i])>8)return false;sample.x[i]=f[i];}for(unsigned i=0;i<2;++i)sample.u[i]=in.applied_u[i];
 if(!p.ingest(sample))return false;o.p=p.predict_horizon(in.applied_u);o.plan=p.plan_action(in.target);
 o.m=m.recall(in.memory_key,f.data());if(in.label_valid&&previous_context_)c.learn(in.label);o.c=c.project(t,f,o.m.found?o.m.confidence:0.f);previous_context_=o.c.valid;o.d=d.process(t,in.raw_discrimination,in.quality);
 if(in.evidence_valid){auto ev=in.evidence;ev.timestamp_s=t;r.assert_evidence(ev);}else if(o.d.d2_class>=0){mod_r::EvidencePacket ev{};ev.source_id=in.source_id;ev.hypothesis_id=static_cast<uint16_t>(o.d.d2_class);ev.timestamp_s=t;ev.support=o.d.d2_posterior;ev.reliability=in.quality;r.assert_evidence(ev);}o.r=r.evaluate();
 OdinMotherInput_t x{};for(unsigned i=0;i<4;++i){x.r[i]=static_cast<int32_t>(f[i]*500.f);x.c[i]=o.m.found?static_cast<int32_t>(o.m.features[i]*500.f):0;x.w[i]=in.relevance[i];}x.history=ODIN_X_Roots();if(o.r.abstained)x.symbolic_flags|=2u;if(ODIN_X_Tick(&x,&o.x)!=ODIN_X_OK)return false;
 odin_ka::ObservationFrame obs{};auto ms=static_cast<uint32_t>(t*1000.f);obs.header={0x0100,o.x.sequence,ms,ms+100,0,static_cast<uint8_t>(in.quality*255.f)};obs.context_id=static_cast<uint16_t>(o.d.d3_segment_id);
 for(unsigned i=0;i<4;++i){obs.state_est[i]=static_cast<int32_t>(o.x.x[i]*131.072f);obs.observation[i]=static_cast<int32_t>(f[i]*65536.f);obs.residual_l1[i]=static_cast<int32_t>(std::fabs(static_cast<float>(o.x.residual[i]))*131.072f);}o.ka=ka.step(obs,nullptr);
 mod_m::Sample mem{};mem.key_id=in.memory_key;mem.event_id=in.sensors.frame_id;mem.timestamp_s=t;mem.importance=in.importance;mem.source_id=in.source_id;mem.trusted=in.trusted;for(unsigned i=0;i<4;++i)mem.features[i]=f[i];o.observed=m.observe(mem,&o.memory_reason);o.valid=o.c.valid&&o.d.valid&&m.verify_integrity();return o.valid;
}
}
