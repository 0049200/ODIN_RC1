#include "odin_x_core.hpp"
#include <string.h>
#include <limits.h>
namespace {
struct State {
    int32_t prior[ODIN_X_DIM];
    OdinMemoryRoot_t roots[ODIN_X_ROOTS];
    uint32_t sequence;
    uint8_t root_head, root_count, stable;
};
static State core;
static uint32_t magnitude(int32_t v) { return v < 0 ? static_cast<uint32_t>(-(static_cast<int64_t>(v))) : static_cast<uint32_t>(v); }
static int32_t round_q15(int32_t v, int32_t gain) {
    const int64_t prod = static_cast<int64_t>(v) * gain;
    if (prod >= 0) return static_cast<int32_t>((prod + 16384) / ODIN_X_Q15_ONE);
    return -static_cast<int32_t>((-prod + 16384) / ODIN_X_Q15_ONE);
}
static uint32_t add_u32(uint32_t a, uint32_t b) { return (UINT32_MAX - a < b) ? UINT32_MAX : a + b; }
static int16_t gain_for(uint32_t error, uint8_t relevance) {
    /* bounded adaptive gain, monotonically increases with error energy proxy */
    const uint32_t energy = error > 4096u ? 4096u : error;
    const uint32_t adaptive = (energy * static_cast<uint32_t>(ODIN_X_K_MAX - ODIN_X_K_MIN)) / 4096u;
    const uint32_t weighted = (adaptive * relevance) / 255u;
    return static_cast<int16_t>(ODIN_X_K_MIN + weighted);
}
}
extern "C" void ODIN_X_Init(void) { memset(&core, 0, sizeof(core)); }
extern "C" OdinXStatus_t ODIN_X_Tick(const OdinMotherInput_t *in, OdinXOutput_t *out) {
    if (!out) return ODIN_X_INVALID;
    memset(out, 0, sizeof(*out));
    out->status = ODIN_X_INVALID;
    out->consolidation = ODIN_X_REJECTED;
    if (!in) return out->status;
    /* Reject atomically: do not alter state or memory on invalid inputs. */
    if (in->symbolic_flags & 1u) { out->status = ODIN_X_INHIBITED; return out->status; }
    for (uint32_t i=0; i<ODIN_X_DIM; ++i) {
        if (in->r[i] < -ODIN_X_SENSOR_LIMIT || in->r[i] > ODIN_X_SENSOR_LIMIT ||
            in->c[i] < -ODIN_X_CONTEXT_LIMIT || in->c[i] > ODIN_X_CONTEXT_LIMIT) {
            out->status = ODIN_X_SATURATED; return out->status;
        }
    }
    /* Context is deliberately not mixed into observations without a calibrated model.
       Historical roots are owned internally; external history is read-only metadata. */
    uint32_t innovation_l1=0, residual_l1=0;
    for (uint32_t i=0; i<ODIN_X_DIM; ++i) {
        const int32_t prior = core.prior[i];
        const int32_t innovation = in->r[i] - prior;
        const int16_t gain = gain_for(magnitude(innovation), in->w[i]);
        const int32_t correction = round_q15(innovation, gain);
        const int32_t posterior = prior + correction;
        const int32_t residual = in->r[i] - posterior;
        out->x_hat[i]=prior;
        out->d[i]=innovation;
        out->gain_q15[i]=gain;
        out->x[i]=posterior;
        out->residual[i]=residual;
        innovation_l1=add_u32(innovation_l1,magnitude(innovation));
        residual_l1=add_u32(residual_l1,magnitude(residual));
    }
    out->innovation_l1=innovation_l1;
    out->residual_l1=residual_l1;
    out->sequence=++core.sequence;
    out->status=ODIN_X_OK;
    out->consolidation=ODIN_X_PROVISIONAL;
    /* Prior for next cycle is posterior; this yields contraction for constant inputs
       outside integer quantization deadbands. */
    for (uint32_t i=0; i<ODIN_X_DIM; ++i) core.prior[i]=out->x[i];
    if (residual_l1 <= ODIN_X_EPSILON && !(in->symbolic_flags & 2u)) {
        if (core.stable < ODIN_X_STABLE_CYCLES) ++core.stable;
    } else core.stable=0;
    if (core.stable >= ODIN_X_STABLE_CYCLES) {
        OdinMemoryRoot_t *root=&core.roots[core.root_head];
        for (uint32_t i=0; i<ODIN_X_DIM; ++i) root->x[i]=out->x[i];
        root->sequence=out->sequence;
        root->valid=1;
        core.root_head=static_cast<uint8_t>((core.root_head+1u)%ODIN_X_ROOTS);
        if (core.root_count<ODIN_X_ROOTS) ++core.root_count;
        out->consolidation=ODIN_X_CONSOLIDATED;
    }
    return out->status;
}
extern "C" const OdinMemoryRoot_t *ODIN_X_Roots(void) { return core.roots; }
extern "C" uint8_t ODIN_X_RootCount(void) { return core.root_count; }
