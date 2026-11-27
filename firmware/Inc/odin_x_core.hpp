#ifndef ODIN_X_CORE_HPP
#define ODIN_X_CORE_HPP
#include <stdint.h>
#include <stdbool.h>
#define ODIN_X_DIM 4u
#define ODIN_X_ROOTS 16u
#define ODIN_X_Q15_ONE 32768
#define ODIN_X_K_MIN 1638
#define ODIN_X_K_MAX 16384
#define ODIN_X_EPSILON 12u
#define ODIN_X_SENSOR_LIMIT 30000
#define ODIN_X_CONTEXT_LIMIT 30000
#define ODIN_X_STABLE_CYCLES 4u
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    int32_t x[ODIN_X_DIM];
    uint32_t sequence;
    uint8_t valid;
} OdinMemoryRoot_t;
typedef struct {
    int32_t r[ODIN_X_DIM];
    int32_t c[ODIN_X_DIM];
    uint8_t w[ODIN_X_DIM]; /* 0..255: relevance */
    uint32_t symbolic_flags; /* bit 0 = reject, bit 1 = inhibit consolidation */
    const OdinMemoryRoot_t *history; /* optional, read-only */
} OdinMotherInput_t;
typedef enum { ODIN_X_REJECTED=0, ODIN_X_PROVISIONAL=1, ODIN_X_CONSOLIDATED=2 } OdinXConsolidation_t;
typedef enum { ODIN_X_OK=0, ODIN_X_SATURATED=1, ODIN_X_INVALID=2, ODIN_X_INHIBITED=3 } OdinXStatus_t;
typedef struct {
    int32_t x[ODIN_X_DIM];
    int32_t x_hat[ODIN_X_DIM];
    int32_t d[ODIN_X_DIM]; /* innovation, BEFORE correction */
    int32_t residual[ODIN_X_DIM]; /* r - posterior */
    uint32_t residual_l1;
    uint32_t innovation_l1;
    int16_t gain_q15[ODIN_X_DIM];
    uint32_t sequence;
    OdinXConsolidation_t consolidation;
    OdinXStatus_t status;
    bool actuator_permitted; /* always false: separate safety controller required */
} OdinXOutput_t;
void ODIN_X_Init(void);
OdinXStatus_t ODIN_X_Tick(const OdinMotherInput_t *in, OdinXOutput_t *out);
const OdinMemoryRoot_t *ODIN_X_Roots(void);
uint8_t ODIN_X_RootCount(void);

#ifdef __cplusplus
}
#endif
#endif
