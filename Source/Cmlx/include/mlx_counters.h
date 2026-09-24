#ifndef MLX_COUNTERS_H
#define MLX_COUNTERS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Process-wide Metal work counters from the numen-tech/mlx fork
 * (`mlx::core::metal::counters()`), summed over every stream since process
 * start or the last `mlx_counters_reset()`:
 *
 * - dispatches: compute kernel dispatches
 * - commits: command-buffer commits (including automatic buffer splits)
 * - syncs: explicit stream synchronizations (`synchronize()`)
 * - waits: host blocking waits on GPU completion, counted per wait call
 *   (event waits from eval()/item() on an in-flight array, the
 *   waitUntilCompleted of every sync, CPU-stream fence spins)
 *
 * Each counter is an independent relaxed atomic with no happens-before
 * relation to the GPU or to other encoding threads; a snapshot is four
 * independent loads and a reset four independent stores. Both are only
 * meaningful while no eval is in flight on any thread. Without the Metal
 * backend every counter reads 0. Any out-parameter may be NULL.
 *
 * Not part of mlx-c: this shim lives in mlx-swift's Cmlx target only
 * (numen-tech/gemma4-qat#181).
 */
void mlx_counters_snapshot(
    uint64_t* dispatches,
    uint64_t* commits,
    uint64_t* syncs,
    uint64_t* waits);

/** Zero all four counters (only while no eval is in flight). */
void mlx_counters_reset(void);

#ifdef __cplusplus
}
#endif

#endif // MLX_COUNTERS_H
