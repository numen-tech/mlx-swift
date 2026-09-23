#ifndef MLX_COUNTERS_H
#define MLX_COUNTERS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Process-wide Metal work counters from the numen-tech/mlx fork
 * (`mlx::core::metal::counters()`): kernel dispatches, command-buffer commits
 * and blocking `synchronize()` calls, summed over every stream since process
 * start or the last `mlx_counters_reset()`.
 *
 * The counters are relaxed atomics with no happens-before relation to the GPU
 * or to other encoding threads. Read them only after the eval you are
 * measuring has completed and while no other thread is encoding work.
 *
 * Not part of mlx-c: this shim lives in mlx-swift's Cmlx target only
 * (numen-tech/gemma4-qat#181).
 */
void mlx_counters_snapshot(
    uint64_t* dispatches,
    uint64_t* commits,
    uint64_t* syncs);

/** Zero all three counters. */
void mlx_counters_reset(void);

#ifdef __cplusplus
}
#endif

#endif // MLX_COUNTERS_H
