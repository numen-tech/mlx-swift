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
 *   waitUntilCompleted of every sync, CPU-stream fence spins, and the
 *   scheduler's throttle waits: eval() blocking until an in-flight task
 *   completes because too many tasks are active or active memory is over the
 *   limit, counted only when a GPU task completed during the wait)
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

/**
 * The version of the Apple6 (A13) MMA-free route the vendored numen-tech/mlx
 * core carries (`mlx::core::metal::kApple6RouteVersion`, gpu_family.h), 0 if
 * it carries none. Constant per build; a package that opens an A13 allowlist
 * must require >= 1 (numen-tech/gemma4-qat#703). Not part of mlx-c.
 */
int mlx_apple6_route_version(void);

/**
 * The version of the NAX kill switch the vendored numen-tech/mlx core carries
 * (`mlx::core::metal::kNaxDisableEnvVersion`, device.h), 0 if it carries
 * none. 1: `MLX_DISABLE_NAX=1`, set before the first GPU op, turns every NAX
 * (gen >= 18) route off (numen-tech/gemma4-qat#1085). 2: also `MLX_METAL_NAX=1`
 * (NAX from gen 17) and `=0` (off), and the kernel trace below
 * (numen-tech/gemma4-qat#312). Constant per build. Not part of mlx-c.
 */
int mlx_nax_disable_env_version(void);

/**
 * `mlx::core::metal::is_nax_available()`: whether this process takes the NAX
 * kernels. The first call fixes the answer for the process (it reads
 * `MLX_DISABLE_NAX` and `MLX_METAL_NAX` then), so a caller that sets either
 * must do so before this or any GPU op. Not part of mlx-c.
 */
int mlx_nax_available(void);

/**
 * `mlx::core::metal::kernel_trace_start()`: from now until the next
 * `mlx_kernel_trace_stop()`, record the name of every kernel looked up, cached
 * or not (numen-tech/gemma4-qat#312). Process-wide; one trace at a time. Not
 * part of mlx-c.
 */
void mlx_kernel_trace_start(void);

/**
 * `mlx::core::metal::kernel_trace_stop()`: ends the trace and returns its names,
 * in lookup order, joined by newlines in a malloc'ed NUL-terminated string the
 * caller frees. Without the Metal backend, an empty string. Not part of mlx-c.
 */
char* mlx_kernel_trace_stop(void);

#ifdef __cplusplus
}
#endif

#endif // MLX_COUNTERS_H
