// numen-tech fork addition (numen-tech/gemma4-qat#181).

#if MLX_GPU_COUNTERS

    import Cmlx

    /// Process-wide Metal work counters, summed over every stream.
    ///
    /// - `dispatches`: compute kernel dispatches (`dispatchThreadgroups` / `dispatchThreads`)
    /// - `commits`: command-buffer commits
    /// - `syncs`: blocking `synchronize()` calls (`waitUntilCompleted`), e.g.
    ///   `Stream.synchronize()`. Waiting for an array's result (`eval(_:)`,
    ///   `item()`) is *not* a sync.
    ///
    /// The counters are relaxed atomics in the numen-tech/mlx fork's Metal backend
    /// (`mlx::core::metal::counters()`), exposed through the Cmlx `mlx_counters_*` shim
    /// (mlx-c is not involved). They have no happens-before relation with the GPU or
    /// with other threads encoding work, so:
    ///
    /// - take a ``snapshot()`` only after the `eval(_:)` whose work you are
    ///   measuring has returned (never across an in-flight `asyncEval`), and
    /// - make sure no other thread is encoding MLX work while you measure.
    ///
    /// Measure a region as the difference of two snapshots, or ``reset()`` first.
    ///
    /// Only available in the SwiftPM build of this fork (`MLX_GPU_COUNTERS`); the CMake
    /// and Xcode-project builds do not compile the Cmlx shim.
    public enum GPUCounters {

        /// The current counter values.
        public static func snapshot() -> (dispatches: UInt64, commits: UInt64, syncs: UInt64) {
            var dispatches: UInt64 = 0
            var commits: UInt64 = 0
            var syncs: UInt64 = 0
            mlx_counters_snapshot(&dispatches, &commits, &syncs)
            return (dispatches, commits, syncs)
        }

        /// Zero all three counters.
        public static func reset() {
            mlx_counters_reset()
        }
    }

#endif
