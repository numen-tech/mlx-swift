// numen-tech fork addition (numen-tech/gemma4-qat#181).

#if MLX_GPU_COUNTERS

    import Cmlx

    /// Process-wide Metal work counters, summed over every stream.
    ///
    /// - `dispatches`: compute kernel dispatches (`dispatchThreadgroups` / `dispatchThreads`)
    /// - `commits`: command-buffer commits, including the automatic splits MLX makes when a
    ///   buffer exceeds `MLX_MAX_OPS_PER_BUFFER` / `MLX_MAX_MB_PER_BUFFER`. A commit is not a
    ///   host wait.
    /// - `syncs`: explicit stream synchronizations (`synchronize()`, e.g.
    ///   `Stream.synchronize()`).
    /// - `waits`: host blocking waits on GPU completion, counted once per wait *call* (not
    ///   blocking time): `eval(_:)` / `item()` on an array still in flight, a CPU stream
    ///   waiting on a GPU event, and the `waitUntilCompleted` of every sync. MLX skips the
    ///   wait when the array's event has already signaled, so an `eval(_:)` whose GPU work
    ///   finished before the host checked records none.
    ///
    /// The counters are independent relaxed atomics in the numen-tech/mlx fork's Metal
    /// backend (`mlx::core::metal::counters()`), exposed through the Cmlx `mlx_counters_*`
    /// shim (mlx-c is not involved). ``snapshot()`` is four independent loads and
    /// ``reset()`` four independent stores, with no happens-before relation to the GPU or
    /// to other threads encoding work, so both are only meaningful while no eval is in
    /// flight on any thread:
    ///
    /// - call them only after the `eval(_:)` whose work you are measuring has returned
    ///   (never across an in-flight `asyncEval`), and
    /// - make sure no other thread is encoding MLX work while you measure.
    ///
    /// Measure a region as the difference of two snapshots, or ``reset()`` first. Without
    /// the Metal backend (e.g. Linux) every counter reads 0.
    ///
    /// Only available in the SwiftPM build of this fork (`MLX_GPU_COUNTERS`); the CMake
    /// and Xcode-project builds do not compile the Cmlx shim.
    public enum GPUCounters {

        /// One reading of the four counters.
        public struct Snapshot: Sendable, Equatable {
            public var dispatches: UInt64
            public var commits: UInt64
            public var syncs: UInt64
            public var waits: UInt64

            public init(dispatches: UInt64, commits: UInt64, syncs: UInt64, waits: UInt64) {
                self.dispatches = dispatches
                self.commits = commits
                self.syncs = syncs
                self.waits = waits
            }

            /// The work done between `earlier` and `self` (counters are monotonic
            /// between resets; call this only on snapshots with no reset in between).
            public static func - (later: Snapshot, earlier: Snapshot) -> Snapshot {
                Snapshot(
                    dispatches: later.dispatches - earlier.dispatches,
                    commits: later.commits - earlier.commits,
                    syncs: later.syncs - earlier.syncs,
                    waits: later.waits - earlier.waits)
            }
        }

        /// The current counter values.
        public static func snapshot() -> Snapshot {
            var dispatches: UInt64 = 0
            var commits: UInt64 = 0
            var syncs: UInt64 = 0
            var waits: UInt64 = 0
            mlx_counters_snapshot(&dispatches, &commits, &syncs, &waits)
            return Snapshot(dispatches: dispatches, commits: commits, syncs: syncs, waits: waits)
        }

        /// Zero all four counters (only while no eval is in flight).
        public static func reset() {
            mlx_counters_reset()
        }
    }

#endif
