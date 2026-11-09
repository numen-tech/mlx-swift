// numen-tech fork addition (numen-tech/gemma4-qat#703).

#if MLX_GPU_COUNTERS

    import Cmlx
    import Foundation

    /// The Apple6 (A13 / G12) route of the vendored numen-tech/mlx core.
    ///
    /// A13 GPUs cannot compile `simdgroup_multiply_accumulate`; the fork routes quantized
    /// matmuls, small dense matmuls and attention to MMA-free kernels on those GPUs and throws a
    /// typed error for anything else. A package that admits models on an A13 must require
    /// ``apple6MMAFreeVersion`` `>= 1` first, so a dependency graph whose mlx core lacks the route
    /// keeps refusing instead of reaching the GPU compiler.
    ///
    /// - 1: quantized matmul, dense matmul and full attention of at most 8 rows
    ///   (numen-tech/gemma4-qat#703).
    /// - 2: affine 2/4/8-bit quantized matmul of any row count through the SIMT GEMM
    ///   (`affine_qgemm_simt`), full attention of any query length through `sdpa_vector`, dense
    ///   matmul of up to 64 rows (numen-tech/gemma4-qat#991). A caller feeds prefill chunks wider
    ///   than 8 rows only when the version is `>= 2`.
    ///
    /// Constant for a build: the value of `mlx::core::metal::kApple6RouteVersion` the core was
    /// compiled with, read through the Cmlx `mlx_apple6_route_version()` shim (mlx-c is not
    /// involved). Only available in the SwiftPM build (`MLX_GPU_COUNTERS`); other builds read 0.
    public enum GPUFamilyRoute {
        /// The route version the core carries; 0 when it carries none.
        public static var apple6MMAFreeVersion: Int { Int(mlx_apple6_route_version()) }

        /// The version of the NAX switches the core carries
        /// (`mlx::core::metal::kNaxDisableEnvVersion`); 0 when it carries none.
        ///
        /// - 1: ``naxDisableEnvironmentKey`` set to `1` before the first GPU op turns every NAX
        ///   (GPU generation 18 and later) route off, so matmul, quantized matmul, attention and
        ///   gated delta take the kernels earlier generations take (numen-tech/gemma4-qat#1085).
        /// - 2: also ``naxForceEnvironmentKey``, `1` to take NAX from GPU generation 17 and `0` to
        ///   turn it off, and ``traceKernels(_:)`` (numen-tech/gemma4-qat#312).
        ///
        /// Constant for a build.
        public static var naxDisableEnvVersion: Int { Int(mlx_nax_disable_env_version()) }

        /// The version of the split-K quantized matmul's NAX route the core carries
        /// (`mlx::core::metal::kQmmSplitkNaxVersion`); 0 when it carries none.
        ///
        /// - 1: on a GPU with NAX (``isNAXAvailable``), an affine 2/4/8-bit split-K quantized
        ///   matmul of group size 32/64/128 (the verify widths from the qmv batch limit up, where
        ///   few threadgroups cover the output) takes the tensor-unit body for fp16 input, and fp32
        ///   input under `MLX_ENABLE_TF32`. The kernel is the `_nax_true` one either way, but at fp16
        ///   the body runs only on 32-row tiles holding at most 16 rows (M <= 16, or the last tile);
        ///   fuller fp16 tiles keep the SIMD body inside that kernel. ``qmmSplitKNAXDisableEnvironmentKey`` set to `1` turns it
        ///   off (read on every call). A GPU without NAX never compiles the body
        ///   (numen-tech/gemma4-qat#315).
        ///
        /// Constant for a build.
        public static var qmmSplitKNAXVersion: Int { Int(mlx_qmm_splitk_nax_version()) }

        /// The version of the non-resident no-copy array the core carries
        /// (`mlx::core::metal::kNonresidentArrayVersion`); 0 when it carries none.
        ///
        /// - 1: ``MLXArray/init(nonResidentRawPointer:_:dtype:finalizer:)`` wraps caller memory
        ///   without a copy, like ``MLXArray/init(rawPointer:_:dtype:finalizer:)``, but its Metal
        ///   buffer never joins the residency sets, so a raised wired limit
        ///   (``WiredMemoryManager``) never wires it (numen-tech/gemma4-qat#1228).
        ///
        /// Constant for a build.
        public static var nonResidentNoCopyVersion: Int { Int(mlx_nonresident_array_version()) }

        /// The bytes the Metal residency sets wire right now: every tracked allocation that fits
        /// the wired limit (0 where residency sets are unavailable). For a test that has to prove
        /// what a raised wired limit covers (numen-tech/gemma4-qat#1228).
        public static var residencyWiredBytes: Int { Int(mlx_residency_wired_bytes()) }

        /// Whether the GPU's Metal residency sets are on (a Metal 3 GPU on macOS 15 / iOS 18 or
        /// later). Where they are not, nothing is tracked and ``residencyWiredBytes`` is always 0,
        /// so a test that judges wiring by it checks this first (numen-tech/gemma4-qat#1228).
        public static var residencySetsAvailable: Bool { mlx_residency_sets_enabled() != 0 }

        /// Whether this process takes the NAX kernels (`mlx::core::metal::is_nax_available()`).
        /// The first read (or the first GPU op that asks) fixes the answer for the process, since
        /// the core reads ``naxDisableEnvironmentKey`` and ``naxForceEnvironmentKey`` then: set
        /// them before it.
        public static var isNAXAvailable: Bool { mlx_nax_available() != 0 }

        /// Runs `body` and returns the name of every Metal kernel looked up meanwhile, cached or
        /// not, in lookup order with repeats (needs ``naxDisableEnvVersion`` `>= 2`). `body` must
        /// `eval` the op it wants traced. Process-wide and not reentrant: nothing else may run MLX
        /// work during the call. For a harness that has to prove which kernel served an op, since
        /// a fallback kernel also gives a plausible answer (numen-tech/gemma4-qat#312).
        public static func traceKernels<R>(_ body: () throws -> R) rethrows -> (
            result: R, kernels: [String]
        ) {
            mlx_kernel_trace_start()
            func stop() -> [String] {
                guard let joined = mlx_kernel_trace_stop() else { return [] }
                defer { free(joined) }
                return String(cString: joined).split(separator: "\n").map(String.init)
            }
            do {
                let result = try body()
                return (result, stop())
            } catch {
                _ = stop()
                throw error
            }
        }
    }

#else

    public enum GPUFamilyRoute {
        public static var apple6MMAFreeVersion: Int { 0 }
        public static var naxDisableEnvVersion: Int { 0 }
        public static var qmmSplitKNAXVersion: Int { 0 }
        public static var nonResidentNoCopyVersion: Int { 0 }
        public static var residencyWiredBytes: Int { 0 }
        public static var residencySetsAvailable: Bool { false }
        public static var isNAXAvailable: Bool { false }
        public static func traceKernels<R>(_ body: () throws -> R) rethrows -> (
            result: R, kernels: [String]
        ) {
            (try body(), [])
        }
    }

#endif

extension GPUFamilyRoute {
    /// The environment variable the core's NAX kill switch reads (``naxDisableEnvVersion`` >= 1).
    public static let naxDisableEnvironmentKey = "MLX_DISABLE_NAX"
    /// The environment variable that overrides the core's NAX rule (``naxDisableEnvVersion`` >= 2).
    public static let naxForceEnvironmentKey = "MLX_METAL_NAX"
    /// The environment variable that turns the split-K NAX body off (``qmmSplitKNAXVersion`` >= 1).
    public static let qmmSplitKNAXDisableEnvironmentKey = "MLX_DISABLE_QMM_SPLITK_NAX"
}
