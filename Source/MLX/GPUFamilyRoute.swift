// numen-tech fork addition (numen-tech/gemma4-qat#703).

#if MLX_GPU_COUNTERS

    import Cmlx

    /// The Apple6 (A13 / G12) route of the vendored numen-tech/mlx core.
    ///
    /// A13 GPUs cannot compile `simdgroup_multiply_accumulate`; the fork routes small quantized
    /// matmuls and attention to MMA-free kernels on those GPUs and throws a typed error for
    /// anything else. A package that admits models on an A13 must require
    /// ``apple6MMAFreeVersion`` `>= 1` first, so a dependency graph whose mlx core lacks the route
    /// keeps refusing instead of reaching the GPU compiler.
    ///
    /// Constant for a build: the value of `mlx::core::metal::kApple6RouteVersion` the core was
    /// compiled with, read through the Cmlx `mlx_apple6_route_version()` shim (mlx-c is not
    /// involved). Only available in the SwiftPM build (`MLX_GPU_COUNTERS`); other builds read 0.
    public enum GPUFamilyRoute {
        /// The route version the core carries; 0 when it carries none.
        public static var apple6MMAFreeVersion: Int { Int(mlx_apple6_route_version()) }
    }

#else

    public enum GPUFamilyRoute {
        public static var apple6MMAFreeVersion: Int { 0 }
    }

#endif
