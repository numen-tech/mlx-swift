// numen-tech fork addition (numen-tech/gemma4-qat#703).

#if MLX_GPU_COUNTERS

    import MLX
    import Metal
    import XCTest

    class GPUFamilyRouteTests: XCTestCase {

        /// The vendored core carries the Apple6 route, so a package that gates an A13 allowlist on
        /// it opens it.
        func testVendoredCoreCarriesTheApple6Route() {
            XCTAssertGreaterThanOrEqual(GPUFamilyRoute.apple6MMAFreeVersion, 1)
        }

        /// The vendored core carries route version 2 (the SIMT quantized GEMM, any-length
        /// `sdpa_vector`, `gemv_wide` to 64 rows; numen-tech/gemma4-qat#991), so a package that
        /// gates its wider A13 prefill chunk on it opens it.
        func testVendoredCoreCarriesTheApple6RouteVersion2() {
            XCTAssertGreaterThanOrEqual(GPUFamilyRoute.apple6MMAFreeVersion, 2)
        }

        /// The vendored core carries the NAX kill switch (numen-tech/gemma4-qat#1085), so a
        /// package that sets `MLX_DISABLE_NAX` for an unverified GPU generation gets the effect.
        func testVendoredCoreCarriesTheNAXKillSwitch() {
            XCTAssertGreaterThanOrEqual(GPUFamilyRoute.naxDisableEnvVersion, 2)
            XCTAssertEqual(GPUFamilyRoute.naxDisableEnvironmentKey, "MLX_DISABLE_NAX")
            XCTAssertEqual(GPUFamilyRoute.naxForceEnvironmentKey, "MLX_METAL_NAX")
        }

        /// The vendored core carries the 2/4/8-bit split-K NAX route (numen-tech/gemma4-qat#315), so
        /// a package that asserts it, or turns it off by its key, gets the route it names.
        func testVendoredCoreCarriesTheSplitKNAXRoute() {
            XCTAssertGreaterThanOrEqual(GPUFamilyRoute.qmmSplitKNAXVersion, 2)
            XCTAssertEqual(
                GPUFamilyRoute.qmmSplitKNAXDisableEnvironmentKey, "MLX_DISABLE_QMM_SPLITK_NAX")
            XCTAssertEqual(
                GPUFamilyRoute.qmmSplitKFallthroughDisableEnvironmentKey,
                "MLX_DISABLE_QMM_SPLITK_FALLTHROUGH")
        }

        /// A split-K quantized matmul at a verify width names its NAX body flag: `_nax_true` where
        /// the core takes NAX, `_nax_false` elsewhere and under the off switch, which the core
        /// reads on every call. The JIT builds either kernel from its source here, so a GPU without
        /// NAX also proves its split-K kernels still compile (numen-tech/gemma4-qat#315).
        func testSplitKNAXKernelNamesItsRoute() {
            // K > 4096 keeps M = 16 at or above every GPU's qmv batch limit; N = 1536 leaves few
            // enough tiles for split-K.
            let (wq, scales, biases) = quantized(
                MLXRandom.normal([1536, 5120]).asType(.float16),
                groupSize: 64, bits: 4, stream: .gpu)
            let x = MLXRandom.normal([16, 5120]).asType(.float16)
            func splitKKernels() -> [String] {
                GPUFamilyRoute.traceKernels {
                    eval(
                        quantizedMM(
                            x, wq, scales: scales, biases: biases, groupSize: 64, bits: 4,
                            stream: .gpu))
                }.kernels.filter { $0.contains("_qmm_t_splitk_") }
            }
            let key = GPUFamilyRoute.qmmSplitKNAXDisableEnvironmentKey
            let saved = ProcessInfo.processInfo.environment[key]
            defer {
                if let saved { setenv(key, saved, 1) } else { unsetenv(key) }
            }
            unsetenv(key)
            let flag = GPUFamilyRoute.isNAXAvailable ? "_nax_true" : "_nax_false"
            let defaultRoute = splitKKernels()
            XCTAssertFalse(defaultRoute.isEmpty, "M = 16 at K 5120 is a split-K qmm")
            XCTAssertTrue(defaultRoute.allSatisfy { $0.hasSuffix(flag) }, "\(defaultRoute)")
            setenv(key, "1", 1)
            let off = splitKKernels()
            XCTAssertFalse(off.isEmpty)
            XCTAssertTrue(off.allSatisfy { $0.hasSuffix("_nax_false") }, "\(off)")
        }

        /// An fp16 split-K matmul of more than 16 rows takes the NAX quantized matmul on NAX; 16 rows,
        /// the off switch and a GPU without NAX keep split-K.
        func testSplitKFallthroughAbove16Rows() {
            // K > 4096 keeps every M here at or above every GPU's qmv batch limit.
            let (wq, scales, biases) = quantized(
                MLXRandom.normal([512, 5120]).asType(.float16),
                groupSize: 64, bits: 4, stream: .gpu)
            func kernels(rows: Int) -> [String] {
                let x = MLXRandom.normal([rows, 5120]).asType(.float16)
                return GPUFamilyRoute.traceKernels {
                    eval(
                        quantizedMM(
                            x, wq, scales: scales, biases: biases, groupSize: 64, bits: 4,
                            stream: .gpu))
                }.kernels.filter { $0.contains("_qmm_t_") }
            }
            let key = GPUFamilyRoute.qmmSplitKFallthroughDisableEnvironmentKey
            let saved = ProcessInfo.processInfo.environment[key]
            defer {
                if let saved { setenv(key, saved, 1) } else { unsetenv(key) }
            }
            unsetenv(key)
            let nax = GPUFamilyRoute.isNAXAvailable
            for rows in [17, 32, 64] {
                let taken = kernels(rows: rows)
                XCTAssertFalse(taken.isEmpty, "M = \(rows)")
                if nax {
                    XCTAssertTrue(taken.allSatisfy { $0.contains("_qmm_t_nax_") }, "\(taken)")
                } else {
                    XCTAssertTrue(taken.allSatisfy { $0.contains("_qmm_t_splitk_") }, "\(taken)")
                }
            }
            let sixteen = kernels(rows: 16)
            XCTAssertTrue(
                !sixteen.isEmpty && sixteen.allSatisfy { $0.contains("_qmm_t_splitk_") },
                "\(sixteen)")
            setenv(key, "1", 1)
            for rows in [16, 17, 32, 64] {
                let taken = kernels(rows: rows)
                XCTAssertTrue(
                    !taken.isEmpty && taken.allSatisfy { $0.contains("_qmm_t_splitk_") },
                    "M = \(rows): \(taken)")
            }
        }

        /// 2048 x 2048 is 1,024 tiles of 64 x 64, enough for an Ultra part too.
        func testQmmTile64KernelNamesItsRoute() throws {
            // The core's Apple6 test (`Device::is_apple6()`).
            #if os(macOS)
                let device = MTLCopyAllDevices().first ?? MTLCreateSystemDefaultDevice()
            #else
                let device = MTLCreateSystemDefaultDevice()
            #endif
            let apple6 =
                ProcessInfo.processInfo.environment["MLX_FORCE_GPU_FAMILY"] == "apple6"
                || (device?.supportsFamily(.apple6) == true
                    && device?.supportsFamily(.apple7) == false)
            try XCTSkipIf(apple6, "the Apple6 route never reaches qmm()")
            let w = MLXRandom.normal([2048, 256], key: MLXRandom.key(1)).asType(.float16)
            let (wq, scales, biases) = quantized(w, groupSize: 64, bits: 4, stream: .gpu)
            let x = MLXRandom.normal([2048, 256], key: MLXRandom.key(2)).asType(.float16)
            let reference = matmul(
                x.asType(.float32),
                dequantized(wq, scales: scales, biases: biases, groupSize: 64, bits: 4)
                    .asType(.float32).transposed())
            func run() -> (MLXArray, [String]) {
                let (y, kernels) = GPUFamilyRoute.traceKernels { () -> MLXArray in
                    let y = quantizedMM(
                        x, wq, scales: scales, biases: biases, groupSize: 64, bits: 4, stream: .gpu)
                    eval(y)
                    return y
                }
                return (y, kernels.filter { $0.contains("_qmm_t_") })
            }
            let key = GPUFamilyRoute.qmmTile64EnvironmentKey
            let saved = ProcessInfo.processInfo.environment[key]
            defer {
                if let saved { setenv(key, saved, 1) } else { unsetenv(key) }
            }
            let nax = GPUFamilyRoute.isNAXAvailable
            for on in [false, true] {
                if on { setenv(key, "1", 1) } else { unsetenv(key) }
                let (y, taken) = run()
                XCTAssertFalse(taken.isEmpty, "on = \(on)")
                let tile64 = taken.contains { $0.contains("_bm_64_bn_64") }
                XCTAssertEqual(tile64, on && !nax, "on = \(on): \(taken)")
                let error = abs(y.asType(.float32) - reference).max().item(Float.self)
                let scale = abs(reference).max().item(Float.self)
                XCTAssertLessThan(error / scale, 2e-3, "on = \(on)")
            }
            XCTAssertEqual(key, "MLX_QMM_TILE64")
        }

        /// The trace sees a kernel on every lookup, not only its first load, and nothing after it ends.
        func testKernelTraceSeesEveryLookup() {
            let a = MLXArray.ones([8, 8], dtype: .float32, stream: .gpu)
            let b = MLXArray.ones([8, 8], dtype: .float32, stream: .gpu)
            eval(a, b)
            for _ in 0 ..< 2 {
                let (_, kernels) = GPUFamilyRoute.traceKernels {
                    eval(MLX.add(a, b, stream: .gpu))
                }
                #if canImport(Metal)
                    XCTAssertFalse(kernels.isEmpty)
                #else
                    XCTAssertTrue(kernels.isEmpty, "no Metal backend: the trace stays empty")
                #endif
            }
            eval(MLX.subtract(a, b, stream: .gpu))
            XCTAssertTrue(GPUFamilyRoute.traceKernels {}.kernels.isEmpty)
        }

        /// In a process started with `MLX_DISABLE_NAX=1` the core takes no NAX kernel. Skips
        /// otherwise: the core reads the variable once, so a test cannot set it after another
        /// test has run a GPU op.
        func testNAXIsOffUnderTheKillSwitch() throws {
            guard ProcessInfo.processInfo.environment["MLX_DISABLE_NAX"] == "1" else {
                throw XCTSkip("run with MLX_DISABLE_NAX=1")
            }
            XCTAssertFalse(GPUFamilyRoute.isNAXAvailable)
        }
    }

#endif
