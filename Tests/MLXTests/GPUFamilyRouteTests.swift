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

        /// A 4-row affine 4-bit product names the qmv_mma kernel only under its switch, and only on a
        /// GPU without NAX (not on Apple6, which has no simdgroup matrices, nor an Ultra); 2 rows never
        /// take it (numen-tech/gemma4-qat#1701).
        func testQmvMMAKernelNamesItsRoute() throws {
            XCTAssertGreaterThanOrEqual(GPUFamilyRoute.qmvMMAVersion, 2)
            #if os(macOS)
                let device = MTLCopyAllDevices().first ?? MTLCreateSystemDefaultDevice()
            #else
                let device = MTLCreateSystemDefaultDevice()
            #endif
            let apple6 =
                ProcessInfo.processInfo.environment["MLX_FORCE_GPU_FAMILY"] == "apple6"
                || (device?.supportsFamily(.apple6) == true
                    && device?.supportsFamily(.apple7) == false)
            let w = MLXRandom.normal([2048, 1024], key: MLXRandom.key(3)).asType(.float16)
            let (wq, scales, biases) = quantized(w, groupSize: 64, bits: 4, stream: .gpu)
            let key = GPUFamilyRoute.qmvMMAEnvironmentKey
            let saved = ProcessInfo.processInfo.environment[key]
            defer {
                if let saved { setenv(key, saved, 1) } else { unsetenv(key) }
            }
            // The core keeps an Ultra part (architecture ending `d`) on the vector kernels.
            let ultra = GPU.deviceInfo().architecture.hasSuffix("d")
            let eligible = !GPUFamilyRoute.isNAXAvailable && !apple6 && !ultra
            XCTAssertEqual(GPUFamilyRoute.qmvMMADevice, eligible)
            for rows in [2, 4] {
                let x = MLXRandom.normal([rows, 1024], key: MLXRandom.key(4)).asType(.float16)
                let reference = matmul(
                    x.asType(.float32),
                    dequantized(wq, scales: scales, biases: biases, groupSize: 64, bits: 4)
                        .asType(.float32).transposed())
                for on in [false, true] {
                    if on { setenv(key, "1", 1) } else { unsetenv(key) }
                    let (y, kernels) = GPUFamilyRoute.traceKernels { () -> MLXArray in
                        let y = quantizedMM(
                            x, wq, scales: scales, biases: biases, groupSize: 64, bits: 4,
                            stream: .gpu)
                        eval(y)
                        return y
                    }
                    let taken = kernels.contains { $0.hasPrefix("affine_qmv_mma_") }
                    XCTAssertEqual(taken, on && eligible && rows == 4, "rows \(rows), on \(on): \(kernels)")
                    let error = abs(y.asType(.float32) - reference).max().item(Float.self)
                    let scale = abs(reference).max().item(Float.self)
                    XCTAssertLessThan(error / scale, 2e-3, "rows \(rows), on \(on)")
                }
            }
            XCTAssertEqual(key, "MLX_QMV_MMA")
        }

        /// ``GPUFamilyRoute/setSDPAD512Prefill(_:)`` lets a causal head-dim-512 call of 9 to
        /// 1,023 rows take a fused d512 path: the blocked path (`sdpa_blocked_*`) without NAX, the
        /// NAX d512 kernel (`_bd512`) with it. Off, the call stays unfused; 8 rows never route
        /// (numen-tech/gemma4-qat#1972).
        func testSDPAD512PrefillTakesTheFusedPath() throws {
            XCTAssertGreaterThanOrEqual(GPUFamilyRoute.sdpaD512PrefillVersion, 1)
            XCTAssertFalse(GPUFamilyRoute.sdpaD512Prefill)
            defer { GPUFamilyRoute.setSDPAD512Prefill(false) }
            func fused(_ kernels: [String]) -> (blocked: Bool, nax: Bool) {
                (
                    kernels.contains { $0.hasPrefix("sdpa_blocked_causal_mask_") },
                    kernels.contains { $0.hasPrefix("steel_attention") && $0.contains("_bd512") }
                )
            }
            for (rows, keys) in [(512, 1536), (280, 811), (9, 300), (8, 600)] {
                let q = (MLXRandom.normal([1, 16, rows, 512], key: MLXRandom.key(1)) * 0.5)
                    .asType(.float16)
                let k = (MLXRandom.normal([1, 2, keys, 512], key: MLXRandom.key(2)) * 0.5)
                    .asType(.float16)
                let v = (MLXRandom.normal([1, 2, keys, 512], key: MLXRandom.key(3)) * 0.5)
                    .asType(.float16)
                var outputs = [Bool: MLXArray]()
                for on in [false, true] {
                    GPUFamilyRoute.setSDPAD512Prefill(on)
                    XCTAssertEqual(GPUFamilyRoute.sdpaD512Prefill, on)
                    let (y, kernels) = GPUFamilyRoute.traceKernels { () -> MLXArray in
                        let y = MLXFast.scaledDotProductAttention(
                            queries: q, keys: k, values: v, scale: Float(1 / 512.0.squareRoot()), mask: .causal)
                        eval(y)
                        return y
                    }
                    outputs[on] = y
                    let route = fused(kernels)
                    if on && rows > 8 {
                        XCTAssertEqual(route.blocked, !GPUFamilyRoute.isNAXAvailable, "\(rows): \(kernels)")
                        XCTAssertEqual(route.nax, GPUFamilyRoute.isNAXAvailable, "\(rows): \(kernels)")
                    } else {
                        XCTAssertFalse(route.blocked || route.nax, "on \(on), \(rows): \(kernels)")
                    }
                }
                let (off, on) = (outputs[false]!, outputs[true]!)
                XCTAssertTrue(
                    off.asType(.float32).allClose(on.asType(.float32), rtol: 1e-3, atol: 1e-3)
                        .item(Bool.self),
                    "\(rows) rows")
            }
        }

        /// ``GPUFamilyRoute/setQmvMMA(_:)`` takes the route as the variable does, with the variable
        /// unset, and off again leaves it to the variable (numen-tech/gemma4-qat#1701).
        func testSetQmvMMATakesTheRouteWithoutTheEnvironment() throws {
            let key = GPUFamilyRoute.qmvMMAEnvironmentKey
            let saved = ProcessInfo.processInfo.environment[key]
            unsetenv(key)
            defer {
                GPUFamilyRoute.setQmvMMA(false)
                if let saved { setenv(key, saved, 1) }
            }
            let w = MLXRandom.normal([2048, 1024], key: MLXRandom.key(3)).asType(.float16)
            let (wq, scales, biases) = quantized(w, groupSize: 64, bits: 4, stream: .gpu)
            let x = MLXRandom.normal([4, 1024], key: MLXRandom.key(4)).asType(.float16)
            for on in [true, false] {
                GPUFamilyRoute.setQmvMMA(on)
                let (_, kernels) = GPUFamilyRoute.traceKernels { () -> MLXArray in
                    let y = quantizedMM(
                        x, wq, scales: scales, biases: biases, groupSize: 64, bits: 4, stream: .gpu)
                    eval(y)
                    return y
                }
                let taken = kernels.contains { $0.hasPrefix("affine_qmv_mma_") }
                XCTAssertEqual(taken, on && GPUFamilyRoute.qmvMMADevice, "on \(on): \(kernels)")
            }
        }

        func testVendoredCoreCarriesTheQuantizedRowStrideRoute() {
            XCTAssertGreaterThanOrEqual(GPUFamilyRoute.quantizedRowStrideVersion, 1)
            XCTAssertEqual(GPUFamilyRoute.quantizedRowStrideEnvironmentKey, "MLX_QMM_ROW_STRIDE")
        }

        /// Under ``GPUFamilyRoute/setQuantizedRowStride(_:)`` the two 512-value views of a packed
        /// 640-value row (K = values 0..<512, V = values 128..<640: a compact K=V KV cache), batched
        /// as quantized attention passes them (queries `[1, kvHeads, 8, M, 512]`), go through one
        /// `_rs` kernel each with no copy and give the bit patterns of their packed copies, at
        /// decode, verify and prefill widths; off, the core copies the three arrays of each view
        /// and takes the stock kernel (numen-tech/gemma4-qat#1973).
        func testQuantizedRowStrideReadsViewsInPlace() throws {
            #if os(macOS)
                let device = MTLCopyAllDevices().first ?? MTLCreateSystemDefaultDevice()
            #else
                let device = MTLCreateSystemDefaultDevice()
            #endif
            let apple6 =
                ProcessInfo.processInfo.environment["MLX_FORCE_GPU_FAMILY"] == "apple6"
                || (device?.supportsFamily(.apple6) == true
                    && device?.supportsFamily(.apple7) == false)
            try XCTSkipIf(apple6, "Apple6 has no row-stride kernels: it copies")
            let key = GPUFamilyRoute.quantizedRowStrideEnvironmentKey
            let saved = ProcessInfo.processInfo.environment[key]
            unsetenv(key)
            defer {
                GPUFamilyRoute.setQuantizedRowStride(false)
                if let saved { setenv(key, saved, 1) }
            }
            typealias Triple = (MLXArray, MLXArray, MLXArray)
            func packed(_ t: Triple) -> Triple {
                let out = (contiguous(t.0), contiguous(t.1), contiguous(t.2))
                eval(out.0, out.1, out.2)
                return out
            }
            func isCopy(_ name: String) -> Bool { name.contains("_copy") }
            var taken = Set<String>()
            // The shipped format (8-bit, group size 64) at the 26B's and the 31B's KV head counts,
            // and the other formats the route takes.
            for (kvHeads, keys, bits, groupSize) in [
                (2, 640, 8, 64), (2, 1536, 8, 64), (4, 2304, 8, 64), (2, 640, 4, 32),
                (2, 1024, 8, 128),
            ] {
                let rows = MLXRandom.normal(
                    [1, kvHeads, 1, keys + 96, 640], key: MLXRandom.key(1973)
                ).asType(.float16)
                let (wq, scales, packedBiases) = quantized(
                    rows, groupSize: groupSize, bits: bits, stream: .gpu)
                let biases = try XCTUnwrap(packedBiases)
                eval(wq, scales, biases)
                func view(values: Range<Int>) -> Triple {
                    let words = values.lowerBound * bits / 32 ..< values.upperBound * bits / 32
                    let groups = values.lowerBound / groupSize ..< values.upperBound / groupSize
                    return (
                        wq[.ellipsis, ..<keys, words], scales[.ellipsis, ..<keys, groups],
                        biases[.ellipsis, ..<keys, groups]
                    )
                }
                func run(_ x: MLXArray, _ t: Triple, transpose: Bool, on: Bool) -> (
                    [UInt16], [String]
                ) {
                    GPUFamilyRoute.setQuantizedRowStride(on)
                    let (y, kernels) = GPUFamilyRoute.traceKernels { () -> MLXArray in
                        let y = quantizedMM(
                            x, t.0, scales: t.1, biases: t.2, transpose: transpose,
                            groupSize: groupSize, bits: bits, stream: .gpu)
                        eval(y)
                        return y
                    }
                    return (y.asArray(Float16.self).map(\.bitPattern), kernels)
                }
                let keyView = view(values: 0 ..< 512)
                let valueView = view(values: 128 ..< 640)
                for m in [1, 4, 8, 33, 512] {
                    let q = MLXRandom.normal(
                        [1, kvHeads, 8, m, 512], key: MLXRandom.key(UInt64(m))
                    ).asType(.float16)
                    let p = MLXRandom.normal(
                        [1, kvHeads, 8, m, keys], key: MLXRandom.key(UInt64(m + 1))
                    ).asType(.float16)
                    eval(q, p)
                    for (x, t, transpose) in [(q, keyView, true), (p, valueView, false)] {
                        let label =
                            "heads \(kvHeads) keys \(keys) \(bits)-bit g\(groupSize) M \(m) transpose \(transpose)"
                        let stock = run(x, packed(t), transpose: transpose, on: true)
                        let inPlace = run(x, t, transpose: transpose, on: true)
                        let copied = run(x, t, transpose: transpose, on: false)
                        XCTAssertEqual(inPlace.0, stock.0, label)
                        XCTAssertEqual(copied.0, stock.0, label)
                        XCTAssertFalse(
                            stock.1.contains { $0.hasSuffix("_rs") }, "\(label): \(stock.1)")
                        XCTAssertEqual(
                            inPlace.1.map { $0.hasSuffix("_rs") ? String($0.dropLast(3)) : $0 },
                            stock.1, label)
                        let twins = inPlace.1.filter { $0.hasSuffix("_rs") }
                        XCTAssertEqual(twins.count, 1, "\(label): \(inPlace.1)")
                        taken.formUnion(twins)
                        XCTAssertEqual(copied.1.filter { !isCopy($0) }, stock.1, label)
                        XCTAssertEqual(copied.1.count(where: isCopy), 3, "\(label): \(copied.1)")
                    }
                }
            }
            print("[QuantizedRowStride] kernels: \(taken.sorted())")
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

        /// A 0-d `biases` factor through `gatherQuantizedMM` with one row per matrix (what a MoE
        /// layer presents) takes an `_ib` gather kernel, `gather_qmv_fast` / `gather_qmv` for
        /// few rows and the sorted `gather_qmm_rhs` for many, and gives the bit patterns of the
        /// explicit bias `scales * T(f)`, for the shipped factor and for one whose product rounds
        /// (a kernel that derived the factor from the bit width would miss it). More rows per
        /// matrix still get a materialized bias and no `_ib` kernel (numen-tech/gemma4-qat#1833).
        func testGatherImpliedBiasKernelNamesItsRoute() {
            XCTAssertGreaterThanOrEqual(GPUFamilyRoute.gatherImpliedBiasVersion, 1)
            // The core's Apple6 test (`Device::is_apple6()`): no simdgroup matrices there, so the
            // sorted kernel throws and only the vector kernels run.
            #if os(macOS)
                let device = MTLCopyAllDevices().first ?? MTLCreateSystemDefaultDevice()
            #else
                let device = MTLCreateSystemDefaultDevice()
            #endif
            let apple6 =
                ProcessInfo.processInfo.environment["MLX_FORCE_GPU_FAMILY"] == "apple6"
                || (device?.supportsFamily(.apple6) == true
                    && device?.supportsFamily(.apple7) == false)
            let experts = 8
            struct Route {
                let rows: [UInt32]
                let perMatrix: Int
                let sorted: Bool
                let family: String
                let implied: Bool
            }
            // K % 256 == 0 with N % 8 == 0 is gather_qmv_fast, anything else gather_qmv.
            for (k, n, fast) in [(512, 64, true), (704, 512, false)] {
                let w = MLXRandom.normal([experts, n, k], key: MLXRandom.key(5)).asType(.float16)
                let (wq, scales, _) = quantized(w, groupSize: 32, bits: 4, stream: .gpu)
                func run(_ x: MLXArray, _ biases: MLXArray, _ rhs: MLXArray, sorted: Bool) -> (
                    [UInt16], [String]
                ) {
                    let (y, kernels) = GPUFamilyRoute.traceKernels { () -> MLXArray in
                        let y = gatherQuantizedMM(
                            x, wq, scales: scales, biases: biases, rhsIndices: rhs,
                            transpose: true, groupSize: 32, bits: 4, sortedIndices: sorted,
                            stream: .gpu)
                        eval(y)
                        return y
                    }
                    XCTAssertEqual(y.dtype, .float16)
                    return (
                        y.asArray(Float16.self).map(\.bitPattern),
                        kernels.filter { $0.contains("_gather_q") }
                    )
                }
                let few = fast ? "_gather_qmv_fast_" : "_gather_qmv_"
                var routes = [
                    Route(
                        rows: [5, 0, 2], perMatrix: 1, sorted: false, family: few, implied: true),
                    Route(
                        rows: [5, 0, 2], perMatrix: 3, sorted: false, family: few, implied: false),
                ]
                if !apple6 {
                    routes.append(
                        Route(
                            rows: (0 ..< 64).map { UInt32($0 * experts / 64) }, perMatrix: 1,
                            sorted: true, family: "_gather_qmm_rhs_", implied: true))
                }
                for value in [Float(-8), -5.3] {
                    let factor = MLXArray(value).asType(.float16)
                    let explicit = scales * factor
                    eval(wq, scales, explicit)
                    for route in routes {
                        let x = MLXRandom.normal(
                            [route.rows.count, route.perMatrix, k], key: MLXRandom.key(6)
                        ).asType(.float16)
                        let rhs = MLXArray(route.rows)
                        let (yImplied, kImplied) = run(x, factor, rhs, sorted: route.sorted)
                        let (yExplicit, kExplicit) = run(x, explicit, rhs, sorted: route.sorted)
                        let label =
                            "k \(k), f \(value), rows \(route.rows.count) x \(route.perMatrix), "
                            + "sorted \(route.sorted)"
                        XCTAssertEqual(yImplied, yExplicit, label)
                        XCTAssertEqual(kImplied.count, 1, "\(label): \(kImplied)")
                        XCTAssertEqual(kExplicit.count, 1, "\(label): \(kExplicit)")
                        XCTAssertTrue(
                            kImplied.allSatisfy { $0.contains(route.family) },
                            "\(label): \(kImplied)")
                        XCTAssertEqual(
                            kImplied.contains { $0.contains("_gather_qmv_fast_") },
                            fast && !route.sorted, "\(label): \(kImplied)")
                        // The sorted kernels' names end with their alignment constants.
                        XCTAssertEqual(
                            kImplied.allSatisfy {
                                $0.hasSuffix("_ib") || $0.contains("_ib_align_")
                            },
                            route.implied, "\(label): \(kImplied)")
                        XCTAssertFalse(
                            kExplicit.contains { $0.contains("_ib") }, "\(label): \(kExplicit)")
                    }
                }
            }
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
