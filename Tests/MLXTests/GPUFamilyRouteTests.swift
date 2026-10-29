// numen-tech fork addition (numen-tech/gemma4-qat#703).

#if MLX_GPU_COUNTERS

    import MLX
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
    }

#endif
