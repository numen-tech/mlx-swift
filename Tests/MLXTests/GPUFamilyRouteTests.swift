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
    }

#endif
