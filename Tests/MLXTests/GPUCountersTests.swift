// numen-tech fork addition (numen-tech/gemma4-qat#181).

#if MLX_GPU_COUNTERS

    import Foundation
    import MLX
    import XCTest

    class GPUCountersTests: XCTestCase {

        func testMatmulEvalIncrementsDispatchesAndCommits() {
            let a = MLXRandom.normal([64, 64], key: MLXRandom.key(0))
            let b = MLXRandom.normal([64, 64], key: MLXRandom.key(1))
            eval(a, b)

            GPUCounters.reset()
            let zero = GPUCounters.snapshot()
            XCTAssertEqual(zero.dispatches, 0)
            XCTAssertEqual(zero.commits, 0)
            XCTAssertEqual(zero.syncs, 0)

            let c = matmul(a, b)
            eval(c)  // snapshot only after the eval has completed
            let afterEval = GPUCounters.snapshot()
            XCTAssertGreaterThanOrEqual(afterEval.dispatches, 1)
            XCTAssertGreaterThanOrEqual(afterEval.commits, 1)
            XCTAssertEqual(afterEval.syncs, 0, "waiting on eval is not a sync")

            GPUCounters.reset()
            let reset = GPUCounters.snapshot()
            XCTAssertEqual(reset.dispatches, 0)
            XCTAssertEqual(reset.commits, 0)
            XCTAssertEqual(reset.syncs, 0)
        }

        func testSynchronizeCountsASync() {
            eval(MLXArray(1) + MLXArray(2))
            GPUCounters.reset()
            Stream.defaultStream(.gpu).synchronize()
            let s = GPUCounters.snapshot()
            XCTAssertEqual(s.syncs, 1)
            XCTAssertGreaterThanOrEqual(s.commits, 1)
            XCTAssertEqual(s.dispatches, 0)
        }
    }

#endif
