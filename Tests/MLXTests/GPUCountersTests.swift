// numen-tech fork addition (numen-tech/gemma4-qat#181).

#if MLX_GPU_COUNTERS

    import Foundation
    import MLX
    import XCTest

    class GPUCountersTests: XCTestCase {

        private let zero = GPUCounters.Snapshot(dispatches: 0, commits: 0, syncs: 0, waits: 0)

        func testMatmulEvalIncrementsDispatchesAndCommits() {
            let a = MLXRandom.normal([64, 64], key: MLXRandom.key(0))
            let b = MLXRandom.normal([64, 64], key: MLXRandom.key(1))
            eval(a, b)

            GPUCounters.reset()
            XCTAssertEqual(GPUCounters.snapshot(), zero)

            let c = matmul(a, b)
            eval(c)  // snapshot only after the eval has completed
            let afterEval = GPUCounters.snapshot()
            #if canImport(Metal)
                XCTAssertGreaterThanOrEqual(afterEval.dispatches, 1)
                XCTAssertGreaterThanOrEqual(afterEval.commits, 1)
                XCTAssertEqual(afterEval.syncs, 0, "waiting on eval is not a sync")
                // eval blocks on c's event unless the GPU signaled it before the host checked.
                XCTAssertLessThanOrEqual(afterEval.waits, 1)
                eval(c)  // already evaluated: no further wait
                XCTAssertEqual(GPUCounters.snapshot(), afterEval)
            #else
                XCTAssertEqual(afterEval, zero, "no Metal backend: counters stay 0")
            #endif

            GPUCounters.reset()
            XCTAssertEqual(GPUCounters.snapshot(), zero)
        }

        func testSynchronizeCountsASyncAndAWait() {
            eval(MLXArray(1) + MLXArray(2))
            GPUCounters.reset()
            Stream.defaultStream(.gpu).synchronize()
            let s = GPUCounters.snapshot()
            #if canImport(Metal)
                XCTAssertEqual(s.syncs, 1)
                XCTAssertEqual(s.waits, 1, "every sync is one host wait")
                XCTAssertGreaterThanOrEqual(s.commits, 1)
                XCTAssertEqual(s.dispatches, 0)
            #else
                XCTAssertEqual(s, zero, "no Metal backend: counters stay 0")
            #endif
        }
    }

#endif
