// Copyright © 2026 Numen Technologies.

import Foundation
import XCTest

@testable import MLX

/// MLX_METAL_FAST_SYNCH=1 routes cross-stream dependencies through the `fence_wait` /
/// `fence_update` kernels of fence.metal (gemma4-qat#301). Run with
/// `TEST_RUNNER_MLX_METAL_FAST_SYNCH=1`; the core reads the variable once, so it has to be
/// set for the whole test process. Without it the test skips.
class FenceFastSynchTests: XCTestCase {
    func testCrossStreamEvalWithFastSynch() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["MLX_METAL_FAST_SYNCH"] == "1",
            "set TEST_RUNNER_MLX_METAL_FAST_SYNCH=1")
        guard #available(macOS 15, iOS 18, *) else {
            throw XCTSkip("the core uses the fast fence only on macOS 15 / iOS 18+")
        }
        let s1 = Stream(.gpu)
        let s2 = Stream(.gpu)
        let a = MLXArray(0 ..< 1024).asType(.float32) + 1
        let b = MLX.multiply(a, MLXArray(Float(2)), stream: .stream(s1))
        let c = MLX.add(b, MLXArray(Float(3)), stream: .stream(s2))
        let d = MLX.sum(c, stream: .cpu)
        eval(d)
        // sum over i in 0..<1024 of 2(i + 1) + 3 = 1_052_672; every partial sum is an
        // integer below 2^24, so float32 is exact.
        XCTAssertEqual(d.item(Float.self), Float(1024 * 1025 + 3 * 1024))
    }
}
