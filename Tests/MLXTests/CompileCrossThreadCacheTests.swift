// Copyright © 2026 Apple Inc.

import Cmlx
import Foundation
import XCTest

@testable import MLX

/// The compile cache is per thread, and a compiled function's id is its own
/// address, which the allocator hands out again once the function is freed.
/// Erasing a function must therefore reach the entries other threads built for
/// it, or a new function with the same id replays the freed function's graph on
/// those threads (gemma4-qat#355).
class CompileCrossThreadCacheTests: XCTestCase {

    /// A thread that stays alive between jobs, so its thread-local compile cache
    /// keeps the entries earlier jobs built.
    private final class Worker: @unchecked Sendable {
        private let jobs = DispatchSemaphore(value: 0)
        private let done = DispatchSemaphore(value: 0)
        private var job: (() -> Void)?
        private var stopped = false
        private var thread: Thread!

        init() {
            thread = Thread { [self] in
                while true {
                    jobs.wait()
                    if stopped { return }
                    job!()
                    job = nil
                    done.signal()
                }
            }
            thread.start()
        }

        /// Runs `body` on the worker thread and waits for it.
        func run(_ body: @escaping () -> Void) {
            job = body
            jobs.signal()
            done.wait()
        }

        func stop() {
            stopped = true
            jobs.signal()
        }
    }

    /// Applies the closure `compiled` to a scalar 0 and returns the result.
    private func apply(_ compiled: mlx_closure) -> Float {
        let input = new_mlx_vector_array([MLXArray(Float(0))])
        defer { mlx_vector_array_free(input) }
        var output = mlx_vector_array_new()
        defer { mlx_vector_array_free(output) }
        XCTAssertEqual(mlx_closure_apply(&output, compiled, input), 0)
        return mlx_vector_array_values(output)[0].item(Float.self)
    }

    /// Compiles `x + constant` under the explicit id `id`.
    private func compileAddingConstant(_ constant: Float, id: UInt) -> mlx_closure {
        let closure = new_mlx_closure { [MLXArray(constant) + $0[0]] }
        defer { mlx_closure_free(closure) }
        var compiled = mlx_closure_new()
        XCTAssertEqual(mlx_detail_compile(&compiled, closure, id, false, [], 0), 0)
        return compiled
    }

    func testEraseReachesTheCacheOfAnotherThread() {
        let id: UInt = 0x355
        let worker = Worker()
        defer { worker.stop() }

        let first = compileAddingConstant(1, id: id)
        defer { mlx_closure_free(first) }
        var firstResult: Float = -1
        worker.run { withEvalLock { firstResult = self.apply(first) } }
        XCTAssertEqual(firstResult, 1)

        // erase from this thread; the worker's cache holds the entry
        withEvalLock {
            var cache = mlx_compile_cache_new()
            mlx_detail_compile_cache(&cache)
            defer { mlx_compile_cache_free(cache) }
            mlx_detail_compile_erase(cache, id)
        }

        // a new function under the same id must not replay the old graph
        let second = compileAddingConstant(2, id: id)
        defer { mlx_closure_free(second) }
        var secondResult: Float = -1
        worker.run { withEvalLock { secondResult = self.apply(second) } }
        XCTAssertEqual(secondResult, 2, "the worker thread replayed the erased function's graph")
    }

    func testReleasedCompiledFunctionsLeaveNoEntryOnOtherThreads() {
        // `compile` keys by the address of the CompiledFunction, and a freed
        // address is typically reused by the next allocation of the same size.
        let worker = Worker()
        defer { worker.stop() }

        for i in 1 ... 64 {
            let constant = Float(i)
            let compiled = compile { (x: MLXArray) in x + MLXArray(constant) }
            var result: Float = -1
            worker.run { result = compiled(MLXArray(Float(0))).item(Float.self) }
            XCTAssertEqual(result, constant, "iteration \(i)")
        }
    }
}
