// numen-tech fork addition (numen-tech/gemma4-qat#1228).

#if MLX_GPU_COUNTERS

    import Darwin
    import Foundation
    import MLX
    import Metal
    import XCTest

    private final class ReleaseCount: @unchecked Sendable {
        private let lock = NSLock()
        private var count = 0
        var value: Int { lock.withLock { count } }
        func increment() { lock.withLock { count += 1 } }
    }

    final class NonResidentNoCopyTests: XCTestCase {

        private let mib = 1 << 20

        private struct WiredSumPolicy: WiredMemoryPolicy, Hashable, Sendable {
            let id = UUID(uuidString: "0E3F8C51-7A2B-4D6E-9F10-1228A1B2C3D4")!
            func limit(baseline: Int, activeSizes: [Int]) -> Int {
                baseline + activeSizes.reduce(0, +)
            }
        }

        /// Page-aligned memory holding `0, 1, ..., 6, 0, 1, ...` as float32, `bytes` long.
        private func pages(_ bytes: Int) -> UnsafeMutableRawPointer {
            var raw: UnsafeMutableRawPointer?
            XCTAssertEqual(posix_memalign(&raw, Int(getpagesize()), bytes), 0)
            let floats = raw!.bindMemory(to: Float.self, capacity: bytes / 4)
            for i in 0 ..< bytes / 4 { floats[i] = Float(i % 7) }
            return raw!
        }

        func testVendoredCoreCarriesTheNonResidentArray() {
            XCTAssertGreaterThanOrEqual(GPUFamilyRoute.nonResidentNoCopyVersion, 1)
        }

        /// The core builds residency sets on a Metal 3 GPU from macOS 15 / iOS 18; anywhere else
        /// nothing is tracked, and `residencyWiredBytes` reads 0 however high the limit is.
        func testResidencySetAvailabilityMatchesTheCoreGate() {
            let metal3 = MTLCreateSystemDefaultDevice()?.supportsFamily(.metal3) ?? false
            var osSupportsSets = false
            if #available(macOS 15, iOS 18, *) { osSupportsSets = true }
            XCTAssertEqual(GPUFamilyRoute.residencySetsAvailable, metal3 && osSupportsSets)
            if !GPUFamilyRoute.residencySetsAvailable {
                XCTAssertEqual(GPUFamilyRoute.residencyWiredBytes, 0)
            }
        }

        /// A raised wired limit wires a plain no-copy array but never a non-resident one, and the
        /// non-resident one still reads the caller's bytes without a copy, on the CPU stream and
        /// on the GPU. Without residency sets nothing is ever wired, so only the wiring checks
        /// are skipped there; the data and finalizer checks run everywhere.
        func testRaisedWiredLimitLeavesTheNonResidentArrayOut() async throws {
            guard Device.defaultDevice().deviceType == .gpu,
                GPU.deviceInfo().maxRecommendedWorkingSetSize > 64 * mib
            else { throw XCTSkip("needs a GPU with a working set") }
            let bytes = 16 * mib
            let count = bytes / 4
            let ticket = WiredMemoryTicket(
                size: 256 * mib, policy: WiredSumPolicy(), manager: .shared, kind: .active)
            _ = await ticket.start()
            let baseline = GPUFamilyRoute.residencyWiredBytes
            let tracksResidency = GPUFamilyRoute.residencySetsAvailable

            let released = ReleaseCount()
            do {
                let pointer = pages(bytes)
                let nonResident = MLXArray(nonResidentRawPointer: pointer, [count], dtype: .float32)
                {
                    released.increment()
                    free(pointer)
                }
                let backing = nonResident.asData(access: .noCopy).data
                backing.withUnsafeBytes {
                    XCTAssertEqual($0.baseAddress, UnsafeRawPointer(pointer), "copied")
                }
                if tracksResidency {
                    XCTAssertLessThan(GPUFamilyRoute.residencyWiredBytes - baseline, bytes)
                }

                let rows = nonResident.take(MLXArray([0, 1, 2, 3, 4, 5, 6, 7] as [Int32]), stream: .cpu)
                XCTAssertEqual(rows.asArray(Float.self), [0, 1, 2, 3, 4, 5, 6, 0])
                let total = nonResident.sum(stream: .gpu)
                eval(total)
                XCTAssertEqual(total.item(Float.self), Float((0 ..< count).reduce(0) { $0 + $1 % 7 }))
                if tracksResidency {
                    XCTAssertLessThan(GPUFamilyRoute.residencyWiredBytes - baseline, bytes)
                }

                // The control: the same kind of memory through the plain initializer is wired.
                let plainPointer = pages(bytes)
                let plain = MLXArray(rawPointer: plainPointer, [count], dtype: .float32) {
                    free(plainPointer)
                }
                eval(plain)
                if tracksResidency {
                    XCTAssertGreaterThanOrEqual(GPUFamilyRoute.residencyWiredBytes, baseline + bytes)
                }
            }
            // The last reference can drop on a command buffer's completion handler.
            Stream.gpu.synchronize()
            let deadline = Date().addingTimeInterval(2)
            while released.value == 0, Date() < deadline { usleep(1_000) }
            XCTAssertEqual(released.value, 1)
            _ = await ticket.end()
        }
    }

#endif
