// numen-tech fork addition (numen-tech/gemma4-qat#991).

import Foundation
import MLX
import XCTest

/// The Apple6 route version 2 kernels, built through this package's JIT sources
/// (`Source/Cmlx/mlx-generated`), against the CPU.
///
/// `MLX_FORCE_GPU_FAMILY=apple6` is read once, when the Metal device is created, so these tests
/// only run in a process started with it set
/// (`MLX_FORCE_GPU_FAMILY=apple6 xcrun swift test --filter MLXTests.Apple6RouteTests`) and skip
/// otherwise. On any other GPU the same calls take the normal route, which the rest of the suite
/// covers.
class Apple6RouteTests: XCTestCase {

    override func setUpWithError() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["MLX_FORCE_GPU_FAMILY"] == "apple6",
            "needs MLX_FORCE_GPU_FAMILY=apple6 set for the whole process")
    }

    /// `affine_qgemm_simt` (M >= 5): the three row tiles (BM 8 / 16 / 32, so M 5, 16 and 33
    /// with a ragged second tile), 2/4/8 bits, explicit and implied (0-d factor) biases, and an N
    /// ragged against the 128 / 64 column tiles.
    func testSIMTQuantizedGEMMMatchesCPU() {
        MLXRandom.seed(991)
        let K = 512
        let N = 200
        for bits in [2, 4, 8] {
            let w = MLXRandom.normal([N, K]).asType(.float16) * 0.05
            let (wq, scales, quantBiases) = MLX.quantized(w, groupSize: 64, bits: bits)
            let implied = MLXArray(-Float(1 << (bits - 1))).asType(.float16)
            for biases in [quantBiases!, implied] {
                for M in [5, 16, 33] {
                    let x = MLXRandom.normal([M, K]).asType(.float16)
                    let gpu = quantizedMM(
                        x, wq, scales: scales, biases: biases, transpose: true, groupSize: 64,
                        bits: bits, stream: .gpu)
                    let cpu = quantizedMM(
                        x, wq, scales: scales, biases: biases, transpose: true, groupSize: 64,
                        bits: bits, stream: .cpu)
                    XCTAssertEqual(gpu.shape, [M, N])
                    assertEqual(
                        gpu.asType(.float32), cpu.asType(.float32), rtol: 1e-2, atol: 1e-2)
                }
            }
        }
    }

    /// `sdpa_vector` past 8 query rows and past query length x GQA 32 (16 x 4 = 64), causal.
    func testVectorAttentionTakesLongQueriesMatchesCPU() {
        MLXRandom.seed(9910)
        let q = MLXRandom.normal([1, 8, 16, 128]).asType(.float16)
        let k = MLXRandom.normal([1, 2, 64, 128]).asType(.float16)
        let v = MLXRandom.normal([1, 2, 64, 128]).asType(.float16)
        let scale = Float(1.0 / sqrt(128.0))
        let gpu = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: .causal, stream: .gpu)
        let cpu = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: .causal, stream: .cpu)
        assertEqual(gpu.asType(.float32), cpu.asType(.float32), rtol: 1e-2, atol: 1e-2)
    }
}
