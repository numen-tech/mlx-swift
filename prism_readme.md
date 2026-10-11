# MLX-Swift with 1-Bit Quantization Support

This fork of [mlx-swift](https://github.com/ml-explore/mlx-swift) adds native 1-bit weight quantization, enabling any MLX 1-bit model to run efficiently on iPhone and iPad at significantly reduced memory footprints.

## Quick Start

### Option A: Swift Package Dependency

Add this fork as a dependency (replacing the standard mlx-swift URL):

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/numen-tech/mlx-swift.git", branch: "prism"),
]
```

No changes are needed in your model loading code. If a model's `config.json` specifies `"bits": 1`, the 1-bit kernels are dispatched automatically through `QuantizedLinear`.

### Option B: Build from Source

```bash
git clone https://github.com/numen-tech/mlx-swift.git
cd mlx-swift
git checkout prism
git submodule update --init
swift build
```

Nothing is patched on top of the submodules: every kernel and host-dispatch change lives in the
`Source/Cmlx/mlx` fork itself, and the generated sources in `Source/Cmlx/mlx-generated/` are
checked in, already regenerated from that submodule.

### Changing kernels or host dispatch

Kernel and host changes go to the MLX C++ fork, [numen-tech/mlx](https://github.com/numen-tech/mlx)
(Metal kernels in `mlx/backend/metal/kernels/quantized.h`, dispatch in `mlx/backend/metal/quantized.cpp`).
This branch tracks upstream MLX main (`version.h` 0.32.4, unreleased; `v0.32.3` + 86 commits): the submodule sits on numen-tech/mlx
`fork/282-sync-upstream-main` ([numen-tech/mlx#10](https://github.com/numen-tech/mlx/pull/10)),
upstream `06eb7483f` plus the fork's patch series (bias-free `affine_sym` kernels, implied bias,
qmv_fast/qmv_wide routing, the dispatch/commit/sync/wait counters that `GPUCounters` reads, the
kernel memo, the NAX gate, precise transcendentals and `spec_decode_verify`). Never regenerate from a
core without those patches: without them the header has no `affine_sym`, so a regeneration silently
drops the bias-free 1/2-bit kernels, and `mlx::core::metal::counters()` does not exist. Changes land
here by bumping the `Source/Cmlx/mlx` submodule and regenerating:

```bash
LC_ALL=C CC=$(xcrun -f clang) CXX=$(xcrun -f clang++) ./tools/update-mlx.sh
```

**Never hand-edit `Source/Cmlx/mlx-generated/`** (or the other files that script writes): the script
deletes and rebuilds them from the submodule on every run, so a hand edit is silently lost the next
time anyone regenerates. `LC_ALL=C` keeps `mlx-generated/cuda/cuda_jit_sources.h` in the checked-in
order (the script enumerates the CUDA sources with a locale-collated glob), and `CC`/`CXX` pin cmake
to Xcode's clang on machines where a toolchain shim shadows `cc`.

## Quantization Format

| Property | Value |
|----------|-------|
| Bits per weight | 1 (packed into uint32, 32 values per word) |
| Group size | 32, 64, or 128 |
| Per-group parameters | fp16 `scale` + fp16 `bias` |
| Effective storage | ~1.1–1.5 bits/weight (depending on group size) |
| Dequantization | `w = scale * bit + bias` |

Models use SafeTensors format with `config.json` containing:
```json
{"quantization": {"bits": 1, "group_size": 128}}
```

## Related Repositories

- [numen-tech/mlx](https://github.com/numen-tech/mlx) — MLX C++ core with 1-bit kernel support (the fork `.gitmodules` names; forked from [PrismML-Eng/mlx](https://github.com/PrismML-Eng/mlx/tree/prism))
- [ml-explore/mlx-swift](https://github.com/ml-explore/mlx-swift) — Upstream mlx-swift
- [ml-explore/mlx](https://github.com/ml-explore/mlx) — Upstream MLX framework

---

## Appendix

### What Changed

#### MLX C++ core (submodule: `Source/Cmlx/mlx`)

The mlx submodule points to [numen-tech/mlx](https://github.com/numen-tech/mlx) (forked from [PrismML-Eng/mlx](https://github.com/PrismML-Eng/mlx/tree/prism)) which adds:

- **Validation** (`ops.cpp`): Accepts `bits=1` in quantize/dequantize operations
- **Metal kernels** (`quantized.h`, `quantized_nax.h`): 1-bit `load_vector`, `qdot`, `qouter`, `dequantize` using bit extraction and `select()` intrinsics; the bias-free (`affine_sym`) 1/2-bit `qmv`/`qmv_fast` path with its in-kernel derived bias (`sym_derived_bias`) and the uint32 wide-load 1-bit `qdot`
- **Kernel instantiation** (`quantized.metal`): `instantiate_quantized_groups(1)` for all group sizes, plus the `affine_sym_qmv[_fast]` set (bits 1 and 2) so the metallib build serves the same names the JIT path builds
- **CPU backend** (`cpu/quantized.cpp`): 1-bit dequantization path

#### Patches

None. Earlier revisions carried a `patches/` directory with manual `git apply` steps on top of the
submodules; both patches are gone. `mlx-quantized-dispatch-1bit.patch` (deleted in `2f4241ef`) added
a host-side `bits >= 2` guard that kept 1-bit matmuls off `qmv_fast`. The guard is unnecessary: the host
fast gate `qmv_fast_k_alignment(bits, mode)` (numen-tech/mlx#10 patches 2 and 4) sends a 1-bit
affine matmul to `qmv_fast` when K is a multiple of 512 (affine 4-bit: 256; other widths and modes:
the full kernel block), and everything else to the generic `qmv`. The partial tail of
`qmv_fast_impl` handles the half block, with upstream's lane predicate and #4516 row clamp
(numen-tech/mlx#10, "qmv_fast tail predicate"). The original `global_scale` patch (passing
`std::nullopt` for core arguments mlx-c did not bind) is gone: upstream mlx-c binds `global_scale`
for quantize/dequantize/qqmm itself. Upstream MLX 0.32.3 then added `global_scale` to `gather_qmm`,
which mlx-c v0.7.0 does not pass, so the mlx-c submodule points at numen-tech/mlx-c branch
`numen/0.7.0-gather-qmm-global-scale` (`cfc471f4`: v0.7.0 plus the open upstream fix
ml-explore/mlx-c#137, which passes `std::nullopt` there). Re-pin to upstream mlx-c once #137 is in a
release. Kernel and host changes live only in
numen-tech/mlx — see "Changing kernels or host dispatch" above.

#### MLX-Swift level

- **`tools/update-mlx.sh`**: Added `steel_conv_3d` build target (required after merging upstream mlx changes)
- **`.gitmodules`**: Points mlx submodule to the 1-bit fork
- **`Source/Cmlx/mlx-generated/`**: Regenerated Metal shaders with 1-bit support
- **`Source/Cmlx/mlx-counters/`, `Source/MLX/GPUCounters.swift`**: C shim + Swift wrapper over the fork's `mlx::core::metal::counters()` (dispatch/commit counts, explicit stream synchronizations and host GPU-completion waits; relaxed atomics, snapshot/reset only while no eval is in flight). SwiftPM-only, gated by `MLX_GPU_COUNTERS`
