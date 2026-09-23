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
The submodule must sit at the head of [numen-tech/mlx#1](https://github.com/numen-tech/mlx/pull/1)
(`fork/179-affine-sym-kernels`; the `prism-0.31.1-fixes` tip once that PR has merged), never at the
pre-#1 `prism-0.31.1-fixes` tip: at `b2b8a3d8` the header has no `affine_sym`, so a regeneration from
it silently drops the bias-free 1/2-bit kernels. Changes land here by bumping the `Source/Cmlx/mlx`
submodule and regenerating:

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
a host-side `bits >= 2` guard that kept 1-bit matmuls off `qmv_fast`. The guard is unnecessary: the
fork's 1-bit `qmv_fast` handles the `K % 512` remainder in-kernel (the partial tail block of
`qmv_fast_impl`) and, as of numen-tech/mlx `c431c502` (numen-tech/mlx#1, Codex P1), skips the
inactive-lane weight reads in that tail, so the 1-bit fast path stays on (mlx#3, ~+11% decode). The
`global_scale` patch is unnecessary because the pinned mlx-c (v0.6.0) handles `global_scale` in its
own bindings. Kernel and host changes live only in numen-tech/mlx — see "Changing kernels or host
dispatch" above.

#### MLX-Swift level

- **`tools/update-mlx.sh`**: Added `steel_conv_3d` build target (required after merging upstream mlx changes)
- **`.gitmodules`**: Points mlx submodule to the 1-bit fork
- **`Source/Cmlx/mlx-generated/`**: Regenerated Metal shaders with 1-bit support
