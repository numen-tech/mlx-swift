# MLX-Swift with 1-Bit Quantization Support

This fork of [mlx-swift](https://github.com/ml-explore/mlx-swift) adds native 1-bit weight quantization, enabling any MLX 1-bit model to run efficiently on iPhone and iPad at significantly reduced memory footprints.

## Quick Start

### Option A: Swift Package Dependency

Add this fork as a dependency (replacing the standard mlx-swift URL):

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/PrismML-Eng/mlx-swift.git", branch: "prism"),
]
```

No changes are needed in your model loading code. If a model's `config.json` specifies `"bits": 1`, the 1-bit kernels are dispatched automatically through `QuantizedLinear`.

### Option B: Build from Source

```bash
git clone https://github.com/PrismML-Eng/mlx-swift.git
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
(Metal kernels in `mlx/backend/metal/kernels/quantized.h`, dispatch in `mlx/backend/metal/quantized.cpp`;
the `prism-0.31.1-fixes` branch today). They land here by bumping the `Source/Cmlx/mlx` submodule and
regenerating:

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

- [PrismML-Eng/mlx](https://github.com/PrismML-Eng/mlx/tree/prism) — MLX C++ core with 1-bit kernel support
- [ml-explore/mlx-swift](https://github.com/ml-explore/mlx-swift) — Upstream mlx-swift
- [ml-explore/mlx](https://github.com/ml-explore/mlx) — Upstream MLX framework

---

## Appendix

### What Changed

#### MLX C++ core (submodule: `Source/Cmlx/mlx`)

The mlx submodule points to [PrismML-Eng/mlx](https://github.com/PrismML-Eng/mlx/tree/prism) which adds:

- **Validation** (`ops.cpp`): Accepts `bits=1` in quantize/dequantize operations
- **Metal kernels** (`quantized.h`, `quantized_nax.h`): 1-bit `load_vector`, `qdot`, `qouter`, `dequantize` using bit extraction and `select()` intrinsics
- **Kernel instantiation** (`quantized.metal`): `instantiate_quantized_groups(1)` for all group sizes
- **CPU backend** (`cpu/quantized.cpp`): 1-bit dequantization path

#### Patches

None. Earlier revisions carried a `patches/` directory with manual `git apply` steps on top of the
submodules; both patches are gone. The 1-bit host-dispatch guard was never needed (the fork's 1-bit
`qmv_fast` path is correct and covered by its consumers' tests), and the pinned mlx-c (v0.6.0) handles
`global_scale` in its own bindings. Kernel and host changes live only in numen-tech/mlx — see
"Changing kernels or host dispatch" above.

#### MLX-Swift level

- **`tools/update-mlx.sh`**: Added `steel_conv_3d` build target (required after merging upstream mlx changes)
- **`.gitmodules`**: Points mlx submodule to the 1-bit fork
- **`Source/Cmlx/mlx-generated/`**: Regenerated Metal shaders with 1-bit support
