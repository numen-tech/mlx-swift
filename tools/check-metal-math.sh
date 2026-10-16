#!/bin/bash
# Check that the SwiftPM-precompiled Metal kernels match MLX's CMake build.
#
# SwiftPM compiles Source/Cmlx/mlx-generated/metal/*.metal with
# `-fmetal-math-mode=fast -fmetal-math-fp32-functions=fast`; MLX's CMake build
# uses `-fno-fast-math`. tools/fix-metal-includes.sh closes the gap in the
# generated copies (a safe-math pragma, and metal::precise:: spellings for
# exp2/log2). This tool verifies the result on the LLVM IR: for each kernel in
# KERNEL_LIST it compiles the generated file both ways and compares
#   * the multiset of `@air.<name>` call targets, and
#   * the number of arithmetic instructions carrying the `fast` flag
#     (fadd/fsub/fmul/fdiv/fneg/frem/fcmp).
#
# Run it manually after tools/update-mlx.sh. It prints one line per kernel and
# exits non-zero if any kernel differs in anything other than
#   * air.fast_fmax.* vs air.fmax.* (these differ only for NaN inputs, per the
#     MSL spec), and
#   * the accepted residuals in ACCEPTED_FAST_INSNS below.
# IR is written to a temp dir that is removed on exit.
#
# Usage: tools/check-metal-math.sh

set -euo pipefail
export LC_ALL=C

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(realpath "${SCRIPT_DIR}/..")
METAL_DIR="${ROOT_DIR}/Source/Cmlx/mlx-generated/metal"

# Keep in sync with KERNEL_LIST in tools/fix-metal-includes.sh.
KERNEL_LIST=" \
arg_reduce.metal \
conv.metal \
dot.metal \
gated_delta_update.metal \
gated_delta_update_vjp.metal \
gather_mm_offsets.metal \
layer_norm.metal \
random.metal \
rms_norm.metal \
rope.metal \
searchsorted.metal \
scaled_dot_product_attention.metal \
scaled_dot_product_attention_vjp.metal \
spec_decode.metal \
steel/attn/kernels/steel_attention.metal"

# Accepted residual `fast`-flagged instructions per kernel (file=count). The
# count must match exactly, so any change in either direction fails the check.
# rope.metal: `-d * base` in rope_single/rope/rope_large. The frontend gives
# the unary minus (fneg) the compile-wide fast flag whatever the safe-math
# pragma says, and InstCombine hoists it into the preceding fdiv, giving 2
# instructions (fneg + fdiv) in each of the 9 kernels. Both are a sign flip
# and a division of two exactly-converted integers; on an Apple GPU the
# resulting inv_freq is bit-identical to the -fno-fast-math build (checked
# over 90k (x, grid, base) points, and the check does see the difference when
# the pragma is removed).
ACCEPTED_FAST_INSNS="rope.metal=18"

TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/check-metal-math.XXXXXX")
trap 'rm -rf "${TMP_DIR}"' EXIT

FAST_FLAGS="-fmetal-math-mode=fast -fmetal-math-fp32-functions=fast"
PRECISE_FLAGS="-fno-fast-math"

# air call-target multiset: one "<count> <name>" line per target, sorted.
function air_calls {
    { grep -oE '@air\.[A-Za-z0-9_.]+' "${1}" || true; } | sort | uniq -c | awk '{print $2, $1}'
}

# count of arithmetic instructions carrying the fast flag.
function fast_insns {
    grep -cE '= (fadd|fsub|fmul|fdiv|fneg|frem|fcmp) fast' "${1}" || true
}

status=0
for file in ${KERNEL_LIST}; do
    name=$(echo "${file}" | tr '/' '_')
    dir=$(dirname "${METAL_DIR}/${file}")
    base=$(basename "${file}")

    # Compile from the file's own directory: includes are relative to it.
    # shellcheck disable=SC2086
    (cd "${dir}" && xcrun metal ${FAST_FLAGS} -S -emit-llvm -o "${TMP_DIR}/${name}.swiftpm.ll" "${base}") 2> "${TMP_DIR}/${name}.swiftpm.err" || {
        echo "FAIL ${file}: compile (SwiftPM flags) failed"; cat "${TMP_DIR}/${name}.swiftpm.err"; exit 2; }
    # shellcheck disable=SC2086
    (cd "${dir}" && xcrun metal ${PRECISE_FLAGS} -S -emit-llvm -o "${TMP_DIR}/${name}.cmake.ll" "${base}") 2> "${TMP_DIR}/${name}.cmake.err" || {
        echo "FAIL ${file}: compile (-fno-fast-math) failed"; cat "${TMP_DIR}/${name}.cmake.err"; exit 2; }

    air_calls "${TMP_DIR}/${name}.swiftpm.ll" > "${TMP_DIR}/${name}.swiftpm.air"
    air_calls "${TMP_DIR}/${name}.cmake.ll" > "${TMP_DIR}/${name}.cmake.air"
    fast_swiftpm=$(fast_insns "${TMP_DIR}/${name}.swiftpm.ll")
    fast_cmake=$(fast_insns "${TMP_DIR}/${name}.cmake.ll")

    # Call targets whose counts differ between the two builds, as
    # name:swiftpm/cmake tokens, minus the accepted fast_fmax/fmax pair.
    diffs=$(awk 'NR == FNR {a[$1] = $2; next} {b[$1] = $2}
        END {
            for (n in a) names[n] = 1
            for (n in b) names[n] = 1
            for (n in names) if (a[n] + 0 != b[n] + 0) printf "%s:%d/%d\n", n, a[n], b[n]
        }' "${TMP_DIR}/${name}.swiftpm.air" "${TMP_DIR}/${name}.cmake.air" | sort)
    bad=$(echo "${diffs}" | grep -vE '^(@air\.fast_fmax\.|@air\.fmax\.)' | grep -v '^$' | tr '\n' ' ' || true)
    fmax=$(echo "${diffs}" | grep -E '^(@air\.fast_fmax\.|@air\.fmax\.)' | tr '\n' ' ' || true)
    fmax_note=""
    [ -n "${fmax}" ] && fmax_note=" (accepted fmax diff: ${fmax% })"

    accepted_fast=0
    for entry in ${ACCEPTED_FAST_INSNS}; do
        [ "${entry%%=*}" = "${file}" ] && accepted_fast=${entry#*=}
    done
    fast_expected=$((fast_cmake + accepted_fast))
    [ "${accepted_fast}" != 0 ] && fmax_note="${fmax_note} (accepted ${accepted_fast} fast-flag insns)"

    if [ -n "${bad}" ] || [ "${fast_swiftpm}" != "${fast_expected}" ]; then
        echo "DIFF ${file}: air calls: ${bad:-none}; fast-flag insns swiftpm=${fast_swiftpm} cmake=${fast_cmake} (expected swiftpm=${fast_expected})"
        status=1
    else
        echo "OK   ${file}: air calls match, fast-flag insns swiftpm=${fast_swiftpm} cmake=${fast_cmake}${fmax_note}"
    fi
done

exit ${status}
