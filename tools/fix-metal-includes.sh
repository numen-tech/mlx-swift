#!/bin/bash
# Fixing include path for mlx-swift metal headers

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(realpath "${SCRIPT_DIR}/..")

# Where the files end up
OUTPUT_DIR="${ROOT_DIR}/Source/Cmlx/mlx-generated/metal"

# The Cmlx source dir
CMLX_MLX_DIR="${ROOT_DIR}/Source/Cmlx/mlx"

# sub-directory of Cmlx source containing the kernels
KERNELS_INCLUDE_PATH="mlx/backend/metal/kernels"

KERNELS_DIR="${CMLX_MLX_DIR}/${KERNELS_INCLUDE_PATH}"

# list of kernels files to process
# see Source/Cmlx/mlx/mlx/backend/metal/kernels/CMakeLists.txt
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

# We fixup all the header files AND the listed kernel files
HEADERS=$(find "${KERNELS_DIR}" -name "*.h")
KERNELS=$(for file in ${KERNEL_LIST}; do  echo "${KERNELS_DIR}/${file}"; done)

# Regular expression to replace include directives
PATTERN="^#include \"${KERNELS_INCLUDE_PATH}/([^\"]*)\""

mkdir -p "${OUTPUT_DIR}"

# Mimic the original logic in  PrepareMetalShaders::transformIncludes
# Returns rootPath, a string containing a sequence of "../../" to prefix the
# include path
function replaceIncludePrefix {
    #Extract components up to the output dir and drop the last one
    #swift: let pathUnderKernels = url.pathComponents.drop { $0 != "output" }.dropLast()

    absolutePath=$(realpath "${1}")
    absoluteOut=$(realpath "${OUTPUT_DIR}")
    remainingPath=${absolutePath#"$absoluteOut"/}

    # Doing the `dropLast` with `dirname`, handling the case where it returns `.``
    remainingPath=$(dirname "${remainingPath}" | sed -E 's|^\.$||')

    # Build the root path
    # swift:  let rootPath =Array(repeating: "..", count: pathUnderKernels.count - 1).joined(separator: "/")
    #       + ((pathUnderKernels.count - 1 == 0) ? "" : "/")
    IFS='/' read -r -a path <<< "${remainingPath}"
    count=${#path[@]}

    if [ "$count" -le 0 ]; then
        root_path=""
    else
        root_path=$(printf "../%.0s" $(seq 1 "${count}"))
    fi
    echo "${root_path}"
}

# First pass : copy the files if needed
for src in ${HEADERS} ${KERNELS}; do

    relative_path=${src#"$KERNELS_DIR"/}
    dest=${OUTPUT_DIR}/${relative_path}

    # If destination file doesn't exist or if it's older than the source
    # copy from source and replace the #include directives
    if [ ! -e "$dest" ] || [ "$src" -nt "$dest" ]; then
        echo "${src} -> ${dest}"
        mkdir -p "$(dirname "${dest}")"
        cp -p "${src}" "${dest}"
    else
        echo "Skipping $src (more recent destination)"
    fi

done

# second pass: update the include lines
# iterating on src to only process the list of files we copied
# (in case the destination directory has other unrelated files)
for src in ${HEADERS} ${KERNELS}; do

    relative_path=${src#"$KERNELS_DIR"/}
    dest=${OUTPUT_DIR}/${relative_path}
    prefix=$(replaceIncludePrefix "${dest}")

    # for each matching input line, compute the relative path, then replace the line
    while read -r includeLine; do
        includePath=$(echo "${includeLine}" | sed -E -n "s|${PATTERN}|\1|p")

        # Note the absence of "/" between the prefix and the path
        replace="${prefix}${includePath}"

        # Replace the include line with the new one
        echo sed -i '' -e "s|${KERNELS_INCLUDE_PATH}/${includePath}|${replace}|" "${dest}"
        sed -i '' -e "s|${KERNELS_INCLUDE_PATH}/${includePath}|${replace}|" "${dest}"

    done < <(grep -E -o "${PATTERN}" "${dest}")
done

# third pass: compile the precompiled kernels with safe math.
# SwiftPM compiles these with Xcode's fast-math default, while MLX's CMake
# build uses -fno-fast-math. The pragma switches the math mode (arithmetic and
# the intrinsics the kernel body lowers to) to safe, but it is only half of
# -fno-fast-math: the default-namespace fp32 functions follow
# -fmetal-math-fp32-functions, which the fourth pass handles.
# Kernels only (not headers). Idempotent: the first pass recopies a file only
# when the source is newer, so skip any file that already carries the pragma.
SAFE_MATH_PRAGMA="#pragma METAL fp math_mode(safe)"
SAFE_MATH_COMMENT="// SwiftPM compiles this with fast math, MLX's CMake build uses -fno-fast-math; keep them bit-identical."
for src in ${KERNELS}; do

    relative_path=${src#"$KERNELS_DIR"/}
    dest=${OUTPUT_DIR}/${relative_path}

    if ! grep -qxF "${SAFE_MATH_PRAGMA}" "${dest}"; then
        echo "safe math: ${dest}"
        sed -i '' -e "1i\\
${SAFE_MATH_COMMENT}\\
${SAFE_MATH_PRAGMA}
" "${dest}"
    fi
done

# fourth pass: spell the default-namespace fp32 transcendentals precise.
# SwiftPM also passes -fmetal-math-fp32-functions=fast, which sets
# __METAL_MATH_FP32_FUNCTIONS_FAST__ so metal::exp2/log2 lower to
# air.fast_exp2/air.fast_log2; a pragma can't override that, and MLX's CMake
# build (-fno-fast-math) gets the precise functions. Spell the calls precise
# instead. Applies to the generated copies of the kernels and the headers.
# Every site takes a float (metal::precise::exp2/log2 promote half to float,
# which would change meaning). Idempotent: the rewritten text doesn't match.
# tools/check-metal-math.sh verifies the result at the IR level.
for src in ${HEADERS} ${KERNELS}; do

    relative_path=${src#"$KERNELS_DIR"/}
    dest=${OUTPUT_DIR}/${relative_path}

    if grep -qE 'metal::(exp2|log2)\(' "${dest}"; then
        echo "precise fp32 functions: ${dest}"
        sed -i '' -E -e 's/metal::(exp2|log2)\(/metal::precise::\1(/g' "${dest}"
    fi
done
