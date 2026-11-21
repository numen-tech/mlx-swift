#include "mlx_counters.h"

#include <cstdlib>
#include <cstring>
#include <string>

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/gpu_family.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/c/error.h"
#include "mlx/c/private/mlx.h"

extern "C" void mlx_counters_snapshot(
    uint64_t* dispatches,
    uint64_t* commits,
    uint64_t* syncs,
    uint64_t* waits) {
  auto c = mlx::core::metal::counters();
  if (dispatches) {
    *dispatches = c.dispatches;
  }
  if (commits) {
    *commits = c.commits;
  }
  if (syncs) {
    *syncs = c.syncs;
  }
  if (waits) {
    *waits = c.waits;
  }
}

extern "C" void mlx_counters_reset(void) {
  mlx::core::metal::reset();
}

extern "C" int mlx_apple6_route_version(void) {
  return mlx::core::metal::kApple6RouteVersion;
}

extern "C" int mlx_nax_disable_env_version(void) {
  return mlx::core::metal::kNaxDisableEnvVersion;
}

extern "C" int mlx_qmm_splitk_nax_version(void) {
  return mlx::core::metal::kQmmSplitkNaxVersion;
}

extern "C" int mlx_nonresident_array_version(void) {
  return mlx::core::metal::kNonresidentArrayVersion;
}

extern "C" int mlx_qmv_mma_version(void) {
  return mlx::core::metal::kQmvMmaVersion;
}

extern "C" int mlx_gather_implied_bias_version(void) {
  return mlx::core::metal::kGatherImpliedBiasVersion;
}

extern "C" int mlx_sdpa_d512_prefill_version(void) {
  return mlx::core::metal::kSdpaD512PrefillVersion;
}

extern "C" mlx_array mlx_array_new_data_managed_payload_nonresident(
    void* data,
    const int* shape,
    int dim,
    mlx_dtype dtype,
    void* payload,
    void (*dtor)(void*)) {
  try {
    mlx::core::Shape cpp_shape(shape, shape + dim);
    std::function<void(void*)> cpp_deleter = [dtor, payload](void*) {
      dtor(payload);
    };
    return mlx_array_new_(mlx::core::metal::nonresident_array(
        data, std::move(cpp_shape), mlx_dtype_to_cpp(dtype), cpp_deleter));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_array_new_();
  }
}

extern "C" size_t mlx_residency_wired_bytes(void) {
  if (!mlx::core::metal::is_available()) {
    return 0;
  }
  return mlx::core::metal::device(mlx::core::Device::gpu)
      .residency_sets()
      .wired_size();
}

extern "C" int mlx_residency_sets_enabled(void) {
  if (!mlx::core::metal::is_available()) {
    return 0;
  }
  return mlx::core::metal::device(mlx::core::Device::gpu)
             .residency_sets()
             .enabled()
      ? 1
      : 0;
}

extern "C" int mlx_nax_available(void) {
  return mlx::core::metal::is_nax_available() ? 1 : 0;
}

extern "C" void mlx_set_qmv_mma(int enabled) {
  mlx::core::metal::set_qmv_mma(enabled != 0);
}

extern "C" int mlx_qmv_mma_device(void) {
  return mlx::core::metal::qmv_mma_device() ? 1 : 0;
}

extern "C" void mlx_set_sdpa_d512_prefill(int enabled) {
  mlx::core::metal::set_sdpa_d512_prefill(enabled != 0);
}

extern "C" int mlx_sdpa_d512_prefill(void) {
  return mlx::core::metal::sdpa_d512_prefill() ? 1 : 0;
}

extern "C" void mlx_kernel_trace_start(void) {
  mlx::core::metal::kernel_trace_start();
}

extern "C" char* mlx_kernel_trace_stop(void) {
  std::string joined;
  for (const auto& name : mlx::core::metal::kernel_trace_stop()) {
    joined += name;
    joined += '\n';
  }
  return strdup(joined.c_str());
}
