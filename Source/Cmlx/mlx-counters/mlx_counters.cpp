#include "mlx_counters.h"

#include <cstdlib>
#include <cstring>
#include <string>

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/gpu_family.h"
#include "mlx/backend/metal/metal.h"

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

extern "C" int mlx_nax_available(void) {
  return mlx::core::metal::is_nax_available() ? 1 : 0;
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
