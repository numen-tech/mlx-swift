#include "mlx_counters.h"

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
