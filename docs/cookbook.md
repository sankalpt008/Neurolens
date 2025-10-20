# NeuroLens Performance Cookbook

This cookbook summarises common GPU performance bottlenecks observed in transformer and
inference workloads. Each entry explains what the signal means, why it matters, and how
NeuroLens surfaces it through metrics, fingerprints, and rules.

## Low Occupancy
- **Signals:** `achieved_occupancy < 0.35`, `warp_execution_efficiency < 0.6`
- **Why it matters:** SMs leave capacity idle, limiting throughput even when kernels are
  short. Often caused by register pressure, block size choices, or divergent control
  flow.
- **What to try:** Increase block size, reduce shared memory usage, enable compiler
  optimisations that limit register allocation, or restructure kernels for uniform work.

## Memory-Bound Operations
- **Signals:** Arithmetic intensity (`ai_flops_per_byte`) below 8 with high `dram_norm`
  and latency share > 2%.
- **Impact:** DRAM bandwidth throttles performance; SMs stall waiting for memory.
- **Mitigations:** Fuse kernels, increase tile reuse, exploit tensor cores with reduced
  precision (FP16/INT8), or prefetch data.

## Poor L2 Locality
- **Signals:** `l2_hit_rate < 0.4`, simultaneous `dram_norm > 0.5`.
- **Impact:** L2 misses cause repeated DRAM fetches.
- **Mitigations:** Reorder tensors for spatial locality, adjust block shapes, or utilise
  shared memory caches.

## Launch-Bound Workloads
- **Signals:** Host launch overhead (`api_launch_overhead_ms`) dominates total latency.
- **Impact:** CPU dispatch time limits throughput, especially with many tiny kernels.
- **Mitigations:** Batch launches, use CUDA Graphs, or fuse trivial kernels.

## GPU Under-utilisation
- **Signals:** `summary.gpu_utilization < 0.4` despite active kernels.
- **Impact:** GPU sits idle due to host gaps or blocking transfers.
- **Mitigations:** Overlap compute with transfers, pipeline mini-batches, use multiple
  streams, or increase batch size.

## Warp Divergence
- **Signals:** `warp_execution_efficiency < 0.5` alongside moderate arithmetic intensity.
- **Impact:** Divergent branches serialise warp execution.
- **Mitigations:** Refactor kernels to minimise conditional divergence or rely on warp
  specialisation.

## High Memcpy Share
- **Signals:** `memcpy_htod_ms`/`memcpy_dtoh_ms` consuming a large portion of total
  latency.
- **Impact:** PCIe/NVLink transfers dominate runtime.
- **Mitigations:** Keep tensors resident on device, use pagelocked buffers, overlap
  transfers with compute, and avoid unnecessary host round-trips.

## Reading the Metrics
- **Arithmetic Intensity (AI):** Ratio of FLOPs to bytes moved. Higher indicates better
  compute utilisation. AI below the memory roofline implies memory-bound behaviour.
- **Warp Execution Efficiency:** Fraction of threads active per warp. Values near 1.0
  indicate uniform work; lower values highlight divergence or masking.
- **L2 Hit Rate:** Percentage of cache hits in L2. Low values suggest data thrashing or
  poor locality.

NeuroLens fingerprints encode these metrics per operation, while the insights engine
(`neurolens report`) triggers rules when thresholds are breached. Combine fingerprint
diffs with the cookbook to quickly diagnose regressions and prioritise fixes.
