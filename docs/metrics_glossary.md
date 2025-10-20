# NeuroLens Metrics Glossary

| Metric | Description | Units / Formula | Dependencies |
| --- | --- | --- | --- |
| latency_ms | Per-operation wall-clock latency. | milliseconds | Raw op timing from backend. |
| total_latency_ms | Sum of latencies over timeline interval. | milliseconds | Aggregates `latency_ms` across ops. |
| sm_efficiency | Ratio of active SM cycles to total cycles. | fraction (0..1) | Requires SM active cycle counters. |
| achieved_occupancy | Achieved resident warps vs theoretical max. | fraction (0..1) | Derived from kernel occupancy metrics. |
| warp_execution_efficiency | Fraction of active threads per warp. | fraction (0..1) | Kernel execution statistics. |
| ai_flops_per_byte | Arithmetic intensity (FLOPs / bytes moved). | FLOPs/byte | Needs FLOP count and memory traffic. |
| dram_throughput_gbps | DRAM bandwidth utilization. | gigabytes per second | Derived from dram bytes / duration. |
| l2_hit_rate | Fraction of memory accesses served by L2. | fraction (0..1) | L2 hit/miss counters. |
| api_launch_overhead_ms | Host API launch latency. | milliseconds | CPU-side profiling hooks. |
| memcpy_htod_ms | Host-to-device transfer time. | milliseconds | cudaMemcpy HTOD logs. |
| memcpy_dtoh_ms | Device-to-host transfer time. | milliseconds | cudaMemcpy DTOH logs. |
| gpu_utilization | Average GPU active time fraction. | fraction (0..1) | Active cycles vs elapsed time. |
| kernel_duration_ms | Individual kernel duration. | milliseconds | Kernel timestamps. |
| bytes_read | Total bytes read from global memory. | bytes | Memory counters. |
| bytes_write | Total bytes written to global memory. | bytes | Memory counters. |
| p50_ms | 50th percentile latency for an op/category. | milliseconds | Latency distribution; requires histogram. |
| p95_ms | 95th percentile latency for an op/category. | milliseconds | Latency distribution; requires histogram. |
| dram_read_gb | DRAM bytes read per kernel converted to GB. | gigabytes | `bytes_read / 1e9`. |
| dram_write_gb | DRAM bytes written per kernel converted to GB. | gigabytes | `bytes_write / 1e9`. |
| tensor_bytes | Size of tensor payload processed. | bytes | Model metadata or runtime introspection. |
| achieved_flops | Actual floating-point operations executed. | FLOPs | Derived from instruction counters. |
| queue_delay_ms | Time spent waiting in execution queue. | milliseconds | Host-side scheduler data. |
| launch_throughput_ops_s | Operations launched per second. | ops/second | Number of launches / elapsed time. |

## Derived Metric Notes
- `ai_flops_per_byte` requires both `achieved_flops` and `(bytes_read + bytes_write)`; report `null` if data missing.
- `dram_throughput_gbps` is computed as `(dram_read_gb + dram_write_gb) / (kernel_duration_ms / 1000)`.
- `sm_efficiency`, `achieved_occupancy`, `warp_execution_efficiency`, and `gpu_utilization` must be bounded between 0 and 1 inclusive; schema enforces this.
- Percentile metrics (`p50_ms`, `p95_ms`) are optional and should be omitted if backend cannot produce them rather than reporting 0.
