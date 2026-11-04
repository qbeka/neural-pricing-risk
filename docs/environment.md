# Environment

Recorded 2026-11-04 on the machine that produced every result in this repository.

| Item | Value |
|---|---|
| CPU | Apple M5 Pro, 18 cores (6 performance, 12 efficiency) |
| Memory | 48 GB |
| Accelerator | Apple M5 Pro GPU, 20 cores, Metal 4, available to PyTorch as MPS (32-bit floats only) |
| Disk | 1.2 TB free |
| OS | macOS, Darwin 25.6.0 |
| Python | 3.12 through uv 0.11.24 |
| Pinned packages | `uv.lock`: numpy 2.5.4, scipy 1.18.1, torch 2.15.0, QuantLib 1.44, matplotlib 3.11.2, pytest 9.2.0 |

No paid compute, licensed data or customer data is used. All inputs are synthetic and generated from saved seeds.

## Kernel timing that set the engine choice

Time for one payoff evaluation on 2^19 paths with 12 fixings. Best of 5 repeats, measured once on an idle machine, so read these as rough guides rather than final results.

| Backend | Precision | Threads | Time (ms) |
|---|---|---|---|
| NumPy | 64-bit | 1 | 23.9 |
| PyTorch CPU | 64-bit | 1 | 14.7 |
| PyTorch CPU | 64-bit | 6 | 3.9 |
| PyTorch CPU | 32-bit | 6 | 2.2 |
| PyTorch MPS | 32-bit | GPU | 2.1 |

Moving the 2^19 by 12 array to the GPU took 3.0 ms and back took 1.0 ms. The GPU gives no speedup over six CPU threads at the same precision on this workload, and it adds transfers and removes 64-bit floats. PyTorch on the CPU in 64-bit is the primary engine.
