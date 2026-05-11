# AI4PDEs — Multi-GPU / Multi-Node Domain Decomposition

Distributed PyTorch implementation of the AI4PDE CFD solver with NCCL
inter-GPU communication. Supports several domain-decomposition
strategies, halo-exchange variants, and an optional per-region
profiler.

## Repository layout

```
src/         Solver code (main entry point: src/main.py)
scripts/     SLURM job scripts and matrix runners for SDumont
tools/       Post-processing utilities
```

## The solver (`src/main.py`)

Launched with `torch.distributed.run`. Required arguments:

| Flag                | Values                              | Purpose                                                      |
| ------------------- | ----------------------------------- | ------------------------------------------------------------ |
| `--nx --ny --nz`    | int                                 | Global grid dimensions; must divide cleanly by the process grid |
| `--topology`        | `1d-x` / `1d-y` / `1d-z` / `3d`     | Decomposition strategy; `3d` keeps subdomains cube-like and preserves multigrid depth |
| `--halo-strategy`   | `blocking` / `async_b`              | `blocking` is the baseline; `async_b` issues `batch_isend_irecv` (validated, no measurable gain on V100) |
| `--steps`           | int (default 40)                    | Number of timesteps                                          |
| `--io-mode`         | `none` / `naive` / `async`          | Per-rank writes of `u/v/w/p` to `FPS/{field}{rank}_{step}.npy` every `n_out` steps. `naive` is synchronous; `async` offloads `np.save` to a per-rank thread pool. `none` skips I/O entirely. |
| `--profile`         | `0` / `1`                           | Per-region timing with mandatory `cuda.synchronize`; small overhead |
| `--profile-nvtx`    | `0` / `1`                           | Add NVTX ranges for `nsys` (requires `--profile=1`)          |
| `--debug`           | `0` / `1`                           | Verbose step prints                                          |

## Running on Santos Dumont

The `scripts/` directory has SLURM templates ready for the LNCC Sequana
GPU partitions:

| Script                       | Purpose                                                                      |
| ---------------------------- | ---------------------------------------------------------------------------- |
| `run_exp0.sbatch`            | Single solver run (Exp 0: halo-strategy validation)                          |
| `run_exp0_all.sh`            | Wrapper that submits the full Exp 0 matrix (strategies × runs × topologies)  |
| `run_profile.sbatch`         | Single profile run (`--profile=1`, `--save=0`)                               |
| `run_profile_all.sh`         | Weak-scaling matrix for the profiler                                         |
| `analyze_exp0.sh`            | Numerical validation + speedup summary across Exp 0 runs                     |

The wrappers write logs under `$SCRATCH/logs_comm/`. Each run isolates
its `FPS/` and profile CSVs in its own working directory.

### Minimal example

```bash
sbatch --nodes=2 scripts/run_profile.sbatch 2400 600 1024 0 3d 40
```

This requests 2 nodes, runs the `3d` decomposition on a 2400×600×1024
grid for 40 steps, with the profiler enabled.

## Post-processing (`tools/`)

| Tool                       | Input                                            | Output                                              |
| -------------------------- | ------------------------------------------------ | --------------------------------------------------- |
| `aggregate_profile.py`     | `profile_runs/` (per-rank CSVs)                  | `profile_per_run.csv`, `profile_per_config.csv`, `metrics_per_config.csv` |
| `compare_halo_outputs.py`  | Two `FPS/` directories from different runs       | Per-tensor relative error to verify halo correctness|

## Environment

- Python 3.10, PyTorch 2.9, CUDA 12.8, NCCL 2.27.5
- OpenMPI 4.1.8 (CUDA-aware)
