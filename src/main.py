#!/usr/bin/env python
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.distributed as distributed

import halo_exchange as halo_exchange_mod
from halo_exchange import Topology, init_process
from profiling import profiler
from solver import AI4Urban, wA

# # # ################################### # # #
# # # ######   Numerial parameters ###### # # #
# # # ################################### # # #
dx = 0.0125
dy = 0.0125
dz = 0.0125
Re = 0.001
dt = 0.01
ub = -1.0
iteration = 10
ntime = 40  # default; pode ser sobrescrito via --steps
n_out = 10
LIBM = True
diag = wA[0, 0, 1, 1, 1].item()


def calculate_max_nlevel(local_nx, local_ny, local_nz):
    level = 1
    current_x, current_y, current_z = local_nx, local_ny, local_nz

    while True:
        if (current_x % 2 != 0) or (current_y % 2 != 0) or (current_z % 2 != 0):
            break
        if (current_x < 4) or (current_y < 4) or (current_z < 4):
            break
        current_x //= 2
        current_y //= 2
        current_z //= 2
        level += 1
    return level


def train(topo, local_rank, nlevel):
    device = torch.device(f"cuda:{local_rank}")

    local_shape = (1, 1, topo.local_nz, topo.local_ny, topo.local_nx)
    local_shape_padded = (1, 1, topo.local_nz + 2, topo.local_ny + 2, topo.local_nx + 2)

    # Initialize tensors
    values_u = torch.zeros(local_shape, device=device)
    values_v = torch.zeros(local_shape, device=device)
    values_w = torch.zeros(local_shape, device=device)
    values_p = torch.zeros(local_shape, device=device)
    k1 = torch.ones(local_shape, device=device) * 2.0

    # Padded
    values_uu = torch.zeros(local_shape_padded, device=device)
    values_vv = torch.zeros(local_shape_padded, device=device)
    values_ww = torch.zeros(local_shape_padded, device=device)
    values_pp = torch.zeros(local_shape_padded, device=device)

    b_uu = torch.zeros(local_shape_padded, device=device)
    b_vv = torch.zeros(local_shape_padded, device=device)
    b_ww = torch.zeros(local_shape_padded, device=device)
    k_uu = torch.zeros(local_shape_padded, device=device)
    k_vv = torch.zeros(local_shape_padded, device=device)
    k_ww = torch.zeros(local_shape_padded, device=device)

    # LIBM Sigma com deslocamento dinâmico
    sigma = torch.zeros(local_shape, dtype=torch.float32, device=device)
    if LIBM:
        z_start_val = topo.pz * topo.local_nz * dz
        y_start_val = topo.py * topo.local_ny * dy
        x_start_val = topo.px * topo.local_nx * dx

        z_coords = (
            torch.arange(topo.local_nz, device=device).float() * dz
        ) + z_start_val
        y_coords = (
            torch.arange(topo.local_ny, device=device).float() * dy
        ) + y_start_val
        x_coords = (
            torch.arange(topo.local_nx, device=device).float() * dx
        ) + x_start_val

        Z, Y, X = torch.meshgrid(z_coords, y_coords, x_coords, indexing="ij")
        dist_ = ((X - 2) ** 2 + (Y - 2) ** 2 + (Z - 2) ** 2) ** 0.5
        sigma[0, 0, dist_ <= 0.5] = 1e08
        del X, Y, Z, dist_

    model = AI4Urban().to(device)

    # Reset peak memory após alocar todos os tensores estáticos e o modelo.
    # Mede o pico atribuível ao train loop (multigrid temps, cudnn workspace,
    # buffers NCCL), não ao setup.
    if profiler.enabled and torch.cuda.is_available():
        profiler.record_metric(
            "mem_static_alloc_gb", torch.cuda.memory_allocated() / 1e9
        )
        torch.cuda.reset_peak_memory_stats()

    save_time_accumulator = 0.0
    # Each rank writes its own subdomain — no gather, no central bottleneck.
    # In async mode every rank owns a single-worker executor that overlaps
    # np.save with the next solver steps.
    io_executor = None
    if IO_MODE == "async":
        io_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"io-r{topo.rank}"
        )
    if IO_MODE != "none":
        os.makedirs("FPS", exist_ok=True)

    start = time.time()

    with torch.no_grad():
        for itime in range(1, ntime + 1):
            if topo.rank == 0 and DEBUG_PRINTS:
                print(f"Step {itime}/{ntime}")

            profiler.start("step_total")
            [values_u, values_v, values_w, values_p, w, r] = model(
                topo,
                local_rank,
                values_u,
                values_uu,
                values_v,
                values_vv,
                values_w,
                values_ww,
                values_p,
                values_pp,
                b_uu,
                b_vv,
                b_ww,
                k1,
                dt,
                iteration,
                k_uu,
                k_vv,
                k_ww,
                sigma,
                nlevel,
                ub,
                Re,
            )

            # Outputs (Unificados)
            if IO_MODE != "none" and itime % n_out == 0:
                start_save = time.time()
                profiler.start("io_save")

                if topo.rank == 0:
                    print(f"Saving step {itime}")

                # The .cpu() call is synchronous — the resulting NumPy array
                # is independent of CUDA, so the executor can write it safely.
                arrays = (
                    ("u", values_u[0, 0].cpu().numpy()),
                    ("v", values_v[0, 0].cpu().numpy()),
                    ("w", values_w[0, 0].cpu().numpy()),
                    ("p", values_p[0, 0].cpu().numpy()),
                )
                if io_executor is not None:
                    for name, arr in arrays:
                        io_executor.submit(
                            np.save, f"FPS/{name}{topo.rank}_{itime}", arr
                        )
                else:
                    for name, arr in arrays:
                        np.save(f"FPS/{name}{topo.rank}_{itime}", arr)

                profiler.end("io_save")
                save_time_accumulator += time.time() - start_save
            profiler.end("step_total")

    # Drain pending writes before stopping the wall clock so the metric
    # captures the full I/O cost, not just the submit overhead.
    flush_time = 0.0
    if io_executor is not None:
        flush_start = time.time()
        io_executor.shutdown(wait=True)
        flush_time = time.time() - flush_start

    end = time.time()

    if topo.rank == 0:
        print(f"\nExecution_time,{end - start:.2f}s")
        print(f"\nSave_time,{save_time_accumulator:.2f}s")
        print(f"\nFlush_time,{flush_time:.2f}s")

    if profiler.enabled:
        if torch.cuda.is_available():
            profiler.record_metric(
                "mem_peak_alloc_gb", torch.cuda.max_memory_allocated() / 1e9
            )
            profiler.record_metric(
                "mem_peak_reserved_gb", torch.cuda.max_memory_reserved() / 1e9
            )

        extra_cols = {
            "rank": topo.rank,
            "world_size": topo.world_size,
            "topology": topo.decomp_type,
            "halo_strategy": halo_exchange_mod.get_strategy(),
            "nx": topo.local_nx * topo.PX,
            "ny": topo.local_ny * topo.PY,
            "nz": topo.local_nz * topo.PZ,
        }
        profile_path = f"profile_rank{topo.rank:02d}.csv"
        profiler.dump_csv(profile_path, extra_cols=extra_cols)
        metrics_path = f"metrics_rank{topo.rank:02d}.csv"
        profiler.dump_metrics_csv(metrics_path, extra_cols=extra_cols)
        if topo.rank == 0:
            print(f"\nProfile dumped: {profile_path} (per rank)")
            print(f"Metrics dumped: {metrics_path} (per rank)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, required=True, help="Tamanho global em X")
    parser.add_argument("--ny", type=int, required=True, help="Tamanho global em Y")
    parser.add_argument("--nz", type=int, required=True, help="Tamanho global em Z")
    parser.add_argument(
        "--io-mode",
        type=str,
        default="none",
        choices=["none", "naive", "async"],
        help="I/O mode: none (skip), naive (synchronous np.save), "
             "async (rank 0 ThreadPoolExecutor)",
    )
    parser.add_argument(
        "--debug",
        type=int,
        default=0,
        choices=[0, 1],
        help="Debug prints (1) ou não (0)",
    )
    # Novo argumento de decomposição
    parser.add_argument(
        "--topology",
        type=str,
        default="1d-z",
        choices=["1d-x", "1d-y", "1d-z", "3d", "slab-y-2d"],
        help="Estratégia de divisão da malha (Decomposition Topology). "
             "slab-y-2d: slabs em Y entre nós + 2D XY intra-nó (NVLink).",
    )
    parser.add_argument(
        "--halo-strategy",
        type=str,
        default="blocking",
        choices=["blocking", "async_b"],
        help="Halo exchange: blocking (baseline), async_b (batch_isend_irecv).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=ntime,
        help=f"Número de timesteps (default: {ntime})",
    )
    parser.add_argument(
        "--profile",
        type=int,
        default=0,
        choices=[0, 1],
        help="Habilita profiler com torch.cuda.synchronize() por região "
             "(default 0; tem ~0.5%% overhead por causa do sync)",
    )
    parser.add_argument(
        "--profile-nvtx",
        type=int,
        default=0,
        choices=[0, 1],
        help="Adiciona NVTX ranges para visualização em nsys (requer --profile=1)",
    )

    args, unknown = parser.parse_known_args()
    halo_exchange_mod.set_strategy(args.halo_strategy)
    ntime = args.steps
    if args.profile:
        profiler.enable(sync=True, use_nvtx=bool(args.profile_nvtx))

    nx = args.nx
    ny = args.ny
    nz = args.nz
    IO_MODE = args.io_mode
    DEBUG_PRINTS = bool(args.debug)

    rank, world_size, local_rank, local_world_size = init_process(backend="nccl")

    try:
        topo = Topology(
            args.topology, rank, world_size, nx, ny, nz,
            gpus_per_node=local_world_size,
        )
        nlevel = calculate_max_nlevel(topo.local_nx, topo.local_ny, topo.local_nz)

        if rank == 0 and DEBUG_PRINTS:
            print("============== Configurações ================")
            print(f"Grid Global: {nx}x{ny}x{nz}")
            print(
                f"Decomposition Topology: {args.topology.upper()} (Process Grid: {topo.PX}x{topo.PY}x{topo.PZ})"
            )
            print(
                f"Local shape on node: {topo.local_nx}x{topo.local_ny}x{topo.local_nz}"
            )
            print(f"Max Multigrid Levels: {nlevel}")
            print(f"Halo strategy: {args.halo_strategy}")
            print(f"Profile: {'on' if args.profile else 'off'}"
                  f"{' (+NVTX)' if args.profile and args.profile_nvtx else ''}")
            print("=============================================")

        train(topo, local_rank, nlevel)

    except KeyboardInterrupt:
        if rank == 0:
            print("\nSimulação interrompida pelo usuário.")
    except Exception as e:
        print(f"[Rank {rank}] Falha Crítica: {e}")
    finally:
        if distributed.is_initialized():
            distributed.destroy_process_group()
