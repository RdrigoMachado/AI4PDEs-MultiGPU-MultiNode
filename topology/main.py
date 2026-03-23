#!/usr/bin/env python
import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as distributed

from halo_exchange import Topology, gather_all_data, init_process
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
ntime = 40
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

    save_time_accumulator = 0.0
    start = time.time()

    with torch.no_grad():
        for itime in range(1, ntime + 1):
            if topo.rank == 0 and DEBUG_PRINTS:
                print(f"Step {itime}/{ntime}")

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
            if SAVE and itime % n_out == 0:
                start_save = time.time()

                global_u = gather_all_data(values_u, topo)
                global_v = gather_all_data(values_v, topo)
                global_w = gather_all_data(values_w, topo)
                global_p = gather_all_data(values_p, topo)

                if topo.rank == 0:
                    save_path = "FPS"
                    os.makedirs(save_path, exist_ok=True)
                    print(f"Saving step {itime}")
                    np.save(
                        save_path + "/w" + str(itime), arr=global_w.numpy()[0, 0, :, :]
                    )
                    np.save(
                        save_path + "/v" + str(itime), arr=global_v.numpy()[0, 0, :, :]
                    )
                    np.save(
                        save_path + "/u" + str(itime), arr=global_u.numpy()[0, 0, :, :]
                    )
                    np.save(
                        save_path + "/p" + str(itime), arr=global_p.numpy()[0, 0, :, :]
                    )

                    # Opcional: Salvar Plot
                    try:
                        z_slice_idx = global_u.shape[2] // 2
                        u_plot = global_u.numpy()[0, 0, z_slice_idx, :, :]
                        plt.figure(figsize=(10, 6))
                        plt.imshow(u_plot, cmap="jet", origin="upper")
                        plt.colorbar(label="Velocity U (m/s)")
                        plt.title(f"Velocity U - Z={z_slice_idx} - Step {itime}")
                        plt.gca().invert_yaxis()
                        plt.savefig(f"{save_path}/Flow_U_step_{itime:05d}.jpg", dpi=100)
                        plt.close()
                    except Exception as e:
                        print(f"Erro salvando Plot: {e}")

                save_time_accumulator += time.time() - start_save

    end = time.time()

    if topo.rank == 0:
        print(f"\nExecution_time,{end - start:.2f}s")
        print(f"\nSave_time,{save_time_accumulator:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, required=True, help="Tamanho global em X")
    parser.add_argument("--ny", type=int, required=True, help="Tamanho global em Y")
    parser.add_argument("--nz", type=int, required=True, help="Tamanho global em Z")
    parser.add_argument(
        "--save",
        type=int,
        default=0,
        choices=[0, 1],
        help="Salvar resultados (1) ou não (0)",
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
        choices=["1d-x", "1d-y", "1d-z", "3d"],
        help="Estratégia de divisão da malha (Decomposition Topology)",
    )

    args, unknown = parser.parse_known_args()

    nx = args.nx
    ny = args.ny
    nz = args.nz
    SAVE = bool(args.save)
    DEBUG_PRINTS = bool(args.debug)

    rank, world_size, local_rank = init_process(backend="nccl")

    try:
        topo = Topology(args.topology, rank, world_size, nx, ny, nz)
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
