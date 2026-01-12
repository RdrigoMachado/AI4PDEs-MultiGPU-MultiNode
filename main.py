#!/usr/bin/env python
import os
import numpy as np
import time
import math
import torch
import torch.distributed as distributed
import matplotlib.pyplot as plt

from halo_exchange import init_process, gather_all_data
from solver import AI4Urban

# # # ################################### # # #
# # # ######   Numerial parameters ###### # # #
# # # ################################### # # #
DEBUG_PRINTS = False
SAVE = True
nx = 800 ; ny = 320 ; nz = 320
dx = 0.0125 ; dy = 0.0125 ; dz = 0.0125
Re = 0.001
dt = 0.01
ub = -1.0
iteration = 10
ntime = 40
n_out = 10
LIBM = True

ratio_x = int(nx/nz)
ratio_y = int(ny/nz)
diag = wA[0,0,1,1,1].item()       # Diagonal component
#nlevels is defined by calculate_max_nlevel after backend is initialized

def get_neighbors(rank, world_size):
    neighbors = {
        'left': -1, 'right': -1,
        'top': -1, 'bottom': -1,
        'back': -1, 'front': -1
    }

    # --- X-Axis (Left/Right) ---
    if rank % 2 == 0: # Left Column (0, 2, 4...)
        neighbors['left'] = -1
        neighbors['right'] = rank + 1
    else:             # Right Column (1, 3, 5...)
        neighbors['left'] = rank - 1
        neighbors['right'] = -1

    # --- Y-Axis (Top/Bottom) ---
    if rank % 4 < 2:  # Top Rank: Local (0, 1)
        neighbors['top'] = -1
        neighbors['bottom'] = rank + 2
    else:             # Bottom Rank: Local (2, 3)
        neighbors['top'] = rank - 2
        neighbors['bottom'] = -1

    # --- Z-Axis (Inter Nodes) ---
    # Z is not in the first slice
    if rank >= 4:
        neighbors['back'] = rank - 4

    # Z is not on the last Slice
    if rank < (world_size - 4):
        neighbors['front'] = rank + 4

    return neighbors

def train(rank, world_size, local_rank,nlevel, ratio_x, ratio_y):
    my_neighbors = get_neighbors(rank, world_size)
    # Get Device ID per local rank
    device = torch.device(f"cuda:{local_rank}")

    # compute z local size
    local_nz = nz // world_size
    z_start = rank * local_nz
    z_end = (rank + 1) * local_nz

    # Initialize tensors shapes
    local_shape = (1,1, local_nz, ny, nx)

    # CORREÇÃO 2: Padding de +2 (1 de cada lado) para bater com boundary_conditions.py
    # Antes era +4, o que gera inconsistência de memória
    local_shape_padded = (1,1, local_nz + 2, ny + 2, nx + 2)

    # Initialize tensors
    values_u = torch.zeros(local_shape, device=device)
    values_v = torch.zeros(local_shape, device=device)
    values_w = torch.zeros(local_shape, device=device)
    values_p = torch.zeros(local_shape, device=device)
    k1 = torch.ones(local_shape, device=device)*2.0

    # Initialize local tensors padded
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

    if Restart == True:
        nrestart = 8000
        ctime_old = nrestart * dt
        if rank == 0: print('Restart solver!')

    # Otimização do Sigma (Vetorizado)
    sigma = torch.zeros(local_shape, dtype=torch.float32, device=device)
    if LIBM == True:
        z_coords = (torch.arange(local_nz, device=device).float() + z_start) * dz
        y_coords = torch.arange(ny, device=device).float() * dy
        x_coords = torch.arange(nx, device=device).float() * dx
        Z, Y, X = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
        dist = ((X - 2)**2 + (Y - 2)**2 + (Z - 2)**2)**0.5
        sigma[0, 0, dist <= 0.5] = 1e08
        del X, Y, Z, dist

    model = AI4Urban().to(device)

    if local_rank == 0:
        print('============== Numerical parameters ===============')
        print(f'Global Mesh resolution: (1, 1, {nz}, {ny}, {nx})')
        print(f'Local Mesh resolution (Rank 0): {local_shape}')
        print(f'Time step: {dt}, Steps: {ntime}')
        print('Diagonal componet:', diag)
        os.makedirs('FPS', exist_ok=True)

    start = time.time()
    with torch.no_grad():
        for itime in range(nrestart + 1, ntime + 1):
            if (DEBUG_PRINTS == True and rank == 0):
                print(f'Step {itime}/{ntime}')

            [values_u, values_v, values_w, values_p, w, r] = model(
                rank, world_size,
                values_u, values_uu, values_v, values_vv, values_w, values_ww,
                values_p, values_pp, b_uu, b_vv, b_ww,
                k1, dt, iteration, k_uu, k_vv, k_ww, sigma,
                nlevel, ratio_x, ratio_y,
                my_neighbors
            )

            # Outputs
            if save_fig(itime, n_out):
                save_results(values_u, values_v, values_w, values_p, itime, rank)

    end = time.time()
    if rank == 0:
        print(f'\nSimulation completed. Execution time: {end-start:.2f}s')

def save_fig(itime, n_out):
    return SAVE and itime % n_out == 0

def save_results(u, v, w, p, itime, rank):
    # Função auxiliar para salvar resultados
    global_u = gather_all_data(u)
    global_v = gather_all_data(v)
    global_w = gather_all_data(w)
    global_p = gather_all_data(p)

    if rank == 0:
        print(f"Saving step {itime}")
        try:
            z_slice = global_u.shape[2] // 2
            u_plot = global_u.numpy()[0, 0, z_slice, :, :]
            plt.figure(figsize=(10, 6))
            plt.imshow(u_plot, cmap='jet', origin='upper')
            plt.colorbar(label='U (m/s)')
            plt.title(f'Velocity U - Z={z_slice} - Step {itime}')
            plt.gca().invert_yaxis()
            plt.savefig(f"FPS/Flow_U_step_{itime:05d}.jpg", dpi=100)
            plt.close()
        except Exception as e:
            print(f"Error saving image: {e}")

def calculate_max_nlevel(nx_global, ny_global, nz_global, world_size):
    # 1. Calcula as dimensões locais (Subdomínio de cada GPU)
    # Assumindo sua topologia: 2 em X, 2 em Y, e (world_size/4) em Z
    # Se world_size=16 (4 nós), temos 4 fatias em Z.

    num_z_slices = max(1, world_size // 4)

    local_nx = nx_global // 2  # Dividido em 2 colunas
    local_ny = ny_global // 2  # Dividido em 2 linhas
    local_nz = nz_global // num_z_slices

    print(f"Dimensões Locais: X={local_nx}, Y={local_ny}, Z={local_nz}")

    # 2. Simula o Multigrid para encontrar o gargalo
    level = 1
    current_x, current_y, current_z = local_nx, local_ny, local_nz

    while True:
        # Verifica se alguma dimensão ficaria ímpar ou pequena demais (< 2)
        if (current_x % 2 != 0) or (current_y % 2 != 0) or (current_z % 2 != 0):
            print(f"Parou no Nível {level}: Dimensão ímpar encontrada ({current_x}, {current_y}, {current_z})")
            break

        if (current_x < 4) or (current_y < 4) or (current_z < 4):
            print(f"Parou no Nível {level}: Dimensão muito pequena (< 4)")
            break

        # Avança para o próximo nível (Restrição)
        current_x //= 2
        current_y //= 2
        current_z //= 2
        level += 1

    return level

if __name__ == "__main__":
    rank, world_size, local_rank = init_process(backend='nccl')
    nlevel = calculate_max_nlevel(nx, ny, nz, world_size)

    if rank == 0:
        print('How many levels in multigrid:', nlevel)
        print('Aspect ratio:', ratio_x)
        print('Grid spacing:', dx)
        print(f"Grid Global: {nx}x{ny}x{nz}")
        print(f"Global z domain: {nz}, World size: {world_size}, Local Z length: {nz // world_size}")

        if nz % world_size != 0:
            print(f"Aviso Crítico: nz ({nz}) não é perfeitamente divisível por world_size ({world_size}).")
            print("Isso pode causar erros na troca de halos (halo_exchange). Ajuste nz ou o número de GPUs.")
            if distributed.is_initialized(): distributed.destroy_process_group()
            exit()

    try:
        train(rank, world_size, local_rank, nlevel, ratio_x, ratio_y)
    except KeyboardInterrupt:
        if rank == 0:
            print("\nSimulação interrompida pelo usuário.")
    finally:
        if distributed.is_initialized():
            distributed.destroy_process_group()
