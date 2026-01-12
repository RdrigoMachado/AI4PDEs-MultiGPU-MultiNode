#!/usr/bin/env python

#-- Import general libraries
import os
import numpy as np
import pandas as pd
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.distributed as distributed
import torch.multiprocessing as mp

from halo_exchange import halo_exchange_Z, init_process, distribute_tensor, gather_all_data
# CORREÇÃO 4: Adicionado apply_BC_k nos imports
from boundary_conditions import apply_BC_k, apply_BC_u, apply_BC_v, apply_BC_w, apply_BC_p, apply_BC_cw

# # # ################################### # # #
# # # ######   Numerial parameters ###### # # #
# # # ################################### # # #
DEBUG_PRINTS = False
GATHER = True
io_time = 0
nx = 800
# ny = 320
nz = 320
# nx = 960
ny = 320
# nz = 480

dx = 0.0125 ; dy = 0.0125 ; dz = 0.0125
Re = 0.001
dt = 0.01
ub = -1.0
ratio_x = int(nx/nz)
ratio_y = int(ny/nz)

# AVISO: Com nz=40, nlevel=6 pode ser muito alto (40 -> 20 -> 10 -> 5 -> 2 -> 1).
# Se der erro no Upsample, reduza para 4 ou 5.
nlevel = 5

# # # ################################### # # #
# # # ######    Linear Filter      ###### # # #
# # # ################################### # # #
bias_initializer = torch.tensor([0.0])
# Laplacian filters
pd1 = torch.tensor([[2/26, 3/26, 2/26], [3/26, 6/26, 3/26], [2/26, 3/26, 2/26]])
pd2 = torch.tensor([[3/26, 6/26, 3/26], [6/26, -88/26, 6/26], [3/26, 6/26, 3/26]])
pd3 = torch.tensor([[2/26, 3/26, 2/26], [3/26, 6/26, 3/26], [2/26, 3/26, 2/26]])

w1 = torch.zeros([1, 1, 3, 3, 3])
wA = torch.zeros([1, 1, 3, 3, 3])
w1[0, 0, 0,:,:] = pd1/dx**2; w1[0, 0, 1,:,:] = pd2/dx**2; w1[0, 0, 2,:,:] = pd3/dx**2
wA[0, 0, 0,:,:] = -pd1/dx**2; wA[0, 0, 1,:,:] = -pd2/dx**2; wA[0, 0, 2,:,:] = -pd3/dx**2

# Gradient filters
p_div_x1 = torch.tensor([[-0.014, 0.0, 0.014], [-0.056, 0.0, 0.056], [-0.014, 0.0, 0.014]])
p_div_x2 = torch.tensor([[-0.056, 0.0, 0.056], [-0.22, 0.0, 0.22],   [-0.056, 0.0, 0.056]])
p_div_x3 = torch.tensor([[-0.014, 0.0, 0.014], [-0.056, 0.0, 0.056], [-0.014, 0.0, 0.014]])
p_div_y1 = torch.tensor([[0.014, 0.056, 0.014],  [0.0, 0.0, 0.0], [-0.014, -0.056, -0.014]])
p_div_y2 = torch.tensor([[0.056, 0.22, 0.056],   [0.0, 0.0, 0.0], [-0.056, -0.22, -0.056]])
p_div_y3 = torch.tensor([[0.014, 0.056, 0.014],  [0.0, 0.0, 0.0], [-0.014, -0.056, -0.014]])
p_div_z1 = torch.tensor([[0.014, 0.056, 0.014],  [0.056, 0.22, 0.056], [0.014, 0.056, 0.014]])
p_div_z2 = torch.tensor([[0.0, 0.0, 0.0],        [0.0, 0.0, 0.0],      [0.0, 0.0, 0.0]])
p_div_z3 = torch.tensor([[-0.014, -0.056, -0.014], [-0.056, -0.22, -0.056], [-0.014, -0.056, -0.014]])

w2 = torch.zeros([1,1,3,3,3]); w3 = torch.zeros([1,1,3,3,3]); w4 = torch.zeros([1,1,3,3,3])
w2[0,0,0,:,:] = -p_div_x1/dx*0.5; w2[0,0,1,:,:] = -p_div_x2/dx*0.5; w2[0,0,2,:,:] = -p_div_x3/dx*0.5
w3[0,0,0,:,:] = -p_div_y1/dx*0.5; w3[0,0,1,:,:] = -p_div_y2/dx*0.5; w3[0,0,2,:,:] = -p_div_y3/dx*0.5
w4[0,0,0,:,:] = -p_div_z1/dx*0.5; w4[0,0,1,:,:] = -p_div_z2/dx*0.5; w4[0,0,2,:,:] = -p_div_z3/dx*0.5

# Restriction filters
w_res = torch.zeros([1,1,2,2,2])
w_res[0,0,:,:,:] = 0.125

################# Numerical parameters ################
ntime = 40                       # Time steps
n_out = 10                        # Results output
iteration = 10                    # Multigrid iteration
nrestart = 0                      # Last time step for restart
ctime_old = 0                     # Last ctime for restart
LIBM = True                       # Immersed boundary method
ctime = 0                         # Initialise ctime
save_fig = True                   # Save results
Restart = False                   # Restart
eplsion_k = 1e-04                 # Stablisatin factor in Petrov-Galerkin for velocity
diag = wA[0,0,1,1,1].item()       # Diagonal component

class AI4Urban(nn.Module):
    """docstring for AI4Urban"""
    def __init__(self):
        super(AI4Urban, self).__init__()
        self.xadv = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=0)
        self.yadv = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=0)
        self.zadv = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=0)
        self.diff = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=0)

        self.A = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=0)
        self.res = nn.Conv3d(1, 1, kernel_size=2, stride=2, padding=0)
        self.prol = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),)

        self.A.weight.data = wA
        self.res.weight.data = w_res
        self.diff.weight.data = w1
        self.xadv.weight.data = w2
        self.yadv.weight.data = w3
        self.zadv.weight.data = w4

        # Bias init
        self.A.bias.data = bias_initializer
        self.res.bias.data = bias_initializer
        self.diff.bias.data = bias_initializer
        self.xadv.bias.data = bias_initializer
        self.yadv.bias.data = bias_initializer
        self.zadv.bias.data = bias_initializer

    def solid_body(self, values_u, values_v, values_w, sigma, dt):
        values_u = values_u / (1+dt*sigma)
        values_v = values_v / (1+dt*sigma)
        values_w = values_w / (1+dt*sigma)
        return values_u, values_v, values_w

    def F_cycle_MG(self, rank, world_size, local_rank, values_uu, values_vv, values_ww, values_p, values_pp, iteration, diag, dt, nlevel, ratio_x, ratio_y):
        global io_time
        b = -(self.xadv(values_uu) + self.yadv(values_vv) + self.zadv(values_ww)) / dt

        for MG in range(iteration):
            # CORREÇÃO 3: w deve ter o mesmo tamanho que values_p (ou b), não (1,1,1,1,1)
            # w = torch.zeros_like(b)
            w = torch.zeros((1,1,1,1,1), device=f"cuda:{local_rank}")

            r = self.A(values_pp) - b
            r_s = []
            r_s.append(r)

            # Restriction
            for i in range(1, nlevel-1):
                r = self.res(r)
                r_s.append(r)

            # Prolongation
            for i in reversed(range(1,nlevel-1)):
                ww = apply_BC_cw(w, rank, world_size)

                #start = time.time()
                ww = halo_exchange_Z(ww)
                #io_time += time.time()-start

                w = w - self.A(ww) / diag + r_s[i] / diag
                w = self.prol(w)

            values_p = values_p - w
            values_p = values_p - self.A(values_pp) / diag + b / diag
            values_pp = apply_BC_p(values_p, values_pp, rank, world_size)

            #start = time.time()
            values_pp = halo_exchange_Z(values_pp)
            #io_time += time.time()-start

        return values_p, w, r

    def PG_vector(self, rank, world_size, values_uu, values_vv, values_ww, values_u, values_v, values_w, k1, k_uu, k_vv, k_ww, sigma):
        global io_time

        k_u = torch.ones_like(values_u).detach()
        k_v = torch.ones_like(values_v).detach()
        k_w = torch.ones_like(values_w).detach()

        k_uu = apply_BC_k(k_u, k_uu, rank, world_size)
        k_vv = apply_BC_k(k_v, k_vv, rank, world_size)
        k_ww = apply_BC_k(k_w, k_ww, rank, world_size)

        #start = time.time()
        k_uu = halo_exchange_Z(k_uu)
        k_vv = halo_exchange_Z(k_vv)
        k_ww = halo_exchange_Z(k_ww)
        #io_time += time.time()-start

        k_u = 0.5 * (k_u * self.diff(values_uu) + self.diff(values_uu * k_uu) - values_u * self.diff(k_uu))
        k_v = 0.5 * (k_v * self.diff(values_vv) + self.diff(values_vv * k_vv) - values_v * self.diff(k_vv))
        k_w = 0.5 * (k_w * self.diff(values_ww) + self.diff(values_ww * k_ww) - values_w * self.diff(k_ww))
        return k_u, k_v, k_w

    def forward(self,rank, world_size, values_u, values_uu, values_v, values_vv, values_w, values_ww, values_p, values_pp, b_uu, b_vv, b_ww, k1, dt, iteration, k_uu, k_vv, k_ww,sigma):
        global io_time
        # Solid body
        if LIBM == True: [values_u, values_v, values_w] = self.solid_body(values_u, values_v, values_w, sigma, dt)

        # Padding velocity vectors
        if (DEBUG_PRINTS == True):
            print('Apply rules and exchange: U, V, W, P')

        values_uu = apply_BC_u(values_u, values_uu, rank,world_size, ub)
        values_vv = apply_BC_v(values_v, values_vv, rank,world_size)
        values_ww = apply_BC_w(values_w, values_ww, rank,world_size)
        values_pp = apply_BC_p(values_p, values_pp, rank,world_size)

        #start = time.time()
        values_uu = halo_exchange_Z(values_uu)
        values_vv = halo_exchange_Z(values_vv)
        values_ww = halo_exchange_Z(values_ww)
        values_pp = halo_exchange_Z(values_pp)
        #io_time += time.time()-start

        # First step for solving uvw
        if (DEBUG_PRINTS == True):
            print('Apply rules and exchange: bU, bV, bW')

        [k_u, k_v, k_w] = self.PG_vector(rank, world_size, values_uu, values_vv, values_ww, values_u, values_v, values_w, k1, k_uu, k_vv, k_ww, sigma)
        b_u = values_u + 0.5 * (Re * k_u * dt - values_u * self.xadv(values_uu) * dt - values_v * self.yadv(values_uu) * dt - values_w * self.zadv(values_uu) * dt) - self.xadv(values_pp) * dt
        b_v = values_v + 0.5 * (Re * k_v * dt - values_u * self.xadv(values_vv) * dt - values_v * self.yadv(values_vv) * dt - values_w * self.zadv(values_vv) * dt) - self.yadv(values_pp) * dt
        b_w = values_w + 0.5 * (Re * k_w * dt - values_u * self.xadv(values_ww) * dt - values_v * self.yadv(values_ww) * dt - values_w * self.zadv(values_ww) * dt) - self.zadv(values_pp) * dt

        # Solid body
        if LIBM == True: [b_u, b_v, b_w] = self.solid_body(b_u, b_v, b_w, sigma, dt)

        # Padding velocity vectors
        if (DEBUG_PRINTS == True):
            print('Apply rules and exchange: buu, bvv, bww')

        b_uu = apply_BC_u(b_u,b_uu, rank, world_size, ub)
        b_vv = apply_BC_v(b_v,b_vv, rank, world_size)
        b_ww = apply_BC_w(b_w,b_ww, rank, world_size)

        #start = time.time()
        b_uu = halo_exchange_Z(b_uu)
        b_vv = halo_exchange_Z(b_vv)
        b_ww = halo_exchange_Z(b_ww)
        #io_time += time.time()-start

        # Second step for solving uvw
        if (DEBUG_PRINTS == True):
            print('Apply rules and exchange: U, V, W')

        [k_u, k_v, k_w] = self.PG_vector(rank,world_size, b_uu, b_vv, b_ww, b_u, b_v, b_w, k1, k_uu, k_vv, k_ww, sigma)
        values_u = values_u + Re * k_u * dt - b_u * self.xadv(b_uu) * dt - b_v * self.yadv(b_uu) * dt - b_w * self.zadv(b_uu) * dt - self.xadv(values_pp) * dt
        values_v = values_v + Re * k_v * dt - b_u * self.xadv(b_vv) * dt - b_v * self.yadv(b_vv) * dt - b_w * self.zadv(b_vv) * dt - self.yadv(values_pp) * dt
        values_w = values_w + Re * k_w * dt - b_u * self.xadv(b_ww) * dt - b_v * self.yadv(b_ww) * dt - b_w * self.zadv(b_ww) * dt - self.zadv(values_pp) * dt

        # Solid body
        if LIBM == True: [values_u, values_v, values_w] = self.solid_body(values_u, values_v, values_w, sigma, dt)

        # pressure
        if (DEBUG_PRINTS == True):
            print('Apply rules and exchange: Uu, Vv, Ww')

        values_uu = apply_BC_u(values_u,values_uu, rank, world_size, ub)
        values_vv = apply_BC_v(values_v,values_vv, rank, world_size)
        values_ww = apply_BC_w(values_w,values_ww, rank, world_size)

        #start = time.time()
        values_uu = halo_exchange_Z(values_uu)
        values_vv = halo_exchange_Z(values_vv)
        values_ww = halo_exchange_Z(values_ww)
        #io_time += time.time()-start

        [values_p, w ,r] = self.F_cycle_MG(rank, world_size, local_rank, values_uu, values_vv, values_ww, values_p, values_pp, iteration, diag, dt, nlevel, ratio_x, ratio_y)

        # Pressure gradient correction
        if (DEBUG_PRINTS == True):
            print('Apply rules and exchange: PP, U, V, W')

        values_pp = apply_BC_p(values_p, values_pp, rank, world_size)

        #start = time.time()
        values_pp = halo_exchange_Z(values_pp)
        #io_time += time.time()-start

        values_u = values_u - self.xadv(values_pp) * dt
        values_v = values_v - self.yadv(values_pp) * dt
        values_w = values_w - self.zadv(values_pp) * dt

        # Solid body
        if LIBM == True: [values_u, values_v, values_w] = self.solid_body(values_u, values_v, values_w, sigma, dt)
        return values_u, values_v, values_w, values_p, w, r

def train(rank, world_size, local_rank):
    global dt, ntime, nx, ny, nz, n_out, iteration, save_fig, diag, ub, Re, LIBM, Restart, nrestart, ctime, ctime_old, wA, w_res, w1, w2, w3, w4, io_time, comm_time

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

    if local_rank == 0:
        print('============== Numerical parameters ===============')
        print(f'Global Mesh resolution: (1, 1, {nz}, {ny}, {nx})')
        print(f'Local Mesh resolution (Rank 0): {local_shape}')
        print('Time step:', ntime)
        print('Initial time:', ctime)
        print('Diagonal componet:', diag)
        # save_path = 'FPS'
        save_path = 'FPS'
        os.makedirs(save_path, exist_ok=True)

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

    start = time.time()
    with torch.no_grad():
        if (DEBUG_PRINTS == True):
            print('Main loop start, rank: ', rank)

        for itime in range(nrestart + 1, ntime + 1):
            if (DEBUG_PRINTS == True and rank == 0):
                print(f'Time step: {itime}')

            [values_u, values_v, values_w, values_p, w, r] = model(
                rank, world_size,
                values_u, values_uu, values_v, values_vv, values_w, values_ww,
                values_p, values_pp, b_uu, b_vv, b_ww,
                k1, dt, iteration, k_uu, k_vv, k_ww, sigma
            )

            if save_fig == GATHER and itime % n_out == 0:
                gather_start = time.time()

                # Importante: use .cpu() antes de gather para economizar VRAM no Mestre
                global_u = gather_all_data(values_u)
                global_v = gather_all_data(values_v)
                global_w = gather_all_data(values_w)
                global_p = gather_all_data(values_p)

                if rank == 0:
                    print(f"Saving weights - step {itime}")
                    np.save(save_path+"/w"+str(itime), arr=global_w.numpy()[0,0,:,:])
                    np.save(save_path+"/v"+str(itime), arr=global_v.numpy()[0,0,:,:])
                    np.save(save_path+"/u"+str(itime), arr=global_u.numpy()[0,0,:,:])
                    np.save(save_path+"/p"+str(itime), arr=global_p.numpy()[0,0,:,:])

                    try:
                        # 1. Seleciona o slice central em Z (ou o índice 160 fixo se preferir)
                        # global_u shape: (1, 1, nz, ny, nx)
                        z_slice_idx = global_u.shape[2] // 2

                        # Extrai os dados para Numpy (já está na CPU pois veio do gather)
                        # [0, 0, z, y, x]
                        u_plot = global_u.numpy()[0, 0, z_slice_idx, :, :]

                        plt.figure(figsize=(10, 6))

                        # Plot com colormap 'jet' ou 'RdBu_r' (comum para velocidade)
                        plt.imshow(u_plot, cmap='jet', origin='upper')
                        plt.colorbar(label='Velocity U (m/s)')

                        # Título com o passo de tempo
                        plt.title(f'Velocity U - Z={z_slice_idx} - Step {itime}')

                        # Inverte Y conforme seu script original
                        plt.gca().invert_yaxis()

                        # Salva na pasta correta com nome dinâmico
                        filename = f"{save_path}/Flow_U_step_{itime:05d}.jpg"
                        plt.savefig(filename, dpi=150)

                        # Limpa a memória da figura (CRUCIAL em loops)
                        plt.close()

                    except Exception as e:
                        print(f"Erro ao salvar imagem: {e}")

                    gather_end = time.time()
                    print(f'\nGather weights enlapsed time: {gather_end-gather_start:.2f}s')


        end = time.time()
        if rank == 0:
            # print(f'\nComm time: {comm_time:.2f}s')
            print(f'\nSimulation completed. Execution time: {end-start:.2f}s')

if __name__ == "__main__":
    rank, world_size, local_rank = init_process(backend='nccl')

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
        train(rank, world_size, local_rank)
    except KeyboardInterrupt:
        if rank == 0:
            print("\nSimulação interrompida pelo usuário.")
    finally:
        if distributed.is_initialized():
            distributed.destroy_process_group()
