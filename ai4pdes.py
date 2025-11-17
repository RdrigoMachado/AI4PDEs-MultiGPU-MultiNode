#!/usr/bin/env python

#  Copyright (C) 2023
#
#  Boyang Chen, Claire Heaney, Christopher Pain
#  Applied Modelling and Computation Group
#  Department of Earth Science and Engineering
#  Imperial College London
#  ++++++++++++++++++++++++++++++++++++++++
#  Jiangnan Wu, Pin Wu
#  Shanghai Univeristy
#
#  boyang.chen16@imperial.ac.uk
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation,
#  version 3.0 of the License.
#
#  This library is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#  Lesser General Public License for more details.

'''
Freight train under cross wind
-------------------------------------------------------------------------------------
+                                        +                                        +
+                                        +                                        +
+              Rank 0                    +                Rank 1                  +
+                                        +                                        +
+                                        +                                        +
-------------------------------------------------------------------------------------
+                                        +                                        +
+                                        +                                        +
+              Rank 2                    +                Rank 3                  +
+                                        +                                        +
+                                        +                                        +
-------------------------------------------------------------------------------------
'''
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
import torch.multiprocessing as mp
from halo_exchange import halo_exchange_Z, split_and_distribute_tensors, init_process, gather_all_data
from boundary_conditions import apply_BC_u, apply_BC_v, apply_BC_w, apply_BC_p

# # # ################################### # # #
# # # ######   Numerial parameters ###### # # #
# # # ################################### # # #
nx = 800
ny = 320
nz = 320
dx = 0.0125 ; dy = 0.0125 ; dz = 0.0125
Re = 0.001
dt = 0.01
ub = -1.0
ratio_x = int(nx/nz)
ratio_y = int(ny/nz)
nlevel = int(math.log(min(nx, ny, nz), 2)) + 1
# ===> Marcelo
print("nlevel: ", nlevel)
nlevel = 6
print("nlevel: ", nlevel)
# ===> Marcelo
print('How many levels in multigrid:', nlevel)
print('Aspect ratio:', ratio_x)
print('Grid spacing:', dx)
# # # ################################### # # #
# # # ######    Linear Filter      ###### # # #
# # # ################################### # # #
bias_initializer = torch.tensor([0.0])
# Laplacian filters
pd1 = torch.tensor([[2/26, 3/26, 2/26],
       [3/26, 6/26, 3/26],
       [2/26, 3/26, 2/26]])
pd2 = torch.tensor([[3/26, 6/26, 3/26],
       [6/26, -88/26, 6/26],
       [3/26, 6/26, 3/26]])
pd3 = torch.tensor([[2/26, 3/26, 2/26],
       [3/26, 6/26, 3/26],
       [2/26, 3/26, 2/26]])
w1 = torch.zeros([1, 1, 3, 3, 3])
wA = torch.zeros([1, 1, 3, 3, 3])
w1[0, 0, 0,:,:] = pd1/dx**2
w1[0, 0, 1,:,:] = pd2/dx**2
w1[0, 0, 2,:,:] = pd3/dx**2
wA[0, 0, 0,:,:] = -pd1/dx**2
wA[0, 0, 1,:,:] = -pd2/dx**2
wA[0, 0, 2,:,:] = -pd3/dx**2
# Gradient filters
p_div_x1 = torch.tensor([[-0.014, 0.0, 0.014],
       [-0.056, 0.0, 0.056],
       [-0.014, 0.0, 0.014]])
p_div_x2 = torch.tensor([[-0.056, 0.0, 0.056],
       [-0.22, 0.0, 0.22],
       [-0.056, 0.0, 0.056]])
p_div_x3 = torch.tensor([[-0.014, 0.0, 0.014],
       [-0.056, 0.0, 0.056],
       [-0.014, 0.0, 0.014]])
p_div_y1 = torch.tensor([[0.014, 0.056, 0.014],
       [0.0, 0.0, 0.0],
       [-0.014, -0.056, -0.014]])
p_div_y2 = torch.tensor([[0.056, 0.22, 0.056],
       [0.0, 0.0, 0.0],
       [-0.056, -0.22, -0.056]])
p_div_y3 = torch.tensor([[0.014, 0.056, 0.014],
       [0.0, 0.0, 0.0],
       [-0.014, -0.056, -0.014]])
p_div_z1 = torch.tensor([[0.014, 0.056, 0.014],
       [0.056, 0.22, 0.056],
       [0.014, 0.056, 0.014]])
p_div_z2 = torch.tensor([[0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0]])
p_div_z3 = torch.tensor([[-0.014, -0.056, -0.014],
       [-0.056, -0.22, -0.056],
       [-0.014, -0.056, -0.014]])
w2 = torch.zeros([1,1,3,3,3])
w3 = torch.zeros([1,1,3,3,3])
w4 = torch.zeros([1,1,3,3,3])
w2[0,0,0,:,:] = -p_div_x1/dx*0.5
w2[0,0,1,:,:] = -p_div_x2/dx*0.5
w2[0,0,2,:,:] = -p_div_x3/dx*0.5
w3[0,0,0,:,:] = -p_div_y1/dx*0.5
w3[0,0,1,:,:] = -p_div_y2/dx*0.5
w3[0,0,2,:,:] = -p_div_y3/dx*0.5
w4[0,0,0,:,:] = -p_div_z1/dx*0.5
w4[0,0,1,:,:] = -p_div_z2/dx*0.5
w4[0,0,2,:,:] = -p_div_z3/dx*0.5
# Restriction filters
w_res = torch.zeros([1,1,2,2,2])
w_res[0,0,:,:,:] = 0.125
################# Numerical parameters ################
ntime = 20000                     # Time steps
n_out = 1000                       # Results output
iteration = 10                    # Multigrid iteration
nrestart = 0                      # Last time step for restart
ctime_old = 0                     # Last ctime for restart
LIBM = True                      # Immersed boundary method
ctime = 0                         # Initialise ctime
save_fig = True                   # Save results
Restart = False                   # Restart
eplsion_k = 1e-04                 # Stablisatin factor in Petrov-Galerkin for velocity

#diag = np.array(wA)[0,0,1,1,1]    # Diagonal component
diag = wA[0,0,1,1,1].item()

#######################################################
# # # ################################### # # #
# # # #########  AI4Urban MAIN ########## # # #
# # # ################################### # # #
class AI4Urban(nn.Module):
    """docstring for AI4Urban"""
    def __init__(self):
        super(AI4Urban, self).__init__()
        # self.arg = arg
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

        self.A.bias.data = bias_initializer
        self.res.bias.data = bias_initializer
        self.diff.bias.data = bias_initializer
        self.xadv.bias.data = bias_initializer
        self.yadv.bias.data = bias_initializer
        self.zadv.bias.data = bias_initializer

###############################################################
    def solid_body(self, values_u, values_v, values_w, sigma, dt):
        values_u = values_u / (1+dt*sigma)
        values_v = values_v / (1+dt*sigma)
        values_w = values_w / (1+dt*sigma)
        return values_u, values_v, values_w

    def F_cycle_MG(self, rank, world_size, values_uu, values_vv, values_ww, values_p, values_pp, iteration, diag, dt, nlevel, ratio_x, ratio_y):
        boundary_condition_p = self.implementations_p.get(rank, None)
        boundary_condition_cw = self.implementations_cw.get(rank, None)
        b = -(self.xadv(values_uu) + self.yadv(values_vv) + self.zadv(values_ww)) / dt
        for MG in range(iteration):
            w = torch.zeros((1,1,1,1,1), device=f"cuda:{rank}")
            r = self.A(values_pp) - b
            r_s = []
            r_s.append(r)
            for i in range(1,nlevel-1):
                r = self.res(r)
                r_s.append(r)
            for i in reversed(range(1,nlevel-1)):
                ww_padded = apply_BC_cw(w, rank, world_size)
                ww = halo_exchange_Z(ww_padded)
                w = w - self.A(ww) / diag + r_s[i] / diag
                w = self.prol(w)
            values_p = values_p - w
            values_p = values_p - self.A(values_pp) / diag + b / diag
            pp_padded = apply_BC_p(values_p, rank, world_size)
            values_pp = halo_exchange_Z(pp_padded)
        return values_p, w, r

    def PG_vector(self, rank, world_size, values_uu, values_vv, values_ww, values_u, values_v, values_w, k1, k_uu, k_vv, k_ww, sigma):#, ADx_u, ADy_u, ADz_u, ADx_v, ADy_v, ADz_v, ADx_w, ADy_w, ADz_w):
        k_u = torch.ones_like(values_u).clone().detach().to(f"cuda:{rank}")
        k_v = torch.ones_like(values_v).clone().detach().to(f"cuda:{rank}")
        k_w = torch.ones_like(values_w).clone().detach().to(f"cuda:{rank}")
        # ===> Marcelo

        k_uu_padded = apply_BC_k(k_u, rank, world_size)
        k_uu = halo_exchange_Z(k_uu_padded)

        k_vv_padded = apply_BC_k(k_v, rank, world_size)
        k_vv = halo_exchange_Z(k_vv_padded)

        k_ww_padded = apply_BC_k(k_w, rank, world_size)
        k_ww = halo_exchange_Z(k_ww_padded)

        k_u = 0.5 * (k_u * self.diff(values_uu) + self.diff(values_uu * k_uu) - values_u * self.diff(k_uu))
        k_v = 0.5 * (k_v * self.diff(values_vv) + self.diff(values_vv * k_vv) - values_v * self.diff(k_vv))
        k_w = 0.5 * (k_w * self.diff(values_ww) + self.diff(values_ww * k_ww) - values_w * self.diff(k_ww))

        return k_u, k_v, k_w

    def forward(self,rank, world_size, values_u, values_uu, values_v, values_vv, values_w, values_ww, values_p, values_pp, b_uu, b_vv, b_ww, k1, dt, iteration, k_uu, k_vv, k_ww,sigma):
    # Solid body
        if LIBM == True: [values_u, values_v, values_w] = self.solid_body(values_u, values_v, values_w, sigma, dt)
    # Padding velocity vectors
        values_uu = apply_BC_u(values_u, rank, world_size, ub)
        values_vv = apply_BC_v(values_v, rank, world_size)
        values_ww = apply_BC_w(values_w, rank, world_size)
        values_pp = apply_BC_p(values_p, rank, world_size)

        values_uu = halo_exchange_Z(values_uu)
        values_vv = halo_exchange_Z(values_vv)
        values_ww = halo_exchange_Z(values_ww)
        values_pp = halo_exchange_Z(values_pp)
        # Grapx_p = self.xadv(values_pp) * dt ; Grapy_p = self.yadv(values_pp) * dt ; Grapz_p = self.zadv(values_pp) * dt
        # ADx_u = self.xadv(values_uu) ; ADy_u = self.yadv(values_uu) ; ADz_u = self.zadv(values_uu)
        # ADx_v = self.xadv(values_vv) ; ADy_v = self.yadv(values_vv) ; ADz_v = self.zadv(values_vv)
        # ADx_w = self.xadv(values_ww) ; ADy_w = self.yadv(values_ww) ; ADz_w = self.zadv(values_ww)
        # AD2_u = self.diff(values_uu) ; AD2_v = self.diff(values_vv) ; AD2_w = self.diff(values_ww)
    # First step for solving uvw
        [k_u, k_v, k_w] = self.PG_vector(rank, world_size, values_uu, values_vv, values_ww, values_u, values_v, values_w, k1, k_uu, k_vv, k_ww, sigma)
        # ,
        #                                ADx_u, ADy_u, ADz_u, ADx_v, ADy_v, ADz_v, ADx_w, ADy_w, ADz_w)

        b_u = values_u + 0.5 * (Re * k_u * dt - values_u * self.xadv(values_uu) * dt - values_v * self.yadv(values_uu) * dt - values_w * self.zadv(values_uu) * dt) - self.xadv(values_pp) * dt #dtGrapx_p
        b_v = values_v + 0.5 * (Re * k_v * dt - values_u * self.xadv(values_vv) * dt - values_v * self.yadv(values_vv) * dt - values_w * self.zadv(values_vv) * dt) - self.yadv(values_pp) * dt #dtGrapy_p
        b_w = values_w + 0.5 * (Re * k_w * dt - values_u * self.xadv(values_ww) * dt - values_v * self.yadv(values_ww) * dt - values_w * self.zadv(values_ww) * dt) - self.zadv(values_pp) * dt #dtGrapz_p
    # Solid body
        if LIBM == True: [b_u, b_v, b_w] = self.solid_body(b_u, b_v, b_w, sigma, dt)
    # Padding velocity vectors
        b_uu = apply_BC_u(b_u, rank, world_size, ub)
        b_vv = apply_BC_v(b_v, rank, world_size)
        b_ww = apply_BC_w(b_w, rank, world_size)

        b_uu = halo_exchange_Z(b_uu)
        b_vv = halo_exchange_Z(b_vv)
        b_ww = halo_exchange_Z(b_ww)
    # Second step for solving uvw
        [k_u, k_v, k_w] = self.PG_vector(rank, world_size, b_uu, b_vv, b_ww, b_u, b_v, b_w, k1, k_uu, k_vv, k_ww, sigma)
        # ,
        #                                ADx_u, ADy_u, ADz_u, ADx_v, ADy_v, ADz_v, ADx_w, ADy_w, ADz_w)

        values_u = values_u + Re * k_u * dt - b_u * self.xadv(b_uu) * dt - b_v * self.yadv(b_uu) * dt - b_w * self.zadv(b_uu) * dt - self.xadv(values_pp) * dt #dtGrapx_p
        values_v = values_v + Re * k_v * dt - b_u * self.xadv(b_vv) * dt - b_v * self.yadv(b_vv) * dt - b_w * self.zadv(b_vv) * dt - self.yadv(values_pp) * dt #dtGrapy_p
        values_w = values_w + Re * k_w * dt - b_u * self.xadv(b_ww) * dt - b_v * self.yadv(b_ww) * dt - b_w * self.zadv(b_ww) * dt - self.zadv(values_pp) * dt #dtGrapz_p
    # Solid body
        if LIBM == True: [values_u, values_v, values_w] = self.solid_body(values_u, values_v, values_w, sigma, dt)
    # pressure
        values_uu = apply_BC_u(values_u, rank, world_size, ub)
        values_vv = apply_BC_v(values_v, rank, world_size)
        values_ww = apply_BC_w(values_w, rank, world_size)

        values_uu = halo_exchange_Z(values_uu)
        values_vv = halo_exchange_Z(values_vv)
        values_ww = halo_exchange_Z(values_ww)
        [values_p, w ,r] = self.F_cycle_MG(rank, world_size, values_uu, values_vv, values_ww, values_p, values_pp, iteration, diag, dt, nlevel, ratio_x, ratio_y)
    # Pressure gradient correction
        values_pp = apply_BC_p(values_p, rank, world_size)
        values_pp = halo_exchange_Z(values_pp)

        values_u = values_u - self.xadv(values_pp) * dt
        values_v = values_v - self.yadv(values_pp) * dt
        values_w = values_w - self.zadv(values_pp) * dt
    # Solid body
        if LIBM == True: [values_u, values_v, values_w] = self.solid_body(values_u, values_v, values_w, sigma, dt)
        return values_u, values_v, values_w, values_p, w, r

def train(rank, world_size):
    global dt, ntime, nx, ny, nz, n_out, iteration, save_fig, diag, ub, Re, LIBM, Restart, nrestart, ctime, ctime_old

    # 1. INICIALIZAR PROCESSO
    init_process(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # 2. CALCULAR DIMENSÕES LOCAIS (Z)
    z_per_rank = nz // world_size
    z_start = rank * z_per_rank
    z_end = (rank + 1) * z_per_rank

    # Lidar com 'nz' não perfeitamente divisível
    if rank == world_size - 1:
        z_end = nz

    local_nz = z_end - z_start

    if rank == 0:
        print(f"Domínio Global Z: {nz}. Ranks: {world_size}. Z/rank (aprox): {z_per_rank}")
    print(f"Rank {rank}: Z-Slice de {z_start} a {z_end} (local_nz={local_nz})")

    # 3. INICIALIZAR TENSORES LOCAIS
    local_shape = (1, 1, local_nz, ny, nx)
    # Padding de Z é 1, padding de X/Y é 2 (baseado no seu input_shape_pad)
    local_shape_pad = (1, 1, local_nz + 2, ny + 4, nx + 4)

    values_u = torch.zeros(local_shape, device=device)
    values_v = torch.zeros(local_shape, device=device)
    values_w = torch.zeros(local_shape, device=device)
    values_p = torch.zeros(local_shape, device=device)
    k1 = torch.ones(local_shape, device=device) * 2.0

    values_uu = torch.zeros(local_shape_pad, device=device)
    values_vv = torch.zeros(local_shape_pad, device=device)
    values_ww = torch.zeros(local_shape_pad, device=device)
    values_pp = torch.zeros(local_shape_pad, device=device)
    b_uu = torch.zeros(local_shape_pad, device=device)
    b_vv = torch.zeros(local_shape_pad, device=device)
    b_ww = torch.zeros(local_shape_pad, device=device)
    k_uu = torch.zeros(local_shape_pad, device=device)
    k_vv = torch.zeros(local_shape_pad, device=device)
    k_ww = torch.zeros(local_shape_pad, device=device)

    #######################################################
    if rank == 0:
        print('============== Numerical parameters ===============')
        print(f'Global Mesh resolution: (1, 1, {nz}, {ny}, {nx})')
        print(f'Local Mesh resolution (Rank 0): {local_shape}')
        print('Time step:', ntime)
        print('Initial time:', ctime)
        print('Diagonal componet:', diag)
    #######################################################
    ################# Only for restart ####################
    if Restart == True:
        nrestart = 8000
        ctime_old = nrestart*dt
        if rank == 0: print('Restart solver!')
    # #######################################################
    # ################# Only for IBM (LOCALIZADO) ########################
    sigma = torch.zeros(local_shape, dtype=torch.float32, device=device)
    if LIBM == True:
        for k_local in range(local_nz):
            k_global = k_local + z_start  # Converte k local para k global
            for j in range(ny):
                for i in range(nx):
                    dist = ((i * dx - 2)**2 + (j * dy - 2)**2 + (k_global * dz - 2)**2)**0.5
                    if dist <= 0.5:
                        sigma[0,0,k_local,j,i] = 1e08

        if rank == 0:
            # Salvar uma fatia do sigma do rank 0 para verificação
            try:
                slice_k_local = local_nz // 2
                plt.imshow(sigma.cpu()[0,0,slice_k_local,:,:])
                plt.colorbar()
                plt.gca().invert_yaxis()
                plt.savefig('Flow_past_sphere_RANK0_SLICE.jpg')
                plt.close()
                print("Imagem 'sigma' (IBM) do Rank 0 salva.")
            except Exception as e:
                print(f"Erro ao salvar imagem do sigma: {e}")

    # 4. INICIALIZAR MODELO
    model = AI4Urban().to(device)
    # Não use DDP a menos que saiba que é necessário.
    # O seu 'model(rank,...)' sugere um modelo 'Data Parallel' manual.
    # model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # 5. LOOP DE TEMPO
    start = time.time()
    with torch.no_grad():
        for itime in range(nrestart + 1, ntime + 1):

            # CHAME O MODELO com 'world_size'
            [values_u, values_v, values_w, values_p, w, r] = model(
                rank, world_size, # <-- Adicionado world_size
                values_u, values_uu, values_v, values_vv, values_w, values_ww,
                values_p, values_pp, b_uu, b_vv, b_ww,
                k1, dt, iteration, k_uu, k_vv, k_ww, sigma
            )

            # output
            if rank == 0 and itime % 100 == 0: # Imprimir com menos frequência
                print('Time step:', itime)
                print('Pressure error:', np.max(np.abs(w.cpu().detach().numpy())), 'cty equation residual:', np.max(np.abs(r.cpu().detach().numpy())))
                print('========================================================')

            if np.max(np.abs(w.cpu().detach().numpy())) > 80000.0:
                if rank == 0: print(f'Rank {rank} não convergiu !!!!!!')
                break

            if save_fig == True and itime % n_out == 0:
                save_path = 'FPS'
                if rank == 0:
                    os.makedirs(save_path, exist_ok=True)

                # Sincronizar processos antes de salvar
                torch.distributed.barrier()

                # Coletar dados de TODOS os ranks para o rank 0
                gathered_w = gather_all_data(values_w.cpu())
                gathered_u = gather_all_data(values_u.cpu())
                gathered_v = gather_all_data(values_v.cpu())
                gathered_p = gather_all_data(values_p.cpu())

                # Apenas o Rank 0 salva os dados globais
                if rank == 0 and gathered_w is not None:
                    print(f"Salvando dados globais no passo de tempo {itime}...")
                    np.save(save_path+"/w"+str(itime), arr=gathered_w.detach().numpy()[0,0,:,:,:]) # Salvar grid 3D
                    np.save(save_path+"/v"+str(itime), arr=gathered_v.detach().numpy()[0,0,:,:,:])
                    np.save(save_path+"/u"+str(itime), arr=gathered_u.detach().numpy()[0,0,:,:,:])
                    np.save(save_path+"/p"+str(itime), arr=gathered_p.detach().numpy()[0,0,:,:,:])

        end = time.time()
        if rank == 0:
            print('Tempo total de simulação:', (end-start))

        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    # Use o número de GPUs disponíveis, ou defina manualmente
    world_size = torch.cuda.device_count()
    if world_size < 1:
        print("Nenhuma GPU encontrada. Abortando.")
        exit()

    if nz % world_size != 0:
        print(f"Aviso: nz ({nz}) não é perfeitamente divisível por world_size ({world_size}).")
        print("A lógica de 'gather' pode precisar de ajustes se isso causar problemas.")

    # Parâmetros globais precisam estar definidos antes de mp.spawn
    nx = 800
    ny = 320
    nz = 320
    # ... (outras variáveis globais que 'train' usa) ...
    dt = 0.01
    ntime = 20000
    n_out = 1000
    iteration = 10
    save_fig = True
    # diag = np.array(wA)[0,0,1,1,1]
    diag = wA[0,0,1,1,1].item()

    ub = -1.0
    Re = 0.001
    LIBM = True
    Restart = False
    nrestart = 0
    ctime = 0
    ctime_old = 0

    print(f"Iniciando {world_size} processos...")
    # REMOVA os tensores globais dos argumentos
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
