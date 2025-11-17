# Em boundary_manager.py (ou similar)
import torch
import torch.nn.functional as F

# --- VELOCIDADE U ---
def apply_BC_u(values_u, rank, world_size, ub):
    """
    Aplica BCs Físicas em U.
    Baseado no seu 'boundary_condition_3D_u' de inspiração.
    """
    local_nz = values_u.shape[2]

    # 1. Cria o padding (halo=1). Os dados de 'values_u' são copiados para o centro.
    values_uu = F.pad(values_u, (1, 1, 1, 1, 1, 1), mode='constant', value=0)

    # 2. Aplicar BCs Físicas (X e Y) - TODOS os ranks fazem isso

    # X-Min (Inflow)
    values_uu[0, 0, :, :, 0].fill_(ub)
    # X-Max (Outflow)
    values_uu[0, 0, :, :, -1].fill_(ub)

    # Y-Min (Neumann/slip)
    values_uu[0, 0, :, 0, :] = values_uu[0, 0, :, 1, :]
    # Y-Max (Neumann/slip)
    values_uu[0, 0, :, -1, :] = values_uu[0, 0, :, -2, :]

    # 3. Aplicar BCs Físicas (Z) - SÓ ranks das pontas

    if rank == 0:
        # Z-Min (Fundo, no-slip)
        values_uu[0, 0, 0, :, :].fill_(0.0)

    if rank == (world_size - 1):
        # Z-Max (Topo, free-slip)
        values_uu[0, 0, -1, :, :] = values_uu[0, 0, -2, :, :]

    return values_uu

# --- VELOCIDADE V ---
def apply_BC_v(values_v, rank, world_size):
    """ Aplica BCs Físicas em V. """
    local_nz = values_v.shape[2]
    values_vv = F.pad(values_v, (1, 1, 1, 1, 1, 1), mode='constant', value=0)

    # V é 0 em todas as bordas X e Y
    values_vv[0, 0, :, :, 0].fill_(0.0)
    values_vv[0, 0, :, :, -1].fill_(0.0)
    values_vv[0, 0, :, 0, :].fill_(0.0)
    values_vv[0, 0, :, -1, :].fill_(0.0)

    if rank == 0:
        # Z-Min (Fundo, no-slip)
        values_vv[0, 0, 0, :, :].fill_(0.0)

    if rank == (world_size - 1):
        # Z-Max (Topo, free-slip)
        values_vv[0, 0, -1, :, :] = values_vv[0, 0, -2, :, :]

    return values_vv

# --- VELOCIDADE W ---
def apply_BC_w(values_w, rank, world_size):
    """ Aplica BCs Físicas em W. """
    local_nz = values_w.shape[2]
    values_ww = F.pad(values_w, (1, 1, 1, 1, 1, 1), mode='constant', value=0)

    # W é 0 em todas as bordas X e Y
    values_ww[0, 0, :, :, 0].fill_(0.0)
    values_ww[0, 0, :, :, -1].fill_(0.0)
    values_ww[0, 0, :, 0, :].fill_(0.0)
    values_ww[0, 0, :, -1, :].fill_(0.0)

    if rank == 0:
        # Z-Min (Fundo, no-slip)
        values_ww[0, 0, 0, :, :].fill_(0.0)

    if rank == (world_size - 1):
        # Z-Max (Topo, free-slip)
        # Nota: A lógica do seu '3D_w' era Neumann/slip,
        # diferente de U e V que eram no-slip.
        values_ww[0, 0, -1, :, :] = values_ww[0, 0, -2, :, :]

    return values_ww

# --- PRESSÃO P ---
def apply_BC_p(values_p, rank, world_size):
    """ Aplica BCs Físicas em P (Pressão). """
    local_nz = values_p.shape[2]
    values_pp = F.pad(values_p, (1, 1, 1, 1, 1, 1), mode='constant', value=0)

    # Neumann (dp/dn = 0) em quase tudo
    values_pp[0, 0, :, :, 0] = values_pp[0, 0, :, :, 1]    # X-Min
    values_pp[0, 0, :, 0, :] = values_pp[0, 0, :, 1, :]    # Y-Min
    values_pp[0, 0, :, -1, :] = values_pp[0, 0, :, -2, :]  # Y-Max

    # Dirichlet 0 (Outflow) em X-Max
    values_pp[0, 0, :, :, -1].fill_(0.0)

    if rank == 0:
        # Z-Min (Neumann)
        values_pp[0, 0, 0, :, :] = values_pp[0, 0, 1, :, :]

    if rank == (world_size - 1):
        # Z-Max (Neumann)
        values_pp[0, 0, -1, :, :] = values_pp[0, 0, -2, :, :]

    return values_pp
