import torch
import torch.nn.functional as F

# --- VELOCIDADE U ---
def apply_BC_u(u, u_padded, rank, world_size, ub):
    """
    Aplica BCs Físicas em U.
    Copia 'u' para o centro de 'u_padded' e aplica bordas.
    """
    # 1. Copia o miolo (Interior)
    # Assumindo u: (1, 1, nz, ny, nx) e u_padded com halo de 1
    u_padded[:, :, 1:-1, 1:-1, 1:-1] = u

    # 2. Aplicar BCs Físicas (X e Y) - TODOS os ranks fazem isso

    # X-Min (Inflow)
    u_padded[0, 0, :, :, 0].fill_(ub)
    # X-Max (Outflow)
    u_padded[0, 0, :, :, -1].fill_(ub)

    # Y-Min (Neumann/slip)
    u_padded[0, 0, :, 0, :] = u_padded[0, 0, :, 1, :]
    # Y-Max (Neumann/slip)
    u_padded[0, 0, :, -1, :] = u_padded[0, 0, :, -2, :]

    # 3. Aplicar BCs Físicas (Z) - SÓ ranks das pontas

    if rank == 0:
        # Z-Min (Fundo, free-slip)
        u_padded[0, 0, 0, :, :] = u_padded[0, 0, 1, :, :]
    if rank == (world_size - 1):
        # Z-Max (Topo, free-slip)
        u_padded[0, 0, -1, :, :] = u_padded[0, 0, -2, :, :]

    return u_padded

# --- VELOCIDADE V ---
def apply_BC_v(v, v_padded, rank, world_size):
    """ Aplica BCs Físicas em V. """

    # 1. Copia o miolo
    v_padded[:, :, 1:-1, 1:-1, 1:-1] = v

    # 2. Bordas
    # V é 0 em todas as bordas X e Y
    v_padded[0, 0, :, :, 0].fill_(0.0)
    v_padded[0, 0, :, :, -1].fill_(0.0)
    v_padded[0, 0, :, 0, :].fill_(0.0)
    v_padded[0, 0, :, -1, :].fill_(0.0)

    if rank == 0:
        # Z-Min (Fundo, free-slip)
        v_padded[0, 0, 0, :, :] = v_padded[0, 0, 1, :, :]

    if rank == (world_size - 1):
        # Z-Max (Topo, free-slip)
        v_padded[0, 0, -1, :, :] = v_padded[0, 0, -2, :, :]

    return v_padded

# --- VELOCIDADE W ---
def apply_BC_w(w, w_padded, rank, world_size):
    """ Aplica BCs Físicas em W. """

    # 1. Copia o miolo
    w_padded[:, :, 1:-1, 1:-1, 1:-1] = w

    # 2. Bordas
    # W é 0 em todas as bordas X e Y
    w_padded[0, 0, :, :, 0].fill_(0.0)
    w_padded[0, 0, :, :, -1].fill_(0.0)
    w_padded[0, 0, :, 0, :] = w_padded[0, 0, :, 1, :]   # Y-Min
    w_padded[0, 0, :, -1, :] = w_padded[0, 0, :, -2, :] # Y-Max
    # w_padded[0, 0, :, 0, :].fill_(0.0)
    # w_padded[0, 0, :, -1, :].fill_(0.0)

    if rank == 0:
        # Z-Min (Fundo, no-slip)
        w_padded[0, 0, 0, :, :].fill_(0.0)

    if rank == (world_size - 1):
        # Z-Max (Topo, no-slip)
        w_padded[0, 0, -1, :, :].fill_(0.0)

    return w_padded

# --- PRESSÃO P ---
def apply_BC_p(p, p_padded, rank, world_size):
    """ Aplica BCs Físicas em P (Pressão). """

    # 1. Copia o miolo
    p_padded[:, :, 1:-1, 1:-1, 1:-1] = p

    # 2. Bordas
    # Neumann (dp/dn = 0) em quase tudo
    p_padded[0, 0, :, :, 0] = p_padded[0, 0, :, :, 1]    # X-Min
    p_padded[0, 0, :, 0, :] = p_padded[0, 0, :, 1, :]    # Y-Min
    p_padded[0, 0, :, -1, :] = p_padded[0, 0, :, -2, :]  # Y-Max

    # Dirichlet 0 (Outflow) em X-Max
    p_padded[0, 0, :, :, -1].fill_(0.0)

    if rank == 0:
        # Z-Min (Neumann)
        p_padded[0, 0, 0, :, :] = p_padded[0, 0, 1, :, :]

    if rank == (world_size - 1):
        # Z-Max (Neumann)
        p_padded[0, 0, -1, :, :] = p_padded[0, 0, -2, :, :]

    return p_padded

# --- CORRECTION W (Multigrid) ---
def apply_BC_cw(values_w, rank, world_size):
    """
    Aplica BCs em 'cw' (Correction W).
    Gera padding internamente pois é usado no Multigrid (tamanhos variáveis).
    """
    # 1. Cria o padding (halo=1) com zeros
    ww = F.pad(values_w, (1, 1, 1, 1, 1, 1), mode='constant', value=0)

    # 2. Bordas X (West/East) -> Dirichlet 0
    ww[0, 0, :, :, 0].fill_(0.0)
    ww[0, 0, :, :, -1].fill_(0.0)

    # 3. Bordas Y (South/North) -> Dirichlet 0
    ww[0, 0, :, 0, :].fill_(0.0)
    ww[0, 0, :, -1, :].fill_(0.0)

    # 4. Bordas Z (Bottom/Top) - Dependem do Rank
    if rank == 0:
        ww[0, 0, 0, :, :].fill_(0.0)

    if rank == (world_size - 1):
        ww[0, 0, -1, :, :].fill_(0.0)

    return ww

def apply_BC_k(k, k_padded, rank, world_size):
    """
    Aplica BCs para o termo 'k' (energia cinética turbulenta ou similar).
    Consolida a lógica: Zera (Dirichlet 0) em todas as paredes físicas.
    """

    # 1. Copia o miolo (Interior)
    # k_padded tem halo de 1, então preenchemos do índice 1 até o penúltimo
    k_padded[:, :, 1:-1, 1:-1, 1:-1] = k

    # 2. Bordas X (West/East) - Domínio Inteiro
    # No código legado: Left zerava X=0, Right zerava X=End. Agora fazemos ambos.
    k_padded[0, 0, :, :, 0].fill_(0.0)      # X-Min
    k_padded[0, 0, :, :, -1].fill_(0.0)     # X-Max

    # 3. Bordas Y (South/North) - Domínio Inteiro
    # No código legado: Top zerava Y=0, Bottom zerava Y=End. Agora fazemos ambos.
    k_padded[0, 0, :, 0, :].fill_(0.0)      # Y-Min
    k_padded[0, 0, :, -1, :].fill_(0.0)     # Y-Max

    # 4. Bordas Z (Bottom/Top) - Dependem do Rank (Decomposição)

    if rank == 0:
        # Z-Min físico (Fundo) -> Zero
        k_padded[0, 0, 0, :, :].fill_(0.0)

    if rank == (world_size - 1):
        # Z-Max físico (Topo) -> Zero
        k_padded[0, 0, -1, :, :].fill_(0.0)

    return k_padded
