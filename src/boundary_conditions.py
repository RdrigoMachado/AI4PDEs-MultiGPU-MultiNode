import torch
import torch.nn.functional as F

X_MIN, X_MAX = 0, -1
Y_MIN, Y_MAX = 0, -1
Z_MIN, Z_MAX = 0, -1


def apply_BC_u(u, u_padded, topo, ub):
    u_padded[:, :, 1:-1, 1:-1, 1:-1] = u

    # Padrão Neumann
    u_padded[0, 0, :, :, X_MIN] = u_padded[0, 0, :, :, X_MIN + 1]
    u_padded[0, 0, :, :, X_MAX] = u_padded[0, 0, :, :, X_MAX - 1]
    u_padded[0, 0, :, Y_MIN, :] = u_padded[0, 0, :, Y_MIN + 1, :]
    u_padded[0, 0, :, Y_MAX, :] = u_padded[0, 0, :, Y_MAX - 1, :]
    u_padded[0, 0, Z_MIN, :, :] = u_padded[0, 0, Z_MIN + 1, :, :]
    u_padded[0, 0, Z_MAX, :, :] = u_padded[0, 0, Z_MAX - 1, :, :]

    # Condições Físicas
    if topo.is_xmin:
        u_padded[0, 0, :, :, X_MIN].fill_(ub)
    if topo.is_xmax:
        u_padded[0, 0, :, :, X_MAX].fill_(ub)
    return u_padded


def apply_BC_v(v, v_padded, topo):
    v_padded[:, :, 1:-1, 1:-1, 1:-1] = v

    # Padrão Neumann
    v_padded[0, 0, :, :, X_MIN] = v_padded[0, 0, :, :, X_MIN + 1]
    v_padded[0, 0, :, :, X_MAX] = v_padded[0, 0, :, :, X_MAX - 1]
    v_padded[0, 0, :, Y_MIN, :] = v_padded[0, 0, :, Y_MIN + 1, :]
    v_padded[0, 0, :, Y_MAX, :] = v_padded[0, 0, :, Y_MAX - 1, :]
    v_padded[0, 0, Z_MIN, :, :] = v_padded[0, 0, Z_MIN + 1, :, :]
    v_padded[0, 0, Z_MAX, :, :] = v_padded[0, 0, Z_MAX - 1, :, :]

    # Condições Físicas
    if topo.is_xmin:
        v_padded[0, 0, :, :, X_MIN].fill_(0.0)
    if topo.is_xmax:
        v_padded[0, 0, :, :, X_MAX].fill_(0.0)
    if topo.is_ymin:
        v_padded[0, 0, :, Y_MIN, :].fill_(0.0)
    if topo.is_ymax:
        v_padded[0, 0, :, Y_MAX, :].fill_(0.0)
    return v_padded


def apply_BC_w(w, w_padded, topo):
    w_padded[:, :, 1:-1, 1:-1, 1:-1] = w

    # Padrão Neumann
    w_padded[0, 0, :, :, X_MIN] = w_padded[0, 0, :, :, X_MIN + 1]
    w_padded[0, 0, :, :, X_MAX] = w_padded[0, 0, :, :, X_MAX - 1]
    w_padded[0, 0, :, Y_MIN, :] = w_padded[0, 0, :, Y_MIN + 1, :]
    w_padded[0, 0, :, Y_MAX, :] = w_padded[0, 0, :, Y_MAX - 1, :]
    w_padded[0, 0, Z_MIN, :, :] = w_padded[0, 0, Z_MIN + 1, :, :]
    w_padded[0, 0, Z_MAX, :, :] = w_padded[0, 0, Z_MAX - 1, :, :]

    # Condições Físicas
    if topo.is_xmin:
        w_padded[0, 0, :, :, X_MIN].fill_(0.0)
    if topo.is_xmax:
        w_padded[0, 0, :, :, X_MAX].fill_(0.0)
    if topo.is_zmin:
        w_padded[0, 0, Z_MIN, :, :].fill_(0.0)
    if topo.is_zmax:
        w_padded[0, 0, Z_MAX, :, :].fill_(0.0)
    return w_padded


def apply_BC_p(p, p_padded, topo):
    p_padded[:, :, 1:-1, 1:-1, 1:-1] = p

    # Padrão Neumann (dp/dn = 0)
    p_padded[0, 0, :, :, X_MIN] = p_padded[0, 0, :, :, X_MIN + 1]
    p_padded[0, 0, :, :, X_MAX] = p_padded[0, 0, :, :, X_MAX - 1]
    p_padded[0, 0, :, Y_MIN, :] = p_padded[0, 0, :, Y_MIN + 1, :]
    p_padded[0, 0, :, Y_MAX, :] = p_padded[0, 0, :, Y_MAX - 1, :]
    p_padded[0, 0, Z_MIN, :, :] = p_padded[0, 0, Z_MIN + 1, :, :]
    p_padded[0, 0, Z_MAX, :, :] = p_padded[0, 0, Z_MAX - 1, :, :]

    # Condições Físicas (Dirichlet outflow)
    if topo.is_xmax:
        p_padded[0, 0, :, :, X_MAX].fill_(0.0)
    return p_padded


def apply_BC_k(k, k_padded, topo):
    k_padded[:, :, 1:-1, 1:-1, 1:-1] = k

    # Padrão Neumann
    k_padded[0, 0, :, :, X_MIN] = k_padded[0, 0, :, :, X_MIN + 1]
    k_padded[0, 0, :, :, X_MAX] = k_padded[0, 0, :, :, X_MAX - 1]
    k_padded[0, 0, :, Y_MIN, :] = k_padded[0, 0, :, Y_MIN + 1, :]
    k_padded[0, 0, :, Y_MAX, :] = k_padded[0, 0, :, Y_MAX - 1, :]
    k_padded[0, 0, Z_MIN, :, :] = k_padded[0, 0, Z_MIN + 1, :, :]
    k_padded[0, 0, Z_MAX, :, :] = k_padded[0, 0, Z_MAX - 1, :, :]

    # Condições Físicas
    if topo.is_xmin:
        k_padded[0, 0, :, :, X_MIN].fill_(0.0)
    if topo.is_xmax:
        k_padded[0, 0, :, :, X_MAX].fill_(0.0)
    if topo.is_ymin:
        k_padded[0, 0, :, Y_MIN, :].fill_(0.0)
    if topo.is_ymax:
        k_padded[0, 0, :, Y_MAX, :].fill_(0.0)
    if topo.is_zmin:
        k_padded[0, 0, Z_MIN, :, :].fill_(0.0)
    if topo.is_zmax:
        k_padded[0, 0, Z_MAX, :, :].fill_(0.0)
    return k_padded


def apply_BC_cw(w, topo):
    ww = F.pad(w, (1, 1, 1, 1, 1, 1), mode="constant", value=0)

    # Padrão Neumann
    ww[0, 0, :, :, X_MIN] = ww[0, 0, :, :, X_MIN + 1]
    ww[0, 0, :, :, X_MAX] = ww[0, 0, :, :, X_MAX - 1]
    ww[0, 0, :, Y_MIN, :] = ww[0, 0, :, Y_MIN + 1, :]
    ww[0, 0, :, Y_MAX, :] = ww[0, 0, :, Y_MAX - 1, :]
    ww[0, 0, Z_MIN, :, :] = ww[0, 0, Z_MIN + 1, :, :]
    ww[0, 0, Z_MAX, :, :] = ww[0, 0, Z_MAX - 1, :, :]

    # Condições Físicas
    if topo.is_xmin:
        ww[0, 0, :, :, X_MIN].fill_(0.0)
    if topo.is_xmax:
        ww[0, 0, :, :, X_MAX].fill_(0.0)
    if topo.is_ymin:
        ww[0, 0, :, Y_MIN, :].fill_(0.0)
    if topo.is_ymax:
        ww[0, 0, :, Y_MAX, :].fill_(0.0)
    if topo.is_zmin:
        ww[0, 0, Z_MIN, :, :].fill_(0.0)
    if topo.is_zmax:
        ww[0, 0, Z_MAX, :, :].fill_(0.0)
    return ww
