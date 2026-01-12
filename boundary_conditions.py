import torch
import torch.nn.functional as F

'''
Intra node division in quadrants X,Y

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

Between nodes divison in Z
'''

# Borders Indexes
X_MIN, X_MAX = 0, -1
Y_MIN, Y_MAX = 0, -1
Z_MIN, Z_MAX = 0, -1

def is_left_x(rank):
    return rank % 2 == 0

def is_top_y(rank):
    return rank % 4 < 2

def is_min_z(rank):
    return rank // 4 == 0

def is_max_z(rank, world_size):
    return rank // 4 == (world_size // 4) -1

def apply_BC_u(u, u_padded, rank, world_size, ub):
    # Copy u to padded tensor
    u_padded[:, :, 1:-1, 1:-1, 1:-1] = u

    # ============================
    # X direction
    # ============================
    u_padded[0, 0, :, :, X_MIN] = u_padded[0, 0, :, :, X_MIN + 1]
    u_padded[0, 0, :, :, X_MAX] = u_padded[0, 0, :, :, X_MAX - 1]
    # ============================
    # Y direction
    # ============================
    u_padded[0, 0, :, Y_MIN, :] = u_padded[0, 0, :, Y_MIN + 1, :]
    u_padded[0, 0, :, Y_MAX, :] = u_padded[0, 0, :, Y_MAX - 1, :]
    # ============================
    # Z direction
    # ============================
    u_padded[0, 0, Z_MIN, :, :] = u_padded[0, 0, Z_MIN + 1, :, :]
    u_padded[0, 0, Z_MAX, :, :] = u_padded[0, 0, Z_MAX - 1, :, :]

    # ============================
    # X direction conditions
    # ============================
    if is_left_x(rank):
        u_padded[0, 0, :, :, X_MIN].fill_(ub)
    else:
        u_padded[0, 0, :, :, X_MAX].fill_(ub)


    return u_padded

def apply_BC_v(v, v_padded, rank, world_size):
    # Copy v to padded tensor
    v_padded[:, :, 1:-1, 1:-1, 1:-1] = v

    # ============================
    # X direction
    # ============================
    v_padded[0, 0, :, :, X_MIN] = v_padded[0, 0, :, :, X_MIN + 1]
    v_padded[0, 0, :, :, X_MAX] = v_padded[0, 0, :, :, X_MAX - 1]
    # ============================
    # Y direction
    # ============================
    v_padded[0, 0, :, Y_MIN, :] = v_padded[0, 0, :, Y_MIN + 1, :]
    v_padded[0, 0, :, Y_MAX, :] = v_padded[0, 0, :, Y_MAX - 1, :]
    # ============================
    # Z direction
    # ============================
    v_padded[0, 0, Z_MIN, :, :] = v_padded[0, 0, Z_MIN + 1, :, :]
    v_padded[0, 0, Z_MAX, :, :] = v_padded[0, 0, Z_MAX - 1, :, :]

    # ============================
    # X direction condition
    # ============================
    if is_left_x(rank):
        v_padded[0, 0, :, :, X_MIN].fill_(0.0)
    else:
        v_padded[0, 0, :, :, X_MAX].fill_(0.0)
    # ============================
    # Y direction condition
    # ============================
    if is_top_y(rank):
        v_padded[0, 0, :, Y_MIN, :].fill_(0.0)
    else:
        v_padded[0, 0, :, Y_MAX, :].fill_(0.0)

    return v_padded

def apply_BC_w(w, w_padded, rank, world_size):
    # Copy w to padded tensor
    w_padded[:, :, 1:-1, 1:-1, 1:-1] = w

    # ============================
    # X direction
    # ============================
    w_padded[0, 0, :, :, X_MIN] = w_padded[0, 0, :, :, X_MIN + 1]
    w_padded[0, 0, :, :, X_MAX] = w_padded[0, 0, :, :, X_MAX - 1]
    # ============================
    # Y direction
    # ============================
    w_padded[0, 0, :, Y_MIN, :] = w_padded[0, 0, :, Y_MIN + 1, :]
    w_padded[0, 0, :, Y_MAX, :] = w_padded[0, 0, :, Y_MAX -1, :]
    # ============================
    # Z direction
    # ============================
    w_padded[0, 0, Z_MIN, :, :] = w_padded[0, 0, Z_MIN + 1, :, :]
    w_padded[0, 0, Z_MAX, :, :] = w_padded[0, 0, Z_MAX - 1, :, :]

    # ============================
    # X direction condition
    # ============================
    if is_left_x(rank):
        w_padded[0, 0, :, :, X_MIN].fill_(0.0)
    else:
        w_padded[0, 0, :, :, X_MAX].fill_(0.0)
    # ============================
    # Z direction condition
    # ============================
    if is_min_z(rank, ):
        w_padded[0, 0, Z_MIN, :, :].fill_(0.0)
    if is_max_z(rank, world_size):
        w_padded[0, 0, Z_MAX, :, :].fill_(0.0)

    return w_padded

def apply_BC_p(p, p_padded, rank, world_size):
    # Copy w to padded tensor
    p_padded[:, :, 1:-1, 1:-1, 1:-1] = p

    # ============================
    # X direction
    # ============================
    p_padded[0, 0, :, :, X_MIN] = p_padded[0, 0, :, :, X_MIN + 1]
    p_padded[0, 0, :, :, X_MAX] = p_padded[0, 0, :, :, X_MAX - 1]
    # ============================
    # Y direction
    # ============================
    p_padded[0, 0, :, Y_MIN, :] = p_padded[0, 0, :, Y_MIN + 1, :]
    p_padded[0, 0, :, Y_MAX, :] = p_padded[0, 0, :, Y_MAX -1, :]
    # ============================
    # Z direction
    # ============================
    p_padded[0, 0, Z_MIN, :, :] = p_padded[0, 0, Z_MIN + 1, :, :]
    p_padded[0, 0, Z_MAX, :, :] = p_padded[0, 0, Z_MAX - 1, :, :]

    # ============================
    # X direction condition
    # ============================
    if not is_left_x(rank):
        p_padded[0, 0, :, :, X_MAX].fill_(0.0)

    return p_padded


def apply_BC_k(k, k_padded, rank, world_size):
    # Copy k to padded tensor
    k_padded[:, :, 1:-1, 1:-1, 1:-1] = k

    # ============================
    # X direction
    # ============================
    k_padded[0, 0, :, :, X_MIN] = k_padded[0, 0, :, :, X_MIN + 1]
    k_padded[0, 0, :, :, X_MAX] = k_padded[0, 0, :, :, X_MAX - 1]
    # ============================
    # Y direction
    # ============================
    k_padded[0, 0, :, Y_MIN, :] = k_padded[0, 0, :, Y_MIN + 1, :]
    k_padded[0, 0, :, Y_MAX, :] = k_padded[0, 0, :, Y_MAX - 1, :]
    # ============================
    # Z direction
    # ============================
    k_padded[0, 0, Z_MIN, :, :] = k_padded[0, 0, Z_MIN + 1, :, :]
    k_padded[0, 0, Z_MAX, :, :] = k_padded[0, 0, Z_MAX - 1, :, :]

    # ============================
    # X direction condition
    # ============================
    if is_left_x(rank):
        k_padded[0, 0, :, :, X_MIN].fill_(0.0)
    else:
        k_padded[0, 0, :, :, X_MAX].fill_(0.0)
    # ============================
    # Y direction condition
    # ============================
    if is_top_y(rank):
        k_padded[0, 0, :, Y_MIN, :].fill_(0.0)
    else:
        k_padded[0, 0, :, Y_MAX, :].fill_(0.0)
    # ============================
    # Z direction condition
    # ============================
    if is_min_z(rank, ):
        k_padded[0, 0, Z_MIN, :, :].fill_(0.0)
    if is_max_z(rank, world_size):
        k_padded[0, 0, Z_MAX, :, :].fill_(0.0)

    return k_padded



def apply_BC_cw(w, rank, world_size):
    # Create padded tensor (Multigrid requires dynamic sizing)
    ww = F.pad(w, (1, 1, 1, 1, 1, 1), mode='constant', value=0)

    # ============================
    # X direction
    # ============================
    ww[0, 0, :, :, X_MIN] = ww[0, 0, :, :, X_MIN + 1]
    ww[0, 0, :, :, X_MAX] = ww[0, 0, :, :, X_MAX - 1]

    # ============================
    # Y direction
    # ============================
    ww[0, 0, :, Y_MIN, :] = ww[0, 0, :, Y_MIN + 1, :]
    ww[0, 0, :, Y_MAX, :] = ww[0, 0, :, Y_MAX - 1, :]

    # ============================
    # Z direction
    # ============================
    ww[0, 0, Z_MIN, :, :] = ww[0, 0, Z_MIN + 1, :, :]
    ww[0, 0, Z_MAX, :, :] = ww[0, 0, Z_MAX - 1, :, :]

    # ============================
    # X direction condition
    # ============================
    if is_left_x(rank):
        ww[0, 0, :, :, X_MIN].fill_(0.0)
    else:
        ww[0, 0, :, :, X_MAX].fill_(0.0)

    # ============================
    # Y direction condition
    # ============================
    if is_top_y(rank):
        ww[0, 0, :, Y_MIN, :].fill_(0.0)
    else:
        ww[0, 0, :, Y_MAX, :].fill_(0.0)

    # ============================
    # Z direction condition
    # ============================
    if is_min_z(rank):
        ww[0, 0, Z_MIN, :, :].fill_(0.0)
    if is_max_z(rank, world_size):
        ww[0, 0, Z_MAX, :, :].fill_(0.0)

    return ww
