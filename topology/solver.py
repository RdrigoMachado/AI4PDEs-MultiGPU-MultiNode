import torch
import torch.nn as nn

from boundary_conditions import (
    apply_BC_cw,
    apply_BC_k,
    apply_BC_p,
    apply_BC_u,
    apply_BC_v,
    apply_BC_w,
)
from halo_exchange import halo_exchange

# # # ################################### # # #
# # # ######    Linear Filter Setup  ###### # # #
# # # ################################### # # #
dx = 0.0125
bias_initializer = torch.tensor([0.0])

pd1 = torch.tensor(
    [[2 / 26, 3 / 26, 2 / 26], [3 / 26, 6 / 26, 3 / 26], [2 / 26, 3 / 26, 2 / 26]]
)
pd2 = torch.tensor(
    [[3 / 26, 6 / 26, 3 / 26], [6 / 26, -88 / 26, 6 / 26], [3 / 26, 6 / 26, 3 / 26]]
)
pd3 = torch.tensor(
    [[2 / 26, 3 / 26, 2 / 26], [3 / 26, 6 / 26, 3 / 26], [2 / 26, 3 / 26, 2 / 26]]
)

w1 = torch.zeros([1, 1, 3, 3, 3])
wA = torch.zeros([1, 1, 3, 3, 3])
w1[0, 0, 0, :, :] = pd1 / dx**2
w1[0, 0, 1, :, :] = pd2 / dx**2
w1[0, 0, 2, :, :] = pd3 / dx**2
wA[0, 0, 0, :, :] = -pd1 / dx**2
wA[0, 0, 1, :, :] = -pd2 / dx**2
wA[0, 0, 2, :, :] = -pd3 / dx**2

p_div_x1 = torch.tensor(
    [[-0.014, 0.0, 0.014], [-0.056, 0.0, 0.056], [-0.014, 0.0, 0.014]]
)
p_div_x2 = torch.tensor(
    [[-0.056, 0.0, 0.056], [-0.22, 0.0, 0.22], [-0.056, 0.0, 0.056]]
)
p_div_x3 = torch.tensor(
    [[-0.014, 0.0, 0.014], [-0.056, 0.0, 0.056], [-0.014, 0.0, 0.014]]
)
p_div_y1 = torch.tensor(
    [[0.014, 0.056, 0.014], [0.0, 0.0, 0.0], [-0.014, -0.056, -0.014]]
)
p_div_y2 = torch.tensor(
    [[0.056, 0.22, 0.056], [0.0, 0.0, 0.0], [-0.056, -0.22, -0.056]]
)
p_div_y3 = torch.tensor(
    [[0.014, 0.056, 0.014], [0.0, 0.0, 0.0], [-0.014, -0.056, -0.014]]
)
p_div_z1 = torch.tensor(
    [[0.014, 0.056, 0.014], [0.056, 0.22, 0.056], [0.014, 0.056, 0.014]]
)
p_div_z2 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
p_div_z3 = torch.tensor(
    [[-0.014, -0.056, -0.014], [-0.056, -0.22, -0.056], [-0.014, -0.056, -0.014]]
)

w2 = torch.zeros([1, 1, 3, 3, 3])
w3 = torch.zeros([1, 1, 3, 3, 3])
w4 = torch.zeros([1, 1, 3, 3, 3])
w2[0, 0, 0, :, :] = -p_div_x1 / dx * 0.5
w2[0, 0, 1, :, :] = -p_div_x2 / dx * 0.5
w2[0, 0, 2, :, :] = -p_div_x3 / dx * 0.5
w3[0, 0, 0, :, :] = -p_div_y1 / dx * 0.5
w3[0, 0, 1, :, :] = -p_div_y2 / dx * 0.5
w3[0, 0, 2, :, :] = -p_div_y3 / dx * 0.5
w4[0, 0, 0, :, :] = -p_div_z1 / dx * 0.5
w4[0, 0, 1, :, :] = -p_div_z2 / dx * 0.5
w4[0, 0, 2, :, :] = -p_div_z3 / dx * 0.5

w_res = torch.zeros([1, 1, 2, 2, 2])
w_res[0, 0, :, :, :] = 0.125
diag = wA[0, 0, 1, 1, 1].item()


class AI4Urban(nn.Module):
    def __init__(self):
        super(AI4Urban, self).__init__()
        self.xadv = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=0)
        self.yadv = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=0)
        self.zadv = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=0)
        self.diff = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=0)

        self.A = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=0)
        self.res = nn.Conv3d(1, 1, kernel_size=2, stride=2, padding=0)
        self.prol = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
        )

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

    def solid_body(self, values_u, values_v, values_w, sigma, dt):
        values_u = values_u / (1 + dt * sigma)
        values_v = values_v / (1 + dt * sigma)
        values_w = values_w / (1 + dt * sigma)
        return values_u, values_v, values_w

    def F_cycle_MG(
        self,
        topo,
        local_rank,
        values_uu,
        values_vv,
        values_ww,
        values_p,
        values_pp,
        iteration,
        diag,
        dt,
        nlevel,
    ):
        b = -(self.xadv(values_uu) + self.yadv(values_vv) + self.zadv(values_ww)) / dt

        for MG in range(iteration):
            w = torch.zeros((1, 1, 1, 1, 1), device=f"cuda:{local_rank}")
            r = self.A(values_pp) - b
            r_s = [r]

            for i in range(1, nlevel - 1):
                r = self.res(r)
                r_s.append(r)

            for i in reversed(range(1, nlevel - 1)):
                ww = apply_BC_cw(w, topo)
                ww = halo_exchange(ww, topo)
                w = w - self.A(ww) / diag + r_s[i] / diag
                w = self.prol(w)

            values_p = values_p - w
            values_p = values_p - self.A(values_pp) / diag + b / diag
            values_pp = apply_BC_p(values_p, values_pp, topo)
            values_pp = halo_exchange(values_pp, topo)

        return values_p, w, r

    def PG_vector(
        self,
        topo,
        values_uu,
        values_vv,
        values_ww,
        values_u,
        values_v,
        values_w,
        k1,
        k_uu,
        k_vv,
        k_ww,
        sigma,
    ):
        k_u = torch.ones_like(values_u).detach()
        k_v = torch.ones_like(values_v).detach()
        k_w = torch.ones_like(values_w).detach()

        k_uu = apply_BC_k(k_u, k_uu, topo)
        k_vv = apply_BC_k(k_v, k_vv, topo)
        k_ww = apply_BC_k(k_w, k_ww, topo)

        k_uu = halo_exchange(k_uu, topo)
        k_vv = halo_exchange(k_vv, topo)
        k_ww = halo_exchange(k_ww, topo)

        k_u = 0.5 * (
            k_u * self.diff(values_uu)
            + self.diff(values_uu * k_uu)
            - values_u * self.diff(k_uu)
        )
        k_v = 0.5 * (
            k_v * self.diff(values_vv)
            + self.diff(values_vv * k_vv)
            - values_v * self.diff(k_vv)
        )
        k_w = 0.5 * (
            k_w * self.diff(values_ww)
            + self.diff(values_ww * k_ww)
            - values_w * self.diff(k_ww)
        )
        return k_u, k_v, k_w

    def forward(
        self,
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
    ):
        if True:
            [values_u, values_v, values_w] = self.solid_body(
                values_u, values_v, values_w, sigma, dt
            )

        # 1. Apply BCs
        values_uu = apply_BC_u(values_u, values_uu, topo, ub)
        values_vv = apply_BC_v(values_v, values_vv, topo)
        values_ww = apply_BC_w(values_w, values_ww, topo)
        values_pp = apply_BC_p(values_p, values_pp, topo)

        # 2. Halo Exchange with Neighbors
        values_uu = halo_exchange(values_uu, topo)
        values_vv = halo_exchange(values_vv, topo)
        values_ww = halo_exchange(values_ww, topo)
        values_pp = halo_exchange(values_pp, topo)

        # First step
        [k_u, k_v, k_w] = self.PG_vector(
            topo,
            values_uu,
            values_vv,
            values_ww,
            values_u,
            values_v,
            values_w,
            k1,
            k_uu,
            k_vv,
            k_ww,
            sigma,
        )

        b_u = (
            values_u
            + 0.5
            * (
                Re * k_u * dt
                - values_u * self.xadv(values_uu) * dt
                - values_v * self.yadv(values_uu) * dt
                - values_w * self.zadv(values_uu) * dt
            )
            - self.xadv(values_pp) * dt
        )
        b_v = (
            values_v
            + 0.5
            * (
                Re * k_v * dt
                - values_u * self.xadv(values_vv) * dt
                - values_v * self.yadv(values_vv) * dt
                - values_w * self.zadv(values_vv) * dt
            )
            - self.yadv(values_pp) * dt
        )
        b_w = (
            values_w
            + 0.5
            * (
                Re * k_w * dt
                - values_u * self.xadv(values_ww) * dt
                - values_v * self.yadv(values_ww) * dt
                - values_w * self.zadv(values_ww) * dt
            )
            - self.zadv(values_pp) * dt
        )

        if True:
            [b_u, b_v, b_w] = self.solid_body(b_u, b_v, b_w, sigma, dt)

        b_uu = apply_BC_u(b_u, b_uu, topo, ub)
        b_vv = apply_BC_v(b_v, b_vv, topo)
        b_ww = apply_BC_w(b_w, b_ww, topo)

        b_uu = halo_exchange(b_uu, topo)
        b_vv = halo_exchange(b_vv, topo)
        b_ww = halo_exchange(b_ww, topo)

        # Second step
        [k_u, k_v, k_w] = self.PG_vector(
            topo, b_uu, b_vv, b_ww, b_u, b_v, b_w, k1, k_uu, k_vv, k_ww, sigma
        )
        values_u = (
            values_u
            + Re * k_u * dt
            - b_u * self.xadv(b_uu) * dt
            - b_v * self.yadv(b_uu) * dt
            - b_w * self.zadv(b_uu) * dt
            - self.xadv(values_pp) * dt
        )
        values_v = (
            values_v
            + Re * k_v * dt
            - b_u * self.xadv(b_vv) * dt
            - b_v * self.yadv(b_vv) * dt
            - b_w * self.zadv(b_vv) * dt
            - self.yadv(values_pp) * dt
        )
        values_w = (
            values_w
            + Re * k_w * dt
            - b_u * self.xadv(b_ww) * dt
            - b_v * self.yadv(b_ww) * dt
            - b_w * self.zadv(b_ww) * dt
            - self.zadv(values_pp) * dt
        )

        if True:
            [values_u, values_v, values_w] = self.solid_body(
                values_u, values_v, values_w, sigma, dt
            )

        # Pressure
        values_uu = apply_BC_u(values_u, values_uu, topo, ub)
        values_vv = apply_BC_v(values_v, values_vv, topo)
        values_ww = apply_BC_w(values_w, values_ww, topo)

        values_uu = halo_exchange(values_uu, topo)
        values_vv = halo_exchange(values_vv, topo)
        values_ww = halo_exchange(values_ww, topo)

        [values_p, w, r] = self.F_cycle_MG(
            topo,
            local_rank,
            values_uu,
            values_vv,
            values_ww,
            values_p,
            values_pp,
            iteration,
            diag,
            dt,
            nlevel,
        )

        values_pp = apply_BC_p(values_p, values_pp, topo)
        values_pp = halo_exchange(values_pp, topo)

        values_u = values_u - self.xadv(values_pp) * dt
        values_v = values_v - self.yadv(values_pp) * dt
        values_w = values_w - self.zadv(values_pp) * dt

        if True:
            [values_u, values_v, values_w] = self.solid_body(
                values_u, values_v, values_w, sigma, dt
            )
        return values_u, values_v, values_w, values_p, w, r
