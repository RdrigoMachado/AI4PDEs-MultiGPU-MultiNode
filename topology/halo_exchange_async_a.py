"""Strategy A — isend/irecv per-axis (non-blocking).

Mantém a estrutura de 2 sub-fases checkerboard por eixo da versão bloqueante,
mas dentro de cada sub-fase disparam-se send e recv simultaneamente via
`dist.isend` + `dist.irecv` para tentar aproveitar o full-duplex do link
(NVLink intra-nó, InfiniBand inter-nó).

Speedup-alvo por sub-fase: até 2x (T_block ~ 2*(alpha+B/beta) -> T_async ~ alpha+B/beta).
"""

import torch
import torch.distributed as dist


def halo_exchange_async_a(tensor, topo):
    if topo.world_size == 1:
        return tensor

    # ======================= AXIS X =======================
    if topo.PX > 1:
        is_even_x = topo.px % 2 == 0
        # Sub-fase 1: par px <-> par+1 (right side of par; left side of ímpar)
        if is_even_x and topo.neighbors["right"] != -1:
            send_buf = tensor[..., -2].contiguous()
            recv_buf = torch.empty_like(tensor[..., -1])
            req_s = dist.isend(send_buf, dst=topo.neighbors["right"])
            req_r = dist.irecv(recv_buf, src=topo.neighbors["right"])
            req_s.wait()
            req_r.wait()
            tensor[..., -1] = recv_buf
        elif (not is_even_x) and topo.neighbors["left"] != -1:
            send_buf = tensor[..., 1].contiguous()
            recv_buf = torch.empty_like(tensor[..., 0])
            req_s = dist.isend(send_buf, dst=topo.neighbors["left"])
            req_r = dist.irecv(recv_buf, src=topo.neighbors["left"])
            req_s.wait()
            req_r.wait()
            tensor[..., 0] = recv_buf

        # Sub-fase 2: ímpar px <-> ímpar+1 (right side of ímpar; left side of par)
        if (not is_even_x) and topo.neighbors["right"] != -1:
            send_buf = tensor[..., -2].contiguous()
            recv_buf = torch.empty_like(tensor[..., -1])
            req_s = dist.isend(send_buf, dst=topo.neighbors["right"])
            req_r = dist.irecv(recv_buf, src=topo.neighbors["right"])
            req_s.wait()
            req_r.wait()
            tensor[..., -1] = recv_buf
        elif is_even_x and topo.neighbors["left"] != -1:
            send_buf = tensor[..., 1].contiguous()
            recv_buf = torch.empty_like(tensor[..., 0])
            req_s = dist.isend(send_buf, dst=topo.neighbors["left"])
            req_r = dist.irecv(recv_buf, src=topo.neighbors["left"])
            req_s.wait()
            req_r.wait()
            tensor[..., 0] = recv_buf

    # ======================= AXIS Y =======================
    if topo.PY > 1:
        is_even_y = topo.py % 2 == 0
        if is_even_y and topo.neighbors["bottom"] != -1:
            send_buf = tensor[..., -2, :].contiguous()
            recv_buf = torch.empty_like(tensor[..., -1, :])
            req_s = dist.isend(send_buf, dst=topo.neighbors["bottom"])
            req_r = dist.irecv(recv_buf, src=topo.neighbors["bottom"])
            req_s.wait()
            req_r.wait()
            tensor[..., -1, :] = recv_buf
        elif (not is_even_y) and topo.neighbors["top"] != -1:
            send_buf = tensor[..., 1, :].contiguous()
            recv_buf = torch.empty_like(tensor[..., 0, :])
            req_s = dist.isend(send_buf, dst=topo.neighbors["top"])
            req_r = dist.irecv(recv_buf, src=topo.neighbors["top"])
            req_s.wait()
            req_r.wait()
            tensor[..., 0, :] = recv_buf

        if (not is_even_y) and topo.neighbors["bottom"] != -1:
            send_buf = tensor[..., -2, :].contiguous()
            recv_buf = torch.empty_like(tensor[..., -1, :])
            req_s = dist.isend(send_buf, dst=topo.neighbors["bottom"])
            req_r = dist.irecv(recv_buf, src=topo.neighbors["bottom"])
            req_s.wait()
            req_r.wait()
            tensor[..., -1, :] = recv_buf
        elif is_even_y and topo.neighbors["top"] != -1:
            send_buf = tensor[..., 1, :].contiguous()
            recv_buf = torch.empty_like(tensor[..., 0, :])
            req_s = dist.isend(send_buf, dst=topo.neighbors["top"])
            req_r = dist.irecv(recv_buf, src=topo.neighbors["top"])
            req_s.wait()
            req_r.wait()
            tensor[..., 0, :] = recv_buf

    # ======================= AXIS Z =======================
    if topo.PZ > 1:
        is_even_z = topo.pz % 2 == 0
        if is_even_z and topo.neighbors["front"] != -1:
            send_buf = tensor[..., -2, :, :].contiguous()
            recv_buf = torch.empty_like(tensor[..., -1, :, :])
            req_s = dist.isend(send_buf, dst=topo.neighbors["front"])
            req_r = dist.irecv(recv_buf, src=topo.neighbors["front"])
            req_s.wait()
            req_r.wait()
            tensor[..., -1, :, :] = recv_buf
        elif (not is_even_z) and topo.neighbors["back"] != -1:
            send_buf = tensor[..., 1, :, :].contiguous()
            recv_buf = torch.empty_like(tensor[..., 0, :, :])
            req_s = dist.isend(send_buf, dst=topo.neighbors["back"])
            req_r = dist.irecv(recv_buf, src=topo.neighbors["back"])
            req_s.wait()
            req_r.wait()
            tensor[..., 0, :, :] = recv_buf

        if (not is_even_z) and topo.neighbors["front"] != -1:
            send_buf = tensor[..., -2, :, :].contiguous()
            recv_buf = torch.empty_like(tensor[..., -1, :, :])
            req_s = dist.isend(send_buf, dst=topo.neighbors["front"])
            req_r = dist.irecv(recv_buf, src=topo.neighbors["front"])
            req_s.wait()
            req_r.wait()
            tensor[..., -1, :, :] = recv_buf
        elif is_even_z and topo.neighbors["back"] != -1:
            send_buf = tensor[..., 1, :, :].contiguous()
            recv_buf = torch.empty_like(tensor[..., 0, :, :])
            req_s = dist.isend(send_buf, dst=topo.neighbors["back"])
            req_r = dist.irecv(recv_buf, src=topo.neighbors["back"])
            req_s.wait()
            req_r.wait()
            tensor[..., 0, :, :] = recv_buf

    return tensor
