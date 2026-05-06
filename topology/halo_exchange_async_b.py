"""Strategy B — batch_isend_irecv (single batched non-blocking call).

Junta TODAS as operações de halo (X, Y, Z, ambos sentidos) em uma única
chamada `dist.batch_isend_irecv`. A API canônica do PyTorch para non-blocking
P2P em NCCL: o backend agrupa as ops em uma única chamada coletiva, o que
permite execução concorrente de fato (diferente de isend/irecv soltos, que
podem serializar dependendo da versão do NCCL).

Sem fases checkerboard — não são necessárias quando a comunicação é
não-bloqueante (não há risco de deadlock por send-then-recv simétrico).
"""

import torch
import torch.distributed as dist


def halo_exchange_async_b(tensor, topo):
    if topo.world_size == 1:
        return tensor

    ops = []
    # pending guarda (tag, recv_buf, send_buf). send_buf é mantido vivo até
    # depois do wait() — caso contrário, o GC poderia liberar o buffer
    # contíguo enquanto a operação assíncrona ainda está em voo.
    pending = []

    # ======================= AXIS X =======================
    if topo.PX > 1:
        if topo.neighbors["right"] != -1:
            send_buf = tensor[..., -2].contiguous()
            recv_buf = torch.empty_like(tensor[..., -1])
            ops.append(dist.P2POp(dist.isend, send_buf, topo.neighbors["right"]))
            ops.append(dist.P2POp(dist.irecv, recv_buf, topo.neighbors["right"]))
            pending.append(("X_max", recv_buf, send_buf))
        if topo.neighbors["left"] != -1:
            send_buf = tensor[..., 1].contiguous()
            recv_buf = torch.empty_like(tensor[..., 0])
            ops.append(dist.P2POp(dist.isend, send_buf, topo.neighbors["left"]))
            ops.append(dist.P2POp(dist.irecv, recv_buf, topo.neighbors["left"]))
            pending.append(("X_min", recv_buf, send_buf))

    # ======================= AXIS Y =======================
    if topo.PY > 1:
        if topo.neighbors["bottom"] != -1:
            send_buf = tensor[..., -2, :].contiguous()
            recv_buf = torch.empty_like(tensor[..., -1, :])
            ops.append(dist.P2POp(dist.isend, send_buf, topo.neighbors["bottom"]))
            ops.append(dist.P2POp(dist.irecv, recv_buf, topo.neighbors["bottom"]))
            pending.append(("Y_max", recv_buf, send_buf))
        if topo.neighbors["top"] != -1:
            send_buf = tensor[..., 1, :].contiguous()
            recv_buf = torch.empty_like(tensor[..., 0, :])
            ops.append(dist.P2POp(dist.isend, send_buf, topo.neighbors["top"]))
            ops.append(dist.P2POp(dist.irecv, recv_buf, topo.neighbors["top"]))
            pending.append(("Y_min", recv_buf, send_buf))

    # ======================= AXIS Z =======================
    if topo.PZ > 1:
        if topo.neighbors["front"] != -1:
            send_buf = tensor[..., -2, :, :].contiguous()
            recv_buf = torch.empty_like(tensor[..., -1, :, :])
            ops.append(dist.P2POp(dist.isend, send_buf, topo.neighbors["front"]))
            ops.append(dist.P2POp(dist.irecv, recv_buf, topo.neighbors["front"]))
            pending.append(("Z_max", recv_buf, send_buf))
        if topo.neighbors["back"] != -1:
            send_buf = tensor[..., 1, :, :].contiguous()
            recv_buf = torch.empty_like(tensor[..., 0, :, :])
            ops.append(dist.P2POp(dist.isend, send_buf, topo.neighbors["back"]))
            ops.append(dist.P2POp(dist.irecv, recv_buf, topo.neighbors["back"]))
            pending.append(("Z_min", recv_buf, send_buf))

    if ops:
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

    for tag, recv_buf, _send_buf_keepalive in pending:
        if tag == "X_max":
            tensor[..., -1] = recv_buf
        elif tag == "X_min":
            tensor[..., 0] = recv_buf
        elif tag == "Y_max":
            tensor[..., -1, :] = recv_buf
        elif tag == "Y_min":
            tensor[..., 0, :] = recv_buf
        elif tag == "Z_max":
            tensor[..., -1, :, :] = recv_buf
        elif tag == "Z_min":
            tensor[..., 0, :, :] = recv_buf

    return tensor
