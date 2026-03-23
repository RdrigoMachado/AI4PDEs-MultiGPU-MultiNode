import os

import torch
import torch.distributed as dist


class Topology:
    def __init__(self, decomp_type, rank, world_size, nx, ny, nz):
        self.decomp_type = decomp_type
        self.rank = rank
        self.world_size = world_size

        self.PZ, self.PY, self.PX = 1, 1, 1

        if world_size == 1:
            pass
        elif decomp_type == "1d-z":
            self.PZ = world_size
        elif decomp_type == "1d-y":
            self.PY = world_size
        elif decomp_type == "1d-x":
            self.PX = world_size
        elif decomp_type == "3d":
            if world_size % 4 != 0:
                raise ValueError(
                    "3D decomposition requires world_size to be a multiple of 4."
                )
            self.PX = 2
            self.PY = 2
            self.PZ = max(1, world_size // 4)
        else:
            raise ValueError(f"Unknown decomposition: {decomp_type}")

        if nx % self.PX != 0 or ny % self.PY != 0 or nz % self.PZ != 0:
            raise ValueError(
                f"Grid {nx}x{ny}x{nz} not divisible by process grid {self.PX}x{self.PY}x{self.PZ}"
            )

        # Rank's grid coordinates
        self.pz = rank // (self.PY * self.PX)
        rem = rank % (self.PY * self.PX)
        self.py = rem // self.PX
        self.px = rem % self.PX

        # Local size
        self.local_nx = nx // self.PX
        self.local_ny = ny // self.PY
        self.local_nz = nz // self.PZ

        # Borders
        self.is_xmin = self.px == 0
        self.is_xmax = self.px == self.PX - 1
        self.is_ymin = self.py == 0
        self.is_ymax = self.py == self.PY - 1
        self.is_zmin = self.pz == 0
        self.is_zmax = self.pz == self.PZ - 1

        # Neighbors
        self.neighbors = {
            "left": self.get_rank(self.pz, self.py, self.px - 1),
            "right": self.get_rank(self.pz, self.py, self.px + 1),
            "top": self.get_rank(self.pz, self.py - 1, self.px),
            "bottom": self.get_rank(self.pz, self.py + 1, self.px),
            "back": self.get_rank(self.pz - 1, self.py, self.px),
            "front": self.get_rank(self.pz + 1, self.py, self.px),
        }

    def get_rank(self, z, y, x):
        if 0 <= z < self.PZ and 0 <= y < self.PY and 0 <= x < self.PX:
            return z * (self.PY * self.PX) + y * self.PX + x
        return -1


def init_process(backend="nccl"):
    try:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    except KeyError:
        print("Error")
        exit(1)

    device_id = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device_id)

    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend, rank=rank, world_size=world_size, device_id=device_id
        )

    dist.barrier()
    return rank, world_size, local_rank


def gather_all_data(local_tensor, topo):
    if topo.world_size == 1:
        return local_tensor.cpu()

    tensor_gpu = local_tensor.contiguous()
    gathered_list = (
        [torch.empty_like(tensor_gpu) for _ in range(topo.world_size)]
        if topo.rank == 0
        else None
    )

    dist.gather(tensor_gpu, gather_list=gathered_list, dst=0)

    if topo.rank == 0:
        slices_z = []
        for z in range(topo.PZ):
            slices_y = []
            for y in range(topo.PY):
                slices_x = []
                for x in range(topo.PX):
                    idx = z * (topo.PY * topo.PX) + y * topo.PX + x
                    slices_x.append(gathered_list[idx])
                row_x = torch.cat(slices_x, dim=4)
                slices_y.append(row_x)
            plane_y = torch.cat(slices_y, dim=3)
            slices_z.append(plane_y)
        return torch.cat(slices_z, dim=2).cpu()
    return None


def halo_exchange(tensor, topo):
    if topo.world_size == 1:
        return tensor

    # ======================= AXIS X =======================
    if topo.PX > 1:
        is_even_x = topo.px % 2 == 0
        # Passo 1
        if is_even_x and topo.neighbors["right"] != -1:
            dist.send(tensor[..., -2].contiguous(), dst=topo.neighbors["right"])
            rb = torch.empty_like(tensor[..., -1]).contiguous()
            dist.recv(rb, src=topo.neighbors["right"])
            tensor[..., -1] = rb
        elif not is_even_x and topo.neighbors["left"] != -1:
            rb = torch.empty_like(tensor[..., 0]).contiguous()
            dist.recv(rb, src=topo.neighbors["left"])
            tensor[..., 0] = rb
            dist.send(tensor[..., 1].contiguous(), dst=topo.neighbors["left"])
        # Passo 2
        if not is_even_x and topo.neighbors["right"] != -1:
            dist.send(tensor[..., -2].contiguous(), dst=topo.neighbors["right"])
            rb = torch.empty_like(tensor[..., -1]).contiguous()
            dist.recv(rb, src=topo.neighbors["right"])
            tensor[..., -1] = rb
        elif is_even_x and topo.neighbors["left"] != -1:
            rb = torch.empty_like(tensor[..., 0]).contiguous()
            dist.recv(rb, src=topo.neighbors["left"])
            tensor[..., 0] = rb
            dist.send(tensor[..., 1].contiguous(), dst=topo.neighbors["left"])

    # ======================= AXIS Y =======================
    if topo.PY > 1:
        is_even_y = topo.py % 2 == 0
        # Passo 1
        if is_even_y and topo.neighbors["bottom"] != -1:
            dist.send(tensor[..., -2, :].contiguous(), dst=topo.neighbors["bottom"])
            rb = torch.empty_like(tensor[..., -1, :]).contiguous()
            dist.recv(rb, src=topo.neighbors["bottom"])
            tensor[..., -1, :] = rb
        elif not is_even_y and topo.neighbors["top"] != -1:
            rb = torch.empty_like(tensor[..., 0, :]).contiguous()
            dist.recv(rb, src=topo.neighbors["top"])
            tensor[..., 0, :] = rb
            dist.send(tensor[..., 1, :].contiguous(), dst=topo.neighbors["top"])
        # Passo 2
        if not is_even_y and topo.neighbors["bottom"] != -1:
            dist.send(tensor[..., -2, :].contiguous(), dst=topo.neighbors["bottom"])
            rb = torch.empty_like(tensor[..., -1, :]).contiguous()
            dist.recv(rb, src=topo.neighbors["bottom"])
            tensor[..., -1, :] = rb
        elif is_even_y and topo.neighbors["top"] != -1:
            rb = torch.empty_like(tensor[..., 0, :]).contiguous()
            dist.recv(rb, src=topo.neighbors["top"])
            tensor[..., 0, :] = rb
            dist.send(tensor[..., 1, :].contiguous(), dst=topo.neighbors["top"])

    # ======================= AXIS Z =======================
    if topo.PZ > 1:
        is_even_z = topo.pz % 2 == 0
        # Passo 1
        if is_even_z and topo.neighbors["front"] != -1:
            dist.send(tensor[..., -2, :, :].contiguous(), dst=topo.neighbors["front"])
            rb = torch.empty_like(tensor[..., -1, :, :]).contiguous()
            dist.recv(rb, src=topo.neighbors["front"])
            tensor[..., -1, :, :] = rb
        elif not is_even_z and topo.neighbors["back"] != -1:
            rb = torch.empty_like(tensor[..., 0, :, :]).contiguous()
            dist.recv(rb, src=topo.neighbors["back"])
            tensor[..., 0, :, :] = rb
            dist.send(tensor[..., 1, :, :].contiguous(), dst=topo.neighbors["back"])
        # Passo 2
        if not is_even_z and topo.neighbors["front"] != -1:
            dist.send(tensor[..., -2, :, :].contiguous(), dst=topo.neighbors["front"])
            rb = torch.empty_like(tensor[..., -1, :, :]).contiguous()
            dist.recv(rb, src=topo.neighbors["front"])
            tensor[..., -1, :, :] = rb
        elif is_even_z and topo.neighbors["back"] != -1:
            rb = torch.empty_like(tensor[..., 0, :, :]).contiguous()
            dist.recv(rb, src=topo.neighbors["back"])
            tensor[..., 0, :, :] = rb
            dist.send(tensor[..., 1, :, :].contiguous(), dst=topo.neighbors["back"])

    return tensor
