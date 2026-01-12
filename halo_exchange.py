import torch
import torch.distributed as dist
import os

# Indexes
X_MIN, X_MAX = 0, -1
Y_MIN, Y_MAX = 0, -1
Z_MIN, Z_MAX = 0, -1

def init_process(backend='nccl'):
    """
    Initialize NCCL backend for GPU communication
    """
    try:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    except KeyError:
        print("Erro: Torchrun variables (RANK, WORLD_SIZE) not found.")
        exit(1)

    device_id = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device_id)
    torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        print(f"[Rank {rank}] Initialize backend {backend}...", flush=True)
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            device_id=device_id
        )

    dist.barrier()
    return rank, world_size, local_rank

def gather_all_data(local_tensor):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if(world_size == 1):
        return local_tensor.cpu()

    tensor_gpu = local_tensor.contiguous()

    if rank == 0:
        gathered_list = [torch.empty_like(tensor_gpu) for _ in range(world_size)]
    else:
        gathered_list = None

    dist.gather(tensor_gpu, gather_list=gathered_list, dst=0)

    if rank == 0:
        num_z_slices = world_size // 4
        z_slices = []
        for i in range(num_z_slices):
            base = i * 4
            tl = gathered_list[base]     # Rank 0, 4...
            tr = gathered_list[base + 1] # Rank 1, 5...
            bl = gathered_list[base + 2] # Rank 2, 6...
            br = gathered_list[base + 3] # Rank 3, 7...

            row_top = torch.cat([tl, tr], dim=4)
            row_bot = torch.cat([bl, br], dim=4)

            z_slice = torch.cat([row_top, row_bot], dim=3)

            z_slices.append(z_slice)

        full_tensor = torch.cat(z_slices, dim=2)
        return full_tensor.cpu()

    return None

def halo_exchange(tensor, neighbors):
    """
    Asynchronous halo exchange for a 5D tensor [batch, channel, z, y, x] with padding.
    neighbors: Dict com chaves 'left', 'right', 'top', 'bottom', 'back', 'front'.
    """

    # List of asynchronous requests
    reqs = []

    # ==========================================
    # 1. X-Axis (Left / Right)
    # ==========================================
    if neighbors['left'] != -1:
        reqs.append(dist.irecv(tensor[..., X_MIN], src=neighbors['left']))
        send_buff_l = tensor[..., X_MIN + 1].contiguous()
        reqs.append(dist.isend(send_buff_l, dst=neighbors['left']))

    if neighbors['right'] != -1:
        reqs.append(dist.irecv(tensor[..., X_MAX], src=neighbors['right']))
        send_buff_r = tensor[..., X_MAX - 1].contiguous()
        reqs.append(dist.isend(send_buff_r, dst=neighbors['right']))

    # ==========================================
    # 2. Y-Axis (Top / Bottom)
    # ==========================================
    if neighbors['top'] != -1:
        reqs.append(dist.irecv(tensor[..., Y_MIN, :], src=neighbors['top']))
        send_buff_t = tensor[..., Y_MIN + 1, :].contiguous()
        reqs.append(dist.isend(send_buff_t, dst=neighbors['top']))

    if neighbors['bottom'] != -1:
        reqs.append(dist.irecv(tensor[..., Y_MAX, :], src=neighbors['bottom']))
        send_buff_b = tensor[..., Y_MAX - 1, :].contiguous()
        reqs.append(dist.isend(send_buff_b, dst=neighbors['bottom']))

    # ==========================================
    # 3. Z-Axis - Inter nodes
    # ==========================================
    if neighbors['back'] != -1:
        reqs.append(dist.irecv(tensor[..., Z_MIN, :, :], src=neighbors['back']))
        send_buff_zm = tensor[..., Z_MIN + 1, :, :].contiguous()
        reqs.append(dist.isend(send_buff_zm, dst=neighbors['back']))

    if neighbors['front'] != -1:
        reqs.append(dist.irecv(tensor[..., Z_MAX, :, :], src=neighbors['front']))
        send_buff_zmx = tensor[..., Z_MAX - 1, :, :].contiguous()
        reqs.append(dist.isend(send_buff_zmx, dst=neighbors['front']))

    # Wait for all communications to complete
    for req in reqs:
        req.wait()

    return tensor
