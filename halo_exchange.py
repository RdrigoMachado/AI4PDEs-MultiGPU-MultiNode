from pandas.core.arrays.base import rank
import torch
import torch.distributed as distributed
from torch.multiprocessing import Process, spawn
import os

# Halos em Z (profundidade)
HALO_FRONT = 0          # camada mais rasa do subdomínio (halo frontal)
HALO_BACK = -1          # camada mais profunda do subdomínio (halo posterior)
INTERIOR_FRONT = 1      # primeira camada de dados internos
INTERIOR_BACK = -2      # última camada de dados internos


def init_process(rank, size, backend='nccl'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    distributed.init_process_group(backend, rank=rank, world_size=size)

def split_and_distribute_tensors(tensor_list):
    comm_size = distributed.get_world_size()

    for tensor in tensor_list:
        tensor = tensor.squeeze(dim=0).squeeze(dim=0)
        z, y, x = tensor.size()
        z_per_rank = z // comm_size

        for rank in range(1, comm_size):
            z_start = rank * z_per_rank
            z_end   = z_start + z_per_rank if rank != comm_size - 1 else z
            distributed.send(tensor=tensor[z_start:z_end, :, :].contiguous(), dst=rank)


def halo_exchange_Z(input_data):
    """Troca de halos ao longo do eixo Z entre processos MPI-like (PyTorch Distributed)."""
    rank = distributed.get_rank()
    comm_size = distributed.get_world_size()

    # Remove dimensões extras (batch, channel)
    input_data = input_data.squeeze(dim=0).squeeze(dim=0)
    ny, nx = input_data.shape[1], input_data.shape[2]

    if rank == 0:
        recv_data_next = torch.empty((ny, nx), dtype=input_data.dtype, device=input_data.device)

        # Envia última camada interna para o próximo rank
        request_send_next = distributed.isend(
            tensor=input_data[INTERIOR_BACK, :, :].contiguous(),
            dst=rank + 1
        )
        request_recv_next = distributed.irecv(
            tensor=recv_data_next,
            src=rank + 1
        )

        request_send_next.wait()
        request_recv_next.wait()

        input_data[HALO_BACK, :, :] = recv_data_next

    elif rank == comm_size - 1:
        recv_data_prev = torch.empty((ny, nx), dtype=input_data.dtype, device=input_data.device)

        request_send_prev = distributed.isend(
            tensor=input_data[INTERIOR_FRONT, :, :].contiguous(),
            dst=rank - 1
        )
        request_recv_prev = distributed.irecv(
            tensor=recv_data_prev,
            src=rank - 1
        )

        request_send_prev.wait()
        request_recv_prev.wait()

        input_data[HALO_FRONT, :, :] = recv_data_prev

    else:
        recv_data_next = torch.empty((ny, nx), dtype=input_data.dtype, device=input_data.device)
        recv_data_prev = torch.empty((ny, nx), dtype=input_data.dtype, device=input_data.device)

        request_send_next = distributed.isend(
            tensor=input_data[INTERIOR_BACK, :, :].contiguous(),
            dst=rank + 1
        )
        request_recv_next = distributed.irecv(
            tensor=recv_data_next,
            src=rank + 1
        )

        request_send_prev = distributed.isend(
            tensor=input_data[INTERIOR_FRONT, :, :].contiguous(),
            dst=rank - 1
        )
        request_recv_prev = distributed.irecv(
            tensor=recv_data_prev,
            src=rank - 1
        )

        request_send_next.wait()
        request_recv_next.wait()
        request_send_prev.wait()
        request_recv_prev.wait()

        input_data[HALO_FRONT, :, :] = recv_data_prev
        input_data[HALO_BACK, :, :] = recv_data_next

    return input_data.unsqueeze(dim=0).unsqueeze(dim=0)

def gather_all_data(local_tensor):
    rank = distributed.get_rank()
    comm_size = distributed.get_world_size()

    local_tensor_cpu = local_tensor.cpu()

    if rank == 0:
        gathered_list = [torch.empty_like(local_tensor_cpu) for _ in range(world_size)]
    else:
        gathered_list = None

    distributed.gather(local_tensor_cpu, gather_list=gathered_list, dst=0)

    if rank == 0:
        return torch.cat(gathered_list, dim=2)
    else:
        return None
