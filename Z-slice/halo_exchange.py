import torch
import torch.distributed as dist
import os
import time

# Constantes de Fatiamento
HALO_FRONT = 0
HALO_BACK = -1
INTERIOR_FRONT = 1
INTERIOR_BACK = -2

# Adicione essa variável global no topo do arquivo
total_bytes_moved = 0
comm_time = 0

DEBUG_COMM = False

def init_process(backend='nccl'):
    """
    Inicializa o backend NCCL usando variáveis do Torchrun.
    NCCL é otimizado para GPU e não precisa de fallback para CPU.
    """
    try:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    except KeyError:
        print("Erro: Variáveis do Torchrun (RANK, WORLD_SIZE) não encontradas.")
        exit(1)

    device_id = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device_id)
    # Configura GPU Local (CRUCIAL para NCCL)
    torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        print(f"[Rank {rank}] Inicializando backend {backend}...", flush=True)
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            device_id=device_id
        )

    # Barreira inicial
    dist.barrier()
    return rank, world_size, local_rank

def distribute_tensor(global_tensor_on_rank0, local_shape):
    rank = dist.get_rank()

    # Define dispositivo
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    # Prepara tensor local na GPU
    local_tensor = torch.empty(local_shape, device=device)

    # Prepara Scatter List (apenas no Rank 0)
    scatter_list = None
    if rank == 0:
        if global_tensor_on_rank0 is None: raise ValueError("Rank 0 precisa do tensor")

        # Garante que a fonte está na GPU para o NCCL ser rápido
        tensor_src = global_tensor_on_rank0.to(device)

        local_nz = local_shape[2]
        scatter_list = list(torch.split(tensor_src, local_nz, dim=2))

    # Executa o Scatter (NCCL lida direto na GPU)
    dist.scatter(local_tensor, scatter_list, src=0)

    return local_tensor

# def halo_exchange_Z(input_data):
#     global comm_time
#     torch.cuda.synchronize()
#     start = time.time()
#     """
#     Troca de Halos Bloqueante com Estratégia Par-Ímpar (Odd-Even).
#     Garante ausência de deadlocks em N processos.
#     """
#     rank = dist.get_rank()
#     world_size = dist.get_world_size()

#     if(world_size == 1):
#         return input_data

#     # Remove dimensões extras
#     squeezed_data = input_data.squeeze(0).squeeze(0)
#     ny, nx = squeezed_data.shape[1], squeezed_data.shape[2]

#     # Identifica vizinhos
#     left_peer = rank - 1 if rank > 0 else None
#     right_peer = rank + 1 if rank < world_size - 1 else None

#     # Buffers temporários na GPU
#     # Usamos buffers dedicados para evitar condições de corrida na memória
#     recv_buff_left = torch.empty((ny, nx), device=input_data.device)
#     recv_buff_right = torch.empty((ny, nx), device=input_data.device)

#     # --- FASE 1: COMUNICAÇÃO NAS FRONTEIRAS PARES (0-1, 2-3, etc) ---
#     # Pares falam com a Direita (Ímpares)

#     if rank % 2 == 0:
#         # Sou PAR. Se tenho vizinho à direita, troco com ele.
#         if right_peer is not None:
#             # 1. Envio para direita (meu fim -> inicio dele)
#             send_tensor = squeezed_data[INTERIOR_BACK].contiguous()
#             dist.send(send_tensor, dst=right_peer)

#             # 2. Recebo da direita (meu halo fim <- interior dele)
#             dist.recv(recv_buff_right, src=right_peer)
#             squeezed_data[HALO_BACK] = recv_buff_right

#     else:
#         # Sou ÍMPAR. Se tenho vizinho à esquerda (que é par), troco com ele.
#         if left_peer is not None:
#             # 1. Recebo da esquerda (meu halo inicio <- interior dele)
#             # Nota: A ordem aqui é RECV depois SEND, para casar com o SEND depois RECV do par.
#             dist.recv(recv_buff_left, src=left_peer)
#             squeezed_data[HALO_FRONT] = recv_buff_left

#             # 2. Envio para esquerda (meu inicio -> fim dele)
#             send_tensor = squeezed_data[INTERIOR_FRONT].contiguous()
#             dist.send(send_tensor, dst=left_peer)


#     # --- FASE 2: COMUNICAÇÃO NAS FRONTEIRAS ÍMPARES (1-2, 3-4, etc) ---
#     # Ímpares falam com a Direita (Pares)

#     if rank % 2 == 1:
#         # Sou ÍMPAR. Agora falo com quem está à minha direita (que é Par)
#         if right_peer is not None:
#             # 1. Envio para direita
#             send_tensor = squeezed_data[INTERIOR_BACK].contiguous()
#             dist.send(send_tensor, dst=right_peer)

#             # 2. Recebo da direita
#             dist.recv(recv_buff_right, src=right_peer)
#             squeezed_data[HALO_BACK] = recv_buff_right

#     else:
#         # Sou PAR. Agora falo com quem está à minha esquerda (que é Ímpar)
#         if left_peer is not None:
#             # 1. Recebo da esquerda
#             dist.recv(recv_buff_left, src=left_peer)
#             squeezed_data[HALO_FRONT] = recv_buff_left

#             # 2. Envio para esquerda
#             send_tensor = squeezed_data[INTERIOR_FRONT].contiguous()
#             dist.send(send_tensor, dst=left_peer)

#     torch.cuda.synchronize()
#     end = time.time()
#     if(rank == 0):
#         # print(f'\nHalo exchange: {end-start:.2f}s')
#         comm_time += end-start
#         print(f'\nComm Acumulado: {comm_time:.2f}s')

#     return squeezed_data.unsqueeze(0).unsqueeze(0)

def halo_exchange_Z(input_data):
    global comm_time, total_bytes_moved

    # 1. Sincroniza GPU antes do timer
    torch.cuda.synchronize()
    start = time.time()

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if world_size == 1:
        return input_data

    # Remove dimensões extras e pega dimensões atuais (variam no Multigrid)
    squeezed_data = input_data.squeeze(0).squeeze(0)
    ny, nx = squeezed_data.shape[1], squeezed_data.shape[2]

    # Identifica vizinhos
    left_peer = rank - 1 if rank > 0 else None
    right_peer = rank + 1 if rank < world_size - 1 else None

    # --- CÁLCULO DE DADOS (BYTES) ---
    # float32 = 4 bytes.
    # Cada vizinho implica em 1 Send e 1 Recv (Bidirecional)
    # Payload = Área da face * 4 bytes
    face_size_bytes = ny * nx * input_data.element_size()

    local_bytes_step = 0
    if left_peer is not None:
        local_bytes_step += 2 * face_size_bytes  # 1 Send + 1 Recv
    if right_peer is not None:
        local_bytes_step += 2 * face_size_bytes  # 1 Send + 1 Recv

    # Acumula
    total_bytes_moved += local_bytes_step

    # Buffers temporários
    recv_buff_left = torch.empty((ny, nx), device=input_data.device)
    recv_buff_right = torch.empty((ny, nx), device=input_data.device)

    # --- FASE 1: Pares ---
    if rank % 2 == 0:
        if right_peer is not None:
            send_tensor = squeezed_data[INTERIOR_BACK].contiguous()
            dist.send(send_tensor, dst=right_peer)
            dist.recv(recv_buff_right, src=right_peer)
            squeezed_data[HALO_BACK] = recv_buff_right
    else:
        if left_peer is not None:
            dist.recv(recv_buff_left, src=left_peer)
            squeezed_data[HALO_FRONT] = recv_buff_left
            send_tensor = squeezed_data[INTERIOR_FRONT].contiguous()
            dist.send(send_tensor, dst=left_peer)

    # --- FASE 2: Ímpares ---
    if rank % 2 == 1:
        if right_peer is not None:
            send_tensor = squeezed_data[INTERIOR_BACK].contiguous()
            dist.send(send_tensor, dst=right_peer)
            dist.recv(recv_buff_right, src=right_peer)
            squeezed_data[HALO_BACK] = recv_buff_right
    else:
        if left_peer is not None:
            dist.recv(recv_buff_left, src=left_peer)
            squeezed_data[HALO_FRONT] = recv_buff_left
            send_tensor = squeezed_data[INTERIOR_FRONT].contiguous()
            dist.send(send_tensor, dst=left_peer)

    # 2. Sincroniza GPU antes de parar o timer
    torch.cuda.synchronize()
    end = time.time()

    elapsed = end - start
    comm_time += elapsed

    if rank == 0 and DEBUG_COMM:
        # Conversão: Bytes -> Bits (*8) -> Giga (/1e9)
        total_gbits = (total_bytes_moved * 8) / 1e9

        # Velocidade Média desde o início
        avg_speed = total_gbits / comm_time if comm_time > 0 else 0

        # Velocidade Instantânea desta chamada
        inst_speed = ((local_bytes_step * 8) / 1e9) / elapsed if elapsed > 0 else 0

        print(
            f"\n[Rank 0] Time: {comm_time:.2f}s | Vol: {total_gbits / 8:.2f} GB | Speed (Avg): {avg_speed:.2f} Gbit/s"
        )

    return squeezed_data.unsqueeze(0).unsqueeze(0)


def gather_all_data(local_tensor):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if(world_size == 1):
        return local_tensor.cpu()

    # 1. Manter na GPU para o NCCL funcionar
    # Garantimos que é contíguo para evitar erros de memória
    tensor_gpu = local_tensor.contiguous()

    if rank == 0:
        # Prepara lista de buffers na GPU
        gathered_list = [torch.empty_like(tensor_gpu) for _ in range(world_size)]
    else:
        gathered_list = None

    # 2. Executa o Gather na GPU (NCCL Suporta)
    dist.gather(tensor_gpu, gather_list=gathered_list, dst=0)

    if rank == 0:
        # 3. Concatena na GPU e move para CPU apenas no final
        full_tensor = torch.cat(gathered_list, dim=2)
        return full_tensor.cpu()

    return None
