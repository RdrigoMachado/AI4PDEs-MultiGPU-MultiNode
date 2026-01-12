import torch
import torch.distributed as dist
import os

# Indexes
X_MIN, X_MAX = 0, -1
Y_MIN, Y_MAX = 0, -1
Z_MIN, Z_MAX = 0, -1

def init_process(backend='nccl'):
    try:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    except KeyError:
        print("Erro: Torchrun variables not found.")
        exit(1)

    device_id = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device_id)

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
    # (Mantive a função de gather igual, pois ela já estava correta para o pós-processamento)
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
        if world_size < 4 or world_size % 4 != 0:
             return torch.cat(gathered_list, dim=2).cpu()

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
    Troca de Halos SÍNCRONA e ORDENADA para evitar Deadlocks.
    Usa buffers contíguos para evitar erro de memória não densa.
    """

    # Identificadores de Topologia
    rank = dist.get_rank()

    # Definição de Paridade (Quem envia primeiro?)
    # X: Colunas Pares (0, 2...) vs Colunas Ímpares (1, 3...)
    is_even_col = (rank % 2 == 0)

    # Y: Linhas de Cima (0, 1...) vs Linhas de Baixo (2, 3...)
    # Nota: No seu setup, Rank%4 < 2 é Topo.
    is_top_row = (rank % 4 < 2)

    # Z: Índice do Nó na Pilha Z (0, 1, 2, 3...)
    z_idx = rank // 4
    is_even_z = (z_idx % 2 == 0)

    # ==========================================
    # 1. Eixo X (Left / Right) - Checkerboard
    # ==========================================
    # Passo A: Pares enviam para Direita, Ímpares recebem da Esquerda
    if is_even_col:
        if neighbors['right'] != -1:
            send_buff = tensor[..., X_MAX - 1].contiguous()
            dist.send(send_buff, dst=neighbors['right'])

            # Agora recebe a resposta (Halo Direito)
            recv_buff = torch.empty_like(tensor[..., X_MAX]).contiguous()
            dist.recv(recv_buff, src=neighbors['right'])
            tensor[..., X_MAX] = recv_buff

    else: # is_odd_col
        if neighbors['left'] != -1:
            # Ímpar primeiro recebe (Halo Esquerdo)
            recv_buff = torch.empty_like(tensor[..., X_MIN]).contiguous()
            dist.recv(recv_buff, src=neighbors['left'])
            tensor[..., X_MIN] = recv_buff

            # Depois envia a resposta
            send_buff = tensor[..., X_MIN + 1].contiguous()
            dist.send(send_buff, dst=neighbors['left'])

    # ==========================================
    # 2. Eixo Y (Top / Bottom) - Checkerboard
    # ==========================================
    # Passo A: Topo envia para Baixo, Baixo recebe de Cima
    if is_top_row:
        if neighbors['bottom'] != -1:
            send_buff = tensor[..., Y_MAX - 1, :].contiguous()
            dist.send(send_buff, dst=neighbors['bottom'])

            recv_buff = torch.empty_like(tensor[..., Y_MAX, :]).contiguous()
            dist.recv(recv_buff, src=neighbors['bottom'])
            tensor[..., Y_MAX, :] = recv_buff

    else: # is_bottom_row
        if neighbors['top'] != -1:
            recv_buff = torch.empty_like(tensor[..., Y_MIN, :]).contiguous()
            dist.recv(recv_buff, src=neighbors['top'])
            tensor[..., Y_MIN, :] = recv_buff

            send_buff = tensor[..., Y_MIN + 1, :].contiguous()
            dist.send(send_buff, dst=neighbors['top'])

    # ==========================================
    # 3. Eixo Z (Back / Front) - 2 Fases
    # ==========================================
    # Aqui é mais complexo pois é uma cadeia (0-1-2-3...).
    # Fase 1: Pares trocam com Front (0->1, 2->3). Ímpares trocam com Back (1<-0, 3<-2).
    # Fase 2: Ímpares trocam com Front (1->2). Pares trocam com Back (2<-1).

    # --- FASE 1: Conexões (0-1), (2-3), (4-5)... ---
    if is_even_z:
        # Sou par (ex: 0), falo com meu Front (ex: 1)
        if neighbors['front'] != -1:
            # Envio pra Frente
            sb = tensor[..., Z_MAX - 1, :, :].contiguous()
            dist.send(sb, dst=neighbors['front'])

            # Recebo da Frente
            rb = torch.empty_like(tensor[..., Z_MAX, :, :]).contiguous()
            dist.recv(rb, src=neighbors['front'])
            tensor[..., Z_MAX, :, :] = rb

    else: # is_odd_z
        # Sou ímpar (ex: 1), falo com meu Back (ex: 0)
        if neighbors['back'] != -1:
            # Recebo de Trás
            rb = torch.empty_like(tensor[..., Z_MIN, :, :]).contiguous()
            dist.recv(rb, src=neighbors['back'])
            tensor[..., Z_MIN, :, :] = rb

            # Envio pra Trás
            sb = tensor[..., Z_MIN + 1, :, :].contiguous()
            dist.send(sb, dst=neighbors['back'])

    # --- FASE 2: Conexões (1-2), (3-4)... ---
    if not is_even_z: # Sou ímpar (ex: 1)
        # Agora falo com meu Front (ex: 2)
        if neighbors['front'] != -1:
            sb = tensor[..., Z_MAX - 1, :, :].contiguous()
            dist.send(sb, dst=neighbors['front'])

            rb = torch.empty_like(tensor[..., Z_MAX, :, :]).contiguous()
            dist.recv(rb, src=neighbors['front'])
            tensor[..., Z_MAX, :, :] = rb

    else: # Sou par (ex: 2)
        # Agora falo com meu Back (ex: 1)
        if neighbors['back'] != -1:
            rb = torch.empty_like(tensor[..., Z_MIN, :, :]).contiguous()
            dist.recv(rb, src=neighbors['back'])
            tensor[..., Z_MIN, :, :] = rb

            sb = tensor[..., Z_MIN + 1, :, :].contiguous()
            dist.send(sb, dst=neighbors['back'])

    return tensor
