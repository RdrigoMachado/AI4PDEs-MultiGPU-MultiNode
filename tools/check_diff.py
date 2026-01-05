import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os

def compare_results(file_ref, file_new, tolerance=1e-5):
    print(f"--- Comparando arquivos ---")
    print(f"Ref (Z):      {file_ref}")
    print(f"Teste (Quad): {file_new}")

    # Carrega os arquivos
    # Tenta carregar como .npy (formato numpy)
    try:
        data_ref = np.load(file_ref)
        data_new = np.load(file_new)
    except Exception as e:
        print(f"\nERRO ao abrir arquivos: {e}")
        print("Certifique-se que são arquivos .npy salvos com np.save()")
        sys.exit(1)

    # 1. Verificação de Formato
    if data_ref.shape != data_new.shape:
        print(f"\nERRO CRÍTICO: Formatos diferentes!")
        print(f"Ref:   {data_ref.shape}")
        print(f"Teste: {data_new.shape}")
        sys.exit(1)

    # 2. Cálculos de Diferença
    diff = np.abs(data_ref - data_new)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    # Norma L2 para erro relativo
    norm_diff = np.linalg.norm(data_ref - data_new)
    norm_ref = np.linalg.norm(data_ref)

    # Evita divisão por zero
    if norm_ref == 0:
        relative_error = 0.0 if norm_diff == 0 else float('inf')
    else:
        relative_error = norm_diff / norm_ref

    # 3. Relatório
    print(f"\n--- Resultados Estatísticos ---")
    print(f"Máxima Diferença Absoluta: {max_diff:.8e}")
    print(f"Média da Diferença:        {mean_diff:.8e}")
    print(f"Erro Relativo (Norma L2):  {relative_error:.8e}")

    # Verificação de Tolerância
    passed = np.allclose(data_ref, data_new, rtol=tolerance, atol=tolerance)
    status = "SUCESSO (SIM)" if passed else "FALHA (NÃO)"

    print(f"\nPassou na tolerância ({tolerance})?  ===> {status}")

    # 4. Gerar Gráfico de Diagnóstico (Salvar em disco)
    try:
        # Pega a fatia central do eixo Z (ou eixo 0)
        mid_idx = data_ref.shape[0] // 2

        # Se for 3D ou mais, plota uma fatia 2D
        if data_ref.ndim >= 3:
            slice_to_plot = diff[mid_idx, :, :]
            title_str = f"Erro Absoluto - Fatia Z={mid_idx}"
        elif data_ref.ndim == 2:
            slice_to_plot = diff
            title_str = "Erro Absoluto - 2D"
        else:
            print("Dados 1D detectados, pulando gráfico 2D.")
            return

        plt.figure(figsize=(10, 6))
        plt.imshow(slice_to_plot, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Diferença Absoluta')
        plt.title(f"{title_str}\nMax Diff Local: {np.max(slice_to_plot):.2e}")

        output_img = "mapa_de_erro.png"
        plt.savefig(output_img)
        print(f"\n[Visualização] Mapa de erro salvo em: {output_img}")
        plt.close()

    except Exception as e:
        print(f"Não foi possível gerar o gráfico: {e}")

if __name__ == "__main__":
    # Configura argumentos da linha de comando
    parser = argparse.ArgumentParser(description='Compara resultados de simulação numérica.')
    parser.add_argument('ref', help='Arquivo de referência (ex: versao_z.npy)')
    parser.add_argument('test', help='Arquivo de teste (ex: versao_quadrantes.npy)')
    parser.add_argument('--tol', type=float, default=1e-5, help='Tolerância para aprovação (default: 1e-5)')

    args = parser.parse_args()

    # Verifica se arquivos existem
    if not os.path.exists(args.ref):
        print(f"Arquivo não encontrado: {args.ref}")
        sys.exit(1)
    if not os.path.exists(args.test):
        print(f"Arquivo não encontrado: {args.test}")
        sys.exit(1)

    compare_results(args.ref, args.test, args.tol)
