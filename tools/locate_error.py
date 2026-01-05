import numpy as np
import sys

def locate_max_error(file_ref, file_test):
    data_ref = np.load(file_ref)
    data_test = np.load(file_test)

    diff = np.abs(data_ref - data_test)
    max_val = np.max(diff)

    # Encontra os índices onde o erro é máximo
    # np.where retorna uma tupla de arrays (z_indices, y_indices, x_indices)
    indices = np.where(diff == max_val)

    # Pega a primeira ocorrência
    z, y, x = indices[0][0], indices[1][0], indices[2][0]

    print(f"--- Diagnóstico de Localização ---")
    print(f"Shape dos dados: {data_ref.shape}")
    print(f"Erro Máximo: {max_val}")
    print(f"Ocorre na coordenada: Z={z}, Y={y}, X={x}")
    print(f"Valor Ref:   {data_ref[z,y,x]}")
    print(f"Valor Teste: {data_test[z,y,x]}")

    # Verifica se está na borda
    nz, ny, nx = data_ref.shape
    is_border = (x == 0 or x == nx-1 or y == 0 or y == ny-1 or z == 0 or z == nz-1)

    print(f"\nÉ borda do domínio global? {'SIM' if is_border else 'NÃO'}")

    # Checagem de vizinhança
    print("\nVizinhança do erro (Ref vs Teste):")
    try:
        print(f"Ref:\n{data_ref[z, y-1:y+2, x-1:x+2]}")
        print(f"Teste:\n{data_test[z, y-1:y+2, x-1:x+2]}")
    except:
        print("Não foi possível mostrar vizinhança (borda muito próxima).")

if __name__ == "__main__":
    locate_max_error(sys.argv[1], sys.argv[2])
