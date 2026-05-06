#!/usr/bin/env python3
"""Compara saídas .npy de duas execuções (baseline vs estratégia async).

Uso:
    python compare_halo_outputs.py <dir_ref> <dir_test> [--rtol 1e-5] [--atol 1e-5]

Itera sobre todos os arquivos .npy comuns às duas pastas e reporta:
- max abs diff
- mean abs diff
- L2 relative error
- pass/fail por tolerância

Critério de validação Exp 0 (PAPER_CONTEXT, Seção 0):
    Saída deve ser numericamente igual ao baseline até precisão de FP32.
    rtol=1e-5, atol=1e-5 são apropriados para FP32.

Exit code 0 se tudo passou; 1 se ao menos um arquivo falhou.
"""

import argparse
import os
import sys

import numpy as np


def compare_pair(ref_path, test_path, rtol, atol):
    ref = np.load(ref_path)
    test = np.load(test_path)

    if ref.shape != test.shape:
        return {
            "status": "SHAPE_MISMATCH",
            "ref_shape": ref.shape,
            "test_shape": test.shape,
        }

    diff = np.abs(ref.astype(np.float64) - test.astype(np.float64))
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())

    norm_diff = float(np.linalg.norm(ref.astype(np.float64) - test.astype(np.float64)))
    norm_ref = float(np.linalg.norm(ref.astype(np.float64)))
    rel_l2 = norm_diff / (norm_ref + 1e-30)

    passed = bool(np.allclose(ref, test, rtol=rtol, atol=atol))

    return {
        "status": "PASS" if passed else "FAIL",
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rel_l2": rel_l2,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dir_ref", help="Pasta de referência (baseline)")
    parser.add_argument("dir_test", help="Pasta da estratégia em teste")
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Imprime só o resumo final",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.dir_ref):
        print(f"ERRO: pasta de referência não existe: {args.dir_ref}", file=sys.stderr)
        return 2
    if not os.path.isdir(args.dir_test):
        print(f"ERRO: pasta de teste não existe: {args.dir_test}", file=sys.stderr)
        return 2

    ref_files = {f for f in os.listdir(args.dir_ref) if f.endswith(".npy")}
    test_files = {f for f in os.listdir(args.dir_test) if f.endswith(".npy")}

    common = sorted(ref_files & test_files)
    only_ref = sorted(ref_files - test_files)
    only_test = sorted(test_files - ref_files)

    if not common:
        print(f"ERRO: nenhum .npy em comum entre {args.dir_ref} e {args.dir_test}",
              file=sys.stderr)
        return 2

    if only_ref:
        print(f"AVISO: {len(only_ref)} arquivo(s) só em {args.dir_ref}: {only_ref[:5]}{'...' if len(only_ref) > 5 else ''}")
    if only_test:
        print(f"AVISO: {len(only_test)} arquivo(s) só em {args.dir_test}: {only_test[:5]}{'...' if len(only_test) > 5 else ''}")

    print(f"\nComparando {len(common)} arquivo(s) — rtol={args.rtol}, atol={args.atol}")
    print(f"REF:  {args.dir_ref}")
    print(f"TEST: {args.dir_test}\n")

    n_pass = 0
    n_fail = 0
    n_shape = 0
    worst_max_abs = 0.0
    worst_rel_l2 = 0.0
    worst_file = None

    if not args.quiet:
        print(f"{'file':<24} {'status':<14} {'max_abs':>14} {'mean_abs':>14} {'rel_l2':>14}")
        print("-" * 84)

    for fn in common:
        result = compare_pair(
            os.path.join(args.dir_ref, fn),
            os.path.join(args.dir_test, fn),
            args.rtol,
            args.atol,
        )
        if result["status"] == "SHAPE_MISMATCH":
            n_shape += 1
            if not args.quiet:
                print(f"{fn:<24} SHAPE_MISMATCH ref={result['ref_shape']} test={result['test_shape']}")
            continue

        if result["status"] == "PASS":
            n_pass += 1
        else:
            n_fail += 1

        if result["max_abs"] > worst_max_abs:
            worst_max_abs = result["max_abs"]
            worst_rel_l2 = result["rel_l2"]
            worst_file = fn

        if not args.quiet:
            print(f"{fn:<24} {result['status']:<14} {result['max_abs']:>14.6e} "
                  f"{result['mean_abs']:>14.6e} {result['rel_l2']:>14.6e}")

    print("\n" + "=" * 84)
    print(f"PASS:  {n_pass}/{len(common)}")
    print(f"FAIL:  {n_fail}/{len(common)}")
    if n_shape:
        print(f"SHAPE MISMATCH: {n_shape}")
    if worst_file is not None:
        print(f"Pior caso: {worst_file}  max_abs={worst_max_abs:.6e}  rel_l2={worst_rel_l2:.6e}")
    print("=" * 84)

    return 0 if (n_fail == 0 and n_shape == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
