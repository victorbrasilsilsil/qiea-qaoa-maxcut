"""Testes estatísticos pareados sobre o r final.

Wilcoxon pareado por (grafo, seed): para um dado p, cada par de otimizadores
é comparado nas mesmas instâncias e seeds, sobre a razão de aproximação
final. Reporta mediana por otimizador, a diferença e o p-valor.
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PAIRS = [("qiea", "random"), ("qiea", "cobyla"), ("qiea", "cobyla_restart")]


def _paired(df, p, a, b):
    # r_final pareado por (grafo_id, seed) pros dois otimizadores. A interseção
    # garante comparar só os pares que existem nos dois (defensivo).
    sub = df[df["p"] == p]
    key = ["grafo_id", "seed"]
    da = sub[sub["optimizer"] == a].set_index(key)["r_final"]
    db = sub[sub["optimizer"] == b].set_index(key)["r_final"]
    common = da.index.intersection(db.index)
    return da.loc[common].to_numpy(), db.loc[common].to_numpy()


def wilcoxon_table(csv_path, p_list=(1, 2), pairs=PAIRS, alpha=0.05):
    df = pd.read_csv(csv_path)
    print(f"Wilcoxon pareado por (grafo, seed) — {csv_path}\n")
    for p in p_list:
        print(f"p = {p}")
        print(f"  {'par':<26} {'med_A':>7} {'med_B':>7} {'dif':>8} "
              f"{'p-valor':>9}  signif")
        for a, b in pairs:
            xa, xb = _paired(df, p, a, b)
            ma, mb = np.median(xa), np.median(xb)
            diff = ma - mb
            # wilcoxon quebra se todas as diferenças são zero; o guard cobre o
            # empate total devolvendo p=1.
            if np.allclose(xa, xb):
                pval = 1.0
            else:
                pval = wilcoxon(xa, xb).pvalue
            sig = "sim" if pval < alpha else "nao"
            star = "  (A>B)" if diff > 0 else ("  (A<B)" if diff < 0 else "")
            print(f"  {a+' vs '+b:<26} {ma:>7.4f} {mb:>7.4f} {diff:>+8.4f} "
                  f"{pval:>9.2e}  {sig}{star}")
        print()


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "results_laptop_scaled.csv"
    wilcoxon_table(path)
