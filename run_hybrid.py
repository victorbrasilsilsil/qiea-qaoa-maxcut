"""Probe do híbrido QIEA -> COBYLA curto (explorador + refinador).

Re-roda o QIEA (determinístico), decodifica a melhor solução em ângulos e
roda um COBYLA curto (~200 avaliações) a partir dela. Compara o r do híbrido
contra qiea puro, cobyla single e cobyla_restart (lidos do summary.csv, os
mesmos números do benchmark), com Wilcoxon pareado por (grafo, seed).

O híbrido gasta o orçamento do QIEA MAIS ~200 avaliações do refinador; não é
comparação de orçamento igual, é um teste da hipótese explorador+refinador.
Não altera qiea.py nem baselines.py.
"""

import csv

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import wilcoxon

from graphs import load_instances, brute_force_maxcut
from encoding import bits_to_angles, m_bits
from objective import make_qaoa_fitness
from qiea import qiea_optimize
from qaoa import build_cut_counts, approx_ratio, reset_evals, get_evals

QIEA_POP = 20
REFINE_EVALS = 200
B_FACTOR = 1000   # orçamento escalado do QIEA: B = 1000*2p (igual ao benchmark)


def refine_from(n, edges, cmax, p, gammas, betas, max_evals=REFINE_EVALS):
    # COBYLA curto partindo dos ângulos do QIEA. Como x0 é o próprio ponto do
    # QIEA, a 1ª avaliação reproduz o r dele, então o híbrido nunca piora.
    # Retorna (best_r, evals).
    cut_counts = build_cut_counts(n, edges)
    best = {"r": -np.inf}

    def neg(x):
        r = approx_ratio(n, edges, x[:p], x[p:], cmax, cut_counts=cut_counts)
        best["r"] = max(best["r"], r)
        return -r

    x0 = np.concatenate([gammas, betas])
    reset_evals()
    minimize(neg, x0, method="COBYLA", options={"maxiter": max_evals})
    return best["r"], get_evals()


def run(out="results_hybrid.csv", n_seeds=20, b=8, summary="summary.csv"):
    base = pd.read_csv(summary)
    insts = load_instances()
    cmax = {g["id"]: brute_force_maxcut(g["n"], g["edges"])[0] for g in insts}

    def base_r(gid, p, seed, opt):
        m = base[(base.grafo_id == gid) & (base.p == p) &
                 (base.seed == seed) & (base.optimizer == opt)]
        return float(m["r_final"].iloc[0])

    rows = 0
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["grafo_id", "n", "p", "seed", "r_qiea", "r_hybrid",
                    "r_cobyla", "r_cobyla_restart", "qiea_evals", "refine_evals"])
        for g in insts:
            gid, n, edges = g["id"], g["n"], g["edges"]
            for p in (1, 2):
                B = B_FACTOR * 2 * p
                fitness = make_qaoa_fitness(n, edges, cmax[gid], p, b=b)
                for seed in range(n_seeds):
                    # mesma seed reproduz o MESMO best_bits do benchmark
                    # (determinístico); por isso leio cobyla/restart do summary.csv
                    # em vez de recomputar, garantindo números idênticos.
                    reset_evals()
                    best_bits, r_qiea, _ = qiea_optimize(
                        fitness, m_bits(p, b), N=QIEA_POP, G=B // QIEA_POP, seed=seed)
                    q_evals = get_evals()
                    gammas, betas = bits_to_angles(best_bits, p, b)
                    r_hyb, r_evals = refine_from(n, edges, cmax[gid], p, gammas, betas)
                    w.writerow([gid, n, p, seed,
                                round(r_qiea, 6), round(r_hyb, 6),
                                round(base_r(gid, p, seed, "cobyla"), 6),
                                round(base_r(gid, p, seed, "cobyla_restart"), 6),
                                q_evals, r_evals])
                    rows += 1
            print(f"  grafo {gid} (n={n}) concluido")
    print(f"{rows} linhas gravadas em {out}")
    return out


def summarize(out="results_hybrid.csv"):
    df = pd.read_csv(out)
    cols = ["r_qiea", "r_hybrid", "r_cobyla", "r_cobyla_restart"]
    print("\nr medio por p (qiea puro / hibrido / cobyla single / multi-start):")
    print(df.groupby("p")[cols].mean().round(4).to_string())
    print(f"\navaliacoes medias do refinador: {df['refine_evals'].mean():.0f} "
          f"(QIEA usa {df['qiea_evals'].mean():.0f}; hibrido = soma)")

    print("\nWilcoxon pareado por (grafo, seed):")
    for p in (1, 2):
        sub = df[df.p == p]
        print(f"p = {p}")
        for other in ["r_qiea", "r_cobyla", "r_cobyla_restart"]:
            a, bb = sub["r_hybrid"].to_numpy(), sub[other].to_numpy()
            med_diff = float(np.median(a - bb))
            if np.allclose(a, bb):
                pval = 1.0
            else:
                pval = wilcoxon(a, bb).pvalue
            sig = "sim" if pval < 0.05 else "nao"
            tag = {"r_qiea": "hibrido vs qiea", "r_cobyla": "hibrido vs cobyla",
                   "r_cobyla_restart": "hibrido vs cobyla_restart"}[other]
            arrow = "(hib>)" if med_diff > 0 else ("(hib<)" if med_diff < 0 else "")
            print(f"  {tag:<28} med_dif={med_diff:+.4f}  p={pval:.2e}  signif={sig} {arrow}")
        print()


if __name__ == "__main__":
    run()
    summarize()
