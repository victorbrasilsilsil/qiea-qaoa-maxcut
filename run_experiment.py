"""Orquestra a varredura de otimizadores e grava o CSV cru.

Para cada grafo de cada n, cada p, cada otimizador e cada seed (mesma lista
de seeds entre otimizadores, comparação pareada), roda sob o mesmo orçamento
B de avaliações e grava uma linha. O history é o melhor-r-até-agora indexado
por avaliação, em JSON, para a curva de convergência.
"""

import argparse
import csv
import json

import numpy as np

from graphs import load_instances, brute_force_maxcut
from encoding import m_bits
from objective import make_qaoa_fitness
from qiea import qiea_optimize
from qaoa import reset_evals
from baselines import random_search, cobyla, cobyla_restart

QIEA_POP = 20   # N; G = budget // N


def _run_qiea(n, edges, cmax, p, B, seed, b):
    # qiea_optimize devolve history por geração; aqui embrulho o fitness pra
    # gravar o melhor-r por AVALIAÇÃO. Assim qiea e baselines saem no mesmo
    # formato de history, que é o eixo x da curva de convergência.
    base = make_qaoa_fitness(n, edges, cmax, p, b=b)
    state = {"best": -np.inf, "history": []}

    def fitness(bits):
        r = base(bits)
        if r > state["best"]:
            state["best"] = r
        state["history"].append(state["best"])
        return r

    reset_evals()
    G = max(1, B // QIEA_POP)   # orçamento = N*G; G sai do B
    qiea_optimize(fitness, m_bits(p, b), N=QIEA_POP, G=G, seed=seed)
    return state["best"], state["history"]


def run_one(opt, n, edges, cmax, p, B, seed, b):
    if opt == "qiea":
        best, hist = _run_qiea(n, edges, cmax, p, B, seed, b)
    elif opt == "random":
        best, hist = random_search(n, edges, cmax, p, B, seed)
    elif opt == "cobyla":
        best, hist = cobyla(n, edges, cmax, p, B, seed)
    elif opt == "cobyla_restart":
        best, hist = cobyla_restart(n, edges, cmax, p, B, seed)
    else:
        raise ValueError(f"otimizador desconhecido: {opt}")
    return best, len(hist), hist


def budget_for(p, base_budget, scaled):
    # scaled => B = 1000*2p, o orçamento "justo" que cresce com o nº de
    # parâmetros, aplicado igual a todos. senão, B fixo pra todo p.
    return 1000 * 2 * p if scaled else base_budget


def sweep(n_list, p_list, optimizers, n_seeds, budget, out, b=8, scaled=False):
    instances = load_instances()
    insts = [g for g in instances if g["n"] in n_list]
    cmax = {g["id"]: brute_force_maxcut(g["n"], g["edges"])[0] for g in insts}
    seeds = list(range(n_seeds))   # mesma lista entre otimizadores => Wilcoxon pareado

    rows = 0
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["grafo_id", "n", "p", "optimizer", "seed",
                    "r_final", "evaluations", "history"])
        for g in insts:
            for p in p_list:
                B = budget_for(p, budget, scaled)
                for opt in optimizers:
                    for seed in seeds:
                        r, ev, hist = run_one(
                            opt, g["n"], g["edges"], cmax[g["id"]],
                            p, B, seed, b)
                        # round(5) encolhe o CSV; history cheio em float dá dezenas de MB
                        hist_json = json.dumps([round(x, 5) for x in hist])
                        w.writerow([g["id"], g["n"], p, opt, seed,
                                    round(r, 6), ev, hist_json])
                        rows += 1
            print(f"  grafo {g['id']} (n={g['n']}) concluido")
    print(f"{rows} linhas gravadas em {out}")
    return out


def summarize(out, p_list):
    import pandas as pd
    df = pd.read_csv(out)
    print("\nr medio por otimizador por p:")
    table = df.groupby(["p", "optimizer"])["r_final"].mean().unstack()
    print(table.to_string(float_format=lambda x: f"{x:.4f}"))

    # cuidado: este flag é só média. O cobyla single tem cauda de falha, então
    # ganha do qiea na mediana mesmo perdendo aqui. Ver stats.py e run_hybrid.py.
    for p in p_list:
        if "qiea" in table.columns and "cobyla" in table.columns:
            dq = table.loc[p, "qiea"]
            dc = table.loc[p, "cobyla"]
            flag = "SUPERA" if dq > dc else "nao supera"
            print(f"p={p}: qiea {dq:.4f} vs cobyla {dc:.4f}  -> qiea {flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_list", nargs="+", type=int, default=[8, 10, 12])
    ap.add_argument("--p_list", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--optimizers", nargs="+",
                    default=["qiea", "cobyla", "random", "cobyla_restart"])
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--budget", type=int, default=2000)
    ap.add_argument("--b", type=int, default=8)
    ap.add_argument("--scaled", action="store_true",
                    help="orcamento B=1000*2p por p (em vez de fixo)")
    ap.add_argument("--out", default="results_laptop.csv")
    args = ap.parse_args()

    sweep(args.n_list, args.p_list, args.optimizers,
          args.seeds, args.budget, args.out, args.b, args.scaled)
    summarize(args.out, args.p_list)


if __name__ == "__main__":
    main()
