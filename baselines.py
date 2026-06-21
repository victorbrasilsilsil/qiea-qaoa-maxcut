"""Baselines clássicos sob o mesmo orçamento de avaliações do QIEA.

Todos operam no espaço contínuo de ângulos (gamma em [0,2pi), beta em
[0,pi)), compartilham o contador global de avaliações de qaoa e devolvem
(best_r, history). O history é o melhor-r-até-agora indexado por número de
avaliações de circuito, que é o eixo x da figura de convergência.

COBYLA usa scipy; random_search é NumPy puro.
"""

import numpy as np
from scipy.optimize import minimize

from qaoa import build_cut_counts, approx_ratio, reset_evals, get_evals


def _tracked_ratio(n, edges, cmax, p):
    # objetivo que, além de avaliar, registra o melhor-r-até-agora por avaliação.
    # Mesmo formato de history do qiea (ver run_experiment), pro eixo x da curva.
    cut_counts = build_cut_counts(n, edges)
    state = {"best": -np.inf, "history": []}

    def ratio(gammas, betas):
        r = approx_ratio(n, edges, gammas, betas, cmax, cut_counts=cut_counts)
        if r > state["best"]:
            state["best"] = r
        state["history"].append(state["best"])
        return r

    return ratio, state


def _rand_angles(rng, p):
    return rng.uniform(0, 2 * np.pi, p), rng.uniform(0, np.pi, p)


def random_search(n, edges, cmax, p, B, seed):
    # B amostras uniformes; o piso honesto da comparação
    reset_evals()
    rng = np.random.default_rng(seed)
    ratio, state = _tracked_ratio(n, edges, cmax, p)
    for _ in range(B):
        gammas, betas = _rand_angles(rng, p)
        ratio(gammas, betas)
    return state["best"], state["history"]


def cobyla(n, edges, cmax, p, B, seed):
    # COBYLA minimizando -r de um start aleatório. maxiter=B é só teto: ele quase
    # sempre converge e PARA bem antes (vimos ~40 aval. em p=1). Daí às vezes
    # morrer cedo num basin ruim, que é a cauda de falha do single-start.
    reset_evals()
    rng = np.random.default_rng(seed)
    ratio, state = _tracked_ratio(n, edges, cmax, p)

    def neg(x):
        return -ratio(x[:p], x[p:])

    g0, b0 = _rand_angles(rng, p)
    x0 = np.concatenate([g0, b0])
    minimize(neg, x0, method="COBYLA", options={"maxiter": B})
    return state["best"], state["history"]


def cobyla_restart(n, edges, cmax, p, B, seed):
    # reinícios aleatórios até gastar B no total. É o baseline forte: cada
    # restart é uma chance nova de escapar de mínimo local.
    reset_evals()
    rng = np.random.default_rng(seed)
    ratio, state = _tracked_ratio(n, edges, cmax, p)

    def neg(x):
        return -ratio(x[:p], x[p:])

    while get_evals() < B:
        remaining = B - get_evals()   # se cair abaixo de num_vars+2, scipy avisa MAXFUN; benigno
        g0, b0 = _rand_angles(rng, p)
        x0 = np.concatenate([g0, b0])
        minimize(neg, x0, method="COBYLA", options={"maxiter": remaining})

    # COBYLA pode passar do maxiter ao fechar uma iteração; apara pra B manter
    # o orçamento honesto e comparável aos outros otimizadores.
    state["history"] = state["history"][:B]
    state["best"] = state["history"][-1]
    return state["best"], state["history"]


if __name__ == "__main__":
    from graphs import load_instances, brute_force_maxcut
    from objective import make_qaoa_fitness
    from encoding import m_bits
    from qiea import qiea_optimize

    inst = load_instances()[0]
    n, edges = inst["n"], inst["edges"]
    cmax, _ = brute_force_maxcut(n, edges)
    p, B, seed = 1, 2000, 0

    r_rand, h_rand = random_search(n, edges, cmax, p, B, seed)
    r_cob, h_cob = cobyla(n, edges, cmax, p, B, seed)
    r_res, h_res = cobyla_restart(n, edges, cmax, p, B, seed)

    fitness = make_qaoa_fitness(n, edges, cmax, p)
    reset_evals()
    _, r_qiea, _ = qiea_optimize(fitness, m_bits(p), N=20, G=100, seed=seed)

    print(f"grafo id=0, n={n}, p={p}, B={B}, seed={seed}")
    print(f"  random        : r = {r_rand:.4f}   ({len(h_rand)} aval.)")
    print(f"  cobyla        : r = {r_cob:.4f}   ({len(h_cob)} aval.)")
    print(f"  cobyla_restart: r = {r_res:.4f}   ({len(h_res)} aval.)")
    print(f"  qiea          : r = {r_qiea:.4f}   (2000 aval.)")
