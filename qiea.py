"""QIEA binário (Han e Kim, 2002), NumPy puro.

Atalho que adotei: cada Q-bit é um ângulo theta só, não o par (alpha, beta).
Com isso P(medir 1) = sin(theta)^2 e a tabela de rotação vira um sinal em
theta. A cada geração observo a população, avalio e giro rumo ao melhor
global. Maximização; orçamento = N*G chamadas de fitness.
"""

import numpy as np

# theta preso longe de 0 e de pi/2. Mantém sin^2 em [0,1], mas o que importa
# mesmo é não deixar o Q-bit saturar em 0 ou 1: lá P(flip) zera e ele nunca
# mais volta atrás, matando a exploração.
_LO = 1e-3
_HI = np.pi / 2 - 1e-3


def observe(thetas, rng):
    # colapsa cada Q-bit em 0/1; P(1) = sin(theta)^2
    p1 = np.sin(thetas) ** 2
    return (rng.random(thetas.shape) < p1).astype(np.int8)


def rotate(thetas, observed, best, delta=0.04 * np.pi, better=True):
    """Gira cada theta rumo ao bit do best (tabela de Han e Kim colapsada).

    Como o Q-bit é um ângulo só, a tabela vira sinal: best quer 1 e observei 0,
    sobe P(1); quer 0 e observei 1, desce P(1); bit igual não mexe. better=False
    (indivíduo melhor que o best) inverteria, mas qiea_optimize sempre passa o
    best global, então na prática é sempre True.
    """
    d = np.zeros_like(thetas)
    d[(observed == 0) & (best == 1)] = +delta   # subir P(1)
    d[(observed == 1) & (best == 0)] = -delta   # descer P(1)
    if not better:
        d = -d
    return np.clip(thetas + d, _LO, _HI)


def qiea_optimize(fitness_fn, m, N=20, G=100, delta=0.04 * np.pi, seed=0):
    """Maximiza fitness_fn sobre bitstrings de m bits. Orçamento = N*G.

    Retorna (best_bits, best_fit, history). Pegadinha: history é por geração
    (melhor global acumulado, monótono), não por avaliação. Quando preciso da
    curva por avaliação (convergência), embrulho o fitness em run_experiment.
    Quem conta avaliação é o fitness_fn (ver qaoa), aqui só chamo uma por bits.
    """
    rng = np.random.default_rng(seed)
    thetas = np.full((N, m), np.pi / 4)   # pi/4 => P(1)=0.5, superposição uniforme

    best_bits = None
    best_fit = -np.inf
    history = []

    for _ in range(G):
        obs = np.empty((N, m), dtype=np.int8)
        for i in range(N):
            obs[i] = observe(thetas[i], rng)
            fit = fitness_fn(obs[i])
            if fit > best_fit:
                best_fit = fit
                best_bits = obs[i].copy()   # sem copy, best_bits viraria view de obs

        # giro depois de avaliar todo mundo: todos rumo ao mesmo best da geração
        for i in range(N):
            thetas[i] = rotate(thetas[i], obs[i], best_bits, delta)

        history.append(best_fit)

    return best_bits, best_fit, history


def _onemax_test():
    m = 20

    def fitness(bits):
        return int(bits.sum())

    best_bits, best_fit, history = qiea_optimize(fitness, m, N=20, G=100, seed=0)
    hit = next((g for g, h in enumerate(history) if h >= m), None)
    print(f"OneMax m={m}: melhor fitness = {best_fit} (esperado {m})")
    print(f"atingiu {m} na geracao {hit}")
    assert best_fit == m

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(history, lw=2)
    ax.axhline(m, color="gray", ls="--", lw=1)
    ax.set_xlabel("geração")
    ax.set_ylabel("melhor fitness")
    ax.set_title(f"QIEA no OneMax (m={m})")
    fig.tight_layout()
    fig.savefig("figs/onemax.png", dpi=150)
    plt.close(fig)
    print("curva salva em figs/onemax.png")


if __name__ == "__main__":
    _onemax_test()
