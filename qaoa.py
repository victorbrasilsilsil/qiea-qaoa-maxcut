"""Simulador de QAOA Max-Cut em NumPy puro (sem Pennylane).

Statevector de 2^n amplitudes. Invariante que não pode quebrar: o qubit q
ocupa o bit de peso 2^(n-1-q) do índice (qubit 0 = MSB). É essa convenção que
casa o reshape do mixer com o cut_counts; mexer numa sem mexer na outra erra a
expectativa em silêncio.
"""

import numpy as np

# contador global de propósito: os otimizadores compartilham, e uma avaliação =
# uma chamada do objetivo, que é a unidade de orçamento que dá justiça à comparação.
_EVAL_COUNT = 0


def reset_evals():
    global _EVAL_COUNT
    _EVAL_COUNT = 0


def get_evals():
    return _EVAL_COUNT


def build_cut_counts(n, edges):
    # diagonal de H_C: nº de arestas cortadas por cada bitstring x (vetor 2^n).
    # O bit (n-1-u) segue a convenção de qubit do mixer; se divergir, a fase de
    # custo cai no qubit errado e tudo parece certo menos o resultado.
    edges = np.asarray(edges, dtype=np.int64)
    x = np.arange(2 ** n, dtype=np.int64)
    counts = np.zeros(2 ** n, dtype=np.float64)
    for u, v in edges:
        bu = (x >> (n - 1 - u)) & 1
        bv = (x >> (n - 1 - v)) & 1
        counts += (bu != bv)
    return counts


def initial_state(n):
    # |+>^n: 2^n amplitudes iguais a 1/sqrt(2^n)
    return np.full(2 ** n, 1.0 / np.sqrt(2 ** n), dtype=np.complex128)


def apply_cost(psi, gamma, cut_counts):
    # U_C(gamma): H_C é diagonal na base computacional, então o custo é só uma
    # fase por amplitude. É o termo barato.
    return psi * np.exp(-1j * gamma * cut_counts)


def apply_mixer(psi, beta, n):
    """U_B(beta) = prod_q exp(-i*beta*X_q), aplicado qubit a qubit.

    Truque pra não montar matriz 2^n x 2^n: o reshape isola o eixo do qubit q
    (tamanho 2) e r[:, ::-1, :] é o X (troca |0>/|1> nesse eixo). Cada qubit
    custa O(2^n), não O(4^n). exp(-i beta X) = cos(beta) I - i sin(beta) X.
    """
    c = np.cos(beta)
    s = np.sin(beta)
    for q in range(n):
        r = psi.reshape(2 ** q, 2, 2 ** (n - q - 1))
        r = c * r - 1j * s * r[:, ::-1, :]
        psi = r.reshape(-1)
    return psi


def qaoa_state(n, cut_counts, gammas, betas):
    # estado final: por camada l, U_C(gamma_l) e depois U_B(beta_l)
    psi = initial_state(n)
    for gamma, beta in zip(gammas, betas):
        psi = apply_cost(psi, gamma, cut_counts)
        psi = apply_mixer(psi, beta, n)
    return psi


def _expectation(n, cut_counts, gammas, betas):
    # <H_C> SEM tocar no contador. Separado de propósito: o grid_scan chama isto
    # res^2 vezes e não pode inflar o orçamento dos otimizadores.
    psi = qaoa_state(n, cut_counts, gammas, betas)
    probs = np.abs(psi) ** 2
    return float(np.dot(probs, cut_counts))


def qaoa_expectation(n, edges, gammas, betas, cut_counts=None):
    # <H_C> contando uma avaliação. Passe cut_counts pronto pra não reconstruir
    # o vetor 2^n a cada chamada; no sweep isso vira minutos de diferença.
    global _EVAL_COUNT
    if cut_counts is None:
        cut_counts = build_cut_counts(n, edges)
    _EVAL_COUNT += 1
    return _expectation(n, cut_counts, gammas, betas)


def approx_ratio(n, edges, gammas, betas, cmax, cut_counts=None):
    # razão de aproximação = <H_C> / cmax
    return qaoa_expectation(n, edges, gammas, betas, cut_counts) / cmax


def grid_scan_p1(n, edges, cmax, res=100, save=True):
    """Varredura p=1 em grade res x res: gamma em [0,2pi), beta em [0,pi).

    Usa _expectation de propósito, pra não contar essas res^2 avaliações como
    orçamento. Retorna (melhor_razao, melhor_gamma, melhor_beta) e, com save,
    grava o heatmap em figs/landscape_n{n}.png (vai pro painel).
    """
    cut_counts = build_cut_counts(n, edges)
    gammas = np.linspace(0, 2 * np.pi, res, endpoint=False)
    betas = np.linspace(0, np.pi, res, endpoint=False)

    ratios = np.empty((res, res))  # linhas=beta, colunas=gamma
    for i, beta in enumerate(betas):
        for j, gamma in enumerate(gammas):
            ratios[i, j] = _expectation(n, cut_counts, [gamma], [beta]) / cmax

    bi, bj = np.unravel_index(ratios.argmax(), ratios.shape)
    best_r = float(ratios[bi, bj])
    best_gamma = float(gammas[bj])
    best_beta = float(betas[bi])

    if save:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(
            ratios, origin="lower", aspect="auto", cmap="viridis",
            extent=[0, 2 * np.pi, 0, np.pi],
        )
        ax.plot(best_gamma, best_beta, "r*", markersize=12)
        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel(r"$\beta$")
        ax.set_title(f"QAOA p=1 landscape, n={n} (best r={best_r:.4f})")
        fig.colorbar(im, ax=ax, label="approx. ratio")
        fig.tight_layout()
        fig.savefig(f"figs/landscape_n{n}.png", dpi=150)
        plt.close(fig)

    return best_r, best_gamma, best_beta


if __name__ == "__main__":
    from graphs import load_instances

    inst = load_instances()[0]  # primeiro grafo n=8
    n, edges = inst["n"], inst["edges"]
    from graphs import brute_force_maxcut
    cmax, _ = brute_force_maxcut(n, edges)

    best_r, g, b = grid_scan_p1(n, edges, cmax)
    print(f"grafo id=0, n={n}, |E|={len(edges)}, cmax={cmax}")
    print(f"melhor razao da grade (p=1): {best_r:.4f}")
    print(f"  em gamma={g:.4f}, beta={b:.4f}")
    print(f"heatmap salvo em figs/landscape_n{n}.png")
