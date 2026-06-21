"""Liga o fitness do QIEA ao objetivo do QAOA.

make_qaoa_fitness fecha sobre o grafo e devolve fitness(bits) -> razão de
aproximação, decodificando os bits em ângulos e simulando o circuito.
Cada chamada conta uma avaliação de circuito (via qaoa.approx_ratio).
"""

import numpy as np

from encoding import bits_to_angles, m_bits
from qaoa import build_cut_counts, approx_ratio


def make_qaoa_fitness(n, edges, cmax, p, b=8):
    # cut_counts uma vez só, preso no closure: cada fitness reusa em vez de
    # reconstruir o vetor 2^n. É o que torna o sweep viável no laptop.
    cut_counts = build_cut_counts(n, edges)

    def fitness(bits):
        gammas, betas = bits_to_angles(bits, p, b)
        return approx_ratio(n, edges, gammas, betas, cmax, cut_counts=cut_counts)

    return fitness


if __name__ == "__main__":
    from graphs import load_instances, brute_force_maxcut
    from qaoa import grid_scan_p1, reset_evals, get_evals
    from qiea import qiea_optimize

    inst = load_instances()[0]   # primeiro grafo n=8
    n, edges = inst["n"], inst["edges"]
    cmax, _ = brute_force_maxcut(n, edges)

    grid_r, _, _ = grid_scan_p1(n, edges, cmax, save=False)

    p = 1
    fitness = make_qaoa_fitness(n, edges, cmax, p)
    reset_evals()
    _, qiea_r, _ = qiea_optimize(fitness, m_bits(p), N=20, G=100, seed=0)
    evals = get_evals()

    print(f"grafo id=0, n={n}, p={p}")
    print(f"  otimo da grade : r = {grid_r:.4f}")
    print(f"  melhor do QIEA : r = {qiea_r:.4f}   ({evals} avaliacoes)")
    print(f"  diferenca      : {abs(grid_r - qiea_r):.4f}  (limite 0.02)")
    assert abs(grid_r - qiea_r) < 0.02, "QIEA longe do otimo: ver encoding/rotacao"
    print("INTEGRACAO OK")
