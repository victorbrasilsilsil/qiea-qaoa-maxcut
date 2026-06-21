"""Geração de instâncias de Max-Cut e força bruta do corte máximo.

Grafos 3-regulares aleatórios para n em {8, 10, 12}, 5 por tamanho,
determinísticos a partir de uma seed mestra. Instâncias salvas em
instances.json (lista de arestas por grafo). NumPy puro + networkx.
"""

import json
import itertools

import numpy as np
import networkx as nx

N_LIST = (8, 10, 12)
GRAPHS_PER_N = 5
INSTANCES_FILE = "instances.json"


def gen_instances(seed=42, n_list=N_LIST, graphs_per_n=GRAPHS_PER_N,
                  out=INSTANCES_FILE):
    """Gera grafos 3-regulares determinísticos e salva em JSON.

    Cada grafo recebe uma seed derivada da seed mestra, de modo que a
    coleção inteira é reprodutível. Retorna a lista de instâncias, onde
    cada instância é um dict {id, n, edges}.
    """
    instances = []
    gid = 0
    for n in n_list:
        accepted = []           # grafos nx já aceitos para este n
        k = 0
        while len(accepted) < graphs_per_n:
            # k avança e pula grafo isomorfo a algum já aceito, pra não ter
            # instância duplicada inflando a estatística pareada. Foi assim que
            # o 5º grafo de n=10 trocou (o naive gerava um isomorfo ao 1º).
            g_seed = seed + 1000 * n + k
            G = nx.random_regular_graph(3, n, seed=g_seed)
            k += 1
            if any(nx.is_isomorphic(G, H) for H in accepted):
                continue
            accepted.append(G)
            edges = [[int(u), int(v)] for u, v in G.edges()]
            instances.append({"id": gid, "n": int(n), "edges": edges})
            gid += 1

    with open(out, "w") as f:
        json.dump(instances, f, indent=2)
    return instances


def load_instances(path=INSTANCES_FILE):
    # carrega o JSON; arestas voltam como tuplas
    with open(path) as f:
        instances = json.load(f)
    for inst in instances:
        inst["edges"] = [tuple(e) for e in inst["edges"]]
    return instances


def brute_force_maxcut(n, edges):
    """Enumera os 2^n cortes e retorna (valor_max, bitstring_max).

    Vetorizado: avalia os 2^n cortes de uma vez. A ordem de bits aqui ((x>>q)&1)
    NÃO é a do qaoa; tudo bem, o maxcut é invariante a relabel de nó e só importa
    o máximo, então não "conserte" pra casar com o simulador.
    """
    edges = np.asarray(edges, dtype=np.int64)
    if edges.size == 0:
        return 0, (0,) * n

    x = np.arange(2 ** n, dtype=np.int64)
    # bit do nó q em cada um dos 2^n estados: matriz (2^n, n)
    qbits = ((x[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int8)

    u = edges[:, 0]
    v = edges[:, 1]
    # aresta cortada quando os dois nós caem em lados diferentes
    cut = qbits[:, u] != qbits[:, v]            # (2^n, |E|)
    cut_counts = cut.sum(axis=1)                # (2^n,)

    best = int(cut_counts.argmax())
    value = int(cut_counts[best])
    bitstring = tuple(int(b) for b in qbits[best])
    return value, bitstring


def _selftest():
    # ciclo de 4 nós: maxcut = 4
    cycle4 = [(0, 1), (1, 2), (2, 3), (3, 0)]
    v4, _ = brute_force_maxcut(4, cycle4)
    print(f"ciclo de 4 nós  -> maxcut = {v4} (esperado 4)")
    assert v4 == 4, v4

    # triângulo: maxcut = 2
    tri = [(0, 1), (1, 2), (2, 0)]
    v3, _ = brute_force_maxcut(3, tri)
    print(f"triângulo       -> maxcut = {v3} (esperado 2)")
    assert v3 == 2, v3

    print("gabaritos OK\n")

    instances = gen_instances()
    print(f"{len(instances)} grafos gerados, salvos em {INSTANCES_FILE}")
    print(f"{'id':>3} {'n':>3} {'|E|':>4} {'maxcut':>7}")
    for inst in instances:
        cmax, _ = brute_force_maxcut(inst["n"], inst["edges"])
        print(f"{inst['id']:>3} {inst['n']:>3} {len(inst['edges']):>4} {cmax:>7}")


if __name__ == "__main__":
    _selftest()
