"""Conversão bitstring <-> ângulos do QAOA.

Cada ângulo é um inteiro de b bits escalado pra sua faixa: gamma em [0,2pi),
beta em [0,pi), as mesmas faixas da varredura em grade do qaoa. Layout de
m=2pb bits: os primeiros p*b são os gammas, os p*b seguintes os betas.
"""

import numpy as np

GAMMA_HI = 2 * np.pi
BETA_HI = np.pi


def m_bits(p, b=8):
    return 2 * p * b


def _bits_to_ints(bits, p, b):
    # reagrupa bits em p inteiros de b bits. MSB primeiro é só convenção
    # interna; encode e decode usam a mesma, então não precisa casar com o qaoa.
    bits = np.asarray(bits, dtype=np.int64).reshape(p, b)
    weights = (1 << np.arange(b)[::-1])
    return bits @ weights


def _ints_to_bits(ks, b):
    # inteiros -> linhas de b bits (MSB primeiro, par do _bits_to_ints)
    ks = np.asarray(ks, dtype=np.int64)
    shifts = np.arange(b)[::-1]
    return ((ks[:, None] >> shifts) & 1).astype(np.int8).reshape(-1)


def bits_to_angles(bits, p, b=8):
    # decodifica m=2pb bits em (gammas, betas)
    bits = np.asarray(bits, dtype=np.int64)
    gk = _bits_to_ints(bits[: p * b], p, b)
    bk = _bits_to_ints(bits[p * b:], p, b)
    # span = 2^b - 1, NÃO 2^b: assim os níveis 0..2^b-1 cobrem [0, hi] fechado.
    # Com 2^b, um ângulo perto do topo arredonda pro nível 2^b, o clip puxa pra
    # 2^b-1 e o round-trip estoura meia resolução. Foi o bug que quebrou o teste.
    span = (1 << b) - 1
    gammas = gk / span * GAMMA_HI
    betas = bk / span * BETA_HI
    return gammas, betas


def angles_to_bits(gammas, betas, b=8):
    # codifica (gammas, betas) em 2pb bits; mesmo span do decode (ver lá)
    span = (1 << b) - 1
    gk = np.clip(np.round(np.asarray(gammas) / GAMMA_HI * span), 0, span)
    bk = np.clip(np.round(np.asarray(betas) / BETA_HI * span), 0, span)
    return np.concatenate([_ints_to_bits(gk, b), _ints_to_bits(bk, b)])


def _roundtrip_test(n_angles=1000, b=8, seed=0):
    rng = np.random.default_rng(seed)
    gammas = rng.uniform(0, GAMMA_HI, n_angles)
    betas = rng.uniform(0, BETA_HI, n_angles)

    g_err = b_err = 0.0
    for g, bt in zip(gammas, betas):
        bits = angles_to_bits([g], [bt], b)
        gr, br = bits_to_angles(bits, p=1, b=b)
        g_err = max(g_err, abs(gr[0] - g))
        b_err = max(b_err, abs(br[0] - bt))

    span = (1 << b) - 1
    half_g = GAMMA_HI / (2 * span)        # meia resolução do gamma
    half_b = BETA_HI / (2 * span)         # meia resolução do beta
    print(f"round-trip {n_angles} angulos, b={b}")
    print(f"  erro max gamma = {g_err:.3e}  (meia res = {half_g:.3e} ~ pi/255)")
    print(f"  erro max beta  = {b_err:.3e}  (meia res = {half_b:.3e} ~ pi/510)")
    assert g_err <= half_g + 1e-12
    assert b_err <= half_b + 1e-12
    print("round-trip OK")


if __name__ == "__main__":
    _roundtrip_test()
