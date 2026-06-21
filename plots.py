"""Figuras a partir do CSV de resultados.

Figura principal do abstract: curvas de convergência (r médio vs avaliações,
com banda de desvio), painéis p=1 e p=2. Figura de estabilidade para o
painel: boxplot do r final por otimizador, por p. O heatmap de landscape é
gerado por qaoa.grid_scan_p1 e fica reservado ao painel.

Lê por padrão results_laptop_scaled.csv (orçamento B=1000*2p, justo).
"""

import argparse
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ordem e cor fixas: restart no topo, qiea/random no meio, cobyla single embaixo
OPT_ORDER = ["cobyla_restart", "qiea", "random", "cobyla"]
OPT_LABEL = {
    "cobyla_restart": "COBYLA multi-start",
    "qiea": "QIEA",
    "random": "random search",
    "cobyla": "COBYLA single",
}
OPT_COLOR = {
    "cobyla_restart": "#1f77b4",
    "qiea": "#ff7f0e",
    "random": "#2ca02c",
    "cobyla": "#d62728",
}

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "lines.linewidth": 1.3,
})

COL_W = 3.4   # largura de uma coluna (polegadas)


def load(csv_path):
    df = pd.read_csv(csv_path)
    df["hist"] = df["history"].apply(json.loads)
    return df


def _padded_stack(hists, length):
    # empilha históricos preenchendo até `length` com o último valor. O
    # melhor-r-até-agora satura quando o otimizador para (ex. cobyla single para
    # cedo), então o padding é o platô real, não invenção.
    out = np.empty((len(hists), length))
    for i, h in enumerate(hists):
        h = np.asarray(h, dtype=float)
        out[i, : len(h)] = h
        out[i, len(h):] = h[-1]
    return out


def convergence_figure(df, p_list=(1, 2), out="figs/convergence.pdf"):
    fig, axes = plt.subplots(len(p_list), 1, figsize=(COL_W, 2.1 * len(p_list)),
                             sharex=False)
    if len(p_list) == 1:
        axes = [axes]

    for ax, p in zip(axes, p_list):
        sub = df[df["p"] == p]
        B = int(sub["evaluations"].max())   # orçamento pleno para este p
        # amostragem log para densidade na fase inicial
        idx = np.unique(np.round(np.logspace(0, np.log10(B), 400)).astype(int))
        idx = np.clip(idx, 1, B) - 1
        x = idx + 1

        finals = []
        for opt in OPT_ORDER:
            hs = sub[sub["optimizer"] == opt]["hist"].tolist()
            if not hs:
                continue
            stack = _padded_stack(hs, B)
            mean = stack.mean(axis=0)
            std = stack.std(axis=0)
            finals.append(mean[-1])
            c = OPT_COLOR[opt]
            ax.plot(x, mean[idx], color=c, label=OPT_LABEL[opt])
            ax.fill_between(x, (mean - std)[idx], (mean + std)[idx],
                            color=c, alpha=0.12, linewidth=0)
        ax.set_xscale("log")   # log abre a fase inicial: cobyla single dispara em ~40 aval.
        lo, hi = min(finals), max(finals)
        # zoom no patamar das médias. As bandas de desvio se cruzam muito (a
        # separação real é ~0.01), então foco nas médias; a cauda fica no boxplot.
        ax.set_ylim(lo - 0.045, hi + 0.02)
        ax.set_title(f"p = {p}")
        ax.set_ylabel("approx. ratio")
        ax.grid(True, which="both", alpha=0.3)
    axes[-1].set_xlabel("circuit evaluations (log)")
    axes[0].legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"figura de convergencia salva em {out}")


def stability_figure(df, p_list=(1, 2), out="figs/stability.pdf"):
    fig, axes = plt.subplots(1, len(p_list), figsize=(COL_W, 2.4), sharey=False)
    if len(p_list) == 1:
        axes = [axes]

    for ax, p in zip(axes, p_list):
        sub = df[df["p"] == p]
        data = [sub[sub["optimizer"] == o]["r_final"].to_numpy() for o in OPT_ORDER]
        bp = ax.boxplot(data, showfliers=True, patch_artist=True, widths=0.6,
                        flierprops=dict(marker="o", markersize=2, alpha=0.4))
        for patch, opt in zip(bp["boxes"], OPT_ORDER):
            patch.set_facecolor(OPT_COLOR[opt])
            patch.set_alpha(0.55)
        for med in bp["medians"]:
            med.set_color("black")
        ax.set_xticks(range(1, len(OPT_ORDER) + 1))
        ax.set_xticklabels(["restart", "QIEA", "random", "single"],
                           rotation=30, ha="right")
        ax.set_title(f"p = {p}")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("final approx. ratio")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"figura de estabilidade salva em {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results_laptop_scaled.csv")
    ap.add_argument("--p_list", nargs="+", type=int, default=[1, 2])
    args = ap.parse_args()

    df = load(args.csv)
    convergence_figure(df, args.p_list)
    stability_figure(df, args.p_list)


if __name__ == "__main__":
    main()
