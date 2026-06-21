# QGA-QAOA

QIEA (Quantum-Inspired Evolutionary Algorithm, Han & Kim) otimizando os parâmetros 
variacionais do QAOA para Max-Cut, comparado contra baselines clássicos (COBYLA, 
busca aleatória, COBYLA com restart) sob o mesmo orçamento de avaliações de circuito.

## Regra de arquitetura

O núcleo é NumPy puro, sem Pennylane. O simulador QAOA é statevector
(`qaoa.py`) para que o mesmo código rode tanto em uma máquina comum quanto numa QLM, 
que só roda notebook e só tem numpy garantido (scipy via `!pip install`, usado apenas
pelo COBYLA).

## Módulos

| Arquivo            | Papel                                                        |
|--------------------|-------------------------------------------------------------|
| `graphs.py`        | gera/salva instâncias; força bruta do Max-Cut               |
| `qaoa.py`          | simulador statevector NumPy; objetivo ⟨H_C⟩; contador        |
| `encoding.py`      | bitstring ↔ ângulos                                          |
| `qiea.py`          | otimizador QIEA binário (observe, rotate, optimize)         |
| `objective.py`     | liga o fitness do QIEA ao objetivo do QAOA                  |
| `baselines.py`     | COBYLA, busca aleatória, COBYLA com restart                |
| `run_experiment.py`| orquestra a varredura, grava CSV                            |
| `plots.py`         | figuras a partir do CSV                                     |
| `stats.py`         | Wilcoxon pareado por seed                                   |
| `run_qlm.ipynb`    | notebook driver para a QLM do LNCC                          |

## Setup

```
python -m venv venv
venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -r requirements.txt
```

## Reprodutibilidade

Seeds fixas, instâncias em `instances.json`, resultados crus em CSV,
plot separado da execução. Seeds pareadas entre otimizadores.
