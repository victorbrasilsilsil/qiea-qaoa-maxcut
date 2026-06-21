#!/bin/sh
# Compila o resumo: pdflatex, bibtex, pdflatex x2. Rode de dentro de paper/.
set -e
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
echo "OK: main.pdf gerado"
