# LaTeX paper — Hybrid CNN–Transformer with Clinical Metadata (ISIC-2019)

Overleaf/PRISM-ready project. Formatting follows the BracU CSE400 Final Year
Thesis Template (documentclass, packages, biblatex IEEE style, front matter,
chapter ordering).

## Build
- **Compiler:** pdfLaTeX
- **Bibliography:** Biber (biblatex, IEEE style)
- On Overleaf: set Compiler = pdfLaTeX (Menu → Settings). It runs biber
  automatically. Locally: `pdflatex main` → `biber main` → `pdflatex main` → `pdflatex main`.

## Structure
```
main.tex                  # entry point (mirrors the template)
core/                     # titlepage, abstract, declaration, approval, etc.
chapters/                 # chapter_1,2,3,5,6,9 (see mapping below)
bibliography/references.bib
figures/                  # the 7 generated figures (copied from ../figures)
appendix/appendix_1.tex   # reproducibility appendix
```

## Chapter mapping (template chapters kept verbatim)
| Template chapter | Logical content |
|---|---|
| 1 Introduction | Introduction |
| 2 Literature Review | Related Work |
| 3 Requirements, Impacts and Constraints | Dataset + requirements + constraints |
| 4 Proposed Methodology | Methodology |
| 5 Result Analysis | Experiments + Results + Discussion |
| 6 Conclusion | Conclusion |

## Figures
The 7 result figures are real, copied from the codebase `figures/` output
(`01_training_curves` … `07_dekan_ablation`). The **architecture diagram
(Figure 4.1) is drawn inline with TikZ** in `chapters/chapter_5.tex` — no
external image file is needed.

## To complete before submission
- Fill student names/IDs, semester/year, supervisor block on the title page,
  declaration, and approval pages.
- All quantitative values are from `results/*/seed42/` (single seed, seed 42).
