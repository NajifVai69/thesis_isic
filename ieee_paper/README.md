# IEEE-format paper — Hybrid CNN–Transformer with Clinical Metadata (ISIC-2019)

This folder is a **structural reformat** of the thesis in `../final_report_latex/`
into IEEE two-column conference style (`IEEEtran`). No academic content — text,
numbers, equations, citations, figures, or captions — was changed. Only the
document structure, sectioning, and LaTeX formatting commands were converted.

## Structure
```
main.tex                 # IEEEtran master document
sections/
  abstract.tex           # IEEE abstract environment + IEEEkeywords
  introduction.tex       # was Chapter 1 (Introduction)
  related_work.tex       # was Chapter 2 (Literature Review)
  requirements.tex       # was Chapter 3 (Requirements, Impacts and Constraints)
  methodology.tex        # was Chapter 4 / chapter_5.tex (Proposed Methodology)
  results.tex            # was Chapter 5 / chapter_6.tex (Result Analysis)
  conclusion.tex         # was Chapter 6 / chapter_9.tex (Conclusion)
figures/                 # exact figures copied from the thesis (unchanged)
references.bib           # exact .bib copied from the thesis (unchanged)
```

## IEEEtran.cls
`IEEEtran.cls` is **not** bundled here. It ships with every standard TeX
distribution (TeX Live, MiKTeX) and is preinstalled on Overleaf, so you do not
normally need a local copy:

- **Overleaf:** upload this folder, set *Compiler = pdfLaTeX*. It just works.
- **Local TeX Live / MiKTeX:** `IEEEtran.cls` is already on your path. If not,
  install it with `tlmgr install ieeetran` (TeX Live) or via the MiKTeX package
  manager, or download `IEEEtran.cls` from CTAN
  (https://ctan.org/pkg/ieeetran) and drop it in this folder.

## Build
```
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```
The bibliography uses the traditional `\bibliographystyle{IEEEtran}` +
`\bibliography{references}` path (BibTeX), per the requested IEEE format. The
original thesis used biblatex/biber; the `.bib` file itself is unchanged and is
compatible with both.

## Notes on the conversion
- **Heading levels were shifted down one rank** so thesis chapters become IEEE
  `\section`s: each chapter's internal `\section`→`\subsection` and
  `\subsection`→`\subsubsection`. All section titles are otherwise verbatim.
- **Wide tables and full-width figures** (main results, requirement tables,
  schedule/risk/TCO tables, the TikZ architecture diagram, the training-curve
  and model-comparison plots) use the two-column-spanning `table*` / `figure*`
  environments with `[!t]` placement. Narrower tables and the square
  confusion-matrix / scatter / bar figures use single-column `table` / `figure`
  with `[!t]`. `\resizebox`/`tabularx` targets were switched between
  `\textwidth` and `\columnwidth` to match the chosen float width.
- **Two display equations are numbered** (the CO₂ formula in the requirements
  section and the Class-Balanced Focal Loss); the focal-loss equation was
  already numbered in the thesis, and the CO₂ formula was promoted from an
  unnumbered display to a numbered `equation` per the "give formula serial
  numbers if needed" instruction.
- **Thesis-only front matter was not carried over**, as it has no IEEE paper
  equivalent: the title page, Declaration, Approval, and the Nomenclature list.
  The title and author names/IDs are preserved in the IEEE `\title`/`\author`
  block; the supervisor/committee block (Approval page) is omitted.
- **In-text cross-references to "Chapter 4/5", "Section 3.1.4", and
  "Appendix A" were left exactly as written in the source** to honour the
  verbatim-preservation rule. Because the paper is now flat-sectioned, these
  literal references no longer resolve to numbered targets; adjust them by hand
  if you want them to point at the new section numbers.
- `figures/A1_baseline_confusion_matrices.pdf`, `A2_all_training_curves.pdf`,
  and `summary_table.csv` were copied for completeness but are not referenced in
  the body (they were unreferenced in the thesis chapters as well), so they are
  not `\includegraphics`'d.
