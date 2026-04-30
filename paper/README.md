# Paper

This directory contains the TalentCLEF 2026 Task A working-notes paper source and compiled PDF.

Main files:

- `paper.tex`
- `references.bib`
- `paper.xmpdata`
- `paper.pdf`

Template support files included for reproducibility:

- `ceurart.cls`
- `elsarticle-num-names.bst`
- `pdfa.xmpi`
- `cc-by.pdf`
- `cc-by.png`

## Build

From this directory:

```bash
pdflatex -interaction=nonstopmode paper.tex
bibtex paper
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex
```

The committed `paper.pdf` is the compiled version of the current source.

## Notes

- Auxiliary LaTeX build files are intentionally not committed.
- The paper points back to the public repository for code and official search runs.
