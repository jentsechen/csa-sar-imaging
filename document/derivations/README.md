# Derivations

Mathematical derivations behind the code, one standalone LaTeX note per topic
(`<topic>.tex` + compiled `<topic>.pdf`).

| Note | Covers |
|---|---|
| [`thresholding.tex`](thresholding.tex) | Soft (L1), hard (L0) and half (L1/2) thresholding of a complex image: the problem on C^N reduces to a magnitude problem, so each operator keeps the phase and thresholds the magnitude |

Build: `latexmk -pdf <topic>.tex` (build files are git-ignored).
