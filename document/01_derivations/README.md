# Derivations

Mathematical derivations behind the code, one standalone LaTeX note per topic
(`<topic>.tex` + compiled `<topic>.pdf`).

| Note | Covers |
|---|---|
| [`thresholding.tex`](thresholding.tex) | Soft (L1), hard (L0) and half (L1/2) thresholding of a complex image: the problem on C^N reduces to a magnitude problem, so each operator keeps the phase and thresholds the magnitude |
| [`l12tv_splitting.tex`](l12tv_splitting.tex) | Eqs. (14)-(19) of Liu et al., "A Sparse SAR Imaging Method for Low-Oversampled Staggered Mode via Compound Regularization," Remote Sens. 16(8), 1459, 2024: auxiliary variables turn the L1/2&TV problem into a constrained one, and alternating minimization of its Lagrangian splits it into three subproblems (quadratic, L1/2, TV); subproblem 1 has a per-pixel closed form because the CSA operator is unitary |
| [`csa_unitary.tex`](csa_unitary.tex) | The CSA operator, written as a matrix on the vectorized echo, is a product of DFTs and unit-magnitude phase multiplications, hence unitary; so the echo-domain fidelity equals the image-domain fidelity (ported from the midterm review response, Appendix A) |

Build: `latexmk -pdf <topic>.tex` (build files are git-ignored).
