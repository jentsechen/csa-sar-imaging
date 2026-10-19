$pdf_mode = 4;
$lualatex = 'lualatex --shell-escape %O %S';
# VSCode LaTeX Workshop's built-in "latexmk" tool passes -pdf on the command
# line, which forces pdf_mode=1 (pdflatex) and OVERRIDES $pdf_mode above --
# the preamble's fontspec package fatally errors under plain pdflatex. To
# survive that override, make $pdflatex itself invoke lualatex too, so mode
# 1 and mode 4 both end up running the same (correct) engine.
$pdflatex = 'lualatex --shell-escape %O %S';
