# Group Meeting Reports

Each report lives in its own `YYYY-MM-DD/` folder (the date of the
presentation), e.g. `2026-08-24/`.

## Compiling

**Use `latexmk`, not `pdflatex` directly:**

```bash
cd 2026-08-24
latexmk report.tex
```

Do not run `pdflatex report.tex` -- it fails immediately with a fatal
`fontspec` error and appears to hang (it's actually stuck waiting on stdin
at pdflatex's error prompt). The shared preamble
(`guidelines/dsrlab_beamer_preamble.tex`) uses the `fontspec` package to
load the lab's Carlito font, and `fontspec` only works under LuaLaTeX or
XeLaTeX -- never plain `pdflatex`.

Each dated folder's `.latexmkrc` is already configured to invoke the right
engine:

```perl
$pdf_mode = 4;
$lualatex = 'lualatex --shell-escape %O %S';
```

so `latexmk report.tex` alone picks the correct engine automatically.

**If compiling via VSCode's LaTeX Workshop extension** (or any tool that
invokes `latexmk -pdf ...`): the `-pdf` command-line flag forces
`pdf_mode=1` (pdflatex) and **overrides** `$pdf_mode = 4` from `.latexmkrc`
-- latexmk always lets CLI flags win over the rc file. This silently
reintroduces the same fontspec fatal error (no PDF, no obvious error dialog
in the extension). The `.latexmkrc` in each dated folder guards against
this by also redefining `$pdflatex` itself to invoke lualatex:

```perl
$pdflatex = 'lualatex --shell-escape %O %S';
```

so mode 1 and mode 4 both end up running lualatex regardless of which flag
wins. If you create a new dated folder, copy the full 3-line `.latexmkrc`
(not just the first two lines) to keep this protection.

To clean up build artifacts (`.aux`, `.log`, `.fdb_latexmk`, etc.):

```bash
latexmk -c report.tex
```

## New report checklist

1. Create a new `YYYY-MM-DD/` folder with `report.tex` and a `.latexmkrc`
   (copy an existing dated folder's `.latexmkrc` verbatim -- all 3 lines,
   see the VSCode note above for why the third line matters).
2. `\input{../guidelines/dsrlab_beamer_preamble.tex}` at the top of
   `report.tex` for the lab's Beamer theme (colors/fonts/title page/footer
   all defined there -- see that file's own comments for details).
3. Set `\title{}`, `\author{}`, `\date{YYYY-MM-DD}`, `\advisor{Name}`.
4. Follow the mandatory section order and checklist documented in
   `guidelines/報告注意事項.pdf` (also summarized in a trailing comment
   block in past `report.tex` files).

## Directories

- `guidelines/` -- shared Beamer preamble (`dsrlab_beamer_preamble.tex`),
  the lab's PowerPoint template (`template.potx`, source of the extracted
  colors/fonts/positions), and the presentation checklist
  (`報告注意事項.pdf`)
- `YYYY-MM-DD/` -- one folder per report; only `.tex`, `.pdf`, and
  `.latexmkrc` are meant to be kept/committed (see repo-root `.gitignore`
  for the build-artifact exclusions)
