# 教育部獎學金申請：編譯各份文件，再合併為 moe_scholarship.pdf
#   make        編譯全部並合併
#   make clean  清除 LaTeX 暫存檔（保留 PDF）
#
# TeX Live 與 latexmk 所需之 Perl 模組安裝於 $HOME（無 root 權限）；
# taskset 避開本工作站故障的 CPU 1。

export PATH     := $(HOME)/texlive/2026/bin/x86_64-linux:$(PATH)
export PERL5LIB := $(HOME)/perl5/lib/perl5

LATEXMK := taskset -c 0,2-23 latexmk -lualatex -interaction=nonstopmode -halt-on-error

# Order here only matters for building; the merge order is set in moe_scholarship.tex.
DOCS := 01_research_proposal/proposal 02_research_excellence/excellence 03_other_documents/profile

.PHONY: all docs clean $(DOCS)

all: docs
	$(LATEXMK) moe_scholarship.tex

docs: $(DOCS)

# latexmk decides itself whether each document is out of date.
$(DOCS):
	cd $(dir $@) && $(LATEXMK) $(notdir $@).tex

clean:
	for d in $(DOCS); do (cd $$(dirname $$d) && latexmk -c $$(basename $$d).tex); done
	latexmk -c moe_scholarship.tex
	rm -f */*.synctex.gz *.synctex.gz
