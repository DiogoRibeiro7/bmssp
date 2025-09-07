.PHONY: joss

joss:
	cffconvert --validate --infile CITATION.cff
	@if command -v pandoc >/dev/null 2>&1; then \
		pandoc paper/paper.md --standalone --from markdown --to pdf \
			--output paper/paper.pdf --bibliography paper/paper.bib; \
	else \
		echo 'pandoc not installed; skipping PDF build'; \
	fi
