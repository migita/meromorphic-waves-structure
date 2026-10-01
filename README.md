# Meromorphic travelling waves for pure-power equations

**One pole orbit and a two-wave bound**

Alexander Migita · 1 October 2026

The submission repository for this paper. Read [main.pdf](main.pdf).

| File | Contents |
|---|---|
| `main.tex` | Complete manuscript |
| `references.bib` | The 41 references cited in the manuscript |
| `main.bbl` | Generated bibliography for source submission |
| `main.pdf` | Compiled paper |
| `anc/` | Exact checks, helper modules, and JSON certificates |

## Build and reproduce

The PDF builds with Tectonic (tested with version 0.17.0). The checks
require Python 3 and SymPy. To install the tested Python dependency in
a local environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r anc/requirements.txt
make pdf
make check PYTHON=.venv/bin/python
```

`make pdf` rebuilds both `main.pdf` and `main.bbl`. If Tectonic is not on
your path, use `make pdf TECTONIC=/path/to/tectonic`. Alternatively, with
a TeX Live installation containing the packages used by the manuscript:

```sh
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

`make check` runs all 13 ancillary scripts and regenerates the JSON
certificates. See [anc/README.md](anc/README.md) for the correspondence
between scripts and statements in the paper, including the exact
elimination used in Proposition 9.5. These computations supplement the
proofs in the manuscript. The full check suite requires only Python
and SymPy; no external polynomial solver is needed.

## Prepare the submission files

```sh
make archives PYTHON=.venv/bin/python
```

This rebuilds the paper, runs the checks, and creates:

- `dist/arxiv-source.tar.gz`: `main.tex`, its bibliography, and `anc/`.
- `dist/zenodo-bundle.zip`: the PDF, source, bibliography, ancillary
  materials, and build instructions.
- `dist/SHA256SUMS.txt`: checksums of the two archives.

The arXiv archive uses the required `anc/` directory and excludes the
compiled PDF and TeX temporary files; see arXiv's
[source submission instructions](https://info.arxiv.org/help/submit_tex.html)
and [ancillary file instructions](https://info.arxiv.org/help/ancillary_files.html).
For Zenodo, `main.pdf` is also available separately as a readable download.
The packaging script includes an explicit file list, so local notes and
build logs cannot enter the archives. It does not submit or upload them.

## Version

Prepared from revision `196cf7572e6e826bb4f06f73697a13a0ea63deb0`
of `migita/meromorphic-waves-modulo-prime`, including the final abstract
correction. The standalone layout renames the manuscript, consolidates
its cited bibliography entries, and changes the ancillary folder name
to `anc/`. The mathematical text and ancillary programs are unchanged.
